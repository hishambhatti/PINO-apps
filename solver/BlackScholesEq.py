import math
import torch
import matplotlib.pyplot as plt


class BlackScholesEq1D:
    """
    Reference solver for the 1D Black-Scholes PDE.

    PDE in original (S, t) coordinates:
        V_t + 0.5 * sigma^2 * S^2 * V_SS + r * S * V_S - r * V = 0

    We solve in transformed coordinates:
        x   = log(S)          (uniform grid, constant-coefficient PDE)
        tau = T - t           (forward time; the payoff becomes the "initial" condition)

    The transformed PDE is a constant-coefficient parabolic equation:
        v_tau = a * v_xx + b * v_x - r * v
    with
        a = 0.5 * sigma^2
        b = r - 0.5 * sigma^2

    Time stepping uses Crank-Nicolson on the interior nodes:
        (I - 0.5 * dtau * L) v_new = (I + 0.5 * dtau * L) v_old + boundary corrections
    L is the tridiagonal discretization of  a * d^2/dx^2 + b * d/dx - r * I.
    The two boundary nodes are imposed via Dirichlet BCs supplied at driver time.
    """

    def __init__(self,
                 S_min=1.0, S_max=200.0,
                 Nx=256,
                 r=0.05, sigma=0.2,
                 dtau=1e-3, T=1.0,
                 device=None, dtype=torch.float64):
        self.S_min = S_min
        self.S_max = S_max
        self.Nx = Nx
        self.r = r
        self.sigma = sigma
        self.dtau = dtau
        self.tend = T
        self.device = device
        self.dtype = dtype

        # Log-price grid: uniform in x = log(S) gives constant FD coefficients.
        self.x_grid = torch.linspace(math.log(S_min), math.log(S_max),
                                     Nx, device=device, dtype=dtype)
        self.dx = (self.x_grid[1] - self.x_grid[0]).item()
        self.S_grid = torch.exp(self.x_grid)

        # Field state (analogue of WaveEq1D's self.phi / self.phi0).
        # BS is first-order in time, so there's no second field like self.psi.
        self.v  = torch.zeros(Nx, device=device, dtype=dtype)
        self.v0 = torch.zeros(Nx, device=device, dtype=dtype)

        # Tau loop counters; snapshot buffers are allocated inside bs_driver
        # because their size depends on save_interval.
        self.tau = 0.0
        self.it  = 0
        self.V = None
        self.T_list = None

        # --- Precompute the Crank-Nicolson tridiagonal matrices once. ---
        # The three stencil coefficients come from central differences:
        #   v_xx_i = (v_{i-1} - 2 v_i + v_{i+1}) / dx^2
        #   v_x_i  = (v_{i+1} - v_{i-1}) / (2 dx)
        # so the operator L = a*d^2/dx^2 + b*d/dx - r*I has stencil
        #   alpha on v_{i-1},  beta on v_i,  gamma on v_{i+1}.
        a = 0.5 * sigma ** 2
        b = r - 0.5 * sigma ** 2
        self.alpha = a / self.dx ** 2 - b / (2.0 * self.dx)
        self.beta  = -2.0 * a / self.dx ** 2 - r
        self.gamma = a / self.dx ** 2 + b / (2.0 * self.dx)

        # Interior system: the two endpoints are Dirichlet, so the unknown
        # vector has size Nx - 2.
        n = Nx - 2
        ones_n   = torch.ones(n,     device=device, dtype=dtype)
        ones_nm1 = torch.ones(n - 1, device=device, dtype=dtype)
        L = (torch.diag(self.alpha * ones_nm1, diagonal=-1)
             + torch.diag(self.beta  * ones_n,   diagonal=0)
             + torch.diag(self.gamma * ones_nm1, diagonal=+1))
        I = torch.eye(n, device=device, dtype=dtype)
        # A is applied implicitly (LHS), B explicitly (RHS).
        self.A = I - 0.5 * dtau * L
        self.B = I + 0.5 * dtau * L

        # Tridiagonal storage of A. We keep the dense A above for readability,
        # but the actual implicit solve uses these vectors with our own
        # Thomas algorithm
        #   a_sub   = sub-diagonal of A,    length n-1
        #   a_diag  = main diagonal of A,   length n
        #   a_super = super-diagonal of A,  length n-1
        self.a_sub   = torch.full((n - 1,), -0.5 * dtau * self.alpha,
                                  device=device, dtype=dtype)
        self.a_diag  = torch.full((n,),     1.0 - 0.5 * dtau * self.beta,
                                  device=device, dtype=dtype)
        self.a_super = torch.full((n - 1,), -0.5 * dtau * self.gamma,
                                  device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Payoff helpers.  In tau = T - t coordinates the terminal payoff at
    # maturity is the INITIAL CONDITION for the tau-marching problem.
    # These are convenience functions; the driver also accepts any v0.
    # ------------------------------------------------------------------
    def call_payoff(self, K):
        """European call payoff  max(S - K, 0) on the asset grid."""
        return torch.clamp(self.S_grid - K, min=0.0)

    def put_payoff(self, K):
        """European put payoff  max(K - S, 0) on the asset grid."""
        return torch.clamp(K - self.S_grid, min=0.0)

    # ------------------------------------------------------------------
    # Dirichlet boundary conditions at S_min and S_max.  These return
    # CALLABLES of tau so the driver can ask for V at both the old and
    # new time levels (Crank-Nicolson needs both for boundary correction).
    #
    # For a European call:
    #   V(S_min, tau) -> 0                            (deep out-of-the-money)
    #   V(S_max, tau) -> S_max - K * exp(-r * tau)    (deep in-the-money)
    # For a European put:
    #   V(S_min, tau) -> K * exp(-r * tau) - S_min
    #   V(S_max, tau) -> 0
    # ------------------------------------------------------------------
    def call_bcs(self, K):
        S_lo = self.S_grid[0].item()
        S_hi = self.S_grid[-1].item()
        def lower(tau):
            return 0.0
        def upper(tau):
            return S_hi - K * math.exp(-self.r * tau)
        return lower, upper

    def put_bcs(self, K):
        S_lo = self.S_grid[0].item()
        S_hi = self.S_grid[-1].item()
        def lower(tau):
            return K * math.exp(-self.r * tau) - S_lo
        def upper(tau):
            return 0.0
        return lower, upper

    # ------------------------------------------------------------------
    # Thomas algorithm: O(n) direct solver for a tridiagonal system
    #     A x = d
    # written from scratch (no library PDE / banded solver is called).
    # Inputs are 1-D tensors describing the three non-zero diagonals.
    # ------------------------------------------------------------------
    def _thomas_solve(self, a_sub, a_diag, a_super, d):
        n = d.shape[0]
        # Modified super-diagonal (c') and modified RHS (d') after the
        # forward sweep of Gaussian elimination on a tridiagonal matrix.
        c_p = torch.empty(n - 1, device=d.device, dtype=d.dtype)
        d_p = torch.empty(n,     device=d.device, dtype=d.dtype)

        # Forward sweep: eliminate the sub-diagonal row by row.
        c_p[0] = a_super[0] / a_diag[0]
        d_p[0] = d[0]        / a_diag[0]
        for i in range(1, n):
            denom = a_diag[i] - a_sub[i - 1] * c_p[i - 1]
            if i < n - 1:
                c_p[i] = a_super[i] / denom
            d_p[i] = (d[i] - a_sub[i - 1] * d_p[i - 1]) / denom

        # Back substitution: the matrix is now upper-bidiagonal.
        x = torch.empty(n, device=d.device, dtype=d.dtype)
        x[-1] = d_p[-1]
        for i in range(n - 2, -1, -1):
            x[i] = d_p[i] - c_p[i] * x[i + 1]
        return x

    # ------------------------------------------------------------------
    # One Crank-Nicolson step in tau:
    #     A v_new = B v_old + boundary_correction
    # where A and B are the constant-coefficient tridiagonal operators
    # precomputed in __init__, and the boundary correction folds the
    # known Dirichlet values at S_min and S_max into rows 0 and -1 of
    # the interior system (averaging old and new time levels).
    # ------------------------------------------------------------------
    def bs_step(self, v, tau, bc_lower, bc_upper):
        # Boundary values at the start and end of this step.  CN uses
        # the average of the two so we need both.
        v_lo_old = bc_lower(tau)
        v_hi_old = bc_upper(tau)
        tau_new  = tau + self.dtau
        v_lo_new = bc_lower(tau_new)
        v_hi_new = bc_upper(tau_new)

        # Interior unknowns (the two endpoints are Dirichlet).
        v_in = v[1:-1]

        # Explicit half-step:  B @ v_old_in.  This is just a tridiagonal
        # mat-vec, so doing it through self.B is fine -- it is not a PDE
        # library call.
        rhs = self.B @ v_in

        # Boundary correction.  The first and last rows of L touch the
        # Dirichlet endpoints v_lo and v_hi; CN averages those across the
        # two time levels and multiplies by 0.5 * dtau.
        rhs[0]  = rhs[0]  + 0.5 * self.dtau * self.alpha * (v_lo_old + v_lo_new)
        rhs[-1] = rhs[-1] + 0.5 * self.dtau * self.gamma * (v_hi_old + v_hi_new)

        # Implicit half-step:  solve A x = rhs with our Thomas routine.
        v_new_in = self._thomas_solve(self.a_sub, self.a_diag, self.a_super, rhs)

        # Reassemble the full vector with the updated boundary values.
        v_new = torch.empty_like(v)
        v_new[0]    = v_lo_new
        v_new[-1]   = v_hi_new
        v_new[1:-1] = v_new_in
        return v_new, tau_new

    # ------------------------------------------------------------------
    # Plot helper.  Mirrors WaveEq1D.plot_data / BurgersEq1D.plot_data
    # but plots V against the asset price S (not the log-price x).
    # ------------------------------------------------------------------
    def plot_data(self, fig_num=0, title='', xlabel='S', ylabel='V',
                  vmin=None, vmax=None):
        plt.ion()
        fig = plt.figure(fig_num)
        plt.cla()
        plt.clf()
        plt.plot(self.S_grid.detach().cpu(), self.v.detach().cpu())
        if vmin is not None or vmax is not None:
            plt.ylim(vmin, vmax)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.draw()
        plt.pause(1e-17)
        plt.show()

    # ------------------------------------------------------------------
    # Top-level driver, modelled on wave_driver / burgers_driver:
    #   - takes the tau=0 initial condition v0 (typically the payoff),
    #   - takes two BC callables lower(tau), upper(tau),
    #   - marches tau from 0 to self.tend in steps of self.dtau,
    #   - snapshots every save_interval steps and returns the stack.
    # ------------------------------------------------------------------
    def bs_driver(self, v0, bc_lower, bc_upper,
                  save_interval=10, plot_interval=0):
        # Reset state so the driver can be called repeatedly.
        self.v0 = v0[:self.Nx].clone()
        self.v  = self.v0.clone()
        self.tau = 0.0
        self.it  = 0
        self.V = []
        self.T_list = []

        # Enforce Dirichlet endpoints on the initial condition as well so
        # that v and v0 are consistent with the BC callables at tau=0.
        # (Assignment handles both Python scalars and 0-d tensors.)
        self.v[0]  = bc_lower(0.0)
        self.v[-1] = bc_upper(0.0)

        # Save / plot the initial frame.
        if save_interval != 0 and self.it % save_interval == 0:
            self.V.append(self.v.clone())
            self.T_list.append(self.tau)
        if plot_interval != 0 and self.it % plot_interval == 0:
            self.plot_data(title=r'V(S, $\tau$=0)')

        # March in tau until we reach maturity (T).  A small epsilon
        # avoids fencepost issues from floating-point accumulation.
        while self.tau < self.tend - 1e-12:
            self.v, self.tau = self.bs_step(self.v, self.tau,
                                            bc_lower, bc_upper)
            self.it += 1

            if plot_interval != 0 and self.it % plot_interval == 0:
                self.plot_data(title=fr'V(S, $\tau$={self.tau:.3f})')
            if save_interval != 0 and self.it % save_interval == 0:
                self.V.append(self.v.clone())
                self.T_list.append(self.tau)

        return torch.stack(self.V)
