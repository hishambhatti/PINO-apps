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
