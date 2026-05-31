"""
Real-World Options Data for Black-Scholes PINO
===============================================

This script handles two workflows:

A) VALIDATION (train on synthetic, validate on real):
    from real_data_validation import (
        fetch_spy_options, filter_liquid_calls,
        build_validation_set, evaluate_pino_vs_market,
        plot_validation_results,
    )
    df = fetch_spy_options()
    calls = filter_liquid_calls(df)
    val_x, val_meta = build_validation_set(calls, config)
    results = evaluate_pino_vs_market(model, val_x, val_meta, config, device)
    plot_validation_results(results)

B) SPARSE TRAINING (train directly on real data):
    from real_data_validation import (
        fetch_spy_options, filter_liquid_calls,
        build_sparse_training_set, DataLoaderBSSparse,
        train_bs_real,
    )
    df = fetch_spy_options()
    calls = filter_liquid_calls(df)
    x_data, y_sparse, mask, meta = build_sparse_training_set(calls, config)
    dataset = DataLoaderBSSparse(x_data, y_sparse, mask,
                                  config['data']['nx'], config['data']['nt'],
                                  config['data']['sub'], config['data']['sub_t'])
    train_loader = dataset.make_loader(n_train, batch_size, start=0, train=True)
    train_bs_real(model, train_loader, optimizer, scheduler, config, rank=device)
"""

import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------------------------------------------------------------------
# 1. FETCH REAL OPTIONS DATA
# ---------------------------------------------------------------------------

def fetch_spy_options(ticker_symbol='SPY', max_expirations=6):
    """
    Pull live option chains from Yahoo Finance.

    Returns a DataFrame with columns:
        strike, expiration, mid_price, implied_vol, option_type,
        bid, ask, volume, openInterest, time_to_expiry_years, spot_price

    Parameters
    ----------
    ticker_symbol : str
        Underlying ticker (default SPY — European-style index, best BS fit)
    max_expirations : int
        How many expiration dates to pull (closest ones first)
    """
    import yfinance as yf

    tk = yf.Ticker(ticker_symbol)
    spot = tk.info.get('regularMarketPrice') or tk.info.get('previousClose')
    if spot is None:
        hist = tk.history(period='1d')
        spot = hist['Close'].iloc[-1]

    expirations = tk.options[:max_expirations]
    today = datetime.today()

    rows = []
    for exp_str in expirations:
        exp_date = datetime.strptime(exp_str, '%Y-%m-%d')
        T_years = (exp_date - today).days / 365.25
        if T_years <= 0:
            continue

        chain = tk.option_chain(exp_str)
        for opt_type, df in [('call', chain.calls), ('put', chain.puts)]:
            for _, row in df.iterrows():
                bid = row.get('bid', 0) or 0
                ask = row.get('ask', 0) or 0
                mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else row['lastPrice']
                iv = row.get('impliedVolatility', np.nan)

                rows.append({
                    'strike':              row['strike'],
                    'expiration':          exp_str,
                    'mid_price':           mid,
                    'implied_vol':         iv,
                    'option_type':         opt_type,
                    'bid':                 bid,
                    'ask':                 ask,
                    'volume':              row.get('volume', 0) or 0,
                    'openInterest':        row.get('openInterest', 0) or 0,
                    'time_to_expiry_years': T_years,
                    'spot_price':          spot,
                })

    df = pd.DataFrame(rows)
    print(f"Fetched {len(df)} contracts for {ticker_symbol} "
          f"(spot={spot:.2f}) across {len(expirations)} expirations")
    return df


def filter_liquid_calls(df, moneyness_range=(0.8, 1.2), min_volume=10):
    """
    Keep only liquid, near-the-money call options for cleaner validation.
    Moneyness = S / K.
    """
    calls = df[df['option_type'] == 'call'].copy()
    calls['moneyness'] = calls['spot_price'] / calls['strike']
    mask = (
        (calls['moneyness'] >= moneyness_range[0]) &
        (calls['moneyness'] <= moneyness_range[1]) &
        (calls['volume'] >= min_volume) &
        (calls['mid_price'] > 0.10)
    )
    filtered = calls[mask].copy()
    print(f"Filtered to {len(filtered)} liquid near-the-money calls")
    return filtered


# ---------------------------------------------------------------------------
# 2. BUILD VALIDATION TENSORS MATCHING PINO INPUT FORMAT
# ---------------------------------------------------------------------------

def build_validation_set(options_df, config):
    """
    Convert real option data into the tensor format DataLoaderBS produces.

    For each real option (with strike K), we:
      - Build the payoff max(S - K, 0) on the model's S-grid
      - Attach normalized x and tau coordinates
      - Record the real market price + metadata for comparison

    Returns
    -------
    val_x : Tensor [N_options, Nt, Nx, 3]
        Model input (same format as training data)
    val_meta : list of dicts
        One per option: strike, market_price, implied_vol, T, spot, etc.
        Plus the grid indices (S_idx, tau_idx) where the real observation lives.
    """
    nx_full = config['data']['nx']
    nt_full = config['data']['nt']
    sub_x   = config['data']['sub']
    sub_t   = config['data']['sub_t']
    S_min   = config['data']['S_min']
    S_max   = config['data']['S_max']
    T_max   = config['data']['T']

    Nx = nx_full // sub_x
    Nt = nt_full // sub_t + 1

    # Normalized grids (matching DataLoaderBS.make_loader)
    gridx = np.linspace(0, 1, Nx + 1)[:-1]           # [Nx]
    gridt = np.linspace(0, 1, Nt)                     # [Nt]

    # Physical S-grid
    Lx = math.log(S_max / S_min)
    S_grid = np.exp(np.log(S_min) + gridx * Lx)      # [Nx]

    val_inputs = []
    val_meta   = []

    for _, row in options_df.iterrows():
        K     = row['strike']
        T_opt = row['time_to_expiry_years']
        spot  = row['spot_price']
        price = row['mid_price']
        iv    = row['implied_vol']

        # Skip if this option's T exceeds the model's T range
        if T_opt > T_max * 1.1:
            continue

        # Skip if spot is outside the S-grid
        if spot < S_min or spot > S_max:
            continue

        # --- Build the payoff (initial condition) on the S-grid ---
        payoff = np.maximum(S_grid - K, 0.0).astype(np.float32)   # [Nx]

        # --- Assemble input tensor [Nt, Nx, 3] ---
        #   channel 0: payoff (repeated across all tau)
        #   channel 1: x_norm (spatial coordinate)
        #   channel 2: tau_norm (temporal coordinate)
        x_input = np.zeros((Nt, Nx, 3), dtype=np.float32)
        x_input[:, :, 0] = payoff[None, :]                       # broadcast payoff
        x_input[:, :, 1] = gridx[None, :]                        # broadcast x_norm
        x_input[:, :, 2] = gridt[:, None]                        # broadcast tau_norm

        # --- Find closest grid indices for the real observation ---
        # S index: where on the S-grid is the current spot price?
        S_idx = int(np.argmin(np.abs(S_grid - spot)))
        # tau index: tau = T - t, and for a current option with time_to_expiry T_opt,
        # the "today" price corresponds to tau = T_opt.
        # In normalized coords: tau_norm = T_opt / T_max (clamped to [0, 1])
        tau_norm_real = min(T_opt / T_max, 1.0)
        tau_idx = int(np.argmin(np.abs(gridt - tau_norm_real)))

        val_inputs.append(x_input)
        val_meta.append({
            'strike':       K,
            'market_price': price,
            'implied_vol':  iv,
            'T_years':      T_opt,
            'spot':         spot,
            'S_idx':        S_idx,
            'S_at_idx':     S_grid[S_idx],
            'tau_idx':      tau_idx,
            'tau_norm':     gridt[tau_idx],
            'moneyness':    spot / K,
        })

    val_x = torch.tensor(np.array(val_inputs), dtype=torch.float32)
    print(f"Built validation set: {val_x.shape[0]} options, "
          f"input shape per sample: {tuple(val_x.shape[1:])}")
    return val_x, val_meta


# ---------------------------------------------------------------------------
# 3. EVALUATE PINO vs MARKET
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_pino_vs_market(model, val_x, val_meta, config, device,
                            batch_size=32):
    """
    Run the trained PINO on real-option inputs and compare predictions
    to market prices.

    Returns a DataFrame with columns:
        strike, market_price, pino_price, bs_analytical_price,
        pino_error, bs_error, implied_vol, moneyness, T_years
    """
    model.eval()
    r     = config['data']['r']
    sigma = config['data']['sigma']

    results = []

    for i in range(0, len(val_meta), batch_size):
        batch_x = val_x[i:i+batch_size].to(device)
        pred = model(batch_x)                                   # [B, Nt, Nx] or [B, Nt, Nx, 1]

        for j in range(batch_x.shape[0]):
            meta = val_meta[i + j]
            # Extract PINO's prediction at the real (S, tau) point
            pred_j = pred[j].squeeze()                          # [Nt, Nx]
            pino_price = pred_j[meta['tau_idx'], meta['S_idx']].item()

            # Analytical BS price for comparison
            bs_price = _bs_call_analytical(
                S=meta['spot'], K=meta['strike'],
                r=r, sigma=sigma, T=meta['T_years'])

            results.append({
                'strike':        meta['strike'],
                'market_price':  meta['market_price'],
                'pino_price':    pino_price,
                'bs_analytical': bs_price,
                'pino_error':    pino_price - meta['market_price'],
                'bs_error':      bs_price - meta['market_price'],
                'implied_vol':   meta['implied_vol'],
                'moneyness':     meta['moneyness'],
                'T_years':       meta['T_years'],
            })

    df = pd.DataFrame(results)
    print(f"\n=== Validation Summary ({len(df)} options) ===")
    print(f"PINO  mean abs error: ${df['pino_error'].abs().mean():.4f}")
    print(f"BS    mean abs error: ${df['bs_error'].abs().mean():.4f}")
    print(f"PINO  RMSE:           ${np.sqrt((df['pino_error']**2).mean()):.4f}")
    print(f"BS    RMSE:           ${np.sqrt((df['bs_error']**2).mean()):.4f}")
    return df


def _bs_call_analytical(S, K, r, sigma, T):
    """Closed-form European call price."""
    if T <= 0:
        return max(S - K, 0.0)
    from scipy.stats import norm
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


# ---------------------------------------------------------------------------
# 4. DIAGNOSTIC PLOTS
# ---------------------------------------------------------------------------

def plot_validation_results(results_df, save_dir=None):
    """
    Three-panel diagnostic plot:
      1. PINO price vs market price (scatter)
      2. Error vs moneyness (where does BS/PINO fail?)
      3. Error vs time-to-expiry
    """
    df = results_df.copy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Panel 1: predicted vs actual ---
    ax = axes[0]
    price_max = max(df['market_price'].max(), df['pino_price'].max()) * 1.1
    ax.scatter(df['market_price'], df['pino_price'],  alpha=0.5, s=20, label='PINO')
    ax.scatter(df['market_price'], df['bs_analytical'], alpha=0.5, s=20, label='BS analytical', marker='x')
    ax.plot([0, price_max], [0, price_max], 'k--', alpha=0.3, label='perfect')
    ax.set_xlabel('Market Price ($)')
    ax.set_ylabel('Model Price ($)')
    ax.set_title('Predicted vs Market Price')
    ax.legend()
    ax.set_xlim(0, price_max)
    ax.set_ylim(0, price_max)
    ax.grid(True, alpha=0.2)

    # --- Panel 2: error vs moneyness ---
    ax = axes[1]
    ax.scatter(df['moneyness'], df['pino_error'],  alpha=0.5, s=20, label='PINO')
    ax.scatter(df['moneyness'], df['bs_error'], alpha=0.5, s=20, label='BS analytical', marker='x')
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Moneyness (S/K)')
    ax.set_ylabel('Price Error ($)')
    ax.set_title('Error vs Moneyness')
    ax.legend()
    ax.grid(True, alpha=0.2)

    # --- Panel 3: error vs time to expiry ---
    ax = axes[2]
    ax.scatter(df['T_years'], df['pino_error'].abs(),  alpha=0.5, s=20, label='PINO |error|')
    ax.scatter(df['T_years'], df['bs_error'].abs(), alpha=0.5, s=20, label='BS |error|', marker='x')
    ax.set_xlabel('Time to Expiry (years)')
    ax.set_ylabel('|Price Error| ($)')
    ax.set_title('Absolute Error vs Time to Expiry')
    ax.legend()
    ax.grid(True, alpha=0.2)

    plt.tight_layout()

    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, 'real_data_validation.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved to {path}")
    plt.show()


def plot_vol_smile_comparison(results_df, save_dir=None):
    """
    For each expiration, plot the implied vol smile from market data
    vs the "effective vol" implied by PINO prices.
    """
    from scipy.optimize import brentq
    from scipy.stats import norm as sp_norm

    df = results_df.copy()

    def implied_vol_from_price(price, S, K, r, T, tol=1e-6):
        """Invert BS formula to get implied vol from a model price."""
        if T <= 0 or price <= max(S - K * math.exp(-r * T), 0) + tol:
            return np.nan
        def objective(sigma):
            return _bs_call_analytical(S, K, r, sigma, T) - price
        try:
            return brentq(objective, 1e-4, 5.0, xtol=tol)
        except (ValueError, RuntimeError):
            return np.nan

    # Compute PINO implied vol for each option
    pino_ivs = []
    for _, row in df.iterrows():
        iv = implied_vol_from_price(
            price=row['pino_price'],
            S=row['moneyness'] * row['strike'],
            K=row['strike'], r=0.05, T=row['T_years'])
        pino_ivs.append(iv)
    df['pino_implied_vol'] = pino_ivs

    expirations = df['T_years'].unique()
    n_exp = min(len(expirations), 4)
    expirations = sorted(expirations)[:n_exp]

    fig, axes = plt.subplots(1, n_exp, figsize=(5 * n_exp, 4), squeeze=False)
    for i, T_val in enumerate(expirations):
        ax = axes[0, i]
        sub = df[df['T_years'] == T_val].sort_values('strike')
        ax.plot(sub['strike'], sub['implied_vol'],    'o-', label='Market IV', markersize=4)
        ax.plot(sub['strike'], sub['pino_implied_vol'], 's--', label='PINO IV', markersize=4)
        ax.set_xlabel('Strike')
        ax.set_ylabel('Implied Volatility')
        ax.set_title(f'T = {T_val:.3f} yr')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    plt.suptitle('Volatility Smile: Market vs PINO', y=1.02)
    plt.tight_layout()

    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, 'vol_smile_comparison.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved to {path}")
    plt.show()


# ---------------------------------------------------------------------------
# 5. SPARSE TRAINING SET (train on real data)
# ---------------------------------------------------------------------------

def build_sparse_training_set(options_df, config):
    """
    Group real options by strike K.  For each unique K, build:
      - payoff:    max(S - K, 0) on the model's S-grid        [Nx]
      - y_sparse:  mostly zeros, real prices at observed pts   [Nt, Nx]
      - mask:      1 at observed pts, 0 elsewhere              [Nt, Nx]

    Multiple expirations for the same K become multiple filled-in
    points along the tau axis of a single sample.

    Returns
    -------
    x_data  : Tensor [Nsamples, Nx]           payoffs
    y_sparse: Tensor [Nsamples, Nt, Nx]       sparse targets
    mask    : Tensor [Nsamples, Nt, Nx]        observation mask
    meta    : list of dicts                    per-sample info
    """
    nx_full = config['data']['nx']
    nt_full = config['data']['nt']
    sub_x   = config['data']['sub']
    sub_t   = config['data']['sub_t']
    S_min   = config['data']['S_min']
    S_max   = config['data']['S_max']
    T_max   = config['data']['T']

    Nx = nx_full // sub_x
    Nt = nt_full // sub_t + 1

    # Physical grids
    gridx = np.linspace(0, 1, Nx + 1)[:-1]
    gridt = np.linspace(0, 1, Nt)
    Lx = math.log(S_max / S_min)
    S_grid = np.exp(np.log(S_min) + gridx * Lx)

    # Group options by strike
    grouped = options_df.groupby('strike')

    x_list      = []
    y_sparse_list = []
    mask_list   = []
    meta_list   = []

    for K, group in grouped:
        spot = group['spot_price'].iloc[0]

        # Skip if spot is outside the model's S-grid
        if spot < S_min or spot > S_max:
            continue

        S_idx = int(np.argmin(np.abs(S_grid - spot)))

        # Payoff (initial condition)
        payoff = np.maximum(S_grid - K, 0.0).astype(np.float32)

        # Build sparse target and mask for this strike
        y_sp = np.zeros((Nt, Nx), dtype=np.float32)
        m    = np.zeros((Nt, Nx), dtype=np.float32)
        obs_count = 0

        for _, row in group.iterrows():
            T_opt = row['time_to_expiry_years']
            price = row['mid_price']

            if T_opt > T_max * 1.1 or T_opt <= 0:
                continue
            if price <= 0:
                continue

            tau_norm = min(T_opt / T_max, 1.0)
            tau_idx  = int(np.argmin(np.abs(gridt - tau_norm)))

            y_sp[tau_idx, S_idx] = price
            m[tau_idx, S_idx]    = 1.0
            obs_count += 1

        # Only keep samples that have at least one observation
        if obs_count == 0:
            continue

        x_list.append(payoff)
        y_sparse_list.append(y_sp)
        mask_list.append(m)
        meta_list.append({
            'strike': K,
            'spot': spot,
            'S_idx': S_idx,
            'n_observations': obs_count,
        })

    x_data   = torch.tensor(np.array(x_list),        dtype=torch.float32)
    y_sparse = torch.tensor(np.array(y_sparse_list),  dtype=torch.float32)
    mask     = torch.tensor(np.array(mask_list),       dtype=torch.float32)

    total_obs = int(mask.sum().item())
    total_pts = mask.numel()
    print(f"Built sparse training set: {len(meta_list)} samples (unique strikes)")
    print(f"  Grid size per sample: [{Nt}, {Nx}] = {Nt * Nx} points")
    print(f"  Total observations:   {total_obs} / {total_pts} "
          f"({100 * total_obs / total_pts:.4f}% filled)")
    return x_data, y_sparse, mask, meta_list


# ---------------------------------------------------------------------------
# 6. SPARSE DATALOADER
# ---------------------------------------------------------------------------

class DataLoaderBSSparse:
    """
    Like DataLoaderBS, but yields (x, y_sparse, mask) triples.

    x       : [batch, Nt, Nx, 3]   (payoff, x_norm, tau_norm)
    y_sparse: [batch, Nt, Nx]       sparse real prices
    mask    : [batch, Nt, Nx]       1 where observed, 0 elsewhere
    """
    def __init__(self, x_data, y_sparse, mask, nx, nt, sub=1, sub_t=1):
        s = nx
        if s % 2 == 1:
            s = s - 1
        self.s = s // sub
        self.T = nt // sub_t + 1

        # x_data is already [Nsamples, Nx] at subsampled resolution
        # y_sparse and mask are already [Nsamples, Nt, Nx] at subsampled resolution
        self.x_data  = x_data
        self.y_sparse = y_sparse
        self.mask     = mask

    def make_loader(self, n_sample, batch_size, start=0, train=True):
        Xs   = self.x_data[start:start + n_sample]
        ys   = self.y_sparse[start:start + n_sample]
        ms   = self.mask[start:start + n_sample]

        # Build coordinate grids (same as DataLoaderBS)
        gridx = torch.tensor(np.linspace(0, 1, self.s + 1)[:-1], dtype=torch.float)
        gridt = torch.tensor(np.linspace(0, 1, self.T), dtype=torch.float)
        gridx = gridx.reshape(1, 1, self.s)
        gridt = gridt.reshape(1, self.T, 1)

        # Expand payoff across time and stack with coordinates
        Xs = Xs.reshape(n_sample, 1, self.s).repeat([1, self.T, 1])
        Xs = torch.stack([
            Xs,
            gridx.repeat([n_sample, self.T, 1]),
            gridt.repeat([n_sample, 1, self.s]),
        ], dim=3)

        dataset = torch.utils.data.TensorDataset(Xs, ys, ms)
        loader  = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=train)
        return loader


# ---------------------------------------------------------------------------
# 7. TRAINING LOOP WITH SPARSE REAL DATA
# ---------------------------------------------------------------------------

def PINO_loss_bs_sparse(u, u0, mask, y_sparse,
                        r=0.05, sigma=0.2, T=1.0,
                        S_min=1.0, S_max=200.0):
    """
    PINO loss with sparse real-data supervision.

    u        : [batch, Nt, Nx]  model prediction (full surface)
    u0       : [batch, Nx]      payoff (initial condition)
    mask     : [batch, Nt, Nx]  1 at observed market points, 0 elsewhere
    y_sparse : [batch, Nt, Nx]  real market prices at observed points

    Returns
    -------
    loss_ic   : IC matching loss (payoff at tau=0)
    loss_f    : PDE residual loss (Black-Scholes equation)
    loss_data : masked data loss (only at observed points)
    """
    batchsize = u.size(0)
    nt = u.size(1)
    nx = u.size(2)
    u = u.reshape(batchsize, nt, nx)

    # --- IC loss: predicted surface at tau=0 must match payoff ---
    loss_ic = F.mse_loss(u[:, 0, :], u0)

    # --- PDE residual loss: BS equation everywhere on the grid ---
    Du = FDM_BlackScholes(u, r=r, sigma=sigma, T=T, S_min=S_min, S_max=S_max)
    loss_f = F.mse_loss(Du, torch.zeros_like(Du))

    # --- Sparse data loss: only at observed market points ---
    n_obs = mask.sum()
    if n_obs > 0:
        diff = (u - y_sparse) * mask
        loss_data = (diff ** 2).sum() / n_obs
    else:
        loss_data = torch.tensor(0.0, device=u.device)

    return loss_ic, loss_f, loss_data


def FDM_BlackScholes(u, D=1.0, r=0.05, sigma=0.2, T=1.0, S_min=1.0, S_max=200.0):
    """
    Black-Scholes PDE residual on a normalized (tau_norm, x_norm) grid.
    Duplicated here so this file is self-contained.
    """
    batchsize = u.size(0)
    nt = u.size(1)
    nx = u.size(2)
    u = u.reshape(batchsize, nt, nx)

    dt = D / (nt - 1)
    dx = D / nx

    Lx    = math.log(S_max / S_min)
    a     = 0.5 * sigma ** 2
    b     = r - 0.5 * sigma ** 2
    a_eff = a * T / Lx ** 2
    b_eff = b * T / Lx
    r_eff = r * T

    ux  = (u[:, :, 2:] - u[:, :, :-2])                        / (2.0 * dx)
    uxx = (u[:, :, 2:] - 2.0 * u[:, :, 1:-1] + u[:, :, :-2]) / dx ** 2
    utau = (u[:, 2:, :] - u[:, :-2, :]) / (2.0 * dt)

    Du = (utau[:, :, 1:-1]
          - a_eff * uxx[:, 1:-1, :]
          - b_eff * ux[:, 1:-1, :]
          + r_eff * u[:, 1:-1, 1:-1])
    return Du


def train_bs_real(model,
                  train_loader,
                  optimizer,
                  scheduler,
                  config,
                  rank=0,
                  log=False,
                  project='PINO-BS-RealData',
                  group='default',
                  tags=['real-data'],
                  use_tqdm=True):
    """
    Training loop for sparse real-data supervision.
    Same interface as train_bs but the loader yields (x, y_sparse, mask).
    """
    try:
        import wandb
    except ImportError:
        wandb = None

    if rank == 0 and wandb and log:
        run = wandb.init(project=project,
                         entity='shawngr2',
                         group=group,
                         config=config,
                         tags=tags, reinit=True,
                         settings=wandb.Settings(start_method='fork'))

    data_weight = config['train']['xy_loss']
    f_weight    = config['train']['f_loss']
    ic_weight   = config['train']['ic_loss']
    r     = config['data']['r']
    sigma = config['data']['sigma']
    T     = config['data']['T']
    S_min = config['data']['S_min']
    S_max = config['data']['S_max']
    ckpt_freq = config['train']['ckpt_freq']

    from train_utils.losses import LpLoss
    from train_utils.utils import save_checkpoint
    from tqdm import tqdm

    model.train()
    pbar = range(config['train']['epochs'])
    if use_tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)

    for e in pbar:
        model.train()
        train_pino = 0.0
        train_data = 0.0
        train_ic   = 0.0
        train_loss = 0.0

        for x, y_sparse, mask in train_loader:
            x, y_sparse, mask = x.to(rank), y_sparse.to(rank), mask.to(rank)
            out = model(x).reshape(y_sparse.shape)

            loss_ic, loss_f, loss_data = PINO_loss_bs_sparse(
                out, x[:, 0, :, 0], mask, y_sparse,
                r=r, sigma=sigma, T=T, S_min=S_min, S_max=S_max)

            total_loss = (loss_ic * ic_weight
                          + loss_f * f_weight
                          + loss_data * data_weight)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_data += loss_data.item()
            train_pino += loss_f.item()
            train_loss += total_loss.item()
            train_ic   += loss_ic.item()

        scheduler.step()
        n = len(train_loader)
        train_data /= n
        train_pino /= n
        train_loss /= n
        train_ic   /= n

        if use_tqdm:
            pbar.set_description(
                f'Epoch {e}, loss: {train_loss:.5f} '
                f'PDE: {train_pino:.5f} '
                f'data: {train_data:.5f} '
                f'IC: {train_ic:.5f}'
            )
        if wandb and log:
            wandb.log({
                'Train PDE error':  train_pino,
                'Train data error': train_data,
                'Train IC error':   train_ic,
                'Train loss':       train_loss,
            })

        if e % ckpt_freq == 0:
            save_checkpoint(config['train']['save_dir'],
                            config['train']['save_name'].replace('.pt', f'_real_{e}.pt'),
                            model, optimizer)

    save_checkpoint(config['train']['save_dir'],
                    config['train']['save_name'].replace('.pt', '_real_final.pt'),
                    model, optimizer)
    print('Done!')


# ---------------------------------------------------------------------------
# 8. STANDALONE ENTRY POINT (data fetch + save only, no model needed)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("Fetching real options data from Yahoo Finance...")
    print("=" * 60)

    df = fetch_spy_options(ticker_symbol='SPY', max_expirations=6)
    calls = filter_liquid_calls(df)

    # Save raw data
    out_path = 'data/spy_options_raw.csv'
    import os
    os.makedirs('data', exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved full dataset to {out_path}")

    calls_path = 'data/spy_calls_filtered.csv'
    calls.to_csv(calls_path, index=False)
    print(f"Saved filtered calls to {calls_path}")

    # Quick summary
    print(f"\n--- Data Summary ---")
    print(f"Spot price:        ${calls['spot_price'].iloc[0]:.2f}")
    print(f"Expirations:       {sorted(calls['expiration'].unique())}")
    print(f"Strike range:      ${calls['strike'].min():.0f} - ${calls['strike'].max():.0f}")
    print(f"Avg implied vol:   {calls['implied_vol'].mean():.4f}")
    print(f"Total contracts:   {len(calls)}")
    print(f"\nNOTE: S_min/S_max in your config is [{1.0}, {200.0}].")
    print(f"SPY is ~${calls['spot_price'].iloc[0]:.0f}, so you'll need to update")
    print(f"config S_min/S_max to cover SPY's range (e.g. S_min=300, S_max=800)")
    print(f"and retrain, OR use a cheaper underlying like IWM or a single stock.")