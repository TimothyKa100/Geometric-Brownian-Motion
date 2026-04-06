from __future__ import annotations

import numpy as np


Array = np.ndarray


def portfolio_pnl(paths: Array, weights: Array) -> Array:
    """Portfolio terminal return from simulated multi-asset paths."""
    weights = np.asarray(weights, dtype=float)
    if paths.ndim != 3:
        raise ValueError("paths must have shape (n_paths, n_assets, n_steps+1).")
    if paths.shape[1] != weights.size:
        raise ValueError("weights length must equal number of assets in paths.")

    s0 = paths[:, :, 0]
    s_t = paths[:, :, -1]
    returns = (s_t - s0) / s0
    return np.sum(returns * weights, axis=1)


def var_cvar(pnl: Array, alpha: float = 0.95) -> dict[str, float]:
    """Historical VaR and CVaR from simulated PnL samples."""
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1).")

    pnl = np.asarray(pnl, dtype=float)
    if pnl.size == 0:
        raise ValueError("pnl must be non-empty.")

    var = -np.quantile(pnl, 1.0 - alpha)
    tail = pnl[pnl <= -var]
    cvar = -tail.mean() if tail.size > 0 else var
    return {
        "VaR": float(var),
        "CVaR": float(cvar),
        "alpha": float(alpha),
        "n_tail": float(tail.size),
    }
