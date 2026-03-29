from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class GBMMLE:
    mu: float
    sigma: float
    nll: float


@dataclass(frozen=True)
class OUMLE:
    theta: float
    mu: float
    sigma: float
    nll: float


def _minimize_grid(
    objective: Callable[[Array], float],
    start: Array,
    step: Array,
    n_iter: int = 120,
) -> tuple[Array, float]:
    """Simple coordinate-search optimizer with step shrinkage."""
    x = start.astype(float).copy()
    best = objective(x)

    for _ in range(n_iter):
        improved = False
        for i in range(len(x)):
            candidates = [x.copy(), x.copy()]
            candidates[0][i] += step[i]
            candidates[1][i] -= step[i]
            for cand in candidates:
                score = objective(cand)
                if score < best:
                    x = cand
                    best = score
                    improved = True

        if not improved:
            step *= 0.6
            if np.all(step < 1e-8):
                break

    return x, best


def gbm_negative_log_likelihood_transformed(params_t: Array, prices: Array, dt: float) -> float:
    """NLL for GBM under transformed parameters [mu, log_sigma].

    For log returns r_t = log(S_t/S_{t-1}):
        r_t ~ N((mu - 0.5 sigma^2) dt, sigma^2 dt)
    """
    mu = params_t[0]
    sigma = np.exp(params_t[1])

    if sigma <= 0 or dt <= 0:
        return np.inf

    if np.any(prices <= 0):
        return np.inf

    returns = np.diff(np.log(prices))
    mean = (mu - 0.5 * sigma**2) * dt
    var = sigma**2 * dt

    n = len(returns)
    if n == 0:
        return np.inf

    residual = returns - mean
    nll = 0.5 * n * np.log(2.0 * np.pi * var) + 0.5 * np.sum((residual**2) / var)
    return float(nll)


def estimate_gbm_mle(prices: Array, dt: float) -> GBMMLE:
    """Estimate GBM parameters via MLE with positivity transform for sigma."""
    if np.any(prices <= 0):
        raise ValueError("GBM estimation requires strictly positive prices.")

    returns = np.diff(np.log(prices))
    if len(returns) < 2:
        raise ValueError("At least 3 price points are needed for stable MLE.")

    sample_mean = np.mean(returns)
    sample_std = np.std(returns, ddof=1)
    sigma0 = max(sample_std / np.sqrt(dt), 1e-6)
    mu0 = sample_mean / dt + 0.5 * sigma0**2

    start = np.array([mu0, np.log(sigma0)], dtype=float)
    step = np.array([0.1 * max(abs(mu0), 1.0), 0.2], dtype=float)

    obj = lambda x: gbm_negative_log_likelihood_transformed(x, prices, dt)
    best_t, best_nll = _minimize_grid(obj, start=start, step=step)

    return GBMMLE(mu=float(best_t[0]), sigma=float(np.exp(best_t[1])), nll=float(best_nll))


def ou_negative_log_likelihood_transformed(params_t: Array, series: Array, dt: float) -> float:
    """Gaussian transition density approximation for OU under EM discretization.

    X_{t+1} | X_t ~ N(X_t + theta(mu - X_t)dt, sigma^2 dt)
    with transformed parameters [log_theta, mu, log_sigma].
    """
    theta = np.exp(params_t[0])
    mu = params_t[1]
    sigma = np.exp(params_t[2])

    if theta < 0 or sigma <= 0 or dt <= 0:
        return np.inf

    x_prev = series[:-1]
    x_next = series[1:]
    if len(x_prev) == 0:
        return np.inf

    mean = x_prev + theta * (mu - x_prev) * dt
    var = sigma**2 * dt

    residual = x_next - mean
    n = len(residual)
    nll = 0.5 * n * np.log(2.0 * np.pi * var) + 0.5 * np.sum((residual**2) / var)
    return float(nll)


def estimate_ou_mle(series: Array, dt: float) -> OUMLE:
    """Estimate OU parameters with transformed MLE objective."""
    if len(series) < 3:
        raise ValueError("At least 3 observations are needed for OU MLE.")

    x_prev = series[:-1]
    x_next = series[1:]
    dx = x_next - x_prev

    beta = np.polyfit(x_prev, dx / dt, 1)
    theta0 = max(-beta[0], 1e-4)
    mu0 = beta[1] / theta0 if theta0 > 0 else np.mean(series)
    sigma0 = max(np.std(dx, ddof=1) / np.sqrt(dt), 1e-6)

    start = np.array([np.log(theta0), mu0, np.log(sigma0)], dtype=float)
    step = np.array([0.3, 0.2 * max(np.std(series), 1.0), 0.3], dtype=float)

    obj = lambda x: ou_negative_log_likelihood_transformed(x, series, dt)
    best_t, best_nll = _minimize_grid(obj, start=start, step=step)

    return OUMLE(
        theta=float(np.exp(best_t[0])),
        mu=float(best_t[1]),
        sigma=float(np.exp(best_t[2])),
        nll=float(best_nll),
    )
