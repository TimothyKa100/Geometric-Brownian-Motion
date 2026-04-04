from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from models import GBMParams, OUParams, euler_maruyama_gbm, euler_maruyama_ou, exact_step_gbm


@dataclass(frozen=True)
class OptionPriceEstimate:
    """Monte Carlo option estimate with 95% confidence interval."""

    price: float
    se: float
    ci_low: float
    ci_high: float
    n_paths: int


def _mc_summary(discounted_payoffs: np.ndarray, confidence: float = 0.95) -> OptionPriceEstimate:
    n_paths = int(discounted_payoffs.size)
    price = float(np.mean(discounted_payoffs))
    if n_paths <= 1:
        return OptionPriceEstimate(price=price, se=0.0, ci_low=price, ci_high=price, n_paths=n_paths)

    se = float(np.std(discounted_payoffs, ddof=1) / np.sqrt(n_paths))
    if abs(confidence - 0.95) > 1e-12:
        raise ValueError("Only 95% confidence intervals are currently supported.")
    z = 1.96
    return OptionPriceEstimate(
        price=price,
        se=se,
        ci_low=float(price - z * se),
        ci_high=float(price + z * se),
        n_paths=n_paths,
    )


def mc_price_european_gbm(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option: Literal["call", "put"] = "call",
    n_paths: int = 50_000,
    n_steps: int = 252,
    seed: Optional[int] = None,
    use_exact: bool = True,
) -> OptionPriceEstimate:
    if S0 <= 0 or K <= 0:
        raise ValueError("S0 and K must be strictly positive.")
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")
    if option not in ("call", "put"):
        raise ValueError("option must be 'call' or 'put'.")

    params = GBMParams(mu=r, sigma=sigma)
    if use_exact:
        _, paths = exact_step_gbm(s0=S0, params=params, horizon=T, n_steps=n_steps, n_paths=n_paths, seed=seed)
    else:
        _, paths = euler_maruyama_gbm(s0=S0, params=params, horizon=T, n_steps=n_steps, n_paths=n_paths, seed=seed)

    terminal = paths[:, -1]
    if option == "call":
        payoffs = np.maximum(terminal - K, 0.0)
    else:
        payoffs = np.maximum(K - terminal, 0.0)

    discounted = np.exp(-r * T) * payoffs
    return _mc_summary(discounted)


def mc_price_asian_arithmetic_call_gbm(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_paths: int = 50_000,
    n_steps: int = 252,
    seed: Optional[int] = None,
) -> OptionPriceEstimate:
    params = GBMParams(mu=r, sigma=sigma)
    _, paths = exact_step_gbm(s0=S0, params=params, horizon=T, n_steps=n_steps, n_paths=n_paths, seed=seed)
    avg_price = np.mean(paths[:, 1:], axis=1)
    payoffs = np.maximum(avg_price - K, 0.0)
    discounted = np.exp(-r * T) * payoffs
    return _mc_summary(discounted)


def mc_price_barrier_gbm(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    barrier: float,
    direction: Literal["down", "up"] = "down",
    knock: Literal["in", "out"] = "out",
    option: Literal["call", "put"] = "call",
    n_paths: int = 80_000,
    n_steps: int = 252,
    seed: Optional[int] = None,
) -> OptionPriceEstimate:
    if barrier <= 0:
        raise ValueError("barrier must be strictly positive for GBM asset prices.")
    if direction not in ("down", "up"):
        raise ValueError("direction must be 'down' or 'up'.")
    if knock not in ("in", "out"):
        raise ValueError("knock must be 'in' or 'out'.")

    params = GBMParams(mu=r, sigma=sigma)
    _, paths = exact_step_gbm(s0=S0, params=params, horizon=T, n_steps=n_steps, n_paths=n_paths, seed=seed)
    terminal = paths[:, -1]

    if option == "call":
        base_payoff = np.maximum(terminal - K, 0.0)
    else:
        base_payoff = np.maximum(K - terminal, 0.0)

    if direction == "down":
        hit = np.min(paths, axis=1) <= barrier
    else:
        hit = np.max(paths, axis=1) >= barrier

    active = hit if knock == "in" else ~hit
    payoffs = base_payoff * active.astype(float)
    discounted = np.exp(-r * T) * payoffs
    return _mc_summary(discounted)


def mc_price_european_ou_log_price(
    x0: float,
    ou_params: OUParams,
    K: float,
    T: float,
    r: float,
    option: Literal["call", "put"] = "call",
    n_paths: int = 40_000,
    n_steps: int = 252,
    seed: Optional[int] = None,
) -> OptionPriceEstimate:
    if K <= 0:
        raise ValueError("K must be strictly positive.")
    if option not in ("call", "put"):
        raise ValueError("option must be 'call' or 'put'.")

    _, x_paths = euler_maruyama_ou(
        x0=x0,
        params=ou_params,
        horizon=T,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
    )
    s_terminal = np.exp(x_paths[:, -1])

    if option == "call":
        payoffs = np.maximum(s_terminal - K, 0.0)
    else:
        payoffs = np.maximum(K - s_terminal, 0.0)

    discounted = np.exp(-r * T) * payoffs
    return _mc_summary(discounted)
