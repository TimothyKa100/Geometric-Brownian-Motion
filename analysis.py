from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from estimation import GBMMLE, OUMLE
from models import GBMParams, OUParams


Array = np.ndarray


@dataclass(frozen=True)
class SummaryStats:
    mean: float
    std: float
    skew: float
    kurtosis: float


@dataclass(frozen=True)
class EstimationError:
    abs_error: float
    rel_error: float


def _moments(x: Array) -> SummaryStats:
    centered = x - np.mean(x)
    std = np.std(x)
    if std == 0:
        return SummaryStats(mean=float(np.mean(x)), std=0.0, skew=0.0, kurtosis=-3.0)

    z = centered / std
    skew = np.mean(z**3)
    kurtosis = np.mean(z**4) - 3.0
    return SummaryStats(
        mean=float(np.mean(x)),
        std=float(std),
        skew=float(skew),
        kurtosis=float(kurtosis),
    )


def gbm_log_return_stats(paths: Array) -> SummaryStats:
    returns = np.diff(np.log(paths), axis=1).ravel()
    return _moments(returns)


def ou_increment_stats(paths: Array) -> SummaryStats:
    increments = np.diff(paths, axis=1).ravel()
    return _moments(increments)


def error_metrics(true_value: float, estimated_value: float) -> EstimationError:
    abs_err = abs(estimated_value - true_value)
    rel_err = abs_err / max(abs(true_value), 1e-12)
    return EstimationError(abs_error=float(abs_err), rel_error=float(rel_err))


def validate_gbm_estimates(true_params: GBMParams, estimated: GBMMLE) -> dict[str, EstimationError]:
    return {
        "mu": error_metrics(true_params.mu, estimated.mu),
        "sigma": error_metrics(true_params.sigma, estimated.sigma),
    }


def validate_ou_estimates(true_params: OUParams, estimated: OUMLE) -> dict[str, EstimationError]:
    return {
        "theta": error_metrics(true_params.theta, estimated.theta),
        "mu": error_metrics(true_params.mu, estimated.mu),
        "sigma": error_metrics(true_params.sigma, estimated.sigma),
    }


def gbm_empirical_mean(paths: Array) -> Array:
    return np.mean(paths, axis=0)


def gbm_theoretical_mean(s0: float, mu: float, times: Array) -> Array:
    return s0 * np.exp(mu * times)


def ou_empirical_mean(paths: Array) -> Array:
    return np.mean(paths, axis=0)


def ou_empirical_variance(paths: Array) -> Array:
    return np.var(paths, axis=0, ddof=1)


def ou_theoretical_mean(x0: float, theta: float, mu: float, times: Array) -> Array:
    return mu + (x0 - mu) * np.exp(-theta * times)


def ou_theoretical_variance(theta: float, sigma: float, times: Array) -> Array:
    return (sigma**2 / (2.0 * theta)) * (1.0 - np.exp(-2.0 * theta * times))


def max_relative_error(empirical: Array, theoretical: Array, eps: float = 1e-12) -> float:
    denom = np.maximum(np.abs(theoretical), eps)
    return float(np.max(np.abs(empirical - theoretical) / denom))


def gbm_standardized_residuals(prices: Array, mu_hat: float, sigma_hat: float, dt: float) -> Array:
    returns = np.diff(np.log(prices))
    mean = (mu_hat - 0.5 * sigma_hat**2) * dt
    std = max(sigma_hat * np.sqrt(dt), 1e-12)
    return (returns - mean) / std


def ou_standardized_residuals(series: Array, theta_hat: float, mu_hat: float, sigma_hat: float, dt: float) -> Array:
    x_prev = series[:-1]
    x_next = series[1:]
    mean = x_prev + theta_hat * (mu_hat - x_prev) * dt
    std = max(sigma_hat * np.sqrt(dt), 1e-12)
    return (x_next - mean) / std


def autocorrelation(x: Array, max_lag: int = 20) -> Array:
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    denom = np.sum(x**2)
    if denom <= 0:
        return np.zeros(max_lag + 1)

    acf = np.empty(max_lag + 1, dtype=float)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        acf[lag] = np.sum(x[:-lag] * x[lag:]) / denom
    return acf


def ljung_box_q(residuals: Array, lags: int = 10) -> float:
    n = len(residuals)
    if n <= lags + 1:
        return float("nan")
    acf = autocorrelation(residuals, max_lag=lags)
    q = n * (n + 2.0) * np.sum((acf[1:] ** 2) / (n - np.arange(1, lags + 1)))
    return float(q)


def qq_data_normal(residuals: Array) -> tuple[Array, Array]:
    """Return theoretical and empirical quantiles for QQ visualization.

    Uses a large normal sample approximation to avoid external SciPy dependency.
    """
    n = len(residuals)
    if n == 0:
        return np.array([]), np.array([])

    rng = np.random.default_rng(12345)
    normal_sample = np.sort(rng.standard_normal(200_000))
    probs = (np.arange(1, n + 1) - 0.5) / n
    idx = np.minimum((probs * (len(normal_sample) - 1)).astype(int), len(normal_sample) - 1)
    theo = normal_sample[idx]
    emp = np.sort(residuals)
    return theo, emp


def likelihood_ratio_test_ou_theta_zero(series: Array, theta_hat: float, mu_hat: float, sigma_hat: float, dt: float) -> tuple[float, float]:
    """LR test for H0: theta=0 (OU reduces to Brownian with drift)."""
    x_prev = series[:-1]
    x_next = series[1:]

    mean_full = x_prev + theta_hat * (mu_hat - x_prev) * dt
    var_full = sigma_hat**2 * dt
    ll_full = -0.5 * len(x_prev) * np.log(2.0 * np.pi * var_full) - 0.5 * np.sum((x_next - mean_full) ** 2 / var_full)

    drift0 = np.mean((x_next - x_prev) / dt)
    residual0 = x_next - (x_prev + drift0 * dt)
    sigma0 = np.sqrt(max(np.mean(residual0**2) / dt, 1e-12))
    var0 = sigma0**2 * dt
    ll_null = -0.5 * len(x_prev) * np.log(2.0 * np.pi * var0) - 0.5 * np.sum((residual0**2) / var0)

    lr = max(2.0 * (ll_full - ll_null), 0.0)
    p_value = math.erfc(np.sqrt(lr / 2.0))
    return float(lr), float(p_value)
