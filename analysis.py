from __future__ import annotations

from dataclasses import dataclass

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
