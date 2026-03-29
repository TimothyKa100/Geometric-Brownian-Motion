from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class FirstPassageSummary:
    hit_probability: float
    mean_hitting_time: float
    median_hitting_time: float


def first_hitting_times(paths: Array, level: float, above: bool = True) -> Array:
    """Return first index at which each path crosses level; np.inf if never hit."""
    n_paths, n_points = paths.shape
    result = np.full(n_paths, np.inf)

    for i in range(n_paths):
        path = paths[i]
        if above:
            idx = np.where(path >= level)[0]
        else:
            idx = np.where(path <= level)[0]
        if idx.size > 0:
            result[i] = float(idx[0])

    return result


def summarize_hitting_times(hitting_indices: Array, dt: float) -> FirstPassageSummary:
    finite = hitting_indices[np.isfinite(hitting_indices)]
    hit_probability = len(finite) / len(hitting_indices) if len(hitting_indices) > 0 else 0.0

    if len(finite) == 0:
        return FirstPassageSummary(
            hit_probability=float(hit_probability),
            mean_hitting_time=float("inf"),
            median_hitting_time=float("inf"),
        )

    times = finite * dt
    return FirstPassageSummary(
        hit_probability=float(hit_probability),
        mean_hitting_time=float(np.mean(times)),
        median_hitting_time=float(np.median(times)),
    )
