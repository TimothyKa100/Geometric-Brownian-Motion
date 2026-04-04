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
    """
    Return first index at which each path crosses level; np.inf if never hit.
    Fully vectorized — no Python loop over paths.
    """
    if above:
        crossed = paths >= level   # (n_paths, n_points) bool
    else:
        crossed = paths <= level

    ever_hit = crossed.any(axis=1)                        # (n_paths,) bool
    # argmax returns 0 for all-False rows — we mask those out
    first_idx = crossed.argmax(axis=1).astype(float)      # (n_paths,)
    first_idx[~ever_hit] = np.inf
    return first_idx


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
