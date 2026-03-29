from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class GBMParams:
	"""Parameters for geometric Brownian motion.

	SDE:
		dS_t = mu * S_t dt + sigma * S_t dW_t
	"""

	mu: float
	sigma: float


@dataclass(frozen=True)
class OUParams:
	"""Parameters for Ornstein-Uhlenbeck process.

	SDE:
		dX_t = theta * (mu - X_t) dt + sigma dW_t
	"""

	theta: float
	mu: float
	sigma: float


def euler_maruyama_gbm(
	s0: float,
	params: GBMParams,
	horizon: float,
	n_steps: int,
	n_paths: int = 1,
	seed: Optional[int] = None,
) -> tuple[Array, Array]:
	"""Simulate GBM paths with Euler-Maruyama discretization."""
	if s0 <= 0:
		raise ValueError("s0 must be strictly positive for GBM.")
	if params.sigma < 0:
		raise ValueError("sigma must be non-negative.")
	if n_steps <= 0 or n_paths <= 0:
		raise ValueError("n_steps and n_paths must be positive.")

	rng = np.random.default_rng(seed)
	dt = horizon / n_steps
	sqrt_dt = np.sqrt(dt)
	times = np.linspace(0.0, horizon, n_steps + 1)
	paths = np.empty((n_paths, n_steps + 1), dtype=float)
	paths[:, 0] = s0

	drift = params.mu * dt
	for step in range(1, n_steps + 1):
		z = rng.standard_normal(n_paths)
		increment = 1.0 + drift + params.sigma * sqrt_dt * z
		paths[:, step] = np.maximum(paths[:, step - 1] * increment, 1e-12)

	return times, paths


def euler_maruyama_ou(
	x0: float,
	params: OUParams,
	horizon: float,
	n_steps: int,
	n_paths: int = 1,
	seed: Optional[int] = None,
) -> tuple[Array, Array]:
	"""Simulate OU paths with Euler-Maruyama discretization."""
	if params.theta < 0:
		raise ValueError("theta must be non-negative.")
	if params.sigma < 0:
		raise ValueError("sigma must be non-negative.")
	if n_steps <= 0 or n_paths <= 0:
		raise ValueError("n_steps and n_paths must be positive.")

	rng = np.random.default_rng(seed)
	dt = horizon / n_steps
	sqrt_dt = np.sqrt(dt)
	times = np.linspace(0.0, horizon, n_steps + 1)
	paths = np.empty((n_paths, n_steps + 1), dtype=float)
	paths[:, 0] = x0

	for step in range(1, n_steps + 1):
		z = rng.standard_normal(n_paths)
		prev = paths[:, step - 1]
		paths[:, step] = prev + params.theta * (params.mu - prev) * dt + params.sigma * sqrt_dt * z

	return times, paths

