from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

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


def exact_step_gbm(
	s0: float,
	params: GBMParams,
	horizon: float,
	n_steps: int,
	n_paths: int = 1,
	seed: Optional[int] = None,
) -> tuple[Array, Array]:
	"""Simulate GBM using exact exponential transition."""
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

	drift = (params.mu - 0.5 * params.sigma**2) * dt
	for step in range(1, n_steps + 1):
		z = rng.standard_normal(n_paths)
		paths[:, step] = paths[:, step - 1] * np.exp(drift + params.sigma * sqrt_dt * z)

	return times, paths


def euler_maruyama_jump_diffusion_gbm(
	s0: float,
	params: GBMParams,
	jump_intensity: float,
	jump_mean: float,
	jump_std: float,
	horizon: float,
	n_steps: int,
	n_paths: int = 1,
	seed: Optional[int] = None,
) -> tuple[Array, Array]:
	"""Merton-style jump diffusion with multiplicative log-normal jumps."""
	if jump_intensity < 0 or jump_std < 0:
		raise ValueError("jump_intensity and jump_std must be non-negative.")

	rng = np.random.default_rng(seed)
	dt = horizon / n_steps
	sqrt_dt = np.sqrt(dt)
	times = np.linspace(0.0, horizon, n_steps + 1)
	paths = np.empty((n_paths, n_steps + 1), dtype=float)
	paths[:, 0] = s0

	kappa = np.exp(jump_mean + 0.5 * jump_std**2) - 1.0
	drift = (params.mu - jump_intensity * kappa) * dt

	for step in range(1, n_steps + 1):
		z = rng.standard_normal(n_paths)
		n_jumps = rng.poisson(jump_intensity * dt, size=n_paths)
		jump_component = np.ones(n_paths)
		jump_paths = n_jumps > 0
		if np.any(jump_paths):
			counts = n_jumps[jump_paths]
			jump_sums = np.array([
				np.sum(rng.normal(loc=jump_mean, scale=jump_std, size=c)) for c in counts
			])
			jump_component[jump_paths] = np.exp(jump_sums)

		diffusion = np.exp((drift - 0.5 * params.sigma**2 * dt) + params.sigma * sqrt_dt * z)
		paths[:, step] = np.maximum(paths[:, step - 1] * diffusion * jump_component, 1e-12)

	return times, paths


def euler_maruyama_cir(
	x0: float,
	kappa: float,
	theta: float,
	sigma: float,
	horizon: float,
	n_steps: int,
	n_paths: int = 1,
	seed: Optional[int] = None,
) -> tuple[Array, Array]:
	"""Simulate CIR process with full truncation Euler scheme.

	dX_t = kappa(theta - X_t)dt + sigma * sqrt(max(X_t,0)) dW_t
	"""
	if kappa < 0 or theta < 0 or sigma < 0:
		raise ValueError("kappa, theta, sigma must be non-negative.")

	rng = np.random.default_rng(seed)
	dt = horizon / n_steps
	sqrt_dt = np.sqrt(dt)
	times = np.linspace(0.0, horizon, n_steps + 1)
	paths = np.empty((n_paths, n_steps + 1), dtype=float)
	paths[:, 0] = max(x0, 0.0)

	for step in range(1, n_steps + 1):
		prev = np.maximum(paths[:, step - 1], 0.0)
		z = rng.standard_normal(n_paths)
		nxt = prev + kappa * (theta - prev) * dt + sigma * np.sqrt(prev) * sqrt_dt * z
		paths[:, step] = np.maximum(nxt, 0.0)

	return times, paths


def euler_maruyama_gbm_time_varying(
	s0: float,
	mu_fn: Callable[[float], float],
	sigma_fn: Callable[[float], float],
	horizon: float,
	n_steps: int,
	n_paths: int = 1,
	seed: Optional[int] = None,
) -> tuple[Array, Array]:
	"""GBM with time-varying drift and volatility under Euler-Maruyama."""
	if s0 <= 0:
		raise ValueError("s0 must be strictly positive for GBM.")

	rng = np.random.default_rng(seed)
	dt = horizon / n_steps
	sqrt_dt = np.sqrt(dt)
	times = np.linspace(0.0, horizon, n_steps + 1)
	paths = np.empty((n_paths, n_steps + 1), dtype=float)
	paths[:, 0] = s0

	for step in range(1, n_steps + 1):
		t_prev = times[step - 1]
		mu_t = mu_fn(t_prev)
		sigma_t = sigma_fn(t_prev)
		if sigma_t < 0:
			raise ValueError("sigma_fn must be non-negative for all t.")

		z = rng.standard_normal(n_paths)
		paths[:, step] = np.maximum(
			paths[:, step - 1] * (1.0 + mu_t * dt + sigma_t * sqrt_dt * z),
			1e-12,
		)

	return times, paths


def euler_maruyama_ou_time_varying(
	x0: float,
	theta_fn: Callable[[float], float],
	mu_fn: Callable[[float], float],
	sigma_fn: Callable[[float], float],
	horizon: float,
	n_steps: int,
	n_paths: int = 1,
	seed: Optional[int] = None,
) -> tuple[Array, Array]:
	"""OU process with time-varying coefficients under Euler-Maruyama."""
	rng = np.random.default_rng(seed)
	dt = horizon / n_steps
	sqrt_dt = np.sqrt(dt)
	times = np.linspace(0.0, horizon, n_steps + 1)
	paths = np.empty((n_paths, n_steps + 1), dtype=float)
	paths[:, 0] = x0

	for step in range(1, n_steps + 1):
		t_prev = times[step - 1]
		theta_t = theta_fn(t_prev)
		mu_t = mu_fn(t_prev)
		sigma_t = sigma_fn(t_prev)
		if theta_t < 0 or sigma_t < 0:
			raise ValueError("theta_fn and sigma_fn must be non-negative for all t.")

		z = rng.standard_normal(n_paths)
		prev = paths[:, step - 1]
		paths[:, step] = prev + theta_t * (mu_t - prev) * dt + sigma_t * sqrt_dt * z

	return times, paths

