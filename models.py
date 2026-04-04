from __future__ import annotations

import math
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


@dataclass(frozen=True)
class BlackScholesState:
	"""d1/d2 state for Black-Scholes diffusion under lognormal assumptions."""

	d1: float
	d2: float


def normal_cdf(x: float) -> float:
	"""Standard normal CDF using the error function (no SciPy dependency)."""
	return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_state(S0: float, K: float, T: float, r: float, sigma: float) -> BlackScholesState:
	"""Return Black-Scholes d1/d2 terms used across analytical formulas."""
	if S0 <= 0 or K <= 0:
		raise ValueError("S0 and K must be strictly positive.")
	if T <= 0:
		raise ValueError("T must be strictly positive.")
	if sigma <= 0:
		raise ValueError("sigma must be strictly positive.")

	d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
	d2 = d1 - sigma * math.sqrt(T)
	return BlackScholesState(d1=float(d1), d2=float(d2))


def black_scholes_log_moments(S0: float, T: float, r: float, sigma: float) -> tuple[float, float]:
	"""Mean and std of log(S_T) under Black-Scholes risk-neutral dynamics."""
	if S0 <= 0:
		raise ValueError("S0 must be strictly positive.")
	if T < 0:
		raise ValueError("T must be non-negative.")
	if sigma < 0:
		raise ValueError("sigma must be non-negative.")

	mean = math.log(S0) + (r - 0.5 * sigma**2) * T
	std = sigma * math.sqrt(T)
	return float(mean), float(std)


def simulate_correlated_gbm_paths(
	S0s: Array,
	mus: Array,
	sigmas: Array,
	corr_matrix: Array,
	horizon: float,
	n_steps: int,
	n_paths: int,
	seed: Optional[int] = None,
) -> tuple[Array, Array]:
	"""Simulate correlated multi-asset GBM paths using Cholesky factorization."""
	S0s = np.asarray(S0s, dtype=float)
	mus = np.asarray(mus, dtype=float)
	sigmas = np.asarray(sigmas, dtype=float)
	corr_matrix = np.asarray(corr_matrix, dtype=float)

	n_assets = S0s.size
	if n_assets == 0:
		raise ValueError("S0s must contain at least one asset.")
	if mus.size != n_assets or sigmas.size != n_assets:
		raise ValueError("S0s, mus, sigmas must have matching lengths.")
	if corr_matrix.shape != (n_assets, n_assets):
		raise ValueError("corr_matrix must have shape (n_assets, n_assets).")

	rng = np.random.default_rng(seed)
	dt = horizon / n_steps
	sqrt_dt = np.sqrt(dt)
	times = np.linspace(0.0, horizon, n_steps + 1)
	paths = np.empty((n_paths, n_assets, n_steps + 1), dtype=float)
	paths[:, :, 0] = S0s[np.newaxis, :]

	cov = np.outer(sigmas, sigmas) * corr_matrix
	L = np.linalg.cholesky(cov)
	log_drift = (mus - 0.5 * sigmas**2) * dt

	for step in range(1, n_steps + 1):
		z_indep = rng.standard_normal((n_assets, n_paths))
		z_corr = L @ z_indep
		diffusion = z_corr.T * sqrt_dt
		paths[:, :, step] = paths[:, :, step - 1] * np.exp(log_drift + diffusion)

	return times, paths

