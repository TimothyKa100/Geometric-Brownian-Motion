from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from models import OUParams, euler_maruyama_ou


Array = np.ndarray


@dataclass(frozen=True)
class LangevinParams:
	"""Parameters for 1D Langevin dynamics with OU velocity.

	Velocity SDE:
		dV_t = -gamma * V_t dt + sigma dW_t
	Position update:
		dX_t = V_t dt
	"""

	gamma: float
	sigma: float


def simulate_langevin_1d(
	x0: float,
	v0: float,
	params: LangevinParams,
	horizon: float,
	n_steps: int,
	n_paths: int = 1,
	seed: Optional[int] = None,
) -> tuple[Array, Array, Array]:
	"""Simulate 1D Langevin particle paths (position and velocity)."""
	if params.gamma < 0:
		raise ValueError("gamma must be non-negative.")
	if params.sigma < 0:
		raise ValueError("sigma must be non-negative.")
	if n_steps <= 0 or n_paths <= 0:
		raise ValueError("n_steps and n_paths must be positive.")

	ou_params = OUParams(theta=params.gamma, mu=0.0, sigma=params.sigma)
	times, velocities = euler_maruyama_ou(
		x0=v0,
		params=ou_params,
		horizon=horizon,
		n_steps=n_steps,
		n_paths=n_paths,
		seed=seed,
	)

	dt = horizon / n_steps
	positions = np.empty_like(velocities)
	positions[:, 0] = x0
	positions[:, 1:] = x0 + np.cumsum(velocities[:, :-1] * dt, axis=1)
	return times, positions, velocities


def velocity_theoretical_moments(v0: float, params: LangevinParams, times: Array) -> tuple[Array, Array]:
	"""Return exact mean/variance of OU velocity at each time."""
	if params.gamma < 0:
		raise ValueError("gamma must be non-negative.")
	if params.sigma < 0:
		raise ValueError("sigma must be non-negative.")

	gamma = params.gamma
	sigma = params.sigma
	t = np.asarray(times, dtype=float)

	if gamma == 0.0:
		mean = np.full_like(t, v0, dtype=float)
		var = sigma**2 * t
		return mean, var

	mean = v0 * np.exp(-gamma * t)
	var = (sigma**2 / (2.0 * gamma)) * (1.0 - np.exp(-2.0 * gamma * t))
	return mean, var
