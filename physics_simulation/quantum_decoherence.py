from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from models import GBMParams, exact_step_gbm


Array = np.ndarray


@dataclass(frozen=True)
class QuantumGBMDecoherenceParams:
	"""Parameters for a two-level dephasing model driven by GBM log-noise.

	The off-diagonal density matrix element is modeled as:
		c_t = c_0 * exp(-gamma_phi * t) * exp(i * phi_t)
	where phi_t is coupled to log-price increments.
	"""

	dephasing_rate: float
	coupling: float
	excited_population: float = 0.5
	visibility: float = 1.0
	phase0: float = 0.0
	carrier_frequency: float = 0.0


def theoretical_coherence_envelope(
	params: QuantumGBMDecoherenceParams,
	gbm_sigma: float,
	times: Array,
) -> Array:
	"""Return |E[c_t]| under GBM-coupled phase diffusion.

	If dphi_t includes coupling * sigma dW_t, then
		|E[c_t]| = |c_0| * exp(-(gamma_phi + 0.5 * coupling^2 * sigma^2) * t).
	"""
	t = np.asarray(times, dtype=float)
	c0_max = np.sqrt(params.excited_population * (1.0 - params.excited_population))
	c0 = params.visibility * c0_max
	rate = params.dephasing_rate + 0.5 * (params.coupling**2) * (gbm_sigma**2)
	return np.abs(c0) * np.exp(-rate * t)


def simulate_quantum_decoherence_gbm(
	s0: float,
	gbm_params: GBMParams,
	q_params: QuantumGBMDecoherenceParams,
	horizon: float,
	n_steps: int,
	n_paths: int = 1,
	seed: Optional[int] = None,
) -> tuple[Array, Array, Array, Array, Array, Array]:
	"""Simulate GBM-driven dephasing and return ensemble decoherence metrics.

	Returns:
		times,
		price_paths,
		coherence_paths,
		ensemble_coherence,
		ensemble_purity,
		theoretical_envelope
	"""
	if s0 <= 0:
		raise ValueError("s0 must be strictly positive.")
	if gbm_params.sigma < 0:
		raise ValueError("GBM sigma must be non-negative.")
	if q_params.dephasing_rate < 0:
		raise ValueError("dephasing_rate must be non-negative.")
	if not (0.0 <= q_params.excited_population <= 1.0):
		raise ValueError("excited_population must be in [0, 1].")
	if not (0.0 <= q_params.visibility <= 1.0):
		raise ValueError("visibility must be in [0, 1].")
	if n_steps <= 0 or n_paths <= 0:
		raise ValueError("n_steps and n_paths must be positive.")

	times, price_paths = exact_step_gbm(
		s0=s0,
		params=gbm_params,
		horizon=horizon,
		n_steps=n_steps,
		n_paths=n_paths,
		seed=seed,
	)

	log_prices = np.log(price_paths)
	dlog = np.diff(log_prices, axis=1)
	phase_noise = q_params.coupling * np.cumsum(dlog, axis=1)

	p = q_params.excited_population
	c0_max = np.sqrt(p * (1.0 - p))
	c0 = q_params.visibility * c0_max

	coherence_paths = np.empty((n_paths, n_steps + 1), dtype=np.complex128)
	coherence_paths[:, 0] = c0 * np.exp(1j * q_params.phase0)

	for i in range(1, n_steps + 1):
		t = times[i]
		deterministic_phase = q_params.phase0 - q_params.carrier_frequency * t
		coherence_paths[:, i] = c0 * np.exp(-q_params.dephasing_rate * t) * np.exp(
			1j * (deterministic_phase + phase_noise[:, i - 1])
		)

	ensemble_coherence = np.mean(coherence_paths, axis=0)
	ensemble_purity = p**2 + (1.0 - p) ** 2 + 2.0 * np.abs(ensemble_coherence) ** 2
	theory_envelope = theoretical_coherence_envelope(q_params, gbm_params.sigma, times)

	return times, price_paths, coherence_paths, ensemble_coherence, ensemble_purity, theory_envelope