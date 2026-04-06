from __future__ import annotations

import math

import numpy as np

from models import HestonParams
from options.black_scholes import black_scholes_implied_vol


def _heston_characteristic_function(
	phi: np.ndarray,
	S0: float,
	r: float,
	T: float,
	params: HestonParams,
	j: int,
) -> np.ndarray:
	i = 1j
	x = math.log(S0)
	kappa = params.kappa
	theta = params.theta
	sigma = params.sigma
	rho = params.rho
	v0 = params.v0
	a = kappa * theta

	if j == 1:
		u = 0.5
		b = kappa - rho * sigma
	else:
		u = -0.5
		b = kappa

	phi = np.asarray(phi, dtype=np.complex128)
	d = np.sqrt((rho * sigma * i * phi - b) ** 2 - sigma**2 * (2.0 * u * i * phi - phi**2))
	g = (b - rho * sigma * i * phi + d) / (b - rho * sigma * i * phi - d)
	exp_dt = np.exp(d * T)

	c = (
		i * phi * r * T
		+ (a / sigma**2)
		* ((b - rho * sigma * i * phi + d) * T - 2.0 * np.log((1.0 - g * exp_dt) / (1.0 - g)))
	)
	D = ((b - rho * sigma * i * phi + d) / sigma**2) * ((1.0 - exp_dt) / (1.0 - g * exp_dt))

	return np.exp(c + D * v0 + i * phi * x)


def _heston_probability(
	S0: float,
	K: float,
	T: float,
	r: float,
	params: HestonParams,
	j: int,
	n_integration: int = 2000,
	upper_limit: float = 250.0,
) -> float:
	lnK = math.log(K)
	phi = np.linspace(1e-6, upper_limit, n_integration)
	cf = _heston_characteristic_function(phi, S0, r, T, params, j)
	integrand = np.real(np.exp(-1j * phi * lnK) * cf / (1j * phi))
	integral = np.trapezoid(integrand, phi)
	return 0.5 + integral / math.pi


def heston_call_price(
	S0: float,
	K: float,
	T: float,
	r: float,
	params: HestonParams,
	n_integration: int = 2000,
	upper_limit: float = 250.0,
) -> float:
	if S0 <= 0 or K <= 0:
		raise ValueError("S0 and K must be strictly positive.")
	if T <= 0:
		return float(max(S0 - K, 0.0))
	if params.sigma < 0 or params.v0 < 0:
		raise ValueError("Heston sigma and v0 must be non-negative.")

	p1 = _heston_probability(S0=S0, K=K, T=T, r=r, params=params, j=1, n_integration=n_integration, upper_limit=upper_limit)
	p2 = _heston_probability(S0=S0, K=K, T=T, r=r, params=params, j=2, n_integration=n_integration, upper_limit=upper_limit)
	return float(S0 * p1 - K * math.exp(-r * T) * p2)


def heston_put_price(
	S0: float,
	K: float,
	T: float,
	r: float,
	params: HestonParams,
	n_integration: int = 2000,
	upper_limit: float = 250.0,
) -> float:
	call = heston_call_price(
		S0=S0,
		K=K,
		T=T,
		r=r,
		params=params,
		n_integration=n_integration,
		upper_limit=upper_limit,
	)
	return float(call - S0 + K * math.exp(-r * T))


def heston_implied_vol_surface(
	S0: float,
	r: float,
	params: HestonParams,
	strikes: np.ndarray,
	maturities: np.ndarray,
	n_integration: int = 2000,
	upper_limit: float = 250.0,
) -> np.ndarray:
	strikes = np.asarray(strikes, dtype=float)
	maturities = np.asarray(maturities, dtype=float)
	surface = np.empty((len(maturities), len(strikes)), dtype=float)

	for i, T in enumerate(maturities):
		for j, K in enumerate(strikes):
			price = heston_call_price(
				S0=S0,
				K=float(K),
				T=float(T),
				r=r,
				params=params,
				n_integration=n_integration,
				upper_limit=upper_limit,
			)
			surface[i, j] = black_scholes_implied_vol(price, S0, float(K), float(T), r)

	return surface