from __future__ import annotations

import math

from models import black_scholes_state, normal_cdf


def black_scholes_call_price(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European call option price."""
    if S0 <= 0 or K <= 0:
        raise ValueError("S0 and K must be strictly positive.")
    if T <= 0 or sigma <= 0:
        return max(S0 - K, 0.0)

    state = black_scholes_state(S0=S0, K=K, T=T, r=r, sigma=sigma)
    return float(S0 * normal_cdf(state.d1) - K * math.exp(-r * T) * normal_cdf(state.d2))


def black_scholes_put_price(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European put option price via put-call parity."""
    return float(black_scholes_call_price(S0, K, T, r, sigma) - S0 + K * math.exp(-r * T))


def black_scholes_digital_call_price(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes digital call (cash-or-nothing, unit payoff)."""
    if S0 <= 0 or K <= 0:
        raise ValueError("S0 and K must be strictly positive.")
    if T <= 0 or sigma <= 0:
        return math.exp(-r * max(T, 0.0)) * float(S0 > K)

    state = black_scholes_state(S0=S0, K=K, T=T, r=r, sigma=sigma)
    return float(math.exp(-r * T) * normal_cdf(state.d2))


def black_scholes_greeks(S0: float, K: float, T: float, r: float, sigma: float) -> dict[str, float]:
    """Black-Scholes Greeks for call and put options."""
    if S0 <= 0 or K <= 0:
        raise ValueError("S0 and K must be strictly positive.")
    if T <= 0 or sigma <= 0:
        call_delta = 1.0 if S0 > K else 0.0
        put_delta = call_delta - 1.0
        return {
            "call_delta": call_delta,
            "put_delta": put_delta,
            "gamma": 0.0,
            "vega": 0.0,
            "call_theta": 0.0,
            "put_theta": 0.0,
            "call_rho": 0.0,
            "put_rho": 0.0,
        }

    sqrt_t = math.sqrt(T)
    state = black_scholes_state(S0=S0, K=K, T=T, r=r, sigma=sigma)
    phi_d1 = math.exp(-0.5 * state.d1**2) / math.sqrt(2.0 * math.pi)

    call_delta = normal_cdf(state.d1)
    put_delta = call_delta - 1.0
    gamma = phi_d1 / (S0 * sigma * sqrt_t)
    vega = S0 * phi_d1 * sqrt_t
    call_theta = -(S0 * phi_d1 * sigma) / (2.0 * sqrt_t) - r * K * math.exp(-r * T) * normal_cdf(state.d2)
    put_theta = -(S0 * phi_d1 * sigma) / (2.0 * sqrt_t) + r * K * math.exp(-r * T) * normal_cdf(-state.d2)
    call_rho = K * T * math.exp(-r * T) * normal_cdf(state.d2)
    put_rho = -K * T * math.exp(-r * T) * normal_cdf(-state.d2)

    return {
        "call_delta": float(call_delta),
        "put_delta": float(put_delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "call_theta": float(call_theta),
        "put_theta": float(put_theta),
        "call_rho": float(call_rho),
        "put_rho": float(put_rho),
    }
