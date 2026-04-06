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


def black_scholes_implied_vol(
    price: float,
    S0: float,
    K: float,
    T: float,
    r: float,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """Invert the Black-Scholes call price to implied volatility using bisection."""
    if S0 <= 0 or K <= 0:
        raise ValueError("S0 and K must be strictly positive.")
    if T < 0:
        raise ValueError("T must be non-negative.")

    if T == 0.0:
        return 0.0 if price <= max(S0 - K, 0.0) else 1.0

    intrinsic = max(S0 - K * math.exp(-r * T), 0.0)
    if price <= intrinsic:
        return 0.0

    low = 1e-8
    high = 4.0
    high_price = black_scholes_call_price(S0, K, T, r, high)
    while high_price < price and high < 10.0:
        high *= 2.0
        high_price = black_scholes_call_price(S0, K, T, r, high)

    if price >= high_price:
        return high

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        mid_price = black_scholes_call_price(S0, K, T, r, mid)
        if abs(mid_price - price) < tol:
            return float(mid)
        if mid_price > price:
            high = mid
        else:
            low = mid

    return float(0.5 * (low + high))
