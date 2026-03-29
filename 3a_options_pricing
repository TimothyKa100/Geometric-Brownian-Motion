"""
Phase 3a — Options Pricing: Monte Carlo vs Black-Scholes
=========================================================

Prices European and Asian options using GBM Monte Carlo simulation,
then benchmarks against closed-form Black-Scholes.

Covered:
  1. European call/put  — MC vs BS analytical formula
  2. Asian (arithmetic average) call — MC only (no closed form)
  3. Implied volatility extraction from MC prices
  4. Price vs Strike table (option chain)
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
from models import GBMParams, euler_maruyama_gbm


# =============================================================================
# 1.  Black-Scholes closed forms
# =============================================================================

def bs_call(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European call price."""
    if T <= 0 or sigma <= 0:
        return max(S0 - K, 0.0)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def bs_put(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European put price  (put-call parity)."""
    return float(bs_call(S0, K, T, r, sigma) - S0 + K * np.exp(-r * T))


def bs_delta(S0: float, K: float, T: float, r: float, sigma: float,
             option: str = "call") -> float:
    """BS delta."""
    if T <= 0 or sigma <= 0:
        return 1.0 if (option == "call" and S0 > K) else 0.0
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return float(norm.cdf(d1) if option == "call" else norm.cdf(d1) - 1)


# =============================================================================
# 2.  Monte Carlo pricers
# =============================================================================

def mc_european(
    S0: float, K: float, T: float, r: float, sigma: float,
    n_paths: int = 50_000, n_steps: int = 252, seed: int = 0,
) -> dict:
    """
    MC price for European call and put.
    Uses risk-neutral drift r (not physical mu).
    Returns price, std error, and 95% CI for both call and put.
    """
    params = GBMParams(mu=r, sigma=sigma)
    _, paths = euler_maruyama_gbm(S0, params, T, n_steps, n_paths, seed=seed)
    S_T = paths[:, -1]
    discount = np.exp(-r * T)

    call_payoffs = np.maximum(S_T - K, 0.0)
    put_payoffs  = np.maximum(K - S_T, 0.0)

    call_price = discount * call_payoffs.mean()
    put_price  = discount * put_payoffs.mean()
    call_se    = discount * call_payoffs.std() / np.sqrt(n_paths)
    put_se     = discount * put_payoffs.std()  / np.sqrt(n_paths)

    return {
        "call_price": call_price, "call_se": call_se,
        "call_ci":    (call_price - 1.96 * call_se, call_price + 1.96 * call_se),
        "put_price":  put_price,  "put_se":  put_se,
        "put_ci":     (put_price  - 1.96 * put_se,  put_price  + 1.96 * put_se),
        "n_paths": n_paths,
    }


def mc_asian_call(
    S0: float, K: float, T: float, r: float, sigma: float,
    n_paths: int = 50_000, n_steps: int = 252, seed: int = 0,
) -> dict:
    """
    Arithmetic-average Asian call  E[max(mean(S) - K, 0)] * discount.
    No closed form exists — pure MC.
    """
    params = GBMParams(mu=r, sigma=sigma)
    _, paths = euler_maruyama_gbm(S0, params, T, n_steps, n_paths, seed=seed)

    # Arithmetic average over the path (excluding t=0)
    avg_S = paths[:, 1:].mean(axis=1)
    payoffs = np.maximum(avg_S - K, 0.0)
    discount = np.exp(-r * T)

    price = discount * payoffs.mean()
    se    = discount * payoffs.std() / np.sqrt(n_paths)

    return {
        "asian_call_price": price,
        "asian_call_se":    se,
        "asian_call_ci":    (price - 1.96 * se, price + 1.96 * se),
    }


def mc_digital_call(
    S0: float, K: float, T: float, r: float, sigma: float,
    n_paths: int = 50_000, n_steps: int = 252, seed: int = 0,
) -> dict:
    """Digital (binary) call: pays $1 if S_T > K."""
    params = GBMParams(mu=r, sigma=sigma)
    _, paths = euler_maruyama_gbm(S0, params, T, n_steps, n_paths, seed=seed)
    S_T = paths[:, -1]
    discount = np.exp(-r * T)

    payoffs = (S_T > K).astype(float)
    # BS closed form: discount * N(d2)
    if T > 0 and sigma > 0:
        d2 = (np.log(S0 / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        bs_price = discount * norm.cdf(d2)
    else:
        bs_price = discount * float(S0 > K)

    price = discount * payoffs.mean()
    se    = discount * payoffs.std() / np.sqrt(n_paths)

    return {
        "digital_mc":  price,
        "digital_se":  se,
        "digital_bs":  bs_price,
    }


# =============================================================================
# 3.  Option chain: price vs strike
# =============================================================================

def option_chain(
    S0: float, T: float, r: float, sigma: float,
    strikes: np.ndarray, n_paths: int = 30_000, seed: int = 0,
) -> dict:
    """Compute MC and BS prices across a range of strikes."""
    params = GBMParams(mu=r, sigma=sigma)
    _, paths = euler_maruyama_gbm(S0, params, T, 252, n_paths, seed=seed)
    S_T = paths[:, -1]
    discount = np.exp(-r * T)

    mc_calls, mc_puts, bs_calls, bs_puts = [], [], [], []
    for K in strikes:
        mc_calls.append(discount * np.maximum(S_T - K, 0).mean())
        mc_puts.append( discount * np.maximum(K - S_T, 0).mean())
        bs_calls.append(bs_call(S0, K, T, r, sigma))
        bs_puts.append( bs_put( S0, K, T, r, sigma))

    return {
        "strikes":   strikes,
        "mc_calls":  np.array(mc_calls),
        "mc_puts":   np.array(mc_puts),
        "bs_calls":  np.array(bs_calls),
        "bs_puts":   np.array(bs_puts),
    }


# =============================================================================
# 4.  Convergence: MC price vs n_paths
# =============================================================================

def pricing_convergence(
    S0: float, K: float, T: float, r: float, sigma: float,
    path_counts: list[int], seed: int = 0,
) -> dict:
    """MC call price as a function of number of paths — shows convergence to BS."""
    bs = bs_call(S0, K, T, r, sigma)
    prices, errors, ses = [], [], []

    for n in path_counts:
        res = mc_european(S0, K, T, r, sigma, n_paths=n, seed=seed)
        prices.append(res["call_price"])
        ses.append(res["call_se"])
        errors.append(abs(res["call_price"] - bs))

    return {
        "n_paths": path_counts,
        "mc_prices": prices,
        "ses": ses,
        "errors": errors,
        "bs_price": bs,
    }


# =============================================================================
# 5.  Plots
# =============================================================================

def plot_all(S0, K, T, r, sigma, n_paths=50_000, seed=0):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(15, 11))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

    # ── A: Payoff diagram ───────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    S_range = np.linspace(60, 160, 300)
    ax_a.plot(S_range, np.maximum(S_range - K, 0), color="#3C3489", lw=2, label="Call payoff")
    ax_a.plot(S_range, np.maximum(K - S_range, 0), color="#0F6E56", lw=2, label="Put payoff")
    ax_a.axvline(K, color="gray", lw=1, ls="--", label=f"Strike K={K}")
    ax_a.axvline(S0, color="#BA7517", lw=1, ls=":", label=f"S₀={S0}")
    ax_a.set_title("Payoff at expiry")
    ax_a.set_xlabel("S_T"); ax_a.set_ylabel("Payoff")
    ax_a.legend(fontsize=8); ax_a.grid(True, alpha=0.25)

    # ── B: Option price vs sigma ─────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    sigmas = np.linspace(0.05, 0.60, 60)
    bs_c = [bs_call(S0, K, T, r, s) for s in sigmas]
    bs_p = [bs_put( S0, K, T, r, s) for s in sigmas]
    ax_b.plot(sigmas, bs_c, color="#3C3489", lw=2, label="Call (BS)")
    ax_b.plot(sigmas, bs_p, color="#0F6E56", lw=2, label="Put (BS)")
    ax_b.axvline(sigma, color="#BA7517", lw=1, ls="--", label=f"σ={sigma}")
    ax_b.set_title("Price vs volatility (vega)")
    ax_b.set_xlabel("σ"); ax_b.set_ylabel("Option price")
    ax_b.legend(fontsize=8); ax_b.grid(True, alpha=0.25)

    # ── C: Option chain (call prices) ──────────────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    strikes = np.arange(70, 145, 5, dtype=float)
    chain = option_chain(S0, T, r, sigma, strikes, n_paths=30_000, seed=seed)
    ax_c.plot(chain["strikes"], chain["bs_calls"],  "o-", color="#3C3489", lw=2, label="BS call")
    ax_c.plot(chain["strikes"], chain["mc_calls"], "x--", color="#E85D24", lw=1.5, ms=6, label="MC call")
    ax_c.plot(chain["strikes"], chain["bs_puts"],  "s-", color="#0F6E56", lw=2, label="BS put")
    ax_c.plot(chain["strikes"], chain["mc_puts"], "+--", color="#BA7517", lw=1.5, ms=6, label="MC put")
    ax_c.axvline(S0, color="gray", lw=1, ls=":")
    ax_c.set_title("Option chain: MC vs BS")
    ax_c.set_xlabel("Strike K"); ax_c.set_ylabel("Price")
    ax_c.legend(fontsize=8); ax_c.grid(True, alpha=0.25)

    # ── D: MC convergence ───────────────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 0])
    path_counts = [100, 300, 600, 1000, 3000, 6000, 10000, 30000, 50000]
    conv = pricing_convergence(S0, K, T, r, sigma, path_counts, seed=seed)
    ax_d.semilogx(conv["n_paths"], conv["mc_prices"], "o-", color="#3C3489", lw=2, label="MC price")
    ax_d.axhline(conv["bs_price"], color="#0F6E56", lw=1.5, ls="--", label=f"BS = {conv['bs_price']:.3f}")
    ses = np.array(conv["ses"])
    mcp = np.array(conv["mc_prices"])
    ax_d.fill_between(conv["n_paths"], mcp - 1.96*ses, mcp + 1.96*ses, alpha=0.15, color="#3C3489")
    ax_d.set_title("MC call convergence to BS")
    ax_d.set_xlabel("Number of paths (log scale)"); ax_d.set_ylabel("Call price")
    ax_d.legend(fontsize=8); ax_d.grid(True, alpha=0.25)

    # ── E: European vs Asian call prices vs strike ──────────────────────────
    ax_e = fig.add_subplot(gs[1, 1])
    asian_prices = []
    euro_bs      = []
    for Kk in strikes:
        res_a = mc_asian_call(S0, Kk, T, r, sigma, n_paths=30_000, seed=seed)
        asian_prices.append(res_a["asian_call_price"])
        euro_bs.append(bs_call(S0, Kk, T, r, sigma))

    ax_e.plot(strikes, euro_bs,      "o-",  color="#3C3489", lw=2, label="European (BS)")
    ax_e.plot(strikes, asian_prices, "x--", color="#E85D24", lw=2, ms=6, label="Asian arith. (MC)")
    ax_e.axvline(S0, color="gray", lw=1, ls=":", label=f"S₀={S0}")
    ax_e.set_title("European vs Asian call prices")
    ax_e.set_xlabel("Strike K"); ax_e.set_ylabel("Call price")
    ax_e.legend(fontsize=8); ax_e.grid(True, alpha=0.25)

    # ── F: Digital call MC vs BS ─────────────────────────────────────────────
    ax_f = fig.add_subplot(gs[1, 2])
    digital_mc_prices, digital_bs_prices = [], []
    for Kk in strikes:
        res_d = mc_digital_call(S0, Kk, T, r, sigma, n_paths=30_000, seed=seed)
        digital_mc_prices.append(res_d["digital_mc"])
        digital_bs_prices.append(res_d["digital_bs"])

    ax_f.plot(strikes, digital_bs_prices, "o-",  color="#3C3489", lw=2, label="Digital (BS)")
    ax_f.plot(strikes, digital_mc_prices, "x--", color="#E85D24", lw=2, ms=6, label="Digital (MC)")
    ax_f.axvline(S0, color="gray", lw=1, ls=":")
    ax_f.set_title("Digital call: MC vs BS")
    ax_f.set_xlabel("Strike K"); ax_f.set_ylabel("Price ($1 payoff)")
    ax_f.legend(fontsize=8); ax_f.grid(True, alpha=0.25)

    fig.suptitle("Phase 3a — Options Pricing (MC vs Black-Scholes)", fontsize=14)
    plt.savefig("results/phase3a_options.png", dpi=140, bbox_inches="tight")
    print("Saved → results/phase3a_options.png")
    plt.show()


# =============================================================================
# 6.  Main
# =============================================================================

if __name__ == "__main__":
    import os; os.makedirs("results", exist_ok=True)

    S0    = 100.0
    K     = 105.0
    T     = 1.0
    r     = 0.05
    sigma = 0.20

    print("=" * 55)
    print("Phase 3a — Options Pricing")
    print("=" * 55)
    print(f"S0={S0}, K={K}, T={T}y, r={r:.0%}, sigma={sigma:.0%}\n")

    # European
    res = mc_european(S0, K, T, r, sigma, n_paths=50_000)
    print("── European options ──")
    print(f"  BS  call = {bs_call(S0,K,T,r,sigma):.4f}")
    print(f"  MC  call = {res['call_price']:.4f}  ±{1.96*res['call_se']:.4f}  (95% CI)")
    print(f"  BS  put  = {bs_put(S0,K,T,r,sigma):.4f}")
    print(f"  MC  put  = {res['put_price']:.4f}  ±{1.96*res['put_se']:.4f}  (95% CI)")

    # Asian
    res_a = mc_asian_call(S0, K, T, r, sigma, n_paths=50_000)
    print(f"\n── Asian arithmetic call ──")
    print(f"  MC price = {res_a['asian_call_price']:.4f}  ±{1.96*res_a['asian_call_se']:.4f}")
    print(f"  (cheaper than European: averaging reduces payoff variance)")

    # Digital
    res_d = mc_digital_call(S0, K, T, r, sigma, n_paths=50_000)
    print(f"\n── Digital call (pays $1 if S_T > K) ──")
    print(f"  BS  price = {res_d['digital_bs']:.4f}")
    print(f"  MC  price = {res_d['digital_mc']:.4f}  ±{1.96*res_d['digital_se']:.4f}")

    # Convergence
    print("\n── MC convergence (call price vs n_paths) ──")
    conv = pricing_convergence(S0, K, T, r, sigma,
                               [500, 1000, 5000, 10000, 50000])
    print(f"  BS benchmark = {conv['bs_price']:.4f}")
    for n, p, e in zip(conv["n_paths"], conv["mc_prices"], conv["errors"]):
        print(f"  n={n:>6,}: price={p:.4f}, |error|={e:.4f}")

    plot_all(S0, K, T, r, sigma)