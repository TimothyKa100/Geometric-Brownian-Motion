"""
Phase 3b — Barrier Options (Path-Dependent Payoffs)
====================================================

Barrier options are knocked in or out depending on whether the path
crosses a barrier level during the option's life. No general closed
form exists for arithmetic barriers — MC with full path simulation
is the natural approach.

Covered:
  1. Down-and-out call      (knocked out if S drops below barrier)
  2. Down-and-in  call      (activated only if S drops below barrier)
  3. Up-and-out   call      (knocked out if S rises above barrier)
  4. Up-and-in    call      (activated only if S rises above barrier)
  5. Knock-in + Knock-out parity check  (should equal European)
  6. Price sensitivity to barrier level
  7. Continuous vs discrete monitoring comparison
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
# 1.  Black-Scholes reference
# =============================================================================

def bs_call(S0, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return max(S0 - K, 0.0)
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return float(S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2))


# =============================================================================
# 2.  Generic barrier pricer
# =============================================================================

# NEW
def mc_barrier_call(
    S0: float, K: float, T: float, r: float, sigma: float,
    barrier: float, barrier_type: str,
    n_paths: int = 80_000, n_steps: int = 252, seed: int = 0,
    use_bgk_correction: bool = True,
) -> dict:
    """
    Price a barrier call option via Monte Carlo.

    Discrete monitoring underestimates knock-out probability vs continuous
    monitoring. When use_bgk_correction=True (default), applies the
    Broadie-Glasserman-Kou (1997) continuity correction, which shifts the
    barrier by exp(±β σ √dt) where β ≈ 0.5826, giving O(dt) accuracy
    instead of O(√dt) for the discrete-monitoring bias.

    barrier_type options:
        'down-and-out', 'down-and-in', 'up-and-out', 'up-and-in'
    """
    params = GBMParams(mu=r, sigma=sigma)
    _, paths = euler_maruyama_gbm(S0, params, T, n_steps, n_paths, seed=seed)

    dt = T / n_steps

    # BGK continuity correction: shift barrier to approximate continuous monitoring
    BETA = 0.5826  # = -zeta(1/2) / sqrt(2*pi), Broadie-Glasserman-Kou 1997
    if use_bgk_correction:
        shift = np.exp(BETA * sigma * np.sqrt(dt))
        if barrier_type in ("down-and-out", "down-and-in"):
            effective_barrier = barrier * shift   # shift barrier UP for down-crossings
        else:
            effective_barrier = barrier / shift   # shift barrier DOWN for up-crossings
    else:
        effective_barrier = barrier

    path_min = paths.min(axis=1)
    path_max = paths.max(axis=1)
    S_T      = paths[:, -1]
    discount = np.exp(-r * T)

    vanilla_payoff = np.maximum(S_T - K, 0.0)

    if barrier_type == "down-and-out":
        alive = path_min > effective_barrier
    elif barrier_type == "down-and-in":
        alive = path_min <= effective_barrier
    elif barrier_type == "up-and-out":
        alive = path_max < effective_barrier
    elif barrier_type == "up-and-in":
        alive = path_max >= effective_barrier
    else:
        raise ValueError(f"Unknown barrier_type: {barrier_type}")

    payoffs = vanilla_payoff * alive.astype(float)
    price   = discount * payoffs.mean()
    se      = discount * payoffs.std() / np.sqrt(n_paths)
    hit_prob = alive.mean()

    return {
        "price":              price,
        "se":                 se,
        "ci":                 (price - 1.96*se, price + 1.96*se),
        "hit_prob":           float(hit_prob),
        "barrier_type":       barrier_type,
        "effective_barrier":  effective_barrier,
        "bgk_correction":     use_bgk_correction,
    }


# =============================================================================
# 3.  Parity check: knock-in + knock-out = European
# =============================================================================

def parity_check(S0, K, T, r, sigma, barrier, direction="down",
                 n_paths=80_000, seed=0):
    """
    Knock-in price + Knock-out price == European call price.
    Holds exactly in continuous monitoring, approximately with discrete.
    """
    if direction == "down":
        res_in  = mc_barrier_call(S0, K, T, r, sigma, barrier, "down-and-in",  n_paths=n_paths, seed=seed)
        res_out = mc_barrier_call(S0, K, T, r, sigma, barrier, "down-and-out", n_paths=n_paths, seed=seed)
    else:
        res_in  = mc_barrier_call(S0, K, T, r, sigma, barrier, "up-and-in",  n_paths=n_paths, seed=seed)
        res_out = mc_barrier_call(S0, K, T, r, sigma, barrier, "up-and-out", n_paths=n_paths, seed=seed)

    euro    = bs_call(S0, K, T, r, sigma)
    combined = res_in["price"] + res_out["price"]

    return {
        "knock_in":  res_in["price"],
        "knock_out": res_out["price"],
        "combined":  combined,
        "european":  euro,
        "parity_error": abs(combined - euro),
    }


# =============================================================================
# 4.  Sensitivity: price vs barrier level
# =============================================================================

def barrier_sensitivity(
    S0, K, T, r, sigma,
    barriers: np.ndarray, barrier_type: str,
    n_paths: int = 40_000, seed: int = 0,
) -> dict:
    prices, ses = [], []
    for B in barriers:
        res = mc_barrier_call(S0, K, T, r, sigma, B, barrier_type,
                              n_paths=n_paths, seed=seed)
        prices.append(res["price"])
        ses.append(res["se"])
    return {
        "barriers": barriers,
        "prices":   np.array(prices),
        "ses":      np.array(ses),
        "barrier_type": barrier_type,
        "european": bs_call(S0, K, T, r, sigma),
    }


# =============================================================================
# 5.  Discrete vs continuous monitoring
# =============================================================================

def monitoring_frequency_study(
    S0, K, T, r, sigma, barrier, barrier_type,
    step_counts: list[int], n_paths: int = 40_000, seed: int = 0,
) -> dict:
    """
    Shows how price changes with monitoring frequency.
    More steps → closer to continuous monitoring.
    """
    prices, ses = [], []
    for n in step_counts:
        res = mc_barrier_call(S0, K, T, r, sigma, barrier, barrier_type,
                              n_paths=n_paths, n_steps=n, seed=seed)
        prices.append(res["price"])
        ses.append(res["se"])
    return {
        "step_counts": step_counts,
        "prices":      np.array(prices),
        "ses":         np.array(ses),
    }


# =============================================================================
# 6.  Plots
# =============================================================================

def plot_all(S0, K, T, r, sigma):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(15, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    down_barrier = 85.0
    up_barrier   = 120.0
    euro_price   = bs_call(S0, K, T, r, sigma)

    # ── A: Sample paths with barrier ────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    params = GBMParams(mu=r, sigma=sigma)
    from models import euler_maruyama_gbm
    times, paths = euler_maruyama_gbm(S0, params, T, 252, n_paths=60, seed=3)
    knocked_out = paths.min(axis=1) <= down_barrier
    for i in range(len(paths)):
        color = "#BBBBBB" if knocked_out[i] else "#3C3489"
        alpha = 0.3      if knocked_out[i] else 0.6
        ax_a.plot(times, paths[i], lw=0.7, alpha=alpha, color=color)
    ax_a.axhline(down_barrier, color="#E85D24", lw=1.5, ls="--", label=f"Barrier={down_barrier}")
    ax_a.axhline(K, color="#0F6E56", lw=1, ls=":", label=f"Strike K={K}")
    ax_a.set_title("Down-and-out paths\n(gray = knocked out)")
    ax_a.set_xlabel("Time"); ax_a.set_ylabel("S_t")
    ax_a.legend(fontsize=8); ax_a.grid(True, alpha=0.2)

    # ── B: All 4 barrier types — price vs strike ─────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    strikes = np.arange(80, 130, 5, dtype=float)
    barrier_types = [
        ("down-and-out", "#3C3489", down_barrier),
        ("down-and-in",  "#0F6E56", down_barrier),
        ("up-and-out",   "#E85D24", up_barrier),
        ("up-and-in",    "#BA7517", up_barrier),
    ]
    for btype, color, barr in barrier_types:
        prices_k = []
        for Kk in strikes:
            res = mc_barrier_call(S0, Kk, T, r, sigma, barr, btype,
                                  n_paths=20_000, seed=0)
            prices_k.append(res["price"])
        ax_b.plot(strikes, prices_k, "o-", color=color, lw=1.5, ms=4, label=btype)
    bs_line = [bs_call(S0, Kk, T, r, sigma) for Kk in strikes]
    ax_b.plot(strikes, bs_line, "k--", lw=1.5, label="European (BS)")
    ax_b.axvline(S0, color="gray", lw=1, ls=":")
    ax_b.set_title("All barrier types vs strike")
    ax_b.set_xlabel("Strike K"); ax_b.set_ylabel("Price")
    ax_b.legend(fontsize=7); ax_b.grid(True, alpha=0.25)

    # ── C: Sensitivity — price vs barrier level ──────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    barriers_down = np.arange(60, 100, 3, dtype=float)
    sens_dao = barrier_sensitivity(S0, K, T, r, sigma, barriers_down,
                                   "down-and-out", n_paths=20_000)
    sens_dai = barrier_sensitivity(S0, K, T, r, sigma, barriers_down,
                                   "down-and-in",  n_paths=20_000)
    ax_c.plot(sens_dao["barriers"], sens_dao["prices"], "o-", color="#3C3489",
              lw=2, label="Down-and-out")
    ax_c.plot(sens_dai["barriers"], sens_dai["prices"], "s-", color="#0F6E56",
              lw=2, label="Down-and-in")
    ax_c.axhline(euro_price, color="k", lw=1.2, ls="--",
                 label=f"European = {euro_price:.2f}")
    ax_c.axvline(S0, color="gray", lw=1, ls=":", label=f"S₀={S0}")
    ax_c.set_title("Price sensitivity to barrier")
    ax_c.set_xlabel("Down barrier level"); ax_c.set_ylabel("Price")
    ax_c.legend(fontsize=8); ax_c.grid(True, alpha=0.25)

    # ── D: Parity check (in + out = European) ───────────────────────────────
    ax_d = fig.add_subplot(gs[1, 0])
    barriers_test = np.arange(65, 98, 4, dtype=float)
    parity_errors, combined_prices = [], []
    for B in barriers_test:
        pc = parity_check(S0, K, T, r, sigma, B, direction="down", n_paths=30_000)
        parity_errors.append(pc["parity_error"])
        combined_prices.append(pc["combined"])
    ax_d.bar(barriers_test, parity_errors, width=3, color="#E85D24", alpha=0.7)
    ax_d.set_title("Parity error: in+out vs European\n(BGK-corrected; residual = discretisation noise)")
    ax_d.set_xlabel("Barrier level"); ax_d.set_ylabel("|combined − European|")
    ax_d.grid(True, alpha=0.25)

    # ── E: Monitoring frequency ──────────────────────────────────────────────
    ax_e = fig.add_subplot(gs[1, 1])
    step_counts = [5, 10, 21, 42, 63, 126, 252, 504]
    mon = monitoring_frequency_study(S0, K, T, r, sigma,
                                     down_barrier, "down-and-out",
                                     step_counts, n_paths=40_000)
    ax_e.semilogx(mon["step_counts"], mon["prices"], "o-",
                  color="#3C3489", lw=2, label="MC price")
    ax_e.fill_between(mon["step_counts"],
                      mon["prices"] - 1.96*mon["ses"],
                      mon["prices"] + 1.96*mon["ses"],
                      alpha=0.15, color="#3C3489")
    ax_e.set_title("Discrete vs continuous monitoring\n(down-and-out)")
    ax_e.set_xlabel("Monitoring steps per year (log)"); ax_e.set_ylabel("Price")
    ax_e.legend(fontsize=8); ax_e.grid(True, alpha=0.25)

    # ── F: Up-barrier sensitivity ────────────────────────────────────────────
    ax_f = fig.add_subplot(gs[1, 2])
    barriers_up = np.arange(105, 145, 4, dtype=float)
    sens_uao = barrier_sensitivity(S0, K, T, r, sigma, barriers_up,
                                   "up-and-out", n_paths=20_000)
    sens_uai = barrier_sensitivity(S0, K, T, r, sigma, barriers_up,
                                   "up-and-in",  n_paths=20_000)
    ax_f.plot(sens_uao["barriers"], sens_uao["prices"], "o-", color="#E85D24",
              lw=2, label="Up-and-out")
    ax_f.plot(sens_uai["barriers"], sens_uai["prices"], "s-", color="#BA7517",
              lw=2, label="Up-and-in")
    ax_f.axhline(euro_price, color="k", lw=1.2, ls="--",
                 label=f"European = {euro_price:.2f}")
    ax_f.axvline(S0, color="gray", lw=1, ls=":")
    ax_f.set_title("Price sensitivity — up barriers")
    ax_f.set_xlabel("Up barrier level"); ax_f.set_ylabel("Price")
    ax_f.legend(fontsize=8); ax_f.grid(True, alpha=0.25)

    fig.suptitle("Phase 3b — Barrier Options (Path-Dependent Payoffs)", fontsize=14)
    plt.savefig("results/phase3b_barrier_options.png", dpi=140, bbox_inches="tight")
    print("Saved → results/phase3b_barrier_options.png")
    plt.show()


# =============================================================================
# 7.  Main
# =============================================================================

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    S0    = 100.0
    K     = 100.0
    T     = 1.0
    r     = 0.05
    sigma = 0.20

    down_B = 85.0
    up_B   = 120.0

    print("=" * 55)
    print("Phase 3b — Barrier Options")
    print("=" * 55)
    print(f"S0={S0}, K={K}, T={T}y, r={r:.0%}, sigma={sigma:.0%}")
    print(f"Down barrier={down_B}, Up barrier={up_B}\n")

    euro = bs_call(S0, K, T, r, sigma)
    print(f"European call (BS) = {euro:.4f}\n")

    types = [
        ("down-and-out", down_B),
        ("down-and-in",  down_B),
        ("up-and-out",   up_B),
        ("up-and-in",    up_B),
    ]
    for btype, barr in types:
        res = mc_barrier_call(S0, K, T, r, sigma, barr, btype, n_paths=80_000)
        print(f"  {btype:<16} barrier={barr:5.0f} | "
              f"price={res['price']:.4f}  ±{1.96*res['se']:.4f}  "
              f"hit_prob={res['hit_prob']:.3f}")

    print("\n── Parity check (down-and-in + down-and-out) ──")
    pc = parity_check(S0, K, T, r, sigma, down_B, n_paths=80_000)
    print(f"  Knock-in  = {pc['knock_in']:.4f}")
    print(f"  Knock-out = {pc['knock_out']:.4f}")
    print(f"  Combined  = {pc['combined']:.4f}")
    print(f"  European  = {pc['european']:.4f}")
    print(f"  Parity error = {pc['parity_error']:.5f}  (should be ≈ 0)")

    plot_all(S0, K, T, r, sigma)