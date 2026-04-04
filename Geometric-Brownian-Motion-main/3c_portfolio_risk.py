"""
Phase 3c — Portfolio Risk: VaR & CVaR with Correlated Multi-Asset GBM
======================================================================

Simulates a portfolio of correlated assets using multi-dimensional GBM.
Correlation structure is introduced via Cholesky decomposition of the
covariance matrix — the standard industry approach.

Covered:
  1. Cholesky decomposition for correlated Brownian motions
  2. Multi-asset GBM simulation
  3. Portfolio P&L distribution at horizon T
  4. Value-at-Risk (VaR) at 95% and 99% confidence
  5. Conditional VaR / Expected Shortfall (CVaR)
  6. Sensitivity: VaR vs correlation, volatility, horizon
  7. Diversification benefit: VaR(portfolio) < sum of individual VaRs
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm


# =============================================================================
# 1.  Correlated multi-asset GBM
# =============================================================================

def simulate_correlated_gbm(
    S0s: np.ndarray,
    mus: np.ndarray,
    sigmas: np.ndarray,
    corr_matrix: np.ndarray,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate n_assets correlated GBM paths via Cholesky decomposition.

    The covariance matrix Σ is decomposed as Σ = L L^T (Cholesky).
    Correlated increments: dW_corr = L @ dW_indep
    where dW_indep are independent standard Brownian increments.

    Parameters
    ----------
    S0s         : Initial prices, shape (n_assets,)
    mus         : Drift vector,   shape (n_assets,)
    sigmas      : Vol vector,     shape (n_assets,)
    corr_matrix : Correlation matrix, shape (n_assets, n_assets)

    Returns
    -------
    times : shape (n_steps+1,)
    paths : shape (n_paths, n_assets, n_steps+1)
    """
    n_assets = len(S0s)
    rng  = np.random.default_rng(seed)
    dt   = T / n_steps
    sqrt_dt = np.sqrt(dt)

    # Covariance matrix and Cholesky factor
    # Cov[i,j] = sigma_i * sigma_j * rho_ij
    cov = np.outer(sigmas, sigmas) * corr_matrix
    L   = np.linalg.cholesky(cov)           # lower-triangular, L @ L.T = Cov

    times = np.linspace(0.0, T, n_steps + 1)
    paths = np.empty((n_paths, n_assets, n_steps + 1))
    paths[:, :, 0] = S0s[np.newaxis, :]    # broadcast S0s across paths

    for step in range(1, n_steps + 1):
        Z_indep = rng.standard_normal((n_assets, n_paths))   # (n_assets, n_paths)
        Z_corr  = L @ Z_indep                                 # (n_assets, n_paths)
        # Each row is the correlated increment for asset i across all paths
        dW = Z_corr.T * sqrt_dt                               # (n_paths, n_assets)

        prev = paths[:, :, step - 1]
        # EM step: S_{t+1} = S_t * exp((mu - 0.5*sigma^2)*dt + dW)
        # (using exact log-step to avoid negativity issues)
        log_drift = (mus - 0.5 * sigmas**2) * dt
        paths[:, :, step] = prev * np.exp(log_drift + dW)

    return times, paths


# =============================================================================
# 2.  Portfolio P&L
# =============================================================================

def portfolio_pnl(paths: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Compute portfolio P&L at terminal time for each path.

    weights : fractional weights summing to 1, shape (n_assets,).
              Interpreted as the fraction of total portfolio value in each asset.
              P&L is expressed as portfolio return (dimensionless), consistent
              with VaR/CVaR being reported as fractions of portfolio value.

    Returns : portfolio return array, shape (n_paths,)
              i.e.  sum_i  w_i * (S_T_i - S0_i) / S0_i
    """
    if not np.isclose(weights.sum(), 1.0, atol=1e-6):
        raise ValueError(
            f"weights must sum to 1 for return-based P&L, got {weights.sum():.6f}. "
            "If using dollar weights, divide by total portfolio value first."
        )
    S_0 = paths[:, :, 0]                       # (n_paths, n_assets)
    S_T = paths[:, :, -1]                       # (n_paths, n_assets)
    asset_returns = (S_T - S_0) / S_0           # simple returns per asset
    pnl = (asset_returns * weights).sum(axis=1) # weighted portfolio return
    return pnl


# =============================================================================
# 3.  Risk measures
# =============================================================================

def var_cvar(pnl: np.ndarray, alpha: float = 0.95) -> dict:
    """
    VaR and CVaR (Expected Shortfall) at confidence level alpha.

    VaR_alpha   = -quantile(PnL, 1-alpha)   (loss as positive number)
    CVaR_alpha  = -E[PnL | PnL <= -VaR]      (average loss in tail)
    """
    var  = -np.quantile(pnl, 1 - alpha)
    tail = pnl[pnl <= -var]
    cvar = -tail.mean() if len(tail) > 0 else var

    return {
        "VaR":   float(var),
        "CVaR":  float(cvar),
        "alpha": alpha,
        "n_tail": len(tail),
    }


def analytical_var_normal(mu_port: float, sigma_port: float,
                           T: float, alpha: float = 0.95) -> float:
    """
    Analytical VaR under the assumption of normally distributed returns.
    Under GBM: log-return is normal, but P&L (arithmetic) is log-normal.
    This is an approximation useful for benchmarking.
    """
    z = norm.ppf(1 - alpha)
    return -(mu_port * T + sigma_port * np.sqrt(T) * z)


# =============================================================================
# 4.  Sensitivity studies
# =============================================================================

def var_vs_correlation(
    S0s, mus, sigmas, weights, T, n_steps, n_paths,
    rho_values: np.ndarray, alpha=0.95, seed=0,
) -> dict:
    """VaR and CVaR as a function of pairwise correlation (2-asset case)."""
    vars_, cvars_ = [], []
    for rho in rho_values:
        corr = np.array([[1.0, rho], [rho, 1.0]])
        _, paths = simulate_correlated_gbm(S0s, mus, sigmas, corr,
                                           T, n_steps, n_paths, seed=seed)
        pnl = portfolio_pnl(paths, weights)
        rm  = var_cvar(pnl, alpha)
        vars_.append(rm["VaR"])
        cvars_.append(rm["CVaR"])
    return {"rho": rho_values, "VaR": np.array(vars_), "CVaR": np.array(cvars_)}


def var_vs_horizon(
    S0s, mus, sigmas, corr, weights,
    horizons: np.ndarray, n_steps_per_year=52,
    n_paths=20_000, alpha=0.95, seed=0,
) -> dict:
    vars_, cvars_ = [], []
    for T in horizons:
        n_steps = max(int(T * n_steps_per_year), 1)
        _, paths = simulate_correlated_gbm(S0s, mus, sigmas, corr,
                                           T, n_steps, n_paths, seed=seed)
        pnl = portfolio_pnl(paths, weights)
        rm  = var_cvar(pnl, alpha)
        vars_.append(rm["VaR"])
        cvars_.append(rm["CVaR"])
    return {"T": horizons, "VaR": np.array(vars_), "CVaR": np.array(cvars_)}


def diversification_benefit(
    S0s, mus, sigmas, corr, weights,
    T, n_steps, n_paths, alpha=0.95, seed=0,
) -> dict:
    """
    Show that portfolio VaR < sum of individual VaRs.
    This is the quantitative argument for diversification.
    """
    n_assets = len(S0s)
    _, paths = simulate_correlated_gbm(S0s, mus, sigmas, corr,
                                       T, n_steps, n_paths, seed=seed)
    # Portfolio VaR
    pnl_port = portfolio_pnl(paths, weights)
    rm_port  = var_cvar(pnl_port, alpha)

    # Individual asset VaRs (unit weight on each)
    individual_vars = []
    for i in range(n_assets):
        e_i = np.zeros(n_assets); e_i[i] = 1.0
        pnl_i = portfolio_pnl(paths, e_i) * weights[i]
        individual_vars.append(var_cvar(pnl_i, alpha)["VaR"])

    sum_individual = sum(individual_vars)
    benefit = sum_individual - rm_port["VaR"]

    return {
        "portfolio_VaR":   rm_port["VaR"],
        "individual_VaRs": individual_vars,
        "sum_individual":  sum_individual,
        "diversification_benefit": benefit,
    }


# =============================================================================
# 5.  Plots
# =============================================================================

def plot_all(S0s, mus, sigmas, corr, weights, T=1.0, n_paths=30_000):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(15, 11))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)
    n_steps = 252

    _, paths = simulate_correlated_gbm(S0s, mus, sigmas, corr,
                                       T, n_steps, n_paths, seed=0)
    pnl = portfolio_pnl(paths, weights)
    rm95  = var_cvar(pnl, 0.95)
    rm99  = var_cvar(pnl, 0.99)

    # ── A: Correlated asset paths ────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    times = np.linspace(0, T, n_steps + 1)
    n_assets = len(S0s)
    colors = ["#3C3489", "#0F6E56", "#E85D24", "#BA7517"]
    for i in range(n_assets):
        for j in range(min(20, n_paths)):
            ax_a.plot(times, paths[j, i, :], lw=0.5, alpha=0.25,
                      color=colors[i % len(colors)])
        ax_a.plot(times, paths[:, i, :].mean(axis=0), lw=2,
                  color=colors[i % len(colors)], label=f"Asset {i+1}")
    ax_a.set_title("Correlated GBM paths (means bold)")
    ax_a.set_xlabel("Time"); ax_a.set_ylabel("Price")
    ax_a.legend(fontsize=8); ax_a.grid(True, alpha=0.2)

    # ── B: Portfolio P&L histogram + VaR/CVaR ───────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.hist(pnl, bins=80, density=True, alpha=0.55, color="#3C3489")
    ax_b.axvline(-rm95["VaR"],  color="#E85D24", lw=2, ls="--",
                 label=f"VaR 95% = {rm95['VaR']:.3f}")
    ax_b.axvline(-rm99["VaR"],  color="#BA7517", lw=2, ls="-.",
                 label=f"VaR 99% = {rm99['VaR']:.3f}")
    ax_b.axvline(-rm95["CVaR"], color="#0F6E56", lw=2, ls=":",
                 label=f"CVaR 95% = {rm95['CVaR']:.3f}")
    # Shade tail
    tail_mask = pnl <= -rm95["VaR"]
    if tail_mask.sum() > 0:
        counts, bins = np.histogram(pnl[tail_mask], bins=30, density=True)
        ax_b.fill_between(bins[:-1], 0, counts * tail_mask.mean(),
                          alpha=0.25, color="#E85D24", label="5% tail")
    ax_b.set_title("Portfolio P&L distribution")
    ax_b.set_xlabel("Portfolio return"); ax_b.set_ylabel("Density")
    ax_b.legend(fontsize=7); ax_b.grid(True, alpha=0.25)

    # ── C: VaR vs correlation ────────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    rho_vals = np.linspace(-0.9, 0.9, 19)
    sens_rho = var_vs_correlation(
        S0s[:2], mus[:2], sigmas[:2], weights[:2],
        T, n_steps, n_paths=15_000, rho_values=rho_vals, seed=0,
    )
    ax_c.plot(rho_vals, sens_rho["VaR"],  "o-", color="#E85D24", lw=2, label="VaR 95%")
    ax_c.plot(rho_vals, sens_rho["CVaR"], "s-", color="#BA7517", lw=2, label="CVaR 95%")
    ax_c.set_title("VaR vs asset correlation")
    ax_c.set_xlabel("Pairwise ρ"); ax_c.set_ylabel("VaR / CVaR (portfolio return)")
    ax_c.legend(fontsize=8); ax_c.grid(True, alpha=0.25)

    # ── D: VaR vs horizon ────────────────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 0])
    horizons = np.array([1/52, 1/12, 1/4, 1/2, 1.0, 2.0])
    sens_T   = var_vs_horizon(S0s, mus, sigmas, corr, weights, horizons,
                              n_paths=15_000, seed=0)
    ax_d.plot(horizons * 12, sens_T["VaR"],  "o-", color="#E85D24", lw=2, label="VaR 95%")
    ax_d.plot(horizons * 12, sens_T["CVaR"], "s-", color="#BA7517", lw=2, label="CVaR 95%")
    # sqrt(T) reference
    ref = sens_T["VaR"][0] * np.sqrt(horizons / horizons[0])
    ax_d.plot(horizons * 12, ref, "k--", lw=1.2, label=r"$\propto\sqrt{T}$")
    ax_d.set_title("VaR vs horizon")
    ax_d.set_xlabel("Horizon (months)"); ax_d.set_ylabel("VaR")
    ax_d.legend(fontsize=8); ax_d.grid(True, alpha=0.25)

    # ── E: Diversification benefit bar chart ─────────────────────────────────
    ax_e = fig.add_subplot(gs[1, 1])
    div = diversification_benefit(S0s, mus, sigmas, corr, weights,
                                   T, n_steps, n_paths=20_000)
    labels = [f"Asset {i+1}" for i in range(len(S0s))] + ["Portfolio", "Sum of indiv."]
    values = div["individual_VaRs"] + [div["portfolio_VaR"], div["sum_individual"]]
    bar_colors = [colors[i % len(colors)] for i in range(len(S0s))] + ["#3C3489", "#BBBBBB"]
    bars = ax_e.bar(labels, values, color=bar_colors, alpha=0.8)
    ax_e.bar_label(bars, fmt="%.3f", fontsize=8)
    benefit_pct = 100 * div["diversification_benefit"] / div["sum_individual"]
    ax_e.set_title(f"Diversification benefit\n(−{benefit_pct:.1f}% vs undiversified)")
    ax_e.set_ylabel("VaR 95%"); ax_e.grid(True, alpha=0.25, axis="y")

    # ── F: Individual vs portfolio cumulative return distribution ────────────
    ax_f = fig.add_subplot(gs[1, 2])
    pnl_sorted = np.sort(pnl)
    cdf = np.arange(1, len(pnl_sorted) + 1) / len(pnl_sorted)
    ax_f.plot(pnl_sorted, cdf, color="#3C3489", lw=2, label="Portfolio")
    for i in range(n_assets):
        e_i = np.zeros(n_assets); e_i[i] = 1.0
        pnl_i = np.sort(portfolio_pnl(paths, e_i) * weights[i])
        ax_f.plot(pnl_i, cdf, lw=1.2, ls="--", color=colors[i % len(colors)],
                  alpha=0.8, label=f"Asset {i+1} only")
    ax_f.axvline(-rm95["VaR"], color="#E85D24", lw=1.5, ls=":",
                 label=f"Portfolio VaR 95%")
    ax_f.set_title("Return CDFs: portfolio vs individual")
    ax_f.set_xlabel("Return"); ax_f.set_ylabel("CDF")
    ax_f.legend(fontsize=7); ax_f.grid(True, alpha=0.25)
    ax_f.set_xlim(np.percentile(pnl, 0.5), np.percentile(pnl, 99.5))

    fig.suptitle("Phase 3c — Portfolio Risk: VaR & CVaR (Correlated GBM)", fontsize=14)
    plt.savefig("results/phase3c_portfolio_risk.png", dpi=140, bbox_inches="tight")
    print("Saved → results/phase3c_portfolio_risk.png")
    plt.show()


# =============================================================================
# 6.  Main
# =============================================================================

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    # 3-asset portfolio
    S0s     = np.array([100.0, 100.0, 100.0])
    mus     = np.array([0.08,  0.06,  0.10])
    sigmas  = np.array([0.20,  0.15,  0.25])
    weights = np.array([0.4,   0.35,  0.25])   # fractional weights summing to 1; VaR is expressed as portfolio return

    corr = np.array([
        [1.00, 0.45, 0.20],
        [0.45, 1.00, 0.30],
        [0.20, 0.30, 1.00],
    ])

    T       = 1.0
    n_steps = 252
    n_paths = 50_000

    print("=" * 55)
    print("Phase 3c — Portfolio Risk: VaR & CVaR")
    print("=" * 55)
    print(f"Assets: {len(S0s)}  |  Horizon: {T}y  |  Paths: {n_paths:,}\n")
    print(f"Weights:  {weights}")
    print(f"Drifts:   {mus}")
    print(f"Vols:     {sigmas}")
    print(f"Correlation matrix:\n{corr}\n")

    _, paths = simulate_correlated_gbm(S0s, mus, sigmas, corr,
                                       T, n_steps, n_paths, seed=0)
    pnl = portfolio_pnl(paths, weights)

    print("── Portfolio P&L statistics ──")
    print(f"  Mean return : {pnl.mean():.4f}  ({pnl.mean()*100:.2f}%)")
    print(f"  Std return  : {pnl.std():.4f}")
    print(f"  Skewness    : {float(((pnl - pnl.mean())**3).mean() / pnl.std()**3):.4f}")

    for alpha in [0.90, 0.95, 0.99]:
        rm = var_cvar(pnl, alpha)
        print(f"\n  α = {alpha:.0%}")
        print(f"    VaR  = {rm['VaR']:.4f}  ({rm['VaR']*100:.2f}% loss)")
        print(f"    CVaR = {rm['CVaR']:.4f}  ({rm['CVaR']*100:.2f}% expected loss in tail)")

    print("\n── Diversification benefit (VaR 95%) ──")
    div = diversification_benefit(S0s, mus, sigmas, corr, weights,
                                   T, n_steps, n_paths=30_000)
    for i, v in enumerate(div["individual_VaRs"]):
        print(f"  Asset {i+1} VaR (standalone):  {v:.4f}")
    print(f"  Sum of individual VaRs:       {div['sum_individual']:.4f}")
    print(f"  Portfolio VaR:                {div['portfolio_VaR']:.4f}")
    benefit_pct = 100 * div["diversification_benefit"] / div["sum_individual"]
    print(f"  Diversification benefit:      {div['diversification_benefit']:.4f}  ({benefit_pct:.1f}% reduction)")

    plot_all(S0s, mus, sigmas, corr, weights, T=T, n_paths=30_000)