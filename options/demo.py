from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from models import GBMParams, HestonParams, euler_maruyama_gbm
from options.black_scholes import (
	black_scholes_call_price,
	black_scholes_digital_call_price,
	black_scholes_greeks,
	black_scholes_put_price,
)
from options.heston import heston_call_price, heston_implied_vol_surface
from options.greeks import run_greeks_demo
from options.monte_carlo import (
	mc_price_asian_arithmetic_call_gbm,
	mc_price_barrier_gbm,
	mc_price_digital_call_gbm,
	mc_price_european_gbm,
)


def _option_chain(S0: float, T: float, r: float, sigma: float, strikes: np.ndarray, n_paths: int = 30_000, seed: int = 0) -> dict[str, np.ndarray]:
	params = GBMParams(mu=r, sigma=sigma)
	_, paths = euler_maruyama_gbm(S0, params, T, 252, n_paths, seed=seed)
	s_t = paths[:, -1]
	discount = np.exp(-r * T)
	return {
		"strikes": strikes,
		"bs_calls": np.array([black_scholes_call_price(S0, float(k), T, r, sigma) for k in strikes]),
		"bs_puts": np.array([black_scholes_put_price(S0, float(k), T, r, sigma) for k in strikes]),
		"mc_calls": np.array([discount * np.maximum(s_t - float(k), 0.0).mean() for k in strikes]),
		"mc_puts": np.array([discount * np.maximum(float(k) - s_t, 0.0).mean() for k in strikes]),
	}


def _pricing_convergence(S0: float, K: float, T: float, r: float, sigma: float, path_counts: list[int], seed: int = 0) -> dict[str, np.ndarray | float]:
	bs = black_scholes_call_price(S0, K, T, r, sigma)
	prices, errors, ses = [], [], []
	for n in path_counts:
		res = mc_price_european_gbm(S0, K, T, r, sigma, n_paths=n, seed=seed)
		prices.append(res.price)
		ses.append(res.se)
		errors.append(abs(res.price - bs))
	return {
		"n_paths": np.array(path_counts, dtype=float),
		"mc_prices": np.array(prices, dtype=float),
		"ses": np.array(ses, dtype=float),
		"errors": np.array(errors, dtype=float),
		"bs_price": float(bs),
	}


def _plot_options_dashboard(output_dir: Path, S0: float, K: float, T: float, r: float, sigma: float) -> Path:
	plt.style.use("seaborn-v0_8-whitegrid")
	fig, axes = plt.subplots(2, 3, figsize=(15, 11))
	ax_a, ax_b, ax_c = axes[0]
	ax_d, ax_e, ax_f = axes[1]

	S_range = np.linspace(60.0, 160.0, 300)
	ax_a.plot(S_range, np.maximum(S_range - K, 0.0), color="#3C3489", lw=2, label="Call payoff")
	ax_a.plot(S_range, np.maximum(K - S_range, 0.0), color="#0F6E56", lw=2, label="Put payoff")
	ax_a.axvline(K, color="gray", lw=1, ls="--", label=f"Strike K={K}")
	ax_a.axvline(S0, color="#BA7517", lw=1, ls=":", label=f"S0={S0}")
	ax_a.set_title("Payoff at expiry")
	ax_a.set_xlabel("ST")
	ax_a.set_ylabel("Payoff")
	ax_a.legend(fontsize=8)

	sigmas = np.linspace(0.05, 0.60, 60)
	ax_b.plot(sigmas, [black_scholes_call_price(S0, K, T, r, s) for s in sigmas], color="#3C3489", lw=2, label="Call (BS)")
	ax_b.plot(sigmas, [black_scholes_put_price(S0, K, T, r, s) for s in sigmas], color="#0F6E56", lw=2, label="Put (BS)")
	ax_b.axvline(sigma, color="#BA7517", lw=1, ls="--", label=f"σ={sigma}")
	ax_b.set_title("Price vs volatility")
	ax_b.set_xlabel("σ")
	ax_b.set_ylabel("Option price")
	ax_b.legend(fontsize=8)

	strikes = np.arange(70.0, 145.0, 5.0, dtype=float)
	chain = _option_chain(S0, T, r, sigma, strikes, n_paths=30_000, seed=0)
	ax_c.plot(chain["strikes"], chain["bs_calls"], "o-", color="#3C3489", lw=2, label="BS call")
	ax_c.plot(chain["strikes"], chain["mc_calls"], "x--", color="#E85D24", lw=1.5, ms=6, label="MC call")
	ax_c.plot(chain["strikes"], chain["bs_puts"], "s-", color="#0F6E56", lw=2, label="BS put")
	ax_c.plot(chain["strikes"], chain["mc_puts"], "+--", color="#BA7517", lw=1.5, ms=6, label="MC put")
	ax_c.axvline(S0, color="gray", lw=1, ls=":")
	ax_c.set_title("Option chain: MC vs BS")
	ax_c.set_xlabel("Strike K")
	ax_c.set_ylabel("Price")
	ax_c.legend(fontsize=8)

	path_counts = [100, 300, 600, 1000, 3000, 6000, 10_000, 30_000, 50_000]
	conv = _pricing_convergence(S0, K, T, r, sigma, path_counts, seed=0)
	ax_d.semilogx(conv["n_paths"], conv["mc_prices"], "o-", color="#3C3489", lw=2, label="MC price")
	ax_d.axhline(conv["bs_price"], color="#0F6E56", lw=1.5, ls="--", label=f"BS = {conv['bs_price']:.3f}")
	mcp = np.asarray(conv["mc_prices"])
	ses = np.asarray(conv["ses"])
	ax_d.fill_between(conv["n_paths"], mcp - 1.96 * ses, mcp + 1.96 * ses, alpha=0.15, color="#3C3489")
	ax_d.set_title("MC call convergence to BS")
	ax_d.set_xlabel("Number of paths")
	ax_d.set_ylabel("Call price")
	ax_d.legend(fontsize=8)

	asian_prices = []
	euro_bs = []
	for strike in strikes:
		res_a = mc_price_asian_arithmetic_call_gbm(S0, float(strike), T, r, sigma, n_paths=30_000, seed=1)
		asian_prices.append(res_a.price)
		euro_bs.append(black_scholes_call_price(S0, float(strike), T, r, sigma))
	ax_e.plot(strikes, euro_bs, "o-", color="#3C3489", lw=2, label="European (BS)")
	ax_e.plot(strikes, asian_prices, "x--", color="#E85D24", lw=2, ms=6, label="Asian arith. (MC)")
	ax_e.axvline(S0, color="gray", lw=1, ls=":")
	ax_e.set_title("European vs Asian call prices")
	ax_e.set_xlabel("Strike K")
	ax_e.set_ylabel("Call price")
	ax_e.legend(fontsize=8)

	digital_mc_prices, digital_bs_prices = [], []
	for strike in strikes:
		digital_bs_prices.append(black_scholes_digital_call_price(S0, float(strike), T, r, sigma))
		digital_mc_prices.append(mc_price_digital_call_gbm(S0, float(strike), T, r, sigma, n_paths=30_000, seed=2).price)
	ax_f.plot(strikes, digital_bs_prices, "o-", color="#3C3489", lw=2, label="Digital (BS)")
	ax_f.plot(strikes, digital_mc_prices, "x--", color="#E85D24", lw=2, ms=6, label="Digital (MC)")
	ax_f.axvline(S0, color="gray", lw=1, ls=":")
	ax_f.set_title("Digital call: MC vs BS")
	ax_f.set_xlabel("Strike K")
	ax_f.set_ylabel("Price ($1 payoff)")
	ax_f.legend(fontsize=8)

	fig.suptitle("Options Pricing (MC vs Black-Scholes)", fontsize=14)
	fig.tight_layout()
	path = output_dir / "options_pricing_dashboard.png"
	fig.savefig(path, dpi=140, bbox_inches="tight")
	plt.show()
	plt.close(fig)
	return path


def _plot_barrier_dashboard(output_dir: Path, S0: float, K: float, T: float, r: float, sigma: float) -> Path:
	fig, ax = plt.subplots(figsize=(9, 5))
	barriers = np.arange(60.0, 140.0, 3.0, dtype=float)
	down_out = []
	down_in = []
	for barrier in barriers:
		down_out.append(mc_price_barrier_gbm(S0, K, T, r, sigma, barrier=barrier, direction="down", knock="out", option="call", n_paths=20_000, seed=0).price)
		down_in.append(mc_price_barrier_gbm(S0, K, T, r, sigma, barrier=barrier, direction="down", knock="in", option="call", n_paths=20_000, seed=0).price)
	ax.plot(barriers, down_out, label="Down-and-out")
	ax.plot(barriers, down_in, label="Down-and-in")
	ax.axhline(black_scholes_call_price(S0, K, T, r, sigma), color="k", linestyle="--", label="European BS")
	ax.axvline(S0, color="gray", linestyle=":")
	ax.set_title("Barrier sensitivity")
	ax.set_xlabel("Barrier")
	ax.set_ylabel("Price")
	ax.grid(True, alpha=0.25)
	ax.legend()
	fig.tight_layout()
	path = output_dir / "options_barrier_dashboard.png"
	fig.savefig(path, dpi=140, bbox_inches="tight")
	plt.show()
	plt.close(fig)
	return path


def _plot_heston_pricing_curve(output_dir: Path, S0: float, T: float, r: float, params: HestonParams) -> Path:
	strikes = np.linspace(70.0, 140.0, 35, dtype=float)
	heston_prices = np.array([heston_call_price(S0, float(K), T, r, params) for K in strikes])
	bs_sigma = math.sqrt(params.theta)
	bs_prices = np.array([black_scholes_call_price(S0, float(K), T, r, bs_sigma) for K in strikes])

	fig, ax = plt.subplots(figsize=(9, 5))
	ax.plot(strikes, heston_prices, "o-", color="#3C3489", lw=2, label="Heston call")
	ax.plot(strikes, bs_prices, "x--", color="#0F6E56", lw=1.5, label=f"BS call (σ={bs_sigma:.3f})")
	ax.axvline(S0, color="gray", lw=1, ls=":")
	ax.set_title("Heston vs Black-Scholes call prices")
	ax.set_xlabel("Strike K")
	ax.set_ylabel("Call price")
	ax.legend(fontsize=8)
	ax.grid(True, alpha=0.25)
	fig.tight_layout()
	path = output_dir / "heston_pricing_vs_bs.png"
	fig.savefig(path, dpi=140, bbox_inches="tight")
	plt.close(fig)
	return path


def _plot_heston_implied_vol_surface(output_dir: Path, S0: float, r: float, params: HestonParams) -> Path:
	strikes = np.linspace(70.0, 140.0, 15, dtype=float)
	maturities = np.linspace(0.25, 2.0, 10, dtype=float)
	surface = heston_implied_vol_surface(S0=S0, r=r, params=params, strikes=strikes, maturities=maturities)

	fig, ax = plt.subplots(figsize=(10, 6))
	im = ax.imshow(
		surface,
		origin="lower",
		aspect="auto",
		cmap="viridis",
		extent=[strikes[0], strikes[-1], maturities[0], maturities[-1]],
	)
	cbar = fig.colorbar(im, ax=ax)
	cbar.set_label("Implied volatility")
	ax.set_title("Heston implied volatility surface")
	ax.set_xlabel("Strike K")
	ax.set_ylabel("Maturity T")
	ax.set_xticks(np.linspace(strikes[0], strikes[-1], 6))
	ax.set_yticks(np.linspace(maturities[0], maturities[-1], 5))
	fig.tight_layout()
	path = output_dir / "heston_implied_vol_surface.png"
	fig.savefig(path, dpi=140, bbox_inches="tight")
	plt.close(fig)
	return path


def run_options_demo(output_dir: Path) -> None:
	"""Standalone options application with the final multi-panel plots."""
	print("\nRunning standalone options dashboard...")
	output_dir.mkdir(parents=True, exist_ok=True)

	S0 = 100.0
	K = 105.0
	T = 1.0
	r = 0.05
	sigma = 0.20

	bs_call = black_scholes_call_price(S0, K, T, r, sigma)
	bs_put = black_scholes_put_price(S0, K, T, r, sigma)
	bs_digital = black_scholes_digital_call_price(S0, K, T, r, sigma)
	greeks = black_scholes_greeks(S0, K, T, r, sigma)
	mc_call = mc_price_european_gbm(S0, K, T, r, sigma, option="call", n_paths=50_000, seed=123)
	mc_put = mc_price_european_gbm(S0, K, T, r, sigma, option="put", n_paths=50_000, seed=123)
	mc_asian = mc_price_asian_arithmetic_call_gbm(S0, K, T, r, sigma, n_paths=50_000, seed=124)

	print("Options summary:")
	print(f"  BS call={bs_call:.4f}, MC call={mc_call.price:.4f} +/- {1.96 * mc_call.se:.4f}")
	print(f"  BS put ={bs_put:.4f}, MC put ={mc_put.price:.4f} +/- {1.96 * mc_put.se:.4f}")
	print(f"  BS digital call={bs_digital:.4f}")
	print(f"  Asian arithmetic call (MC)={mc_asian.price:.4f} +/- {1.96 * mc_asian.se:.4f}")
	print(
		"  Greeks: "
		f"delta(call)={greeks['call_delta']:.4f}, gamma={greeks['gamma']:.5f}, "
		f"vega={greeks['vega']:.4f}, theta(call)={greeks['call_theta']:.4f}"
	)

	pricing_path = _plot_options_dashboard(output_dir, S0, K, T, r, sigma)
	barrier_path = _plot_barrier_dashboard(output_dir, S0, K, T, r, sigma)
	greeks_path = run_greeks_demo(output_dir, S0=S0, K=K, T=T, r=r, sigma=sigma)

	params = HestonParams(kappa=2.0, theta=0.04, sigma=0.25, rho=-0.7, v0=0.04)
	heston_path = _plot_heston_pricing_curve(output_dir, S0, T, r, params)
	heston_surface_path = _plot_heston_implied_vol_surface(output_dir, S0, r, params)

	print(f"Saved options pricing dashboard -> {pricing_path.name}")
	print(f"Saved barrier dashboard -> {barrier_path.name}")
	print(f"Saved greeks dashboard -> {greeks_path.name}")
	print(f"Saved Heston pricing comparison -> {heston_path.name}")
	print(f"Saved Heston implied volatility surface -> {heston_surface_path.name}")


def run_heston_demo(output_dir: Path) -> None:
	"""Standalone Heston pricing demonstration."""
	print("\nRunning standalone Heston demo...")
	output_dir.mkdir(parents=True, exist_ok=True)

	S0 = 100.0
	T = 1.0
	r = 0.03
	params = HestonParams(kappa=2.0, theta=0.04, sigma=0.25, rho=-0.7, v0=0.04)

	heston_price = heston_call_price(S0=S0, K=S0, T=T, r=r, params=params)
	print(f"Heston ATM call price = {heston_price:.4f}")

	heston_path = _plot_heston_pricing_curve(output_dir, S0, T, r, params)
	heston_surface_path = _plot_heston_implied_vol_surface(output_dir, S0, r, params)
	print(f"Saved Heston pricing comparison -> {heston_path.name}")
	print(f"Saved Heston implied volatility surface -> {heston_surface_path.name}")
