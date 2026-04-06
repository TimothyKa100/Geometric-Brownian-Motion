from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from options.black_scholes import black_scholes_greeks


def _simulate_terminal(S0: float, r: float, sigma: float, T: float, n_paths: int, seed: Optional[int]):
	rng = np.random.default_rng(seed)
	z = rng.standard_normal(n_paths)
	s_t = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
	return s_t, z


def pathwise_delta(
	S0: float,
	K: float,
	T: float,
	r: float,
	sigma: float,
	n_paths: int = 100_000,
	seed: int = 0,
) -> tuple[float, float]:
	s_t, _ = _simulate_terminal(S0, r, sigma, T, n_paths, seed)
	discount = np.exp(-r * T)
	indicator = (s_t > K).astype(float)
	values = indicator * s_t / S0
	return float(discount * np.mean(values)), float(discount * np.std(values) / np.sqrt(n_paths))


def pathwise_vega(
	S0: float,
	K: float,
	T: float,
	r: float,
	sigma: float,
	n_paths: int = 100_000,
	seed: int = 0,
) -> tuple[float, float]:
	s_t, z = _simulate_terminal(S0, r, sigma, T, n_paths, seed)
	discount = np.exp(-r * T)
	indicator = (s_t > K).astype(float)
	values = indicator * s_t * (z * np.sqrt(T) - sigma * T)
	return float(discount * np.mean(values)), float(discount * np.std(values) / np.sqrt(n_paths))


def pathwise_rho(
	S0: float,
	K: float,
	T: float,
	r: float,
	sigma: float,
	n_paths: int = 100_000,
	seed: int = 0,
) -> tuple[float, float]:
	s_t, _ = _simulate_terminal(S0, r, sigma, T, n_paths, seed)
	discount = np.exp(-r * T)
	indicator = (s_t > K).astype(float)
	payoff = np.maximum(s_t - K, 0.0)
	values = T * (indicator * s_t - payoff)
	return float(discount * np.mean(values)), float(discount * np.std(values) / np.sqrt(n_paths))


def lr_delta(
	S0: float,
	K: float,
	T: float,
	r: float,
	sigma: float,
	n_paths: int = 100_000,
	seed: int = 0,
) -> tuple[float, float]:
	s_t, z = _simulate_terminal(S0, r, sigma, T, n_paths, seed)
	discount = np.exp(-r * T)
	payoff = np.maximum(s_t - K, 0.0)
	score = z / (S0 * sigma * np.sqrt(T))
	values = payoff * score
	return float(discount * np.mean(values)), float(discount * np.std(values) / np.sqrt(n_paths))


def lr_vega(
	S0: float,
	K: float,
	T: float,
	r: float,
	sigma: float,
	n_paths: int = 100_000,
	seed: int = 0,
) -> tuple[float, float]:
	s_t, z = _simulate_terminal(S0, r, sigma, T, n_paths, seed)
	discount = np.exp(-r * T)
	payoff = np.maximum(s_t - K, 0.0)
	score = (z**2 - 1.0) / sigma - z * np.sqrt(T)
	values = payoff * score
	return float(discount * np.mean(values)), float(discount * np.std(values) / np.sqrt(n_paths))


def lr_gamma(
	S0: float,
	K: float,
	T: float,
	r: float,
	sigma: float,
	n_paths: int = 100_000,
	seed: int = 0,
) -> tuple[float, float]:
	s_t, z = _simulate_terminal(S0, r, sigma, T, n_paths, seed)
	discount = np.exp(-r * T)
	payoff = np.maximum(s_t - K, 0.0)
	h = S0 * sigma * np.sqrt(T)
	score = (z**2 - 1.0) / (h**2) - z / (h * S0)
	values = payoff * score
	return float(discount * np.mean(values)), float(discount * np.std(values) / np.sqrt(n_paths))


def greek_profiles(
	K: float,
	T: float,
	r: float,
	sigma: float,
	S0_range: np.ndarray,
	n_paths: int = 50_000,
	seed: int = 0,
) -> dict[str, np.ndarray]:
	pw_deltas, lr_deltas = [], []
	pw_vegas, lr_vegas = [], []
	lr_gammas = []
	bs_deltas, bs_gammas, bs_vegas = [], [], []
	for S0 in S0_range:
		bs = black_scholes_greeks(S0, K, T, r, sigma)
		bs_deltas.append(bs["call_delta"])
		bs_gammas.append(bs["gamma"])
		bs_vegas.append(bs["vega"])
		pw_d, _ = pathwise_delta(S0, K, T, r, sigma, n_paths, seed)
		lr_d, _ = lr_delta(S0, K, T, r, sigma, n_paths, seed)
		pw_v, _ = pathwise_vega(S0, K, T, r, sigma, n_paths, seed)
		lr_v, _ = lr_vega(S0, K, T, r, sigma, n_paths, seed)
		lr_g, _ = lr_gamma(S0, K, T, r, sigma, n_paths, seed)
		pw_deltas.append(pw_d)
		lr_deltas.append(lr_d)
		pw_vegas.append(pw_v)
		lr_vegas.append(lr_v)
		lr_gammas.append(lr_g)
	return {
		"S0_range": np.asarray(S0_range, dtype=float),
		"bs_delta": np.asarray(bs_deltas, dtype=float),
		"bs_gamma": np.asarray(bs_gammas, dtype=float),
		"bs_vega": np.asarray(bs_vegas, dtype=float),
		"pw_delta": np.asarray(pw_deltas, dtype=float),
		"lr_delta": np.asarray(lr_deltas, dtype=float),
		"pw_vega": np.asarray(pw_vegas, dtype=float),
		"lr_vega": np.asarray(lr_vegas, dtype=float),
		"lr_gamma": np.asarray(lr_gammas, dtype=float),
	}


def greek_convergence(
	S0: float,
	K: float,
	T: float,
	r: float,
	sigma: float,
	path_counts: list[int],
	seed: int = 0,
) -> dict[str, np.ndarray]:
	bs = black_scholes_greeks(S0, K, T, r, sigma)
	pw_d_errors, lr_d_errors, pw_v_errors, lr_v_errors = [], [], [], []
	for n in path_counts:
		pw_d, _ = pathwise_delta(S0, K, T, r, sigma, n, seed)
		lr_d, _ = lr_delta(S0, K, T, r, sigma, n, seed)
		pw_v, _ = pathwise_vega(S0, K, T, r, sigma, n, seed)
		lr_v, _ = lr_vega(S0, K, T, r, sigma, n, seed)
		pw_d_errors.append(abs(pw_d - bs["call_delta"]))
		lr_d_errors.append(abs(lr_d - bs["call_delta"]))
		pw_v_errors.append(abs(pw_v - bs["vega"]))
		lr_v_errors.append(abs(lr_v - bs["vega"]))
	return {
		"n_paths": np.asarray(path_counts, dtype=float),
		"pw_delta_err": np.asarray(pw_d_errors, dtype=float),
		"lr_delta_err": np.asarray(lr_d_errors, dtype=float),
		"pw_vega_err": np.asarray(pw_v_errors, dtype=float),
		"lr_vega_err": np.asarray(lr_v_errors, dtype=float),
		"bs_delta": float(bs["call_delta"]),
		"bs_vega": float(bs["vega"]),
	}


def plot_greeks_dashboard(
	output_dir: Path,
	S0: float,
	K: float,
	T: float,
	r: float,
	sigma: float,
	*,
	show: bool = True,
) -> Path:
	plt.style.use("seaborn-v0_8-whitegrid")
	fig = plt.figure(figsize=(15, 11))
	gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

	S0_range = np.arange(max(40.0, 0.7 * K), 1.4 * K, 5.0, dtype=float)
	profiles = greek_profiles(K, T, r, sigma, S0_range, n_paths=30_000, seed=0)

	ax_a = fig.add_subplot(gs[0, 0])
	ax_a.plot(S0_range, profiles["bs_delta"], "-", color="k", lw=2, label="BS")
	ax_a.plot(S0_range, profiles["pw_delta"], "o--", color="#3C3489", lw=1.5, ms=5, label="Pathwise")
	ax_a.plot(S0_range, profiles["lr_delta"], "x--", color="#0F6E56", lw=1.5, ms=6, label="LRM")
	ax_a.axvline(K, color="gray", lw=1, ls=":")
	ax_a.set_title("Delta")
	ax_a.set_xlabel("S0")
	ax_a.set_ylabel("Delta")
	ax_a.legend(fontsize=8)
	ax_a.grid(True, alpha=0.25)

	ax_b = fig.add_subplot(gs[0, 1])
	ax_b.plot(S0_range, profiles["bs_vega"], "-", color="k", lw=2, label="BS")
	ax_b.plot(S0_range, profiles["pw_vega"], "o--", color="#3C3489", lw=1.5, ms=5, label="Pathwise")
	ax_b.plot(S0_range, profiles["lr_vega"], "x--", color="#0F6E56", lw=1.5, ms=6, label="LRM")
	ax_b.axvline(K, color="gray", lw=1, ls=":")
	ax_b.set_title("Vega")
	ax_b.set_xlabel("S0")
	ax_b.set_ylabel("Vega")
	ax_b.legend(fontsize=8)
	ax_b.grid(True, alpha=0.25)

	ax_c = fig.add_subplot(gs[0, 2])
	ax_c.plot(S0_range, profiles["bs_gamma"], "-", color="k", lw=2, label="BS")
	ax_c.plot(S0_range, profiles["lr_gamma"], "x--", color="#E85D24", lw=1.5, ms=6, label="LRM")
	ax_c.axvline(K, color="gray", lw=1, ls=":")
	ax_c.set_title("Gamma")
	ax_c.set_xlabel("S0")
	ax_c.set_ylabel("Gamma")
	ax_c.legend(fontsize=8)
	ax_c.grid(True, alpha=0.25)

	ax_d = fig.add_subplot(gs[1, 0])
	path_counts = [500, 1000, 3000, 5000, 10_000, 30_000, 50_000, 100_000]
	conv = greek_convergence(S0, K, T, r, sigma, path_counts)
	ax_d.loglog(conv["n_paths"], conv["pw_delta_err"], "o-", color="#3C3489", lw=2, label="Pathwise delta")
	ax_d.loglog(conv["n_paths"], conv["lr_delta_err"], "s-", color="#0F6E56", lw=2, label="LRM delta")
	ref = np.array(conv["n_paths"], dtype=float) ** (-0.5)
	ref *= conv["pw_delta_err"][0] / ref[0]
	ax_d.loglog(conv["n_paths"], ref, "k--", lw=1, label=r"$O(n^{-1/2})$")
	ax_d.set_title("Delta convergence")
	ax_d.set_xlabel("n_paths")
	ax_d.set_ylabel("|error|")
	ax_d.legend(fontsize=8)
	ax_d.grid(True, which="both", alpha=0.25)

	ax_e = fig.add_subplot(gs[1, 1])
	rng = np.random.default_rng(1)
	pw_samples = [pathwise_delta(S0, K, T, r, sigma, 2000, int(rng.integers(1_000_000)))[0] for _ in range(80)]
	lr_samples = [lr_delta(S0, K, T, r, sigma, 2000, int(rng.integers(1_000_000)))[0] for _ in range(80)]
	bs_d = black_scholes_greeks(S0, K, T, r, sigma)["call_delta"]
	bins = np.linspace(min(min(pw_samples), min(lr_samples)) - 0.01, max(max(pw_samples), max(lr_samples)) + 0.01, 30)
	ax_e.hist(pw_samples, bins=bins, alpha=0.55, color="#3C3489", label=f"Pathwise std={np.std(pw_samples):.4f}")
	ax_e.hist(lr_samples, bins=bins, alpha=0.55, color="#0F6E56", label=f"LRM std={np.std(lr_samples):.4f}")
	ax_e.axvline(bs_d, color="k", lw=2, ls="--", label=f"BS delta = {bs_d:.4f}")
	ax_e.set_title("Pathwise vs LRM variance")
	ax_e.set_xlabel("Delta estimate")
	ax_e.set_ylabel("Count")
	ax_e.legend(fontsize=7)
	ax_e.grid(True, alpha=0.25)

	ax_f = fig.add_subplot(gs[1, 2])
	ax_f.axis("off")
	bs = black_scholes_greeks(S0, K, T, r, sigma)
	pw_d, pw_d_se = pathwise_delta(S0, K, T, r, sigma, 100_000)
	lr_d, lr_d_se = lr_delta(S0, K, T, r, sigma, 100_000)
	pw_v, pw_v_se = pathwise_vega(S0, K, T, r, sigma, 100_000)
	lr_v, lr_v_se = lr_vega(S0, K, T, r, sigma, 100_000)
	lr_g, lr_g_se = lr_gamma(S0, K, T, r, sigma, 100_000)
	rows = [
		["Greek", "BS analytical", "Pathwise MC", "LRM MC"],
		["Delta", f"{bs['call_delta']:.4f}", f"{pw_d:.4f}±{pw_d_se:.4f}", f"{lr_d:.4f}±{lr_d_se:.4f}"],
		["Vega", f"{bs['vega']:.4f}", f"{pw_v:.4f}±{pw_v_se:.4f}", f"{lr_v:.4f}±{lr_v_se:.4f}"],
		["Gamma", f"{bs['gamma']:.4f}", "—", f"{lr_g:.4f}±{lr_g_se:.4f}"],
		["Theta", f"{bs['call_theta']:.4f}", "—", "—"],
		["Rho", f"{bs['call_rho']:.4f}", "—", "—"],
	]
	table = ax_f.table(cellText=rows[1:], colLabels=rows[0], cellLoc="center", loc="center")
	table.auto_set_font_size(False)
	table.set_fontsize(9)
	table.scale(1.1, 1.6)
	ax_f.set_title(f"Greeks at ATM (S0=K={S0})", pad=20)

	fig.suptitle("Greeks: Pathwise vs Likelihood Ratio Method", fontsize=14)
	fig.tight_layout()
	path = output_dir / "options_greeks_dashboard.png"
	fig.savefig(path, dpi=140, bbox_inches="tight")
	if show:
		plt.show()
	plt.close(fig)
	return path


def run_greeks_demo(
	output_dir: Path,
	S0: float = 100.0,
	K: float = 100.0,
	T: float = 1.0,
	r: float = 0.05,
	sigma: float = 0.20,
) -> Path:
	return plot_greeks_dashboard(output_dir, S0, K, T, r, sigma, show=True)
