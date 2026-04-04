from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis import (
	autocorrelation,
	gbm_log_return_stats,
	gbm_standardized_residuals,
	likelihood_ratio_test_ou_theta_zero,
	ljung_box_q,
	ou_increment_stats,
	ou_standardized_residuals,
	qq_data_normal,
	validate_gbm_estimates,
	validate_ou_estimates,
)
from estimation import (
	ParameterCI,
	estimate_gbm_mle,
	estimate_gbm_mle_panel,
	estimate_ou_exact_mle,
	estimate_ou_mle,
	estimate_ou_mle_panel,
	gbm_asymptotic_ci,
	gbm_bootstrap_ci,
	information_criteria,
	ou_asymptotic_ci,
	ou_bootstrap_ci,
)
from first_passage import first_hitting_times, summarize_hitting_times
from models import (
	GBMParams,
	OUParams,
	euler_maruyama_cir,
	euler_maruyama_gbm,
	euler_maruyama_gbm_time_varying,
	euler_maruyama_jump_diffusion_gbm,
	euler_maruyama_ou,
	euler_maruyama_ou_time_varying,
	exact_step_gbm,
	simulate_correlated_gbm_paths,
)
from options.black_scholes import (
	black_scholes_call_price,
	black_scholes_digital_call_price,
	black_scholes_greeks,
	black_scholes_put_price,
)
from options.monte_carlo import (
	mc_price_asian_arithmetic_call_gbm,
	mc_price_barrier_gbm,
	mc_price_european_gbm,
	mc_price_european_ou_log_price,
)
from options.portfolio_risk import portfolio_pnl, var_cvar
from physics_simulation import LangevinParams, simulate_langevin_1d, velocity_theoretical_moments
from plots import (
	plot_acf,
	plot_heatmap,
	plot_hitting_time_histogram,
	plot_qq,
	plot_sample_paths,
	plot_terminal_histogram,
)


def _print_estimation_report(model_name: str, true_params: dict[str, float], est_params: dict[str, float]) -> None:
	print(f"\n{model_name} estimation report")
	print("-" * (len(model_name) + 19))
	for key, true_value in true_params.items():
		estimate = est_params[key]
		abs_err = abs(estimate - true_value)
		rel_err = abs_err / max(abs(true_value), 1e-12)
		print(
			f"{key:>8s}: true={true_value: .6f}, est={estimate: .6f}, "
			f"abs_err={abs_err: .6f}, rel_err={rel_err: .2%}"
		)


def _write_csv(path: Path, header: list[str], rows: list[list[float | str]]) -> None:
	with path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(header)
		writer.writerows(rows)


def _parameter_summary(values: np.ndarray, true_value: float) -> dict[str, float]:
	bias = float(np.mean(values) - true_value)
	variance = float(np.var(values, ddof=1)) if len(values) > 1 else 0.0
	rmse = float(np.sqrt(np.mean((values - true_value) ** 2)))
	return {
		"true": float(true_value),
		"mean_estimate": float(np.mean(values)),
		"bias": bias,
		"std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
		"variance": variance,
		"rmse": rmse,
	}


def _save_monte_carlo_histograms(
	output_dir: Path,
	gbm_mu: np.ndarray,
	gbm_sigma: np.ndarray,
	ou_theta: np.ndarray,
	ou_mu: np.ndarray,
	ou_sigma: np.ndarray,
) -> None:
	fig, axes = plt.subplots(2, 3, figsize=(13, 8))

	axes[0, 0].hist(gbm_mu, bins=30, alpha=0.8)
	axes[0, 0].set_title("GBM mu estimates")
	axes[0, 1].hist(gbm_sigma, bins=30, alpha=0.8)
	axes[0, 1].set_title("GBM sigma estimates")
	axes[0, 2].axis("off")

	axes[1, 0].hist(ou_theta, bins=30, alpha=0.8)
	axes[1, 0].set_title("OU theta estimates")
	axes[1, 1].hist(ou_mu, bins=30, alpha=0.8)
	axes[1, 1].set_title("OU mu estimates")
	axes[1, 2].hist(ou_sigma, bins=30, alpha=0.8)
	axes[1, 2].set_title("OU sigma estimates")

	for ax in axes.ravel():
		ax.grid(True, alpha=0.25)

	fig.suptitle("Monte Carlo estimation distributions")
	fig.tight_layout()
	fig.savefig(output_dir / "monte_carlo_estimate_histograms.png", dpi=130)
	plt.close(fig)


def _ci_contains(ci: ParameterCI, true_value: float) -> int:
	return int(ci.lower <= true_value <= ci.upper)


def run_gbm_experiment(output_dir: Path) -> tuple[np.ndarray, float, GBMParams, object]:
	params = GBMParams(mu=0.08, sigma=0.22)
	s0 = 100.0
	horizon = 2.0
	n_steps = 504
	n_paths = 4000
	seed = 7
	dt = horizon / n_steps

	times, em_paths = euler_maruyama_gbm(
		s0=s0,
		params=params,
		horizon=horizon,
		n_steps=n_steps,
		n_paths=n_paths,
		seed=seed,
	)
	_, exact_paths = exact_step_gbm(
		s0=s0,
		params=params,
		horizon=horizon,
		n_steps=n_steps,
		n_paths=n_paths,
		seed=seed,
	)

	stats = gbm_log_return_stats(em_paths)
	print("GBM log-return moments (EM paths):")
	print(f"mean={stats.mean:.6f}, std={stats.std:.6f}, skew={stats.skew:.6f}, kurtosis={stats.kurtosis:.6f}")

	representative_path = em_paths[0]
	est = estimate_gbm_mle(representative_path, dt=dt)
	validation = validate_gbm_estimates(params, est)

	_print_estimation_report(
		"GBM",
		true_params={"mu": params.mu, "sigma": params.sigma},
		est_params={"mu": est.mu, "sigma": est.sigma},
	)
	print(f"GBM objective (NLL): {est.nll:.6f}")
	ic = information_criteria(nll=est.nll, n_obs=len(representative_path) - 1, n_params=2)
	print(f"GBM AIC={ic.aic:.3f}, BIC={ic.bic:.3f}")
	print("Validation summary:", {k: {"abs": v.abs_error, "rel": v.rel_error} for k, v in validation.items()})

	upper_barrier = 140.0
	hit_idx = first_hitting_times(em_paths, level=upper_barrier, above=True)
	hit_summary = summarize_hitting_times(hit_idx, dt=dt)
	finite_times = hit_idx[np.isfinite(hit_idx)] * dt
	print(
		f"GBM first passage to {upper_barrier}: p_hit={hit_summary.hit_probability:.3f}, "
		f"mean_tau={hit_summary.mean_hitting_time:.3f}, median_tau={hit_summary.median_hitting_time:.3f}"
	)

	fig_paths = plot_sample_paths(times, em_paths[:40], "GBM sample paths (Euler-Maruyama)")
	fig_exact = plot_sample_paths(times, exact_paths[:40], "GBM sample paths (Exact transition)")
	fig_terminal = plot_terminal_histogram(em_paths, "GBM terminal distribution")
	fig_hit = plot_hitting_time_histogram(finite_times, f"GBM hitting time histogram (level={upper_barrier})")

	fig_paths.savefig(output_dir / "gbm_paths.png", dpi=130)
	fig_exact.savefig(output_dir / "gbm_exact_paths.png", dpi=130)
	fig_terminal.savefig(output_dir / "gbm_terminal_hist.png", dpi=130)
	fig_hit.savefig(output_dir / "gbm_hitting_times.png", dpi=130)

	plt.close(fig_paths)
	plt.close(fig_exact)
	plt.close(fig_terminal)
	plt.close(fig_hit)

	return representative_path, dt, params, est


def run_ou_experiment(output_dir: Path) -> tuple[np.ndarray, float, OUParams, object, object]:
	params = OUParams(theta=1.4, mu=1.0, sigma=0.35)
	x0 = -0.4
	horizon = 3.0
	n_steps = 900
	n_paths = 4000
	seed = 11
	dt = horizon / n_steps

	times, paths = euler_maruyama_ou(
		x0=x0,
		params=params,
		horizon=horizon,
		n_steps=n_steps,
		n_paths=n_paths,
		seed=seed,
	)

	stats = ou_increment_stats(paths)
	print("\nOU increment moments:")
	print(f"mean={stats.mean:.6f}, std={stats.std:.6f}, skew={stats.skew:.6f}, kurtosis={stats.kurtosis:.6f}")

	representative_path = paths[0]
	est_em = estimate_ou_mle(representative_path, dt=dt)
	est_exact = estimate_ou_exact_mle(representative_path, dt=dt)
	validation = validate_ou_estimates(params, est_em)

	_print_estimation_report(
		"OU (EM-MLE)",
		true_params={"theta": params.theta, "mu": params.mu, "sigma": params.sigma},
		est_params={"theta": est_em.theta, "mu": est_em.mu, "sigma": est_em.sigma},
	)
	print(f"OU EM objective (NLL): {est_em.nll:.6f}")
	ic_em = information_criteria(nll=est_em.nll, n_obs=len(representative_path) - 1, n_params=3)
	print(f"OU EM AIC={ic_em.aic:.3f}, BIC={ic_em.bic:.3f}")

	_print_estimation_report(
		"OU (Exact-MLE)",
		true_params={"theta": params.theta, "mu": params.mu, "sigma": params.sigma},
		est_params={"theta": est_exact.theta, "mu": est_exact.mu, "sigma": est_exact.sigma},
	)
	print(f"OU Exact objective (NLL): {est_exact.nll:.6f}")
	ic_exact = information_criteria(nll=est_exact.nll, n_obs=len(representative_path) - 1, n_params=3)
	print(f"OU Exact AIC={ic_exact.aic:.3f}, BIC={ic_exact.bic:.3f}")

	lr_stat, lr_p = likelihood_ratio_test_ou_theta_zero(
		representative_path,
		theta_hat=est_em.theta,
		mu_hat=est_em.mu,
		sigma_hat=est_em.sigma,
		dt=dt,
	)
	print(f"OU LR test H0(theta=0): LR={lr_stat:.4f}, p={lr_p:.4f}")
	print("Validation summary:", {k: {"abs": v.abs_error, "rel": v.rel_error} for k, v in validation.items()})

	upper_barrier = 1.7
	hit_idx = first_hitting_times(paths, level=upper_barrier, above=True)
	hit_summary = summarize_hitting_times(hit_idx, dt=dt)
	finite_times = hit_idx[np.isfinite(hit_idx)] * dt
	print(
		f"OU first passage to {upper_barrier}: p_hit={hit_summary.hit_probability:.3f}, "
		f"mean_tau={hit_summary.mean_hitting_time:.3f}, median_tau={hit_summary.median_hitting_time:.3f}"
	)

	fig_paths = plot_sample_paths(times, paths[:40], "OU sample paths (Euler-Maruyama)")
	fig_terminal = plot_terminal_histogram(paths, "OU terminal distribution")
	fig_hit = plot_hitting_time_histogram(finite_times, f"OU hitting time histogram (level={upper_barrier})")

	fig_paths.savefig(output_dir / "ou_paths.png", dpi=130)
	fig_terminal.savefig(output_dir / "ou_terminal_hist.png", dpi=130)
	fig_hit.savefig(output_dir / "ou_hitting_times.png", dpi=130)

	plt.close(fig_paths)
	plt.close(fig_terminal)
	plt.close(fig_hit)

	return representative_path, dt, params, est_em, est_exact


def run_residual_diagnostics(
	output_dir: Path,
	gbm_path: np.ndarray,
	gbm_dt: float,
	gbm_est: object,
	ou_path: np.ndarray,
	ou_dt: float,
	ou_est: object,
) -> None:
	gbm_resid = gbm_standardized_residuals(gbm_path, gbm_est.mu, gbm_est.sigma, gbm_dt)
	ou_resid = ou_standardized_residuals(ou_path, ou_est.theta, ou_est.mu, ou_est.sigma, ou_dt)

	gbm_q = ljung_box_q(gbm_resid, lags=10)
	ou_q = ljung_box_q(ou_resid, lags=10)
	print(f"\nLjung-Box Q(10): GBM={gbm_q:.3f}, OU={ou_q:.3f}; chi-square(10,0.95)≈18.307")

	gbm_theo, gbm_emp = qq_data_normal(gbm_resid)
	ou_theo, ou_emp = qq_data_normal(ou_resid)
	gbm_acf = autocorrelation(gbm_resid, max_lag=20)
	ou_acf = autocorrelation(ou_resid, max_lag=20)

	fig_gbm_qq = plot_qq(gbm_theo, gbm_emp, "GBM residual QQ plot")
	fig_ou_qq = plot_qq(ou_theo, ou_emp, "OU residual QQ plot")
	fig_gbm_acf = plot_acf(gbm_acf, "GBM residual ACF")
	fig_ou_acf = plot_acf(ou_acf, "OU residual ACF")

	fig_gbm_qq.savefig(output_dir / "gbm_residual_qq.png", dpi=130)
	fig_ou_qq.savefig(output_dir / "ou_residual_qq.png", dpi=130)
	fig_gbm_acf.savefig(output_dir / "gbm_residual_acf.png", dpi=130)
	fig_ou_acf.savefig(output_dir / "ou_residual_acf.png", dpi=130)

	plt.close(fig_gbm_qq)
	plt.close(fig_ou_qq)
	plt.close(fig_gbm_acf)
	plt.close(fig_ou_acf)


def run_monte_carlo_study(output_dir: Path, n_replications: int = 300) -> None:
	print(f"\nRunning Monte Carlo study with {n_replications} replications...")

	gbm_params = GBMParams(mu=0.08, sigma=0.22)
	ou_params = OUParams(theta=1.4, mu=1.0, sigma=0.35)

	gbm_s0 = 100.0
	gbm_horizon = 2.0
	gbm_n_steps = 504
	gbm_dt = gbm_horizon / gbm_n_steps

	ou_x0 = -0.4
	ou_horizon = 3.0
	ou_n_steps = 900
	ou_dt = ou_horizon / ou_n_steps

	gbm_rows: list[list[float | str]] = []
	ou_rows: list[list[float | str]] = []
	print_every = max(1, n_replications // 5)

	for i in range(n_replications):
		seed = 1000 + i

		_, gbm_path = euler_maruyama_gbm(
			s0=gbm_s0,
			params=gbm_params,
			horizon=gbm_horizon,
			n_steps=gbm_n_steps,
			n_paths=1,
			seed=seed,
		)
		gbm_est = estimate_gbm_mle(gbm_path[0], dt=gbm_dt)
		gbm_rows.append([float(i), float(seed), gbm_est.mu, gbm_est.sigma, gbm_est.nll])

		_, ou_path = euler_maruyama_ou(
			x0=ou_x0,
			params=ou_params,
			horizon=ou_horizon,
			n_steps=ou_n_steps,
			n_paths=1,
			seed=seed,
		)
		ou_est = estimate_ou_mle(ou_path[0], dt=ou_dt)
		ou_rows.append([float(i), float(seed), ou_est.theta, ou_est.mu, ou_est.sigma, ou_est.nll])

		if (i + 1) % print_every == 0 or i == n_replications - 1:
			print(f"  completed {i + 1}/{n_replications}")

	gbm_arr = np.asarray(gbm_rows, dtype=float)
	ou_arr = np.asarray(ou_rows, dtype=float)

	gbm_summary = {
		"mu": _parameter_summary(gbm_arr[:, 2], gbm_params.mu),
		"sigma": _parameter_summary(gbm_arr[:, 3], gbm_params.sigma),
	}
	ou_summary = {
		"theta": _parameter_summary(ou_arr[:, 2], ou_params.theta),
		"mu": _parameter_summary(ou_arr[:, 3], ou_params.mu),
		"sigma": _parameter_summary(ou_arr[:, 4], ou_params.sigma),
	}

	_write_csv(
		output_dir / "monte_carlo_gbm_raw.csv",
		header=["replication", "seed", "mu_hat", "sigma_hat", "nll"],
		rows=gbm_rows,
	)
	_write_csv(
		output_dir / "monte_carlo_ou_raw.csv",
		header=["replication", "seed", "theta_hat", "mu_hat", "sigma_hat", "nll"],
		rows=ou_rows,
	)

	summary_rows: list[list[float | str]] = []
	for model_name, summary in (("GBM", gbm_summary), ("OU", ou_summary)):
		for param_name, stats in summary.items():
			summary_rows.append(
				[
					model_name,
					param_name,
					stats["true"],
					stats["mean_estimate"],
					stats["bias"],
					stats["std"],
					stats["variance"],
					stats["rmse"],
				]
			)
	_write_csv(
		output_dir / "monte_carlo_summary.csv",
		header=["model", "parameter", "true", "mean_estimate", "bias", "std", "variance", "rmse"],
		rows=summary_rows,
	)

	_save_monte_carlo_histograms(
		output_dir=output_dir,
		gbm_mu=gbm_arr[:, 2],
		gbm_sigma=gbm_arr[:, 3],
		ou_theta=ou_arr[:, 2],
		ou_mu=ou_arr[:, 3],
		ou_sigma=ou_arr[:, 4],
	)

	print("\nMonte Carlo summary (mean ± std, bias, RMSE)")
	for model_name, summary in (("GBM", gbm_summary), ("OU", ou_summary)):
		print(f"{model_name}:")
		for param_name, stats in summary.items():
			print(
				f"  {param_name:>6s}: true={stats['true']:.6f}, mean={stats['mean_estimate']:.6f}, "
				f"std={stats['std']:.6f}, bias={stats['bias']:.6f}, rmse={stats['rmse']:.6f}"
			)


def run_uncertainty_and_coverage(
	output_dir: Path,
	n_replications: int = 80,
	n_bootstrap: int = 80,
) -> None:
	print(f"\nRunning CI coverage study ({n_replications} reps, {n_bootstrap} bootstrap samples)...")

	gbm_params = GBMParams(mu=0.08, sigma=0.22)
	ou_params = OUParams(theta=1.4, mu=1.0, sigma=0.35)

	gbm_s0 = 100.0
	gbm_horizon = 2.0
	gbm_n_steps = 504
	gbm_dt = gbm_horizon / gbm_n_steps

	ou_x0 = -0.4
	ou_horizon = 3.0
	ou_n_steps = 900
	ou_dt = ou_horizon / ou_n_steps

	gbm_asym_hits = {"mu": 0, "sigma": 0}
	gbm_boot_hits = {"mu": 0, "sigma": 0}
	ou_asym_hits = {"theta": 0, "mu": 0, "sigma": 0}
	ou_boot_hits = {"theta": 0, "mu": 0, "sigma": 0}

	for i in range(n_replications):
		seed = 2000 + i

		_, gbm_path = exact_step_gbm(
			s0=gbm_s0,
			params=gbm_params,
			horizon=gbm_horizon,
			n_steps=gbm_n_steps,
			n_paths=1,
			seed=seed,
		)
		gbm_series = gbm_path[0]
		gbm_ci_asym = gbm_asymptotic_ci(gbm_series, dt=gbm_dt)
		gbm_ci_boot = gbm_bootstrap_ci(gbm_series, dt=gbm_dt, n_bootstrap=n_bootstrap, seed=seed + 9999)

		gbm_asym_hits["mu"] += _ci_contains(gbm_ci_asym["mu"], gbm_params.mu)
		gbm_asym_hits["sigma"] += _ci_contains(gbm_ci_asym["sigma"], gbm_params.sigma)
		gbm_boot_hits["mu"] += _ci_contains(gbm_ci_boot["mu"], gbm_params.mu)
		gbm_boot_hits["sigma"] += _ci_contains(gbm_ci_boot["sigma"], gbm_params.sigma)

		_, ou_path = euler_maruyama_ou(
			x0=ou_x0,
			params=ou_params,
			horizon=ou_horizon,
			n_steps=ou_n_steps,
			n_paths=1,
			seed=seed,
		)
		ou_series = ou_path[0]
		ou_ci_asym = ou_asymptotic_ci(ou_series, dt=ou_dt)
		ou_ci_boot = ou_bootstrap_ci(ou_series, dt=ou_dt, n_bootstrap=n_bootstrap, seed=seed + 19999)

		for key in ("theta", "mu", "sigma"):
			true_val = getattr(ou_params, key)
			ou_asym_hits[key] += _ci_contains(ou_ci_asym[key], true_val)
			ou_boot_hits[key] += _ci_contains(ou_ci_boot[key], true_val)

	rows: list[list[float | str]] = []
	for param in ("mu", "sigma"):
		rows.append(["GBM", param, "asymptotic", gbm_asym_hits[param] / n_replications])
		rows.append(["GBM", param, "bootstrap", gbm_boot_hits[param] / n_replications])
	for param in ("theta", "mu", "sigma"):
		rows.append(["OU", param, "asymptotic", ou_asym_hits[param] / n_replications])
		rows.append(["OU", param, "bootstrap", ou_boot_hits[param] / n_replications])

	_write_csv(
		output_dir / "ci_coverage_summary.csv",
		header=["model", "parameter", "ci_type", "empirical_coverage"],
		rows=rows,
	)

	print("CI coverage summary saved to results/ci_coverage_summary.csv")


def run_grid_study(output_dir: Path, n_replications: int = 60) -> None:
	print(f"\nRunning grid study across (Δt, T, N) with {n_replications} reps per cell...")

	gbm_params = GBMParams(mu=0.08, sigma=0.22)
	ou_params = OUParams(theta=1.4, mu=1.0, sigma=0.35)
	dt_grid = [1.0 / 252.0, 1.0 / 126.0, 1.0 / 63.0]
	t_grid = [1.0, 2.0, 3.0]
	n_grid = [1, 5, 20]
	rows: list[list[float | str]] = []

	for n_paths_est in n_grid:
		gbm_matrix = np.zeros((len(dt_grid), len(t_grid)))
		ou_matrix = np.zeros((len(dt_grid), len(t_grid)))

		for i, dt in enumerate(dt_grid):
			for j, horizon in enumerate(t_grid):
				n_steps = max(10, int(round(horizon / dt)))
				dt_eff = horizon / n_steps

				gbm_rel_sq_errors = []
				ou_rel_sq_errors = []

				for rep in range(n_replications):
					seed = 3000 + rep + 100 * i + 1000 * j + 10_000 * n_paths_est
					_, gbm_paths = exact_step_gbm(
						s0=100.0,
						params=gbm_params,
						horizon=horizon,
						n_steps=n_steps,
						n_paths=n_paths_est,
						seed=seed,
					)
					gbm_est = estimate_gbm_mle_panel(gbm_paths, dt=dt_eff)
					gbm_rel_sq_errors.append(
						0.5
						* (
							((gbm_est.mu - gbm_params.mu) / gbm_params.mu) ** 2
							+ ((gbm_est.sigma - gbm_params.sigma) / gbm_params.sigma) ** 2
						)
					)

					_, ou_paths = euler_maruyama_ou(
						x0=-0.4,
						params=ou_params,
						horizon=horizon,
						n_steps=n_steps,
						n_paths=n_paths_est,
						seed=seed,
					)
					ou_est = estimate_ou_mle_panel(ou_paths, dt=dt_eff)
					ou_rel_sq_errors.append(
						(
							((ou_est.theta - ou_params.theta) / ou_params.theta) ** 2
							+ ((ou_est.mu - ou_params.mu) / max(abs(ou_params.mu), 1e-12)) ** 2
							+ ((ou_est.sigma - ou_params.sigma) / ou_params.sigma) ** 2
						)
						/ 3.0
					)

				gbm_matrix[i, j] = math.sqrt(float(np.mean(gbm_rel_sq_errors)))
				ou_matrix[i, j] = math.sqrt(float(np.mean(ou_rel_sq_errors)))

				rows.append(["GBM", dt_eff, horizon, n_paths_est, gbm_matrix[i, j]])
				rows.append(["OU", dt_eff, horizon, n_paths_est, ou_matrix[i, j]])

		x_labels = [f"{t:.1f}" for t in t_grid]
		y_labels = [f"{dt:.4f}" for dt in dt_grid]
		fig_gbm = plot_heatmap(
			matrix=gbm_matrix,
			x_labels=x_labels,
			y_labels=y_labels,
			title=f"GBM relative RMSE heatmap (N={n_paths_est})",
			colorbar_label="Relative RMSE",
		)
		fig_ou = plot_heatmap(
			matrix=ou_matrix,
			x_labels=x_labels,
			y_labels=y_labels,
			title=f"OU relative RMSE heatmap (N={n_paths_est})",
			colorbar_label="Relative RMSE",
		)

		fig_gbm.savefig(output_dir / f"grid_heatmap_gbm_N{n_paths_est}.png", dpi=130)
		fig_ou.savefig(output_dir / f"grid_heatmap_ou_N{n_paths_est}.png", dpi=130)
		plt.close(fig_gbm)
		plt.close(fig_ou)

	_write_csv(
		output_dir / "grid_study_summary.csv",
		header=["model", "dt", "T", "N", "relative_rmse"],
		rows=rows,
	)
	print("Grid study outputs saved to results/")


def run_regime_extension_experiments(output_dir: Path) -> None:
	print("\nRunning extension simulators (jump diffusion, CIR, time-varying coefficients)...")

	horizon = 2.0
	n_steps = 504
	seed = 77

	times, jump_paths = euler_maruyama_jump_diffusion_gbm(
		s0=100.0,
		params=GBMParams(mu=0.07, sigma=0.20),
		jump_intensity=0.8,
		jump_mean=-0.03,
		jump_std=0.12,
		horizon=horizon,
		n_steps=n_steps,
		n_paths=200,
		seed=seed,
	)
	fig_jump = plot_sample_paths(times, jump_paths[:30], "Jump-diffusion GBM sample paths")
	fig_jump.savefig(output_dir / "jump_diffusion_gbm_paths.png", dpi=130)
	plt.close(fig_jump)

	times, cir_paths = euler_maruyama_cir(
		x0=0.04,
		kappa=2.0,
		theta=0.05,
		sigma=0.25,
		horizon=horizon,
		n_steps=n_steps,
		n_paths=200,
		seed=seed + 1,
	)
	fig_cir = plot_sample_paths(times, cir_paths[:30], "CIR sample paths")
	fig_cir.savefig(output_dir / "cir_paths.png", dpi=130)
	plt.close(fig_cir)

	mu_fn = lambda t: 0.06 + 0.02 * np.sin(2.0 * np.pi * t)
	sigma_fn = lambda t: 0.18 + 0.05 * (0.5 + 0.5 * np.cos(2.0 * np.pi * t))
	times, tv_gbm_paths = euler_maruyama_gbm_time_varying(
		s0=100.0,
		mu_fn=mu_fn,
		sigma_fn=sigma_fn,
		horizon=horizon,
		n_steps=n_steps,
		n_paths=200,
		seed=seed + 2,
	)
	fig_tv_gbm = plot_sample_paths(times, tv_gbm_paths[:30], "Time-varying GBM sample paths")
	fig_tv_gbm.savefig(output_dir / "time_varying_gbm_paths.png", dpi=130)
	plt.close(fig_tv_gbm)

	theta_fn = lambda t: 1.0 + 0.6 * (0.5 + 0.5 * np.sin(2.0 * np.pi * t))
	ou_mu_fn = lambda t: 0.8 + 0.4 * np.cos(2.0 * np.pi * t)
	ou_sigma_fn = lambda t: 0.25 + 0.1 * (0.5 + 0.5 * np.sin(4.0 * np.pi * t))
	times, tv_ou_paths = euler_maruyama_ou_time_varying(
		x0=0.0,
		theta_fn=theta_fn,
		mu_fn=ou_mu_fn,
		sigma_fn=ou_sigma_fn,
		horizon=horizon,
		n_steps=n_steps,
		n_paths=200,
		seed=seed + 3,
	)
	fig_tv_ou = plot_sample_paths(times, tv_ou_paths[:30], "Time-varying OU sample paths")
	fig_tv_ou.savefig(output_dir / "time_varying_ou_paths.png", dpi=130)
	plt.close(fig_tv_ou)


def run_integrated_options_and_risk(output_dir: Path) -> None:
	"""Integrated options/risk demo backed directly by models.py APIs."""
	print("\nRunning integrated options and portfolio-risk experiments...")

	S0 = 100.0
	K = 105.0
	T = 1.0
	r = 0.05
	sigma = 0.20

	bs_call = black_scholes_call_price(S0=S0, K=K, T=T, r=r, sigma=sigma)
	bs_put = black_scholes_put_price(S0=S0, K=K, T=T, r=r, sigma=sigma)
	bs_digital = black_scholes_digital_call_price(S0=S0, K=K, T=T, r=r, sigma=sigma)
	greeks = black_scholes_greeks(S0=S0, K=K, T=T, r=r, sigma=sigma)

	mc_call = mc_price_european_gbm(
		S0=S0,
		K=K,
		T=T,
		r=r,
		sigma=sigma,
		option="call",
		n_paths=50_000,
		seed=123,
	)
	mc_put = mc_price_european_gbm(
		S0=S0,
		K=K,
		T=T,
		r=r,
		sigma=sigma,
		option="put",
		n_paths=50_000,
		seed=123,
	)
	mc_asian = mc_price_asian_arithmetic_call_gbm(
		S0=S0,
		K=K,
		T=T,
		r=r,
		sigma=sigma,
		n_paths=50_000,
		seed=124,
	)
	mc_barrier = mc_price_barrier_gbm(
		S0=S0,
		K=K,
		T=T,
		r=r,
		sigma=sigma,
		barrier=85.0,
		direction="down",
		knock="out",
		option="call",
		n_paths=60_000,
		seed=125,
	)

	ou_log_params = OUParams(theta=1.2, mu=np.log(100.0), sigma=0.25)
	mc_ou_call = mc_price_european_ou_log_price(
		x0=np.log(S0),
		ou_params=ou_log_params,
		K=K,
		T=T,
		r=r,
		option="call",
		n_paths=50_000,
		seed=321,
	)

	print("Options summary:")
	print(f"  BS call={bs_call:.4f}, MC call={mc_call.price:.4f} +/- {1.96 * mc_call.se:.4f}")
	print(f"  BS put ={bs_put:.4f}, MC put ={mc_put.price:.4f} +/- {1.96 * mc_put.se:.4f}")
	print(f"  BS digital call={bs_digital:.4f}")
	print(f"  Asian arithmetic call (MC)={mc_asian.price:.4f} +/- {1.96 * mc_asian.se:.4f}")
	print(f"  Down-and-out call (MC)={mc_barrier.price:.4f} +/- {1.96 * mc_barrier.se:.4f}")
	print(f"  OU mean-reverting log-price call (MC)={mc_ou_call.price:.4f} +/- {1.96 * mc_ou_call.se:.4f}")
	print(
		"  Greeks: "
		f"delta(call)={greeks['call_delta']:.4f}, gamma={greeks['gamma']:.5f}, "
		f"vega={greeks['vega']:.4f}, theta(call)={greeks['call_theta']:.4f}"
	)

	strikes = np.arange(70.0, 141.0, 5.0)
	bs_calls = np.array([black_scholes_call_price(S0=S0, K=float(k), T=T, r=r, sigma=sigma) for k in strikes])
	mc_calls = np.array(
		[
			mc_price_european_gbm(
				S0=S0,
				K=float(k),
				T=T,
				r=r,
				sigma=sigma,
				option="call",
				n_paths=20_000,
				seed=700 + i,
			).price
			for i, k in enumerate(strikes)
		]
	)

	fig_chain, ax_chain = plt.subplots(figsize=(9, 5))
	ax_chain.plot(strikes, bs_calls, "o-", label="BS call")
	ax_chain.plot(strikes, mc_calls, "x--", label="MC call")
	ax_chain.axvline(S0, color="gray", linestyle=":", linewidth=1.0)
	ax_chain.set_title("Integrated option chain (GBM): MC vs Black-Scholes")
	ax_chain.set_xlabel("Strike")
	ax_chain.set_ylabel("Price")
	ax_chain.grid(True, alpha=0.25)
	ax_chain.legend()
	fig_chain.tight_layout()
	fig_chain.savefig(output_dir / "integrated_option_chain.png", dpi=130)
	plt.close(fig_chain)

	S0s = np.array([100.0, 100.0, 100.0])
	mus = np.array([0.08, 0.06, 0.10])
	sigmas = np.array([0.20, 0.15, 0.25])
	weights = np.array([0.40, 0.35, 0.25])
	corr = np.array([
		[1.00, 0.45, 0.20],
		[0.45, 1.00, 0.30],
		[0.20, 0.30, 1.00],
	])

	_, corr_paths = simulate_correlated_gbm_paths(
		S0s=S0s,
		mus=mus,
		sigmas=sigmas,
		corr_matrix=corr,
		horizon=1.0,
		n_steps=252,
		n_paths=50_000,
		seed=900,
	)
	pnl = portfolio_pnl(corr_paths, weights)
	rm95 = var_cvar(pnl, alpha=0.95)
	rm99 = var_cvar(pnl, alpha=0.99)

	print("Portfolio risk summary:")
	print(f"  VaR 95%={rm95['VaR']:.4f}, CVaR 95%={rm95['CVaR']:.4f}")
	print(f"  VaR 99%={rm99['VaR']:.4f}, CVaR 99%={rm99['CVaR']:.4f}")

	fig_pnl, ax_pnl = plt.subplots(figsize=(9, 5))
	ax_pnl.hist(pnl, bins=80, density=True, alpha=0.6)
	ax_pnl.axvline(-rm95["VaR"], color="tab:red", linestyle="--", label=f"VaR 95%={rm95['VaR']:.3f}")
	ax_pnl.axvline(-rm95["CVaR"], color="tab:green", linestyle=":", label=f"CVaR 95%={rm95['CVaR']:.3f}")
	ax_pnl.set_title("Integrated portfolio PnL distribution (correlated GBM)")
	ax_pnl.set_xlabel("Portfolio return")
	ax_pnl.set_ylabel("Density")
	ax_pnl.grid(True, alpha=0.25)
	ax_pnl.legend()
	fig_pnl.tight_layout()
	fig_pnl.savefig(output_dir / "integrated_portfolio_pnl.png", dpi=130)
	plt.close(fig_pnl)


def run_langevin_physics_simulation(output_dir: Path) -> None:
	"""Physics simulation: underdamped Langevin particle with OU velocity."""
	print("\nRunning physics simulation (Langevin particle)...")

	params = LangevinParams(gamma=1.8, sigma=1.2)
	x0 = 0.0
	v0 = 0.8
	horizon = 8.0
	n_steps = 1200
	n_paths = 4000
	seed = 2026

	times, positions, velocities = simulate_langevin_1d(
		x0=x0,
		v0=v0,
		params=params,
		horizon=horizon,
		n_steps=n_steps,
		n_paths=n_paths,
		seed=seed,
	)

	v_mean_th, v_var_th = velocity_theoretical_moments(v0=v0, params=params, times=times)
	v_mean_emp = np.mean(velocities, axis=0)
	v_var_emp = np.var(velocities, axis=0)
	x_mean_emp = np.mean(positions, axis=0)
	x_var_emp = np.var(positions, axis=0)

	print("Langevin summary:")
	print(f"  gamma={params.gamma:.3f}, sigma={params.sigma:.3f}, horizon={horizon:.2f}, n_paths={n_paths}")
	print(f"  E[V_T] empirical={v_mean_emp[-1]:.4f}, theory={v_mean_th[-1]:.4f}")
	print(f"  Var[V_T] empirical={v_var_emp[-1]:.4f}, theory={v_var_th[-1]:.4f}")
	print(f"  E[X_T] empirical={x_mean_emp[-1]:.4f}, Var[X_T] empirical={x_var_emp[-1]:.4f}")

	fig_v_paths = plot_sample_paths(times, velocities[:35], "Langevin velocity paths (OU)")
	fig_v_paths.savefig(output_dir / "physics_langevin_velocity_paths.png", dpi=130)
	plt.close(fig_v_paths)

	fig_x_paths = plot_sample_paths(times, positions[:35], "Langevin position paths")
	fig_x_paths.savefig(output_dir / "physics_langevin_position_paths.png", dpi=130)
	plt.close(fig_x_paths)

	fig_v_mom, ax_v_mom = plt.subplots(figsize=(9, 5))
	ax_v_mom.plot(times, v_mean_emp, label="Empirical mean velocity")
	ax_v_mom.plot(times, v_mean_th, "--", label="Theoretical mean velocity")
	ax_v_mom.plot(times, v_var_emp, label="Empirical velocity variance")
	ax_v_mom.plot(times, v_var_th, "--", label="Theoretical velocity variance")
	ax_v_mom.set_title("Langevin velocity moments: empirical vs theory")
	ax_v_mom.set_xlabel("Time")
	ax_v_mom.set_ylabel("Moment value")
	ax_v_mom.grid(True, alpha=0.25)
	ax_v_mom.legend()
	fig_v_mom.tight_layout()
	fig_v_mom.savefig(output_dir / "physics_langevin_velocity_moments.png", dpi=130)
	plt.close(fig_v_mom)

	fig_x_final, ax_x_final = plt.subplots(figsize=(9, 5))
	ax_x_final.hist(positions[:, -1], bins=80, density=True, alpha=0.7)
	ax_x_final.set_title("Langevin terminal position distribution")
	ax_x_final.set_xlabel("X(T)")
	ax_x_final.set_ylabel("Density")
	ax_x_final.grid(True, alpha=0.25)
	fig_x_final.tight_layout()
	fig_x_final.savefig(output_dir / "physics_langevin_terminal_position.png", dpi=130)
	plt.close(fig_x_final)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="SDE simulation and estimation runner")
	parser.add_argument(
		"--mode",
		choices=["single", "study", "all", "full"],
		default="all",
		help="single: one-shot experiments; study: Monte Carlo only; all: single+study; full: all advanced modules",
	)
	parser.add_argument("--mc-reps", type=int, default=300, help="Replications for Monte Carlo estimation study")
	parser.add_argument("--coverage-reps", type=int, default=60, help="Replications for CI coverage study")
	parser.add_argument("--bootstrap-reps", type=int, default=80, help="Bootstrap samples per replication")
	parser.add_argument("--grid-reps", type=int, default=40, help="Replications per (dt,T,N) grid cell")
	parser.add_argument(
		"--with-options",
		action="store_true",
		help="Run integrated options + risk module backed by models.py",
	)
	parser.add_argument(
		"--with-physics",
		action="store_true",
		help="Run physics simulation module (Langevin particle with OU velocity)",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	output_dir = Path("results")
	output_dir.mkdir(parents=True, exist_ok=True)

	if args.mode in ("single", "all", "full"):
		gbm_path, gbm_dt, _gbm_params, gbm_est = run_gbm_experiment(output_dir=output_dir)
		ou_path, ou_dt, _ou_params, ou_est, _ou_exact = run_ou_experiment(output_dir=output_dir)
		run_residual_diagnostics(
			output_dir=output_dir,
			gbm_path=gbm_path,
			gbm_dt=gbm_dt,
			gbm_est=gbm_est,
			ou_path=ou_path,
			ou_dt=ou_dt,
			ou_est=ou_est,
		)

	if args.mode in ("study", "all", "full"):
		run_monte_carlo_study(output_dir=output_dir, n_replications=args.mc_reps)

	if args.mode == "full":
		run_uncertainty_and_coverage(
			output_dir=output_dir,
			n_replications=args.coverage_reps,
			n_bootstrap=args.bootstrap_reps,
		)
		run_grid_study(output_dir=output_dir, n_replications=args.grid_reps)
		run_regime_extension_experiments(output_dir=output_dir)

	if args.with_options:
		run_integrated_options_and_risk(output_dir=output_dir)

	if args.with_physics:
		run_langevin_physics_simulation(output_dir=output_dir)

	print("\nAll requested tasks completed.")


if __name__ == "__main__":
	main()

