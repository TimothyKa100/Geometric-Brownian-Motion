from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis import (
	gbm_log_return_stats,
	ou_increment_stats,
	validate_gbm_estimates,
	validate_ou_estimates,
)
from estimation import estimate_gbm_mle, estimate_ou_mle
from first_passage import first_hitting_times, summarize_hitting_times
from models import GBMParams, OUParams, euler_maruyama_gbm, euler_maruyama_ou
from plots import plot_hitting_time_histogram, plot_sample_paths, plot_terminal_histogram


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


def run_gbm_experiment(output_dir: Path) -> None:
	params = GBMParams(mu=0.08, sigma=0.22)
	s0 = 100.0
	horizon = 2.0
	n_steps = 504
	n_paths = 4000
	seed = 7
	dt = horizon / n_steps

	times, paths = euler_maruyama_gbm(
		s0=s0,
		params=params,
		horizon=horizon,
		n_steps=n_steps,
		n_paths=n_paths,
		seed=seed,
	)

	stats = gbm_log_return_stats(paths)
	print("GBM log-return moments:")
	print(f"mean={stats.mean:.6f}, std={stats.std:.6f}, skew={stats.skew:.6f}, kurtosis={stats.kurtosis:.6f}")

	representative_path = paths[0]
	est = estimate_gbm_mle(representative_path, dt=dt)
	validation = validate_gbm_estimates(params, est)

	_print_estimation_report(
		"GBM",
		true_params={"mu": params.mu, "sigma": params.sigma},
		est_params={"mu": est.mu, "sigma": est.sigma},
	)
	print(f"GBM objective (NLL): {est.nll:.6f}")
	print(
		"Validation summary:",
		{k: {"abs": v.abs_error, "rel": v.rel_error} for k, v in validation.items()},
	)

	upper_barrier = 140.0
	hit_idx = first_hitting_times(paths, level=upper_barrier, above=True)
	hit_summary = summarize_hitting_times(hit_idx, dt=dt)
	finite_times = hit_idx[np.isfinite(hit_idx)] * dt
	print(
		f"GBM first passage to {upper_barrier}: p_hit={hit_summary.hit_probability:.3f}, "
		f"mean_tau={hit_summary.mean_hitting_time:.3f}, median_tau={hit_summary.median_hitting_time:.3f}"
	)

	fig_paths = plot_sample_paths(times, paths[:40], "GBM sample paths (Euler-Maruyama)")
	fig_terminal = plot_terminal_histogram(paths, "GBM terminal distribution")
	fig_hit = plot_hitting_time_histogram(finite_times, f"GBM hitting time histogram (level={upper_barrier})")

	fig_paths.savefig(output_dir / "gbm_paths.png", dpi=130)
	fig_terminal.savefig(output_dir / "gbm_terminal_hist.png", dpi=130)
	fig_hit.savefig(output_dir / "gbm_hitting_times.png", dpi=130)

	plt.close(fig_paths)
	plt.close(fig_terminal)
	plt.close(fig_hit)


def run_ou_experiment(output_dir: Path) -> None:
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
	est = estimate_ou_mle(representative_path, dt=dt)
	validation = validate_ou_estimates(params, est)

	_print_estimation_report(
		"OU",
		true_params={"theta": params.theta, "mu": params.mu, "sigma": params.sigma},
		est_params={"theta": est.theta, "mu": est.mu, "sigma": est.sigma},
	)
	print(f"OU objective (NLL): {est.nll:.6f}")
	print(
		"Validation summary:",
		{k: {"abs": v.abs_error, "rel": v.rel_error} for k, v in validation.items()},
	)

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


def main() -> None:
	output_dir = Path("results")
	output_dir.mkdir(parents=True, exist_ok=True)

	run_gbm_experiment(output_dir=output_dir)
	run_ou_experiment(output_dir=output_dir)

	print("\nSaved figures to ./results")


if __name__ == "__main__":
	main()

