from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


Array = np.ndarray


def plot_sample_paths(times: Array, paths: Array, title: str, max_paths: int = 20) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    n_show = min(paths.shape[0], max_paths)

    for i in range(n_show):
        ax.plot(times, paths[i], linewidth=1.2, alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_terminal_histogram(paths: Array, title: str, bins: int = 40) -> plt.Figure:
    terminal = paths[:, -1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(terminal, bins=bins, density=True, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Terminal value")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_hitting_time_histogram(hitting_times: Array, title: str, bins: int = 30) -> plt.Figure:
    finite = hitting_times[np.isfinite(hitting_times)]
    fig, ax = plt.subplots(figsize=(8, 5))

    if finite.size > 0:
        ax.hist(finite, bins=bins, alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel("Hitting time")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_qq(theoretical_quantiles: Array, empirical_quantiles: Array, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(theoretical_quantiles, empirical_quantiles, alpha=0.65, s=12)

    if len(theoretical_quantiles) > 0:
        lower = min(theoretical_quantiles.min(), empirical_quantiles.min())
        upper = max(theoretical_quantiles.max(), empirical_quantiles.max())
        ax.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1.2)

    ax.set_title(title)
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Empirical quantiles")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_acf(acf_values: Array, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    lags = np.arange(len(acf_values))
    ax.vlines(lags, 0.0, acf_values, linewidth=1.6)
    ax.scatter(lags, acf_values, s=16)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_heatmap(
    matrix: Array,
    x_labels: list[str],
    y_labels: list[str],
    title: str,
    colorbar_label: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)

    ax.set_title(title)
    ax.set_xlabel("T")
    ax.set_ylabel("Δt")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)

    fig.tight_layout()
    return fig
