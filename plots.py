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
