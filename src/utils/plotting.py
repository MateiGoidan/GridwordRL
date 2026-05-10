"""
Plotting utilities for MAS HW1.
Provides functions to generate the required learning curve plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def smooth(values: list, window: int = 20) -> np.ndarray:
    """Running mean smoothing for noisy learning curves."""
    arr = np.array(values, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def plot_learning_curves(
    results: dict,          # {"Agent Name": {"episode_lengths": [...], ...}}
    title: str = "Learning Curves",
    metric: str = "episode_lengths",
    ylabel: str = "Steps per Episode",
    smooth_window: int = 20,
    save_name: str | None = None,
) -> None:
    """
    Plot smoothed learning curves for multiple agents on the same axes.

    Parameters
    ----------
    results      : dict mapping agent name → metrics dict
    title        : plot title
    metric       : which metric to plot
    ylabel       : y-axis label
    smooth_window: rolling average window size
    save_name    : filename (without extension) to save to results/
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for agent_name, data in results.items():
        raw = data[metric]
        smoothed = smooth(raw, window=smooth_window)
        episodes = np.arange(smooth_window - 1, len(raw))
        ax.plot(episodes, smoothed, label=agent_name, linewidth=1.8)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_name:
        path = RESULTS_DIR / f"{save_name}.png"
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")

    plt.show()


def plot_hyperparameter_comparison(
    param_name: str,
    param_values: list,
    results_per_value: list,    # list of dicts {"episode_lengths": [...]}
    agent_name: str = "Agent",
    metric: str = "episode_lengths",
    ylabel: str = "Steps per Episode",
    smooth_window: int = 20,
    save_name: str | None = None,
) -> None:
    """
    Plot the effect of varying a single hyperparameter for one agent.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for val, data in zip(param_values, results_per_value):
        raw = data[metric]
        smoothed = smooth(raw, window=smooth_window)
        episodes = np.arange(smooth_window - 1, len(raw))
        ax.plot(episodes, smoothed, label=f"{param_name}={val}", linewidth=1.8)

    ax.set_title(f"{agent_name} — effect of {param_name}", fontsize=13)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_name:
        path = RESULTS_DIR / f"{save_name}.png"
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")

    plt.show()