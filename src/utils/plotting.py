import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
RESULTS_DIR.mkdir(exist_ok=True)


ALGORITHM_COLORS = {
    "SARSA":             "blue",
    "Q-Learning":        "orange",
    "Double Q-Learning": "red",
}


def smooth_per_run(runs: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window) / window
    return np.array([np.convolve(run, kernel, mode="valid") for run in runs])


def plot_learning_curves(
    results: dict,
    title: str = "Learning Curves",
    metric: str = "episode_lengths",
    ylabel: str = "Steps per Episode",
    window: int = 50,
    save_name: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for label, runs_or_single in results.items():
        color = ALGORITHM_COLORS.get(label)
        
        if isinstance(runs_or_single, dict):
            runs = np.array([runs_or_single[metric]])
        else:
            runs = np.array([r[metric] for r in runs_or_single])
        
        if runs.shape[1] < window:
            print(f"{label}: not enough data ({runs.shape[1]} < window={window})")
            continue
        
        smoothed = smooth_per_run(runs, window)         
        mean = smoothed.mean(axis=0)
        std = smoothed.std(axis=0)
        x = np.arange(window - 1, window - 1 + len(mean))
        
        line, = ax.plot(x, mean, color=color, label=label, linewidth=2)
        if runs.shape[0] > 1:
            ax.fill_between(
            x, mean - std, mean + std,
            alpha=0.2,
            color=line.get_color(),   
        )
    
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel(f"{ylabel}", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_name:
        path = RESULTS_DIR / f"{save_name}.png"
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")
    # plt.show()