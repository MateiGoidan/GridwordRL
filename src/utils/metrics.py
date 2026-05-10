import numpy as np

def convergence_episode(
    episode_lengths: list[int],
    threshold_factor: float = 1.5,
    optimal_estimate: int | None = None,
    window: int = 50,
) -> int | None:
    if optimal_estimate is None:
        optimal_estimate = min(episode_lengths)
    threshold = threshold_factor * optimal_estimate
    
    arr = np.array(episode_lengths)
    if len(arr) < window:
        return None
    
    kernel = np.ones(window) / window
    rolling = np.convolve(arr, kernel, mode="valid")
    
    below = np.where(rolling <= threshold)[0]
    return int(below[0] + window - 1) if len(below) > 0 else None

def path_efficiency(episode_lengths: list[int], last_n: int = 50) -> float:
    if len(episode_lengths) == 0:
        return float("inf")
    return float(np.mean(episode_lengths[-last_n:]))

def robustness(per_seed_episode_lengths: list[list[int]], last_n: int = 50) -> float:
    if len(per_seed_episode_lengths) < 2:
        return 0.0
    final_per_seed = [path_efficiency(el, last_n) for el in per_seed_episode_lengths]
    return float(np.std(final_per_seed))