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

def path_efficiency(eval_steps: list[float], last_n: int = 3) -> float:
    if len(eval_steps) == 0:
        return float("inf")
    return float(np.mean(eval_steps[-last_n:]))

def robustness(per_seed_eval_steps: list[list[float]], last_n: int = 3) -> float:
    if len(per_seed_eval_steps) < 2:
        return 0.0
    final_per_seed = [path_efficiency(es, last_n) for es in per_seed_eval_steps]
    return float(np.std(final_per_seed))