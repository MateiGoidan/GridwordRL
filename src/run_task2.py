import numpy as np

from envs.grids import gridworld_b
from envs.gridworld_b import GridworldB
from agents.sarsa import SARSAAgent
from utils.output import save_experiment
from utils.metrics import convergence_episode, path_efficiency, robustness
from utils.plotting import plot_learning_curves


NUMB_EPISODES = 500
NUMB_SEEDS    = 5         
EVAL_EVERY = 50

ALPHA   = 0.1          
EPSILON = 0.1
GAMMA   = 1.0


def run_sarsa_on_gridB(numb_actions: int, numb_seeds: int) -> list:
    grid_config = gridworld_b()
    runs = []
    for seed in range(numb_seeds):
        env = GridworldB(grid_config, numb_actions=numb_actions)
        agent = SARSAAgent(
            numb_states=env.numb_states,
            numb_actions=env.numb_actions,
            alpha=ALPHA,
            epsilon=EPSILON,
            gamma=GAMMA,
            seed=seed,
        )
        metrics = agent.train(env, numb_episodes=NUMB_EPISODES, eval_every=EVAL_EVERY)
        runs.append(metrics)
    return runs


def summarise(runs: list) -> dict:
    """Compute the homework metrics from a list of seed runs."""
    convergence_eps = [
        convergence_episode(r["episode_lengths"]) for r in runs
    ]
    convergence_eps_clean = [c for c in convergence_eps if c is not None]
    
    return {
        "numb_seeds": len(runs),
        "convergence_episode_mean": (
            float(np.mean(convergence_eps_clean)) if convergence_eps_clean else None
        ),
        "path_efficiency_mean": float(np.mean([
            path_efficiency(r["episode_lengths"]) for r in runs
        ])),
        "robustness_std": robustness([r["episode_lengths"] for r in runs]),
    }


def main():
    print("=" * 70)
    print("TASK 2 — SARSA on Gridworld B: 4 vs 8 actions")
    print(f"Episodes: {NUMB_EPISODES} | Seeds: {NUMB_SEEDS} | alpha={ALPHA}, epsilon={EPSILON}")
    print("=" * 70)
    
    results = {}
    summaries = {}
    for numb_actions in [4, 8]:
        label = f"SARSA ({numb_actions} actions)"
        print(f"\nRunning {label}...", end=" ")
        runs = run_sarsa_on_gridB(numb_actions, NUMB_SEEDS)
        results[label] = runs
        summaries[label] = summarise(runs)
        print(
            f"convergence≈{summaries[label]['convergence_episode_mean']}, "
            f"path_eff={summaries[label]['path_efficiency_mean']:.1f}, "
            f"robustness={summaries[label]['robustness_std']:.2f}"
        )
    
    # ── Save JSON ─────────────────────────────────────────────────────────────
    save_experiment("task2_sarsa_4_vs_8_actions", {
        "task":            "Task 2 — SARSA 4 vs 8 actions on Gridworld B",
        "hyperparameters": {"alpha": ALPHA, "epsilon": EPSILON, "gamma": GAMMA},
        "numb_episodes":      NUMB_EPISODES,
        "numb_seeds":         NUMB_SEEDS,
        "results":         results,
        "summaries":       summaries,
    })
    
    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_learning_curves(
        results,
        title=f"Task 2 — SARSA on Gridworld B: 4 vs 8 actions (alpha={ALPHA}, epsilon={EPSILON})",
        save_name="task2_sarsa_4_vs_8_actions",
    )
    
    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Final comparison:")
    for label, summary in summaries.items():
        print(f"  {label}:")
        print(f"    Convergence at: episode {summary['convergence_episode_mean']}")
        print(f"    Path efficiency: {summary['path_efficiency_mean']:.2f} steps")
        print(f"    Robustness (std): {summary['robustness_std']:.2f}")
    
    eff_4 = summaries["SARSA (4 actions)"]["path_efficiency_mean"]
    eff_8 = summaries["SARSA (8 actions)"]["path_efficiency_mean"]
    if eff_4 > 0:
        improvement = (eff_4 - eff_8) / eff_4 * 100
        print(f"\n  → 8-action SARSA is {improvement:.1f}% more efficient than 4-action SARSA.")
    
    print("=" * 70)


if __name__ == "__main__":
    main()