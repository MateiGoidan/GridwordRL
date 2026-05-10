import numpy as np

from envs.grids import gridworld_a, gridworld_b
from envs.gridworld_a import GridworldA
from envs.gridworld_b import GridworldB
from agents.sarsa import SARSAAgent
from agents.qlearning import QLearningAgent
from agents.double_qlearning import DoubleQLearningAgent
from utils.output import save_experiment
from utils.metrics import convergence_episode, path_efficiency, robustness
from utils.plotting import plot_learning_curves

NUMB_EPISODES = 500
NUMB_SEEDS = 8
EVAL_EVERY = 50

ALPHA_SWEEP   = [0.1, 0.5, 0.9]
EPSILON_SWEEP = [0.1, 0.5, 0.8]
ALPHA_FIXED   = 0.1
EPSILON_FIXED = 0.1
GAMMA         = 1.0

ALGORITHMS = [
    ("SARSA",             SARSAAgent),
    ("Q-Learning",        QLearningAgent),
    ("Double Q-Learning", DoubleQLearningAgent),
]

# (name, config-fn, EnvClass, extra env kwargs)
GRIDWORLDS = [
    ("GridA", gridworld_a, GridworldA, {}),
    ("GridB", gridworld_b, GridworldB, {"numb_actions": 4}),
]


def run_one_config(AgentClass, env_factory, alpha, epsilon, numb_seeds):
    runs = []
    for seed in range(numb_seeds):
        env = env_factory()
        agent = AgentClass(
            numb_states=env.numb_states,
            numb_actions=env.numb_actions,
            alpha=alpha,
            epsilon=epsilon,
            gamma=GAMMA,
            seed=seed,
        )
        metrics = agent.train(env, numb_episodes=NUMB_EPISODES, eval_every=EVAL_EVERY)
        runs.append(metrics)
    return runs


def summarise(runs):
    convergence_eps = [
        convergence_episode(r["episode_lengths"]) for r in runs
    ]
    convergence_eps_clean = [c for c in convergence_eps if c is not None]
    
    return {
        "numb_seeds": len(runs),
        "convergence_episodes_per_seed": convergence_eps,
        "convergence_episode_mean": (
            float(np.mean(convergence_eps_clean)) if convergence_eps_clean else None
        ),
        "path_efficiency_mean": float(np.mean([
            path_efficiency(r["episode_lengths"]) for r in runs
        ])),
        "robustness_std": robustness([r["episode_lengths"] for r in runs]),
    }

def safe_filename(s: str) -> str:
    """Make a string safe for use in a filename."""
    return s.replace(" ", "_").replace("-", "_")

def run_algorithm_comparison(grid_name, env_factory, alpha, epsilon, numb_seeds):
    """
    Run all three algorithms with the SAME hyperparameters and produce
    one comparison plot + JSON.
    """
    print(f"\n=== Algorithm comparison on {grid_name} (alpha={alpha}, epsilon={epsilon}) ===")
    
    results = {}
    summaries = {}
    for algo_name, AgentClass in ALGORITHMS:
        print(f"  Running {algo_name}...", end=" ")
        runs = run_one_config(AgentClass, env_factory, alpha, epsilon, numb_seeds)
        results[algo_name] = runs
        summaries[algo_name] = summarise(runs)
        print(
            f"convergence≈{summaries[algo_name]['convergence_episode_mean']}, "
            f"path_eff={summaries[algo_name]['path_efficiency_mean']:.1f}, "
            f"robustness={summaries[algo_name]['robustness_std']:.2f}"
        )
    
    fname = f"{grid_name}_algorithm_comparison"
    save_experiment(fname, {
        "grid":           grid_name,
        "experiment":     "algorithm_comparison",
        "hyperparameters": {"alpha": alpha, "epsilon": epsilon, "gamma": GAMMA},
        "numb_episodes":     NUMB_EPISODES,
        "numb_seeds":        numb_seeds,
        "results":        results,
        "summaries":      summaries,
    })
    
    plot_learning_curves(
        results,
        title=f"Algorithm Comparison on {grid_name} (alpha={alpha}, epsilon={epsilon})",
        save_name=fname,
    )

def main():
    print("=" * 70)
    print("TASK 1 — Full Experiment Sweep")
    print(f"Episodes: {NUMB_EPISODES} | Seeds: {NUMB_SEEDS} | Eval every: {EVAL_EVERY}")
    print("=" * 70)
    
    for grid_name, grid_config_fn, EnvClass, env_kwargs in GRIDWORLDS:
        print(f"\n{'═' * 70}\n  GRIDWORLD: {grid_name}\n{'═' * 70}")
        
        grid_config = grid_config_fn()
        env_factory = lambda: EnvClass(grid_config, **env_kwargs)
        
        for sweep_name, sweep_values, fixed_param, fixed_value in [
            ("alpha values",   ALPHA_SWEEP,   "epsilon", EPSILON_FIXED),
            ("epsilon values", EPSILON_SWEEP, "alpha",   ALPHA_FIXED),
        ]:
            for algo_name, AgentClass in ALGORITHMS:
                print(f"\n--- {grid_name} / {algo_name} / {sweep_name} ---")
                
                results_per_value = {}
                for value in sweep_values:
                    if sweep_name == "alpha_sweep":
                        alpha, epsilon = value, fixed_value
                    else:
                        alpha, epsilon = fixed_value, value
                    
                    print(f"  alpha={alpha}, epsilon={epsilon}, {NUMB_SEEDS} seeds...", end=" ")
                    runs = run_one_config(
                        AgentClass, env_factory, alpha, epsilon, NUMB_SEEDS
                    )
                    summary = summarise(runs)
                    print(
                        f"convergence≈{summary['convergence_episode_mean']}, "
                        f"path_eff={summary['path_efficiency_mean']:.1f}, "
                        f"robustness={summary['robustness_std']:.2f}"
                    )
                    
                    results_per_value[str(value)] = {
                        "alpha": alpha,
                        "epsilon": epsilon,
                        "runs": runs,
                        "summary": summary,
                    }
                
                # ── Save JSON ─────────────────────────────────────────────────
                fname = f"{grid_name}_{safe_filename(algo_name)}_{sweep_name}"
                save_experiment(fname, {
                    "grid":       grid_name,
                    "algorithm":  algo_name,
                    "sweep":      sweep_name,
                    "fixed":      {fixed_param: fixed_value, "gamma": GAMMA},
                    "numb_episodes": NUMB_EPISODES,
                    "numb_seeds":    NUMB_SEEDS,
                    "results_per_value": results_per_value,
                })
                
                # ── Plot: one line per swept value (3 lines per plot) ─────────
                swept_param = sweep_name.split("_")[0]    # "alpha" or "epsilon"
                plot_data = {
                    f"{swept_param}={v}": results_per_value[str(v)]["runs"]
                    for v in sweep_values
                }
                plot_learning_curves(
                    plot_data,
                    title=f"{algo_name} on {grid_name} with {sweep_name}",
                    save_name=fname,
                )
    
    print("\n" + "=" * 70)
    print("All experiments complete. Results in results/experiments/")
    print("=" * 70)

    print(f"\n{'═' * 70}\n  ALGORITHM COMPARISONS\n{'═' * 70}")
    
    COMPARISON_ALPHA   = 0.5    
    COMPARISON_EPSILON = 0.1
    
    for grid_name, grid_config_fn, EnvClass, env_kwargs in GRIDWORLDS:
        grid_config = grid_config_fn()
        env_factory = lambda: EnvClass(grid_config, **env_kwargs)
        run_algorithm_comparison(
            grid_name, env_factory,
            COMPARISON_ALPHA, COMPARISON_EPSILON, NUMB_SEEDS,
        )

if __name__ == "__main__":
    main()