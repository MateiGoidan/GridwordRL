from envs.generator import generate_gridworld_a
from envs.gridworld_a import GridworldA
from agents.sarsa import SARSAAgent
from utils.plotting import plot_learning_curves

# ── 1. Generate environment ────────────────────────────────────────────────────
config = generate_gridworld_a(seed=42)
env = GridworldA(config)

print(f"Grid: {env.rows}x{env.cols}")
print(f"S={env.start}, G={env.goal}")
print(f"Obstacles: {len(env.obstacles)}")
print(f"State space size: {env.n_states}")
print(f"Action space size: {env.n_actions}")

# ── 2. Create and train agent ──────────────────────────────────────────────────
agent = SARSAAgent(
    n_states=env.n_states,
    n_actions=env.n_actions,
    alpha=0.2,
    epsilon=0.1,
    gamma=1.0,
    seed=0,
)

print("\nTraining SARSA for 500 episodes...")
metrics = agent.train(env, n_episodes=500)

# ── 3. Quick stats ─────────────────────────────────────────────────────────────
import numpy as np
lengths = metrics["episode_lengths"]
print(f"\nFirst 10 episodes (steps): {lengths[:10]}")
print(f"Last 10 episodes (steps):  {lengths[-10:]}")
print(f"Average over last 50:      {np.mean(lengths[-50:]):.2f}")
print(f"Min episode length:        {min(lengths)}")

# ── 4. Plot learning curve ─────────────────────────────────────────────────────
plot_learning_curves(
    {"SARSA": metrics},
    title=f"SARSA on Gridworld A (α=0.2, ε=0.1)",
    save_name="first_training_run",
)