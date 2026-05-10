import numpy as np

from envs.grids import *
from envs.gridworld_a import GridworldA
from agents.qlearning import QLearningAgent
from agents.sarsa import SARSAAgent
from agents.double_qlearning import DoubleQLearningAgent
from utils.plotting import plot_learning_curves

# agent = QLearningAgent(
#     n_states=env.n_states,
#     n_actions=env.n_actions,
#     alpha=0.2,
#     epsilon=0.1,
#     gamma=1.0,
#     seed=0,
# )

# agent = SARSAAgent(
#     n_states=env.n_states,
#     n_actions=env.n_actions,
#     alpha=0.2,
#     epsilon=0.1,
#     # gamma=1.0,
#     seed=0,
# )

# agent = DoubleQLearningAgent(
#     n_states=env.n_states,
#     n_actions=env.n_actions,
#     alpha=0.2,
#     epsilon=0.1,
#     gamma=1.0,
#     seed=0,
# )

grid = gridworld_a()
env = GridworldA(grid)

print(f"Grid: {env.rows}x{env.cols}")
print(f"S={env.start}, G={env.goal}")
print(f"Obstacles: {len(env.obstacles)}")
print(f"State space size: {env.n_states}")
print(f"Action space size: {env.n_actions}")

for AgentClass, name in [
    (SARSAAgent, "SARSA"),
    (QLearningAgent, "Q-Learning"),
    (DoubleQLearningAgent, "Double Q-Learning"),
]:
    agent = AgentClass(env.n_states, env.n_actions, alpha=0.2, epsilon=0.1, seed=0)
    n_episodes = 500

    print(f"\nTraining {name} for {n_episodes} episodes...")
    metrics = agent.train(env, n_episodes=n_episodes)
    
    lengths = metrics["episode_lengths"]
    print(f"\nFirst 10 episodes (steps): {lengths[:10]}")
    print(f"Last 10 episodes (steps):  {lengths[-10:]}")
    print(f"Average over last 50:      {np.mean(lengths[-50:]):.2f}")
    print(f"Min episode length:        {min(lengths)}")

    print("\nEvaluation results (greedy policy):")
    for ep, steps in zip(metrics["eval_episodes"], metrics["eval_steps"]):
        print(f"  After episode {ep}: avg {steps:.1f} steps to goal")

    safe_name = name.replace(" ", "_").replace("-", "_")
    save_name=f"alpha02_{safe_name}_test"

    plot_learning_curves(
        {f"{name} Learning": metrics},
        title=f"{name} on Gridworld A",
        save_name=save_name,
    )