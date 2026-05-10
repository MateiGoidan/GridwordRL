import numpy as np

from utils.plotting import *
from envs.grids import *
from envs.gridworld_a import GridworldA
from envs.gridworld_b import GridworldB
from agents.qlearning import QLearningAgent
from agents.sarsa import SARSAAgent
from agents.double_qlearning import DoubleQLearningAgent

grid = gridworld_a()
env = GridworldA(grid)

print(f"Grid: {env.rows}x{env.cols}")
print(f"S={env.start}, G={env.goal}")
print(f"Obstacles: {len(env.obstacles)}")
print(f"State space size: {env.numb_states}")
print(f"Action space size: {env.numb_actions}")

for AgentClass, name in [
    (SARSAAgent, "SARSA"),
    (QLearningAgent, "Q-Learning"),
    (DoubleQLearningAgent, "Double Q-Learning"),
]:
    agent = AgentClass(env.numb_states, env.numb_actions, alpha=0.2, epsilon=0.1, seed=0)
    numb_episodes = 500

    print(f"\nTraining {name} for {numb_episodes} episodes...")
    metrics = agent.train(env, numb_episodes)
    
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