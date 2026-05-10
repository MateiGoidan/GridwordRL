import numpy as np

from utils.plotting import *
from envs.grids import *
from envs.gridworld_a import GridworldA
from envs.gridworld_b import GridworldB
from agents.qlearning import QLearningAgent
from agents.sarsa import SARSAAgent
from agents.double_qlearning import DoubleQLearningAgent

results = {}
for AgentClass, name in [
    (SARSAAgent, "SARSA"),
    (QLearningAgent, "Q-Learning"),
    (DoubleQLearningAgent, "Double Q-Learning"),
]:
    grid = gridworld_a()
    env = GridworldA(grid)
    agent = AgentClass(env.numb_states, env.numb_actions, alpha=0.2, epsilon=0.1, seed=0)
    numb_episodes = 500

    print(f"\nTraining {name} for {numb_episodes} episodes...")
    metrics = agent.train(env, numb_episodes)
    results[name] = metrics
    
    lengths = metrics["episode_lengths"]
    print(f"\nFirst 10 episodes (steps): {lengths[:10]}")
    print(f"Last 10 episodes (steps):  {lengths[-10:]}")
    print(f"Average over last 50:      {np.mean(lengths[-50:]):.2f}")
    print(f"Min episode length:        {min(lengths)}")

    print("\nEvaluation results (greedy policy):")
    for ep, steps in zip(metrics["eval_episodes"], metrics["eval_steps"]):
        print(f"  After episode {ep}: avg {steps:.1f} steps to goal")

plot_learning_curves(
    results,
    title="Gridworld A Comparison",
    save_name="gridworld_a_comparison",
)