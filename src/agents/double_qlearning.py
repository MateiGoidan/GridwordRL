"""
Double Q-Learning Agent
========================
Off-policy TD control with two Q-tables to reduce maximisation bias.

The problem with Q-Learning: when all Q values are 0 (or noisy), taking
max Q(s', ·) consistently overestimates the true value — this is called
maximisation bias. In stochastic environments it can cause the agent to
think risky paths are better than they are.

Double Q-Learning fix:
  - Maintain two Q-tables: Q_A and Q_B
  - On each update, randomly choose one to UPDATE and use the OTHER to EVALUATE:

    With 50% probability:
      a* = argmax_a  Q_A(s', a)          ← use Q_A to SELECT action
      Q_A(s,a) ← Q_A(s,a) + α * [r + γ*Q_B(s', a*) - Q_A(s,a)]   ← use Q_B to EVALUATE

    Otherwise swap A and B.

  This decoupling of selection and evaluation removes the upward bias.
"""

import numpy as np
from typing import Optional


class DoubleQLearningAgent:
    """
    Double Q-Learning agent.

    Parameters match QLearningAgent exactly — drop-in replacement.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        epsilon: float = 0.1,
        gamma: float = 1.0,
        seed: Optional[int] = None,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)

        # Two independent Q-tables, both initialised to 0
        self.Q_A = np.zeros((n_states, n_actions))
        self.Q_B = np.zeros((n_states, n_actions))

    @property
    def Q(self) -> np.ndarray:
        """Combined Q estimate used for action selection and policy evaluation."""
        return self.Q_A + self.Q_B

    def select_action(self, state_idx: int) -> int:
        """ε-greedy on the COMBINED Q = Q_A + Q_B."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.Q[state_idx]))

    def update(
        self,
        state_idx: int,
        action: int,
        reward: float,
        next_state_idx: int,
        done: bool,
    ) -> None:
        """
        Double Q-Learning update.

        We flip a fair coin each step to decide which table gets updated.
        """
        if self.rng.random() < 0.5:
            # Update Q_A, evaluate with Q_B
            if done:
                target = reward
            else:
                best_action = int(np.argmax(self.Q_A[next_state_idx]))
                target = reward + self.gamma * self.Q_B[next_state_idx, best_action]
            self.Q_A[state_idx, action] += self.alpha * (target - self.Q_A[state_idx, action])
        else:
            # Update Q_B, evaluate with Q_A
            if done:
                target = reward
            else:
                best_action = int(np.argmax(self.Q_B[next_state_idx]))
                target = reward + self.gamma * self.Q_A[next_state_idx, best_action]
            self.Q_B[state_idx, action] += self.alpha * (target - self.Q_B[state_idx, action])

    def evaluate(
        self,
        env,
        n_episodes: int = 20,
        max_steps: int = 1000,
        eval_eps: float = 0.01,
    ) -> float:
        saved_epsilon = self.epsilon
        self.epsilon = eval_eps
    
        total_steps = 0
        for _ in range(n_episodes):
            state = env.reset()
            s_idx = env.state_to_index(state)
            steps = 0
        
            while steps < max_steps:
                action = self.select_action(s_idx)
                next_state, _, done = env.step(action)
                s_idx = env.state_to_index(next_state)
                steps += 1
                if done:
                    break
        
            total_steps += steps
    
        self.epsilon = saved_epsilon
        return total_steps / n_episodes

    def train(
        self,
        env,
        n_episodes: int,
        eval_every: int = 50,
        eval_episodes: int = 20,
    ) -> dict:
        """
        Train Double Q-Learning for n_episodes, with periodic policy evaluation.
        """
        episode_lengths = []
        episode_rewards = []
        eval_episodes_at = []
        eval_steps = []
    
        for ep in range(n_episodes):
            state = env.reset()
            s_idx = env.state_to_index(state)
            total_reward = 0.0
            steps = 0
        
            while True:
                action = self.select_action(s_idx)
                next_state, reward, done = env.step(action)
                ns_idx = env.state_to_index(next_state)
            
                self.update(s_idx, action, reward, ns_idx, done)
            
                s_idx = ns_idx
                total_reward += reward
                steps += 1
            
                if done:
                    break
        
            episode_lengths.append(steps)
            episode_rewards.append(total_reward)
        
            if (ep + 1) % eval_every == 0:
                avg_steps = self.evaluate(env, n_episodes=eval_episodes)
                eval_episodes_at.append(ep + 1)
                eval_steps.append(avg_steps)
    
        return {
            "episode_lengths": episode_lengths,
            "episode_rewards": episode_rewards,
            "eval_episodes":   eval_episodes_at,
            "eval_steps":      eval_steps,
        }

    def reset_q(self) -> None:
        self.Q_A = np.zeros((self.n_states, self.n_actions))
        self.Q_B = np.zeros((self.n_states, self.n_actions))
