"""
Q-Learning Agent
=================
Off-policy TD control.

Update rule:
    Q(s,a) ← Q(s,a) + α * [r + γ*max_a' Q(s',a') - Q(s,a)]

Key property: Q-Learning is OFF-POLICY — it updates toward the GREEDY action
(max over Q), regardless of what action the agent actually took.
This means it learns the optimal policy even while exploring,
but can be overoptimistic (maximisation bias) — that's what Double Q-Learning fixes.
"""

import numpy as np
from typing import Optional


class QLearningAgent:
    """
    Q-Learning agent.

    Parameters
    ----------
    n_states  : int
    n_actions : int
    alpha     : float — learning rate
    epsilon   : float — ε-greedy exploration rate
    gamma     : float — discount factor
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
        self.Q = np.zeros((n_states, n_actions))

    def select_action(self, state_idx: int) -> int:
        """ε-greedy action selection (same as SARSA)."""
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
        Q-Learning update.

        Notice: we only need (s, a, r, s') — no next_action needed
        because we bootstrap from max Q(s', ·).
        """
        target = reward
        if not done:
            target += self.gamma * np.max(self.Q[next_state_idx])

        td_error = target - self.Q[state_idx, action]
        self.Q[state_idx, action] += self.alpha * td_error

    def evaluate(
        self,
        env,
        n_episodes: int = 20,
        max_steps: int = 1000,
        eval_eps: float = 0.01,
    ) -> float:
        """
        Evaluate the policy with near-greedy action selection.
        Returns average steps to goal over n_episodes.
        """
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
        Train Q-Learning for n_episodes, with periodic policy evaluation.
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
            
                self.update(s_idx, action, reward, ns_idx, done)  # ← no next_action param
            
                s_idx = ns_idx
                total_reward += reward
                steps += 1
            
                if done:
                    break
        
            episode_lengths.append(steps)
            episode_rewards.append(total_reward)
        
            # ── Periodic evaluation ───────────────────────────────────────────────
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
        self.Q = np.zeros((self.n_states, self.n_actions))