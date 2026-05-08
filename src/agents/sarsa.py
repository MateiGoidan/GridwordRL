"""
SARSA Agent
============
On-policy TD control.

Update rule:
    Q(s,a) ← Q(s,a) + α * [r + γ*Q(s',a') - Q(s,a)]

Key property: SARSA is ON-POLICY — it updates using the action a' that was
actually chosen by the current (epsilon-greedy) policy. This means the agent
"lives with the consequences" of its own exploration, making it more
conservative near dangerous states.
"""

import numpy as np
from typing import Optional


class SARSAAgent:
    """
    SARSA (State-Action-Reward-State-Action) agent.

    Parameters
    ----------
    n_states  : int   — number of states in the environment
    n_actions : int   — number of available actions
    alpha     : float — learning rate  (e.g. 0.1, 0.3, 0.5)
    epsilon   : float — exploration rate for ε-greedy policy (e.g. 0.1)
    gamma     : float — discount factor (1.0 for undiscounted episodic tasks)
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

        # Q-table: shape (n_states, n_actions), initialised to 0 as per homework
        self.Q = np.zeros((n_states, n_actions))

    # ── Policy ─────────────────────────────────────────────────────────────────

    def select_action(self, state_idx: int) -> int:
        """
        ε-greedy action selection.

        With probability ε  → pick a random action (explore)
        With probability 1-ε → pick argmax Q(s, ·) (exploit)
        Ties in argmax are broken randomly.
        """
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        q_values = self.Q[state_idx]
        max_q = np.max(q_values)
        # break ties randomly — important for symmetrical initial Q=0
        best_actions = np.where(q_values == max_q)[0]
        return int(self.rng.choice(best_actions))

    # ── Learning ───────────────────────────────────────────────────────────────

    def update(
        self,
        state_idx: int,
        action: int,
        reward: float,
        next_state_idx: int,
        next_action: int,
        done: bool,
    ) -> None:
        """
        SARSA update.

        If the episode is done, the bootstrap target is just `reward`
        (no next state value), which correctly handles the terminal condition.
        """
        target = reward
        if not done:
            target += self.gamma * self.Q[next_state_idx, next_action]

        td_error = target - self.Q[state_idx, action]
        self.Q[state_idx, action] += self.alpha * td_error

    # ── Training loop ──────────────────────────────────────────────────────────

    def train(self, env, n_episodes: int) -> dict:
        """
        Run SARSA for n_episodes and collect metrics.

        Returns
        -------
        dict with keys:
          'episode_lengths' : list[int]   — steps per episode
          'episode_rewards' : list[float] — cumulative reward per episode
        """
        episode_lengths = []
        episode_rewards = []

        for _ in range(n_episodes):
            state = env.reset()
            s_idx = env.state_to_index(state)
            action = self.select_action(s_idx)

            total_reward = 0.0
            steps = 0

            while True:
                next_state, reward, done = env.step(action)
                ns_idx = env.state_to_index(next_state)
                next_action = self.select_action(ns_idx)

                self.update(s_idx, action, reward, ns_idx, next_action, done)

                s_idx = ns_idx
                action = next_action
                total_reward += reward
                steps += 1

                if done:
                    break

            episode_lengths.append(steps)
            episode_rewards.append(total_reward)

        return {
            "episode_lengths": episode_lengths,
            "episode_rewards": episode_rewards,
        }

    def reset_q(self) -> None:
        """Reset Q-table to zeros (useful for fresh runs with new hyperparams)."""
        self.Q = np.zeros((self.n_states, self.n_actions))
