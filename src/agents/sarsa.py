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
        return int(np.argmax(self.Q[state_idx]))

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

    def evaluate(self, env, n_episodes: int = 20, max_steps: int = 1000, eval_eps: float = 0.01) -> float:
        """
        Evaluate the current policy with PURE GREEDY action selection (no exploration).
    
        Parameters
        ----------
        env        : the environment to evaluate on
        n_episodes : how many evaluation episodes to average over
        max_steps  : safety cap — if the policy is bad and gets stuck, give up after this
    
        Returns
        -------
        float : average number of steps to reach the goal across `n_episodes`.
            Episodes that hit `max_steps` count as `max_steps` (penalising bad policies).
        """
        # Save the training epsilon so we can restore it after evaluation
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
    
        # Restore original epsilon
        self.epsilon = saved_epsilon
        return total_steps / n_episodes

    # ── Training loop ──────────────────────────────────────────────────────────

    def train(
        self,
        env,
        n_episodes: int,
        eval_every: int = 50,
        eval_episodes: int = 20,
    ) -> dict:
        """
        Train SARSA for n_episodes, with periodic policy evaluation.
    
        Parameters
        ----------
        env           : environment to train on
        n_episodes    : total training episodes
        eval_every    : run a greedy evaluation every N training episodes
        eval_episodes : how many episodes to average per evaluation
    
        Returns
        -------
        dict with:
            'episode_lengths' : list[int]   — steps per training episode (with ε exploration)
            'episode_rewards' : list[float] — total reward per training episode
            'eval_episodes'   : list[int]   — episode number when each eval was done
            'eval_steps'      : list[float] — avg greedy steps to goal at each eval point
        """
        episode_lengths = []
        episode_rewards = []
        eval_episodes_at = []
        eval_steps = []
    
        for ep in range(n_episodes):
            # ── Standard SARSA training episode ────────────────────────────────────
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
        """Reset Q-table to zeros (useful for fresh runs with new hyperparams)."""
        self.Q = np.zeros((self.n_states, self.n_actions))
