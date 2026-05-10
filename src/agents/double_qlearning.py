import numpy as np
from typing import Optional

class DoubleQLearningAgent:
    def __init__(
        self,
        numb_states: int,
        numb_actions: int,
        alpha: float = 0.1,
        epsilon: float = 0.1,
        gamma: float = 1.0,
        seed: Optional[int] = None,
    ):
        self.numb_states = numb_states
        self.numb_actions = numb_actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)

        self.Q_A = np.zeros((numb_states, numb_actions))
        self.Q_B = np.zeros((numb_states, numb_actions))

    @property
    def Q(self) -> np.ndarray:
        # Combined Q estimate
        return self.Q_A + self.Q_B

    def select_action(self, state_idx: int) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.numb_actions))
        return int(np.argmax(self.Q[state_idx]))

    def update(
        self,
        state_idx: int,
        action: int,
        reward: float,
        next_state_idx: int,
        done: bool,
    ) -> None:
        if self.rng.random() < 0.5:
            target = reward
            if not done:
                best_action = int(np.argmax(self.Q_A[next_state_idx]))
                target += self.gamma * self.Q_B[next_state_idx, best_action]
            error = target - self.Q_A[state_idx, action]
            self.Q_A[state_idx, action] += self.alpha * error 
        else:
            target = reward
            if not done:
                best_action = int(np.argmax(self.Q_B[next_state_idx]))
                target = reward + self.gamma * self.Q_A[next_state_idx, best_action]
            error = target - self.Q_B[state_idx, action]
            self.Q_B[state_idx, action] += self.alpha * error 

    def evaluate(
        self,
        env,
        numb_episodes: int = 20,
        max_steps: int = 1000,
        eval_eps: float = 0.01,
    ) -> float:
        saved_epsilon = self.epsilon
        self.epsilon = eval_eps
    
        total_steps = 0
        for _ in range(numb_episodes):
            state = env.reset()
            state_idx = env.get_index(state)
            steps = 0
        
            while steps < max_steps:
                action = self.select_action(state_idx)
                next_state, _, done = env.step(action)
                state_idx = env.get_index(next_state)
                steps += 1
                if done:
                    break
        
            total_steps += steps
    
        self.epsilon = saved_epsilon
        return total_steps / numb_episodes

    def train(
        self,
        env,
        numb_episodes: int,
        eval_every: int = 50,
        eval_episodes: int = 20,
    ) -> dict:
        episode_lengths = []
        episode_rewards = []
        eval_episodes_at = []
        eval_steps = []
    
        for episode in range(numb_episodes):
            state = env.reset()
            state_idx = env.get_index(state)
            total_reward = 0.0
            steps = 0
        
            while True:
                action = self.select_action(state_idx)
                next_state, reward, done = env.step(action)
                next_state_idx = env.get_index(next_state)
            
                self.update(state_idx, action, reward, next_state_idx, done)
            
                state_idx = next_state_idx
                total_reward += reward
                steps += 1
            
                if done:
                    break
        
            episode_lengths.append(steps)
            episode_rewards.append(total_reward)
        
            if (episode + 1) % eval_every == 0:
                avg_steps = self.evaluate(env, eval_episodes)
                eval_episodes_at.append(episode + 1)
                eval_steps.append(avg_steps)
    
        return {
            "episode_lengths": episode_lengths,
            "episode_rewards": episode_rewards,
            "eval_episodes":   eval_episodes_at,
            "eval_steps":      eval_steps,
        }

    def reset_q(self) -> None:
        self.Q_A = np.zeros((self.numb_states, self.numb_actions))
        self.Q_B = np.zeros((self.numb_states, self.numb_actions))