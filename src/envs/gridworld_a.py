from envs.base import BaseGridworld

class GridworldA(BaseGridworld):
    def __init__(self, config: dict, n_actions: int = 4):
        super().__init__(config, n_actions)
        self.obstacles = config["obstacles"]
    
    def step(self, action: int) -> tuple[tuple[int, int], float, bool]:
        dr, dc = self.actions[action]
        r, c = self.agent_pos
        nr, nc = r + dr, c + dc
        
        nr, nc = self._clamp(nr, nc)
        
        if (nr, nc) in self.obstacles:
            nr, nc = r, c
        
        self.agent_pos = (nr, nc)
        done = (self.agent_pos == self.goal)
        reward = self._compute_reward(done)
        return self.agent_pos, reward, done