from envs.base import BaseGridworld


class GridworldB(BaseGridworld):
    def __init__(self, config: dict, numb_actions: int = 4):
        super().__init__(config, numb_actions)
        self.wind = config["wind"]
    
    def step(self, action: int) -> tuple[tuple[int, int], float, bool]:
        dr, dc = self.actions[action]
        r, c = self.agent_pos
        
        nr = r + dr - self.wind[c]
        nc = c + dc
        
        nr, nc = self._clamp(nr, nc)
        
        self.agent_pos = (nr, nc)
        done = (self.agent_pos == self.goal)
        reward = self._compute_reward(done)
        return self.agent_pos, reward, done