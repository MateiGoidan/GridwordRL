from abc import ABC, abstractmethod


ACTIONS_4 = {
    0: (-1,  0),   # UP
    1: ( 1,  0),   # DOWN
    2: ( 0, -1),   # LEFT
    3: ( 0,  1),   # RIGHT
}

ACTIONS_8 = {
    **ACTIONS_4,
    4: (-1,  1),   # UP-RIGHT
    5: ( 1,  1),   # DOWN-RIGHT
    6: ( 1, -1),   # DOWN-LEFT
    7: (-1, -1),   # UP-LEFT
}


class BaseGridworld(ABC):
    def __init__(self, config: dict, numb_actions: int = 4):
        assert numb_actions in (4, 8), "n_actions must be 4 or 8"
        
        self.rows = config["rows"]
        self.cols = config["cols"]
        self.start = config["start"]
        self.goal = config["goal"]
        
        self.numb_actions = numb_actions
        self.actions = ACTIONS_4 if numb_actions == 4 else ACTIONS_8
        self.numb_states = self.rows * self.cols
        
        self.agent_pos = self.start
    
    def reset(self) -> tuple[int, int]:
        self.agent_pos = self.start
        return self.agent_pos
    
    @abstractmethod
    def step(self, action: int) -> tuple[tuple[int, int], float, bool]:
        """
        (next_state, reward, done)
        """
    
    def get_index(self, state: tuple[int, int]) -> int:
        return state[0] * self.cols + state[1]
    
    def _clamp(self, r: int, c: int) -> tuple[int, int]:
        r = max(0, min(self.rows - 1, r))
        c = max(0, min(self.cols - 1, c))
        return r, c
    
    def _compute_reward(self, done: bool) -> float:
        return 1.0 if done else -1.0