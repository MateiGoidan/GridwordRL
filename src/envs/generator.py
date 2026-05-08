import numpy as np
import math
from collections import deque

def manhattan_distance(p1: tuple[int, int], p2: tuple[int, int]) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def generate_grid_size(rng: np.random.Generator) -> tuple[int, int]:
    rows = int(rng.integers(5, 9))
    cols = int(rng.integers(8, 13))
    return rows, cols

def place_start_and_goal(
    rows: int,
    cols: int,
    rng: np.random.Generator,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Place S and G randomly such that Manhattan distance is greater or equal than half the grid diagonal.
    """
    max_manhattan = rows + cols - 2
    min_distance = math.ceil(max_manhattan / 2)
    
    start = (int(rng.integers(0, rows)), int(rng.integers(0, cols)))
    
    while True:
        goal = (int(rng.integers(0, rows)), int(rng.integers(0, cols)))
        if manhattan_distance(start, goal) >= min_distance:
            return start, goal

def check_path(
    rows: int,
    cols: int,
    start: tuple[int, int],
    goal: tuple[int, int],
    obstacles: set[tuple[int, int]],
) -> bool:
    if start in obstacles or goal in obstacles:
        return False
    
    visited = {start}
    queue = deque([start])
    
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # BFS route check
    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if (nr, nc) in visited or (nr, nc) in obstacles:
                continue
            visited.add((nr, nc))
            queue.append((nr, nc))
    
    return False

def place_obstacles(
    rows: int,
    cols: int,
    start: tuple[int, int],
    goal: tuple[int, int],
    rng: np.random.Generator,
) -> set[tuple[int, int]]:
    """
    Place a random number of obstacles ([rows // 2, cols // 2]) such that a path from S to G remains.
    """
    min_obstacles = rows // 2
    max_obstacles = cols // 2 + rows // 2
    n_obstacles = int(rng.integers(min_obstacles, max_obstacles + 1))
    
    forbidden = {start, goal}
    obstacles: set[tuple[int, int]] = set()
    
    candidates = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in forbidden]
    rng.shuffle(candidates)
    
    for cell in candidates:
        if len(obstacles) >= n_obstacles:
            break
        obstacles.add(cell)
        if not check_path(rows, cols, start, goal, obstacles):
            obstacles.remove(cell)
    
    return obstacles

def generate_wind(cols: int, rng: np.random.Generator) -> list[int]:
    return [int(rng.integers(0, 3)) for _ in range(cols)]

def generate_gridworld_a(seed: int) -> dict:
    """Generate Gridworld A: random size + obstacles."""
    rng = np.random.default_rng(seed)
    rows, cols = generate_grid_size(rng)
    start, goal = place_start_and_goal(rows, cols, rng)
    obstacles = place_obstacles(rows, cols, start, goal, rng)
    return {
        "rows": rows,
        "cols": cols,
        "start": start,
        "goal": goal,
        "obstacles": obstacles,
    }

def generate_gridworld_b(seed: int) -> dict:
    """Generate Gridworld B: random size + wind."""
    rng = np.random.default_rng(seed)
    rows, cols = generate_grid_size(rng)
    start, goal = place_start_and_goal(rows, cols, rng)
    wind = generate_wind(cols, rng)
    return {
        "rows": rows,
        "cols": cols,
        "start": start,
        "goal": goal,
        "wind": wind,
    }