from envs.grid_generator import generate_gridworld_a, generate_gridworld_b
from envs.gridworld_a import GridworldA
from envs.gridworld_b import GridworldB

seed = 42

config_a = generate_gridworld_a(seed)
env_a = GridworldA(config_a)
print(f"Gridworld A: {env_a.rows}x{env_a.cols}, numb_states={env_a.numb_states}")

state = env_a.reset()
print(f"Initial state: {state}")

for action_name, action in [("UP", 0), ("DOWN", 1), ("LEFT", 2), ("RIGHT", 3)]:
    env_a.reset()
    next_state, reward, done = env_a.step(action)
    print(f"  Action {action_name}: {state} -> {next_state}, r={reward}, done={done}")

print(f"Obstacles: {sorted(env_a.obstacles)}")

# Test Gridworld B
config_b = generate_gridworld_b(seed)
env_b_4 = GridworldB(config_b, numb_actions=4)
env_b_8 = GridworldB(config_b, numb_actions=8)
print(f"\nGridworld B (4 actions): numb_actions={env_b_4.numb_actions}")
print(f"Gridworld B (8 actions): numb_actions={env_b_8.numb_actions}")
print(f"Wind: {env_b_4.wind}")