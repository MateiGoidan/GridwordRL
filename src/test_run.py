from envs.generator import generate_gridworld_a
from envs.gridworld_a import GridworldA

config_a = generate_gridworld_a(seed=2)
env = GridworldA(config_a)

print(f"S = {env.start}")
print(f"G = {env.goal}")

state = env.reset()
print(f"\nStart: {state}")

# Move toward the goal step by step
for step in range(50):
    sr, sc = state
    gr, gc = env.goal
    
    # Pick a greedy action toward the goal
    if sr < gr:
        action = 1  # DOWN
    elif sr > gr:
        action = 0  # UP
    elif sc < gc:
        action = 3  # RIGHT
    else:
        action = 2  # LEFT
    
    state, reward, done = env.step(action)
    print(f"  step {step+1}: pos={state}, r={reward}, done={done}")
    
    if done:
        print(f"\nReached goal in {step+1} steps!")
        break