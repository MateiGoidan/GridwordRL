from envs.generator import generate_gridworld_a, generate_gridworld_b

seed = 42

# Gridworld A
grid_a = generate_gridworld_a(seed)
print("=== Gridworld A ===")
print(f"Size: {grid_a['rows']} x {grid_a['cols']}")
print(f"S={grid_a['start']}, G={grid_a['goal']}")
print(f"Obstacles: {len(grid_a['obstacles'])}")

print("\nGrid layout:")
for r in range(grid_a['rows']):
    row_str = ""
    for c in range(grid_a['cols']):
        if (r, c) == grid_a['start']:
            row_str += "S "
        elif (r, c) == grid_a['goal']:
            row_str += "G "
        elif (r, c) in grid_a['obstacles']:
            row_str += "# "
        else:
            row_str += ". "
    print(row_str)

# Gridworld B
grid_b = generate_gridworld_b(seed)
print("\n=== Gridworld B ===")
print(f"Size: {grid_b['rows']} x {grid_b['cols']}")
print(f"S={grid_b['start']}, G={grid_b['goal']}")
print(f"Wind: {grid_b['wind']}")

# Visualize Gridworld B with wind row at bottom
print("\nGridworld B layout:")
for r in range(grid_b['rows']):
    row_str = ""
    for c in range(grid_b['cols']):
        if (r, c) == grid_b['start']:
            row_str += "S "
        elif (r, c) == grid_b['goal']:
            row_str += "G "
        else:
            row_str += ". "
    print(row_str)
print(" ".join(str(w) for w in grid_b['wind']))