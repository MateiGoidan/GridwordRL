def gridworld_a() -> dict:
    return {
        "rows": 7,
        "cols": 10,
        "start": (3, 0),
        "goal": (3, 7),
        "obstacles": {
            (1, 5),
            (2, 5),
            (3, 5),
            (4, 5),
        },
    }


def gridworld_b() -> dict:
    return {
        "rows": 7,
        "cols": 10,
        "start": (3, 0),
        "goal": (3, 7),
        "wind": [0, 0, 0, 1, 1, 1, 2, 2, 1, 0],
    }