import json
from pathlib import Path
from datetime import datetime
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

def _to_python(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    return obj

def save_experiment(filename: str, data: dict) -> Path:
    path = EXPERIMENTS_DIR / f"{filename}.json"
    
    data_with_meta = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        **data,
    }
    
    with open(path, "w") as f:
        json.dump(_to_python(data_with_meta), f, indent=2)
    
    print(f"Saved: {path.name}")
    return path

def load_experiment(filename: str) -> dict:
    path = EXPERIMENTS_DIR / f"{filename}.json"
    with open(path, "r") as f:
        return json.load(f)