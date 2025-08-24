import os, json, torch
from typing import Any, Dict

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_ckpt(state: Dict[str, Any], path: str):
    ensure_dir(os.path.dirname(path))
    torch.save(state, path)

def load_ckpt(path: str) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu")

def append_jsonl(path: str, record: Dict[str, Any]):
    ensure_dir(os.path.dirname(path))
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
