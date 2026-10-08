"""
Persistence and Serialization Utilities using Joblib and JSON.
"""

from pathlib import Path
import json
from typing import Any, Dict
import joblib


def save_artifact(obj: Any, file_path: str):
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, p)


def load_artifact(file_path: str) -> Any:
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Artifact not found at {file_path}")
    return joblib.load(p)


def save_json(data: Dict[str, Any], file_path: str):
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(file_path: str) -> Dict[str, Any]:
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found at {file_path}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
