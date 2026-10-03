"""Load configs/experiment.yaml and stamp runs with a config hash."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "configs" / "experiment.yaml"

_CACHED: Optional[Dict[str, Any]] = None


def load_experiment(path: Optional[str] = None, reload: bool = False) -> Dict[str, Any]:
    global _CACHED
    if _CACHED is not None and not reload and path is None:
        return _CACHED
    p = Path(path) if path else DEFAULT_PATH
    if not p.exists():
        raise FileNotFoundError(f"Experiment config not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data["_path"] = str(p)
    if path is None:
        _CACHED = data
    return data


def config_hash(path: Optional[str] = None) -> str:
    p = Path(path) if path else DEFAULT_PATH
    raw = p.read_bytes()
    return hashlib.sha256(raw).hexdigest()


def git_commit(root: Optional[Path] = None) -> str:
    cwd = str(root or ROOT)
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def provenance(path: Optional[str] = None) -> Dict[str, str]:
    return {
        "git_commit": git_commit(),
        "config_hash": config_hash(path),
        "config_path": str(Path(path) if path else DEFAULT_PATH),
    }


def is_paper_mode(path: Optional[str] = None) -> bool:
    try:
        return load_experiment(path).get("mode") == "paper"
    except Exception:
        return False


def strict_default(path: Optional[str] = None) -> bool:
    try:
        exp = load_experiment(path)
        if exp.get("mode") == "paper":
            return True
        return bool(exp.get("strict", False))
    except FileNotFoundError:
        return True
