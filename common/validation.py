import json
import yaml
import numpy as np
from pathlib import Path
from common.errors import StrictFailure

def get_allowed_annotators():
    try:
        with open("configs/experiment.yaml", "r") as f:
            config = yaml.safe_load(f)
            return config.get("allowed_annotators", [])
    except Exception:
        return []

def get_signal_keys():
    try:
        with open("configs/signals.yaml", "r") as f:
            config = yaml.safe_load(f)
            return [k for k in config.keys() if k.startswith("s_")]
    except Exception:
        return ["s_ret", "s_ver", "s_nli", "s_ent", "s_div"]

def load_and_validate_data(path: str, strict: bool = True) -> list:
    res = []
    allowed_annotators = get_allowed_annotators()
    forbidden_annotators = {"judge", "pipeline", "mock"}
    
    for l in open(path, encoding="utf-8"):
        if not l.strip(): continue
        row = json.loads(l)
        if strict:
            if row.get("synthetic"):
                raise StrictFailure(f"Synthetic data found in {path}. Use real validation data.")
            if "label" in row:
                ann = row.get("annotator_id")
                if not ann:
                    raise StrictFailure(f"Missing annotator_id on labeled record in {path}.")
                if ann in forbidden_annotators:
                    raise StrictFailure(f"Forbidden annotator_id '{ann}' in {path}.")
                if allowed_annotators and ann not in allowed_annotators:
                    raise StrictFailure(f"Annotator '{ann}' not in allowed list in {path}.")
        res.append(row)
        
    if strict and res:
        signals = get_signal_keys()
        if "s_mech" in signals:
            raise StrictFailure(f"s_mech must not appear in signals config. (Found in {path})")
            
        try:
            from aggregate.dataset import FEATURE_COLS
        except ImportError:
            FEATURE_COLS = []
            
        for sig in signals:
            vals = [float(r.get(sig)) for r in res if r.get(sig) is not None]
            n_total = len(res)
            n_missing = n_total - len(vals)
            
            missing_indicator = f"{sig}_skipped"
            has_indicator = missing_indicator in FEATURE_COLS
            
            if n_missing / n_total > 0.20 and not has_indicator:
                raise StrictFailure(f"Signal column '{sig}' is missing > 20% of values in {path} and has no explicit missing-indicator feature.")
                
            if len(vals) > 1:
                std = np.std(vals)
                unique_frac = len(set(vals)) / len(vals)
                if std < 1e-3:
                    raise StrictFailure(f"Signal column '{sig}' is practically constant (std < 1e-3) in {path}.")
                if unique_frac < 0.05:
                    raise StrictFailure(f"Signal column '{sig}' has < 5% unique values in {path}.")
    return res
