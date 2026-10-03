import json
import yaml
import numpy as np
from pathlib import Path
from common.errors import StrictFailure

def get_allowed_annotators(strict=True):
    try:
        with open("configs/experiment.yaml", "r") as f:
            config = yaml.safe_load(f)
            return config.get("allowed_annotators", [])
    except Exception as e:
        if strict:
            raise StrictFailure(f"Failed to read configs/experiment.yaml: {e}")
        return []

def get_signal_config(strict=True):
    try:
        with open("configs/signals.yaml", "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        if strict:
            raise StrictFailure(f"Failed to read configs/signals.yaml: {e}")
        return {}

def load_and_validate_data(path: str, strict: bool = True) -> list:
    res = []
    allowed_annotators = get_allowed_annotators(strict)
    if strict and not allowed_annotators:
        raise StrictFailure("allowed_annotators list is empty or missing in configs/experiment.yaml")

    forbidden_annotators = {"judge", "pipeline", "mock"}
    
    for l in open(path, encoding="utf-8"):
        if not l.strip(): continue
        row = json.loads(l)
        if "s_mech" in row:
            raise StrictFailure(f"Row contains forbidden key 's_mech' in {path}")
            
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
        sig_config = get_signal_config(strict)
        signals = {k: v for k, v in sig_config.items() if k.startswith("s_")}
        if "s_mech" in signals:
            raise StrictFailure(f"s_mech must not appear in signals config. (Found in {path})")
            
        try:
            from aggregate.dataset import FEATURE_COLS
        except ImportError as e:
            raise StrictFailure(f"Failed to import FEATURE_COLS: {e}")
            
        for sig, cfg in signals.items():
            sig_type = cfg.get("type", "continuous")
            raw_vals = [r.get(sig) for r in res if r.get(sig) is not None]
            n_total = len(res)
            n_missing = n_total - len(raw_vals)
            
            missing_indicator = f"{sig}_skipped"
            has_indicator = missing_indicator in FEATURE_COLS
            
            if n_missing / n_total > 0.20 and not has_indicator:
                raise StrictFailure(f"Signal column '{sig}' is missing > 20% of values in {path} and has no explicit missing-indicator feature.")
            
            if len(raw_vals) > 1:
                if sig_type == "categorical":
                    # For categorical signals: require >= 2 distinct values and rarest value >= 2% of rows
                    # do not call float()
                    unique_counts = {}
                    for v in raw_vals:
                        unique_counts[v] = unique_counts.get(v, 0) + 1
                    
                    if len(unique_counts) < 2:
                        raise StrictFailure(f"Categorical signal '{sig}' has < 2 distinct values in {path}.")
                    
                    rarest_count = min(unique_counts.values())
                    if rarest_count / len(raw_vals) < 0.02:
                        raise StrictFailure(f"Categorical signal '{sig}' rarest value appears in < 2% of present rows in {path}.")
                else:
                    # Continuous
                    vals = [float(v) for v in raw_vals]
                    std = np.std(vals)
                    unique_frac = len(set(vals)) / len(vals)
                    if std < 1e-3:
                        raise StrictFailure(f"Signal column '{sig}' is practically constant (std < 1e-3) in {path}.")
                    if unique_frac < 0.05:
                        raise StrictFailure(f"Signal column '{sig}' has < 5% unique values in {path}.")
    return res
