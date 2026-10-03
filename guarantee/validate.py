import random
import numpy as np
from collections import defaultdict
from guarantee.ltt import build_mondrian_thresholds, StrataStatus

def validate_guarantee(calib_pool, eps, delta, n_splits=500, seed=42, system_name=None):
    """
    Repeatedly split the calibration pool into (calibrate, validate) parts BY DOCUMENT.
    Run LTT on calibrate part, measure selective risk and coverage on validate part.
    """
    rng = random.Random(seed)
    
    # Group atoms by doc_id
    doc_to_atoms = defaultdict(list)
    for row in calib_pool:
        doc_id = row.get("doc_id", "unknown_doc")
        doc_to_atoms[doc_id].append(row)
        
    doc_ids = list(doc_to_atoms.keys())
    
    violations = 0
    selective_risks = []
    coverages = []
    status_counts = defaultdict(int)
    
    for _ in range(n_splits):
        rng.shuffle(doc_ids)
        split_idx = int(len(doc_ids) * 0.5)
        calib_docs = doc_ids[:split_idx]
        val_docs = doc_ids[split_idx:]
        
        calib_rows = []
        for d in calib_docs:
            calib_rows.extend(doc_to_atoms[d])
            
        val_rows = []
        for d in val_docs:
            val_rows.extend(doc_to_atoms[d])
            
        if not calib_rows or not val_rows:
            continue
            
        if system_name is None or system_name in ("RegGuard (ours)", "regguard"):
            # Run LTT on calib
            thresholds = build_mondrian_thresholds(calib_rows, eps, delta, use_regulator=False)
            
            # Collect status counts
            for stratum, res in thresholds.items():
                status_counts[f"{stratum}_{res.status}"] += 1
                
            # Evaluate on val
            n_accepted = 0
            n_wrong = 0
            
            for row in val_rows:
                atom_type = row.get("atom_type", "UNKNOWN")
                stratum_keys = [atom_type, "*|*"]
                
                accepted = False
                for sk in stratum_keys:
                    if sk in thresholds and thresholds[sk].status == StrataStatus.CERTIFIED:
                        th = thresholds[sk].accept_below
                        if row.get("risk", 0.0) <= th:
                            accepted = True
                            break
                            
                if accepted:
                    n_accepted += 1
                    if row.get("label") in ("unsupported", "outdated", "1", 1):
                        n_wrong += 1
        else:
            from eval.systems import risk_vector, tune_threshold
            cal_risk = risk_vector(calib_rows, system_name)
            if cal_risk is None:
                continue
            cal_labels = [r.get("label") for r in calib_rows]
            thr = tune_threshold(cal_risk, cal_labels, eps)
            
            val_risk = risk_vector(val_rows, system_name)
            val_labels = [r.get("label") for r in val_rows]
            
            n_accepted = 0
            n_wrong = 0
            if thr == thr: # not NaN
                for rsk, lbl in zip(val_risk, val_labels):
                    if rsk <= thr:
                        n_accepted += 1
                        if lbl in ("unsupported", "outdated", "1", 1):
                            n_wrong += 1
                            
        coverages.append(n_accepted / len(val_rows))
        
        if n_accepted > 0:
            sr = n_wrong / n_accepted
            selective_risks.append(sr)
            if sr > eps:
                violations += 1
                
    mean_coverage = float(np.mean(coverages)) if coverages else 0.0
    vr = violations / max(n_splits, 1)
    
    return {
        "n_splits": n_splits,
        "n_violations": violations,
        "violation_rate": vr,
        "mean_selective_risk": float(np.mean(selective_risks)) if selective_risks else 0.0,
        "max_selective_risk": float(np.max(selective_risks)) if selective_risks else 0.0,
        "mean_coverage": mean_coverage,
        "status_counts": dict(status_counts) if (system_name is None or system_name in ("RegGuard (ours)", "regguard")) else {},
        "valid": bool(mean_coverage > 0.0)
    }
