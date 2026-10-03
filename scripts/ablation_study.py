import argparse
import json
import logging
import math
import numpy as np
from pathlib import Path

from common.validation import load_and_validate_data
from common.experiment import load_experiment
from aggregate.dataset import FEATURE_COLS, row_to_features
from guarantee.ltt import build_mondrian_thresholds, StrataStatus
from eval.metrics import aurc, coverage, hallucination_rate, risk_coverage_curve
from eval.stats import paired_bootstrap, mcnemar
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

logger = logging.getLogger("ablation")
logging.basicConfig(level=logging.INFO)

COMPONENTS = {
    "s_ver": ["s_ver"],
    "s_nli": ["s_nli"],
    "s_ent": ["s_ent", "s_ent_max"],
    "s_div": ["s_div"],
    "s_ret": ["s_ret"],
    "V1": ["s_ver", "v1_match", "v1_mismatch", "v1_not_found", "v1_na"],
    "V2": ["s_nli", "nli_skipped"],
    "V3 judge band": ["s_ent", "s_ent_max", "s_ent_skipped", "s_div", "s_div_skipped"],
    "Mondrian": [], # Special case handled in logic
}
CORE_COMPONENTS = {"V1", "V2", "Mondrian"}

def _matrix(rows, drop_cols):
    cols = []
    for row in rows:
        vec = []
        for name in FEATURE_COLS:
            if name in drop_cols or row.get(name) is None or (isinstance(row.get(name), float) and math.isnan(row[name])):
                vec.append(0.0)
            else:
                vec.append(float(row[name]))
        cols.append(vec)
    return np.asarray(cols, dtype=float)

def _labels(rows):
    return np.asarray([1 if r.get("label") in ("unsupported", "outdated", 1, "1") else 0 for r in rows], dtype=int)

def fit_and_predict(train_rows, calib_rows, test_rows, drop_cols, seed, use_mondrian=True, eps=0.05, delta=0.10):
    X_train = _matrix(train_rows, drop_cols)
    y_train = _labels(train_rows)
    
    # We must use grouped CV by doc_id for retraining as per prompt
    # Wait, prompt says: "Retrain the aggregator whenever a signal is removed (grouped CV by doc_id)."
    # Does this mean the model is evaluated via CV on train, or trained on train using CV to pick best model?
    # Or just use the OOF predictions for the evaluation?
    # If the user wants the evaluation on test set, we just fit on train.
    # Let's fit on train. If the user meant "do GroupKFold on train to report CV AUROC", we can compute that too.
    groups = np.array([r.get("doc_id", "") for r in train_rows])
    gkf = GroupKFold(n_splits=5)
    cv_aurocs = []
    if len(set(y_train)) > 1 and len(set(groups)) >= 5:
        for tr_idx, val_idx in gkf.split(X_train, y_train, groups):
            if len(set(y_train[tr_idx])) < 2 or len(set(y_train[val_idx])) < 2: continue
            clf = LogisticRegression(max_iter=500, random_state=seed)
            clf.fit(X_train[tr_idx], y_train[tr_idx])
            p = clf.predict_proba(X_train[val_idx])[:, 1]
            cv_aurocs.append(roc_auc_score(y_train[val_idx], p))
    cv_auroc = float(np.mean(cv_aurocs)) if cv_aurocs else 0.5
    
    clf = LogisticRegression(max_iter=500, random_state=seed)
    if len(set(y_train)) > 1:
        clf.fit(X_train, y_train)
        cal_p = clf.predict_proba(_matrix(calib_rows, drop_cols))[:, 1]
        test_p = clf.predict_proba(_matrix(test_rows, drop_cols))[:, 1]
    else:
        cal_p = np.zeros(len(calib_rows))
        test_p = np.zeros(len(test_rows))
        
    for i, r in enumerate(calib_rows):
        r["temp_risk"] = cal_p[i]
    for i, r in enumerate(test_rows):
        r["temp_risk"] = test_p[i]
        
    # Calibrate
    # For LTT we need to temporarily set the 'risk' field to our model's predictions
    # to reuse build_mondrian_thresholds, which reads row["risk"].
    # So we backup risk.
    for r in calib_rows + test_rows:
        r["_orig_risk"] = r.get("risk", 0.0)
        r["risk"] = r["temp_risk"]
        
    thresholds = build_mondrian_thresholds(calib_rows, eps, delta, use_regulator=use_mondrian)
    # If we are disabling Mondrian, build_mondrian_thresholds already groups by *|atom_type and *|*
    # Wait, if we disable Mondrian, we should map everything to *|*.
    if not use_mondrian:
        # Override all strata to just use *|*
        global_th = thresholds.get("*|*")
        thresholds = {k: global_th for k in thresholds} if global_th else {}
        
    decs = []
    for row in test_rows:
        atom_type = row.get("atom_type", "UNKNOWN")
        sk_list = [atom_type, "*|*"]
        if use_mondrian:
            sk_list = [atom_type, "*|*"] # Simplified, usually Regulator is disabled by default unless use_regulator=True
        else:
            sk_list = ["*|*"]
            
        accepted = False
        for sk in sk_list:
            if sk in thresholds and thresholds[sk].status == StrataStatus.CERTIFIED:
                th = thresholds[sk].accept_below
                if row["risk"] <= th:
                    accepted = True
                    break
        decs.append("SUPPORTED" if accepted else "ABSTAINED")
        
    # Restore risk
    for r in calib_rows + test_rows:
        r["risk"] = r["_orig_risk"]
        
    test_labels_arr = _labels(test_rows)
    test_labels_str = [r.get("label") for r in test_rows]
    test_auroc = roc_auc_score(test_labels_arr, test_p) if len(set(test_labels_arr)) > 1 else 0.5
    rc = risk_coverage_curve(test_p.tolist(), test_labels_str)
    
    correct = [(d == "SUPPORTED" and l == "supported") or (d == "ABSTAINED" and l != "supported") for d, l in zip(decs, test_labels_str)]
    
    return {
        "cv_auroc": cv_auroc,
        "auroc": test_auroc,
        "aurc": aurc(rc),
        "coverage": coverage(decs),
        "hallucination_rate": hallucination_rate(test_labels_str, decs),
        "correct": correct,
        "decs": decs,
        "probs": test_p.tolist()
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/splits/train.jsonl")
    parser.add_argument("--calib", default="data/splits/calib.jsonl")
    parser.add_argument("--test", default="data/splits/test.jsonl")
    parser.add_argument("--summary", default="results/summary.json")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    
    train_rows = load_and_validate_data(args.train, strict=args.strict)
    calib_rows = load_and_validate_data(args.calib, strict=args.strict)
    test_rows = load_and_validate_data(args.test, strict=args.strict)
    
    # Feature transform
    for r in train_rows + calib_rows + test_rows:
        r.update(row_to_features(r))
        
    # Full System
    print("Evaluating Full System...")
    full_res = fit_and_predict(train_rows, calib_rows, test_rows, [], seed=42, use_mondrian=True)
    
    # Load summary for output
    try:
        with open(args.summary, "r") as f:
            summary = json.load(f)
    except Exception:
        summary = {"hallucination_rate": full_res["hallucination_rate"], "coverage": full_res["coverage"], "aurc": full_res["aurc"]}
        
    print(f"{'Condition':<20} | {'AUROC':<7} | {'AURC':<7} | {'Coverage':<8} | {'HR':<7} | {'Action'}")
    print("-" * 80)
    print(f"{'Full RegGuard':<20} | {full_res['auroc']:.4f} | {full_res['aurc']:.4f} | {full_res['coverage']:.4f}   | {full_res['hallucination_rate']:.4f} | CORE")
    
    for comp, drop_cols in COMPONENTS.items():
        use_mondrian = (comp != "Mondrian")
        res = fit_and_predict(train_rows, calib_rows, test_rows, drop_cols, seed=42, use_mondrian=use_mondrian)
        
        # Paired bootstrap vs full system
        # Marginal effect = Full - Ablated (so drop in AUROC and drop in coverage)
        # Wait, the rule says "marginal-effect interval on AUROC is < 0.01 AND on coverage-at-fixed-risk is < 0.01"
        # We can bootstrap the AUROC and Coverage differences.
        
        n_boot = 10000
        np.random.seed(42)
        n = len(test_rows)
        auroc_diffs = []
        cov_diffs = []
        labels_arr = _labels(test_rows)
        for _ in range(n_boot):
            idx = np.random.choice(n, n, replace=True)
            y_b = labels_arr[idx]
            if len(set(y_b)) < 2: continue
            
            p_full = np.array(full_res["probs"])[idx]
            p_abl = np.array(res["probs"])[idx]
            
            d_full = np.array(full_res["decs"])[idx]
            d_abl = np.array(res["decs"])[idx]
            
            a_full = roc_auc_score(y_b, p_full)
            a_abl = roc_auc_score(y_b, p_abl)
            auroc_diffs.append(a_full - a_abl)
            
            c_full = sum(1 for d in d_full if d in ("SUPPORTED", "VERIFIED")) / n
            c_abl = sum(1 for d in d_abl if d in ("SUPPORTED", "VERIFIED")) / n
            cov_diffs.append(c_full - c_abl)
            
        auroc_ub = np.percentile(auroc_diffs, 97.5) if auroc_diffs else 0.0
        cov_ub = np.percentile(cov_diffs, 97.5) if cov_diffs else 0.0
        
        mc = mcnemar(full_res["correct"], res["correct"])
        
        action = "KEEP"
        if auroc_ub < 0.01 and cov_ub < 0.01:
            action = "REMOVE-CANDIDATE"
            
        if comp in CORE_COMPONENTS:
            action = "CORE (NEVER REMOVE)"
            
        print(f"{comp:<20} | {res['auroc']:.4f} | {res['aurc']:.4f} | {res['coverage']:.4f}   | {res['hallucination_rate']:.4f} | {action}")

if __name__ == "__main__":
    main()
