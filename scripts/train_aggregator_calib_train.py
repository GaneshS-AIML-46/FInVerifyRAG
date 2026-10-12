"""Train logistic aggregator on results/train.jsonl + results/calib.jsonl."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import IsotonicRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aggregate.dataset import ATOM_TYPES

TRAIN = ROOT / "results" / "train.jsonl"
CALIB = ROOT / "results" / "calib.jsonl"
RAW = ROOT / "results" / "raw_atoms.jsonl"
OUT_DIR = ROOT / "models" / "aggregator"
MODEL_PATH = OUT_DIR / "aggregator_v1.pkl"
CAL_PATH = OUT_DIR / "calibrator_v1.pkl"

FEATURE_ORDER = (
    ["s_ret", "s_ver", "s_nli", "s_ent", "s_div"]
    + ["v1_match", "v1_mismatch", "v1_not_found", "v1_na"]
    + ["V2_output"]
    + [f"type_{t}" for t in ATOM_TYPES]
)

_V1_SCORE = {"MATCH": 0.1, "MISMATCH": 0.9, "NOT_FOUND": 0.7, "NA": 0.5}


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _load_raw_index() -> dict[str, dict]:
    idx = {}
    for rec in _load_jsonl(RAW):
        idx[rec["id"]] = rec
    return idx


def _v1_onehot(status: str) -> dict[str, int]:
    s = (status or "NA").upper()
    return {
        "v1_match": int(s == "MATCH"),
        "v1_mismatch": int(s == "MISMATCH"),
        "v1_not_found": int(s == "NOT_FOUND"),
        "v1_na": int(s == "NA"),
    }


def _feature_row(labeled: dict, raw: dict) -> dict:
    v1 = raw.get("v1_status", "NA")
    v1oh = _v1_onehot(v1)
    atom_type = raw.get("atom_type", "UNKNOWN")
    if atom_type not in ATOM_TYPES:
        atom_type = "UNKNOWN"

    def _sig(name: str, default=np.nan):
        v = raw.get(name)
        return float(v) if v is not None else default

    s_ver = raw.get("s_ver")
    if s_ver is None:
        s_ver = _V1_SCORE.get(str(v1).upper(), 0.5)

    row = {
        "atom_id": labeled["atom_id"],
        "doc_id": labeled["doc_id"],
        "label_text": labeled["label"],
        "s_ret": _sig("s_ret"),
        "s_ver": float(s_ver),
        "s_nli": _sig("s_nli"),
        "s_ent": _sig("s_ent"),
        "s_div": _sig("s_div"),
        "V2_output": _sig("v2_entail_prob"),
        **v1oh,
    }
    for t in ATOM_TYPES:
        row[f"type_{t}"] = int(atom_type == t)
    return row


def main() -> None:
    if not TRAIN.is_file() or not CALIB.is_file():
        raise FileNotFoundError("Missing results/train.jsonl or results/calib.jsonl")
    if not RAW.is_file():
        raise FileNotFoundError(RAW)

    raw_idx = _load_raw_index()
    combined = _load_jsonl(TRAIN) + _load_jsonl(CALIB)

    rows = []
    for lab in combined:
        lab_label = (lab.get("label") or "").strip().lower()
        if lab_label == "unclear":
            continue
        raw = raw_idx.get(lab["atom_id"])
        if raw is None:
            raise ValueError(f"No raw_atoms row for atom_id={lab['atom_id']}")
        rows.append(_feature_row(lab, raw))

    if len(rows) < 10:
        raise ValueError(f"Too few labeled atoms after excluding unclear: {len(rows)}")

    df = pd.DataFrame(rows)
    y = np.array([0 if r["label_text"] == "supported" else 1 for r in rows])
    groups = df["doc_id"].astype(str).values
    n_docs = len(set(groups))
    if n_docs < 5:
        raise ValueError(f"Need >=5 doc_id groups for 5-fold CV, found {n_docs}")

    X = df[FEATURE_ORDER].astype(float).values
    imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    X = imputer.fit_transform(X)

    gkf = GroupKFold(n_splits=5)
    aurocs, briers, precisions, recalls, f1s = [], [], [], [], []

    print("Per-fold metrics (val fold; isotonic calibration applied on val probabilities):")
    for fold, (tr, va) in enumerate(gkf.split(X, y, groups), start=1):
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        clf.fit(X[tr], y[tr])
        raw_p = clf.predict_proba(X[va])[:, 1]
        y_va = y[va]

        if len(set(y_va.tolist())) > 1:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(raw_p, y_va)
            cal_p = iso.predict(raw_p)
            auroc = roc_auc_score(y_va, cal_p)
            brier = brier_score_loss(y_va, cal_p)
        else:
            cal_p = raw_p
            auroc = float("nan")
            brier = brier_score_loss(y_va, cal_p)

        y_hat = (cal_p >= 0.5).astype(int)
        prec = precision_score(y_va, y_hat, zero_division=0)
        rec = recall_score(y_va, y_hat, zero_division=0)
        f1 = f1_score(y_va, y_hat, zero_division=0)

        aurocs.append(auroc)
        briers.append(brier)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        print(
            f"  Fold {fold}: AUROC={auroc:.4f}  Brier={brier:.4f}  "
            f"Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}"
        )

    # Final model: hold out last doc group for isotonic, fit on rest
    hold_docs = sorted(set(groups))[-1:]
    cal_mask = np.isin(groups, hold_docs)
    final_clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    final_clf.fit(X[~cal_mask], y[~cal_mask])
    calibrator = None
    if cal_mask.sum() >= 3 and len(set(y[cal_mask].tolist())) > 1:
        raw_hold = final_clf.predict_proba(X[cal_mask])[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_hold, y[cal_mask])
        calibrator = {"method": "isotonic", "calibrator": iso}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": final_clf,
        "feature_order": list(FEATURE_ORDER),
        "kind": "logreg",
        "imputer_fill": 0.0,
        "imputer": imputer,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    if calibrator:
        with open(CAL_PATH, "wb") as f:
            pickle.dump(calibrator, f)

    auroc_mean = float(np.nanmean(aurocs))
    auroc_std = float(np.nanstd(aurocs))
    brier_mean = float(np.mean(briers))
    print(
        f"Aggregator trained. CV AUROC: {auroc_mean:.2f} ± {auroc_std:.2f}. "
        f"Brier: {brier_mean:.2f}."
    )
    print(f"Saved {MODEL_PATH} ({len(FEATURE_ORDER)} features)")


if __name__ == "__main__":
    main()
