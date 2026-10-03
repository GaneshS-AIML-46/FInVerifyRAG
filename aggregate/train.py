"""
aggregate/train.py
-------------------
Train the multi-signal risk aggregator.

Steps
-----
1. Load feature table (Parquet built by dataset.py).
2. Logistic Regression baseline.
3. LightGBM classifier (main model).
4. Cross-validate by question-family column (if present), else k-fold.
5. Signal ablation: drop one signal at a time, re-train, compare AUROC.
6. Save the best model as `aggregator_v1.pkl` and the feature list as
   `features.json`.

Usage
-----
    python -m aggregate.train \\
        --features data/processed/feature_table.parquet \\
        --out      models/aggregator/ \\
        --ablation

Pitfall: this script must ONLY see agg_train data. Calibration and test
splits must be held out for M7.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd  # type: ignore

from aggregate.dataset import FEATURE_COLS, SIGNAL_COLS

logger = logging.getLogger("aggregate.train")
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_table(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    logger.info("Loaded feature table: %d rows", len(df))
    return df


def _check_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Add any missing feature columns as 0.5 (neutral) so training never errors."""
    for col in FEATURE_COLS:
        if col not in df.columns:
            logger.warning("Feature column '%s' missing — filling with 0.5", col)
            df[col] = 0.5
    return df


def _get_X_y(df: pd.DataFrame):
    X = df[FEATURE_COLS].fillna(0.5).values
    y = df["label"].values
    return X, y


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_logreg(X_train, y_train, X_val, y_val):
    """Logistic Regression baseline."""
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.metrics import roc_auc_score             # type: ignore

    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_val)[:, 1]
    auroc = roc_auc_score(y_val, proba)
    logger.info("LogReg AUROC (val): %.4f", auroc)
    return clf, auroc


def train_lgbm(X_train, y_train, X_val, y_val):
    """LightGBM main model."""
    from lightgbm import LGBMClassifier   # type: ignore
    from sklearn.metrics import roc_auc_score  # type: ignore

    clf = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=31,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[],
    )
    proba = clf.predict_proba(X_val)[:, 1]
    auroc = roc_auc_score(y_val, proba)
    logger.info("LightGBM AUROC (val): %.4f", auroc)
    return clf, auroc


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------

def run_ablation(X_train, y_train, X_val, y_val, feature_names: List[str]) -> dict:
    """
    Drop one SIGNAL column at a time, retrain LightGBM, record delta AUROC.
    Returns dict {signal_name: auroc_without_it}.
    """
    from lightgbm import LGBMClassifier  # type: ignore
    from sklearn.metrics import roc_auc_score  # type: ignore

    results = {}
    signal_indices = [feature_names.index(s) for s in SIGNAL_COLS if s in feature_names]

    for idx in signal_indices:
        sig_name = feature_names[idx]
        mask = [i for i in range(X_train.shape[1]) if i != idx]
        Xt = X_train[:, mask]
        Xv = X_val[:, mask]

        clf = LGBMClassifier(n_estimators=200, learning_rate=0.05,
                             max_depth=4, random_state=42, verbose=-1)
        clf.fit(Xt, y_train)
        auc = roc_auc_score(y_val, clf.predict_proba(Xv)[:, 1])
        results[sig_name] = round(auc, 4)
        logger.info("Ablation drop %s → AUROC=%.4f", sig_name, auc)

    return results


# ---------------------------------------------------------------------------
# Cross-validation split
# ---------------------------------------------------------------------------

def _train_val_split(df: pd.DataFrame, val_frac: float = 0.15, seed: int = 42):
    """
    Split by question_family if present, else random.
    """
    if "question_family" in df.columns:
        families = df["question_family"].unique()
        rng = np.random.default_rng(seed)
        n_val = max(1, int(len(families) * val_frac))
        val_fams = set(rng.choice(families, n_val, replace=False))
        val_mask = df["question_family"].isin(val_fams)
        return df[~val_mask], df[val_mask]
    else:
        val_mask = np.random.default_rng(seed).random(len(df)) < val_frac
        return df[~val_mask], df[val_mask]


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    features_path: str,
    out_dir: str,
    run_ablation_flag: bool = False,
    val_frac: float = 0.15,
    seed: int = 42,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = _load_table(features_path)
    df = _check_cols(df)

    train_df, val_df = _train_val_split(df, val_frac=val_frac, seed=seed)
    logger.info("Split: train=%d, val=%d", len(train_df), len(val_df))

    X_train, y_train = _get_X_y(train_df)
    X_val,   y_val   = _get_X_y(val_df)

    # --- Baseline: LogReg ---
    lr_clf, lr_auroc = train_logreg(X_train, y_train, X_val, y_val)

    # --- Main: LightGBM ---
    lgbm_clf, lgbm_auroc = train_lgbm(X_train, y_train, X_val, y_val)

    # --- Save best model ---
    best_clf = lgbm_clf if lgbm_auroc >= lr_auroc else lr_clf
    model_path = out / "aggregator_v1.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_clf, f)
    logger.info("Saved best model to %s (AUROC=%.4f)", model_path,
                max(lgbm_auroc, lr_auroc))

    # --- Save feature list ---
    feat_path = out / "features.json"
    with open(feat_path, "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)
    logger.info("Saved feature list to %s", feat_path)

    # --- Save metrics ---
    metrics = {
        "logreg_auroc": lr_auroc,
        "lgbm_auroc":   lgbm_auroc,
        "n_train":      len(train_df),
        "n_val":        len(val_df),
    }

    # --- Ablation (optional) ---
    if run_ablation_flag:
        ablation = run_ablation(X_train, y_train, X_val, y_val, FEATURE_COLS)
        metrics["ablation"] = ablation
        abl_path = out / "ablation.json"
        with open(abl_path, "w") as f:
            json.dump(ablation, f, indent=2)
        logger.info("Ablation saved to %s", abl_path)

    metrics_path = out / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics: %s", metrics)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def train_from_jsonl(
    jsonl_path: str,
    out_dir: str,
    model: str = "logreg",
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    """
    Train on labeled train.jsonl. Grouped CV is by document id.
    Probability calibration uses a held-out slice of train only.
    Exits via SystemExit if labels or enough documents are missing.
    """
    import sys
    path = Path(jsonl_path)
    if not path.exists():
        logger.error("Refusing to train: %s does not exist.", path)
        sys.exit(1)
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    labeled = [r for r in rows if r.get("label") in ("supported", "unsupported", "outdated", 0, 1)]
    if not labeled:
        logger.error("Refusing to train: no human labels in %s.", path)
        sys.exit(1)
    from aggregate.dataset import row_to_features
    frame = pd.DataFrame([row_to_features(r) for r in labeled])
    if "doc_id" not in frame.columns:
        frame["doc_id"] = [r.get("doc_id") or r.get("document_id") or "" for r in labeled]
    groups = frame["doc_id"].astype(str)
    if groups.nunique() < n_splits:
        logger.error(
            "Need at least %d distinct documents for grouped CV, found %d.",
            n_splits, int(groups.nunique()),
        )
        sys.exit(1)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.impute import SimpleImputer

    y = np.array([
        1 if r.get("label") in ("unsupported", "outdated", 1) else 0
        for r in labeled
    ])
    X = frame.reindex(columns=FEATURE_COLS).to_numpy(dtype=float)
    imputer = SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True)
    X = imputer.fit_transform(X)

    if model == "gbm":
        try:
            from lightgbm import LGBMClassifier
            def _make():
                return LGBMClassifier(
                    n_estimators=200, learning_rate=0.05, max_depth=4,
                    random_state=seed, verbose=-1,
                )
            kind = "gbm"
        except Exception:
            logger.error("Gradient boosting was requested but lightgbm is not importable.")
            sys.exit(1)
    else:
        def _make():
            return LogisticRegression(max_iter=500, random_state=seed)
        kind = "logreg"

    gkf = GroupKFold(n_splits=n_splits)
    aurocs = []
    briers = []
    for train_idx, val_idx in gkf.split(X, y, groups):
        clf = _make()
        clf.fit(X[train_idx], y[train_idx])
        proba = clf.predict_proba(X[val_idx])[:, 1]
        if len(set(y[val_idx].tolist())) > 1:
            aurocs.append(float(roc_auc_score(y[val_idx], proba)))
        briers.append(float(brier_score_loss(y[val_idx], proba)))

    # Final fit on all but the last group's documents, calibrate on those.
    hold_docs = sorted(groups.unique())[-1:]
    cal_mask = groups.isin(hold_docs).to_numpy()
    final = _make()
    final.fit(X[~cal_mask], y[~cal_mask])
    raw = final.predict_proba(X[cal_mask])[:, 1]
    calibrator = None
    if cal_mask.sum() >= 5 and len(set(y[cal_mask].tolist())) > 1:
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw, y[cal_mask])
        calibrator = {"method": "isotonic", "calibrator": iso}

    from common.experiment import config_hash
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": final,
        "feature_order": list(FEATURE_COLS),
        "kind": kind,
        "config_hash": config_hash(),
        "imputer_fill": 0.0,
    }
    model_path = out / "aggregator_v1.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)
    if calibrator is not None:
        with open(out / "calibrator_v1.pkl", "wb") as f:
            pickle.dump(calibrator, f)
    metrics = {
        "auroc_mean": float(np.mean(aurocs)) if aurocs else None,
        "brier_mean": float(np.mean(briers)) if briers else None,
        "auroc_folds": aurocs,
        "brier_folds": briers,
        "n_labeled": len(labeled),
        "n_documents": int(groups.nunique()),
        "model": kind,
        "config_hash": bundle["config_hash"],
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "features.json").write_text(json.dumps(FEATURE_COLS, indent=2), encoding="utf-8")
    logger.info("Saved %s  AUROC=%s  Brier=%s", model_path, metrics["auroc_mean"], metrics["brier_mean"])
    # Print feature importance
    if kind == "gbm":
        importances = final.feature_importances_
    else:
        importances = np.abs(final.coef_[0])
        
    feat_imps = sorted(zip(FEATURE_COLS, importances), key=lambda x: x[1], reverse=True)
    logger.info("--- Feature Importance ---")
    for fname, fval in feat_imps:
        if fval > 0:
            logger.info("  %s: %.4f", fname, fval)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train M6 risk aggregator.")
    parser.add_argument("--features", default=None, help="Path to feature_table.parquet")
    parser.add_argument("--jsonl", default=None, help="Labeled train.jsonl (document-grouped CV)")
    parser.add_argument("--out", default="models/aggregator/", help="Output directory for model")
    parser.add_argument("--model", choices=["logreg", "gbm"], default="logreg")
    parser.add_argument("--ablation", action="store_true", help="Run signal ablation")
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--seed",     type=int,   default=42)
    args = parser.parse_args()

    if args.jsonl:
        train_from_jsonl(args.jsonl, args.out, model=args.model, seed=args.seed)
    elif args.features:
        train(
            features_path=args.features,
            out_dir=args.out,
            run_ablation_flag=args.ablation,
            val_frac=args.val_frac,
            seed=args.seed,
        )
    else:
        parser.error("Pass --jsonl data/splits/train.jsonl or --features feature_table.parquet")
