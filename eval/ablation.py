"""Retrain the aggregator with one signal removed. No noise injection."""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np

from eval.metrics import aurc, coverage, hallucination_rate, risk_coverage_curve
from eval.stats import mcnemar, paired_bootstrap
from eval.systems import decisions_at, tune_threshold

SIGNAL_DROPS = [
    ("No V1 (skip deterministic verifier)", ["s_ver"]),
    ("No V2 (skip NLI verifier)", ["s_nli"]),
    ("No s_div signal", ["s_div"]),
    ("No s_ret signal", ["s_ret"]),
    ("No s_nli signal", ["s_nli"]),
    ("No s_ent signal", ["s_ent"]),
]

FEATURES = ["s_ver", "s_nli", "s_ent", "s_ret", "s_div"]


def _matrix(rows: Sequence[dict], drop: Sequence[str]) -> np.ndarray:
    cols = []
    for row in rows:
        vec = []
        for name in FEATURES:
            if name in drop or row.get(name) is None or (isinstance(row.get(name), float) and math.isnan(row[name])):
                vec.append(0.0)
            else:
                vec.append(float(row[name]))
        cols.append(vec)
    return np.asarray(cols, dtype=float)


def _labels(rows: Sequence[dict]) -> np.ndarray:
    wrong = {"unsupported", "outdated", 1, "1"}
    return np.asarray([1 if r.get("label") in wrong else 0 for r in rows], dtype=int)


def _fit_and_score(train_rows, cal_rows, test_rows, drop, eps, seed):
    from sklearn.linear_model import LogisticRegression
    X = _matrix(train_rows, drop)
    y = _labels(train_rows)
    if len(set(y.tolist())) < 2:
        return None
    clf = LogisticRegression(max_iter=500, random_state=seed)
    clf.fit(X, y)
    cal_p = clf.predict_proba(_matrix(cal_rows, drop))[:, 1]
    test_p = clf.predict_proba(_matrix(test_rows, drop))[:, 1]
    thr = tune_threshold(cal_p.tolist(), [r.get("label") for r in cal_rows], eps)
    decs = decisions_at(test_p.tolist(), thr)
    labels = [r.get("label") for r in test_rows]
    rc = risk_coverage_curve(test_p.tolist(), labels)
    correct = [d == "SUPPORTED" and lab == "supported" or d == "ABSTAINED" and lab != "supported"
               for d, lab in zip(decs, labels)]
    return {
        "hallucination_rate": hallucination_rate(labels, decs),
        "coverage": coverage(decs),
        "aurc": aurc(rc),
        "correct": correct,
        "threshold": None if thr != thr else thr,
    }


def retrain_ablations(
    train_rows: Sequence[dict],
    cal_rows: Sequence[dict],
    test_rows: Sequence[dict],
    full_row: dict,
    seed: int = 42,
    n_bootstrap: int = 10000,
    eps: float = 0.05,
) -> dict:
    if not train_rows or not test_rows:
        return {
            "status": "not_certifiable",
            "reason": "train or test rows are empty",
            "simulated": False,
            "full_regguard": full_row,
        }
    missing = [c for c in FEATURES if c not in train_rows[0]]
    if missing:
        return {
            "status": "not_certifiable",
            "reason": f"cannot retrain; missing columns {missing}",
            "simulated": False,
            "full_regguard": full_row,
        }

    full = _fit_and_score(train_rows, cal_rows, test_rows, drop=[], eps=eps, seed=seed)
    # The published Full RegGuard numbers are the main-table numbers.
    rows: List[dict] = [dict(full_row)]
    if full is None:
        return {
            "status": "not_certifiable",
            "reason": "training labels have a single class",
            "simulated": False,
            "rows": rows,
            "full_regguard": full_row,
        }

    comparisons = []
    for name, drop in SIGNAL_DROPS:
        scored = _fit_and_score(train_rows, cal_rows, test_rows, drop=drop, eps=eps, seed=seed)
        if scored is None:
            rows.append({"condition": name, "status": "not_certifiable"})
            continue
        boot = paired_bootstrap(full["correct"], scored["correct"], n_resamples=n_bootstrap, seed=seed)
        mc = mcnemar(full["correct"], scored["correct"])
        # correct flags are bools; bootstrap wants floats
        rows.append({
            "condition": name,
            "hallucination_rate": scored["hallucination_rate"],
            "coverage": scored["coverage"],
            "aurc": scored["aurc"],
            "threshold": scored["threshold"],
            "bootstrap_mean_diff": boot["mean_diff"],
            "bootstrap_ci95": [boot["ci_low"], boot["ci_high"]],
            "mcnemar_p": mc["p_value"],
        })
        comparisons.append(name)

    # Force the full row to stay identical to the caller-supplied main-table slice.
    rows[0] = {
        "condition": full_row.get("condition", "Full RegGuard (all components)"),
        "hallucination_rate": full_row.get("hallucination_rate"),
        "coverage": full_row.get("coverage"),
        "aurc": full_row.get("aurc"),
    }
    return {
        "status": "computed",
        "simulated": False,
        "rows": rows,
        "full_regguard": rows[0],
        "n_bootstrap": n_bootstrap,
    }
