"""Per-system risk scores. Thresholds are tuned on calibration rows only."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from eval.metrics import aurc, coverage, hallucination_rate, risk_coverage_curve

_WRONG = {"unsupported", "outdated", "1", 1}


def _wrong_label(label) -> bool:
    return label in _WRONG or label == 1


def risk_vector(records: Sequence[dict], system: str) -> Optional[List[float]]:
    """
    Each baseline has its own risk. Returns None if the required field is absent.
    """
    def col(name: str) -> Optional[List[float]]:
        if records and name not in records[0]:
            return None
        return [float(r[name]) for r in records]

    if system in {"RegGuard (ours)", "regguard"}:
        return col("risk")
    if system == "plain_rag":
        return [0.0 for _ in records]
    if system == "selfcheck":
        return col("s_ent") or col("selfcheck_risk")
    if system == "nli_only":
        return col("s_nli")
    if system == "llm_judge_all":
        return col("s_ver")
    if system == "conformal_no_mondrian":
        # Global score: unweighted mean of the raw signals, not the aggregator.
        needed = ("s_ver", "s_nli", "s_ent", "s_ret", "s_div")
        if not records or any(name not in records[0] for name in needed):
            return None
        out = []
        for rec in records:
            vals = []
            for name in needed:
                val = rec[name]
                if val is None or val != val:
                    continue
                vals.append(float(val))
            out.append(sum(vals) / len(vals) if vals else 1.0)
        return out
    raise KeyError(system)


def tune_threshold(risks: Sequence[float], labels: Sequence, eps: float) -> float:
    """Largest lambda whose calibration selective risk is <= eps. Never -1."""
    best = None
    grid = np.linspace(0.0, 1.0, 201)
    for lam in grid:
        idx = [i for i, r in enumerate(risks) if r <= lam]
        if not idx:
            continue
        k = sum(1 for i in idx if _wrong_label(labels[i]))
        if k / len(idx) <= eps:
            best = float(lam)
    if best is None:
        return float("nan")
    return best


def decisions_at(risks: Sequence[float], threshold: float) -> List[str]:
    if threshold != threshold:  # NaN
        return ["ABSTAINED"] * len(risks)
    return ["SUPPORTED" if r <= threshold else "ABSTAINED" for r in risks]


def hallucination_at_coverage(
    risks: Sequence[float],
    labels: Sequence,
    target: float,
) -> Optional[float]:
    """Error rate among the lowest-risk atoms that make up `target` coverage."""
    if not risks:
        return None
    order = sorted(range(len(risks)), key=lambda i: risks[i])
    n_take = max(1, int(round(target * len(risks))))
    chosen = order[:n_take]
    k = sum(1 for i in chosen if _wrong_label(labels[i]))
    return k / len(chosen)


def system_row(
    name: str,
    cal_records: Sequence[dict],
    test_records: Sequence[dict],
    eps: float,
) -> dict:
    cal_risk = risk_vector(cal_records, name)
    test_risk = risk_vector(test_records, name)
    if cal_risk is None or test_risk is None:
        return {"system": name, "status": "not_available", "reason": "missing risk field"}
    labels = [r.get("label") for r in test_records]
    cal_labels = [r.get("label") for r in cal_records]
    thr = tune_threshold(cal_risk, cal_labels, eps)
    decs = decisions_at(test_risk, thr)
    rc = risk_coverage_curve(list(test_risk), list(labels))
    judge_calls = 0
    if name == "llm_judge_all":
        judge_calls = len(test_records)
    elif name == "RegGuard (ours)":
        judge_calls = sum(int(r.get("judge_calls", 0) or (r.get("decision") == "UNCERTAIN")) for r in test_records)
    row = {
        "system": name,
        "status": "computed",
        "threshold": None if thr != thr else thr,
        "hallucination_rate": hallucination_rate(list(labels), decs),
        "coverage": coverage(decs),
        "aurc": aurc(rc),
        "judge_call_rate": judge_calls / max(len(test_records), 1),
        "cost_judge_calls": judge_calls,
        "rc_curve": rc,
    }
    for target in (0.40, 0.60, 0.80):
        row[f"hallucination_at_{int(target * 100)}"] = hallucination_at_coverage(
            test_risk, labels, target,
        )
    if row["coverage"] == 0.0 or (thr != thr):
        row["status"] = "invalid_zero_coverage"
        row["valid"] = False
    else:
        row["valid"] = True
    return row
