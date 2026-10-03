"""
eval/metrics.py
----------------
M10 Evaluation: All metrics for the RegGuard paper.

Metrics implemented
-------------------
Atom-level:
  hallucination_rate(labels, decisions)
      — fraction of *accepted* atoms that are wrong.
  coverage(decisions)
      — fraction of atoms that are SUPPORTED or VERIFIED (not ABSTAINED).

Risk-coverage:
  risk_coverage_curve(risks, labels, thresholds)
      — list of (coverage, hallucination_rate) at each threshold value.
  aurc(rc_curve)
      — Area Under the Risk-Coverage curve (lower is better).

Calibration:
  empirical_violation_rate(risks, labels, accept_thresholds, eps)
      — fraction of random calibration/test splits where the empirical
        selective risk exceeds eps.  Should be <= delta.

Retrieval (M2):
  recall_at_k(retrieved_ids, gold_ids, k)
  mrr(retrieved_ids_list, gold_ids_list)

Extraction (M4):
  extraction_recall(predicted_atoms, gold_atoms, atom_type)
  extraction_precision(predicted_atoms, gold_atoms, atom_type)

System cost / latency:
  judge_call_rate(decisions)  — fraction of atoms that required the judge.
  mean_latency(latency_dicts, stage)

Public API
----------
    from eval.metrics import (
        hallucination_rate, coverage, risk_coverage_curve, aurc,
        empirical_violation_rate, recall_at_k, mrr,
        extraction_recall, extraction_precision,
        judge_call_rate, mean_latency,
    )
"""
from __future__ import annotations

import logging
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import beta

logger = logging.getLogger("eval.metrics")

def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    lower = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    upper = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return (float(lower), float(upper))

# ── Atom-level metrics ────────────────────────────────────────────────────────

def hallucination_rate(
    labels:    List[str],      # ground-truth: "supported"|"unsupported"|"outdated"
    decisions: List[str],      # pipeline decisions: "SUPPORTED"|"VERIFIED"|"ABSTAINED"|...
) -> float:
    """
    Hallucination rate = fraction of *accepted* atoms that are wrong.

    Accepted atoms are those with decision SUPPORTED or VERIFIED.
    Wrong atoms are those with label "unsupported" or "outdated".

    Returns nan if no atoms are accepted.
    """
    accepted_wrong = 0
    n_accepted     = 0
    wrong_labels   = {"unsupported", "outdated"}
    accepted_decs  = {"SUPPORTED", "VERIFIED"}

    for label, decision in zip(labels, decisions):
        if decision in accepted_decs:
            n_accepted += 1
            if label in wrong_labels:
                accepted_wrong += 1

    if n_accepted == 0:
        return float("nan")
    return round(accepted_wrong / n_accepted, 6)


def coverage(decisions: List[str]) -> float:
    """
    Coverage = fraction of atoms that are SUPPORTED or VERIFIED.
    (Atoms that abstain are excluded from the answer, reducing coverage.)
    """
    if not decisions:
        return 0.0
    accepted = sum(1 for d in decisions if d in ("SUPPORTED", "VERIFIED"))
    return round(accepted / len(decisions), 6)


# ── Risk-coverage curve ───────────────────────────────────────────────────────

def risk_coverage_curve(
    risks:      List[float],    # risk score for each atom
    labels:     List[str],      # ground-truth labels
    n_points:   int = 100,
) -> List[Tuple[float, float]]:
    """
    Compute (coverage, hallucination_rate) at each lambda threshold.

    The curve sweeps lambda from 0 to 1 in n_points steps.
    At lambda=0: no atoms accepted, coverage=0.
    At lambda=1: all atoms accepted.

    Returns
    -------
    List of (coverage, hallucination_rate) tuples, one per threshold.
    Points where n_accepted=0 are skipped.
    """
    thresholds = np.linspace(0.0, 1.0, n_points + 1)
    curve = []
    wrong_labels = {"unsupported", "outdated"}
    n = len(risks)

    for lam in thresholds:
        accepted_idx = [i for i, r in enumerate(risks) if r <= lam]
        n_accepted   = len(accepted_idx)
        if n_accepted == 0:
            continue
        n_wrong = sum(1 for i in accepted_idx if labels[i] in wrong_labels)
        cov      = n_accepted / n
        h_rate   = n_wrong   / n_accepted
        curve.append((round(cov, 6), round(h_rate, 6)))

    return curve


def aurc(rc_curve: List[Tuple[float, float]]) -> float:
    """
    Area Under the Risk-Coverage curve (trapezoid rule).

    Lower AURC is better (lower hallucination rate at all coverage levels).
    Returns 0.0 for empty or length-1 curves.
    """
    if len(rc_curve) < 2:
        return 0.0
    coverages = [p[0] for p in rc_curve]
    h_rates   = [p[1] for p in rc_curve]
    return round(float(np.trapezoid(h_rates, coverages)), 6)


# ── Empirical violation rate ──────────────────────────────────────────────────

def empirical_violation_rate(
    risks:           List[float],
    labels:          List[str],
    accept_threshold: float,
    eps:             float,
    n_splits:        int = 1000,
    calib_frac:      float = 0.50,
    seed:            int = 42,
) -> dict:
    """
    Empirical violation rate: how often the certified guarantee is violated.

    For each random calibration/test split:
      1. Use calibration half to determine the lambda (accept_threshold).
      2. On the test half, compute the selective risk (hallucination_rate
         among atoms with risk <= accept_threshold).
      3. A "violation" occurs when selective_risk > eps.

    Parameters
    ----------
    risks            : Risk scores for all atoms.
    labels           : Ground-truth labels (same length).
    accept_threshold : The certified lambda (accept_below from M7).
    eps              : Target error rate from M7 config.
    n_splits         : Number of random splits.
    calib_frac       : Fraction used for calibration in each split.
    seed             : RNG seed.

    Returns
    -------
    dict with keys: violation_rate, n_violations, n_splits, mean_selective_risk,
                    max_selective_risk.
    """
    rng = random.Random(seed)
    n   = len(risks)
    wrong_labels = {"unsupported", "outdated"}
    violations = 0
    selective_risks = []

    for _ in range(n_splits):
        idx = list(range(n))
        rng.shuffle(idx)
        split = int(n * calib_frac)
        test_idx = idx[split:]

        accepted = [i for i in test_idx if risks[i] <= accept_threshold]
        if not accepted:
            continue
        n_wrong = sum(1 for i in accepted if labels[i] in wrong_labels)
        sr = n_wrong / len(accepted)
        selective_risks.append(sr)
        if sr > eps:
            violations += 1

    vr = violations / max(n_splits, 1)
    return {
        "violation_rate":      round(vr, 6),
        "n_violations":        violations,
        "n_splits":            n_splits,
        "mean_selective_risk": round(float(np.mean(selective_risks)) if selective_risks else 0.0, 6),
        "max_selective_risk":  round(float(np.max(selective_risks))  if selective_risks else 0.0, 6),
        "eps":                 eps,
        "accept_threshold":    accept_threshold,
    }


# ── Retrieval metrics ─────────────────────────────────────────────────────────

def recall_at_k(
    retrieved_ids: List[str],
    gold_ids:      List[str],
    k:             int,
) -> float:
    """Recall@k: fraction of gold chunks appearing in top-k retrieved."""
    if not gold_ids:
        return 0.0
    top_k  = set(retrieved_ids[:k])
    gold   = set(gold_ids)
    return round(len(top_k & gold) / len(gold), 6)


def mrr(
    retrieved_ids_list: List[List[str]],
    gold_ids_list:      List[List[str]],
) -> float:
    """Mean Reciprocal Rank."""
    rrs = []
    for retrieved, gold in zip(retrieved_ids_list, gold_ids_list):
        gold_set = set(gold)
        for rank, cid in enumerate(retrieved, start=1):
            if cid in gold_set:
                rrs.append(1.0 / rank)
                break
        else:
            rrs.append(0.0)
    return round(float(np.mean(rrs)) if rrs else 0.0, 6)


def ndcg_at_k(
    retrieved_ids: List[str],
    gold_ids:      List[str],
    k:             int,
) -> float:
    """nDCG@k with binary relevance."""
    import math
    gold = set(gold_ids)
    dcg = 0.0
    for rank, cid in enumerate(retrieved_ids[:k], start=1):
        if cid in gold:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(k, len(gold))
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return round(dcg / idcg, 6) if idcg else 0.0


def precision_at_k(
    retrieved_ids: List[str],
    gold_ids:      List[str],
    k:             int,
) -> float:
    """Precision@k."""
    if not retrieved_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    gold  = set(gold_ids)
    return round(sum(1 for c in top_k if c in gold) / k, 6)


# ── Extraction metrics ────────────────────────────────────────────────────────

def extraction_recall(
    predicted_texts: List[str],
    gold_texts:      List[str],
    normalize:       bool = True,
    fuzzy_threshold: float = 1.0,
) -> float:
    """
    Recall of atom extraction (per type, on text level).
    Normalises by lowercasing and stripping whitespace.
    Supports fuzzy matching via difflib if fuzzy_threshold < 1.0.
    """
    if not gold_texts:
        return 1.0
    norm = (lambda s: s.lower().strip()) if normalize else (lambda s: s)
    pred_set = {norm(p) for p in predicted_texts}
    
    hits = 0
    import difflib
    for g in gold_texts:
        gn = norm(g)
        if gn in pred_set:
            hits += 1
        elif fuzzy_threshold < 1.0:
            # Check for close matches
            for p in pred_set:
                if difflib.SequenceMatcher(None, gn, p).ratio() >= fuzzy_threshold:
                    hits += 1
                    break
                    
    return round(hits / len(gold_texts), 6)


def extraction_precision(
    predicted_texts: List[str],
    gold_texts:      List[str],
    normalize:       bool = True,
    fuzzy_threshold: float = 1.0,
) -> float:
    """Precision of atom extraction."""
    if not predicted_texts:
        return 0.0
    norm = (lambda s: s.lower().strip()) if normalize else (lambda s: s)
    gold_set = {norm(g) for g in gold_texts}
    
    hits = 0
    import difflib
    for p in predicted_texts:
        pn = norm(p)
        if pn in gold_set:
            hits += 1
        elif fuzzy_threshold < 1.0:
            # Check for close matches
            for g in gold_set:
                if difflib.SequenceMatcher(None, pn, g).ratio() >= fuzzy_threshold:
                    hits += 1
                    break
                    
    return round(hits / len(predicted_texts), 6)


def extraction_f1(recall: float, precision: float) -> float:
    """F1 from recall and precision."""
    if recall + precision == 0:
        return 0.0
    return round(2 * recall * precision / (recall + precision), 6)


# ── System cost / latency ─────────────────────────────────────────────────────

def judge_call_rate(decisions: List[str]) -> float:
    """Fraction of atoms that required a V3 judge call (VERIFIED or NOT_VERIFIED)."""
    if not decisions:
        return 0.0
    judged = sum(1 for d in decisions if d in ("VERIFIED", "NOT_VERIFIED"))
    return round(judged / len(decisions), 6)


def mean_latency(latency_dicts: List[Dict[str, float]], stage: str = "total") -> float:
    """Mean latency for a given pipeline stage across queries."""
    vals = [d.get(stage, 0.0) for d in latency_dicts if stage in d]
    return round(float(np.mean(vals)) if vals else 0.0, 4)


# ── Aggregate evaluation report ───────────────────────────────────────────────

def compute_all_metrics(
    risks:        List[float],
    labels:       List[str],
    decisions:    List[str],
    eps:          float = 0.05,
    accept_threshold: float = 0.5,
    retrieved_ids_list: Optional[List[List[str]]] = None,
    gold_ids_list:      Optional[List[List[str]]] = None,
    latency_dicts:      Optional[List[Dict[str, float]]] = None,
    k:            int = 6,
) -> dict:
    """
    Compute all standard evaluation metrics in one call.

    Returns a flat dict suitable for JSON serialization.
    """
    rc_curve = risk_coverage_curve(risks, labels)

    report = {
        "hallucination_rate":  hallucination_rate(labels, decisions),
        "coverage":            coverage(decisions),
        "aurc":                aurc(rc_curve),
        "risk_coverage_curve": rc_curve[:20],   # Truncate for readability
        "judge_call_rate":     judge_call_rate(decisions),
    }

    # Violation rate (expensive — use fewer splits for quick eval)
    vr = empirical_violation_rate(
        risks=risks, labels=labels,
        accept_threshold=accept_threshold, eps=eps,
        n_splits=200,  # Reduced for speed; use 1000 for paper
    )
    report["empirical_violation_rate"] = vr

    # Retrieval metrics
    if retrieved_ids_list and gold_ids_list:
        report["mrr"]        = mrr(retrieved_ids_list, gold_ids_list)
        report["recall_at_k"] = {str(k): float(np.mean([
            recall_at_k(ret, gold, k)
            for ret, gold in zip(retrieved_ids_list, gold_ids_list)
        ]))}

    # Latency
    if latency_dicts:
        for stage in ("retrieve", "generate", "verify", "score", "decide", "total"):
            v = mean_latency(latency_dicts, stage)
            if v > 0:
                report[f"mean_latency_{stage}s"] = v

    return report
