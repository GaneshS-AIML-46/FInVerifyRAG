"""
experiments/drift/drift_exp.py
------------------------------
Standalone drift experiment — NOT part of the main RegGuard pipeline or paper claims.
See experiments/drift/README.md for status and known limitations.

Experiment design
-----------------
1. Calibrate on pre-amendment data (static baseline).
2. Test on post-amendment data (simulating regulatory drift).
3. Compare:
   a. Static         — threshold from old data, applied to new.
   b. Sliding window — recalibrate on last N months only.
   c. Recency-weighted — exponential decay weighting.
   d. Triggered       — recalibrate only when drift detector fires.

Public API
----------
    from experiments.drift.drift_exp import run_drift_experiment, DriftExperimentResult
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("experiments.drift")


@dataclass
class DriftWindowResult:
    """Result for one strategy on a pre/post split."""
    strategy:          str
    violation_rate:    float
    hallucination_rate: float
    coverage:          float
    n_calibration:     int
    n_test:            int
    threshold:         float = 0.0
    status:            str = "computed"
    recalibrated:      bool = False


@dataclass
class DriftExperimentResult:
    """Full drift experiment result across all strategies."""
    cutoff_date:    str
    n_pre:          int
    n_post:         int
    eps:            float
    delta:          float
    results:        List[DriftWindowResult] = field(default_factory=list)

    def summary_table(self) -> List[dict]:
        return [asdict(r) for r in self.results]

    def best_strategy(self) -> Optional[str]:
        """Strategy with the lowest hallucination_rate (among non-violating)."""
        valid = [r for r in self.results if r.violation_rate <= self.delta]
        if not valid:
            return None
        return min(valid, key=lambda r: r.hallucination_rate).strategy


def calibrate_threshold(
    strategy: str,
    risks: List[float],
    labels: List[str],
    eps: float,
    dates: Optional[List[str]] = None,
    cutoff_date: str = "2023-01-01",
    window_months: int = 6,
    half_life_days: float = 180.0,
    static_threshold: Optional[float] = None,
    static_violation: Optional[float] = None,
    delta: float = 0.10,
) -> Tuple[float, int, bool]:
    """
    Fit one threshold. `strategy` is required and selects the calibration set.

    Detector for `triggered`: selective_risk_monitor. Recalibrate when the
    static threshold's violation rate on the calibration pool exceeds delta.
    """
    if strategy == "static":
        return _compute_threshold_static(risks, labels, eps), len(risks), False

    if strategy == "sliding_window":
        idx = _window_indices(dates, cutoff_date, window_months, len(risks))
        if not idx:
            idx = list(range(len(risks)))
        sub_r = [risks[i] for i in idx]
        sub_l = [labels[i] for i in idx]
        return _compute_threshold_static(sub_r, sub_l, eps), len(idx), True

    if strategy == "recency_weighted":
        weights = _recency_weights(dates, cutoff_date, half_life_days, len(risks))
        return _weighted_threshold(risks, labels, weights, eps), len(risks), True

    if strategy == "triggered":
        if static_threshold is None or static_violation is None:
            raise ValueError("triggered calibration needs the static threshold and its violation rate")
        if static_violation > delta:
            thr, n_cal, _ = calibrate_threshold(
                "sliding_window", risks, labels, eps,
                dates=dates, cutoff_date=cutoff_date, window_months=window_months,
            )
            return thr, n_cal, True
        return static_threshold, len(risks), False

    raise ValueError(f"Unknown drift strategy: {strategy}")


def _window_indices(dates, cutoff_date, window_months, n) -> List[int]:
    if not dates:
        return list(range(n // 2, n))
    from datetime import date as _date, timedelta
    try:
        cutoff = _date.fromisoformat(cutoff_date)
        window_cutoff = (cutoff - timedelta(days=window_months * 30)).isoformat()
        return [i for i, d in enumerate(dates) if d >= window_cutoff]
    except Exception:
        return list(range(n // 2, n))


def _recency_weights(dates, cutoff_date, half_life_days, n) -> np.ndarray:
    if dates:
        from datetime import date as _date
        try:
            cutoff = _date.fromisoformat(cutoff_date)
            weights = []
            for d in dates:
                try:
                    days_ago = max(0, (cutoff - _date.fromisoformat(d)).days)
                    w = np.exp(-days_ago * np.log(2) / half_life_days)
                except Exception:
                    w = 1.0
                weights.append(max(float(w), 1e-6))
            arr = np.array(weights, dtype=float)
            return arr / arr.sum()
        except Exception:
            pass
    arr = np.array([(i + 1) / n for i in range(n)], dtype=float)
    return arr / arr.sum()


def _weighted_threshold(risks, labels, weights, eps) -> float:
    wrong = {"unsupported", "outdated", 1, "1"}
    best = 0.0
    for lam in np.linspace(0.0, 1.0, 200):
        acc_idx = [i for i, r in enumerate(risks) if r <= lam]
        if not acc_idx:
            continue
        w_total = sum(float(weights[i]) for i in acc_idx)
        w_wrong = sum(float(weights[i]) for i in acc_idx if labels[i] in wrong or labels[i] == 1)
        wsr = w_wrong / w_total if w_total > 0 else 0.0
        if wsr <= eps:
            best = float(lam)
    return best


def _compute_threshold_static(risks: List[float], labels: List[str], eps: float) -> float:
    """Find the largest lambda such that selective risk <= eps on calibration data."""
    wrong = {"unsupported", "outdated"}
    grid = np.linspace(0.0, 1.0, 200)
    best = 0.0
    for lam in grid:
        accepted = [i for i, r in enumerate(risks) if r <= lam]
        if not accepted:
            continue
        n_wrong = sum(1 for i in accepted if labels[i] in wrong)
        sr = n_wrong / len(accepted)
        if sr <= eps:
            best = lam
    return best


def _selective_risk(risks: List[float], labels: List[str], threshold: float) -> Tuple[float, float]:
    """Return (hallucination_rate, coverage) for a threshold on a test set."""
    wrong  = {"unsupported", "outdated"}
    idx    = [i for i, r in enumerate(risks) if r <= threshold]
    cov    = len(idx) / max(len(risks), 1)
    if not idx:
        return float("nan"), cov
    n_wrong = sum(1 for i in idx if labels[i] in wrong)
    return round(n_wrong / len(idx), 6), round(cov, 6)


def _violation_rate(
    risks_test:  List[float],
    labels_test: List[str],
    threshold:   float,
    eps:         float,
    n_splits:    int = 200,
    seed:        int = 42,
) -> float:
    """Estimate empirical violation rate on test data via random sub-splits."""
    rng   = random.Random(seed)
    wrong = {"unsupported", "outdated"}
    n = len(risks_test)
    violations = 0

    for _ in range(n_splits):
        idx = list(range(n))
        rng.shuffle(idx)
        test_half = idx[n // 2 :]
        accepted  = [i for i in test_half if risks_test[i] <= threshold]
        if not accepted:
            continue
        n_wrong = sum(1 for i in accepted if labels_test[i] in wrong)
        if n_wrong / len(accepted) > eps:
            violations += 1

    return round(violations / n_splits, 6)


def run_drift_experiment(
    risks_pre:    List[float],     # Risk scores for pre-amendment atoms
    labels_pre:   List[str],       # Ground-truth labels for pre-amendment atoms
    risks_post:   List[float],     # Risk scores for post-amendment atoms
    labels_post:  List[str],       # Ground-truth labels for post-amendment atoms
    dates_pre:    Optional[List[str]] = None,  # ISO dates for pre atoms
    dates_post:   Optional[List[str]] = None,  # ISO dates for post atoms
    cutoff_date:  str = "2023-01-01",
    eps:          float = 0.05,
    delta:        float = 0.10,
    window_months: int  = 6,
    half_life_days: float = 180.0,
    seed:         int = 42,
) -> DriftExperimentResult:
    """
    Run the drift experiment comparing all 4 calibration strategies.

    Parameters
    ----------
    risks_pre / labels_pre   : Atoms from before the amendment.
    risks_post / labels_post : Atoms from after the amendment.
    dates_pre / dates_post   : Optional ISO date strings per atom.
    cutoff_date              : Amendment date (split boundary).
    eps                      : Target selective risk.
    delta                    : Allowed violation probability.
    window_months            : Sliding window size (months).
    half_life_days           : Exponential decay half-life for recency weights.
    seed                     : RNG seed.

    Returns
    -------
    DriftExperimentResult with per-strategy metrics.
    """
    result = DriftExperimentResult(
        cutoff_date=cutoff_date,
        n_pre=len(risks_pre),
        n_post=len(risks_post),
        eps=eps,
        delta=delta,
    )

    if not risks_pre or not risks_post:
        logger.warning("Empty pre or post split; skipping drift experiment.")
        return result

    too_small = len(risks_pre) < 150 or len(risks_post) < 150
    size_status = "not_certifiable" if too_small else "computed"
    if too_small:
        logger.warning(
            "Drift split is below 150 atoms per side (pre=%d, post=%d). "
            "Results are not publishable.",
            len(risks_pre), len(risks_post),
        )

    static_threshold, n_static, rec_static = calibrate_threshold(
        "static", risks_pre, labels_pre, eps,
        dates=dates_pre, cutoff_date=cutoff_date,
        window_months=window_months, half_life_days=half_life_days,
    )
    vr_static = _violation_rate(risks_post, labels_post, static_threshold, eps, seed=seed)

    plans = [("static", static_threshold, n_static, rec_static)]
    for name in ("sliding_window", "recency_weighted"):
        thr, n_cal, rec = calibrate_threshold(
            name, risks_pre, labels_pre, eps,
            dates=dates_pre, cutoff_date=cutoff_date,
            window_months=window_months, half_life_days=half_life_days,
        )
        plans.append((name, thr, n_cal, rec))
    thr_t, n_t, rec_t = calibrate_threshold(
        "triggered", risks_pre, labels_pre, eps,
        dates=dates_pre, cutoff_date=cutoff_date,
        window_months=window_months, half_life_days=half_life_days,
        static_threshold=static_threshold, static_violation=vr_static, delta=delta,
    )
    plans.append(("triggered", thr_t, n_t, rec_t))

    for name, thr, n_cal, rec in plans:
        hr, cov = _selective_risk(risks_post, labels_post, thr)
        vr = vr_static if name == "static" else _violation_rate(
            risks_post, labels_post, thr, eps, seed=seed,
        )
        result.results.append(DriftWindowResult(
            strategy=name,
            violation_rate=vr,
            hallucination_rate=hr if not np.isnan(hr) else 1.0,
            coverage=cov,
            n_calibration=n_cal,
            n_test=len(risks_post),
            threshold=float(thr),
            status=size_status,
            recalibrated=rec,
        ))

    logger.info("Drift experiment complete: %d strategies", len(result.results))
    for r in result.results:
        logger.info(
            "  %s: vr=%.4f  hr=%.4f  cov=%.4f",
            r.strategy, r.violation_rate, r.hallucination_rate, r.coverage,
        )
    return result
