"""
guarantee/simulate.py
----------------------
Repeated random calibration/test split experiments for M7.

This script validates that the LTT guarantee holds empirically by running
many random calibration/test splits on the calibration data and measuring:

  1. The empirical violation rate (fraction of splits where the certified
     threshold admits > epsilon wrong atoms on the held-out test set).
     Expected: violation_rate <= delta.

  2. The coverage distribution (fraction of atoms accepted per split).
     Expected: a tight distribution around the mean coverage.

  3. Drift experiments: compare static vs. drift-aware strategies when
     calibrated on pre-amendment data and tested on post-amendment data.

Usage
-----
    python -m guarantee.simulate \\
        --calib_path data/calibration/atoms_calib.jsonl \\
        --eps        0.05 \\
        --delta      0.10 \\
        --n_splits   1000 \\
        --out_path   results/simulation_results.json

Public API
----------
    from guarantee.simulate import run_simulation, drift_experiment
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from guarantee.ltt import ltt_threshold, simulate_violation_rate
from guarantee.mondrian import build_strata, allocate_delta
from guarantee.drift import sliding_window_filter, recency_weights, effective_sample_size

logger = logging.getLogger("guarantee.simulate")
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Full simulation: violation rate + coverage
# ---------------------------------------------------------------------------

def run_simulation(
    risk:        np.ndarray,
    wrong:       np.ndarray,
    eps:         float = 0.05,
    delta:       float = 0.10,
    n_splits:    int   = 1000,
    calib_frac:  float = 0.5,
    n_grid:      int   = 200,
    use_hb:      bool  = True,
    seed:        int   = 42,
    verbose:     bool  = True,
) -> Dict[str, Any]:
    """
    Validate the LTT guarantee by repeatedly splitting the data into
    calibration and test sets and measuring the empirical violation rate.

    A violation occurs when the certified threshold yields selective risk > eps
    on the held-out test atoms.  The expected violation rate is <= delta.

    Parameters
    ----------
    risk, wrong  : full dataset (all calibration atoms).
    eps          : target risk.
    delta        : confidence level (violation rate should be <= delta).
    n_splits     : number of random splits.
    calib_frac   : fraction of data used for calibration.
    n_grid       : LTT grid resolution.
    use_hb       : use Hoeffding-Bentkus bound.
    seed         : RNG seed.
    verbose      : log per-split progress at DEBUG level.

    Returns
    -------
    dict with summary statistics:
      violation_rate, target_delta, mean_lambda, std_lambda,
      mean_coverage, std_coverage, n_valid_splits, n_abstain_splits.
    """
    result = simulate_violation_rate(
        risk, wrong, eps=eps, delta=delta,
        n_splits=n_splits, calib_frac=calib_frac,
        n_grid=n_grid, use_hb=use_hb, seed=seed,
    )

    # Additional: compute coverage distribution
    rng = np.random.default_rng(seed + 1)
    n   = len(risk)
    coverages = []

    for _ in range(min(n_splits, 200)):  # Sample 200 splits for coverage
        idx   = rng.permutation(n)
        cal_n = max(1, int(calib_frac * n))
        cal_idx, test_idx = idx[:cal_n], idx[cal_n:]

        lam = ltt_threshold(
            risk[cal_idx], wrong[cal_idx], eps, delta, n_grid=n_grid, use_hb=use_hb
        )
        if lam is None:
            coverages.append(0.0)
            continue

        accepted = (risk[test_idx] <= lam).sum()
        coverages.append(accepted / len(test_idx) if len(test_idx) > 0 else 0.0)

    result["mean_coverage"] = float(np.mean(coverages)) if coverages else float("nan")
    result["std_coverage"]  = float(np.std(coverages))  if coverages else float("nan")
    # mean_lambda is already set by simulate_violation_rate; std_lambda is a placeholder
    result["std_lambda"]    = 0.0

    logger.info(
        "Simulation summary: violation_rate=%.4f (target <=%.4f), "
        "mean_coverage=%.3f, mean_lambda=%.3f",
        result["violation_rate"], delta,
        result.get("mean_coverage", float("nan")),
        result.get("mean_lambda",  float("nan")),
    )
    return result


# ---------------------------------------------------------------------------
# Mondrian simulation: per-stratum violation rate
# ---------------------------------------------------------------------------

def run_mondrian_simulation(
    risk:        np.ndarray,
    wrong:       np.ndarray,
    regulators:  List[str],
    atom_types:  List[str],
    eps:         float = 0.05,
    delta:       float = 0.10,
    n_splits:    int   = 500,
    calib_frac:  float = 0.5,
    n_min:       int   = 50,
    n_grid:      int   = 200,
    use_hb:      bool  = True,
    seed:        int   = 42,
) -> Dict[str, Any]:
    """
    Run the violation rate simulation separately per Mondrian stratum.

    For each random split:
      1. Build strata on the calibration half.
      2. Allocate delta (Bonferroni).
      3. Run LTT per stratum.
      4. Measure selective risk on the test half per stratum.

    Returns
    -------
    dict mapping stratum_name -> {violation_rate, n_valid_splits, ...}.
    """
    rng = np.random.default_rng(seed)
    n   = len(risk)

    stratum_results: Dict[str, Dict[str, Any]] = {}

    for split_idx in range(n_splits):
        idx   = rng.permutation(n)
        cal_n = max(1, int(calib_frac * n))
        cal_idx, test_idx = idx[:cal_n], idx[cal_n:]

        cal_regs  = [regulators[i]  for i in cal_idx]
        cal_types = [atom_types[i]  for i in cal_idx]

        strata   = build_strata(
            risk[cal_idx], wrong[cal_idx], cal_regs, cal_types, n_min=n_min
        )
        delta_map = allocate_delta(delta, strata)

        for name, stratum in strata.items():
            d   = delta_map[name]
            lam = ltt_threshold(
                stratum.risk, stratum.wrong, eps, d, n_grid=n_grid, use_hb=use_hb
            )

            if name not in stratum_results:
                stratum_results[name] = {
                    "violations": 0, "valid": 0, "abstains": 0
                }

            if lam is None:
                stratum_results[name]["abstains"] += 1
                continue

            stratum_results[name]["valid"] += 1

            # Test atoms for this stratum
            test_mask = np.array([
                (regulators[i].upper() == name.split("|")[0] or name.startswith("*")) and
                (atom_types[i].upper()  == name.split("|")[1] or name.endswith("*"))
                for i in test_idx
            ], dtype=bool)

            if test_mask.sum() == 0:
                continue

            t_risk  = risk[test_idx][test_mask]
            t_wrong = wrong[test_idx][test_mask]

            accepted = t_risk <= lam
            n_acc    = accepted.sum()
            if n_acc == 0:
                continue

            k        = t_wrong[accepted].sum()
            test_risk = k / n_acc
            if test_risk > eps:
                stratum_results[name]["violations"] += 1

    # Compute violation rates
    summary: Dict[str, Any] = {}
    for name, counts in stratum_results.items():
        v = counts["valid"]
        summary[name] = {
            "violation_rate": counts["violations"] / v if v > 0 else 0.0,
            "n_valid_splits":  v,
            "n_abstain_splits": counts["abstains"],
        }

    logger.info("Mondrian simulation complete for %d strata.", len(summary))
    return summary


# ---------------------------------------------------------------------------
# Drift experiment: pre- vs. post-amendment
# ---------------------------------------------------------------------------

def drift_experiment(
    pre_risk:     np.ndarray,
    pre_wrong:    np.ndarray,
    post_risk:    np.ndarray,
    post_wrong:   np.ndarray,
    pre_dates:    Optional[List[str]] = None,
    post_dates:   Optional[List[str]] = None,
    eps:          float = 0.05,
    delta:        float = 0.10,
    n_grid:       int   = 200,
    use_hb:       bool  = True,
    window_months: int  = 6,
    half_life_days: float = 180.0,
    reference_date: Optional[str] = None,
    seed:          int  = 42,
) -> Dict[str, Any]:
    """
    Compare static vs. drift-aware calibration:
      1. Calibrate all strategies on ``pre_risk``/``pre_wrong``.
      2. Measure selective risk on ``post_risk``/``post_wrong`` (post-amendment).

    Strategies
    ----------
    - static              : calibrate once on all pre-amendment data.
    - sliding_window      : use only recent months of pre-amendment data.
    - recency_weighted    : weight pre-amendment atoms by recency.

    Returns
    -------
    dict with keys: static, sliding_window, recency_weighted.
    Each value is a dict: {accept_below, empirical_risk_post, accepted_fraction}.
    """
    def _evaluate(lam: Optional[float], test_risk: np.ndarray, test_wrong: np.ndarray) -> Dict[str, Any]:
        if lam is None or lam < 0:
            return {"accept_below": lam, "empirical_risk_post": float("nan"), "accepted_fraction": 0.0}
        accepted = test_risk <= lam
        n_acc    = accepted.sum()
        k        = test_wrong[accepted].sum() if n_acc > 0 else 0
        return {
            "accept_below":         lam,
            "empirical_risk_post":  float(k / n_acc) if n_acc > 0 else float("nan"),
            "accepted_fraction":    float(n_acc / len(test_risk)) if len(test_risk) > 0 else 0.0,
        }

    results: Dict[str, Any] = {}

    # Strategy 1: Static
    lam_static = ltt_threshold(pre_risk, pre_wrong, eps, delta, n_grid=n_grid, use_hb=use_hb)
    results["static"] = _evaluate(lam_static, post_risk, post_wrong)

    # Strategy 2: Sliding window
    if pre_dates is not None:
        r_sw, w_sw = sliding_window_filter(
            pre_dates, pre_risk, pre_wrong,
            window_months=window_months, reference_date=reference_date,
        )
        lam_sw = ltt_threshold(r_sw, w_sw, eps, delta, n_grid=n_grid, use_hb=use_hb)
    else:
        lam_sw = lam_static  # fallback
    results["sliding_window"] = _evaluate(lam_sw, post_risk, post_wrong)

    # Strategy 3: Recency-weighted
    if pre_dates is not None:
        w_rw   = recency_weights(pre_dates, half_life_days=half_life_days,
                                 reference_date=reference_date)
        n_eff  = effective_sample_size(w_rw)
        rng    = np.random.default_rng(seed)
        prob   = w_rw / w_rw.sum()
        idx    = rng.choice(len(pre_risk), size=int(n_eff), replace=True, p=prob)
        r_rw, w_rw_vals = pre_risk[idx], pre_wrong[idx]
        lam_rw = ltt_threshold(r_rw, w_rw_vals, eps, delta, n_grid=n_grid, use_hb=use_hb)
    else:
        lam_rw = lam_static
    results["recency_weighted"] = _evaluate(lam_rw, post_risk, post_wrong)

    logger.info(
        "Drift experiment:\n"
        "  static:           accept_below=%.3f  post_risk=%.3f  coverage=%.3f\n"
        "  sliding_window:   accept_below=%.3f  post_risk=%.3f  coverage=%.3f\n"
        "  recency_weighted: accept_below=%.3f  post_risk=%.3f  coverage=%.3f",
        results["static"].get("accept_below", float("nan")),
        results["static"].get("empirical_risk_post", float("nan")),
        results["static"].get("accepted_fraction", 0.0),
        results["sliding_window"].get("accept_below", float("nan")),
        results["sliding_window"].get("empirical_risk_post", float("nan")),
        results["sliding_window"].get("accepted_fraction", 0.0),
        results["recency_weighted"].get("accept_below", float("nan")),
        results["recency_weighted"].get("empirical_risk_post", float("nan")),
        results["recency_weighted"].get("accepted_fraction", 0.0),
    )
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M7: Run LTT simulation experiments.")
    parser.add_argument("--calib_path", required=True,
                        help="Calibration JSONL (risk + label + regulator + atom_type + date)")
    parser.add_argument("--out_path",   required=True,
                        help="Output JSON path for simulation results")
    parser.add_argument("--eps",       type=float, default=0.05)
    parser.add_argument("--delta",     type=float, default=0.10)
    parser.add_argument("--n_splits",  type=int,   default=1000)
    parser.add_argument("--calib_frac",type=float, default=0.5)
    parser.add_argument("--n_grid",    type=int,   default=200)
    parser.add_argument("--use_hb",    action="store_true", default=True)
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    # Load data
    records = []
    with open(args.calib_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    risk  = np.array([float(r.get("risk", 0.5)) for r in records])
    wrong = np.array([int(r.get("label", 0))    for r in records])

    sim_result = run_simulation(
        risk, wrong,
        eps=args.eps, delta=args.delta,
        n_splits=args.n_splits, calib_frac=args.calib_frac,
        n_grid=args.n_grid, use_hb=args.use_hb, seed=args.seed,
    )

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(sim_result, f, indent=2)
    logger.info("Saved simulation results to %s", args.out_path)
