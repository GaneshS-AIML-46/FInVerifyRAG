"""
guarantee/certify.py
---------------------
End-to-end pipeline for producing certified thresholds.

This is the main orchestration script for M7.  It:
  1. Reads calibration atoms (JSONL with risk scores and labels).
  2. Builds Mondrian strata by (regulator, atom_type).
  3. Runs LTT threshold selection per stratum.
  4. Saves certified ``Thresholds`` objects to ``thresholds.json``.
  5. Optionally runs the LTT simulation to validate the violation rate.
  6. Supports drift-aware recalibration strategies.

Usage (CLI)
-----------
    python -m guarantee.certify \\
        --calib_path  data/calibration/atoms_calib.jsonl \\
        --out_path    models/guarantee/thresholds.json \\
        --eps_accept  0.05 \\
        --eps_abstain 0.20 \\
        --delta       0.10 \\
        --strategy    static

    # Drift-aware strategies: static | sliding_window | recency_weighted | triggered
    python -m guarantee.certify \\
        --calib_path  data/calibration/atoms_calib.jsonl \\
        --out_path    models/guarantee/thresholds.json \\
        --strategy    sliding_window \\
        --window_months 6

Public API
----------
    from guarantee.certify import certify, load_thresholds
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from common.schemas import Thresholds
from guarantee.ltt import ltt_thresholds_pair, min_accepted, simulate_violation_rate
from guarantee.mondrian import build_strata, allocate_delta, Stratum
from guarantee.drift import (
    sliding_window_filter,
    recency_weights,
    effective_sample_size,
)

logger = logging.getLogger("guarantee.certify")
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Load calibration data from JSONL
# ---------------------------------------------------------------------------

def _load_calib_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Read calibration JSONL.  Expected fields per row:
        risk        : float in [0, 1]  (from M6 aggregator)
        label       : int, 1 = wrong atom
        regulator   : str (SEBI, RBI, INCOMETAX, MF, OTHER)
        atom_type   : str (RATE, THRESHOLD, DATE, SECTION, ENTITY, APPLICABILITY)
        date        : str "YYYY-MM-DD"  (when the atom was produced, for drift)
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Loaded %d calibration records from %s", len(records), path)
    return records


def _records_to_arrays(
    records: List[Dict[str, Any]],
) -> tuple:
    """
    Convert list of dicts to numpy arrays.

    Returns
    -------
    (risk, wrong, regulators, atom_types, dates)
    """
    risk       = np.array([float(r.get("risk", 0.5))  for r in records])
    wrong      = np.array([int(r.get("label", 0))     for r in records])
    regulators = [str(r.get("regulator", ""))          for r in records]
    atom_types = [str(r.get("atom_type", ""))          for r in records]
    dates      = [str(r.get("date", "2000-01-01"))     for r in records]
    return risk, wrong, regulators, atom_types, dates


# ---------------------------------------------------------------------------
# Certify: main function
# ---------------------------------------------------------------------------

def certify(
    calib_path:     str,
    out_path:       str,
    eps_accept:     float = 0.05,
    eps_abstain:    float = 0.20,   # kept for API compat but now controls judge_budget_quantile
    delta:          float = 0.10,
    n_min:          int   = 100,
    n_grid:         int   = 200,
    use_hb:         bool  = True,
    n_start:        int   = 50,     # Bug 4.2: minimum accepted atoms before a lambda is considered
    strategy:       str   = "static",
    window_months:  int   = 6,
    half_life_days: float = 180.0,
    run_simulation: bool  = True,
    n_sim_splits:   int   = 500,
    seed:           int   = 42,
    reference_date: Optional[str] = None,
    strict:         bool  = False,
) -> Dict[str, Thresholds]:
    """
    Build certified thresholds for every Mondrian stratum and save to JSON.

    Parameters
    ----------
    calib_path     : path to calibration JSONL (risk + label per atom).
    out_path       : path where thresholds.json will be written.
    eps_accept     : target selective risk for SUPPORTED atoms.
    eps_abstain    : looser epsilon for ABSTAINED atoms.
    delta          : family-wise Type-I error rate.
    n_min          : minimum atoms per stratum (smaller ones are merged).
    n_grid         : LTT grid resolution.
    use_hb         : use Hoeffding-Bentkus bound (True) or binomial (False).
    strategy       : one of: static | sliding_window | recency_weighted | triggered.
    window_months  : months to keep for sliding_window strategy.
    half_life_days : decay half-life for recency_weighted strategy.
    run_simulation : run LTT simulation to validate violation rate.
    n_sim_splits   : number of simulation splits.
    seed           : RNG seed.
    reference_date : today's date (used by drift strategies).

    Returns
    -------
    Dict[stratum_name -> Thresholds].
    """
    # 1. Load data
    records = _load_calib_jsonl(calib_path)
    if not records:
        raise ValueError(f"No calibration records found in {calib_path}")

    risk, wrong, regulators, atom_types, dates = _records_to_arrays(records)

    # 2. Apply drift-aware calibration strategy
    weights: Optional[np.ndarray] = None

    if strategy == "static":
        pass  # Use all data as-is

    elif strategy == "sliding_window":
        risk, wrong = sliding_window_filter(
            dates, risk, wrong,
            window_months=window_months,
            reference_date=reference_date,
        )
        # Rebuild regulators / atom_types to match filtered arrays
        mask = _sliding_mask(dates, window_months, reference_date)
        regulators = [r for r, m in zip(regulators, mask) if m]
        atom_types = [a for a, m in zip(atom_types, mask) if m]
        logger.info("Sliding window: kept %d / %d atoms", len(risk), len(risk) + (~mask).sum())

    elif strategy == "recency_weighted":
        weights = recency_weights(dates, half_life_days=half_life_days,
                                  reference_date=reference_date)
        n_eff = effective_sample_size(weights)
        logger.info("Recency-weighted: n_eff=%.1f (actual n=%d)", n_eff, len(risk))

    elif strategy == "triggered":
        # In triggered mode the caller is responsible for deciding when to
        # recalibrate; this function is called after the trigger fires, so
        # we just use the provided data as-is (same as static).
        pass

    else:
        raise ValueError(f"Unknown strategy: {strategy!r}. "
                         "Choose: static | sliding_window | recency_weighted | triggered")

    # 3. Build Mondrian strata
    strata: Dict[str, Stratum] = build_strata(
        risk, wrong, regulators, atom_types, n_min=n_min
    )

    # 4. Allocate delta (Bonferroni)
    delta_map = allocate_delta(delta, strata)

    # 5. Run LTT per stratum
    thresholds_map: Dict[str, Thresholds] = {}

    for name, stratum in strata.items():
        d = delta_map[name]

        if len(stratum.risk) == 0:
            logger.warning("Stratum %s: empty — skipping.", name)
            continue

        # For recency-weighted: extract per-stratum weights
        if weights is not None and len(weights) == len(risk):
            # Re-index to stratum's atoms
            s_indices = _stratum_indices(
                name, regulators, atom_types, len(risk)
            )
            s_weights = weights[s_indices]
            n_eff_s   = effective_sample_size(s_weights)
            # Scale n_eff into LTT by resampling proportional to weights
            s_risk, s_wrong = _weighted_resample(
                stratum.risk, stratum.wrong, s_weights, n=int(n_eff_s), seed=seed
            )
        else:
            s_risk, s_wrong = stratum.risk, stratum.wrong

        accept_below, abstain_above = ltt_thresholds_pair(
            s_risk, s_wrong,
            eps_accept=eps_accept,
            delta=d,
            n_grid=n_grid,
            use_hb=use_hb,
            n_start=n_start,
            # Map eps_abstain -> judge_budget_quantile: e.g. 0.20 -> 80th percentile
            judge_budget_quantile=1.0 - eps_abstain,
        )

        # Bug 4.3 fix: accept_below=None is the canonical fail-closed state.
        # Do NOT replace None with -1.0 sentinel; keep None in the schema.
        need = min_accepted(eps_accept, delta)
        n_acc = 0 if accept_below is None else int((np.asarray(s_risk) <= accept_below).sum())
        if accept_below is None or accept_below < 0 or n_acc == 0:
            status = "not_certifiable"
            if n_acc == 0:
                accept_below = None
            msg = f"Stratum '{name}' failed M7 certification: threshold={accept_below} accepts {n_acc} atoms."
            print(msg)
            if strict:
                from common.errors import StrictFailure
                raise StrictFailure(msg)
            from common.errors import print_not_paper_valid_banner
            print_not_paper_valid_banner(msg)
            logger.warning(
                "Stratum %s: not_certifiable (empirical_risk=%.3f, n=%d, need=%d).",
                name, stratum.empirical_risk, stratum.n_atoms, need,
            )
        elif stratum.n_atoms < need or stratum.merged_from:
            status = "merged"
        else:
            status = "certified"

        stratum.accept_below  = accept_below
        stratum.abstain_above = abstain_above

        thresholds_map[name] = Thresholds(
            stratum_name=name,
            accept_below=(
                round(float(accept_below), 4) if accept_below is not None else None
            ),
            abstain_above=round(float(abstain_above), 4),
            metadata={
                "status": status,
                "n_atoms": int(stratum.n_atoms),
                "n_accepted": int(n_acc),
                "min_accepted": int(need),
                "coverage": (n_acc / stratum.n_atoms) if stratum.n_atoms else 0.0,
            },
        )
        logger.info(
            "Stratum %-20s  n=%4d  emp_risk=%.3f  accept_below=%s  abstain_above=%.3f  delta=%.4f",
            name, stratum.n_atoms, stratum.empirical_risk,
            str(accept_below), abstain_above, d,
        )

    # 6. Run simulation (optional — validates that violation_rate <= delta)
    if run_simulation and len(risk) >= 50:
        logger.info("Running LTT simulation on global pool (n=%d, n_splits=%d)...", len(risk), n_sim_splits)
        sim = simulate_violation_rate(
            risk, wrong, eps=eps_accept, delta=delta,
            n_splits=n_sim_splits, use_hb=use_hb, seed=seed,
        )
        logger.info(
            "Simulation: violation_rate=%.4f (target <= %.4f)",
            sim["violation_rate"], delta,
        )
        # Attach simulation result to metadata
        for th in thresholds_map.values():
            th.metadata = {  # type: ignore[attr-defined]
                "simulation": sim,
                "strategy": strategy,
                "eps_accept": eps_accept,
                "eps_abstain": eps_abstain,
                "delta": delta,
                "use_hb": use_hb,
            }

    # 7. Save thresholds.json
    _save_thresholds(thresholds_map, out_path)

    return thresholds_map


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def _save_thresholds(thresholds_map: Dict[str, Thresholds], out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    data = {}
    for name, th in thresholds_map.items():
        entry = {
            "stratum_name":  th.stratum_name,
            "accept_below":  th.accept_below,
            "abstain_above": th.abstain_above,
            "schema_version": th.schema_version,
        }
        # Include simulation metadata if present
        if hasattr(th, "metadata") and th.metadata:
            entry["metadata"] = th.metadata  # type: ignore[attr-defined]
        data[name] = entry

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved %d threshold entries to %s", len(data), out_path)


def load_thresholds(path: str) -> Dict[str, Thresholds]:
    """
    Load thresholds.json and return a dict mapping stratum_name -> Thresholds.

    Includes the ``*|*`` global fallback.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = {}
    for name, entry in data.items():
        th = Thresholds(
            stratum_name=entry["stratum_name"],
            accept_below=entry["accept_below"],
            abstain_above=entry["abstain_above"],
        )
        if "metadata" in entry:
            th.metadata = entry["metadata"]
        result[name] = th
    logger.info("Loaded %d thresholds from %s", len(result), path)
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sliding_mask(
    dates:          List[str],
    window_months:  int,
    reference_date: Optional[str],
) -> np.ndarray:
    """Boolean mask: True for atoms within the sliding window."""
    from datetime import datetime, timedelta
    if reference_date is None:
        ref = datetime.utcnow()
    else:
        ref = datetime.strptime(reference_date[:10], "%Y-%m-%d")
    cutoff = ref - timedelta(days=window_months * 30)
    mask = []
    for d in dates:
        try:
            dt = datetime.strptime(d[:10], "%Y-%m-%d")
            mask.append(dt >= cutoff)
        except ValueError:
            mask.append(True)
    return np.array(mask, dtype=bool)


def _stratum_indices(
    stratum_name: str,
    regulators:   List[str],
    atom_types:   List[str],
    n:            int,
) -> np.ndarray:
    """Return indices of atoms belonging to the given stratum."""
    from guarantee.mondrian import get_stratum
    indices = []
    for i in range(n):
        if get_stratum(regulators[i], atom_types[i]) == stratum_name:
            indices.append(i)
    return np.array(indices, dtype=int)


def _weighted_resample(
    risk:    np.ndarray,
    wrong:   np.ndarray,
    weights: np.ndarray,
    n:       int,
    seed:    int = 42,
) -> tuple:
    """Resample risk/wrong proportional to weights (importance sampling)."""
    if n <= 0 or len(risk) == 0:
        return risk, wrong
    rng  = np.random.default_rng(seed)
    prob = weights / weights.sum()
    idx  = rng.choice(len(risk), size=min(n, len(risk)), replace=True, p=prob)
    return risk[idx], wrong[idx]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="M7: certify risk-control thresholds using Learn-then-Test."
    )
    parser.add_argument("--calib_path",     required=True,          help="Calibration JSONL path")
    parser.add_argument("--out_path",       required=True,          help="Output thresholds.json path")
    parser.add_argument("--eps_accept",     type=float, default=0.05)
    parser.add_argument("--eps_abstain",    type=float, default=0.20)
    parser.add_argument("--delta",          type=float, default=0.10)
    parser.add_argument("--n_min",          type=int,   default=100)
    parser.add_argument("--n_grid",         type=int,   default=200)
    parser.add_argument("--use_hb",         action="store_true",    default=True)
    parser.add_argument("--strategy",       default="static",
                        choices=["static", "sliding_window", "recency_weighted", "triggered"])
    parser.add_argument("--window_months",  type=int,   default=6)
    parser.add_argument("--half_life_days", type=float, default=180.0)
    parser.add_argument("--no_simulation",  action="store_true")
    parser.add_argument("--n_sim_splits",   type=int,   default=500)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--reference_date", default=None)
    from common.experiment import strict_default
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=strict_default())

    args = parser.parse_args()
    certify(
        calib_path=args.calib_path,
        out_path=args.out_path,
        eps_accept=args.eps_accept,
        eps_abstain=args.eps_abstain,
        delta=args.delta,
        n_min=args.n_min,
        n_grid=args.n_grid,
        use_hb=args.use_hb,
        strategy=args.strategy,
        window_months=args.window_months,
        half_life_days=args.half_life_days,
        run_simulation=not args.no_simulation,
        n_sim_splits=args.n_sim_splits,
        seed=args.seed,
        reference_date=args.reference_date,
        strict=args.strict,
    )
