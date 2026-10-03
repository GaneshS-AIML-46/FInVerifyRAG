"""
guarantee/certify.py
--------------------
High-level interface for running LTT certification on calibration data and
saving the resulting per-stratum thresholds to disk.

Usage (CLI)::

    python -m guarantee.certify \\
        --calib_path data/calibration/atoms_calib.jsonl \\
        --out_path   models/guarantee/thresholds.json \\
        --eps_accept 0.05 --delta 0.10 --strict

Public API::

    from guarantee.certify import certify, load_thresholds
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from common.schemas import Thresholds
from common.errors import StrictFailure, print_not_paper_valid_banner
from guarantee.ltt import build_mondrian_thresholds, StrataStatus, ltt_thresholds_pair

logger = logging.getLogger("guarantee.certify")


def certify(
    calib_path: str,
    out_path: str,
    eps_accept: float = 0.05,
    eps_abstain: float = 0.20,
    delta: float = 0.10,
    n_min: int = 50,
    n_grid: int = 200,
    run_simulation: bool = False,
    strict: bool = False,
    strategy: str = "static",
    window_months: int = 12,
    half_life_days: float = 180.0,
    reference_date: Optional[str] = None,
) -> Dict[str, Thresholds]:
    """
    Run LTT certification on calibration data and write thresholds.json.

    Parameters
    ----------
    calib_path      : Path to calibration JSONL (risk, label, regulator, atom_type, date).
    out_path        : Where to write the thresholds JSON.
    eps_accept      : Target selective risk for accept threshold.
    eps_abstain     : Target selective risk for abstain threshold (judge-budget).
    delta           : LTT confidence level (violation rate <= delta with prob >= 1-delta).
    n_min           : Minimum atoms per stratum to attempt standalone certification.
                      Strata below this are merged into the global stratum.
    n_grid          : Number of grid points for lambda search.
    run_simulation  : If True, also run the violation-rate simulation after certifying.
    strict          : If True, raise StrictFailure when any stratum has accept_below=None.
    strategy        : Calibration strategy: "static" | "sliding_window" | "recency_weighted".
    window_months   : For "sliding_window" strategy: how many months to keep.
    half_life_days  : For "recency_weighted" strategy: exponential half-life in days.
    reference_date  : Reference date for recency/window strategies (ISO 8601).
                      Defaults to today if not set.

    Returns
    -------
    Dict[stratum_name, Thresholds]
    """
    # ── Load calibration data ─────────────────────────────────────────────────
    rows = []
    with open(calib_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        msg = f"certify: calib_path {calib_path!r} is empty."
        if strict:
            raise StrictFailure(msg)
        logger.warning(msg)
        rows = []

    # ── Apply strategy filter / weights ──────────────────────────────────────
    if strategy != "static" and rows:
        dates = [r.get("date", "") for r in rows]

        if strategy == "sliding_window":
            from guarantee.drift import sliding_window_filter
            risks_arr  = np.array([float(r.get("risk", 0.5)) for r in rows])
            labels_arr = np.array([int(r.get("label", 0)) for r in rows])
            risks_arr, labels_arr = sliding_window_filter(
                dates, risks_arr, labels_arr,
                window_months=window_months,
                reference_date=reference_date,
            )
            # Rebuild rows subset
            rows = [
                {"risk": float(risks_arr[i]), "label": int(labels_arr[i]),
                 "regulator": rows[i].get("regulator", "*"),
                 "atom_type": rows[i].get("atom_type", "*"),
                 "date": dates[i]}
                for i in range(len(risks_arr))
            ]

        elif strategy == "recency_weighted":
            from guarantee.drift import recency_weights, effective_sample_size
            risks_arr  = np.array([float(r.get("risk", 0.5)) for r in rows])
            labels_arr = np.array([int(r.get("label", 0)) for r in rows])
            w = recency_weights(dates, half_life_days=half_life_days,
                                reference_date=reference_date)
            n_eff = int(effective_sample_size(w))
            rng = np.random.default_rng(42)
            prob = w / w.sum()
            idx = rng.choice(len(rows), size=max(n_eff, 1), replace=True, p=prob)
            rows = [
                {"risk": float(risks_arr[i]), "label": int(labels_arr[i]),
                 "regulator": rows[i].get("regulator", "*"),
                 "atom_type": rows[i].get("atom_type", "*"),
                 "date": dates[i]}
                for i in idx
            ]

    # ── Run Mondrian LTT ──────────────────────────────────────────────────────
    mondrian_results = build_mondrian_thresholds(
        rows, eps_accept, delta, use_regulator=False
    )

    # Ensure global stratum always present
    if "*|*" not in mondrian_results:
        from guarantee.ltt import calibrate_ltt
        all_risks  = np.array([float(r.get("risk", 0.5)) for r in rows])
        all_labels = np.array([1 if r.get("label") in (1, "unsupported", "outdated") else 0
                                for r in rows])
        mondrian_results["*|*"] = calibrate_ltt(all_risks, all_labels, eps_accept, delta)

    # ── Build Thresholds objects ──────────────────────────────────────────────
    thresholds_dict: Dict[str, Thresholds] = {}

    for stratum_name, res in mondrian_results.items():
        if res.status == StrataStatus.CERTIFIED and res.accept_below is not None:
            accept_below = float(res.accept_below)
            # abstain_above: judge-budget threshold slightly above accept_below
            abstain_above = min(1.0, accept_below * 1.5 + 0.05)
            if abstain_above <= accept_below:
                abstain_above = min(1.0, accept_below + 0.05)
        else:
            accept_below = None
            # Fall back to eps-based abstain budget
            abstain_above = min(1.0, eps_abstain)

        n_accepted = res.n_accepted if res.status == StrataStatus.CERTIFIED else 0

        # Strict mode: raise on any uncertified stratum
        if accept_below is None:
            msg = (
                f"Stratum '{stratum_name}' failed M7 certification: "
                f"threshold={accept_below} accepts {n_accepted} atoms."
            )
            print(msg)
            if strict:
                raise StrictFailure(msg)
            print_not_paper_valid_banner(msg)

        thresholds_dict[stratum_name] = Thresholds(
            stratum_name=stratum_name,
            accept_below=accept_below,
            abstain_above=abstain_above,
            metadata={
                "n_accepted": n_accepted,
                "n_calib":    res.n_calib,
                "status":     res.status,
                "strategy":   strategy,
            },
        )

    # ── Save to disk ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {k: v.model_dump() for k, v in thresholds_dict.items()},
            f,
            indent=2,
        )
    logger.info("Thresholds written to %s (%d strata)", out_path, len(thresholds_dict))

    # ── Optional simulation ───────────────────────────────────────────────────
    if run_simulation and rows:
        from guarantee.simulate import run_simulation as _run_sim
        all_risks  = np.array([float(r.get("risk", 0.5)) for r in rows])
        all_labels = np.array([1 if r.get("label") in (1, "unsupported", "outdated") else 0
                                for r in rows])
        sim = _run_sim(all_risks, all_labels, eps=eps_accept, delta=delta, n_splits=200)
        logger.info(
            "Simulation: violation_rate=%.4f (target <=%.2f), "
            "mean_coverage=%.3f, mean_lambda=%.3f",
            sim["violation_rate"], delta,
            sim.get("mean_coverage", float("nan")),
            sim.get("mean_lambda", float("nan")),
        )

    return thresholds_dict


def load_thresholds(path: str) -> Dict[str, Thresholds]:
    """
    Load thresholds.json and return a dict of stratum_name → Thresholds.

    Handles both the new schema (with stratum_name, abstain_above) and
    the legacy schema (accept_below only) by filling in defaults.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    result: Dict[str, Thresholds] = {}
    for k, v in data.items():
        # Backward-compat: fill missing required fields
        if "stratum_name" not in v:
            v = dict(v, stratum_name=k)
        if "abstain_above" not in v:
            ab = v.get("accept_below")
            v = dict(v, abstain_above=min(1.0, (ab * 1.5 + 0.05) if ab is not None else 0.70))
        result[k] = Thresholds(**v)

    return result
