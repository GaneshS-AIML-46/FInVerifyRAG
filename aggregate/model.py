"""
aggregate/model.py
-------------------
Inference-time aggregator: turns VerifiedAtoms into ScoredAtoms.

This module loads the saved LightGBM / LogReg model + probability calibrator
and applies them to a list of VerifiedAtoms to produce per-atom risk scores
in [0, 1].

Public API
----------
    from aggregate.model import score

    scored_atoms = score(verified_atoms, answer, rr)

    # Each ScoredAtom has:
    #   .verified  — the original VerifiedAtom
    #   .risk      — float in [0,1], 0=safe, 1=high risk of being wrong
    #   .metadata  — includes the raw signal feature dict

Usage without a trained model
------------------------------
If `aggregator_v1.pkl` is not found, `score()` falls back to a simple
weighted average of the available signals so the pipeline can still run
end-to-end before training data is collected.

Configuration
-------------
Model paths are loaded from `configs/signals.yaml` (key: aggregator_model_path
and calibrator_path). Defaults point to `models/aggregator/`.
"""
from __future__ import annotations

import logging
import math
import os
import pickle
from pathlib import Path
from typing import List, Optional

from common.errors import StrictFailure
from common.schemas import GeneratedAnswer, RetrievalResult, ScoredAtom, VerifiedAtom
from signals import compute_all_signals
from aggregate.dataset import FEATURE_COLS

logger = logging.getLogger("aggregate.model")

# Default paths — overridden by signals.yaml / environment variables
_DEFAULT_MODEL_PATH = Path("models/aggregator/aggregator_v1.pkl")
_DEFAULT_CAL_PATH   = Path("models/aggregator/calibrator_v1.pkl")

# -----------------------------------------------------------------------
# Lazy singleton model loader
# -----------------------------------------------------------------------

_model = None
_calibrator = None


def _load_model(model_path: Optional[str] = None) -> Optional[object]:
    global _model
    if _model is not None:
        return _model

    p = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
    if not p.exists():
        logger.error(
            "Aggregator model not found at %s. "
            "Strict paper runs refuse to score. "
            "Train with: python -m aggregate.train",
            p,
        )
        return None

    try:
        with open(p, "rb") as f:
            _model = pickle.load(f)
        logger.info("Loaded aggregator model from %s", p)
        return _model
    except Exception as exc:
        logger.error(
            "Aggregator model at %s is corrupt and cannot be loaded (%s). "
            "Treat as missing. Re-train with: python -m aggregate.train",
            p, exc,
        )
        return None


def _load_calibrator(cal_path: Optional[str] = None) -> Optional[object]:
    global _calibrator
    if _calibrator is not None:
        return _calibrator

    p = Path(cal_path) if cal_path else _DEFAULT_CAL_PATH
    if not p.exists():
        return None

    with open(p, "rb") as f:
        _calibrator = pickle.load(f)
    logger.info("Loaded calibrator from %s", p)
    return _calibrator


# -----------------------------------------------------------------------
# Fallback: simple weighted average when no model is trained yet
# -----------------------------------------------------------------------

_FALLBACK_WEIGHTS = {
    "s_ver":   0.35,
    "s_nli":   0.30,
    "s_ent":   0.15,
    "s_ret":   0.10,   # inverted: high s_ret → low risk
    "s_div":   0.10,   # inverted: high s_div → low risk
}


def _is_missing(val) -> bool:
    if val is None:
        return True
    try:
        return math.isnan(float(val))
    except (TypeError, ValueError):
        return False


def _fallback_risk(feats: dict) -> float:
    """
    Non-strict weighted average. Skipped signals (NaN) are left out.
    They are not treated as 0.5. Returns NaN if every signal is missing.
    """
    total_w = 0.0
    score   = 0.0
    for sig, w in _FALLBACK_WEIGHTS.items():
        if sig not in feats or _is_missing(feats[sig]):
            continue
        val = float(feats[sig])
        if sig in {"s_ret", "s_div"}:
            val = 1.0 - val
        score   += w * val
        total_w += w
    if total_w <= 0:
        return float("nan")
    return round(score / total_w, 4)


# -----------------------------------------------------------------------
# Feature vector builder
# -----------------------------------------------------------------------

def _build_feature_vector(feats: dict, columns: Optional[List[str]] = None) -> list:
    """
    Feature vector in training column order.

    A missing signal is 0 in the value column and 1 in its skip indicator.
    It is not filled with 0.5.
    """
    cols = columns or FEATURE_COLS
    vec = []
    for col in cols:
        val = feats.get(col, float("nan"))
        if _is_missing(val):
            val = 0.0
        vec.append(float(val))
    return vec


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def score(
    verified_atoms: List[VerifiedAtom],
    answer: GeneratedAnswer,
    rr: RetrievalResult,
    model_path: Optional[str] = None,
    calibrator_path: Optional[str] = None,
    strict: bool = False,
) -> List[ScoredAtom]:
    """
    Compute a risk score in [0, 1] for every VerifiedAtom.

    Parameters
    ----------
    verified_atoms   : List of VerifiedAtom objects from M5.
    answer           : GeneratedAnswer with token_logprobs and metadata.
    rr               : RetrievalResult with rerank_scores in metadata.
    model_path       : Override path to aggregator_v1.pkl.
    calibrator_path  : Override path to calibrator_v1.pkl.

    Returns
    -------
    List[ScoredAtom] in the same order as input, each with .risk in [0, 1].
    """
    bundle = _load_model(model_path)
    model = bundle
    feature_order = None
    imputer = None
    if isinstance(bundle, dict) and "model" in bundle:
        feature_order = bundle.get("feature_order")
        imputer = bundle.get("imputer")
        model = bundle["model"]
    if model is None:
        if strict:
            raise StrictFailure(
                "models/aggregator/aggregator_v1.pkl is missing (no weighted-average fallback)."
            )
        from common.errors import print_not_paper_valid_banner
        print_not_paper_valid_banner(
            "models/aggregator/aggregator_v1.pkl is missing (falling back to weighted average)."
        )
    calibrator = _load_calibrator(calibrator_path)

    scored: List[ScoredAtom] = []

    # Build feature matrix for batch prediction (if model available)
    all_feats = []
    for va in verified_atoms:
        feats = compute_all_signals(va, answer, rr, strict=strict)

        # Add atom-type one-hot
        atom_type = va.atom.type
        from aggregate.dataset import ATOM_TYPES, REGULATORS
        for t in ATOM_TYPES:
            feats[f"type_{t}"] = int(atom_type == t)

        # Add regulator one-hot
        reg = (va.atom.metadata.get("regulator", "") if hasattr(va.atom, "metadata")
               else "").upper()
        if reg not in REGULATORS:
            reg = "OTHER"
        for r in REGULATORS:
            feats[f"reg_{r}"] = int(reg == r)

        v2 = getattr(va, "v2_entail_prob", None)
        if v2 is not None:
            feats["V2_output"] = float(v2)

        all_feats.append(feats)

    from signals import validate_signals
    validate_signals(all_feats, strict=strict)

    if model is not None:
        import numpy as np
        X = np.array(
            [_build_feature_vector(f, feature_order) for f in all_feats],
            dtype=float,
        )
        if imputer is not None:
            X = imputer.transform(X)
        raw_probs = model.predict_proba(X)[:, 1]

        # Apply calibration if available
        if calibrator is not None:
            cal_obj = calibrator
            method  = cal_obj.get("method", "isotonic") if isinstance(cal_obj, dict) else "isotonic"
            cal     = cal_obj.get("calibrator", cal_obj) if isinstance(cal_obj, dict) else cal_obj
            if method == "isotonic":
                calibrated = cal.predict(raw_probs)
            elif method == "platt":
                calibrated = cal.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
            else:
                calibrated = raw_probs
        else:
            calibrated = raw_probs

        for va, feats, risk_val in zip(verified_atoms, all_feats, calibrated):
            scored.append(
                ScoredAtom(
                    verified=va,
                    risk=round(float(risk_val), 4),
                    metadata={
                        "signals": {k: v for k, v in feats.items()
                                    if k.startswith("s_") or k.startswith("v1_")
                                    or k.endswith("_skipped")},
                        "model_mode": "lightgbm" if hasattr(model, "predict_proba") else "logreg",
                        "calibrated": calibrator is not None,
                    },
                )
            )
    else:
        # Non-strict only. Skipped signals are excluded, not scored as 0.5.
        for va, feats in zip(verified_atoms, all_feats):
            risk_val = _fallback_risk(feats)
            if _is_missing(risk_val):
                risk_val = float("nan")
            scored.append(
                ScoredAtom(
                    verified=va,
                    risk=risk_val,
                    metadata={
                        "signals": {k: v for k, v in feats.items()
                                    if k.startswith("s_") or k.startswith("v1_")
                                    or k.endswith("_skipped")},
                        "model_mode": "fallback_weighted_average",
                        "calibrated": False,
                    },
                )
            )

    logger.debug(
        "Scored %d atoms. Risks: %s",
        len(scored),
        [s.risk for s in scored],
    )
    return scored
