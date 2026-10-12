"""LTT calibration on results/calib.jsonl using trained aggregator."""
from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aggregate.dataset import ATOM_TYPES
from guarantee.ltt import binom_pvalue

CALIB = ROOT / "results" / "calib.jsonl"
RAW = ROOT / "results" / "raw_atoms.jsonl"
MODEL_PATH = ROOT / "models" / "aggregator" / "aggregator_v1.pkl"
CAL_PATH = ROOT / "models" / "aggregator" / "calibrator_v1.pkl"
OUT = ROOT / "results" / "ltt_strata.json"
CFG = ROOT / "configs" / "guarantee.yaml"

FEATURE_ORDER = (
    ["s_ret", "s_ver", "s_nli", "s_ent", "s_div"]
    + ["v1_match", "v1_mismatch", "v1_not_found", "v1_na"]
    + ["V2_output"]
    + [f"type_{t}" for t in ATOM_TYPES]
)

_V1_SCORE = {"MATCH": 0.1, "MISMATCH": 0.9, "NOT_FOUND": 0.7, "NA": 0.5}
N_MIN_STRATUM = 20
N_START = 5
GRID = np.round(np.arange(0.1, 0.951, 0.05), 2)
PARENT_ID = "*|*"


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _v1_onehot(status: str) -> dict[str, int]:
    s = (status or "NA").upper()
    return {
        "v1_match": int(s == "MATCH"),
        "v1_mismatch": int(s == "MISMATCH"),
        "v1_not_found": int(s == "NOT_FOUND"),
        "v1_na": int(s == "NA"),
    }


def _feature_vector(raw: dict) -> np.ndarray:
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
    return np.array([row[c] for c in FEATURE_ORDER], dtype=float)


def _score_atoms(raw_by_id: dict, bundle: dict, calibrator) -> dict[str, float]:
    model = bundle["model"]
    imputer = bundle.get("imputer")
    risks = {}
    for aid, raw in raw_by_id.items():
        x = _feature_vector(raw).reshape(1, -1)
        if imputer is not None:
            x = imputer.transform(x)
        p = float(model.predict_proba(x)[0, 1])
        if calibrator is not None:
            cal = calibrator.get("calibrator", calibrator)
            p = float(cal.predict(np.array([p]))[0])
        risks[aid] = p
    return risks


def _wrong(label: str) -> int:
    lab = (label or "").strip().lower()
    if lab == "unclear":
        raise ValueError("unclear labels must be excluded before LTT")
    return 0 if lab == "supported" else 1


def _largest_passing_threshold(
    risk: np.ndarray,
    wrong: np.ndarray,
    eps: float,
    delta: float,
) -> tuple[float | None, int, float, float]:
    """Return (threshold, n_accepted, selective_risk, p_value_at_choice)."""
    best_lam = None
    best_n = 0
    best_k = 0
    best_p = 1.0
    for lam in GRID:
        acc = risk <= lam
        n_acc = int(np.sum(acc))
        if n_acc < N_START:
            continue
        k = int(np.sum(wrong[acc]))
        p = binom_pvalue(k, n_acc, eps)
        if p <= delta:
            if n_acc > best_n:
                best_n = n_acc
                best_k = k
                best_lam = float(lam)
                best_p = p
    if best_lam is None:
        return None, 0, 0.0, best_p
    sel = best_k / best_n if best_n else 0.0
    return best_lam, best_n, sel, best_p


def main() -> None:
    with open(CFG, encoding="utf-8") as f:
        gcfg = yaml.safe_load(f)["ltt"]
    eps = float(gcfg.get("eps_accept", 0.05))
    delta = float(gcfg.get("delta", 0.10))

    calib = _load_jsonl(CALIB)
    raw_idx = {r["id"]: r for r in _load_jsonl(RAW)}

    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    calibrator = None
    if CAL_PATH.is_file():
        with open(CAL_PATH, "rb") as f:
            calibrator = pickle.load(f)

    # Build calibration rows with risk
    rows = []
    for lab in calib:
        lab_label = (lab.get("label") or "").strip().lower()
        if lab_label == "unclear":
            continue
        aid = lab["atom_id"]
        raw = raw_idx.get(aid)
        if raw is None:
            raise ValueError(f"Missing raw_atoms for {aid}")
        risk_map = _score_atoms({aid: raw}, bundle, calibrator)
        atom_type = raw.get("atom_type", "UNKNOWN")
        if atom_type not in ATOM_TYPES:
            atom_type = "UNKNOWN"
        rows.append(
            {
                "atom_id": aid,
                "atom_type": atom_type,
                "risk": risk_map[aid],
                "wrong": _wrong(lab_label),
            }
        )

    by_type: dict[str, list] = defaultdict(list)
    for r in rows:
        by_type[r["atom_type"]].append(r)

    # Strata to attempt: 6 canonical types + parent
    strata_specs: list[tuple[str, list]] = []
    merged_into_parent: list[dict] = []
    for t in ATOM_TYPES:
        data = by_type.get(t, [])
        if len(data) < N_MIN_STRATUM:
            merged_into_parent.extend(data)
            strata_specs.append((t, []))  # placeholder merged
        else:
            strata_specs.append((t, data))

    parent_data = list(rows)  # parent uses all calib atoms
    strata_specs.append((PARENT_ID, parent_data))

    n_cert_attempts = sum(1 for sid, data in strata_specs if data and sid != PARENT_ID) + 1
    delta_stratum = delta / max(n_cert_attempts, 1)

    out_strata = []
    n_cert, n_merged, n_not = 0, 0, 0

    for stratum_id, data in strata_specs:
        if stratum_id != PARENT_ID and len(data) < N_MIN_STRATUM:
            n_cal = len(by_type.get(stratum_id, []))
            out_strata.append(
                {
                    "stratum_id": stratum_id,
                    "n_calib": n_cal,
                    "n_accepted": 0,
                    "threshold": None,
                    "selective_risk": None,
                    "status": "merged",
                }
            )
            n_merged += 1
            continue

        risk = np.array([d["risk"] for d in data], dtype=float)
        wrong = np.array([d["wrong"] for d in data], dtype=int)
        n_calib = len(data)
        lam, n_acc, sel, _p = _largest_passing_threshold(risk, wrong, eps, delta_stratum)

        if lam is not None:
            status = "certified"
            n_cert += 1
        else:
            status = "not_certifiable"
            n_not += 1

        out_strata.append(
            {
                "stratum_id": stratum_id,
                "n_calib": n_calib,
                "n_accepted": n_acc,
                "threshold": lam,
                "selective_risk": round(sel, 4) if lam is not None else None,
                "status": status,
            }
        )

    OUT.write_text(json.dumps(out_strata, indent=2), encoding="utf-8")
    print(
        f"LTT calibration done. {n_cert} strata certified, {n_merged} merged, {n_not} not_certifiable."
    )
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
