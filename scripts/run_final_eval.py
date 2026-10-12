"""Final evaluation on results/test.jsonl — NO-FILTER mode (LTT not certified)."""
from __future__ import annotations

import hashlib
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aggregate.dataset import ATOM_TYPES
from scripts.run_ltt_calib import _score_atoms, _feature_vector

TEST = ROOT / "results" / "test.jsonl"
RAW = ROOT / "results" / "raw_atoms.jsonl"
MODEL_PATH = ROOT / "models" / "aggregator" / "aggregator_v1.pkl"
CAL_PATH = ROOT / "models" / "aggregator" / "calibrator_v1.pkl"
LOCK_PATH = ROOT / "data" / "splits" / "test.lock"
SUMMARY_OUT = ROOT / "results" / "summary.json"
TABLE_OUT = ROOT / "results" / "main_table.json"

NOTE = (
    "LTT threshold not applied. Conformal guarantee not certified. "
    "Pilot size insufficient."
)


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _verify_lock(path: Path, lock_path: Path) -> None:
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = lock_path.read_text().strip()
    if sha != expected:
        raise RuntimeError(
            f"test.jsonl checksum mismatch!\n  got:      {sha}\n  expected: {expected}"
        )


def _clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact (Clopper-Pearson) 95% CI for a binomial proportion."""
    from scipy.stats import beta as beta_dist
    lo = beta_dist.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta_dist.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def _aurc(risk: np.ndarray, wrong: np.ndarray) -> float:
    """Area under risk-coverage curve (sorted descending by coverage threshold).

    AURC = integral over coverage c in [0,1] of selective_risk(c).
    We approximate by trapezoidal rule over the grid of thresholds.
    """
    order = np.argsort(risk)
    sorted_wrong = wrong[order]
    n = len(risk)
    coverages, risks = [], []
    for i in range(1, n + 1):
        c = i / n
        r = float(sorted_wrong[:i].mean())
        coverages.append(c)
        risks.append(r)
    # Trapezoidal integration
    coverages = np.array(coverages)
    risks_arr = np.array(risks)
    aurc = float(np.trapezoid(risks_arr, coverages))
    return aurc


def main() -> None:
    # 1. Verify test lock
    _verify_lock(TEST, LOCK_PATH)

    # 2. Load data
    test_rows = _load_jsonl(TEST)
    raw_idx = {r["id"]: r for r in _load_jsonl(RAW)}

    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    calibrator = None
    if CAL_PATH.is_file():
        with open(CAL_PATH, "rb") as f:
            calibrator = pickle.load(f)

    # 3. Score + label
    records = []
    skipped_unclear = 0
    for row in test_rows:
        label = (row.get("label") or "").strip().lower()
        if label == "unclear":
            skipped_unclear += 1
            continue
        aid = row["atom_id"]
        raw = raw_idx.get(aid)
        if raw is None:
            print(f"[WARN] {aid} not in raw_atoms; skipping.", file=sys.stderr)
            continue
        risk_val = _score_atoms({aid: raw}, bundle, calibrator)[aid]
        wrong = 0 if label == "supported" else 1
        records.append(
            {
                "atom_id": aid,
                "atom_text": row.get("atom_text", ""),
                "label": label,
                "risk": risk_val,
                "wrong": wrong,
            }
        )

    n_total = len(records)
    n_accepted = n_total  # NO-FILTER: accept all
    wrong_arr = np.array([r["wrong"] for r in records], dtype=int)
    risk_arr = np.array([r["risk"] for r in records], dtype=float)
    n_unsupported = int(wrong_arr.sum())

    coverage = 1.0
    hall_rate = n_unsupported / n_total if n_total else 0.0
    ci_lo, ci_hi = _clopper_pearson(n_unsupported, n_total)
    aurc = _aurc(risk_arr, wrong_arr)

    # 4. Print failure examples
    failures = [r for r in records if r["wrong"] == 1]
    print("\n-- Failure examples (unsupported atoms) --")
    for ex in failures[:5]:
        print(
            f"  atom_id : {ex['atom_id']}\n"
            f"  atom    : {ex['atom_text'][:120]}\n"
            f"  label   : {ex['label']}\n"
            f"  risk    : {ex['risk']:.4f}\n"
        )

    # 5. Write outputs
    summary = {
        "n_total": n_total,
        "n_accepted": n_accepted,
        "n_unsupported": n_unsupported,
        "n_unclear_skipped": skipped_unclear,
        "coverage": coverage,
        "hallucination_rate": round(hall_rate, 4),
        "hallucination_ci_low": round(ci_lo, 4),
        "hallucination_ci_high": round(ci_hi, 4),
        "aurc": round(aurc, 4),
        "ltt_certified": False,
        "note": NOTE,
    }
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    main_table = {
        "system": "RegGuard-pilot-v1",
        "coverage": coverage,
        "hallucination_rate": round(hall_rate, 4),
        "hallucination_ci_low": round(ci_lo, 4),
        "hallucination_ci_high": round(ci_hi, 4),
        "aurc": round(aurc, 4),
        "ltt_certified": False,
        "note": NOTE,
    }
    TABLE_OUT.write_text(json.dumps(main_table, indent=2), encoding="utf-8")

    # 6. Refresh lock
    sha = hashlib.sha256(TEST.read_bytes()).hexdigest()
    LOCK_PATH.write_text(sha, encoding="utf-8")

    print(
        f"Final evaluation done. "
        f"Coverage: {coverage*100:.0f}%. "
        f"Hallucination: {hall_rate*100:.1f}% "
        f"(CI: {ci_lo*100:.1f}%–{ci_hi*100:.1f}%). "
        f"AURC: {aurc:.4f}. "
        f"LTT: not certified."
    )
    print(f"Wrote {SUMMARY_OUT} and {TABLE_OUT}")


if __name__ == "__main__":
    main()
