"""
eval/run_all.py
----------------
M10 Evaluation: Master runner that reproduces ALL tables and figures
with a single command.

    python -m eval.run_all --results_dir results/ --seed 42

Output files
------------
results/
  main_table.json         — RegGuard vs. all baselines
  ablation_table.json     — Signal / component ablation
  violation_rate.json     — Empirical violation rate distribution
  retrieval_metrics.json  — M2 recall/MRR evaluation
  extraction_metrics.json — M4 per-type recall/precision
  latency_breakdown.json  — Per-stage mean latency
  rc_curve.json           — Risk-coverage curve data (for plotting)

Usage
-----
    python -m eval.run_all [--results_dir RESULTS_DIR] [--seed SEED]
                           [--n_splits N_SPLITS] [--eps EPS] [--delta DELTA]
                           [--test_data TEST_JSONL] [--calib_data CALIB_JSONL]
                           [--skip_baselines]

Note
----
The drift calibration experiment has been moved to experiments/drift/.
It is NOT part of the main pipeline, summary.json, main_table, or ablation_table.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("eval.run_all")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── Synthetic data helpers (used when real data is unavailable) ───────────────

def _make_synthetic_data(
    n: int = 500,
    error_rate: float = 0.08,
    seed: int = 42,
) -> Tuple[List[float], List[str], List[str]]:
    """
    Generate synthetic (risk, label, decision) triples for smoke-test evaluation.

    Returns (risks, labels, decisions).
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    wrong_labels = ["unsupported", "outdated"]

    labels    = []
    risks     = []
    decisions = []

    for _ in range(n):
        is_wrong = rng.random() < error_rate
        label    = rng.choice(wrong_labels) if is_wrong else "supported"
        labels.append(label)

        # Wrong atoms get higher risk scores on average
        if is_wrong:
            risk = float(np.clip(np_rng.beta(5, 2), 0.0, 1.0))
        else:
            risk = float(np.clip(np_rng.beta(2, 6), 0.0, 1.0))
        risks.append(round(risk, 4))

        # Decision based on a simple threshold
        if risk <= 0.35:
            decisions.append("SUPPORTED")
        elif risk >= 0.70:
            decisions.append("ABSTAINED")
        else:
            decisions.append("UNCERTAIN")

    return risks, labels, decisions


SYSTEM_NAMES = [
    "RegGuard (ours)",
    "plain_rag",
    "selfcheck",
    "nli_only",
    "llm_judge_all",
    "conformal_no_mondrian",
]


def _build_main_table(cal_rows: list, test_rows: list, eps: float) -> dict:
    """One row per system, each with its own risk. No shared curve and no fixed 0.9."""
    from eval.systems import system_row
    rows = [system_row(name, cal_rows, test_rows, eps) for name in SYSTEM_NAMES]
    aurcs = [r["aurc"] for r in rows if r.get("aurc") is not None and r.get("valid")]
    distinct = len(aurcs) == len(set(round(a, 6) for a in aurcs)) and len(aurcs) == len(SYSTEM_NAMES)
    full = next(r for r in rows if r["system"] == "RegGuard (ours)")
    return {"rows": rows, "full_regguard": full, "aurc_distinct": distinct, "simulated": False}


def _build_ablation_table(train_rows, cal_rows, test_rows, full_row: dict, seed: int) -> dict:
    """Retrain without each signal. Full RegGuard is copied from the main table."""
    from eval.ablation import retrain_ablations
    return retrain_ablations(train_rows, cal_rows, test_rows, full_row, seed=seed)


def _not_available(reason: str) -> dict:
    return {"status": "not_available", "reason": reason, "simulated": False}


# ── Master runner ─────────────────────────────────────────────────────────────

def run_all(
    results_dir:      str = "results/",
    seed:             int = 42,
    n_splits:         int = 1000,
    eps:              float = 0.05,
    delta:            float = 0.10,
    test_data:        Optional[str] = None,
    calib_data:       Optional[str] = None,
    train_data:       Optional[str] = None,
    skip_baselines:   bool = False,
    strict:           bool = False,
    gold_chunks:      Optional[str] = None,
    gold_atoms:       Optional[str] = None,
) -> dict:
    """
    Run all evaluation experiments and write JSON result files.

    Parameters
    ----------
    results_dir    : Directory to write output JSON files.
    seed           : Global random seed.
    n_splits       : Number of calibration/test splits for violation rate.
    eps            : Target error rate (from M7 config).
    delta          : Allowed violation probability.
    test_data      : Path to test JSONL (risk, label, decision per atom).
                     If None, uses synthetic data.
    calib_data     : Path to calibration JSONL.
    skip_baselines : Skip baseline comparison (for speed).

    Returns
    -------
    dict with all results (also written to files).

    Note
    ----
    The drift experiment is in experiments/drift/ and is not run here.
    """
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)

    all_results = {}

    from common.errors import StrictFailure, print_not_paper_valid_banner
    from common.experiment import provenance

    paper_valid = True

    # ── Check Condition 1: Missing aggregator ──────────────────────────────────
    agg_path = ROOT / "models" / "aggregator" / "aggregator_v1.pkl"
    if not agg_path.exists():
        msg = "models/aggregator/aggregator_v1.pkl is missing (no weighted-average fallback)."
        if strict:
            raise StrictFailure(msg)
        print_not_paper_valid_banner(msg)
        paper_valid = False

    simulated = False
    from common.validation import load_and_validate_data
    def _load_jsonl(path: str) -> list:
        return load_and_validate_data(path, strict)

    # ── Load or generate data ──────────────────────────────────────────────────
    if test_data and Path(test_data).exists():
        if "fixture" in str(test_data).lower():
            msg = f"Metrics computed from fixtures ({test_data})."
            if strict:
                raise StrictFailure(msg)
            print_not_paper_valid_banner(msg)
            paper_valid = False
        rows = _load_jsonl(test_data)
        risks     = [r.get("risk", 0.0) for r in rows]
        labels    = [r.get("label", "unsupported") for r in rows]
        decisions = [r.get("decision", "SUPPORTED") for r in rows]
        logger.info("Loaded %d test atoms from %s", len(risks), test_data)
    elif strict:
        raise StrictFailure(
            "Strict run has no held-out test JSONL. Refusing to synthesize metrics."
        )
    else:
        logger.info("No test_data provided; using synthetic data for a non-paper smoke path.")
        msg = "Metrics computed from simulation (no test_data provided)."
        print_not_paper_valid_banner(msg)
        paper_valid = False
        risks, labels, decisions = _make_synthetic_data(n=500, error_rate=0.08, seed=seed)
        rows = [
            {"risk": r, "label": lab, "decision": dec}
            for r, lab, dec in zip(risks, labels, decisions)
        ]
        simulated = True

    # ── Check Condition 2: Signal validation on rows ───────────────────────────
    if len(rows) >= 2:
        from signals import validate_signals
        if not validate_signals(rows, strict=strict):
            paper_valid = False

    cal_rows = rows
    if calib_data and Path(calib_data).exists():
        cal_rows = _load_jsonl(calib_data)
    train_rows = []
    if train_data and Path(train_data).exists():
        train_rows = _load_jsonl(train_data)

    accept_threshold = 0.35
    # ── Check Condition 3: M7 heuristic threshold fallback ────────────────────
    tp = ROOT / "models" / "guarantee" / "thresholds.json"
    if tp.exists():
        from guarantee.certify import load_thresholds
        thresholds = load_thresholds(str(tp))
        for s_name, th in thresholds.items():
            n_acc = th.metadata.get("n_accepted") if hasattr(th, "metadata") and th.metadata else None
            if th.accept_below is None or (n_acc is not None and n_acc == 0):
                msg = f"Stratum '{s_name}' failed M7 certification: threshold={th.accept_below} accepts {n_acc} atoms."
                print(msg)
                if strict:
                    raise StrictFailure(msg)
                print_not_paper_valid_banner(msg)
                paper_valid = False
        if "*|*" in thresholds and thresholds["*|*"].accept_below is not None:
            accept_threshold = thresholds["*|*"].accept_below
    else:
        msg = "Stratum '*|*' failed M7 certification: falling back to heuristic thresholds (accept<0.35, abstain>0.70)."
        print(msg)
        if strict:
            raise StrictFailure(msg)
        print_not_paper_valid_banner(msg)
        paper_valid = False

    # ── Main table ─────────────────────────────────────────────────────────────
    logger.info("Building main comparison table...")
    from eval.metrics import (
        hallucination_rate, coverage, aurc, risk_coverage_curve,
        empirical_violation_rate, judge_call_rate,
    )
    rc_curve = risk_coverage_curve(risks, labels)
    main_metrics = {
        "regguard": {
            "hallucination_rate": hallucination_rate(labels, decisions),
            "coverage":           coverage(decisions),
            "aurc":               aurc(rc_curve),
            "judge_call_rate":    judge_call_rate(decisions),
        }
    }

    full_row = {
        "condition": "Full RegGuard (all components)",
        "hallucination_rate": main_metrics["regguard"]["hallucination_rate"],
        "coverage": main_metrics["regguard"]["coverage"],
        "aurc": main_metrics["regguard"]["aurc"],
    }

    if not skip_baselines:
        if simulated or "s_nli" not in rows[0]:
            main_table = _not_available(
                "per-system risks require signal columns on held-out atoms"
            )
        else:
            main_table = _build_main_table(cal_rows, rows, eps)
            if strict and not main_table.get("aurc_distinct"):
                raise StrictFailure("Per-system AURC values are not distinct.")
            full = main_table.get("full_regguard") or {}
            if full.get("hallucination_rate") is not None:
                full_row = {
                    "condition": "Full RegGuard (all components)",
                    "hallucination_rate": full["hallucination_rate"],
                    "coverage": full["coverage"],
                    "aurc": full["aurc"],
                }
        all_results["main_table"] = main_table
        _write(out_dir / "main_table.json", main_table)
        logger.info("main_table.json written.")

    # ── Ablation table ─────────────────────────────────────────────────────────
    logger.info("Building ablation table...")
    if simulated or not train_rows:
        ablation = {
            "status": "not_certifiable",
            "reason": "ablations retrain on labeled train.jsonl; that file is absent",
            "simulated": False,
            "full_regguard": full_row,
        }
    else:
        ablation = _build_ablation_table(train_rows, cal_rows, rows, full_row, seed)
    all_results["ablation_table"] = ablation
    _write(out_dir / "ablation_table.json", ablation)
    logger.info("ablation_table.json written.")

    # ── Violation rate distribution ────────────────────────────────────────────
    logger.info("Computing empirical violation rate (n_splits=%d)...", n_splits)
    vr_result = empirical_violation_rate(
        risks, labels, accept_threshold, eps,
        n_splits=n_splits, seed=seed,
    )
    all_results["violation_rate"] = vr_result
    _write(out_dir / "violation_rate.json", vr_result)
    logger.info(
        "violation_rate.json: violation_rate=%.4f (delta=%.2f) -> %s",
        vr_result["violation_rate"], delta,
        "✅ PASSES" if vr_result["violation_rate"] <= delta else "❌ FAILS",
    )

    # ── Risk-coverage curve ────────────────────────────────────────────────────
    all_results["rc_curve"] = {"points": rc_curve, "aurc": aurc(rc_curve)}
    _write(out_dir / "rc_curve.json", all_results["rc_curve"])
    logger.info("rc_curve.json written (AURC=%.4f).", aurc(rc_curve))

    # ── Retrieval and extraction (gold files only; never simulated) ─────────
    from eval.gold import extraction_from_gold, retrieval_from_gold
    gold_c = gold_chunks or "data/gold/chunks.jsonl"
    gold_a = gold_atoms or "data/gold/atoms.jsonl"
    if "fixture" in str(gold_c).lower() or "fixture" in str(gold_a).lower():
        msg = "Metrics computed from fixtures."
        if strict:
            raise StrictFailure(msg)
        print_not_paper_valid_banner(msg)
        paper_valid = False
    ret_metrics = retrieval_from_gold(gold_c)
    ext_metrics = extraction_from_gold(gold_a)
    if strict and ret_metrics.get("status") != "computed":
        raise StrictFailure(ret_metrics.get("reason", "retrieval gold missing"))
    if strict and ext_metrics.get("status") != "computed":
        raise StrictFailure(ext_metrics.get("reason", "extraction gold missing"))
    all_results["retrieval_metrics"] = ret_metrics
    all_results["extraction_metrics"] = ext_metrics
    _write(out_dir / "retrieval_metrics.json", ret_metrics)
    _write(out_dir / "extraction_metrics.json", ext_metrics)

    # ── Latency from recorded query timings only ───────────────────────────────
    latencies = [r.get("latency") for r in rows if isinstance(r.get("latency"), dict)]
    if latencies:
        stages = set().union(*[d.keys() for d in latencies])
        latency = {
            stage: round(float(np.mean([d.get(stage, 0.0) for d in latencies])), 4)
            for stage in sorted(stages)
        }
    else:
        latency = _not_available("no recorded per-query latency")
    all_results["latency_breakdown"] = latency
    _write(out_dir / "latency_breakdown.json", latency)

    # ── Summary (one object; tables above are slices of it) ───────────────────
    summary = {
        "seed":                seed,
        "n_atoms":             len(risks),
        "eps":                 eps,
        "delta":               delta,
        "accept_threshold":    accept_threshold,
        "hallucination_rate":  main_metrics["regguard"]["hallucination_rate"],
        "coverage":            main_metrics["regguard"]["coverage"],
        "aurc":                main_metrics["regguard"]["aurc"],
        "violation_rate":      vr_result["violation_rate"],
        "violation_rate_ok":   vr_result["violation_rate"] <= delta,
        "retrieval_recall_at_6": ret_metrics.get("recall_at_6"),
        "retrieval_mrr":         ret_metrics.get("mrr"),
        "simulated":           simulated,
        "paper_valid":         paper_valid,
        "full_regguard":       full_row,
        "generated_on":        date.today().isoformat(),
        **provenance(),
    }
    all_results["summary"] = summary
    _write(out_dir / "summary.json", summary)

    logger.info("=" * 60)
    logger.info("RegGuard Evaluation Complete")
    logger.info("  Hallucination rate (accepted): %.4f (target: <= %.3f)",
                summary["hallucination_rate"], eps)
    logger.info("  Coverage:                      %.4f", summary["coverage"])
    logger.info("  AURC:                          %.4f", summary["aurc"])
    logger.info("  Violation rate:                %.4f (delta=%.2f) -> %s",
                summary["violation_rate"], delta,
                "✅ PASSES" if summary["violation_rate_ok"] else "❌ FAILS")
    logger.info("  Output: %s", out_dir.resolve())
    logger.info("=" * 60)

    return all_results


def _write(path: Path, data) -> None:
    """Write data to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)


def _json_default(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def run_seeds(
    seeds: List[int],
    results_dir: str = "results/",
    **kwargs,
) -> dict:
    """Run every seed into its own directory and summarise intervals."""
    import math
    per_seed = []
    for seed in seeds:
        dest = str(Path(results_dir) / f"seed_{seed}")
        per_seed.append(run_all(results_dir=dest, seed=seed, **kwargs))
    def _col(key):
        vals = []
        for item in per_seed:
            val = item["summary"].get(key)
            if isinstance(val, (int, float)) and val == val:
                vals.append(float(val))
        return vals
    summary = {"seeds": seeds, "per_seed": [p["summary"] for p in per_seed], "intervals": {}}
    for key in ("hallucination_rate", "coverage", "aurc", "violation_rate"):
        vals = _col(key)
        if len(vals) < 2:
            summary["intervals"][key] = None
            continue
        mu = sum(vals) / len(vals)
        var = sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)
        half = 1.96 * math.sqrt(var / len(vals))
        summary["intervals"][key] = {"mean": mu, "ci95": [mu - half, mu + half]}
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "seeds_summary.json", summary)
    from eval.latex_tables import write_latex
    write_latex(per_seed[-1], out / "tables.tex")
    return summary


def _cli():
    parser = argparse.ArgumentParser(
        description="RegGuard M10: Run all evaluation experiments."
    )
    parser.add_argument("--results_dir",    default="results/", help="Output directory")
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--seeds",          default=None,
                        help="Comma-separated seeds. Writes seeds_summary.json and tables.tex")
    parser.add_argument("--n_splits",       type=int,   default=1000,
                        help="Number of calibration/test splits for violation rate")
    parser.add_argument("--eps",            type=float, default=0.05)
    parser.add_argument("--delta",          type=float, default=0.10)
    parser.add_argument("--test_data",      default=None,
                        help="JSONL file with {risk, label, decision} per atom")
    parser.add_argument("--calib_data",     default=None)
    parser.add_argument("--train_data",     default=None)
    parser.add_argument("--gold_chunks",    default=None)
    parser.add_argument("--gold_atoms",     default=None)
    parser.add_argument("--skip_baselines", action="store_true")
    from common.experiment import strict_default
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=strict_default())
    args = parser.parse_args()

    common = dict(
        n_splits=args.n_splits,
        eps=args.eps,
        delta=args.delta,
        test_data=args.test_data,
        calib_data=args.calib_data,
        train_data=args.train_data,
        skip_baselines=args.skip_baselines,
        strict=args.strict,
        gold_chunks=args.gold_chunks,
        gold_atoms=args.gold_atoms,
    )
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        run_seeds(seeds, results_dir=args.results_dir, **common)
    else:
        run_all(results_dir=args.results_dir, seed=args.seed, **common)


if __name__ == "__main__":
    _cli()
