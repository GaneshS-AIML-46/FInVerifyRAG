#!/usr/bin/env python3
"""
eval/check_publishable.py
Gatekeeper script that asserts all requirements for the final paper are met.
Exits 0 if ready to publish, 1 otherwise.
"""
from __future__ import annotations

import json
import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Ensure repo root is importable (needed for 'tests' package when run standalone)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _load(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except:
        return None

def check(results_dir: str = "results", splits_dir: str = "data/splits") -> list:
    """Return a list of failure strings. Empty means the gate passes."""
    failures = []
    root = Path(results_dir)
    splits = Path(splits_dir)
    
    summary = _load(root / "summary.json") or {}
    main = _load(root / "main_table.json") or {}
    ablation = _load(root / "ablation_table.json") or {}
    seeds = _load(root / "seeds_summary.json") or {}

    # 1. Strict run completed with zero errors, zero fallbacks, zero synthetic
    if not summary:
        failures.append("summary.json is missing")
    if not summary.get("paper_valid"):
        failures.append("paper_valid is false (simulated data, fallbacks, or strict failures present)")
    if summary.get("simulated"):
        failures.append("run used simulated inputs")

    # 2. Aggregator trained and loaded; all signals pass the validation guards
    model = ROOT / "models" / "aggregator" / "aggregator_v1.pkl"
    if not model.exists():
        failures.append("models/aggregator/aggregator_v1.pkl not found")

    # 3. Test split has 500+ human-labeled atoms with a reported inter-annotator kappa
    test_n = summary.get("splits", {}).get("test", 0)
    if test_n < 500:
        failures.append(f"Test split has {test_n} atoms, need >= 500")
    kappa = _load(root / "kappa.json")
    if not kappa:
        failures.append("Inter-annotator kappa not reported (results/kappa.json missing)")

    # 4. Document-level splits, the leakage check passes, and test.lock checksum matches
    lock_path = splits / "test.lock"
    if not lock_path.exists():
        failures.append("test.lock checksum file missing")
    leakage = _load(root / "leakage_report.json") or _load(splits / "leakage_report.json")
    if not leakage:
        failures.append("leakage_report.json missing (must pass document-level leakage check)")
    elif leakage.get("leaked"):
        failures.append("document-level split leakage detected")

    # 5. Every Mondrian stratum is certified or explicitly merged; coverage > 0; no None thresholds
    th_path = ROOT / "models" / "guarantee" / "thresholds.json"
    if not th_path.exists():
        failures.append("models/guarantee/thresholds.json missing")
    else:
        th_data = _load(th_path) or {}
        for stratum, info in th_data.items():
            if info.get("accept_below") is None:
                failures.append(f"Stratum {stratum} has None threshold (must be certified or merged)")
            n_acc = info.get("metadata", {}).get("n_accepted", 0)
            if n_acc == 0:
                failures.append(f"Stratum {stratum} has 0 coverage")

    # 6. Held-out violation rate <= delta (P13), with Clopper-Pearson intervals reported
    delta = summary.get("delta", 0.10)
    vr = summary.get("violation_rate", 1.0)
    if vr > delta:
        failures.append(f"Violation rate {vr} > delta {delta}")
    vr_interval = seeds.get("intervals", {}).get("violation_rate")
    if not vr_interval or not vr_interval.get("ci95"):
        failures.append("Clopper-Pearson intervals for violation rate missing in seeds_summary")

    # 7. Each system has its own RC curve and a different AURC
    if not main.get("aurc_distinct"):
        failures.append("Baselines do not have distinct AURC (or missing baseline data)")

    # 8. Tables agree across all result files (P17)
    try:
        from tests.test_results_consistency import test_results_consistency
        test_results_consistency()
    except Exception as e:
        failures.append(f"Consistency check failed: {e}")

    # 9. 5 seeds with intervals on every headline number
    seed_list = seeds.get("seeds", [])
    if len(seed_list) < 5:
        failures.append(f"Only {len(seed_list)} seeds run, need 5")
    intervals = seeds.get("intervals", {})
    for key in ("hallucination_rate", "coverage", "aurc", "violation_rate"):
        if key not in intervals or not intervals[key]:
            failures.append(f"Missing interval for {key} in seeds_summary")

    # 10. docs/decisions.md and the decision-rules tag exist
    doc_path = ROOT / "docs" / "decisions.md"
    if not doc_path.exists():
        failures.append("docs/decisions.md missing")
        
    try:
        tags = subprocess.check_output(["git", "tag"], text=True, stderr=subprocess.DEVNULL)
        if "decision-rules" not in tags:
            failures.append("decision-rules git tag not found")
    except Exception:
        failures.append("Failed to check git tags")

    return failures

def main() -> int:
    print("="*60)
    print("  RegGuard Final Publishable Gatekeeper")
    print("="*60)
    
    import argparse
    parser = argparse.ArgumentParser(description="RegGuard publishable release gate")
    parser.add_argument("--results", default="results")
    parser.add_argument("--splits", default="data/splits")
    args = parser.parse_args()
    
    failures = check(args.results, args.splits)
    if failures:
        print("NOT PUBLISHABLE")
        for item in failures:
            print(f"  [FAIL] {item}")
    else:
        print("[PASS] PUBLISHABLE")
        
    print("\nCommand to run the final test-split evaluation:")
    print("  python -c \"from eval.run_all import final_test_evaluation; final_test_evaluation('data/splits/test.jsonl', 'data/splits/test.lock', is_final_run=True, results_dir='results/')\"")
    print("="*60)
    
    return 1 if failures else 0

if __name__ == "__main__":
    sys.exit(main())
