from __future__ import annotations
import json, logging, sys, time
from datetime import date
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(name)-24s %(levelname)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("run_real_pipeline")
QUERIES = [
    {"question": "What is the minimum Capital Adequacy Ratio required for scheduled commercial banks under RBI guidelines?", "query_date": "2024-01-01", "family": "rate"},
    {"question": "What are the KYC norms for opening a savings account as per RBI master circular?", "query_date": "2024-01-01", "family": "applicability"},
    {"question": "What is the timeline for submission of Suspicious Transaction Report to FIU-IND?", "query_date": "2024-01-01", "family": "threshold"},
    {"question": "What are the disclosure requirements for listed companies under SEBI LODR regulations?", "query_date": "2024-01-01", "family": "section"},
    {"question": "What is the penalty for non-compliance with SEBI Insider Trading regulations?", "query_date": "2024-01-01", "family": "entity"},
]
def run_queries(strict: bool = True):
    from app.pipeline import run_pipeline
    all_atoms, query_results = [], []
    for i, q in enumerate(QUERIES, 1):
        logger.info("QUERY %d/%d: %s", i, len(QUERIES), q["question"][:70])
        t0 = time.time()
        result = run_pipeline(question=q["question"], query_date=q["query_date"], skip_judge=False, rewrite_mode="flag", strict=strict)
        elapsed = time.time() - t0
        print(f"\n{'='*70}")
        print(f"QUERY {i}: {q['question']}")
        print(f"Date: {q['query_date']}  | Family: {q['family']}")
        print(f"{'='*70}")
        print(f"STATUS: {result.verification.status_label}  | Coverage: {result.verification.coverage:.1%}")
        if result.disclaimer:
            print(f"DISCLAIMER: {result.disclaimer}")
        ans = result.answer or result.original_answer
        print(f"\nANSWER:\n{ans[:500]}{'...' if len(ans) > 500 else ''}")
        print(f"\n--- ATOMS ({len(result.atoms)}) ---")
        for a in result.atoms:
            bar = "#" * int(a.risk * 10) + "." * (10 - int(a.risk * 10))
            print(f"  [{a.status:12s}] ({a.atom_type:15s}) risk={a.risk:.2f} [{bar}]")
            print(f"    -> '{a.text[:80]}'")
            if a.evidence_quote:
                print(f"    Evidence: '{a.evidence_quote[:60]}...'")
        print(f"\n--- SOURCES ({len(result.sources)}) ---")
        for s in result.sources[:3]:
            print(f"  [{s.chunk_id}] {s.regulator} | {s.section}")
        print(f"\n--- LATENCY ---")
        for stage, sec in result.latency.items():
            print(f"  {stage:12s}: {sec:.3f}s")
        print(f"  {'TOTAL':12s}: {elapsed:.3f}s")
        for a in result.atoms:
            atom_label = "supported" if a.status in ("SUPPORTED","VERIFIED") else "unsupported"
            all_atoms.append({"risk": a.risk, "label": atom_label, "decision": a.status,
                "query_idx": i, "family": q["family"], "atom_type": a.atom_type, "stratum": a.stratum})
        query_results.append({"query": q["question"], "family": q["family"], "status": result.verification.status,
            "coverage": result.verification.coverage, "n_atoms": len(result.atoms),
            "latency": result.latency, "elapsed": elapsed})
    return all_atoms, query_results


def run_m7(strict: bool = True):
    import numpy as np
    from common.errors import StrictFailure, print_not_paper_valid_banner
    from guarantee.ltt import ltt_threshold, simulate_violation_rate
    logger.info("Running M7 guarantee certification (LTT)...")
    if strict:
        raise StrictFailure("run_m7 in run_real_pipeline computes metrics from simulation. Refusing simulated calibration pool in strict mode.")
    print_not_paper_valid_banner("M7 metrics computed from simulation.")
    rng = np.random.RandomState(42)
    n_calib = 300
    risk_arr = rng.beta(2, 5, n_calib)
    wrong_arr = (rng.random(n_calib) < 0.06).astype(float)
    threshold = ltt_threshold(risk_arr, wrong_arr, eps=0.05, delta=0.10)
    sim = simulate_violation_rate(risk_arr, wrong_arr, eps=0.05, delta=0.10, n_splits=200, seed=42)
    logger.info("M7 threshold=%.4f viol_rate=%.4f", threshold or -1, sim.get("violation_rate", -1))
    return {"threshold": threshold, "simulate": sim}


def print_report(qr, m10, m7):
    print("\n\n" + "="*70)
    print("  RegGuard M0->M10 REAL DATASET PIPELINE REPORT")
    print(f"  Generated: {date.today().isoformat()}")
    print("="*70)
    print(f"\n{'#':<4} {'Family':<16} {'Status':<22} {'Coverage':>10} {'Atoms':>6} {'Time':>7}")
    print("-"*70)
    for i, r in enumerate(qr, 1):
        print(f"{i:<4} {r['family']:<16} {r['status']:<22} {r['coverage']:>9.1%} {r['n_atoms']:>6} {r['elapsed']:>6.1f}s")
    s = m10.get("summary", {})
    print("\nM10 EVALUATION (real data):")
    print(f"  Atoms evaluated:    {s.get('n_atoms','N/A')}")
    print(f"  Hallucination rate: {s.get('hallucination_rate',0):.4f}  (target <= {s.get('eps',0.05):.2f})")
    print(f"  Coverage:           {s.get('coverage',0):.4f}")
    print(f"  AURC:               {s.get('aurc',0):.4f}")
    vr = s.get("violation_rate",1.0); vr_ok = s.get("violation_rate_ok",False)
    print(f"  Violation rate:     {vr:.4f}  {'PASSES' if vr_ok else 'FAILS'} (delta=0.10)")
    print(f"  Recall@6:           {s.get('retrieval_recall_at_6',0):.4f}")
    print(f"  MRR:                {s.get('retrieval_mrr',0):.4f}")
    thr = m7.get("threshold"); sim = m7.get("simulate",{})
    print(f"\nM7 GUARANTEE:")
    print(f"  LTT threshold:      {thr:.4f}" if thr else "  LTT threshold:      None (no valid threshold)")
    print(f"  Simulated viol_rate:{sim.get('violation_rate',0):.4f}")
    print(f"  Bound satisfied:    {'YES' if sim.get('violation_rate',1)<0.10 else 'NO'}")
    print("\nOutput files in results/:")
    for fname in ["summary.json","main_table.json","ablation_table.json","violation_rate.json",
                  "drift_table.json","retrieval_metrics.json","extraction_metrics.json",
                  "latency_breakdown.json","rc_curve.json","pipeline_report.json"]:
        p = ROOT / "results" / fname
        print(f"  {'OK' if p.exists() else 'MISSING'} {fname}")
    print("="*70)


def run_m10(all_atoms, strict: bool = True):
    from eval.run_all import run_all
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    test_jsonl = results_dir / "real_test_atoms.jsonl"
    with open(test_jsonl, "w") as f:
        for a in all_atoms:
            f.write(json.dumps(a) + "\n")
    logger.info("Written %d real atoms to %s", len(all_atoms), test_jsonl)
    return run_all(results_dir=str(results_dir), seed=42, n_splits=500, eps=0.05, delta=0.10, test_data=str(test_jsonl), strict=strict)


def main():
    import argparse
    from common.experiment import strict_default
    parser = argparse.ArgumentParser(description="RegGuard real dataset pipeline runner")
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=strict_default())
    args = parser.parse_args()

    print("\n" + "="*70)
    print("  RegGuard: End-to-End Pipeline with REAL DATASET")
    print("  M0->M1(done)->M2->M3->M4->M5->M6->M7->M8->M9->M10")
    print(f"  Dataset: 42,986 chunks (1,827 RBI + 638 SEBI PDFs)")
    print("="*70+"\n")
    all_atoms, query_results = run_queries(strict=args.strict)
    logger.info("Collected %d atoms from %d queries.", len(all_atoms), len(QUERIES))
    m7_result = run_m7(strict=args.strict)
    m10_results = run_m10(all_atoms, strict=args.strict)
    print_report(query_results, m10_results, m7_result)
    report_path = ROOT / "results" / "pipeline_report.json"
    with open(report_path, "w") as f:
        json.dump({"generated": date.today().isoformat(),
            "dataset": {"chroma_chunks": 42986, "rbi_pdfs": 1827, "sebi_pdfs": 638},
            "queries": query_results, "m7": {
                "threshold": m7_result.get("threshold"),
                "violation_rate": m7_result.get("simulate",{}).get("violation_rate")
            }, "m10_summary": m10_results.get("summary", {})},
            f, indent=2, default=str)
    logger.info("Report saved: %s", report_path)


if __name__ == "__main__":
    main()
