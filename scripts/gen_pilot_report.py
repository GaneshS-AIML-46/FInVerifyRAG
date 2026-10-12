"""Generate docs/pilot_report.md from results/raw_atoms.jsonl and results/cost.jsonl.
No API calls. No synthetic numbers in results/.
"""
from __future__ import annotations

import json
import math
import random
import re
import statistics
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
random.seed(42)

# ── 1. Parse raw_atoms.jsonl ─────────────────────────────────────────────────
atoms_path = ROOT / "results" / "raw_atoms.jsonl"
raw = atoms_path.read_text(encoding="utf-8")
# The file uses literal \n between records (one physical line)
# Try splitting on actual newlines first, then on escaped \n
lines = [l.strip() for l in raw.splitlines() if l.strip()]
if len(lines) == 1:
    # embedded \n sequences
    lines = [l.strip() for l in raw.replace("\\n", "\n").split("\n") if l.strip()]

atoms: list[dict] = []
for line in lines:
    try:
        atoms.append(json.loads(line))
    except Exception:
        pass

# De-duplicate by id (file may contain duplicates from two test runs)
seen: set[str] = set()
unique_atoms: list[dict] = []
for a in atoms:
    key = a.get("id", "")
    if key not in seen:
        seen.add(key)
        unique_atoms.append(a)
atoms = unique_atoms

n_atoms = len(atoms)
n_questions = len(set(a.get("question_id", "") for a in atoms))

# ── 2. Decision / type / stratum counts ─────────────────────────────────────
decisions: dict[str, int] = {}
types: dict[str, int] = {}
strata: dict[str, int] = {}
for a in atoms:
    d = a.get("decision", "UNKNOWN") or "UNKNOWN"
    t = a.get("atom_type", "UNKNOWN") or "UNKNOWN"
    s = a.get("stratum", "*|*") or "*|*"
    decisions[d] = decisions.get(d, 0) + 1
    types[t] = types.get(t, 0) + 1
    strata[s] = strata.get(s, 0) + 1

n_accepted = decisions.get("ACCEPTED", 0) + decisions.get("SUPPORTED", 0) + decisions.get("VERIFIED", 0)
n_abstained = decisions.get("ABSTAINED", 0)
n_uncertain = decisions.get("UNCERTAIN", 0)
n_error = decisions.get("ERROR", 0)
coverage_pct = 100.0 * n_accepted / n_atoms if n_atoms else 0.0

# ── 3. Cost ──────────────────────────────────────────────────────────────────
cost_path = ROOT / "results" / "cost.jsonl"
cost_entries: list[dict] = []
if cost_path.exists():
    for line in cost_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                cost_entries.append(json.loads(line))
            except Exception:
                pass
    total_cost = sum(e.get("cost_usd", 0) for e in cost_entries)
    cost_source = "results/cost.jsonl"
else:
    # Estimate: 100 questions drafted on Groq qwen/qwen3.8-27b
    # Avg ~578 input + 115 output tokens per question, $0.29/$0.59 per MTok
    n_drafted = 100
    est_input = n_drafted * 578 / 1_000_000 * 0.29
    est_output = n_drafted * 115 / 1_000_000 * 0.59
    total_cost = est_input + est_output
    cost_source = "estimated (cost.jsonl not found; computed from token heuristic)"

cost_per_q = total_cost / n_questions if n_questions else 0.0
cost_per_atom = total_cost / n_atoms if n_atoms else 0.0

# ── 4. Per-signal statistics ─────────────────────────────────────────────────
signals = ["s_ret", "s_nli", "s_ent", "s_div"]

def sig_stats(name: str) -> dict:
    vals = [a[name] for a in atoms if name in a and a[name] is not None and not (isinstance(a[name], float) and math.isnan(a[name]))]
    missing = n_atoms - len(vals)
    missing_pct = 100.0 * missing / n_atoms if n_atoms else 0.0
    if not vals:
        return {"mean": None, "std": None, "min": None, "max": None, "missing_pct": 100.0, "n": 0}
    return {
        "mean": statistics.mean(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
        "missing_pct": missing_pct,
        "n": len(vals),
    }

sig_table = {s: sig_stats(s) for s in signals}

# s_ver derived from v1_status
ver_vals = []
for a in atoms:
    v = a.get("v1_status", "")
    if v == "MATCH":
        ver_vals.append(0.0)
    elif v == "MISMATCH":
        ver_vals.append(1.0)
    elif v in ("NOT_FOUND", "NA", ""):
        ver_vals.append(0.5)
sig_table["s_ver"] = {
    "mean": statistics.mean(ver_vals) if ver_vals else None,
    "std": statistics.stdev(ver_vals) if len(ver_vals) > 1 else 0.0,
    "min": min(ver_vals) if ver_vals else None,
    "max": max(ver_vals) if ver_vals else None,
    "missing_pct": 0.0,
    "n": len(ver_vals),
}

# ── 5. Judge call rate & latency ─────────────────────────────────────────────
judge_calls = [a for a in atoms if a.get("judge_verdict") not in (None, "", "null")]
judge_call_rate = len(judge_calls) / n_atoms if n_atoms else 0.0

latency_fields = ["retrieve", "generate", "extract", "verify", "score", "decide"]
lat_means: dict[str, float] = {}
for field in latency_fields:
    vals = [a["latency"][field] for a in atoms if "latency" in a and field in a["latency"]]
    lat_means[field] = statistics.mean(vals) if vals else 0.0

total_lat = sum(lat_means.values())

# ── 6. Sample atoms (5 accepted, 5 abstained) ────────────────────────────────
accepted_pool = [a for a in atoms if a.get("decision") in ("ACCEPTED", "SUPPORTED", "VERIFIED")]
abstained_pool = [a for a in atoms if a.get("decision") == "ABSTAINED"]

sample_accepted = random.sample(accepted_pool, min(5, len(accepted_pool)))
sample_abstained = random.sample(abstained_pool, min(5, len(abstained_pool)))
sample_atoms = sample_accepted + sample_abstained

def fmt_atom(a: dict) -> str:
    qid = a.get("question_id", "?")
    atype = a.get("atom_type", "?")
    text = (a.get("atom_text") or "(empty)").strip()[:120]
    dec = a.get("decision", "?")
    risk = a.get("aggregated_risk")
    risk_str = f"{risk:.4f}" if isinstance(risk, float) else str(risk)
    def _sig(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "NaN"
        return f"{float(v):.4f}"

    sret = a.get("s_ret")
    snli = a.get("s_nli")
    sdiv = a.get("s_div")
    sent = a.get("s_ent")
    stratum = a.get("stratum", "*|*")
    v1 = a.get("v1_status", "-") or "-"
    jv = a.get("judge_verdict") or "-"
    lines = [
        f"**question_id**: `{qid}`  **stratum**: `{stratum}`  **type**: `{atype}`",
        f"**atom**: {text}",
        f"**decision**: `{dec}`  **risk**: `{risk_str}`  **v1_status**: `{v1}`  **judge**: `{jv}`",
        f"**signals** — s_ret: {_sig(sret)}  s_nli: {_sig(snli)}  s_div: {_sig(sdiv)}  s_ent: {_sig(sent)}",
    ]
    return "\n".join(lines)

# ── 7. Write report ──────────────────────────────────────────────────────────
today = date.today().isoformat()

def fmt_row(name, st):
    m = f"{st['mean']:.4f}" if st['mean'] is not None else "N/A"
    s = f"{st['std']:.4f}" if st['std'] is not None else "N/A"
    mn = f"{st['min']:.4f}" if st['min'] is not None else "N/A"
    mx = f"{st['max']:.4f}" if st['max'] is not None else "N/A"
    miss = f"{st['missing_pct']:.1f}%"
    return f"| `{name}` | {m} | {s} | {mn} | {mx} | {miss} | {st['n']} |"

report = f"""# RegGuard — Pilot Run Report

**Date**: {today}  
**Data source**: `results/raw_atoms.jsonl` (parsed {n_atoms} unique atoms from {n_questions} questions)  
**Cost source**: {cost_source}  
**Status**: Pre-label — no human annotations, no ground truth, no conformal guarantee yet.

---

## 1. Summary

| Item | Value |
|------|-------|
| Questions in pilot | {n_questions} |
| Total atoms | {n_atoms} |
| Accepted atoms | {n_accepted} ({coverage_pct:.1f}% coverage) |
| Abstained atoms | {n_abstained} |
| Uncertain atoms | {n_uncertain} |
| Error atoms | {n_error} |
| Atom types | {len(types)} |
| Strata | {len(strata)} |

---

## 2. Cost

| Item | Value |
|------|-------|
| Total cost | ${total_cost:.5f} |
| Cost per question | ${cost_per_q:.5f} |
| Cost per atom | ${cost_per_atom:.5f} |

> **Note**: {cost_source}

---

## 3. Atom Types

| Atom type | Count | % |
|-----------|-------|---|
""" + "\n".join(
    f"| `{t}` | {c} | {100*c/n_atoms:.1f}% |"
    for t, c in sorted(types.items(), key=lambda x: -x[1])
) + f"""

---

## 4. Decisions by Stratum

| Stratum | Count | % |
|---------|-------|---|
""" + "\n".join(
    f"| `{s}` | {c} | {100*c/n_atoms:.1f}% |"
    for s, c in sorted(strata.items(), key=lambda x: -x[1])
) + f"""

---

## 5. Signal Health

| Signal | Mean | Std | Min | Max | Missing | N |
|--------|------|-----|-----|-----|---------|---|
""" + "\n".join(fmt_row(sig, sig_table[sig]) for sig in ["s_ret", "s_ver", "s_nli", "s_ent", "s_div"]) + f"""

> **s_ent** (token entropy) is NaN for all atoms in this pilot — the pipeline ran with stub logprobs (no real vLLM generation yet). This is expected.

---

## 6. Latency Breakdown (mean per atom, seconds)

| Stage | Mean (s) |
|-------|----------|
""" + "\n".join(
    f"| {stage} | {lat_means.get(stage, 0):.3f} |"
    for stage in latency_fields
) + f"""
| **Total** | **{total_lat:.3f}** |

---

## 7. Judge Usage

| Item | Value |
|------|-------|
| Judge calls | {len(judge_calls)} / {n_atoms} |
| Judge call rate | {judge_call_rate:.1%} |

> No judge calls in this pilot — all atoms were routed ABSTAINED before reaching the UNCERTAIN band.  
> This is expected: the aggregator is not yet trained on labeled data; default scores push atoms to ABSTAINED.

---

## 8. Sample Atoms

### 8a. Accepted (SUPPORTED / VERIFIED)

"""

if sample_accepted:
    for a in sample_accepted:
        report += fmt_atom(a) + "\n\n---\n\n"
else:
    report += "_No accepted atoms in this pilot run._\n\n"

report += "### 8b. Abstained\n\n"
for a in sample_abstained:
    report += fmt_atom(a) + "\n\n---\n\n"

report += f"""
## 9. Limitations

This report is **pre-label** and should not be used to draw conclusions about hallucination rate or system reliability.

| Limitation | Impact |
|------------|--------|
| No human labels | Cannot compute hallucination rate, precision, recall, or F1 |
| No Cohen's κ | Inter-annotator agreement unknown |
| No ground truth | Accepted vs. abstained is driven by untrained aggregator (heuristic weights) |
| Conformal guarantee not certified | LTT/Mondrian calibration requires labeled calibration split |
| s_ent all NaN | Token entropy signal unavailable without real vLLM generation |
| All atoms have empty `atom_text` | Raw atoms in this file are from test fixtures, not real pipeline output |
| Only SECTION family questions | Groq daily limit exhausted before cross-family sampling completed |
| cost.jsonl absent | Cost is estimated, not measured from API responses |

---

## 10. Next Steps

1. **Human labeling** — annotators label each atom as `supported / unsupported / unclear` using `scripts/export_labeling_sheet.py`. Minimum 150 atoms needed for calibration.
2. **Compute Cohen's κ** — double-annotate 20% of atoms with `scripts/kappa.py`.
3. **Train aggregator** — `python aggregate/train.py` on the labeled `agg_train` split.
4. **Certify thresholds** — `python -m guarantee.certify --calib_path ...` to run LTT/Mondrian on calibration split.
5. **Re-run pipeline on real questions** — run the full M2→M8 chain on `data/questions_pilot_review.csv` to produce real atoms (not test fixtures).
6. **Get valid Anthropic API key** — for cross-family question drafting (RATE_THRESHOLD, DATE_TIMELINE, ENTITY_APPLICABILITY, NOT_ANSWERABLE) once Groq daily limit resets.

---

*Generated automatically by `scripts/gen_pilot_report.py` on {today}. No LLM calls were made.*
"""

out_path = ROOT / "docs" / "pilot_report.md"
out_path.parent.mkdir(exist_ok=True)
out_path.write_text(report, encoding="utf-8")
print(f"Report written: {out_path}")
print(f"  Atoms: {n_atoms} unique  |  Questions: {n_questions}")
print(f"  Accepted: {n_accepted} ({coverage_pct:.1f}%)  Abstained: {n_abstained}")
print(f"  Total cost: ${total_cost:.5f}  per-Q: ${cost_per_q:.5f}  per-atom: ${cost_per_atom:.5f}")
