"""
scripts/finalize_questions.py
-----------------------------
Strict validation and finalization of human-reviewed question CSV.

Required columns:
  question_id, question, family, answerable, regulator, split, query_date,
  source_doc_id, seed_chunk_id, evidence_quote, gold_answer_short

Rejects:
  - Missing required columns (ValueError)
  - Template-stem questions (ValueError)
  - Rows with empty query_date or split (ValueError)

Valid rows (keep=yes/YES) are written to:
  - data/questions.jsonl
  - data/questions_pilot.jsonl (split=train, up to 50)
  - data/gold_chunks.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REQUIRED_COLUMNS = [
    "question_id", "question", "family", "answerable", "regulator",
    "split", "query_date", "source_doc_id", "seed_chunk_id",
    "evidence_quote", "gold_answer_short",
]

TEMPLATE_STEMS = [
    "What is the specific rate or threshold mentioned in this",
    "Under which section or provision does",
    "What is the effective date or deadline specified in this",
    "Which categories of entities are required to comply with the",
    "What are the penalties for non-compliance with the",
]


def finalize_questions(csv_path: Path, data_dir: Path | None = None) -> dict:
    """Validate and finalize questions. Returns summary dict."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    if data_dir is None:
        data_dir = ROOT / "data"

    # data/ must exist
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    # Read CSV
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row")
        fieldnames = [fn.strip() for fn in reader.fieldnames]
        rows = [row for row in reader]

    # Check required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in fieldnames]
    if missing_cols:
        raise ValueError(
            f"CSV is missing required columns: {missing_cols}. "
            f"Found columns: {fieldnames}"
        )

    # Check for template stems
    template_offenders = []
    for row in rows:
        q = row.get("question", "")
        for stem in TEMPLATE_STEMS:
            if stem.lower() in q.lower():
                template_offenders.append(row.get("question_id", "?"))
                break
    if template_offenders:
        raise ValueError(
            f"Template-stem questions detected in question_ids: {template_offenders}. "
            "These must be replaced with LLM-generated questions."
        )

    # Check for empty query_date
    missing_date = [row.get("question_id", "?") for row in rows if not row.get("query_date", "").strip()]
    if missing_date:
        raise ValueError(f"Rows with empty query_date: {missing_date}")

    # Check for empty split
    missing_split = [row.get("question_id", "?") for row in rows if not row.get("split", "").strip()]
    if missing_split:
        raise ValueError(f"Rows with empty split: {missing_split}")

    # Filter valid rows
    valid_rows = [r for r in rows if r.get("keep", "").strip().lower() == "yes"]

    # Write data/questions.jsonl
    questions_path = data_dir / "questions.jsonl"
    with open(questions_path, "w", encoding="utf-8") as f:
        for row in valid_rows:
            obj = {c: row.get(c, "") for c in REQUIRED_COLUMNS}
            f.write(json.dumps(obj) + "\n")

    # Write data/questions_pilot.jsonl (train split, up to 50)
    pilot_rows = [r for r in valid_rows if r.get("split", "").strip().lower() == "train"][:50]
    pilot_path = data_dir / "questions_pilot.jsonl"
    with open(pilot_path, "w", encoding="utf-8") as f:
        for row in pilot_rows:
            obj = {c: row.get(c, "") for c in REQUIRED_COLUMNS}
            f.write(json.dumps(obj) + "\n")

    # Write data/gold_chunks.jsonl
    gold_path = data_dir / "gold_chunks.jsonl"
    with open(gold_path, "w", encoding="utf-8") as f:
        for row in valid_rows:
            obj = {
                "question_id": row.get("question_id", ""),
                "gold_chunk_ids": [row.get("seed_chunk_id", "")],
                "note": "lower bound for recall",
            }
            f.write(json.dumps(obj) + "\n")

    summary = {
        "total_rows": len(rows),
        "valid_rows": len(valid_rows),
        "pilot_rows": len(pilot_rows),
        "questions_path": str(questions_path),
        "pilot_path": str(pilot_path),
        "gold_path": str(gold_path),
    }
    print(f"Finalized {len(valid_rows)} valid questions → {questions_path}")
    print(f"Pilot (train, ≤50): {len(pilot_rows)} → {pilot_path}")
    print(f"Gold chunks: {len(valid_rows)} → {gold_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize and validate question CSV.")
    parser.add_argument("csv_path", nargs="?",
                        default=str(ROOT / "data" / "questions_reviewed.csv"),
                        help="Path to reviewed question CSV")
    parser.add_argument("--data-dir", default=None, help="Override data directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else None
    finalize_questions(Path(args.csv_path), data_dir)


if __name__ == "__main__":
    main()
