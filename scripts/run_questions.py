"""Run the pipeline over a question JSONL file and write one atom per line.

Question line:
    {"question_id", "question", "query_date", "doc_id"}

Atom line (README schema plus scores). Labels are left empty.
The runner checkpoints after every question and refuses to invent labels.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.checkpoint import append_jsonl, resume_pending
from common.errors import StrictFailure
from common.experiment import load_experiment, provenance


def load_questions(path: str) -> list:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def atom_record(question: dict, atom, signals: dict, source_text: str) -> dict:
    return {
        "question_id": question["question_id"],
        "query": question["question"],
        "query_date": question["query_date"],
        "doc_id": question.get("doc_id") or "",
        "atom_id": atom.atom_id,
        "type": atom.type,
        "text": atom.text,
        "claim": atom.claim,
        "cited_chunk": atom.cited_chunk,
        "source_text": source_text,
        "label": None,
        "signals": signals,
        **{k: signals.get(k) for k in (
            "s_div", "s_ret", "s_ver", "s_nli", "s_ent",
            "s_div_skipped", "s_ret_skipped", "nli_skipped",
            "s_ent_skipped",
        )},
        "parse_failures": question.get("parse_failures", 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run questions and write atom JSONL")
    parser.add_argument("--questions", default=None)
    parser.add_argument("--out", default="data/processed/atoms.jsonl")
    parser.add_argument("--checkpoint", default="results/checkpoints/questions.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    exp = load_experiment()
    qpath = args.questions or exp["paths"]["questions"]
    questions = load_questions(qpath)
    target = exp.get("n_questions")
    if target and not (400 <= target <= 1000):
        raise StrictFailure("n_questions must be between 400 and 1000 for a paper run")
    pending = resume_pending(questions, args.checkpoint, key="question_id")
    if args.limit:
        pending = pending[: args.limit]
    if not pending:
        print("No pending questions.")
        return 0

    # Importing the live pipeline is deferred so --help and resume checks
    # do not load the index. A missing index fails the run; it does not
    # synthesize atoms.
    from app.pipeline import run_pipeline

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stamp = provenance()
    for question in pending:
        if "query_date" not in question:
            raise StrictFailure(f"{question.get('question_id')} has no query_date")
        try:
            result = run_pipeline(
                question["question"],
                query_date=question["query_date"],
                strict=True,
            )
        except Exception as exc:
            append_jsonl(args.checkpoint, {
                "question_id": question["question_id"],
                "status": "ERROR",
                "error": str(exc),
                **stamp,
            })
            raise StrictFailure(f"Query {question['question_id']} ended in ERROR: {exc}") from exc
        append_jsonl(args.checkpoint, {
            "question_id": question["question_id"],
            "status": "OK",
            "n_atoms": len(result.atoms),
            **stamp,
        })
        # The pipeline result does not carry raw signal dicts. Callers that
        # need them re-score from the saved answer. This runner records the
        # atom text and leaves label null.
        for atom in result.atoms:
            record = {
                "question_id": question["question_id"],
                "query": question["question"],
                "query_date": question["query_date"],
                "doc_id": question.get("doc_id", ""),
                "atom_id": getattr(atom, "atom_id", ""),
                "text": getattr(atom, "text", ""),
                "label": None,
                **stamp,
            }
            append_jsonl(out, record)
    print(f"Wrote atoms for {len(pending)} questions to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
