"""Draft regulatory Q&A questions from seed chunks.

Usage:
    python scripts/draft_questions.py [--dry-run]

Outputs:
    data/questions_pilot_review.csv  (50 questions, stratified)
    data/questions_draft.csv         (300-500 questions)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

# Make repo root importable when run as a script
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.errors import LLMError

SYSTEM_PROMPT = (
    "You are a regulatory Q&A expert drafting questions over Indian financial "
    "regulation documents. Return JSON only."
)

NOT_ANSWERABLE_SYSTEM = (
    "You are a regulatory Q&A expert. Return JSON only."
)

SELF_REF_PATTERNS = re.compile(
    r"\bthis circular\b|\bthis document\b|\bthe above\b", re.IGNORECASE
)

CHECKPOINT_FILE = ROOT / "results" / "draft_checkpoint.jsonl"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _build_prompt(doc_id: str, regulator: str, family: str, chunk_text: str) -> str:
    return (
        f"CHUNK (doc_id={doc_id}, regulator={regulator}, family={family}):\n"
        f"{chunk_text[:2000]}\n\n"
        f"Draft ONE question that:\n"
        f"- Is standalone: names the regulator and instrument/topic explicitly "
        f"(no \"this circular\", \"this document\", \"the above\")\n"
        f"- Is complete and in English\n"
        f"- Has a specific answerable fact of type {family} as the answer\n"
        f"- Does NOT reveal the answer in the question\n\n"
        f"Return JSON:\n"
        f"{{\n"
        f'  "question": "...",\n'
        f'  "evidence_quote": "exact sentence from chunk",\n'
        f'  "gold_answer_short": "brief answer",\n'
        f'  "family": "{family}"\n'
        f"}}"
    )


def _build_not_answerable_prompt(doc_id: str, regulator: str, family: str, chunk_text: str) -> str:
    return (
        f"CHUNK (doc_id={doc_id}, regulator={regulator}, family={family}):\n"
        f"{chunk_text[:2000]}\n\n"
        f"Draft ONE plausible question about {regulator} regulations that is NOT answerable "
        f"from the chunk above. The question must be standalone (no 'this circular', "
        f"'this document', 'the above'), complete, and in English.\n\n"
        f"Return JSON:\n"
        f"{{\n"
        f'  "question": "...",\n'
        f'  "evidence_quote": "",\n'
        f'  "gold_answer_short": "",\n'
        f'  "family": "{family}"\n'
        f"}}"
    )


def _validate_response(
    resp_text: str,
    chunk_text: str,
    requested_family: str,
) -> tuple[dict | None, str | None]:
    """Parse and validate LLM JSON response.

    Returns (parsed_dict, drop_reason) — one of them is None.
    """
    try:
        data = json.loads(resp_text)
    except json.JSONDecodeError:
        # Try extracting JSON from response
        m = re.search(r"\{.*\}", resp_text, re.DOTALL)
        if not m:
            return None, "invalid_json"
        try:
            data = json.loads(m.group())
        except json.JSONDecodeError:
            return None, "invalid_json"

    question = data.get("question", "")
    evidence_quote = data.get("evidence_quote", "")
    family = data.get("family", "")

    # Validate family match
    if family and family != requested_family:
        return None, "family_mismatch"

    # Validate self-referential
    if SELF_REF_PATTERNS.search(question):
        return None, "self_referential"

    # Validate evidence quote is substring (for answerable questions)
    if evidence_quote:
        norm_chunk = _normalize(chunk_text)
        norm_quote = _normalize(evidence_quote)
        if norm_quote and norm_quote not in norm_chunk:
            return None, "evidence_not_substring"

    return data, None


def _make_row(
    question: str,
    family: str,
    answerable: str,
    regulator: str,
    doc_id: str,
    seed_chunk_id: str,
    evidence_quote: str,
    gold_answer_short: str,
    split: str = "draft",
) -> dict:
    return {
        "question_id": str(uuid.uuid4()),
        "question": question,
        "family": family,
        "answerable": answerable,
        "regulator": regulator,
        "split": split,
        "query_date": datetime.now(timezone.utc).date().isoformat(),
        "source_doc_id": doc_id,
        "seed_chunk_id": seed_chunk_id,
        "evidence_quote": evidence_quote,
        "gold_answer_short": gold_answer_short,
        "retrieval_check": "pending",
        "keep": "",
        "edited_question": "",
        "notes": "",
    }


def _write_checkpoint(rows: list[dict]) -> None:
    CHECKPOINT_FILE.parent.mkdir(exist_ok=True)
    with open(CHECKPOINT_FILE, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _synthetic_seed_chunks() -> list[dict]:
    """5 synthetic in-memory seed chunks for --dry-run when no CSV exists."""
    return [
        {
            "seed_chunk_id": f"SYNTH_{i:03d}",
            "doc_id": f"RBI_SYNTH_{i:03d}",
            "regulator": "RBI",
            "family_hint": fam,
            "chunk_text": (
                f"The Reserve Bank of India, under Circular No. {i}/2024, mandates that "
                f"all scheduled commercial banks must maintain a minimum {fam.lower()} ratio "
                f"of {5+i}% effective from April 1, 2024. Non-compliance will attract a penalty "
                f"of INR {i*10} lakh per instance. Banks must report compliance quarterly."
            ),
        }
        for i, fam in enumerate(
            ["RATE_THRESHOLD", "SECTION", "DATE_TIMELINE", "ENTITY_APPLICABILITY", "RATE_THRESHOLD"], 1
        )
    ]


def _try_llm(llm_client, user_prompt: str, system_prompt: str, seed_chunk_id: str) -> dict | None:
    """Call llm_client.chat() and return result dict, or None on failure. No template fallback."""
    try:
        result = llm_client.chat(user=user_prompt, system=system_prompt)
        return result
    except Exception as e:
        raise LLMError(f"LLM call failed for chunk {seed_chunk_id}: {e}") from e


def run_draft(cfg: dict, llm_client, dry_run: bool = False) -> list[dict]:
    """Core drafting logic. llm_client must have .chat() interface."""
    seed_path = ROOT / "data" / "seed_chunks.csv"

    if seed_path.exists():
        df = pd.read_csv(seed_path)
        chunks = df.to_dict("records")
    else:
        if dry_run:
            chunks = _synthetic_seed_chunks()
        else:
            raise FileNotFoundError(
                f"seed_chunks.csv not found at {seed_path}. "
                "Run select_seed_chunks.py first."
            )

    family_weights = cfg.get("question_families", {})
    not_answerable_pct = family_weights.get("NOT_ANSWERABLE", 0.10)

    rows: list[dict] = []
    seen_questions: set[str] = set()
    batch_since_checkpoint: list[dict] = []
    drop_log: list[dict] = []

    import random as _random
    _random.seed(cfg.get("seed", 42))

    for i, chunk in enumerate(chunks):
        doc_id = str(chunk.get("doc_id", chunk.get("source_doc_id", f"UNKNOWN_{i}")))
        regulator = str(chunk.get("regulator", "RBI"))
        family = str(chunk.get("family_hint", chunk.get("family", "SECTION")))
        chunk_text = str(chunk.get("chunk_text", chunk.get("text", "")))
        seed_chunk_id = str(chunk.get("seed_chunk_id", f"chunk_{i}"))

        # Decide if NOT_ANSWERABLE
        is_not_answerable = _random.random() < not_answerable_pct

        if is_not_answerable:
            user_prompt = _build_not_answerable_prompt(doc_id, regulator, family, chunk_text)
            answerable_label = "no"
        else:
            user_prompt = _build_prompt(doc_id, regulator, family, chunk_text)
            answerable_label = "yes"

        if dry_run:
            print(f"\n{'='*60}")
            print(f"PROMPT #{i+1} | doc={doc_id} | family={family} | answerable={answerable_label}")
            print(f"{'='*60}")
            print(f"SYSTEM: {SYSTEM_PROMPT}")
            print(f"USER:\n{user_prompt}")
            if i >= 4:
                break
            continue

        # Real run — no template fallback allowed
        llm_result = _try_llm(llm_client, user_prompt, SYSTEM_PROMPT, seed_chunk_id)
        llm_failed = llm_result is None
        quota_exceeded = False
        error_detail = f"chunk {seed_chunk_id}"
        if llm_failed or quota_exceeded:
            raise RuntimeError(
                f"LLM call failed or quota exceeded: {error_detail}. "
                "No template fallback allowed. Fix the API key/quota and retry."
            )

        parsed, drop_reason = _validate_response(
            llm_result["text"], chunk_text, family
        )

        if drop_reason:
            drop_log.append({"seed_chunk_id": seed_chunk_id, "reason": drop_reason})
            continue

        question = parsed.get("question", "")
        norm_q = _normalize(question.lower())
        if norm_q in seen_questions:
            drop_log.append({"seed_chunk_id": seed_chunk_id, "reason": "duplicate"})
            continue
        seen_questions.add(norm_q)

        row = _make_row(
            question=question,
            family=parsed.get("family", family),
            answerable=answerable_label,
            regulator=regulator,
            doc_id=doc_id,
            seed_chunk_id=seed_chunk_id,
            evidence_quote=parsed.get("evidence_quote", ""),
            gold_answer_short=parsed.get("gold_answer_short", ""),
        )
        rows.append(row)
        batch_since_checkpoint.append(row)

        if len(batch_since_checkpoint) >= 10:
            _write_checkpoint(batch_since_checkpoint)
            batch_since_checkpoint = []

    if batch_since_checkpoint:
        _write_checkpoint(batch_since_checkpoint)

    if drop_log:
        print(f"[draft_questions] Dropped {len(drop_log)} items: "
              + ", ".join(f"{d['reason']}({sum(1 for x in drop_log if x['reason']==d['reason'])})"
                          for d in {d['reason']: d for d in drop_log}.values()))

    return rows


def main():
    parser = argparse.ArgumentParser(description="Draft regulatory Q&A questions")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompts for 5 seed chunks, do not call API")
    parser.add_argument("--config", default=str(ROOT / "configs" / "experiment.yaml"),
                        help="Path to experiment.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        import yaml as _yaml
        cfg = _yaml.safe_load(f)

    if args.dry_run:
        print("[draft_questions] --dry-run mode: showing prompts only, no API calls")
        run_draft(cfg, llm_client=None, dry_run=True)
        return

    from common.llm_client import LLMClient
    client = LLMClient(role="question_drafting", cfg=cfg, check_prices=False)

    rows = run_draft(cfg, llm_client=client, dry_run=False)

    if not rows:
        print("[draft_questions] No questions generated.")
        return

    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)

    # Pilot review: up to 50 questions, stratified by family
    pilot = df.groupby("family", group_keys=False).apply(
        lambda g: g.sample(min(len(g), max(1, int(50 * len(g) / len(df)))), random_state=42)
    ).head(50)
    pilot.to_csv(data_dir / "questions_pilot_review.csv", index=False)

    df.to_csv(data_dir / "questions_draft.csv", index=False)
    print(f"[draft_questions] Wrote {len(df)} questions to data/questions_draft.csv")
    print(f"[draft_questions] Wrote {len(pilot)} pilot questions to data/questions_pilot_review.csv")


if __name__ == "__main__":
    main()
