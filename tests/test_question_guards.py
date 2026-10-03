"""
tests/test_question_guards.py
------------------------------
Tests for question validation guards (T0).
All fixtures are synthetic (no real data).
synthetic=true
"""
from __future__ import annotations

import csv
import io
import sys
import types
from pathlib import Path

import pytest

# Add scripts/ to path so we can import finalize_questions
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from finalize_questions import finalize_questions  # noqa: E402

REQUIRED_COLS = [
    "question_id", "question", "family", "answerable", "regulator",
    "split", "query_date", "source_doc_id", "seed_chunk_id",
    "evidence_quote", "gold_answer_short", "keep",
]


def _make_csv(rows: list[dict], fields: list[str] | None = None, tmp_path: Path = None) -> Path:
    """Write synthetic CSV to tmp_path, return path."""
    if fields is None:
        fields = REQUIRED_COLS
    p = tmp_path / "test_questions.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return p


def _good_row(qid="Q00001") -> dict:
    """Return a valid synthetic row."""
    return {
        "question_id": qid,
        "question": "What is the capital adequacy ratio requirement for commercial banks?",
        "family": "RATE_THRESHOLD",
        "answerable": "yes",
        "regulator": "RBI",
        "split": "train",
        "query_date": "2026-10-03",
        "source_doc_id": "RBI_doc_001",
        "seed_chunk_id": "RBI_doc_001_chunk_1",
        "evidence_quote": "Banks must maintain a CAR of 9%.",
        "gold_answer_short": "9%",
        "keep": "yes",
    }


# a) Missing required column
def test_finalize_rejects_missing_columns(tmp_path):
    """CSV missing 'evidence_quote' → ValueError mentioning 'evidence_quote'."""
    fields = [c for c in REQUIRED_COLS if c != "evidence_quote"]
    row = {k: v for k, v in _good_row().items() if k != "evidence_quote"}
    csv_path = _make_csv([row], fields=fields, tmp_path=tmp_path)
    with pytest.raises(ValueError, match="evidence_quote"):
        finalize_questions(csv_path, data_dir=tmp_path / "data_out")


# b) Template-stem question
def test_finalize_rejects_template_questions(tmp_path):
    """Row with template-stem question → ValueError."""
    row = _good_row()
    row["question"] = "What is the specific rate or threshold mentioned in this RBI circular regarding capital?"
    csv_path = _make_csv([row], tmp_path=tmp_path)
    with pytest.raises(ValueError, match="[Tt]emplate"):
        finalize_questions(csv_path, data_dir=tmp_path / "data_out")


# c) Missing query_date
def test_finalize_rejects_missing_query_date(tmp_path):
    """Row with empty query_date → ValueError."""
    row = _good_row()
    row["query_date"] = ""
    csv_path = _make_csv([row], tmp_path=tmp_path)
    with pytest.raises(ValueError, match="query_date"):
        finalize_questions(csv_path, data_dir=tmp_path / "data_out")


# d) Missing split
def test_finalize_rejects_missing_split(tmp_path):
    """Row with empty split → ValueError."""
    row = _good_row()
    row["split"] = ""
    csv_path = _make_csv([row], tmp_path=tmp_path)
    with pytest.raises(ValueError, match="split"):
        finalize_questions(csv_path, data_dir=tmp_path / "data_out")


# e) draft_questions raises RuntimeError when _try_llm returns None
def test_draft_questions_fails_on_llm_error(tmp_path, monkeypatch):
    """
    When _try_llm is monkeypatched to return None, run_draft must
    raise RuntimeError (NOT write a fallback template question).
    synthetic=true
    """
    import draft_questions as dq

    # Patch _try_llm to always return None (simulates LLM failure)
    monkeypatch.setattr(dq, "_try_llm", lambda *a, **kw: None)

    cfg = {
        "question_families": {"RATE_THRESHOLD": 0.0, "NOT_ANSWERABLE": 0.0},
        "seed": 42,
    }

    # Synthetic seed chunks CSV so run_draft doesn't need real files
    (tmp_path / "data").mkdir()
    seed_csv = tmp_path / "data" / "seed_chunks.csv"
    seed_csv.write_text(
        "chunk_id,doc_id,regulator,split,doc_title,issue_date,text_length,ascii_ratio,family_hint,chunk_text\n"
        "chunk_1,doc_1,RBI,train,Test Doc,2025-01-01,400,0.99,RATE_THRESHOLD," + ("word " * 80) + "\n",
        encoding="utf-8",
    )

    # Monkeypatch seed path
    monkeypatch.setattr(dq, "ROOT", tmp_path)

    # Fake llm_client that is non-None (so the LLM branch executes)
    class _FakeLLMClient:
        def chat(self, **kw):
            return {"text": '{"question": "Q?", "evidence_quote": "", "gold_answer_short": "", "family": "RATE_THRESHOLD"}'}

    out_path = tmp_path / "draft_out.csv"

    with pytest.raises(RuntimeError, match="No template fallback allowed"):
        dq.run_draft(cfg, llm_client=_FakeLLMClient(), dry_run=False)

    # Ensure no output was written to data/questions_draft.csv
    draft_csv = tmp_path / "data" / "questions_draft.csv"
    assert not draft_csv.exists(), "run_draft must NOT write output on LLM failure"
