"""
tests/test_select_seed_chunks.py
---------------------------------
Tests for select_seed_chunks.py filters and family-hint logic.
All fixtures are synthetic (no real file I/O).
synthetic=true
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from select_seed_chunks import (
    _count_skip_reason,
    _family_hint,
    _build_newest_doc_ids,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_inventory(doc_id="doc_1", title="RBI Circular", issue_date="2025-01-01", regulator="RBI"):
    return {doc_id: {"regulator": regulator, "doc_type": "circular", "title": title, "issue_date": issue_date}}


def _skip_reason(text, doc_id="doc_1", inventory=None, doc_assignment=None,
                  newest=None, selected=None, min_ascii=0.70):
    if inventory is None:
        inventory = _make_inventory(doc_id)
    if doc_assignment is None:
        doc_assignment = {doc_id: "train"}
    if newest is None:
        newest = {doc_id}
    if selected is None:
        selected = {}
    return _count_skip_reason(text, doc_id, inventory, doc_assignment, newest, selected, min_ascii)


# a) too short
def test_skip_too_short():
    text = "A" * 250  # < 300 chars
    assert _skip_reason(text) == "skip_too_short"


# b) low ASCII (>30% non-ASCII)
def test_skip_low_ascii():
    # 70% non-ASCII: use CJK characters
    ascii_part = "A" * 30
    non_ascii = "中" * 70  # each is 1 char, non-ASCII
    text = (ascii_part + non_ascii) * 5  # 500 chars, 30% ASCII < 0.70
    assert _skip_reason(text) == "skip_low_ascii"


# c) letterhead
def test_skip_letterhead():
    # First 200 chars contains RBI header, total text ≥300 chars but < 600 chars
    header = "Reserve Bank of India\nMumbai - 400 001\n" + "x" * 160  # 200 chars
    # Add more to reach ≥300 but keep < 600
    body = "Some regulatory content here. " * 8  # ~240 chars
    text = header + body
    assert len(text) >= 300, f"text too short: {len(text)}"
    assert len(text) < 600, f"text too long: {len(text)}"
    assert _skip_reason(text) == "skip_letterhead"


# d) table rows (>50% lines have ≥3 '|')
def test_skip_table_rows():
    # Build a chunk where most lines are table rows
    table_lines = ["col1 | col2 | col3 | col4"] * 20
    text_lines = ["Some normal text here for padding."] * 5
    text = "\n".join(table_lines + text_lines) * 3
    # Ensure long enough (>300 chars)
    assert len(text) >= 300
    assert _skip_reason(text) == "skip_table_rows"


# e) family_hint → RATE_THRESHOLD
def test_family_hint_rate():
    text = "Banks shall maintain a capital adequacy ratio of at least 9 per cent at all times."
    assert _family_hint(text) == "RATE_THRESHOLD"


# f) family_hint → SECTION
def test_family_hint_section():
    text = "Section 12 of the Banking Regulation Act governs the requirements."
    assert _family_hint(text) == "SECTION"


# g) oldest version skipped
def test_oldest_version_skipped():
    """Two docs with same normalized title; only the newer one should be 'newest'."""
    inventory = {
        "doc_old": {"regulator": "RBI", "doc_type": "circular",
                    "title": "Master Circular on KYC Norms", "issue_date": "2023-01-01"},
        "doc_new": {"regulator": "RBI", "doc_type": "circular",
                    "title": "Master Circular on KYC Norms", "issue_date": "2025-06-01"},
    }
    newest = _build_newest_doc_ids(inventory)
    assert "doc_new" in newest
    assert "doc_old" not in newest

    # Chunk from old doc → skip_old_version
    text = "X " * 200  # long enough, all ASCII
    doc_assignment = {"doc_old": "train", "doc_new": "train"}
    reason = _count_skip_reason(text, "doc_old", inventory, doc_assignment, newest, {}, 0.70)
    assert reason == "skip_old_version"

    # Chunk from new doc → accepted (None)
    reason_new = _count_skip_reason(text, "doc_new", inventory, doc_assignment, newest, {}, 0.70)
    assert reason_new is None


# h) over quota (at most 2 per doc)
def test_over_quota():
    """
    Simulate a doc with 5 passing chunks; only 2 should be accepted, rest get skip_over_quota.
    """
    inventory = _make_inventory()
    doc_assignment = {"doc_1": "train"}
    newest = {"doc_1"}
    text = "X " * 200  # passes all filters

    selected: dict[str, int] = {}
    accepted = 0
    skip_over = 0
    for _ in range(5):
        reason = _count_skip_reason(text, "doc_1", inventory, doc_assignment, newest, selected, 0.70)
        if reason is None:
            selected["doc_1"] = selected.get("doc_1", 0) + 1
            accepted += 1
        elif reason == "skip_over_quota":
            skip_over += 1
        else:
            pytest.fail(f"Unexpected skip reason: {reason}")

    assert accepted == 2
    assert skip_over == 3
