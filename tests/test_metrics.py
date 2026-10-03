"""Tests for T4 retrieval and extraction metrics.

All data is synthetic. No real data files used.
"""
import json
import math
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

SYNTHETIC_GOLD_CHUNKS = [
    {"question_id": "Q1", "gold_chunk_ids": ["c1", "c2"]},
    {"question_id": "Q2", "gold_chunk_ids": ["c3"]},
]
# _synthetic marker (not written to file, just for clarity)

SYNTHETIC_RETRIEVED = [
    {"question_id": "Q1", "retrieved_chunk_ids": ["c1", "x", "x", "x", "x", "x"]},  # c1 at rank 1
    {"question_id": "Q2", "retrieved_chunk_ids": ["x", "c3", "x", "x", "x", "x"]},  # c3 at rank 2
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _load_metrics():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "retrieval_metrics",
        Path(__file__).parent.parent / "eval" / "retrieval_metrics.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_latency():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "latency_logger",
        Path(__file__).parent.parent / "eval" / "latency_logger.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_perfect_recall_at_6(tmp_path):
    """Both Q1 and Q2 have their gold chunks in top-6 -> Recall@6=1.0."""
    metrics = _load_metrics()
    gold_path = tmp_path / "gold.jsonl"
    ret_path = tmp_path / "retrieved.jsonl"
    _write_jsonl(SYNTHETIC_GOLD_CHUNKS, gold_path)
    _write_jsonl(SYNTHETIC_RETRIEVED, ret_path)

    result = metrics.compute_retrieval_metrics(str(gold_path), str(ret_path))
    assert result["recall_at_6"] == pytest.approx(1.0)


def test_mrr_correct(tmp_path):
    """Q1 rank 1 -> RR=1.0; Q2 rank 2 -> RR=0.5; MRR=0.75."""
    metrics = _load_metrics()
    gold_path = tmp_path / "gold.jsonl"
    ret_path = tmp_path / "retrieved.jsonl"
    _write_jsonl(SYNTHETIC_GOLD_CHUNKS, gold_path)
    _write_jsonl(SYNTHETIC_RETRIEVED, ret_path)

    result = metrics.compute_retrieval_metrics(str(gold_path), str(ret_path))
    assert result["mrr"] == pytest.approx(0.75)


def test_missing_gold_returns_not_available(tmp_path):
    """Missing gold file -> status='not_available'."""
    metrics = _load_metrics()
    ret_path = tmp_path / "retrieved.jsonl"
    _write_jsonl(SYNTHETIC_RETRIEVED, ret_path)

    result = metrics.compute_retrieval_metrics(
        str(tmp_path / "nonexistent_gold.jsonl"),
        str(ret_path)
    )
    assert result.get("status") == "not_available"
    assert "gold" in result.get("reason", "").lower()


def test_missing_retrieved_returns_not_available(tmp_path):
    """Missing retrieved file -> status='not_available'."""
    metrics = _load_metrics()
    gold_path = tmp_path / "gold.jsonl"
    _write_jsonl(SYNTHETIC_GOLD_CHUNKS, gold_path)

    result = metrics.compute_retrieval_metrics(
        str(gold_path),
        str(tmp_path / "nonexistent_retrieved.jsonl")
    )
    assert result.get("status") == "not_available"


def test_latency_p50_p95(tmp_path):
    """Log 10 timings; p50 <= p95."""
    latency_mod = _load_latency()
    logger = latency_mod.LatencyLogger()

    # Record 10 timings: small sleeps proportional to index
    import time as _time
    for i in range(10):
        with logger.stage("retrieve"):
            _time.sleep(0.001 * (i + 1))  # 1ms to 10ms

    summary = logger.summary()
    assert "retrieve" in summary
    info = summary["retrieve"]
    assert info["count"] == 10
    assert info["p50_ms"] <= info["p95_ms"], "p50 should be <= p95"
    assert info["p95_ms"] > 0
