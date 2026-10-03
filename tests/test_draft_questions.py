"""Tests for draft_questions.py and LLMClient.

All tests use fake clients and synthetic data only.
No data/ directory writes. No real API calls.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.errors import BudgetExceededError, LLMError, PriceMissingError


# ---------------------------------------------------------------------------
# Fake LLM Client
# ---------------------------------------------------------------------------

class FakeLLMClient:
    def __init__(self, responses=None, fail_after=None, cache_hit_on=None):
        self.calls = 0
        self.responses = responses or []
        self.fail_after = fail_after
        self.cache_hit_on = cache_hit_on  # set of call indices (1-based) to return cache_hit=True

    def chat(self, user, system="", **kwargs):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise LLMError("Quota exceeded")
        resp = (
            self.responses[self.calls - 1]
            if self.calls <= len(self.responses)
            else '{"question":"q","evidence_quote":"e","gold_answer_short":"a","family":"SECTION"}'
        )
        cache_hit = self.cache_hit_on is not None and self.calls in self.cache_hit_on
        return {
            "text": resp,
            "model": "test",
            "cache_hit": cache_hit,
            "input_tokens": 10,
            "output_tokens": 10,
            "cost_usd": 0.0 if cache_hit else 0.001,
        }


# ---------------------------------------------------------------------------
# Minimal experiment cfg fixture
# ---------------------------------------------------------------------------

def _base_cfg(*, price_input=1.0, price_output=1.0, budget_cap=100.0) -> dict:
    return {
        "seed": 42,
        "roles": {
            "question_drafting": {
                "provider": "anthropic",
                "model": "claude-test",
                "api_key_env": "ANTHROPIC_API_KEY",
                "max_tokens": 64,
                "temperature": 0.0,
            }
        },
        "llm_prices": {
            "claude-test": {
                "input": price_input,
                "output": price_output,
            }
        },
        "budget_caps": {"question_drafting": budget_cap},
        "question_families": {
            "RATE_THRESHOLD": 0.30,
            "SECTION": 0.60,
            "NOT_ANSWERABLE": 0.10,
        },
    }


# ---------------------------------------------------------------------------
# a. test_cache_hit_costs_zero
# ---------------------------------------------------------------------------

def test_cache_hit_costs_zero(tmp_path):
    """Cache-hit result must have cost_usd=0.0."""
    fake = FakeLLMClient(cache_hit_on={1})
    result = fake.chat("hello")
    assert result["cache_hit"] is True
    assert result["cost_usd"] == 0.0


# ---------------------------------------------------------------------------
# b. test_retry_on_429
# ---------------------------------------------------------------------------

def test_retry_on_429():
    """LLMClient retries on RateLimitError and succeeds after 3 calls total."""
    import anthropic as _anthropic

    call_count = 0

    def fake_create(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise _anthropic.RateLimitError(
                "rate limit",
                response=MagicMock(status_code=429, headers={}),
                body={"error": {"message": "rate limit"}},
            )
        resp = MagicMock()
        resp.content = [MagicMock(text="answer")]
        resp.model = "claude-test"
        resp.usage = MagicMock(input_tokens=5, output_tokens=5)
        return resp

    from common.llm_client import LLMClient
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-fake"}):
        client = LLMClient("question_drafting", _base_cfg(), check_prices=True)

    # Override the internal anthropic client's messages.create
    client._anthropic_client = MagicMock()
    client._anthropic_client.messages.create.side_effect = fake_create

    # Patch cache read to always miss (prevent stale cache from masking retries)
    with patch.object(client, "_read_cache", return_value=None), \
         patch.object(client, "_write_cache"), \
         patch("time.sleep"):
        result = client.chat(f"retry-test-{id(client)}", system="sys")

    assert call_count == 3
    assert result["text"] == "answer"


# ---------------------------------------------------------------------------
# c. test_budget_cap_stops_run
# ---------------------------------------------------------------------------

def test_budget_cap_stops_run(tmp_path):
    """budget_caps.question_drafting=0.001; first call should trigger BudgetExceededError."""
    cfg = _base_cfg(price_input=1000.0, price_output=1000.0, budget_cap=0.001)

    from common.llm_client import LLMClient, _role_costs
    _role_costs.clear()

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-fake"}):
        client = LLMClient("question_drafting", cfg, check_prices=True)

    resp_mock = MagicMock()
    resp_mock.content = [MagicMock(text="result")]
    resp_mock.model = "claude-test"
    resp_mock.usage = MagicMock(input_tokens=10000, output_tokens=10000)
    client._anthropic_client = MagicMock()
    client._anthropic_client.messages.create.return_value = resp_mock

    with pytest.raises(BudgetExceededError):
        client.chat("some prompt")


# ---------------------------------------------------------------------------
# d. test_missing_price_raises
# ---------------------------------------------------------------------------

def test_missing_price_raises():
    """price=null in config -> PriceMissingError on LLMClient init (not dry-run)."""
    cfg = _base_cfg(price_input=None, price_output=None)
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-fake"}):
        with pytest.raises(PriceMissingError):
            from common.llm_client import LLMClient
            LLMClient("question_drafting", cfg, check_prices=True)


# ---------------------------------------------------------------------------
# e. test_missing_api_key_raises
# ---------------------------------------------------------------------------

def test_missing_api_key_raises():
    """No env var set -> RuntimeError on init."""
    cfg = _base_cfg()
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    env.pop("ANTHROPIC_API_KEY", None)
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(RuntimeError, match="API key env var"):
            from common.llm_client import LLMClient
            LLMClient("question_drafting", cfg, check_prices=True)


# ---------------------------------------------------------------------------
# f. test_evidence_not_substring_dropped
# ---------------------------------------------------------------------------

def test_evidence_not_substring_dropped():
    """evidence_quote not in chunk -> question is dropped, no crash."""
    from scripts.draft_questions import _validate_response

    resp_json = json.dumps({
        "question": "What is the penalty under RBI circular?",
        "evidence_quote": "THIS QUOTE DOES NOT EXIST IN CHUNK",
        "gold_answer_short": "10 lakh",
        "family": "RATE_THRESHOLD",
    })
    chunk_text = "The Reserve Bank of India mandates a 5% ratio under RBI Circular 2024."
    parsed, reason = _validate_response(resp_json, chunk_text, "RATE_THRESHOLD")
    assert parsed is None
    assert reason == "evidence_not_substring"


# ---------------------------------------------------------------------------
# g. test_failing_client_stops_script
# ---------------------------------------------------------------------------

def test_failing_client_stops_script(tmp_path, monkeypatch):
    """LLM raises on first call -> LLMError propagated, no template fallback."""
    failing_client = FakeLLMClient(fail_after=0)

    # Patch seed_chunks.csv path to use tmp synthetic data (no file = use synthetic in dry_run)
    # For non-dry-run, we create a minimal CSV
    seed_csv = tmp_path / "seed_chunks.csv"
    import pandas as pd
    pd.DataFrame([{
        "seed_chunk_id": "C001",
        "doc_id": "RBI_001",
        "regulator": "RBI",
        "family_hint": "SECTION",
        "chunk_text": "RBI mandates all banks maintain capital ratio of 10%.",
    }]).to_csv(seed_csv, index=False)

    from scripts import draft_questions as dq_module
    monkeypatch.setattr(dq_module, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    import shutil
    shutil.copy(seed_csv, tmp_path / "data" / "seed_chunks.csv")
    (tmp_path / "results").mkdir()

    cfg = _base_cfg()
    cfg["question_families"] = {"SECTION": 1.0, "NOT_ANSWERABLE": 0.0}

    with pytest.raises(LLMError):
        dq_module.run_draft(cfg, llm_client=failing_client, dry_run=False)


# ---------------------------------------------------------------------------
# h. test_dry_run_prints_prompts
# ---------------------------------------------------------------------------

def test_dry_run_prints_prompts(tmp_path, monkeypatch, capsys):
    """--dry-run flag -> prints 5 prompts, makes 0 API calls."""
    from scripts import draft_questions as dq_module
    monkeypatch.setattr(dq_module, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    # No seed_chunks.csv -> synthetic chunks used

    cfg = _base_cfg()
    cfg["question_families"] = {"RATE_THRESHOLD": 0.9, "NOT_ANSWERABLE": 0.1}

    fake = FakeLLMClient()
    dq_module.run_draft(cfg, llm_client=fake, dry_run=True)

    captured = capsys.readouterr()
    assert fake.calls == 0, "dry-run must not call the LLM"
    assert "PROMPT #1" in captured.out
    assert "PROMPT #5" in captured.out
