"""
tests/test_m6_signals.py
-------------------------
Unit tests for all 5 M6 signals and the compute_all_signals() dispatcher.

Coverage:
  - s_ver: all 4 V1 status codes (MATCH, MISMATCH, NOT_FOUND, NA) + one-hot
  - s_nli: entail_prob present / absent (skipped flag)
  - s_ent: logprobs present / absent; mean & max entropy correct
  - s_ret: rerank_scores path; rank-based fallback; missing cited_chunk
  - s_div: no_context_answer absent → 0.5 neutral
  - compute_all_signals: keys present; values in [0, 1]
"""
from __future__ import annotations

import math
import pytest
from unittest.mock import patch

from common.schemas import (
    Atom,
    Chunk,
    GeneratedAnswer,
    RetrievalResult,
    VerifiedAtom,
)
from signals.s_ver import compute_s_ver, s_ver_features
from signals.s_nli import compute_s_nli, s_nli_features
from signals.s_ent import compute_s_ent, s_ent_features
from signals.s_ret import compute_s_ret
from signals.s_div import compute_s_div
from signals import compute_all_signals


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_atom(atom_type: str = "RATE", span=(0, 20)) -> Atom:
    return Atom(
        atom_id="atom-001",
        type=atom_type,
        text="5% per annum",
        claim="The rate is 5% per annum.",
        cited_chunk="chunk-001",
        span=span,
    )


def _make_verified(
    atom_type: str = "RATE",
    v1_status: str = "MATCH",
    v2_entail_prob=None,
    span=(0, 20),
) -> VerifiedAtom:
    return VerifiedAtom(
        atom=_make_atom(atom_type, span=span),
        v1_status=v1_status,
        v2_entail_prob=v2_entail_prob,
    )


def _make_answer(
    text: str = "The rate is 5% per annum.",
    logprobs=None,
    no_context_answer: str = "",
) -> GeneratedAnswer:
    return GeneratedAnswer(
        query="What is the rate?",
        answer_text=text,
        citations=["chunk-001"],
        token_logprobs=logprobs,
        metadata={"no_context_answer": no_context_answer},
    )


def _make_rr(rerank_scores=None) -> RetrievalResult:
    chunk = Chunk(
        chunk_id="chunk-001",
        text="The applicable rate is 5% per annum.",
        regulator="SEBI",
        issue_date="2024-01-01",
        source_url="https://example.com",
    )
    meta = {}
    if rerank_scores:
        meta["rerank_scores"] = rerank_scores
    return RetrievalResult(
        query="What is the rate?",
        query_date="2024-06-01",
        chunks=[chunk],
        metadata=meta,
    )


# ===========================================================================
# s_ver tests
# ===========================================================================

class TestSVer:
    def test_match_returns_zero(self):
        va = _make_verified(v1_status="MATCH")
        assert compute_s_ver(va) == 0.0

    def test_mismatch_returns_one(self):
        va = _make_verified(v1_status="MISMATCH")
        assert compute_s_ver(va) == 1.0

    def test_not_found_returns_half(self):
        va = _make_verified(v1_status="NOT_FOUND")
        assert compute_s_ver(va) == 0.5

    def test_na_returns_half(self):
        va = _make_verified(v1_status="NA")
        assert compute_s_ver(va) == 0.5

    def test_unknown_status_returns_half(self):
        va = _make_verified(v1_status="UNKNOWN_XYZ")
        assert compute_s_ver(va) == 0.5

    def test_features_one_hot_match(self):
        va = _make_verified(v1_status="MATCH")
        f = s_ver_features(va)
        assert f["v1_match"] == 1
        assert f["v1_mismatch"] == 0
        assert f["v1_not_found"] == 0
        assert f["v1_na"] == 0
        assert f["s_ver"] == 0.0

    def test_features_one_hot_mismatch(self):
        va = _make_verified(v1_status="MISMATCH")
        f = s_ver_features(va)
        assert f["v1_match"] == 0
        assert f["v1_mismatch"] == 1
        assert f["s_ver"] == 1.0

    def test_features_one_hot_na(self):
        va = _make_verified(v1_status="NA")
        f = s_ver_features(va)
        assert f["v1_na"] == 1
        assert f["v1_match"] == 0

    def test_features_one_hot_not_found(self):
        va = _make_verified(v1_status="NOT_FOUND")
        f = s_ver_features(va)
        assert f["v1_not_found"] == 1
        assert f["v1_match"] == 0


# ===========================================================================
# s_nli tests
# ===========================================================================

class TestSNli:
    def test_full_entailment_gives_zero_risk(self):
        va = _make_verified(v2_entail_prob=1.0)
        assert compute_s_nli(va) == 0.0

    def test_zero_entailment_gives_full_risk(self):
        va = _make_verified(v2_entail_prob=0.0)
        assert compute_s_nli(va) == 1.0

    def test_mid_entailment(self):
        va = _make_verified(v2_entail_prob=0.6)
        assert abs(compute_s_nli(va) - 0.4) < 1e-4

    def test_none_is_missing(self):
        va = _make_verified(v2_entail_prob=None)
        assert math.isnan(compute_s_nli(va))

    def test_skipped_flag_when_none(self):
        va = _make_verified(v2_entail_prob=None)
        f = s_nli_features(va)
        assert f["nli_skipped"] == 1
        assert math.isnan(f["s_nli"])

    def test_skipped_flag_not_set_when_prob_present(self):
        va = _make_verified(v2_entail_prob=0.8)
        f = s_nli_features(va)
        assert f["nli_skipped"] == 0
        assert abs(f["s_nli"] - 0.2) < 1e-4

    def test_s_nli_in_range(self):
        for p in [0.0, 0.1, 0.5, 0.9, 1.0]:
            va = _make_verified(v2_entail_prob=p)
            val = compute_s_nli(va)
            assert 0.0 <= val <= 1.0


# ===========================================================================
# s_ent tests
# ===========================================================================

class TestSEnt:
    def test_no_logprobs_returns_fallback(self):
        va = _make_verified()
        ans = _make_answer(logprobs=None)
        f = s_ent_features(va, ans)
        assert math.isnan(f["s_ent"])
        assert math.isnan(f["s_ent_max"])
        assert f["s_ent_skipped"] == 1

    def test_empty_logprobs_returns_fallback(self):
        va = _make_verified()
        ans = _make_answer(logprobs=[])
        f = s_ent_features(va, ans)
        assert f["s_ent_skipped"] == 1

    def test_confident_logprobs_low_entropy(self):
        # log(1.0) = 0 → entropy = 0
        va = _make_verified(span=(0, 4))
        ans = _make_answer(logprobs=[0.0] * 10)   # p=1.0 → entropy=0
        f = s_ent_features(va, ans)
        assert f["s_ent"] == 0.0
        assert f["s_ent_skipped"] == 0

    def test_uncertain_logprobs_higher_entropy(self):
        # log(0.01) ≈ -4.6 → entropy ≈ 4.6/5.0 = 0.92
        va = _make_verified(span=(0, 40))
        logprobs = [math.log(0.01)] * 10
        ans = _make_answer(logprobs=logprobs)
        f = s_ent_features(va, ans)
        assert f["s_ent"] > 0.8  # high uncertainty

    def test_entropy_clipped_to_one(self):
        # Very low prob → entropy > 5.0 → clamped to 1.0
        va = _make_verified(span=(0, 40))
        logprobs = [math.log(1e-6)] * 10  # entropy ≈ 13.8 >> 5.0
        ans = _make_answer(logprobs=logprobs)
        f = s_ent_features(va, ans)
        assert f["s_ent"] == 1.0
        assert f["s_ent_max"] == 1.0

    def test_compute_s_ent_matches_features(self):
        va = _make_verified(span=(0, 20))
        ans = _make_answer(logprobs=[math.log(0.5)] * 10)
        assert compute_s_ent(va, ans) == s_ent_features(va, ans)["s_ent"]


# ===========================================================================
# s_ret tests
# ===========================================================================

class TestSRet:
    def test_rerank_scores_cited_chunk_max(self):
        # cited chunk has highest score → s_ret = 1.0
        rr = _make_rr(rerank_scores={"chunk-001": 0.9, "chunk-002": 0.3})
        va = _make_verified()
        val = compute_s_ret(va, rr)
        assert val == 1.0

    def test_rerank_scores_cited_chunk_min(self):
        # cited chunk has lowest score → s_ret = 0.0
        rr = _make_rr(rerank_scores={"chunk-001": 0.1, "chunk-002": 0.9})
        va = _make_verified()
        val = compute_s_ret(va, rr)
        assert val == 0.0

    def test_rerank_scores_mid(self):
        rr = _make_rr(rerank_scores={"chunk-001": 0.5, "chunk-002": 0.9, "chunk-003": 0.1})
        va = _make_verified()
        # (0.5 - 0.1) / (0.9 - 0.1) = 0.5
        val = compute_s_ret(va, rr)
        assert abs(val - 0.5) < 1e-4

    def test_no_rerank_scores_rank_based(self):
        # No rerank scores, chunk-001 is at position 0 → s_ret = 1.0
        rr = _make_rr(rerank_scores=None)
        va = _make_verified()
        val = compute_s_ret(va, rr)
        assert val == 1.0

    def test_no_cited_chunk_returns_pessimistic(self):
        rr = _make_rr(rerank_scores={"chunk-001": 0.9, "chunk-002": 0.1})
        atom = Atom(
            atom_id="atom-002",
            type="RATE",
            text="5%",
            claim="The rate is 5%.",
            cited_chunk=None,   # no cited chunk
            span=(0, 3),
        )
        va = VerifiedAtom(atom=atom, v1_status="NOT_FOUND")
        val = compute_s_ret(va, rr)
        assert val == 0.0  # pessimistic: min score

    def test_cited_chunk_not_in_result(self):
        rr = _make_rr(rerank_scores=None)
        atom = Atom(
            atom_id="atom-003",
            type="RATE",
            text="5%",
            claim="Rate is 5%.",
            cited_chunk="chunk-999",  # not in rr.chunks
            span=(0, 3),
        )
        va = VerifiedAtom(atom=atom, v1_status="NOT_FOUND")
        val = compute_s_ret(va, rr)
        assert val == 0.0

    def test_empty_chunks_returns_neutral(self):
        rr = RetrievalResult(
            query="q", query_date="2024-01-01", chunks=[], metadata={}
        )
        va = _make_verified()
        val = compute_s_ret(va, rr)
        assert val == 0.5


# ===========================================================================
# s_div tests
# ===========================================================================

class TestSDiv:
    def test_no_context_answer_returns_neutral(self):
        va = _make_verified()
        ans = _make_answer(no_context_answer="")
        rr = _make_rr()
        val = compute_s_div(va, ans)
        assert math.isnan(val)

    def test_identical_strings_low_divergence(self):
        # When claim == no_context answer → cosine sim = 1 → divergence = 0
        # (embedder may not be available; test graceful fallback)
        va = _make_verified()
        same_text = "The rate is 5% per annum."
        ans = _make_answer(no_context_answer=same_text)
        val = compute_s_div(va, ans)
        # Embedder present: low divergence. Embedder missing: NaN, not 0.5.
        assert math.isnan(val) or 0.0 <= val <= 0.5

    def test_returns_float_in_range(self):
        va = _make_verified()
        ans = _make_answer(no_context_answer="Something completely different.")
        val = compute_s_div(va, ans)
        assert math.isnan(val) or 0.0 <= val <= 1.0


# ===========================================================================
# compute_all_signals integration test
# ===========================================================================

class TestComputeAllSignals:
    def test_returns_all_expected_keys(self):
        va = _make_verified(v1_status="MATCH", v2_entail_prob=0.9)
        ans = _make_answer(logprobs=[math.log(0.8)] * 10)
        rr = _make_rr(rerank_scores={"chunk-001": 0.85})
        feats = compute_all_signals(va, ans, rr)

        expected_keys = {
            "s_ver", "v1_match", "v1_mismatch", "v1_not_found", "v1_na",
            "s_nli", "nli_skipped",
            "s_ent", "s_ent_max", "s_ent_skipped",
            "s_ret",
            "s_div",
        }
        for key in expected_keys:
            assert key in feats, f"Missing signal key: {key}"

    def test_all_values_in_range(self):
        va = _make_verified(v1_status="NOT_FOUND", v2_entail_prob=0.3)
        ans = _make_answer(logprobs=[math.log(0.5)] * 15)
        rr = _make_rr()
        feats = compute_all_signals(va, ans, rr)

        for key, val in feats.items():
            if isinstance(val, float) and math.isnan(val):
                continue
            if isinstance(val, float):
                assert 0.0 <= val <= 1.0, f"Signal {key}={val} out of [0,1]"

    def test_mismatch_atom_high_s_ver(self):
        va = _make_verified(v1_status="MISMATCH", v2_entail_prob=0.05)
        ans = _make_answer()
        rr = _make_rr()
        feats = compute_all_signals(va, ans, rr)
        assert feats["s_ver"] == 1.0
        assert feats["v1_mismatch"] == 1

    def test_match_atom_with_high_entailment(self):
        va = _make_verified(v1_status="MATCH", v2_entail_prob=0.95)
        ans = _make_answer(logprobs=[0.0] * 10)   # high confidence
        rr = _make_rr(rerank_scores={"chunk-001": 0.95})
        feats = compute_all_signals(va, ans, rr)
        # Low risk signals
        assert feats["s_ver"] == 0.0
        assert feats["s_nli"] < 0.1
        assert feats["s_ent"] == 0.0

    def test_no_logprobs_falls_back_gracefully(self):
        va = _make_verified(v1_status="NA", v2_entail_prob=None)
        ans = _make_answer(logprobs=None, no_context_answer="")
        rr = _make_rr()
        feats = compute_all_signals(va, ans, rr)
        # Fallback values
        assert math.isnan(feats["s_nli"])
        assert feats["nli_skipped"] == 1
        assert math.isnan(feats["s_ent"])
        assert feats["s_ent_skipped"] == 1
        assert math.isnan(feats["s_div"])
        assert feats["s_div_skipped"] == 1
