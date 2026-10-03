"""
tests/test_m6_aggregate.py
---------------------------
Unit tests for the M6 aggregator:
  - aggregate/dataset.py  — row_to_features(), build_feature_table()
  - aggregate/model.py    — score() with fallback and with synthetic model
  - aggregate/calibrate_probs.py — apply_calibration()

All tests are self-contained (no real data files needed).
"""
from __future__ import annotations

import math
import pickle
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import pandas as pd
import numpy as np

from common.schemas import (
    Atom, Chunk, GeneratedAnswer, RetrievalResult, ScoredAtom, VerifiedAtom,
)
from aggregate.dataset import (
    row_to_features, build_feature_table, FEATURE_COLS, SIGNAL_COLS,
    ATOM_TYPES, REGULATORS,
)
from aggregate.model import score, _fallback_risk
import aggregate.model as _agg_model


# ---------------------------------------------------------------------------
# Reset the model singleton between tests so injected models don't bleed over
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_model_singleton():
    """Reset aggregate.model singletons before each test."""
    _agg_model._model = None
    _agg_model._calibrator = None
    yield
    _agg_model._model = None
    _agg_model._calibrator = None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_atom_row(overrides: dict | None = None) -> dict:
    """Return a minimal labeled-atom dict for row_to_features()."""
    row = {
        "atom_id":        "atom-001",
        "type":           "RATE",
        "text":           "5% per annum",
        "claim":          "The rate is 5% per annum.",
        "cited_chunk":    "chunk-001",
        "regulator":      "SEBI",
        "v1_status":      "MATCH",
        "v2_entail_prob": 0.9,
        "rerank_score":   0.8,
        "rrf_rank":       0,
        "n_chunks":       5,
        "token_logprobs": [math.log(0.8)] * 10,
        "no_context_answer": "The rate is around 5%.",
        "atom_span":      [0, 12],
        "answer_text":    "The rate is 5% per annum.",
        "query":          "What is the rate?",
        "label":          0,
    }
    if overrides:
        row.update(overrides)
    return row


def _make_verified(
    atom_type: str = "RATE",
    v1_status: str = "MATCH",
    v2_entail_prob=0.9,
) -> VerifiedAtom:
    return VerifiedAtom(
        atom=Atom(
            atom_id="atom-001",
            type=atom_type,
            text="5% per annum",
            claim="The rate is 5% per annum.",
            cited_chunk="chunk-001",
            span=(0, 12),
        ),
        v1_status=v1_status,
        v2_entail_prob=v2_entail_prob,
    )


def _make_answer(logprobs=None) -> GeneratedAnswer:
    return GeneratedAnswer(
        query="What is the rate?",
        answer_text="The rate is 5% per annum.",
        citations=["chunk-001"],
        token_logprobs=logprobs,
        metadata={"no_context_answer": "Around 5%."},
    )


def _make_rr(rerank_scores=None) -> RetrievalResult:
    chunk = Chunk(
        chunk_id="chunk-001",
        text="The rate is 5% per annum.",
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
# aggregate/dataset.py tests
# ===========================================================================

class TestRowToFeatures:
    def test_basic_row_produces_feature_dict(self):
        row = _make_atom_row()
        feats = row_to_features(row)
        # Must have all FEATURE_COLS
        for col in FEATURE_COLS:
            assert col in feats, f"Missing column: {col}"

    def test_match_s_ver_is_zero(self):
        row = _make_atom_row({"v1_status": "MATCH"})
        feats = row_to_features(row)
        assert feats["s_ver"] == 0.0
        assert feats["v1_match"] == 1

    def test_mismatch_s_ver_is_one(self):
        row = _make_atom_row({"v1_status": "MISMATCH"})
        feats = row_to_features(row)
        assert feats["s_ver"] == 1.0
        assert feats["v1_mismatch"] == 1

    def test_s_nli_computed_from_entail_prob(self):
        row = _make_atom_row({"v2_entail_prob": 0.7})
        feats = row_to_features(row)
        assert abs(feats["s_nli"] - 0.3) < 1e-4
        assert feats["nli_skipped"] == 0

    def test_nli_skipped_when_none(self):
        row = _make_atom_row({"v2_entail_prob": None})
        feats = row_to_features(row)
        assert feats["nli_skipped"] == 1
        assert math.isnan(feats["s_nli"])

    def test_s_ret_from_rerank_score(self):
        row = _make_atom_row({"rerank_score": 0.75})
        feats = row_to_features(row)
        assert feats["s_ret"] == 0.75

    def test_s_ret_from_rrf_rank_when_no_rerank(self):
        row = _make_atom_row({"rerank_score": None, "rrf_rank": 2, "n_chunks": 5})
        feats = row_to_features(row)
        # 1 - 2/5 = 0.6
        assert abs(feats["s_ret"] - 0.6) < 1e-4

    def test_s_ret_neutral_when_both_absent(self):
        row = _make_atom_row({"rerank_score": None, "rrf_rank": None})
        feats = row_to_features(row)
        assert math.isnan(feats["s_ret"])
        assert feats["s_ret_skipped"] == 1

    def test_atom_type_one_hot_rate(self):
        row = _make_atom_row({"type": "RATE"})
        feats = row_to_features(row)
        assert feats["type_RATE"] == 1
        for t in ATOM_TYPES:
            if t != "RATE":
                assert feats[f"type_{t}"] == 0

    def test_atom_type_one_hot_entity(self):
        row = _make_atom_row({"type": "ENTITY"})
        feats = row_to_features(row)
        assert feats["type_ENTITY"] == 1
        assert feats["type_RATE"] == 0

    def test_regulator_one_hot_sebi(self):
        row = _make_atom_row({"regulator": "SEBI"})
        feats = row_to_features(row)
        assert feats["reg_SEBI"] == 1
        for r in REGULATORS:
            if r != "SEBI":
                assert feats[f"reg_{r}"] == 0

    def test_unknown_regulator_maps_to_other(self):
        row = _make_atom_row({"regulator": "UNKNOWN"})
        feats = row_to_features(row)
        assert feats["reg_OTHER"] == 1
        assert feats["reg_SEBI"] == 0

    def test_label_preserved(self):
        row = _make_atom_row({"label": 1})
        feats = row_to_features(row)
        assert feats["label"] == 1

    def test_missing_optional_fields_handled(self):
        row = {
            "atom_id": "a-min",
            "type": "DATE",
            "regulator": "RBI",
            "v1_status": "NOT_FOUND",
            "answer_text": "The date is 12 March 2024.",
            "label": 0,
        }
        feats = row_to_features(row)
        # Should not raise and should have all required columns
        assert "s_ver" in feats
        assert "s_nli" in feats


class TestBuildFeatureTable:
    def test_builds_table_from_jsonl(self, tmp_path):
        import json
        rows = [_make_atom_row({"atom_id": f"atom-{i}", "label": i % 2}) for i in range(5)]
        jsonl_path = tmp_path / "atoms.jsonl"
        with open(jsonl_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        df = build_feature_table(str(tmp_path))
        assert len(df) == 5
        assert "label" in df.columns
        for col in FEATURE_COLS:
            assert col in df.columns

    def test_saves_parquet(self, tmp_path):
        import json
        row = _make_atom_row()
        jsonl_path = tmp_path / "data.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(json.dumps(row) + "\n")

        out_path = str(tmp_path / "out.parquet")
        df = build_feature_table(str(tmp_path), out_path=out_path)
        assert Path(out_path).exists()
        df2 = pd.read_parquet(out_path)
        assert len(df2) == 1

    def test_empty_dir_returns_empty_df(self, tmp_path):
        df = build_feature_table(str(tmp_path))
        assert len(df) == 0


# ===========================================================================
# aggregate/model.py tests
# ===========================================================================

class TestFallbackRisk:
    def test_all_zero_signals_low_risk(self):
        feats = {
            "s_ver": 0.0, "s_nli": 0.0, "s_ent": 0.0,
            "s_ret": 1.0,  # inverted: high s_ret → low risk
            "s_div": 1.0,  # inverted: high s_div → low risk
        }
        risk = _fallback_risk(feats)
        assert risk < 0.2

    def test_all_max_signals_high_risk(self):
        feats = {
            "s_ver": 1.0, "s_nli": 1.0, "s_ent": 1.0,
            "s_ret": 0.0,   # inverted: low s_ret → high risk
            "s_div": 0.0,   # inverted: low s_div → high risk
        }
        risk = _fallback_risk(feats)
        assert risk > 0.8

    def test_mid_signals_mid_risk(self):
        feats = {
            "s_ver": 0.5, "s_nli": 0.5, "s_ent": 0.5,
            "s_ret": 0.5, "s_div": 0.5,
        }
        risk = _fallback_risk(feats)
        assert abs(risk - 0.5) < 0.1

    def test_output_in_range(self):
        feats = {"s_ver": 0.3, "s_nli": 0.4, "s_ent": 0.2, "s_ret": 0.7}
        risk = _fallback_risk(feats)
        assert 0.0 <= risk <= 1.0


class TestScoreFunction:
    def test_returns_scored_atoms_same_length(self):
        vas = [
            _make_verified("RATE", "MATCH", 0.95),
            _make_verified("ENTITY", "NA", None),
            _make_verified("DATE", "MISMATCH", 0.05),
        ]
        ans = _make_answer(logprobs=[math.log(0.8)] * 20)
        rr = _make_rr(rerank_scores={"chunk-001": 0.85})

        result = score(vas, ans, rr)
        assert len(result) == len(vas)

    def test_returns_scored_atom_type(self):
        va = [_make_verified()]
        ans = _make_answer()
        rr = _make_rr()
        result = score(va, ans, rr)
        assert all(isinstance(s, ScoredAtom) for s in result)

    def test_risk_in_range(self):
        vas = [_make_verified(v1_status=s) for s in ["MATCH", "MISMATCH", "NOT_FOUND", "NA"]]
        ans = _make_answer()
        rr = _make_rr()
        result = score(vas, ans, rr)
        for s in result:
            assert 0.0 <= s.risk <= 1.0

    def test_mismatch_higher_risk_than_match(self):
        va_match = _make_verified(v1_status="MATCH", v2_entail_prob=0.95)
        va_mismatch = _make_verified(v1_status="MISMATCH", v2_entail_prob=0.05)
        ans = _make_answer(logprobs=[math.log(0.8)] * 20)
        rr = _make_rr(rerank_scores={"chunk-001": 0.9})

        result = score([va_match, va_mismatch], ans, rr)
        risk_match    = result[0].risk
        risk_mismatch = result[1].risk
        assert risk_mismatch > risk_match, (
            f"Expected mismatch risk ({risk_mismatch}) > match risk ({risk_match})"
        )

    def test_metadata_contains_signals(self):
        va = [_make_verified()]
        ans = _make_answer()
        rr = _make_rr()
        result = score(va, ans, rr)
        assert "signals" in result[0].metadata
        signals = result[0].metadata["signals"]
        assert "s_ver" in signals
        assert "s_nli" in signals

    def test_fallback_mode_when_model_missing(self, tmp_path):
        """score() uses fallback if model_path does not exist."""
        va = [_make_verified()]
        ans = _make_answer()
        rr = _make_rr()
        result = score(
            va, ans, rr,
            model_path=str(tmp_path / "nonexistent_model.pkl"),
        )
        assert len(result) == 1
        assert result[0].metadata["model_mode"] == "fallback_weighted_average"

    def test_with_synthetic_sklearn_model(self, tmp_path):
        """Score with a trivially-trained LogisticRegression model."""
        from sklearn.linear_model import LogisticRegression  # type: ignore

        # Train a minimal model on synthetic data
        n_feats = len(FEATURE_COLS)
        X = np.random.rand(100, n_feats)
        y = (X[:, 0] > 0.5).astype(int)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)

        model_path = tmp_path / "aggregator_v1.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(clf, f)

        va = [_make_verified()]
        ans = _make_answer(logprobs=[math.log(0.7)] * 20)
        rr = _make_rr(rerank_scores={"chunk-001": 0.8})

        result = score(
            va, ans, rr,
            model_path=str(model_path),
        )
        assert len(result) == 1
        assert 0.0 <= result[0].risk <= 1.0
        assert result[0].metadata["model_mode"] in {"lightgbm", "logreg"}


# ===========================================================================
# Feature column consistency check
# ===========================================================================

class TestFeatureCols:
    def test_feature_cols_no_duplicates(self):
        assert len(FEATURE_COLS) == len(set(FEATURE_COLS))

    def test_signal_cols_subset_of_feature_cols(self):
        for sig in SIGNAL_COLS:
            assert sig in FEATURE_COLS, f"Signal {sig} not in FEATURE_COLS"

    def test_type_cols_in_feature_cols(self):
        for t in ATOM_TYPES:
            assert f"type_{t}" in FEATURE_COLS

    def test_reg_cols_in_feature_cols(self):
        for r in REGULATORS:
            assert f"reg_{r}" in FEATURE_COLS
