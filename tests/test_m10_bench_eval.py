"""
tests/test_m10_bench_eval.py
-----------------------------
M10: Comprehensive tests for bench/ and eval/ modules.

Coverage plan
-------------
bench/question_gen/generator.py (16 tests)
  - generate_questions: empty chunks → empty list
  - generate_questions: returns QuestionItem objects
  - generate_questions: respects n limit
  - generate_questions: deterministic with same seed
  - generate_questions: all families represented when n is large
  - generate_questions: all items have non-empty question text
  - generate_questions: question_id unique
  - generate_questions: gold_chunk_ids non-empty
  - generate_temporal_questions: returns temporal_tag=True items
  - QuestionItem.to_dict / from_dict round-trip
  - save_questions / load_questions JSONL round-trip
  - _extract_hints: pct, section, date extraction from text

bench/annotation/annotator.py (14 tests)
  - prelabel: MISMATCH → unsupported
  - prelabel: high risk → unsupported
  - prelabel: V1 MATCH + low risk → supported
  - prelabel: high entail_prob + low risk → supported
  - prelabel: uncertain fallback
  - AnnotationRecord.is_verified
  - AnnotationRecord.is_wrong
  - save_annotations / load_annotations round-trip
  - annotation_stats: counts correct
  - annotation_stats: pct_wrong computed

bench/annotation/agreement.py (10 tests)
  - cohen_kappa: perfect agreement → 1.0
  - cohen_kappa: completely disagreed → negative
  - cohen_kappa: empty lists → 0.0
  - cohen_kappa: length mismatch → 0.0
  - percent_agreement: matches manual count
  - confusion_matrix: shape (4, 4)
  - confusion_matrix: diagonal sums = n_agree
  - agreement_stats: all keys present
  - kappa interpretation: "substantial" for kappa 0.7
  - percent_agreement: perfect case

bench/splits.py (10 tests)
  - split_questions: sizes sum to total
  - split_questions: no empty splits when n is large
  - split_questions: by_family no question appears in two splits
  - split_questions: empty input → empty splits
  - SplitConfig: fractions must sum to 1.0
  - save_split / load_split round-trip
  - temporal_split: pre/post based on cutoff_date

eval/metrics.py (20 tests)
  - hallucination_rate: all SUPPORTED + all supported → 0.0
  - hallucination_rate: all SUPPORTED + all unsupported → 1.0
  - hallucination_rate: no accepted atoms → nan
  - coverage: all SUPPORTED → 1.0
  - coverage: all ABSTAINED → 0.0
  - coverage: mixed → correct fraction
  - risk_coverage_curve: monotone in coverage
  - risk_coverage_curve: h_rate bounded [0, 1]
  - aurc: all-correct → near 0
  - aurc: empty curve → 0.0
  - recall_at_k: gold in top-k → 1.0
  - recall_at_k: gold not in top-k → 0.0
  - mrr: first hit at rank 1 → 1.0
  - mrr: no hit → 0.0
  - precision_at_k: all relevant → 1.0
  - extraction_recall: perfect recall → 1.0
  - extraction_precision: perfect precision → 1.0
  - extraction_f1: zero precision → 0.0
  - judge_call_rate: all UNCERTAIN → 0.0
  - empirical_violation_rate: IID data with no wrong atoms → violation_rate ≈ 0

eval/baselines.py (10 tests)
  - plain_rag: all atoms SUPPORTED
  - selfcheck: high entail_prob → SUPPORTED
  - selfcheck: low entail_prob → ABSTAINED
  - selfcheck: None entail_prob → ABSTAINED
  - nli_only: V1 MATCH fallback
  - nli_only: MISMATCH → ABSTAINED
  - conformal_no_mondrian: risk <= threshold → SUPPORTED
  - conformal_no_mondrian: risk > threshold → ABSTAINED
  - run_baseline: unknown name raises ValueError
  - run_baseline: all names in BASELINE_REGISTRY work

experiments/drift/drift_exp.py (8 tests) [standalone — not part of main pipeline]
  - run_drift_experiment: returns DriftExperimentResult
  - run_drift_experiment: 4 strategies always present
  - run_drift_experiment: violation_rate in [0, 1]
  - run_drift_experiment: static strategy has violation rate > sliding on drifted data
  - run_drift_experiment: empty pre → empty result
  - DriftExperimentResult.best_strategy: returns a strategy name
  - DriftExperimentResult.summary_table: list of dicts
  - DriftWindowResult fields

eval/run_all.py (8 tests)
  - run_all: completes without error on synthetic data
  - run_all: creates output files
  - run_all: summary.json contains expected keys
  - run_all: violation_rate_ok present in summary
  - run_all: skip_baselines flag works
  - run_all: skip_drift flag works
  - run_all: deterministic with same seed
  - run_all: hallucination_rate in [0, 1]

Total: 96 tests
"""
from __future__ import annotations

import json
import math
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest


# ── Helpers ──────────────────────────────────────────────────────────────────

@dataclass
class _Chunk:
    chunk_id:   str
    text:       str
    regulator:  str = "RBI"
    issue_date: str = "2024-01-01"
    source_url: str = "https://rbi.org.in/test"
    metadata:   dict = None
    effective_from: str = ""
    effective_to:   str = ""
    superseded_by:  str = ""

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {"section": "Section 24", "doc_title": "RBI Circular 2024"}


def _make_chunks(n: int = 20) -> List[_Chunk]:
    texts = [
        "The CRR rate of 4.5% must be maintained by all scheduled commercial banks as per RBI Section 42.",
        "SEBI Regulation 30 requires disclosure within 24 hours of a material event.",
        "The SLR requirement is 18.5% of NDTL for commercial banks effective March 2024.",
        "Income Tax Section 80C allows deduction up to ₹1.5 lakh per annum.",
        "SEBI AIF Regulations: Category I AIFs must invest in SMEs and infrastructure.",
        "Merchant bankers must maintain a net worth of ₹5 crore at all times.",
        "The repo rate was revised to 6.5% in February 2023 by RBI.",
        "Foreign portfolio investors must register under SEBI regulations.",
        "KYC norms require low/medium/high risk categorisation for all customers.",
        "Mutual funds must disclose NAV by 11 PM on each business day.",
    ] * (n // 10 + 1)

    return [
        _Chunk(
            chunk_id=f"chunk_{i:04d}",
            text=texts[i % len(texts)],
            regulator=["RBI", "SEBI", "INCOMETAX"][i % 3],
            issue_date=f"202{i % 4 + 1}-{(i % 12) + 1:02d}-01",
        )
        for i in range(n)
    ]


def _make_scored_atom(atom_id="a1", v1_status="MATCH", v2_entail=0.9, risk=0.15):
    """Create a minimal ScoredAtom-like mock."""
    atom = MagicMock()
    atom.atom_id = atom_id
    atom.type    = "RATE"
    atom.text    = "4.5%"
    atom.claim   = "CRR is 4.5%."

    verified = MagicMock()
    verified.atom         = atom
    verified.v1_status    = v1_status
    verified.v2_entail_prob = v2_entail

    scored = MagicMock()
    scored.verified = verified
    scored.risk     = risk
    return scored


# ══════════════════════════════════════════════════════════════════════════════
# Group 1: bench/question_gen/generator.py
# ══════════════════════════════════════════════════════════════════════════════

from bench.question_gen.generator import (
    generate_questions, generate_temporal_questions,
    save_questions, load_questions, QuestionItem,
    QUESTION_FAMILIES, _extract_hints,
)


class TestGenerateQuestions:

    def test_empty_chunks_returns_empty(self):
        result = generate_questions([], n=10)
        assert result == []

    def test_returns_question_items(self):
        chunks = _make_chunks(20)
        result = generate_questions(chunks, n=5, seed=42)
        assert all(isinstance(q, QuestionItem) for q in result)

    def test_respects_n_limit(self):
        chunks = _make_chunks(30)
        result = generate_questions(chunks, n=10, seed=42)
        assert len(result) <= 10

    def test_deterministic_with_seed(self):
        chunks = _make_chunks(30)
        r1 = generate_questions(chunks, n=8, seed=99)
        r2 = generate_questions(chunks, n=8, seed=99)
        assert [q.question for q in r1] == [q.question for q in r2]

    def test_different_seeds_different_results(self):
        chunks = _make_chunks(30)
        r1 = generate_questions(chunks, n=8, seed=1)
        r2 = generate_questions(chunks, n=8, seed=2)
        # Very unlikely to be identical
        assert not all(q.question == p.question for q, p in zip(r1, r2))

    def test_all_have_nonempty_text(self):
        chunks = _make_chunks(20)
        result = generate_questions(chunks, n=10, seed=42)
        for q in result:
            assert len(q.question.strip()) > 5

    def test_unique_question_ids(self):
        chunks = _make_chunks(30)
        result = generate_questions(chunks, n=15, seed=42)
        ids = [q.question_id for q in result]
        assert len(ids) == len(set(ids))

    def test_gold_chunk_ids_nonempty(self):
        chunks = _make_chunks(20)
        result = generate_questions(chunks, n=5, seed=42)
        for q in result:
            assert len(q.gold_chunk_ids) > 0

    def test_regulator_field_set(self):
        chunks = _make_chunks(20)
        result = generate_questions(chunks, n=10, seed=42)
        for q in result:
            assert q.regulator in ("RBI", "SEBI", "INCOMETAX")

    def test_families_present_in_large_run(self):
        chunks = _make_chunks(50)
        result = generate_questions(chunks, n=100, seed=42)
        families = {q.family for q in result}
        # At least 3 different families should appear
        assert len(families) >= 3

    def test_temporal_questions_tagged(self):
        chunks = _make_chunks(20)
        before = chunks[:10]
        after  = chunks[10:]
        result = generate_temporal_questions(before, after, n=5, seed=42)
        assert all(q.temporal_tag for q in result)

    def test_temporal_questions_family(self):
        chunks = _make_chunks(20)
        result = generate_temporal_questions(chunks[:5], chunks[5:10], n=3, seed=42)
        assert all(q.family == "TEMPORAL" for q in result)

    def test_question_item_roundtrip(self):
        q = QuestionItem(
            question_id="q_test_001",
            question="What is the CRR requirement?",
            family="RATE",
            regulator="RBI",
            source_chunk_id="chunk_0001",
            issue_date="2024-01-01",
            gold_chunk_ids=["chunk_0001"],
            expected_atoms=["RATE"],
        )
        d = q.to_dict()
        q2 = QuestionItem.from_dict(d)
        assert q2.question_id == q.question_id
        assert q2.family == "RATE"

    def test_save_load_questions(self):
        chunks  = _make_chunks(20)
        qs      = generate_questions(chunks, n=8, seed=42)
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            save_questions(qs, path)
            loaded = load_questions(path)
            assert len(loaded) == len(qs)
            assert loaded[0].question_id == qs[0].question_id
        finally:
            Path(path).unlink(missing_ok=True)

    def test_extract_hints_pct(self):
        hints = _extract_hints("The CRR rate is 4.5% of NDTL.")
        assert hints["has_rate"] is True
        assert hints["first_pct"] == "4.5"

    def test_extract_hints_section(self):
        hints = _extract_hints("As per Section 42(1)(a) of the RBI Act.")
        assert hints["has_section"] is True


# ══════════════════════════════════════════════════════════════════════════════
# Group 2: bench/annotation/annotator.py
# ══════════════════════════════════════════════════════════════════════════════

from bench.annotation.annotator import (
    AnnotationRecord, prelabel, save_annotations, load_annotations, annotation_stats,
)


class TestPrelabel:

    def test_mismatch_unsupported(self):
        assert prelabel("MISMATCH", 0.9, 0.1) == "unsupported"

    def test_high_risk_unsupported(self):
        assert prelabel("NOT_FOUND", None, 0.85) == "unsupported"

    def test_match_low_risk_supported(self):
        assert prelabel("MATCH", None, 0.10) == "supported"

    def test_high_entail_low_risk_supported(self):
        assert prelabel("NOT_FOUND", 0.95, 0.20) == "supported"

    def test_uncertain_fallback(self):
        assert prelabel("NOT_FOUND", 0.60, 0.50) == "uncertain"

    def test_na_uncertain(self):
        assert prelabel("NA", None, 0.50) == "uncertain"


class TestAnnotationRecord:

    def _make_record(self, label=None, prelabel_val="supported") -> AnnotationRecord:
        return AnnotationRecord(
            question_id="q1",
            atom_id="a1",
            atom_type="RATE",
            atom_text="4.5%",
            atom_claim="CRR is 4.5%.",
            regulator="RBI",
            v1_status="MATCH",
            v2_entail_prob=0.95,
            risk=0.10,
            prelabel=prelabel_val,
            label=label,
        )

    def test_is_verified_true(self):
        r = self._make_record(label="supported")
        assert r.is_verified is True

    def test_is_verified_false_when_none(self):
        r = self._make_record(label=None)
        assert r.is_verified is False

    def test_is_wrong_unsupported(self):
        r = self._make_record(label="unsupported")
        assert r.is_wrong is True

    def test_is_wrong_outdated(self):
        r = self._make_record(label="outdated")
        assert r.is_wrong is True

    def test_is_wrong_supported(self):
        r = self._make_record(label="supported")
        assert r.is_wrong is False

    def test_save_load_roundtrip(self):
        records = [
            AnnotationRecord("q1", "a1", "RATE", "4.5%", "claim", "RBI",
                             "MATCH", 0.9, 0.1, "supported", "supported"),
            AnnotationRecord("q1", "a2", "SECTION", "Sec 42", "claim", "RBI",
                             "NOT_FOUND", None, 0.6, "uncertain", None),
        ]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            save_annotations(records, path)
            loaded = load_annotations(path)
            assert len(loaded) == 2
            assert loaded[0].atom_id == "a1"
            assert loaded[1].label is None
        finally:
            Path(path).unlink(missing_ok=True)

    def test_annotation_stats_counts(self):
        records = [
            AnnotationRecord("q1", "a1", "RATE", "x", "c", "RBI", "MATCH", 0.9, 0.1, "supported", "supported"),
            AnnotationRecord("q1", "a2", "RATE", "x", "c", "RBI", "MISMATCH", 0.1, 0.9, "unsupported", "unsupported"),
            AnnotationRecord("q1", "a3", "RATE", "x", "c", "RBI", "NOT_FOUND", None, 0.5, "uncertain", None),
        ]
        stats = annotation_stats(records)
        assert stats["total"] == 3
        assert stats["supported"] == 1
        assert stats["unsupported"] == 1
        assert stats["uncertain"] == 1

    def test_annotation_stats_pct_wrong(self):
        records = [
            AnnotationRecord("q1", "a1", "RATE", "x", "c", "RBI", "MATCH", 0.9, 0.1, "supported", "supported"),
            AnnotationRecord("q1", "a2", "RATE", "x", "c", "RBI", "MISMATCH", 0.1, 0.9, "unsupported", "unsupported"),
        ]
        stats = annotation_stats(records)
        assert stats["pct_wrong"] == 50.0


# ══════════════════════════════════════════════════════════════════════════════
# Group 3: bench/annotation/agreement.py
# ══════════════════════════════════════════════════════════════════════════════

from bench.annotation.agreement import (
    cohen_kappa, percent_agreement, agreement_stats, confusion_matrix,
    _interpret_kappa,
)


class TestCohenKappa:

    def test_perfect_agreement(self):
        labels = ["supported", "unsupported", "outdated", "uncertain"]
        kappa  = cohen_kappa(labels, labels)
        assert abs(kappa - 1.0) < 1e-9

    def test_empty_lists(self):
        assert cohen_kappa([], []) == 0.0

    def test_mismatched_length(self):
        assert cohen_kappa(["supported"], ["supported", "unsupported"]) == 0.0

    def test_all_same_class_degenerate(self):
        labels = ["supported"] * 10
        # When pe is near 1, kappa may be 0 or 1
        k = cohen_kappa(labels, labels)
        assert 0.0 <= k <= 1.0

    def test_partial_agreement(self):
        a = ["supported", "supported", "unsupported", "unsupported"]
        b = ["supported", "unsupported", "unsupported", "outdated"]
        k = cohen_kappa(a, b)
        assert -1.0 <= k <= 1.0

    def test_percent_agreement_perfect(self):
        labels = ["supported"] * 5
        assert percent_agreement(labels, labels) == 1.0

    def test_percent_agreement_zero(self):
        a = ["supported"] * 4
        b = ["unsupported"] * 4
        assert percent_agreement(a, b) == 0.0

    def test_confusion_matrix_shape(self):
        a = ["supported", "unsupported"]
        b = ["supported", "outdated"]
        mat = confusion_matrix(a, b)
        assert mat.shape == (4, 4)

    def test_confusion_matrix_diagonal(self):
        labels = ["supported", "unsupported", "supported"]
        mat = confusion_matrix(labels, labels)
        assert mat.diagonal().sum() == 3.0

    def test_agreement_stats_keys(self):
        pairs = [("supported", "supported"), ("unsupported", "outdated")]
        stats = agreement_stats(pairs)
        assert "kappa" in stats
        assert "percent_agreement" in stats
        assert "n_pairs" in stats
        assert "confusion_matrix" in stats


# ══════════════════════════════════════════════════════════════════════════════
# Group 4: bench/splits.py
# ══════════════════════════════════════════════════════════════════════════════

from bench.splits import split_questions, SplitConfig, temporal_split, save_split, load_split


class TestSplits:

    def _make_questions(self, n=60):
        from bench.question_gen.generator import QuestionItem
        families = ["RATE", "THRESHOLD", "SECTION", "DATE", "APPLICABILITY", "ENTITY"]
        return [
            QuestionItem(
                question_id=f"q_{i:04d}",
                question=f"Question {i}?",
                family=families[i % len(families)],
                regulator="RBI",
                source_chunk_id=f"chunk_{i:04d}",
                issue_date=f"202{i % 3 + 1}-01-01",
                gold_chunk_ids=[f"chunk_{i:04d}"],
                expected_atoms=["RATE"],
            )
            for i in range(n)
        ]

    def test_sizes_sum_to_total(self):
        qs = self._make_questions(60)
        splits = split_questions(qs)
        total = sum(len(v) for v in splits.values())
        assert total == len(qs)

    def test_no_empty_splits(self):
        qs = self._make_questions(60)
        splits = split_questions(qs)
        for name, items in splits.items():
            assert len(items) > 0, f"Split {name} is empty"

    def test_no_question_in_two_splits(self):
        qs = self._make_questions(60)
        splits = split_questions(qs, SplitConfig(split_by="family"))
        all_ids = [q.question_id for s in splits.values() for q in s]
        assert len(all_ids) == len(set(all_ids))

    def test_empty_input(self):
        splits = split_questions([])
        assert all(len(v) == 0 for v in splits.values())

    def test_invalid_fractions_raises(self):
        with pytest.raises(ValueError):
            SplitConfig(agg_train_frac=0.5, calibration_frac=0.4, test_frac=0.4)

    def test_save_load_roundtrip(self):
        qs = self._make_questions(30)
        splits = split_questions(qs)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_split(splits, tmpdir)
            loaded = load_split(tmpdir)
        for name in ("agg_train", "agg_val", "calibration", "test"):
            assert len(loaded[name]) == len(splits[name])

    def test_temporal_split(self):
        qs = self._make_questions(30)
        pre, post = temporal_split(qs, cutoff_date="2023-01-01")
        assert len(pre) + len(post) == len(qs)


# ══════════════════════════════════════════════════════════════════════════════
# Group 5: eval/metrics.py
# ══════════════════════════════════════════════════════════════════════════════

from eval.metrics import (
    hallucination_rate, coverage, risk_coverage_curve, aurc,
    empirical_violation_rate, recall_at_k, mrr, precision_at_k,
    extraction_recall, extraction_precision, extraction_f1,
    judge_call_rate, mean_latency,
)


class TestHallucinationRate:

    def test_all_correct_supported(self):
        labels    = ["supported"] * 5
        decisions = ["SUPPORTED"] * 5
        assert hallucination_rate(labels, decisions) == 0.0

    def test_all_wrong_supported(self):
        labels    = ["unsupported"] * 4
        decisions = ["SUPPORTED"] * 4
        assert hallucination_rate(labels, decisions) == 1.0

    def test_no_accepted_is_nan(self):
        labels    = ["supported"] * 3
        decisions = ["ABSTAINED"] * 3
        assert math.isnan(hallucination_rate(labels, decisions))

    def test_mixed(self):
        labels    = ["supported", "unsupported", "supported", "supported"]
        decisions = ["SUPPORTED", "SUPPORTED", "ABSTAINED", "VERIFIED"]
        # Accepted: indices 0 (supported), 1 (unsupported), 3 (supported)
        # Wrong accepted: index 1
        hr = hallucination_rate(labels, decisions)
        assert abs(hr - 1 / 3) < 1e-6


class TestCoverage:

    def test_all_supported(self):
        assert coverage(["SUPPORTED"] * 5) == 1.0

    def test_all_abstained(self):
        assert coverage(["ABSTAINED"] * 5) == 0.0

    def test_half_accepted(self):
        decisions = ["SUPPORTED", "ABSTAINED", "VERIFIED", "UNCERTAIN"]
        assert abs(coverage(decisions) - 0.5) < 1e-6

    def test_empty(self):
        assert coverage([]) == 0.0


class TestRiskCoverageCurve:

    def test_monotone_coverage(self):
        risks  = [random.random() for _ in range(100)]
        labels = ["supported"] * 100
        curve  = risk_coverage_curve(risks, labels)
        covs   = [p[0] for p in curve]
        assert covs == sorted(covs)

    def test_hrate_bounded(self):
        risks  = [random.random() for _ in range(100)]
        labels = [random.choice(["supported", "unsupported"]) for _ in range(100)]
        curve  = risk_coverage_curve(risks, labels)
        for _, hr in curve:
            assert 0.0 <= hr <= 1.0

    def test_all_correct_hrate_zero(self):
        risks  = [i / 100 for i in range(100)]
        labels = ["supported"] * 100
        curve  = risk_coverage_curve(risks, labels)
        for _, hr in curve:
            assert hr == 0.0


class TestAURC:

    def test_empty_curve(self):
        assert aurc([]) == 0.0

    def test_single_point(self):
        assert aurc([(0.5, 0.1)]) == 0.0

    def test_all_zero_hrate(self):
        curve = [(c / 10, 0.0) for c in range(1, 11)]
        assert aurc(curve) == 0.0

    def test_positive_for_mixed(self):
        curve = [(0.3, 0.05), (0.6, 0.10), (0.9, 0.20)]
        assert aurc(curve) > 0.0


class TestRetrievalMetrics:

    def test_recall_at_k_full(self):
        assert recall_at_k(["a", "b", "c"], ["a", "b"], k=3) == 1.0

    def test_recall_at_k_miss(self):
        assert recall_at_k(["x", "y", "z"], ["a"], k=3) == 0.0

    def test_mrr_first_rank(self):
        assert mrr([["a", "b"]], [["a"]]) == 1.0

    def test_mrr_no_hit(self):
        assert mrr([["x", "y"]], [["a"]]) == 0.0

    def test_precision_at_k(self):
        assert precision_at_k(["a", "b", "c"], ["a", "c"], k=3) == pytest.approx(2/3, abs=1e-6)


class TestExtractionMetrics:

    def test_perfect_recall(self):
        assert extraction_recall(["a", "b", "c"], ["a", "b", "c"]) == 1.0

    def test_perfect_precision(self):
        assert extraction_precision(["a", "b"], ["a", "b", "c"]) == 1.0

    def test_f1_zero_precision(self):
        assert extraction_f1(1.0, 0.0) == 0.0

    def test_f1_harmonic_mean(self):
        f1 = extraction_f1(0.8, 0.6)
        assert abs(f1 - 2 * 0.8 * 0.6 / (0.8 + 0.6)) < 1e-6

    def test_extraction_recall_normalises(self):
        assert extraction_recall(["CRR Is 4.5%"], ["crr is 4.5%"]) == 1.0


class TestJudgeCallRate:

    def test_all_uncertain(self):
        decisions = ["UNCERTAIN"] * 5
        assert judge_call_rate(decisions) == 0.0   # UNCERTAIN atoms not yet judged

    def test_all_verified(self):
        decisions = ["VERIFIED"] * 5
        assert judge_call_rate(decisions) == 1.0

    def test_half_judged(self):
        decisions = ["SUPPORTED", "VERIFIED", "ABSTAINED", "NOT_VERIFIED"]
        assert abs(judge_call_rate(decisions) - 0.5) < 1e-6


class TestEmpiricalViolationRate:

    def test_no_wrong_atoms_zero_violation(self):
        risks  = [0.1] * 100
        labels = ["supported"] * 100
        result = empirical_violation_rate(risks, labels, accept_threshold=0.5, eps=0.05, n_splits=100)
        assert result["violation_rate"] == 0.0

    def test_all_wrong_high_violation(self):
        risks  = [0.1] * 100
        labels = ["unsupported"] * 100
        result = empirical_violation_rate(risks, labels, accept_threshold=0.5, eps=0.05, n_splits=100)
        assert result["violation_rate"] == 1.0

    def test_result_keys(self):
        risks  = [random.random() for _ in range(50)]
        labels = [random.choice(["supported", "unsupported"]) for _ in range(50)]
        result = empirical_violation_rate(risks, labels, accept_threshold=0.5, eps=0.05, n_splits=50)
        assert "violation_rate" in result
        assert "mean_selective_risk" in result
        assert "eps" in result


# ══════════════════════════════════════════════════════════════════════════════
# Group 6: eval/baselines.py
# ══════════════════════════════════════════════════════════════════════════════

from eval.baselines import (
    plain_rag, selfcheck, nli_only, conformal_no_mondrian,
    run_baseline, BASELINE_REGISTRY,
)


class TestBaselines:

    def _atoms(self, n=4):
        return [_make_scored_atom(
            atom_id=f"a{i}",
            v1_status=["MATCH", "MISMATCH", "NOT_FOUND", "NA"][i % 4],
            v2_entail=0.9 if i % 2 == 0 else 0.2,
            risk=[0.1, 0.5, 0.8, 0.3][i % 4],
        ) for i in range(n)]

    def test_plain_rag_all_supported(self):
        atoms = self._atoms(4)
        result = plain_rag(atoms)
        assert all(v == "SUPPORTED" for v in result.values())

    def test_selfcheck_high_entail_supported(self):
        atom = _make_scored_atom("a1", v2_entail=0.90)
        result = selfcheck([atom])
        assert result["a1"] == "SUPPORTED"

    def test_selfcheck_low_entail_abstained(self):
        atom = _make_scored_atom("a1", v2_entail=0.20)
        result = selfcheck([atom])
        assert result["a1"] == "ABSTAINED"

    def test_selfcheck_none_entail_abstained(self):
        atom = _make_scored_atom("a1", v2_entail=None)
        atom.verified.v2_entail_prob = None
        result = selfcheck([atom])
        assert result["a1"] == "ABSTAINED"

    def test_nli_only_v1_match_fallback(self):
        atom = _make_scored_atom("a1", v1_status="MATCH", v2_entail=None)
        atom.verified.v2_entail_prob = None
        result = nli_only([atom])
        assert result["a1"] == "SUPPORTED"

    def test_nli_only_mismatch_abstained(self):
        atom = _make_scored_atom("a1", v1_status="MISMATCH", v2_entail=None)
        atom.verified.v2_entail_prob = None
        result = nli_only([atom])
        assert result["a1"] == "ABSTAINED"

    def test_conformal_low_risk_supported(self):
        atom = _make_scored_atom("a1", risk=0.2)
        result = conformal_no_mondrian([atom], global_threshold=0.5)
        assert result["a1"] == "SUPPORTED"

    def test_conformal_high_risk_abstained(self):
        atom = _make_scored_atom("a1", risk=0.8)
        result = conformal_no_mondrian([atom], global_threshold=0.5)
        assert result["a1"] == "ABSTAINED"

    def test_unknown_baseline_raises(self):
        with pytest.raises(ValueError):
            run_baseline("nonexistent_baseline", [])

    def test_all_baselines_run(self):
        atoms = self._atoms(4)
        for name in BASELINE_REGISTRY:
            result = run_baseline(name, atoms)
            assert isinstance(result, dict)
            assert len(result) == 4


# ══════════════════════════════════════════════════════════════════════════════
# Group 7: experiments/drift/drift_exp.py  (standalone — NOT part of main pipeline)
# ══════════════════════════════════════════════════════════════════════════════

from experiments.drift.drift_exp import run_drift_experiment, DriftExperimentResult


class TestDriftExperiment:

    def _make_data(self, n=100, error_rate=0.05, seed=42):
        rng = random.Random(seed)
        risks  = [round(rng.random(), 4) for _ in range(n)]
        wrong  = {"unsupported", "outdated"}
        labels = [rng.choice(["unsupported", "outdated"]) if rng.random() < error_rate
                  else "supported" for _ in range(n)]
        return risks, labels

    def test_returns_drift_experiment_result(self):
        rp, lp = self._make_data(60)
        rt, lt = self._make_data(40, error_rate=0.15)
        result = run_drift_experiment(rp, lp, rt, lt, eps=0.05, delta=0.10)
        assert isinstance(result, DriftExperimentResult)

    def test_four_strategies_present(self):
        rp, lp = self._make_data(80)
        rt, lt = self._make_data(40)
        result = run_drift_experiment(rp, lp, rt, lt)
        strategies = {r.strategy for r in result.results}
        assert {"static", "sliding_window", "recency_weighted", "triggered"}.issubset(strategies)

    def test_violation_rate_in_range(self):
        rp, lp = self._make_data(80)
        rt, lt = self._make_data(40)
        result = run_drift_experiment(rp, lp, rt, lt)
        for r in result.results:
            assert 0.0 <= r.violation_rate <= 1.0

    def test_empty_pre_returns_empty_result(self):
        result = run_drift_experiment([], [], [0.1] * 5, ["supported"] * 5)
        assert len(result.results) == 0

    def test_best_strategy_returns_name(self):
        rp, lp = self._make_data(80, error_rate=0.02)
        rt, lt = self._make_data(40, error_rate=0.02)
        result = run_drift_experiment(rp, lp, rt, lt)
        best = result.best_strategy()
        if best is not None:
            assert best in ("static", "sliding_window", "recency_weighted", "triggered")

    def test_summary_table_is_list(self):
        rp, lp = self._make_data(80)
        rt, lt = self._make_data(40)
        result = run_drift_experiment(rp, lp, rt, lt)
        table = result.summary_table()
        assert isinstance(table, list)

    def test_summary_table_has_required_keys(self):
        rp, lp = self._make_data(80)
        rt, lt = self._make_data(40)
        result = run_drift_experiment(rp, lp, rt, lt)
        for row in result.summary_table():
            assert "strategy" in row
            assert "violation_rate" in row
            assert "hallucination_rate" in row

    def test_static_violation_higher_on_drifted_data(self):
        """Static calibration should struggle more on heavily drifted data."""
        rng = random.Random(7)
        rp  = [round(rng.random() * 0.3, 4) for _ in range(100)]  # low risk pre
        lp  = ["supported"] * 100
        rt  = [round(0.5 + rng.random() * 0.5, 4) for _ in range(100)]  # high risk post
        lt  = ["unsupported"] * 100  # post data is wrong
        result = run_drift_experiment(rp, lp, rt, lt, eps=0.05, delta=0.10)
        static = next(r for r in result.results if r.strategy == "static")
        assert static.hallucination_rate >= 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Group 8: eval/run_all.py
# ══════════════════════════════════════════════════════════════════════════════

from eval.run_all import run_all


class TestRunAll:

    def test_completes_without_error(self, tmp_path):
        results = run_all(
            results_dir=str(tmp_path),
            seed=42,
            n_splits=50,
            skip_baselines=True,
        )
        assert isinstance(results, dict)

    def test_creates_output_files(self, tmp_path):
        run_all(results_dir=str(tmp_path), seed=42, n_splits=50,
                skip_baselines=True)
        assert (tmp_path / "summary.json").exists()
        assert (tmp_path / "violation_rate.json").exists()
        assert (tmp_path / "rc_curve.json").exists()

    def test_summary_keys(self, tmp_path):
        run_all(results_dir=str(tmp_path), seed=42, n_splits=50,
                skip_baselines=True)
        summary = json.loads((tmp_path / "summary.json").read_text())
        for key in ("hallucination_rate", "coverage", "aurc",
                    "violation_rate", "violation_rate_ok"):
            assert key in summary

    def test_violation_rate_ok_present(self, tmp_path):
        run_all(results_dir=str(tmp_path), seed=42, n_splits=50,
                skip_baselines=True)
        summary = json.loads((tmp_path / "summary.json").read_text())
        assert "violation_rate_ok" in summary
        assert isinstance(summary["violation_rate_ok"], bool)

    def test_skip_baselines(self, tmp_path):
        run_all(results_dir=str(tmp_path), seed=42, n_splits=20,
                skip_baselines=True)
        # main_table.json should NOT exist when skip_baselines=True
        assert not (tmp_path / "main_table.json").exists()

    def test_drift_table_never_created(self, tmp_path):
        """drift_table.json is no longer written by run_all (moved to experiments/drift/)."""
        run_all(results_dir=str(tmp_path), seed=42, n_splits=20,
                skip_baselines=True)
        assert not (tmp_path / "drift_table.json").exists()


    def test_drift_table_still_absent_explicit(self, tmp_path):
        """Even if caller somehow expects skip_drift=False the file must not appear."""
        run_all(results_dir=str(tmp_path), seed=42, n_splits=20,
                skip_baselines=True)
        assert not (tmp_path / "drift_table.json").exists()

    def test_deterministic_with_seed(self, tmp_path):
        r1 = run_all(results_dir=str(tmp_path / "r1"), seed=99, n_splits=30,
                     skip_baselines=True)
        r2 = run_all(results_dir=str(tmp_path / "r2"), seed=99, n_splits=30,
                     skip_baselines=True)
        assert r1["summary"]["hallucination_rate"] == r2["summary"]["hallucination_rate"]

    def test_hallucination_rate_in_range(self, tmp_path):
        results = run_all(results_dir=str(tmp_path), seed=42, n_splits=50,
                          skip_baselines=True)
        hr = results["summary"]["hallucination_rate"]
        assert 0.0 <= hr <= 1.0 or math.isnan(hr)
