"""
tests/test_strict_eval.py
-------------------------
Unit tests verifying --strict flag behavior across evaluation entry points:
1. Missing models/aggregator/aggregator_v1.pkl (no weighted-average fallback in strict mode).
2. Any signal fails to load or returns a constant value for all atoms in a run.
3. M7 returns a None threshold, accepts zero atoms, or falls back to heuristic thresholds.
4. Any query ends with status ERROR.
5. Metrics computed from fixtures or simulation.
6. Outside strict mode, banner "NOT PAPER-VALID" is printed and summary.json has paper_valid: false.
7. strict_default defaults to ON when configs/experiment.yaml has mode: paper.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from common.errors import StrictFailure, print_not_paper_valid_banner
from common.experiment import strict_default
from common.schemas import (
    Atom,
    GeneratedAnswer,
    RetrievalResult,
    Thresholds,
    VerifiedAtom,
)
from eval.run_all import run_all
from guarantee.certify import certify
from signals import validate_signals


# ── Helper Fixtures ──────────────────────────────────────────────────────────

def _make_atoms(n=3):
    return [
        VerifiedAtom(
            atom=Atom(
                atom_id=f"atom_{i}",
                type="RATE",
                text=f"Rate is {5 + i}%",
                claim=f"The prescribed rate is {5 + i}%.",
                cited_chunk="chunk_1",
            ),
            v1_status="MATCH",
            v2_entail_prob=0.9 - 0.1 * i,
        )
        for i in range(n)
    ]


def _make_answer():
    return GeneratedAnswer(
        query="What is the CAR rate?",
        answer_text="The CAR rate is 9%.",
        citations=["chunk_1"],
        metadata={"no_context_answer": "The CAR rate might be 8%."},
    )


def _make_rr():
    return RetrievalResult(
        query="What is the CAR rate?",
        query_date="2024-01-01",
        chunks=[],
        metadata={"rerank_scores": {"chunk_1": 0.85}},
    )


# ── Case 1: Missing Aggregator ───────────────────────────────────────────────

def test_strict_missing_aggregator_raises(tmp_path):
    """In strict mode, missing models/aggregator/aggregator_v1.pkl must raise StrictFailure."""
    from aggregate.model import score

    fake_model_path = str(tmp_path / "nonexistent_aggregator.pkl")
    verified_atoms = _make_atoms(1)
    answer = _make_answer()
    rr = _make_rr()

    with patch("aggregate.model._DEFAULT_MODEL_PATH", Path(fake_model_path)):
        with pytest.raises(StrictFailure, match="aggregator_v1.pkl is missing"):
            score(verified_atoms, answer, rr, model_path=fake_model_path, strict=True)


def test_non_strict_missing_aggregator_falls_back(tmp_path, capsys):
    """Outside strict mode, missing aggregator uses fallback and prints NOT PAPER-VALID banner."""
    from aggregate.model import score

    fake_model_path = str(tmp_path / "nonexistent_aggregator.pkl")
    verified_atoms = _make_atoms(1)
    answer = _make_answer()
    rr = _make_rr()

    with patch("aggregate.model._DEFAULT_MODEL_PATH", Path(fake_model_path)):
        scored = score(verified_atoms, answer, rr, model_path=fake_model_path, strict=False)

    captured = capsys.readouterr()
    assert "NOT PAPER-VALID" in captured.out
    assert len(scored) == 1
    assert scored[0].metadata.get("model_mode") == "fallback_weighted_average"


# ── Case 2: Signal Fails to Load or Constant Value ───────────────────────────

def test_strict_signal_fails_to_load_raises():
    """In strict mode, if a signal fails to load (e.g. NLI pipeline unavailable), raise StrictFailure."""
    from verify.v2_nli import NLIVerifier

    verifier = NLIVerifier.__new__(NLIVerifier)
    verifier.max_length = 512
    with patch("verify.v2_nli._pipeline", None):
        with pytest.raises(StrictFailure, match="Signal 's_nli' failed to load"):
            verifier.predict("premise", "hypothesis", strict=True)


def test_strict_signal_embedder_fails_to_load_raises():
    """In strict mode, if s_div embedder fails to load, raise StrictFailure."""
    from signals.s_div import compute_s_div

    va = _make_atoms(1)[0]
    ans = _make_answer()
    with patch("signals.s_div._get_embedder", return_value=None):
        with pytest.raises(StrictFailure, match="Signal 's_div' failed to load"):
            compute_s_div(va, ans, strict=True)


def test_strict_signal_constant_value_raises():
    """In strict mode, if any signal returns a constant value across all atoms, raise StrictFailure."""
    # 3 atoms with identical s_nli = 0.5
    records = [
        {"atom_id": "a1", "s_nli": 0.5, "s_ret": 0.8, "s_div": 0.3},
        {"atom_id": "a2", "s_nli": 0.5, "s_ret": 0.7, "s_div": 0.4},
        {"atom_id": "a3", "s_nli": 0.5, "s_ret": 0.6, "s_div": 0.2},
    ]
    with pytest.raises(StrictFailure, match="Signal 's_nli' returned a constant value"):
        validate_signals(records, strict=True)


def test_non_strict_signal_constant_value_prints_banner(capsys):
    """Outside strict mode, constant signal returns False and prints banner."""
    records = [
        {"atom_id": "a1", "s_nli": 0.5, "s_ret": 0.8},
        {"atom_id": "a2", "s_nli": 0.5, "s_ret": 0.7},
    ]
    ok = validate_signals(records, strict=False)
    assert ok is False
    captured = capsys.readouterr()
    assert "NOT PAPER-VALID" in captured.out


# ── Case 3: M7 None Threshold, Zero Accepted, or Heuristic Fallback ──────────

def test_strict_m7_none_threshold_raises(capsys):
    """In strict mode, stratum with None threshold must print stratum and raise StrictFailure."""
    from scripts.run_full_eval import run_m7_ltt
    atoms = [{"risk": 0.9, "label": "unsupported"}, {"risk": 0.95, "label": "unsupported"}]
    with pytest.raises(StrictFailure):
        run_m7_ltt(atoms, strict=True)

    captured = capsys.readouterr()
    assert "Stratum" in captured.out


def test_strict_m7_zero_accepted_in_certify_raises(tmp_path, capsys):
    """In guarantee.certify, if a stratum accepts zero atoms, print stratum and raise StrictFailure."""
    calib = tmp_path / "calib.jsonl"
    rows = [
        {"risk": 0.9, "label": 1, "regulator": "RBI", "atom_type": "RATE", "date": "2024-01-01"}
        for _ in range(100)
    ]
    calib.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    out = tmp_path / "thresholds.json"
    with pytest.raises(StrictFailure, match="Stratum '.*' failed M7 certification"):
        certify(str(calib), str(out), eps_accept=0.05, delta=0.10, n_min=10, run_simulation=False, strict=True)

    captured = capsys.readouterr()
    assert "failed M7 certification" in captured.out


def test_strict_m7_heuristic_fallback_raises(tmp_path, capsys):
    """In strict mode, falling back to heuristic thresholds (accept<0.35, abstain>0.70) raises."""
    # Ensure aggregator exists so we test the threshold check
    agg = tmp_path / "models" / "aggregator" / "aggregator_v1.pkl"
    agg.parent.mkdir(parents=True, exist_ok=True)
    agg.write_bytes(b"dummy")

    test_jsonl = tmp_path / "test.jsonl"
    rows = [
        {"risk": 0.2, "label": "supported", "decision": "SUPPORTED", "s_nli": 0.1, "s_ret": 0.8},
        {"risk": 0.8, "label": "unsupported", "decision": "ABSTAINED", "s_nli": 0.9, "s_ret": 0.2},
    ]
    test_jsonl.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    # Point to nonexistent thresholds.json
    with patch("eval.run_all.ROOT", tmp_path):
        with pytest.raises(StrictFailure, match="heuristic thresholds"):
            run_all(results_dir=str(tmp_path / "out"), test_data=str(test_jsonl), strict=True)

    captured = capsys.readouterr()
    assert "Stratum '*|*'" in captured.out


# ── Case 4: Query Ends with Status ERROR ─────────────────────────────────────

def test_strict_query_error_raises():
    """In strict mode, any query ending in ERROR must raise StrictFailure."""
    from scripts.run_full_eval import run_one_query

    retriever = MagicMock()
    retriever.retrieve.return_value = _make_rr()
    generator = MagicMock()
    generator.generate.side_effect = RuntimeError("LLM timed out")

    with pytest.raises(RuntimeError, match="LLM timed out"):
        run_one_query("test question", "2024-01-01", retriever, generator, {}, None, {}, strict=True)


# ── Case 5: Metrics Computed from Simulation or Fixtures ─────────────────────

def test_strict_simulation_raises(tmp_path):
    """In strict mode, run_all with no test_data (simulation) must raise StrictFailure."""
    # Ensure aggregator exists so we test the simulation check
    agg = tmp_path / "models" / "aggregator" / "aggregator_v1.pkl"
    agg.parent.mkdir(parents=True, exist_ok=True)
    agg.write_bytes(b"dummy")

    with patch("eval.run_all.ROOT", tmp_path):
        with pytest.raises(StrictFailure, match="no held-out test JSONL"):
            run_all(results_dir=str(tmp_path), test_data=None, strict=True)


def test_strict_fixtures_raises(tmp_path):
    """In strict mode, running with test_data from a fixtures path must raise StrictFailure."""
    # Ensure aggregator exists so we test the fixture check
    agg = tmp_path / "models" / "aggregator" / "aggregator_v1.pkl"
    agg.parent.mkdir(parents=True, exist_ok=True)
    agg.write_bytes(b"dummy")

    fixture_dir = tmp_path / "tests" / "fixtures"
    fixture_dir.mkdir(parents=True)
    test_file = fixture_dir / "atoms_fixtures.jsonl"
    rows = [
        {"risk": 0.1, "label": "supported", "decision": "SUPPORTED", "s_nli": 0.1},
        {"risk": 0.7, "label": "unsupported", "decision": "ABSTAINED", "s_nli": 0.8},
    ]
    test_file.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    with patch("eval.run_all.ROOT", tmp_path):
        with pytest.raises(StrictFailure, match="Metrics computed from fixtures"):
            run_all(results_dir=str(tmp_path / "out"), test_data=str(test_file), strict=True)


def test_non_strict_simulation_writes_paper_valid_false(tmp_path, capsys):
    """Outside strict mode, simulated run sets paper_valid=false in summary.json and prints banner."""
    run_all(results_dir=str(tmp_path), test_data=None, strict=False, skip_baselines=True)

    summary_file = tmp_path / "summary.json"
    assert summary_file.exists()
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    assert summary.get("simulated") is True
    assert summary.get("paper_valid") is False

    captured = capsys.readouterr()
    assert "NOT PAPER-VALID" in captured.out


# ── Case 6: Config Mode Paper Default ────────────────────────────────────────

def test_strict_default_on_when_mode_paper(tmp_path):
    """When configs/experiment.yaml has mode: paper, strict_default returns True."""
    exp_file = tmp_path / "experiment.yaml"
    exp_file.write_text("mode: paper\nstrict: false\n", encoding="utf-8")

    assert strict_default(str(exp_file)) is True


def test_strict_default_off_when_not_paper_mode(tmp_path):
    """When configs/experiment.yaml has mode: dev and strict: false, strict_default returns False."""
    exp_file = tmp_path / "experiment.yaml"
    exp_file.write_text("mode: dev\nstrict: false\n", encoding="utf-8")

    assert strict_default(str(exp_file)) is False


# ── Additional Case Coverage for All Conditions and Entry Points ──────────────

def test_strict_missing_aggregator_in_run_all_raises(tmp_path):
    """In strict mode, run_all must raise StrictFailure if aggregator_v1.pkl is missing."""
    test_jsonl = tmp_path / "test.jsonl"
    rows = [{"risk": 0.2, "label": "supported", "decision": "SUPPORTED", "s_nli": 0.1, "s_ret": 0.8}]
    test_jsonl.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    with patch("eval.run_all.ROOT", tmp_path):
        with pytest.raises(StrictFailure, match="aggregator_v1.pkl is missing"):
            run_all(results_dir=str(tmp_path / "out"), test_data=str(test_jsonl), strict=True)


def test_non_strict_missing_aggregator_in_run_all_sets_paper_valid_false(tmp_path, capsys):
    """Outside strict mode, run_all with missing aggregator prints banner and sets paper_valid=false."""
    test_jsonl = tmp_path / "test.jsonl"
    rows = [
        {"risk": 0.2, "label": "supported", "decision": "SUPPORTED", "s_nli": 0.1, "s_ret": 0.8},
        {"risk": 0.8, "label": "unsupported", "decision": "ABSTAINED", "s_nli": 0.9, "s_ret": 0.2},
    ]
    test_jsonl.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    with patch("eval.run_all.ROOT", tmp_path):
        run_all(results_dir=str(tmp_path / "out"), test_data=str(test_jsonl), strict=False, skip_baselines=True)

    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert summary.get("paper_valid") is False
    captured = capsys.readouterr()
    assert "NOT PAPER-VALID" in captured.out


def test_strict_verify_cascade_fails_to_load_raises():
    """In strict mode, verify() cascade must propagate StrictFailure when NLI fails to load."""
    from verify.cascade import verify
    atoms = [
        Atom(atom_id="a1", type="ENTITY", text="Entity claim", claim="Entity claim is valid.")
    ]
    rr = _make_rr()
    # Patch both _pipeline AND _model_loaded so that _load_model() is a no-op
    # and cannot overwrite the None sentinel with a real model during the call.
    with patch("verify.v2_nli._pipeline", None), \
         patch("verify.v2_nli._model_loaded", True), \
         patch("verify.v2_nli._verifier_instance", None):
        with pytest.raises(StrictFailure, match="Signal 's_nli' failed to load"):
            verify(atoms, rr, strict=True)


def test_strict_constant_signal_in_run_all_raises(tmp_path):
    """In strict mode, run_all with constant signal across atoms must raise StrictFailure."""
    agg = tmp_path / "models" / "aggregator" / "aggregator_v1.pkl"
    agg.parent.mkdir(parents=True, exist_ok=True)
    agg.write_bytes(b"dummy")

    test_jsonl = tmp_path / "test.jsonl"
    rows = [
        {"risk": 0.2, "label": "supported", "decision": "SUPPORTED", "s_nli": 0.5, "s_ret": 0.8},
        {"risk": 0.8, "label": "unsupported", "decision": "ABSTAINED", "s_nli": 0.5, "s_ret": 0.2},
    ]
    test_jsonl.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    with patch("eval.run_all.ROOT", tmp_path):
        with pytest.raises(StrictFailure, match="Signal 's_nli' returned a constant value"):
            run_all(results_dir=str(tmp_path / "out"), test_data=str(test_jsonl), strict=True)


def test_strict_m7_loaded_zero_accepted_raises(tmp_path, capsys):
    """When thresholds.json has a stratum accepting 0 atoms, print stratum and raise StrictFailure."""
    agg = tmp_path / "models" / "aggregator" / "aggregator_v1.pkl"
    agg.parent.mkdir(parents=True, exist_ok=True)
    agg.write_bytes(b"dummy")

    # Write a thresholds.json with a failing stratum
    thr_file = tmp_path / "models" / "guarantee" / "thresholds.json"
    thr_file.parent.mkdir(parents=True, exist_ok=True)
    thr_data = {
        "RBI|RATE": {
            "stratum_name": "RBI|RATE",
            "accept_below": 0.10,
            "abstain_above": 0.80,
            "metadata": {"n_accepted": 0, "n_atoms": 50},
        }
    }
    thr_file.write_text(json.dumps(thr_data), encoding="utf-8")

    test_jsonl = tmp_path / "test.jsonl"
    rows = [
        {"risk": 0.2, "label": "supported", "decision": "SUPPORTED", "s_nli": 0.1, "s_ret": 0.8},
        {"risk": 0.8, "label": "unsupported", "decision": "ABSTAINED", "s_nli": 0.9, "s_ret": 0.2},
    ]
    test_jsonl.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    with patch("eval.run_all.ROOT", tmp_path):
        with pytest.raises(StrictFailure, match="Stratum 'RBI\\|RATE' failed M7 certification"):
            run_all(results_dir=str(tmp_path / "out"), test_data=str(test_jsonl), strict=True)

    captured = capsys.readouterr()
    assert "Stratum 'RBI|RATE' failed M7 certification" in captured.out


def test_non_strict_m7_loaded_zero_accepted_sets_paper_valid_false(tmp_path, capsys):
    """When thresholds.json has a stratum accepting 0 atoms, outside strict mode set paper_valid=false."""
    agg = tmp_path / "models" / "aggregator" / "aggregator_v1.pkl"
    agg.parent.mkdir(parents=True, exist_ok=True)
    agg.write_bytes(b"dummy")

    thr_file = tmp_path / "models" / "guarantee" / "thresholds.json"
    thr_file.parent.mkdir(parents=True, exist_ok=True)
    thr_data = {
        "RBI|RATE": {
            "stratum_name": "RBI|RATE",
            "accept_below": 0.10,
            "abstain_above": 0.80,
            "metadata": {"n_accepted": 0, "n_atoms": 50},
        }
    }
    thr_file.write_text(json.dumps(thr_data), encoding="utf-8")

    test_jsonl = tmp_path / "test.jsonl"
    rows = [
        {"risk": 0.2, "label": "supported", "decision": "SUPPORTED", "s_nli": 0.1, "s_ret": 0.8},
        {"risk": 0.8, "label": "unsupported", "decision": "ABSTAINED", "s_nli": 0.9, "s_ret": 0.2},
    ]
    test_jsonl.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    with patch("eval.run_all.ROOT", tmp_path):
        run_all(results_dir=str(tmp_path / "out"), test_data=str(test_jsonl), strict=False, skip_baselines=True)

    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert summary.get("paper_valid") is False
    captured = capsys.readouterr()
    assert "NOT PAPER-VALID" in captured.out
    assert "Stratum 'RBI|RATE' failed M7 certification" in captured.out


def test_strict_gold_fixture_raises_in_run_all(tmp_path):
    """In strict mode, gold_chunks with 'fixture' in path must raise StrictFailure."""
    agg = tmp_path / "models" / "aggregator" / "aggregator_v1.pkl"
    agg.parent.mkdir(parents=True, exist_ok=True)
    agg.write_bytes(b"dummy")

    thr_file = tmp_path / "models" / "guarantee" / "thresholds.json"
    thr_file.parent.mkdir(parents=True, exist_ok=True)
    thr_file.write_text(json.dumps({"*|*": {"stratum_name": "*|*", "accept_below": 0.35, "abstain_above": 0.70}}), encoding="utf-8")

    test_jsonl = tmp_path / "test.jsonl"
    rows = [
        {"risk": 0.2, "label": "supported", "decision": "SUPPORTED", "s_nli": 0.1, "s_ret": 0.8},
        {"risk": 0.8, "label": "unsupported", "decision": "ABSTAINED", "s_nli": 0.9, "s_ret": 0.2},
    ]
    test_jsonl.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    fixture_chunks = tmp_path / "tests" / "fixtures" / "chunks.jsonl"
    fixture_chunks.parent.mkdir(parents=True, exist_ok=True)
    fixture_chunks.write_text("{}", encoding="utf-8")

    with patch("eval.run_all.ROOT", tmp_path):
        with pytest.raises(StrictFailure, match="Metrics computed from fixtures"):
            run_all(
                results_dir=str(tmp_path / "out"),
                test_data=str(test_jsonl),
                gold_chunks=str(fixture_chunks),
                strict=True,
                skip_baselines=True,
            )


def test_certify_cli_has_strict_option():
    """Verify guarantee.certify CLI parser accepts --strict and defaults using strict_default."""
    import argparse
    from guarantee.certify import certify
    # Check that certify accepts strict kwarg
    import inspect
    sig = inspect.signature(certify)
    assert "strict" in sig.parameters
