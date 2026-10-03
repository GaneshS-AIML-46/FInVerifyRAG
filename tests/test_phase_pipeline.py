"""Tests for the publishable-pipeline phases. These do not write paper metrics."""
from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from common.checkpoint import append_jsonl, completed_ids, resume_pending
from common.errors import StrictFailure
from common.experiment import config_hash, load_experiment
from common.rate_limit import TokenBucket
from atoms.llm_extract import loads_atom_json
from guarantee.ltt import (
    certify_lambda,
    clopper_pearson,
    evaluate_untouched_test,
    min_accepted,
    simulate_violation_rate,
)


def test_min_accepted_eps_005_delta_01():
    assert min_accepted(0.05, 0.1) == 45


def test_config_hash_stable():
    assert len(config_hash()) == 64
    assert load_experiment()["strict"] is True


def test_token_bucket_disabled_does_not_block():
    bucket = TokenBucket(0.0, 1.0)
    assert bucket.acquire() == 0.0


def test_checkpoint_resume(tmp_path):
    path = tmp_path / "ckpt.jsonl"
    append_jsonl(path, {"question_id": "q1", "status": "OK"})
    append_jsonl(path, {"question_id": "q2", "status": "ERROR"})
    assert completed_ids(path) == {"q1"}
    pending = resume_pending(
        [{"question_id": "q1"}, {"question_id": "q2"}, {"question_id": "q3"}],
        path,
    )
    assert [r["question_id"] for r in pending] == ["q2", "q3"]


def test_json_repair_pulls_array_out_of_prose():
    raw = "here you go\n[{\"type\": \"RATE\", \"text\": \"5%\"}]\nthanks"
    parsed = loads_atom_json(raw)
    assert parsed[0]["type"] == "RATE"


def test_strict_score_refuses_missing_aggregator(tmp_path):
    from aggregate.model import score
    from common.schemas import Atom, GeneratedAnswer, RetrievalResult, VerifiedAtom
    va = VerifiedAtom(
        atom=Atom(atom_id="a", type="RATE", text="4.5%", claim="4.5%"),
        v1_status="MATCH", v2_entail_prob=0.9,
    )
    ans = GeneratedAnswer(query="q", answer_text="4.5%", citations=[], token_logprobs=[0.0], metadata={"no_context_answer": "x"})
    rr = RetrievalResult(query="q", query_date="2026-01-01", chunks=[], metadata={})
    with pytest.raises(StrictFailure):
        score([va], ans, rr, model_path=str(tmp_path / "missing.pkl"), strict=True)


def test_ltt_certificate_not_a_bare_none_on_tiny_data():
    risk = np.linspace(0.1, 0.9, 10)
    wrong = np.zeros(10)
    cert = certify_lambda(risk, wrong, eps=0.05, delta=0.1, n_start=50, use_hb=False)
    assert cert.status == "not_certifiable"
    assert cert.threshold is None
    assert cert.valid is False


def test_ltt_holds_and_fails():
    rng = np.random.default_rng(0)
    risk = rng.uniform(0, 1, 400)
    # Holds: the lowest-risk atoms are all correct, and n_start meets
    # min_accepted so a zero-error prefix can pass the binomial tail.
    wrong_ok = (risk > 0.55).astype(float)
    good = certify_lambda(risk, wrong_ok, eps=0.05, delta=0.1, n_start=45, n_grid=80, use_hb=False)
    assert good.status == "certified"
    assert good.valid
    assert good.coverage > 0

    # Fails: every atom is wrong, so no lambda can certify.
    bad = certify_lambda(risk, np.ones(400), eps=0.05, delta=0.1, n_start=20, n_grid=40, use_hb=False)
    assert bad.status == "not_certifiable"
    assert bad.valid is False


def test_zero_coverage_is_invalid():
    risk = np.array([0.2, 0.4, 0.6, 0.8])
    wrong = np.array([1, 1, 1, 1], dtype=float)
    sim = simulate_violation_rate(
        risk, wrong, eps=0.05, delta=0.1, n_splits=20, n_start=50, n_grid=10, use_hb=False,
    )
    assert sim["valid"] is False
    assert math.isnan(sim["violation_rate"])


def test_clopper_pearson_contains_rate():
    lo, hi = clopper_pearson(5, 100)
    assert lo < 0.05 < hi


def test_untouched_test_is_separate_from_calibration():
    rng = np.random.default_rng(1)
    risk = np.sort(rng.uniform(0, 1, 300))
    wrong = (risk > 0.7).astype(float)
    out = evaluate_untouched_test(
        risk[:200], wrong[:200], risk[200:], wrong[200:],
        eps=0.05, delta=0.1, n_start=15, n_grid=30, use_hb=False,
    )
    assert "certificate" in out
    assert out["certificate"]["status"] in {"certified", "merged", "not_certifiable"}


def test_drift_strategies_differ_on_synthetic_drift():
    from experiments.drift.drift_exp import run_drift_experiment
    # Early pre period is correct and low risk. Recent pre and all post are wrong.
    risks_pre = [0.1] * 120 + [0.9] * 40
    labels_pre = ["supported"] * 120 + ["unsupported"] * 40
    dates_pre = ["2022-01-01"] * 120 + ["2022-11-01"] * 40
    risks_post = [0.95] * 160
    labels_post = ["unsupported"] * 160
    dates_post = ["2023-06-01"] * 160
    result = run_drift_experiment(
        risks_pre, labels_pre, risks_post, labels_post,
        dates_pre=dates_pre, dates_post=dates_post,
        cutoff_date="2023-01-01", eps=0.05, delta=0.10,
        window_months=6, seed=0,
    )
    thresholds = {r.strategy: r.threshold for r in result.results}
    assert len(set(thresholds.values())) >= 2
    assert thresholds["static"] != thresholds["sliding_window"]


def test_split_refuses_unlabeled(tmp_path):
    from bench.splits import split_labeled_atoms
    rows = [{"doc_id": f"d{i}", "label": None, "family": "rate"} for i in range(6)]
    with pytest.raises(ValueError):
        split_labeled_atoms(rows, out_dir=str(tmp_path))


def test_split_leakage_check(tmp_path):
    from bench.splits import document_leakage, split_labeled_atoms
    rows = []
    for i in range(9):
        rows.append({
            "doc_id": f"doc{i}",
            "label": "supported" if i % 2 == 0 else "unsupported",
            "family": "rate",
            "question_id": f"q{i}",
        })
    splits = split_labeled_atoms(rows, out_dir=str(tmp_path))
    report = json.loads((tmp_path / "leakage_report.json").read_text(encoding="utf-8"))
    assert report["leaked"] == []
    owners = {}
    for name in ("agg_train", "calibration", "test"):
        for row in splits[name]:
            doc = row["doc_id"]
            assert doc not in owners
            owners[doc] = name
    leaked = document_leakage({
        "agg_train": [{"doc_id": "same"}],
        "test": [{"doc_id": "same"}],
    })
    assert leaked["leaked"]


def test_train_refuses_missing_labels(tmp_path):
    import subprocess, sys
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps({"doc_id": "d", "text": "x"}) + "\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "aggregate.train", "--jsonl", str(path), "--out", str(tmp_path / "m")],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True, text=True,
    )
    assert proc.returncode != 0


def test_baselines_have_distinct_aurc():
    from eval.systems import system_row
    cal, test = [], []
    for i in range(30):
        row = {
            "label": "unsupported" if i % 5 == 0 else "supported",
            "risk": 0.1 + (i % 5) * 0.15,
            "s_ent": 0.2 if i % 2 == 0 else 0.8,
            "s_nli": 0.1 if i % 3 else 0.9,
            "s_ver": 0.0 if i % 4 else 1.0,
            "s_ret": 0.7,
            "s_div": 0.3 + (i % 7) * 0.1,
        }
        (cal if i < 15 else test).append(row)
    names = ["RegGuard (ours)", "plain_rag", "selfcheck", "nli_only", "llm_judge_all", "conformal_no_mondrian"]
    aurcs = []
    for name in names:
        row = system_row(name, cal, test, eps=0.2)
        if row.get("aurc") is not None:
            aurcs.append(round(row["aurc"], 6))
    assert len(set(aurcs)) == len(aurcs)


def test_check_publishable_fails_on_empty(tmp_path):
    from eval.check_publishable import check
    failures = check(str(tmp_path), str(tmp_path))
    assert failures


def test_mcnemar_and_bootstrap():
    from eval.stats import mcnemar, paired_bootstrap
    a = [True] * 20 + [False] * 5
    b = [True] * 10 + [False] * 15
    stats = mcnemar(a, b)
    assert stats["p_value"] < 0.05
    boot = paired_bootstrap([float(x) for x in a], [float(x) for x in b], n_resamples=200, seed=1)
    assert boot["ci_low"] <= boot["mean_diff"] <= boot["ci_high"]
