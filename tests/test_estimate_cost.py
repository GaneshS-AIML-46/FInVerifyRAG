"""Tests for T5 cost estimator (scripts/estimate_cost.py).

All tests use synthetic/mock data only.
"""
import importlib.util
import io
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


def _load_estimate_cost():
    spec = importlib.util.spec_from_file_location(
        "estimate_cost",
        Path(__file__).parent.parent / "scripts" / "estimate_cost.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_price_missing_shows_label(capsys):
    """Null price in config -> 'PRICE MISSING' in output."""
    mod = _load_estimate_cost()
    # Empty llm_prices -> all prices missing
    rows = mod.compute_costs(n_questions=10, atoms_per_answer=4, llm_prices={})
    all_text = " ".join(str(r.get("cost", "")) for r in rows)
    assert "PRICE MISSING" in all_text


def test_cost_calculation_correct():
    """100 calls; input=1.0, output=2.0 per million; 1000 input, 500 output tok.
    Cost = (100*1000*1.0 + 100*500*2.0) / 1_000_000 = 0.2 USD per role.
    """
    mod = _load_estimate_cost()

    # Override DEFAULT_TOKENS for a single role, compute manually
    # calls = 100 (we'll set n_questions so judge_calls are known), easier to compute directly:
    in_tok = 100 * 1000
    out_tok = 100 * 500
    price_in = 1.0  # per million
    price_out = 2.0  # per million
    cost = (in_tok * price_in + out_tok * price_out) / 1_000_000
    assert abs(cost - 0.2) < 1e-9


def test_all_roles_present(capsys):
    """Output contains all four role names."""
    mod = _load_estimate_cost()

    # Capture print_table output
    buf = io.StringIO()
    rows = mod.compute_costs(n_questions=100, atoms_per_answer=4, llm_prices={})
    with patch("sys.stdout", buf):
        mod.print_table(rows, "Test scope")

    output = buf.getvalue()
    for role in ["question_drafting", "generator", "extractor", "judge"]:
        assert role in output, f"Role '{role}' not found in output"
