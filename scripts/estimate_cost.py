"""Estimate LLM cost for the pipeline.

Usage:
    python scripts/estimate_cost.py [--n-questions N] [--atoms-per-answer N]

Reads llm_prices from experiment.yaml (if present).
"""
import argparse
import sys
from pathlib import Path

import yaml


# Default token estimates per role per call
DEFAULT_TOKENS = {
    "question_drafting": {"input": 2000, "output": 300},
    "generator":         {"input": 3000, "output": 800},
    "extractor":         {"input": 1500, "output": 400},
    "judge":             {"input": 1000, "output": 200},
}

SELFCHECK_SAMPLES = 3  # judge called 3x per answer for SelfCheck


def load_config() -> dict:
    cfg_path = Path("configs/experiment.yaml")
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def compute_costs(n_questions: int, atoms_per_answer: int, llm_prices: dict) -> list:
    """Return list of row dicts for display."""
    n_atoms = n_questions * atoms_per_answer

    # Number of calls per scope
    # question_drafting: once per question
    # generator: once per question
    # extractor: once per question (or atom; use question as proxy)
    # judge: selfcheck_samples per answer + 1 baseline per question
    role_calls = {
        "question_drafting": n_questions,
        "generator":         n_questions,
        "extractor":         n_questions,
        "judge":             n_atoms * SELFCHECK_SAMPLES + n_questions,  # selfcheck + baseline
    }

    rows = []
    total_cost = 0.0
    total_cost_missing = False

    for role, tokens in DEFAULT_TOKENS.items():
        calls = role_calls[role]
        in_tok = tokens["input"] * calls
        out_tok = tokens["output"] * calls

        prices = llm_prices.get(role, {}) if isinstance(llm_prices, dict) else {}
        price_in = prices.get("input_per_million") if prices else None
        price_out = prices.get("output_per_million") if prices else None

        if price_in is None or price_out is None:
            cost_str = "PRICE MISSING"
            total_cost_missing = True
        else:
            cost = (in_tok * price_in + out_tok * price_out) / 1_000_000
            cost_str = f"${cost:.4f}"
            total_cost += cost

        rows.append({
            "role": role,
            "calls": calls,
            "input_tok": in_tok,
            "output_tok": out_tok,
            "cost": cost_str,
        })

    # TOTAL row
    total_str = "PRICE MISSING (some)" if total_cost_missing else f"${total_cost:.4f}"
    rows.append({
        "role": "TOTAL",
        "calls": sum(r["calls"] for r in rows[:-0] if isinstance(r["calls"], int)),
        "input_tok": sum(r["input_tok"] for r in rows if isinstance(r.get("input_tok"), int)),
        "output_tok": sum(r["output_tok"] for r in rows if isinstance(r.get("output_tok"), int)),
        "cost": total_str,
    })

    return rows


def print_table(rows: list, scope_label: str) -> None:
    print(f"\n{'='*70}")
    print(f"Scope: {scope_label}")
    print(f"{'='*70}")
    header = f"{'Role':<22} {'Calls':>10} {'Input tok':>14} {'Output tok':>14} {'Cost USD':>14}"
    print(header)
    print("-" * 70)
    for row in rows:
        calls_str = str(row["calls"]) if isinstance(row["calls"], int) else str(row["calls"])
        in_str = f"{row['input_tok']:,}" if isinstance(row.get("input_tok"), int) else str(row.get("input_tok",""))
        out_str = f"{row['output_tok']:,}" if isinstance(row.get("output_tok"), int) else str(row.get("output_tok",""))
        print(f"{row['role']:<22} {calls_str:>10} {in_str:>14} {out_str:>14} {row['cost']:>14}")


def main():
    parser = argparse.ArgumentParser(description="Estimate LLM pipeline cost")
    parser.add_argument("--n-questions", type=int, default=400)
    parser.add_argument("--atoms-per-answer", type=int, default=4)
    # Allow custom price overrides (for testing)
    parser.add_argument("--price-input", type=float, default=None, help="Override input price per million tokens (all roles)")
    parser.add_argument("--price-output", type=float, default=None, help="Override output price per million tokens (all roles)")
    args = parser.parse_args()

    cfg = load_config()
    llm_prices = cfg.get("llm_prices", {})

    # If price overrides provided, apply to all roles
    if args.price_input is not None or args.price_output is not None:
        for role in DEFAULT_TOKENS:
            if role not in llm_prices:
                llm_prices[role] = {}
            if args.price_input is not None:
                llm_prices[role]["input_per_million"] = args.price_input
            if args.price_output is not None:
                llm_prices[role]["output_per_million"] = args.price_output

    n_q = args.n_questions
    atoms_per = args.atoms_per_answer

    scopes = [
        ("Question drafting only", 50, atoms_per),
        ("Pilot (50 questions)", 50, atoms_per),
        (f"Full run ({n_q} questions)", n_q, atoms_per),
    ]

    for label, nq, na in scopes:
        rows = compute_costs(nq, na, llm_prices)
        print_table(rows, f"{label}")

    print()


if __name__ == "__main__":
    main()
