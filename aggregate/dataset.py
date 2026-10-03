"""
aggregate/dataset.py
---------------------
Build the feature table for the aggregator: one row per atom, with all
6 signal columns, atom-type one-hot, regulator one-hot, and a binary label
`wrong = 1` (atom is wrong/unsupported/outdated).

Input format (JSONL — one record per labeled atom):
    {
        "atom_id":        str,
        "type":           str,           # RATE | THRESHOLD | DATE | SECTION | ENTITY | APPLICABILITY
        "regulator":      str,           # SEBI | RBI | INCOMETAX | MF
        "v1_status":      str,           # MATCH | MISMATCH | NOT_FOUND | NA
        "v2_entail_prob": float | null,
        "rerank_score":   float | null,  # from retrieval metadata
        "rrf_rank":       int   | null,  # fallback if rerank absent
        "token_logprobs":   list[float] | null,
        "no_context_answer": str | null,
        "atom_span":      [int, int] | null,
        "answer_text":    str,
        "label":          int             # 0 = correct, 1 = wrong/unsupported
    }

Usage:
    python -m aggregate.dataset \\
        --atoms_dir data/agg_train/ \\
        --out       data/processed/feature_table.parquet
"""
from __future__ import annotations

import argparse
import json
import math
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd  # type: ignore

from signals.s_ver import s_ver_features
from signals.s_nli import s_nli_features
from signals.s_ent import s_ent_features
from common.schemas import VerifiedAtom, Atom, GeneratedAnswer

logger = logging.getLogger("aggregate.dataset")
logging.basicConfig(level=logging.INFO)

# Canonical atom types and regulators (for one-hot encoding)
ATOM_TYPES = ["RATE", "THRESHOLD", "DATE", "SECTION", "ENTITY", "APPLICABILITY"]
REGULATORS = ["SEBI", "RBI", "INCOMETAX", "MF", "OTHER"]


# ---------------------------------------------------------------------------
# Row → feature dict
# ---------------------------------------------------------------------------

def row_to_features(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert one labeled-atom row to a flat feature dict.
    Missing values are handled with sensible defaults (0.5 for signals).
    """
    # Re-construct lightweight schema objects for signal functions
    atom = Atom(
        atom_id=row.get("atom_id", ""),
        type=row.get("type", ""),
        text=row.get("text", ""),
        claim=row.get("claim", ""),
        cited_chunk=row.get("cited_chunk"),
        span=tuple(row["atom_span"]) if row.get("atom_span") else None,
    )
    va = VerifiedAtom(
        atom=atom,
        v1_status=row.get("v1_status", "NOT_FOUND"),
        v2_entail_prob=row.get("v2_entail_prob"),
    )

    # Lightweight GeneratedAnswer just for s_ent
    answer = GeneratedAnswer(
        query=row.get("query", ""),
        answer_text=row.get("answer_text", ""),
        citations=[],
        token_logprobs=row.get("token_logprobs"),
        metadata={"no_context_answer": row.get("no_context_answer", "")},
    )

    # --- Signals ---
    feats: Dict[str, Any] = {}

    # s_ver (pure)
    feats.update(s_ver_features(va))

    # s_nli
    if "s_nli" in row and row["s_nli"] is not None:
        feats["s_nli"] = float(row["s_nli"])
        feats["nli_skipped"] = int(row.get("nli_skipped", 0) if "nli_skipped" in row else math.isnan(row["s_nli"]))
    else:
        feats.update(s_nli_features(va))

    # s_ent
    if "s_ent" in row and row["s_ent"] is not None:
        feats["s_ent"] = float(row["s_ent"])
        feats["s_ent_max"] = float(row.get("s_ent_max", row["s_ent"]))
        feats["s_ent_skipped"] = int(row.get("s_ent_skipped", 0) if "s_ent_skipped" in row else math.isnan(row["s_ent"]))
    else:
        feats.update(s_ent_features(va, answer))

    # s_ret (needs rerank_score or rrf_rank). Missing evidence is an
    # indicator, not the constant 0.5.
    rerank_score = row.get("rerank_score")
    rrf_rank     = row.get("rrf_rank")
    n_chunks     = row.get("n_chunks", 6)
    
    if "s_ret" in row and row["s_ret"] is not None:
        feats["s_ret"] = float(row["s_ret"])
        feats["s_ret_skipped"] = 0
    elif rerank_score is not None:
        feats["s_ret"] = float(rerank_score)
        feats["s_ret_skipped"] = 0
    elif rrf_rank is not None:
        feats["s_ret"] = round(1.0 - float(rrf_rank) / max(1, n_chunks), 4)
        feats["s_ret_skipped"] = 0
    else:
        feats["s_ret"] = float("nan")
        feats["s_ret_skipped"] = 1

    # s_div: use the stored value, or mark missing.
    if row.get("s_div") is None:
        feats["s_div"] = float("nan")
        feats["s_div_skipped"] = 1
    else:
        feats["s_div"] = float(row["s_div"])
        feats["s_div_skipped"] = int(row.get("s_div_skipped", 0))

    # --- Atom type one-hot ---
    atom_type = row.get("type", "")
    for t in ATOM_TYPES:
        feats[f"type_{t}"] = int(atom_type == t)

    # --- Regulator one-hot ---
    reg = row.get("regulator", "OTHER").upper()
    if reg not in REGULATORS:
        reg = "OTHER"
    for r in REGULATORS:
        feats[f"reg_{r}"] = int(reg == r)

    # --- Metadata ---
    feats["atom_id"]   = row.get("atom_id", "")
    feats["atom_type"] = atom_type
    feats["regulator"] = row.get("regulator", "")
    feats["label"]     = int(row.get("label", 0))

    return feats


# ---------------------------------------------------------------------------
# Build full feature table from a directory of JSONL files
# ---------------------------------------------------------------------------

def build_feature_table(
    atoms_dir: str,
    out_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Walk `atoms_dir` for all *.jsonl files, build a feature row per atom,
    and return a DataFrame.  Optionally save as Parquet.
    """
    records = []
    p = Path(atoms_dir)
    jsonl_files = list(p.rglob("*.jsonl"))
    logger.info("Found %d JSONL files in %s", len(jsonl_files), atoms_dir)

    for fpath in jsonl_files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    feats = row_to_features(row)
                    records.append(feats)
                except Exception as exc:
                    logger.warning("Skipping row in %s: %s", fpath.name, exc)

    df = pd.DataFrame(records)
    logger.info("Feature table: %d rows × %d cols", len(df), len(df.columns))

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        logger.info("Saved feature table to %s", out_path)

    return df


# ---------------------------------------------------------------------------
# Column order used by the aggregator (must match training)
# ---------------------------------------------------------------------------

SIGNAL_COLS = [
    "s_div", "s_div_skipped",
    "s_ret", "s_ret_skipped",
    "s_ver",
    "s_nli", "nli_skipped",
    "s_ent", "s_ent_max", "s_ent_skipped",
    "v1_match", "v1_mismatch", "v1_not_found", "v1_na",
]

TYPE_COLS = [f"type_{t}" for t in ATOM_TYPES]
REG_COLS  = [f"reg_{r}" for r in REGULATORS]

FEATURE_COLS = SIGNAL_COLS + TYPE_COLS + REG_COLS


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build M6 feature table from labeled atoms.")
    parser.add_argument("--atoms_dir", required=True)
    parser.add_argument("--out",       required=True)
    args = parser.parse_args()
    build_feature_table(args.atoms_dir, args.out)
