"""
scripts/draft_questions.py
--------------------------
Phase C: Draft candidate questions from the BM25 chunk index.

Family mix (from experiment.yaml question_families, must sum to 1.0):
  RATE_THRESHOLD:       0.30  – numeric rates, limits, thresholds, ratios
  SECTION:              0.20  – which section/clause/provision governs X
  DATE_TIMELINE:        0.15  – specific dates, deadlines, transitions
  ENTITY_APPLICABILITY: 0.25  – which entities/persons/transactions are subject
  NOT_ANSWERABLE:       0.10  – plausible question NOT answerable from corpus

Every question gets query_date = config.snapshot_date (2026-10-03).
Superseded/date-dependent family REMOVED.

Chunks are drawn only from train-split documents (data/splits/doc_assignment.csv).
Excluded documents (data/excluded_docs.csv) are never sampled.

Output: data/questions_draft.csv (or --out PATH)

Usage:
    python scripts/draft_questions.py [--n N] [--out PATH] [--seed N]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import pickle
import random
import sys
from pathlib import Path
from typing import Optional

import yaml

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("draft_questions")

ROOT = Path(__file__).resolve().parent.parent

# ── Family system prompts ─────────────────────────────────────────────────────

_FAMILY_PROMPTS: dict[str, str] = {
    "RATE_THRESHOLD": (
        "You are a regulatory compliance analyst. "
        "Given the regulatory passage below, write ONE question that asks about a "
        "specific numeric rate, threshold, limit, ratio, or percentage EXPLICITLY "
        "stated in the passage. The question must be answerable solely from this passage. "
        "Output ONLY a JSON object: "
        '{\"question\": \"...\", \"family\": \"RATE_THRESHOLD\", \"regulator\": \"RBI or SEBI\"}'
    ),
    "SECTION": (
        "You are a regulatory compliance analyst. "
        "Given the regulatory passage below, write ONE question that asks which specific "
        "section, clause, provision, regulation, or direction governs a particular "
        "activity or requirement described in the passage. "
        "Output ONLY a JSON object: "
        '{\"question\": \"...\", \"family\": \"SECTION\", \"regulator\": \"RBI or SEBI\"}'
    ),
    "DATE_TIMELINE": (
        "You are a regulatory compliance analyst. "
        "Given the regulatory passage below, write ONE question about a specific date, "
        "deadline, implementation period, or transitional provision mentioned in the "
        "passage. The question must be answerable solely from this passage. "
        "Output ONLY a JSON object: "
        '{\"question\": \"...\", \"family\": \"DATE_TIMELINE\", \"regulator\": \"RBI or SEBI\"}'
    ),
    "ENTITY_APPLICABILITY": (
        "You are a regulatory compliance analyst. "
        "Given the regulatory passage below, write ONE question about which category of "
        "entity, institution, person, or transaction is subject to (or exempt from) a "
        "specific requirement described in the passage. "
        "Output ONLY a JSON object: "
        '{\"question\": \"...\", \"family\": \"ENTITY_APPLICABILITY\", \"regulator\": \"RBI or SEBI\"}'
    ),
    "NOT_ANSWERABLE": (
        "You are a regulatory compliance analyst. "
        "Given the regulatory passage below, write ONE plausible compliance question "
        "that a regulated entity might ask about Indian financial regulation, but which "
        "is NOT explicitly answered by this passage (the answer requires information "
        "absent from the text). The question should sound realistic and specific. "
        "Output ONLY a JSON object: "
        '{\"question\": \"...\", \"family\": \"NOT_ANSWERABLE\", \"regulator\": \"RBI or SEBI\"}'
    ),
}

# ── Template fallback (when LLM is unavailable) ───────────────────────────────

_FAMILY_TEMPLATES: dict[str, str] = {
    "RATE_THRESHOLD":       "What is the specific rate or threshold mentioned in this {regulator} circular regarding {topic}?",
    "SECTION":              "Under which section or provision does {regulator} regulate {topic}?",
    "DATE_TIMELINE":        "What is the effective date or deadline specified in this {regulator} circular for {topic}?",
    "ENTITY_APPLICABILITY": "Which categories of entities are required to comply with the {regulator} rules on {topic}?",
    "NOT_ANSWERABLE":       "What are the penalties for non-compliance with the {regulator} rules on {topic}?",
}


def _template_question(family: str, regulator: str, chunk_text: str) -> str:
    """Generate a fallback question from a template when LLM is unavailable."""
    # Extract a brief topic from the chunk text (first 80 chars, stripped)
    topic_raw = chunk_text.strip()[:80].replace("\n", " ")
    # Truncate at sentence boundary
    for stop in (".", ";", ":"):
        idx = topic_raw.find(stop)
        if 15 < idx < 60:
            topic_raw = topic_raw[:idx]
            break
    topic = topic_raw.strip(" .,;:")[:50] or "the stated requirement"
    return _FAMILY_TEMPLATES[family].format(regulator=regulator, topic=topic)


# ── Config / data loading ─────────────────────────────────────────────────────

def _load_config() -> dict:
    cfg_path = ROOT / "configs" / "experiment.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_train_doc_ids(cfg: dict) -> set:
    """Return set of doc_ids assigned to the train split."""
    paths   = cfg.get("paths", {})
    split_path = ROOT / paths.get("doc_splits", "data/splits/doc_assignment.csv")
    if not split_path.exists():
        logger.warning("Split file not found: %s", split_path)
        return set()
    train_ids: set[str] = set()
    with open(split_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("split", "").strip() == "train":
                train_ids.add(row["doc_id"].strip())
    logger.info("Loaded %d train-split doc_ids", len(train_ids))
    return train_ids


def _load_doc_regulator(cfg: dict) -> dict:
    """Return {doc_id: regulator} from inventory."""
    paths = cfg.get("paths", {})
    inv_path = ROOT / paths.get("doc_inventory", "data/doc_inventory.csv")
    result: dict[str, str] = {}
    if not inv_path.exists():
        return result
    with open(inv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            result[row["doc_id"]] = row.get("regulator", "")
    return result


class _ChunkStub:
    """Minimal duck-type replacement for RegulatoryChunk when rag_module is absent."""
    __slots__ = ("text", "chunk_id", "doc_id", "metadata", "breadcrumb_text")

    def __init__(self, **kw: object) -> None:
        for s in self.__slots__:
            setattr(self, s, kw.get(s, ""))

    def __setstate__(self, state: dict) -> None:
        for s in self.__slots__:
            setattr(self, s, state.get(s, ""))


class _StubUnpickler(pickle.Unpickler):
    """Replace any unknown class with _ChunkStub so the index can be loaded."""
    def find_class(self, module: str, name: str):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            logger.debug("Replacing missing class %s.%s with _ChunkStub", module, name)
            return _ChunkStub


def _load_bm25_chunks(cfg: dict) -> list:
    """Load chunks from .bm25_index.pkl, restricted to train-split docs."""
    index_path = ROOT / ".bm25_index.pkl"
    if not index_path.exists():
        logger.warning("BM25 index not found: %s", index_path)
        return []
    logger.info("Loading BM25 index from %s ...", index_path)
    with open(index_path, "rb") as f:
        raw = _StubUnpickler(f).load()
    all_chunks = raw.get("chunks", [])
    logger.info("BM25 index: %d total chunks", len(all_chunks))
    return all_chunks


def _get_regulator(chunk, doc_regulator: dict) -> str:
    reg = (getattr(chunk, "metadata", {}) or {}).get("issuer", "")
    if not reg:
        reg = doc_regulator.get(getattr(chunk, "doc_id", ""), "")
    if not reg:
        reg = "RBI"   # default
    return reg.upper()


# ── LLM client wrapper ────────────────────────────────────────────────────────

def _try_llm(llm, family: str, chunk_text: str) -> Optional[dict]:
    """Call the LLM and parse JSON. Returns dict or None on failure."""
    system = _FAMILY_PROMPTS[family]
    prompt = f"PASSAGE:\n{chunk_text[:1500]}"
    try:
        resp = llm.chat(user=prompt, system=system,
                        response_format={"type": "json_object"})
        data = json.loads(resp["text"])
        if data.get("question"):
            return data
    except Exception as exc:
        logger.debug("LLM call failed: %s", exc)
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def draft_questions(
    n_total: int,
    out_path: Path,
    seed: int,
    cfg: dict,
) -> list[dict]:
    snapshot_date = cfg.get("snapshot_date", "2026-10-03")

    # Family mix from config
    fam_weights_raw: dict[str, float] = cfg.get("question_families", {
        "RATE_THRESHOLD":       0.30,
        "SECTION":              0.20,
        "DATE_TIMELINE":        0.15,
        "ENTITY_APPLICABILITY": 0.25,
        "NOT_ANSWERABLE":       0.10,
    })
    # Normalise weights just in case
    total_w = sum(fam_weights_raw.values())
    fam_weights = {k: v / total_w for k, v in fam_weights_raw.items()}

    # Compute per-family quotas (ceil so we meet n_total minimum)
    quotas: dict[str, int] = {}
    assigned = 0
    families = list(fam_weights.keys())
    for i, fam in enumerate(families):
        if i < len(families) - 1:
            q = math.ceil(n_total * fam_weights[fam])
        else:
            q = max(0, n_total - assigned)
        quotas[fam] = q
        assigned += q
    logger.info("Family quotas: %s  (total=%d)", quotas, sum(quotas.values()))

    # Load train doc_ids and doc→regulator map
    train_ids    = _load_train_doc_ids(cfg)
    doc_regulator = _load_doc_regulator(cfg)

    # Load chunks and filter to train docs
    all_chunks = _load_bm25_chunks(cfg)
    train_chunks = [
        c for c in all_chunks
        if getattr(c, "doc_id", None) in train_ids
        and len(getattr(c, "text", "") or "") >= 100
    ]
    logger.info("Filtered to %d train-split chunks", len(train_chunks))

    if not train_chunks:
        logger.error("No train-split chunks found. Aborting.")
        sys.exit(1)

    # Try to load LLM client (skip if --no-llm or unavailable)
    llm = None
    use_llm: bool = cfg.get("_use_llm", True)
    if use_llm:
        try:
            sys.path.insert(0, str(ROOT))
            from common.llm_client import LLMClient
            llm = LLMClient()
            logger.info("LLM client loaded")
        except Exception as exc:
            logger.warning("LLM client unavailable (%s). Using template fallback.", exc)

    rng = random.Random(seed)
    results: list[dict] = []
    qid_counter = 1

    for family, quota in quotas.items():
        logger.info("Drafting %d questions for family %s ...", quota, family)
        # Sample chunks for this family (with replacement if corpus smaller than quota)
        if len(train_chunks) >= quota:
            samples = rng.sample(train_chunks, quota)
        else:
            samples = [rng.choice(train_chunks) for _ in range(quota)]

        for chunk in samples:
            text      = (getattr(chunk, "text", "") or "").strip()
            doc_id    = getattr(chunk, "doc_id", "")
            chunk_id  = getattr(chunk, "chunk_id", "")
            regulator = _get_regulator(chunk, doc_regulator)

            question = ""
            # Try LLM
            if llm:
                data = _try_llm(llm, family, text)
                if data:
                    question  = data.get("question", "")
                    regulator = data.get("regulator", regulator).upper()

            # Fallback to template
            if not question:
                question = _template_question(family, regulator, text)

            results.append({
                "question_id":      f"Q{qid_counter:05d}",
                "status":           "DRAFT - needs human review",
                "question":         question,
                "family":           family,
                "answerable":       "no" if family == "NOT_ANSWERABLE" else "yes",
                "regulator":        regulator,
                "query_date":       snapshot_date,
                "source_doc_id":    doc_id,
                "seed_chunk_id":    chunk_id,
                "seed_chunk_preview": text[:180].replace("\n", " "),
            })
            qid_counter += 1

    # Shuffle final list so families are interleaved
    rng.shuffle(results)
    # Re-number after shuffle
    for i, r in enumerate(results, 1):
        r["question_id"] = f"Q{i:05d}"

    # Write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question_id", "status", "question", "family", "answerable",
        "regulator", "query_date", "source_doc_id", "seed_chunk_id",
        "seed_chunk_preview",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logger.info("Wrote %d questions to %s", len(results), out_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    from collections import Counter
    fam_cnt = Counter(r["family"] for r in results)
    reg_cnt = Counter(r["regulator"] for r in results)

    print(f"\n{'='*60}")
    print(f"  Draft Questions Summary")
    print(f"{'='*60}")
    print(f"  Total drafted : {len(results)}")
    print(f"  query_date    : {snapshot_date}")
    print(f"  llm_used      : {'yes' if llm else 'no (template fallback)'}")
    print(f"\n  By family:")
    for fam in families:
        print(f"    {fam:<22}: {fam_cnt.get(fam, 0):>4}")
    print(f"\n  By regulator:")
    for reg, cnt in sorted(reg_cnt.items(), key=lambda x: -x[1]):
        print(f"    {reg:<8}: {cnt:>4}")
    print(f"\n  Output: {out_path}")
    print(f"{'='*60}\n")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = _load_config()
    paths = cfg.get("paths", {})
    default_out = str(ROOT / "data" / "questions_draft.csv")

    parser = argparse.ArgumentParser(description="Draft candidate questions (Phase C).")
    parser.add_argument("--n",    type=int, default=200,
                        help="Total questions to draft (default 200)")
    parser.add_argument("--out",  default=default_out,
                        help="Output CSV path")
    parser.add_argument("--seed", type=int, default=int(cfg.get("seed", 42)),
                        help="Random seed")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM calls; use template questions only (fast).")
    args = parser.parse_args()

    if args.no_llm:
        cfg["_use_llm"] = False

    draft_questions(
        n_total=args.n,
        out_path=Path(args.out),
        seed=args.seed,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
