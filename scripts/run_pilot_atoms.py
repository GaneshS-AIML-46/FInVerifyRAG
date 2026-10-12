"""
scripts/run_pilot_atoms.py
--------------------------
Run the full M2→M8 pipeline on data/questions_pilot_review.csv using:
  M2  – BM25 search directly on .bm25_index.pkl (no dense index needed)
  M3  – Ollama qwen2.5:7b-instruct-q4_K_M via openai-compatible API
  M4  – Regex fast-pass + LLM extractor (Ollama)
  M5  – V1 deterministic + V2 zero-shot DeBERTa NLI cascade
  M6  – Heuristic aggregator (real model is a stub; uses signal averages)
  M7  – Loads thresholds from models/guarantee/thresholds.json
  M8  – Decision router (no judge calls in pilot: UNCERTAIN → ABSTAINED)

Output : results/raw_atoms.jsonl  (one JSON per atom, appended)
         results/cost.jsonl        (one JSON per question, cost tracked)

No LLM calls beyond Ollama (local, free). No API keys consumed.
"""
from __future__ import annotations

import json
import logging
import math
import os
import pickle
import re
import sys
import time
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_pilot_atoms")

# ── Config ────────────────────────────────────────────────────────────────────
with open(ROOT / "configs" / "experiment.yaml", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

QUERY_DATE = CFG.get("snapshot_date", date.today().isoformat())
QUESTIONS_PATH = ROOT / "data" / "questions_pilot_review.csv"
BM25_PATH = ROOT / ".bm25_index.pkl"
ATOMS_OUT = ROOT / "results" / "raw_atoms.jsonl"
COST_OUT = ROOT / "results" / "cost.jsonl"
CHECKPOINT = ROOT / "results" / "pilot_atoms_checkpoint.jsonl"

# ── Ollama LLM config (no API key needed) ────────────────────────────────────
# vLLM served via WSL Ubuntu-24.04 on port 8000 (OpenAI-compatible API)
# Model: Qwen/Qwen2.5-7B-Instruct-AWQ (AWQ quantized, ~4.8GB VRAM)
# Falls back to Ollama on port 11434 if vLLM is not reachable
import urllib.request

def _vllm_available(port: int = 8000) -> bool:
    try:
        urllib.request.urlopen(f"http://localhost:{port}/health", timeout=3)
        return True
    except Exception:
        return False

_USE_VLLM = _vllm_available(8000)
_BACKEND = "vLLM:8000" if _USE_VLLM else "Ollama:11434"
_MODEL = "Qwen/Qwen2.5-7B-Instruct" if _USE_VLLM else "qwen2.5:7b-instruct-q4_K_M"
_BASE_URL = "http://localhost:8000/v1" if _USE_VLLM else "http://localhost:11434/v1"
logger.info("LLM backend: %s  model: %s", _BACKEND, _MODEL)

OLLAMA_CFG = {
    "roles": {
        "generator": {
            "provider": "ollama",  # openai-compatible — works for both Ollama and vLLM
            "model": _MODEL,
            "api_key_env": "LLM_API_KEY",
            "base_url": _BASE_URL,
            "max_tokens": 768 if _USE_VLLM else 1024,
            "temperature": 0.0,
        },
        "extractor": {
            "provider": "ollama",
            "model": _MODEL,
            "api_key_env": "LLM_API_KEY",
            "base_url": _BASE_URL,
            "max_tokens": 512,
            "temperature": 0.0,
        },
    },
    "llm_prices": {
        _MODEL: {"input": 0.0, "output": 0.0}  # local = free
    },
    "budget_caps": {},
}

# ── Load BM25 index ───────────────────────────────────────────────────────────
logger.info("Loading BM25 index from %s …", BM25_PATH)
_t = time.perf_counter()
with open(BM25_PATH, "rb") as f:
    _bm25_data = pickle.load(f)

_raw_chunks: list = _bm25_data["chunks"]           # RegulatoryChunk objects
_tokenized: list = _bm25_data["tokenized_corpus"]  # list of token lists

from rank_bm25 import BM25Okapi
_bm25 = BM25Okapi(_tokenized)

# Build chunk_id → raw_chunk lookup
_chunk_map: Dict[str, Any] = {c.chunk_id: c for c in _raw_chunks}
logger.info("BM25 loaded: %d chunks  (%.1fs)", len(_raw_chunks), time.perf_counter() - _t)


# ── Adapt RegulatoryChunk → common.schemas.Chunk ─────────────────────────────
from common.schemas import Chunk, RetrievalResult, GeneratedAnswer

def _to_schema_chunk(rc) -> Chunk:
    """Convert a RegulatoryChunk to a common.schemas.Chunk."""
    meta = rc.metadata or {}
    regulator = str(meta.get("issuer", rc.doc_id.split("_")[0]))
    # Normalise regulator to RBI / SEBI
    if "RBI" in regulator.upper() or "RESERVE" in regulator.upper():
        regulator = "RBI"
    elif "SEBI" in regulator.upper():
        regulator = "SEBI"
    return Chunk(
        chunk_id=rc.chunk_id,
        doc_id=rc.doc_id,
        text=rc.text,
        regulator=regulator,
        issue_date=str(meta.get("year", "2024")),
        source_url=str(meta.get("file_path", "")),
        metadata=meta,
    )


def bm25_retrieve(query: str, query_date: str, top_k: int = 6) -> RetrievalResult:
    """Simple BM25 retrieval returning common.schemas.RetrievalResult."""
    tokens = query.lower().split()
    scores = _bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k * 3]
    # Filter out very-low-scoring
    top_indices = [i for i in top_indices if scores[i] > 0][:top_k]
    chunks = [_to_schema_chunk(_raw_chunks[i]) for i in top_indices]
    return RetrievalResult(query=query, query_date=query_date, chunks=chunks)


# ── Build LLM clients ─────────────────────────────────────────────────────────
from common.llm_client import LLMClient

os.environ.setdefault("LLM_API_KEY", "ollama")  # Ollama ignores API key
gen_client = LLMClient(role="generator", cfg=OLLAMA_CFG, check_prices=False)
ext_client = LLMClient(role="extractor", cfg=OLLAMA_CFG, check_prices=False)


# ── M3: Generator ─────────────────────────────────────────────────────────────
from generation.generator import Generator
_generator = Generator(llm_client=gen_client)


# ── M4: Atom extractor ────────────────────────────────────────────────────────
from atoms.splitter import AtomSplitter
_splitter = AtomSplitter(llm=ext_client)

def extract_atoms(answer: GeneratedAnswer) -> list:
    try:
        return _splitter.split(answer)
    except Exception as e:
        logger.warning("Atom extraction failed: %s", e)
        return []


# ── M5: Verification cascade ──────────────────────────────────────────────────
try:
    from verify.cascade import verify as cascade_verify
    logger.info("M5 cascade loaded")
except Exception as e:
    logger.warning("M5 cascade unavailable (%s); using stub", e)
    def cascade_verify(atoms, rr, **_kw):
        from common.schemas import VerifiedAtom
        return [VerifiedAtom(atom=a, v1_status="NA") for a in atoms]


# ── M6: Aggregator ────────────────────────────────────────────────────────────
try:
    from aggregate.model import score as agg_score
    logger.info("M6 aggregator loaded")
except Exception as e:
    logger.warning("M6 aggregator unavailable (%s); using heuristic", e)
    from common.schemas import ScoredAtom
    def agg_score(verified_atoms, answer, rr, **_kw):
        results = []
        for va in verified_atoms:
            # Heuristic: average of available signals
            s_nli = getattr(va, "v2_entail_prob", None)
            v1 = getattr(va, "v1_status", "NA")
            s_ver = {"MATCH": 0.0, "MISMATCH": 1.0, "NOT_FOUND": 0.5, "NA": 0.5}.get(v1, 0.5)
            risk = (1.0 - s_nli) * 0.5 + s_ver * 0.5 if s_nli is not None else s_ver
            results.append(ScoredAtom(verified=va, risk=round(risk, 4)))
        return results


# ── M7: Thresholds ────────────────────────────────────────────────────────────
try:
    from guarantee.certify import load_thresholds
    _thresholds = load_thresholds(str(ROOT / "models" / "guarantee" / "thresholds.json"))
except Exception as e:
    logger.warning("Thresholds unavailable (%s); UNCERTAIN band disabled", e)
    _thresholds = {}


# ── M8: Decision ──────────────────────────────────────────────────────────────
try:
    from decision.combine import decide as m8_decide
    from judge.judge_llm import JudgeLLM
    _judge = None   # skip judge for pilot
    logger.info("M8 decision loaded")
except Exception as e:
    logger.warning("M8 decision unavailable (%s)", e)
    m8_decide = None


def _route_atom(scored_atom, thresholds: dict) -> str:
    """Simple router without judge: risk <= 0.3 → SUPPORTED, else ABSTAINED."""
    risk = scored_atom.risk
    stratum = getattr(scored_atom.verified.atom, "atom_type", "*")
    th = thresholds.get(f"*|{stratum}", thresholds.get("*|*", None))
    accept_below = getattr(th, "accept_below", 0.3) if th else 0.3
    abstain_above = getattr(th, "abstain_above", 0.7) if th else 0.7
    if risk <= accept_below:
        return "SUPPORTED"
    elif risk >= abstain_above:
        return "ABSTAINED"
    return "UNCERTAIN"   # Would go to judge; for pilot we ABSTAIN


# ── Helpers ───────────────────────────────────────────────────────────────────
def _safe(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def _atom_to_dict(question_id: str, question: str, regulator: str,
                  scored_atom, decision: str, latency: dict) -> dict:
    va = scored_atom.verified
    a = va.atom
    return {
        "id": f"{question_id}_{a.atom_id}",
        "question_id": question_id,
        "question": question,
        "regulator": regulator,
        "query_date": QUERY_DATE,
        "atom_type": getattr(a, "atom_type", "UNKNOWN"),
        "atom_text": getattr(a, "claim", getattr(a, "text", "")),
        "stratum": f"*|{getattr(a, 'atom_type', 'UNKNOWN')}",
        "cited_chunk_ids": list(getattr(a, "cited_chunk_ids", [])),
        "v1_status": getattr(va, "v1_status", "NA"),
        "v2_entail_prob": _safe(getattr(va, "v2_entail_prob", None)),
        "s_ret": _safe(getattr(scored_atom, "s_ret", 0.0)),
        "s_nli": _safe(getattr(scored_atom, "s_nli", None)),
        "s_ent": _safe(getattr(scored_atom, "s_ent", None)),
        "s_div": _safe(getattr(scored_atom, "s_div", None)),
        "s_ver": _safe(getattr(scored_atom, "s_ver", None)),
        "aggregated_risk": scored_atom.risk,
        "decision": decision,
        "judge_verdict": None,
        "latency": latency,
    }


def _append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ── Load already-completed question IDs ──────────────────────────────────────
done_ids: set[str] = set()
if CHECKPOINT.exists():
    for line in CHECKPOINT.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                done_ids.add(json.loads(line)["question_id"])
            except Exception:
                pass
logger.info("Already done: %d questions", len(done_ids))


# ── Main loop ─────────────────────────────────────────────────────────────────
df = pd.read_csv(QUESTIONS_PATH)
total_q = len(df)
total_atoms = 0
total_cost = 0.0
target_atoms = 300

logger.info("Running pipeline on %d questions (target %d atoms) …", total_q, target_atoms)

for idx, row in df.iterrows():
    q_id = str(row.get("question_id", f"q{idx}"))
    question = str(row["question"])
    regulator = str(row.get("regulator", "RBI"))
    query_date = str(row.get("query_date", QUERY_DATE))

    if q_id in done_ids:
        logger.info("[%d/%d] SKIP (already done): %s", idx + 1, total_q, q_id)
        continue

    if total_atoms >= target_atoms:
        logger.info("Reached %d atoms target. Stopping.", target_atoms)
        break

    logger.info("[%d/%d] %s | %s", idx + 1, total_q, q_id, question[:80])
    run_latency: dict = {}
    run_atoms: list = []

    try:
        # M2: Retrieve
        t = time.perf_counter()
        rr = bm25_retrieve(question, query_date)
        run_latency["retrieve"] = round(time.perf_counter() - t, 3)
        logger.info("  M2: %d chunks (%.2fs)", len(rr.chunks), run_latency["retrieve"])

        # M3: Generate
        t = time.perf_counter()
        answer: GeneratedAnswer = _generator.generate(rr)
        run_latency["generate"] = round(time.perf_counter() - t, 3)
        logger.info("  M3: %d chars (%.2fs)", len(answer.answer_text), run_latency["generate"])

        # M4: Extract atoms
        t = time.perf_counter()
        atoms = extract_atoms(answer)
        run_latency["extract"] = round(time.perf_counter() - t, 3)
        logger.info("  M4: %d atoms (%.2fs)", len(atoms), run_latency["extract"])

        if not atoms:
            logger.warning("  No atoms extracted; skipping verification.")
            _append_jsonl(CHECKPOINT, {"question_id": q_id, "n_atoms": 0, "status": "no_atoms"})
            continue

        # M5: Verify
        t = time.perf_counter()
        verified = cascade_verify(atoms, rr)
        run_latency["verify"] = round(time.perf_counter() - t, 3)
        logger.info("  M5: %d verified (%.2fs)", len(verified), run_latency["verify"])

        # M6: Score
        t = time.perf_counter()
        scored = agg_score(verified, answer, rr)
        run_latency["score"] = round(time.perf_counter() - t, 3)
        logger.info("  M6: %d scored (%.2fs)", len(scored), run_latency["score"])

        # M8: Route (no judge in pilot)
        decisions = [(sa, _route_atom(sa, _thresholds)) for sa in scored]
        n_sup = sum(1 for _, d in decisions if d == "SUPPORTED")
        n_abs = sum(1 for _, d in decisions if d == "ABSTAINED")
        n_unc = sum(1 for _, d in decisions if d == "UNCERTAIN")
        logger.info("  M8: SUPPORTED=%d  ABSTAINED=%d  UNCERTAIN=%d", n_sup, n_abs, n_unc)

        # Write atoms
        for sa, decision in decisions:
            row_dict = _atom_to_dict(q_id, question, regulator, sa, decision, run_latency)
            _append_jsonl(ATOMS_OUT, row_dict)
            run_atoms.append(row_dict)

        total_atoms += len(run_atoms)
        _append_jsonl(CHECKPOINT, {
            "question_id": q_id, "n_atoms": len(run_atoms),
            "status": "ok", "latency": run_latency,
        })
        _append_jsonl(COST_OUT, {
            "question_id": q_id, "role": "generator",
            "model": _MODEL,
            "cost_usd": 0.0,
            "note": f"local {_BACKEND} — no cost",
            "ts": datetime.now(timezone.utc).isoformat(),
        })

    except KeyboardInterrupt:
        logger.warning("Interrupted by user. %d atoms written so far.", total_atoms)
        break
    except Exception as e:
        logger.error("  ERROR on %s: %s", q_id, e)
        _append_jsonl(CHECKPOINT, {"question_id": q_id, "n_atoms": 0, "status": f"error: {e}"})
        continue

logger.info("=" * 60)
logger.info("Done. Total atoms: %d / %d target", total_atoms, target_atoms)
logger.info("Output: %s", ATOMS_OUT)
