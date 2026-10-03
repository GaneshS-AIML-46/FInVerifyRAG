"""
app/pipeline.py
---------------
M9: End-to-end pipeline runner that chains M2 → M8.

This is the single entry point used by both the FastAPI layer (app/api.py)
and the Streamlit UI.  It accepts a plain question (+ optional date and
session history) and returns a structured PipelineResult.

Architecture
------------
    1. M2  retrieve(query, query_date)              -> RetrievalResult
    2. M3  generate(retrieval_result)               -> GeneratedAnswer
    3. M4  extract_atoms(answer)                    -> List[Atom]
    4. M5  verify(atoms, retrieval_result)          -> List[VerifiedAtom]
    5. M6  score(verified_atoms, answer, rr)        -> List[ScoredAtom]
    6. M7  load_thresholds()                        -> Dict[str, Thresholds]
    7. M8  decide(scored, thresholds, judge, rr, answer) -> AnswerDecision
    8. M8  rewrite_answer(answer, decision)         -> RewrittenAnswer
    9. App build PipelineResult

All heavy objects (retriever, generator, NLI model, aggregator, thresholds)
are loaded once and cached in module-level singletons.  Call
``reset_pipeline()`` to force re-initialisation (e.g. after re-ingestion).

Public API
----------
    from app.pipeline import run_pipeline, PipelineResult, reset_pipeline
    result = run_pipeline("What is the SLR requirement?", query_date="2024-01-15")
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional

logger = logging.getLogger("app.pipeline")

# ── Schema imports ────────────────────────────────────────────────────────────
from common.schemas import (
    GeneratedAnswer, RetrievalResult, ScoredAtom, Thresholds,
)
from decision.combine import AnswerDecision
from decision.answer_rewrite import RewrittenAnswer


# ══════════════════════════════════════════════════════════════════════════════
# Result dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AtomResult:
    """Per-atom output for the API / UI."""
    atom_id:      str
    atom_type:    str
    text:         str           # Atom text snippet
    claim:        str           # Full self-contained sentence
    status:       str           # SUPPORTED | VERIFIED | UNCERTAIN | ABSTAINED | NOT_VERIFIED
    risk:         float
    risk_label:   str           # LOW | MEDIUM | HIGH
    stratum:      str
    v1_status:    Optional[str] = None
    v2_entail_prob: Optional[float] = None
    rationale:    Optional[str] = None
    evidence_quote: Optional[str] = None


@dataclass
class SourceResult:
    """Per-source-chunk output for the API / UI."""
    chunk_id:   str
    regulator:  str
    issue_date: str
    source_url: str
    section:    str
    preview:    str             # First N chars of chunk text


@dataclass
class VerificationSummary:
    """Answer-level verification summary."""
    status:        str          # FULLY_SUPPORTED | PARTIALLY_VERIFIED | ABSTAINED
    status_label:  str          # Human-readable label
    coverage:      float        # Fraction of atoms SUPPORTED or VERIFIED
    n_total:       int
    n_supported:   int
    n_verified:    int
    n_uncertain:   int
    n_abstained:   int
    n_not_verified: int
    judge_calls:   int
    explanation:   str          # Short human-readable explanation for UI


@dataclass
class PipelineResult:
    """Full structured output from one pipeline run."""
    query:          str
    query_date:     str
    answer:         str                     # Rewritten (or original) answer text
    original_answer: str                    # Raw generator output
    disclaimer:     str                     # Prepended disclaimer (if any)
    rewrite_mode:   str                     # "none" | "flag" | "trim" | "stub"
    atoms:          List[AtomResult]
    sources:        List[SourceResult]
    verification:   VerificationSummary
    latency:        Dict[str, float] = field(default_factory=dict)  # stage -> seconds
    error:          Optional[str]    = None


# ══════════════════════════════════════════════════════════════════════════════
# Lazy singleton loaders
# ══════════════════════════════════════════════════════════════════════════════

_retriever    = None
_generator    = None
_atom_extractor = None
_cascade      = None
_aggregator   = None
_judge        = None
_thresholds: Dict[str, Any] = {}

_THRESHOLDS_PATH = "models/guarantee/thresholds.json"
_REWRITE_MODE    = "flag"
_SKIP_JUDGE      = False
_MAX_SOURCES     = 6


def _load_retriever():
    global _retriever
    if _retriever is None:
        try:
            from retrieval.retriever import retrieve  # M2
            _retriever = retrieve
        except ImportError:
            logger.warning("M2 retriever not available; using stub.")
            _retriever = _stub_retriever
    return _retriever


def _load_generator():
    global _generator
    if _generator is None:
        try:
            from generation.generator import generate  # M3
            _generator = generate
        except ImportError:
            logger.warning("M3 generator not available; using stub.")
            _generator = _stub_generator
    return _generator


def _load_atom_extractor():
    global _atom_extractor
    if _atom_extractor is None:
        try:
            from atoms.extractor import extract  # M4
            _atom_extractor = extract
        except ImportError:
            logger.warning("M4 atom extractor not available; using stub.")
            _atom_extractor = _stub_extract
    return _atom_extractor


def _load_cascade():
    global _cascade
    if _cascade is None:
        try:
            from verify.cascade import verify  # M5
            _cascade = verify
        except ImportError:
            logger.warning("M5 cascade not available; using stub.")
            _cascade = _stub_verify
    return _cascade


def _load_aggregator():
    global _aggregator
    if _aggregator is None:
        try:
            from aggregate.model import score  # M6
            _aggregator = score
        except ImportError:
            logger.warning("M6 aggregator not available; using stub.")
            _aggregator = _stub_score
    return _aggregator


def _load_thresholds() -> Dict[str, Any]:
    global _thresholds
    if not _thresholds:
        try:
            from guarantee.certify import load_thresholds  # M7
            _thresholds = load_thresholds(_THRESHOLDS_PATH)
        except Exception as exc:
            logger.warning("Could not load thresholds (%s); using empty dict.", exc)
            _thresholds = {}
    return _thresholds


def _load_judge():
    global _judge
    if _judge is None:
        try:
            from judge.judge_llm import JudgeLLM  # M8
            _judge = JudgeLLM()
        except Exception as exc:
            logger.warning("Judge LLM not available (%s); using stub.", exc)
            _judge = _StubJudge()
    return _judge


def reset_pipeline() -> None:
    """Force re-initialisation of all cached objects (e.g. after re-ingest)."""
    global _retriever, _generator, _atom_extractor, _cascade
    global _aggregator, _judge, _thresholds
    _retriever = _generator = _atom_extractor = _cascade = None
    _aggregator = _judge = None
    _thresholds = {}
    logger.info("Pipeline singletons reset.")


# ══════════════════════════════════════════════════════════════════════════════
# Stubs (used when real modules are unavailable / not yet trained)
# ══════════════════════════════════════════════════════════════════════════════

def _stub_retriever(query: str, query_date: str, **_kw) -> RetrievalResult:
    return RetrievalResult(query=query, query_date=query_date, chunks=[])


def _stub_generator(rr: RetrievalResult, **_kw) -> GeneratedAnswer:
    return GeneratedAnswer(
        query=rr.query,
        answer_text="[Generator not available]",
        citations=[],
        token_logprobs=None,
        model_id="stub",
    )


def _stub_extract(answer: GeneratedAnswer, **_kw):
    return []


def _stub_verify(atoms, rr, **_kw):
    from common.schemas import VerifiedAtom
    return [VerifiedAtom(atom=a, v1_status="NA") for a in atoms]


def _stub_score(verified_atoms, answer, rr, **_kw) -> List[ScoredAtom]:
    return [ScoredAtom(verified=va, risk=0.5) for va in verified_atoms]


class _StubJudge:
    """No-op judge that marks everything NOT_VERIFIED."""
    def reset_call_counter(self): pass
    def judge_batch(self, uncertain_atoms, rr):
        from judge.schemas import JudgeResult, NOT_VERIFIED
        return {
            sa.verified.atom.atom_id: JudgeResult(
                atom_id=sa.verified.atom.atom_id,
                verdict=NOT_VERIFIED,
                rationale="Judge not available.",
                evidence_quote=None,
            )
            for sa in uncertain_atoms
        }


# ══════════════════════════════════════════════════════════════════════════════
# Helper builders
# ══════════════════════════════════════════════════════════════════════════════

def _build_atom_result(ad, badge_fn) -> AtomResult:
    """Convert an AtomDecision to an AtomResult."""
    rb = badge_fn(ad.risk)
    return AtomResult(
        atom_id=ad.atom_id,
        atom_type=ad.atom_type,
        text=ad.atom_text,
        claim=ad.atom_claim,
        status=ad.final_status,
        risk=round(ad.risk, 4),
        risk_label=rb.label,
        stratum=ad.stratum,
        v1_status=ad.v1_status,
        v2_entail_prob=ad.v2_entail_prob,
        rationale=ad.rationale,
        evidence_quote=ad.evidence_quote,
    )


def _build_source(chunk, max_chars: int = 300) -> SourceResult:
    """Convert a Chunk to a SourceResult."""
    meta = getattr(chunk, "metadata", {}) or {}
    return SourceResult(
        chunk_id=chunk.chunk_id,
        regulator=chunk.regulator,
        issue_date=chunk.issue_date,
        source_url=chunk.source_url,
        section=str(meta.get("section", meta.get("breadcrumb", ""))),
        preview=chunk.text[:max_chars],
    )


def _build_verification_summary(ad: AnswerDecision) -> VerificationSummary:
    status = ad.answer_status
    label_map = {
        "FULLY_SUPPORTED":    "Fully Verified",
        "PARTIALLY_VERIFIED": "Partially Verified",
        "ABSTAINED":          "Cannot Be Verified",
    }
    label = label_map.get(status, status)

    if status == "FULLY_SUPPORTED":
        explanation = (
            f"All {ad.n_supported + ad.n_verified} atom(s) verified against regulatory evidence."
        )
    elif status == "ABSTAINED":
        explanation = (
            "No atoms could be verified. The answer should not be relied upon without "
            "consulting primary regulatory sources."
        )
    else:
        explanation = (
            f"{ad.n_supported + ad.n_verified} of {len(ad.atom_decisions)} atom(s) verified. "
            f"{ad.n_abstained} abstained, {ad.n_not_verified} not verified, "
            f"{ad.n_uncertain} uncertain."
        )

    return VerificationSummary(
        status=status,
        status_label=label,
        coverage=ad.coverage,
        n_total=len(ad.atom_decisions),
        n_supported=ad.n_supported,
        n_verified=ad.n_verified,
        n_uncertain=ad.n_uncertain,
        n_abstained=ad.n_abstained,
        n_not_verified=ad.n_not_verified,
        judge_calls=ad.judge_calls,
        explanation=explanation,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline entry point
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    question:     str,
    query_date:   Optional[str] = None,
    history:      Optional[List[Dict[str, str]]] = None,
    skip_judge:   bool = _SKIP_JUDGE,
    rewrite_mode: str  = _REWRITE_MODE,
    max_sources:  int  = _MAX_SOURCES,
    strict:       bool = False,
    checkpoint_path: Optional[str] = None,
) -> PipelineResult:
    """
    Run the full M2 → M8 pipeline for one question.

    Parameters
    ----------
    question     : User question (plain text).
    query_date   : ISO date string (e.g. "2024-01-15").  Defaults to today.
    history      : Optional list of {role: "user"|"assistant", content: "..."} turns.
    skip_judge   : If True, skip M8 judge (UNCERTAIN atoms remain UNCERTAIN).
    rewrite_mode : "flag" | "trim" | "stub_on_abstain" | "none".
    max_sources  : Max source chunks in the response.

    Returns
    -------
    PipelineResult.
    """
    from app.badge import risk_badge  # avoid circular at module level

    if not query_date:
        if strict:
            raise ValueError("query_date is required on each question in strict mode")
        query_date = date.today().isoformat()

    latency: Dict[str, float] = {}
    t0 = time.perf_counter()

    try:
        # ── M2: Retrieve ──────────────────────────────────────────────────────
        t = time.perf_counter()
        retriever = _load_retriever()
        rr: RetrievalResult = retriever(query=question, query_date=query_date)
        latency["retrieve"] = round(time.perf_counter() - t, 3)
        logger.info("M2 done: %d chunks  (%.3fs)", len(rr.chunks), latency["retrieve"])

        # ── M3: Generate ──────────────────────────────────────────────────────
        t = time.perf_counter()
        generator = _load_generator()
        answer: GeneratedAnswer = generator(rr)
        latency["generate"] = round(time.perf_counter() - t, 3)
        logger.info("M3 done: %d chars  (%.3fs)", len(answer.answer_text), latency["generate"])

        # ── M4: Extract atoms ─────────────────────────────────────────────────
        t = time.perf_counter()
        extractor = _load_atom_extractor()
        atoms = extractor(answer)
        latency["extract"] = round(time.perf_counter() - t, 3)
        logger.info("M4 done: %d atoms  (%.3fs)", len(atoms), latency["extract"])

        # ── M5: Verify (cascade V1 + V2) ──────────────────────────────────────
        t = time.perf_counter()
        cascade = _load_cascade()
        verified_atoms = cascade(atoms, rr, strict=strict)
        latency["verify"] = round(time.perf_counter() - t, 3)
        logger.info("M5 done: %d verified  (%.3fs)", len(verified_atoms), latency["verify"])

        # ── M6: Score (aggregator) ─────────────────────────────────────────────
        t = time.perf_counter()
        aggregator = _load_aggregator()
        scored_atoms: List[ScoredAtom] = aggregator(verified_atoms, answer, rr, strict=strict)
        latency["score"] = round(time.perf_counter() - t, 3)
        logger.info("M6 done: %d scored  (%.3fs)", len(scored_atoms), latency["score"])

        # ── M7: Load thresholds ────────────────────────────────────────────────
        thresholds = _load_thresholds()
        if strict and not thresholds:
            from common.errors import StrictFailure
            raise StrictFailure("No certified thresholds at models/guarantee/thresholds.json. Refusing heuristic thresholds.")

        # ── M8: Decide (router + judge + combine) ──────────────────────────────
        t = time.perf_counter()
        from decision.combine import decide as m8_decide
        judge = _load_judge()
        answer_decision: AnswerDecision = m8_decide(
            scored_atoms=scored_atoms,
            thresholds=thresholds,
            judge=judge,
            rr=rr,
            answer=answer,
            skip_judge=skip_judge,
        )
        latency["decide"] = round(time.perf_counter() - t, 3)
        logger.info("M8 done: %s  (%.3fs)", answer_decision.summary(), latency["decide"])

        # ── M8: Rewrite ────────────────────────────────────────────────────────
        t = time.perf_counter()
        from decision.answer_rewrite import rewrite_answer
        rewritten: RewrittenAnswer = rewrite_answer(
            answer=answer,
            answer_decision=answer_decision,
            mode=rewrite_mode,
        )
        latency["rewrite"] = round(time.perf_counter() - t, 3)

        latency["total"] = round(time.perf_counter() - t0, 3)
        logger.info("Pipeline total: %.3fs", latency["total"])

        # ── Assemble result ────────────────────────────────────────────────────
        atom_results = [_build_atom_result(ad, risk_badge) for ad in answer_decision.atom_decisions]

        # Source chunks: deduplicate and limit
        seen_chunk_ids: set[str] = set()
        source_results: List[SourceResult] = []
        for chunk in rr.chunks[:max_sources]:
            if chunk.chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk.chunk_id)
                source_results.append(_build_source(chunk))

        verification = _build_verification_summary(answer_decision)

        result = PipelineResult(
            query=question,
            query_date=query_date,
            answer=rewritten.text,
            original_answer=answer.answer_text,
            disclaimer=rewritten.disclaimer,
            rewrite_mode=rewritten.rewrite_mode,
            atoms=atom_results,
            sources=source_results,
            verification=verification,
            latency=latency,
        )
        if checkpoint_path:
            from common.checkpoint import append_jsonl
            append_jsonl(checkpoint_path, {
                "question": question,
                "query_date": query_date,
                "status": verification.status,
                "n_atoms": len(atom_results),
            })
        return result

    except Exception as exc:
        if strict:
            raise
        logger.exception("Pipeline error: %s", exc)
        latency["total"] = round(time.perf_counter() - t0, 3)
        return PipelineResult(
            query=question,
            query_date=query_date,
            answer="",
            original_answer="",
            disclaimer="",
            rewrite_mode="none",
            atoms=[],
            sources=[],
            verification=VerificationSummary(
                status="ABSTAINED",
                status_label="Error",
                coverage=0.0,
                n_total=0,
                n_supported=0,
                n_verified=0,
                n_uncertain=0,
                n_abstained=0,
                n_not_verified=0,
                judge_calls=0,
                explanation=f"Pipeline error: {exc}",
            ),
            latency=latency,
            error=str(exc),
        )
