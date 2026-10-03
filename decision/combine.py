"""
decision/combine.py
--------------------
M8 Answer-level combination: aggregates per-atom decisions into a final
answer-level outcome and structured response object.

Answer-level rule (conservative / paper-aligned)
-------------------------------------------------
An answer is "SUPPORTED" only if ALL atoms pass (SUPPORTED or judge-VERIFIED).
If ANY atom is ABSTAINED or NOT_VERIFIED, the answer cannot be fully supported.

Combination table
-----------------
| All atoms                | Answer status    |
|--------------------------|------------------|
| All SUPPORTED            | FULLY_SUPPORTED  |
| All SUPPORTED + VERIFIED | FULLY_SUPPORTED  |
| Some UNCERTAIN remain    | PARTIALLY_VERIFIED |
| Any ABSTAINED            | PARTIALLY_VERIFIED |
| Any NOT_VERIFIED         | PARTIALLY_VERIFIED |
| All ABSTAINED            | ABSTAINED        |

The per-atom breakdown is always returned so the UI can annotate each claim.

Public API
----------
    from decision.combine import combine, AnswerDecision, AtomDecision

    answer_decision = combine(
        scored_atoms, route_results, judge_results, answer
    )
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from common.schemas import Decision, GeneratedAnswer, ScoredAtom
from decision.router import RouteResult, SUPPORTED, UNCERTAIN, ABSTAINED
from judge.schemas import JudgeResult, VERIFIED, NOT_VERIFIED

logger = logging.getLogger("decision.combine")

# Answer-level status values
FULLY_SUPPORTED     = "FULLY_SUPPORTED"
PARTIALLY_VERIFIED  = "PARTIALLY_VERIFIED"
ANSWER_ABSTAINED    = "ABSTAINED"


# ---------------------------------------------------------------------------
# Per-atom decision (the M0 Decision schema + extra fields for the API)
# ---------------------------------------------------------------------------

@dataclass
class AtomDecision:
    """
    Final per-atom decision combining router + judge outcomes.
    Maps to the ``Decision`` Pydantic schema for serialization.
    """
    atom_id:        str
    atom_type:      str
    atom_text:      str
    atom_claim:     str
    final_status:   str    # SUPPORTED | VERIFIED | ABSTAINED | NOT_VERIFIED | UNCERTAIN
    risk:           float
    stratum:        str
    rationale:      Optional[str] = None   # Judge rationale (for VERIFIED/NOT_VERIFIED)
    evidence_quote: Optional[str] = None   # Verbatim judge evidence quote
    v1_status:      Optional[str] = None
    v2_entail_prob: Optional[float] = None

    def to_schema(self) -> Decision:
        """Convert to the common Decision schema."""
        return Decision(
            atom_id=self.atom_id,
            status=self.final_status,
            risk=self.risk,
        )


# ---------------------------------------------------------------------------
# Answer-level decision
# ---------------------------------------------------------------------------

@dataclass
class AnswerDecision:
    """Aggregated decision for the full answer."""
    answer_status:   str                    # FULLY_SUPPORTED | PARTIALLY_VERIFIED | ABSTAINED
    atom_decisions:  List[AtomDecision]
    n_supported:     int = 0
    n_verified:      int = 0
    n_uncertain:     int = 0
    n_abstained:     int = 0
    n_not_verified:  int = 0
    judge_calls:     int = 0
    parse_errors:    int = 0
    coverage:        float = 0.0            # fraction of atoms that are SUPPORTED or VERIFIED
    answer_text:     str = ""
    query:           str = ""

    def to_decisions(self) -> List[Decision]:
        """Return the list of Decision schema objects for downstream use."""
        return [ad.to_schema() for ad in self.atom_decisions]

    def summary(self) -> str:
        """Human-readable one-liner for logging."""
        return (
            f"answer={self.answer_status}  atoms={len(self.atom_decisions)}  "
            f"SUPPORTED={self.n_supported}  VERIFIED={self.n_verified}  "
            f"UNCERTAIN={self.n_uncertain}  ABSTAINED={self.n_abstained}  "
            f"NOT_VERIFIED={self.n_not_verified}  judge_calls={self.judge_calls}  "
            f"coverage={self.coverage:.2%}"
        )


# ---------------------------------------------------------------------------
# Core combine logic
# ---------------------------------------------------------------------------

def combine(
    scored_atoms:  List[ScoredAtom],
    route_results: List[RouteResult],
    judge_results: Dict[str, JudgeResult],
    answer:        GeneratedAnswer,
) -> AnswerDecision:
    """
    Combine per-atom routing + judge results into the final answer decision.

    Parameters
    ----------
    scored_atoms  : All ScoredAtoms for the query (from M6).
    route_results : Per-atom RouteResult from decision.router (same order).
    judge_results : Dict[atom_id -> JudgeResult] for UNCERTAIN atoms (from judge).
    answer        : GeneratedAnswer for context (query, answer_text).

    Returns
    -------
    AnswerDecision with per-atom breakdown and answer-level status.
    """
    atom_decisions: List[AtomDecision] = []
    n_supported    = 0
    n_verified     = 0
    n_uncertain    = 0
    n_abstained    = 0
    n_not_verified = 0
    parse_errors   = 0
    judge_calls    = len(judge_results)

    for scored, route in zip(scored_atoms, route_results):
        va       = scored.verified
        atom     = va.atom
        atom_id  = atom.atom_id

        # Determine final_status
        if route.status == SUPPORTED:
            final_status = SUPPORTED
            rationale    = None
            ev_quote     = None
            n_supported += 1

        elif route.status == ABSTAINED:
            final_status = ABSTAINED
            rationale    = "Risk score exceeds certified threshold; atom not shown as fact."
            ev_quote     = None
            n_abstained += 1

        elif route.status == UNCERTAIN:
            jr = judge_results.get(atom_id)
            if jr is None:
                # Judge was not called (e.g. cap hit) — stay UNCERTAIN
                final_status = UNCERTAIN
                rationale    = "Judge not called (call cap reached or skipped)."
                ev_quote     = None
                n_uncertain += 1
            elif jr.verdict == VERIFIED:
                final_status = VERIFIED
                rationale    = jr.rationale
                ev_quote     = jr.evidence_quote
                n_verified  += 1
            else:  # NOT_VERIFIED
                final_status = NOT_VERIFIED
                rationale    = jr.rationale
                ev_quote     = jr.evidence_quote
                n_not_verified += 1
                if getattr(jr, 'parse_error', False):
                    parse_errors += 1
        else:
            # Fallback: treat unknown status as ABSTAINED
            final_status = ABSTAINED
            rationale    = f"Unknown route status: {route.status!r}"
            ev_quote     = None
            n_abstained += 1

        atom_decisions.append(AtomDecision(
            atom_id=atom_id,
            atom_type=atom.type,
            atom_text=atom.text,
            atom_claim=atom.claim,
            final_status=final_status,
            risk=scored.risk,
            stratum=route.stratum,
            rationale=rationale,
            evidence_quote=ev_quote,
            v1_status=va.v1_status,
            v2_entail_prob=va.v2_entail_prob,
        ))

    # Answer-level aggregation
    n_total = len(atom_decisions)
    n_pass  = n_supported + n_verified
    coverage = n_pass / n_total if n_total > 0 else 0.0

    if n_total == 0:
        answer_status = FULLY_SUPPORTED   # No atoms to fail
    elif n_abstained == n_total:
        answer_status = ANSWER_ABSTAINED
    elif n_abstained > 0 or n_not_verified > 0 or n_uncertain > 0:
        answer_status = PARTIALLY_VERIFIED
    else:
        answer_status = FULLY_SUPPORTED

    decision = AnswerDecision(
        answer_status=answer_status,
        atom_decisions=atom_decisions,
        n_supported=n_supported,
        n_verified=n_verified,
        n_uncertain=n_uncertain,
        n_abstained=n_abstained,
        n_not_verified=n_not_verified,
        judge_calls=judge_calls,
        parse_errors=parse_errors,
        coverage=round(coverage, 4),
        answer_text=answer.answer_text,
        query=answer.query,
    )

    logger.info("Combine: %s", decision.summary())
    return decision


# ---------------------------------------------------------------------------
# Convenience: resolve UNCERTAIN atoms through judge and combine in one call
# ---------------------------------------------------------------------------

def decide(
    scored_atoms:   List[ScoredAtom],
    thresholds:     Dict,
    judge,                          # JudgeLLM instance (avoid circular import)
    rr,                             # RetrievalResult
    answer:         GeneratedAnswer,
    skip_judge:     bool = False,
    mode:           str = 'selective',
) -> AnswerDecision:
    """
    Full M8 pipeline in one call:
      1. Route all scored atoms (router.route).
      2. Send UNCERTAIN atoms to the judge (unless skip_judge=True).
      3. Combine results (combine()).

    Parameters
    ----------
    scored_atoms : from M6.
    thresholds   : Dict[stratum -> Thresholds] from M7.
    judge        : JudgeLLM instance.
    rr           : RetrievalResult (for judge evidence).
    answer       : GeneratedAnswer.
    skip_judge   : If True, skip judge calls (UNCERTAIN atoms stay UNCERTAIN).

    Returns
    -------
    AnswerDecision.
    """
    from decision.router import route as _route

    route_results = _route(scored_atoms, thresholds)

    if mode == 'llm_judge_all':
        for r in route_results:
            r.status = UNCERTAIN

    judge_results: Dict[str, JudgeResult] = {}
    if not skip_judge:
        uncertain = [
            sa for sa, rr_res in zip(scored_atoms, route_results)
            if rr_res.status == UNCERTAIN
        ]
        if uncertain:
            judge.reset_call_counter()
            judge_results = judge.judge_batch(uncertain, rr)

    return combine(scored_atoms, route_results, judge_results, answer)
