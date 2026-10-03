"""
signals/s_div.py  —  Context Divergence Signal
------------------------------------------------
s_div measures how much the atom's claim sentence DIVERGES from what the
model says when there is NO context (no-context answer).

High divergence  →  the evidence actually changed the model's output
                    →  the claim is grounded in retrieved evidence  →  lower risk.
Low divergence   →  model would say the same thing even without evidence
                    →  possible hallucination or prior knowledge  →  higher risk.

Computation
-----------
Given:
  • atom.claim        : the specific sentence to evaluate
  • answer.answer_text: the evidence-grounded answer
  • no_context_text   : the answer generated WITHOUT evidence (stored in
                        answer.metadata["no_context_answer"])

Strategy A – embedding cosine distance (default, no logprob required):
    embed(atom.claim from answer)  vs  embed(matching sentence from no-context)
    s_div = 1 - cosine_similarity   ∈ [0, 1]
    0 → identical (bad, hallucination risk)  1 → very different (good, grounded)

Strategy B – KL divergence on token logprobs (if both logprob lists available):
    KL(context_distribution || no_context_distribution) over the atom's token span.
    (Used when answer.token_logprobs AND no_context_logprobs are both stored.)

We default to Strategy A (always available). Strategy B is engaged when
both logprob lists are present.
"""
from __future__ import annotations

import math
import logging
from typing import Optional, List

from common.schemas import VerifiedAtom, GeneratedAnswer

logger = logging.getLogger("signals.s_div")

# ---------------------------------------------------------------------------
# Embedding backend (lazy import)
# ---------------------------------------------------------------------------

_embed_model = None


def _get_embedder():
    """Lazy-load a small sentence-transformer for embedding."""
    global _embed_model
    if _embed_model is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            _embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
            logger.info("Loaded BGE-small embedder for s_div.")
        except Exception as exc:
            logger.warning("Could not load SentenceTransformer for s_div: %s", exc)
            _embed_model = None
    return _embed_model


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Strategy B: KL divergence on logprobs
# ---------------------------------------------------------------------------

def _kl_from_logprobs(
    lp_context: List[float],
    lp_nocontext: List[float],
    start: int,
    end: int,
) -> Optional[float]:
    """
    Estimate KL(P_ctx || P_noctx) for a token span.
    Both lists are per-token log-probabilities.
    Returns None if span is out of range.
    """
    span_ctx = lp_context[start:end]
    span_noc = lp_nocontext[start:end]
    if not span_ctx or len(span_ctx) != len(span_noc):
        return None
    kl = 0.0
    for lp_p, lp_q in zip(span_ctx, span_noc):
        p = math.exp(lp_p)
        q = math.exp(lp_q)
        if p > 0 and q > 0:
            kl += p * (lp_p - lp_q)
    return max(0.0, kl)


# ---------------------------------------------------------------------------
# Public signal function
# ---------------------------------------------------------------------------

def compute_s_div(
    verified_atom: VerifiedAtom,
    answer: GeneratedAnswer,
    no_context_logprobs: Optional[List[float]] = None,
    strict: bool = False,
) -> float:
    """
    Compute the context-divergence signal for one atom.

    Parameters
    ----------
    verified_atom    : VerifiedAtom from M5 cascade.
    answer           : GeneratedAnswer with answer_text, token_logprobs,
                       and optionally metadata["no_context_answer"].
    no_context_logprobs : Optional per-token logprobs of the no-context answer.
    strict           : If True, raise StrictFailure if signal fails to load.

    Returns
    -------
    float in [0, 1].
    0 → no divergence (atom same in context/no-context → riskier).
    1 → maximum divergence (evidence changed the answer → more grounded).
    """
    atom = verified_atom.atom
    claim = atom.claim or atom.text

    # --- Strategy B: logprob KL if both available ---
    if (
        answer.token_logprobs
        and no_context_logprobs
        and atom.span is not None
    ):
        # Convert character span to rough token span (heuristic: ~4 chars/token)
        char_start, char_end = atom.span
        tok_start = char_start // 4
        tok_end = max(tok_start + 1, char_end // 4)
        kl = _kl_from_logprobs(answer.token_logprobs, no_context_logprobs,
                                tok_start, tok_end)
        if kl is not None:
            # Normalize: clamp KL to [0, 5] then scale to [0, 1]
            return min(1.0, kl / 5.0)

    # --- Strategy A: embedding cosine distance ---
    no_context_text: str = answer.metadata.get("no_context_answer", "")
    if not no_context_text:
        logger.debug("s_div: no no_context_answer; marking missing.")
        return float("nan")

    embedder = _get_embedder()
    if embedder is None:
        if strict:
            from common.errors import StrictFailure
            raise StrictFailure("Signal 's_div' failed to load embedder.")
        logger.error("s_div: embedder unavailable; marking missing.")
        return float("nan")

    try:
        vecs = embedder.encode([claim, no_context_text], normalize_embeddings=True)
        sim = float(_cosine(vecs[0].tolist(), vecs[1].tolist()))
        return round(1.0 - sim, 4)   # divergence = 1 - similarity
    except Exception as exc:
        if strict:
            from common.errors import StrictFailure
            raise StrictFailure(f"Signal 's_div' embedding failed: {exc}") from exc
        logger.warning("s_div embedding failed: %s", exc)
        return float("nan")
