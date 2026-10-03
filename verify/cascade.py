"""
verify/cascade.py
-----------------
M5 top-level cascade: V1 → V2(NLI) → VerifiedAtom

Decision logic
--------------
1. Run V1 (deterministic) for all atoms that have a numeric/date/section
   type.  V1 status is used as a scoring SIGNAL, not as a bypass gate.
2. Run V2 (NLI) for **every atom** (Design Gap 6.4 fix):
   - Skipping NLI when V1=MATCH is unsafe because a retrieved chunk can
     contain the same number with a different semantic proposition, causing
     a false match (wrong claim → accidental lexical match → NLI skipped
     → low risk → SUPPORTED).
   - DeBERTa-large inference on short pairs is fast enough.
   - premise  = best_evidence(atom.claim, retrieved_chunk_texts)
   - hypothesis = atom.claim
   - Store raw p_entail in VerifiedAtom.v2_entail_prob.
   - The *threshold* is NOT applied here; that is M7's job.

Public API
----------
    from verify.cascade import verify
    verified: list[VerifiedAtom] = verify(atoms, retrieval_result, nli_model_path=...)
"""
from __future__ import annotations

import logging
from typing import List, Optional

from common.schemas import Atom, RetrievalResult, VerifiedAtom
from verify.v1_deterministic import v1_check, MATCH, MISMATCH, NOT_FOUND, NA
from verify.v2_nli import get_verifier, best_evidence

logger = logging.getLogger("verify.cascade")

# Atom types where V1 gives a definitive answer (MATCH or MISMATCH)
_V1_DEFINITIVE_TYPES = {"RATE", "THRESHOLD", "DATE", "SECTION"}


def _atom_to_sentence(atom: Atom) -> str:
    """Return the full claim sentence for NLI; fall back to atom.text."""
    return atom.claim if atom.claim else atom.text


def _chunk_texts(rr: RetrievalResult) -> List[str]:
    return [c.text for c in rr.chunks]


# ---------------------------------------------------------------------------
# Main cascade entry point
# ---------------------------------------------------------------------------

def verify(
    atoms: List[Atom],
    rr: RetrievalResult,
    nli_model_path: Optional[str] = None,
    skip_v2: bool = False,
    strict: bool = False,
) -> List[VerifiedAtom]:
    """
    Run the V1 → V2 verification cascade on a list of atoms.

    Parameters
    ----------
    atoms : list of Atom
        Output of the M4 atom extractor.
    rr : RetrievalResult
        The evidence bundle (retrieved chunks) for the query.
    nli_model_path : str, optional
        Override path to the fine-tuned NLI checkpoint.
        If None, uses env-var NLI_MODEL_PATH or HF default.
    skip_v2 : bool
        If True, skip V2 entirely (useful for fast tests / when model
        is not yet trained). Atoms that would go to V2 get v2_entail_prob=None.
    strict : bool
        If True, raise exceptions if NLI model fails to load or predict.

    Returns
    -------
    list of VerifiedAtom — one per input atom, in the same order.
    Every atom is guaranteed an entry; no silent drops.
    """
    if not skip_v2:
        nli = get_verifier(nli_model_path)

    chunk_text_list = _chunk_texts(rr)
    verified_atoms: List[VerifiedAtom] = []

    for atom in atoms:
        # ------------------------------------------------------------------
        # Step 1: V1 deterministic  (informational signal, not bypass gate)
        # ------------------------------------------------------------------
        v1_status = v1_check(atom, rr)

        # ------------------------------------------------------------------
        # Step 2: Always run NLI (Design Gap 6.4 fix)
        # ------------------------------------------------------------------
        # V1=MATCH no longer skips NLI.  Lexical coincidence in a chunk is
        # NOT semantic support — NLI must confirm the proposition.
        # ENTITY / APPLICABILITY always ran V2 before; that is preserved.
        # The only legitimate skip is skip_v2=True (test/no-model mode).
        run_v2 = not skip_v2
        v2_skipped_reason: Optional[str] = None
        if skip_v2:
            v2_skipped_reason = "skip_v2_flag"

        v2_entail_prob: Optional[float] = None

        if run_v2:
            hypothesis = _atom_to_sentence(atom)
            premise = best_evidence(hypothesis, chunk_text_list)
            try:
                p_e, p_c, p_n = nli.predict(premise, hypothesis, strict=strict)
                v2_entail_prob = round(p_e, 4)
                logger.debug(
                    "V2 for atom %s: p_entail=%.3f, p_contra=%.3f, p_neutral=%.3f",
                    atom.atom_id, p_e, p_c, p_n,
                )
            except Exception as exc:
                if strict:
                    raise
                logger.warning("V2 prediction failed for atom %s: %s", atom.atom_id, exc)
                v2_entail_prob = None

        verified_atoms.append(
            VerifiedAtom(
                atom=atom,
                v1_status=v1_status,
                v2_entail_prob=v2_entail_prob,
                metadata={
                    "v2_ran": run_v2,
                    "v2_skipped_reason": v2_skipped_reason,
                    # Record whether V1 would have been definitive pre-fix
                    "v1_was_definitive": v1_status in {MATCH, MISMATCH},
                },
            )
        )

    logger.info(
        "Cascade complete: %d atoms | MATCH=%d MISMATCH=%d NOT_FOUND=%d NA=%d | V2_ran=%d",
        len(verified_atoms),
        sum(1 for v in verified_atoms if v.v1_status == MATCH),
        sum(1 for v in verified_atoms if v.v1_status == MISMATCH),
        sum(1 for v in verified_atoms if v.v1_status == NOT_FOUND),
        sum(1 for v in verified_atoms if v.v1_status == NA),
        sum(1 for v in verified_atoms if v.metadata.get("v2_ran")),
    )

    return verified_atoms
