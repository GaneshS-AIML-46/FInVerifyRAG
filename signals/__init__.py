"""
M6: signals/ — Multi-signal atom risk computation.

Public API
----------
    from signals import compute_all_signals

    feats = compute_all_signals(
        verified_atom,
        answer,
        rr,
        no_context_logprobs=None,   # optional
    )
    # feats is a flat dict with all signal columns ready for the aggregator:
    # {s_div, s_ret, s_ver, s_nli, nli_skipped, s_ent, s_ent_max,
    #  s_ent_skipped, v1_match, v1_mismatch, v1_not_found, v1_na}
"""
from __future__ import annotations

import math
from typing import List, Optional

from common.schemas import GeneratedAnswer, RetrievalResult, VerifiedAtom

from signals.s_div import compute_s_div
from signals.s_ent import s_ent_features
from signals.s_nli import s_nli_features
from signals.s_ret import compute_s_ret
from signals.s_ver import s_ver_features


def compute_all_signals(
    verified_atom: VerifiedAtom,
    answer: GeneratedAnswer,
    rr: RetrievalResult,
    no_context_logprobs: Optional[List[float]] = None,
    strict: bool = False,
) -> dict:
    """
    Compute all M6 signals for one VerifiedAtom and return a flat feature dict.

    Parameters
    ----------
    verified_atom       : VerifiedAtom from M5 cascade.
    answer              : GeneratedAnswer (with token_logprobs if available).
    rr                  : RetrievalResult (with rerank_scores in metadata if available).
    no_context_logprobs : Optional per-token logprobs of the no-context answer
                          (used for s_div Strategy B; Strategy A uses
                          answer.metadata[\"no_context_answer\"]).

    Returns
    -------
    dict with keys:
        s_div, s_ret, s_ver, s_nli, nli_skipped,
        s_ent, s_ent_max, s_ent_skipped,
        v1_match, v1_mismatch, v1_not_found, v1_na
    """
    feats: dict = {}

    # --- s_ver: V1 deterministic encoding ---
    feats.update(s_ver_features(verified_atom))

    # --- s_nli: 1 - p(entail) ---
    feats.update(s_nli_features(verified_atom))

    # --- s_ent: model token entropy ---
    feats.update(s_ent_features(verified_atom, answer))

    # --- s_ret: retrieval reranker strength ---
    ret = compute_s_ret(verified_atom, rr)
    # compute_s_ret still returns 0.5 when evidence is absent. Treat that
    # sentinel as missing only when there is no cited chunk and no scores.
    cited = verified_atom.atom.cited_chunk
    has_scores = bool((rr.metadata or {}).get("rerank_scores"))
    if not cited and not has_scores and not rr.chunks:
        feats["s_ret"] = float("nan")
        feats["s_ret_skipped"] = 1
    else:
        feats["s_ret"] = ret
        feats["s_ret_skipped"] = 0

    # --- s_div: context divergence ---
    div = compute_s_div(verified_atom, answer, no_context_logprobs, strict=strict)
    feats["s_div"] = div
    feats["s_div_skipped"] = int(isinstance(div, float) and math.isnan(div))

    return feats


def validate_signals(records: List[dict], strict: bool = False) -> bool:
    """
    Validate that signals in a run did not fail or return a constant value
    for all atoms in the run.

    Raises StrictFailure in strict mode if any signal returned a constant value
    across all atoms. Outside strict mode, prints the NOT PAPER-VALID banner and
    returns False.
    """
    if len(records) < 2:
        return True

    from common.errors import StrictFailure, print_not_paper_valid_banner

    signal_keys = ("s_div", "s_ret", "s_ver", "s_nli", "s_ent")
    for k in signal_keys:
        vals = []
        for r in records:
            if k in r:
                vals.append(r[k])
            elif "signals" in r and isinstance(r["signals"], dict) and k in r["signals"]:
                vals.append(r["signals"][k])

        if not vals or len(vals) < len(records):
            continue

        first = vals[0]
        is_const = False
        if first is None or (isinstance(first, float) and math.isnan(first)):
            if all(v is None or (isinstance(v, float) and math.isnan(v)) for v in vals):
                is_const = True
        else:
            if all(v == first for v in vals):
                is_const = True

        if is_const:
            msg = f"Signal '{k}' returned a constant value ({first}) for all {len(records)} atoms in run."
            if strict:
                raise StrictFailure(msg)
            print_not_paper_valid_banner(msg)
            return False

    return True


__all__ = ["compute_all_signals", "validate_signals"]
