"""
judge/judge_llm.py
-------------------
M8 V3 Judge: strong LLM fact-verification for UNCERTAIN atoms.

The judge is only invoked for atoms that fall in the UNCERTAIN band
(accept_below < risk <= abstain_above).  It must never be called for
SUPPORTED or ABSTAINED atoms — those are already settled by the M7 thresholds.

Architecture
------------
- Uses a strong, separate LLM (Qwen2.5-72B or an API model) via
  ``common.llm_client.LLMClient``.
- The prompt (judge_v1.txt) enforces evidence-only verdicts and JSON output.
- Responses are parsed and validated; malformed JSON is retried once, then
  treated as NOT_VERIFIED (safe default).
- Judge calls are capped per query to control cost.
- Latency and call counts are logged for M10 metrics.

Guarantee handling
------------------
We use approach (a) from the implementation plan:
  "Treat system + judge as one pipeline and calibrate end-to-end risk on the
  calibration set (accepted set = SUPPORTED plus judge-VERIFIED)."
This means the judge's decisions are within the M7 guarantee only when the
end-to-end pipeline (including judge) has been calibrated jointly.  If
operating before that joint calibration, judge verdicts are provided as-is
with a ``guarantee_mode='empirical'`` flag.

Public API
----------
    from judge.judge_llm import JudgeResult, JudgeLLM

    judge = JudgeLLM(llm_client)
    result = judge.judge(scored_atom, retrieval_result)
    # result.verdict in {"VERIFIED", "NOT_VERIFIED"}
    # result.rationale: str
    # result.evidence_quote: str
    # result.latency_ms: float
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

# Auto-load .env from the project root so API keys are always available
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass  # python-dotenv not installed; rely on shell env vars

from common.schemas import Chunk, RetrievalResult, ScoredAtom
from judge.schemas import JudgeResult, VERIFIED, NOT_VERIFIED

if TYPE_CHECKING:
    from common.llm_client import LLMClient

logger = logging.getLogger("judge.judge_llm")

# Default prompt path
_DEFAULT_PROMPT_PATH = Path(__file__).parent / "prompts" / "judge_v1.txt"

# Max evidence characters sent to the judge (to stay within context limits)
_MAX_EVIDENCE_CHARS = 8000


# ---------------------------------------------------------------------------
# Evidence preparation helpers
# ---------------------------------------------------------------------------

def _format_evidence(chunks: List[Chunk], atom_claim: str) -> str:
    """
    Format retrieved chunks as numbered evidence paragraphs for the judge.
    Prioritises the cited chunk if available; truncates to _MAX_EVIDENCE_CHARS.
    """
    lines = []
    for i, c in enumerate(chunks, 1):
        header = f"[{i}] ({c.regulator}, {c.issue_date}, {c.chunk_id})"
        lines.append(f"{header}\n{c.text.strip()}")

    full = "\n\n".join(lines)
    if len(full) > _MAX_EVIDENCE_CHARS:
        full = full[:_MAX_EVIDENCE_CHARS] + "\n[...evidence truncated...]"
    return full


def _build_computed_annotation(atom) -> str:
    """
    Phase 7: Build a structured annotation block for COMPUTED atoms.

    The judge needs to see the derivation formula and inputs so it can
    check whether the computed result is consistent with the evidence,
    not just whether the text appears verbatim.

    Returns an empty string for non-COMPUTED atoms.
    """
    if atom.type != "COMPUTED" or atom.computed is None:
        return ""
    c = atom.computed
    lines = [
        "\n--- COMPUTED ATOM DERIVATION ---",
        f"Formula:   {c.formula}",
        f"Operation: {c.operation}",
        f"Inputs:    {', '.join(str(x) for x in c.inputs)}",
    ]
    if c.constants:
        lines.append(f"Constants: {c.constants}")
    lines.append(f"Result:    {c.result}")
    if c.result_numeric is not None:
        lines.append(f"Result (numeric): {c.result_numeric}")
    lines.append("--- END DERIVATION ---")
    return "\n".join(lines)


def _build_user_prompt(claim: str, evidence: str, atom_type: str, computed_annotation: str = "") -> str:
    """Build the user-turn message for the judge."""
    base = (
        f"ATOM TYPE: {atom_type}\n\n"
        f"CLAIM TO VERIFY:\n{claim}\n"
        f"{computed_annotation}\n\n"
        f"PROVIDED EVIDENCE:\n{evidence}\n\n"
        "Now give your verdict as JSON."
    )
    return base



# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def _parse_judge_json(raw: str) -> Dict[str, str]:
    """
    Parse the judge's JSON response.  Handles:
    - Clean JSON
    - JSON embedded inside a markdown code block ```json ... ```
    - Partial truncation (best-effort)

    Returns dict with keys: verdict, rationale, evidence_quote.
    Raises ValueError on complete failure.
    """
    text = raw.strip()

    # Strip markdown code fences if present
    if "```" in text:
        import re
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            text = m.group(1)
        else:
            text = text.replace("```json", "").replace("```", "").strip()

    # Try direct parse
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to extract just the JSON object
        import re
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
        else:
            raise ValueError(f"No JSON object found in response: {text[:200]!r}")

    # Validate required fields
    verdict = str(data.get("verdict", "")).upper().strip()
    if verdict not in {VERIFIED, NOT_VERIFIED}:
        raise ValueError(f"Invalid verdict: {verdict!r}")

    return {
        "verdict":        verdict,
        "rationale":      str(data.get("rationale",      "")).strip(),
        "evidence_quote": str(data.get("evidence_quote", "none")).strip(),
    }


# ---------------------------------------------------------------------------
# JudgeLLM class
# ---------------------------------------------------------------------------

class JudgeLLM:
    """
    V3 Judge: strong LLM fact-verifier for UNCERTAIN atoms.

    Parameters
    ----------
    llm_client     : LLMClient configured for the strong judge model
                     (e.g. Qwen2.5-72B or an API model).
    prompt_path    : Path to judge_v1.txt system prompt.
    max_retries    : Number of JSON parse retries on malformed output.
    max_calls_per_query : Hard cap on judge invocations per query.
    guarantee_mode : 'calibrated' if end-to-end pipeline has been jointly
                     calibrated with the judge; 'empirical' otherwise.
    """

    def __init__(
        self,
        llm_client:          Optional[Any] = None,
        prompt_path:         Optional[str] = None,
        max_retries:         int           = 1,
        max_calls_per_query: int           = 20,
        guarantee_mode:      str           = "empirical",
        config:              Optional[dict] = None,
    ) -> None:
        if llm_client is None:
            from common.llm_client import LLMClient
            llm_client = LLMClient(config=config)

        self.llm                  = llm_client
        self.max_retries          = max_retries
        self.max_calls_per_query  = max_calls_per_query
        self.guarantee_mode       = guarantee_mode

        p = Path(prompt_path) if prompt_path else _DEFAULT_PROMPT_PATH
        with open(p, "r", encoding="utf-8") as f:
            self._system_prompt = f.read().strip()

        # Per-query call counter (reset by caller via reset_call_counter)
        self._call_count: int = 0

    def reset_call_counter(self) -> None:
        """Reset per-query call counter. Call at the start of each new query."""
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    def judge(
        self,
        scored_atom: ScoredAtom,
        rr:          RetrievalResult,
    ) -> JudgeResult:
        """
        Judge a single UNCERTAIN atom.

        Parameters
        ----------
        scored_atom : ScoredAtom from M6 (must have risk in UNCERTAIN band).
        rr          : RetrievalResult with the evidence chunks.

        Returns
        -------
        JudgeResult with verdict, rationale, evidence_quote, latency_ms.
        If the judge call cap is reached, returns NOT_VERIFIED immediately.
        """
        atom     = scored_atom.verified.atom
        atom_id  = atom.atom_id
        risk_val = scored_atom.risk

        # -- Call cap guard --
        if self._call_count >= self.max_calls_per_query:
            logger.warning(
                "Judge call cap (%d) reached; atom %s -> NOT_VERIFIED (capped).",
                self.max_calls_per_query, atom_id,
            )
            return JudgeResult(
                verdict=NOT_VERIFIED,
                rationale="Judge call cap reached for this query.",
                evidence_quote="none",
                atom_id=atom_id,
                risk=risk_val,
                parse_error=False,
            )

        # -- Phase 7: Format prompt (extend with COMPUTED annotation if needed) --
        evidence            = _format_evidence(rr.chunks, atom.claim)
        computed_annotation = _build_computed_annotation(atom)
        user_prompt         = _build_user_prompt(atom.claim, evidence, atom.type, computed_annotation)

        # -- Call LLM with timing --
        t0 = time.perf_counter()
        raw_response = ""
        model_id     = ""
        parse_error  = False

        for attempt in range(self.max_retries + 1):
            try:
                resp         = self.llm.chat(
                    user=user_prompt,
                    system=self._system_prompt,
                    temperature=0.0,
                )
                raw_response = resp.get("text", "")
                model_id     = resp.get("model", "")
                parsed       = _parse_judge_json(raw_response)
                break
            except Exception as exc:
                logger.warning(
                    "Judge parse error (attempt %d/%d) for atom %s: %s",
                    attempt + 1, self.max_retries + 1, atom_id, exc,
                )
                if attempt == self.max_retries:
                    parsed      = {
                        "verdict":        NOT_VERIFIED,
                        "rationale":      f"JSON parse failed after {self.max_retries+1} attempts.",
                        "evidence_quote": "none",
                    }
                    parse_error = True

        latency_ms = (time.perf_counter() - t0) * 1000.0
        self._call_count += 1

        result = JudgeResult(
            verdict=parsed["verdict"],
            rationale=parsed["rationale"],
            evidence_quote=parsed["evidence_quote"],
            atom_id=atom_id,
            risk=risk_val,
            latency_ms=round(latency_ms, 1),
            model_id=model_id,
            raw_response=raw_response,
            parse_error=parse_error,
        )

        logger.info(
            "Judge atom=%s  type=%s  verdict=%s  latency=%.0fms  risk=%.3f  "
            "calls=%d/%d  parse_error=%s",
            atom_id, atom.type, result.verdict, latency_ms, risk_val,
            self._call_count, self.max_calls_per_query, parse_error,
        )
        return result

    def judge_batch(
        self,
        uncertain_atoms: List[ScoredAtom],
        rr:              RetrievalResult,
    ) -> Dict[str, JudgeResult]:
        """
        Judge all UNCERTAIN atoms, respecting the per-query call cap.

        Returns
        -------
        Dict mapping atom_id -> JudgeResult.
        """
        results: Dict[str, JudgeResult] = {}
        for sa in uncertain_atoms:
            atom_id = sa.verified.atom.atom_id
            results[atom_id] = self.judge(sa, rr)
        return results
