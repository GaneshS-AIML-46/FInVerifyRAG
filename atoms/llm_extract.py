import json
import re
import uuid
import logging
from typing import List, Optional
from common.schemas import Atom, ATOM_TYPES, ComputedAtomEvidence, AtomOperator
from common.llm_client import LLMClient
from atoms.regex_extract import find_span

logger = logging.getLogger("atoms.llm_extract")


# ---------------------------------------------------------------------------
# Operator normalisation map  (Design Gap 6.1)
# ---------------------------------------------------------------------------

_OPERATOR_MAP = {
    "eq":            AtomOperator.EQ,
    "=":             AtomOperator.EQ,
    "equal":         AtomOperator.EQ,
    "equals":        AtomOperator.EQ,
    "is":            AtomOperator.EQ,
    "lt":            AtomOperator.LT,
    "<":             AtomOperator.LT,
    "less than":     AtomOperator.LT,
    "below":         AtomOperator.LT,
    "lte":           AtomOperator.LTE,
    "<=":            AtomOperator.LTE,
    "at most":       AtomOperator.LTE,
    "not more than": AtomOperator.LTE,
    "gt":            AtomOperator.GT,
    ">":             AtomOperator.GT,
    "greater than":  AtomOperator.GT,
    "more than":     AtomOperator.GT,
    "gte":           AtomOperator.GTE,
    ">=":            AtomOperator.GTE,
    "at least":      AtomOperator.GTE,
    "minimum":       AtomOperator.GTE,
    "up to":         AtomOperator.UP_TO,
    "upto":          AtomOperator.UP_TO,
    "up_to":         AtomOperator.UP_TO,
    "exceeding":     AtomOperator.EXCEEDING,
    "above":         AtomOperator.EXCEEDING,
    "over":          AtomOperator.EXCEEDING,
    "between":       AtomOperator.BETWEEN,
    "approximately": AtomOperator.APPROXIMATELY,
    "around":        AtomOperator.APPROXIMATELY,
    "about":         AtomOperator.APPROXIMATELY,
    "unknown":       AtomOperator.UNKNOWN,
    "":              AtomOperator.UNKNOWN,
}


def _normalise_operator(raw: str) -> str:
    """Convert raw operator string from LLM to AtomOperator enum value."""
    return _OPERATOR_MAP.get(raw.strip().lower(), AtomOperator.UNKNOWN)


def loads_atom_json(raw_text: str):
    """
    Parse an atom JSON array. Strips fences and pulls out the first
    JSON array or object if the model wrapped it in prose.
    Raises json.JSONDecodeError if nothing parses.
    """
    text = (raw_text or "").strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(\[.*\]|\{.*\})", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(1))


def _try_float(val) -> Optional[float]:
    """Safely parse a float from an arbitrary value."""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# LLMAtomExtractor
# ---------------------------------------------------------------------------

class LLMAtomExtractor:
    """
    LLM-powered atom extractor using the extract_v1 prompt.

    Phase 2 improvements:
    ---------------------
    - Populates all Design Gap 6.1 structured semantic fields:
        subject, operator, value, unit, condition, effective_period, source_span.
    - COMPUTED atoms (Design Gap 7) receive a ``ComputedAtomEvidence`` block
      with inputs, operation, formula, constants, result and result_numeric.
    - Operator strings are normalised to AtomOperator vocabulary so that
        'up to ₹5 lakh'  -> operator=UP_TO, value='5', unit='lakh'
        'exceeding ₹5 lakh' -> operator=EXCEEDING, value='5', unit='lakh'
      are NEVER treated as equivalent.
    - A fast regex pass is run first; the LLM fills in claims and catches
      what regex misses.
    """

    def __init__(self, llm: LLMClient, prompt_path: str = "atoms/prompts/extract_v1.txt"):
        self.llm = llm
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read().strip()
        self.parse_failures = 0
        self.parse_attempts = 0

    def _request_json(self, user_prompt: str) -> str:
        kwargs = {}
        try:
            from common.experiment import load_experiment
            if (load_experiment().get("llm_extract") or {}).get("json_mode", True):
                kwargs["response_format"] = {"type": "json_object"}
        except Exception:
            kwargs["response_format"] = {"type": "json_object"}
        response = self.llm.chat(
            user=user_prompt,
            system=self.system_prompt,
            temperature=0.0,
            logprobs=False,
            **kwargs,
        )
        return response["text"].strip()

    def extract(self, answer_text: str) -> List[Atom]:
        """
        Extract atoms from a generated answer using the LLM.
        Returns a list of Atom objects with fully populated structured fields.
        """
        # Short-circuit: if the model said "not found", return nothing
        if "not found in evidence" in answer_text.lower():
            return []

        user_prompt = f"ANSWER:\n{answer_text}\n\nReturn a JSON array of atoms."
        self.parse_attempts += 1
        raw_text = self._request_json(user_prompt)

        try:
            raw_atoms = loads_atom_json(raw_text)
        except Exception as exc:
            logger.warning("Invalid JSON from atom extraction. Retrying once. Error: %s", exc)
            repair_prompt = (
                "The following text was supposed to be a JSON array of atoms. "
                "It failed to parse with the following error:\n"
                f"{str(exc)}\n\n"
                "Return only a valid JSON array, with no prose.\n\n"
                + raw_text
            )
            repaired = self._request_json(repair_prompt)
            try:
                raw_atoms = loads_atom_json(repaired)
            except Exception:
                self.parse_failures += 1
                logger.error(
                    "Atom JSON parse failed after repair (%d/%d).",
                    self.parse_failures, self.parse_attempts,
                )
                return []

        if isinstance(raw_atoms, dict):
            raw_atoms = raw_atoms.get("atoms") or raw_atoms.get("items") or []

        atoms = []
        seen_texts = set()

        for i, item in enumerate(raw_atoms):
            atom_type = item.get("type", "").strip().upper()
            text = item.get("text", "").strip()
            claim = item.get("claim", "").strip()
            cited_chunk = item.get("cited_chunk") or None

            # Validate atom type first
            if atom_type not in ATOM_TYPES:
                logger.warning(f"Unknown atom type '{atom_type}' — skipping.")
                continue

            # Skip empty text
            if not text:
                continue

            # Deduplicate by (type, text)
            key = (atom_type, text)
            if key in seen_texts:
                continue
            seen_texts.add(key)

            # Align span: find exact occurrence in the answer
            span = find_span(answer_text, text)

            # ── Phase 2: Design Gap 6.1 — structured semantic fields ─────────
            subject          = item.get("subject") or None
            raw_operator     = str(item.get("operator", "unknown")).lower()
            value            = item.get("value") or None
            unit             = item.get("unit") or None
            condition        = item.get("condition") or None
            effective_period = item.get("effective_period") or None
            source_span      = item.get("source_span") or None

            # Normalise operator so UP_TO != EXCEEDING at schema level
            operator = _normalise_operator(raw_operator)

            # ── Phase 2: Design Gap 7 — COMPUTED atom evidence ────────────────
            computed: Optional[ComputedAtomEvidence] = None
            if atom_type == "COMPUTED":
                comp_data = item.get("computed") or {}
                if isinstance(comp_data, dict):
                    computed = ComputedAtomEvidence(
                        inputs=comp_data.get("inputs", [text]),
                        operation=comp_data.get("operation", "unknown"),
                        formula=comp_data.get("formula", claim or text),
                        constants=comp_data.get("constants", {}),
                        result=comp_data.get("result", value or text),
                        result_numeric=_try_float(comp_data.get("result_numeric")),
                        input_evidence=comp_data.get(
                            "input_evidence",
                            [cited_chunk] if cited_chunk else [],
                        ),
                        verified=False,   # deterministic check not yet run
                    )
                else:
                    # Minimal fallback evidence when LLM omits the computed block
                    computed = ComputedAtomEvidence(
                        inputs=[text],
                        operation="derived",
                        formula=claim or text,
                        result=value or text,
                    )
                logger.debug(
                    "COMPUTED atom '%s': op=%s formula='%s' result='%s'",
                    text[:40], computed.operation, computed.formula[:40], computed.result,
                )

            atoms.append(Atom(
                atom_id=f"atom_{uuid.uuid4().hex[:8]}",
                type=atom_type,
                text=text,
                # Phase 2 structured fields
                subject=subject,
                operator=operator,
                value=value,
                unit=unit,
                condition=condition,
                effective_period=effective_period,
                source_span=source_span,
                # legacy NLI / citation fields
                claim=claim,
                cited_chunk=cited_chunk,
                span=span,
                # COMPUTED evidence block
                computed=computed,
            ))

        logger.info(f"Extracted {len(atoms)} atoms from answer ({len(answer_text)} chars).")
        return atoms
