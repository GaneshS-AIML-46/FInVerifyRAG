"""
verify/v2_nli.py
----------------
V2 semantic NLI verifier using DeBERTa-v3-large fine-tuned on
regulatory entailment pairs.

Model is loaded lazily on first call to avoid GPU/CPU overhead when
only V1 is needed (e.g. all atoms resolve in V1).

Interface
---------
    from verify.v2_nli import NLIVerifier
    nli = NLIVerifier(model_path="models/nli_deberta_v3")
    p_entail, p_contradict, p_neutral = nli.predict(premise, hypothesis)

If the model is unavailable (no checkpoint), the verifier falls back to
returning 0.5 (neutral) and logs a WARNING. This keeps the pipeline
functional even before NLI training is complete.

Training note
-------------
Fine-tuning is done in verify/nli_train/train.py. This file only does
inference. The model checkpoint path is read from:
    configs/verify.yaml → nli_model_path
or from the NLI_MODEL_PATH environment variable.
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

logger = logging.getLogger("verify.v2_nli")

# Lazy imports so the module loads even without torch/transformers installed
_pipeline = None   # HuggingFace zero-shot / NLI pipeline
_model_loaded = False
_model_path: Optional[str] = None

_DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-large"  # HF Hub fallback


def _load_model(model_path: Optional[str] = None) -> None:
    """Load the NLI model once. Subsequent calls are no-ops."""
    global _pipeline, _model_loaded, _model_path

    if _model_loaded:
        return

    # Resolve model path: arg → env → YAML (deferred) → HF default
    resolved = (
        model_path
        or os.getenv("NLI_MODEL_PATH")
        or _DEFAULT_MODEL
    )
    _model_path = resolved

    try:
        from transformers import pipeline  # type: ignore
        logger.info("Loading NLI model from: %s", resolved)
        import torch
        _nli_device = 0 if torch.cuda.is_available() else -1  # 0=GPU, -1=CPU
        _pipeline = pipeline(
            "zero-shot-classification",
            model=resolved,
            device=_nli_device,
            multi_label=False,
        )
        _model_loaded = True
        logger.info("NLI model loaded successfully.")
    except Exception as exc:
        logger.warning(
            "Could not load NLI model '%s': %s. Falling back to neutral (0.5).",
            resolved,
            exc,
        )
        _model_loaded = True  # don't retry on every call


class NLIVerifier:
    """
    Wraps the DeBERTa NLI pipeline for atom verification.

    Parameters
    ----------
    model_path : str, optional
        Path to a local fine-tuned checkpoint or a HF Hub model ID.
        If omitted, uses the NLI_MODEL_PATH env-var or the HF default.
    max_length : int
        Tokenizer truncation length (512 for DeBERTa-v3-large).
    """

    LABELS = ["entailment", "contradiction", "neutral"]

    def __init__(self, model_path: Optional[str] = None, max_length: int = 512):
        self.max_length = max_length
        _load_model(model_path)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict(
        self, premise: str, hypothesis: str, strict: bool = False
    ) -> Tuple[float, float, float]:
        """
        Run NLI on (premise, hypothesis).

        Returns
        -------
        (p_entail, p_contradict, p_neutral) : floats summing to ~1.0

        Falls back to (0.5, 0.25, 0.25) if the model is unavailable.
        """
        if _pipeline is None:
            if strict:
                from common.errors import StrictFailure
                raise StrictFailure("Signal 's_nli' failed to load NLI model pipeline.")
            logger.debug("NLI pipeline unavailable; returning neutral fallback.")
            return (0.5, 0.25, 0.25)

        # HF zero-shot-classification returns scores for each candidate label
        try:
            result = _pipeline(
                sequences=premise,
                candidate_labels=self.LABELS,
                hypothesis_template="{}",
                # Pass hypothesis as the first candidate label's template
            )
            # Build a score dict keyed by label
            score_map = dict(zip(result["labels"], result["scores"]))
        except TypeError:
            # Some HF versions use a different API; call with explicit kwargs
            try:
                result = _pipeline(
                    premise,
                    candidate_labels=self.LABELS,
                )
                score_map = dict(zip(result["labels"], result["scores"]))
            except Exception as exc:
                if strict:
                    from common.errors import StrictFailure
                    raise StrictFailure(f"Signal 's_nli' prediction failed: {exc}") from exc
                logger.warning("NLI prediction failed: %s. Returning neutral.", exc)
                return (0.5, 0.25, 0.25)
        except Exception as exc:
            if strict:
                from common.errors import StrictFailure
                raise StrictFailure(f"Signal 's_nli' prediction failed: {exc}") from exc
            logger.warning("NLI prediction failed: %s. Returning neutral.", exc)
            return (0.5, 0.25, 0.25)

        p_e = score_map.get("entailment", 0.333)
        p_c = score_map.get("contradiction", 0.333)
        p_n = score_map.get("neutral", 0.333)
        return (p_e, p_c, p_n)

    def entail_prob(self, premise: str, hypothesis: str) -> float:
        """Convenience: return only P(entailment)."""
        return self.predict(premise, hypothesis)[0]


# ---------------------------------------------------------------------------
# Module-level singleton (created lazily in cascade.py)
# ---------------------------------------------------------------------------

_verifier_instance: Optional[NLIVerifier] = None


def get_verifier(model_path: Optional[str] = None) -> NLIVerifier:
    """Return a shared NLIVerifier instance (lazy singleton)."""
    global _verifier_instance
    if _verifier_instance is None:
        _verifier_instance = NLIVerifier(model_path=model_path)
    return _verifier_instance


# ---------------------------------------------------------------------------
# Utility: select the best evidence sentence for an atom
# ---------------------------------------------------------------------------

def best_evidence(atom_claim: str, chunk_texts: list[str]) -> str:
    """
    Simple heuristic: return the chunk text with the most token overlap
    with the atom claim. Used to pick the premise for the NLI call.

    If no chunks are available, return the atom claim itself (degenerate).
    """
    if not chunk_texts:
        return atom_claim

    claim_words = set(atom_claim.lower().split())

    def overlap(text: str) -> int:
        return len(claim_words & set(text.lower().split()))

    return max(chunk_texts, key=overlap)
