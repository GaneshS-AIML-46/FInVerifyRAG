"""
bench/temporal_tags.py
------------------------
M10 Benchmark: Mark questions whose answer changed after an amendment.

These temporal-tagged questions form the drift test set used in the
drift experiment (experiments/drift/drift_exp.py).

A question is temporal if:
  1. It has ``temporal_tag=True`` (set by generate_temporal_questions), OR
  2. Its answer contains atom types DATE or RATE and the source chunk was
     superseded after a cutoff date.

Public API
----------
    from bench.temporal_tags import tag_temporal, TemporalTag, load_temporal_tags
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("bench.temporal_tags")


@dataclass
class TemporalTag:
    """Temporal tag for a benchmark question."""
    question_id:     str
    is_temporal:     bool
    amendment_date:  Optional[str]   = None   # ISO date when answer changed
    superseded_by:   Optional[str]   = None   # chunk_id that replaced source
    drift_type:      Optional[str]   = None   # "value" | "supersession" | "addition"
    notes:           str             = ""


def tag_temporal(
    questions: list,           # List[QuestionItem]
    version_graph=None,        # Optional VersionGraph from M1
    cutoff_date: Optional[str] = None,
) -> List[TemporalTag]:
    """
    Assign temporal tags to questions.

    Rules
    -----
    1. If question.temporal_tag is already True → tag with drift_type="value".
    2. If version_graph is provided and the source chunk is superseded
       after cutoff_date → tag with drift_type="supersession".
    3. If the question family is "TEMPORAL" → always tagged.
    4. Otherwise → not temporal.

    Parameters
    ----------
    questions    : List of QuestionItem objects.
    version_graph: Optional M1 VersionGraph instance.
    cutoff_date  : ISO date string (e.g. "2023-01-01").

    Returns
    -------
    List of TemporalTag objects (one per question).
    """
    tags = []
    for q in questions:
        qid        = _get_field(q, "question_id", "")
        family     = _get_field(q, "family", "")
        is_temp    = bool(_get_field(q, "temporal_tag", False))
        amend_date = _get_field(q, "amendment_date", None)
        src_chunk  = _get_field(q, "source_chunk_id", "")

        drift_type    = None
        superseded_by = None

        if family == "TEMPORAL":
            is_temp    = True
            drift_type = "value"

        elif is_temp:
            drift_type = "value"

        elif version_graph is not None:
            try:
                history = version_graph.history(src_chunk)
                for v in history:
                    if v.superseded_by and (
                        cutoff_date is None or
                        getattr(v, "effective_from", "") >= cutoff_date
                    ):
                        is_temp    = True
                        drift_type = "supersession"
                        superseded_by = v.superseded_by
                        amend_date    = getattr(v, "effective_from", None)
                        break
            except Exception as exc:
                logger.debug("version_graph.history(%s): %s", src_chunk, exc)

        tags.append(TemporalTag(
            question_id=qid,
            is_temporal=is_temp,
            amendment_date=amend_date,
            superseded_by=superseded_by,
            drift_type=drift_type,
        ))

    n_temporal = sum(1 for t in tags if t.is_temporal)
    logger.info("Temporal tags: %d/%d questions are temporal", n_temporal, len(tags))
    return tags


def _get_field(obj, field: str, default=None):
    if hasattr(obj, field):
        return getattr(obj, field, default)
    if isinstance(obj, dict):
        return obj.get(field, default)
    return default


def save_temporal_tags(tags: List[TemporalTag], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for t in tags:
            f.write(json.dumps(asdict(t)) + "\n")
    logger.info("Saved %d temporal tags to %s", len(tags), path)


def load_temporal_tags(path: str) -> List[TemporalTag]:
    tags = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                tags.append(TemporalTag(**{k: v for k, v in d.items()
                                           if k in TemporalTag.__dataclass_fields__}))
    return tags
