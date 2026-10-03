"""
scripts/select_seed_chunks.py
------------------------------
Phase C-pre: Select seed chunks for question drafting.

Loads the BM25 index, applies quality filters, deduplicates by title-series
(keeping newest version), stratifies by regulator × split × family_hint,
and writes data/seed_chunks.csv.
"""
from __future__ import annotations

import argparse
import csv
import logging
import pickle
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("select_seed_chunks")

ROOT = Path(__file__).resolve().parent.parent


# ── Stub unpickler ─────────────────────────────────────────────────────────────

class _ChunkStub:
    __slots__ = ("text", "chunk_id", "doc_id", "metadata", "breadcrumb_text")

    def __init__(self, **kw):
        for s in self.__slots__:
            setattr(self, s, kw.get(s, ""))

    def __setstate__(self, state):
        for s in self.__slots__:
            setattr(self, s, state.get(s, ""))


class _StubUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            return _ChunkStub


# ── Title normalisation (from assign_doc_splits.py) ──────────────────────────

_MONTHS = (
    r"(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december|"
    r"jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
)


def normalize_title(title: str) -> str:
    t: str = unicodedata.normalize("NFKD", title)
    t = t.encode("ascii", "ignore").decode("ascii").lower()
    t = re.sub(r"\b(?:rbi|sebi)[/\-][\w/\-\.]{3,}", "", t)
    _date_qual = r"\b(?:dated?|as\s+on|updated?|revised?|amended?)\s+"
    t = re.sub(_date_qual + rf"{_MONTHS}\s+\d{{1,2}},?\s+\d{{4}}", "", t, flags=re.I)
    t = re.sub(_date_qual + rf"{_MONTHS}\s+\d{{4}}", "", t, flags=re.I)
    t = re.sub(_date_qual + r"\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}", "", t, flags=re.I)
    t = re.sub(r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b", "", t)
    t = re.sub(rf"\b{_MONTHS}\s+\d{{1,2}},?\s+\d{{4}}\b", "", t, flags=re.I)
    t = re.sub(rf"\b{_MONTHS}\s+\d{{4}}\b", "", t, flags=re.I)
    t = re.sub(r"\b20\d{2}\b", "", t)
    t = re.sub(
        r"\(\s*(?:amendment|corrigendum|addendum|erratum|"
        r"no\.?\s*\d+|part\s*[ivx\d]+|vol\.?\s*\d+|"
        r"serial\s*\d+|s\.?\s*no\.?\s*\d+)\s*\)",
        " ", t, flags=re.I,
    )
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ── Config ────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    cfg_path = ROOT / "configs" / "experiment.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_doc_assignment(path: Path) -> dict[str, str]:
    """Returns {doc_id: split}."""
    if not path.exists():
        raise FileNotFoundError(f"doc_assignment.csv not found: {path}")
    result: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            result[row["doc_id"].strip()] = row.get("split", "").strip()
    return result


def _load_inventory(path: Path) -> dict[str, dict]:
    """Returns {doc_id: {regulator, doc_type, title, issue_date}}."""
    if not path.exists():
        raise FileNotFoundError(f"doc_inventory.csv not found: {path}")
    result: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            result[row["doc_id"].strip()] = {
                "regulator": row.get("regulator", "").strip(),
                "doc_type": row.get("doc_type", "").strip(),
                "title": row.get("title", row.get("title_raw", "")).strip(),
                "issue_date": row.get("issue_date", "").strip(),
            }
    return result


# ── Newest-version detection ──────────────────────────────────────────────────

def _build_newest_doc_ids(inventory: dict[str, dict]) -> set[str]:
    """
    Per title-series (normalized title), keep only the doc with the max issue_date.
    Returns set of doc_ids that are the newest in their series.
    """
    series: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for doc_id, info in inventory.items():
        norm = normalize_title(info.get("title", ""))
        if not norm:
            norm = doc_id
        series[norm].append((info.get("issue_date", ""), doc_id))

    newest: set[str] = set()
    for norm, entries in series.items():
        # Sort by issue_date descending; take first
        entries_sorted = sorted(entries, key=lambda x: x[0], reverse=True)
        newest.add(entries_sorted[0][1])
    return newest


# ── Skip predicates ───────────────────────────────────────────────────────────

_TOC_LINE_PAT = re.compile(r"^\s*\.{3,}\s*\d+\s*$|^\s*\d+\s*$")


def _is_toc(text: str) -> bool:
    lines = text.splitlines()
    if not lines:
        return False
    short_lines = [l for l in lines if len(l.strip()) <= 60]
    if len(short_lines) < 3:
        return False
    toc_lines = sum(1 for l in short_lines if _TOC_LINE_PAT.match(l) or l.rstrip().endswith("..."))
    return (toc_lines / len(short_lines)) >= 0.60


_LETTERHEAD_PAT = re.compile(
    r"(?i)(reserve bank of india|securities and exchange board|भारतीय|रिज़र्व)"
)
_NAME_PAT = re.compile(r"^\s*[A-Z][a-z]+\s+[A-Z][a-z]+")
_TABLE_ROW_PAT = re.compile(r"\|")


def _count_skip_reason(
    text: str,
    doc_id: str,
    inventory: dict,
    doc_assignment: dict,
    newest_doc_ids: set,
    doc_selected_count: dict[str, int],
    min_ascii_ratio: float,
) -> Optional[str]:
    """Return the first skip reason, or None if the chunk should be kept."""
    if len(text.strip()) < 300:
        return "skip_too_short"

    ascii_count = sum(1 for c in text if ord(c) < 128)
    ascii_ratio = ascii_count / max(len(text), 1)
    if ascii_ratio < min_ascii_ratio:
        return "skip_low_ascii"

    if _is_toc(text):
        return "skip_toc"

    if _LETTERHEAD_PAT.search(text[:200]) and len(text) < 600:
        return "skip_letterhead"

    lines = text.splitlines()
    name_lines = sum(1 for l in lines if _NAME_PAT.match(l))
    if name_lines >= 5 and len(text) < 800:
        return "skip_names_addresses"

    if lines:
        pipe_lines = sum(1 for l in lines if l.count("|") >= 3)
        if pipe_lines / len(lines) > 0.5:
            return "skip_table_rows"

    if doc_id not in inventory:
        return "skip_not_in_inventory"

    if doc_id not in doc_assignment:
        return "skip_excluded_doc"

    if doc_id not in newest_doc_ids:
        return "skip_old_version"

    if doc_selected_count.get(doc_id, 0) >= 2:
        return "skip_over_quota"

    return None


# ── Family hint ───────────────────────────────────────────────────────────────

def _family_hint(text: str) -> str:
    if re.search(r"(?i)\bpercent|per cent|\brate\b|\blimit\b|\bratio\b|\bthreshold\b", text):
        return "RATE_THRESHOLD"
    if re.search(r"(?i)\bsection\b|\bregulation\b|\bpara\b|\bclause\b|\bprovision\b", text):
        return "SECTION"
    if re.search(r"(?i)within \d+ days|effective from|by \d{1,2}[/-]\d{1,2}|deadline|transitional", text):
        return "DATE_TIMELINE"
    if re.search(r"(?i)applicable to|shall apply|are required to|entities|institutions", text):
        return "ENTITY_APPLICABILITY"
    return "GENERAL"


# ── Main ──────────────────────────────────────────────────────────────────────

def select_seed_chunks(cfg: dict | None = None, out_path: Path | None = None) -> list[dict]:
    if cfg is None:
        cfg = _load_config()

    seed_cfg = cfg.get("seed_chunks", {})
    target = seed_cfg.get("target", 1200)
    min_ascii_ratio = seed_cfg.get("min_ascii_ratio", 0.70)
    rng_seed = cfg.get("seed", 42)

    if out_path is None:
        out_path = ROOT / "data" / "seed_chunks.csv"

    # Load BM25 index
    index_path = ROOT / ".bm25_index.pkl"
    if not index_path.exists():
        raise FileNotFoundError(f"BM25 index not found: {index_path}")
    logger.info("Loading BM25 index from %s ...", index_path)
    with open(index_path, "rb") as f:
        raw = _StubUnpickler(f).load()
    chunks = raw["chunks"]
    logger.info("Loaded %d total chunks", len(chunks))

    # Load splits and inventory
    split_path = ROOT / "data" / "splits" / "doc_assignment.csv"
    inv_path = ROOT / "data" / "doc_inventory.csv"
    doc_assignment = _load_doc_assignment(split_path)
    inventory = _load_inventory(inv_path)

    # Build newest-doc set
    newest_doc_ids = _build_newest_doc_ids(inventory)
    logger.info("Newest doc IDs: %d", len(newest_doc_ids))

    # Filter chunks
    skip_counts: Counter = Counter()
    candidates: list[dict] = []
    doc_selected_count: dict[str, int] = {}

    for chunk in chunks:
        text = (getattr(chunk, "text", "") or "").strip()
        doc_id = getattr(chunk, "doc_id", "")
        chunk_id = getattr(chunk, "chunk_id", "")
        metadata = getattr(chunk, "metadata", {}) or {}

        reason = _count_skip_reason(
            text, doc_id, inventory, doc_assignment,
            newest_doc_ids, doc_selected_count, min_ascii_ratio,
        )
        if reason:
            skip_counts[reason] += 1
            continue

        # Accept
        doc_selected_count[doc_id] = doc_selected_count.get(doc_id, 0) + 1

        inv_info = inventory[doc_id]
        regulator = inv_info.get("regulator") or metadata.get("issuer", "UNKNOWN")
        split = doc_assignment.get(doc_id, "")
        doc_title = inv_info.get("title", "")
        issue_date = inv_info.get("issue_date", "")
        ascii_count = sum(1 for c in text if ord(c) < 128)
        ascii_ratio = ascii_count / max(len(text), 1)
        hint = _family_hint(text)

        candidates.append({
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "regulator": regulator,
            "split": split,
            "doc_title": doc_title,
            "issue_date": issue_date,
            "text_length": len(text),
            "ascii_ratio": round(ascii_ratio, 4),
            "family_hint": hint,
            "chunk_text": text,
        })

    logger.info("Candidate pool before sampling: %d", len(candidates))

    # Stratify sampling if over target
    rng = random.Random(rng_seed)
    if len(candidates) > target:
        # Group by stratum
        strata: dict[tuple, list] = defaultdict(list)
        for c in candidates:
            key = (c["regulator"], c["split"], c["family_hint"])
            strata[key].append(c)
        # Proportional sample
        total = len(candidates)
        sampled: list[dict] = []
        for key, items in strata.items():
            n = max(1, round(target * len(items) / total))
            n = min(n, len(items))
            sampled.extend(rng.sample(items, n))
        # If still over/under due to rounding, trim/pad
        if len(sampled) > target:
            rng.shuffle(sampled)
            sampled = sampled[:target]
        candidates = sampled

    logger.info("Final pool: %d chunks", len(candidates))

    # Write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "chunk_id", "doc_id", "regulator", "split", "doc_title",
        "issue_date", "text_length", "ascii_ratio", "family_hint", "chunk_text",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(candidates)
    logger.info("Wrote %d seed chunks to %s", len(candidates), out_path)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  SKIP COUNTS")
    print(f"{'='*65}")
    for reason, cnt in sorted(skip_counts.items()):
        print(f"  {reason:<35}: {cnt:>6}")
    print(f"  {'TOTAL SKIPPED':<35}: {sum(skip_counts.values()):>6}")
    print(f"  {'TOTAL ACCEPTED':<35}: {len(candidates):>6}")

    # Pool table: regulator × split × family_hint
    print(f"\n{'='*65}")
    print("  POOL COUNTS (regulator × split × family_hint)")
    print(f"{'='*65}")
    pool_counts: Counter = Counter()
    for c in candidates:
        pool_counts[(c["regulator"], c["split"], c["family_hint"])] += 1
    print(f"  {'Regulator':<10} {'Split':<8} {'Family':<22} {'Count':>6}")
    print(f"  {'-'*10} {'-'*8} {'-'*22} {'-'*6}")
    for (reg, split, hint), cnt in sorted(pool_counts.items()):
        print(f"  {reg:<10} {split:<8} {hint:<22} {cnt:>6}")

    # 5 random previews
    print(f"\n{'='*65}")
    print("  5 RANDOM CHUNK PREVIEWS (first 200 chars)")
    print(f"{'='*65}")
    previews = rng.sample(candidates, min(5, len(candidates)))
    for i, c in enumerate(previews, 1):
        preview = c["chunk_text"][:200].replace("\n", " ")
        # Encode safely for Windows console
        def _safe(s): return s.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8")
        print(f"\n  [{i}] chunk_id: {_safe(c['chunk_id'])}")
        print(f"      doc_id  : {_safe(c['doc_id'])}")
        print(f"      hint    : {c['family_hint']}  split={c['split']}  reg={c['regulator']}")
        print(f"      preview : {_safe(preview)}")
    print(f"\n{'='*65}\n")

    return candidates


def main() -> None:
    cfg = _load_config()
    select_seed_chunks(cfg)


if __name__ == "__main__":
    main()
