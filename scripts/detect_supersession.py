"""
scripts/detect_supersession.py
--------------------------------
Phase B: Detect supersession relationships in regulatory documents.

Searches document titles and (optionally) page-1 text for phrases that
indicate a document has been superseded, withdrawn, or updated.

Outputs data/supersession_proposed.csv for HUMAN REVIEW.
Does NOT modify data/doc_inventory.csv automatically.

Phrases detected
----------------
  superseded by X           → proposed_superseded_by = X
  in supersession of X      → proposed_superseded_by = X
  stands superseded / withdrawn / hereby withdrawn
  supersedes X              → the current doc supersedes X
  (Updated as on DATE)      → proposed_effective_to = DATE
  (Amended up to DATE)      → proposed_effective_to = DATE

For each hit the script also tries to match the referenced document name
against the inventory using normalized-title token overlap.

Usage
-----
    python scripts/detect_supersession.py [--inventory PATH] [--out PATH]
                                          [--title-only] [--max-docs N]

    --title-only  : scan only inventory titles (fast, ~1 s for 2500 docs)
    --max-docs N  : limit PDF scanning to first N docs (for testing)
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Optional

import yaml

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("detect_supersession")

ROOT = Path(__file__).resolve().parent.parent


# ── Supersession phrase patterns ──────────────────────────────────────────────

# Each entry: (phrase_type, compiled_regex)
# Group 1 (if present) = referenced document name or date qualifier
_PATTERNS = [
    ("superseded_by",
     re.compile(r"superseded\s+by\s+(.{5,150}?)(?:\.|,\s*$|\n|$)", re.I)),
    ("in_supersession_of",
     re.compile(r"in\s+supersession\s+of\s+(.{5,150}?)(?:\.|,\s*$|\n|$)", re.I)),
    ("stands_superseded",
     re.compile(r"(?:stands?|is|are|hereby)\s+superseded\b", re.I)),
    ("stands_withdrawn",
     re.compile(r"(?:stands?|is|are|hereby)\s+withdrawn\b", re.I)),
    ("supersedes",
     re.compile(r"\bsupersedes?\s+(.{5,150}?)(?:\.|,\s*$|\n|$)", re.I)),
    ("updated_as_on",
     re.compile(
         r"\(\s*(?:updated|amended|revised)\s+(?:as\s+on|up\s+to|upto|till)\s+"
         r"([\w\s,./\-]+?)\s*\)",
         re.I,
     )),
    ("amended_up_to",
     re.compile(
         r"\(\s*(?:amended|revised)\s+up\s+to\s+([\w\s,./\-]+?)\s*\)",
         re.I,
     )),
]

_DATE_FMTS_PARSE = [
    "%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y",
    "%d %B %Y", "%B %d, %Y", "%B %d %Y",
    "%d %b %Y",  "%b %d, %Y",
]

_DATE_EXTRACT_RE = re.compile(
    r"(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})|"
    r"(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{4})|"
    r"((?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{1,2},?\s+\d{4})|"
    r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4})",
    re.I,
)


def _parse_date(raw: str) -> Optional[str]:
    from datetime import datetime
    raw = raw.strip().rstrip(",")
    for fmt in _DATE_FMTS_PARSE:
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None


def _extract_date(text: str) -> Optional[str]:
    for m in _DATE_EXTRACT_RE.finditer(text):
        raw = next(g for g in m.groups() if g)
        parsed = _parse_date(raw)
        if parsed:
            return parsed
    return None


# ── Title normalisation (mirrors assign_doc_splits.py) ────────────────────────

_MONTHS = (
    r"(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december|"
    r"jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
)


def _normalize_title(title: str) -> str:
    t = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii").lower()
    t = re.sub(r"\b(?:rbi|sebi)[/\-][\w/\-\.]{3,}", "", t)
    _dq = r"\b(?:dated?|as\s+on|updated?|revised?|amended?)\s+"
    t = re.sub(_dq + rf"{_MONTHS}\s+\d{{1,2}},?\s+\d{{4}}", "", t, flags=re.I)
    t = re.sub(_dq + rf"{_MONTHS}\s+\d{{4}}", "", t, flags=re.I)
    t = re.sub(_dq + r"\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}", "", t, flags=re.I)
    t = re.sub(r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b", "", t)
    t = re.sub(rf"\b{_MONTHS}\s+\d{{1,2}},?\s+\d{{4}}\b", "", t, flags=re.I)
    t = re.sub(rf"\b{_MONTHS}\s+\d{{4}}\b", "", t, flags=re.I)
    t = re.sub(r"\b20\d{2}\b", "", t)
    t = re.sub(
        r"\(\s*(?:amendment|corrigendum|addendum|erratum|"
        r"no\.?\s*\d+|part\s*[ivx\d]+|vol\.?\s*\d+)\s*\)",
        " ", t, flags=re.I,
    )
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ── Inventory-doc matching ────────────────────────────────────────────────────

def _find_matching_doc(
    extracted_text: str,
    norm_to_doc: dict[str, str],
) -> tuple[Optional[str], str]:
    """
    Try to match *extracted_text* (a referenced doc name) to an inventory doc_id.

    Returns (doc_id_or_None, confidence_string).
    Confidence strings: "exact_norm", "token_overlap_0.XX", "none".
    """
    if not extracted_text.strip():
        return None, "none"

    norm_ext = _normalize_title(extracted_text)
    if not norm_ext:
        return None, "none"

    # Exact normalized match
    if norm_ext in norm_to_doc:
        return norm_to_doc[norm_ext], "exact_norm"

    # Token overlap (Jaccard-like)
    ext_tokens = set(norm_ext.split())
    best_doc_id: Optional[str] = None
    best_score = 0.0
    for norm_t, doc_id in norm_to_doc.items():
        t_tokens = set(norm_t.split())
        if not t_tokens or not ext_tokens:
            continue
        inter = ext_tokens & t_tokens
        score = len(inter) / (len(ext_tokens | t_tokens))
        if score > best_score and score >= 0.55:
            best_score = score
            best_doc_id = doc_id

    if best_doc_id:
        return best_doc_id, f"token_overlap_{best_score:.2f}"
    return None, "none"


# ── PDF page-1 extraction ─────────────────────────────────────────────────────

def _extract_page1(pdf_path: Path) -> str:
    try:
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz  # type: ignore
        doc = fitz.open(str(pdf_path))
        text = doc[0].get_text("text") if len(doc) > 0 else ""
        doc.close()
        return text[:3000]
    except Exception as exc:
        logger.debug("PDF parse failed for %s: %s", pdf_path.name, exc)
        return ""


# ── Phrase scanning ───────────────────────────────────────────────────────────

def _scan_text(text: str, source: str) -> list[dict]:
    """Return list of hit-dicts for all detected supersession phrases in *text*."""
    hits = []
    for phrase_type, pat in _PATTERNS:
        for m in pat.finditer(text):
            captured = ""
            if m.lastindex and m.lastindex >= 1:
                captured = (m.group(1) or "").strip()
            # Context window
            start = max(0, m.start() - 40)
            end   = min(len(text), m.end() + 100)
            ctx   = text[start:end].replace("\n", " ").strip()
            # Date extraction for "updated as on" / "amended up to"
            proposed_date = ""
            if phrase_type in ("updated_as_on", "amended_up_to"):
                proposed_date = _extract_date(captured) or ""
            hits.append({
                "phrase_type":     phrase_type,
                "captured":        captured[:200],
                "context":         ctx[:300],
                "proposed_date":   proposed_date,
                "source":          source,
            })
    return hits


# ── Main ──────────────────────────────────────────────────────────────────────

def run(
    inventory_path: Path,
    out_path: Path,
    scan_pdf: bool = True,
    max_docs: Optional[int] = None,
) -> None:
    # Load inventory
    inventory: list[dict] = []
    with open(inventory_path, newline="", encoding="utf-8") as f:
        inventory = list(csv.DictReader(f))
    logger.info("Loaded %d docs from inventory", len(inventory))

    if max_docs:
        logger.info("Limiting PDF scan to first %d docs (--max-docs)", max_docs)

    # Build normalised-title → doc_id lookup
    norm_to_doc: dict[str, str] = {}
    for r in inventory:
        norm = _normalize_title(r.get("title", "") or "")
        if norm:
            norm_to_doc[norm] = r["doc_id"]

    results: list[dict] = []
    n_title_hits = 0
    n_page1_hits = 0
    pdf_scanned  = 0

    for i, row in enumerate(inventory):
        doc_id      = row["doc_id"]
        title       = row.get("title", "") or ""
        source_file = row.get("source_file", "")

        if (i + 1) % 300 == 0:
            logger.info("Progress: %d/%d scanned", i + 1, len(inventory))

        # ── Title scan ─────────────────────────────────────────────────────
        title_hits = _scan_text(title, "title")
        n_title_hits += len(title_hits)

        # ── Page-1 scan ────────────────────────────────────────────────────
        page1_hits: list[dict] = []
        if scan_pdf and (max_docs is None or pdf_scanned < max_docs):
            pdf_path = ROOT / source_file if source_file else None
            if pdf_path and pdf_path.exists():
                text = _extract_page1(pdf_path)
                if text:
                    page1_hits = _scan_text(text, "page1_text")
                pdf_scanned += 1
                n_page1_hits += len(page1_hits)

        all_hits = title_hits + page1_hits

        for hit in all_hits:
            matched_doc_id, match_conf = _find_matching_doc(
                hit["captured"], norm_to_doc)
            results.append({
                "doc_id":                doc_id,
                "regulator":             row.get("regulator", ""),
                "title":                 title,
                "issue_date":            row.get("issue_date", ""),
                "effective_from":        row.get("effective_from", ""),
                "phrase_type":           hit["phrase_type"],
                "phrase_source":         hit["source"],
                "extracted_context":     hit["context"],
                "proposed_effective_to": hit["proposed_date"],
                "proposed_superseded_by": hit["captured"],
                "matched_doc_id":        matched_doc_id or "",
                "match_confidence":      match_conf,
            })

    # ── Write output ──────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "doc_id", "regulator", "title", "issue_date", "effective_from",
        "phrase_type", "phrase_source", "extracted_context",
        "proposed_effective_to", "proposed_superseded_by",
        "matched_doc_id", "match_confidence",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    # ── Summary ───────────────────────────────────────────────────────────────
    from collections import Counter
    type_counts: Counter = Counter(r["phrase_type"] for r in results)
    reg_counts:  Counter = Counter(r["regulator"]   for r in results)

    print(f"\n{'='*62}")
    print("  Supersession Detection Results")
    print(f"{'='*62}")
    print(f"  Documents in inventory : {len(inventory)}")
    print(f"  Title hits             : {n_title_hits}")
    if scan_pdf:
        print(f"  PDFs scanned           : {pdf_scanned}")
        print(f"  Page-1 hits            : {n_page1_hits}")
    print(f"  Total rows proposed    : {len(results)}")

    print("\n  By phrase type:")
    for pt, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {pt:<30}: {cnt:>4}")

    print("\n  By regulator:")
    for reg, cnt in sorted(reg_counts.items(), key=lambda x: -x[1]):
        print(f"    {reg:<10}: {cnt:>4}")

    matched = [r for r in results if r["matched_doc_id"]]
    with_date = [r for r in results if r["proposed_effective_to"]]
    print(f"\n  Proposed superseded_by with matched inventory doc: {len(matched)}")
    print(f"  Proposed effective_to date extracted:              {len(with_date)}")

    # Show top proposed effective_to (from "updated as on")
    if with_date:
        print("\n  Rows with proposed_effective_to (first 10):")
        print(f"  {'doc_id':<45} {'phrase_type':<20} {'proposed_effective_to'}")
        print("  " + "-" * 80)
        for r in with_date[:10]:
            print(f"  {r['doc_id'][:43]:<45} {r['phrase_type']:<20} {r['proposed_effective_to']}")

    # Show top matched supersessions
    if matched:
        print("\n  Rows with matched inventory doc (first 10):")
        print(f"  {'doc_id':<40} → {'matched_doc_id':<40} {'confidence'}")
        print("  " + "-" * 100)
        for r in matched[:10]:
            print(f"  {r['doc_id'][:38]:<40}   {r['matched_doc_id'][:38]:<40} {r['match_confidence']}")

    print(f"\n  Output: {out_path}")
    print(f"  Review this file and add effective_to / superseded_by to doc_inventory.csv by hand.")
    print(f"{'='*62}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    cfg_path = ROOT / "configs" / "experiment.yaml"
    try:
        with open(cfg_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


def main() -> None:
    cfg   = _load_config()
    paths = cfg.get("paths", {})
    default_inv = str(ROOT / paths.get("doc_inventory", "data/doc_inventory.csv"))
    default_out = str(ROOT / "data" / "supersession_proposed.csv")

    parser = argparse.ArgumentParser(
        description="Detect supersession phrases in regulatory documents.")
    parser.add_argument("--inventory",  default=default_inv)
    parser.add_argument("--out",        default=default_out)
    parser.add_argument(
        "--title-only", action="store_true",
        help="Scan only inventory titles (fast); skip page-1 PDF extraction.",
    )
    parser.add_argument(
        "--max-docs", type=int, default=None,
        metavar="N", help="Limit PDF scanning to first N documents (for testing).",
    )
    args = parser.parse_args()

    inv_path = Path(args.inventory)
    if not inv_path.exists():
        print(f"ERROR: inventory not found: {inv_path}", file=sys.stderr)
        sys.exit(1)

    run(
        inventory_path=inv_path,
        out_path=Path(args.out),
        scan_pdf=not args.title_only,
        max_docs=args.max_docs,
    )


if __name__ == "__main__":
    main()
