"""
scripts/make_doc_inventory.py
-----------------------------
Phase B: Scan the PDF folder and build data/doc_inventory.csv.

Sources (priority order):
  1. manifest.json  per sub-corpus  → doc_type, regulator, sha256, local_path
  2. index CSV per sub-corpus        → title, issue_date
  3. data/doc_dates.csv             → effective_from, effective_to
  4. PDF page-1 text extraction     → fallback title / date

Output columns:
  doc_id, regulator, doc_type, title, issue_date, effective_from, effective_to,
  superseded_by, source_file, needs_review

doc_id is stable: {REGULATOR}_{stem}  (stem = PDF filename without .pdf).
The stem itself already contains a content-hash suffix so it is stable.

Usage:
    python scripts/make_doc_inventory.py [--pdf_dir PATH] [--out PATH]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import yaml

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("make_doc_inventory")

ROOT = Path(__file__).resolve().parent.parent

# ── Date parsing ──────────────────────────────────────────────────────────────

_DATE_FMTS = [
    "%B %d, %Y",       # September 29, 2026
    "%b %d, %Y",       # Sep 09, 2026
    "%d-%b-%Y",        # 01-Jan-2024
    "%d/%m/%Y",        # 01/01/2024
    "%Y-%m-%d",        # 2024-01-01
    "%d %B %Y",        # 01 January 2024
    "%d %b %Y",        # 01 Jan 2024
]

_YEAR_RE = re.compile(r"\b(20\d{2})\b")


def parse_date(raw: Optional[str]) -> tuple[Optional[str], bool]:
    """
    Try to parse raw into YYYY-MM-DD.
    Returns (iso_date_or_None, needs_review).
    needs_review=True when date is missing or ambiguous.
    """
    if not raw or not raw.strip():
        return None, True

    raw = raw.strip().rstrip(",")

    from datetime import datetime

    # Try known formats
    for fmt in _DATE_FMTS:
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.strftime("%Y-%m-%d"), False
        except ValueError:
            pass

    # Last resort: extract year only
    m = _YEAR_RE.search(raw)
    if m:
        return f"{m.group(1)}-01-01", True  # year-only → needs_review

    return None, True


# ── doc_type inference ────────────────────────────────────────────────────────

_DOCTYPE_PATTERNS = [
    (re.compile(r"master\s+(circular|direction)", re.I), "master_circular"),
    (re.compile(r"master", re.I), "master_circular"),
    (re.compile(r"direction", re.I), "direction"),
    (re.compile(r"notification", re.I), "notification"),
    (re.compile(r"circular", re.I), "circular"),
    (re.compile(r"regulation", re.I), "regulation"),
    (re.compile(r"press\s+release", re.I), "press_release"),
    (re.compile(r"report", re.I), "report"),
    (re.compile(r"guideline", re.I), "guideline"),
    (re.compile(r"amendment", re.I), "amendment"),
]


def infer_doc_type(manifest_type: str, instrument_id: str, title: str) -> str:
    if manifest_type and manifest_type not in ("", "unknown"):
        return manifest_type.lower().replace(" ", "_")
    search_text = f"{instrument_id} {title}"
    for pat, dtype in _DOCTYPE_PATTERNS:
        if pat.search(search_text):
            return dtype
    return "document"


# ── Index CSV loaders ─────────────────────────────────────────────────────────

def _load_rbi_index(path: Path) -> dict:
    """Returns {stem: {title, issue_date, instrument_id, doc_type}} keyed by file stem."""
    result = {}
    if not path.exists():
        logger.warning("RBI index not found: %s", path)
        return result
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lpath = row.get("local_pdf_path", "")
            if not lpath:
                continue
            stem = Path(lpath).stem
            iid = row.get("instrument_id", "")
            title = row.get("title", "").strip()
            raw_date = row.get("issue_date", "").strip().strip('"')
            result[stem] = {
                "title": title,
                "raw_date": raw_date,
                "instrument_id": iid,
            }
    logger.info("Loaded %d entries from RBI index %s", len(result), path.name)
    return result


def _load_sebi_index(path: Path, doc_type: str) -> dict:
    """Returns {stem: {title, issue_date, doc_type}} keyed by file stem."""
    result = {}
    if not path.exists():
        logger.warning("SEBI index not found: %s", path)
        return result
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lpath = row.get("local_path", "")
            if not lpath:
                continue
            stem = Path(lpath).stem
            raw_date = row.get("date", "").strip().strip('"')
            result[stem] = {
                "title": row.get("title", "").strip(),
                "raw_date": raw_date,
                "doc_type": doc_type,
            }
    logger.info("Loaded %d entries from SEBI index %s", len(result), path.name)
    return result


def _load_doc_dates(path: Path) -> dict:
    """Returns {doc_id: {effective_from, effective_to}} from data/doc_dates.csv."""
    result = {}
    if not path.exists():
        return result
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            did = row.get("doc_id", "").strip()
            if did:
                result[did] = {
                    "effective_from": row.get("effective_from", "").strip() or None,
                    "effective_to":   row.get("effective_to", "").strip() or None,
                }
    logger.info("Loaded %d effective-date entries from %s", len(result), path.name)
    return result


def _load_manifest(path: Path) -> dict:
    """Returns {stem: {doc_type, sha256, fetched_at}} from manifest.json."""
    result = {}
    if not path.exists():
        return result
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for _url, info in data.items():
        lpath = info.get("local_path", "")
        if not lpath:
            continue
        stem = Path(lpath).stem
        result[stem] = {
            "doc_type":  info.get("doc_type", ""),
            "sha256":    info.get("sha256", ""),
            "fetched_at": info.get("fetched_at", ""),
        }
    logger.info("Loaded %d entries from manifest %s", len(result), path.name)
    return result


# ── PDF page-1 fallback ───────────────────────────────────────────────────────

def _extract_page1(pdf_path: Path) -> str:
    """Extract first page text from PDF. Returns '' on failure."""
    try:
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz  # type: ignore
        doc = fitz.open(str(pdf_path))
        text = doc[0].get_text("text") if len(doc) > 0 else ""
        doc.close()
        return text[:2000]
    except Exception as exc:
        logger.debug("PDF parse failed for %s: %s", pdf_path.name, exc)
        return ""


def _extract_date_from_text(text: str) -> tuple[Optional[str], bool]:
    """Try to find a date in page-1 text."""
    patterns = [
        r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+20\d{2})\b",
        r"\b((?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{1,2},?\s+20\d{2})\b",
        r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+20\d{2})\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if m:
            iso, nr = parse_date(m.group(1))
            if iso:
                return iso, nr
    return None, True


def _extract_title_from_text(text: str) -> Optional[str]:
    """Extract a candidate title from page-1 text (first non-empty line)."""
    for line in text.splitlines():
        line = line.strip()
        if len(line) > 10:
            return line[:200]
    return None


# ── Main inventory builder ────────────────────────────────────────────────────

def build_inventory(pdf_dir: Path, out_path: Path) -> list:
    """
    Scan all PDFs under pdf_dir, build and write doc_inventory.csv.
    Returns the list of inventory records.
    """
    dataset_root = pdf_dir  # e.g. Dataset/data/full_dataset

    # Load existing doc_dates for effective_from/effective_to
    doc_dates = _load_doc_dates(ROOT / "data" / "doc_dates.csv")

    # Load RBI data
    rbi_manifest = _load_manifest(dataset_root / "rbi" / "manifest.json")
    rbi_index    = _load_rbi_index(dataset_root / "rbi" / "rbi_index.csv")
    rbi_pdf_dir  = dataset_root / "rbi" / "pdf"

    # Load SEBI data
    sebi_circ_manifest = _load_manifest(
        dataset_root / "sebi" / "circulars" / "manifest.json")
    sebi_mc_manifest   = _load_manifest(
        dataset_root / "sebi" / "master_circulars" / "manifest.json")
    sebi_circ_index    = _load_sebi_index(
        dataset_root / "sebi" / "circulars" / "sebi_circulars_index.csv",
        "circular")
    sebi_mc_index      = _load_sebi_index(
        dataset_root / "sebi" / "master_circulars" / "sebi_master_circulars_index.csv",
        "master_circular")
    sebi_circ_pdf_dir  = dataset_root / "sebi" / "circulars" / "pdf"
    sebi_mc_pdf_dir    = dataset_root / "sebi" / "master_circulars" / "pdf"

    # Build records
    records: list[dict] = []
    needs_review_list: list[dict] = []

    def _process_pdf(
        pdf_path: Path,
        regulator: str,
        manifest_lookup: dict,
        index_lookup: dict,
    ):
        stem = pdf_path.stem
        doc_id = f"{regulator}_{stem}"

        m_info = manifest_lookup.get(stem, {})
        i_info = index_lookup.get(stem, {})

        # Title
        title = i_info.get("title", "").strip()
        if not title:
            title = _extract_title_from_text(_extract_page1(pdf_path)) or ""

        # raw date
        raw_date = i_info.get("raw_date", "")
        issue_date, date_nr = parse_date(raw_date)
        if date_nr and not issue_date:
            # Fallback to page-1 extraction
            txt = _extract_page1(pdf_path)
            issue_date, date_nr = _extract_date_from_text(txt)
            if not title:
                title = _extract_title_from_text(txt) or ""

        # doc_type
        doc_type = infer_doc_type(
            m_info.get("doc_type", ""),
            i_info.get("instrument_id", ""),
            title,
        )

        # effective dates from doc_dates.csv
        dd = doc_dates.get(doc_id, {})
        effective_from = dd.get("effective_from") or issue_date
        effective_to   = dd.get("effective_to")

        # needs_review logic
        nr_reasons = []
        if date_nr or not issue_date:
            nr_reasons.append("date_missing_or_ambiguous")
        if not title:
            nr_reasons.append("title_missing")
        if regulator not in ("RBI", "SEBI"):
            nr_reasons.append("regulator_unclear")
        if "supersed" in title.lower():
            nr_reasons.append("superseded_title")

        needs_review = bool(nr_reasons)
        superseded_by = ""  # populated later if needed

        rec = {
            "doc_id":         doc_id,
            "regulator":      regulator,
            "doc_type":       doc_type,
            "title":          title,
            "issue_date":     issue_date or "",
            "effective_from": effective_from or "",
            "effective_to":   effective_to or "",
            "superseded_by":  superseded_by,
            "source_file":    str(pdf_path.relative_to(ROOT)),
            "needs_review":   "true" if needs_review else "false",
            "_nr_reasons":    "; ".join(nr_reasons),
        }
        records.append(rec)
        if needs_review:
            needs_review_list.append(rec)

    # Scan RBI
    if rbi_pdf_dir.exists():
        rbi_pdfs = sorted(rbi_pdf_dir.glob("*.pdf"))
        logger.info("Scanning %d RBI PDFs...", len(rbi_pdfs))
        for pdf_path in rbi_pdfs:
            _process_pdf(pdf_path, "RBI", rbi_manifest, rbi_index)
    else:
        logger.warning("RBI pdf dir not found: %s", rbi_pdf_dir)

    # Scan SEBI circulars
    if sebi_circ_pdf_dir.exists():
        sebi_circ_pdfs = sorted(sebi_circ_pdf_dir.glob("*.pdf"))
        logger.info("Scanning %d SEBI circular PDFs...", len(sebi_circ_pdfs))
        for pdf_path in sebi_circ_pdfs:
            _process_pdf(pdf_path, "SEBI", sebi_circ_manifest, sebi_circ_index)
    else:
        logger.warning("SEBI circulars pdf dir not found: %s", sebi_circ_pdf_dir)

    # Scan SEBI master circulars
    if sebi_mc_pdf_dir.exists():
        sebi_mc_pdfs = sorted(sebi_mc_pdf_dir.glob("*.pdf"))
        logger.info("Scanning %d SEBI master_circular PDFs...", len(sebi_mc_pdfs))
        for pdf_path in sebi_mc_pdfs:
            _process_pdf(pdf_path, "SEBI", sebi_mc_manifest, sebi_mc_index)
    else:
        logger.warning("SEBI master_circulars pdf dir not found: %s", sebi_mc_pdf_dir)

    # Write CSV
    fieldnames = [
        "doc_id", "regulator", "doc_type", "title", "issue_date",
        "effective_from", "effective_to", "superseded_by", "source_file", "needs_review",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    logger.info("Wrote %d records to %s", len(records), out_path)

    # ── Print summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Doc Inventory Summary: {len(records)} documents total")
    print(f"{'='*60}")

    # Counts per regulator
    reg_counts: dict[str, int] = defaultdict(int)
    for r in records:
        reg_counts[r["regulator"]] += 1
    print("\nPer regulator:")
    for reg, cnt in sorted(reg_counts.items()):
        print(f"  {reg:8s}: {cnt:5d}")

    # Counts per year
    year_counts: dict[str, int] = defaultdict(int)
    for r in records:
        y = r["issue_date"][:4] if len(r["issue_date"]) >= 4 else "unknown"
        year_counts[y] += 1
    print("\nPer issue year:")
    for yr, cnt in sorted(year_counts.items()):
        print(f"  {yr}: {cnt:5d}")

    # Counts per regulator+doc_type
    type_counts: dict[str, int] = defaultdict(int)
    for r in records:
        type_counts[f"{r['regulator']}|{r['doc_type']}"] += 1
    print("\nPer regulator|doc_type:")
    for k, cnt in sorted(type_counts.items()):
        print(f"  {k:40s}: {cnt:5d}")

    # needs_review
    print(f"\nneeds_review: {len(needs_review_list)} documents")
    if needs_review_list:
        print(f"\n{'doc_id':<45} {'regulator':<6} {'reason'}")
        print("-" * 80)
        for r in needs_review_list[:50]:
            print(f"  {r['doc_id'][:43]:<45} {r['regulator']:<6} {r.get('_nr_reasons','')}")
        if len(needs_review_list) > 50:
            print(f"  ... and {len(needs_review_list) - 50} more (see {out_path})")

    # Docs with unknown effective dates
    unknown_eff = [r for r in records if not r["effective_from"]]
    print(f"\nDocuments with unknown effective_from: {len(unknown_eff)}")
    if unknown_eff:
        print(f"  First 10:")
        for r in unknown_eff[:10]:
            print(f"  {r['doc_id'][:60]}")

    print(f"\n{'='*60}")
    return records


# ── CLI ───────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    cfg_path = ROOT / "configs" / "experiment.yaml"
    try:
        with open(cfg_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


def main():
    cfg = _load_config()
    paths = cfg.get("paths", {})
    default_pdf_dir = str(ROOT / paths.get("pdf_dir", "Dataset/data/full_dataset"))
    default_out     = str(ROOT / paths.get("doc_inventory", "data/doc_inventory.csv"))

    parser = argparse.ArgumentParser(description="Build document inventory CSV.")
    parser.add_argument("--pdf_dir", default=default_pdf_dir,
                        help="Root of the PDF dataset (contains rbi/ and sebi/ subdirs)")
    parser.add_argument("--out", default=default_out,
                        help="Output path for doc_inventory.csv")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists():
        print(f"ERROR: pdf_dir not found: {pdf_dir}", file=sys.stderr)
        sys.exit(1)

    build_inventory(pdf_dir, Path(args.out))


if __name__ == "__main__":
    main()
