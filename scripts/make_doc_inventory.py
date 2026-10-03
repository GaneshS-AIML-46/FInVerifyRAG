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
  doc_id, regulator, doc_type, title, title_raw, issue_date,
  effective_from, effective_to, superseded_by, source_file, needs_review

  title_raw: original (garbage) title when we replaced it; empty otherwise.

doc_id is stable: {REGULATOR}_{stem}  (stem = PDF filename without .pdf).

Exclusions:
  Documents whose title matches configs/experiment.yaml inventory.excluded_title_patterns
  are dropped from the inventory and written to data/excluded_docs.csv instead.

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
    "%B %d, %Y",   # September 29, 2026
    "%b %d, %Y",   # Sep 09, 2026
    "%d-%b-%Y",    # 01-Jan-2024
    "%d/%m/%Y",    # 01/01/2024
    "%Y-%m-%d",    # 2024-01-01
    "%d %B %Y",    # 01 January 2024
    "%d %b %Y",    # 01 Jan 2024
]

_YEAR_RE = re.compile(r"\b(20\d{2})\b")


def parse_date(raw: Optional[str]) -> tuple[Optional[str], bool]:
    """Try to parse *raw* into YYYY-MM-DD.  Returns (iso_or_None, needs_review)."""
    if not raw or not raw.strip():
        return None, True
    raw = raw.strip().rstrip(",")
    from datetime import datetime
    for fmt in _DATE_FMTS:
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.strftime("%Y-%m-%d"), False
        except ValueError:
            pass
    m = _YEAR_RE.search(raw)
    if m:
        return f"{m.group(1)}-01-01", True   # year-only → needs_review
    return None, True


# ── doc_type inference ────────────────────────────────────────────────────────

_DOCTYPE_PATTERNS = [
    (re.compile(r"master\s+(circular|direction)", re.I), "master_circular"),
    (re.compile(r"master", re.I),                       "master_circular"),
    (re.compile(r"direction", re.I),                    "direction"),
    (re.compile(r"notification", re.I),                 "notification"),
    (re.compile(r"circular", re.I),                     "circular"),
    (re.compile(r"regulation", re.I),                   "regulation"),
    (re.compile(r"press\s+release", re.I),              "press_release"),
    (re.compile(r"report", re.I),                       "report"),
    (re.compile(r"guideline", re.I),                    "guideline"),
    (re.compile(r"amendment", re.I),                    "amendment"),
]


def infer_doc_type(manifest_type: str, instrument_id: str, title: str) -> str:
    if manifest_type and manifest_type not in ("", "unknown"):
        return manifest_type.lower().replace(" ", "_")
    search_text = f"{instrument_id} {title}"
    for pat, dtype in _DOCTYPE_PATTERNS:
        if pat.search(search_text):
            return dtype
    return "document"


# ── Exclusion filter ──────────────────────────────────────────────────────────

def _load_excluded_patterns(cfg: dict) -> list:
    """Return list of compiled regex patterns from inventory.excluded_title_patterns."""
    raw = cfg.get("inventory", {}).get("excluded_title_patterns", [])
    compiled = []
    for p in raw:
        try:
            compiled.append(re.compile(p, re.I))
        except re.error as e:
            logger.warning("Bad excluded_title_pattern %r: %s", p, e)
    return compiled


def _load_excluded_doc_ids(cfg: dict) -> set:
    """Return set of doc_ids explicitly excluded via inventory.excluded_doc_ids."""
    return set(cfg.get("inventory", {}).get("excluded_doc_ids", []))


def _is_excluded(title: str, patterns: list) -> bool:
    return any(pat.search(title) for pat in patterns)


# ── Garbage-title detection ───────────────────────────────────────────────────

# Fragment patterns: "1.", "(a)", "(iv)", "3.2" etc.
_GARBAGE_FRAGMENT_RE = re.compile(
    r"^[\(\[]?(?:\d{1,4}(?:\.\d{1,4})?|[ivxlcIVXLC]{1,5}|[a-zA-Z]{1,3})[\.\)\]]\s*$"
)
# Single-word section / structural headers
_GARBAGE_HEADER_RE = re.compile(
    r"^(annex|annexure|schedule|appendix|exhibit|nil|na|none|table|figure|"
    r"note|draft|corrigendum|foreword|preface|contents?|index|tier\s*\d*)\s*\d*$",
    re.I,
)
# Single-letter section label: "A. Individuals"
_SECTION_LABEL_RE = re.compile(r"^[A-Z][.\:]\s+\w+\s*$")
# Pure date-qualifier title: "(Updated as on April 27, 2026)"
_DATE_QUALIFIER_RE = re.compile(
    r"^[\(\[]\s*(?:updated|amended|revised)\s+(?:as\s+on|till|up\s+to|upto)\s+"
    r"[\w\s,./\-]+\s*[\)\]]$",
    re.I,
)


def _is_garbage_title(
    title: str,
    min_len: int = 8,
    max_nonascii_ratio: float = 0.30,
) -> bool:
    """Return True if *title* is unsuitable as a document title."""
    t = title.strip()
    if not t:
        return True
    if len(t) < min_len:
        return True
    non_ascii = sum(1 for c in t if ord(c) > 127)
    if non_ascii / len(t) > max_nonascii_ratio:
        return True
    if _GARBAGE_FRAGMENT_RE.match(t):
        return True
    if _GARBAGE_HEADER_RE.match(t):
        return True
    if _SECTION_LABEL_RE.match(t):
        return True
    if _DATE_QUALIFIER_RE.match(t):
        return True
    return False


def _extract_english_subject_line(text: str, min_len: int = 15) -> Optional[str]:
    """
    Return the first line in *text* that looks like an English document title:
    - ≥ min_len chars
    - ≤ 30 % non-ASCII
    - ≥ 35 % alphabetic content
    - Does not start with a bare number (page / section number)
    - Is not a pure date line
    """
    for line in text.splitlines():
        line = line.strip()
        if len(line) < min_len:
            continue
        non_ascii = sum(1 for c in line if ord(c) > 127)
        if non_ascii / len(line) > 0.30:
            continue
        alpha = sum(1 for c in line if c.isalpha())
        if alpha / len(line) < 0.35:
            continue
        # Skip "12 January 2024" style date lines
        if re.match(r"^\d{1,2}\s+[A-Za-z]", line) and re.search(r"\d{4}", line):
            continue
        # Skip lines starting with a bare page/section number
        if re.match(r"^\d+[\.\)]\s", line):
            continue
        return line[:200]
    return None


# ── Index CSV loaders ─────────────────────────────────────────────────────────

def _load_rbi_index(path: Path) -> dict:
    result: dict = {}
    if not path.exists():
        logger.warning("RBI index not found: %s", path)
        return result
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lpath = row.get("local_pdf_path", "")
            if not lpath:
                continue
            stem = Path(lpath).stem
            result[stem] = {
                "title":         row.get("title", "").strip(),
                "raw_date":      row.get("issue_date", "").strip().strip('"'),
                "instrument_id": row.get("instrument_id", ""),
            }
    logger.info("Loaded %d entries from RBI index %s", len(result), path.name)
    return result


def _load_sebi_index(path: Path, doc_type: str) -> dict:
    result: dict = {}
    if not path.exists():
        logger.warning("SEBI index not found: %s", path)
        return result
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lpath = row.get("local_path", "")
            if not lpath:
                continue
            stem = Path(lpath).stem
            result[stem] = {
                "title":    row.get("title", "").strip(),
                "raw_date": row.get("date", "").strip().strip('"'),
                "doc_type": doc_type,
            }
    logger.info("Loaded %d entries from SEBI index %s", len(result), path.name)
    return result


def _load_doc_dates(path: Path) -> dict:
    result: dict = {}
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
    result: dict = {}
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
            "doc_type":   info.get("doc_type", ""),
            "sha256":     info.get("sha256", ""),
            "fetched_at": info.get("fetched_at", ""),
        }
    logger.info("Loaded %d entries from manifest %s", len(result), path.name)
    return result


# ── PDF page-1 fallback ───────────────────────────────────────────────────────

def _extract_page1(pdf_path: Path) -> str:
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


# ── Main inventory builder ────────────────────────────────────────────────────

def build_inventory(
    pdf_dir: Path,
    out_path: Path,
    excluded_patterns: list,
    excluded_doc_ids: set,
    garbage_min_len: int = 8,
    garbage_max_nonascii: float = 0.30,
) -> list:
    """Scan all PDFs under *pdf_dir*, build and write doc_inventory.csv."""

    dataset_root = pdf_dir

    doc_dates = _load_doc_dates(ROOT / "data" / "doc_dates.csv")

    rbi_manifest      = _load_manifest(dataset_root / "rbi" / "manifest.json")
    rbi_index         = _load_rbi_index(dataset_root / "rbi" / "rbi_index.csv")
    rbi_pdf_dir       = dataset_root / "rbi" / "pdf"

    sebi_circ_manifest = _load_manifest(
        dataset_root / "sebi" / "circulars" / "manifest.json")
    sebi_mc_manifest   = _load_manifest(
        dataset_root / "sebi" / "master_circulars" / "manifest.json")
    sebi_circ_index    = _load_sebi_index(
        dataset_root / "sebi" / "circulars" / "sebi_circulars_index.csv", "circular")
    sebi_mc_index      = _load_sebi_index(
        dataset_root / "sebi" / "master_circulars" / "sebi_master_circulars_index.csv",
        "master_circular")
    sebi_circ_pdf_dir  = dataset_root / "sebi" / "circulars" / "pdf"
    sebi_mc_pdf_dir    = dataset_root / "sebi" / "master_circulars" / "pdf"

    records: list[dict] = []
    excluded_list: list[dict] = []
    needs_review_list: list[dict] = []
    title_fixed_count = 0
    title_unfixed_count = 0

    def _process_pdf(
        pdf_path: Path,
        regulator: str,
        manifest_lookup: dict,
        index_lookup: dict,
    ) -> None:
        nonlocal title_fixed_count, title_unfixed_count

        stem = pdf_path.stem
        doc_id = f"{regulator}_{stem}"

        m_info = manifest_lookup.get(stem, {})
        i_info = index_lookup.get(stem, {})

        # ── Title from index (fast path) ──────────────────────────────────────
        title = i_info.get("title", "").strip()

        # ── Exclusion check (title pattern first, then explicit doc_id list) ──
        excl_reason = ""
        if _is_excluded(title, excluded_patterns):
            excl_reason = "title_pattern"
        elif doc_id in excluded_doc_ids:
            excl_reason = "excluded_doc_id"
        if excl_reason:
            excluded_list.append({
                "doc_id":      doc_id,
                "regulator":   regulator,
                "title":       title,
                "source_file": str(pdf_path.relative_to(ROOT)),
                "reason":      excl_reason,
            })
            return

        # ── Garbage-title replacement ─────────────────────────────────────────
        title_raw = ""
        _page1_cache: list[str] = []   # lazy-load page-1 text once

        def get_page1() -> str:
            if not _page1_cache:
                _page1_cache.append(_extract_page1(pdf_path))
            return _page1_cache[0]

        if not title or _is_garbage_title(title, garbage_min_len, garbage_max_nonascii):
            title_raw = title          # preserve original
            txt = get_page1()
            new_title = _extract_english_subject_line(txt)
            if new_title:
                title = new_title
                title_fixed_count += 1
            else:
                title_unfixed_count += 1
                # keep title_raw set; mark for review below

        # ── raw date / issue date ─────────────────────────────────────────────
        raw_date = i_info.get("raw_date", "")
        issue_date, date_nr = parse_date(raw_date)
        if date_nr and not issue_date:
            txt = get_page1()
            issue_date, date_nr = _extract_date_from_text(txt)
            if not title and not title_raw:
                # try title extraction if we haven't done so
                candidate = _extract_english_subject_line(txt)
                if candidate:
                    title = candidate

        # ── doc_type ──────────────────────────────────────────────────────────
        doc_type = infer_doc_type(
            m_info.get("doc_type", ""),
            i_info.get("instrument_id", ""),
            title,
        )

        # ── Effective dates ───────────────────────────────────────────────────
        dd = doc_dates.get(doc_id, {})
        effective_from = dd.get("effective_from") or issue_date
        effective_to   = dd.get("effective_to")

        # ── needs_review reasons ──────────────────────────────────────────────
        nr_reasons = []
        if date_nr or not issue_date:
            nr_reasons.append("date_missing_or_ambiguous")
        if not title:
            nr_reasons.append("title_missing")
        elif title_raw and not title:
            nr_reasons.append("title_garbage_unfixed")
        elif title_raw:
            nr_reasons.append("title_replaced_from_page1")
        if regulator not in ("RBI", "SEBI"):
            nr_reasons.append("regulator_unclear")
        if "supersed" in title.lower() or "withdrawn" in title.lower():
            nr_reasons.append("superseded_title")

        rec = {
            "doc_id":         doc_id,
            "regulator":      regulator,
            "doc_type":       doc_type,
            "title":          title,
            "title_raw":      title_raw,
            "issue_date":     issue_date or "",
            "effective_from": effective_from or "",
            "effective_to":   effective_to or "",
            "superseded_by":  "",
            "source_file":    str(pdf_path.relative_to(ROOT)),
            "needs_review":   "true" if nr_reasons else "false",
            "_nr_reasons":    "; ".join(nr_reasons),
        }
        records.append(rec)
        if nr_reasons:
            needs_review_list.append(rec)

    # ── Scan PDFs ─────────────────────────────────────────────────────────────
    if rbi_pdf_dir.exists():
        rbi_pdfs = sorted(rbi_pdf_dir.glob("*.pdf"))
        logger.info("Scanning %d RBI PDFs...", len(rbi_pdfs))
        for p in rbi_pdfs:
            _process_pdf(p, "RBI", rbi_manifest, rbi_index)
    else:
        logger.warning("RBI pdf dir not found: %s", rbi_pdf_dir)

    if sebi_circ_pdf_dir.exists():
        sebi_circ_pdfs = sorted(sebi_circ_pdf_dir.glob("*.pdf"))
        logger.info("Scanning %d SEBI circular PDFs...", len(sebi_circ_pdfs))
        for p in sebi_circ_pdfs:
            _process_pdf(p, "SEBI", sebi_circ_manifest, sebi_circ_index)
    else:
        logger.warning("SEBI circulars pdf dir not found: %s", sebi_circ_pdf_dir)

    if sebi_mc_pdf_dir.exists():
        sebi_mc_pdfs = sorted(sebi_mc_pdf_dir.glob("*.pdf"))
        logger.info("Scanning %d SEBI master_circular PDFs...", len(sebi_mc_pdfs))
        for p in sebi_mc_pdfs:
            _process_pdf(p, "SEBI", sebi_mc_manifest, sebi_mc_index)
    else:
        logger.warning("SEBI master_circulars pdf dir not found: %s", sebi_mc_pdf_dir)

    # ── Write excluded_docs.csv ───────────────────────────────────────────────
    excl_path = ROOT / "data" / "excluded_docs.csv"
    excl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(excl_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["doc_id", "regulator", "title", "reason", "source_file"],
            extrasaction="ignore")
        writer.writeheader()
        writer.writerows(excluded_list)

    # ── Write doc_inventory.csv ───────────────────────────────────────────────
    fieldnames = [
        "doc_id", "regulator", "doc_type", "title", "title_raw",
        "issue_date", "effective_from", "effective_to",
        "superseded_by", "source_file", "needs_review",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    logger.info("Wrote %d records to %s", len(records), out_path)

    # ══════════════════════════════════════════════════════════════════════════
    #   Summary
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*62}")
    print(f"  Doc Inventory Summary")
    print(f"{'='*62}")
    print(f"  Total in inventory : {len(records)}")
    print(f"  Excluded (by title): {len(excluded_list)}")
    if excluded_list:
        pat_counts: dict[str, int] = defaultdict(int)
        for ex in excluded_list:
            for pat in excluded_patterns:
                if pat.search(ex["title"]):
                    pat_counts[pat.pattern] += 1
                    break
        for pat_str, cnt in sorted(pat_counts.items(), key=lambda x: -x[1]):
            print(f"    pattern {pat_str!r}: {cnt} docs")

    # Per regulator
    reg_counts: dict[str, int] = defaultdict(int)
    for r in records:
        reg_counts[r["regulator"]] += 1
    print("\n  Per regulator:")
    for reg, cnt in sorted(reg_counts.items()):
        print(f"    {reg:8s}: {cnt:5d}")

    # Per year
    year_counts: dict[str, int] = defaultdict(int)
    for r in records:
        y = r["issue_date"][:4] if len(r["issue_date"]) >= 4 else "unknown"
        year_counts[y] += 1
    print("\n  Per issue year:")
    for yr, cnt in sorted(year_counts.items()):
        print(f"    {yr}: {cnt:5d}")

    # Per regulator+doc_type
    type_counts: dict[str, int] = defaultdict(int)
    for r in records:
        type_counts[f"{r['regulator']}|{r['doc_type']}"] += 1
    print("\n  Per regulator|doc_type:")
    for k, cnt in sorted(type_counts.items()):
        print(f"    {k:42s}: {cnt:5d}")

    # Title fixes
    print(f"\n  Titles replaced (garbage→page-1): {title_fixed_count}")
    print(f"  Titles still garbage (no page-1 English found): {title_unfixed_count}")
    unfixed = [r for r in records if "title_garbage_unfixed" in r.get("_nr_reasons", "")]
    for r in unfixed[:20]:
        print(f"    {r['doc_id'][:55]}  raw={r['title_raw'][:40]!r}")

    # needs_review
    nr_by_reason: dict[str, int] = defaultdict(int)
    for r in needs_review_list:
        for reason in r.get("_nr_reasons", "").split("; "):
            if reason:
                nr_by_reason[reason] += 1
    print(f"\n  needs_review: {len(needs_review_list)} documents")
    for reason, cnt in sorted(nr_by_reason.items(), key=lambda x: -x[1]):
        print(f"    {reason}: {cnt}")
    if needs_review_list:
        print(f"\n  {'doc_id':<45} {'regulator':<6} reason")
        print("  " + "-"*80)
        skip_reasons = {"title_replaced_from_page1"}   # not actionable by human
        actionable = [r for r in needs_review_list
                      if not set(r.get("_nr_reasons","").split("; ")) <= skip_reasons]
        for r in actionable[:40]:
            print(f"  {r['doc_id'][:43]:<45} {r['regulator']:<6} {r.get('_nr_reasons','')}")
        if len(actionable) > 40:
            print(f"  ... and {len(actionable)-40} more")

    # Unknown effective_from
    unknown_eff = [r for r in records if not r["effective_from"]]
    print(f"\n  Documents with unknown effective_from: {len(unknown_eff)}")
    for r in unknown_eff[:10]:
        print(f"    {r['doc_id'][:60]}")

    # ── Effective-date anomaly report (task 4) ────────────────────────────────
    print(f"\n{'='*62}")
    print("  Effective-from anomaly report")
    print(f"{'='*62}")

    from collections import Counter
    eff_counter = Counter(r["effective_from"] for r in records if r["effective_from"])
    bulk_dates = sorted(
        [(dt, cnt) for dt, cnt in eff_counter.items() if cnt >= 20],
        key=lambda x: -x[1],
    )
    if bulk_dates:
        print(f"\n  effective_from values shared by ≥20 documents ({len(bulk_dates)} found):")
        print(f"  NOTE: A large shared date usually means doc_dates.csv used one date")
        print(f"        for a whole batch. Spot-check the sample docs below.")
        hdr = f"  {'effective_from':<13} {'count':>5}  sample doc_ids (first 3)"
        print(hdr)
        print("  " + "-" * 70)
        for dt, cnt in bulk_dates:
            sample = [r["doc_id"][:32] for r in records if r["effective_from"] == dt][:3]
            print(f"  {dt:<13} {cnt:>5}  {', '.join(sample)}")
    else:
        print("\n  No effective_from date shared by ≥20 documents. ✓")

    inversions = [
        r for r in records
        if r["effective_from"] and r["issue_date"]
        and len(r["effective_from"]) >= 10 and len(r["issue_date"]) >= 10
        and r["effective_from"][:10] < r["issue_date"][:10]
    ]
    print(f"\n  Documents where effective_from < issue_date: {len(inversions)}")
    if inversions:
        print(f"  {'doc_id':<44} {'eff_from':<12} {'issue_date':<12} source_file (last 40)")
        print("  " + "-" * 100)
        for r in inversions[:30]:
            src = r["source_file"][-40:]
            print(f"  {r['doc_id'][:42]:<44} {r['effective_from']:<12} {r['issue_date']:<12} ...{src}")
        if len(inversions) > 30:
            print(f"  ... and {len(inversions)-30} more. Check data/doc_inventory.csv.")

    print(f"\n{'='*62}")
    return records


# ── Config / CLI ──────────────────────────────────────────────────────────────

def _load_config() -> dict:
    cfg_path = ROOT / "configs" / "experiment.yaml"
    try:
        with open(cfg_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


def main() -> None:
    cfg = _load_config()
    paths   = cfg.get("paths", {})
    inv_cfg = cfg.get("inventory", {})

    default_pdf_dir = str(ROOT / paths.get("pdf_dir", "Dataset/data/full_dataset"))
    default_out     = str(ROOT / paths.get("doc_inventory", "data/doc_inventory.csv"))

    parser = argparse.ArgumentParser(description="Build document inventory CSV.")
    parser.add_argument("--pdf_dir", default=default_pdf_dir)
    parser.add_argument("--out",     default=default_out)
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists():
        print(f"ERROR: pdf_dir not found: {pdf_dir}", file=sys.stderr)
        sys.exit(1)

    excluded_patterns = _load_excluded_patterns(cfg)
    excluded_doc_ids  = _load_excluded_doc_ids(cfg)
    garbage_min_len   = int(inv_cfg.get("garbage_title_min_len", 8))
    garbage_max_na    = float(inv_cfg.get("garbage_title_max_nonascii_ratio", 0.30))

    build_inventory(
        pdf_dir=pdf_dir,
        out_path=Path(args.out),
        excluded_patterns=excluded_patterns,
        excluded_doc_ids=excluded_doc_ids,
        garbage_min_len=garbage_min_len,
        garbage_max_nonascii=garbage_max_na,
    )


if __name__ == "__main__":
    main()
