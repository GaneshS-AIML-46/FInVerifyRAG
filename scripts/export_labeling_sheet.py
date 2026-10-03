"""Export atoms to a labeling sheet (CSV or XLSX) for human annotation.

Usage:
    python scripts/export_labeling_sheet.py --atoms <atoms.jsonl> --out <sheet.csv|sheet.xlsx>
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

FORBIDDEN_COLUMNS = {"decision", "risk", "s_ret", "s_nli", "s_ent", "s_div", "s_ver", "judge_verdict"}
OUTPUT_COLUMNS = ["atom_id", "question", "query_date", "atom_text", "cited_passage_highlighted",
                  "label", "reason", "needs_close_look"]

# Patterns for numbers, dates, and section references
_NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_DATE_RE = re.compile(r"\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}", re.IGNORECASE)
_SEC_RE = re.compile(r"\b(?:Section|Clause|Para|Regulation)\s+\d+\b", re.IGNORECASE)


def _extract_tokens(text: str) -> set:
    tokens = set()
    tokens.update(_NUM_RE.findall(text))
    tokens.update(m.group() for m in _DATE_RE.finditer(text))
    tokens.update(m.group() for m in _SEC_RE.finditer(text))
    return tokens


def highlight_passage(atom_text: str, cited_passage: str) -> str:
    """Bold-highlight tokens from atom_text that also appear in cited_passage."""
    atom_tokens = _extract_tokens(atom_text)
    passage_tokens = _extract_tokens(cited_passage)
    common = atom_tokens & passage_tokens
    if not common:
        return cited_passage

    # Sort longest first to avoid partial replacements
    sorted_tokens = sorted(common, key=len, reverse=True)
    result = cited_passage
    for tok in sorted_tokens:
        result = result.replace(tok, f"**{tok}**")
    return result


def needs_close_look(atom_text: str, cited_passage: str) -> str:
    """YES if any token in atom_text is NOT in cited_passage (possible hallucination)."""
    atom_tokens = _extract_tokens(atom_text)
    if not atom_tokens:
        return "NO"
    passage_tokens = _extract_tokens(cited_passage)
    missing = atom_tokens - passage_tokens
    return "YES" if missing else "NO"


def load_atoms(path: str) -> list:
    atoms = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            bad = FORBIDDEN_COLUMNS & set(record.keys())
            if bad:
                raise ValueError(
                    f"Input contains forbidden column(s): {sorted(bad)}. "
                    "Remove pipeline-internal fields before exporting."
                )
            atoms.append(record)
    return atoms


def build_rows(atoms: list) -> list:
    rows = []
    for atom in atoms:
        cited = atom.get("cited_passage", "")
        atext = atom.get("atom_text", "")
        row = {
            "atom_id": atom.get("atom_id", ""),
            "question": atom.get("question", ""),
            "query_date": atom.get("query_date", ""),
            "atom_text": atext,
            "cited_passage_highlighted": highlight_passage(atext, cited),
            "label": "",
            "reason": "",
            "needs_close_look": needs_close_look(atext, cited),
        }
        rows.append(row)
    return rows


def write_csv(rows: list, out_path: str) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_xlsx(rows: list, out_path: str) -> None:
    try:
        import openpyxl
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl", "-q"])
        import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(OUTPUT_COLUMNS)
    for row in rows:
        ws.append([row[c] for c in OUTPUT_COLUMNS])
    wb.save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Export atoms to labeling sheet")
    parser.add_argument("--atoms", required=True, help="Atoms JSONL file")
    parser.add_argument("--out", required=True, help="Output CSV or XLSX path")
    args = parser.parse_args()

    atoms = load_atoms(args.atoms)
    rows = build_rows(atoms)

    out = args.out.lower()
    if out.endswith(".xlsx"):
        write_xlsx(rows, args.out)
    else:
        write_csv(rows, args.out)

    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
