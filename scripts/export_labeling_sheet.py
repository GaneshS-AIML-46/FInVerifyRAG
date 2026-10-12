"""Export atoms to a labeling sheet (CSV or XLSX) for human annotation.

Usage:
    python scripts/export_labeling_sheet.py --atoms <atoms.jsonl> --out <sheet.csv|sheet.xlsx>

Blind sheet from pipeline raw_atoms.jsonl (no decision/risk/signals):
    python scripts/export_labeling_sheet.py --from-pipeline-raw results/raw_atoms.jsonl \\
        --out results/labeling_sheet_blind.csv --bm25-index .bm25_index.pkl
"""
import argparse
import csv
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List

FORBIDDEN_COLUMNS = {"decision", "risk", "s_ret", "s_nli", "s_ent", "s_div", "s_ver", "judge_verdict"}
OUTPUT_COLUMNS = ["atom_id", "question", "query_date", "atom_text", "cited_passage_highlighted",
                  "label", "reason", "needs_close_look"]

BLIND_COLUMNS = ["atom_id", "question", "atom_text", "source_passage", "label", "reason"]

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


def _make_bm25_retriever(bm25_index_path: Path) -> Callable[[str, str], List[str]]:
    """Return a function (question, query_date) -> list of chunk texts (top-6 BM25)."""
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    with open(bm25_index_path, "rb") as f:
        data = pickle.load(f)
    raw_chunks: list = data["chunks"]
    tokenized: list = data["tokenized_corpus"]
    from rank_bm25 import BM25Okapi

    bm25 = BM25Okapi(tokenized)
    top_k = 6

    def retrieve_texts(query: str, _query_date: str) -> List[str]:
        tokens = query.lower().split()
        scores = bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[: top_k * 3]
        top_indices = [i for i in top_indices if scores[i] > 0][:top_k]
        return [raw_chunks[i].text for i in top_indices]

    return retrieve_texts


def load_pipeline_raw_atoms(path: str) -> list:
    """Load pipeline results/raw_atoms.jsonl (may contain decision, risk, signals)."""
    atoms = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                atoms.append(json.loads(line))
    return atoms


def build_blind_rows(
    pipeline_atoms: list,
    retrieve_texts: Callable[[str, str], List[str]],
) -> List[Dict[str, str]]:
    """Blind labeling rows; source_passage = BM25 evidence bundle for the question."""
    passage_cache: Dict[str, str] = {}
    rows: List[Dict[str, str]] = []

    for atom in pipeline_atoms:
        atom_id = atom.get("atom_id") or atom.get("id") or ""
        question = (atom.get("question") or "").strip()
        query_date = atom.get("query_date") or ""
        atom_text = (atom.get("atom_text") or atom.get("claim") or atom.get("text") or "").strip()

        cache_key = atom.get("question_id") or question
        if cache_key not in passage_cache:
            cited = (atom.get("cited_passage") or atom.get("source_passage") or "").strip()
            if cited:
                passage_cache[cache_key] = cited
            else:
                chunk_texts = retrieve_texts(question, query_date)
                if not chunk_texts:
                    raise ValueError(
                        f"No BM25 passages for question_id={atom.get('question_id')!r}; "
                        "cannot build source_passage."
                    )
                passage_cache[cache_key] = "\n\n---\n\n".join(t.strip() for t in chunk_texts if t.strip())

        source_passage = passage_cache[cache_key].strip()
        if not atom_id or not atom_text or not source_passage:
            raise ValueError(
                f"Missing required field for atom_id={atom_id!r}: "
                f"atom_text={bool(atom_text)} source_passage={bool(source_passage)}"
            )

        rows.append(
            {
                "atom_id": atom_id,
                "question": question,
                "atom_text": atom_text,
                "source_passage": source_passage,
                "label": "",
                "reason": "",
            }
        )
    return rows


def write_blind_csv(rows: list, out_path: str) -> None:
    forbidden = {"decision", "risk", "s_ret", "s_nli", "s_ent", "s_div", "s_ver", "judge_verdict", "stratum"}
    if forbidden & set(BLIND_COLUMNS):
        raise RuntimeError(f"Blind columns must not include forbidden fields: {forbidden & set(BLIND_COLUMNS)}")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=BLIND_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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
    parser.add_argument("--atoms", help="Annotator-ready atoms JSONL (no pipeline fields)")
    parser.add_argument("--from-pipeline-raw", metavar="PATH", help="Pipeline raw_atoms.jsonl")
    parser.add_argument("--out", required=True, help="Output CSV or XLSX path")
    parser.add_argument(
        "--bm25-index",
        default=".bm25_index.pkl",
        help="BM25 index for source_passage when exporting from pipeline raw (default: .bm25_index.pkl)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    bm25_path = Path(args.bm25_index)
    if not bm25_path.is_absolute():
        bm25_path = root / bm25_path

    if args.from_pipeline_raw:
        raw_path = Path(args.from_pipeline_raw)
        if not raw_path.is_absolute():
            raw_path = root / raw_path
        if not bm25_path.is_file():
            raise FileNotFoundError(f"BM25 index not found: {bm25_path}")
        retrieve = _make_bm25_retriever(bm25_path)
        pipeline_atoms = load_pipeline_raw_atoms(str(raw_path))
        rows = build_blind_rows(pipeline_atoms, retrieve)
        write_blind_csv(rows, args.out)
        print(f"Wrote {len(rows)} blind labeling rows to {args.out}")
        return

    if not args.atoms:
        parser.error("Provide --atoms or --from-pipeline-raw")

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
