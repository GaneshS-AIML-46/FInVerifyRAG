"""Merge LLM train + human test labels, assign doc splits, write train/calib/test JSONL."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TRAIN_FINAL = ROOT / "results" / "train_atoms_final.jsonl"
TRAIN_LLM = ROOT / "results" / "train_atoms_llm_labeled.jsonl"
TEST_HUMAN_CSV = ROOT / "results" / "test_atoms_human_labeled.csv"
DOC_ASSIGN = ROOT / "data" / "splits" / "doc_assignment.csv"
QUESTIONS = ROOT / "data" / "questions_pilot_review.csv"
MERGED_OUT = ROOT / "results" / "raw_atoms_labeled.jsonl"
RESULTS_DIR = ROOT / "results"
LOCK_PATH = ROOT / "data" / "splits" / "test.lock"
VALID_LABELS = {"supported", "unsupported", "unclear"}


def _norm_label(s: str) -> str:
    lab = (s or "").strip().lower()
    if lab not in VALID_LABELS:
        raise ValueError(f"Invalid label {s!r}")
    return lab


def _question_id(atom_id: str) -> str:
    if "_atom_" in atom_id:
        return atom_id.rsplit("_atom_", 1)[0]
    return atom_id


def _load_train_llm_records() -> list[dict]:
    path = TRAIN_FINAL if TRAIN_FINAL.is_file() else TRAIN_LLM
    if not path.is_file():
        raise FileNotFoundError(
            f"Neither {TRAIN_FINAL} nor {TRAIN_LLM} found. "
            "Run train labeling or pass audit before merge."
        )
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        src = rec.get("label_source", "llm")
        if src != "llm":
            raise ValueError(f"Train record {rec.get('id')} has label_source={src!r}, expected 'llm'")
        aid = rec.get("atom_id") or rec.get("id")
        rows.append(
            {
                "atom_id": aid,
                "question_id": _question_id(aid),
                "question": rec.get("question", ""),
                "atom_text": rec.get("atom_text", ""),
                "source_passage": rec.get("source_passage", ""),
                "label": _norm_label(rec.get("label", "")),
                "reason": rec.get("reason", ""),
                "label_source": "llm",
                "model": rec.get("model", "Qwen/Qwen2.5-7B-Instruct"),
            }
        )
    if path == TRAIN_LLM and not TRAIN_FINAL.is_file():
        TRAIN_FINAL.write_text(TRAIN_LLM.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Note: copied {TRAIN_LLM.name} -> {TRAIN_FINAL.name} (audit had not created final).")
    return rows


def _load_test_human_records() -> list[dict]:
    if not TEST_HUMAN_CSV.is_file():
        raise FileNotFoundError(TEST_HUMAN_CSV)
    df = pd.read_csv(TEST_HUMAN_CSV, encoding="utf-8", encoding_errors="replace")
    rows = []
    for _, r in df.iterrows():
        aid = str(r["atom_id"])
        rows.append(
            {
                "atom_id": aid,
                "question_id": _question_id(aid),
                "question": str(r.get("question", "")),
                "atom_text": str(r.get("atom_text", "")),
                "source_passage": str(r.get("source_passage", "")),
                "label": _norm_label(str(r.get("label", ""))),
                "reason": str(r.get("reason", "")),
                "label_source": "human",
                "model": None,
            }
        )
    return rows


def _attach_doc_split(records: list[dict], q_to_doc: dict, doc_split: dict) -> list[dict]:
    out = []
    for rec in records:
        qid = rec["question_id"]
        doc_id = q_to_doc.get(qid)
        if not doc_id:
            raise ValueError(f"No source_doc_id for question_id={qid} (atom {rec['atom_id']})")
        split = doc_split.get(doc_id)
        if not split:
            raise ValueError(f"doc_id {doc_id!r} not in doc_assignment.csv")
        rec = dict(rec)
        rec["doc_id"] = doc_id
        rec["split"] = split
        out.append(rec)
    return out


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _leakage_check(train: list[dict], calib: list[dict], test: list[dict]) -> bool:
    d_train = {r["doc_id"] for r in train}
    d_calib = {r["doc_id"] for r in calib}
    d_test = {r["doc_id"] for r in test}
    overlaps = [
        ("train", "calib", d_train & d_calib),
        ("train", "test", d_train & d_test),
        ("calib", "test", d_calib & d_test),
    ]
    bad = [(a, b, len(o), sorted(o)[:5]) for a, b, o in overlaps if o]
    if bad:
        print("Leakage check: FAIL")
        for a, b, n, sample in bad:
            print(f"  {a}/{b}: {n} shared doc_id(s), e.g. {sample}")
        return False
    print("Leakage check: PASS")
    return True


def _label_dist(rows: list[dict]) -> dict[str, int]:
    return dict(Counter(r["label"] for r in rows))


def main() -> None:
    train_llm = _load_train_llm_records()
    test_human = _load_test_human_records()

    train_ids = {r["atom_id"] for r in train_llm}
    test_ids = {r["atom_id"] for r in test_human}
    overlap = train_ids & test_ids
    if overlap:
        raise ValueError(f"Train/test atom_id overlap: {len(overlap)} ids")

    merged = train_llm + test_human
    _write_jsonl(MERGED_OUT, merged)

    q_df = pd.read_csv(QUESTIONS)
    q_to_doc = dict(zip(q_df["question_id"].astype(str), q_df["source_doc_id"].astype(str)))
    doc_df = pd.read_csv(DOC_ASSIGN)
    doc_split = dict(zip(doc_df["doc_id"].astype(str), doc_df["split"].astype(str)))

    labeled = _attach_doc_split(merged, q_to_doc, doc_split)

    by_split: dict[str, list[dict]] = defaultdict(list)
    for rec in labeled:
        by_split[rec["split"]].append(rec)

    train_rows = by_split.get("train", [])
    calib_rows = by_split.get("calib", [])
    test_rows = by_split.get("test", [])

    _write_jsonl(RESULTS_DIR / "train.jsonl", train_rows)
    _write_jsonl(RESULTS_DIR / "calib.jsonl", calib_rows)
    _write_jsonl(RESULTS_DIR / "test.jsonl", test_rows)

    if not _leakage_check(train_rows, calib_rows, test_rows):
        sys.exit(1)

    test_bytes = (RESULTS_DIR / "test.jsonl").read_bytes()
    digest = hashlib.sha256(test_bytes).hexdigest()
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.write_text(digest, encoding="utf-8")

    n_train_llm = sum(1 for r in train_rows if r["label_source"] == "llm")
    n_calib_human = sum(1 for r in calib_rows if r["label_source"] == "human")
    n_test_human = sum(1 for r in test_rows if r["label_source"] == "human")

    print("=" * 60)
    print(f"Merged labeled atoms: {len(merged)} -> {MERGED_OUT.name}")
    print(f"N_train atoms (LLM):  {n_train_llm}  (total train split rows: {len(train_rows)})")
    print(f"N_calib atoms (human): {n_calib_human}  (total calib split rows: {len(calib_rows)})")
    print(f"N_test atoms (human):  {n_test_human}  (total test split rows: {len(test_rows)})")
    for name, rows in [("train", train_rows), ("calib", calib_rows), ("test", test_rows)]:
        print(f"  {name} label distribution: {_label_dist(rows)}")
    print(f"Wrote {LOCK_PATH} (sha256 of results/test.jsonl)")


if __name__ == "__main__":
    main()
