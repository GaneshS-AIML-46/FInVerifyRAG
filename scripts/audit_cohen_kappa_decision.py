"""Merge human vs Qwen audit labels, Cohen's kappa, accept/reject train LLM labels."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HUMAN_CSV = ROOT / "results" / "audit_50_atoms_human_labeled.csv"
LLM_CSV = ROOT / "results" / "audit_50_atoms_llm_labels.csv"
TRAIN_LLM = ROOT / "results" / "train_atoms_llm_labeled.jsonl"
TRAIN_FINAL = ROOT / "results" / "train_atoms_final.jsonl"
KAPPA_THRESHOLD = 0.65
CATEGORIES = ["supported", "unsupported", "unclear"]


def _norm_label(s: str) -> str:
    lab = (s or "").strip().lower()
    if lab not in CATEGORIES:
        raise ValueError(f"Invalid label {s!r}; expected one of {CATEGORIES}")
    return lab


def _load_train_qwen() -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(TRAIN_LLM, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                out[rec["id"]] = rec
    return out


def _ensure_llm_csv(human: pd.DataFrame) -> pd.DataFrame:
    train = _load_train_qwen()
    rows = []
    missing = []
    for _, r in human.iterrows():
        aid = r["atom_id"]
        if aid in train:
            t = train[aid]
            rows.append(
                {
                    "atom_id": aid,
                    "label": t["label"],
                    "reason": t.get("reason", ""),
                    "label_source": t.get("label_source", "llm"),
                    "model": t.get("model", "Qwen/Qwen2.5-7B-Instruct"),
                }
            )
        else:
            missing.append(r)

    if missing:
        import yaml

        from common.llm_client import LLMClient
        from scripts.label_train_atoms_llm import _label_one, _vllm_up

        if _vllm_up():
            from scripts.label_train_atoms_llm import _build_llm_client

            client, model = _build_llm_client()
        else:
            with open(ROOT / "configs" / "experiment.yaml", encoding="utf-8") as f:
                base = yaml.safe_load(f)
            role = dict(base["roles"]["generator"])
            role["max_tokens"] = 256
            role["temperature"] = 0.0
            cfg = {
                "roles": {"atom_labeler": role},
                "llm_prices": base.get("llm_prices", {}),
                "budget_caps": {},
            }
            model = role["model"]
            print(f"[audit] vLLM down; labeling {len(missing)} audit atoms with {model}.")
            client = LLMClient(role="atom_labeler", cfg=cfg, check_prices=False)
        for r in missing:
            parsed, _ = _label_one(
                client,
                str(r["question"]),
                str(r["atom_text"]),
                str(r["source_passage"]),
            )
            rows.append(
                {
                    "atom_id": r["atom_id"],
                    "label": parsed["label"],
                    "reason": parsed["reason"],
                    "label_source": "llm",
                    "model": model,
                }
            )

    llm_df = pd.DataFrame(rows)
    llm_df.to_csv(LLM_CSV, index=False)
    return llm_df


def main() -> None:
    if not HUMAN_CSV.is_file():
        raise FileNotFoundError(HUMAN_CSV)

    human = pd.read_csv(HUMAN_CSV, encoding="utf-8", encoding_errors="replace")
    if len(human) != 50:
        raise ValueError(f"Expected 50 human-labeled audit atoms, got {len(human)}")

    if LLM_CSV.is_file():
        llm = pd.read_csv(LLM_CSV, encoding="utf-8", encoding_errors="replace")
    else:
        print(f"Building {LLM_CSV.name} from train Qwen labels (+ LLM for any missing IDs).")
        llm = _ensure_llm_csv(human)

    merged = human.merge(llm, on="atom_id", suffixes=("_human", "_llm"))
    if len(merged) != 50:
        raise ValueError(f"Merge produced {len(merged)} rows, expected 50")

    y_human = [_norm_label(x) for x in merged["label_human"]]
    y_llm = [_norm_label(x) for x in merged["label_llm"]]

    kappa = cohen_kappa_score(y_human, y_llm, labels=CATEGORIES)
    print(f"Cohen's kappa: {kappa:.2f}")

    cm = confusion_matrix(y_llm, y_human, labels=CATEGORIES)
    print("Confusion matrix (rows=LLM, cols=human):")
    header = "           " + "  ".join(f"{c:>12}" for c in CATEGORIES)
    print(header)
    for i, row_name in enumerate(CATEGORIES):
        vals = "  ".join(f"{cm[i, j]:>12}" for j in range(len(CATEGORIES)))
        print(f"{row_name:>12} {vals}")

    print("Per-label agreement (human stratum, LLM matches human):")
    for cat in CATEGORIES:
        idx = [i for i, h in enumerate(y_human) if h == cat]
        if not idx:
            pct = 0.0
        else:
            pct = 100.0 * sum(1 for i in idx if y_llm[i] == y_human[i]) / len(idx)
        print(f"  {cat}: {pct:.1f}%")

    if kappa >= KAPPA_THRESHOLD:
        with open(TRAIN_LLM, encoding="utf-8") as fin, open(TRAIN_FINAL, "w", encoding="utf-8") as fout:
            for line in fin:
                if line.strip():
                    fout.write(line if line.endswith("\n") else line + "\n")
        print("Audit passed. LLM labels accepted for train split.")
        print(f"Wrote {TRAIN_FINAL}")
    else:
        print(
            f"Audit failed (kappa {kappa:.2f}). LLM labels rejected. "
            "Human labeling of all 87 train atoms required."
        )


if __name__ == "__main__":
    main()
