"""LLM-label train split of pilot atoms; hold out test split unlabeled.

Usage:
    python scripts/label_train_atoms_llm.py
"""
from __future__ import annotations

import json
import random
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.llm_client import LLMClient
from scripts.export_labeling_sheet import _make_bm25_retriever, load_pipeline_raw_atoms

RAW_ATOMS = ROOT / "results" / "raw_atoms.jsonl"
BM25_INDEX = ROOT / ".bm25_index.pkl"
TRAIN_OUT = ROOT / "results" / "train_atoms_llm_labeled.jsonl"
TEST_OUT = ROOT / "results" / "test_atoms_unlabeled.jsonl"
SPLIT_SEED = 42
REPORT_MODEL_VLLM = "Qwen/Qwen2.5-7B-Instruct"
REPORT_MODEL_CLAUDE = "claude-sonnet-4-6"

LABEL_PROMPT = (
    'Is this atom SUPPORTED (passage states or entails it), UNSUPPORTED (contradicted or absent), '
    "or UNCLEAR? Respond JSON: {{label: 'supported'|'unsupported'|'unclear', reason: 'short'}}"
)

VALID_LABELS = {"supported", "unsupported", "unclear"}


def _vllm_up() -> bool:
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=3)
        return True
    except Exception:
        return False


def _build_llm_client() -> tuple[LLMClient, str]:
    if _vllm_up():
        cfg = {
            "roles": {
                "atom_labeler": {
                    "provider": "ollama",
                    "model": REPORT_MODEL_VLLM,
                    "api_key_env": "LLM_API_KEY",
                    "base_url": "http://127.0.0.1:8000/v1",
                    "max_tokens": 256,
                    "temperature": 0.0,
                }
            },
            "llm_prices": {REPORT_MODEL_VLLM: {"input": 0.0, "output": 0.0}},
            "budget_caps": {},
        }
        return LLMClient(role="atom_labeler", cfg=cfg, check_prices=False), REPORT_MODEL_VLLM

    cfg_path = ROOT / "configs" / "experiment.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        base = yaml.safe_load(f)
    cfg = {
        "roles": {
            "atom_labeler": {
                "provider": "anthropic",
                "model": REPORT_MODEL_CLAUDE,
                "api_key_env": "LLM_API_KEY",
                "max_tokens": 256,
                "temperature": 0.0,
            }
        },
        "llm_prices": base.get("llm_prices", {}),
        "budget_caps": base.get("budget_caps", {}),
    }
    if REPORT_MODEL_CLAUDE not in cfg["llm_prices"]:
        cfg["llm_prices"][REPORT_MODEL_CLAUDE] = {"input": 3.0, "output": 15.0}
    print("[label_train_atoms_llm] vLLM not reachable; using Anthropic fallback.")
    return LLMClient(role="atom_labeler", cfg=cfg, check_prices=False), REPORT_MODEL_CLAUDE


def _source_passage_for_atom(atom: dict, retrieve_texts, cache: dict) -> str:
    question = (atom.get("question") or "").strip()
    query_date = atom.get("query_date") or ""
    cache_key = atom.get("question_id") or question
    if cache_key not in cache:
        cited = (atom.get("cited_passage") or atom.get("source_passage") or "").strip()
        if cited:
            cache[cache_key] = cited
        else:
            texts = retrieve_texts(question, query_date)
            if not texts:
                raise ValueError(f"No BM25 passage for question_id={atom.get('question_id')!r}")
            cache[cache_key] = "\n\n---\n\n".join(t.strip() for t in texts if t.strip())
    return cache[cache_key].strip()


def split_train_test(atoms: list[dict]) -> tuple[list[dict], list[dict]]:
    atoms = sorted(atoms, key=lambda a: (a.get("id") or a.get("atom_id") or ""))
    shuffled = list(atoms)
    random.Random(SPLIT_SEED).shuffle(shuffled)
    n_train = round(len(shuffled) * 2 / 3)
    return shuffled[:n_train], shuffled[n_train:]


def _parse_label_json(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty LLM response")
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)

    obj = None
    for candidate in (text,):
        try:
            obj = json.loads(candidate)
            break
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                blob = m.group()
                try:
                    obj = json.loads(blob)
                    break
                except json.JSONDecodeError:
                    blob2 = re.sub(
                        r"(?P<prefix>[\{,]\s*)'(?P<key>[^']+)'\s*:",
                        r'\1"\g<key>":',
                        blob,
                    )
                    blob2 = re.sub(
                        r":\s*'(?P<val>[^']*)'",
                        r': "\g<val>"',
                        blob2,
                    )
                    try:
                        obj = json.loads(blob2)
                        break
                    except json.JSONDecodeError:
                        pass

    if obj is None:
        m_label = re.search(
            r"label\s*:\s*['\"]?(supported|unsupported|unclear)['\"]?",
            text,
            re.IGNORECASE,
        )
        m_reason = re.search(
            r"reason\s*:\s*['\"]([^'\"]{1,500})['\"]",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if not m_label or not m_reason:
            raise ValueError(f"no parseable JSON in response: {text[:200]!r}")
        label = m_label.group(1).lower()
        reason = m_reason.group(1).strip()
    else:
        if isinstance(obj, list) and obj:
            obj = obj[0]
        label = str(obj.get("label", "")).strip().lower()
        reason = str(obj.get("reason", "")).strip()

    if label not in VALID_LABELS:
        raise ValueError(f"invalid label {label!r}")
    if not reason:
        raise ValueError("missing reason")
    return {"label": label, "reason": reason}


def _label_one(client: LLMClient, question: str, atom_text: str, source_passage: str) -> tuple[dict, float]:
    user = (
        f"Question:\n{question}\n\n"
        f"Atom:\n{atom_text}\n\n"
        f"Source passage:\n{source_passage}\n\n"
        f"{LABEL_PROMPT}"
    )
    system = "You label regulatory QA atoms. Output only valid JSON."
    last_err: Exception | None = None
    total_cost = 0.0
    for attempt in range(2):
        result = client.chat(user, system=system)
        total_cost += float(result.get("cost_usd") or 0.0)
        client.record_cost("atom_labeler", result)
        try:
            return _parse_label_json(result.get("text", "")), total_cost
        except ValueError as e:
            last_err = e
            if attempt == 0:
                user = user + "\n\nYour previous reply was not valid JSON. Reply with JSON only."
    raise RuntimeError(f"JSON parse failed after 2 attempts: {last_err}")


def _atom_record(atom: dict, source_passage: str) -> dict:
    return {
        "id": atom.get("id") or atom.get("atom_id"),
        "question": atom.get("question", ""),
        "atom_text": (atom.get("atom_text") or atom.get("claim") or "").strip(),
        "source_passage": source_passage,
    }


def main() -> None:
    if not RAW_ATOMS.is_file():
        raise FileNotFoundError(f"Missing {RAW_ATOMS}")
    if not BM25_INDEX.is_file():
        raise FileNotFoundError(f"Missing {BM25_INDEX}")

    pipeline_atoms = load_pipeline_raw_atoms(str(RAW_ATOMS))
    train_atoms, test_atoms = split_train_test(pipeline_atoms)
    retrieve = _make_bm25_retriever(BM25_INDEX)
    passage_cache: dict = {}

    client, report_model = _build_llm_client()
    total_cost = 0.0
    labels: list[str] = []

    TRAIN_OUT.parent.mkdir(exist_ok=True)
    with open(TRAIN_OUT, "w", encoding="utf-8") as train_f:
        for atom in train_atoms:
            sp = _source_passage_for_atom(atom, retrieve, passage_cache)
            base = _atom_record(atom, sp)
            parsed, cost = _label_one(
                client, base["question"], base["atom_text"], base["source_passage"]
            )
            total_cost += cost
            labels.append(parsed["label"])
            row = {
                **base,
                "label": parsed["label"],
                "reason": parsed["reason"],
                "label_source": "llm",
                "model": report_model,
            }
            train_f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(TEST_OUT, "w", encoding="utf-8") as test_f:
        for atom in test_atoms:
            sp = _source_passage_for_atom(atom, retrieve, passage_cache)
            test_f.write(json.dumps(_atom_record(atom, sp), ensure_ascii=False) + "\n")

    n = len(labels)
    dist = Counter(labels)
    print("=" * 60)
    print(f"Train atoms labeled: {n}")
    print(f"Test atoms (unlabeled): {len(test_atoms)}")
    print(f"Total cost (USD): ${total_cost:.4f}")
    if n:
        print(
            "Label distribution: "
            f"{100 * dist['supported'] / n:.1f}% supported, "
            f"{100 * dist['unsupported'] / n:.1f}% unsupported, "
            f"{100 * dist['unclear'] / n:.1f}% unclear"
        )
    print(f"Wrote {TRAIN_OUT}")
    print(f"Wrote {TEST_OUT}")
    print(f"Backend model field: {report_model}")


if __name__ == "__main__":
    main()
