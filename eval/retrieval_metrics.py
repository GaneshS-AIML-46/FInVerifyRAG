"""Retrieval and extraction evaluation metrics.

No LLM/network calls. Uses only local files.
"""
import json
import math
from difflib import SequenceMatcher
from pathlib import Path


def compute_retrieval_metrics(gold_path: str, retrieved_path: str) -> dict:
    """Compute retrieval metrics: Recall@6, Recall@10, MRR, nDCG@10.

    Returns:
        dict with keys: recall_at_6, recall_at_10, mrr, ndcg_at_10, n_queries
        OR {status: 'not_available', reason: ...} if a file is missing.
    """
    if not Path(gold_path).exists():
        return {"status": "not_available", "reason": "gold_chunks.jsonl not found"}
    if not Path(retrieved_path).exists():
        return {"status": "not_available", "reason": "retrieved results not found"}

    gold = {}
    with open(gold_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            gold[rec["question_id"]] = set(rec["gold_chunk_ids"])

    retrieved = {}
    with open(retrieved_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            retrieved[rec["question_id"]] = rec["retrieved_chunk_ids"]

    recall_6_scores = []
    recall_10_scores = []
    mrr_scores = []
    ndcg_10_scores = []

    for qid, gold_ids in gold.items():
        ret = retrieved.get(qid, [])

        # Recall@6
        recall_6_scores.append(1.0 if any(c in gold_ids for c in ret[:6]) else 0.0)
        # Recall@10
        recall_10_scores.append(1.0 if any(c in gold_ids for c in ret[:10]) else 0.0)

        # MRR
        rr = 0.0
        for rank, chunk in enumerate(ret[:10], start=1):
            if chunk in gold_ids:
                rr = 1.0 / rank
                break
        mrr_scores.append(rr)

        # nDCG@10
        dcg = 0.0
        for rank, chunk in enumerate(ret[:10], start=1):
            if chunk in gold_ids:
                dcg += 1.0 / math.log2(rank + 1)
        # Ideal DCG: all gold chunks at top positions
        ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold_ids), 10)))
        ndcg_10_scores.append(dcg / ideal_dcg if ideal_dcg > 0 else 0.0)

    n = len(gold)
    return {
        "n_queries": n,
        "recall_at_6": sum(recall_6_scores) / n if n else 0.0,
        "recall_at_10": sum(recall_10_scores) / n if n else 0.0,
        "mrr": sum(mrr_scores) / n if n else 0.0,
        "ndcg_at_10": sum(ndcg_10_scores) / n if n else 0.0,
    }


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def compute_extraction_metrics(
    gold_atoms_path: str,
    predicted_atoms_path: str,
    fuzzy_threshold: float = 0.0,
) -> dict:
    """Compute precision/recall/F1 per atom_type.

    Returns:
        dict keyed by atom_type with precision, recall, f1
        OR {status: 'not_available', reason: ...} if a file is missing.
    """
    if not Path(gold_atoms_path).exists():
        return {"status": "not_available", "reason": "gold atoms file not found"}
    if not Path(predicted_atoms_path).exists():
        return {"status": "not_available", "reason": "predicted atoms file not found"}

    gold_atoms = []
    with open(gold_atoms_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            gold_atoms.append(json.loads(line))

    pred_atoms = []
    with open(predicted_atoms_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pred_atoms.append(json.loads(line))

    # Group by question_id and atom_type
    from collections import defaultdict

    gold_by_qtype = defaultdict(list)
    pred_by_qtype = defaultdict(list)

    for a in gold_atoms:
        key = (a["question_id"], a.get("atom_type", "unknown"))
        gold_by_qtype[key].append(_normalize(a["atom_text"]))

    for a in pred_atoms:
        key = (a["question_id"], a.get("atom_type", "unknown"))
        pred_by_qtype[key].append(_normalize(a["atom_text"]))

    # All types
    all_types = set(k[1] for k in set(gold_by_qtype) | set(pred_by_qtype))

    def _match(gold_texts, pred_texts):
        matched_gold = set()
        matched_pred = set()
        for pi, pt in enumerate(pred_texts):
            for gi, gt in enumerate(gold_texts):
                if gi in matched_gold:
                    continue
                if fuzzy_threshold > 0:
                    ratio = SequenceMatcher(None, pt, gt).ratio()
                    hit = ratio >= fuzzy_threshold
                else:
                    hit = pt == gt
                if hit:
                    matched_gold.add(gi)
                    matched_pred.add(pi)
                    break
        return len(matched_gold)

    results = {}
    for atom_type in sorted(all_types):
        all_questions = set(k[0] for k in set(gold_by_qtype) | set(pred_by_qtype) if k[1] == atom_type)
        tp_total = 0
        gold_total = 0
        pred_total = 0
        for qid in all_questions:
            key = (qid, atom_type)
            g = gold_by_qtype.get(key, [])
            p = pred_by_qtype.get(key, [])
            tp = _match(g, p)
            tp_total += tp
            gold_total += len(g)
            pred_total += len(p)
        precision = tp_total / pred_total if pred_total else 0.0
        recall = tp_total / gold_total if gold_total else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        results[atom_type] = {"precision": precision, "recall": recall, "f1": f1}

    return results


def get_gold_template_csv() -> str:
    """Return CSV template text for annotating 200+ answers."""
    import csv, io
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["atom_id", "question_id", "atom_text", "atom_type", "answer", "notes"])
    writer.writeheader()
    return buf.getvalue()
