"""Retrieval and extraction metrics from gold annotation files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from eval.metrics import (
    extraction_f1,
    extraction_precision,
    extraction_recall,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def _read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _get_pipeline_queries(pipeline_report_path: str, questions_path: str) -> dict:
    """Returns dict mapping question_id to query result."""
    q_id_map = {}
    if Path(questions_path).exists():
        q_rows = _read_jsonl(Path(questions_path))
        for r in q_rows:
            q_id_map[r["question"]] = r.get("id", r.get("question_id"))
            
    reports = {}
    p = Path(pipeline_report_path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            for qr in data.get("queries", []):
                q_text = qr.get("question", "")
                qid = q_id_map.get(q_text, q_text)
                reports[qid] = qr
    return reports


def retrieval_from_gold(path: str, pipeline_report_path: str = "results/pipeline_report.json", questions_path: str = "data/questions.jsonl", ks: Optional[List[int]] = None) -> dict:
    """
    Format: {"question_id": str, "gold_chunk_ids": [...]}
    Returns not_available when the file is missing or empty.
    """
    ks = ks or [6, 10]
    p = Path(path)
    if not p.exists():
        return {"status": "not_available", "reason": f"gold file missing: {path}", "simulated": False}
        
    rows = _read_jsonl(p)
    if not rows:
        return {"status": "not_available", "reason": f"no gold chunk annotations at {path}", "simulated": False}
        
    pipeline_queries = _get_pipeline_queries(pipeline_report_path, questions_path)
    
    retrieved_list = []
    gold_list = []
    for r in rows:
        qid = r["question_id"]
        if qid not in pipeline_queries:
            continue
        gold_ids = r.get("gold_chunk_ids", [])
        if not gold_ids:
            continue
            
        ret_chunks = pipeline_queries[qid].get("sources", [])
        ret_ids = [c["chunk_id"] for c in ret_chunks]
        
        retrieved_list.append(ret_ids)
        gold_list.append(gold_ids)
        
    if not gold_list:
        return {"status": "not_available", "reason": "no pipeline results matched gold annotations", "simulated": False}
        
    out = {"status": "computed", "simulated": False, "n_queries": len(gold_list)}
    for k in ks:
        out[f"recall_at_{k}"] = round(sum(recall_at_k(r, g, k) for r, g in zip(retrieved_list, gold_list)) / len(gold_list), 6)
        out[f"precision_at_{k}"] = round(sum(precision_at_k(r, g, k) for r, g in zip(retrieved_list, gold_list)) / len(gold_list), 6)
        out[f"ndcg_at_{k}"] = round(sum(ndcg_at_k(r, g, k) for r, g in zip(retrieved_list, gold_list)) / len(gold_list), 6)
    out["mrr"] = mrr(retrieved_list, gold_list)
    return out


def extraction_from_gold(path: str, pipeline_report_path: str = "results/pipeline_report.json", questions_path: str = "data/questions.jsonl", config_path: str = "configs/experiment.yaml") -> dict:
    """
    Format: {"question_id": str, "answer_id": str, "gold_atoms": [{"type": "...", "text": "..."}]}
    """
    p = Path(path)
    if not p.exists():
        return {"status": "not_available", "reason": f"gold file missing: {path}", "simulated": False}
        
    rows = _read_jsonl(p)
    if not rows:
        return {"status": "not_available", "reason": f"no gold atom annotations at {path}", "simulated": False}
        
    pipeline_queries = _get_pipeline_queries(pipeline_report_path, questions_path)
    
    # Check config for fuzzy threshold
    fuzzy_threshold = 1.0 # 1.0 means exact match
    if Path(config_path).exists():
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            fuzzy_threshold = float(cfg.get("fuzzy_match_threshold", 1.0))
            
    by_type = {}
    for r in rows:
        qid = r["question_id"]
        if qid not in pipeline_queries:
            continue
            
        gold_atoms = r.get("gold_atoms", [])
        pred_atoms = pipeline_queries[qid].get("atoms", [])
        
        # Group by type
        g_by_t = {}
        for ga in gold_atoms:
            g_by_t.setdefault(ga["type"], []).append(ga["text"])
            
        p_by_t = {}
        for pa in pred_atoms:
            # handle case where pred_atoms is flat dict vs objects
            atype = pa.get("type", pa.get("atom_type", "UNKNOWN"))
            p_by_t.setdefault(atype, []).append(pa.get("text", ""))
            
        # Combine types found in either
        all_types = set(g_by_t.keys()) | set(p_by_t.keys())
        for t in all_types:
            by_type.setdefault(t, {"preds": [], "golds": []})
            by_type[t]["preds"].append(p_by_t.get(t, []))
            by_type[t]["golds"].append(g_by_t.get(t, []))
            
    if not by_type:
        return {"status": "not_available", "reason": "no pipeline results matched gold annotations", "simulated": False}
        
    table = {"status": "computed", "simulated": False}
    for atom_type, group in sorted(by_type.items()):
        # Flatten
        preds = [p for sub in group["preds"] for p in sub]
        golds = [g for sub in group["golds"] for g in sub]
        
        rec = extraction_recall(preds, golds, fuzzy_threshold=fuzzy_threshold)
        prec = extraction_precision(preds, golds, fuzzy_threshold=fuzzy_threshold)
        table[atom_type] = {
            "recall": rec,
            "precision": prec,
            "f1": extraction_f1(rec, prec),
            "n_gold": len(golds),
            "n_pred": len(preds)
        }
    return table
