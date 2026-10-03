import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import time
import argparse
import logging
from common.experiment import load_experiment
from common.llm_client import LLMClient
from scripts.run_full_eval import build_retriever, run_one_query
from judge.judge_llm import JudgeLLM
from generation.generator import Generator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("generate_dataset")

def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def append_jsonl(path, data):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\\n")

def generate_dataset(questions_file, out_file, checkpoint_file):
    if not os.path.exists(questions_file):
        raise FileNotFoundError(f"{questions_file} not found")
        
    questions = load_jsonl(questions_file)
    logger.info(f"Loaded {len(questions)} questions from {questions_file}")
    
    # Read checkpoint to skip already processed question IDs
    processed = set()
    if os.path.exists(checkpoint_file):
        lines = load_jsonl(checkpoint_file)
        processed = {r["question_id"] for r in lines if "question_id" in r}
        
    exp = load_experiment()
    retriever = build_retriever()
    llm_gen = LLMClient(config=exp.get("llm_generate"))
    generator = Generator(llm_client=llm_gen)
    thresholds = exp.get("thresholds", [])
    judge = JudgeLLM(config=exp.get("llm_judge"))
    
    max_retries = exp.get("query_retries", 1)
    
    for i, q in enumerate(questions):
        qid = q.get("id")
        if not qid:
            logger.error(f"Question missing id: {q}")
            continue
            
        if qid in processed:
            logger.info(f"Skipping {qid} (already processed)")
            continue
            
        q_text = q.get("question")
        q_date = q.get("query_date")
        
        logger.info(f"Processing [{i+1}/{len(questions)}] {qid}: {q_text}")
        
        for attempt in range(max_retries + 1):
            try:
                # We need to run the query and get the atoms
                r = run_one_query(q_text, q_date, retriever, generator, thresholds, judge, exp, strict=False)
                atom_rows = r.pop("atoms")
                
                # Write each atom to out_file
                for row in atom_rows:
                    atom_record = {
                        "id": f"{qid}_{row.get('atom_id', row.get('id', 'unknown'))}",
                        "question_id": qid,
                        "doc_id": row.get("cited_chunk_id", ""),
                        "query_date": q_date,
                        "atom_type": row.get("atom_type", ""),
                        "stratum": row.get("stratum", ""),
                        "atom_text": row.get("text", ""),
                        "cited_chunk_ids": [row.get("cited_chunk_id")] if row.get("cited_chunk_id") else [],
                        "s_ret": row.get("s_ret", 0),
                        "s_nli": row.get("s_nli", 0),
                        "s_div": row.get("s_div", 0),
                        "s_ent": row.get("s_ent", 0),
                        "v1_status": row.get("v1_status", ""),
                        "v2_entail_prob": row.get("v2_entail_prob"),
                        "aggregated_risk": row.get("risk", 0),
                        "decision": row.get("decision", ""),
                        "judge_verdict": row.get("judge_reason", ""),
                        "latency": r.get("latency", {}),
                        "synthetic": True
                    }
                    append_jsonl(out_file, atom_record)
                    
                # Write to checkpoint so we don't re-run this question
                append_jsonl(checkpoint_file, {"question_id": qid, "status": "OK"})
                break
            except Exception as exc:
                if attempt < max_retries:
                    logger.warning(f"Failed attempt {attempt+1}/{max_retries}. Retrying... {exc}")
                    time.sleep(2)
                else:
                    logger.error(f"Failed completely on {qid}: {exc}")
                    append_jsonl(checkpoint_file, {"question_id": qid, "status": "ERROR", "error": str(exc)})
                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/questions.jsonl")
    parser.add_argument("--output", default="results_mock/raw_atoms.jsonl")
    parser.add_argument("--checkpoint", default="results_mock/checkpoints/generate_dataset_cp.jsonl")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
    generate_dataset(args.input, args.output, args.checkpoint)
