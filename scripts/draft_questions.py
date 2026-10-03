import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv
import pickle
import random
import json
from common.llm_client import LLMClient

def draft_questions(n_samples=10, out_csv="data/draft_questions.csv"):
    print("Loading index...")
    with open(".bm25_index.pkl", "rb") as f:
        raw = pickle.load(f)
    chunks = raw["chunks"]
    
    print(f"Selecting {n_samples} random chunks...")
    samples = random.sample(chunks, n_samples)
    
    llm = LLMClient()
    system_prompt = (
        "You are a regulatory QA expert. I will give you a chunk of a regulatory document. "
        "Your task is to draft a realistic question that a user might ask, which can be answered by this chunk. "
        "Also determine the regulator (RBI or SEBI) and the 'family' of the question (one of: rate, threshold, section, date, entity, applicability). "
        "Output ONLY a JSON object with keys: 'question', 'regulator', 'family'."
    )
    
    results = []
    
    for i, c in enumerate(samples):
        print(f"Drafting question {i+1}/{n_samples}...")
        prompt = f"CHUNK:\\n{c.text[:1500]}"
        try:
            resp = llm.chat(user=prompt, system=system_prompt, response_format={"type": "json_object"})
            data = json.loads(resp["text"])
            
            # The chunk issue date / effective date
            q_date = c.metadata.get("effective_from") or c.issue_date or "2024-01-01"
            
            results.append({
                "status": "DRAFT - needs human review",
                "question": data.get("question", ""),
                "query_date": q_date,
                "regulator": data.get("regulator", c.regulator),
                "family": data.get("family", "entity"),
                "gold_doc_ids": c.chunk_id
            })
        except Exception as e:
            print(f"Error drafting for {c.chunk_id}: {e}")
            
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["status", "question", "query_date", "regulator", "family", "gold_doc_ids"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Drafted {len(results)} questions to {out_csv}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Number of questions to draft")
    args = parser.parse_args()
    draft_questions(n_samples=args.n)
