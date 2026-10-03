import json
import csv
from pathlib import Path

def generate_templates(questions_path="data/questions.jsonl", out_dir="data/gold"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    questions = []
    if Path(questions_path).exists():
        with open(questions_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))
                    
    retrieval_path = Path(out_dir) / "template_retrieval.csv"
    with open(retrieval_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["question_id", "question", "gold_chunk_ids_comma_separated"])
        for q in questions:
            qid = q.get("id", q.get("question_id", q.get("question", "")))
            writer.writerow([qid, q.get("question", ""), ""])
            
    extraction_path = Path(out_dir) / "template_extraction.csv"
    with open(extraction_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["question_id", "answer_id", "gold_atoms_json_or_csv", "notes"])
        for q in questions:
            qid = q.get("id", q.get("question_id", q.get("question", "")))
            writer.writerow([qid, "", '[{"type": "RATE", "text": "example"}]', ""])
            
    print(f"Templates generated in {out_dir}")

if __name__ == "__main__":
    generate_templates()
