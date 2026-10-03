import json
import os
import shutil
from pathlib import Path
from eval.gold import retrieval_from_gold, extraction_from_gold
from scripts.generate_gold_templates import generate_templates

def test_gold_evaluation():
    test_dir = Path("data/test_gold_temp")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Mock pipeline_report.json
        pipeline_report = {
            "queries": [
                {
                    "question": "q1",
                    "sources": [{"chunk_id": "c1"}, {"chunk_id": "c2"}],
                    "atoms": [{"type": "RATE", "text": "5%"}, {"type": "DATE", "text": "2024-01-01"}]
                }
            ]
        }
        with open(test_dir / "pipeline_report.json", "w") as f:
            json.dump(pipeline_report, f)
            
        # Mock questions.jsonl
        with open(test_dir / "questions.jsonl", "w") as f:
            f.write(json.dumps({"id": "q1", "question": "q1"}) + "\n")
            
        # Mock chunks.jsonl
        with open(test_dir / "chunks.jsonl", "w") as f:
            f.write(json.dumps({"question_id": "q1", "gold_chunk_ids": ["c1"]}) + "\n")
            
        # Mock atoms.jsonl
        with open(test_dir / "atoms.jsonl", "w") as f:
            f.write(json.dumps({
                "question_id": "q1",
                "answer_id": "a1",
                "gold_atoms": [{"type": "RATE", "text": "5% "}, {"type": "DATE", "text": "2024-01-02"}]
            }) + "\n")
            
        # Test missing file gives not_available
        res = retrieval_from_gold("data/does_not_exist.jsonl", str(test_dir / "pipeline_report.json"), str(test_dir / "questions.jsonl"))
        assert res["status"] == "not_available"
        print("Missing file check passed.")
        
        # Test perfect retrieval
        ret_metrics = retrieval_from_gold(str(test_dir / "chunks.jsonl"), str(test_dir / "pipeline_report.json"), str(test_dir / "questions.jsonl"))
        assert ret_metrics["status"] == "computed"
        assert ret_metrics["recall_at_6"] == 1.0, f"Expected 1.0, got {ret_metrics['recall_at_6']}"
        assert ret_metrics["mrr"] == 1.0, f"Expected 1.0, got {ret_metrics['mrr']}"
        print("Retrieval perfect match passed.")
        
        # Test extraction matching (fuzzy threshold 1.0 by default -> exact match after normalize)
        # Gold RATE is "5% ", Pred RATE is "5%" -> Match (normalized)
        # Gold DATE is "2024-01-02", Pred DATE is "2024-01-01" -> No Match
        # So RATE F1 should be 1.0, DATE F1 should be 0.0
        ext_metrics = extraction_from_gold(str(test_dir / "atoms.jsonl"), str(test_dir / "pipeline_report.json"), str(test_dir / "questions.jsonl"), "configs/does_not_exist.yaml")
        assert ext_metrics["status"] == "computed"
        assert ext_metrics["RATE"]["f1"] == 1.0, f"Expected 1.0, got {ext_metrics['RATE']['f1']}"
        assert ext_metrics["DATE"]["f1"] == 0.0, f"Expected 0.0, got {ext_metrics['DATE']['f1']}"
        print("Extraction F1 match passed.")
        
        # Test templates
        generate_templates(questions_path=str(test_dir / "questions.jsonl"), out_dir=str(test_dir))
        assert (test_dir / "template_retrieval.csv").exists()
        assert (test_dir / "template_extraction.csv").exists()
        print("Templates generated successfully.")
        with open(test_dir / "template_retrieval.csv") as f:
            print("\nRetrieval Template Head:")
            print("".join([f.readline() for _ in range(2)]))
            
    finally:
        shutil.rmtree(test_dir)

if __name__ == "__main__":
    test_gold_evaluation()
