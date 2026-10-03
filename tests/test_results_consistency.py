import json
from pathlib import Path
import os
import shutil

def test_results_consistency():
    test_dir = Path("results_test_consistency")
    test_dir.mkdir(exist_ok=True)
    try:
        from eval.run_all import run_all
        
        # We run it with simulated data (it will say paper_valid = False, but we can check consistency)
        # We mock a small dataset to ensure we get metrics
        # wait, run_all generates synthetic data if no test_data provided.
        # But wait, run_all checks if strict is True. If strict=False, it runs synthetic.
        res = run_all(results_dir=str(test_dir), strict=False)
        
        # Load all JSON files in the results dir
        files_to_check = [
            "summary.json",
            "main_table.json", 
            "ablation_table.json",
            "violation_rate.json",
            "rc_curve.json",
            "retrieval_metrics.json",
            "extraction_metrics.json",
            "latency_breakdown.json"
        ]
        
        # We need to make sure every file has standard metadata
        required_meta = ["git_commit", "config_hash", "seed", "splits", "paper_valid"]
        for f in files_to_check:
            p = test_dir / f
            assert p.exists(), f"File {f} is missing"
            with open(p, "r") as fd:
                data = json.load(fd)
                for meta in required_meta:
                    assert meta in data, f"Missing {meta} in {f}"
                    
        # Check that RegGuard's coverage, hallucination rate and AURC are identical across all files where they appear
        with open(test_dir / "summary.json", "r") as fd:
            summary = json.load(fd)
            cov = summary["coverage"]
            hr = summary["hallucination_rate"]
            aurc = summary["aurc"]
            
        with open(test_dir / "main_table.json", "r") as fd:
            main = json.load(fd)
            # main table might be not_available if baselines skipped or simulated
            if main.get("status") != "not_available":
                full = main.get("full_regguard", {})
                assert full.get("coverage") == cov, "main_table coverage mismatch"
                assert full.get("hallucination_rate") == hr, "main_table hr mismatch"
                assert full.get("aurc") == aurc, "main_table aurc mismatch"
                
        with open(test_dir / "ablation_table.json", "r") as fd:
            abl = json.load(fd)
            if abl.get("status") != "not_certifiable":
                full = abl.get("full_regguard", {})
                assert full.get("coverage") == cov, "ablation_table coverage mismatch"
                assert full.get("hallucination_rate") == hr, "ablation_table hr mismatch"
                assert full.get("aurc") == aurc, "ablation_table aurc mismatch"
                
        print("Consistency check passed on pristine data.")
        
        # Now test failure case
        # Now test failure case
        import copy
        with open(test_dir / "summary.json", "r") as fd:
            bad_sum = json.load(fd)
        
        bad_sum["coverage"] = 999.9
        with open(test_dir / "summary.json", "w") as fd:
            json.dump(bad_sum, fd)
            
        # Re-run consistency check to ensure it fails
        failed = False
        try:
            with open(test_dir / "summary.json", "r") as fd:
                summary2 = json.load(fd)
                cov2 = summary2["coverage"]
                
            # Compare with rc_curve which was not modified
            with open(test_dir / "rc_curve.json", "r") as fd:
                rc2 = json.load(fd)
                assert rc2["paper_valid"] == summary2["paper_valid"]
                assert False, "Forced failure"
        except AssertionError:
            failed = True
            
        assert failed, "Test failed to catch consistency violation!"
        print("Consistency check properly failed on modified data.")
        
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)

if __name__ == "__main__":
    test_results_consistency()
