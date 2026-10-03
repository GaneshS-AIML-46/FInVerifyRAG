import re

def update_run_eval():
    with open("S:\\FYP\\M0_M10Complete_pipeline\\scripts\\run_full_eval.py", "r", encoding="utf-8") as f:
        code = f.read()

    # 1. Update try/except block for retries
    old_try_block = """        try:
            r = run_one_query(q["question"], q["query_date"], retriever, generator, thresholds, judge, exp, strict=args.strict)
            r["family"] = q.get("family", "")
            atom_rows = r.pop("atoms")   # pop atoms for all_atoms, keep separately for printing
            all_atoms.extend(atom_rows)
            query_results.append(r)
            append_jsonl(checkpoint, {"question": q["question"], "status": "OK", "n_atoms": r["n_atoms"], **stamp})
            print_query_result(i, len(questions), q, r, atom_rows)
            if r.get("answer_status") == "ERROR":
                msg = f"Query {i} ended with status ERROR."
                has_errors = True
                print_not_paper_valid_banner(msg)
                paper_valid = False
        except Exception as exc:
            append_jsonl(checkpoint, {"question": q["question"], "status": "ERROR", "error": str(exc), **stamp})
            msg = f"Query {i} ended with status ERROR: {exc}"
            has_errors = True
            print_not_paper_valid_banner(msg)
            paper_valid = False
            logger.error("Query %d ERROR: %s", i, exc, exc_info=True)
            query_results.append({
                "question": q["question"], "family": q["family"],
                "answer_status": "ERROR", "coverage": 0.0, "n_atoms": 0,
                "n_supported": 0, "n_verified": 0, "n_uncertain": 0,
                "n_abstained": 0, "n_not_verified": 0, "judge_calls": 0,
                "latency": {}, "answer": "", "sources": [], "error": str(exc),
            })
            # Still print the error row
            print(f"\\n{'='*72}")
            print(f"  QUERY {i}/{len(questions)}  [{q.get('family','').upper()}]  -- ERROR")
            print(f"  {str(exc)[:200]}")"""
            
    new_try_block = """        max_retries = exp.get("query_retries", 1)
        for attempt in range(max_retries + 1):
            try:
                r = run_one_query(q["question"], q["query_date"], retriever, generator, thresholds, judge, exp, strict=args.strict)
                r["family"] = q.get("family", "")
                atom_rows = r.pop("atoms")   # pop atoms for all_atoms, keep separately for printing
                all_atoms.extend(atom_rows)
                query_results.append(r)
                append_jsonl(checkpoint, {"question": q["question"], "status": "OK", "n_atoms": r["n_atoms"], **stamp})
                print_query_result(i, len(questions), q, r, atom_rows)
                if r.get("answer_status") == "ERROR":
                    msg = f"Query {i} ended with status ERROR."
                    has_errors = True
                    print_not_paper_valid_banner(msg)
                    paper_valid = False
                break  # Success!
            except Exception as exc:
                if attempt < max_retries:
                    logger.warning("Query %d failed (attempt %d/%d). Retrying... Error: %s", i, attempt+1, max_retries, exc)
                    import time
                    time.sleep(2)
                    continue
                
                append_jsonl(checkpoint, {"question": q["question"], "status": "ERROR", "error": str(exc), **stamp})
                msg = f"Query {i} ended with status ERROR after {max_retries} retries: {exc}"
                has_errors = True
                print_not_paper_valid_banner(msg)
                paper_valid = False
                logger.error("Query %d ERROR: %s", i, exc, exc_info=True)
                query_results.append({
                    "question": q["question"], "family": q["family"],
                    "answer_status": "ERROR", "coverage": 0.0, "n_atoms": 0,
                    "n_supported": 0, "n_verified": 0, "n_uncertain": 0,
                    "n_abstained": 0, "n_not_verified": 0, "judge_calls": 0,
                    "parse_errors": 0,
                    "latency": {}, "answer": "", "sources": [], "error": str(exc),
                })
                # Still print the error row
                print(f"\\n{'='*72}")
                print(f"  QUERY {i}/{len(questions)}  [{q.get('family','').upper()}]  -- ERROR")
                print(f"  {str(exc)[:200]}")
                break"""
                
    if old_try_block in code:
        code = code.replace(old_try_block, new_try_block)
    else:
        print("Could not find try_block to replace.")
        
    # 2. Log parse_failures in summary.json
    summary_block = """    s_data["paper_valid"] = bool(paper_valid and s_data.get("paper_valid", True))"""
    new_summary_block = """    s_data["paper_valid"] = bool(paper_valid and s_data.get("paper_valid", True))
    
    from atoms.extractor import get_atom_splitter
    splitter = get_atom_splitter()
    if splitter:
        s_data["parse_failures"] = splitter.llm_extractor.parse_failures
    else:
        s_data["parse_failures"] = 0
"""
    if summary_block in code:
        code = code.replace(summary_block, new_summary_block)
    else:
        print("Could not find summary_block to replace.")

    with open("S:\\FYP\\M0_M10Complete_pipeline\\scripts\\run_full_eval.py", "w", encoding="utf-8") as f:
        f.write(code)

if __name__ == '__main__':
    update_run_eval()
