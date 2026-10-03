import re

def update_combine():
    with open("S:\\FYP\\M0_M10Complete_pipeline\\decision\\combine.py", "r", encoding="utf-8") as f:
        code = f.read()

    # Add parse_errors to AnswerDecision
    code = code.replace(
        "    judge_calls:     int = 0\n    coverage:        float = 0.0",
        "    judge_calls:     int = 0\n    parse_errors:    int = 0\n    coverage:        float = 0.0"
    )

    # Add parse_errors counting in combine
    code = code.replace(
        "    n_not_verified = 0\n    judge_calls    = len(judge_results)",
        "    n_not_verified = 0\n    parse_errors   = 0\n    judge_calls    = len(judge_results)"
    )

    code = code.replace(
        "                final_status = NOT_VERIFIED\n                rationale    = jr.rationale\n                ev_quote     = jr.evidence_quote\n                n_not_verified += 1",
        "                final_status = NOT_VERIFIED\n                rationale    = jr.rationale\n                ev_quote     = jr.evidence_quote\n                n_not_verified += 1\n                if getattr(jr, 'parse_error', False):\n                    parse_errors += 1"
    )

    code = code.replace(
        "        judge_calls=judge_calls,\n        coverage=round(coverage, 4),",
        "        judge_calls=judge_calls,\n        parse_errors=parse_errors,\n        coverage=round(coverage, 4),"
    )

    # Modify decide to accept mode
    code = code.replace(
        "    answer:         GeneratedAnswer,\n    skip_judge:     bool = False,\n) -> AnswerDecision:",
        "    answer:         GeneratedAnswer,\n    skip_judge:     bool = False,\n    mode:           str = 'selective',\n) -> AnswerDecision:"
    )

    code = code.replace(
        "    route_results = _route(scored_atoms, thresholds)\n\n    judge_results: Dict[str, JudgeResult] = {}",
        "    route_results = _route(scored_atoms, thresholds)\n\n    if mode == 'llm_judge_all':\n        from decision.router import UNCERTAIN\n        for r in route_results:\n            r.status = UNCERTAIN\n\n    judge_results: Dict[str, JudgeResult] = {}"
    )

    with open("S:\\FYP\\M0_M10Complete_pipeline\\decision\\combine.py", "w", encoding="utf-8") as f:
        f.write(code)

def update_run_eval():
    with open("S:\\FYP\\M0_M10Complete_pipeline\\scripts\\run_full_eval.py", "r", encoding="utf-8") as f:
        code = f.read()

    code = code.replace(
        "ad = m8_decide(scored_atoms=s_atoms, thresholds=thresholds, judge=judge, rr=rr, answer=answer, skip_judge=(judge is None))",
        "ad = m8_decide(scored_atoms=s_atoms, thresholds=thresholds, judge=judge, rr=rr, answer=answer, skip_judge=(judge is None), mode=exp.get('judge_mode', 'selective'))"
    )

    code = code.replace(
        "\"judge_calls\": ad.judge_calls, \"latency\": lat,",
        "\"judge_calls\": ad.judge_calls, \"parse_errors\": getattr(ad, 'parse_errors', 0), \"latency\": lat,"
    )

    # In print_final_report add the rates
    code = code.replace(
        "    print(f\"  Atoms     : {len(all_atoms)} real atoms collected from pipeline\")\n    print(SEP)",
        "    print(f\"  Atoms     : {len(all_atoms)} real atoms collected from pipeline\")\n    \n    total_judge_calls = sum(r.get('judge_calls', 0) for r in qr)\n    total_parse_errors = sum(r.get('parse_errors', 0) for r in qr)\n    total_atoms = len(all_atoms)\n    judge_call_rate = total_judge_calls / total_atoms if total_atoms else 0\n    parse_error_rate = total_parse_errors / total_judge_calls if total_judge_calls else 0\n    print(f\"  Judge calls : {total_judge_calls}/{total_atoms} ({judge_call_rate:.1%})\")\n    print(f\"  Parse errors: {total_parse_errors}/{total_judge_calls} ({parse_error_rate:.1%})\")\n    print(SEP)"
    )

    # Add to pipeline report JSON
    code = code.replace(
        "\"ablation_table\": m10.get(\"ablation_table\",[]),",
        "\"judge_call_rate\": judge_call_rate,\n            \"parse_error_rate\": parse_error_rate,\n            \"ablation_table\": m10.get(\"ablation_table\",[]),"
    )
    # wait, judge_call_rate and parse_error_rate might not be defined in main, only in print_final_report!
    # I should compute them in main before saving pipeline report.
    code = code.replace(
        "    logger.info(\"Report saved: %s  |  Total time: %.1fs\", rp, time.perf_counter()-t_start)",
        "    logger.info(\"Report saved: %s  |  Total time: %.1fs\", rp, time.perf_counter()-t_start)" # dummy
    )
    # actually let's calculate them in main()
    code = code.replace(
        "    # Save pipeline report",
        "    total_jc = sum(r.get('judge_calls', 0) for r in query_results)\n    total_pe = sum(r.get('parse_errors', 0) for r in query_results)\n    jc_rate = total_jc / len(all_atoms) if all_atoms else 0\n    pe_rate = total_pe / total_jc if total_jc else 0\n    logger.info(\"Judge call rate: %.2f%%, Parse error rate: %.2f%%\", jc_rate*100, pe_rate*100)\n\n    # Save pipeline report"
    )
    code = code.replace(
        "\"judge_call_rate\": judge_call_rate,\n            \"parse_error_rate\": parse_error_rate,",
        "\"judge_call_rate\": jc_rate,\n            \"parse_error_rate\": pe_rate,"
    )

    with open("S:\\FYP\\M0_M10Complete_pipeline\\scripts\\run_full_eval.py", "w", encoding="utf-8") as f:
        f.write(code)

if __name__ == '__main__':
    update_combine()
    update_run_eval()
