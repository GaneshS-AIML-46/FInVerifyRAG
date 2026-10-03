import json
from common.validation import load_and_validate_data
from common.errors import StrictFailure

def create_and_test(filename, rows, description, expect_fail=True):
    import os
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    try:
        load_and_validate_data(filename, strict=True)
        if expect_fail:
            print(f"FAIL: {description} did NOT raise an exception.")
        else:
            print(f"PASS: {description} completed without exception.")
    except StrictFailure as e:
        if expect_fail:
            print(f"PASS: {description} raised -> {e}")
        else:
            print(f"FAIL: {description} raised unexpectedly -> {e}")

if __name__ == "__main__":
    import random
    
    # Base valid row for clean test
    # Categorical s_ver must have >= 2 distinct values and rarest >= 2%
    # We will generate 50 rows.
    # To be clean: s_ver = "MATCH" (25), "MISMATCH" (25).
    # s_ret, s_nli, s_ent, s_div = continuous random
    clean_rows = []
    for i in range(50):
        clean_rows.append({
            "label": "unsupported",
            "annotator_id": "human_1",
            "s_ver": "MATCH" if i < 25 else "MISMATCH",
            "s_ret": random.random(),
            "s_nli": random.random(),
            "s_ent": random.random(),
            "s_div": random.random()
        })
        
    create_and_test(
        "results_mock/test_clean.jsonl",
        clean_rows,
        "Clean file (categorical s_ver + continuous)",
        expect_fail=False
    )
    
    # Test categorical s_ver with only one value
    bad_categorical_rows = []
    for i in range(50):
        bad_categorical_rows.append({
            "label": "unsupported",
            "annotator_id": "human_1",
            "s_ver": "MATCH", # Only one value
            "s_ret": random.random(),
            "s_nli": random.random(),
            "s_ent": random.random(),
            "s_div": random.random()
        })
    create_and_test(
        "results_mock/bad_categorical.jsonl",
        bad_categorical_rows,
        "Categorical signal < 2 distinct values check"
    )

    # Test s_mech in row
    bad_mech_rows = clean_rows[:1]
    bad_mech_rows[0]["s_mech"] = 0.5
    create_and_test(
        "results_mock/bad_mech.jsonl",
        bad_mech_rows,
        "Row contains s_mech check"
    )
    
    # Original tests
    # Test 1: Synthetic
    create_and_test(
        "results_mock/bad_synthetic.jsonl",
        [{"label": "unsupported", "annotator_id": "human_1", "synthetic": True, "s_ret": 0.5}],
        "Synthetic flag check"
    )

    # Test 2: Invalid annotator
    create_and_test(
        "results_mock/bad_annotator.jsonl",
        [{"label": "unsupported", "annotator_id": "judge", "s_ret": 0.5}],
        "Annotator validation"
    )

    # Test 3: Missing signal > 20% (s_ver) without indicator
    valid_row = {"label": "unsupported", "annotator_id": "human_1", "s_ret": 0.5, "s_ver": "MATCH", "s_nli": 0.2, "s_ent": 0.3, "s_div": 0.4}
    valid_row2 = {"label": "unsupported", "annotator_id": "human_1", "s_ret": 0.8, "s_ver": "MISMATCH", "s_nli": 0.8, "s_ent": 0.7, "s_div": 0.6}
    
    missing_ver_row = {"label": "unsupported", "annotator_id": "human_1", "s_ret": 0.1, "s_nli": 0.2, "s_ent": 0.3, "s_div": 0.4}
    missing_ver_row2 = {"label": "unsupported", "annotator_id": "human_1", "s_ret": 0.9, "s_nli": 0.8, "s_ent": 0.7, "s_div": 0.6}
    
    create_and_test(
        "results_mock/bad_missing.jsonl",
        [valid_row, valid_row2, valid_row, valid_row2, missing_ver_row, missing_ver_row2],
        "Missing > 20% check"
    )

    # Test 4: Constant signal (std < 1e-3)
    # Using s_div as constant
    bad_constant_rows = []
    for i in range(50):
        bad_constant_rows.append({
            "label": "unsupported",
            "annotator_id": "human_1",
            "s_ver": "MATCH" if i < 25 else "MISMATCH",
            "s_ret": random.random(),
            "s_nli": random.random(),
            "s_ent": random.random(),
            "s_div": 0.5
        })
    create_and_test(
        "results_mock/bad_constant.jsonl",
        bad_constant_rows,
        "Constant signal (std < 1e-3) check"
    )
    
    # Test 5: Unique fraction < 5%
    rows_unique = []
    for i in range(49):
        rows_unique.append({
            "label": "unsupported", "annotator_id": "human_1",
            "s_ret": random.random(), "s_ver": "MATCH" if i < 25 else "MISMATCH", 
            "s_nli": random.random(), "s_ent": 0.5, "s_div": random.random()
        })
    rows_unique.append({
        "label": "unsupported", "annotator_id": "human_1",
        "s_ret": random.random(), "s_ver": "MATCH", 
        "s_nli": random.random(), "s_ent": 1.0, "s_div": random.random()
    })
    
    create_and_test(
        "results_mock/bad_unique.jsonl",
        rows_unique,
        "Unique fraction < 5% check"
    )
