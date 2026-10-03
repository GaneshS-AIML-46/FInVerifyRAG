import json
from common.validation import load_and_validate_data
from common.errors import StrictFailure

def create_and_test(filename, rows, description):
    import os
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    try:
        load_and_validate_data(filename, strict=True)
        print(f"FAIL: {description} did NOT raise an exception.")
    except StrictFailure as e:
        print(f"PASS: {description} raised -> {e}")

if __name__ == "__main__":
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
    # Make other signals not constant
    valid_row = {"label": "unsupported", "annotator_id": "human_1", "s_ret": 0.5, "s_ver": 0.1, "s_nli": 0.2, "s_ent": 0.3, "s_div": 0.4}
    valid_row2 = {"label": "unsupported", "annotator_id": "human_1", "s_ret": 0.8, "s_ver": 0.9, "s_nli": 0.8, "s_ent": 0.7, "s_div": 0.6}
    
    missing_ver_row = {"label": "unsupported", "annotator_id": "human_1", "s_ret": 0.1, "s_nli": 0.2, "s_ent": 0.3, "s_div": 0.4}
    missing_ver_row2 = {"label": "unsupported", "annotator_id": "human_1", "s_ret": 0.9, "s_nli": 0.8, "s_ent": 0.7, "s_div": 0.6}
    
    # 6 rows total. 2 have s_ver missing. 2/6 = 33% > 20%
    create_and_test(
        "results_mock/bad_missing.jsonl",
        [valid_row, valid_row2, valid_row, valid_row2, missing_ver_row, missing_ver_row2],
        "Missing > 20% check"
    )

    # Test 4: Constant signal (std < 1e-3)
    create_and_test(
        "results_mock/bad_constant.jsonl",
        [valid_row, valid_row, valid_row, valid_row, valid_row, valid_row],
        "Constant signal (std < 1e-3) check"
    )
    
    # Test 5: Unique fraction < 5%
    # We need <5% unique. 50 items. 49 are identical, 1 is different.
    # But wait, if they are identical, std might be small? 
    # If 1 is different, std > 0. Let's make sure std > 1e-3.
    # 49 items are 0.5, 1 item is 1.0. std is approx sqrt( (0.5^2)/50 ) ~ 0.07 > 1e-3.
    # But other signals must be valid (not constant, unique > 5%).
    # So let's make all other signals completely random/unique for each row.
    import random
    rows_unique = []
    for i in range(49):
        rows_unique.append({
            "label": "unsupported", "annotator_id": "human_1",
            "s_ret": random.random(), "s_ver": random.random(), 
            "s_nli": random.random(), "s_ent": 0.5, "s_div": random.random()
        })
    rows_unique.append({
        "label": "unsupported", "annotator_id": "human_1",
        "s_ret": random.random(), "s_ver": random.random(), 
        "s_nli": random.random(), "s_ent": 1.0, "s_div": random.random()
    })
    
    create_and_test(
        "results_mock/bad_unique.jsonl",
        rows_unique,
        "Unique fraction < 5% check"
    )
