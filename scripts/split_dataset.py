import json
import random
import os
import hashlib
from collections import defaultdict

def split_dataset(atoms_file, out_dir="data/splits", seed=42):
    random.seed(seed)
    
    with open(atoms_file, "r", encoding="utf-8") as f:
        atoms = [json.loads(line) for line in f if line.strip()]
        
    # Group atoms by doc_id
    doc_groups = defaultdict(list)
    for a in atoms:
        # Default to a random doc_id for atoms that don't have one, just to ensure they don't all cluster
        doc_id = a.get("doc_id") or f"unknown_{a.get('question_id')}"
        doc_groups[doc_id].append(a)
        
    doc_ids = list(doc_groups.keys())
    random.shuffle(doc_ids)
    
    splits = {"train": [], "calib": [], "test": []}
    
    def count_types(split_list):
        counts = defaultdict(int)
        for a in split_list:
            counts[a.get("atom_type", "UNKNOWN")] += 1
        return counts
        
    # Requirements:
    # Train: 600+ atoms
    # Test: 500+ atoms
    # Calib: 600+ atoms AND >=150 per atom type
    
    # We will distribute doc_ids into splits
    for did in doc_ids:
        group = doc_groups[did]
        
        c_types = count_types(splits["calib"])
        
        # Check if calib needs more atoms or more type diversity
        needs_calib = False
        if len(splits["calib"]) < 600:
            needs_calib = True
        else:
            # Check if any atom type has < 150
            for a in group:
                typ = a.get("atom_type", "UNKNOWN")
                if c_types[typ] < 150:
                    needs_calib = True
                    break
                    
        if needs_calib:
            splits["calib"].extend(group)
            continue
            
        if len(splits["train"]) < 600:
            splits["train"].extend(group)
            continue
            
        # Put the rest into test (or until it hits 500, but usually we just dump the rest)
        splits["test"].extend(group)
        
    os.makedirs(out_dir, exist_ok=True)
    
    for s_name, s_atoms in splits.items():
        out_path = os.path.join(out_dir, f"{s_name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for a in s_atoms:
                f.write(json.dumps(a) + "\n")
        print(f"Split '{s_name}': {len(s_atoms)} atoms")
        
        # Print breakdown
        breakdown = count_types(s_atoms)
        for t, c in breakdown.items():
            print(f"  - {t}: {c}")
            
    # Lock test
    test_path = os.path.join(out_dir, "test.jsonl")
    lock_path = os.path.join(out_dir, "test.lock")
    
    with open(test_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
        
    with open(lock_path, "w") as f:
        f.write(digest)
    print(f"Wrote checksum {digest[:8]}... to {lock_path}")
    
    return splits

def check_leakage(train_path, calib_path, test_path):
    def get_docs(path):
        if not os.path.exists(path): return set()
        with open(path, "r", encoding="utf-8") as f:
            return {json.loads(line).get("doc_id") for line in f if line.strip()}
            
    docs_train = get_docs(train_path)
    docs_calib = get_docs(calib_path)
    docs_test = get_docs(test_path)
    
    leak_train_calib = docs_train.intersection(docs_calib)
    leak_train_test = docs_train.intersection(docs_test)
    leak_calib_test = docs_calib.intersection(docs_test)
    
    total_leaks = len(leak_train_calib) + len(leak_train_test) + len(leak_calib_test)
    if total_leaks > 0:
        print(f"LEAKAGE DETECTED!")
        print(f"Train/Calib overlap: {len(leak_train_calib)} docs")
        print(f"Train/Test overlap: {len(leak_train_test)} docs")
        print(f"Calib/Test overlap: {len(leak_calib_test)} docs")
        return False
        
    print("Leakage check PASSED: No doc_id overlap between splits.")
    return True
    
def enforce_test_lock(test_path, lock_path, is_final_run=False):
    if not os.path.exists(test_path) or not os.path.exists(lock_path):
        return True # nothing to check
        
    with open(lock_path, "r") as f:
        expected = f.read().strip()
        
    with open(test_path, "rb") as f:
        actual = hashlib.sha256(f.read()).hexdigest()
        
    if expected != actual:
        raise ValueError(f"TEST DATA TAMPERED: Checksum mismatch. Expected {expected[:8]}, got {actual[:8]}")
        
    if not is_final_run:
        raise PermissionError("EVALUATION REFUSED: test.jsonl is locked. Set final-run flag to evaluate on test set.")
        
    print("Test lock verified and final-run flag provided. Proceeding...")
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--atoms", default="results/raw_atoms.jsonl")
    parser.add_argument("--out", default="data/splits")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    
    if not args.check_only:
        split_dataset(args.atoms, args.out)
    
    check_leakage(
        os.path.join(args.out, "train.jsonl"),
        os.path.join(args.out, "calib.jsonl"),
        os.path.join(args.out, "test.jsonl")
    )
    
    try:
        enforce_test_lock(os.path.join(args.out, "test.jsonl"), os.path.join(args.out, "test.lock"), is_final_run=False)
    except Exception as e:
        print(f"Lock test (expecting refusal without flag): {e}")
