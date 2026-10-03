import csv
import json
import os
import pickle
from collections import defaultdict

def load_chunks(index_path=".bm25_index.pkl"):
    if not os.path.exists(index_path):
        return {}
    with open(index_path, "rb") as f:
        data = pickle.load(f)
    return {c.chunk_id: c.text for c in data["chunks"]}

def export_for_labeling(atoms_file, csv_file, annotator_id=""):
    os.makedirs(os.path.dirname(csv_file) or ".", exist_ok=True)
    chunks = load_chunks()
    
    with open(atoms_file, "r", encoding="utf-8") as f, \
         open(csv_file, "w", newline="", encoding="utf-8") as out:
         
        writer = csv.writer(out)
        writer.writerow(["id", "question_id", "doc_id", "atom_type", "atom_text", "source_text", "label", "annotator_id"])
        
        for line in f:
            if not line.strip(): continue
            a = json.loads(line)
            source_text = ""
            for cid in a.get("cited_chunk_ids", []):
                source_text += chunks.get(cid, "[SOURCE NOT FOUND]") + "\\n\\n"
            
            writer.writerow([
                a.get("id"),
                a.get("question_id"),
                a.get("doc_id"),
                a.get("atom_type"),
                a.get("atom_text"),
                source_text.strip(),
                "", # label to be filled (SUPPORTED, UNSUPPORTED, UNCLEAR)
                annotator_id
            ])
    print(f"Exported atoms for labeling to {csv_file}")

def compute_kappa(file1, file2, reviewer_out):
    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError:
        print("Please install scikit-learn for Cohen's Kappa.")
        return

    data1 = {}
    with open(file1, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            data1[row["id"]] = row
            
    data2 = {}
    with open(file2, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            data2[row["id"]] = row
            
    shared_ids = set(data1.keys()).intersection(data2.keys())
    if not shared_ids:
        print("No shared atoms found.")
        return
        
    labels1, labels2 = [], []
    disagreements = []
    
    for aid in shared_ids:
        l1 = data1[aid].get("label", "").upper()
        l2 = data2[aid].get("label", "").upper()
        if not l1 or not l2: continue
        
        labels1.append(l1)
        labels2.append(l2)
        
        if l1 != l2:
            disagreements.append({
                "id": aid,
                "atom_text": data1[aid].get("atom_text"),
                "source_text": data1[aid].get("source_text"),
                "annotator_1_label": l1,
                "annotator_2_label": l2,
                "reviewer_decision": ""
            })
            
    kappa = cohen_kappa_score(labels1, labels2)
    print(f"Analyzed {len(labels1)} shared labeled atoms.")
    print(f"Cohen's Kappa: {kappa:.4f}")
    
    if disagreements:
        with open(reviewer_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "atom_text", "source_text", "annotator_1_label", "annotator_2_label", "reviewer_decision"])
            writer.writeheader()
            writer.writerows(disagreements)
        print(f"Wrote {len(disagreements)} disagreements to {reviewer_out}")
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd")
    
    p_export = subparsers.add_parser("export")
    p_export.add_argument("--atoms", required=True)
    p_export.add_argument("--out", required=True)
    p_export.add_argument("--annotator", default="")
    
    p_kappa = subparsers.add_parser("kappa")
    p_kappa.add_argument("--f1", required=True)
    p_kappa.add_argument("--f2", required=True)
    p_kappa.add_argument("--out", required=True)
    
    args = parser.parse_args()
    if args.cmd == "export":
        export_for_labeling(args.atoms, args.out, args.annotator)
    elif args.cmd == "kappa":
        compute_kappa(args.f1, args.f2, args.out)
