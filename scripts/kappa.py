"""Compute Cohen's kappa between two annotators.

Usage:
    python scripts/kappa.py --a <a.jsonl> --b <b.jsonl> --out <disagreements.csv> [--adjudicated <adj.csv>]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

CATEGORIES = ["supported", "unsupported", "unclear"]


def load_labels(path: str) -> dict:
    """Load {atom_id: {label, annotator_id}} from a JSONL file."""
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            data[rec["atom_id"]] = rec
    return data


def compute_kappa(labels_a: dict, labels_b: dict):
    """Return (n_shared, kappa, confusion, p_o)."""
    shared_ids = sorted(set(labels_a) & set(labels_b))
    n = len(shared_ids)

    if n == 0:
        return 0, 0.0, {}, 0.0

    # Build lists
    a_list = [labels_a[i]["label"] for i in shared_ids]
    b_list = [labels_b[i]["label"] for i in shared_ids]

    # Confusion matrix: rows=A, cols=B
    confusion = {c: {d: 0 for d in CATEGORIES} for c in CATEGORIES}
    agree = 0
    for la, lb in zip(a_list, b_list):
        confusion[la][lb] += 1
        if la == lb:
            agree += 1

    p_o = agree / n

    # Expected agreement
    count_a = {c: a_list.count(c) for c in CATEGORIES}
    count_b = {c: b_list.count(c) for c in CATEGORIES}
    p_e = sum((count_a[c] / n) * (count_b[c] / n) for c in CATEGORIES)

    if p_e == 1.0:
        kappa = 1.0
    else:
        kappa = (p_o - p_e) / (1 - p_e)

    return n, kappa, confusion, p_o, shared_ids, a_list, b_list


def main():
    parser = argparse.ArgumentParser(description="Compute Cohen's kappa")
    parser.add_argument("--a", required=True, help="Labels JSONL from annotator A")
    parser.add_argument("--b", required=True, help="Labels JSONL from annotator B")
    parser.add_argument("--out", required=True, help="Disagreements CSV output")
    parser.add_argument("--adjudicated", help="Optional adjudication CSV (atom_id, adjudicated_label)")
    args = parser.parse_args()

    labels_a = load_labels(args.a)
    labels_b = load_labels(args.b)

    n, kappa, confusion, p_o, shared_ids, a_list, b_list = compute_kappa(labels_a, labels_b)

    if n < 20:
        print("WARNING: Too few shared atoms for reliable kappa")

    print(f"n_shared: {n}")
    print(f"Cohen's kappa: {kappa:.4f}")
    print(f"Agreement rate: {p_o:.4f}")

    # Print confusion matrix
    print("\nConfusion matrix (rows=A, cols=B):")
    header = f"{'':12s}" + "  ".join(f"{c:12s}" for c in CATEGORIES)
    print(header)
    for c in CATEGORIES:
        row = f"{c:12s}" + "  ".join(f"{confusion[c][d]:12d}" for d in CATEGORIES)
        print(row)

    # Write disagreements CSV
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["atom_id", "label_a", "label_b"])
        writer.writeheader()
        for aid, la, lb in zip(shared_ids, a_list, b_list):
            if la != lb:
                writer.writerow({"atom_id": aid, "label_a": la, "label_b": lb})

    n_disagreements = sum(1 for la, lb in zip(a_list, b_list) if la != lb)
    print(f"\nDisagreements: {n_disagreements} written to {args.out}")

    # Optional adjudication
    if args.adjudicated:
        adj_path = Path(args.adjudicated)
        if not adj_path.exists():
            print(f"Adjudication file not found: {args.adjudicated}")
        else:
            with open(adj_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                print("\nAdjudication output:")
                for row in reader:
                    aid = row["atom_id"]
                    adj_label = row["adjudicated_label"]
                    la = labels_a.get(aid, {}).get("label", "N/A")
                    lb = labels_b.get(aid, {}).get("label", "N/A")
                    ann_a = labels_a.get(aid, {}).get("annotator_id", "A")
                    ann_b = labels_b.get(aid, {}).get("annotator_id", "B")
                    print(f"{aid}, {ann_a}, {la}, {ann_b}, {lb}, {adj_label}")


if __name__ == "__main__":
    main()
