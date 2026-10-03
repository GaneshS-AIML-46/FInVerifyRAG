"""
scripts/assign_doc_splits.py
-----------------------------
Phase B: Assign document-level train/calib/test splits.

Reads data/doc_inventory.csv, assigns each document to exactly one split,
stratified by (regulator, doc_type) with ratios from configs/experiment.yaml:
  train: 0.40  calib: 0.35  test: 0.25

Rules (frozen):
  - Append-only: existing rows in doc_assignment.csv are never changed.
  - A doc that was once test stays test forever.
  - New documents (not yet in the assignment file) are assigned in this run.
  - Warns if any regulator/doc_type stratum has fewer than 3 documents.

Output:
  data/splits/doc_assignment.csv  columns: doc_id, split

Usage:
    python scripts/assign_doc_splits.py [--inventory PATH] [--out PATH] [--seed N]
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("assign_doc_splits")

ROOT = Path(__file__).resolve().parent.parent


# ── Config loader ─────────────────────────────────────────────────────────────

def _load_config() -> dict:
    cfg_path = ROOT / "configs" / "experiment.yaml"
    try:
        with open(cfg_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


# ── Existing assignment loader ────────────────────────────────────────────────

def _load_existing(path: Path) -> dict[str, str]:
    """Returns {doc_id: split} from the existing assignment file (if any)."""
    existing: dict[str, str] = {}
    if not path.exists():
        return existing
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            did = row.get("doc_id", "").strip()
            spl = row.get("split", "").strip()
            if did and spl:
                existing[did] = spl
    logger.info("Loaded %d existing assignments from %s", len(existing), path)
    return existing


# ── Stratified split assignment ───────────────────────────────────────────────

def _assign_stratum(
    doc_ids: list[str],
    ratios: dict[str, float],
    seed: int,
) -> dict[str, str]:
    """
    Shuffle doc_ids with fixed seed, then cut into train/calib/test proportions.
    Returns {doc_id: split}.
    """
    rng = random.Random(seed)
    shuffled = list(doc_ids)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = math.ceil(n * ratios["train"])
    n_calib = math.ceil(n * ratios["calib"])
    # Rest goes to test

    splits: list[str] = (
        ["train"] * n_train +
        ["calib"] * n_calib +
        ["test"]  * max(0, n - n_train - n_calib)
    )
    # Safety: if rounding pushed n_train+n_calib > n
    splits = splits[:n]

    return {did: spl for did, spl in zip(shuffled, splits)}


# ── Main ──────────────────────────────────────────────────────────────────────

def assign_splits(
    inventory_path: Path,
    out_path: Path,
    ratios: dict[str, float],
    seed: int,
) -> dict[str, str]:
    """
    Build or update doc_assignment.csv.
    Returns the full {doc_id: split} mapping after this run.
    """
    # Load inventory
    if not inventory_path.exists():
        print(f"ERROR: inventory file not found: {inventory_path}", file=sys.stderr)
        sys.exit(1)

    inventory: list[dict] = []
    with open(inventory_path, newline="", encoding="utf-8") as f:
        inventory = list(csv.DictReader(f))
    logger.info("Loaded %d documents from inventory", len(inventory))

    # Load existing assignments (append-only guarantee)
    existing = _load_existing(out_path)
    already_assigned = set(existing.keys())

    # Find new documents (not yet assigned)
    new_docs = [r for r in inventory if r["doc_id"] not in already_assigned]
    logger.info("%d documents already assigned, %d new", len(already_assigned), len(new_docs))

    # Group new docs by (regulator, doc_type)
    strata: dict[tuple, list[str]] = defaultdict(list)
    for r in new_docs:
        key = (r["regulator"].upper(), r["doc_type"].lower())
        strata[key].append(r["doc_id"])

    # Assign each stratum with warnings for small groups
    new_assignments: dict[str, str] = {}
    small_strata: list[str] = []

    for (reg, dtype), ids in sorted(strata.items()):
        if len(ids) < 3:
            small_strata.append(
                f"  WARNING: {reg}|{dtype} has only {len(ids)} new document(s) "
                f"(< 3 minimum for reliable stratification)"
            )
        assignments = _assign_stratum(ids, ratios, seed)
        new_assignments.update(assignments)

    if small_strata:
        print("\n" + "\n".join(small_strata))

    # Merge: existing first, then new
    final = dict(existing)
    final.update(new_assignments)

    # Write output (sorted by doc_id for reproducibility)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "split"])
        writer.writeheader()
        for doc_id in sorted(final.keys()):
            writer.writerow({"doc_id": doc_id, "split": final[doc_id]})
    logger.info("Wrote %d assignments to %s", len(final), out_path)

    # ── Print summary ──────────────────────────────────────────────────────────
    split_counts: dict[str, int] = defaultdict(int)
    for spl in final.values():
        split_counts[spl] += 1

    print(f"\n{'='*60}")
    print(f"  Document Split Assignment Summary")
    print(f"{'='*60}")
    print(f"  Total documents assigned: {len(final)}")
    print(f"  New assignments this run: {len(new_assignments)}")
    for spl in ("train", "calib", "test"):
        cnt = split_counts.get(spl, 0)
        pct = 100.0 * cnt / max(len(final), 1)
        print(f"    {spl:6s}: {cnt:5d}  ({pct:.1f}%)")

    # Per-stratum breakdown
    strat_split: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    id_to_meta = {r["doc_id"]: r for r in inventory}
    for doc_id, spl in final.items():
        meta = id_to_meta.get(doc_id, {})
        key = (meta.get("regulator", "?"), meta.get("doc_type", "?"))
        strat_split[key][spl] += 1

    print("\n  Per (regulator|doc_type) split breakdown:")
    print(f"  {'Stratum':<40} {'train':>6} {'calib':>6} {'test':>6} {'total':>6}")
    print("  " + "-" * 62)
    for (reg, dtype), counts in sorted(strat_split.items()):
        tot = sum(counts.values())
        tr = counts.get("train", 0)
        ca = counts.get("calib", 0)
        te = counts.get("test", 0)
        label = f"{reg}|{dtype}"
        warn = " ⚠" if tot < 3 else ""
        print(f"  {label:<40} {tr:>6} {ca:>6} {te:>6} {tot:>6}{warn}")

    print(f"\n  Ratios: train={ratios['train']:.2f} calib={ratios['calib']:.2f} test={ratios['test']:.2f}  seed={seed}")
    print(f"  Output: {out_path}")
    print(f"{'='*60}\n")

    return final


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    cfg = _load_config()
    paths   = cfg.get("paths", {})
    sp_cfg  = cfg.get("splits", {})

    default_inv = str(ROOT / paths.get("doc_inventory", "data/doc_inventory.csv"))
    default_out = str(ROOT / paths.get("doc_splits", "data/splits/doc_assignment.csv"))
    default_seed = int(sp_cfg.get("seed", 42))
    default_ratios = {
        "train": float(sp_cfg.get("train", 0.40)),
        "calib": float(sp_cfg.get("calib", 0.35)),
        "test":  float(sp_cfg.get("test",  0.25)),
    }

    # Validate ratios sum to ~1
    total = sum(default_ratios.values())
    if not (0.99 < total < 1.01):
        print(f"ERROR: split ratios sum to {total:.3f}, must be ~1.0", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Assign document-level train/calib/test splits.")
    parser.add_argument("--inventory", default=default_inv,
                        help="Path to doc_inventory.csv")
    parser.add_argument("--out", default=default_out,
                        help="Output path for doc_assignment.csv")
    parser.add_argument("--seed", type=int, default=default_seed,
                        help="Random seed (default from config)")
    args = parser.parse_args()

    # Re-parse ratios in case seed changed but ratios stayed from config
    assign_splits(
        inventory_path=Path(args.inventory),
        out_path=Path(args.out),
        ratios=default_ratios,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
