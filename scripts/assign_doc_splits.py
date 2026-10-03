"""
scripts/assign_doc_splits.py
-----------------------------
Phase B: Assign document-level train/calib/test splits.

Group-aware (task 5): documents whose titles normalize to the same string are
treated as a series and assigned to a single split together.  This prevents
the same regulatory circular (issued in different years) from leaking across
train/calib/test boundaries.

Normal (append-only) mode:
  - Existing assignments are never changed.
  - New documents that belong to an already-assigned group inherit that group's
    split; new groups are assigned with stratified sampling at the group level.
  - Logs any cross-split group detected in the existing assignment and aborts
    (human must re-run with --regroup to fix).

One-time re-group mode (--regroup):
  - Ignores the previous assignment entirely; computes a fresh group-aware
    assignment for all documents in the current inventory.
  - Logs every document whose split changed vs the previous assignment to
    data/splits/reassignment_log.csv.
  - Safe ONLY before any question or label files exist (Phase B).

Ratios from configs/experiment.yaml: train=0.40  calib=0.35  test=0.25
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import random
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Optional

import yaml

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("assign_doc_splits")

ROOT = Path(__file__).resolve().parent.parent


# ── Config ────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    cfg_path = ROOT / "configs" / "experiment.yaml"
    try:
        with open(cfg_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


# ── Title normalisation ───────────────────────────────────────────────────────

_MONTHS = (
    r"(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december|"
    r"jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
)


def normalize_title(title: str) -> str:
    """
    Produce a canonical form of *title* for series-duplicate detection.

    Strips:
      • Circular reference codes (RBI/2023-24/123, SEBI/HO/...)
      • Date qualifiers ("dated DD/MM/YYYY", "as on Month YYYY", "updated ...")
      • Standalone dates (DD/MM/YYYY, Month YYYY, 4-digit year 2000-2099)
      • Amendment-only parentheticals: (Amendment), (Corrigendum), (No. 1), (Part II)

    Keeps parenthetical subject matter such as "(KYC)" or
    "(Call, Notice and Term Money Markets)" so that genuinely different
    directives are NOT grouped together.
    """
    # ASCII-fold (drops Devanagari etc.)
    t: str = unicodedata.normalize("NFKD", title)
    t = t.encode("ascii", "ignore").decode("ascii").lower()

    # Circular/notification reference codes: RBI/2023-24/23, SEBI/HO/MRD/...
    t = re.sub(r"\b(?:rbi|sebi)[/\-][\w/\-\.]{3,}", "", t)

    # "dated DATE" / "as on DATE" / "updated DATE" qualifiers
    _date_qual = rf"\b(?:dated?|as\s+on|updated?|revised?|amended?)\s+"
    t = re.sub(_date_qual + rf"{_MONTHS}\s+\d{{1,2}},?\s+\d{{4}}", "", t, flags=re.I)
    t = re.sub(_date_qual + rf"{_MONTHS}\s+\d{{4}}", "", t, flags=re.I)
    t = re.sub(_date_qual + r"\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}", "", t, flags=re.I)

    # Standalone dates
    t = re.sub(r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b", "", t)
    t = re.sub(rf"\b{_MONTHS}\s+\d{{1,2}},?\s+\d{{4}}\b", "", t, flags=re.I)
    t = re.sub(rf"\b{_MONTHS}\s+\d{{4}}\b", "", t, flags=re.I)
    t = re.sub(r"\b20\d{2}\b", "", t)     # standalone years 2000-2099

    # Amendment-only parentheticals (NOT subject-matter parentheticals)
    t = re.sub(
        r"\(\s*(?:amendment|corrigendum|addendum|erratum|"
        r"no\.?\s*\d+|part\s*[ivx\d]+|vol\.?\s*\d+|"
        r"serial\s*\d+|s\.?\s*no\.?\s*\d+)\s*\)",
        " ", t, flags=re.I,
    )

    # Collapse punctuation & whitespace
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ── Group building ─────────────────────────────────────────────────────────────

def build_groups(inventory: list[dict]) -> tuple[dict, dict]:
    """
    Group documents by normalized title.

    Returns:
        groups      : {norm_title: [doc_id, ...]}
        doc_to_norm : {doc_id: norm_title}
    """
    groups: dict[str, list[str]] = defaultdict(list)
    doc_to_norm: dict[str, str] = {}

    for r in inventory:
        doc_id = r["doc_id"]
        title  = r.get("title", "") or ""
        norm   = normalize_title(title) if title else f"__no_title__{doc_id}"
        groups[norm].append(doc_id)
        doc_to_norm[doc_id] = norm

    return dict(groups), doc_to_norm


# ── Stratified group-level assignment ─────────────────────────────────────────

def _dominant_stratum(group_doc_ids: list[str], id_to_meta: dict) -> tuple[str, str]:
    from collections import Counter
    c: Counter = Counter()
    for did in group_doc_ids:
        meta = id_to_meta.get(did, {})
        c[(meta.get("regulator", "?").upper(),
           meta.get("doc_type", "?").lower())] += 1
    return c.most_common(1)[0][0]


def _assign_groups_stratified(
    groups: dict[str, list[str]],
    id_to_meta: dict,
    ratios: dict[str, float],
    seed: int,
) -> dict[str, str]:
    """
    Assign each *group* atomically to a split, stratified by dominant
    (regulator, doc_type).  Returns {doc_id: split}.
    """
    strata_groups: dict[tuple, list[str]] = defaultdict(list)
    for norm_title, doc_ids in groups.items():
        stratum = _dominant_stratum(doc_ids, id_to_meta)
        strata_groups[stratum].append(norm_title)

    rng = random.Random(seed)
    group_split: dict[str, str] = {}

    for stratum, norm_titles in sorted(strata_groups.items()):
        shuffled = list(norm_titles)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = math.ceil(n * ratios["train"])
        n_calib = math.ceil(n * ratios["calib"])
        splits: list[str] = (
            ["train"] * n_train +
            ["calib"] * n_calib +
            ["test"]  * max(0, n - n_train - n_calib)
        )[:n]
        for norm_t, spl in zip(shuffled, splits):
            group_split[norm_t] = spl

    doc_split: dict[str, str] = {}
    for norm_title, doc_ids in groups.items():
        spl = group_split.get(norm_title, "train")
        for did in doc_ids:
            doc_split[did] = spl

    return doc_split


# ── Existing assignment loader ────────────────────────────────────────────────

def _load_existing(path: Path) -> dict[str, str]:
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


# ── Main ──────────────────────────────────────────────────────────────────────

def assign_splits(
    inventory_path: Path,
    out_path: Path,
    ratios: dict[str, float],
    seed: int,
    allow_regroup: bool = False,
) -> dict[str, str]:
    """
    Build or update doc_assignment.csv.

    allow_regroup=True : one-time fresh group-aware assignment; logs diffs.
    allow_regroup=False: append-only; existing rows unchanged; new docs inherit
                         their group's split or get freshly assigned.
    """
    if not inventory_path.exists():
        print(f"ERROR: inventory not found: {inventory_path}", file=sys.stderr)
        sys.exit(1)

    inventory: list[dict] = []
    with open(inventory_path, newline="", encoding="utf-8") as f:
        inventory = list(csv.DictReader(f))
    logger.info("Loaded %d documents from inventory", len(inventory))

    id_to_meta = {r["doc_id"]: r for r in inventory}

    # Build groups from full (current) inventory
    groups, doc_to_norm = build_groups(inventory)
    multi_groups = {k: v for k, v in groups.items() if len(v) > 1}
    logger.info(
        "Groups: %d total, %d with >1 doc (%d docs in multi-doc groups)",
        len(groups), len(multi_groups), sum(len(v) for v in multi_groups.values()),
    )

    old_existing = _load_existing(out_path)

    if allow_regroup:
        # ── Fresh group-aware assignment ──────────────────────────────────────
        logger.info("--regroup: computing fresh group-aware assignment for all docs")
        final = _assign_groups_stratified(groups, id_to_meta, ratios, seed)

        # Log diffs vs previous assignment
        reassignments: list[dict] = []
        for did, new_spl in final.items():
            old_spl = old_existing.get(did, "")
            if old_spl and old_spl != new_spl:
                norm = doc_to_norm.get(did, "")
                reassignments.append({
                    "doc_id":     did,
                    "old_split":  old_spl,
                    "new_split":  new_spl,
                    "norm_title": norm,
                    "group_size": len(groups.get(norm, [])),
                })

        if reassignments:
            log_path = out_path.parent / "reassignment_log.csv"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["doc_id", "old_split", "new_split",
                                   "norm_title", "group_size"])
                writer.writeheader()
                writer.writerows(reassignments)
            logger.info("Wrote %d reassignment entries to %s", len(reassignments), log_path)

    else:
        # ── Append-only group-aware mode ──────────────────────────────────────
        existing = dict(old_existing)

        # Check for cross-split groups in existing assignment → abort if found
        group_splits: dict[str, set] = defaultdict(set)
        for did, spl in existing.items():
            norm = doc_to_norm.get(did)
            if norm:
                group_splits[norm].add(spl)
        cross = {norm: splts for norm, splts in group_splits.items() if len(splts) > 1}
        if cross:
            print(
                f"\nERROR: {len(cross)} groups span multiple splits in the existing "
                "assignment.\nRun with --regroup to fix (safe before Phase C).\n",
                file=sys.stderr,
            )
            for norm, splts in list(cross.items())[:5]:
                print(f"  {norm[:60]}  splits={splts}", file=sys.stderr)
            sys.exit(1)

        # Find docs not yet assigned
        already = set(existing.keys())
        new_docs = [r for r in inventory if r["doc_id"] not in already]
        logger.info("%d existing, %d new docs", len(already), len(new_docs))

        if new_docs:
            # For each new doc, assign via its group
            need_fresh: dict[str, list[str]] = {}   # norm → [new doc_ids needing assignment]
            for r in new_docs:
                norm = doc_to_norm[r["doc_id"]]
                # If any existing doc in this group is assigned, inherit
                group_existing = [existing.get(did) for did in groups[norm] if did in existing]
                if group_existing:
                    dominant = max(set(group_existing), key=group_existing.count)
                    existing[r["doc_id"]] = dominant
                else:
                    need_fresh.setdefault(norm, []).append(r["doc_id"])

            if need_fresh:
                # Fresh groups among new docs – assign at group level
                fresh_groups = {norm: groups[norm] for norm in need_fresh}
                new_assignments = _assign_groups_stratified(
                    fresh_groups, id_to_meta, ratios, seed)
                existing.update(new_assignments)

        final = existing
        reassignments = []

    # ── Write doc_assignment.csv ──────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "split"])
        writer.writeheader()
        for did in sorted(final.keys()):
            writer.writerow({"doc_id": did, "split": final[did]})
    logger.info("Wrote %d assignments to %s", len(final), out_path)

    # ── Verify: no group spans two splits ────────────────────────────────────
    check: dict[str, set] = defaultdict(set)
    for did, spl in final.items():
        norm = doc_to_norm.get(did)
        if norm:
            check[norm].add(spl)
    violated = {n: s for n, s in check.items() if len(s) > 1}
    if violated:
        logger.error("POST-CHECK FAILED: %d groups still span multiple splits!", len(violated))
    else:
        logger.info("Group coherence check passed: no group spans multiple splits. ✓")

    # ══════════════════════════════════════════════════════════════════════════
    #   Summary
    # ══════════════════════════════════════════════════════════════════════════
    split_counts: dict[str, int] = defaultdict(int)
    for spl in final.values():
        split_counts[spl] += 1

    print(f"\n{'='*62}")
    print("  Document Split Assignment Summary")
    print(f"{'='*62}")
    print(f"  Mode          : {'--regroup (fresh)' if allow_regroup else 'append-only'}")
    print(f"  Total assigned: {len(final)}")
    if allow_regroup and reassignments:
        print(f"  Reassigned    : {len(reassignments)} docs (see data/splits/reassignment_log.csv)")
    for spl in ("train", "calib", "test"):
        cnt = split_counts.get(spl, 0)
        pct = 100.0 * cnt / max(len(final), 1)
        print(f"    {spl:6s}: {cnt:5d}  ({pct:.1f}%)")

    # Per-stratum breakdown
    strat_split: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for did, spl in final.items():
        meta = id_to_meta.get(did, {})
        key  = (meta.get("regulator", "?"), meta.get("doc_type", "?"))
        strat_split[key][spl] += 1

    print("\n  Per (regulator|doc_type):")
    print(f"  {'Stratum':<42} {'train':>6} {'calib':>6} {'test':>6} {'total':>6}")
    print("  " + "-" * 64)
    for (reg, dtype), counts in sorted(strat_split.items()):
        tot  = sum(counts.values())
        tr   = counts.get("train", 0)
        ca   = counts.get("calib", 0)
        te   = counts.get("test",  0)
        warn = " ⚠" if tot < 3 else ""
        print(f"  {reg}|{dtype:<40} {tr:>6} {ca:>6} {te:>6} {tot:>6}{warn}")

    # Group statistics
    print(f"\n  Group statistics (normalized-title grouping):")
    print(f"    Unique groups          : {len(groups)}")
    print(f"    Multi-doc groups       : {len(multi_groups)}")
    print(f"    Docs in multi-doc grps : {sum(len(v) for v in multi_groups.values())}")

    # Top merged groups
    top_groups = sorted(multi_groups.items(), key=lambda x: -len(x[1]))[:15]
    if top_groups:
        print(f"\n  Top merged groups (≥2 docs, same normalized title):")
        print(f"  {'size':>5}  norm_title[:65]")
        print("  " + "-" * 72)
        for norm_t, ids in top_groups:
            spl = final.get(ids[0], "?")
            print(f"  {len(ids):>5}  [{spl}]  {norm_t[:65]}")

    print(f"\n  Ratios: train={ratios['train']:.2f} calib={ratios['calib']:.2f} "
          f"test={ratios['test']:.2f}  seed={seed}")
    print(f"  Output: {out_path}")
    print(f"{'='*62}\n")

    return final


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg     = _load_config()
    paths   = cfg.get("paths",  {})
    sp_cfg  = cfg.get("splits", {})

    default_inv  = str(ROOT / paths.get("doc_inventory", "data/doc_inventory.csv"))
    default_out  = str(ROOT / paths.get("doc_splits",    "data/splits/doc_assignment.csv"))
    default_seed = int(sp_cfg.get("seed", 42))
    default_ratios = {
        "train": float(sp_cfg.get("train", 0.40)),
        "calib": float(sp_cfg.get("calib", 0.35)),
        "test":  float(sp_cfg.get("test",  0.25)),
    }

    total = sum(default_ratios.values())
    if not (0.99 < total < 1.01):
        print(f"ERROR: split ratios sum to {total:.3f}, must be ~1.0", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Assign document-level train/calib/test splits (group-aware).")
    parser.add_argument("--inventory", default=default_inv)
    parser.add_argument("--out",       default=default_out)
    parser.add_argument("--seed",      type=int, default=default_seed)
    parser.add_argument(
        "--regroup", action="store_true",
        help=(
            "One-time fresh group-aware assignment. Safe only before Phase C "
            "(no questions or labels exist yet). Diffs logged to reassignment_log.csv."
        ),
    )
    args = parser.parse_args()

    assign_splits(
        inventory_path=Path(args.inventory),
        out_path=Path(args.out),
        ratios=default_ratios,
        seed=args.seed,
        allow_regroup=args.regroup,
    )


if __name__ == "__main__":
    main()
