"""Import annotator labels from a CSV and validate them.

Usage:
    python scripts/import_labels.py --labels <labeled.csv> --out <output.jsonl>
"""
import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

VALID_LABELS = {"supported", "unsupported", "unclear"}
RESERVED_ANNOTATORS = {"judge", "pipeline", "mock"}

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_config() -> dict:
    cfg_path = Path("configs/experiment.yaml")
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Import annotator labels")
    parser.add_argument("--labels", required=True, help="Labeled CSV file")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    args = parser.parse_args()

    if not Path(args.labels).exists():
        raise FileNotFoundError(f"Labels file not found: {args.labels}")

    cfg = load_config()
    allowed_annotators = cfg.get("allowed_annotators", [])

    records = []
    skipped = 0

    with open(args.labels, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            atom_id = row.get("atom_id", "").strip()
            label = row.get("label", "").strip()
            reason = row.get("reason", "").strip()
            annotator_id = row.get("annotator_id", "").strip()

            # Validate annotator
            if annotator_id.lower() in RESERVED_ANNOTATORS:
                raise ValueError(
                    f"annotator_id '{annotator_id}' is reserved and not allowed. "
                    f"Valid annotators: {allowed_annotators}"
                )
            if allowed_annotators and annotator_id not in allowed_annotators:
                raise ValueError(
                    f"annotator_id '{annotator_id}' is not in allowed list. "
                    f"Valid annotators: {allowed_annotators}"
                )

            # Validate label
            if not label:
                logger.warning("Skipping atom_id='%s': empty label", atom_id)
                skipped += 1
                continue

            label_norm = label.lower()
            if label_norm not in VALID_LABELS:
                logger.warning(
                    "Skipping atom_id='%s': unknown label '%s'", atom_id, label
                )
                skipped += 1
                continue

            records.append({
                "annotator_id": annotator_id,
                "atom_id": atom_id,
                "label": label_norm,
                "reason": reason,
                "ts": datetime.now(timezone.utc).isoformat(),
            })

    with open(args.out, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Imported {len(records)} records, skipped {skipped}. Written to {args.out}")


if __name__ == "__main__":
    main()
