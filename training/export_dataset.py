#!/usr/bin/env python3
"""Validate and summarize imprint cortex JSONL training exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED = {"schema_version", "task", "split", "id", "input", "target", "artifact_ids", "source_refs"}
V2_REQUIRED = REQUIRED | {
    "source_id",
    "anchor_ids",
    "source_type",
    "visibility",
    "redacted",
    "excluded_or_stale",
}


def load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    if not path.exists():
        return records
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        missing = REQUIRED - record.keys()
        if record.get("schema_version") == 2:
            missing |= V2_REQUIRED - record.keys()
        if missing:
            raise SystemExit(f"{path}:{line_number} missing fields: {sorted(missing)}")
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", required=True, help="Path to imprint memory store")
    args = parser.parse_args()

    training_dir = Path(args.store) / "training"
    files = sorted(training_dir.glob("*.jsonl"))
    summary = {}
    task_counts = {}
    split_counts = {}
    source_types = {}
    for path in files:
        records = load_records(path)
        summary[path.name] = len(records)
        for record in records:
            task_counts[record.get("task", "unknown")] = task_counts.get(record.get("task", "unknown"), 0) + 1
            split_counts[record.get("split", "unknown")] = split_counts.get(record.get("split", "unknown"), 0) + 1
            source_types[record.get("source_type", "unknown")] = source_types.get(record.get("source_type", "unknown"), 0) + 1
    print(json.dumps({
        "training_dir": str(training_dir),
        "files": summary,
        "task_counts": task_counts,
        "split_counts": split_counts,
        "source_types": source_types,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
