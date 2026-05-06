#!/usr/bin/env python3
"""Validate and summarize imprint cortex JSONL training exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED = {"schema_version", "task", "split", "id", "input", "target", "artifact_ids", "source_refs"}


def load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    if not path.exists():
        return records
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        missing = REQUIRED - record.keys()
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
    for path in files:
        records = load_records(path)
        summary[path.name] = len(records)
    print(json.dumps({"training_dir": str(training_dir), "files": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
