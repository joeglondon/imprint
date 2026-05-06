#!/usr/bin/env python3
"""Score simple baseline routing/tool/critique behavior from imprint exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def records(dataset: Path) -> list[dict]:
    values: list[dict] = []
    for path in sorted(dataset.glob("*.eval.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                values.append(json.loads(line))
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to store/training directory")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    eval_records = records(dataset)
    if not eval_records:
        print(json.dumps({"records": 0, "baseline_accuracy": 0.0}, indent=2))
        return

    correct = 0
    for record in eval_records:
        target = record.get("target", "").lower()
        text = record.get("input", "").lower()
        if "route" in record.get("task", "") and "region" in target and "region" in text:
            correct += 1
        elif "tool" in record.get("task", "") and "memory_search" in target:
            correct += 1
        elif "critique" in record.get("task", "") and "anchor" in target:
            correct += 1
        elif "collaboration" in record.get("task", "") and "planner" in target:
            correct += 1
    print(json.dumps({
        "records": len(eval_records),
        "baseline_accuracy": round(correct / len(eval_records), 4),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
