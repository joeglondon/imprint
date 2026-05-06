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
        if record.get("task") in {"route_region", "query_to_region"} and "region" in target:
            correct += 1
        elif record.get("task") == "query_to_source_family" and target.startswith("source_family:"):
            correct += 1
        elif record.get("task") in {"choose_tool", "query_to_tool_plan"} and "memory_search" in target:
            correct += 1
        elif record.get("task") in {"critique_evidence", "weak_evidence_to_next_action"} and (
            "anchor" in target or "weak_evidence" in target
        ):
            correct += 1
        elif record.get("task") == "collaboration_pattern" and "planner" in target:
            correct += 1
        elif record.get("task") == "chunk_to_semantic_address" and "anchor:" in target:
            correct += 1
        elif record.get("task") == "snippet_set_to_citation_boundary" and "cite_anchor_ids" in target:
            correct += 1
        elif record.get("task") == "deleted_or_stale_memory_to_caution" and "caution:" in target:
            correct += 1
        elif record.get("task") == "web_needed_or_not" and "web_search_" in target:
            correct += 1
    print(json.dumps({
        "records": len(eval_records),
        "baseline_accuracy": round(correct / len(eval_records), 4),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
