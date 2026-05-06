#!/usr/bin/env python3
"""Optional MLX-LM LoRA training entrypoint for imprint cortex adapters."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(record, sort_keys=True) for record in records))


def to_mlx_record(record: dict) -> dict:
    return {
        "prompt": (
            "You are imprint's tiny-model memory cortex. "
            f"Task: {record.get('task', 'memory')}\n\n"
            f"{record.get('input', '')}"
        ),
        "completion": record.get("target", ""),
    }


def prepare_mlx_dataset(dataset: Path, output: Path) -> Path:
    data_dir = output / "mlx-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_records: list[dict] = []
    valid_records: list[dict] = []
    for path in sorted(dataset.glob("*.train.jsonl")):
        train_records.extend(to_mlx_record(record) for record in load_jsonl(path))
    for path in sorted(dataset.glob("*.eval.jsonl")):
        valid_records.extend(to_mlx_record(record) for record in load_jsonl(path))
    if not train_records:
        raise SystemExit(f"no training records found in {dataset}")
    if not valid_records:
        valid_records = train_records[:]
    write_jsonl(data_dir / "train.jsonl", train_records)
    write_jsonl(data_dir / "valid.jsonl", valid_records)
    write_jsonl(data_dir / "test.jsonl", valid_records)
    return data_dir


def load_prepared_counts(dataset: Path) -> dict[str, int]:
    return {
        "train_records": len(load_jsonl(dataset / "train.jsonl")),
        "valid_records": len(load_jsonl(dataset / "valid.jsonl")),
        "test_records": len(load_jsonl(dataset / "test.jsonl")),
    }


def dataset_hash(dataset: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(dataset.glob("*.jsonl")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        for record in load_jsonl(path):
            record.pop("created_at", None)
            digest.update(json.dumps(record, separators=(",", ":"), sort_keys=True).encode("utf-8"))
            digest.update(b"\n")
        digest.update(b"\0")
    return digest.hexdigest()


def write_adapter_manifest(
    *,
    model: str,
    dataset: Path,
    source_dataset: Path | None = None,
    output: Path,
    iters: int,
    status: str,
) -> dict:
    manifest = {
        "status": status,
        "base_model": model,
        "dataset": str(dataset),
        "dataset_hash": dataset_hash(dataset),
        "adapter_path": str(output),
        "iters": iters,
        **load_prepared_counts(dataset),
    }
    if source_dataset is not None:
        manifest["source_dataset"] = str(source_dataset)
        manifest["source_dataset_hash"] = dataset_hash(source_dataset)
    (output / "adapter_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Base MLX model name or path")
    parser.add_argument("--dataset", required=True, help="Path to store/training directory")
    parser.add_argument("--output", required=True, help="Adapter output directory")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true", help="Prepare MLX data and print the training command without running it")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    mlx_data = prepare_mlx_dataset(dataset, output)
    command = [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        args.model,
        "--train",
        "--data",
        str(mlx_data),
        "--adapter-path",
        str(output),
        "--iters",
        str(args.iters),
    ]
    if args.dry_run:
        manifest = write_adapter_manifest(
            model=args.model,
            dataset=mlx_data,
            source_dataset=dataset,
            output=output,
            iters=args.iters,
            status="prepared",
        )
        print(json.dumps({
            "status": "ready",
            "dataset": str(mlx_data),
            "manifest": str(output / "adapter_manifest.json"),
            "dataset_hash": manifest["dataset_hash"],
            "command": command,
        }, indent=2))
        return
    if importlib.util.find_spec("mlx_lm") is None:
        raise SystemExit("mlx_lm is not installed. Install optional dependency mlx-lm to train adapters.")
    subprocess.run(command, check=True)
    manifest = write_adapter_manifest(
        model=args.model,
        dataset=mlx_data,
        source_dataset=dataset,
        output=output,
        iters=args.iters,
        status="trained",
    )
    print(json.dumps({"status": "trained", **manifest}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
