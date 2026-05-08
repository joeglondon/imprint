#!/usr/bin/env python3
"""Optional MLX-LM LoRA training entrypoint for imprint cortex adapters."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import subprocess
import sys
import time
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


def prepare_mlx_dataset(dataset: Path, output: Path) -> tuple[Path, list[str]]:
    data_dir = output / "mlx-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_records: list[dict] = []
    eval_records: list[dict] = []
    test_records: list[dict] = []
    warnings: list[str] = []
    for path in sorted(dataset.glob("*.train.jsonl")):
        train_records.extend(to_mlx_record(record) for record in load_jsonl(path))
    for path in sorted(dataset.glob("*.eval.jsonl")):
        eval_records.extend(to_mlx_record(record) for record in load_jsonl(path))
    for path in sorted(dataset.glob("*.test.jsonl")):
        test_records.extend(to_mlx_record(record) for record in load_jsonl(path))
    if not train_records:
        raise SystemExit(f"no training records found in {dataset}")
    valid_records = eval_records
    if not valid_records:
        if test_records:
            valid_records = test_records[:]
            warnings.append("No eval records found; using test records for validation.")
        else:
            valid_records = train_records[:]
            warnings.append("No eval or test records found; using training records for validation.")
    if not test_records:
        if eval_records:
            test_records = eval_records[:]
            warnings.append("No test records found; using eval records for test.")
        else:
            test_records = train_records[:]
            warnings.append("No test records found; using training records for test.")
    write_jsonl(data_dir / "train.jsonl", train_records)
    write_jsonl(data_dir / "valid.jsonl", valid_records)
    write_jsonl(data_dir / "test.jsonl", test_records)
    return data_dir, warnings


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


def now_millis() -> int:
    return int(time.time() * 1000)


def adapter_file_hash(output: Path) -> str | None:
    digest = hashlib.sha256()
    files: list[Path] = []
    for path in output.rglob("*"):
        if not path.is_file():
            continue
        if path.name == "adapter_manifest.json":
            continue
        if "mlx-data" in path.relative_to(output).parts:
            continue
        files.append(path)
    if not files:
        return None
    for path in sorted(files):
        relative = path.relative_to(output).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_adapter_manifest(
    *,
    model: str,
    dataset: Path,
    source_dataset: Path | None = None,
    source_dataset_hash: str | None = None,
    output: Path,
    iters: int,
    status: str,
    command: list[str] | None = None,
    created_at: int | None = None,
    finished_at: int | None = None,
    warnings: list[str] | None = None,
) -> dict:
    prepared_dataset_hash = dataset_hash(dataset)
    manifest = {
        "status": status,
        "base_model": model,
        "dataset": str(dataset),
        "dataset_hash": prepared_dataset_hash,
        "prepared_dataset_hash": prepared_dataset_hash,
        "adapter_path": str(output),
        "adapter_file_hash": adapter_file_hash(output),
        "iters": iters,
        "command": command or [],
        "versions": {
            "python": sys.version.split()[0],
            "mlx_lm": package_version("mlx-lm") or package_version("mlx_lm"),
        },
        "created_at": created_at if created_at is not None else now_millis(),
        "finished_at": finished_at if finished_at is not None else now_millis(),
        "warnings": warnings or [],
        **load_prepared_counts(dataset),
    }
    if source_dataset is not None:
        manifest["source_dataset"] = str(source_dataset)
        manifest["source_dataset_hash"] = source_dataset_hash or dataset_hash(source_dataset)
    (output / "adapter_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Base MLX model name or path")
    parser.add_argument("--dataset", required=True, help="Path to store/training directory")
    parser.add_argument("--output", required=True, help="Adapter output directory")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--source-dataset-hash", help="Canonical Rust training export hash to record in the manifest")
    parser.add_argument("--dry-run", action="store_true", help="Prepare MLX data and print the training command without running it")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started_at = now_millis()
    mlx_data, warnings = prepare_mlx_dataset(dataset, output)
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
            source_dataset_hash=args.source_dataset_hash,
            output=output,
            iters=args.iters,
            status="prepared",
            command=command,
            created_at=started_at,
            finished_at=now_millis(),
            warnings=warnings,
        )
        print(json.dumps({
            "status": "ready",
            "dataset": str(mlx_data),
            "manifest": str(output / "adapter_manifest.json"),
            "dataset_hash": manifest["dataset_hash"],
            "command": command,
            "warnings": warnings,
        }, indent=2))
        return
    if importlib.util.find_spec("mlx_lm") is None:
        raise SystemExit("mlx_lm is not installed. Install optional dependency mlx-lm to train adapters.")
    subprocess.run(command, check=True)
    finished_at = now_millis()
    manifest = write_adapter_manifest(
        model=args.model,
        dataset=mlx_data,
        source_dataset=dataset,
        source_dataset_hash=args.source_dataset_hash,
        output=output,
        iters=args.iters,
        status="trained",
        command=command,
        created_at=started_at,
        finished_at=finished_at,
        warnings=warnings,
    )
    print(json.dumps({"status": "trained", **manifest}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
