#!/usr/bin/env python3
"""Tests for the optional MLX LoRA adapter preparation helper."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).with_name("train_mlx_lora.py")
SPEC = importlib.util.spec_from_file_location("train_mlx_lora", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
train_mlx_lora = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train_mlx_lora)


class TrainMlxLoraTests(unittest.TestCase):
    def write_minimal_dataset(self, dataset: Path) -> None:
        dataset.mkdir()
        (dataset / "route_region.train.jsonl").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "task": "route_region",
                    "split": "train",
                    "id": "route:train:1",
                    "input": "Region card for project notes",
                    "target": "Route to project notes",
                    "artifact_ids": ["brain-region:project"],
                    "source_refs": ["imprint://chunk/project:0"],
                }
            )
            + "\n"
        )
        (dataset / "route_region.eval.jsonl").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "task": "route_region",
                    "split": "eval",
                    "id": "route:eval:1",
                    "input": "Held-out region card for project notes",
                    "target": "Route to project notes",
                    "artifact_ids": ["brain-region:project"],
                    "source_refs": ["imprint://chunk/project:0"],
                }
            )
            + "\n"
        )

    def test_adapter_manifest_records_dataset_hash_and_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "training"
            output = root / "adapter"
            self.write_minimal_dataset(dataset)

            mlx_data = train_mlx_lora.prepare_mlx_dataset(dataset, output)
            (output / "adapters.safetensors").write_bytes(b"fake adapter weights")
            manifest = train_mlx_lora.write_adapter_manifest(
                model="tiny-memory-model",
                dataset=mlx_data,
                source_dataset=dataset,
                output=output,
                iters=25,
                status="prepared",
                command=["python3", "-m", "mlx_lm", "lora"],
                created_at=100,
                finished_at=200,
            )

            self.assertEqual(manifest["status"], "prepared")
            self.assertEqual(manifest["base_model"], "tiny-memory-model")
            self.assertEqual(manifest["command"], ["python3", "-m", "mlx_lm", "lora"])
            self.assertEqual(manifest["created_at"], 100)
            self.assertEqual(manifest["finished_at"], 200)
            self.assertEqual(manifest["train_records"], 1)
            self.assertEqual(manifest["valid_records"], 1)
            self.assertEqual(manifest["test_records"], 1)
            self.assertRegex(manifest["dataset_hash"], r"^[0-9a-f]{64}$")
            self.assertEqual(manifest["prepared_dataset_hash"], manifest["dataset_hash"])
            self.assertRegex(manifest["source_dataset_hash"], r"^[0-9a-f]{64}$")
            self.assertRegex(manifest["adapter_file_hash"], r"^[0-9a-f]{64}$")
            self.assertIn("python", manifest["versions"])
            self.assertIn("mlx_lm", manifest["versions"])
            self.assertNotEqual(manifest["dataset_hash"], manifest["source_dataset_hash"])
            written = json.loads((output / "adapter_manifest.json").read_text())
            self.assertEqual(written, manifest)

    def test_dry_run_prepares_manifest_without_mlx_lm_installed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "training"
            output = root / "adapter"
            self.write_minimal_dataset(dataset)

            argv = [
                "train_mlx_lora.py",
                "--model",
                "tiny-memory-model",
                "--dataset",
                str(dataset),
                "--output",
                str(output),
                "--dry-run",
            ]
            stdout = io.StringIO()
            with mock.patch.object(sys, "argv", argv), mock.patch(
                "importlib.util.find_spec", return_value=None
            ), redirect_stdout(stdout):
                train_mlx_lora.main()

            result = json.loads(stdout.getvalue())
            manifest = json.loads((output / "adapter_manifest.json").read_text())
            self.assertEqual(result["status"], "ready")
            self.assertEqual(manifest["status"], "prepared")
            self.assertEqual(manifest["base_model"], "tiny-memory-model")
            self.assertIn("mlx_lm", manifest["command"])
            self.assertIn("--adapter-path", manifest["command"])
            self.assertIsNone(manifest["adapter_file_hash"])
            self.assertIsInstance(manifest["created_at"], int)
            self.assertIsInstance(manifest["finished_at"], int)
            self.assertEqual(manifest["train_records"], 1)


if __name__ == "__main__":
    unittest.main()
