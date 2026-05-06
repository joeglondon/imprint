#!/usr/bin/env python3
"""Tests for the optional MLX LoRA adapter preparation helper."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("train_mlx_lora.py")
SPEC = importlib.util.spec_from_file_location("train_mlx_lora", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
train_mlx_lora = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train_mlx_lora)


class TrainMlxLoraTests(unittest.TestCase):
    def test_adapter_manifest_records_dataset_hash_and_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "training"
            output = root / "adapter"
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

            mlx_data = train_mlx_lora.prepare_mlx_dataset(dataset, output)
            manifest = train_mlx_lora.write_adapter_manifest(
                model="tiny-memory-model",
                dataset=mlx_data,
                output=output,
                iters=25,
                status="prepared",
            )

            self.assertEqual(manifest["status"], "prepared")
            self.assertEqual(manifest["base_model"], "tiny-memory-model")
            self.assertEqual(manifest["train_records"], 1)
            self.assertEqual(manifest["valid_records"], 1)
            self.assertEqual(manifest["test_records"], 1)
            self.assertRegex(manifest["dataset_hash"], r"^[0-9a-f]{64}$")
            written = json.loads((output / "adapter_manifest.json").read_text())
            self.assertEqual(written, manifest)


if __name__ == "__main__":
    unittest.main()
