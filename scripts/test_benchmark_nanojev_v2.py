#!/usr/bin/env python3
import json
from pathlib import Path
import tempfile
import unittest

from benchmark_nanojev_v2 import (benchmark_manifest, cluster_bootstrap_ci,
                                  percentile, probability_metrics,
                                  resolve_input_files, resolve_seed_label)


class BenchmarkHelpersTest(unittest.TestCase):
    def test_percentile_interpolates(self):
        self.assertEqual(percentile([1, 3, 5, 7], 0.5), 4.0)

    def test_probability_metrics_reports_accuracy_and_calibration(self):
        result = probability_metrics([
            {"probabilities": {"false": 0.1, "true": 0.9}, "gold_index": 1},
            {"probabilities": {"false": 0.8, "true": 0.2}, "gold_index": 0},
        ])
        self.assertEqual(result["questions"], 2)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["invalid_outputs"], 0)

    def test_probability_metrics_counts_invalid_outputs(self):
        result = probability_metrics([
            {"probabilities": {"false": 0.5, "true": 0.6}, "gold_index": 1},
        ])
        self.assertEqual(result["invalid_outputs"], 1)
        self.assertIsNone(result["accuracy"])

    def test_cluster_bootstrap_is_deterministic(self):
        rows = [
            {"probabilities": {"false": 0.1, "true": 0.9}, "gold_index": 1,
             "source_group_id": "a"},
            {"probabilities": {"false": 0.8, "true": 0.2}, "gold_index": 1,
             "source_group_id": "b"},
        ]
        first = cluster_bootstrap_ci(rows, 50, 17)
        second = cluster_bootstrap_ci(rows, 50, 17)
        self.assertEqual(first, second)
        self.assertEqual(first["clusters"], 2)

    def test_manifest_hashes_input_checkpoint_and_itself(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            checkpoint = root / "checkpoint"
            (checkpoint / "backbone_config").mkdir(parents=True)
            (checkpoint / "tokenizer").mkdir()
            dataset.mkdir()
            input_path = dataset / "test.jsonl"
            input_path.write_text("{}\n", encoding="utf-8")
            (dataset / "manifest.json").write_text('{"version":1}\n', encoding="utf-8")
            for path, content in (
                (checkpoint / "config.json", "{}\n"),
                (checkpoint / "best.safetensors", "weights"),
                (checkpoint / "backbone_config/config.json", "{}\n"),
                (checkpoint / "tokenizer/tokenizer.json", "{}\n"),
                (checkpoint / "tokenizer/tokenizer_config.json", "{}\n"),
            ):
                path.write_text(content, encoding="utf-8")
            script = root / "benchmark.py"
            script.write_text("pass\n", encoding="utf-8")
            manifest = benchmark_manifest(input_path, checkpoint, "cpu", script)
        self.assertEqual(len(manifest["manifest_sha256"]), 64)
        self.assertEqual(manifest["input"]["files"][0]["bytes"], 3)
        self.assertEqual(len(manifest["checkpoint"]["files"]), 5)
        self.assertEqual(len(manifest["dependencies"]["sha256"]), 64)
        without_hash = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        from benchmark_nanojev_v2 import canonical_json, sha256_bytes
        self.assertEqual(manifest["manifest_sha256"],
                         sha256_bytes(canonical_json(without_hash).encode("utf-8")))

    def test_directory_input_and_checkpoint_seed_are_resolved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("test", "ood"):
                (root / f"{split}.jsonl").write_text("{}\n", encoding="utf-8")
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text('{"seed":29}\n', encoding="utf-8")
            files = resolve_input_files(root, {"test", "ood"})
            seed = resolve_seed_label(checkpoint, None)
        self.assertEqual([path.name for path in files], ["ood.jsonl", "test.jsonl"])
        self.assertEqual(seed, "seed-29")


if __name__ == "__main__":
    unittest.main()
