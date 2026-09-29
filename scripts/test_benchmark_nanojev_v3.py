#!/usr/bin/env python3
"""Unit tests for the N1 benchmark contract harness.

These tests use temporary text fixtures only. They never import torch, never load
weights, and never touch the network.
"""

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import benchmark_nanojev_v3 as harness


def make_checkpoint(root):
    checkpoint = Path(root) / "checkpoint"
    (checkpoint / "backbone_config").mkdir(parents=True, exist_ok=True)
    (checkpoint / "tokenizer").mkdir(exist_ok=True)
    (checkpoint / "config.json").write_text('{"seed": 17}\n', encoding="utf-8")
    (checkpoint / "best.safetensors").write_text("weights\n", encoding="utf-8")
    (checkpoint / "backbone_config/config.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "tokenizer/tokenizer.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "tokenizer/tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"}) + "\n", encoding="utf-8")
    return checkpoint


class CanonicalizationTest(unittest.TestCase):
    def test_canonical_json_is_sorted_and_compact(self):
        self.assertEqual(harness.canonical_json({"b": 1, "a": 2}), '{"a":2,"b":1}')

    def test_strip_time_varying_drops_volatile_fields_recursively(self):
        report = {
            "keep": 1,
            "latency": {"cold": {"p50_ms": 3}},
            "resources": {"peak_rss_bytes": 99},
            "nested": {"generated_at": 1, "keep": "x"},
        }
        self.assertEqual(harness.strip_time_varying(report),
                         {"keep": 1, "nested": {"keep": "x"}})

    def test_canonical_hash_ignores_time_varying_noise(self):
        base = {"keep": 1, "latency": {"warm": {"p50_ms": 5}}, "generated_at": 10,
                "resources": {"peak_rss_bytes": 1}}
        noisy = {"keep": 1, "latency": {"warm": {"p50_ms": 999}}, "generated_at": 20,
                 "resources": {"peak_rss_bytes": 2}}
        self.assertEqual(harness.canonical_report_sha256(base),
                         harness.canonical_report_sha256(noisy))

    def test_canonical_hash_changes_with_material_change(self):
        self.assertNotEqual(harness.canonical_report_sha256({"keep": 1}),
                            harness.canonical_report_sha256({"keep": 2}))


class StatisticsTest(unittest.TestCase):
    def test_percentile_interpolates(self):
        self.assertEqual(harness.percentile([1, 3, 5, 7], 0.5), 4.0)

    def test_summarize_latencies_reports_required_percentiles(self):
        summary = harness.summarize_latencies([1, 2, 3, 4, 5])
        self.assertEqual(summary["samples"], 5)
        self.assertEqual(summary["p50_ms"], 3.0)
        self.assertIsNotNone(summary["p95_ms"])
        self.assertIsNotNone(summary["p99_ms"])

    def test_summarize_empty_latencies_is_none(self):
        summary = harness.summarize_latencies([])
        self.assertEqual(summary["samples"], 0)
        self.assertIsNone(summary["p50_ms"])
        self.assertIsNone(summary["p95_ms"])


class IdentityTest(unittest.TestCase):
    def test_checkpoint_identity_records_files_and_chat_template(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = make_checkpoint(directory)
            identity = harness.checkpoint_identity(checkpoint)
        self.assertEqual(set(identity["required_files"]),
                         set(harness.REQUIRED_CHECKPOINT_FILES))
        template = identity["tokenizer_chat_template"]
        self.assertIsNotNone(template)
        self.assertEqual(len(template["sha256"]), 64)
        self.assertIn("chat_template", template["source"])

    def test_checkpoint_identity_fails_closed_on_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = make_checkpoint(directory)
            (checkpoint / "best.safetensors").unlink()
            with self.assertRaises(harness.ContractError):
                harness.checkpoint_identity(checkpoint)

    def test_chat_template_from_jinja_file(self):
        with tempfile.TemporaryDirectory() as directory:
            tokenizer = Path(directory)
            (tokenizer / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
            (tokenizer / "chat_template.jinja").write_text("{{ x }}", encoding="utf-8")
            template = harness.chat_template_identity(tokenizer)
        self.assertEqual(template["source"], "chat_template.jinja")

    def test_chat_template_absent_returns_none(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
            self.assertIsNone(harness.chat_template_identity(directory))

    def test_input_identity_directory_requires_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(harness.ContractError):
                harness.input_identity(directory)

    def test_input_identity_and_row_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            row = {"id": "s1", "state": "go", "questions": {"q": {"type": "boolean"}}}
            (Path(directory) / "test.jsonl").write_text(json.dumps(row) + "\n",
                                                        encoding="utf-8")
            identity = harness.input_identity(directory)
            rows, loaded = harness.load_input_rows(directory)
        self.assertEqual(len(identity["files"]), 1)
        self.assertEqual(rows, [row])
        self.assertEqual(loaded["sha256"], identity["sha256"])

    def test_load_input_rows_fails_closed_on_malformed_row(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "test.jsonl").write_text('{"id": "s1"}\n', encoding="utf-8")
            with self.assertRaises(harness.ContractError):
                harness.load_input_rows(directory)

    def test_build_payload_wraps_states(self):
        rows = [{"id": "s1", "state": "go", "questions": {"q": {}}}]
        self.assertEqual(harness.build_payload(rows),
                         {"states": [{"id": "s1", "state": "go", "questions": {"q": {}}}]})


class FailClosedTest(unittest.TestCase):
    def test_network_calls_must_be_zero(self):
        self.assertEqual(harness.validate_network_model_calls({"network_model_calls": 0}), 0)
        with self.assertRaises(harness.ContractError):
            harness.validate_network_model_calls({"network_model_calls": 1})
        with self.assertRaises(harness.ContractError):
            harness.validate_network_model_calls({})
        with self.assertRaises(harness.ContractError):
            harness.validate_network_model_calls(None)

    def test_threads_validation(self):
        self.assertEqual(harness.validate_threads(4), 4)
        self.assertIsNone(harness.validate_threads(None))
        for bad in (0, -1, "4"):
            with self.assertRaises(harness.ContractError):
                harness.validate_threads(bad)

    def test_apply_thread_budget_sets_env(self):
        with mock.patch.dict(harness.os.environ, {}, clear=False):
            self.assertEqual(harness.apply_thread_budget(3), 3)
            self.assertEqual(harness.os.environ["OMP_NUM_THREADS"], "3")
            self.assertEqual(harness.os.environ["MKL_NUM_THREADS"], "3")


class LazyImportTest(unittest.TestCase):
    def test_predictor_is_resolved_from_sys_modules_at_runtime(self):
        fake = types.ModuleType("predict_toy_decisions")
        fake.DecisionPredictor = object
        with mock.patch.dict(sys.modules, {"predict_toy_decisions": fake}):
            resolved = harness.load_predictor_class()
        self.assertIs(resolved, object)

    def test_fresh_module_import_does_not_pull_predictor_or_torch(self):
        saved = {name: sys.modules.pop(name)
                 for name in ("predict_toy_decisions", "torch") if name in sys.modules}
        try:
            spec = importlib.util.spec_from_file_location(
                "benchmark_nanojev_v3_fresh", harness.__file__)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertNotIn("predict_toy_decisions", sys.modules)
            self.assertNotIn("torch", sys.modules)
        finally:
            sys.modules.update(saved)


class ReceiptTest(unittest.TestCase):
    def _receipt(self, checkpoint_sha, input_sha, digest):
        return harness.build_receipt(
            checkpoint_sha256=checkpoint_sha,
            input_sha256=input_sha,
            chat_template_sha256="c" * 64,
            runtime={"device": "cpu", "precision": "fp32", "threads_effective": 4,
                     "temperature": 1.0, "batch_states": 8, "batch_questions": 0},
            network_model_calls=0,
            output_digest=digest,
            cold_samples=1,
            warm_samples=2)

    def test_receipt_is_deterministic_and_content_free(self):
        sentinel = "SECRET-STATE-TEXT"
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "test.jsonl").write_text(
                json.dumps({"id": "s1", "state": sentinel, "questions": {}}) + "\n",
                encoding="utf-8")
            identity = harness.input_identity(directory)
        receipt = self._receipt("a" * 64, identity["sha256"], "d" * 64)
        again = self._receipt("a" * 64, identity["sha256"], "d" * 64)
        self.assertEqual(receipt["receipt_sha256"], again["receipt_sha256"])
        serialized = harness.canonical_json(receipt)
        self.assertNotIn(sentinel, serialized)
        self.assertNotIn(identity["path"], serialized)
        self.assertEqual(receipt["network_model_calls"], 0)
        self.assertFalse(receipt["enables_production_pruning"])

    def test_receipt_hash_changes_when_input_changes(self):
        first = self._receipt("a" * 64, "b" * 64, "d" * 64)
        second = self._receipt("a" * 64, "c" * 64, "d" * 64)
        self.assertNotEqual(first["receipt_sha256"], second["receipt_sha256"])


class ArgumentTest(unittest.TestCase):
    def test_parser_defaults(self):
        args = harness.build_parser().parse_args(
            ["--checkpoint", "/c", "--input", "/i"])
        self.assertEqual(args.device, "auto")
        self.assertEqual(args.precision, "auto")
        self.assertEqual(args.batch_states, 8)
        self.assertEqual(args.temperature, 1.0)
        self.assertEqual(args.repeats, 2)


class _FakeEngine:
    """In-memory stand-in for DecisionPredictor; never loads weights or network."""

    def __init__(self, checkpoint, max_length=None, device_name="auto", precision="auto"):
        self.device = "cpu"
        self.precision = "fp32"
        self.network_model_calls = 0

    def predict(self, payload, batch_questions=0, temperature=1.0):
        states = []
        for state in payload["states"]:
            answers = {qid: {"probabilities": {"false": 0.5, "true": 0.5}}
                       for qid in state["questions"]}
            states.append({"id": state["id"], "answers": answers})
        return {"execution": {"network_model_calls": self.network_model_calls},
                "states": states}


def _run_fixture(directory, *, network_calls=0):
    checkpoint = make_checkpoint(directory)
    row = {"id": "s1", "state": "go",
           "questions": {"q": {"type": "boolean"}}}
    input_path = Path(directory) / "test.jsonl"
    input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    args = harness.build_parser().parse_args([
        "--checkpoint", str(checkpoint), "--input", str(input_path),
        "--batch-states", "1", "--repeats", "2", "--threads", "2",
    ])

    def fake_class(*class_args, **class_kwargs):
        engine = _FakeEngine(*class_args, **class_kwargs)
        engine.network_model_calls = network_calls
        return engine

    with mock.patch.object(harness, "load_predictor_class", return_value=fake_class), \
            mock.patch.object(harness, "effective_thread_count", return_value=2):
        return harness.run(args)


class RunIntegrationTest(unittest.TestCase):
    def test_run_records_full_contract_without_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            report = _run_fixture(directory)
            again = _run_fixture(directory)
        execution = report["execution"]
        self.assertEqual(execution["device"], "cpu")
        self.assertEqual(execution["precision"], "fp32")
        self.assertEqual(execution["threads_requested"], 2)
        self.assertEqual(execution["temperature"], 1.0)
        self.assertEqual(execution["batch_states"], 1)
        self.assertEqual(report["network_model_calls"], 0)
        self.assertGreaterEqual(report["latency"]["cold"]["samples"], 1)
        self.assertGreaterEqual(report["latency"]["warm"]["samples"], 1)
        for key in ("p50_ms", "p95_ms", "p99_ms"):
            self.assertIsNotNone(report["latency"]["cold"][key])
            self.assertIsNotNone(report["latency"]["warm"][key])
        self.assertGreater(report["resources"]["peak_rss_bytes"], 0)
        self.assertEqual(len(report["receipt_sha256"]), 64)
        self.assertEqual(len(report["canonical_sha256"]), 64)
        self.assertIsNotNone(
            report["manifest"]["checkpoint"]["tokenizer_chat_template"]["sha256"])
        self.assertTrue(report["gates"]["zero_network_model_calls"])
        # Deterministic canonicalization ignores timing/memory noise.
        self.assertEqual(report["canonical_sha256"], again["canonical_sha256"])
        self.assertEqual(report["receipt_sha256"], again["receipt_sha256"])
        self.assertEqual(report["output_digest"], again["output_digest"])

    def test_run_fails_closed_on_network_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(harness.ContractError):
                _run_fixture(directory, network_calls=1)


if __name__ == "__main__":
    unittest.main()
