"""Tests for scripts/jevbench_adapter_v1.py (roadmap X1).

Covers: contract-hash binding (changed contract refuses), adapter self-hash
binding, JevBench->NanoJev type mapping, probability projection onto the
exact label set, output record schema, timing fields, fail-closed input
validation, the gated fetch/milestone path, and an end-to-end synthetic run
against the real LoRA seed18 checkpoint on MPS/FP32 (skipped only if the
checkpoint is absent).

Run from the scripts/ directory:

    ../.venv/bin/python -m unittest test_jevbench_adapter_v1 -v
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import jevbench_adapter_v1 as adapter

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "research" / "jevbench_adapter_contract_v1.json"
FIXTURES = ROOT / "research" / "jevbench_fixtures_v1" / "items.jsonl"
CHECKPOINT = ROOT / "checkpoints" / "domain_adaptation_v4_lora_seed18"
FALLBACK_CHECKPOINT = ROOT / "checkpoints" / "local_atomic_seed17" / "variants" / "local_atomic_seed17"


def fixture_items():
    return [json.loads(line) for line in
            FIXTURES.read_text(encoding="utf-8").splitlines() if line.strip()]


class ContractBindingTests(unittest.TestCase):
    def test_real_contract_hash_verifies(self):
        contract, _ = adapter.load_contract(CONTRACT)
        self.assertEqual(contract["schema_version"],
                         "nanojev-jevbench-adapter-contract-v1")
        self.assertEqual(
            contract["contract_sha256"],
            "09b6c6edb9caf0742aec2c61bf30a486d3c7cb8bc52adffd50dbec9ce5b7b114")

    def test_tampered_contract_refuses(self):
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        contract["data_quarantine"]["evaluation_only"] = "maybe"
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path = Path(tmp) / "contract.json"
            path.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaises(adapter.ContractBindingError):
                adapter.load_contract(path)

    def test_rehashed_but_flag_relaxed_contract_refuses(self):
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        contract["data_quarantine"]["training_allowed"] = True
        contract["contract_sha256"] = adapter.digest_value(
            {k: v for k, v in contract.items() if k != "contract_sha256"})
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path = Path(tmp) / "contract.json"
            path.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaises(adapter.ContractBindingError):
                adapter.load_contract(path)

    def test_missing_contract_refuses(self):
        with self.assertRaises((OSError, adapter.ContractBindingError)):
            adapter.load_contract(ROOT / "research" / "nonexistent.json")


class TaskValidationTests(unittest.TestCase):
    def test_all_fixtures_validate(self):
        tasks, _ = adapter.load_tasks(FIXTURES, adapter.SPLITS_SYNTHETIC)
        self.assertEqual(len(tasks), 12)
        self.assertEqual({t["question"]["type"] for t in tasks},
                         {"noul", "choice", "score"})

    def test_fixture_manifest_hash_matches(self):
        manifest = json.loads((FIXTURES.parent / "manifest.json")
                              .read_text(encoding="utf-8"))
        self.assertEqual(manifest["sha256"], adapter.file_hash(FIXTURES))
        self.assertEqual(manifest["items"], 12)
        self.assertFalse(manifest["quarantine"]["training_allowed"])

    def _base(self):
        return {"id": "t1", "family": "f", "state": "s",
                "question": {"type": "noul", "instructions": "q?"},
                "labels": ["no", "yes"], "expected": "no",
                "split": "synthetic",
                "group": None, "provenance": {"synthetic": True}}

    def test_real_benchmark_split_rejected_in_synthetic_mode(self):
        task = self._base()
        task["split"] = "public"
        with self.assertRaisesRegex(ValueError, "split"):
            adapter.validate_task_record(task, adapter.SPLITS_SYNTHETIC)

    def test_missing_synthetic_marker_rejected(self):
        task = self._base()
        task["provenance"] = {}
        with self.assertRaisesRegex(ValueError, "synthetic"):
            adapter.validate_task_record(task, adapter.SPLITS_SYNTHETIC)

    def test_bad_type_rejected(self):
        task = self._base()
        task["question"]["type"] = "essay"
        with self.assertRaisesRegex(ValueError, "type"):
            adapter.validate_task_record(task, adapter.SPLITS_SYNTHETIC)

    def test_choice_labels_must_equal_criteria(self):
        task = self._base()
        task["question"] = {"type": "choice", "instructions": "pick",
                            "criteria": {"a": "A", "b": "B"}}
        task["labels"] = ["a", "c"]
        with self.assertRaisesRegex(ValueError, "criteria keys"):
            adapter.validate_task_record(task, adapter.SPLITS_SYNTHETIC)

    def test_score_labels_must_be_indices(self):
        task = self._base()
        task["question"] = {"type": "score", "instructions": "rate",
                            "criteria": ["lo", "hi"]}
        task["labels"] = ["low", "high"]
        task["expected"] = None
        with self.assertRaisesRegex(ValueError, "level"):
            adapter.validate_task_record(task, adapter.SPLITS_SYNTHETIC)

    def test_label_leak_key_in_state_rejected(self):
        task = self._base()
        task["state"] = {"facts": "x", "answer_key": "yes"}
        with self.assertRaisesRegex(ValueError, "banned"):
            adapter.validate_task_record(task, adapter.SPLITS_SYNTHETIC)

    def test_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path = Path(tmp) / "dup.jsonl"
            line = json.dumps(self._base())
            path.write_text(line + "\n" + line + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                adapter.load_tasks(path, adapter.SPLITS_SYNTHETIC)

    def test_bulk_task_groups_are_deterministic(self):
        class Tok:
            eos_token_id = 0

            def encode(self, text, add_special_tokens=False):
                return [1] * max(1, len(str(text).split()))

        tasks = []
        for i in range(4):
            task = self._base()
            task["id"] = f"t{i}"
            tasks.append(task)
        payloads = {task["id"]: adapter.task_to_nanojev_payload(task)
                    for task in tasks}
        groups, stats = adapter.bulk_task_groups(
            tasks, payloads, Tok(), 2048, max_questions=2,
            max_leaf_tokens=1000)
        self.assertEqual(stats["questions"], 4)
        self.assertEqual(stats["groups"], 2)
        self.assertTrue(all(len(group) <= 2 for group in groups))


class TypeMappingTests(unittest.TestCase):
    def test_noul_maps_to_boolean(self):
        task = next(t for t in fixture_items()
                    if t["id"] == "syn-noul-001")
        payload = adapter.task_to_nanojev_payload(task)
        state = payload["states"][0]
        self.assertEqual(set(state), {"id", "state", "questions"})
        q = state["questions"]["decision"]
        self.assertEqual(q["type"], "boolean")
        self.assertEqual(q["criteria"], task["question"]["criteria"])

    def test_choice_maps_verbatim(self):
        task = next(t for t in fixture_items()
                    if t["id"] == "syn-choice-4opt")
        q = adapter.task_to_nanojev_payload(task)["states"][0]["questions"]["decision"]
        self.assertEqual(q["type"], "choice")
        self.assertEqual(q["criteria"], task["question"]["criteria"])

    def test_score_maps_verbatim(self):
        task = next(t for t in fixture_items()
                    if t["id"] == "syn-score-5lvl")
        q = adapter.task_to_nanojev_payload(task)["states"][0]["questions"]["decision"]
        self.assertEqual(q["type"], "score")
        self.assertEqual(q["criteria"], task["question"]["criteria"])

    def test_gold_never_enters_request(self):
        for task in fixture_items():
            payload = adapter.task_to_nanojev_payload(task)
            blob = json.dumps(payload)
            for forbidden in ("expected", "provenance", "labels", "split",
                              "group"):
                self.assertNotIn(f'"{forbidden}"', blob)

    def test_all_fixture_payloads_pass_nanojev_validator(self):
        for task in fixture_items():
            payload = adapter.task_to_nanojev_payload(task)
            states = adapter.validate_request(payload)
            self.assertEqual(len(states), 1)


class ProbMappingTests(unittest.TestCase):
    def _task(self, qtype, labels, criteria=None):
        q = {"type": qtype, "instructions": "q"}
        if criteria is not None:
            q["criteria"] = criteria
        return {"id": "t", "question": q, "labels": labels}

    def test_noul_projection(self):
        task = self._task("noul", ["no", "yes"])
        probs = adapter.map_probs(task, {"probabilities": {"false": 0.25,
                                                         "true": 0.75}})
        self.assertEqual(probs, {"no": 0.25, "yes": 0.75})

    def test_choice_projection_preserves_label_order(self):
        task = self._task("choice", ["b", "a"], {"a": "A", "b": "B"})
        probs = adapter.map_probs(task, {"probabilities": {"a": 0.9,
                                                         "b": 0.1}})
        self.assertEqual(list(probs), ["b", "a"])
        self.assertAlmostEqual(sum(probs.values()), 1.0)

    def test_score_projection(self):
        task = self._task("score", ["0", "1", "2"], ["l0", "l1", "l2"])
        probs = adapter.map_probs(task, {"probabilities": {"0": 0.2, "1": 0.5,
                                                         "2": 0.3}})
        self.assertEqual(set(probs), {"0", "1", "2"})

    def test_unnormalized_probs_rejected(self):
        task = self._task("noul", ["no", "yes"])
        with self.assertRaisesRegex(ValueError, "sum"):
            adapter.map_probs(task, {"probabilities": {"false": 0.6,
                                                     "true": 0.6}})

    def test_missing_label_rejected(self):
        task = self._task("choice", ["a", "b"], {"a": "A", "b": "B"})
        with self.assertRaises(ValueError):
            adapter.map_probs(task, {"probabilities": {"a": 0.4, "c": 0.6}})

    def test_nonfinite_prob_rejected(self):
        task = self._task("noul", ["no", "yes"])
        with self.assertRaises(ValueError):
            adapter.map_probs(task, {"probabilities": {"false": float("nan"),
                                                     "true": 1.0}})


class GatedPathTests(unittest.TestCase):
    def test_fetch_is_gated(self):
        with self.assertRaises(adapter.BenchmarkFetchGated):
            adapter.fetch_public_tasks()

    def test_milestone_mode_is_gated(self):
        with self.assertRaises(adapter.BenchmarkFetchGated):
            adapter.run(FIXTURES, checkpoint_dir=CHECKPOINT,
                        contract_path=CONTRACT, mode="milestone")

    def test_dry_run_loads_no_model_and_validates(self):
        manifest = adapter.run(FIXTURES, contract_path=CONTRACT,
                               mode="synthetic", dry_run=True)
        self.assertEqual(manifest["status"], "dry_run_validated")
        self.assertEqual(len(manifest["decisions"]), 12)
        self.assertEqual(manifest["quarantine"]["network_calls"], 0)
        self.assertEqual(manifest["quarantine"]["benchmark_rows_consumed"], 0)
        self.assertNotIn("checkpoint", manifest)

    def test_public_diagnostic_dry_run_accepts_public_rows(self):
        task = {"id": "pub-1", "family": "f", "state": "s",
                "question": {"type": "noul", "instructions": "q?"},
                "labels": ["no", "yes"], "expected": "no",
                "split": "public", "group": None,
                "provenance": {"source": "pinned-public-fixture"}}
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path = Path(tmp) / "public.jsonl"
            path.write_text(json.dumps(task) + "\n", encoding="utf-8")
            manifest = adapter.run(path, contract_path=CONTRACT,
                                   mode="public_diagnostic", dry_run=True)
        self.assertEqual(manifest["status"], "dry_run_validated")
        self.assertEqual(manifest["quarantine"]["benchmark_rows_consumed"], 1)


@unittest.skipUnless(CHECKPOINT.is_dir() or FALLBACK_CHECKPOINT.is_dir(),
                     "no local NanoJev checkpoint available")
class SyntheticRunTests(unittest.TestCase):
    """End-to-end: real checkpoint on MPS/FP32 over the synthetic fixtures."""

    @classmethod
    def setUpClass(cls):
        cls.checkpoint = CHECKPOINT if CHECKPOINT.is_dir() else FALLBACK_CHECKPOINT
        cls.manifest = adapter.run(FIXTURES, checkpoint_dir=cls.checkpoint,
                                   contract_path=CONTRACT, mode="synthetic",
                                   device="mps", precision="fp32")

    def test_all_decisions_ok(self):
        self.assertEqual(self.manifest["status"], "completed")
        self.assertTrue(all(d["ok"] for d in self.manifest["decisions"]))

    def test_decision_record_schema(self):
        required = {"schema_version", "task_id", "family", "split", "group",
                    "question_type", "nanojev_type", "ok", "probs",
                    "probs_source", "predicted", "confidence", "expected",
                    "correct", "latency_s", "abstention", "usage"}
        for d in self.manifest["decisions"]:
            self.assertTrue(required <= set(d), d["task_id"])
            self.assertEqual(d["schema_version"], "nanojev-jevbench-decision-v1")
            self.assertEqual(d["probs_source"], "native")

    def test_probs_normalized_over_exact_labels(self):
        tasks = {t["id"]: t for t in fixture_items()}
        for d in self.manifest["decisions"]:
            probs = d["probs"]
            self.assertEqual(set(probs), set(tasks[d["task_id"]]["labels"]))
            self.assertTrue(all(0.0 <= p <= 1.0 for p in probs.values()))
            self.assertAlmostEqual(math.fsum(probs.values()), 1.0, places=5)

    def test_type_mapping_in_output(self):
        expected = {"noul": "boolean", "choice": "choice", "score": "score"}
        for d in self.manifest["decisions"]:
            self.assertEqual(d["nanojev_type"],
                             expected[d["question_type"]])

    def test_usage_counts_scored_candidate_leaves(self):
        for d in self.manifest["decisions"]:
            if d["question_type"] == "noul":
                self.assertEqual(d["usage"]["candidate_paths"], 1)
            else:
                self.assertGreaterEqual(d["usage"]["candidate_paths"], 2)

    def test_timing_fields_present_and_sane(self):
        for d in self.manifest["decisions"]:
            self.assertIsInstance(d["latency_s"], float)
            self.assertGreater(d["latency_s"], 0.0)
        timing = self.manifest["timing"]
        for key in ("min", "max", "mean", "p50", "p95"):
            self.assertIn(key, timing["latency_s"])
        self.assertGreaterEqual(timing["latency_s"]["p95"],
                                timing["latency_s"]["p50"])
        self.assertFalse(self.manifest["execution"]
                         ["model_load_counted_in_latency"])
        self.assertGreater(self.manifest["execution"]["model_load_s"], 0.0)

    def test_manifest_binding_and_quarantine(self):
        self.assertEqual(self.manifest["contract"]["contract_sha256"],
                         "09b6c6edb9caf0742aec2c61bf30a486d3c7cb8bc52adffd5"
                         "0dbec9ce5b7b114")
        self.assertEqual(self.manifest["execution"]["device"], "mps")
        self.assertEqual(self.manifest["execution"]["precision"], "fp32")
        self.assertEqual(self.manifest["quarantine"]["network_calls"], 0)
        self.assertFalse(self.manifest["abstention"]["applied"])

    def test_score_ev_labels_are_index_strings(self):
        for d in self.manifest["decisions"]:
            if d["question_type"] == "score":
                self.assertTrue(all(k.isdigit() for k in d["probs"]))


if __name__ == "__main__":
    unittest.main()
