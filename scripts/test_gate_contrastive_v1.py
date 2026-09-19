"""Tests for the deterministic contrastive gate-data builder (V1).

These tests actually exercise ``scripts/build_gate_contrastive_v1.py``: they
materialize the cohort twice, diff each pair byte-structurally, re-derive the
removal oracle independently, re-run the real byte-preserving shadow core, and
try to feed the builder evaluation corpora it must refuse.
"""

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

from build_gate_contrastive_v1 import (CONTRASTIVE_RULE, DEFAULT_SEED, EXCLUDED_DIR_PARTS, FIELD_POOL,
                                       MANIFEST_NAME, MANIFEST_SCHEMA, MANIFEST_STATUS, MECHANISMS,
                                       PAIR_COUNT, PROVENANCE, RESERVED_TOKENS, SPLITS, WIRES, build,
                                       default_base_fixture, digest_value, excluded_path_reason,
                                       load_base_fixture, self_test, validate_base_fixture,
                                       validate_manifest)
from context_gate_v1 import parse_segments, shadow_request


BUILDER = Path(__file__).resolve().parent / "build_gate_contrastive_v1.py"
ALL_IRRELEVANT_STUB = {"model": "test-all-irrelevant-stub", "revision": "test"}


# --------------------------------------------------------------------------------------
# independent helpers: nothing below is imported from the builder's rule internals
# --------------------------------------------------------------------------------------

def files_tree(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in sorted(Path(root).rglob("*")) if path.is_file()}


def leaf_diff(left, right, path=""):
    """Every JSON leaf that differs, as (path, left, right)."""
    changes = []
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            child = f"{path}/{key}"
            if key not in left or key not in right:
                changes.append((child, left.get(key), right.get(key)))
            else:
                changes.extend(leaf_diff(left[key], right[key], child))
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            changes.append((path, left, right))
        else:
            for index, (first, second) in enumerate(zip(left, right)):
                changes.extend(leaf_diff(first, second, f"{path}/{index}"))
    elif left != right:
        changes.append((path, left, right))
    return changes


def records_of(body, wire_format):
    key = "input" if wire_format == "openai_responses" else "messages"
    return [json.loads(message["content"]) for message in body[key] if message.get("role") == "assistant"]


def pointer_index(pointer, wire_format):
    parts = pointer.strip("/").split("/")
    offset = 0 if wire_format in ("openai_responses", "anthropic_messages") else 1
    return int(parts[1]) - offset


def record_pointer(wire_format, index):
    if wire_format == "openai_responses":
        return f"/input/{index}/content"
    if wire_format == "anthropic_messages":
        return f"/messages/{index}/content"
    return f"/messages/{index + 1}/content"


def at_pointer(body, pointer):
    node = body
    for part in pointer.strip("/").split("/"):
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def oracle_answer(records, entity, fields):
    wanted = set(fields)
    values = {record["field"]: record["value"] for record in records
              if record["entity"] == entity and record["field"] in wanted}
    if set(values) != wanted:
        return None
    return [values[field] for field in fields]


def stub_scorer(records, entity, fields, wire_format):
    """Independent copy of the removal oracle used to drive the real core."""
    def score(payload):
        states = []
        for state in payload["states"]:
            pointer = json.loads(state["state"])["candidate_pointer"]
            index = pointer_index(pointer, wire_format)
            reference = oracle_answer(records, entity, fields)
            dropped = oracle_answer(records[:index] + records[index + 1:], entity, fields)
            probability = 1.0 if reference == dropped else 0.0
            states.append({"id": state["id"], "answers": {"irrelevant": {"type": "boolean",
                          "probabilities": {"false": 1.0 - probability, "true": probability}}}})
        return {"checkpoint": dict(ALL_IRRELEVANT_STUB), "states": states}
    return score


def all_irrelevant_scorer(payload):
    return {"checkpoint": dict(ALL_IRRELEVANT_STUB), "states": [
        {"id": state["id"], "answers": {"irrelevant": {"type": "boolean",
         "probabilities": {"false": 0.0, "true": 1.0}}}} for state in payload["states"]]}


class GateContrastiveBuilderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.output = cls.root / "cohort"
        cls.manifest = build(cls.output, DEFAULT_SEED)
        cls.on_disk = json.loads((cls.output / MANIFEST_NAME).read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def pair(self, pair_id):
        return next(pair for pair in self.manifest["pairs"] if pair["pair_id"] == pair_id)

    def member(self, pair_id, member):
        directory = self.output / "pairs" / pair_id / member
        return (json.loads((directory / "request.json").read_text(encoding="utf-8")),
                (directory / "request.json").read_bytes(),
                json.loads((directory / "sidecar.json").read_text(encoding="utf-8")),
                json.loads((directory / "expected.json").read_text(encoding="utf-8")))

    # ------------------------------------------------------------------ contract

    def test_manifest_integrity_and_frozen_contract(self):
        self.assertEqual(validate_manifest(self.manifest), [])
        self.assertEqual(validate_manifest(self.on_disk), [])
        self.assertEqual(self.on_disk["schema_version"], MANIFEST_SCHEMA)
        self.assertEqual(self.on_disk["status"], MANIFEST_STATUS)
        self.assertEqual(self.on_disk["pair_count"], PAIR_COUNT)
        self.assertEqual(self.on_disk["item_count"], PAIR_COUNT * 2)
        self.assertEqual(self.on_disk["contrastive_rule"], CONTRASTIVE_RULE)
        self.assertEqual(self.on_disk["core"]["threshold"], 0.99)
        self.assertTrue(self.on_disk["core"]["threshold_frozen"])
        self.assertFalse(self.on_disk["core"]["active_filtering"])
        for flag, value in PROVENANCE.items():
            if isinstance(value, bool):
                self.assertIs(self.on_disk["provenance"][flag], value, flag)
        self.assertEqual(self.on_disk["exclusions"]["reserved_token_count"], len(RESERVED_TOKENS))
        self.assertFalse(self.on_disk["exclusions"]["uses_relevance_test_or_ood_as_training_source"])
        self.assertEqual(sorted({pair["mechanism"] for pair in self.manifest["pairs"]}), sorted(MECHANISMS))
        self.assertEqual(len(self.manifest["pairs"]), PAIR_COUNT)
        self.assertEqual(len(self.manifest["items"]), PAIR_COUNT * 2)

    def test_validation_rejects_a_hand_edited_manifest(self):
        for mutate in (
            lambda m: m["pairs"][0].__setitem__("decision_after", "keep"),
            lambda m: m["pairs"][0]["mutation"].__setitem__("fact_field", "entity"),
            lambda m: m["pairs"][0]["mutation"].__setitem__("value_after", 0),
            lambda m: m["provenance"].__setitem__("training_performed", True),
            lambda m: m["core"].__setitem__("threshold", 0.5),
            lambda m: m["pairs"][0].__setitem__("split", "test"),
            lambda m: m.__setitem__("pair_digest", "0" * 64),
        ):
            broken = json.loads(json.dumps(self.on_disk))
            mutate(broken)
            with self.subTest(mutation=broken["pairs"][0]["mutation"]["fact_field"]):
                self.assertTrue(validate_manifest(broken))

    # ------------------------------------------------------------------ determinism

    def test_determinism_across_two_runs_is_byte_identical(self):
        first, second = self.root / "rebuild-a", self.root / "rebuild-b"
        first_manifest, second_manifest = build(first, DEFAULT_SEED), build(second, DEFAULT_SEED)
        self.assertEqual(first_manifest, second_manifest)
        self.assertEqual(files_tree(first), files_tree(second))
        self.assertEqual(files_tree(first), files_tree(self.output))

    def test_seed_changes_the_cohort_but_not_the_frozen_structure(self):
        other_dir = self.root / "seed-other"
        other = build(other_dir, DEFAULT_SEED + 1)
        self.assertEqual([pair["pair_id"] for pair in other["pairs"]],
                         [pair["pair_id"] for pair in self.manifest["pairs"]])
        self.assertEqual([pair["direction"] for pair in other["pairs"]],
                         [pair["direction"] for pair in self.manifest["pairs"]])
        self.assertEqual([pair["split"] for pair in other["pairs"]],
                         [pair["split"] for pair in self.manifest["pairs"]])
        self.assertEqual({pair["source_group_id"] for pair in other["pairs"]}
                         & {pair["source_group_id"] for pair in self.manifest["pairs"]}, set())
        self.assertNotEqual({pair["synthetic_tag"] for pair in other["pairs"]},
                            {pair["synthetic_tag"] for pair in self.manifest["pairs"]})
        for pair, changed in zip(self.manifest["pairs"], other["pairs"]):
            self.assertNotEqual(pair["synthetic_tag"], changed["synthetic_tag"])
            self.assertNotEqual(pair["members"]["keep"]["files"]["request.json"]["sha256"],
                                changed["members"]["keep"]["files"]["request.json"]["sha256"])
        report = self_test(DEFAULT_SEED + 1)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["pairs"], PAIR_COUNT)

    # ------------------------------------------------------------------ contrastive invariant

    def test_one_fact_only_mutation_is_confined_to_the_declared_fact(self):
        for pair in self.manifest["pairs"]:
            with self.subTest(pair=pair["pair_id"]):
                keep_body, _, _, _ = self.member(pair["pair_id"], "keep")
                remove_body, _, _, _ = self.member(pair["pair_id"], "remove")
                changes = leaf_diff(keep_body, remove_body)
                self.assertEqual([path for path, _, _ in changes], [pair["mutation"]["fact_path"]])
                path, keep_content, remove_content = changes[0]
                self.assertEqual(path, pair["mutation"]["fact_path"])
                keep_fact, remove_fact = json.loads(keep_content), json.loads(remove_content)
                fact_diff = leaf_diff(keep_fact, remove_fact)
                self.assertEqual([item[0] for item in fact_diff], ["/" + pair["mutation"]["fact_field"]])
                self.assertEqual(set(keep_fact), set(remove_fact))
                self.assertEqual(len(keep_fact), 3)
                # All other records and both non-record messages are identical.
                records_keep = records_of(keep_body, pair["wire_format"])
                records_remove = records_of(remove_body, pair["wire_format"])
                index = pair["mutation"]["fact_index"]
                self.assertEqual(records_keep[:index], records_remove[:index])
                self.assertEqual(records_keep[index + 1:], records_remove[index + 1:])
                self.assertEqual(record_pointer(pair["wire_format"], index), pair["mutation"]["fact_path"])

    def test_declared_mutation_values_match_the_materialized_members(self):
        for pair in self.manifest["pairs"]:
            with self.subTest(pair=pair["pair_id"]):
                mutation = pair["mutation"]
                keep_body, _, _, _ = self.member(pair["pair_id"], "keep")
                remove_body, _, _, _ = self.member(pair["pair_id"], "remove")
                facts = {"keep": records_of(keep_body, pair["wire_format"])[mutation["fact_index"]],
                         "remove": records_of(remove_body, pair["wire_format"])[mutation["fact_index"]]}
                for member in ("keep", "remove"):
                    self.assertEqual(facts[member][mutation["fact_field"]],
                                     mutation["value_" + ("before" if member == mutation["from_member"] else "after")])
                self.assertNotEqual(mutation["value_before"], mutation["value_after"])
                self.assertEqual(pair["decision_before"], mutation["from_member"])
                self.assertEqual(pair["decision_after"], mutation["to_member"])
                self.assertEqual(pair["direction"], f"{mutation['from_member']}_to_{mutation['to_member']}")
                self.assertEqual(pair["diff"]["changed_paths"], [mutation["fact_path"]])
                self.assertEqual(pair["diff"]["changed_fact_keys"], [mutation["fact_field"]])

    # ------------------------------------------------------------------ gate decision flip

    def test_pair_decisions_flip_and_are_reproduced_by_the_real_core(self):
        query_pattern = re.compile(r"^Using the records in chronological order, return the (\S+) of (\S+)\.$")
        for pair in self.manifest["pairs"]:
            with self.subTest(pair=pair["pair_id"]):
                suggestions = {}
                for member in ("keep", "remove"):
                    body, raw, sidecar, expected = self.member(pair["pair_id"], member)
                    records = records_of(body, pair["wire_format"])
                    # The query is recovered from the request text, never from the expected file.
                    ask = [message["content"] for key in ("messages", "input")
                           for message in body.get(key, []) if message.get("role") == "user"][-1]
                    matched = query_pattern.match(ask)
                    self.assertIsNotNone(matched, ask)
                    field, entity = matched.group(1), matched.group(2)
                    self.assertEqual(expected["query"], {"entity": entity, "fields": [field]})
                    index = pointer_index(expected["candidate_pointer"], pair["wire_format"])
                    # The declared candidate pointer must really address the candidate record.
                    self.assertEqual(json.loads(at_pointer(body, expected["candidate_pointer"])), records[index])
                    self.assertEqual(expected["candidate_pointer"], record_pointer(pair["wire_format"], index))
                    self.assertEqual(sidecar["segments"][expected["candidate_pointer"]], {"eligible": True})
                    reference = oracle_answer(records, entity, [field])
                    self.assertIsNotNone(reference)
                    dropped = oracle_answer(records[:index] + records[index + 1:], entity, [field])
                    decision = "keep" if reference != dropped else "remove"
                    self.assertEqual(decision, expected["decision"])
                    self.assertEqual(decision, member)
                    self.assertEqual(expected["oracle"]["reference"], reference)
                    self.assertEqual(expected["oracle"]["after_removal"], dropped)
                    self.assertEqual(expected["oracle"]["load_bearing"], decision == "keep")
                    self.assertEqual(expected["gate"]["status"], "scored")
                    self.assertEqual(expected["gate"]["policy_threshold"], 0.99)
                    self.assertEqual(expected["gate"]["p_irrelevant"], 0.0 if decision == "keep" else 1.0)
                    _, receipt = shadow_request(raw, pair["wire_format"], sidecar,
                                                stub_scorer(records, entity, [field], pair["wire_format"]),
                                                threshold=0.99)
                    self.assertEqual(receipt["status"], "scored")
                    focus = next(segment for segment in receipt["segments"]
                                 if segment["pointer"] == expected["candidate_pointer"])
                    suggestions[member] = focus["suggestion"]
                    self.assertEqual(focus["suggestion"], "retain" if decision == "keep" else "drop")
                    self.assertEqual(focus["applied"], False)
                    self.assertEqual(receipt["actual_removed_segments"], 0)
                    self.assertEqual(receipt["actual_removed_tokens"], 0)
                self.assertEqual(suggestions, {"keep": "retain", "remove": "drop"})

    def test_protected_segments_are_never_proposed_for_removal(self):
        for pair in self.manifest["pairs"]:
            for member in ("keep", "remove"):
                with self.subTest(pair=pair["pair_id"], member=member):
                    body, raw, sidecar, expected = self.member(pair["pair_id"], member)
                    segments = parse_segments(raw, pair["wire_format"])
                    candidates = {pointer: note for pointer, note in sidecar["segments"].items()}
                    _, receipt = shadow_request(raw, pair["wire_format"], sidecar,
                                                all_irrelevant_scorer, threshold=0.99)
                    self.assertEqual(receipt["status"], "scored")
                    for segment in receipt["segments"]:
                        if segment["pointer"] in candidates:
                            self.assertEqual(segment["suggestion"], "drop")
                        else:
                            self.assertEqual(segment["suggestion"], "retain", segment["pointer"])
                            self.assertIn(segment["reason"], {"protected_structure", "not_explicitly_eligible"})
                    self.assertIn(expected["candidate_pointer"], candidates)
                    self.assertTrue(any(segment.role == "user" for segment in segments))
                    self.assertEqual(receipt["actual_removed_segments"], 0)

    # ------------------------------------------------------------------ hashes

    def test_manifest_hash_stability_and_per_item_content_hashes(self):
        manifest_bytes = (self.output / MANIFEST_NAME).read_bytes()
        rebuild = self.root / "hash-rebuild"
        build(rebuild, DEFAULT_SEED)
        self.assertEqual(manifest_bytes, (rebuild / MANIFEST_NAME).read_bytes())
        on_disk = json.loads(manifest_bytes)
        for item in on_disk["items"]:
            directory = self.output / "pairs" / item["pair_id"] / item["member"]
            request = (directory / "request.json").read_bytes()
            sidecar = json.loads((directory / "sidecar.json").read_text(encoding="utf-8"))
            expected = json.loads((directory / "expected.json").read_text(encoding="utf-8"))
            self.assertEqual(item["files"]["request.json"]["sha256"], hashlib.sha256(request).hexdigest())
            self.assertEqual(item["files"]["request.json"]["bytes"], len(request))
            self.assertEqual(item["files"]["sidecar.json"]["bytes"],
                             (directory / "sidecar.json").stat().st_size)
            self.assertEqual(item["item_content_sha256"], digest_value(
                {"body": json.loads(request), "sidecar": sidecar, "expected": expected}))
        self.assertEqual(on_disk["pair_digest"], digest_value(on_disk["pairs"]))
        self.assertEqual(on_disk["item_digest"], digest_value(on_disk["items"]))
        self.assertEqual(on_disk["base_fixture"]["worlds_sha256"], digest_value(on_disk["base_fixture"]["worlds"]))
        self.assertEqual(on_disk["exclusions"]["reserved_tokens_sha256"], digest_value(list(RESERVED_TOKENS)))

    # ------------------------------------------------------------------ split geometry

    def test_split_geometry_keeps_source_groups_separated(self):
        geometry = self.manifest["split_geometry"]
        self.assertEqual(sorted(geometry), sorted(SPLITS))
        self.assertEqual(sum(len(geometry[split]["pairs"]) for split in SPLITS), PAIR_COUNT)
        groups = {}
        for pair in self.manifest["pairs"]:
            self.assertIn(pair["split"], SPLITS)
            self.assertNotIn(pair["source_group_id"], groups)
            groups[pair["source_group_id"]] = pair["split"]
        for split in SPLITS:
            for pair_id in geometry[split]["pairs"]:
                self.assertEqual(self.pair(pair_id)["split"], split)
        self.assertEqual(set(geometry["test"]["source_groups"])
                         & {group for split in ("train", "dev", "calibration")
                            for group in geometry[split]["source_groups"]}, set())
        self.assertEqual(set(geometry["train"]["mechanisms"]), set(MECHANISMS))
        self.assertGreaterEqual(len(geometry["test"]["mechanisms"]), 2)
        self.assertEqual({pair["direction"] for pair in self.manifest["pairs"]},
                         {"keep_to_remove", "remove_to_keep"})

    # ------------------------------------------------------------------ leakage refusal

    def test_no_relevance_or_ood_source_leaks_into_the_cohort(self):
        for name, content in files_tree(self.output).items():
            text = content.decode("utf-8")
            for token in RESERVED_TOKENS:
                self.assertNotIn(token, text, f"{name} leaked {token!r}")
        self.assertNotIn("build_context_relevance_v1", BUILDER.read_text(encoding="utf-8"))
        entity_tokens = {}
        for pair in self.manifest["pairs"]:
            body, _, _, _ = self.member(pair["pair_id"], "keep")
            rendered = json.dumps(body)
            world = next(world for world in self.manifest["base_fixture"]["worlds"]
                         if world["world_id"] == pair["world_id"])
            entity_tokens[pair["pair_id"]] = [world["entity_a"], world["entity_b"]]
            self.assertIn(world["entity_a"], rendered)
            self.assertIn(world["entity_b"], rendered)
        for pair_id, tokens in entity_tokens.items():
            body, _, _, _ = self.member(pair_id, "keep")
            rendered = json.dumps(body)
            for other_id, other_tokens in entity_tokens.items():
                if other_id == pair_id:
                    continue
                for token in other_tokens:
                    self.assertNotIn(token, rendered, f"{pair_id} leaked {token!r} from {other_id}")

    def test_refuses_evaluation_sources_and_excluded_output_locations(self):
        legitimate = default_base_fixture(DEFAULT_SEED)
        good_path = self.root / "legit" / "base.json"
        good_path.parent.mkdir(parents=True, exist_ok=True)
        good_path.write_text(json.dumps(legitimate), encoding="utf-8")
        loaded = load_base_fixture(good_path)
        self.assertEqual(validate_base_fixture(loaded), [])
        same = self.root / "from-base-fixture"
        manifest = build(same, DEFAULT_SEED, good_path)
        self.assertEqual(manifest["pairs"], self.manifest["pairs"])
        self.assertEqual(files_tree(same), files_tree(self.output))

        for excluded in (self.root / "data" / "base.json",
                         self.root / "context_relevance_v1" / "base.json",
                         self.root / "ood" / "base.json"):
            excluded.parent.mkdir(parents=True, exist_ok=True)
            excluded.write_text(json.dumps(legitimate), encoding="utf-8")
            with self.subTest(path=str(excluded)):
                self.assertIsNotNone(excluded_path_reason(excluded))
                with self.assertRaisesRegex(ValueError, "refused"):
                    load_base_fixture(excluded)
                with self.assertRaisesRegex(ValueError, "refused"):
                    build(self.root / "never", DEFAULT_SEED, excluded)

        poisoned = json.loads(json.dumps(legitimate))
        poisoned["worlds"][0]["entity_a"] = "speed_limit-item"
        poisoned_path = self.root / "legit" / "poisoned.json"
        poisoned_path.write_text(json.dumps(poisoned), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "reserved"):
            load_base_fixture(poisoned_path)

        excluded_output = self.root / "results" / "cohort"
        with self.assertRaisesRegex(ValueError, "refused"):
            build(excluded_output, DEFAULT_SEED)
        self.assertFalse(excluded_output.exists())
        self.assertTrue(EXCLUDED_DIR_PARTS)

    # ------------------------------------------------------------------ CLI / safety

    def test_refuses_a_nonempty_output_directory(self):
        target = self.root / "occupied"
        build(target, DEFAULT_SEED)
        marker = target / "keep.txt"
        marker.write_text("sentinel", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "empty"):
            build(target, DEFAULT_SEED)
        self.assertEqual(marker.read_text(encoding="utf-8"), "sentinel")
        self.assertEqual(files_tree(target)["pairs/supersession-00/keep/request.json"],
                         files_tree(self.output)["pairs/supersession-00/keep/request.json"])

    def test_self_test_and_cli_write_nothing(self):
        report = self_test(DEFAULT_SEED)
        self.assertEqual(report["status"], "ok")
        self.assertFalse(report["wrote_output"])
        self.assertFalse(report["training_performed"])
        self.assertEqual(report["pairs"], PAIR_COUNT)
        self.assertEqual(report["items"], PAIR_COUNT * 2)
        self.assertEqual(report["splits"], {"train": 6, "dev": 2, "calibration": 2, "test": 2})
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run([sys.executable, str(BUILDER), "--self-test"], cwd=temporary,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('"status": "ok"', result.stdout)
            self.assertEqual(list(Path(temporary).iterdir()), [])
        help_result = subprocess.run([sys.executable, str(BUILDER), "--help"], capture_output=True, text=True)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--self-test", help_result.stdout)
        clash = subprocess.run([sys.executable, str(BUILDER), "--self-test", "--output-dir",
                                self.root / "nope"], capture_output=True, text=True)
        self.assertEqual(clash.returncode, 2)
        self.assertFalse((self.root / "nope").exists())
        out = self.root / "cli-build"
        built = subprocess.run([sys.executable, str(BUILDER), "--output-dir", str(out), "--seed",
                                str(DEFAULT_SEED)], capture_output=True, text=True)
        self.assertEqual(built.returncode, 0, built.stderr)
        self.assertIn('"pairs": 12', built.stdout)
        self.assertEqual(files_tree(out), files_tree(self.output))

    def test_builder_has_no_network_or_training_dependency(self):
        source = BUILDER.read_text(encoding="utf-8")
        for forbidden in ("import requests", "import urllib", "import socket", "import torch",
                          "from openai", "subprocess", "http://", "https://"):
            self.assertNotIn(forbidden, source, forbidden)
        self.assertIn("training_performed", source)
        self.assertIn("active_filtering", source)
        self.assertTrue(set(WIRES) <= {"openai_chat", "openai_responses", "anthropic_messages"})
        self.assertTrue(set(FIELD_POOL))


if __name__ == "__main__":
    unittest.main()
