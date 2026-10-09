import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from build_tool_history_fixtures_v1 import (DEFAULT_MANIFEST, KEEP_BOTH, KEEP_BOUNDED, OUTCOMES,
                                            REMOVE_PAIR, REQUIRED_KINDS, build, deterministic_scorer,
                                            load_manifest, self_test, source_group_id, substitute,
                                            synthetic_tag, validate_manifest)
from context_gate_v1 import Bypass, parse_segments, shadow_request


BUILDER = Path(__file__).resolve().parent / "build_tool_history_fixtures_v1.py"
TAG_PLACEHOLDER = "@SYNTH_TAG@"


def tool_linkage(segments):
    """Independent structural linkage detection used to police the declared pairs."""
    calls, results = set(), set()
    for segment in segments:
        value, pointer = segment.value, segment.pointer
        if isinstance(value, dict) and value.get("type") == "tool_use":
            calls.add(pointer)
        elif isinstance(value, dict) and value.get("type") == "tool_result":
            results.add(pointer)
        elif pointer.endswith("/tool_calls"):
            calls.add(pointer)
        elif pointer.endswith("/tool_call_id"):
            results.add(pointer)
        elif segment.role == "tool":
            results.add(pointer)
    return calls, results


def apply_ops(body, ops):
    for operation in ops:
        target, path = body, operation["path"]
        for key in path[:-1]:
            target = target[key]
        if operation["op"] == "set":
            target[path[-1]] = operation["value"]
        else:
            target[path[-1]].append(operation["value"])
    return body


def tree(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(Path(root).rglob("*")) if path.is_file()}


class ToolHistoryFixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = load_manifest()
        cls.seed = cls.manifest["seed"]
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.output = cls.root / "fixtures"
        cls.index = build(cls.output, DEFAULT_MANIFEST)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def case(self, case_id):
        return next(case for case in self.manifest["cases"] if case["case_id"] == case_id)

    def fixture(self, case_id):
        directory = self.output / case_id
        return (directory / "request.json").read_bytes(), json.loads((directory / "sidecar.json").read_text()), \
            json.loads((directory / "expected.json").read_text())

    def test_manifest_integrity_and_required_case_coverage(self):
        self.assertEqual(validate_manifest(self.manifest), [])
        kinds = [case["kind"] for case in self.manifest["cases"]]
        self.assertEqual(sorted(kinds), sorted(REQUIRED_KINDS))
        self.assertEqual(len(self.manifest["cases"]), len(REQUIRED_KINDS))
        self.assertEqual(self.manifest["status"], "frozen_before_shadow_materialization")
        self.assertTrue(self.manifest["gate_contract"]["threshold_frozen"])
        self.assertEqual(self.manifest["gate_contract"]["threshold"], 0.99)
        self.assertFalse(self.manifest["gate_contract"]["active_filtering"])
        for flag, value in self.manifest["provenance"].items():
            if flag.startswith("contains_") or flag == "reuses_frozen_relevance_corpus":
                self.assertIs(value, False, flag)
        for case in self.manifest["cases"]:
            self.assertTrue(case["segments"], case["case_id"])
            statuses = {segment["expected_status"] for segment in case["segments"]}
            self.assertTrue(statuses <= {"protected", "eligible"}, case["case_id"])
            self.assertIn("protected", statuses)

    def test_protected_and_eligible_status_is_machine_verified(self):
        # Every case is cross-checked against the real core during validate_manifest; a
        # deliberate corruption of one declared status must be rejected.
        broken = json.loads(json.dumps(self.manifest))
        broken["cases"][0]["segments"][0]["expected_suggestion"] = "drop"
        self.assertTrue(validate_manifest(broken))
        broken = json.loads(json.dumps(self.manifest))
        broken["cases"][1]["segments"][3]["expected_reason_code"] = "high_irrelevance_score"
        self.assertTrue(validate_manifest(broken))
        broken = json.loads(json.dumps(self.manifest))
        broken["gate_contract"]["threshold"] = 0.5
        self.assertTrue(validate_manifest(broken))

    def test_self_test_mode_validates_without_writing_output(self):
        report = self_test()
        self.assertEqual(report["status"], "ok")
        self.assertFalse(report["wrote_output"])
        self.assertEqual(report["cases"], len(REQUIRED_KINDS))
        self.assertEqual(report["scored_cases"] + report["bypass_cases"], len(REQUIRED_KINDS))
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run([sys.executable, str(BUILDER), "--self-test"], cwd=temporary,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('"status": "ok"', result.stdout)
            self.assertEqual(list(Path(temporary).iterdir()), [])
        help_result = subprocess.run([sys.executable, str(BUILDER), "--help"], capture_output=True, text=True)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--self-test", help_result.stdout)
        clash = subprocess.run([sys.executable, str(BUILDER), "--self-test", "--output-dir", self.root / "nope"],
                               capture_output=True, text=True)
        self.assertEqual(clash.returncode, 2)
        self.assertFalse((self.root / "nope").exists())

    def test_deterministic_rebuild_is_byte_identical(self):
        first, second = self.root / "rebuild-a", self.root / "rebuild-b"
        first_index, second_index = build(first, DEFAULT_MANIFEST), build(second, DEFAULT_MANIFEST)
        self.assertEqual(first_index, second_index)
        self.assertEqual(tree(first), tree(second))
        self.assertEqual(tree(first), tree(self.output))

    def test_refuses_nonempty_output_directory(self):
        target = self.root / "occupied"
        build(target, DEFAULT_MANIFEST)
        marker = target / "keep.txt"
        marker.write_text("sentinel", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "empty"):
            build(target, DEFAULT_MANIFEST)
        self.assertEqual(marker.read_text(encoding="utf-8"), "sentinel")
        self.assertEqual(tree(target)["success/request.json"], tree(self.output)["success/request.json"])

    def test_source_group_isolation_never_spans_fixtures(self):
        groups = self.manifest["source_groups"]
        self.assertEqual([group["fixture_id"] for group in groups], [case["case_id"] for case in self.manifest["cases"]])
        self.assertEqual(len({group["group_id"] for group in groups}), len(groups))
        for group in groups:
            case = self.case(group["fixture_id"])
            self.assertEqual(group["group_id"], source_group_id(case["kind"], case["source_world"]))
            self.assertEqual(group["entity_tokens"], [case["source_world"][key] for key in sorted(case["source_world"])
                                                      if key.startswith("entity")])
        all_tokens = {token for group in groups for token in group["entity_tokens"]}
        self.assertEqual(len(all_tokens), sum(len(group["entity_tokens"]) for group in groups))
        for case in self.manifest["cases"]:
            text = (self.output / case["case_id"] / "request.json").read_text(encoding="utf-8")
            self.assertNotIn(TAG_PLACEHOLDER, text)
            tag = synthetic_tag(case["case_id"], self.seed)
            self.assertIn(tag, text)
            for group in groups:
                other = group["fixture_id"]
                other_tag = synthetic_tag(other, self.seed)
                if other == case["case_id"]:
                    for token in group["entity_tokens"]:
                        self.assertIn(token, text)
                else:
                    self.assertNotIn(other_tag, text)
                    for token in group["entity_tokens"]:
                        self.assertNotIn(token, text)

    def test_no_orphan_results_and_atomic_call_result_pairing(self):
        for case in self.manifest["cases"]:
            with self.subTest(case=case["case_id"]):
                raw, _, expected = self.fixture(case["case_id"])
                declared = {segment["pointer"] for segment in expected["segments"]}
                try:
                    segments = parse_segments(raw, case["wire_format"])
                except Bypass as error:
                    self.assertFalse(case["pairing_resolved"])
                    self.assertEqual(str(error), case["expected_gate"]["reason"])
                    self.assertTrue(case["ambiguity_probe"]["ops"])
                    body = json.loads(raw.decode("utf-8"))
                    probe = substitute(case["ambiguity_probe"], synthetic_tag(case["case_id"], self.seed))
                    segments = parse_segments(
                        ("\n" + json.dumps(apply_ops(body, probe["ops"]), ensure_ascii=False, indent=2) + "\n").encode(),
                        case["wire_format"])
                else:
                    self.assertTrue(case["pairing_resolved"])
                calls, results = tool_linkage(segments)
                pair_calls, pair_results = set(), set()
                for pair in expected["pairing"]:
                    self.assertNotIn(pair["call_segment"], pair_calls)
                    pair_calls.add(pair["call_segment"])
                    self.assertIn(pair["call_segment"], declared)
                    for result in pair["result_segments"]:
                        self.assertNotIn(result, pair_results, "a result may not belong to two pairs")
                        pair_results.add(result)
                        self.assertIn(result, declared)
                self.assertEqual({pointer for pointer in calls if pointer in declared}, pair_calls)
                self.assertEqual({pointer for pointer in results if pointer in declared}, pair_results)
                self.assertTrue(pair_calls <= declared and pair_results <= declared)
                # No orphan result: every result segment in the request is paired.
                self.assertFalse({pointer for pointer in results if pointer in declared} - pair_results)

    def test_shadow_receipts_retain_protected_text_and_never_apply(self):
        for case in self.manifest["cases"]:
            with self.subTest(case=case["case_id"]):
                raw, sidecar, expected = self.fixture(case["case_id"])
                output, receipt = shadow_request(raw, case["wire_format"], sidecar, deterministic_scorer)
                self.assertIs(output, raw)
                self.assertEqual(receipt["request_sha256"], receipt["forwarded_sha256"])
                self.assertEqual(receipt["status"], expected["expected_gate"]["status"])
                self.assertEqual(receipt["reason"], expected["expected_gate"]["reason"])
                self.assertEqual(receipt["actual_removed_segments"], 0)
                self.assertEqual(receipt["actual_removed_tokens"], 0)
                self.assertEqual(receipt["policy_threshold"], 0.99)
                self.assertFalse(receipt["probabilities_calibrated"])
                declared = {segment["pointer"]: segment for segment in expected["segments"]}
                if expected["expected_gate"]["segments_observed"]:
                    self.assertEqual({segment["pointer"] for segment in receipt["segments"]}, set(declared))
                    for segment in receipt["segments"]:
                        self.assertFalse(segment["applied"])
                        self.assertEqual(segment["suggestion"], declared[segment["pointer"]]["expected_suggestion"])
                        self.assertEqual(segment["reason"], declared[segment["pointer"]]["expected_reason_code"])
                else:
                    self.assertEqual(receipt["segments"], [])
                    self.assertTrue(receipt["forwarded_unchanged"])

    def test_protected_segments_are_never_proposed_for_removal(self):
        for case in self.manifest["cases"]:
            raw, sidecar, expected = self.fixture(case["case_id"])
            if not expected["expected_gate"]["segments_observed"]:
                continue
            _, receipt = shadow_request(raw, case["wire_format"], sidecar, deterministic_scorer)
            for segment in receipt["segments"]:
                declared = next(item for item in expected["segments"] if item["pointer"] == segment["pointer"])
                if declared["expected_status"] == "protected":
                    self.assertEqual(segment["suggestion"], "retain", (case["case_id"], segment["pointer"]))
                else:
                    self.assertEqual(segment["suggestion"], declared["expected_suggestion"])

    def test_unresolved_linkage_bypasses_without_a_shadow_plan(self):
        for case_id in ("duplicate_ids", "orphan_result", "pending_call"):
            raw, sidecar, expected = self.fixture(case_id)
            output, receipt = shadow_request(raw, expected["wire_format"], sidecar, deterministic_scorer)
            self.assertIs(output, raw)
            self.assertEqual(receipt["status"], "bypass")
            self.assertEqual(receipt["reason"], "unresolved_tool_link")
            self.assertEqual(receipt["segments"], [])
            self.assertTrue(receipt["forwarded_unchanged"])

    def test_declared_outcomes_respect_dependency_flags(self):
        pairs = [pair for case in self.manifest["cases"] for pair in case["pairing"]]
        self.assertTrue(pairs)
        for pair in pairs:
            self.assertTrue(set(pair["allowed_outcomes"]) <= set(OUTCOMES))
            self.assertIn(KEEP_BOTH, pair["allowed_outcomes"])
            hard = any(pair[flag] for flag in ("non_repeatable", "credential_bearing", "contains_error",
                                               "stale_evidence", "correction_dependent"))
            if hard:
                self.assertNotIn(KEEP_BOUNDED, pair["allowed_outcomes"])
                self.assertNotIn(REMOVE_PAIR, pair["allowed_outcomes"])
            if pair["volatile_source"]:
                self.assertNotIn(REMOVE_PAIR, pair["allowed_outcomes"])
            if not pair["repeatable"]:
                self.assertNotIn(REMOVE_PAIR, pair["allowed_outcomes"])
        self.assertTrue(any(set(pair["allowed_outcomes"]) == set(OUTCOMES) for pair in pairs))
        self.assertTrue(any(KEEP_BOUNDED in pair["allowed_outcomes"] and REMOVE_PAIR not in pair["allowed_outcomes"]
                            for pair in pairs))
        self.assertTrue(any(pair["allowed_outcomes"] == [KEEP_BOTH] for pair in pairs))

    def test_secret_case_is_placeholder_only_and_absent_from_receipts(self):
        case = self.case("secrets_credentials")
        raw, sidecar, _ = self.fixture("secrets_credentials")
        self.assertIn(b"not-a-real-secret-0000", raw)
        _, receipt = shadow_request(raw, case["wire_format"], sidecar, deterministic_scorer)
        rendered = json.dumps(receipt)
        self.assertNotIn("not-a-real-secret-0000", rendered)
        self.assertNotIn("zql", rendered)
        self.assertNotIn(synthetic_tag("secrets_credentials", self.seed), rendered)
        self.assertEqual(receipt["status"], "scored")
        self.assertTrue(all(segment["applied"] is False for segment in receipt["segments"]))

    def test_index_records_every_materialized_fixture(self):
        index = json.loads((self.output / "fixture_index.json").read_text(encoding="utf-8"))
        self.assertEqual(index["case_count"], len(REQUIRED_KINDS))
        self.assertTrue(index["shadow_only"])
        self.assertEqual({entry["case_id"] for entry in index["cases"]},
                         {case["case_id"] for case in self.manifest["cases"]})
        for entry in index["cases"]:
            request = (self.output / entry["case_id"] / "request.json").read_bytes()
            self.assertEqual(entry["request_sha256"], hashlib.sha256(request).hexdigest())
            self.assertEqual(len(request), entry["request_bytes"])


if __name__ == "__main__":
    unittest.main()
