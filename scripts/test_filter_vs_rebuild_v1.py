import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import build_tool_history_fixtures_v1 as a4_builder
from build_tool_history_fixtures_v1 import validate_manifest

import filter_vs_rebuild_v1 as a5
from filter_vs_rebuild_v1 import (ARMS, AUTHORIZATION_FLAGS, DEFAULT_A4_MANIFEST, DEFAULT_PROTOCOL,
                                  EXPECTED_FAMILIES, EXPECTED_A4_REL_PATH, FROZEN_A4_MANIFEST_SHA256,
                                  METRICS, NULL_FIELDS, PROTOCOL_SCHEMA, STATUS_OK, aggregate_gate,
                                  build_bundle, canonical_bytes, derive_protected_families,
                                  load_protocol, preflight, sha256_hex, synthetic_receipts,
                                  validate_protocol, write_exclusive)


SCRIPT = Path(__file__).resolve().parent / "filter_vs_rebuild_v1.py"


def clone(value):
    return json.loads(json.dumps(value))


class A5ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = load_protocol()
        cls.a4_bytes = DEFAULT_A4_MANIFEST.read_bytes()
        cls.a4_manifest = json.loads(cls.a4_bytes.decode("utf-8"))
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    # --- protocol shape -------------------------------------------------

    def test_protocol_schema_and_status(self):
        self.assertEqual(validate_protocol(self.protocol), [])
        self.assertEqual(self.protocol["schema_version"], PROTOCOL_SCHEMA)
        self.assertEqual(self.protocol["status"], "frozen_before_measurement")
        self.assertTrue(self.protocol["frozen"])
        self.assertEqual(self.protocol["execution_mode"], "synthetic_preflight_control_only")
        self.assertTrue(self.protocol["synthetic_only"])

    def test_five_arms_are_frozen_in_order(self):
        self.assertEqual([arm["arm_id"] for arm in self.protocol["arms"]], list(ARMS))
        self.assertEqual(len(self.protocol["arms"]), 5)
        for index, arm in enumerate(self.protocol["arms"]):
            self.assertEqual(arm["order"], index)
            self.assertTrue(arm["fail_open"])
            self.assertIn(arm["transformation_class"],
                          {"identity", "deterministic_exact_duplicate_removal",
                           "verbatim_retain_relevance_removal", "abstractive_reconstruction",
                           "retrieval_reconstruction"})

    def test_a4_manifest_is_bound_by_exact_sha256(self):
        self.assertEqual(sha256_hex(self.a4_bytes), FROZEN_A4_MANIFEST_SHA256)
        self.assertEqual(self.protocol["source_binding"]["a4_manifest_path"], EXPECTED_A4_REL_PATH)
        self.assertEqual(self.protocol["source_binding"]["a4_manifest_sha256"], FROZEN_A4_MANIFEST_SHA256)
        self.assertEqual(validate_manifest(self.a4_manifest), [])
        self.assertEqual(self.protocol["source_binding"]["a4_case_count"], len(self.a4_manifest["cases"]))

    def test_fixture_kind_to_task_family_map_is_total_and_frozen(self):
        family_map = self.protocol["fixture_family_map"]
        kinds = sorted(case["kind"] for case in self.a4_manifest["cases"])
        self.assertEqual(sorted(family_map), kinds)
        families = {family["family_id"]: set(family["kinds"]) for family in self.protocol["task_families"]}
        self.assertEqual(sorted(families), sorted(EXPECTED_FAMILIES))
        for kind, family_id in family_map.items():
            self.assertIn(kind, families[family_id])
        derived = derive_protected_families(self.a4_manifest, family_map)
        self.assertEqual(set(self.protocol["protected_families"]), derived)
        self.assertEqual(derived, {"error_evidence", "ambiguous_linkage", "side_effect_safety",
                                   "snapshot_consistency", "correction_dependency", "credential_handling"})

    def test_metrics_and_provider_only_cost_contract(self):
        ids = [metric["metric_id"] for metric in self.protocol["metrics"]]
        self.assertEqual(sorted(ids), sorted(METRICS))
        for metric in self.protocol["metrics"]:
            if metric["metric_id"] in ("restore_cost", "end_to_end_cost"):
                self.assertEqual(metric["unit"], "provider_tokens")
        cost = self.protocol["cost_contract"]
        self.assertEqual(cost["cost_basis"], "provider_reported_or_tokenizer")
        self.assertFalse(cost["character_estimates_allowed"])
        self.assertFalse(cost["manual_character_counts_allowed"])
        self.assertTrue(cost["provider_prompt_tokens_required"])
        self.assertTrue(cost["tokenizer_id_required_for_tokenizer_counts"])

    # --- fail-closed corruption ----------------------------------------

    def test_fail_closed_on_schema_and_enum_mismatch(self):
        broken = clone(self.protocol)
        broken["schema_version"] = "nanojev-t10-a5-wrong"
        self.assertTrue(validate_protocol(broken))
        broken = clone(self.protocol)
        broken["arms"][2]["arm_id"] = "unknown_arm"
        self.assertTrue(validate_protocol(broken))
        broken = clone(self.protocol)
        broken["metrics"] = broken["metrics"][:-1]
        self.assertTrue(validate_protocol(broken))
        broken = clone(self.protocol)
        broken["fixture_family_map"]["success"] = "mystery_family"
        self.assertTrue(validate_protocol(broken))
        broken = clone(self.protocol)
        broken["protected_families"] = sorted(set(broken["protected_families"]) - {"error_evidence"})
        self.assertTrue(validate_protocol(broken))

    def test_fail_closed_on_hash_and_source_path_mismatch(self):
        broken = clone(self.protocol)
        broken["source_binding"]["a4_manifest_sha256"] = "0" * 64
        self.assertTrue(validate_protocol(broken))
        broken = clone(self.protocol)
        broken["source_binding"]["a4_manifest_path"] = "research/other.json"
        self.assertTrue(validate_protocol(broken))
        # A manifest whose bytes do not hash to the frozen digest must be rejected.
        errors = validate_protocol(self.protocol, a4_manifest=clone(self.a4_manifest), a4_bytes=b"{}")
        self.assertIn("A4 manifest bytes do not match the frozen SHA-256", errors)

    def test_fail_closed_on_unreadable_a4_bytes(self):
        errors = validate_protocol(self.protocol, a4_manifest=None, a4_bytes=b"{")
        self.assertTrue(any(error.startswith("A4 manifest unreadable:") for error in errors))

    def test_fail_closed_on_authorization_and_network(self):
        for flag in AUTHORIZATION_FLAGS:
            broken = clone(self.protocol)
            broken["authorization"][flag] = True
            self.assertTrue(validate_protocol(broken), flag)
        broken = clone(self.protocol)
        broken["network_model_calls"] = 1
        self.assertTrue(validate_protocol(broken))
        broken = clone(self.protocol)
        broken["model_loaded"] = True
        self.assertTrue(validate_protocol(broken))
        broken = clone(self.protocol)
        broken["measurement_evidence"] = True
        self.assertTrue(validate_protocol(broken))

    def test_fail_closed_on_character_estimate_costs(self):
        broken = clone(self.protocol)
        broken["cost_contract"]["character_estimates_allowed"] = True
        self.assertTrue(validate_protocol(broken))

    def test_fail_closed_on_acceptance_contract_drift(self):
        for key in ("require_paired_confidence_intervals", "report_per_task_family"):
            broken = clone(self.protocol)
            broken["acceptance"][key] = False
            self.assertTrue(validate_protocol(broken), key)
        broken = clone(self.protocol)
        broken["acceptance"]["min_task_families"] = 1
        self.assertTrue(validate_protocol(broken))
        broken = clone(self.protocol)
        broken["cost_contract"]["manual_character_counts_allowed"] = True
        self.assertTrue(validate_protocol(broken))
        broken = clone(self.protocol)
        for metric in broken["metrics"]:
            if metric["metric_id"] == "end_to_end_cost":
                metric["unit"] = "characters"
        self.assertTrue(validate_protocol(broken))

    def test_fail_closed_on_raw_content_leak(self):
        token = self.a4_manifest["cases"][0]["leak_tokens"][0]
        broken = clone(self.protocol)
        broken["notes"] = f"accidental raw content {token}"
        errors = validate_protocol(broken)
        self.assertTrue(any("raw fixture content" in error for error in errors))

    # --- preflight ------------------------------------------------------

    def test_preflight_is_read_only_and_not_authorized(self):
        before = sha256_hex(DEFAULT_A4_MANIFEST.read_bytes())
        report = preflight()
        self.assertEqual(report["status"], STATUS_OK)
        self.assertEqual(report["errors"], [])
        self.assertFalse(report["wrote_output"])
        self.assertFalse(report["measurement_evidence"])
        self.assertEqual(report["network_model_calls"], 0)
        self.assertFalse(report["model_loaded"])
        self.assertTrue(report["synthetic_only"])
        self.assertEqual(report["authorization"], {flag: False for flag in AUTHORIZATION_FLAGS})
        self.assertEqual(report["a4_manifest"]["observed_sha256"], FROZEN_A4_MANIFEST_SHA256)
        self.assertEqual(sha256_hex(DEFAULT_A4_MANIFEST.read_bytes()), before)

    def test_preflight_fails_closed_on_path_mismatch(self):
        other = self.root / "other_manifest.json"
        other.write_bytes(self.a4_bytes)
        report = preflight(DEFAULT_PROTOCOL, other)
        self.assertEqual(report["status"], a5.STATUS_BLOCKED)
        self.assertIn("a4_manifest_path_mismatch", report["errors"])
        self.assertEqual(report["authorization"], {flag: False for flag in AUTHORIZATION_FLAGS})

    def test_preflight_fails_closed_on_corrupted_protocol_file(self):
        corrupted = self.root / "corrupted_protocol.json"
        broken = clone(self.protocol)
        broken["authorization"]["network_access"] = True
        corrupted.write_bytes(canonical_bytes(broken))
        report = preflight(corrupted)
        self.assertEqual(report["status"], a5.STATUS_BLOCKED)
        self.assertTrue(any("network_access" in error for error in report["errors"]))

    # --- synthetic receipts --------------------------------------------

    def test_synthetic_receipts_are_placeholders_only(self):
        receipts = synthetic_receipts(self.protocol, sha256_hex(DEFAULT_PROTOCOL.read_bytes()))
        self.assertEqual([receipt["arm_id"] for receipt in receipts], list(ARMS))
        for receipt in receipts:
            self.assertEqual(receipt["schema_version"], a5.RECEIPT_SCHEMA)
            self.assertEqual(receipt["receipt_type"], "synthetic_placeholder")
            self.assertFalse(receipt["applied"])
            self.assertFalse(receipt["measurement_evidence"])
            self.assertFalse(receipt["model_loaded"])
            self.assertEqual(receipt["network_model_calls"], 0)
            self.assertEqual(receipt["protocol_sha256"], sha256_hex(DEFAULT_PROTOCOL.read_bytes()))
            receipt_hash = receipt["receipt_sha256"]
            payload = dict(receipt)
            payload["receipt_sha256"] = None
            self.assertEqual(receipt_hash, sha256_hex(canonical_bytes(payload)))
            self.assertTrue(receipt["synthetic_only"])
            self.assertEqual(receipt["authorization"], {flag: False for flag in AUTHORIZATION_FLAGS})
            for group, fields in NULL_FIELDS.items():
                for field in fields:
                    self.assertIsNone(receipt[group][field], (receipt["arm_id"], group, field))
            self.assertTrue(receipt["fail_open"]["forwarded_unchanged"])
            self.assertEqual(receipt["fail_open"]["action"], "forward_original_bytes")
            self.assertFalse(receipt["fail_open"]["summary_substituted"])

    def test_receipts_never_contain_raw_fixture_content(self):
        rendered = json.dumps(synthetic_receipts(self.protocol), ensure_ascii=False)
        for case in self.a4_manifest["cases"]:
            for token in case["leak_tokens"]:
                self.assertNotIn(token, rendered, case["case_id"])
        self.assertNotIn("SYNTHETIC_ZQL_API_KEY", rendered)
        self.assertNotIn("not-a-real-secret", rendered)

    # --- aggregate gate -------------------------------------------------

    def _measured(self, **overrides):
        result = {"measurement_evidence": True, "protected_segment_deletion": 0,
                  "paired_confidence_intervals": True, "success_regression": 0.0,
                  "token_reduction_fraction": 0.5}
        result.update(overrides)
        return result

    def test_aggregate_gate_protected_loss_blocks_aggregate_win(self):
        results = {family: self._measured() for family in EXPECTED_FAMILIES}
        results["credential_handling"] = self._measured(protected_segment_deletion=1)
        decision = aggregate_gate(self.protocol, results)
        self.assertFalse(decision["aggregate_win"])
        self.assertTrue(decision["blocked"])
        self.assertEqual(decision["reason"], "protected_family_loss")
        self.assertEqual([entry["family"] for entry in decision["blocked_by"]], ["credential_handling"])

    def test_aggregate_gate_protected_regression_blocks_aggregate_win(self):
        results = {family: self._measured(token_reduction_fraction=0.9) for family in EXPECTED_FAMILIES}
        results["ambiguous_linkage"] = self._measured(success_regression=0.02)
        decision = aggregate_gate(self.protocol, results)
        self.assertFalse(decision["aggregate_win"])
        self.assertEqual(decision["reason"], "protected_family_loss")

    def test_aggregate_gate_requires_measurement_evidence(self):
        results = {family: self._measured(measurement_evidence=False) for family in EXPECTED_FAMILIES}
        decision = aggregate_gate(self.protocol, results)
        self.assertFalse(decision["aggregate_win"])
        self.assertFalse(decision["measurement_evidence"])

    def test_aggregate_gate_can_only_pass_when_all_families_measured_and_clean(self):
        results = {family: self._measured() for family in EXPECTED_FAMILIES}
        decision = aggregate_gate(self.protocol, results)
        self.assertTrue(decision["aggregate_win"])
        self.assertFalse(decision["blocked"])
        low = {family: self._measured(token_reduction_fraction=0.01) for family in EXPECTED_FAMILIES}
        self.assertFalse(aggregate_gate(self.protocol, low)["aggregate_win"])

    # --- determinism and CLI -------------------------------------------

    def test_bundle_is_deterministic_and_canonical(self):
        first = build_bundle()
        second = build_bundle()
        self.assertEqual(first, second)
        self.assertEqual(canonical_bytes(first), canonical_bytes(second))
        self.assertEqual([receipt["arm_id"] for receipt in first["receipts"]], list(ARMS))
        self.assertEqual(first["status"], STATUS_OK)

    def test_write_exclusive_is_deterministic_and_refuses_overwrite(self):
        target_a = self.root / "a" / "bundle.json"
        target_b = self.root / "b" / "bundle.json"
        payload = build_bundle()
        data_a = write_exclusive(target_a, payload)
        data_b = write_exclusive(target_b, payload)
        self.assertEqual(data_a, data_b)
        self.assertEqual(target_a.read_bytes(), target_b.read_bytes())
        with self.assertRaises(FileExistsError):
            write_exclusive(target_a, payload)

    def test_cli_self_test_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run([sys.executable, str(SCRIPT), "--self-test"], cwd=temporary,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(STATUS_OK, result.stdout)
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_cli_output_is_exclusive_create(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "bundle.json"
            first = subprocess.run([sys.executable, str(SCRIPT), "--output", str(out)], cwd=temporary,
                                   capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertTrue(out.exists())
            second = subprocess.run([sys.executable, str(SCRIPT), "--output", str(out)], cwd=temporary,
                                    capture_output=True, text=True)
            self.assertEqual(second.returncode, 2)
            self.assertIn("output_exists", second.stdout)
            # stdout mode is equivalent and writes nothing.
            before = set(Path(temporary).iterdir())
            stdout = subprocess.run([sys.executable, str(SCRIPT), "--stdout"], cwd=temporary,
                                    capture_output=True, text=True)
            self.assertEqual(stdout.returncode, 0, stdout.stderr)
            self.assertEqual(set(Path(temporary).iterdir()), before)
            self.assertEqual([arm["arm_id"] for arm in json.loads(stdout.stdout)["receipts"]], list(ARMS))

    def test_cli_mode_requires_exactly_one_action(self):
        result = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        result = subprocess.run([sys.executable, str(SCRIPT), "--self-test", "--stdout"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)

    def test_a4_manifest_and_builder_are_reused_read_only(self):
        before = DEFAULT_A4_MANIFEST.read_bytes()
        preflight()
        build_bundle()
        self.assertEqual(DEFAULT_A4_MANIFEST.read_bytes(), before)
        self.assertTrue(callable(a4_builder.validate_manifest))
        self.assertTrue(callable(a4_builder.load_manifest))


if __name__ == "__main__":
    unittest.main()
