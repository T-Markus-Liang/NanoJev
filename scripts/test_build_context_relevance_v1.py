from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
import unittest

from build_context_relevance_v1 import FAMILIES, SPLITS, KINDS, WIRES, build, make_record, oracle, validate_cohort
from context_gate_v1 import shadow_request, serialized
from train_pipeline_decisions import read_training_records


class ContextRelevanceDatasetTest(unittest.TestCase):
    def record(self, kind="required_field", split="train", wire="openai_chat"):
        return make_record("code", (100, 200, 300, 50), kind, split, wire, random.Random(17))

    def test_oracle_cases_and_metadata_never_enter_input(self):
        for family in FAMILIES:
            for kind in KINDS:
                row = make_record(family, (100, 200, 300, 50), kind, "train", WIRES[0], random.Random(17))
                self.assertEqual(row["gold"]["irrelevant"], kind in KINDS[4:])
                parsed = json.loads(row["state"])
                self.assertEqual(set(parsed), {"conversation", "candidate_pointer", "user_messages_in_order"})
                self.assertNotIn(kind, row["state"])
                self.assertNotIn("train", row["state"])

    def test_exact_live_shadow_renderer_agreement(self):
        for wire in WIRES:
            row = self.record(wire=wire)
            captured = []
            def scorer(payload):
                captured.append(payload)
                return {"states": []}
            metadata = row["metadata"]
            shadow_request(serialized(metadata["body"]).encode(), wire,
                           {"segments": {metadata["candidate_pointer"]: {"eligible": True}}}, scorer)
            self.assertEqual(len(captured), 1)
            self.assertEqual(captured[0]["states"][0]["state"], row["state"])
            self.assertEqual(captured[0]["states"][0]["questions"], row["questions"])

    def test_split_cannot_mask_shared_facts(self):
        first, second = self.record(), self.record("wrong_field", "test")
        self.assertEqual(first["metadata"]["source_group_id"], second["metadata"]["source_group_id"])
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            validate_cohort([first, second])
        second["metadata"]["source_group_id"] = "test:" + second["metadata"]["source_group_id"]
        with self.assertRaisesRegex(ValueError, "actual facts"):
            validate_cohort([first, second])

    def test_duplicate_inputs_and_wrong_labels_rejected(self):
        row = self.record()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_cohort([row, deepcopy(row)])
        row["gold"]["irrelevant"] = True
        with self.assertRaisesRegex(ValueError, "oracle"):
            validate_cohort([row])

    def test_older_fact_and_latest_correction_are_observably_distinct(self):
        old, correction = self.record("superseded"), self.record("latest_correction")
        self.assertNotEqual(old["state"], correction["state"])
        self.assertNotEqual(old["gold"], correction["gold"])
        self.assertIsNone(oracle([], "x", ["y"]))

    def test_build_reproducible_and_semantically_isolated(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp)/"a", Path(tmp)/"b"
            self.assertEqual(build(first), build(second))
            rows, _ = read_training_records(first)
            validate_cohort(rows)
            self.assertEqual(len(rows), sum(SPLITS.values()))
            worlds, model_inputs = {}, set()
            for row in rows:
                world = tuple(row["metadata"]["world"])
                if world in worlds:
                    self.assertEqual(worlds[world], row["split"])
                worlds[world] = row["split"]
                self.assertNotIn(row["state"], model_inputs)
                model_inputs.add(row["state"])
            train_families = {r["family_id"] for r in rows if r["split"] == "train"}
            ood_families = {r["family_id"] for r in rows if r["split"] == "ood"}
            self.assertFalse(train_families & ood_families)
            self.assertEqual(len(worlds), sum(SPLITS.values())//8)
            with self.assertRaisesRegex(ValueError, "empty"):
                build(first)


if __name__ == "__main__":
    unittest.main()
