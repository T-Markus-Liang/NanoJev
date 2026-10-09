#!/usr/bin/env python3
"""Focused tests for the V5 base-plus-repair engineering corpus."""

import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_engineering_corpus_v5 as v5  # noqa: E402
import build_engineering_corpus_v4 as v4  # noqa: E402
from audit_engineering_corpus_v1 import digest, visible_input  # noqa: E402
from build_engineering_corpus_v2 import visible_fingerprint  # noqa: E402
from predict_toy_decisions import validate_request  # noqa: E402
from train_pipeline_decisions import validate_training_row  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
V4_DIR = ROOT / "research" / "engineering_judgment_corpus_v4"
HELDOUT_V1 = ROOT / "research" / "engineering_heldout_v1" / "items.jsonl"
HELDOUT_V2 = ROOT / "research" / "engineering_heldout_v2" / "items.jsonl"


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


class CorpusV5DerivationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = v5.derive_manifest()
        cls.items = cls.manifest["items"]
        cls.repair_items = [item for item in cls.items
                            if item["provenance"]["source_id"] == v5.SOURCE_ID]

    def test_validate_manifest(self):
        self.assertEqual(v5.validate_manifest(self.manifest), [])

    def test_scale_and_repair_targets(self):
        self.assertEqual(self.manifest["item_count"], 5280)
        self.assertEqual(self.manifest["pair_count"], 880)
        self.assertEqual(len(self.manifest["counts_by_family"]), 44)
        self.assertEqual(len(self.manifest["repair_arm"]["families"]), 8)
        counts = self.manifest["counts_by_family"]
        self.assertEqual(counts["six_queue_route"]["choice_candidates"], 6)
        self.assertEqual(counts["six_candidate_failover"]["choice_candidates"], 6)
        self.assertEqual(counts["twelve_lane_dispatch"]["choice_candidates"], 12)
        self.assertEqual(counts["twelve_region_weight_pick"]["choice_candidates"], 12)
        self.assertEqual(counts["twelve_action_triage"]["choice_candidates"], 12)

    def test_v4_base_is_byte_identical(self):
        base = json.loads((V4_DIR / "manifest.json").read_text(encoding="utf-8"))
        base_items = [item for item in self.items
                      if item["item_id"] in {row["item_id"] for row in base["items"]}]
        self.assertEqual(base_items, base["items"])
        self.assertTrue(self.manifest["base_corpus"]["included_byte_identical"])

    def test_repair_items_validate(self):
        self.assertEqual(len(self.repair_items), 960)
        for item in self.repair_items:
            self.assertEqual(item["schema_version"], v5.ITEM_SCHEMA)
            self.assertTrue(item["item_id"].startswith("ej5-"))
            self.assertFalse(item["provenance"]["derived_from_evaluation_corpus"])
            self.assertFalse(item["provenance"]["training_authorized"])
            self.assertEqual(v5.validate_item(item), [], item["item_id"])
            validate_request(item["request"])
            targets = validate_training_row(v5.trainer_row(item))
            self.assertIn(item["qid"], targets)

    def test_split_isolation_and_pair_flips(self):
        fingerprints = {}
        groups = {}
        pairs = {}
        for item in self.items:
            fp = visible_fingerprint(item)
            if fp in fingerprints:
                self.assertEqual(fingerprints[fp], item["split"])
            fingerprints[fp] = item["split"]
            groups.setdefault(item["source_group_id"], item["split"])
            self.assertEqual(groups[item["source_group_id"]], item["split"])
            pairs.setdefault(item["pair_id"], []).append(item)
        for pair_id, members in pairs.items():
            self.assertEqual(len(members), 6)
            bases = {item["question_type"]: item for item in members
                     if item["member"] == "base"}
            variants = {item["question_type"]: item for item in members
                        if item["member"] == "variant"}
            declared = {qtype for qtype, item in bases.items()
                        if item["contrastive"]["is_flip_question"]}
            self.assertTrue(declared)
            for qtype in declared:
                self.assertNotEqual(v4._answer_payload(bases[qtype]),
                                    v4._answer_payload(variants[qtype]))

    def test_repair_has_no_canonical_overlap(self):
        repair_fps = {visible_fingerprint(item) for item in self.repair_items}
        heldout_fps = set()
        for path in (HELDOUT_V1, HELDOUT_V2):
            for row in jsonl(path):
                question = next(iter(row["questions"].values()))
                heldout_fps.add(digest(visible_input(row, question)))
        self.assertFalse(repair_fps & heldout_fps)
        for name in ("v1", "v2", "v3", "v4"):
            manifest = json.loads(
                (ROOT / "research" / f"engineering_judgment_corpus_{name}"
                 / "manifest.json").read_text(encoding="utf-8"))
            old_fps = {visible_fingerprint(item) for item in manifest["items"]}
            self.assertFalse(repair_fps & old_fps, name)


if __name__ == "__main__":
    unittest.main()
