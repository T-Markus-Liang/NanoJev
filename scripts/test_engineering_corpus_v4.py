#!/usr/bin/env python3
"""Tests for the X4 engineering-judgment corpus builder (V4).

These tests re-derive the corpus from its frozen seed, check the emitted
manifest byte-for-byte, re-run the real served request validator and the real
trainer row validator on every row, verify the contrastive-pair invariants and
split isolation, and confirm the new v4 surface area (analytic-chance families,
32/255-candidate choice items, T9c calibration group minimums) plus zero
canonical-input overlap with every earlier cohort.
"""

import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_engineering_corpus_v4 as v4  # noqa: E402
import build_engineering_corpus_v3 as v3  # noqa: E402
from audit_engineering_corpus_v1 import digest, visible_input  # noqa: E402
from build_engineering_corpus_v2 import visible_fingerprint  # noqa: E402
from predict_toy_decisions import validate_request  # noqa: E402
from train_pipeline_decisions import validate_training_row  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "research" / "engineering_judgment_corpus_v4"
V3_DIR = ROOT / "research" / "engineering_judgment_corpus_v3"
HELDOUT_V1 = ROOT / "research" / "engineering_heldout_v1" / "items.jsonl"
HELDOUT_V2 = ROOT / "research" / "engineering_heldout_v2" / "items.jsonl"


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


class CorpusV4DerivationTests(unittest.TestCase):
    """Heavyweight: one re-derivation per test (~40s each)."""

    def test_self_test_passes(self):
        report = v4.self_test()
        self.assertEqual(report["status"], "ok", report["errors"])
        self.assertEqual(report["items"], 4320)
        self.assertEqual(report["families"], 36)
        self.assertEqual(report["new_families"], 18)
        self.assertEqual(report["components"], report["pairs"])
        self.assertFalse(report["training_performed"])

    def test_built_corpus_matches_rederived_bytes(self):
        errors = v4.check(CORPUS_DIR)
        self.assertEqual(errors, [])


class CorpusV4ContentTests(unittest.TestCase):
    """One shared derivation; structural assertions over the manifest."""

    @classmethod
    def setUpClass(cls):
        cls.manifest = v4.derive_manifest()
        cls.items = cls.manifest["items"]

    def test_scale_targets(self):
        self.assertGreaterEqual(self.manifest["item_count"], 4000)
        self.assertGreaterEqual(len(self.manifest["counts_by_family"]), 30)
        self.assertEqual(self.manifest["pair_count"], 720)

    def test_family_cardinality_strata(self):
        by_family = self.manifest["counts_by_family"]
        cardinalities = sorted({f["choice_candidates"] for f in by_family.values()})
        self.assertIn(32, cardinalities)
        self.assertIn(255, cardinalities)
        self.assertEqual(by_family["batch_grid_255"]["choice_candidates"], 255)
        self.assertEqual(by_family["wide_dispatch_32"]["choice_candidates"], 32)

    def test_every_item_validates(self):
        for item in self.items:
            self.assertEqual(v4.validate_item(item), [], item["item_id"])

    def test_every_trainer_row_validates(self):
        for item in self.items:
            row = v4.trainer_row(item)
            targets = validate_training_row(row)
            self.assertIn(item["qid"], targets)

    def test_chance_items_have_no_hard_gold(self):
        chance = [item for item in self.items
                  if item["expected"]["distribution_kind"] == v4.CHANCE_KIND]
        self.assertEqual({item["family"] for item in chance},
                         {"weighted_queue_lottery", "traffic_split_lottery"})
        nondegenerate = 0
        for item in chance:
            self.assertIsNone(item["expected"]["gold"])
            self.assertIsNone(item["expected"]["gold_index"])
            dist = item["expected"]["distribution"]
            self.assertAlmostEqual(sum(dist.values()), 1.0, places=6)
            if any(0.0 < p < 1.0 for p in dist.values()):
                nondegenerate += 1
            row = v4.trainer_row(item)
            self.assertEqual(row["gold"], {})
            self.assertEqual(row["gold_probs_kind"][item["qid"]],
                             "programmatic_conditional_distribution")
            self.assertEqual(row["gold_label_kind"][item["qid"]], "unobserved")
            t = validate_training_row(row)[item["qid"]]
            self.assertIsNone(t["gold_index"])
            self.assertIsNotNone(t["gold_distribution_probs"])
        # a handful of degenerate lotteries (e.g. all weight on one candidate)
        # are legitimate; the cohort must be mostly non-degenerate
        self.assertGreater(nondegenerate / len(chance), 0.8)

    def test_deterministic_items_are_one_hot(self):
        for item in self.items:
            if item["expected"]["distribution_kind"] == v4.CHANCE_KIND:
                continue
            dist = item["expected"]["distribution"]
            self.assertEqual(sum(p == 1.0 for p in dist.values()), 1)
            self.assertEqual(len(dist), len(item["expected"]["candidate_keys"]))

    def test_requests_pass_served_validator(self):
        for item in self.items:
            validate_request(item["request"])

    def test_split_isolation_at_component_level(self):
        seen = {}
        for item in self.items:
            fp = visible_fingerprint(item)
            if fp in seen:
                self.assertEqual(seen[fp], item["split"],
                                 f"canonical input crosses splits: {item['item_id']}")
            else:
                seen[fp] = item["split"]
        by_group = {}
        for item in self.items:
            gid = item["source_group_id"]
            by_group.setdefault(gid, item["split"])
            self.assertEqual(by_group[gid], item["split"])

    def test_pairs_flip_declared(self):
        pairs = {}
        for item in self.items:
            pairs.setdefault(item["pair_id"], []).append(item)
        self.assertEqual(len(pairs), 720)
        for pair_id, members in pairs.items():
            self.assertEqual(len(members), 6)
            self.assertEqual(len({m["split"] for m in members}), 1)
            self.assertEqual(len({m["source_group_id"] for m in members}), 1)
            bases = {m["question_type"]: m for m in members if m["member"] == "base"}
            variants = {m["question_type"]: m for m in members if m["member"] == "variant"}
            for qtype, base in bases.items():
                variant = variants[qtype]
                flipped = (v4._answer_payload(base) != v4._answer_payload(variant))
                self.assertEqual(flipped, base["contrastive"]["is_flip_question"],
                                 f"{pair_id}:{qtype}")
            self.assertTrue(any(m["contrastive"]["is_flip_question"]
                                for m in members if m["member"] == "base"))

    def test_gold_recomputes_from_fact_basis(self):
        for item in self.items:
            qid, gold = v4.gold_for(item["family"],
                                    item["provenance"]["fact_basis"],
                                    item["question_type"])
            self.assertEqual(qid, item["qid"])
            if item["expected"]["distribution_kind"] == v4.CHANCE_KIND:
                self.assertEqual(gold, item["expected"]["distribution"])
            else:
                self.assertEqual(gold, item["expected"]["gold"])

    def test_tamper_detection(self):
        manifest = json.loads(json.dumps(self.manifest))
        item = next(i for i in manifest["items"]
                    if i["expected"]["distribution_kind"] == "hard_label"
                    and i["question_type"] == "score"
                    and len(i["expected"]["candidate_keys"]) > 2)
        item["expected"]["gold"] = 0 if item["expected"]["gold"] != 0 else 1
        errors = v4.validate_manifest(manifest)
        self.assertTrue(any("gold" in e or "differ" in e or "re-derivable" in e
                            for e in errors), errors)

    def test_t9c_calibration_minimums(self):
        rows = [v4.trainer_row(item) for item in self.items
                if item["split"] == "calibration"]
        stats = v4.t9c_group_stats(rows)
        estimable = [k for k, v in stats.items() if v["estimable"]]
        self.assertGreaterEqual(len(estimable), v4.T9C_MIN_GROUPS)
        for key in estimable:
            self.assertGreaterEqual(stats[key]["rows"], v4.T9C_MIN_ROWS)
            self.assertGreaterEqual(stats[key]["states"], v4.T9C_MIN_STATES)

    def test_zero_canonical_overlap_with_prior_cohorts(self):
        v4_fps = {visible_fingerprint(item) for item in self.items}
        v3_manifest = json.loads((V3_DIR / "manifest.json").read_text())
        v3_fps = {v3._visible_fingerprint(item) for item in v3_manifest["items"]}
        self.assertFalse(v4_fps & v3_fps)
        for heldout in (HELDOUT_V1, HELDOUT_V2):
            if not heldout.exists():
                continue
            heldout_fps = set()
            for row in jsonl(heldout):
                question = next(iter(row["questions"].values()))
                heldout_fps.add(digest(visible_input(row, question)))
            self.assertFalse(v4_fps & heldout_fps)

    def test_split_proportions(self):
        counts = self.manifest["counts_by_split"]
        total = sum(counts.values())
        self.assertAlmostEqual(counts["train"] / total, 0.55, places=2)
        self.assertAlmostEqual(counts["dev"] / total, 0.175, places=2)
        self.assertAlmostEqual(counts["calibration"] / total, 0.10, places=2)
        self.assertAlmostEqual(counts["test"] / total, 0.175, places=2)


if __name__ == "__main__":
    unittest.main()
