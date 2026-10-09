#!/usr/bin/env python3
"""Tests for the X4 engineering heldout v2 builder.

Rebuilds the cohort deterministically, re-checks every gold against the frozen
family oracles, runs the real served/trainer validators, and verifies scale
(>=300 rows, >=50 source groups), option-count strata and zero canonical-input
overlap with heldout_v1 and the v4 corpus.
"""

import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_engineering_heldout_v2 as h2  # noqa: E402
from audit_engineering_corpus_v1 import digest, visible_input  # noqa: E402
from predict_toy_decisions import validate_request  # noqa: E402
from train_pipeline_decisions import validate_training_row  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
HELDOUT_DIR = ROOT / "research" / "engineering_heldout_v2"
HELDOUT_V1 = ROOT / "research" / "engineering_heldout_v1" / "items.jsonl"
CORPUS_V4_TRAINER = (ROOT / "research" / "engineering_judgment_corpus_v4"
                     / "trainer_view")


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def fingerprints(path_or_dir):
    if Path(path_or_dir).is_dir():
        rows = [row for split in ("train", "dev", "calibration", "test")
                for row in jsonl(Path(path_or_dir) / f"{split}.jsonl")]
    else:
        rows = jsonl(path_or_dir)
    fps = set()
    for row in rows:
        for question in row["questions"].values():
            fps.add(digest(visible_input(row, question)))
    return fps


class HeldoutV2BuildTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rows = h2.build_rows()
        cls.errors = h2.validate_rows(cls.rows)

    def test_self_test_passes(self):
        self.assertEqual(self.errors, [])

    def test_scale_and_group_targets(self):
        self.assertGreaterEqual(len(self.rows), 300)
        groups = {r["metadata"]["source_group_id"] for r in self.rows}
        self.assertGreaterEqual(len(groups), 50)
        states = {r["state_id"] for r in self.rows}
        self.assertEqual(len(states), len(groups))
        self.assertEqual(len(self.rows), 3 * len(states))

    def test_all_rows_validate_and_gold_is_argmax(self):
        for row in self.rows:
            targets = validate_training_row(row)
            validate_request({"states": [{k: row[k]
                                          for k in ("id", "state", "questions")}]})
            for qid in row["questions"]:
                t = targets[qid]
                self.assertIsNotNone(t["gold_index"])
                self.assertEqual(t["gold_distribution_probs"][t["gold_index"]], 1.0)

    def test_gold_recomputes_from_fact_basis(self):
        for row in self.rows:
            spec = h2.FAMILY_MAP[row["family_id"].removeprefix("eh2_")]
            qid = next(iter(row["questions"]))
            qtype = row["metadata"]["question_type"]
            recomputed = h2._oracle_checked(spec, row["metadata"]["fact_basis"])
            self.assertEqual(recomputed[qtype], row["gold"][qid])

    def test_all_rows_are_test_split_and_disjoint_ids(self):
        ids = [r["id"] for r in self.rows]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(r["split"] == "test" for r in self.rows))
        self.assertTrue(all(r["id"].startswith("eh2-") for r in self.rows))
        self.assertTrue(all(r["family_id"].startswith("eh2_") for r in self.rows))

    def test_option_count_strata(self):
        strata = sorted({len(q["criteria"]) for r in self.rows
                         for q in r["questions"].values()
                         if q["type"] in ("choice", "score")})
        self.assertTrue(set(strata) >= {3, 4, 5, 6, 8, 12}, strata)

    def test_question_types_all_present(self):
        qtypes = {r["metadata"]["question_type"] for r in self.rows}
        self.assertEqual(qtypes, {"boolean", "choice", "score"})

    def test_zero_overlap_with_heldout_v1_and_corpus_v4(self):
        ours = fingerprints(HELDOUT_DIR / "items.jsonl")
        self.assertFalse(ours & fingerprints(HELDOUT_V1))
        self.assertFalse(ours & fingerprints(CORPUS_V4_TRAINER))

    def test_emitted_file_is_deterministic(self):
        on_disk = (HELDOUT_DIR / "items.jsonl").read_text(encoding="utf-8")
        self.assertEqual(on_disk, h2._jsonl(self.rows))
        manifest = json.loads((HELDOUT_DIR / "manifest.json").read_text())
        self.assertEqual(manifest["schema_version"],
                         "nanojev-engineering-heldout-v2-manifest-v1")
        self.assertEqual(manifest["row_count"], len(self.rows))
        self.assertFalse(manifest["training_authorized"])

    def test_no_reserved_tokens_or_forbidden_sources(self):
        import build_engineering_corpus_v1 as v1
        for row in self.rows:
            self.assertEqual(v1.reserved_hits(
                json.dumps(row, ensure_ascii=False, sort_keys=True)), [])
            self.assertIsNone(v1.forbidden_path_reason(
                row["metadata"]["provenance_source_id"]))


if __name__ == "__main__":
    unittest.main()
