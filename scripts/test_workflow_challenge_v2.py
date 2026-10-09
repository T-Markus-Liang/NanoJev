#!/usr/bin/env python3
from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
import unittest

from build_workflow_challenge_v2 import VARIANTS, build, variant
from build_workflow_decisions import smart_home
from train_pipeline_decisions import read_training_records


class WorkflowChallengeTest(unittest.TestCase):
    def row(self, split="test"):
        return smart_home(split, 0, ("kitchen", "heater", "on", True, 0, 12), random.Random(17))

    def test_variants_preserve_source_gold_score_order_and_input(self):
        row = self.row()
        before = deepcopy(row)
        for name in VARIANTS:
            output = variant(row, name)
            self.assertEqual(output["gold"], row["gold"])
            self.assertEqual(output["gold_probs"], row["gold_probs"])
            self.assertEqual(output["state_id"], row["state_id"])
            self.assertEqual(output["metadata"]["source_group_id"], row["metadata"]["source_group_id"])
            self.assertEqual(output["questions"]["risk"]["criteria"], row["questions"]["risk"]["criteria"])
            self.assertNotEqual(output["id"], row["id"])
        self.assertEqual(before, row)

    def test_reversal_and_distractor_are_real_changes(self):
        row = self.row()
        reversed_row = variant(row, "choice_order_reversed")
        self.assertEqual(list(reversed_row["questions"]["device"]["criteria"]),
                         list(reversed(row["questions"]["device"]["criteria"])))
        self.assertIn(row["state"], variant(row, "archived_distractor")["state"])
        self.assertEqual(variant(row, "structured_envelope")["state"]["current_record"], row["state"])

    def test_training_calibration_unknown_and_missing_source_rejected(self):
        for split in ("train", "dev", "calibration"):
            with self.assertRaisesRegex(ValueError, "test/OOD"):
                variant(self.row(split), "original")
        with self.assertRaisesRegex(ValueError, "unknown"):
            variant(self.row(), "typo")
        row = self.row()
        row["metadata"].pop("source_group_id")
        with self.assertRaisesRegex(ValueError, "source group"):
            variant(row, "original")

    def test_reproducible_build_and_refusal_to_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            for split in ("test", "ood"):
                row = self.row(split)
                row["metadata"]["source_group_id"] += split
                (source / f"{split}.jsonl").write_text(json.dumps(row) + "\n")
            manifest = build(source, root / "first")
            second = build(source, root / "second")
            self.assertEqual(manifest, second)
            rows, _ = read_training_records(root / "first")
            self.assertEqual(len(rows), 8)
            self.assertEqual(manifest["splits"]["test"]["questions"], 12)
            self.assertEqual(manifest["splits"]["test"]["source_groups"], 1)
            rebuilt = next(row for row in rows if row["split"] == "test" and row["metadata"]["challenge_variant"] == "choice_order_reversed")
            self.assertEqual(list(rebuilt["questions"]["device"]["criteria"]),
                             list(reversed(self.row()["questions"]["device"]["criteria"])))
            with self.assertRaisesRegex(ValueError, "empty"):
                build(source, root / "first")


if __name__ == "__main__":
    unittest.main()
