import json
import tempfile
import unittest
from pathlib import Path

import audit_engineering_corpus_v2 as audit


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "research/engineering_judgment_corpus_v2"


class EngineeringCorpusV2AuditTests(unittest.TestCase):
    def test_internal_isolation_passes_but_fresh_holdout_is_required(self):
        report = audit.audit_corpus(CORPUS, compare_seed=20260920)
        self.assertEqual(report["status"], "preflight_passed_not_training_authorized")
        self.assertEqual(report["block_reasons"], [])
        self.assertEqual(report["audit"]["canonical_input_cross_split_groups"], 0)
        self.assertEqual(report["audit"]["evaluation_provenance_hits"], [])
        self.assertIn("seed_change_does_not_prove_semantic_holdout", report["holdout_block_reasons"])
        self.assertFalse(report["training_authorized"])

    def test_hashes_are_stable(self):
        first = audit.audit_corpus(CORPUS, compare_seed=20260920)
        second = audit.audit_corpus(CORPUS, compare_seed=20260920)
        self.assertEqual(first["source_hashes"], second["source_hashes"])
        self.assertEqual(first["source_hashes"], first["source_hashes_after"])

    def test_same_seed_is_rejected(self):
        with self.assertRaises(ValueError):
            audit.audit_corpus(CORPUS, compare_seed=20260919)

    def test_authorization_is_fail_closed(self):
        report = audit.audit_corpus(CORPUS, compare_seed=20260920)
        for key in ("training_authorized", "measurement_authorized", "deployment_authorized"):
            self.assertFalse(report[key])
        self.assertEqual(report["merged_rows_written"], 0)


if __name__ == "__main__":
    unittest.main()
