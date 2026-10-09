import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import run_financial_baselines_v1 as runner


class T11RunnerTests(unittest.TestCase):
    def test_missing_authorization_is_fail_closed(self):
        result = runner.run()
        self.assertEqual(result["status"], "blocked_fit_authorization_missing")
        self.assertFalse(result["fit_performed"])
        self.assertFalse(result["training_authorized"])
        self.assertFalse(result["measurement_authorized"])
        self.assertEqual(result["network_model_calls"], 0)
        self.assertEqual(result["folds"], [])

    def test_authorization_receipt_requires_exact_protocol_and_reviewer(self):
        protocol_sha = runner._sha256(runner.PROTOCOL_PATH)
        with tempfile.TemporaryDirectory(dir=runner.ROOT) as tmp:
            path = Path(tmp) / "auth.json"
            base = {
                "schema_version": runner.AUTH_SCHEMA,
                "protocol_sha256": protocol_sha,
                "decision": "approved_for_real_fit",
                "fit_authorized": True,
                "measurement_authorized": True,
                "independent_reviewer": {"id": "reviewer-test", "reviewed_utc": "2026-09-20T00:00:00Z"},
                "network_model_calls": 0,
            }
            path.write_text(json.dumps(base), encoding="utf-8")
            ok, info = runner._check_authorization(path, protocol_sha)
            self.assertTrue(ok)
            self.assertEqual(info["reviewer_id"], "reviewer-test")
            base["protocol_sha256"] = "0" * 64
            path.write_text(json.dumps(base), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different T11 protocol"):
                runner._check_authorization(path, protocol_sha)

    def test_selection_is_dev_only_and_deterministic(self):
        metrics = {
            "z": {"nll": 0.2},
            "a": {"nll": 0.2},
            "b": {"nll": 0.3},
        }
        self.assertEqual(runner._select_dev(metrics), "a")

    def test_rule_candidates_are_probability_bounded(self):
        X = np.array([[-1.0, 0.0], [1.0, 0.0]])
        y = np.array([0.0, 1.0])
        for name in runner.json.loads(runner.PROTOCOL_PATH.read_text())["candidates"]["deterministic"]:
            p = runner._rule_predictions(name, X, y, 1729)
            self.assertTrue(np.all((p >= 0.0) & (p <= 1.0)))


if __name__ == "__main__":
    unittest.main()
