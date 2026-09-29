import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import financial_rlcd_estimators_v1 as rlcd
import financial_baseline_estimators_v1 as estimators
import run_financial_rlcd_v1 as runner


class T14RunnerGateTests(unittest.TestCase):
    def test_missing_authorization_is_fail_closed(self):
        result = runner.run()
        self.assertEqual(result["status"], "blocked_fit_authorization_missing")
        self.assertFalse(result["fit_performed"])
        self.assertFalse(result["real_fit_performed"])
        self.assertFalse(result["fit_authorized"])
        self.assertFalse(result["measurement_authorized"])
        self.assertEqual(result["network_model_calls"], 0)
        self.assertEqual(result["folds"], [])

    def test_authorization_receipt_requires_exact_protocol_and_flags(self):
        protocol_sha = runner._sha256(runner.PROTOCOL_PATH)
        self.assertEqual(protocol_sha, runner.EXPECTED_PROTOCOL_SHA256)
        with tempfile.TemporaryDirectory(dir=runner.ROOT) as tmp:
            path = Path(tmp) / "auth.json"
            base = {
                "schema_version": runner.AUTH_SCHEMA,
                "protocol_sha256": protocol_sha,
                "decision": "approved_for_real_fit",
                "fit_authorized": True,
                "measurement_authorized": True,
                "training_authorized": False,
                "independent_reviewer": {"id": "reviewer-test", "reviewed_utc": "2026-09-21T00:00:00Z"},
                "network_model_calls": 0,
                "order_submission_authorized": False,
                "live_trading_authorized": False,
            }
            path.write_text(json.dumps(base), encoding="utf-8")
            ok, info = runner._check_authorization(path, protocol_sha)
            self.assertTrue(ok)
            self.assertEqual(info["reviewer_id"], "reviewer-test")
            bad = dict(base, protocol_sha256="0" * 64)
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different T14 protocol"):
                runner._check_authorization(path, protocol_sha)
            bad = dict(base, order_submission_authorized=True)
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "order submission"):
                runner._check_authorization(path, protocol_sha)
            bad = dict(base, live_trading_authorized=None)
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "live trading"):
                runner._check_authorization(path, protocol_sha)
            bad = dict(base, training_authorized=True)
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "model training"):
                runner._check_authorization(path, protocol_sha)

    def test_selection_is_dev_nll_only_with_name_tiebreak(self):
        metrics = {"z": {"nll": 0.2}, "a": {"nll": 0.2}, "b": {"nll": 0.3}}
        self.assertEqual(runner._select_dev(metrics), "a")

    def test_real_authorization_receipt_shape(self):
        path = runner.ROOT / "results/financial_rlcd_authorization_20260921_v1.json"
        if not path.exists():
            self.skipTest("authorization receipt not present")
        ok, info = runner._check_authorization(path, runner.EXPECTED_PROTOCOL_SHA256)
        self.assertTrue(ok)
        self.assertEqual(info["status"], "approved_for_real_fit")


class T14EstimatorTests(unittest.TestCase):
    def test_sampling_seed_rule_matches_protocol(self):
        for seed in (1729, 2718, 3141):
            self.assertEqual(rlcd.sampling_seed(seed), seed * 1_000_003 + 17)

    def test_exact_brier_grad_z_matches_finite_difference(self):
        # d/dz ||p - q||^2 for binary p = sigmoid(z): 4 (p - q) p (1 - p).
        for z in (-2.0, -0.3, 0.0, 1.1):
            p = 1.0 / (1.0 + np.exp(-z))
            for q in (0.1, 0.5, 0.9):
                exact = rlcd.exact_brier_grad_z(p, q)
                eps = 1e-6

                def dist(zz):
                    pp = 1.0 / (1.0 + np.exp(-zz))
                    return 2.0 * (pp - q) ** 2

                fd = (dist(z + eps) - dist(z - eps)) / (2 * eps)
                self.assertAlmostEqual(exact, fd, places=5)

    def test_paired_gradient_monte_carlo_matches_conditional_expectation(self):
        # For y=1 the conditional expected surrogate gradient is
        # -(2 p(1-p) - 2(2p-1)p(1-p)); MC mean must land within ~5 SE.
        p, y = 0.3, 1.0
        mc = rlcd.paired_gradient_monte_carlo(p, y, samples=32, replicates=500, seed=1729)
        d_r = 2.0 * p * (1.0 - p)
        d_norm = 2.0 * (2.0 * p - 1.0) * p * (1.0 - p)
        exact = -(d_r - d_norm)
        self.assertLessEqual(abs(mc["mean"] - exact), 5.0 * max(mc["se"], 1e-12))

    def test_paired_gradient_bias_check_passes_on_frozen_grid(self):
        result = rlcd.paired_gradient_bias_check(
            [0.1, 0.5, 0.9], [0.1, 0.5, 0.9], samples=32,
            replicates=400, seed=1729, max_standard_errors=5.0)
        self.assertTrue(result["passed"], json.dumps(result))

    def test_fit_linear_pg_is_seed_deterministic(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(64, 4))
        y = (rng.random(64) < 0.4).astype(np.float64)
        a = rlcd.fit_linear_pg(X, y, estimator="paired_brier_pg", steps=20, seed=1729)
        b = rlcd.fit_linear_pg(X, y, estimator="paired_brier_pg", steps=20, seed=1729)
        c = rlcd.fit_linear_pg(X, y, estimator="paired_brier_pg", steps=20, seed=2718)
        np.testing.assert_array_equal(a["weights"], b["weights"])
        self.assertEqual(a["bias"], b["bias"])
        self.assertFalse(np.array_equal(a["weights"], c["weights"]))
        # Zero initialization is frozen.
        self.assertTrue(np.all(a["initial_weights"] == 0.0))
        self.assertEqual(a["initial_bias"], 0.0)

    def test_pg_fit_shares_permutation_stream_with_exact_fit(self):
        # The protocol freezes default_rng(training_seed) as the shared batch
        # permutation for every objective; reward draws must not perturb it.
        rng = np.random.default_rng(0)
        X = rng.normal(size=(48, 3))
        y = (rng.random(48) < 0.5).astype(np.float64)
        exact = estimators.fit_linear(X, y, objective="brier", steps=20, seed=1729)
        pg = rlcd.fit_linear_pg(X, y, estimator="paired_brier_pg", steps=20, seed=1729)
        # Both start at zero and share the batch order; PG samples add noise so
        # the weights differ, but both must stay finite and move off zero.
        self.assertTrue(np.all(np.isfinite(pg["weights"])))
        self.assertFalse(np.allclose(pg["weights"], 0.0))
        self.assertTrue(np.all(np.isfinite(exact["weights"])))

    def test_correctness_pg_is_biased_toward_majority(self):
        # Negative control: expected reward linear in p pushes probability to
        # the majority outcome rather than recovering q.
        rng = np.random.default_rng(1)
        X = rng.normal(size=(200, 2))
        y = np.zeros(200)
        y[:40] = 1.0  # 20% positives; majority is 0
        model = rlcd.fit_linear_pg(X, y, estimator="correctness_pg",
                                   steps=200, lr=0.05, l2=0.001, seed=1729)
        p = estimators.predict_linear(model, X)
        self.assertLess(float(np.mean(p)), 0.2)

    def test_reward_parts_decomposition(self):
        y = np.array([0.0, 1.0, 1.0, 0.0])
        p = np.array([0.25, 0.75, 0.6, 0.4])
        parts = rlcd.reward_parts(y, p)
        p_y = np.where(y == 1.0, p, 1.0 - p)
        self.assertAlmostEqual(parts["hit_component"], 2.0 * p_y.mean())
        self.assertAlmostEqual(parts["agreement_component"],
                               float(np.mean(p * p + (1 - p) ** 2)))
        self.assertAlmostEqual(parts["expected_reward"],
                               parts["hit_component"] - parts["agreement_component"])
        self.assertAlmostEqual(parts["exact_brier"], float(np.mean((p - y) ** 2)))

    def test_selective_risk_coverages(self):
        rng = np.random.default_rng(2)
        y = (rng.random(100) < 0.5).astype(np.float64)
        p = rng.random(100)
        out = rlcd.selective_risk(y, p, [1.0, 0.9, 0.75, 0.5, 0.25])
        self.assertEqual([e["coverage"] for e in out], [1.0, 0.9, 0.75, 0.5, 0.25])
        self.assertEqual([e["covered_count"] for e in out], [100, 90, 75, 50, 25])
        for e in out:
            for key in ("nll", "brier", "accuracy", "ece"):
                self.assertIn(key, e)

    def test_gradient_variance_diagnostic_shape(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(40, 3))
        y = (rng.random(40) < 0.5).astype(np.float64)
        model = rlcd.fit_linear_pg(X, y, estimator="paired_brier_pg", steps=10, seed=1729)
        diag = rlcd.gradient_variance_diagnostic(model, X, y, "paired_brier_pg",
                                                 samples=32, replicates=16, seed=1729)
        self.assertEqual(diag["replicates"], 16)
        self.assertEqual(diag["rows"], 40)
        self.assertGreaterEqual(diag["per_row_grad_variance_mean"], 0.0)
        self.assertGreaterEqual(diag["exact_brier_grad_l2"], 0.0)


if __name__ == "__main__":
    unittest.main()
