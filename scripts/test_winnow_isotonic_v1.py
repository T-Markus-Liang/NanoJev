#!/usr/bin/env python3
"""T168 isotonic bundle sanity tests: PAVA correctness/monotonicity on
synthetic data, global fallback for unseen domains, and receipt checks on
the emitted bundle (monotone maps, eval ECE actually improved, eval rows
measurement-only). Pure numpy — no scorer or torch needed."""
import json
import unittest
from pathlib import Path

import numpy as np

from winnow_isotonic_v1 import (
    BUNDLE_OUT, apply_map, build_bundle, domain_of, fit_pava, map_knots,
    predict)

ROOT = Path(__file__).resolve().parent.parent


class PavaTest(unittest.TestCase):
    def test_monotone_nondecreasing_on_noise(self):
        rng = np.random.default_rng(7)
        x = rng.uniform(0, 1, 500)
        y = (rng.uniform(0, 1, 500) < x ** 2).astype(float)  # noisy, monotone
        kx, ky = fit_pava(x, y)
        self.assertTrue(np.all(np.diff(kx) > 0))
        self.assertTrue(np.all(np.diff(ky) >= -1e-12))
        self.assertTrue(np.all(ky >= 0.0) and np.all(ky <= 1.0))

    def test_recovers_step(self):
        x = np.concatenate([np.full(50, 0.1), np.full(50, 0.9)])
        y = np.concatenate([np.zeros(50), np.ones(50)])
        kx, ky = fit_pava(x, y)
        self.assertAlmostEqual(float(apply_map((kx, ky), [0.05])[0]), 0.0)
        self.assertAlmostEqual(float(apply_map((kx, ky), [0.95])[0]), 1.0)
        mid = float(apply_map((kx, ky), [0.5])[0])
        self.assertTrue(0.0 <= mid <= 1.0)

    def test_duplicate_x_collapses(self):
        kx, ky = fit_pava([0.5, 0.5, 0.5, 0.9], [0.0, 1.0, 0.0, 1.0])
        self.assertTrue(np.all(np.diff(kx) > 0))
        self.assertAlmostEqual(float(ky[0]), 1.0 / 3.0)


class PredictTest(unittest.TestCase):
    def test_unknown_domain_falls_back_to_global(self):
        rows = [{"group_id": "brand_new_domain:abc", "noul_prob": 0.4}]
        maps = {"global": {"x": [0.0, 1.0], "y": [0.1, 0.9]},
                "per_domain": {"seen": {"x": [0.0, 1.0], "y": [0.5, 0.5]}}}
        out = predict(rows, maps)
        self.assertAlmostEqual(float(out[0]), 0.42)  # global: 0.1+0.4*0.8
        rows[0]["group_id"] = "seen:abc"
        self.assertAlmostEqual(float(predict(rows, maps)[0]), 0.5)

    def test_domain_of(self):
        self.assertEqual(domain_of({"group_id": "src_v1:deadbeef"}), "src_v1")
        self.assertEqual(domain_of({}), "")


@unittest.skipUnless(BUNDLE_OUT.exists(), "bundle not built yet")
class BundleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = json.loads(BUNDLE_OUT.read_text(encoding="utf-8"))

    def test_schema_and_provenance(self):
        b = self.bundle
        self.assertEqual(b["schema_version"],
                         "nanojev-winnow-isotonic-bundle-v1")
        self.assertTrue(b["advisory_only"])
        self.assertEqual(b["data"]["eval_rows"]["usage"],
                         "measurement only; never fitted")
        self.assertEqual(len(b["data"]["fit_rows"]["sha256"]), 64)
        self.assertEqual(len(b["data"]["eval_rows"]["sha256"]), 64)

    def test_all_maps_monotone(self):
        maps = [self.bundle["maps"]["global"],
                *self.bundle["maps"]["per_domain"].values()]
        for m in maps:
            x, y = np.asarray(m["x"]), np.asarray(m["y"])
            self.assertTrue(np.all(np.diff(x) > 0))
            self.assertTrue(np.all(np.diff(y) >= -1e-9))
            self.assertTrue(np.all(y >= 0.0) and np.all(y <= 1.0))

    def test_eval_ece_improved(self):
        ev = self.bundle["metrics"]["eval"]
        self.assertLess(ev["isotonic_global"]["ece"], ev["raw"]["ece"])
        self.assertLess(ev["isotonic_per_domain"]["ece"], ev["raw"]["ece"])
        # holdout must improve too — an artifact that only helps eval would
        # smell like leakage
        ho = self.bundle["metrics"]["cal_holdout"]
        self.assertLess(ho["isotonic_global"]["ece"], ho["raw"]["ece"])

    def test_fit_rows_actually_calfit_only(self):
        """Re-derive the map from cal-fit rows and confirm the bundle's
        global knots match — proves the artifact came from fit data only."""
        b = build_bundle()
        self.assertEqual(b["maps"]["global"], self.bundle["maps"]["global"])
        self.assertEqual(b["maps"]["per_domain"].keys(),
                         self.bundle["maps"]["per_domain"].keys())


if __name__ == "__main__":
    unittest.main()
