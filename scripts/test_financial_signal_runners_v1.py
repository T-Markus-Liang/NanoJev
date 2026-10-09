"""Unit tests for the T63-T72 signal measurement runners.

Synthetic fixtures only — deterministic, no cohort/network access. Each runner
is validated on a tiny constructed cohort where the expected direction is known.
"""
import json
import math
import tempfile
import unittest
from pathlib import Path

import financial_signal_hypotheses_v1 as v1
import financial_signal_hypotheses_v3 as v3
import financial_signal_labels_v4 as v4
import financial_signal_grid_v1 as grid
import financial_signal_robustness_v1 as rob


def _rec(asset, i, mark, funding, basis=0.0, vol=None, rv=None):
    """Minimal PIT record matching the cohort schema."""
    ns = 1_674_691_200_000_000_000 + i * 86_400_000_000_000
    vol = vol if vol is not None else 1_000_000.0
    rv = rv if rv is not None else 0.02
    f = lambda v: {"value": v, "available_ns": ns, "event_ns": ns}
    return {"asset_id": asset, "venue": "binance_um", "decision_ns": ns,
            "id": f"test:{asset}:{i}", "schema_version": "test",
            "universe_available_ns": ns,
            "features": {"mark_price": f(mark), "index_price": f(mark),
                         "mark_index_basis_bps": f(basis),
                         "last_funding_rate": f(funding),
                         "funding_interval_hours": f(8.0),
                         "quote_volume": f(vol), "trade_count": f(1000),
                         "taker_buy_ratio": f(0.5), "realized_vol_24bar": f(rv)},
            "label": {"outcome": mark > 0, "event": "test",
                      "end_ns": ns, "available_ns": ns}}


def _write_cohort(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def _run_with(runner_mod, records, tmp):
    cohort = tmp / "records.jsonl"
    _write_cohort(cohort, records)
    folds = tmp / "folds.json"
    folds.write_text(json.dumps({"folds": [
        {"train": [0, 1], "dev": [0, 1], "calibration": [0, 1],
         "test": [0, 10 ** 30]}]}))
    old_c, old_f = runner_mod.COHORT, getattr(runner_mod, "FOLDS", None)
    runner_mod.COHORT = cohort
    if old_f is not None:
        runner_mod.FOLDS = folds
    try:
        return runner_mod.run()
    finally:
        runner_mod.COHORT = old_c
        if old_f is not None:
            runner_mod.FOLDS = old_f


def _make_asset_records(asset, n=260, funding_fn=None, mark_fn=None):
    funding_fn = funding_fn or (lambda i: 0.0001)
    mark_fn = mark_fn or (lambda i: 100 * (1 + 0.001 * i))
    return [_rec(asset, i, mark_fn(i), funding_fn(i)) for i in range(n)]


class HelpersTest(unittest.TestCase):
    def test_wilson_ci_bounds(self):
        lo, hi = v1.wilson_ci(50, 100)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)

    def test_two_prop_z_sign(self):
        r = v1.two_prop_z(80, 100, 40, 100)
        self.assertGreater(r["z"], 0)
        self.assertLess(r["p"], 0.01)

    def test_spearman_perfect(self):
        r = v4.spearman([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        self.assertAlmostEqual(r["rho"], 1.0, places=3)

    def test_spearman_independent(self):
        r = v4.spearman(list(range(50)), [i % 7 for i in range(50)])
        self.assertIsNotNone(r)


class RunnerIntegrationTest(unittest.TestCase):
    def test_v1_runner_schema_and_arms(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            recs = _make_asset_records("BTCUSDT-PERP")
            out = _run_with(v1, recs, tmp)
        self.assertEqual(out["schema_version"],
                         "nanojev-financial-signal-hypotheses-v1")
        for arm in ("h1_tail", "h2_fade_long", "h3_basis_extreme"):
            self.assertIn(arm, out["results"]["arms"])

    def test_v3_runner_schema(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            recs = _make_asset_records(
                "ETHUSDT-PERP",
                funding_fn=lambda i: ((i // 20) % 10) * 1e-4)
            out = _run_with(v3, recs, tmp)
        self.assertEqual(out["schema_version"],
                         "nanojev-financial-signal-hypotheses-v3")
        self.assertIn("h6_follow_high", out["results"]["arms"])

    def test_labels_runner_detects_planted_signal(self):
        """Funding in 30-day blocks; daily return proportional to same-block
        funding -> labels runner must see a positive l3:funding rank corr."""
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            fund = [((i // 20) % 10) * 1e-4 for i in range(260)]
            price, prices = 100.0, []
            for i in range(260):
                prices.append(price)
                price *= 1 + fund[i] * 20
            recs = [_rec("BTCUSDT-PERP", i, prices[i], fund[i])
                    for i in range(260)]
            out = _run_with(v4, recs, tmp)
        cell = out["results"]["cells"]["l3:funding"]
        self.assertIsNotNone(cell["spearman"])
        self.assertGreater(cell["spearman"]["rho"], 0.3)

    def test_grid_runner_schema(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            recs = _make_asset_records("SOLUSDT-PERP")
            out = _run_with(grid, recs, tmp)
        self.assertEqual(out["schema_version"],
                         "nanojev-financial-signal-grid-v1")
        self.assertIn("f0_v0", out["results"]["grid_mean_bps"])

    def test_robustness_runner_schema(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            recs = _make_asset_records("XRPUSDT-PERP")
            out = _run_with(rob, recs, tmp)
        self.assertEqual(out["schema_version"],
                         "nanojev-financial-signal-robustness-v1")
        self.assertIn("hold_5d", out["results"]["spearmans"])


if __name__ == "__main__":
    unittest.main()
