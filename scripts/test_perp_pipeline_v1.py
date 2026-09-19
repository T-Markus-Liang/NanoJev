#!/usr/bin/env python3
"""Tests for the real-data perp pipeline (fetchers, PIT build, paper trading).

The data-driven tests read the local ignored ``data/`` directory and SKIP when the archive
has not been fetched, so the suite stays runnable on a clean checkout. Tests that need no
downloaded data always run.
"""
import json
import math
import pathlib
import statistics
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from build_perp_pit_v1 import (  # noqa: E402
    DEFINITION_SHA256, EVENT_NAME, PILOT_FEATURES, build_records, load_funding, load_klines,
)
from fetch_venue_perp_v1 import bybit_series, ms  # noqa: E402
from financial_pit_v1 import validate_dataset, validate_record  # noqa: E402
from paper_trade_perp_v1 import (  # noqa: E402
    REGIME_BASIS_BLOWOUT, REGIME_FUNDING_EXTREME, REGIME_LIQUIDITY_LOW, REGIME_NORMAL,
    REGIME_VOL_HIGH, build_regime_windows, collapse_windows, quantile,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / "data" / "binance_vision_v1"
HAVE_ARCHIVE = (ARCHIVE / "markPriceKlines" / "BTCUSDT").is_dir()


class QuantileTest(unittest.TestCase):
    def test_nearest_rank_and_bounds(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertEqual(quantile(values, 0.0), 1.0)
        self.assertEqual(quantile(values, 1.0), 5.0)
        self.assertIn(quantile(values, 0.5), values)

    def test_empty_returns_none_and_no_interpolation(self):
        self.assertIsNone(quantile([], 0.5))
        # Nearest-rank returns an OBSERVED value, never an interpolated one.
        values = [1.0, 10.0]
        self.assertIn(quantile(values, 0.75), values)


class RegimeCausalityTest(unittest.TestCase):
    """A regime label must never depend on a later observation."""

    def _rows(self, count=200):
        rows = []
        for index in range(count):
            rows.append((index * 1_000_000_000, 0.01 + 0.001 * (index % 7),
                         math.log(1000.0 + index), 1e-5 * (index % 5), 2.0 * (index % 3)))
        return rows

    def test_labels_are_identical_when_future_is_appended(self):
        rows = self._rows()
        prefix = build_regime_windows(rows[:120])
        extended = build_regime_windows(rows)
        self.assertEqual(prefix, extended[:120])

    def test_no_stress_label_before_minimum_history(self):
        rows = self._rows(120)
        labels = [label for _, label in build_regime_windows(rows, min_history=48)]
        self.assertTrue(all(label == REGIME_NORMAL for label in labels[:48]))

    def test_stress_labels_fire_when_a_statistic_is_extreme(self):
        rows = []
        for index in range(120):
            vol = 0.01 if index < 100 else 5.0          # a late volatility spike
            rows.append((index * 1_000_000_000, vol, math.log(1000.0), 1e-6, 0.5))
        labels = [label for _, label in build_regime_windows(rows, min_history=48)]
        self.assertIn(REGIME_VOL_HIGH, labels[100:])
        self.assertEqual(labels[0], REGIME_NORMAL)

    def test_priority_order_is_deterministic(self):
        # Normal history first, then a window where EVERY stress condition is extreme, so
        # the highest-priority label must be the one that appears. (A perfectly constant
        # series would fire nothing: its expanding quantile equals the value itself, and the
        # comparison is strict. That is intended, not a bug.)
        rows = []
        for index in range(120):
            if index < 100:
                rows.append((index * 1_000_000_000, 0.01, math.log(1000.0), 1e-6, 0.5))
            else:
                rows.append((index * 1_000_000_000, 9.0, math.log(0.001), 9.0, 900.0))
        labels = [label for _, label in build_regime_windows(rows, min_history=48)]
        self.assertEqual(labels[0], REGIME_NORMAL)
        # While every condition fires, the highest-priority one wins.
        self.assertEqual(labels[100], REGIME_BASIS_BLOWOUT)
        # As the extreme basis observations enter the expanding window they stop being
        # "extreme" against their own history, and the next-priority condition takes over.
        # This transition is the mechanism working, so both labels must appear.
        self.assertIn(REGIME_BASIS_BLOWOUT, labels)
        self.assertIn(REGIME_VOL_HIGH, labels)
        self.assertEqual(set(labels) - {REGIME_NORMAL, REGIME_BASIS_BLOWOUT, REGIME_VOL_HIGH},
                         set())


class CollapseWindowsTest(unittest.TestCase):
    def test_consecutive_same_labels_merge_and_windows_are_disjoint(self):
        buckets = [(0, REGIME_NORMAL), (10, REGIME_NORMAL), (20, REGIME_VOL_HIGH),
                   (30, REGIME_VOL_HIGH), (40, REGIME_NORMAL)]
        windows = collapse_windows(buckets)
        self.assertEqual([w.label for w in windows],
                         [REGIME_NORMAL, REGIME_VOL_HIGH, REGIME_NORMAL])
        for left, right in zip(windows, windows[1:]):
            self.assertEqual(left.end_ns, right.start_ns)


class DefinitionHashTest(unittest.TestCase):
    def test_recorded_definition_hash_reproduces(self):
        """The cohort's label hash must equal the protocol's own recomputation."""
        protocol = json.loads(
            (ROOT / "research" / "financial_experiment_protocol_v1.json").read_text())
        convention = protocol["label_predicate"]["definition_hash_convention"]
        import hashlib
        recomputed = hashlib.sha256(json.dumps(
            convention["frozen_definition_object"], sort_keys=True,
            separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()
        self.assertEqual(recomputed, DEFINITION_SHA256)
        self.assertEqual(EVENT_NAME, convention["frozen_definition_object"]["event"])


class PipeliningUnitTest(unittest.TestCase):
    """Fetch-layer helpers, exercised without network access."""

    def test_month_parsing_is_utc(self):
        import datetime as dt
        self.assertEqual(ms("2024-01-01"),
                         int(dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc).timestamp() * 1000))

    def test_bybit_backward_paging_does_not_loop_forever(self):
        """A stub that always returns one row must terminate, not spin."""
        calls = []

        def fake_get(path):
            calls.append(path)
            return {"list": [["1704067200000", "1", "2", "0.5", "1.5", "10", "15"]]}

        import fetch_venue_perp_v1 as module
        original = module.bybit_get
        module.bybit_get = fake_get
        try:
            rows = bybit_series("BTCUSDT", "kline", 1_600_000_000_000, 1_700_000_000_000)
        finally:
            module.bybit_get = original
        self.assertEqual(len(rows), 1)
        self.assertLessEqual(len(calls), 3)   # bounded, no infinite pagination


@unittest.skipUnless(HAVE_ARCHIVE, "local Binance archive not fetched")
class RealArchiveTest(unittest.TestCase):
    def test_loader_returns_daily_bars_with_required_fields(self):
        marks = load_klines(ARCHIVE, "markPriceKlines", "BTCUSDT")
        self.assertGreater(len(marks), 1000)
        sample = next(iter(marks.values()))
        for field in ("open", "high", "low", "close", "close_time", "volume"):
            self.assertIn(field, sample)
        self.assertGreater(sample["close"], 0)

    def test_funding_rows_are_sorted_and_typed(self):
        rows = load_funding(ARCHIVE, "BTCUSDT")
        self.assertGreater(len(rows), 100)
        self.assertEqual(rows, sorted(rows, key=lambda row: row["calc_time"]))
        self.assertIn(rows[0]["interval_hours"], (1, 2, 4, 8))

    def test_cohort_satisfies_the_frozen_pit_contract(self):
        """Every built record must pass the UNMODIFIED validator, as one frozen cohort."""
        records, _ = build_records(ARCHIVE, "BTCUSDT", "2024-01-01", "2024-03-31")
        self.assertGreater(len(records), 60)
        for record in records:
            validate_record(record)            # raises on any contract violation
        validate_dataset(records)              # frozen feature set + one definition
        self.assertEqual({name for name in records[0]["features"]}, set(PILOT_FEATURES))

    def test_label_predicate_matches_the_declared_formula(self):
        records, _ = build_records(ARCHIVE, "BTCUSDT", "2024-01-01", "2024-03-31")
        marks = load_klines(ARCHIVE, "markPriceKlines", "BTCUSDT")
        by_time = {bar["close_time"]: bar["close"] for bar in marks.values()}
        checked = 0
        for record in records:
            entry = record["features"]["mark_price"]["value"]
            # end_ns = decision_ns + horizon; decision_ns = close_time_ms * 1e6 + 1, so the
            # integer division recovers the EXIT bar's close_time in milliseconds.
            exit_close = by_time.get(record["label"]["end_ns"] // 1_000_000)
            if exit_close is None:
                continue
            expected = (10000.0 * ((exit_close / entry) - 1.0)) >= 25.0
            self.assertEqual(record["label"]["outcome"], expected)
            checked += 1
        self.assertGreater(checked, 60)

    def test_feature_availability_never_exceeds_the_decision(self):
        records, _ = build_records(ARCHIVE, "BTCUSDT", "2024-01-01", "2024-03-31")
        for record in records:
            for feature in record["features"].values():
                self.assertLessEqual(feature["event_ns"], record["decision_ns"])
                self.assertLessEqual(feature["available_ns"], record["decision_ns"])
                self.assertLessEqual(feature["fit_cutoff_ns"], feature["available_ns"])
            label = record["label"]
            self.assertGreater(label["end_ns"], record["decision_ns"])
            self.assertEqual(label["end_ns"] - record["decision_ns"], 86_400_000_000_000)
            self.assertEqual(label["available_ns"] - label["end_ns"], 604_800_000_000_000)


if __name__ == "__main__":
    unittest.main()
