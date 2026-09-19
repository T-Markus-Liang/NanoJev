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
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from build_perp_pit_v1 import (  # noqa: E402
    DEFINITION_SHA256, EVENT_NAME, PILOT_FEATURES, build_records, load_funding, load_klines,
)
from fetch_venue_perp_v1 import bybit_series, ms  # noqa: E402
from financial_pit_v1 import validate_dataset, validate_record  # noqa: E402
from paper_trade_perp_v1 import (  # noqa: E402
    REGIME_BASIS_BLOWOUT, REGIME_FUNDING_EXTREME, REGIME_LIQUIDITY_LOW, REGIME_NORMAL,
    REGIME_VOL_HIGH, build_regime_windows, collapse_windows, day_window_ms, find_divergences,
    quantile,
)
from paper_trade_protocol_v1 import (  # noqa: E402
    ProtocolError, input_manifest_sha256, load_protocol, sha256_file,
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
class B0ProtocolTest(unittest.TestCase):
    """T1 (B0-0): the frozen protocol must be verifiable and must refuse to be ignored."""

    PROTOCOL = ROOT / "research" / "paper_trade_b0_protocol.json"

    def test_frozen_protocol_loads_and_reports_its_digest(self):
        protocol, evidence = load_protocol(self.PROTOCOL)
        self.assertEqual(evidence["protocol_sha256"], sha256_file(self.PROTOCOL))
        self.assertEqual(evidence["protocol_run_id"], protocol["run_id"])
        # Every parameter the run depends on must be present and frozen.
        for name in ("fast", "slow"):
            self.assertIn(name, protocol["strategy"])
        for name in ("seed", "sizing", "capacity", "on_divergence", "participation_fraction",
                     "reference_notional_volume"):
            self.assertIn(name, protocol["policy"])

    def test_digest_mismatch_is_rejected(self):
        with self.assertRaises(ProtocolError):
            load_protocol(self.PROTOCOL, expected="0" * 64)

    def test_matching_digest_is_accepted(self):
        _, evidence = load_protocol(self.PROTOCOL, expected=sha256_file(self.PROTOCOL))
        self.assertTrue(evidence["protocol_verified_against_expected"])

    def test_unknown_schema_and_missing_fields_are_rejected(self):
        original = json.loads(self.PROTOCOL.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            for mutate in ("schema", "missing"):
                candidate = json.loads(json.dumps(original))
                if mutate == "schema":
                    candidate["schema_version"] = "some-other-schema"
                else:
                    del candidate["policy"]["seed"]
                path = pathlib.Path(tmp) / f"{mutate}.json"
                path.write_text(json.dumps(candidate))
                with self.assertRaises(ProtocolError, msg=mutate):
                    load_protocol(path)

    def test_input_manifest_digest_is_computed_from_the_declared_path(self):
        protocol, _ = load_protocol(self.PROTOCOL)
        manifest = input_manifest_sha256(protocol, "binance", ROOT)
        declared = ROOT / protocol["input_manifests"]["binance"]
        if not declared.exists():
            self.skipTest("binance fetch manifest not present")
        self.assertEqual(manifest["sha256"], sha256_file(declared))

    def test_unknown_source_has_no_manifest(self):
        protocol, _ = load_protocol(self.PROTOCOL)
        with self.assertRaises(ProtocolError):
            input_manifest_sha256(protocol, "no-such-venue", ROOT)


class FillDivergenceTest(unittest.TestCase):
    """T2 (B0-A): a fill that differs from the request must be reported, never assumed."""

    @staticmethod
    def _order(status, requested, filled):
        return {"decision_id": "d1", "asset_id": "BTCUSDT-PERP",
                "requested_quantity": requested, "filled_quantity": filled,
                "status_history": [{"status": "submitted"}, {"status": status}]}

    def test_fully_filled_order_is_not_a_divergence(self):
        result = {"orders": [self._order("filled", 10.0, 10.0)]}
        self.assertEqual(find_divergences(result), [])

    def test_rejected_order_is_a_divergence(self):
        result = {"orders": [self._order("rejected", 10.0, 0.0)]}
        divergences = find_divergences(result)
        self.assertEqual(len(divergences), 1)
        self.assertEqual(divergences[0]["terminal_status"], "rejected")
        self.assertEqual(divergences[0]["filled_quantity"], 0.0)

    def test_partial_fill_is_a_divergence_even_when_status_is_filled(self):
        result = {"orders": [self._order("filled", 10.0, 4.0)]}
        self.assertEqual(len(find_divergences(result)), 1)

    def test_expired_order_is_a_divergence(self):
        result = {"orders": [self._order("expired", 10.0, 3.0)]}
        self.assertEqual(len(find_divergences(result)), 1)

    def test_empty_result_has_no_divergences(self):
        self.assertEqual(find_divergences({}), [])


class DayWindowTest(unittest.TestCase):
    """Regression test for the window defect: the receipt must not claim a sample it did not use."""

    def test_window_is_half_open_and_covers_the_last_day(self):
        start, end = day_window_ms("2024-01-01", "2026-08-31")
        import datetime as dt
        self.assertEqual(dt.datetime.fromtimestamp(start / 1000, dt.timezone.utc).date(),
                         dt.date(2024, 1, 1))
        # The last day must be INCLUDED, so the exclusive end is the next midnight.
        self.assertEqual(dt.datetime.fromtimestamp(end / 1000, dt.timezone.utc).date(),
                         dt.date(2026, 9, 1))

    def test_bar_open_times_are_selected_by_the_window(self):
        import datetime as dt
        start, end = day_window_ms("2024-01-01", "2024-01-03")
        def ms(y, m, d):
            return int(dt.datetime(y, m, d, tzinfo=dt.timezone.utc).timestamp() * 1000)
        inside = [ms(2024, 1, 1), ms(2024, 1, 2), ms(2024, 1, 3)]
        outside = [ms(2023, 12, 31), ms(2024, 1, 4)]
        for stamp in inside:
            self.assertTrue(start <= stamp < end, stamp)
        for stamp in outside:
            self.assertFalse(start <= stamp < end, stamp)

    def test_reversed_window_is_rejected(self):
        with self.assertRaises(ValueError):
            day_window_ms("2025-01-01", "2024-01-01")

    def test_driver_applies_the_window_to_the_loaded_days(self):
        # The defect was that the window was recorded but never applied.
        source = (ROOT / "scripts" / "paper_trade_perp_v1.py").read_text()
        self.assertIn("day_window_ms(args.first_day, args.last_day)", source)
        self.assertIn("window_start_ms <= day < window_end_ms", source)


class SizingIndependenceTest(unittest.TestCase):
    """T3 (B0-B): sizing must depend only on initial cash and the decision-day close."""

    def test_sizing_formula_uses_initial_cash_not_equity(self):
        source = (ROOT / "scripts" / "paper_trade_perp_v1.py").read_text()
        # The quantity must be built from initial_cash; an equity term would make the size
        # depend on the PnL path and destroy measurability.
        self.assertIn("args.initial_cash", source)
        for forbidden in ("ledger.get(\"equity\")", "equity /", "* equity"):
            self.assertNotIn(forbidden, source)


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
