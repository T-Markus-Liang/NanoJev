#!/usr/bin/env python3
"""T5 report regressions.

Isolated Worker tests were imported and adapted to the final real receipt.
Most build_report tests mock ONLY loader/hash I/O and clone receipts into
synthetic cells; those are validation tests, not economic simulations.
RealMatrixIntegration below separately checks all 36 actual receipts/hashes
and, when local ignored data exists, recomputes the report without mocks.
"""
import copy
import datetime as dt
import hashlib
import json
import pathlib
import shutil
import sys
import tempfile
import unittest
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import paper_trade_report_v1 as report  # noqa: E402
from paper_trade_attribution_v1 import attribute  # noqa: E402
from paper_trade_protocol_v1 import load_protocol  # noqa: E402

FIXTURE_PATH = HERE.parent / "results/paper_trade_t5_v1_run2/receipts/binance-full-20260919.json"
BASE_PROTOCOL_PATH = HERE.parent / "research/paper_trade_b0_protocol.json"
FULL_WINDOW = ["2024-01-01", "2026-08-31"]


# --------------------------------------------------------------------------- #
# Fixtures / helpers (no absolute outside-workdir path is ever read)
# --------------------------------------------------------------------------- #
def _fake_sha(path):
    """Deterministic 64-hex stand-in for hashing; no file is read."""
    return hashlib.sha256(str(pathlib.Path(path)).encode("utf-8")).hexdigest()


def _study(**overrides):
    study = {
        "schema_version": report.STUDY_SCHEMA,
        "base_protocol": "research/paper_trade_b0_protocol.json",
        "seeds": [20260919],
        "venues": ["binance"],
        "appendix_only_venues": [],
        "windows": {"full": list(FULL_WINDOW)},
        "headline_spread_threshold_fraction_of_initial": 0.05,
    }
    study.update(overrides)
    return study


def _days(window):
    first, last = map(dt.date.fromisoformat, window)
    return (last - first).days + 1


class ReportCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base, _ = load_protocol(BASE_PROTOCOL_PATH)
        cls.fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="t5-report-test-")
        self.addCleanup(shutil.rmtree, self._tmp, True)

    # -- receipt construction -------------------------------------------------
    def make_receipt(self, study, venue, seed, window):
        """Clone the real fixture receipt into a declared matrix cell."""
        receipt = copy.deepcopy(self.fixture)
        days = study["windows"][window]
        receipt["data"]["source_key"] = venue
        receipt["data"]["first_day"], receipt["data"]["last_day"] = days
        span = _days(days)
        for info in receipt["data"]["per_symbol"].values():
            info["quotes"] = span
            info["observed_first_day"] = days[0]
            info["observed_last_day"] = days[1]
        receipt["data"]["quote_count"] = sum(
            info["quotes"] for info in receipt["data"]["per_symbol"].values())
        receipt["reproducibility"]["seed"] = seed
        # Fake, self-describing protocol path; the loader is mocked (see label).
        receipt["reproducibility"]["protocol_path"] = (
            f"/test-protocols/{venue}-{seed}-{window}.json")
        manifest = self.base["input_manifests"][venue]
        receipt["reproducibility"]["input_manifest"] = {
            "path": manifest, "sha256": _fake_sha(report.ROOT / manifest)}
        return receipt

    def write_receipts(self, study, cells=None, mutate=None):
        cells = cells or [
            (v, s, w)
            for v in study["venues"]
            for s in study["seeds"]
            for w in study["windows"]
        ]
        paths = []
        for index, (venue, seed, window) in enumerate(cells):
            receipt = self.make_receipt(study, venue, seed, window)
            if mutate is not None:
                mutate(receipt, venue, seed, window)
            path = pathlib.Path(self._tmp) / (
                f"{index:03d}-{venue}-{seed}-{window}.json")
            path.write_text(json.dumps(receipt), encoding="utf-8")
            paths.append(path)
        return paths

    # -- build_report with mocked I/O only ------------------------------------
    def protocol_stub(self, study):
        base = self.base

        def side(path, expected=None):
            stem = pathlib.Path(path).name[: -len(".json")]
            parts = stem.split("-")
            window, seed = parts[-1], int(parts[-2])
            derived = report.derived_protocol(
                base, window, study["windows"][window], seed)
            return derived, {"protocol_sha256": expected or "0" * 64}

        return side

    def run_build(self, study, paths, protocol_side=None, base_digest="base-digest"):
        side = protocol_side if protocol_side is not None else self.protocol_stub(study)
        with mock.patch.object(report, "load_protocol", side_effect=side), mock.patch.object(
                report, "sha256_file", side_effect=_fake_sha):
            return report.build_report(paths, study, self.base, base_digest)


# --------------------------------------------------------------------------- #
# summary(): finite / NaN / inf / boolean rejection
# --------------------------------------------------------------------------- #
class TestSummary(unittest.TestCase):
    def test_finite_values_summarised(self):
        stats = report.summary([3, 1, 2])
        self.assertEqual(stats, {"n": 3, "min": 1.0, "median": 2.0,
                                 "max": 3.0, "spread": 2.0})

    def test_empty_returns_none(self):
        self.assertIsNone(report.summary([]))

    def test_nonfinite_and_boolean_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf"), True, False):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    report.summary([1.0, value])
                with self.assertRaises(ValueError):
                    report.summary([value])


# --------------------------------------------------------------------------- #
# sensitivity(): axes are independent, threshold is strictly "exceed"
# --------------------------------------------------------------------------- #
class TestSensitivity(unittest.TestCase):
    @staticmethod
    def _rows(venues, seeds, windows, net):
        rows = []
        for window in windows:
            for seed in seeds:
                for venue in venues:
                    value = net(venue, seed, window)
                    rows.append({"venue": venue, "seed": seed, "window": window,
                                 "net_pnl": value,
                                 "net_pnl_fraction_of_initial": value})
        return rows

    def test_axes_are_independent_and_never_pooled(self):
        venues, seeds, windows = ["a", "b", "c"], [1, 2], ["w1", "w2"]
        rows = self._rows(venues, seeds, windows, lambda v, s, w: 1.0)
        result = report.sensitivity(rows, threshold=1.0)

        # venue varies while (seed, window) is held fixed -> 2*2 groups of 3.
        venue_axis = result["venue"]
        self.assertEqual(len(venue_axis), len(seeds) * len(windows))
        for entry in venue_axis:
            self.assertEqual(set(entry["fixed"]), {"seed", "window"})
            self.assertNotIn("venue", entry["fixed"])
            self.assertEqual(sorted(entry["varied"]), venues)
            self.assertEqual(entry["net_pnl"]["n"], len(venues))
        # seed varies while (venue, window) is held fixed -> 3*2 groups of 2.
        seed_axis = result["seed"]
        self.assertEqual(len(seed_axis), len(venues) * len(windows))
        for entry in seed_axis:
            self.assertEqual(set(entry["fixed"]), {"venue", "window"})
            self.assertEqual(sorted(entry["varied"]), seeds)
        # window varies while (venue, seed) is held fixed -> 3*2 groups of 2.
        window_axis = result["window"]
        self.assertEqual(len(window_axis), len(venues) * len(seeds))
        for entry in window_axis:
            self.assertEqual(set(entry["fixed"]), {"venue", "seed"})
            self.assertEqual(sorted(entry["varied"]), windows)

        # A venue-axis group must contain only the three venue PnLs, not all 12.
        for entry in venue_axis:
            self.assertEqual(entry["net_pnl"]["n"], 3)
            self.assertLess(entry["net_pnl"]["n"], len(rows))

    def test_threshold_equality_is_not_exceeded(self):
        rows = self._rows(["a", "b"], [1], ["w"],
                          lambda v, s, w: 0.25 if v == "a" else 0.5)
        at = report.sensitivity(rows, threshold=0.25)["venue"][0]
        self.assertEqual(at["fraction_of_initial"]["spread"], 0.25)
        self.assertFalse(at["spread_exceeds_threshold"])
        above = report.sensitivity(rows, threshold=0.2)["venue"][0]
        self.assertTrue(above["spread_exceeds_threshold"])
        below = report.sensitivity(rows, threshold=0.5)["venue"][0]
        self.assertFalse(below["spread_exceeds_threshold"])


# --------------------------------------------------------------------------- #
# load_study(): duplicate/malformed seeds+windows, Aster exclusion
# --------------------------------------------------------------------------- #
class TestLoadStudy(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="t5-study-test-")
        self.addCleanup(shutil.rmtree, self._tmp, True)

    def load(self, study):
        path = pathlib.Path(self._tmp) / "study.json"
        path.write_text(json.dumps(study), encoding="utf-8")
        return report.load_study(path)

    def test_valid_study_round_trips(self):
        study = _study()
        self.assertEqual(self.load(study), study)

    def test_duplicate_seeds_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate axis: seeds"):
            self.load(_study(seeds=[1, 1]))

    def test_duplicate_venues_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate axis: venues"):
            self.load(_study(venues=["binance", "binance"]))

    def test_malformed_seed_types_rejected(self):
        for seed in (True, 1.5, "20260919", None):
            with self.subTest(seed=seed):
                with self.assertRaisesRegex(ValueError, "seeds must be integers"):
                    self.load(_study(seeds=[seed]))

    def test_reversed_window_rejected(self):
        with self.assertRaisesRegex(ValueError, "reversed window"):
            self.load(_study(windows={"w": ["2024-12-31", "2024-01-01"]}))

    def test_duplicate_window_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate window"):
            self.load(_study(windows={
                "a": ["2024-01-01", "2024-12-31"],
                "b": ["2024-01-01", "2024-12-31"]}))

    def test_empty_window_axis_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty window axis"):
            self.load(_study(windows={}))

    def test_aster_must_stay_appendix_only(self):
        with self.assertRaisesRegex(ValueError, "Aster must remain appendix-only"):
            self.load(_study(venues=["binance", "aster"], appendix_only_venues=[]))
        # The declared appendix-only form is accepted.
        good = _study(venues=["binance", "aster"], appendix_only_venues=["aster"])
        self.assertEqual(self.load(good), good)

    def test_unknown_appendix_source_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown appendix source"):
            self.load(_study(venues=["binance"], appendix_only_venues=["kraken"]))

    def test_unsupported_venue_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported source"):
            self.load(_study(venues=["kraken"]))

    def test_bad_schema_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported sensitivity study"):
            self.load(_study(schema_version="something-else"))

    def test_bad_threshold_rejected(self):
        with self.assertRaises(ValueError):
            self.load(_study(headline_spread_threshold_fraction_of_initial=-0.01))
        with self.assertRaises(ValueError):
            self.load(_study(headline_spread_threshold_fraction_of_initial=float("nan")))


# --------------------------------------------------------------------------- #
# build_report(): happy path on the REAL fixture (loader/hash mocked -- label)
# --------------------------------------------------------------------------- #
class TestBuildReportRealFixture(ReportCase):
    """TEST LIMITATION: the real fixture's protocol/manifest paths are not staged,
    so ``load_protocol`` and ``sha256_file`` are mocked here (and below)."""

    def test_fixture_records_outside_absolute_protocol_path(self):
        # Documents why load_protocol MUST be mocked: the receipt points at an
        # absolute path outside this work directory (no filesystem probe here).
        protocol_path = pathlib.Path(self.fixture["reproducibility"]["protocol_path"])
        self.assertTrue(protocol_path.is_absolute())
        self.assertFalse(str(protocol_path).startswith(str(HERE)))
        manifest_path = pathlib.Path(
            self.fixture["reproducibility"]["input_manifest"]["path"])
        self.assertFalse(manifest_path.is_absolute())

    def test_single_cell_report_from_real_fixture(self):
        study = _study()
        paths = self.write_receipts(
            study, cells=[("binance", 20260919, "full")])
        built = self.run_build(study, paths)

        self.assertEqual(built["schema_version"], "nanojev-paper-trade-b0-report-v1")
        self.assertEqual(built["matrix_cells"], 1)
        self.assertEqual(len(built["primary"]["matrix"]), 1)
        self.assertEqual(built["appendix_licence_conflict"]["matrix"], [])
        row = built["primary"]["matrix"][0]
        self.assertEqual((row["venue"], row["seed"], row["window"]),
                         ("binance", 20260919, "full"))
        self.assertTrue(row["coverage_complete"])
        self.assertEqual(row["duration_days"], _days(FULL_WINDOW))
        self.assertLess(built["max_abs_attribution_residual"], 1e-6)
        self.assertAlmostEqual(
            row["net_pnl_fraction_of_initial"],
            row["net_pnl"] / self.base["policy"]["initial_cash"], places=12)
        self.assertEqual(row["net_pnl_fraction_of_initial"],
                         self.fixture["net_pnl_fraction_of_initial"])

    def test_publication_is_refused_for_provisional_settings(self):
        study = _study(venues=["binance", "bybit"])
        paths = self.write_receipts(study)
        built = self.run_build(study, paths)

        self.assertEqual(built["matrix_cells"], 2)
        self.assertFalse(built["headline_allowed"])
        reasons = set(built["headline_refusal_reasons"])
        self.assertIn(
            "provisional_costs_contracts_and_fill_funding_conventions_pending_R1", reasons)
        self.assertIn("current_snapshot_not_asof_vintage", reasons)
        self.assertIn("reference_strategy_not_model_evidence", reasons)
        self.assertIn("fewer_than_three_non_appendix_venues", reasons)
        self.assertIn("bybit_terms_unverified", reasons)

    def test_aster_excluded_from_primary_and_headline(self):
        study = _study(venues=["binance", "bybit", "aster"],
                       appendix_only_venues=["aster"])
        paths = self.write_receipts(study)
        built = self.run_build(study, paths)

        self.assertEqual(built["matrix_cells"], 3)
        self.assertEqual({r["venue"] for r in built["primary"]["matrix"]},
                         {"binance", "bybit"})
        self.assertEqual({r["venue"] for r in built["appendix_licence_conflict"]["matrix"]},
                         {"aster"})
        self.assertIn("Aster excluded",
                      built["appendix_licence_conflict"]["notice"])
        # Design observation (reported as a risk, not fixed): with only three
        # supported venues and Aster forced appendix-only, primary can never
        # hold 3 venues, so "fewer_than_three_non_appendix_venues" is
        # unsatisfiable and is always present.
        self.assertIn("fewer_than_three_non_appendix_venues",
                      built["headline_refusal_reasons"])


# --------------------------------------------------------------------------- #
# build_report(): fail-closed matrix / receipt validation
# --------------------------------------------------------------------------- #
class TestBuildReportMatrix(ReportCase):
    def test_complete_matrix_positive_control(self):
        study = _study(venues=["binance", "bybit"], seeds=[20260919, 20260920])
        built = self.run_build(study, self.write_receipts(study))
        self.assertEqual(built["matrix_cells"], 4)

    def test_missing_cell_fails_closed(self):
        study = _study(venues=["binance", "bybit"])
        paths = self.write_receipts(study, cells=[("binance", 20260919, "full")])
        with self.assertRaisesRegex(ValueError, "incomplete sensitivity matrix"):
            self.run_build(study, paths)

    def test_duplicate_cell_fails_closed(self):
        study = _study()
        paths = self.write_receipts(study, cells=[
            ("binance", 20260919, "full"), ("binance", 20260919, "full")])
        with self.assertRaisesRegex(ValueError, "duplicate or unexpected cell"):
            self.run_build(study, paths)

    def test_unexpected_cell_fails_closed(self):
        study = _study(venues=["binance"])
        paths = self.write_receipts(study, cells=[("bybit", 20260919, "full")])
        with self.assertRaisesRegex(ValueError, "duplicate or unexpected cell"):
            self.run_build(study, paths)

    def test_undeclared_window_fails_closed(self):
        study = _study()

        def mutate(receipt, venue, seed, window):
            receipt["data"]["first_day"] = "2023-01-01"

        paths = self.write_receipts(study, mutate=mutate)
        with self.assertRaisesRegex(ValueError, "undeclared receipt window"):
            self.run_build(study, paths)


class TestBuildReportReceiptValidation(ReportCase):
    def build_mutated(self, mutate, study=None, cells=None):
        study = study or _study()
        paths = self.write_receipts(study, cells=cells, mutate=mutate)
        return self.run_build(study, paths)

    def test_old_receipt_revision_rejected(self):
        def mutate(receipt, *rest):
            receipt["receipt_revision"] = "legacy-v0"

        with self.assertRaisesRegex(ValueError, "old receipts lack"):
            self.build_mutated(mutate)

    def test_invalid_receipt_seed_rejected(self):
        for bad in (True, "20260919", 20260919.0, None):
            with self.subTest(seed=bad):
                def mutate(receipt, *rest, bad=bad):
                    receipt["reproducibility"]["seed"] = bad

                with self.assertRaisesRegex(ValueError, "invalid receipt seed"):
                    self.build_mutated(mutate)

    def test_nonmatching_protocol_rejected(self):
        study = _study()
        paths = self.write_receipts(study)
        unchanged = lambda path, expected=None: (  # noqa: E731
            copy.deepcopy(self.base),
            {"protocol_sha256": expected or "0" * 64})
        with self.assertRaisesRegex(ValueError, "not a declared base-only"):
            self.run_build(study, paths, protocol_side=unchanged)

    def test_execution_policy_drift_rejected(self):
        study = _study(seeds=[20260919, 20260920])

        def mutate(receipt, venue, seed, window):
            if seed == 20260920:
                receipt["execution_policy"]["fees"]["fee_bps"] = 7.5

        with self.assertRaisesRegex(ValueError, "runtime (policy|contract)"):
            self.build_mutated(mutate, study=study)

    def test_contract_drift_rejected(self):
        study = _study(seeds=[20260919, 20260920])

        def mutate(receipt, venue, seed, window):
            if seed == 20260920:
                receipt["contracts"] = copy.deepcopy(receipt["contracts"])
                receipt["contracts"][0]["tick_size"] = 99.0

        with self.assertRaisesRegex(ValueError, "runtime (policy|contract)"):
            self.build_mutated(mutate, study=study)

    def test_attribution_mismatch_rejected(self):
        def mutate(receipt, *rest):
            receipt["attribution"]["sum"] = receipt["attribution"]["sum"] + 1.0

        with self.assertRaisesRegex(ValueError, "stored attribution differs"):
            self.build_mutated(mutate)

    def test_attribution_is_recomputed_from_evidence(self):
        # Sanity: the real fixture's stored attribution must equal a recompute.
        recomputed = attribute(self.fixture["ledger"], self.fixture["accounting_evidence"])
        self.assertEqual(recomputed, self.fixture["attribution"])

    def test_input_manifest_path_mismatch_rejected(self):
        def mutate(receipt, *rest):
            receipt["reproducibility"]["input_manifest"]["path"] = "data/other.json"

        with self.assertRaisesRegex(ValueError, "does not match source"):
            self.build_mutated(mutate)

    def test_input_manifest_hash_mismatch_rejected(self):
        def mutate(receipt, *rest):
            receipt["reproducibility"]["input_manifest"]["sha256"] = "0" * 64

        with self.assertRaisesRegex(ValueError, "input manifest changed"):
            self.build_mutated(mutate)

    def test_source_manifest_differs_across_cells_rejected(self):
        study = _study(seeds=[20260919, 20260920])

        def mutate(receipt, venue, seed, window):
            if seed == 20260920:
                receipt["reproducibility"]["input_manifest"]["note"] = "extra"

        with self.assertRaisesRegex(ValueError, "source manifest differs across cells"):
            self.build_mutated(mutate, study=study)

    def test_conservation_flag_rejected(self):
        def mutate(receipt, *rest):
            receipt["conservation"] = False

        with self.assertRaisesRegex(ValueError, "failed conservation"):
            self.build_mutated(mutate)

    def test_missing_replay_digest_rejected(self):
        def mutate(receipt, *rest):
            receipt["replay_sha256"] = "short"

        with self.assertRaisesRegex(ValueError, "missing replay digest"):
            self.build_mutated(mutate)

    def test_noncomparable_strategy_rejected(self):
        def mutate(receipt, *rest):
            receipt["data"]["symbols"] = ["BTCUSDT"]

        with self.assertRaisesRegex(ValueError, "noncomparable"):
            self.build_mutated(mutate)

    def test_initial_capital_mismatch_rejected(self):
        def mutate(receipt, *rest):
            receipt["ledger"]["initial_cash"] = 50000.0

        with self.assertRaisesRegex(ValueError, "initial capital mismatch"):
            self.build_mutated(mutate)

    def test_seed_degeneracy_declaration_rejected(self):
        def mutate(receipt, *rest):
            receipt["execution_policy"]["fills"]["fill_probability"] = 0.5

        with self.assertRaisesRegex(ValueError, "seed degeneracy"):
            self.build_mutated(mutate)

    def test_quote_count_reconciliation_rejected(self):
        def mutate(receipt, *rest):
            receipt["data"]["quote_count"] += 1

        with self.assertRaisesRegex(ValueError, "quote counts do not reconcile"):
            self.build_mutated(mutate)

    def test_decision_count_reconciliation_rejected(self):
        def mutate(receipt, *rest):
            receipt["counts"]["decisions"] += 1

        with self.assertRaisesRegex(ValueError, "decision counts do not reconcile"):
            self.build_mutated(mutate)


class TestBuildReportCoverage(ReportCase):
    """Coverage gaps must fail closed (refuse headline), not crash."""

    def _mutated_report(self, mutate):
        study = _study()
        paths = self.write_receipts(study, mutate=mutate)
        return self.run_build(study, paths)

    def test_short_quote_coverage_refuses_headline(self):
        def mutate(receipt, *rest):
            per_symbol = receipt["data"]["per_symbol"]
            first = next(iter(per_symbol))
            per_symbol[first]["quotes"] -= 1
            receipt["data"]["quote_count"] = sum(
                v["quotes"] for v in per_symbol.values())

        built = self._mutated_report(mutate)
        row = built["primary"]["matrix"][0]
        self.assertFalse(row["coverage_complete"])
        self.assertIn("incomplete_window_coverage", built["headline_refusal_reasons"])
        self.assertFalse(built["headline_allowed"])

    def test_observed_day_mismatch_refuses_headline(self):
        def mutate(receipt, *rest):
            first = next(iter(receipt["data"]["per_symbol"]))
            receipt["data"]["per_symbol"][first]["observed_last_day"] = "2026-08-30"

        built = self._mutated_report(mutate)
        self.assertFalse(built["primary"]["matrix"][0]["coverage_complete"])
        self.assertIn("incomplete_window_coverage", built["headline_refusal_reasons"])

    def test_decision_divergence_surfaces_in_reasons(self):
        def mutate(receipt, *rest):
            receipt["divergence_count"] = 1
            order = receipt["accounting_evidence"]["orders"][0]
            order["requested_quantity"] += 1
            order["remaining_quantity"] += 1
            order["status"] = "expired"
            receipt["attribution"] = attribute(receipt["ledger"], receipt["accounting_evidence"])

        built = self._mutated_report(mutate)
        self.assertIn("order_outcome_divergences", built["headline_refusal_reasons"])
        self.assertEqual(built["primary"]["matrix"][0]["divergence_count"], 1)

    def test_fake_divergence_count_rejected(self):
        with self.assertRaisesRegex(ValueError, "counts disagree"):
            self._mutated_report(lambda receipt, *rest: receipt.update(divergence_count=1))


class RealMatrixIntegration(unittest.TestCase):
    def setUp(self):
        self.root = HERE.parent
        self.report = json.loads((self.root / "results/paper_trade_b0_report_v1.json").read_text())
        self.paths = sorted((self.root / "results/paper_trade_t5_v1_run2/receipts").glob("*.json"))

    def test_all_real_cells_independently_reconcile(self):
        self.assertEqual(len(self.paths), 36)
        for path in self.paths:
            receipt = json.loads(path.read_text())
            actual = attribute(receipt["ledger"], receipt["accounting_evidence"])
            self.assertEqual(receipt["attribution"], actual)
            self.assertLessEqual(actual["max_abs_residual"], 1e-6)

    def test_full_window_preserves_corrected_baseline_economics_and_replay(self):
        for venue in ("binance", "bybit", "aster"):
            old = json.loads((self.root / f"results/paper_trade_b0_baseline_{venue}_v1.json").read_text())
            new = json.loads((self.root / f"results/paper_trade_t5_v1_run2/receipts/{venue}-full-20260919.json").read_text())
            self.assertEqual(old["ledger"], new["ledger"])
            self.assertEqual(old["replay_sha256"], new["replay_sha256"])
            self.assertEqual(old["counts"], new["counts"])

    def test_actual_headline_refusal_and_seeds(self):
        self.assertFalse(self.report["headline_allowed"])
        self.assertIn("declared_spread_threshold_exceeded", self.report["headline_refusal_reasons"])
        self.assertTrue(self.report["seed_axis_degenerate"])
        self.assertTrue(all(e["net_pnl"]["spread"] == 0 for e in self.report["primary"]["sensitivity"]["seed"]))
        self.assertEqual(len(self.report["primary"]["matrix"]), 24)
        self.assertEqual(len(self.report["appendix_licence_conflict"]["matrix"]), 12)

    def test_missing_day_and_aster_source_are_explicit(self):
        for path in self.paths:
            receipt = json.loads(path.read_text())
            data = receipt["data"]
            if data["source_key"] == "aster":
                self.assertEqual(data["venue"], "aster_perp")
                self.assertEqual(data["simulation_template_venue"], "binance_um")
            expected = ["2026-06-29"] if data["source_key"] == "binance" and data["last_day"] == "2026-08-31" else []
            for info in data["per_symbol"].values():
                self.assertEqual(info["missing_days"], expected)

    def test_final_source_and_receipt_hashes_and_repeat(self):
        for relative, digest in self.report["execution_verification"]["source_sha256"].items():
            self.assertEqual(hashlib.sha256((self.root / relative).read_bytes()).hexdigest(), digest)
        for row in self.report["primary"]["matrix"] + self.report["appendix_licence_conflict"]["matrix"]:
            path = self.root / "results/paper_trade_t5_v1_run2/receipts" / pathlib.Path(row["receipt_path"]).name
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row["receipt_sha256"])
        self.assertEqual((self.root / "results/paper_trade_t5_v1_run2/repeat_receipt.json").read_bytes(),
                         (self.root / "results/paper_trade_t5_v1_run2/receipts/binance-full-20260917.json").read_bytes())

    def test_recompute_whole_report_with_real_protocols_and_manifests(self):
        if not (self.root / "data/binance_vision_v1/fetch_manifest.json").exists():
            self.skipTest("local ignored raw data is unavailable")
        # Stored protocol paths are machine-local; don't read outside a relocated checkout.
        first = json.loads(self.paths[0].read_text())
        if not pathlib.Path(first["reproducibility"]["protocol_path"]).is_relative_to(self.root):
            self.skipTest("receipt contains original machine-local protocol paths")
        study = report.load_study(self.root / "research/paper_trade_t5_sensitivity_v1.json")
        base, metadata = load_protocol(self.root / study["base_protocol"])
        actual = report.build_report(self.paths, study, base, metadata["protocol_sha256"])
        for key, value in actual.items():
            self.assertEqual(value, self.report[key])


if __name__ == "__main__":
    unittest.main(verbosity=2)
