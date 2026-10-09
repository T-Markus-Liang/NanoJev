"""Pure synthetic unit tests for the execution simulator and the out-of-model risk guards.

No test downloads market data, contacts a broker, or asserts profitability. Every
quote, decision, cost and timestamp below is constructed in this file.
"""

from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

# Support both `unittest discover -s scripts` and `unittest scripts.test_...` invocation.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import financial_risk_v1 as risk  # noqa: E402
from financial_simulator_v1 import (  # noqa: E402
    ACTION_ABSTAIN, ACTION_BUY, ACTION_CLOSE_LONG, ACTION_CLOSE_SHORT, ACTION_FLAT, ACTION_LONG,
    ACTION_NO_TRADE, ACTION_OPEN_LONG, ACTION_OPEN_SHORT, ACTION_SELL, ACTION_SHORT,
    FUNDING_SOURCE_DECLARED_SCHEDULE, FUNDING_SOURCE_POLICY_CONSTANT,
    FUNDING_SOURCE_QUOTE_FIELD, INSTRUMENT_PERP, MARGIN_CROSS, MARGIN_ISOLATED,
    ORDER_ACCEPTED, ORDER_EXPIRED,
    ORDER_FILLED, ORDER_PARTIALLY_FILLED, ORDER_REJECTED, ORDER_SUBMITTED,
    POSITION_HEDGE, POSITION_ONE_WAY, PRICE_SOURCE_LAST,
    CapacityPolicy, ContractSpec, Decision, ExecutionPolicy, FeePolicy, FillPolicy,
    FutureInformationError,
    LatencyPolicy, LedgerConservationError, LeakGuardPolicy, MarkPolicy, NoFutureInformationGuard,
    PerpPolicy, Quote,
    REASON_ABSTAIN, REASON_CAPACITY_REJECT, REASON_INSUFFICIENT_INVENTORY, REASON_NO_TRADE,
    REASON_INSUFFICIENT_CASH, REASON_MARGIN_INSUFFICIENT, REASON_MARGIN_REDUCE,
    REASON_LOT_SIZE_ROUND, REASON_MIN_NOTIONAL, REASON_TICK_ROUND, REASON_NO_POSITION_TO_CLOSE,
    REASON_OPEN_WOULD_FLIP, REASON_PERP_CONTRACT_NOT_DECLARED, REASON_SHORT_DISABLED,
    REASON_LEVERAGE_ABOVE_CONTRACT_MAX, REASON_LIQUIDATION,
    REASON_FUNDING, LEDGER_REASON_FUNDING, LEDGER_REASON_LIQUIDATION_FEE,
    SIDE_BUY, SIDE_SELL, SlippagePolicy, SpreadPolicy, build_synthetic_scenario,
    causal_reference_selector,
    conservation_report, leaky_reference_selector, require_conservation, run_stress_suite,
    uniform_draw,
    validate_status_history,
)
import financial_backtest_v1 as backtest  # noqa: E402

T0 = 1_760_000_000_000_000_000
MS = 1_000_000
SECOND = 1_000_000_000
ASSET = "synthetic-asset"
VENUE = "synthetic-venue"
PERP = "synthetic-perp"
PERP_VENUE = "binance"


def make_quote(index, *, price=100.0, volume=1000.0, delay_ns=0, tick_ns=MS,
               asset=ASSET, venue=VENUE):
    event_ns = T0 + index * tick_ns
    half = price * 0.0001
    return Quote(asset_id=asset, venue=venue, event_ns=event_ns,
                 available_ns=event_ns + delay_ns, bid=price - half, ask=price + half,
                 volume=volume)


def make_decision(index, *, action=ACTION_BUY, quantity=1.0, decision_ns=None, **kwargs):
    return Decision(decision_id=f"d{index}", asset_id=ASSET, venue=VENUE,
                    decision_ns=T0 + index * MS if decision_ns is None else decision_ns,
                    action=action, quantity=quantity, **kwargs)


def instant_policy(**overrides):
    """Zero latency, no rejection, certain fills, ample capacity unless overridden."""
    defaults = dict(latency=LatencyPolicy(decision_to_order_ns=0, order_to_ack_ns=0,
                                          ack_to_execution_ns=0),
                    fills=FillPolicy(reject_probability=0.0, fill_probability=1.0),
                    capacity=CapacityPolicy(max_participation_fraction=1.0))
    defaults.update(overrides)
    return ExecutionPolicy(**defaults)


def run(quotes, decisions, policy=None, *, seed=7, asof_ns=None, **kwargs):
    from financial_simulator_v1 import replay
    if asof_ns is None:
        asof_ns = T0 + 100 * MS
    return replay(quotes, decisions, asof_ns=asof_ns, policy=policy or instant_policy(),
                  seed=seed, **kwargs)


class DeterminismTest(unittest.TestCase):
    def test_same_seed_and_input_produce_a_byte_identical_ledger(self):
        quotes, decisions, asof_ns = build_synthetic_scenario(11, assets=1, ticks=48)
        policy = ExecutionPolicy(fills=FillPolicy(reject_probability=0.3, fill_probability=0.7))
        first = run(quotes, decisions, policy, seed=99, asof_ns=asof_ns)
        second = run(quotes, decisions, policy, seed=99, asof_ns=asof_ns)
        self.assertEqual(first["ledger_sha256"], second["ledger_sha256"])
        self.assertEqual(json.dumps(first["ledger"], sort_keys=True),
                         json.dumps(second["ledger"], sort_keys=True))
        self.assertEqual(first["replay_sha256"], second["replay_sha256"])

    def test_seed_actually_drives_the_synthetic_draws(self):
        quotes, decisions, asof_ns = build_synthetic_scenario(11, assets=1, ticks=48)
        policy = ExecutionPolicy(fills=FillPolicy(reject_probability=0.5, fill_probability=0.5))
        by_seed = {seed: run(quotes, decisions, policy, seed=seed, asof_ns=asof_ns)["ledger_sha256"]
                   for seed in range(6)}
        self.assertGreater(len(set(by_seed.values())), 1)
        self.assertEqual(uniform_draw(3, "purpose", 1), uniform_draw(3, "purpose", 1))
        self.assertNotEqual(uniform_draw(3, "purpose", 1), uniform_draw(3, "purpose", 2))
        for key in range(64):
            self.assertGreaterEqual(uniform_draw(1, "bounds", key), 0.0)
            self.assertLess(uniform_draw(1, "bounds", key), 1.0)


class LedgerConservationTest(unittest.TestCase):
    def test_round_trip_ledger_conserves_cash_position_pnl_and_fees(self):
        quotes = [make_quote(i, volume=10_000.0) for i in range(10)]
        decisions = [make_decision(0, quantity=3.0), make_decision(2, action=ACTION_SELL, quantity=3.0)]
        result = run(quotes, decisions, asof_ns=T0 + 20 * MS)
        ledger = result["ledger"]
        report = conservation_report(ledger)
        self.assertTrue(report["ok"], report["violations"])
        self.assertEqual(report["violations"], [])
        for residual in report["residuals"].values():
            self.assertAlmostEqual(residual, 0.0, places=9)
        self.assertEqual(ledger["positions"][ASSET]["quantity"], 0.0)
        self.assertAlmostEqual(ledger["net_pnl"], ledger["realized_pnl"] - ledger["fees_paid"], places=9)
        self.assertGreaterEqual(ledger["fees_paid"], 0.0)
        self.assertAlmostEqual(
            ledger["net_pnl_excluding_explicit_fees"], ledger["net_pnl"] + ledger["fees_paid"], places=9)
        require_conservation(ledger)

    def test_hand_computed_single_fill_matches_an_independent_reference(self):
        quote = make_quote(0, price=100.0, volume=100.0)
        policy = ExecutionPolicy(
            fees=FeePolicy(fee_bps=1.0, fixed_fee_per_fill=0.0, minimum_fee=0.0),
            spread=SpreadPolicy(half_spread_bps=1.0, fixed_half_spread_price=0.0),
            slippage=SlippagePolicy(fixed_bps=0.0, impact_bps_at_full_capacity=0.0),
            capacity=CapacityPolicy(max_participation_fraction=1.0),
            fills=FillPolicy(reject_probability=0.0, fill_probability=1.0),
            latency=LatencyPolicy(decision_to_order_ns=0, order_to_ack_ns=0, ack_to_execution_ns=0))
        result = run([quote], [make_decision(0, quantity=2.0)], policy,
                     asof_ns=T0 + MS, initial_cash=10_000.0)
        ledger = result["ledger"]
        fill = ledger["fills"][0]
        mid = quote.mid
        expected_price = mid + mid * 1.0 / 10_000.0
        expected_notional = expected_price * 2.0
        expected_fee = expected_notional * 1.0 / 10_000.0
        self.assertAlmostEqual(fill["mid_price"], mid, places=12)
        self.assertAlmostEqual(fill["execution_price"], expected_price, places=12)
        self.assertAlmostEqual(fill["spread_cost"], (expected_price - mid) * 2.0, places=12)
        self.assertAlmostEqual(fill["slippage_cost"], 0.0, places=12)
        self.assertAlmostEqual(fill["explicit_fee"], expected_fee, places=12)
        self.assertAlmostEqual(ledger["cash"], 10_000.0 - expected_notional - expected_fee, places=12)
        self.assertAlmostEqual(ledger["fees_paid"], expected_fee, places=12)
        self.assertAlmostEqual(ledger["positions"][ASSET]["quantity"], 2.0, places=12)
        self.assertAlmostEqual(ledger["positions"][ASSET]["cost_basis"], expected_notional, places=12)
        self.assertAlmostEqual(ledger["realized_pnl"], 0.0, places=12)
        self.assertAlmostEqual(ledger["unrealized_pnl"], 2.0 * mid - expected_notional, places=12)
        self.assertAlmostEqual(ledger["net_pnl"], 2.0 * mid - expected_notional - expected_fee, places=12)
        self.assertTrue(ledger["conservation"]["ok"])

    def test_tampered_fill_or_cash_entry_fails_conservation(self):
        quotes = [make_quote(i, volume=10_000.0) for i in range(6)]
        decisions = [make_decision(0, quantity=2.0)]
        ledger = run(quotes, decisions, asof_ns=T0 + 10 * MS)["ledger"]
        self.assertTrue(ledger["conservation"]["ok"])

        tampered = deepcopy(ledger)
        tampered["fills"][0]["quantity"] += 1.0
        self.assertFalse(conservation_report(tampered)["ok"])
        with self.assertRaises(LedgerConservationError):
            require_conservation(tampered)

        tampered = deepcopy(ledger)
        tampered["cash_entries"][0]["amount"] -= 5.0
        self.assertFalse(conservation_report(tampered)["ok"])

        tampered = deepcopy(ledger)
        tampered["positions"][ASSET]["market_value"] *= 1.5
        self.assertFalse(conservation_report(tampered)["ok"])

        tampered = deepcopy(ledger)
        tampered["fees_paid"] += 1.0
        self.assertFalse(conservation_report(tampered)["ok"])


class LifecycleTest(unittest.TestCase):
    def test_fill_lifecycle_partial_then_complete(self):
        quotes = [make_quote(i, volume=1.0) for i in range(6)]
        policy = instant_policy(capacity=CapacityPolicy(max_participation_fraction=1.0))
        order = run(quotes, [make_decision(0, quantity=3.0)], policy,
                    asof_ns=T0 + 10 * MS)["orders"][0]
        self.assertEqual(order["status"], ORDER_FILLED)
        self.assertEqual([entry["status"] for entry in order["status_history"]],
                         [ORDER_SUBMITTED, ORDER_ACCEPTED, ORDER_PARTIALLY_FILLED,
                          ORDER_PARTIALLY_FILLED, ORDER_FILLED])
        self.assertEqual(order["fill_count"], 3)
        self.assertEqual(order["filled_quantity"], 3.0)
        self.assertEqual(order["remaining_quantity"], 0.0)
        validate_status_history(order)

    def test_partial_fill_then_expiry_keeps_the_remainder_open(self):
        quotes = [make_quote(i, volume=1.0) for i in range(2)]
        policy = instant_policy(fills=FillPolicy(order_time_to_live_ns=2 * MS))
        order = run(quotes, [make_decision(0, quantity=5.0)], policy,
                    asof_ns=T0 + 10 * MS)["orders"][0]
        self.assertEqual(order["status"], ORDER_EXPIRED)
        self.assertEqual(order["filled_quantity"], 2.0)
        self.assertEqual(order["remaining_quantity"], 3.0)
        self.assertEqual(order["status_history"][-1]["status"], ORDER_EXPIRED)
        self.assertIn(ORDER_PARTIALLY_FILLED,
                      [entry["status"] for entry in order["status_history"]])

    def test_order_rejection_leaves_the_ledger_untouched(self):
        quotes = [make_quote(i) for i in range(4)]
        policy = instant_policy(fills=FillPolicy(reject_probability=1.0))
        result = run(quotes, [make_decision(0, quantity=2.0)], policy, asof_ns=T0 + 10 * MS)
        order = result["orders"][0]
        self.assertEqual(order["status"], ORDER_REJECTED)
        self.assertEqual([entry["status"] for entry in order["status_history"]],
                         [ORDER_SUBMITTED, ORDER_REJECTED])
        self.assertEqual(order["filled_quantity"], 0.0)
        ledger = result["ledger"]
        self.assertEqual(ledger["cash"], ledger["initial_cash"])
        self.assertEqual(ledger["fills"], [])
        self.assertEqual(ledger["fees_paid"], 0.0)
        self.assertTrue(ledger["conservation"]["ok"])

    def test_expiry_when_latency_exceeds_time_to_live(self):
        quotes = [make_quote(i) for i in range(4)]
        policy = instant_policy(fills=FillPolicy(order_time_to_live_ns=MS),
                                latency=LatencyPolicy(decision_to_order_ns=5 * MS,
                                                      order_to_ack_ns=0, ack_to_execution_ns=0))
        result = run(quotes, [make_decision(0, quantity=1.0)], policy, asof_ns=T0 + 10 * MS)
        order = result["orders"][0]
        self.assertEqual(order["status"], ORDER_EXPIRED)
        self.assertEqual(order["fills"], [])
        self.assertEqual([entry["status"] for entry in order["status_history"]],
                         [ORDER_SUBMITTED, ORDER_EXPIRED])
        self.assertEqual(result["ledger"]["fill_count"], 0)
        validate_status_history(order)

    def test_illegal_status_transition_is_rejected(self):
        with self.assertRaises(Exception):
            validate_status_history({"status_history": [
                {"status": ORDER_FILLED, "ns": T0, "reason_code": "x"}]})
        with self.assertRaises(Exception):
            validate_status_history({"status_history": [
                {"status": ORDER_SUBMITTED, "ns": T0, "reason_code": "x"},
                {"status": ORDER_ACCEPTED, "ns": T0 + 1, "reason_code": "x"},
                {"status": ORDER_REJECTED, "ns": T0 + 2, "reason_code": "x"}]})
        with self.assertRaises(Exception):
            validate_status_history({"status_history": [
                {"status": ORDER_SUBMITTED, "ns": T0, "reason_code": "x"},
                {"status": ORDER_ACCEPTED, "ns": T0, "reason_code": "x"}]})

    def test_event_timestamps_are_distinct_and_ordered(self):
        policy = ExecutionPolicy(
            latency=LatencyPolicy(decision_to_order_ns=2 * MS, order_to_ack_ns=3 * MS,
                                  ack_to_execution_ns=4 * MS),
            fills=FillPolicy(reject_probability=0.0, fill_probability=1.0))
        quotes = [make_quote(i, delay_ns=1 * MS) for i in range(30)]
        decision = make_decision(1, quantity=1.0)
        order = run(quotes, [decision], policy, asof_ns=T0 + 100 * MS)["orders"][0]
        self.assertEqual(order["submit_ns"], decision.decision_ns + 2 * MS)
        self.assertEqual(order["ack_ns"], order["submit_ns"] + 3 * MS)
        fill = order["fills"][0]
        self.assertGreaterEqual(fill["execution_ns"], order["ack_ns"] + 4 * MS)
        self.assertEqual(fill["execution_ns"], fill["quote_available_ns"])
        self.assertGreaterEqual(order["decision_ns"], 0)
        self.assertLess(order["decision_ns"], order["submit_ns"])
        self.assertLess(order["submit_ns"], order["ack_ns"])
        self.assertLess(order["ack_ns"], fill["execution_ns"])


class CostPolicyTest(unittest.TestCase):
    def scenario(self):
        quotes, decisions, asof_ns = build_synthetic_scenario(3, assets=2, ticks=160)
        return quotes, decisions, asof_ns

    def test_raising_fees_never_increases_net_pnl(self):
        quotes, decisions, asof_ns = self.scenario()
        series = []
        for fee_bps in (0.0, 0.5, 1.0, 5.0, 25.0, 100.0):
            policy = ExecutionPolicy(fees=FeePolicy(fee_bps=fee_bps),
                                     fills=FillPolicy(reject_probability=0.0, fill_probability=1.0),
                                     capacity=CapacityPolicy(max_participation_fraction=0.1))
            result = run(quotes, decisions, policy, seed=5, asof_ns=asof_ns)
            series.append(result["ledger"])
        self.assertGreater(series[0]["fill_count"], 0)
        for lower, higher in zip(series, series[1:]):
            self.assertLessEqual(higher["net_pnl"], lower["net_pnl"] + 1e-9)
            self.assertGreaterEqual(higher["fees_paid"], lower["fees_paid"])
            self.assertAlmostEqual(higher["net_pnl_excluding_explicit_fees"],
                                   lower["net_pnl_excluding_explicit_fees"], places=6)

    def test_raising_spread_and_slippage_increases_costs_on_a_buy(self):
        quotes = [make_quote(i, volume=10_000.0) for i in range(6)]
        decision = [make_decision(0, quantity=2.0)]
        costs = []
        for half_spread_bps, slippage_bps in ((0.0, 0.0), (1.0, 0.0), (1.0, 2.0)):
            policy = instant_policy(spread=SpreadPolicy(half_spread_bps=half_spread_bps),
                                    slippage=SlippagePolicy(fixed_bps=slippage_bps))
            ledger = run(quotes, decision, policy, asof_ns=T0 + 10 * MS)["ledger"]
            fill = ledger["fills"][0]
            costs.append(fill["execution_price"])
            self.assertAlmostEqual(
                abs(fill["execution_price"] - fill["mid_price"]) * fill["quantity"],
                fill["spread_cost"] + fill["slippage_cost"], places=9)
            self.assertGreaterEqual(fill["explicit_fee"], 0.0)
        self.assertEqual(costs, sorted(costs))
        self.assertLess(costs[0], costs[-1])

    def test_limit_price_blocks_execution_and_expires_the_order(self):
        quotes = [make_quote(i, price=100.0) for i in range(6)]
        decision = [make_decision(0, quantity=1.0, limit_price=1.0)]
        order = run(quotes, decision, asof_ns=T0 + 10 * MS)["orders"][0]
        self.assertEqual(order["status"], ORDER_EXPIRED)
        self.assertEqual(order["filled_quantity"], 0.0)

    def test_capacity_reject_policy_rejects_oversized_orders(self):
        quotes = [make_quote(i) for i in range(4)]
        policy = instant_policy(capacity=CapacityPolicy(max_order_quantity=1.0,
                                                        on_order_exceeds_limit="reject"))
        result = run(quotes, [make_decision(0, quantity=5.0)], policy, asof_ns=T0 + 10 * MS)
        self.assertEqual(result["orders"], [])
        self.assertEqual(result["decisions"][0]["outcome"], "rejected")
        self.assertIn(REASON_CAPACITY_REJECT, result["decisions"][0]["reason_codes"])

    def test_mark_fallback_policy_is_explicit_and_parameterized(self):
        quotes = [make_quote(i, volume=10_000.0) for i in range(6)]
        decisions = [make_decision(0, quantity=2.0)]
        observed = {}
        for fallback in ("cost_basis", "zero", "last_fill_price"):
            policy = instant_policy(marks=MarkPolicy(fallback=fallback))
            ledger = run(quotes, decisions, policy, asof_ns=T0 - 1)["ledger"]
            position = ledger["positions"][ASSET]
            observed[fallback] = (position["mark_price"], position["mark_source"])
            self.assertEqual(position["mark_source"], f"mark_fallback::{fallback}")
            self.assertTrue(ledger["conservation"]["ok"])
        self.assertEqual(observed["zero"][0], 0.0)
        self.assertGreater(observed["cost_basis"][0], 0.0)
        self.assertGreater(observed["last_fill_price"][0], 0.0)
        with self.assertRaises(ValueError):
            MarkPolicy(fallback="invented")


class FutureInformationGuardTest(unittest.TestCase):
    def test_deliberately_leaked_decision_price_raises(self):
        quotes = [make_quote(i, delay_ns=0) for i in range(6)]
        decision = make_decision(2, quantity=1.0)
        result = run(quotes, [decision], asof_ns=T0 + 10 * MS)
        self.assertLessEqual(result["decisions"][0]["reference_available_ns"], decision.decision_ns)

        with self.assertRaises(FutureInformationError):
            run(quotes, [decision], reference_selector=leaky_reference_selector,
                asof_ns=T0 + 10 * MS)

    def test_guard_records_the_violation_it_raised(self):
        quotes = [make_quote(i, delay_ns=0) for i in range(6)]
        guard = NoFutureInformationGuard()
        from financial_simulator_v1 import replay
        with self.assertRaises(FutureInformationError):
            replay(quotes, [make_decision(2, quantity=1.0)], asof_ns=T0 + 10 * MS,
                   reference_selector=leaky_reference_selector, guard=guard)
        snapshot = guard.snapshot()
        self.assertEqual(snapshot["decision_violations"], 1)
        self.assertEqual(snapshot["execution_violations"], 0)
        self.assertEqual(snapshot["violation_count"], 1)
        self.assertIn("decision", snapshot["violations"][0]["message"])

    def test_execution_may_not_use_a_price_from_its_own_future(self):
        guard = NoFutureInformationGuard()
        late_quote = make_quote(9)
        with self.assertRaises(FutureInformationError):
            guard.check_execution(T0, T0 + MS, T0 + 5 * MS, late_quote)
        self.assertEqual(guard.snapshot()["execution_violations"], 1)
        with self.assertRaises(FutureInformationError):
            guard.check_execution(T0, T0 + 2 * MS, T0 + MS, make_quote(0))
        self.assertEqual(guard.snapshot()["execution_violations"], 2)

    def test_strict_mode_forbids_any_post_decision_execution_price(self):
        quotes = [make_quote(i, delay_ns=1 * MS) for i in range(8)]
        strict = NoFutureInformationGuard(
            LeakGuardPolicy(forbid_post_decision_prices_in_executions=True))
        from financial_simulator_v1 import replay
        with self.assertRaises(FutureInformationError):
            replay(quotes, [make_decision(1, quantity=1.0)], asof_ns=T0 + 10 * MS, guard=strict)
        lenient = NoFutureInformationGuard()
        result = replay(quotes, [make_decision(1, quantity=1.0)], asof_ns=T0 + 10 * MS, guard=lenient)
        self.assertGreater(result["ledger"]["fill_count"], 0)
        self.assertEqual(lenient.snapshot()["violation_count"], 0)

    def test_causal_selector_never_returns_an_unavailable_quote(self):
        quotes = [make_quote(i, delay_ns=3 * MS) for i in range(8)]
        for index in range(8):
            decision = make_decision(index)
            chosen = causal_reference_selector(quotes, decision)
            if chosen is not None:
                self.assertLessEqual(chosen.available_ns, decision.decision_ns)


class RiskGateIntegrationTest(unittest.TestCase):
    def test_risk_gate_blocks_an_order_and_leaves_no_position(self):
        quotes = [make_quote(i, volume=10_000.0) for i in range(6)]
        engine = risk.RiskEngine(limits=risk.RiskLimits(max_position_quantity=0.0))
        result = run(quotes, [make_decision(0, quantity=3.0)], risk_gate=engine,
                     asof_ns=T0 + 10 * MS)
        self.assertEqual(result["orders"], [])
        record = result["decisions"][0]
        self.assertEqual(record["outcome"], "risk_blocked")
        self.assertEqual(record["risk_decision"], "block")
        self.assertIn(risk.REASON_POSITION_LIMIT, record["risk_reason_codes"])
        self.assertEqual(result["ledger"]["cash"], result["ledger"]["initial_cash"])
        self.assertEqual(result["ledger"]["fill_count"], 0)

    def test_risk_gate_reduces_quantity_instead_of_blocking(self):
        quotes = [make_quote(i, volume=10_000.0) for i in range(6)]
        engine = risk.RiskEngine(limits=risk.RiskLimits(max_position_quantity=4.0))
        result = run(quotes, [make_decision(0, quantity=10.0)], risk_gate=engine,
                     asof_ns=T0 + 10 * MS)
        record = result["decisions"][0]
        self.assertEqual(record["risk_decision"], "reduce")
        self.assertAlmostEqual(record["effective_quantity"], 4.0)
        self.assertEqual(result["orders"][0]["filled_quantity"], 4.0)
        self.assertEqual(result["ledger"]["positions"][ASSET]["quantity"], 4.0)

    def test_stale_data_and_clock_drift_block_every_order(self):
        quotes = [make_quote(0, volume=10_000.0)]
        stale_decision = make_decision(0, quantity=1.0, decision_ns=T0 + 30 * SECOND)
        engine = risk.RiskEngine(limits=risk.RiskLimits(max_data_staleness_ns=SECOND,
                                                        max_clock_drift_ns=3_600 * SECOND))
        result = run(quotes, [stale_decision], risk_gate=engine, asof_ns=T0 + 60 * SECOND)
        self.assertEqual(result["decisions"][0]["risk_reason_codes"], [risk.REASON_STALE_DATA])
        self.assertEqual(result["ledger"]["fill_count"], 0)

        fresh_quotes = [make_quote(i, volume=10_000.0) for i in range(6)]
        engine = risk.RiskEngine(limits=risk.RiskLimits(max_clock_drift_ns=SECOND))
        result = run(fresh_quotes, [make_decision(0, quantity=1.0)], risk_gate=engine,
                     clock_skew_ns=10 * SECOND, asof_ns=T0 + 10 * MS)
        self.assertEqual(result["decisions"][0]["risk_reason_codes"], [risk.REASON_CLOCK_DRIFT])
        self.assertEqual(result["ledger"]["fill_count"], 0)

    def test_kill_switch_blocks_everything_until_released(self):
        quotes = [make_quote(i, volume=10_000.0) for i in range(6)]
        engine = risk.RiskEngine()
        engine.kill_switch.engage("synthetic_operator_halt", T0)
        blocked = run(quotes, [make_decision(0, quantity=1.0)], risk_gate=engine,
                      asof_ns=T0 + 10 * MS)
        self.assertEqual(blocked["ledger"]["fill_count"], 0)
        self.assertEqual(blocked["decisions"][0]["risk_reason_codes"],
                         [risk.REASON_KILL_SWITCH_ENGAGED])
        engine.kill_switch.release(T0 + 1)
        released = run(quotes, [make_decision(0, quantity=1.0)], risk_gate=engine,
                       asof_ns=T0 + 10 * MS)
        self.assertGreater(released["ledger"]["fill_count"], 0)

    def test_turnover_and_order_rate_windows_are_enforced(self):
        engine = risk.RiskEngine(limits=risk.RiskLimits(max_orders_per_window=2,
                                                        order_rate_window_ns=SECOND))
        base = {"decision_ns": T0, "local_ns": T0, "last_data_ns": T0 - MS, "feed_clock_ns": T0,
                "asset_id": ASSET, "venue": VENUE,
                "side": SIDE_BUY, "requested_quantity": 1.0, "reference_price": 100.0,
                "position_quantity": 0.0, "gross_exposure_notional": 0.0,
                "net_exposure_notional": 0.0, "realized_pnl": 0.0, "unrealized_pnl": 0.0,
                "fees_paid": 0.0, "net_pnl": 0.0}
        self.assertEqual(engine.evaluate(base).decision, risk.DECISION_ALLOW)
        engine.record_order(base)
        engine.record_order(base)
        blocked = engine.evaluate(base)
        self.assertEqual(blocked.decision, risk.DECISION_BLOCK)
        self.assertEqual(blocked.reason_codes, (risk.REASON_ORDER_RATE_LIMIT,))
        later = dict(base, decision_ns=T0 + 2 * SECOND, local_ns=T0 + 2 * SECOND,
                     last_data_ns=T0 + 2 * SECOND - MS, feed_clock_ns=T0 + 2 * SECOND)
        self.assertEqual(engine.evaluate(later).decision, risk.DECISION_ALLOW)

        turnover = risk.RiskEngine(limits=risk.RiskLimits(max_turnover_notional=150.0,
                                                          turnover_window_ns=SECOND))
        turnover.record_fill(base, 100.0)
        reduced = turnover.evaluate(base)
        self.assertEqual(reduced.decision, risk.DECISION_REDUCE)
        self.assertAlmostEqual(reduced.allowed_quantity, 0.5)
        self.assertIn(risk.REASON_TURNOVER_LIMIT, reduced.reason_codes)

    def test_sell_reducing_a_long_is_allowed_when_gross_exposure_is_at_the_limit(self):
        engine = risk.RiskEngine(limits=risk.RiskLimits(max_gross_exposure_notional=1000.0,
                                                        max_abs_net_exposure_notional=1000.0))
        context = {"decision_ns": T0, "local_ns": T0, "last_data_ns": T0 - MS, "feed_clock_ns": T0,
                   "asset_id": ASSET, "venue": VENUE,
                   "side": SIDE_SELL, "requested_quantity": 5.0, "reference_price": 100.0,
                   "position_quantity": 10.0, "gross_exposure_notional": 1000.0,
                   "net_exposure_notional": 1000.0, "realized_pnl": 0.0, "unrealized_pnl": 0.0,
                   "fees_paid": 0.0, "net_pnl": 0.0}
        decision = engine.evaluate(context)
        self.assertEqual(decision.decision, risk.DECISION_ALLOW)
        self.assertEqual(decision.reason_codes, (risk.REASON_OK,))

    def test_latched_kill_switch_engages_on_first_block(self):
        engine = risk.RiskEngine(limits=risk.RiskLimits(max_position_quantity=0.0,
                                                        latch_kill_switch_on_block=True))
        context = {"decision_ns": T0, "local_ns": T0, "last_data_ns": T0 - MS, "feed_clock_ns": T0,
                   "asset_id": ASSET, "venue": VENUE,
                   "side": SIDE_BUY, "requested_quantity": 1.0, "reference_price": 100.0,
                   "position_quantity": 0.0, "gross_exposure_notional": 0.0,
                   "net_exposure_notional": 0.0, "realized_pnl": 0.0, "unrealized_pnl": 0.0,
                   "fees_paid": 0.0, "net_pnl": 0.0}
        first = engine.evaluate(context)
        self.assertEqual(first.decision, risk.DECISION_BLOCK)
        self.assertTrue(engine.kill_switch.engaged)
        second = engine.evaluate(context)
        self.assertEqual(second.reason_codes, (risk.REASON_KILL_SWITCH_ENGAGED,))


class SyntheticActionAndValidationTest(unittest.TestCase):
    def test_abstain_and_no_trade_are_first_class_and_change_nothing(self):
        quotes = [make_quote(i, volume=10_000.0) for i in range(6)]
        decisions = [make_decision(0, action=ACTION_ABSTAIN), make_decision(1, action=ACTION_NO_TRADE)]
        result = run(quotes, decisions, asof_ns=T0 + 10 * MS)
        self.assertEqual([record["reason_codes"] for record in result["decisions"]],
                         [[REASON_ABSTAIN], [REASON_NO_TRADE]])
        self.assertEqual([record["order_id"] for record in result["decisions"]], [None, None])
        self.assertEqual(result["orders"], [])
        self.assertEqual(result["counts"]["abstain_decisions"], 1)
        self.assertEqual(result["counts"]["no_trade_decisions"], 1)
        self.assertEqual(result["ledger"]["cash"], result["ledger"]["initial_cash"])

    def test_sell_beyond_inventory_is_rejected_without_shorting(self):
        quotes = [make_quote(i, volume=10_000.0) for i in range(6)]
        result = run(quotes, [make_decision(0, action=ACTION_SELL, quantity=1.0)],
                     asof_ns=T0 + 10 * MS)
        record = result["decisions"][0]
        self.assertEqual(record["outcome"], "rejected")
        self.assertIn(REASON_INSUFFICIENT_INVENTORY, record["reason_codes"])
        self.assertEqual(result["ledger"]["cash"], result["ledger"]["initial_cash"])
        self.assertGreaterEqual(result["ledger"]["positions"].get(ASSET, {"quantity": 0.0})["quantity"], 0.0)

    def test_missing_reference_price_is_rejected_not_silently_filled(self):
        quotes = [make_quote(5, volume=10_000.0)]
        result = run(quotes, [make_decision(0, quantity=1.0)], asof_ns=T0 + 10 * MS)
        self.assertEqual(result["orders"], [])
        self.assertEqual(result["decisions"][0]["outcome"], "rejected")

    def test_quote_and_decision_contract_validation(self):
        with self.assertRaises(ValueError):
            Quote(asset_id=ASSET, venue=VENUE, event_ns=T0 + 1, available_ns=T0, bid=1.0, ask=2.0,
                  volume=1.0)
        with self.assertRaises(ValueError):
            Quote(asset_id=ASSET, venue=VENUE, event_ns=T0, available_ns=T0, bid=2.0, ask=1.0,
                  volume=1.0)
        with self.assertRaises(ValueError):
            make_decision(0, action=ACTION_BUY, quantity=0.0)
        with self.assertRaises(ValueError):
            make_decision(0, action="liquidate", quantity=1.0)
        with self.assertRaises(ValueError):
            make_decision(0, action=ACTION_BUY, quantity=1.0, expire_ns=T0 - 1)
        with self.assertRaises(ValueError):
            FeePolicy(fee_bps=-1.0)
        with self.assertRaises(ValueError):
            FillPolicy(fill_probability=1.5)
        with self.assertRaises(ValueError):
            CapacityPolicy(on_order_exceeds_limit="ignore")

    def test_every_order_status_is_reachable_and_consistent(self):
        quotes = [make_quote(i, volume=2.0) for i in range(8)]
        partial_policy = instant_policy(fills=FillPolicy(order_time_to_live_ns=3 * MS))
        partial = run(quotes, [make_decision(0, quantity=9.0)], partial_policy, asof_ns=T0 + 20 * MS)
        complete = run(quotes, [make_decision(0, quantity=1.0)], partial_policy, asof_ns=T0 + 20 * MS)
        rejected = run(quotes, [make_decision(0, quantity=1.0)],
                       instant_policy(fills=FillPolicy(reject_probability=1.0)), asof_ns=T0 + 20 * MS)
        observed = set()
        for result in (partial, complete, rejected):
            for order in result["orders"]:
                validate_status_history(order)
                observed.update(entry["status"] for entry in order["status_history"])
        self.assertEqual(observed, {ORDER_SUBMITTED, ORDER_ACCEPTED, ORDER_PARTIALLY_FILLED,
                                    ORDER_FILLED, ORDER_EXPIRED, ORDER_REJECTED})


class StressReceiptTest(unittest.TestCase):
    def test_stress_suite_receipt_is_self_consistent(self):
        receipt = run_stress_suite(seed=4_242)
        self.assertEqual(receipt["status"], "passed")
        self.assertEqual(receipt["schema_version"], "nanojev-financial-simulator-stress-v1")
        self.assertTrue(receipt["determinism"]["byte_identical"])
        self.assertTrue(receipt["conservation"]["ok"])
        self.assertEqual(receipt["cost_monotonicity"]["violations"], [])
        self.assertEqual(receipt["guard"]["decision_leaks_blocked"], 200)
        self.assertEqual(receipt["guard"]["strict_execution_leaks_blocked"], 1)
        self.assertIn("synthetic", receipt["scope"].lower())
        self.assertIn("no broker", receipt["scope"].lower())
        self.assertGreater(receipt["counts"]["baseline"]["fills"], 0)
        self.assertGreater(receipt["counts"]["partial"]["fills"], 0)
        self.assertEqual(receipt["counts"]["rejection"]["filled"], 0)
        self.assertEqual(receipt["counts"]["no_fill"]["fills"], 0)
        for name, counts in receipt["counts"].items():
            self.assertTrue(counts["decision_closes"], name)
            self.assertEqual(
                counts["abstain_decisions"] + counts["no_trade_decisions"]
                + counts["risk_blocked_decisions"] + counts["rejected_decisions"]
                + counts["no_order_decisions"] + counts["ordered_decisions"],
                counts["decisions"], name)
        provisional = receipt["provisional_pending_r1"]
        self.assertIn("fees.fee_bps", provisional)
        self.assertIn("latency.ack_to_execution_ns", provisional)
        self.assertIn("limits.max_data_staleness_ns", provisional)
        self.assertEqual({entry["path"].rsplit("/", 1)[-1] for entry in receipt["code"]},
                         {"financial_simulator_v1.py", "financial_risk_v1.py"})

    def test_cli_writes_a_stress_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "stress.json"
            command = [sys.executable, str(Path(__file__).with_name("financial_simulator_v1.py")),
                       "--stress", "--output", str(output), "--seed", "31337"]
            result = subprocess.run(command, capture_output=True, text=True, timeout=900)
            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "passed")
            self.assertEqual(receipt["seed"], 31337)
            self.assertTrue(receipt["determinism"]["byte_identical"])
            self.assertIn("limitations", receipt)
            self.assertIn("no real market data", receipt["scope"])

    def test_cli_receipt_is_byte_identical_across_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            payloads = []
            for index in (1, 2):
                output = Path(tmp) / f"stress-{index}.json"
                command = [sys.executable,
                           str(Path(__file__).with_name("financial_simulator_v1.py")),
                           "--stress", "--output", str(output), "--seed", "909"]
                result = subprocess.run(command, capture_output=True, text=True, timeout=900)
                self.assertEqual(result.returncode, 0, result.stderr)
                payloads.append(output.read_bytes())
            self.assertEqual(payloads[0], payloads[1])
            self.assertNotIn(b"elapsed_ms", payloads[0])
            receipt = json.loads(payloads[0].decode("utf-8"))
            self.assertTrue(receipt["deterministic_receipt"])

    def test_stress_receipt_covers_the_perpetual_contract_layer(self):
        receipt = run_stress_suite(seed=4_242)
        perp = receipt["perpetual_contracts"]
        self.assertEqual(perp["schema_version"], "nanojev-financial-perp-stress-v1")
        self.assertGreater(perp["short_round_trip"]["net_pnl"], 0.0)
        self.assertTrue(perp["short_round_trip"]["conservation_ok"])
        self.assertEqual(perp["funding_monotonicity"]["violations"], [])
        self.assertGreater(perp["funding_monotonicity"]["funding_cost"][-1],
                           perp["funding_monotonicity"]["funding_cost"][0])
        self.assertGreaterEqual(perp["liquidation"]["liquidation_count"], 1)
        self.assertTrue(perp["liquidation"]["position_flat"])
        self.assertEqual(perp["liquidation"]["position_after"], 0.0)
        self.assertTrue(perp["liquidation"]["conservation_ok"])
        self.assertEqual(perp["margin_mode_comparison"]["isolated_liquidations"], 1)
        self.assertEqual(perp["margin_mode_comparison"]["cross_liquidations"], 0)
        self.assertGreater(perp["margin_mode_comparison"]["isolated_final_equity"],
                           perp["margin_mode_comparison"]["cross_final_equity"])
        self.assertEqual(perp["leverage_effect"]["net_pnl_spread"], 0.0)
        self.assertEqual(perp["multiplier_scaling"]["pnl_ratio_error"], 0.0)
        self.assertTrue(perp["margin_sufficiency"]["enforced"])
        self.assertEqual(perp["netting"]["one_way_keys"], 1)
        self.assertEqual(perp["netting"]["hedge_keys"], 2)
        self.assertIn("perp_scope", receipt)
        self.assertIn("PERPETUAL CONTRACT", receipt["perp_scope"])
        self.assertTrue(receipt["backtest"]["byte_identical"])
        self.assertTrue(receipt["backtest"]["attribution_closes"])
        self.assertIn("synthetic-regime-up", receipt["backtest"]["regime_labels"])


class PerpContractSpecTest(unittest.TestCase):
    def test_contract_spec_validation(self):
        with self.assertRaises(ValueError):
            ContractSpec(asset_id=PERP, venue=PERP_VENUE, contract_multiplier=0.0)
        with self.assertRaises(ValueError):
            ContractSpec(asset_id=PERP, venue=PERP_VENUE, margin_mode="portfolio")
        with self.assertRaises(ValueError):
            ContractSpec(asset_id=PERP, venue=PERP_VENUE, settlement_asset="base")
        with self.assertRaises(ValueError):
            ContractSpec(asset_id=PERP, venue=PERP_VENUE, maintenance_margin_rate=1.5)
        with self.assertRaises(ValueError):
            ContractSpec(asset_id="", venue=PERP_VENUE)
        with self.assertRaises(ValueError):
            ContractSpec(asset_id=PERP, venue=PERP_VENUE, max_leverage=0.5)

    def test_conflicting_contract_declarations_are_refused(self):
        quotes = [perp_quote(i) for i in range(8)]
        with self.assertRaises(ValueError):
            run(quotes, [perp_decision(0, action=ACTION_LONG, quantity=1.0)], contracts=[
                perp_contract(), perp_contract()])
        with self.assertRaises(ValueError):
            run(quotes, [perp_decision(0, action=ACTION_LONG, quantity=1.0)], contracts=[
                perp_contract(contract_multiplier=1.0),
                perp_contract(venue="bybit", contract_multiplier=2.0)])
        with self.assertRaises(ValueError):
            pol = instant_policy(perps=PerpPolicy(margin_mode=MARGIN_CROSS))
            run(quotes, [perp_decision(0, action=ACTION_LONG, quantity=1.0)], pol,
                contracts=[perp_contract(margin_mode=MARGIN_ISOLATED)])

    def test_perp_actions_require_a_declared_contract(self):
        quotes = [make_quote(i, volume=10_000.0) for i in range(8)]
        for action, quantity in ((ACTION_LONG, 1.0), (ACTION_SHORT, 1.0), (ACTION_FLAT, 0.0),
                                 (ACTION_OPEN_LONG, 1.0), (ACTION_CLOSE_SHORT, 1.0)):
            result = run(quotes, [make_decision(0, action=action, quantity=quantity)],
                         asof_ns=T0 + 10 * MS)
            self.assertEqual(result["decisions"][0]["outcome"], "rejected", action)
            self.assertIn(REASON_PERP_CONTRACT_NOT_DECLARED, result["decisions"][0]["reason_codes"])

    def test_lot_size_minimum_notional_and_tick_rounding_are_enforced(self):
        quotes = [perp_quote(i, price=100.0, volume=10_000.0) for i in range(8)]
        contract = perp_contract(lot_size=0.5, min_notional=50.0, tick_size=0.03)
        result = run(quotes, [perp_decision(0, action=ACTION_LONG, quantity=3.7)],
                     contracts=[contract], asof_ns=T0 + 10 * MS)
        order = result["orders"][0]
        self.assertEqual(order["filled_quantity"], 3.5)
        self.assertIn(REASON_LOT_SIZE_ROUND, order["selection_reason_codes"])
        self.assertIn(REASON_TICK_ROUND, order["selection_reason_codes"])
        fill = order["fills"][0]
        self.assertAlmostEqual(round(fill["execution_price"] / 0.03), fill["execution_price"] / 0.03,
                               places=6)

        tiny = run(quotes, [perp_decision(0, action=ACTION_LONG, quantity=0.4)],
                   contracts=[perp_contract(lot_size=0.5, min_notional=1.0)],
                   asof_ns=T0 + 10 * MS)
        self.assertEqual(tiny["orders"], [])
        self.assertEqual(tiny["decisions"][0]["outcome"], "rejected")

        below = run(quotes, [perp_decision(0, action=ACTION_LONG, quantity=0.5)],
                    contracts=[perp_contract(lot_size=0.5, min_notional=1_000_000.0)],
                    asof_ns=T0 + 10 * MS)
        self.assertEqual(below["orders"], [])
        self.assertIn(REASON_MIN_NOTIONAL, below["decisions"][0]["reason_codes"])


class PerpShortTest(unittest.TestCase):
    def test_short_round_trip_realizes_profit_and_conserves(self):
        quotes = [perp_quote(i, price=100.0, volume=10_000.0) for i in range(6)]
        quotes = [perp_quote(i, price=(100.0 if i < 3 else 90.0), volume=10_000.0)
                  for i in range(6)]
        decisions = [perp_decision(0, action=ACTION_SHORT, quantity=10.0),
                     perp_decision(3, action=ACTION_FLAT)]
        result = run(quotes, decisions, perp_policy(), contracts=[perp_contract()],
                     asof_ns=T0 + 10 * MS)
        ledger = result["ledger"]
        self.assertEqual(result["counts"]["rejected"], 0)
        self.assertAlmostEqual(ledger["realized_pnl"], 100.0, places=9)
        self.assertAlmostEqual(ledger["net_pnl"], 100.0, places=9)
        self.assertEqual(ledger["positions"][PERP]["quantity"], 0.0)
        self.assertEqual(ledger["fills"][0]["side"], SIDE_SELL)
        self.assertEqual(ledger["fills"][1]["side"], SIDE_BUY)
        self.assertTrue(ledger["conservation"]["ok"], ledger["conservation"]["violations"])

    def test_short_losing_round_trip(self):
        quotes = [perp_quote(i, price=(100.0 if i < 3 else 110.0), volume=10_000.0)
                  for i in range(6)]
        decisions = [perp_decision(0, action=ACTION_SHORT, quantity=10.0),
                     perp_decision(3, action=ACTION_FLAT)]
        ledger = run(quotes, decisions, perp_policy(), contracts=[perp_contract()],
                     asof_ns=T0 + 10 * MS)["ledger"]
        self.assertAlmostEqual(ledger["realized_pnl"], -100.0, places=9)
        self.assertTrue(ledger["conservation"]["ok"])

    def test_spot_still_refuses_short_sales(self):
        quotes = [make_quote(i, volume=10_000.0) for i in range(6)]
        result = run(quotes, [make_decision(0, action=ACTION_SELL, quantity=1.0)],
                     asof_ns=T0 + 10 * MS)
        self.assertIn(REASON_INSUFFICIENT_INVENTORY, result["decisions"][0]["reason_codes"])

    def test_short_can_be_disabled_by_the_perp_policy(self):
        quotes = [perp_quote(i, price=100.0, volume=10_000.0) for i in range(6)]
        policy = perp_policy(allow_short=False)
        result = run(quotes, [perp_decision(0, action=ACTION_SHORT, quantity=1.0)], policy,
                     contracts=[perp_contract()], asof_ns=T0 + 10 * MS)
        self.assertIn(REASON_SHORT_DISABLED, result["decisions"][0]["reason_codes"])
        self.assertEqual(result["ledger"]["fill_count"], 0)

    def test_one_way_netting_closes_then_flips(self):
        quotes = [perp_quote(i, price=100.0, volume=10_000.0) for i in range(8)]
        decisions = [perp_decision(0, action=ACTION_LONG, quantity=5.0),
                     perp_decision(2, action=ACTION_SELL, quantity=8.0)]
        ledger = run(quotes, decisions, perp_policy(), contracts=[perp_contract()],
                     asof_ns=T0 + 10 * MS)["ledger"]
        self.assertEqual(sorted(ledger["positions"]), [PERP])
        self.assertAlmostEqual(ledger["positions"][PERP]["quantity"], -3.0, places=9)
        self.assertAlmostEqual(ledger["realized_pnl"], 0.0, places=9)
        self.assertTrue(ledger["conservation"]["ok"])

    def test_open_effect_refuses_to_flip_a_one_way_position(self):
        quotes = [perp_quote(i, price=100.0, volume=10_000.0) for i in range(8)]
        decisions = [perp_decision(0, action=ACTION_LONG, quantity=5.0),
                     perp_decision(2, action=ACTION_OPEN_SHORT, quantity=8.0)]
        result = run(quotes, decisions, perp_policy(), contracts=[perp_contract()],
                     asof_ns=T0 + 10 * MS)
        self.assertIn(REASON_OPEN_WOULD_FLIP, result["decisions"][1]["reason_codes"])
        self.assertAlmostEqual(result["ledger"]["positions"][PERP]["quantity"], 5.0, places=9)

    def test_close_effect_is_reduce_only(self):
        quotes = [perp_quote(i, price=100.0, volume=10_000.0) for i in range(8)]
        decisions = [perp_decision(0, action=ACTION_CLOSE_LONG, quantity=1.0)]
        result = run(quotes, decisions, perp_policy(), contracts=[perp_contract()],
                     asof_ns=T0 + 10 * MS)
        self.assertIn(REASON_NO_POSITION_TO_CLOSE, result["decisions"][0]["reason_codes"])
        self.assertEqual(result["ledger"]["fill_count"], 0)


class PerpHedgeModeTest(unittest.TestCase):
    def test_hedge_mode_keeps_both_legs_and_flat_closes_them(self):
        quotes = [perp_quote(i, price=100.0, volume=10_000.0) for i in range(10)]
        policy = perp_policy(position_mode=POSITION_HEDGE)
        decisions = [perp_decision(0, action=ACTION_LONG, quantity=4.0),
                     perp_decision(1, action=ACTION_SHORT, quantity=3.0)]
        result = run(quotes, decisions, policy, contracts=[perp_contract()],
                     asof_ns=T0 + 10 * MS)
        ledger = result["ledger"]
        self.assertEqual(sorted(ledger["positions"]),
                         [f"{PERP}::long", f"{PERP}::short"])
        self.assertAlmostEqual(ledger["positions"][f"{PERP}::long"]["quantity"], 4.0, places=9)
        self.assertAlmostEqual(ledger["positions"][f"{PERP}::short"]["quantity"], -3.0, places=9)
        self.assertTrue(ledger["conservation"]["ok"], ledger["conservation"]["violations"])

        flattened = run(quotes, decisions + [perp_decision(3, action=ACTION_FLAT)], policy,
                        contracts=[perp_contract()], asof_ns=T0 + 10 * MS)
        self.assertEqual(len(flattened["decisions"][2]["order_ids"]), 2)
        for entry in flattened["ledger"]["positions"].values():
            self.assertAlmostEqual(entry["quantity"], 0.0, places=9)
        self.assertTrue(flattened["ledger"]["conservation"]["ok"])

    def test_hedge_and_one_way_disagree_on_netting(self):
        quotes = [perp_quote(i, price=100.0, volume=10_000.0) for i in range(8)]
        decisions = [perp_decision(0, action=ACTION_LONG, quantity=5.0),
                     perp_decision(2, action=ACTION_SELL, quantity=2.0)]
        one_way = run(quotes, decisions, perp_policy(), contracts=[perp_contract()],
                      asof_ns=T0 + 10 * MS)["ledger"]
        hedge = run(quotes, decisions, perp_policy(position_mode=POSITION_HEDGE),
                    contracts=[perp_contract()], asof_ns=T0 + 10 * MS)["ledger"]
        self.assertAlmostEqual(one_way["positions"][PERP]["quantity"], 3.0, places=9)
        self.assertAlmostEqual(hedge["positions"][f"{PERP}::long"]["quantity"], 5.0, places=9)
        self.assertAlmostEqual(hedge["positions"][f"{PERP}::short"]["quantity"], -2.0, places=9)


class PerpLeverageAndMarginTest(unittest.TestCase):
    def moving_quotes(self, start=100.0, step=1.0, count=8, volume=10_000.0):
        return [perp_quote(i, price=start + i * step, volume=volume) for i in range(count)]

    def test_leverage_scales_the_margin_requirement_not_the_pnl(self):
        quotes = self.moving_quotes()
        decisions = [perp_decision(0, action=ACTION_LONG, quantity=10.0),
                     perp_decision(4, action=ACTION_FLAT)]
        observed = {}
        for leverage in (1.0, 5.0, 10.0, 20.0):
            result = run(quotes, decisions, perp_policy(default_leverage=leverage),
                         contracts=[perp_contract()], asof_ns=T0 + 10 * MS)
            ledger = result["ledger"]
            observed[leverage] = (ledger["net_pnl"], ledger["fills"][0]["notional"] / leverage,
                                  result["orders"][0]["required_initial_margin"])
            self.assertTrue(ledger["conservation"]["ok"])
        pnls = {round(value[0], 9) for value in observed.values()}
        self.assertEqual(len(pnls), 1)
        for leverage, (_, expected_margin, order_margin) in observed.items():
            self.assertAlmostEqual(order_margin, expected_margin, places=6)
        self.assertAlmostEqual(observed[20.0][2], observed[10.0][2] / 2.0, places=6)

    def test_leverage_above_the_contract_maximum_is_rejected(self):
        quotes = self.moving_quotes()
        decision = perp_decision(0, action=ACTION_LONG, quantity=1.0, leverage=50.0)
        result = run(quotes, [decision], perp_policy(max_leverage=20.0),
                     contracts=[perp_contract(max_leverage=20.0)], asof_ns=T0 + 10 * MS)
        self.assertEqual(result["orders"], [])
        self.assertIn(REASON_LEVERAGE_ABOVE_CONTRACT_MAX, result["decisions"][0]["reason_codes"])

    def test_margin_sufficiency_is_enforced_never_assumed(self):
        quotes = self.moving_quotes(step=0.0)
        policy = perp_policy(default_leverage=10.0, max_leverage=10.0)
        contract = perp_contract(max_leverage=10.0)

        # Requesting far more than the account can margin: reduced, not silently filled.
        reduced = run(quotes, [perp_decision(0, action=ACTION_LONG, quantity=500.0)], policy,
                      contracts=[contract], initial_cash=1_000.0, asof_ns=T0 + 10 * MS)
        record = reduced["decisions"][0]
        self.assertEqual(record["outcome"], "order_submitted")
        self.assertIn(REASON_MARGIN_REDUCE, record["reason_codes"])
        self.assertIn(REASON_MARGIN_INSUFFICIENT, record["reason_codes"])
        self.assertLessEqual(reduced["ledger"]["allocated_margin_total"], 1_000.0 + 1e-6)
        self.assertGreater(reduced["orders"][0]["filled_quantity"], 0.0)
        self.assertLess(reduced["orders"][0]["filled_quantity"], 500.0)

        # With the account fully margined, the next risk-increasing order is refused.
        full = run(quotes, [perp_decision(0, action=ACTION_LONG, quantity=100.0)], policy,
                   contracts=[contract], initial_cash=1_000.0, asof_ns=T0 + 10 * MS)
        self.assertEqual(full["counts"]["rejected"], 0)
        self.assertAlmostEqual(full["ledger"]["allocated_margin_total"], 1_000.0, places=6)
        blocked = run(quotes, [perp_decision(0, action=ACTION_LONG, quantity=100.0),
                               perp_decision(2, action=ACTION_BUY, quantity=10.0)], policy,
                      contracts=[contract], initial_cash=1_000.0, asof_ns=T0 + 10 * MS)
        self.assertEqual(blocked["decisions"][1]["outcome"], "rejected")
        self.assertIn(REASON_MARGIN_INSUFFICIENT, blocked["decisions"][1]["reason_codes"])

        # A reduce-only close never needs new margin, so it stays available at the limit.
        closed = run(quotes, [perp_decision(0, action=ACTION_LONG, quantity=100.0),
                              perp_decision(2, action=ACTION_FLAT)], policy,
                     contracts=[contract], initial_cash=1_000.0, asof_ns=T0 + 10 * MS)
        self.assertEqual(closed["decisions"][1]["outcome"], "order_submitted")
        self.assertAlmostEqual(closed["ledger"]["positions"][PERP]["quantity"], 0.0, places=9)

    def test_spot_purchases_are_limited_by_available_cash(self):
        quotes = [make_quote(i, volume=1e9) for i in range(6)]
        result = run(quotes, [make_decision(0, action=ACTION_BUY, quantity=1_000.0)],
                     instant_policy(), initial_cash=1_000.0, asof_ns=T0 + 10 * MS)
        record = result["decisions"][0]
        self.assertIn(REASON_INSUFFICIENT_CASH, record["reason_codes"])
        self.assertGreater(result["ledger"]["cash"], -1e-6)
        self.assertGreaterEqual(record["effective_quantity"], 0.0)

        strict = instant_policy(account=backtest_account_policy(reject=True))
        blocked = run(quotes, [make_decision(0, action=ACTION_BUY, quantity=1_000.0)], strict,
                      initial_cash=1_000.0, asof_ns=T0 + 10 * MS)
        self.assertEqual(blocked["orders"], [])
        self.assertIn(REASON_INSUFFICIENT_CASH, blocked["decisions"][0]["reason_codes"])

    def test_liquidation_price_moves_towards_entry_as_leverage_rises(self):
        quotes = self.moving_quotes(step=0.0, count=4)
        observed = []
        for leverage in (2.0, 5.0, 10.0, 20.0):
            ledger = run(quotes, [perp_decision(0, action=ACTION_LONG, quantity=1.0)],
                         perp_policy(default_leverage=leverage,
                                     max_leverage=max(20.0, leverage)),
                         contracts=[perp_contract(max_leverage=max(20.0, leverage))],
                         asof_ns=T0 + 10 * MS)["ledger"]
            position = ledger["positions"][PERP]
            self.assertGreater(ledger["fill_count"], 0)
            maintenance_margin_rate = 0.005
            liquidation_price = (
                (position["cost_basis"] - position["allocated_margin"])
                / (position["quantity"] * position["contract_multiplier"]
                   * (1.0 - maintenance_margin_rate)))
            self.assertLess(liquidation_price, position["mark_price"])
            observed.append(liquidation_price)
        self.assertEqual(observed, sorted(observed))
        self.assertLess(observed[0], observed[-1])

    def test_contract_multiplier_scales_notional_pnl_and_margin(self):
        quotes = self.moving_quotes()
        decisions = [perp_decision(0, action=ACTION_LONG, quantity=10.0),
                     perp_decision(4, action=ACTION_FLAT)]
        base_result = run(quotes, decisions, perp_policy(), contracts=[perp_contract()],
                          asof_ns=T0 + 10 * MS)
        base = base_result["ledger"]
        scaled = run(quotes, decisions, perp_policy(),
                     contracts=[perp_contract(contract_multiplier=3.0)], asof_ns=T0 + 10 * MS)
        ledger = scaled["ledger"]
        self.assertAlmostEqual(ledger["net_pnl"], base["net_pnl"] * 3.0, places=6)
        self.assertAlmostEqual(ledger["fills"][0]["notional"], base["fills"][0]["notional"] * 3.0,
                               places=6)
        self.assertTrue(ledger["conservation"]["ok"], ledger["conservation"]["violations"])
        self.assertAlmostEqual(scaled["orders"][0]["required_initial_margin"],
                               base_result["orders"][0]["required_initial_margin"] * 3.0,
                               places=6)


class PerpFundingTest(unittest.TestCase):
    def funding_run(self, rate, *, action=ACTION_LONG, quantity=10.0, price=100.0,
                    contracts_ns=2 * MS, ticks=12, source=FUNDING_SOURCE_POLICY_CONSTANT,
                    schedule=(), quotes_rate=None):
        quotes = [perp_quote(i, price=price, volume=10_000.0,
                             funding_rate=quotes_rate) for i in range(ticks)]
        policy = instant_policy(fees=FeePolicy(fee_bps=0.0),
                                perps=PerpPolicy(allow_short=True, default_leverage=10.0,
                                                 max_leverage=20.0, margin_mode=MARGIN_CROSS,
                                                 position_mode=POSITION_ONE_WAY,
                                                 funding_interval_ns=contracts_ns,
                                                 funding_anchor_ns=0,
                                                 funding_rate_source=source,
                                                 default_funding_rate=rate,
                                                 funding_rate_schedule=schedule))
        decisions = [perp_decision(0, action=action, quantity=quantity)]
        return run(quotes, decisions, policy, contracts=[perp_contract()],
                   asof_ns=T0 + ticks * MS)

    def test_funding_cost_is_monotone_in_the_funding_rate_for_a_long(self):
        series = []
        for rate in (0.0, 0.0001, 0.0005, 0.001):
            ledger = self.funding_run(rate)["ledger"]
            series.append((rate, ledger["funding_cost"], ledger["net_pnl"],
                           ledger["funding_payment_count"]))
            self.assertTrue(ledger["conservation"]["ok"], ledger["conservation"]["violations"])
        self.assertEqual(len({entry[3] for entry in series}), 1)
        payments = series[0][3]
        self.assertGreater(payments, 0)
        for lower, higher in zip(series, series[1:]):
            self.assertGreater(higher[1], lower[1])
            self.assertLess(higher[2], lower[2])
        self.assertAlmostEqual(series[-1][1], payments * 10.0 * 100.0 * 0.001, places=6)
        self.assertAlmostEqual(series[0][1], 0.0, places=9)

    def test_funding_is_received_by_a_short_when_the_rate_is_positive(self):
        ledger = self.funding_run(0.0005, action=ACTION_SHORT)["ledger"]
        self.assertLess(ledger["funding_cost"], 0.0)
        self.assertGreater(ledger["net_pnl"], 0.0)
        self.assertTrue(ledger["conservation"]["ok"])

    def test_declared_schedule_and_quote_field_rate_sources_are_honoured(self):
        scheduled = self.funding_run(
            0.0, source=FUNDING_SOURCE_DECLARED_SCHEDULE,
            schedule=((PERP, PERP_VENUE, T0 + 2 * MS, 0.001),))
        payments = scheduled["ledger"]["funding_payments"]
        self.assertTrue(payments)
        self.assertEqual(payments[0]["funding_rate"], 0.001)
        self.assertEqual(payments[0]["funding_rate_source"], FUNDING_SOURCE_DECLARED_SCHEDULE)
        self.assertAlmostEqual(scheduled["ledger"]["funding_cost"], 1000.0 * 0.001, places=9)
        self.assertAlmostEqual(payments[0]["funding_cost"], 1000.0 * 0.001, places=9)
        self.assertTrue(all(payment["funding_rate"] == 0.0 for payment in payments[1:]))

        quoted = self.funding_run(0.0, source=FUNDING_SOURCE_QUOTE_FIELD, quotes_rate=0.0003)
        self.assertTrue(all(payment["funding_rate"] == 0.0003
                            for payment in quoted["ledger"]["funding_payments"]))
        self.assertEqual(quoted["ledger"]["funding_payments"][0]["funding_rate_source"],
                         FUNDING_SOURCE_QUOTE_FIELD)

    def test_funding_is_recorded_as_its_own_ledger_entry_with_a_reason_code(self):
        ledger = self.funding_run(0.0005)["ledger"]
        entries = [entry for entry in ledger["cash_entries"]
                   if entry["reason"] == LEDGER_REASON_FUNDING]
        self.assertEqual(len(entries), ledger["funding_payment_count"])
        self.assertAlmostEqual(math.fsum(entry["amount"] for entry in entries),
                               -ledger["funding_cost"], places=9)
        for payment in ledger["funding_payments"]:
            self.assertEqual(payment["funding_rate_source"], FUNDING_SOURCE_POLICY_CONSTANT)
            self.assertGreater(payment["notional"], 0.0)
        self.assertEqual(REASON_FUNDING, "declared_funding_payment")


class PerpLiquidationTest(unittest.TestCase):
    def declining_quotes(self, count=40, start=100.0, factor=0.98, volume=1e9):
        price = start
        quotes = []
        for index in range(count):
            quotes.append(perp_quote(index, price=price, volume=volume))
            price *= factor
        return quotes

    def liquidation_run(self, margin_mode, *, leverage=10.0, quantity=50.0, cash=10_000.0,
                        count=40):
        quotes = self.declining_quotes(count=count)
        policy = perp_policy(default_leverage=leverage, max_leverage=leverage,
                             margin_mode=margin_mode, maintenance_margin_rate=0.005,
                             funding_interval_ns=10**15)
        contract = perp_contract(margin_mode=margin_mode, max_leverage=leverage)
        decisions = [perp_decision(0, action=ACTION_LONG, quantity=quantity)]
        return run(quotes, decisions, policy, contracts=[contract], initial_cash=cash,
                   asof_ns=T0 + count * MS + MS)

    def test_liquidation_triggers_and_writes_an_explicit_ledger_entry(self):
        result = self.liquidation_run(MARGIN_ISOLATED)
        ledger = result["ledger"]
        self.assertEqual(ledger["liquidation_count"], 1)
        record = ledger["liquidations"][0]
        self.assertEqual(record["trigger_reason_code"], REASON_LIQUIDATION)
        self.assertEqual(record["position_after"], 0.0)
        self.assertEqual(record["asset_id"], PERP)
        self.assertLess(record["ns"], T0 + 40 * MS + MS)
        self.assertTrue(ledger["conservation"]["ok"], ledger["conservation"]["violations"])
        self.assertEqual(ledger["positions"][PERP]["quantity"], 0.0)
        self.assertEqual(ledger["positions"][PERP]["allocated_margin"], 0.0)
        reasons = [entry["reason"] for entry in ledger["cash_entries"]]
        self.assertIn(LEDGER_REASON_LIQUIDATION_FEE, reasons)
        self.assertIn(record["liquidation_id"],
                      [entry["entry_id"].rsplit("::", 1)[0] for entry in ledger["cash_entries"]])
        self.assertGreaterEqual(len(ledger["liquidations"]), 1)
        self.assertTrue(result["counts"]["liquidations"] >= 1)

    def test_isolated_and_cross_margin_behave_differently(self):
        isolated = self.liquidation_run(MARGIN_ISOLATED)
        cross = self.liquidation_run(MARGIN_CROSS)
        self.assertEqual(isolated["ledger"]["liquidation_count"], 1)
        self.assertEqual(cross["ledger"]["liquidation_count"], 0)
        self.assertEqual(isolated["ledger"]["positions"][PERP]["quantity"], 0.0)
        self.assertGreater(abs(cross["ledger"]["positions"][PERP]["quantity"]), 0.0)
        self.assertGreater(isolated["ledger"]["equity"], cross["ledger"]["equity"])
        self.assertTrue(cross["ledger"]["conservation"]["ok"])

    def test_a_well_margined_position_is_not_liquidated(self):
        quotes = [perp_quote(i, price=100.0, volume=1e9) for i in range(12)]
        policy = perp_policy(default_leverage=20.0, max_leverage=20.0)
        result = run(quotes, [perp_decision(0, action=ACTION_LONG, quantity=1.0)], policy,
                     contracts=[perp_contract()], initial_cash=100_000.0,
                     asof_ns=T0 + 12 * MS)
        self.assertEqual(result["ledger"]["liquidation_count"], 0)
        self.assertGreater(result["ledger"]["allocated_margin_total"], 0.0)

    def test_funding_can_push_an_account_into_liquidation(self):
        quotes = [perp_quote(i, price=100.0, volume=1e9) for i in range(12)]
        policy = instant_policy(
            fees=FeePolicy(fee_bps=0.0),
            perps=PerpPolicy(allow_short=True, default_leverage=20.0, max_leverage=20.0,
                             margin_mode=MARGIN_ISOLATED, position_mode=POSITION_ONE_WAY,
                             maintenance_margin_rate=0.005, funding_interval_ns=1 * MS,
                             funding_anchor_ns=0, funding_rate_source=FUNDING_SOURCE_POLICY_CONSTANT,
                             default_funding_rate=0.02))
        result = run(quotes, [perp_decision(0, action=ACTION_LONG, quantity=19.0)], policy,
                     contracts=[perp_contract(margin_mode=MARGIN_ISOLATED)], initial_cash=100.0,
                     asof_ns=T0 + 11 * MS)
        self.assertGreater(result["ledger"]["funding_payment_count"], 0)
        self.assertGreaterEqual(result["ledger"]["liquidation_count"], 1)

    def test_spot_legs_are_not_liquidated_by_the_perp_margin_engine(self):
        spot_quotes = [make_quote(i, price=100.0, volume=1e9, asset="synthetic-spot")
                       for i in range(40)]
        perp_quotes = self.declining_quotes(count=40)
        decisions = [
            Decision(decision_id="spot-buy", asset_id="synthetic-spot", venue=VENUE,
                     decision_ns=T0, action=ACTION_BUY, quantity=100.0),
            perp_decision(0, action=ACTION_LONG, quantity=50.0),
        ]
        policy = perp_policy(default_leverage=10.0, max_leverage=10.0,
                             margin_mode=MARGIN_ISOLATED)
        result = run(spot_quotes + perp_quotes, decisions, policy,
                     contracts=[perp_contract(max_leverage=10.0, margin_mode=MARGIN_ISOLATED)],
                     initial_cash=50_000.0, asof_ns=T0 + 41 * MS)
        ledger = result["ledger"]
        self.assertEqual(ledger["positions"][PERP]["quantity"], 0.0)
        self.assertGreater(ledger["liquidation_count"], 0)
        self.assertGreater(ledger["positions"]["synthetic-spot"]["quantity"], 0.0)
        for record in ledger["liquidations"]:
            self.assertEqual(record["asset_id"], PERP)
        self.assertTrue(ledger["conservation"]["ok"], ledger["conservation"]["violations"])


class PerpMarkPriceTest(unittest.TestCase):
    def test_mark_and_last_price_are_distinguished_in_valuation(self):
        quotes = [perp_quote(i, price=100.0, volume=1e9, mark_price=101.0, last_price=100.0)
                  for i in range(4)]
        decision = [perp_decision(0, action=ACTION_LONG, quantity=10.0)]
        mark_ledger = run(quotes, decision, perp_policy(), contracts=[perp_contract()],
                          asof_ns=T0 + 4 * MS)["ledger"]
        self.assertAlmostEqual(mark_ledger["positions"][PERP]["mark_price"], 101.0, places=9)
        self.assertAlmostEqual(mark_ledger["unrealized_pnl"], 10.0 * (101.0 - 100.0), places=6)
        self.assertTrue(mark_ledger["conservation"]["ok"], mark_ledger["conservation"]["violations"])

        last_policy = instant_policy(marks=MarkPolicy(fallback="cost_basis",
                                                      price_source=PRICE_SOURCE_LAST))
        last_ledger = run(quotes, decision, last_policy, contracts=[perp_contract()],
                          asof_ns=T0 + 4 * MS)["ledger"]
        self.assertAlmostEqual(last_ledger["positions"][PERP]["mark_price"], 100.0, places=9)
        self.assertNotAlmostEqual(last_ledger["unrealized_pnl"], mark_ledger["unrealized_pnl"],
                                  places=6)
        self.assertTrue(last_ledger["conservation"]["ok"])

    def test_funding_uses_the_mark_price_notional(self):
        quotes = [perp_quote(i, price=100.0, volume=1e9, mark_price=110.0) for i in range(8)]
        policy = instant_policy(
            fees=FeePolicy(fee_bps=0.0),
            perps=PerpPolicy(allow_short=True, default_leverage=10.0, max_leverage=20.0,
                             margin_mode=MARGIN_CROSS, position_mode=POSITION_ONE_WAY,
                             funding_interval_ns=2 * MS, funding_anchor_ns=0,
                             funding_rate_source=FUNDING_SOURCE_POLICY_CONSTANT,
                             default_funding_rate=0.001))
        ledger = run(quotes, [perp_decision(0, action=ACTION_LONG, quantity=10.0)], policy,
                     contracts=[perp_contract()], asof_ns=T0 + 7 * MS)["ledger"]
        self.assertGreater(ledger["funding_payment_count"], 0)
        for payment in ledger["funding_payments"]:
            self.assertAlmostEqual(payment["mark_price"], 110.0, places=9)
            self.assertAlmostEqual(payment["notional"], 10.0 * 110.0, places=6)
        self.assertAlmostEqual(ledger["funding_cost"],
                               ledger["funding_payment_count"] * 1100.0 * 0.001, places=6)


class PerpRiskGuardTest(unittest.TestCase):
    def guard_context(self, **overrides):
        context = {
            "decision_ns": T0, "local_ns": T0, "last_data_ns": T0 - MS, "feed_clock_ns": T0,
            "asset_id": PERP, "venue": PERP_VENUE, "side": SIDE_BUY,
            "requested_quantity": 1.0, "reference_price": 100.0,
            "position_quantity": 0.0, "gross_exposure_notional": 0.0,
            "net_exposure_notional": 0.0, "realized_pnl": 0.0, "unrealized_pnl": 0.0,
            "fees_paid": 0.0, "net_pnl": 0.0,
            "position_notional": 0.0, "contract_multiplier": 1.0,
            "venue_position_notional": 0.0, "margin_mode": MARGIN_CROSS,
            "margin_balance": 10_000.0, "maintenance_margin_required": 0.0,
            "initial_margin_required": 500.0, "available_margin": 9_500.0,
            "margin_ratio": 0.0, "liquidation_distance_fraction": 0.5,
            "requested_leverage": 10.0, "contract_max_leverage": 20.0,
            "funding_cost_to_date": 0.0,
        }
        context.update(overrides)
        return context

    def test_leverage_limit_blocks_and_reduces(self):
        engine = risk.RiskEngine(limits=risk.RiskLimits(max_leverage=5.0))
        blocked = engine.evaluate(self.guard_context(requested_leverage=10.0))
        self.assertEqual(blocked.decision, risk.DECISION_BLOCK)
        self.assertIn(risk.REASON_LEVERAGE_LIMIT, blocked.reason_codes)

        engine = risk.RiskEngine(limits=risk.RiskLimits(max_leverage=20.0))
        reduced = engine.evaluate(self.guard_context(margin_balance=1_000.0,
                                                     gross_exposure_notional=19_950.0,
                                                     net_exposure_notional=19_950.0))
        self.assertEqual(reduced.decision, risk.DECISION_REDUCE)
        self.assertIn(risk.REASON_LEVERAGE_LIMIT, reduced.reason_codes)

        contract_capped = risk.RiskEngine(limits=risk.RiskLimits(max_leverage=100.0)).evaluate(
            self.guard_context(requested_leverage=30.0, contract_max_leverage=20.0))
        self.assertEqual(contract_capped.decision, risk.DECISION_BLOCK)
        self.assertIn(risk.REASON_CONTRACT_LEVERAGE_LIMIT, contract_capped.reason_codes)

    def test_margin_sufficiency_guard_fails_closed(self):
        engine = risk.RiskEngine(limits=risk.RiskLimits(max_leverage=10.0))
        blocked = engine.evaluate(self.guard_context(available_margin=0.0,
                                                     requested_quantity=10.0))
        self.assertEqual(blocked.decision, risk.DECISION_BLOCK)
        self.assertIn(risk.REASON_INSUFFICIENT_MARGIN, blocked.reason_codes)

        reduced = engine.evaluate(self.guard_context(available_margin=5.0,
                                                     requested_quantity=10.0,
                                                     requested_leverage=10.0))
        self.assertEqual(reduced.decision, risk.DECISION_REDUCE)
        self.assertAlmostEqual(reduced.allowed_quantity, 0.5, places=9)

        disabled = risk.RiskEngine(limits=risk.RiskLimits(require_margin_sufficiency=False))
        relaxed = disabled.evaluate(self.guard_context(available_margin=0.0,
                                                       requested_quantity=1.0))
        self.assertEqual(relaxed.decision, risk.DECISION_ALLOW)

    def test_margin_ratio_and_liquidation_distance_guards(self):
        engine = risk.RiskEngine(limits=risk.RiskLimits(max_margin_ratio=0.8,
                                                        min_liquidation_distance_fraction=0.05))
        blocked = engine.evaluate(self.guard_context(margin_ratio=0.9))
        self.assertEqual(blocked.decision, risk.DECISION_BLOCK)
        self.assertIn(risk.REASON_MARGIN_RATIO_LIMIT, blocked.reason_codes)

        distance = engine.evaluate(self.guard_context(liquidation_distance_fraction=0.01))
        self.assertEqual(distance.decision, risk.DECISION_BLOCK)
        self.assertIn(risk.REASON_LIQUIDATION_DISTANCE_LIMIT, distance.reason_codes)

        reducing = engine.evaluate(self.guard_context(
            margin_ratio=0.9, liquidation_distance_fraction=0.01, side=SIDE_SELL,
            position_quantity=5.0, net_exposure_notional=500.0,
            gross_exposure_notional=500.0, reference_price=100.0, requested_quantity=5.0))
        self.assertNotIn(risk.REASON_MARGIN_RATIO_LIMIT, reducing.reason_codes)
        self.assertNotIn(risk.REASON_LIQUIDATION_DISTANCE_LIMIT, reducing.reason_codes)

    def test_funding_cost_and_venue_position_guards(self):
        engine = risk.RiskEngine(limits=risk.RiskLimits(max_funding_cost_notional=100.0,
                                                        max_position_notional_per_venue=1_000.0))
        blocked = engine.evaluate(self.guard_context(funding_cost_to_date=150.0))
        self.assertEqual(blocked.decision, risk.DECISION_BLOCK)
        self.assertIn(risk.REASON_FUNDING_COST_LIMIT, blocked.reason_codes)

        expected = engine.evaluate(self.guard_context(funding_cost_to_date=90.0,
                                                      expected_funding_cost_notional=50.0))
        self.assertEqual(expected.decision, risk.DECISION_BLOCK)

        venue = engine.evaluate(self.guard_context(venue_position_notional=1_000.0,
                                                   requested_quantity=5.0))
        self.assertEqual(venue.decision, risk.DECISION_BLOCK)
        self.assertIn(risk.REASON_VENUE_POSITION_LIMIT, venue.reason_codes)

        partial = engine.evaluate(self.guard_context(venue_position_notional=900.0,
                                                     requested_quantity=5.0))
        self.assertEqual(partial.decision, risk.DECISION_REDUCE)
        self.assertAlmostEqual(partial.allowed_quantity, 1.0, places=9)

    def test_absent_perp_fields_leave_the_spot_guards_untouched(self):
        context = self.guard_context()
        for key in ("available_margin", "margin_ratio", "liquidation_distance_fraction",
                    "funding_cost_to_date", "venue_position_notional", "requested_leverage",
                    "margin_balance", "contract_multiplier", "position_notional",
                    "contract_max_leverage", "margin_mode", "initial_margin_required",
                    "maintenance_margin_required"):
            context.pop(key)
        decision = risk.RiskEngine().evaluate(context)
        self.assertEqual(decision.decision, risk.DECISION_ALLOW)
        for name in ("leverage", "margin_sufficiency", "margin_ratio", "liquidation_distance",
                     "funding_cost", "venue_position"):
            self.assertEqual(decision.checks[name]["decision"], risk.CHECK_NOT_EVALUATED)

    def test_short_opening_increases_projected_gross_exposure(self):
        engine = risk.RiskEngine(limits=risk.RiskLimits(
            max_gross_exposure_notional=300.0, min_position_quantity=-100.0))
        context = self.guard_context(side=SIDE_SELL, requested_quantity=5.0,
                                     position_quantity=0.0, gross_exposure_notional=0.0,
                                     net_exposure_notional=0.0)
        decision = engine.evaluate(context)
        self.assertEqual(decision.decision, risk.DECISION_REDUCE)
        self.assertIn(risk.REASON_GROSS_EXPOSURE_LIMIT, decision.reason_codes)
        self.assertAlmostEqual(decision.allowed_quantity, 3.0, places=9)
        self.assertAlmostEqual(decision.checks["gross_exposure"]["projected"], 500.0, places=9)


class BacktestHarnessTest(unittest.TestCase):
    def small_path(self, **kwargs):
        options = dict(ticks=120, tick_ns=3 * 60 * SECOND, regime_length_ticks=40,
                       drift_bps_per_tick=1.0, noise_bps_per_tick=4.0)
        options.update(kwargs)
        return backtest.build_synthetic_perp_path(2024, **options)

    def test_synthetic_path_is_deterministic_and_labelled(self):
        first = self.small_path()
        second = self.small_path()
        self.assertEqual([quote.to_dict() for quote in first.quotes],
                         [quote.to_dict() for quote in second.quotes])
        different = self.small_path()
        other = backtest.build_synthetic_perp_path(2025, ticks=120,
                                                   tick_ns=3 * 60 * SECOND,
                                                   regime_length_ticks=40,
                                                   drift_bps_per_tick=1.0, noise_bps_per_tick=4.0)
        self.assertNotEqual([quote.mid for quote in different.quotes],
                            [quote.mid for quote in other.quotes])
        labels = {window.label for window in first.regimes}
        self.assertTrue(labels)
        for label in labels:
            self.assertIn("synthetic", label)

    def test_backtest_receipt_is_byte_identical_across_runs(self):
        path = self.small_path()
        strategy = backtest.MovingAverageCrossoverStrategy(fast_ticks=4, slow_ticks=16,
                                                           target_quantity=2.0, rebalance_ticks=2)
        first = backtest.run_backtest(path, strategy=strategy, initial_cash=50_000.0)
        second = backtest.run_backtest(path, strategy=strategy, initial_cash=50_000.0)
        self.assertTrue(first["determinism"]["byte_identical"])
        self.assertEqual(first["backtest_sha256"], second["backtest_sha256"])
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
        self.assertTrue(first["determinism"]["equity_curve_identical"])
        self.assertTrue(first["determinism"]["fills_identical"])
        self.assertTrue(first["determinism"]["funding_identical"])

    def test_backtest_receipt_is_complete_and_attribution_closes(self):
        path = self.small_path()
        strategy = backtest.MovingAverageCrossoverStrategy(fast_ticks=4, slow_ticks=16,
                                                           target_quantity=2.0, rebalance_ticks=2)
        receipt = backtest.run_backtest(path, strategy=strategy, initial_cash=50_000.0)
        for key in ("fills", "orders", "decisions", "ledger", "equity_curve", "drawdown",
                    "turnover", "margin_usage", "liquidations", "funding", "costs",
                    "per_instrument", "per_regime", "determinism", "limitations",
                    "provisional_pending_r1", "code"):
            self.assertIn(key, receipt)
        self.assertGreater(receipt["counts"]["fills"], 0)
        self.assertGreater(receipt["funding"]["payment_count"], 0)
        self.assertGreater(receipt["turnover"]["total_notional"], 0.0)
        self.assertAlmostEqual(receipt["attribution"]["instrument_residual"], 0.0, places=6)
        self.assertAlmostEqual(receipt["attribution"]["regime_residual"], 0.0, places=6)
        self.assertLess(receipt["drawdown"]["max_drawdown_fraction"], 1.0)
        self.assertEqual(receipt["equity_curve"][0]["equity"], receipt["initial_cash"])
        self.assertAlmostEqual(receipt["equity_curve"][-1]["equity"],
                               receipt["ledger"]["equity"], places=6)
        self.assertGreater(len(receipt["per_instrument"]), 1)
        self.assertGreater(len(receipt["per_regime"]), 1)
        self.assertTrue(receipt["ledger"]["conservation"]["ok"])
        self.assertIn("no venue", receipt["scope"])
        self.assertIn("backtest.initial_cash", receipt["provisional_pending_r1"])
        self.assertEqual({entry["path"].rsplit("/", 1)[-1] for entry in receipt["code"]},
                         {"financial_backtest_v1.py", "financial_simulator_v1.py",
                          "financial_risk_v1.py"})
        for entry in receipt["per_instrument"].values():
            self.assertIn("net_pnl", entry)
            self.assertIn("turnover_notional", entry)

    def test_no_op_strategy_produces_an_empty_but_valid_run(self):
        receipt = backtest.run_backtest(self.small_path(), strategy=backtest.NoOpStrategy(),
                                        initial_cash=10_000.0)
        self.assertEqual(receipt["counts"]["fills"], 0)
        self.assertEqual(receipt["ledger"]["net_pnl"], 0.0)
        self.assertEqual(receipt["equity_curve"][-1]["equity"], 10_000.0)
        self.assertEqual(receipt["drawdown"]["max_drawdown_notional"], 0.0)
        self.assertEqual(receipt["turnover"]["total_notional"], 0.0)

    def test_backtest_can_be_forced_into_a_liquidation(self):
        path = backtest.build_synthetic_perp_path(11, ticks=240, tick_ns=3 * 60 * SECOND,
                                                 drift_bps_per_tick=-4.0,
                                                 noise_bps_per_tick=2.0,
                                                 regime_length_ticks=240)
        instrument = path.instruments[0]
        start_ns = min(quote.available_ns for quote in path.quotes)
        decisions = [Decision(decision_id="hold-long", asset_id=instrument[0],
                              venue=instrument[1], decision_ns=start_ns, action=ACTION_LONG,
                              quantity=100.0)]
        policy = backtest.default_execution_policy(
            leverage=20.0, max_leverage=20.0, margin_mode=MARGIN_ISOLATED,
            funding_anchor_ns=start_ns, order_time_to_live_ns=3 * path.tick_ns)
        contracts = backtest.default_contracts(path.instruments, margin_mode=MARGIN_ISOLATED,
                                               max_leverage=20.0)
        receipt = backtest.run_backtest(path, decisions=decisions, execution_policy=policy,
                                        contracts=contracts, initial_cash=20_000.0)
        self.assertGreaterEqual(receipt["counts"]["liquidations"], 1)
        self.assertTrue(receipt["ledger"]["conservation"]["ok"])
        record = receipt["liquidations"][0]
        self.assertEqual(record["trigger_reason_code"], REASON_LIQUIDATION)
        self.assertEqual(record["position_after"], 0.0)
        self.assertGreater(receipt["drawdown"]["max_drawdown_fraction"], 0.0)
        self.assertEqual(sum(entry["liquidations"]
                             for entry in receipt["per_regime"].values()),
                         receipt["counts"]["liquidations"])

    def test_backtest_with_the_perp_risk_gate_blocks_or_reduces(self):
        path = self.small_path()
        strategy = backtest.MovingAverageCrossoverStrategy(fast_ticks=4, slow_ticks=16,
                                                           target_quantity=20.0, rebalance_ticks=2)
        gate = backtest.build_default_risk_gate()
        receipt = backtest.run_backtest(path, strategy=strategy, initial_cash=20_000.0,
                                        risk_gate=gate)
        self.assertEqual(receipt["risk"]["risk_gate"], "financial_risk_v1")
        self.assertTrue(any(record["risk_decision"] in ("reduce", "block")
                            for record in receipt["decisions"]))
        self.assertTrue(receipt["ledger"]["conservation"]["ok"])
        self.assertIn("max_leverage", receipt["risk"]["limits"])
        self.assertIn("min_liquidation_distance_fraction", receipt["risk"]["limits"])

    def test_backtest_cli_writes_a_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "backtest.json"
            command = [sys.executable, str(Path(__file__).with_name("financial_backtest_v1.py")),
                       "--output", str(output), "--seed", "777", "--ticks", "120",
                       "--with-risk-limits"]
            result = subprocess.run(command, capture_output=True, text=True, timeout=900)
            self.assertEqual(result.returncode, 0, result.stderr)
            first = output.read_bytes()
            again = subprocess.run(command, capture_output=True, text=True, timeout=900)
            self.assertEqual(again.returncode, 0, again.stderr)
            self.assertEqual(first, output.read_bytes())
            receipt = json.loads(first.decode("utf-8"))
            self.assertEqual(receipt["schema_version"], "nanojev-financial-backtest-v1")
            self.assertTrue(receipt["determinism"]["byte_identical"])
            self.assertIn("synthetic", receipt["scope"].lower())
            self.assertIn("perpetual", receipt["scope"].lower())
            self.assertTrue(receipt["ledger"]["conservation"]["ok"])


def perp_quote(index, *, price=100.0, volume=10_000.0, delay_ns=0, asset=PERP, venue=PERP_VENUE,
               mark_price=None, last_price=None, funding_rate=None):
    return Quote(asset_id=asset, venue=venue, event_ns=T0 + index * MS,
                 available_ns=T0 + index * MS + delay_ns,
                 bid=price - price * 0.0001, ask=price + price * 0.0001, volume=volume,
                 mark_price=mark_price, last_price=last_price, funding_rate=funding_rate)


def perp_decision(index, *, action=ACTION_LONG, quantity=None, decision_ns=None, **kwargs):
    if quantity is None:
        quantity = 0.0 if action == ACTION_FLAT else 1.0
    return Decision(decision_id=f"p{index}-{action}", asset_id=PERP, venue=PERP_VENUE,
                    decision_ns=T0 + index * MS if decision_ns is None else decision_ns,
                    action=action, quantity=quantity, **kwargs)


def perp_contract(**overrides):
    options = dict(asset_id=PERP, venue=PERP_VENUE, instrument_type=INSTRUMENT_PERP,
                   contract_multiplier=1.0, tick_size=0.0, lot_size=0.0, min_notional=0.0,
                   margin_mode=MARGIN_CROSS, max_leverage=20.0, maintenance_margin_rate=0.005)
    options.update(overrides)
    return ContractSpec(**options)


def perp_policy(**overrides):
    options = dict(allow_short=True, default_leverage=10.0, max_leverage=20.0,
                   margin_mode=MARGIN_CROSS, position_mode=POSITION_ONE_WAY,
                   maintenance_margin_rate=0.005, funding_interval_ns=10**15,
                   funding_rate_source=FUNDING_SOURCE_POLICY_CONSTANT, default_funding_rate=0.0)
    options.update(overrides)
    return instant_policy(fees=FeePolicy(fee_bps=0.0), spread=SpreadPolicy(half_spread_bps=0.0),
                          perps=PerpPolicy(**options))


def backtest_account_policy(reject=False):
    from financial_simulator_v1 import AccountPolicy
    return AccountPolicy(require_sufficient_cash=True,
                         on_insufficient_cash="reject" if reject else "reduce")


def run(quotes, decisions, policy=None, *, seed=7, asof_ns=None, contracts=(), **kwargs):
    from financial_simulator_v1 import replay
    if asof_ns is None:
        asof_ns = T0 + 100 * MS
    return replay(quotes, decisions, asof_ns=asof_ns, policy=policy or instant_policy(),
                  seed=seed, contracts=contracts, **kwargs)


if __name__ == "__main__":
    unittest.main()
