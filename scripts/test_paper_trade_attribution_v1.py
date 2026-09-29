"""Tests for ``paper_trade_attribution_v1`` against the real simulator ledger.

The attribution layer is an independent accounting reconciliation: it rebuilds the
executed-price/signal PnL from signed mid-price cash flows and the terminal marked
inventory, then checks every residual against the ledger aggregates. These tests
exercise that contract with ledger objects produced by
``financial_simulator_v1.Ledger`` (and one full ``replay`` fixture), never with a
hand-rolled fake ledger, so a source change that breaks the integration fails here.

"""
import copy
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import financial_simulator_v1 as sim
import paper_trade_attribution_v1 as pta


# --------------------------------------------------------------------------
# Fixture helpers built on the real simulator ledger/fill objects
# --------------------------------------------------------------------------
def _quote(asset_id="BTC", venue="binance", event_ns=1000, bid=100.0, ask=100.0):
    return sim.Quote(asset_id=asset_id, venue=venue, event_ns=event_ns,
                     available_ns=event_ns, bid=bid, ask=ask, volume=1.0e9)


def _fill(ledger, *, order_id, side, quantity, mid, price, asset_id="BTC",
          venue="binance", execution_ns=1000, fee=0.0, spread_cost=0.0,
          slippage_cost=0.0, is_liquidation=False, liquidation_id=None,
          fill_index=0):
    return ledger.apply_fill(
        order_id=order_id, fill_index=fill_index, asset_id=asset_id, venue=venue,
        side=side, quantity=quantity, execution_ns=execution_ns,
        quote=_quote(asset_id, venue, execution_ns), mid=mid, execution_price=price,
        spread_cost=spread_cost, slippage_cost=slippage_cost, fee=fee, limit_price=None,
        is_liquidation=is_liquidation, liquidation_id=liquidation_id)


def _marks(prices, venue="binance"):
    return {key: {"price": float(price), "ns": 0, "source": "test_quote", "venue": venue}
            for key, price in prices.items()}


def _order(order_id, asset_id, side, requested, filled, remaining=None, status="filled"):
    if remaining is None:
        remaining = requested - filled
    return {"order_id": order_id, "asset_id": asset_id, "side": side, "status": status,
            "requested_quantity": float(requested), "filled_quantity": float(filled),
            "remaining_quantity": float(remaining), "selection_reason_codes": []}


def _evidence(ledger, orders):
    return pta.evidence_from_replay({"ledger": ledger, "orders": orders})


def _single_fill_fixture():
    """One long fill of one contract, fully attributed and reconciled."""
    ledger = sim.Ledger(100_000.0, multipliers={"BTC": 1.0})
    _fill(ledger, order_id="o1", side="buy", quantity=1.0, mid=100.0, price=100.0)
    ledger_dict = ledger.to_dict(_marks({"BTC": 100.0}))
    evidence = _evidence(ledger_dict, [_order("o1", "BTC", "buy", 1.0, 1.0)])
    return ledger_dict, evidence


class NumberValidationTests(unittest.TestCase):
    def test_accepts_finite_plain_numbers(self):
        self.assertEqual(pta.number(3), 3.0)
        self.assertEqual(pta.number(2.5), 2.5)
        self.assertIsInstance(pta.number(1), float)

    def test_rejects_bool_nan_inf_and_non_numbers(self):
        for bad in (True, False, float("nan"), float("inf"), float("-inf"),
                    "1", None, [1], {}, object()):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    pta.number(bad)


class EvidenceProjectionTests(unittest.TestCase):
    def test_projects_the_declared_fields(self):
        ledger = sim.Ledger(100_000.0, multipliers={"BTC": 1.0})
        _fill(ledger, order_id="o1", side="buy", quantity=2.0, mid=100.0, price=100.0)
        # Funding cost (-2) and a liquidation fee (-3) both live in the ledger.
        ledger.apply_funding(payment_id="fp1", asset_id="BTC", venue="binance",
                             position_key="BTC", ns=2000, quantity=2.0, multiplier=1.0,
                             mark_price=100.0, funding_rate=0.01,
                             rate_source=sim.REASON_FUNDING)
        ledger.apply_liquidation_fee(liquidation_id="lq1", asset_id="BTC", ns=3000, fee=3.0)
        ledger_dict = ledger.to_dict(_marks({"BTC": 110.0}))
        orders = [_order("o1", "BTC", "buy", 2.0, 2.0)]

        evidence = _evidence(ledger_dict, orders)

        self.assertEqual(evidence["schema_version"], pta.SCHEMA)
        self.assertEqual(evidence["initial_inventory"], "flat")
        self.assertEqual(evidence["external_cash_flows"], 0.0)
        self.assertIs(evidence["fills"], ledger_dict["fills"])
        self.assertEqual(evidence["funding_cash_flow"], -2.0)
        self.assertEqual(evidence["funding_payment_count"], 1)
        self.assertEqual(evidence["liquidation_fee_cash_flow"], -3.0)
        self.assertEqual(
            set(evidence["orders"][0]),
            {"order_id", "asset_id", "side", "status", "requested_quantity",
             "filled_quantity", "remaining_quantity", "selection_reason_codes"})

        result = pta.attribute(ledger_dict, evidence)
        self.assertTrue(result["reconciled"])
        self.assertEqual(result["components"]["funding"], -2.0)
        self.assertEqual(result["components"]["liquidation_fees"], -3.0)


class IndependentPriceAttributionTests(unittest.TestCase):
    """Price/signal PnL is checked against hand-computed mid-price algebra.

    The expected values below are derived from the signed position segments and the
    mid prices only; they are deliberately *not* reconstructed as
    ``net_pnl + costs`` so the test would still fail if both sides drifted together.
    """

    def test_round_trip_with_multiplier_not_one(self):
        multiplier = 10.0
        ledger = sim.Ledger(1_000_000.0, multipliers={"BTC": multiplier})
        _fill(ledger, order_id="o1", side="buy", quantity=2.0, mid=100.0, price=105.0)
        _fill(ledger, order_id="o2", side="sell", quantity=2.0, mid=110.0, price=115.0,
              execution_ns=2000)
        ledger_dict = ledger.to_dict(_marks({"BTC": 110.0}))
        orders = [_order("o1", "BTC", "buy", 2.0, 2.0),
                  _order("o2", "BTC", "sell", 2.0, 2.0)]

        result = pta.attribute(ledger_dict, _evidence(ledger_dict, orders))

        # Independent algebra: flat -> long 2 -> flat, holding across a 10 mid move.
        expected_price_pnl = 2.0 * multiplier * (110.0 - 100.0)
        self.assertEqual(result["components"]["price_signal_on_executed_path"],
                         expected_price_pnl)
        self.assertEqual(result["components"]["fees"], 0.0)
        self.assertEqual(result["components"]["funding"], 0.0)
        self.assertEqual(result["components"]["spread"], 0.0)
        self.assertEqual(result["components"]["slippage"], 0.0)
        self.assertEqual(result["components"]["tick_rounding"], 0.0)
        self.assertTrue(result["reconciled"])

    def test_buy_then_sell_across_a_flip(self):
        multiplier = 2.0
        ledger = sim.Ledger(1_000_000.0, multipliers={"BTC": multiplier})
        _fill(ledger, order_id="o1", side="buy", quantity=5.0, mid=100.0, price=100.0)
        _fill(ledger, order_id="o2", side="sell", quantity=8.0, mid=120.0, price=120.0,
              execution_ns=2000)
        ledger_dict = ledger.to_dict(_marks({"BTC": 110.0}))
        orders = [_order("o1", "BTC", "buy", 5.0, 5.0),
                  _order("o2", "BTC", "sell", 8.0, 8.0)]

        result = pta.attribute(ledger_dict, _evidence(ledger_dict, orders))

        self.assertAlmostEqual(ledger_dict["positions"]["BTC"]["quantity"], -3.0)
        # Independent algebra: close long 5 from 100->120 plus short 3 from 120->110.
        expected_price_pnl = (5.0 * multiplier * (120.0 - 100.0)
                              + 3.0 * multiplier * (120.0 - 110.0))
        self.assertEqual(result["components"]["price_signal_on_executed_path"],
                         expected_price_pnl)
        self.assertEqual(ledger_dict["realized_pnl"], 5.0 * multiplier * (120.0 - 100.0))
        self.assertTrue(result["reconciled"])

    def test_sell_then_buy_reverse_flip(self):
        ledger = sim.Ledger(1_000_000.0, multipliers={"BTC": 1.0})
        _fill(ledger, order_id="o1", side="sell", quantity=4.0, mid=100.0, price=100.0)
        _fill(ledger, order_id="o2", side="buy", quantity=6.0, mid=90.0, price=90.0,
              execution_ns=2000)
        ledger_dict = ledger.to_dict(_marks({"BTC": 95.0}))
        orders = [_order("o1", "BTC", "sell", 4.0, 4.0),
                  _order("o2", "BTC", "buy", 6.0, 6.0)]

        result = pta.attribute(ledger_dict, _evidence(ledger_dict, orders))

        self.assertAlmostEqual(ledger_dict["positions"]["BTC"]["quantity"], 2.0)
        # Short 4 closed 100->90, then long 2 opened at 90 marked at 95.
        expected_price_pnl = 4.0 * 1.0 * (100.0 - 90.0) + 2.0 * 1.0 * (95.0 - 90.0)
        self.assertEqual(result["components"]["price_signal_on_executed_path"],
                         expected_price_pnl)
        self.assertTrue(result["reconciled"])


class CostComponentTests(unittest.TestCase):
    def test_signed_funding_cost_on_a_long(self):
        ledger = sim.Ledger(100_000.0, multipliers={"BTC": 1.0})
        _fill(ledger, order_id="o1", side="buy", quantity=1.0, mid=100.0, price=100.0)
        ledger.apply_funding(payment_id="fp1", asset_id="BTC", venue="binance",
                             position_key="BTC", ns=2000, quantity=1.0, multiplier=1.0,
                             mark_price=100.0, funding_rate=0.01,
                             rate_source=sim.REASON_FUNDING)
        ledger_dict = ledger.to_dict(_marks({"BTC": 100.0}))
        orders = [_order("o1", "BTC", "buy", 1.0, 1.0)]

        result = pta.attribute(ledger_dict, _evidence(ledger_dict, orders))

        # A long pays when the rate is positive: the signed cash flow is a cost.
        expected_funding_cash = -1.0 * 1.0 * 100.0 * 0.01
        self.assertEqual(result["components"]["funding"], expected_funding_cash)
        self.assertEqual(result["components"]["funding"], -1.0)
        self.assertTrue(result["reconciled"])

    def test_signed_funding_income_on_a_short(self):
        ledger = sim.Ledger(100_000.0, multipliers={"BTC": 1.0})
        _fill(ledger, order_id="o1", side="sell", quantity=1.0, mid=100.0, price=100.0)
        ledger.apply_funding(payment_id="fp1", asset_id="BTC", venue="binance",
                             position_key="BTC", ns=2000, quantity=-1.0, multiplier=1.0,
                             mark_price=100.0, funding_rate=0.01,
                             rate_source=sim.REASON_FUNDING)
        ledger_dict = ledger.to_dict(_marks({"BTC": 100.0}))
        orders = [_order("o1", "BTC", "sell", 1.0, 1.0)]

        result = pta.attribute(ledger_dict, _evidence(ledger_dict, orders))

        # A short receives when the rate is positive: the signed cash flow is income.
        expected_funding_cash = -(-1.0) * 1.0 * 100.0 * 0.01
        self.assertEqual(result["components"]["funding"], expected_funding_cash)
        self.assertEqual(result["components"]["funding"], 1.0)
        self.assertTrue(result["reconciled"])

    def test_tick_rounding_is_the_shortfall_after_spread_and_slippage(self):
        ledger = sim.Ledger(100_000.0, multipliers={"X": 1.0})
        _fill(ledger, order_id="o1", side="buy", quantity=1.0, mid=100.0, price=101.0,
              asset_id="X", spread_cost=0.5, slippage_cost=0.2)
        ledger_dict = ledger.to_dict(_marks({"X": 101.0}))
        orders = [_order("o1", "X", "buy", 1.0, 1.0)]

        result = pta.attribute(ledger_dict, _evidence(ledger_dict, orders))

        shortfall = 1.0 * 1.0 * (101.0 - 100.0)
        self.assertEqual(result["components"]["spread"], -0.5)
        self.assertEqual(result["components"]["slippage"], -0.2)
        self.assertAlmostEqual(result["components"]["tick_rounding"],
                               -(shortfall - 0.5 - 0.2))
        self.assertAlmostEqual(result["components"]["tick_rounding"], -0.3)
        self.assertTrue(result["reconciled"])

    def test_liquidation_fill_and_liquidation_fee(self):
        ledger = sim.Ledger(100_000.0, multipliers={"BTC": 1.0})
        _fill(ledger, order_id="o1", side="buy", quantity=2.0, mid=100.0, price=100.0)
        # A liquidation fill is booked in the ledger but is not tied to submitted orders.
        _fill(ledger, order_id="liquidation::0::order", side="sell", quantity=2.0, mid=90.0,
              price=90.0, execution_ns=2000, is_liquidation=True, liquidation_id="lq1")
        ledger.apply_liquidation_fee(liquidation_id="lq1", asset_id="BTC", ns=2000, fee=5.0)
        ledger_dict = ledger.to_dict(_marks({"BTC": 90.0}))
        orders = [_order("o1", "BTC", "buy", 2.0, 2.0)]

        result = pta.attribute(ledger_dict, _evidence(ledger_dict, orders))

        # Independent algebra: long 2 liquidated from mid 100 to mid 90.
        self.assertEqual(result["components"]["price_signal_on_executed_path"], -20.0)
        self.assertEqual(result["components"]["liquidation_fees"], -5.0)
        self.assertEqual(result["sum"], -25.0)
        self.assertTrue(result["reconciled"])


class UnfilledRemainderTests(unittest.TestCase):
    def test_submitted_remainders_aggregate_per_asset_without_booked_pnl(self):
        ledger = sim.Ledger(100_000.0, multipliers={"BTC": 1.0})
        _fill(ledger, order_id="o1", side="buy", quantity=4.0, mid=100.0, price=100.0)
        ledger_dict = ledger.to_dict(_marks({"BTC": 100.0}))
        orders = [_order("o1", "BTC", "buy", 10.0, 4.0, remaining=6.0),
                  _order("o2", "BTC", "buy", 3.0, 0.0, remaining=3.0, status="expired")]

        result = pta.attribute(ledger_dict, _evidence(ledger_dict, orders))

        self.assertEqual(result["unfilled"]["remaining_quantity_by_asset"], {"BTC": 9.0})
        self.assertEqual(result["unfilled"]["booked_effect"], 0.0)
        self.assertIsNone(result["unfilled"]["counterfactual_pnl"])
        self.assertEqual(result["components"]["unfilled_booked_effect"], 0.0)
        self.assertTrue(result["reconciled"])


class IdentityValidationTests(unittest.TestCase):
    def test_missing_and_duplicate_fill_identity_rejected(self):
        ledger_dict, evidence = _single_fill_fixture()

        missing = copy.deepcopy(evidence)
        missing["fills"][0]["fill_id"] = None
        with self.assertRaises(ValueError):
            pta.attribute(ledger_dict, missing)

        non_string = copy.deepcopy(evidence)
        non_string["fills"][0]["fill_id"] = 7
        with self.assertRaises(ValueError):
            pta.attribute(ledger_dict, non_string)

        duplicate = copy.deepcopy(evidence)
        duplicate["fills"].append(copy.deepcopy(duplicate["fills"][0]))
        with self.assertRaises(ValueError):
            pta.attribute(ledger_dict, duplicate)

    def test_missing_and_duplicate_order_identity_rejected(self):
        ledger_dict, evidence = _single_fill_fixture()

        missing = copy.deepcopy(evidence)
        missing["orders"][0]["order_id"] = None
        with self.assertRaises(ValueError):
            pta.attribute(ledger_dict, missing)

        duplicate = copy.deepcopy(evidence)
        duplicate["orders"].append(copy.deepcopy(duplicate["orders"][0]))
        with self.assertRaises(ValueError):
            pta.attribute(ledger_dict, duplicate)

    def test_orphan_fill_without_order_evidence_rejected(self):
        ledger_dict, evidence = _single_fill_fixture()

        orphan = copy.deepcopy(evidence)
        orphan["fills"][0]["order_id"] = "ghost-order"
        with self.assertRaises(ValueError):
            pta.attribute(ledger_dict, orphan)

    def test_missing_terminal_position_rejected(self):
        ledger_dict, evidence = _single_fill_fixture()

        missing_position = copy.deepcopy(evidence)
        missing_position["positions"] = {}
        with self.assertRaises(ValueError):
            pta.attribute(ledger_dict, missing_position)


class NonFiniteAndBoolEvidenceTests(unittest.TestCase):
    def test_nan_bool_and_non_finite_inputs_rejected(self):
        ledger_dict, evidence = _single_fill_fixture()

        cases = {
            "external_cash_flows_bool": ("external_cash_flows", True),
            "quantity_bool": ("fill", ("quantity", True)),
            "mid_nan": ("fill", ("mid_price", float("nan"))),
            "price_inf": ("fill", ("execution_price", float("inf"))),
            "multiplier_false": ("fill", ("contract_multiplier", False)),
            "funding_nan": ("funding_cash_flow", float("nan")),
            "liquidation_bool": ("liquidation_fee_cash_flow", True),
        }
        for name, spec in cases.items():
            with self.subTest(case=name):
                broken = copy.deepcopy(evidence)
                if spec[0] == "fill":
                    field, value = spec[1]
                    broken["fills"][0][field] = value
                else:
                    broken[spec[0]] = spec[1]
                with self.assertRaises(ValueError):
                    pta.attribute(ledger_dict, broken)

    def test_schema_and_flat_start_preconditions(self):
        ledger_dict, evidence = _single_fill_fixture()

        bad_schema = copy.deepcopy(evidence)
        bad_schema["schema_version"] = "not-the-schema"
        with self.assertRaises(ValueError):
            pta.attribute(ledger_dict, bad_schema)

        not_flat = copy.deepcopy(evidence)
        not_flat["initial_inventory"] = "short"
        with self.assertRaises(ValueError):
            pta.attribute(ledger_dict, not_flat)

        external_cash = copy.deepcopy(evidence)
        external_cash["external_cash_flows"] = 1.0
        with self.assertRaises(ValueError):
            pta.attribute(ledger_dict, external_cash)

        failed_conservation = copy.deepcopy(evidence)
        failed_conservation["conservation"] = {"ok": False}
        with self.assertRaises(ValueError):
            pta.attribute(ledger_dict, failed_conservation)


class LedgerTamperingTests(unittest.TestCase):
    def test_tampered_ledger_aggregates_break_reconciliation(self):
        ledger_dict, evidence = _single_fill_fixture()

        for field in ("net_pnl", "equity", "cash", "initial_cash", "fees_paid", "funding_cost",
                      "liquidation_fees_paid", "gross_traded_notional", "fill_count",
                      "funding_payment_count"):
            with self.subTest(field=field):
                tampered = copy.deepcopy(ledger_dict)
                tampered[field] = tampered[field] + 5.0
                with self.assertRaises(ValueError):
                    pta.attribute(tampered, copy.deepcopy(evidence))

    def test_tampered_fill_flow_breaks_reconciliation(self):
        ledger_dict, evidence = _single_fill_fixture()

        for field, delta in (("execution_price", 1.0), ("quantity", 1.0), ("notional", 1.0)):
            with self.subTest(field=field):
                tampered = copy.deepcopy(evidence)
                tampered["fills"][0][field] += delta
                with self.assertRaises(ValueError):
                    pta.attribute(ledger_dict, copy.deepcopy(tampered))

    def test_mid_shift_redistributes_price_and_tick_but_preserves_total(self):
        # The executed path is defined by the execution price and terminal mark; a
        # shifted mid moves value between the price signal and tick rounding while
        # leaving the reconciled total unchanged. This documents (and pins) that the
        # mid price is a split reference, not an independently cross-checked input.
        ledger_dict, evidence = _single_fill_fixture()
        baseline = pta.attribute(ledger_dict, copy.deepcopy(evidence))

        shifted = copy.deepcopy(evidence)
        shifted["fills"][0]["mid_price"] += 1.0
        result = pta.attribute(ledger_dict, shifted)

        self.assertAlmostEqual(result["sum"], baseline["sum"])
        self.assertAlmostEqual(result["components"]["price_signal_on_executed_path"],
                               baseline["components"]["price_signal_on_executed_path"] - 1.0)
        self.assertAlmostEqual(result["components"]["tick_rounding"],
                               baseline["components"]["tick_rounding"] + 1.0)
        self.assertTrue(result["reconciled"])

    def test_tampered_position_market_value_breaks_reconciliation(self):
        ledger_dict, evidence = _single_fill_fixture()

        tampered = copy.deepcopy(evidence)
        tampered["positions"]["BTC"]["market_value"] += 1.0
        with self.assertRaises(ValueError):
            pta.attribute(ledger_dict, tampered)


class RealReplayIntegrationTests(unittest.TestCase):
    """A full ``replay`` fixture must reconcile through the attribution layer."""

    ORIGIN_NS = 1_760_000_000_000_000_000
    TICK_NS = 1_000_000

    def _scenario(self):
        quotes = []
        for index in range(12):
            event_ns = self.ORIGIN_NS + index * self.TICK_NS
            mid = 100.0 + index * 0.5
            quotes.append(sim.Quote(asset_id="BTC-PERP", venue="binance", event_ns=event_ns,
                                    available_ns=event_ns, bid=mid - 0.01, ask=mid + 0.01,
                                    volume=1.0e6))
        decisions = [
            sim.Decision(decision_id="d1", asset_id="BTC-PERP", venue="binance",
                         decision_ns=self.ORIGIN_NS + 3 * self.TICK_NS,
                         action=sim.ACTION_BUY, quantity=3.0),
            sim.Decision(decision_id="d2", asset_id="BTC-PERP", venue="binance",
                         decision_ns=self.ORIGIN_NS + 6 * self.TICK_NS,
                         action=sim.ACTION_SELL, quantity=3.0),
            # A limit far below the market never executes and leaves a remainder.
            sim.Decision(decision_id="d3", asset_id="BTC-PERP", venue="binance",
                         decision_ns=self.ORIGIN_NS + 8 * self.TICK_NS,
                         action=sim.ACTION_BUY, quantity=4.0, limit_price=1.0),
        ]
        policy = sim.ExecutionPolicy(
            fees=sim.FeePolicy(fee_bps=2.0),
            fills=sim.FillPolicy(reject_probability=0.0, fill_probability=1.0,
                                 order_time_to_live_ns=10_000_000),
            capacity=sim.CapacityPolicy(max_participation_fraction=0.01),
            perps=sim.PerpPolicy(funding_interval_ns=10 ** 18))
        contract = sim.ContractSpec(asset_id="BTC-PERP", venue="binance",
                                    contract_multiplier=2.0, tick_size=0.01)
        asof_ns = self.ORIGIN_NS + 11 * self.TICK_NS
        return sim.replay(quotes, decisions, asof_ns=asof_ns, policy=policy,
                          contracts=(contract,), replay_id="attribution-fixture")

    def test_replay_reconciles_and_price_pnl_is_independent(self):
        replay = self._scenario()
        ledger_dict = replay["ledger"]
        self.assertEqual(ledger_dict["fill_count"], 2)

        evidence = pta.evidence_from_replay(replay)
        result = pta.attribute(ledger_dict, evidence)

        self.assertTrue(result["reconciled"])
        self.assertLessEqual(result["max_abs_residual"], pta.TOLERANCE)
        self.assertAlmostEqual(result["sum"], result["net_pnl"], places=6)

        fills = ledger_dict["fills"]
        self.assertEqual([fill["side"] for fill in fills], ["buy", "sell"])
        buy, sell = fills
        # Independent algebra on the executed path: buy 3, sell 3, flat terminal.
        expected_price_pnl = (buy["quantity"] * buy["contract_multiplier"]
                              * (sell["mid_price"] - buy["mid_price"]))
        self.assertAlmostEqual(result["components"]["price_signal_on_executed_path"],
                               expected_price_pnl, places=9)
        self.assertAlmostEqual(result["components"]["fees"],
                               -ledger_dict["fees_paid"], places=9)

        # The unexecuted third order surfaces as a per-asset remainder only.
        self.assertEqual(result["unfilled"]["remaining_quantity_by_asset"],
                         {"BTC-PERP": 4.0})
        self.assertEqual(result["unfilled"]["booked_effect"], 0.0)


if __name__ == "__main__":
    unittest.main()
