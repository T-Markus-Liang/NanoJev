"""Executed-path accounting, not a cost-free strategy counterfactual.

Zero initial inventory and no external cash flows are required. Price/signal is
reconstructed from signed mid-price cash flows and terminal marked inventory,
independently of the reported net PnL. Unfilled orders have no booked PnL;
their opportunity cost is unknown without a matched counterfactual replay.
"""
import math

SCHEMA = "nanojev-paper-execution-evidence-v1"
TOLERANCE = 1e-6


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("accounting evidence requires finite numbers, not booleans")
    return float(value)


def evidence_from_replay(result):
    ledger = result["ledger"]
    return {
        "schema_version": SCHEMA,
        "initial_inventory": "flat",
        "external_cash_flows": 0.0,
        "fills": ledger["fills"],  # includes liquidation fills, unlike order-only fills
        "positions": ledger["positions"],
        "orders": [{k: order[k] for k in (
            "order_id", "asset_id", "side", "status", "requested_quantity",
            "filled_quantity", "remaining_quantity", "selection_reason_codes")}
                   for order in result["orders"]],
        "funding_cash_flow": math.fsum(p["cash_flow"] for p in ledger["funding_payments"]),
        "funding_payment_count": len(ledger["funding_payments"]),
        "liquidation_fee_cash_flow": math.fsum(
            e["amount"] for e in ledger["cash_entries"] if e["reason"] == "liquidation_fee"),
        "conservation": ledger["conservation"],
    }


def attribute(ledger, evidence):
    if evidence["schema_version"] != SCHEMA:
        raise ValueError("unsupported accounting evidence schema")
    if evidence["initial_inventory"] != "flat" or number(evidence["external_cash_flows"]) != 0:
        raise ValueError("only flat-start, zero-external-flow replay is supported")
    if evidence["conservation"]["ok"] is not True:
        raise ValueError("simulator conservation failed")
    mid_flows, shortfalls, spreads, slippages, fees, notionals = [], [], [], [], [], []
    signed_quantities, by_order, ids, multipliers, order_metadata = {}, {}, set(), {}, {}
    for fill in evidence["fills"]:
        fid = fill["fill_id"]
        if not isinstance(fid, str) or not fid or fid in ids:
            raise ValueError("missing or duplicate fill identity")
        ids.add(fid)
        if fill["side"] not in ("buy", "sell"):
            raise ValueError("invalid fill side")
        sign = 1 if fill["side"] == "buy" else -1
        qty, mult = number(fill["quantity"]), number(fill["contract_multiplier"])
        mid, price = number(fill["mid_price"]), number(fill["execution_price"])
        if min(qty, mult, mid, price) <= 0:
            raise ValueError("fill quantities, multipliers and prices must be positive")
        key = fill["position_key"]
        if key in multipliers and multipliers[key] != mult:
            raise ValueError("inconsistent contract multiplier")
        multipliers[key] = mult
        signed_quantities.setdefault(key, []).append(sign * qty)
        if not isinstance(fill["is_liquidation"], bool):
            raise ValueError("invalid liquidation flag")
        if not fill["is_liquidation"]:
            by_order.setdefault(fill["order_id"], []).append(qty)
            metadata = (fill["asset_id"], fill["side"])
            if fill["order_id"] in order_metadata and order_metadata[fill["order_id"]] != metadata:
                raise ValueError("inconsistent order fill metadata")
            order_metadata[fill["order_id"]] = metadata
        mid_flows.append(sign * qty * mult * mid)
        shortfalls.append(sign * qty * mult * (price - mid))
        spreads.append(number(fill["spread_cost"]) * mult)
        slippages.append(number(fill["slippage_cost"]) * mult)
        fees.append(number(fill["explicit_fee"]))
        if min(spreads[-1], slippages[-1], fees[-1]) < 0:
            raise ValueError("negative costs unsupported by this simulator")
        if abs(number(fill["notional"]) - qty * mult * price) > TOLERANCE:
            raise ValueError("fill notional mismatch")
        notionals.append(qty * mult * price)
    residuals = {}
    terminal_values = []
    positions = evidence["positions"]
    if set(signed_quantities) - set(positions):
        raise ValueError("missing terminal position")
    for key, position in positions.items():
        qty = number(position["quantity"])
        mult, mark = number(position["contract_multiplier"]), number(position["mark_price"])
        if mult <= 0 or mark < 0 or (key in multipliers and multipliers[key] != mult):
            raise ValueError("invalid terminal valuation")
        terminal_values.append(qty * mult * mark)
        residuals[f"position:{key}"] = qty - math.fsum(signed_quantities.get(key, []))
        residuals[f"market_value:{key}"] = number(position["market_value"]) - qty * mult * mark
    remainders, order_ids = {}, set()
    for order in evidence["orders"]:
        oid = order["order_id"]
        if not isinstance(oid, str) or not oid or oid in order_ids:
            raise ValueError("missing or duplicate order identity")
        order_ids.add(oid)
        if oid in order_metadata and order_metadata[oid] != (order["asset_id"], order["side"]):
            raise ValueError("order/fill metadata mismatch")
        requested, filled, remaining = (number(order[k]) for k in (
            "requested_quantity", "filled_quantity", "remaining_quantity"))
        if min(requested, filled, remaining) < 0:
            raise ValueError("negative order quantity")
        residuals[f"order:{oid}"] = requested - filled - remaining
        residuals[f"order_fills:{oid}"] = filled - math.fsum(by_order.get(oid, []))
        if remaining > TOLERANCE:
            # Asset units cannot be summed across assets. Do not infer cause from quantity.
            remainders.setdefault(order["asset_id"], []).append(remaining)
    if set(by_order) - order_ids:
        raise ValueError("orphan fill has no order evidence")
    spread, slippage = math.fsum(spreads), math.fsum(slippages)
    components = {
        "price_signal_on_executed_path": math.fsum(terminal_values) - math.fsum(mid_flows),
        "fees": -number(ledger["fees_paid"]),
        "funding": -number(ledger["funding_cost"]),
        "spread": -spread,
        "slippage": -slippage,
        "tick_rounding": -(math.fsum(shortfalls) - spread - slippage),
        "liquidation_fees": -number(ledger["liquidation_fees_paid"]),
        "unfilled_booked_effect": 0.0,
    }
    total = math.fsum(components.values())
    net = number(ledger["net_pnl"])
    residuals.update({
        "net": total - net,
        "equity": total - (number(ledger["equity"]) - number(ledger["initial_cash"])),
        "cash_and_terminal_value": number(ledger["cash"]) + math.fsum(terminal_values) - number(ledger["equity"]),
        "fees": math.fsum(fees) - number(ledger["fees_paid"]),
        "notional": math.fsum(notionals) - number(ledger["gross_traded_notional"]),
        "fill_count": len(ids) - number(ledger["fill_count"]),
        "funding": number(evidence["funding_cash_flow"]) + number(ledger["funding_cost"]),
        "funding_count": number(evidence["funding_payment_count"]) - number(ledger["funding_payment_count"]),
        "liquidation_fee": number(evidence["liquidation_fee_cash_flow"]) + number(ledger["liquidation_fees_paid"]),
    })
    if any(abs(number(v)) > TOLERANCE for v in residuals.values()):
        raise ValueError(f"attribution reconciliation failed: {residuals}")
    return {
        "method": "terminal marked inventory minus signed mid-price fill flows; fixed executed path",
        "components": components, "sum": total, "net_pnl": net,
        "absolute_tolerance": TOLERANCE, "max_abs_residual": max(map(abs, residuals.values())),
        "residuals": residuals, "reconciled": True,
        "unfilled": {"remaining_quantity_by_asset": {k: math.fsum(v) for k, v in remainders.items()},
                     "booked_effect": 0.0, "counterfactual_pnl": None,
                     "cause": "not inferred; see order status/reason codes",
                     "caveat": "submitted-order remainder only; pre-submission rejections require decision evidence"},
        "caveat": "not a no-cost strategy replay: costs may change later margin/fill paths; legacy net_pnl_excluding_all_costs still includes execution shortfall",
    }
