#!/usr/bin/env python3
"""Out-of-model deterministic risk and deployment guards for synthetic execution tests.

Nothing in this module consults a model, a learned score, a fitted threshold, or a
sampled quantity. Every guard is a deterministic function of an explicit context
mapping plus explicitly configured limits, and every evaluation returns an
explicit ``allow`` / ``reduce`` / ``block`` decision with machine-readable reason
codes. Optional kill-switch latching is the only state transition.

Scope and honesty statement
---------------------------
This module makes no claim about market data authenticity, strategy quality,
calibrated probabilities, or profitability. The default limit values are
engineering guesses, not measured venue or broker facts, and every one of them is
provisional pending review gate R1 (see ``PROVISIONAL_RISK_PARAMETERS``).

Context contract (supplied by the caller, e.g. ``financial_simulator_v1``)
------------------------------------------------------------------------
``decision_ns``               int   decision timestamp, UTC Unix ns
``local_ns``                  int   simulator/risk clock, UTC Unix ns
``last_data_ns``              int   availability timestamp of the newest usable input
``feed_clock_ns``             int   wall clock reported by the data feed itself
``asset_id``, ``venue``       str
``side``                      str   ``buy`` or ``sell``
``requested_quantity``        float positive quantity the caller wants to trade
``reference_price``           float decision-time reference price for sizing
``position_quantity``         float signed current position in the asset
``gross_exposure_notional``   float sum of absolute position notional
``net_exposure_notional``     float signed position notional
``realized_pnl``              float
``unrealized_pnl``            float
``fees_paid``                 float
``net_pnl``                   float realized + unrealized - explicit fees

Optional perpetual-contract keys. An absent key is never evaluated, so the spot-only
guard behaviour is unchanged when they are omitted::

``position_notional``             float absolute notional of the requested instrument
``contract_multiplier``           float declared contract multiplier
``venue_position_notional``       float absolute notional of this venue's positions
``margin_mode``                   str   ``isolated`` or ``cross``
``margin_balance``                float equity backing the positions
``maintenance_margin_required``   float sum of maintenance margin
``initial_margin_required``       float margin already committed
``available_margin``              float ``margin_balance - initial_margin_required``
``margin_ratio``                  float maintenance / margin_balance
``liquidation_distance_fraction`` float buffer remaining before the liquidation trigger
``requested_leverage``            float leverage this order would be sized at
``contract_max_leverage``         float declared contract maximum
``funding_cost_to_date``          float cumulative funding cost already paid
``expected_funding_cost_notional``float projected next funding payment
``target_position_quantity``      float declared target position for target actions

Optional context keys are ignored; unknown keys never change a decision.

Result semantics
----------------
The top-level decision is one of ``allow`` / ``reduce`` / ``block``.
``allowed_quantity`` is the largest quantity every guard permits; ``block`` always
carries ``allowed_quantity == 0.0``. Per-guard entries use ``decision == "allow"``
to mean "this guard does not hard-block; it permits at most ``allowed_quantity``"
(``None`` means unbounded), ``decision == "block"`` for a hard stop, and
``"not_evaluated"`` when the context omitted the required inputs. ``reason_codes``
contains one code per triggered guard and never a free-form message.
"""

from dataclasses import asdict, dataclass
import argparse
import math

from benchmark_nanojev_v2 import canonical_json


RISK_POLICY_VERSION = "v1-provisional-pending-r1"
SCHEMA = "nanojev-financial-risk-v1"

DECISION_ALLOW = "allow"
DECISION_REDUCE = "reduce"
DECISION_BLOCK = "block"
RISK_DECISIONS = (DECISION_ALLOW, DECISION_REDUCE, DECISION_BLOCK)

CHECK_ALLOW = "allow"
CHECK_BLOCK = "block"
CHECK_NOT_EVALUATED = "not_evaluated"

SIDE_BUY = "buy"
SIDE_SELL = "sell"
SIDES = (SIDE_BUY, SIDE_SELL)

REASON_OK = "ok"
REASON_KILL_SWITCH_ENGAGED = "kill_switch_engaged"
REASON_MISSING_FIELD = "missing_context_field"
REASON_NONFINITE_INPUT = "nonfinite_input"
REASON_UNKNOWN_SIDE = "unknown_side"
REASON_NONPOSITIVE_QUANTITY = "nonpositive_requested_quantity"
REASON_INVALID_REFERENCE_PRICE = "invalid_reference_price"
REASON_STALE_DATA = "stale_data"
REASON_DATA_FROM_FUTURE = "data_timestamp_in_future"
REASON_CLOCK_DRIFT = "clock_drift"
REASON_DATA_QUALITY_NOT_EVALUATED = "data_quality_not_evaluated"
REASON_POSITION_LIMIT = "position_limit"
REASON_GROSS_EXPOSURE_LIMIT = "gross_exposure_limit"
REASON_NET_EXPOSURE_LIMIT = "net_exposure_limit"
REASON_NET_LOSS_LIMIT = "net_loss_limit"
REASON_REALIZED_LOSS_LIMIT = "realized_loss_limit"
REASON_TURNOVER_LIMIT = "turnover_limit"
REASON_ORDER_RATE_LIMIT = "order_rate_limit"
REASON_ORDER_NOTIONAL_LIMIT = "order_notional_limit"
REASON_ZERO_ALLOWED_QUANTITY = "zero_allowed_quantity"
REASON_LEVERAGE_LIMIT = "leverage_limit"
REASON_CONTRACT_LEVERAGE_LIMIT = "contract_leverage_limit"
REASON_INSUFFICIENT_MARGIN = "insufficient_margin"
REASON_MARGIN_RATIO_LIMIT = "margin_ratio_limit"
REASON_LIQUIDATION_DISTANCE_LIMIT = "liquidation_distance_limit"
REASON_FUNDING_COST_LIMIT = "funding_cost_limit"
REASON_VENUE_POSITION_LIMIT = "venue_position_limit"
REASON_PERP_GUARD_NOT_EVALUATED = "perp_guard_not_evaluated"

EPSILON = 1e-12

REQUIRED_CONTEXT_FIELDS = (
    "decision_ns", "local_ns", "last_data_ns", "feed_clock_ns", "asset_id", "venue",
    "side", "requested_quantity", "reference_price", "position_quantity",
    "gross_exposure_notional", "net_exposure_notional", "realized_pnl",
    "unrealized_pnl", "fees_paid", "net_pnl",
)

PROVISIONAL_RISK_PARAMETERS = (
    "limits.max_position_quantity",
    "limits.min_position_quantity",
    "limits.max_gross_exposure_notional",
    "limits.max_abs_net_exposure_notional",
    "limits.max_net_loss_notional",
    "limits.max_realized_loss_notional",
    "limits.max_turnover_notional",
    "limits.turnover_window_ns",
    "limits.max_orders_per_window",
    "limits.order_rate_window_ns",
    "limits.max_order_notional",
    "limits.max_data_staleness_ns",
    "limits.max_clock_drift_ns",
    "limits.require_data_quality_fields",
    "limits.latch_kill_switch_on_block",
    "limits.max_leverage",
    "limits.max_position_notional_per_venue",
    "limits.max_funding_cost_notional",
    "limits.max_margin_ratio",
    "limits.min_available_margin_notional",
    "limits.min_liquidation_distance_fraction",
    "limits.max_initial_margin_utilization",
    "limits.require_margin_sufficiency",
)


def _finite(value, field):
    if type(value) not in (int, float) or type(value) is bool or not math.isfinite(value):
        raise ValueError(f"{field} requires a finite number, not {value!r}")
    return float(value)


def _nonnegative(value, field):
    number = _finite(value, field)
    if number < 0:
        raise ValueError(f"{field} requires a nonnegative number")
    return number


def _timestamp(value, field):
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} requires nonnegative UTC Unix nanoseconds")
    return value


def _positive_int(value, field):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} requires a positive integer")
    return value


@dataclass(frozen=True)
class RiskLimits:
    """Deterministic limits. Every field is provisional pending review gate R1."""

    max_position_quantity: float = 100.0
    min_position_quantity: float = 0.0
    max_gross_exposure_notional: float = 100_000.0
    max_abs_net_exposure_notional: float = 100_000.0
    max_net_loss_notional: float = 10_000.0
    max_realized_loss_notional: float = 10_000.0
    max_turnover_notional: float = 250_000.0
    turnover_window_ns: int = 60_000_000_000
    max_orders_per_window: int = 60
    order_rate_window_ns: int = 1_000_000_000
    max_order_notional: float = 25_000.0
    max_data_staleness_ns: int = 5_000_000_000
    max_clock_drift_ns: int = 1_000_000_000
    require_data_quality_fields: bool = True
    latch_kill_switch_on_block: bool = False
    max_leverage: float = 20.0
    max_position_notional_per_venue: float = 1_000_000.0
    max_funding_cost_notional: float = 10_000.0
    max_margin_ratio: float = 0.8
    min_available_margin_notional: float = 0.0
    min_liquidation_distance_fraction: float = 0.05
    max_initial_margin_utilization: float = 0.95
    require_margin_sufficiency: bool = True
    policy_version: str = RISK_POLICY_VERSION

    def __post_init__(self):
        _nonnegative(self.max_position_quantity, "max_position_quantity")
        _finite(self.min_position_quantity, "min_position_quantity")
        if self.min_position_quantity > self.max_position_quantity:
            raise ValueError("min_position_quantity cannot exceed max_position_quantity")
        for name in ("max_gross_exposure_notional", "max_abs_net_exposure_notional",
                     "max_net_loss_notional", "max_realized_loss_notional",
                     "max_turnover_notional", "max_order_notional",
                     "max_position_notional_per_venue", "max_funding_cost_notional",
                     "min_available_margin_notional"):
            _nonnegative(getattr(self, name), name)
        _positive_int(self.turnover_window_ns, "turnover_window_ns")
        _positive_int(self.order_rate_window_ns, "order_rate_window_ns")
        _nonnegative(self.max_data_staleness_ns, "max_data_staleness_ns")
        _nonnegative(self.max_clock_drift_ns, "max_clock_drift_ns")
        if type(self.max_orders_per_window) is not int or self.max_orders_per_window < 0:
            raise ValueError("max_orders_per_window requires a nonnegative integer")
        for name in ("require_data_quality_fields", "latch_kill_switch_on_block",
                     "require_margin_sufficiency"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} requires a boolean policy")
        if _finite(self.max_leverage, "max_leverage") < 1.0:
            raise ValueError("max_leverage requires at least 1x")
        if not 0.0 <= _finite(self.max_margin_ratio, "max_margin_ratio") <= 1.0:
            raise ValueError("max_margin_ratio requires a fraction in [0, 1]")
        if not 0.0 <= _finite(self.max_initial_margin_utilization,
                              "max_initial_margin_utilization") <= 1.0:
            raise ValueError("max_initial_margin_utilization requires a fraction in [0, 1]")
        if _finite(self.min_liquidation_distance_fraction,
                   "min_liquidation_distance_fraction") < 0.0:
            raise ValueError("min_liquidation_distance_fraction requires a nonnegative fraction")

    def to_dict(self):
        return asdict(self)


@dataclass
class KillSwitch:
    """Deterministic latch: once engaged, every evaluation blocks until released."""

    engaged: bool = False
    reason_code: str = ""
    engaged_ns: int = None
    release_ns: int = None

    def engage(self, reason_code, ns=None):
        if not isinstance(reason_code, str) or not reason_code.strip():
            raise ValueError("kill switch requires a nonempty reason code")
        if ns is not None:
            _timestamp(ns, "kill switch ns")
        self.engaged = True
        self.reason_code = reason_code
        self.engaged_ns = ns
        self.release_ns = None
        return self

    def release(self, ns=None):
        if ns is not None:
            _timestamp(ns, "kill switch ns")
        self.engaged = False
        self.release_ns = ns
        return self

    def to_dict(self):
        return {"engaged": self.engaged, "reason_code": self.reason_code,
                "engaged_ns": self.engaged_ns, "release_ns": self.release_ns}


@dataclass(frozen=True)
class RiskDecision:
    decision: str
    reason_codes: tuple
    allowed_quantity: float
    requested_quantity: float
    checks: dict

    def to_dict(self):
        return {"decision": self.decision, "reason_codes": list(self.reason_codes),
                "allowed_quantity": self.allowed_quantity,
                "requested_quantity": self.requested_quantity, "checks": self.checks}

    @property
    def blocked(self):
        return self.decision == DECISION_BLOCK


def _check(decision, reason_code, allowed_quantity=None, **observed):
    entry = {"decision": decision, "reason_code": reason_code,
             "allowed_quantity": allowed_quantity}
    entry.update(observed)
    return entry


class RiskEngine:
    """Stateful, deterministic, model-free guard evaluation.

    ``evaluate`` never mutates limits and never samples. It reads the trailing
    order-rate and turnover windows, which the caller advances with explicit
    ``record_order`` and ``record_fill`` calls so that replay is reproducible.
    The only side effect of ``evaluate`` is kill-switch latching when
    ``limits.latch_kill_switch_on_block`` is enabled and a guard blocks.
    """

    def __init__(self, limits=None, kill_switch=None):
        self.limits = limits or RiskLimits()
        self.kill_switch = kill_switch or KillSwitch()
        self._order_ns = []
        self._fills = []

    # -- window bookkeeping -------------------------------------------------
    def record_order(self, context):
        self._order_ns.append(_timestamp(context["local_ns"], "local_ns"))
        return self

    def record_fill(self, context, notional):
        amount = _nonnegative(notional, "fill notional")
        self._fills.append({"ns": _timestamp(context["local_ns"], "local_ns"), "notional": amount})
        return self

    def _prune(self, now_ns):
        order_horizon = now_ns - self.limits.order_rate_window_ns
        turnover_horizon = now_ns - self.limits.turnover_window_ns
        self._order_ns = [ns for ns in self._order_ns if ns > order_horizon]
        self._fills = [fill for fill in self._fills if fill["ns"] > turnover_horizon]

    def orders_in_window(self, now_ns):
        self._prune(now_ns)
        return len(self._order_ns)

    def turnover_in_window(self, now_ns):
        self._prune(now_ns)
        return sum(fill["notional"] for fill in self._fills)

    def snapshot(self, now_ns=None):
        if now_ns is None:
            self._prune(0)
            return {"schema_version": SCHEMA, "limits": self.limits.to_dict(),
                    "kill_switch": self.kill_switch.to_dict(), "orders_in_window": len(self._order_ns),
                    "turnover_notional_in_window": sum(fill["notional"] for fill in self._fills),
                    "policy_version": self.limits.policy_version}
        return {"schema_version": SCHEMA, "limits": self.limits.to_dict(),
                "kill_switch": self.kill_switch.to_dict(),
                "orders_in_window": self.orders_in_window(now_ns),
                "turnover_notional_in_window": self.turnover_in_window(now_ns),
                "policy_version": self.limits.policy_version}

    # -- guard evaluation ---------------------------------------------------
    def evaluate(self, context):
        if not isinstance(context, dict):
            raise ValueError("risk context must be a mapping")

        if self.kill_switch.engaged:
            decision = RiskDecision(DECISION_BLOCK, (REASON_KILL_SWITCH_ENGAGED,), 0.0,
                                    _safe_quantity(context), {"kill_switch": _check(
                                        CHECK_BLOCK, REASON_KILL_SWITCH_ENGAGED, 0.0,
                                        engaged_ns=self.kill_switch.engaged_ns,
                                        engaged_reason=self.kill_switch.reason_code)})
            return self._maybe_latch(decision, context)

        missing = [name for name in REQUIRED_CONTEXT_FIELDS if name not in context]
        if missing and self.limits.require_data_quality_fields:
            return self._finish(DECISION_BLOCK, (REASON_MISSING_FIELD,), 0.0, _safe_quantity(context),
                                {"context": _check(CHECK_BLOCK, REASON_MISSING_FIELD, 0.0, missing=missing)},
                                context)
        missing = [name for name in missing if name not in ("last_data_ns", "feed_clock_ns")]
        if missing:
            return self._finish(DECISION_BLOCK, (REASON_MISSING_FIELD,), 0.0, _safe_quantity(context),
                                {"context": _check(CHECK_BLOCK, REASON_MISSING_FIELD, 0.0, missing=missing)},
                                context)

        requested = context["requested_quantity"]
        price = context["reference_price"]
        side = context["side"]
        if type(requested) is bool or type(requested) not in (int, float) \
                or not math.isfinite(requested):
            return self._finish(DECISION_BLOCK, (REASON_NONFINITE_INPUT,), 0.0, 0.0,
                                {"context": _check(CHECK_BLOCK, REASON_NONFINITE_INPUT, 0.0)}, context)
        if type(price) is bool or type(price) not in (int, float) or not math.isfinite(price):
            return self._finish(DECISION_BLOCK, (REASON_NONFINITE_INPUT,), 0.0, _safe_quantity(context),
                                {"context": _check(CHECK_BLOCK, REASON_NONFINITE_INPUT, 0.0)}, context)
        requested = float(requested)
        price = float(price)

        checks = {}
        if side not in SIDES:
            checks["side"] = _check(CHECK_BLOCK, REASON_UNKNOWN_SIDE, 0.0, observed=side)
            return self._finish(DECISION_BLOCK, (REASON_UNKNOWN_SIDE,), 0.0, requested, checks, context)
        checks["side"] = _check(CHECK_ALLOW, REASON_OK, requested, observed=side)
        if not requested > 0:
            checks["quantity"] = _check(CHECK_BLOCK, REASON_NONPOSITIVE_QUANTITY, 0.0, observed=requested)
            return self._finish(DECISION_BLOCK, (REASON_NONPOSITIVE_QUANTITY,), 0.0, requested, checks, context)
        checks["quantity"] = _check(CHECK_ALLOW, REASON_OK, requested, observed=requested)
        if not price > 0:
            checks["reference_price"] = _check(CHECK_BLOCK, REASON_INVALID_REFERENCE_PRICE, 0.0, observed=price)
            return self._finish(DECISION_BLOCK, (REASON_INVALID_REFERENCE_PRICE,), 0.0, requested, checks, context)
        checks["reference_price"] = _check(CHECK_ALLOW, REASON_OK, requested, observed=price)

        self._data_quality_checks(context, checks)
        self._loss_checks(context, checks)
        self._position_check(context, requested, side, checks)
        self._exposure_checks(context, requested, price, side, checks)
        self._turnover_check(context, requested, price, checks)
        self._order_rate_check(context, checks)
        self._order_notional_check(requested, price, checks)
        self._leverage_check(context, requested, price, side, checks)
        self._margin_sufficiency_check(context, requested, price, side, checks)
        self._margin_ratio_check(context, side, checks)
        self._liquidation_distance_check(context, side, checks)
        self._funding_cost_check(context, side, checks)
        self._venue_position_limit_check(context, requested, price, side, checks)

        blocking = [name for name, entry in checks.items() if entry["decision"] == CHECK_BLOCK]
        if blocking:
            codes = tuple(_dedupe(checks[name]["reason_code"] for name in blocking))
            return self._finish(DECISION_BLOCK, codes, 0.0, requested, checks, context)

        candidates = [entry["allowed_quantity"] for entry in checks.values()
                      if entry["decision"] == CHECK_ALLOW and entry["allowed_quantity"] is not None]
        allowed = max(0.0, min([requested] + candidates))
        if allowed <= EPSILON:
            codes = tuple(_dedupe(entry["reason_code"] for entry in checks.values()
                                  if entry["allowed_quantity"] is not None and entry["allowed_quantity"] <= EPSILON))
            return self._finish(DECISION_BLOCK, codes or (REASON_ZERO_ALLOWED_QUANTITY,), 0.0,
                                requested, checks, context)
        if allowed < requested - EPSILON:
            limiting = [name for name, entry in checks.items()
                        if entry["allowed_quantity"] is not None and entry["allowed_quantity"] < requested - EPSILON]
            codes = tuple(_dedupe(checks[name]["reason_code"] for name in limiting)) or (REASON_ZERO_ALLOWED_QUANTITY,)
            return self._finish(DECISION_REDUCE, codes, allowed, requested, checks, context)
        return self._finish(DECISION_ALLOW, (REASON_OK,), requested, requested, checks, context)

    # -- individual guards --------------------------------------------------
    def _data_quality_checks(self, context, checks):
        if "last_data_ns" not in context:
            checks["data_staleness"] = _check(CHECK_NOT_EVALUATED, REASON_DATA_QUALITY_NOT_EVALUATED, None)
        else:
            age = context["local_ns"] - context["last_data_ns"]
            if age < 0:
                checks["data_staleness"] = _check(CHECK_BLOCK, REASON_DATA_FROM_FUTURE, 0.0,
                                                  age_ns=age, limit_ns=self.limits.max_data_staleness_ns)
            elif age > self.limits.max_data_staleness_ns:
                checks["data_staleness"] = _check(CHECK_BLOCK, REASON_STALE_DATA, 0.0,
                                                  age_ns=age, limit_ns=self.limits.max_data_staleness_ns)
            else:
                checks["data_staleness"] = _check(CHECK_ALLOW, REASON_OK, None,
                                                  age_ns=age, limit_ns=self.limits.max_data_staleness_ns)
        if "feed_clock_ns" not in context:
            checks["clock_drift"] = _check(CHECK_NOT_EVALUATED, REASON_DATA_QUALITY_NOT_EVALUATED, None)
        else:
            drift = context["local_ns"] - context["feed_clock_ns"]
            if abs(drift) > self.limits.max_clock_drift_ns:
                checks["clock_drift"] = _check(CHECK_BLOCK, REASON_CLOCK_DRIFT, 0.0,
                                               drift_ns=drift, limit_ns=self.limits.max_clock_drift_ns)
            else:
                checks["clock_drift"] = _check(CHECK_ALLOW, REASON_OK, None,
                                               drift_ns=drift, limit_ns=self.limits.max_clock_drift_ns)

    def _loss_checks(self, context, checks):
        net_loss = max(0.0, -float(context["net_pnl"]))
        realized_loss = max(0.0, -float(context["realized_pnl"]))
        if net_loss > self.limits.max_net_loss_notional:
            checks["net_loss"] = _check(CHECK_BLOCK, REASON_NET_LOSS_LIMIT, 0.0,
                                        observed=net_loss, limit=self.limits.max_net_loss_notional)
        else:
            checks["net_loss"] = _check(CHECK_ALLOW, REASON_OK, None,
                                        observed=net_loss, limit=self.limits.max_net_loss_notional)
        if realized_loss > self.limits.max_realized_loss_notional:
            checks["realized_loss"] = _check(CHECK_BLOCK, REASON_REALIZED_LOSS_LIMIT, 0.0,
                                             observed=realized_loss,
                                             limit=self.limits.max_realized_loss_notional)
        else:
            checks["realized_loss"] = _check(CHECK_ALLOW, REASON_OK, None,
                                             observed=realized_loss,
                                             limit=self.limits.max_realized_loss_notional)

    def _position_check(self, context, requested, side, checks):
        signed = requested if side == SIDE_BUY else -requested
        position = float(context["position_quantity"])
        projected = position + signed
        allowed = requested
        if projected > self.limits.max_position_quantity:
            allowed = max(0.0, self.limits.max_position_quantity - position)
            checks["position"] = _check(CHECK_ALLOW, REASON_POSITION_LIMIT, allowed,
                                        observed=position, projected=projected,
                                        limit=self.limits.max_position_quantity)
        elif projected < self.limits.min_position_quantity:
            allowed = max(0.0, position - self.limits.min_position_quantity)
            checks["position"] = _check(CHECK_ALLOW, REASON_POSITION_LIMIT, allowed,
                                        observed=position, projected=projected,
                                        limit=self.limits.min_position_quantity)
        else:
            checks["position"] = _check(CHECK_ALLOW, REASON_OK, None,
                                        observed=position, projected=projected,
                                        limit=self.limits.max_position_quantity)

    def _exposure_checks(self, context, requested, price, side, checks):
        gross = float(context["gross_exposure_notional"])
        net = float(context["net_exposure_notional"])
        signed = requested * price if side == SIDE_BUY else -requested * price
        projected_net = net + signed
        # Gross moves with the change in the absolute net position, so opening a short
        # increases gross exactly as opening a long does, and closing a short releases it.
        projected_gross = max(0.0, gross + (abs(projected_net) - abs(net)))
        if projected_gross > self.limits.max_gross_exposure_notional:
            allowed_gross = max(0.0, (self.limits.max_gross_exposure_notional - gross) / price)
            checks["gross_exposure"] = _check(CHECK_ALLOW, REASON_GROSS_EXPOSURE_LIMIT, allowed_gross,
                                              observed=gross, projected=projected_gross,
                                              limit=self.limits.max_gross_exposure_notional)
        else:
            checks["gross_exposure"] = _check(CHECK_ALLOW, REASON_OK, None, observed=gross,
                                              projected=projected_gross,
                                              limit=self.limits.max_gross_exposure_notional)

        moving_outward = abs(projected_net) > abs(net) + EPSILON
        if moving_outward and abs(projected_net) > self.limits.max_abs_net_exposure_notional:
            allowed_net = max(0.0, (self.limits.max_abs_net_exposure_notional - abs(net)) / price)
            checks["net_exposure"] = _check(CHECK_ALLOW, REASON_NET_EXPOSURE_LIMIT, allowed_net,
                                            observed=net, projected=projected_net,
                                            limit=self.limits.max_abs_net_exposure_notional)
        else:
            checks["net_exposure"] = _check(CHECK_ALLOW, REASON_OK, None, observed=net,
                                            projected=projected_net,
                                            limit=self.limits.max_abs_net_exposure_notional)

    def _turnover_check(self, context, requested, price, checks):
        turnover = self.turnover_in_window(context["local_ns"])
        projected = turnover + requested * price
        if projected > self.limits.max_turnover_notional:
            allowed = max(0.0, (self.limits.max_turnover_notional - turnover) / price)
            checks["turnover"] = _check(CHECK_ALLOW, REASON_TURNOVER_LIMIT, allowed,
                                        observed=turnover, projected=projected,
                                        limit=self.limits.max_turnover_notional,
                                        window_ns=self.limits.turnover_window_ns)
        else:
            checks["turnover"] = _check(CHECK_ALLOW, REASON_OK, None, observed=turnover,
                                        projected=projected, limit=self.limits.max_turnover_notional,
                                        window_ns=self.limits.turnover_window_ns)

    def _order_rate_check(self, context, checks):
        count = self.orders_in_window(context["local_ns"])
        if count >= self.limits.max_orders_per_window:
            checks["order_rate"] = _check(CHECK_BLOCK, REASON_ORDER_RATE_LIMIT, 0.0,
                                          observed=count, limit=self.limits.max_orders_per_window,
                                          window_ns=self.limits.order_rate_window_ns)
        else:
            checks["order_rate"] = _check(CHECK_ALLOW, REASON_OK, None, observed=count,
                                          limit=self.limits.max_orders_per_window,
                                          window_ns=self.limits.order_rate_window_ns)

    def _order_notional_check(self, requested, price, checks):
        projected = requested * price
        if projected > self.limits.max_order_notional:
            checks["order_notional"] = _check(CHECK_ALLOW, REASON_ORDER_NOTIONAL_LIMIT,
                                              max(0.0, self.limits.max_order_notional / price),
                                              observed=projected, limit=self.limits.max_order_notional)
        else:
            checks["order_notional"] = _check(CHECK_ALLOW, REASON_OK, None, observed=projected,
                                              limit=self.limits.max_order_notional)

    # -- perpetual-contract guards -----------------------------------------
    @staticmethod
    def _instrument_multiplier(context):
        value = context.get("contract_multiplier")
        if type(value) is bool or type(value) not in (int, float) or not math.isfinite(value) \
                or value <= 0:
            return 1.0
        return float(value)

    @staticmethod
    def _increases_risk(context, requested, price, side):
        """True when the order moves the position further away from zero."""
        net = float(context["net_exposure_notional"])
        signed = requested * price if side == SIDE_BUY else -requested * price
        projected = net + signed
        return abs(projected) > abs(net) + EPSILON

    @staticmethod
    def _outward_notional(context, requested, price, side):
        net = float(context["net_exposure_notional"])
        signed = requested * price if side == SIDE_BUY else -requested * price
        return max(0.0, abs(net + signed) - abs(net))

    def _leverage_check(self, context, requested, price, side, checks):
        if "requested_leverage" not in context and "margin_balance" not in context:
            checks["leverage"] = _check(CHECK_NOT_EVALUATED, REASON_PERP_GUARD_NOT_EVALUATED, None)
            return
        requested_leverage = context.get("requested_leverage")
        if requested_leverage is not None and type(requested_leverage) in (int, float) \
                and not isinstance(requested_leverage, bool) \
                and math.isfinite(requested_leverage) \
                and requested_leverage > self.limits.max_leverage + EPSILON:
            checks["leverage"] = _check(CHECK_BLOCK, REASON_LEVERAGE_LIMIT, 0.0,
                                        observed=float(requested_leverage),
                                        limit=self.limits.max_leverage)
            return
        contract_max = context.get("contract_max_leverage")
        if contract_max is not None and requested_leverage is not None \
                and type(contract_max) in (int, float) and not isinstance(contract_max, bool) \
                and math.isfinite(contract_max) and math.isfinite(float(requested_leverage)) \
                and float(requested_leverage) > float(contract_max) + EPSILON:
            checks["leverage"] = _check(CHECK_BLOCK, REASON_CONTRACT_LEVERAGE_LIMIT, 0.0,
                                        observed=float(requested_leverage),
                                        contract_max_leverage=float(contract_max))
            return
        balance = context.get("margin_balance")
        if balance is None or type(balance) not in (int, float) or isinstance(balance, bool) \
                or not math.isfinite(balance) or balance <= 0:
            checks["leverage"] = _check(CHECK_NOT_EVALUATED, REASON_PERP_GUARD_NOT_EVALUATED, None)
            return
        gross = float(context["gross_exposure_notional"])
        mult = self._instrument_multiplier(context)
        projected_gross = gross + self._outward_notional(context, requested, price * mult, side)
        effective = projected_gross / float(balance)
        if effective > self.limits.max_leverage + EPSILON:
            allowed_outward = max(0.0, self.limits.max_leverage * float(balance) - gross)
            checks["leverage"] = _check(CHECK_ALLOW, REASON_LEVERAGE_LIMIT,
                                        allowed_outward / (price * mult) if price > 0 else 0.0,
                                        observed=effective, projected=projected_gross,
                                        limit=self.limits.max_leverage)
        else:
            checks["leverage"] = _check(CHECK_ALLOW, REASON_OK, None, observed=effective,
                                        projected=projected_gross,
                                        limit=self.limits.max_leverage)

    def _margin_sufficiency_check(self, context, requested, price, side, checks):
        if not self.limits.require_margin_sufficiency:
            checks["margin_sufficiency"] = _check(CHECK_NOT_EVALUATED, REASON_PERP_GUARD_NOT_EVALUATED,
                                                  None)
            return
        if "available_margin" not in context:
            checks["margin_sufficiency"] = _check(CHECK_NOT_EVALUATED,
                                                  REASON_PERP_GUARD_NOT_EVALUATED, None)
            return
        available = context["available_margin"]
        if type(available) not in (int, float) or isinstance(available, bool) \
                or not math.isfinite(available):
            checks["margin_sufficiency"] = _check(CHECK_BLOCK, REASON_NONFINITE_INPUT, 0.0)
            return
        available = float(available)
        mult = self._instrument_multiplier(context)
        leverage = context.get("requested_leverage")
        if type(leverage) not in (int, float) or isinstance(leverage, bool) \
                or not math.isfinite(leverage) or leverage <= 0:
            leverage = self.limits.max_leverage
        opening = self._outward_notional(context, requested, price * mult, side)
        required = opening / float(leverage)
        if available < self.limits.min_available_margin_notional - EPSILON:
            checks["margin_sufficiency"] = _check(CHECK_BLOCK, REASON_INSUFFICIENT_MARGIN, 0.0,
                                                  available=available,
                                                  min_available=self.limits.min_available_margin_notional)
            return
        if required > available + EPSILON:
            allowed = max(0.0, available * float(leverage) / (price * mult)) if price > 0 else 0.0
            checks["margin_sufficiency"] = _check(CHECK_ALLOW, REASON_INSUFFICIENT_MARGIN, allowed,
                                                  observed=required, available=available,
                                                  leverage=float(leverage))
        else:
            checks["margin_sufficiency"] = _check(CHECK_ALLOW, REASON_OK, None, observed=required,
                                                  available=available, leverage=float(leverage))

    def _margin_ratio_check(self, context, side, checks):
        if "margin_ratio" not in context:
            checks["margin_ratio"] = _check(CHECK_NOT_EVALUATED, REASON_PERP_GUARD_NOT_EVALUATED, None)
            return
        ratio = context["margin_ratio"]
        if ratio is None or type(ratio) not in (int, float) or isinstance(ratio, bool) \
                or not math.isfinite(ratio):
            checks["margin_ratio"] = _check(CHECK_NOT_EVALUATED, REASON_PERP_GUARD_NOT_EVALUATED, None)
            return
        ratio = float(ratio)
        risk_increasing = True
        if "net_exposure_notional" in context and "reference_price" in context \
                and "requested_quantity" in context:
            risk_increasing = self._increases_risk(context, float(context["requested_quantity"]),
                                                   float(context["reference_price"]), side)
        if ratio >= self.limits.max_margin_ratio - EPSILON and risk_increasing:
            checks["margin_ratio"] = _check(CHECK_BLOCK, REASON_MARGIN_RATIO_LIMIT, 0.0,
                                            observed=ratio, limit=self.limits.max_margin_ratio)
        else:
            checks["margin_ratio"] = _check(CHECK_ALLOW, REASON_OK, None, observed=ratio,
                                            limit=self.limits.max_margin_ratio)

    def _liquidation_distance_check(self, context, side, checks):
        if "liquidation_distance_fraction" not in context:
            checks["liquidation_distance"] = _check(CHECK_NOT_EVALUATED,
                                                    REASON_PERP_GUARD_NOT_EVALUATED, None)
            return
        distance = context["liquidation_distance_fraction"]
        if distance is None or type(distance) not in (int, float) or isinstance(distance, bool) \
                or not math.isfinite(distance):
            checks["liquidation_distance"] = _check(CHECK_NOT_EVALUATED,
                                                    REASON_PERP_GUARD_NOT_EVALUATED, None)
            return
        distance = float(distance)
        risk_increasing = True
        if "net_exposure_notional" in context and "reference_price" in context \
                and "requested_quantity" in context:
            risk_increasing = self._increases_risk(context, float(context["requested_quantity"]),
                                                   float(context["reference_price"]), side)
        if distance < self.limits.min_liquidation_distance_fraction - EPSILON and risk_increasing:
            checks["liquidation_distance"] = _check(
                CHECK_BLOCK, REASON_LIQUIDATION_DISTANCE_LIMIT, 0.0, observed=distance,
                limit=self.limits.min_liquidation_distance_fraction)
        else:
            checks["liquidation_distance"] = _check(
                CHECK_ALLOW, REASON_OK, None, observed=distance,
                limit=self.limits.min_liquidation_distance_fraction)

    def _funding_cost_check(self, context, side, checks):
        if "funding_cost_to_date" not in context:
            checks["funding_cost"] = _check(CHECK_NOT_EVALUATED, REASON_PERP_GUARD_NOT_EVALUATED, None)
            return
        paid = context["funding_cost_to_date"]
        if type(paid) not in (int, float) or isinstance(paid, bool) or not math.isfinite(paid):
            checks["funding_cost"] = _check(CHECK_BLOCK, REASON_NONFINITE_INPUT, 0.0)
            return
        paid = float(paid)
        expected = context.get("expected_funding_cost_notional", 0.0)
        if type(expected) not in (int, float) or isinstance(expected, bool) \
                or not math.isfinite(expected):
            expected = 0.0
        projected = paid + max(0.0, float(expected))
        risk_increasing = True
        if "net_exposure_notional" in context and "reference_price" in context \
                and "requested_quantity" in context:
            risk_increasing = self._increases_risk(context, float(context["requested_quantity"]),
                                                   float(context["reference_price"]), side)
        if projected > self.limits.max_funding_cost_notional + EPSILON and risk_increasing:
            checks["funding_cost"] = _check(CHECK_BLOCK, REASON_FUNDING_COST_LIMIT, 0.0,
                                            observed=projected, paid=paid,
                                            limit=self.limits.max_funding_cost_notional)
        else:
            checks["funding_cost"] = _check(CHECK_ALLOW, REASON_OK, None, observed=projected,
                                            paid=paid, limit=self.limits.max_funding_cost_notional)

    def _venue_position_limit_check(self, context, requested, price, side, checks):
        if "venue_position_notional" not in context:
            checks["venue_position"] = _check(CHECK_NOT_EVALUATED, REASON_PERP_GUARD_NOT_EVALUATED,
                                              None)
            return
        venue = context["venue_position_notional"]
        if type(venue) not in (int, float) or isinstance(venue, bool) or not math.isfinite(venue):
            checks["venue_position"] = _check(CHECK_BLOCK, REASON_NONFINITE_INPUT, 0.0)
            return
        venue = float(venue)
        mult = self._instrument_multiplier(context)
        projected = venue + self._outward_notional(context, requested, price * mult, side)
        if projected > self.limits.max_position_notional_per_venue + EPSILON:
            allowed = max(0.0, (self.limits.max_position_notional_per_venue - venue)
                          / (price * mult)) if price > 0 else 0.0
            checks["venue_position"] = _check(CHECK_ALLOW, REASON_VENUE_POSITION_LIMIT, allowed,
                                              observed=venue, projected=projected,
                                              limit=self.limits.max_position_notional_per_venue)
        else:
            checks["venue_position"] = _check(CHECK_ALLOW, REASON_OK, None, observed=venue,
                                              projected=projected,
                                              limit=self.limits.max_position_notional_per_venue)

    # -- result assembly ----------------------------------------------------
    def _finish(self, decision, codes, allowed, requested, checks, context):
        result = RiskDecision(decision, tuple(codes), allowed, requested, checks)
        return self._maybe_latch(result, context)

    def _maybe_latch(self, decision, context):
        if decision.decision == DECISION_BLOCK and self.limits.latch_kill_switch_on_block:
            now = context.get("local_ns")
            if type(now) is not int or now < 0:
                now = None
            if not self.kill_switch.engaged:
                self.kill_switch.engage(decision.reason_codes[0] if decision.reason_codes
                                        else REASON_ZERO_ALLOWED_QUANTITY, now)
        return decision


def _dedupe(values):
    seen = set()
    ordered = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _safe_quantity(context):
    value = context.get("requested_quantity") if isinstance(context, dict) else None
    if type(value) is bool or type(value) not in (int, float) or not math.isfinite(value):
        return 0.0
    return float(value)


def evaluate(limits=None, kill_switch=None, **context):
    """Stateless convenience wrapper for a single check with fresh windows."""
    return RiskEngine(limits=limits, kill_switch=kill_switch).evaluate(context).to_dict()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true",
                        help="run deterministic guard demonstrations and print JSON")
    args = parser.parse_args()
    if not args.self_check:
        parser.print_help()
        raise SystemExit(0)
    demo = RiskEngine()
    base = {"decision_ns": 1_000_000_000, "local_ns": 1_000_000_000, "last_data_ns": 999_000_000,
            "feed_clock_ns": 1_000_000_000, "asset_id": "synthetic-asset", "venue": "synthetic-venue",
            "side": SIDE_BUY, "requested_quantity": 10.0, "reference_price": 100.0,
            "position_quantity": 0.0, "gross_exposure_notional": 0.0, "net_exposure_notional": 0.0,
            "realized_pnl": 0.0, "unrealized_pnl": 0.0, "fees_paid": 0.0, "net_pnl": 0.0}
    print(canonical_json({"allowed": demo.evaluate(base).to_dict(),
                          "blocked": demo.evaluate(dict(base, requested_quantity=10_000.0)).to_dict(),
                          "limits": demo.limits.to_dict(),
                          "provisional_pending_r1": list(PROVISIONAL_RISK_PARAMETERS)}))
