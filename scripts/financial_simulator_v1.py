#!/usr/bin/env python3
"""Deterministic, event-time, replayable execution simulator over synthetic data only.

Scope and honesty statement
---------------------------
This module replays *constructed* decisions against *constructed* quotes. It does
not download market data, contact a broker or venue, model a real matching engine,
or establish any fill authenticity, calibration, or profitability claim. Every
cost, timing, capacity, and fill rule is an explicit policy object; the numeric
defaults are engineering guesses that are provisional pending review gate R1
(see ``PROVISIONAL_POLICY_PARAMETERS``).

Event timeline (distinct timestamps, never collapsed)
-----------------------------------------------------
``decision_ns``        when the decision was made; only information with
                       ``available_ns <= decision_ns`` may influence it.
``order.submit_ns``    ``decision_ns + latency.decision_to_order_ns``
``order.ack_ns``       ``submit_ns + latency.order_to_ack_ns`` (accept or reject)
``execution_ns``       ``quote.available_ns`` for the observation used, and never
                       earlier than ``ack_ns + latency.ack_to_execution_ns``
``order.expire_ns``    decision time to live, or the caller's explicit expiry

Determinism and replayability
-----------------------------
All synthetic variability is a pure function of ``(seed, purpose, *key_parts)``
hashed with SHA-256, so it never depends on call order, dictionary order, the
platform PRNG, or Python's per-process string-hash seed. Every emitted artifact is
JSON-serializable and canonically ordered, so identical inputs and the same seed
produce a byte-identical ledger.

No-future-information guard
---------------------------
``NoFutureInformationGuard`` is a hard gate on both boundaries: a decision may not
consume a price that became available after the decision, and an execution may not
consume a price that became available after that execution. The guard policies are
explicit (``LeakGuardPolicy``); ``forbid_post_decision_prices_in_executions`` is
the strictest reading, disabled by default because a realistic fill necessarily
uses a post-decision price when latency is nonzero. See
``docs/FINANCIAL_SIMULATOR_V1.md`` for the exact interpretation.

Order lifecycle
---------------
``submitted`` -> ``accepted`` or ``rejected``; ``accepted`` -> ``partially_filled``
-> ``filled`` or ``expired``. ``abstain`` and ``no_trade`` are first-class actions
that produce a logged decision and no order.

Perpetual-contract layer (R1 scope: crypto secondary-market perpetuals only)
--------------------------------------------------------------------------
A caller may declare a :class:`ContractSpec` for an ``(asset_id, venue)`` pair with
``instrument_type == "perp"``. Only then do the perp semantics activate; without a
declared contract the simulator behaves exactly as the long-only spot slice did.
With a perp contract declared:

* positions are **signed** (long positive, short negative) and a sell may open a
  short (``PerpPolicy.allow_short``), so two-sided ``long``/``short``/``flat``
  target actions and ``open_long``/``open_short``/``close_long``/``close_short``
  order actions are available;
* **margin sufficiency is enforced, never assumed**: every risk-increasing order
  is checked against ``equity - allocated_margin`` at the declared leverage before
  it can execute, and is reduced or rejected with a reason code otherwise;
* **funding** is charged at declared intervals from a declared
  ``funding_rate_source`` field, on the mark-price notional of the open position;
* **liquidation** is evaluated at every available synthetic price observation and
  at every funding boundary; a breach flattens the position (isolated: the
  breaching leg only; cross: the whole account) and writes an explicit ledger
  entry;
* valuation distinguishes **mark price** from **last price** (``Quote.mark_price``
  / ``Quote.last_price``; both default to the mid).

Linear (quote-settled) contracts only. Inverse contracts, portfolio margin,
insurance funds, auto-deleveraging, bad-debt socialisation and cross-margin
netting across settlement assets are **not** modelled. See
``docs/FINANCIAL_SIMULATOR_V1.md``.
"""

import argparse
from bisect import bisect_right
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

from benchmark_nanojev_v2 import canonical_json, file_identity, sha256_bytes


SCHEMA = "nanojev-financial-simulator-v1"
STRESS_SCHEMA = "nanojev-financial-simulator-stress-v1"
POLICY_VERSION = "v1-provisional-pending-r1"
SCOPE = ("Synthetic constructed quotes and orders only; no real market data, no broker, "
         "no venue, no fill-authenticity, calibration or profitability claim.")
PERP_SCOPE = ("Crypto secondary-market PERPETUAL CONTRACT trading only (Binance, Bybit, Aster, "
              "Hyperliquid named as the declared venue universe; no venue is contacted and no "
              "venue-specific rule is claimed to be replicated). Linear quote-settled contracts; "
              "inverse contracts, portfolio margin, insurance funds and ADL are not modelled.")

ACTION_ABSTAIN = "abstain"
ACTION_NO_TRADE = "no_trade"
ACTION_BUY = "buy"
ACTION_SELL = "sell"
ACTION_LONG = "long"
ACTION_SHORT = "short"
ACTION_FLAT = "flat"
ACTION_OPEN_LONG = "open_long"
ACTION_OPEN_SHORT = "open_short"
ACTION_CLOSE_LONG = "close_long"
ACTION_CLOSE_SHORT = "close_short"
ORDER_ACTIONS = (ACTION_BUY, ACTION_SELL, ACTION_LONG, ACTION_SHORT, ACTION_FLAT,
                 ACTION_OPEN_LONG, ACTION_OPEN_SHORT, ACTION_CLOSE_LONG, ACTION_CLOSE_SHORT)
TARGET_POSITION_ACTIONS = (ACTION_LONG, ACTION_SHORT, ACTION_FLAT)
ACTIONS = (ACTION_ABSTAIN, ACTION_NO_TRADE) + ORDER_ACTIONS

SIDE_BUY = "buy"
SIDE_SELL = "sell"
SIDES = (SIDE_BUY, SIDE_SELL)

INSTRUMENT_SPOT = "spot"
INSTRUMENT_PERP = "perp"
INSTRUMENT_TYPES = (INSTRUMENT_SPOT, INSTRUMENT_PERP)

MARGIN_ISOLATED = "isolated"
MARGIN_CROSS = "cross"
MARGIN_MODES = (MARGIN_ISOLATED, MARGIN_CROSS)

POSITION_ONE_WAY = "one_way"
POSITION_HEDGE = "hedge"
POSITION_MODES = (POSITION_ONE_WAY, POSITION_HEDGE)

EFFECT_AUTO = "auto"
EFFECT_OPEN = "open"
EFFECT_CLOSE = "close"
POSITION_EFFECTS = (EFFECT_AUTO, EFFECT_OPEN, EFFECT_CLOSE)

LEG_LONG = "long"
LEG_SHORT = "short"
LEGS = (LEG_LONG, LEG_SHORT)

FUNDING_SOURCE_POLICY_CONSTANT = "policy_constant"
FUNDING_SOURCE_DECLARED_SCHEDULE = "declared_schedule"
FUNDING_SOURCE_QUOTE_FIELD = "quote_funding_rate_field"
FUNDING_RATE_SOURCES = (FUNDING_SOURCE_POLICY_CONSTANT, FUNDING_SOURCE_DECLARED_SCHEDULE,
                        FUNDING_SOURCE_QUOTE_FIELD)

PRICE_SOURCE_MARK = "mark"
PRICE_SOURCE_LAST = "last"
PRICE_SOURCE_MID = "mid"
PRICE_SOURCES = (PRICE_SOURCE_MARK, PRICE_SOURCE_LAST, PRICE_SOURCE_MID)

# Quantities at or below this relative tolerance are floating-point dust, not positions.
FLAT_QUANTITY_TOLERANCE = 1e-9

CASH_ACTION_REDUCE = "reduce"
CASH_ACTION_REJECT = "reject"
CASH_ACTIONS = (CASH_ACTION_REDUCE, CASH_ACTION_REJECT)

LEDGER_REASON_TRADE = "trade"
LEDGER_REASON_EXPLICIT_FEE = "explicit_fee"
LEDGER_REASON_FUNDING = "funding"
LEDGER_REASON_LIQUIDATION_FEE = "liquidation_fee"

ORDER_SUBMITTED = "submitted"
ORDER_ACCEPTED = "accepted"
ORDER_REJECTED = "rejected"
ORDER_PARTIALLY_FILLED = "partially_filled"
ORDER_FILLED = "filled"
ORDER_EXPIRED = "expired"
ORDER_STATUSES = (ORDER_SUBMITTED, ORDER_ACCEPTED, ORDER_REJECTED,
                  ORDER_PARTIALLY_FILLED, ORDER_FILLED, ORDER_EXPIRED)
ORDER_TERMINAL_STATUSES = (ORDER_REJECTED, ORDER_FILLED, ORDER_EXPIRED)
ALLOWED_TRANSITIONS = {
    ORDER_SUBMITTED: (ORDER_ACCEPTED, ORDER_REJECTED, ORDER_EXPIRED),
    ORDER_ACCEPTED: (ORDER_PARTIALLY_FILLED, ORDER_FILLED, ORDER_EXPIRED),
    ORDER_PARTIALLY_FILLED: (ORDER_PARTIALLY_FILLED, ORDER_FILLED, ORDER_EXPIRED),
    ORDER_FILLED: (),
    ORDER_REJECTED: (),
    ORDER_EXPIRED: (),
}

REASON_SUBMITTED = "order_submitted"
REASON_ACCEPTED = "synthetic_venue_ack"
REASON_ABSTAIN = "model_abstain"
REASON_NO_TRADE = "policy_no_trade"
REASON_RISK_BLOCK = "risk_block"
REASON_RISK_REDUCE = "risk_reduce"
REASON_NO_REFERENCE_PRICE = "no_visible_reference_price"
REASON_SYNTHETIC_REJECT = "synthetic_reject_draw"
REASON_INSUFFICIENT_INVENTORY = "insufficient_inventory_long_only"
REASON_CAPACITY_REJECT = "capacity_order_limit_reject"
REASON_CAPACITY_CLAMP = "capacity_order_limit_clamp"
REASON_FILL_DRAW_MISS = "synthetic_fill_draw_miss"
REASON_ZERO_LIQUIDITY = "zero_synthetic_liquidity"
REASON_BELOW_MINIMUM_FILL = "below_minimum_fill_quantity"
REASON_LIMIT_NOT_REACHED = "limit_price_not_reached"
REASON_FILLED = "fully_filled"
REASON_PARTIAL_FILL = "synthetic_partial_fill"
REASON_EXPIRED_UNFILLED = "no_execution_before_expiry"
REASON_INSUFFICIENT_CASH = "insufficient_cash_for_spot_purchase"
REASON_CASH_REDUCE = "cash_available_quantity_reduce"
REASON_PERP_CONTRACT_NOT_DECLARED = "perp_action_requires_declared_contract"
REASON_SHORT_DISABLED = "short_sales_disabled_by_perp_policy"
REASON_MARGIN_INSUFFICIENT = "insufficient_initial_margin"
REASON_MARGIN_REDUCE = "initial_margin_quantity_reduce"
REASON_LOT_SIZE_ROUND = "quantity_rounded_down_to_contract_lot_size"
REASON_MIN_NOTIONAL = "below_contract_minimum_notional"
REASON_TICK_ROUND = "execution_price_rounded_to_contract_tick"
REASON_TARGET_SATISFIED = "target_position_already_satisfied"
REASON_NO_POSITION_TO_CLOSE = "no_position_to_close"
REASON_REDUCE_ONLY = "reduce_only_order_would_increase_position"
REASON_OPEN_WOULD_FLIP = "open_effect_would_flip_one_way_position"
REASON_LEVERAGE_ABOVE_CONTRACT_MAX = "leverage_above_contract_maximum"
REASON_MULTIPLIER_CONFLICT = "conflicting_contract_multiplier_for_asset"
REASON_LIQUIDATION = "margin_below_maintenance_requirement"
REASON_FUNDING = "declared_funding_payment"
REASON_LIQUIDATION_FEE = "synthetic_liquidation_fee"

GUARD_MODE_CAUSAL = "last_available_at_or_before_decision"

PROVISIONAL_POLICY_PARAMETERS = (
    "fees.fee_bps",
    "fees.fixed_fee_per_fill",
    "fees.minimum_fee",
    "spread.half_spread_bps",
    "spread.fixed_half_spread_price",
    "slippage.fixed_bps",
    "slippage.impact_bps_at_full_capacity",
    "capacity.max_participation_fraction",
    "capacity.max_order_quantity",
    "capacity.max_order_notional",
    "capacity.on_order_exceeds_limit",
    "fills.reject_probability",
    "fills.fill_probability",
    "fills.minimum_fill_quantity",
    "fills.require_inventory_for_sell",
    "fills.order_time_to_live_ns",
    "latency.decision_to_order_ns",
    "latency.order_to_ack_ns",
    "latency.ack_to_execution_ns",
    "guard.forbid_post_decision_prices_in_decisions",
    "guard.forbid_post_execution_prices_in_executions",
    "guard.forbid_post_decision_prices_in_executions",
    "marks.fallback",
    "marks.price_source",
    "account.require_sufficient_cash",
    "account.on_insufficient_cash",
    "account.cash_buffer_fraction",
    "perps.allow_short",
    "perps.default_leverage",
    "perps.max_leverage",
    "perps.margin_mode",
    "perps.position_mode",
    "perps.maintenance_margin_rate",
    "perps.funding_interval_ns",
    "perps.funding_anchor_ns",
    "perps.funding_rate_source",
    "perps.default_funding_rate",
    "perps.funding_rates",
    "perps.funding_rate_schedule",
    "perps.liquidation_fee_bps",
    "perps.trace_perp_curve",
    "contracts.*.contract_multiplier",
    "contracts.*.tick_size",
    "contracts.*.lot_size",
    "contracts.*.min_notional",
    "contracts.*.margin_mode",
    "contracts.*.max_leverage",
    "contracts.*.maintenance_margin_rate",
)


class SimulationError(RuntimeError):
    """Base class for simulator contract violations."""


class FutureInformationError(SimulationError):
    """Raised when a decision or execution consumes a not-yet-available price."""


class LedgerConservationError(SimulationError):
    """Raised when the ledger fails an independent conservation recomputation."""


def _finite(value, field):
    if type(value) not in (int, float) or type(value) is bool or not math.isfinite(value):
        raise ValueError(f"{field} requires a finite number, not {value!r}")
    return float(value)


def _nonnegative(value, field):
    number = _finite(value, field)
    if number < 0:
        raise ValueError(f"{field} requires a nonnegative number")
    return number


def _probability(value, field):
    number = _finite(value, field)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} requires a probability in [0, 1]")
    return number


def _positive(value, field):
    number = _finite(value, field)
    if number <= 0:
        raise ValueError(f"{field} requires a positive number")
    return number


def _choice(value, field, allowed):
    if value not in allowed:
        raise ValueError(f"{field} requires one of {allowed}")
    return value


def _timestamp(value, field):
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} requires nonnegative UTC Unix nanoseconds")
    return value


def uniform_draw(seed, *key_parts):
    """Deterministic uniform draw in [0, 1) as a pure function of the key parts."""
    key = canonical_json([int(seed), *key_parts]).encode()
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


# --------------------------------------------------------------------------
# Explicit, parameterized policy objects (all defaults provisional pending R1)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FeePolicy:
    fee_bps: float = 1.0
    fixed_fee_per_fill: float = 0.0
    minimum_fee: float = 0.0
    policy_version: str = POLICY_VERSION

    def __post_init__(self):
        _nonnegative(self.fee_bps, "fee_bps")
        _nonnegative(self.fixed_fee_per_fill, "fixed_fee_per_fill")
        _nonnegative(self.minimum_fee, "minimum_fee")

    def explicit_fee(self, notional):
        return max(self.minimum_fee, _nonnegative(notional, "notional") * self.fee_bps / 10_000.0) \
            + self.fixed_fee_per_fill


@dataclass(frozen=True)
class SpreadPolicy:
    """Quoted spread is modelled as twice the configured half spread."""

    half_spread_bps: float = 1.0
    fixed_half_spread_price: float = 0.0
    policy_version: str = POLICY_VERSION

    def __post_init__(self):
        _nonnegative(self.half_spread_bps, "half_spread_bps")
        _nonnegative(self.fixed_half_spread_price, "fixed_half_spread_price")

    def half_spread_price(self, mid):
        return _nonnegative(mid, "mid") * self.half_spread_bps / 10_000.0 + self.fixed_half_spread_price


@dataclass(frozen=True)
class SlippagePolicy:
    fixed_bps: float = 0.0
    impact_bps_at_full_capacity: float = 0.0
    policy_version: str = POLICY_VERSION

    def __post_init__(self):
        _nonnegative(self.fixed_bps, "fixed_bps")
        _nonnegative(self.impact_bps_at_full_capacity, "impact_bps_at_full_capacity")

    def slippage_bps(self, participation_ratio):
        ratio = _nonnegative(participation_ratio, "participation_ratio")
        return self.fixed_bps + self.impact_bps_at_full_capacity * ratio


@dataclass(frozen=True)
class CapacityPolicy:
    max_participation_fraction: float = 0.1
    max_order_quantity: float = 1.0e9
    max_order_notional: float = 1.0e12
    on_order_exceeds_limit: str = "clamp"
    policy_version: str = POLICY_VERSION

    def __post_init__(self):
        _probability(self.max_participation_fraction, "max_participation_fraction")
        _nonnegative(self.max_order_quantity, "max_order_quantity")
        _nonnegative(self.max_order_notional, "max_order_notional")
        if self.on_order_exceeds_limit not in ("clamp", "reject"):
            raise ValueError("on_order_exceeds_limit requires 'clamp' or 'reject'")


@dataclass(frozen=True)
class FillPolicy:
    reject_probability: float = 0.0
    fill_probability: float = 1.0
    minimum_fill_quantity: float = 1.0e-9
    require_inventory_for_sell: bool = True
    order_time_to_live_ns: int = 1_000_000_000
    policy_version: str = POLICY_VERSION

    def __post_init__(self):
        _probability(self.reject_probability, "reject_probability")
        _probability(self.fill_probability, "fill_probability")
        _nonnegative(self.minimum_fill_quantity, "minimum_fill_quantity")
        if type(self.require_inventory_for_sell) is not bool:
            raise ValueError("require_inventory_for_sell requires a boolean policy")
        if type(self.order_time_to_live_ns) is not int or self.order_time_to_live_ns <= 0:
            raise ValueError("order_time_to_live_ns requires a positive integer")


@dataclass(frozen=True)
class LatencyPolicy:
    decision_to_order_ns: int = 1_000_000
    order_to_ack_ns: int = 500_000
    ack_to_execution_ns: int = 1_000_000
    policy_version: str = POLICY_VERSION

    def __post_init__(self):
        for name in ("decision_to_order_ns", "order_to_ack_ns", "ack_to_execution_ns"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} requires nonnegative integer nanoseconds")


@dataclass(frozen=True)
class LeakGuardPolicy:
    forbid_post_decision_prices_in_decisions: bool = True
    forbid_post_execution_prices_in_executions: bool = True
    forbid_post_decision_prices_in_executions: bool = False
    policy_version: str = POLICY_VERSION

    def __post_init__(self):
        for name in ("forbid_post_decision_prices_in_decisions",
                     "forbid_post_execution_prices_in_executions",
                     "forbid_post_decision_prices_in_executions"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} requires a boolean switch")


@dataclass(frozen=True)
class MarkPolicy:
    """Valuation rule when a position has no usable quote at the mark timestamp.

    ``cost_basis`` carries the position at its own average cost, ``zero`` writes it
    down to zero (conservative), and ``last_fill_price`` uses the newest simulated
    fill price for that asset.

    ``price_source`` selects which synthetic price values an open position: ``mark``
    (``Quote.mark_price`` when declared, else the mid), ``last`` (``Quote.last_price``
    when declared, else the mid) or ``mid``. Execution prices are unaffected: they
    always cross the synthetic bid/ask.
    """

    fallback: str = "cost_basis"
    price_source: str = PRICE_SOURCE_MARK
    policy_version: str = POLICY_VERSION

    def __post_init__(self):
        if self.fallback not in ("cost_basis", "zero", "last_fill_price"):
            raise ValueError("fallback requires 'cost_basis', 'zero' or 'last_fill_price'")
        _choice(self.price_source, "price_source", PRICE_SOURCES)


@dataclass(frozen=True)
class AccountPolicy:
    """Cash/account sufficiency, enforced instead of assumed.

    The P2 review found that the spot slice had no buying-power check at all. This
    policy closes that gap for spot purchases and is the account-level counterpart
    of the perp margin check: a risk-increasing order that the account cannot fund
    is reduced (``reduce``) or rejected (``reject``) with an explicit reason code,
    never silently filled into negative cash.
    """

    require_sufficient_cash: bool = True
    on_insufficient_cash: str = CASH_ACTION_REDUCE
    cash_buffer_fraction: float = 0.0
    policy_version: str = POLICY_VERSION

    def __post_init__(self):
        if type(self.require_sufficient_cash) is not bool:
            raise ValueError("require_sufficient_cash requires a boolean policy")
        _choice(self.on_insufficient_cash, "on_insufficient_cash", CASH_ACTIONS)
        _probability(self.cash_buffer_fraction, "cash_buffer_fraction")


@dataclass(frozen=True)
class PerpPolicy:
    """Perpetual-contract policy layer. Every field is provisional pending R1.

    ``funding_rate_source`` is the declared field the funding rate is read from:

    ``policy_constant``          every funding boundary uses ``default_funding_rate``;
    ``declared_schedule``        an explicit ``(asset_id, venue, funding_ns) -> rate``
                                 declaration wins, ``default_funding_rate`` is the fallback;
    ``quote_funding_rate_field`` the rate is taken from ``Quote.funding_rate``.

    Whichever mode is declared is recorded on every emitted funding payment, so a
    receipt always states where its funding rates came from.
    """

    allow_short: bool = True
    default_leverage: float = 5.0
    max_leverage: float = 20.0
    margin_mode: str = MARGIN_CROSS
    position_mode: str = POSITION_ONE_WAY
    maintenance_margin_rate: float = 0.005
    funding_interval_ns: int = 8 * 3_600 * 1_000_000_000
    funding_anchor_ns: int = 0
    funding_rate_source: str = FUNDING_SOURCE_POLICY_CONSTANT
    default_funding_rate: float = 0.0
    funding_rates: tuple = ()
    funding_rate_schedule: tuple = ()
    liquidation_fee_bps: float = 0.0
    trace_perp_curve: bool = False
    policy_version: str = POLICY_VERSION

    def __post_init__(self):
        if type(self.allow_short) is not bool:
            raise ValueError("allow_short requires a boolean policy")
        if type(self.trace_perp_curve) is not bool:
            raise ValueError("trace_perp_curve requires a boolean policy")
        _positive(self.default_leverage, "default_leverage")
        _positive(self.max_leverage, "max_leverage")
        if self.default_leverage > self.max_leverage:
            raise ValueError("default_leverage cannot exceed max_leverage")
        _choice(self.margin_mode, "margin_mode", MARGIN_MODES)
        _choice(self.position_mode, "position_mode", POSITION_MODES)
        _probability(self.maintenance_margin_rate, "maintenance_margin_rate")
        if type(self.funding_interval_ns) is not int or self.funding_interval_ns <= 0:
            raise ValueError("funding_interval_ns requires a positive integer")
        _timestamp(self.funding_anchor_ns, "funding_anchor_ns")
        _choice(self.funding_rate_source, "funding_rate_source", FUNDING_RATE_SOURCES)
        _finite(self.default_funding_rate, "default_funding_rate")
        _nonnegative(self.liquidation_fee_bps, "liquidation_fee_bps")
        for entry in self.funding_rates:
            if len(entry) != 3:
                raise ValueError("funding_rates entries require (asset_id, venue, rate)")
            _finite(entry[2], "funding rate")
        for entry in self.funding_rate_schedule:
            if len(entry) != 4:
                raise ValueError(
                    "funding_rate_schedule entries require (asset_id, venue, funding_ns, rate)")
            _timestamp(entry[2], "funding schedule ns")
            _finite(entry[3], "funding schedule rate")

    def declared_rate(self, asset_id, venue, funding_ns):
        """Rate from the declared constant table, if one is declared for the pair."""
        for entry_asset, entry_venue, rate in self.funding_rates:
            if entry_asset == asset_id and entry_venue == venue:
                return float(rate)
        return None

    def scheduled_rate(self, asset_id, venue, funding_ns):
        """Rate from the declared per-boundary schedule, if one matches exactly."""
        for entry_asset, entry_venue, entry_ns, rate in self.funding_rate_schedule:
            if entry_asset == asset_id and entry_venue == venue and entry_ns == funding_ns:
                return float(rate)
        return None

    def funding_rate_at(self, asset_id, venue, funding_ns, quote):
        """Declared funding rate for one boundary, with its declared source mode."""
        mode = self.funding_rate_source
        if mode == FUNDING_SOURCE_QUOTE_FIELD:
            if quote is not None and quote.funding_rate is not None:
                return float(quote.funding_rate), mode
            return float(self.default_funding_rate), mode
        if mode == FUNDING_SOURCE_DECLARED_SCHEDULE:
            rate = self.scheduled_rate(asset_id, venue, funding_ns)
            if rate is None:
                rate = self.declared_rate(asset_id, venue, funding_ns)
            if rate is None:
                rate = float(self.default_funding_rate)
            return rate, mode
        return float(self.default_funding_rate), mode

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ContractSpec:
    """Declared contract specification for one ``(asset_id, venue)`` instrument.

    Declaring a ``perp`` contract for a pair is what activates the perpetual layer
    for that instrument; without it the simulator treats the pair as the long-only
    spot slice. ``tick_size``/``lot_size``/``min_notional`` of ``0`` disable that
    particular constraint. Linear, quote-settled contracts only.
    """

    asset_id: str
    venue: str
    instrument_type: str = INSTRUMENT_PERP
    contract_multiplier: float = 1.0
    tick_size: float = 0.0
    lot_size: float = 0.0
    min_notional: float = 0.0
    margin_mode: str = MARGIN_CROSS
    max_leverage: float = 20.0
    maintenance_margin_rate: float = 0.005
    settlement_asset: str = "quote"
    policy_version: str = POLICY_VERSION

    def __post_init__(self):
        for name in ("asset_id", "venue"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} requires a nonempty string")
        _choice(self.instrument_type, "instrument_type", INSTRUMENT_TYPES)
        _positive(self.contract_multiplier, "contract_multiplier")
        _nonnegative(self.tick_size, "tick_size")
        _nonnegative(self.lot_size, "lot_size")
        _nonnegative(self.min_notional, "min_notional")
        _choice(self.margin_mode, "margin_mode", MARGIN_MODES)
        _positive(self.max_leverage, "max_leverage")
        if self.max_leverage < 1.0:
            raise ValueError("max_leverage requires at least 1x")
        _probability(self.maintenance_margin_rate, "maintenance_margin_rate")
        if self.settlement_asset != "quote":
            raise ValueError("only linear quote-settled contracts are modelled")

    @property
    def key(self):
        return (self.asset_id, self.venue)

    @property
    def is_perp(self):
        return self.instrument_type == INSTRUMENT_PERP

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ExecutionPolicy:
    fees: FeePolicy = FeePolicy()
    spread: SpreadPolicy = SpreadPolicy()
    slippage: SlippagePolicy = SlippagePolicy()
    capacity: CapacityPolicy = CapacityPolicy()
    fills: FillPolicy = FillPolicy()
    latency: LatencyPolicy = LatencyPolicy()
    marks: MarkPolicy = MarkPolicy()
    account: AccountPolicy = AccountPolicy()
    perps: PerpPolicy = PerpPolicy()

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Quote:
    """One synthetic point-in-time price observation.

    ``event_ns`` is when the price was observed at the source; ``available_ns`` is
    the earliest moment the simulator may use it. Both are required because a feed
    delay is not the same thing as the observation time.

    ``mark_price``/``last_price`` are optional declared perp inputs: when omitted the
    mid is used for both, so "mark versus last" is only a distinction the caller
    explicitly declares. ``funding_rate`` is an optional declared funding-rate field
    used only when ``PerpPolicy.funding_rate_source == "quote_funding_rate_field"``.
    """

    asset_id: str
    venue: str
    event_ns: int
    available_ns: int
    bid: float
    ask: float
    volume: float
    source_id: str = "synthetic-only"
    version: str = "v1"
    mark_price: float = None
    last_price: float = None
    funding_rate: float = None

    def __post_init__(self):
        for name in ("asset_id", "venue", "source_id", "version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} requires a nonempty string")
        _timestamp(self.event_ns, "event_ns")
        _timestamp(self.available_ns, "available_ns")
        if self.event_ns > self.available_ns:
            raise ValueError("quote availability cannot precede its event timestamp")
        bid = _finite(self.bid, "bid")
        ask = _finite(self.ask, "ask")
        if bid <= 0 or ask <= 0 or bid > ask:
            raise ValueError("quote requires positive bid <= ask")
        _nonnegative(self.volume, "volume")
        for name in ("mark_price", "last_price"):
            value = getattr(self, name)
            if value is not None:
                if _finite(value, name) <= 0:
                    raise ValueError(f"{name} requires a positive price")
        if self.funding_rate is not None:
            _finite(self.funding_rate, "funding_rate")

    @property
    def mid(self):
        return (float(self.bid) + float(self.ask)) / 2.0

    @property
    def declared_mark_price(self):
        return float(self.mark_price) if self.mark_price is not None else self.mid

    @property
    def declared_last_price(self):
        return float(self.last_price) if self.last_price is not None else self.mid

    def price_for(self, price_source):
        if price_source == PRICE_SOURCE_LAST:
            return self.declared_last_price
        if price_source == PRICE_SOURCE_MID:
            return self.mid
        return self.declared_mark_price

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Decision:
    """A single decision event. ``abstain`` and ``no_trade`` are first-class.

    ``long``/``short`` declare a *target* signed position; ``flat`` closes whatever
    the instrument currently holds. ``buy``/``sell`` are order-direction actions,
    which under one-way netting open a short when they exceed the long position and
    the declared contract allows it. ``position_effect`` makes an order explicitly
    ``open`` (increase only) or ``close`` (reduce only). ``leverage`` overrides the
    policy default for the margin requirement of this order's opening portion.
    """

    decision_id: str
    asset_id: str
    venue: str
    decision_ns: int
    action: str
    quantity: float = 0.0
    limit_price: float = None
    expire_ns: int = None
    reason: str = ""
    position_effect: str = EFFECT_AUTO
    leverage: float = None

    def __post_init__(self):
        for name in ("decision_id", "asset_id", "venue"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} requires a nonempty string")
        _timestamp(self.decision_ns, "decision_ns")
        if self.action not in ACTIONS:
            raise ValueError(f"action must be one of {ACTIONS}")
        quantity = _nonnegative(self.quantity, "quantity")
        if self.action in ORDER_ACTIONS and self.action != ACTION_FLAT and quantity <= 0:
            raise ValueError("order actions require a positive quantity; use no_trade to decline")
        if self.action == ACTION_FLAT and quantity != 0.0:
            raise ValueError("flat takes no quantity; it closes the current position")
        if self.limit_price is not None:
            _finite(self.limit_price, "limit_price")
        if self.expire_ns is not None and _timestamp(self.expire_ns, "expire_ns") < self.decision_ns:
            raise ValueError("expire_ns cannot precede the decision timestamp")
        _choice(self.position_effect, "position_effect", POSITION_EFFECTS)
        if self.leverage is not None:
            _positive(self.leverage, "leverage")

    @property
    def order_side(self):
        if self.action in (ACTION_BUY, ACTION_LONG, ACTION_OPEN_LONG, ACTION_CLOSE_SHORT):
            return SIDE_BUY
        if self.action in (ACTION_SELL, ACTION_SHORT, ACTION_OPEN_SHORT, ACTION_CLOSE_LONG):
            return SIDE_SELL
        return None

    @property
    def is_target_position_action(self):
        return self.action in TARGET_POSITION_ACTIONS

    def to_dict(self):
        return asdict(self)


class NoFutureInformationGuard:
    """Hard gate against consuming a price that was not yet available.

    Three independent switches (see ``LeakGuardPolicy``). The first two are on by
    default and are the causal boundaries; the third is the strictest reading of
    "no execution may use a post-decision price" and is disabled by default
    because any nonzero-latency fill legitimately uses a post-decision price.
    """

    def __init__(self, policy=None):
        self.policy = policy or LeakGuardPolicy()
        self.decision_checks = 0
        self.execution_checks = 0
        self.decision_violations = 0
        self.execution_violations = 0
        self.violations = []

    def check_decision(self, decision_ns, quote):
        self.decision_checks += 1
        if self.policy.forbid_post_decision_prices_in_decisions and quote.available_ns > decision_ns:
            self._violate("decision", f"decision at {decision_ns} consumed quote available at "
                                      f"{quote.available_ns} ({quote.asset_id}@{quote.venue})")
        return quote

    def check_execution(self, decision_ns, submit_ns, execution_ns, quote):
        self.execution_checks += 1
        if execution_ns < submit_ns:
            self._violate("execution", f"execution at {execution_ns} precedes order submit at {submit_ns}")
        if execution_ns < decision_ns:
            self._violate("execution", f"execution at {execution_ns} precedes decision at {decision_ns}")
        if self.policy.forbid_post_execution_prices_in_executions and quote.available_ns > execution_ns:
            self._violate("execution", f"execution at {execution_ns} consumed quote available at "
                                       f"{quote.available_ns}")
        if self.policy.forbid_post_decision_prices_in_executions and quote.available_ns > decision_ns:
            self._violate("execution", f"execution at {execution_ns} consumed a price timestamped "
                                       f"{quote.available_ns}, after the decision at {decision_ns}")
        return quote

    def _violate(self, boundary, message):
        if boundary == "decision":
            self.decision_violations += 1
        else:
            self.execution_violations += 1
        self.violations.append({"boundary": boundary, "message": message})
        raise FutureInformationError(f"{boundary} leaked future information: {message}")

    def snapshot(self):
        return {"policy": asdict(self.policy), "decision_checks": self.decision_checks,
                "execution_checks": self.execution_checks,
                "decision_violations": self.decision_violations,
                "execution_violations": self.execution_violations,
                "violations": list(self.violations),
                "violation_count": len(self.violations)}


def causal_reference_selector(quotes, decision):
    """Default policy: the newest quote already available at the decision time.

    Receives *all* quotes for the asset/venue pair, including future ones, so the
    guard -- not the selector -- is the authority on causality. A deliberately
    leaky selector is therefore still caught. Returns ``None`` when no observation
    was available yet.
    """
    eligible = [quote for quote in quotes if quote.available_ns <= decision.decision_ns]
    if not eligible:
        return None
    return max(eligible, key=lambda quote: (quote.available_ns, quote.event_ns, quote.venue))


# --------------------------------------------------------------------------
# Ledger with an independent conservation recomputation
# --------------------------------------------------------------------------
class Ledger:
    """Cash, signed position, PnL, fee, funding and liquidation accounting for one replay.

    Positions are **signed** in contracts: positive is long, negative is short. The
    quote-currency cost basis carries the same sign, so ``average_cost`` stays
    positive for either direction and ``unrealized = market_value - cost_basis`` is
    correct without a special case.

    A *position key* identifies the accounting bucket. Under ``one_way`` netting it
    is the ``asset_id`` (venues aggregated, as in the spot slice). Under ``hedge``
    mode it is ``asset_id::long`` / ``asset_id::short``, so the two legs never net
    against each other.
    """

    def __init__(self, initial_cash, *, multipliers=None, position_mode=POSITION_ONE_WAY):
        self.initial_cash = _nonnegative(initial_cash, "initial_cash")
        self.cash = self.initial_cash
        self.fees_paid = 0.0
        self.liquidation_fees_paid = 0.0
        self.realized_pnl = 0.0
        self.positions = {}
        self.cost_basis = {}
        self.asset_of_key = {}
        self.venue_of_key = {}
        self.allocated_margin = {}
        self.leverage_by_key = {}
        self.multipliers = dict(multipliers or {})
        self.position_mode = _choice(position_mode, "position_mode", POSITION_MODES)
        self.fill_count = 0
        self.gross_traded_notional = 0.0
        self.turnover_by_asset = {}
        self.realized_pnl_by_asset = {}
        self.fees_by_asset = {}
        self.funding_by_asset = {}
        self.funding_by_key = {}
        self.funding_payments = []
        self.funding_cost = 0.0
        self.liquidations = []
        self.cash_entries = []
        self.fills = []
        self.status_history = []

    def multiplier(self, asset_id):
        return float(self.multipliers.get(asset_id, 1.0))

    def register_key(self, key, asset_id):
        existing = self.asset_of_key.get(key)
        if existing is not None and existing != asset_id:
            raise SimulationError(f"position key {key!r} is already bound to {existing!r}")
        self.asset_of_key[key] = asset_id
        return key

    def key_quantity(self, key):
        return self.positions.get(key, 0.0)

    def asset_net_quantity(self, asset_id):
        return math.fsum(quantity for key, quantity in self.positions.items()
                         if self.asset_of_key.get(key, key) == asset_id)

    def keys_for_asset(self, asset_id):
        return sorted(key for key in self.positions
                      if self.asset_of_key.get(key, key) == asset_id)

    def allocated_margin_total(self):
        return math.fsum(self.allocated_margin.values())

    # -- fills --------------------------------------------------------------
    def apply_fill(self, *, order_id, fill_index, asset_id, venue, side, quantity, execution_ns,
                   quote, mid, execution_price, spread_cost, slippage_cost, fee, limit_price,
                   position_key=None, multiplier=None, margin_rate=0.0, is_liquidation=False,
                   liquidation_id=None, order_effect=EFFECT_AUTO):
        """Apply one fill to the signed position for ``position_key``.

        ``side`` is the order direction (buy increases the signed position, sell
        decreases it). Reducing an existing position realizes PnL against its average
        cost and releases margin pro rata; crossing through zero realizes the closed
        part and opens the remainder at the execution price.
        """
        quantity = _nonnegative(quantity, "fill quantity")
        if quantity <= 0:
            raise ValueError("fill quantity must be positive")
        execution_price = _nonnegative(execution_price, "execution_price")
        fee = _nonnegative(fee, "fee")
        key = self.register_key(position_key if position_key is not None else asset_id, asset_id)
        # Under one-way netting a key can aggregate venues; the most recent fill's venue is
        # recorded so per-venue guard reporting has a declared, reproducible answer.
        self.venue_of_key[key] = venue
        mult = self.multiplier(asset_id) if multiplier is None else _positive(multiplier, "multiplier")
        margin_rate = _nonnegative(margin_rate, "margin_rate")
        notional = execution_price * quantity * mult
        fill_id = f"{order_id}::fill-{fill_index:04d}"

        held_before = self.positions.get(key, 0.0)
        cost_before = self.cost_basis.get(key, 0.0)
        allocated_before = self.allocated_margin.get(key, 0.0)
        direction = 1.0 if side == SIDE_BUY else -1.0
        remaining = quantity
        realized = 0.0
        closing = 0.0

        if held_before != 0.0 and (held_before > 0.0) != (direction > 0.0):
            held_start = held_before
            closing = min(quantity, abs(held_before))
            sign = 1.0 if held_before > 0.0 else -1.0
            average_cost = cost_before / (held_before * mult)
            realized += closing * sign * (execution_price - average_cost) * mult
            cost_before -= closing * sign * average_cost * mult
            held_before -= closing * sign
            released = (closing / abs(held_start)) * allocated_before
            self.allocated_margin[key] = allocated_before - released
            remaining = quantity - closing

        if remaining > 0.0:
            held_before += direction * remaining
            cost_before += direction * remaining * execution_price * mult
            self.allocated_margin[key] = self.allocated_margin.get(key, 0.0) \
                + remaining * execution_price * mult * margin_rate

        if closing > 0.0 and abs(held_before) <= FLAT_QUANTITY_TOLERANCE * max(1.0, quantity):
            # A position reduced to floating-point dust is flat, not a liquidatable residual.
            held_before = 0.0
            cost_before = 0.0
            self.allocated_margin[key] = 0.0

        self.positions[key] = held_before
        self.cost_basis[key] = cost_before
        if margin_rate > 0.0:
            self.leverage_by_key[key] = 1.0 / margin_rate if margin_rate else None
        self.realized_pnl += realized
        self.realized_pnl_by_asset[asset_id] = self.realized_pnl_by_asset.get(asset_id, 0.0) + realized
        self.fees_by_asset[asset_id] = self.fees_by_asset.get(asset_id, 0.0) + fee
        self.turnover_by_asset[asset_id] = self.turnover_by_asset.get(asset_id, 0.0) + notional

        cash_flow = -notional if side == SIDE_BUY else notional
        self.cash += cash_flow
        self.cash_entries.append({"entry_id": f"{fill_id}::trade", "ns": execution_ns,
                                  "order_id": order_id, "fill_id": fill_id,
                                  "reason": LEDGER_REASON_TRADE,
                                  "amount": cash_flow, "cash_after": self.cash,
                                  "is_liquidation": bool(is_liquidation),
                                  "liquidation_id": liquidation_id})
        self.cash -= fee
        self.fees_paid += fee
        self.cash_entries.append({"entry_id": f"{fill_id}::explicit-fee", "ns": execution_ns,
                                  "order_id": order_id, "fill_id": fill_id,
                                  "reason": LEDGER_REASON_EXPLICIT_FEE,
                                  "amount": -fee, "cash_after": self.cash,
                                  "is_liquidation": bool(is_liquidation),
                                  "liquidation_id": liquidation_id})
        fill = {"fill_id": fill_id, "order_id": order_id, "asset_id": asset_id, "venue": venue,
                "position_key": key, "side": side, "quantity": quantity,
                "contract_multiplier": mult, "position_effect": order_effect,
                "execution_ns": execution_ns,
                "quote_event_ns": quote.event_ns, "quote_available_ns": quote.available_ns,
                "quote_source_id": quote.source_id, "quote_version": quote.version,
                "mid_price": mid, "execution_price": execution_price,
                "spread_cost": spread_cost, "slippage_cost": slippage_cost,
                "explicit_fee": fee, "notional": notional, "limit_price": limit_price,
                "realized_pnl": realized, "closed_quantity": closing,
                "position_after": self.positions[key],
                "is_liquidation": bool(is_liquidation), "liquidation_id": liquidation_id,
                "cash_flow": cash_flow, "cash_after": self.cash}
        self.fills.append(fill)
        self.fill_count += 1
        self.gross_traded_notional += notional
        return fill

    # -- funding and liquidation -------------------------------------------
    def apply_funding(self, *, payment_id, asset_id, venue, position_key, ns, quantity, multiplier,
                      mark_price, funding_rate, rate_source):
        """Charge one declared funding payment on the mark-price notional.

        ``amount = -quantity * multiplier * mark_price * funding_rate``: a long pays
        when the declared rate is positive and receives when it is negative, and a
        short does the opposite. ``funding_cost`` accumulates the paid side, so it
        is monotone in the funding rate for a fixed position.
        """
        mark_price = _positive(mark_price, "mark_price")
        funding_rate = _finite(funding_rate, "funding_rate")
        quantity = _finite(quantity, "funding quantity")
        multiplier = _positive(multiplier, "multiplier")
        notional = quantity * multiplier * mark_price
        amount = -notional * funding_rate
        self.cash += amount
        self.funding_cost += -amount
        self.funding_by_asset[asset_id] = self.funding_by_asset.get(asset_id, 0.0) - amount
        self.funding_by_key[position_key] = self.funding_by_key.get(position_key, 0.0) - amount
        self.cash_entries.append({"entry_id": f"{payment_id}::funding", "ns": ns,
                                  "order_id": None, "fill_id": None,
                                  "reason": LEDGER_REASON_FUNDING, "amount": amount,
                                  "cash_after": self.cash})
        payment = {"payment_id": payment_id, "reason_code": REASON_FUNDING, "ns": ns,
                   "asset_id": asset_id, "venue": venue,
                   "position_key": position_key, "position_quantity": quantity,
                   "contract_multiplier": multiplier, "mark_price": mark_price,
                   "funding_rate": funding_rate, "funding_rate_source": rate_source,
                   "notional": notional, "cash_flow": amount, "funding_cost": -amount,
                   "cash_after": self.cash}
        self.funding_payments.append(payment)
        return payment

    def apply_liquidation_fee(self, *, liquidation_id, asset_id, ns, fee):
        fee = _nonnegative(fee, "liquidation fee")
        self.cash -= fee
        self.liquidation_fees_paid += fee
        self.cash_entries.append({"entry_id": f"{liquidation_id}::liquidation-fee", "ns": ns,
                                  "order_id": None, "fill_id": None,
                                  "reason": LEDGER_REASON_LIQUIDATION_FEE,
                                  "amount": -fee, "cash_after": self.cash,
                                  "liquidation_id": liquidation_id})
        return fee

    def record_liquidation(self, record):
        self.liquidations.append(record)
        return record

    # -- reporting ----------------------------------------------------------
    def to_dict(self, marks):
        positions = {}
        unrealized = 0.0
        market_value_total = 0.0
        unrealized_by_asset = {}
        for key in sorted(set(self.positions) | set(marks)):
            quantity = self.positions.get(key, 0.0)
            cost = self.cost_basis.get(key, 0.0)
            asset_id = self.asset_of_key.get(key, key)
            mult = self.multiplier(asset_id)
            mark = marks[key]
            market_value = quantity * mult * mark["price"]
            entry_unrealized = market_value - cost
            unrealized += entry_unrealized
            unrealized_by_asset[asset_id] = unrealized_by_asset.get(asset_id, 0.0) + entry_unrealized
            market_value_total += market_value
            positions[key] = {
                "position_key": key, "asset_id": asset_id, "quantity": quantity,
                "contract_multiplier": mult, "cost_basis": cost,
                "average_cost": cost / (quantity * mult) if quantity else 0.0,
                "allocated_margin": self.allocated_margin.get(key, 0.0),
                "mark_price": mark["price"], "mark_ns": mark["ns"], "mark_source": mark["source"],
                "mark_venue": mark["venue"], "market_value": market_value,
                "unrealized_pnl": entry_unrealized}
        equity = self.cash + market_value_total
        net_pnl = equity - self.initial_cash
        gross_position_notional = math.fsum(
            abs(entry["quantity"]) * entry["contract_multiplier"] * entry["mark_price"]
            for entry in positions.values())
        ledger = {
            "schema_version": SCHEMA,
            "position_scope": ("signed positions in contracts; "
                               + ("one-way netting aggregated per asset_id across venues"
                                  if self.position_mode == POSITION_ONE_WAY
                                  else "hedge mode: long and short legs never net")),
            "position_mode": self.position_mode,
            "initial_cash": self.initial_cash,
            "cash": self.cash,
            "fees_paid": self.fees_paid,
            "liquidation_fees_paid": self.liquidation_fees_paid,
            "funding_cost": self.funding_cost,
            "funding_payment_count": len(self.funding_payments),
            "realized_pnl": self.realized_pnl,
            "realized_pnl_by_asset": self.realized_pnl_by_asset,
            "fees_by_asset": self.fees_by_asset,
            "funding_cost_by_asset": self.funding_by_asset,
            "funding_cost_by_position_key": self.funding_by_key,
            "turnover_by_asset": self.turnover_by_asset,
            "unrealized_pnl": unrealized,
            "unrealized_pnl_by_asset": unrealized_by_asset,
            "market_value": market_value_total,
            "gross_position_notional": gross_position_notional,
            "allocated_margin_total": self.allocated_margin_total(),
            "equity": equity,
            "net_pnl": net_pnl,
            "net_pnl_excluding_explicit_fees": net_pnl + self.fees_paid,
            "net_pnl_excluding_all_costs": (net_pnl + self.fees_paid
                                            + self.liquidation_fees_paid + self.funding_cost),
            "gross_traded_notional": self.gross_traded_notional,
            "fill_count": self.fill_count,
            "liquidation_count": len(self.liquidations),
            "contract_multipliers": dict(self.multipliers),
            "positions": positions,
            "funding_payments": self.funding_payments,
            "liquidations": self.liquidations,
            "cash_entries": self.cash_entries,
            "fills": self.fills,
        }
        ledger["conservation"] = conservation_report(ledger)
        return ledger


def _max_abs_difference(left, right):
    keys = set(left) | set(right)
    return max((abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys), default=0.0)


def conservation_report(ledger):
    """Recompute the ledger from its own raw events and report every residual.

    This is deliberately an independent second implementation of the accounting:
    it replays the fill list into signed positions, cost basis, realized PnL and
    fees, replays the funding list and the liquidation-fee entries, then compares
    that replay against the stored aggregates. Tampering with any fill, funding
    payment or cash entry therefore produces a nonzero residual.
    """
    fills = ledger["fills"]
    entries = ledger["cash_entries"]
    multipliers = ledger.get("contract_multipliers", {})
    replayed_cash = ledger["initial_cash"] + sum(entry["amount"] for entry in entries)
    replayed_fees = 0.0
    replayed_positions = {}
    replayed_cost_basis = {}
    replayed_realized = 0.0
    replayed_fill_cash = 0.0
    liquidation_not_flat = 0
    for fill in fills:
        asset_id = fill["asset_id"]
        key = fill.get("position_key", asset_id)
        quantity = fill["quantity"]
        price = fill["execution_price"]
        mult = float(multipliers.get(asset_id, fill.get("contract_multiplier", 1.0)))
        replayed_fees += fill["explicit_fee"]
        direction = 1.0 if fill["side"] == SIDE_BUY else -1.0
        held = replayed_positions.get(key, 0.0)
        cost = replayed_cost_basis.get(key, 0.0)
        if held != 0.0 and (held > 0.0) != (direction > 0.0):
            closing = min(quantity, abs(held))
            sign = 1.0 if held > 0.0 else -1.0
            average_cost = cost / (held * mult)
            replayed_realized += closing * sign * (price - average_cost) * mult
            cost -= closing * sign * average_cost * mult
            held -= closing * sign
            replayed_fill_cash -= direction * closing * price * mult
            remainder = quantity - closing
        else:
            remainder = quantity
        if remainder > 0.0:
            held += direction * remainder
            cost += direction * remainder * price * mult
            replayed_fill_cash -= direction * remainder * price * mult
        replayed_positions[key] = held
        replayed_cost_basis[key] = cost
        if fill.get("is_liquidation") and abs(held) > 1e-9 * max(1.0, abs(quantity)):
            liquidation_not_flat += 1
    replayed_fill_cash -= replayed_fees

    funding_payments = ledger.get("funding_payments", [])
    replayed_funding = math.fsum(payment["cash_flow"] for payment in funding_payments)
    replayed_funding_entries = math.fsum(entry["amount"] for entry in entries
                                         if entry["reason"] == LEDGER_REASON_FUNDING)
    replayed_liquidation_fee_entries = math.fsum(entry["amount"] for entry in entries
                                                 if entry["reason"] == LEDGER_REASON_LIQUIDATION_FEE)

    stored_positions = {key: entry["quantity"] for key, entry in ledger["positions"].items()}
    stored_cost = {key: entry["cost_basis"] for key, entry in ledger["positions"].items()}
    market_value = math.fsum(entry["market_value"] for entry in ledger["positions"].values())
    scale = max(1.0, ledger["initial_cash"], ledger["gross_traded_notional"],
                abs(ledger["cash"]), sum(abs(entry["amount"]) for entry in entries) or 0.0)
    residuals = {
        "cash_vs_cash_entries": ledger["cash"] - replayed_cash,
        "cash_entries_vs_event_cash_flows":
            replayed_cash - (ledger["initial_cash"] + replayed_fill_cash + replayed_funding
                             + replayed_liquidation_fee_entries),
        "fees_vs_fill_fees": ledger["fees_paid"] - replayed_fees,
        "positions_vs_fill_quantities": _max_abs_difference(stored_positions, replayed_positions),
        "cost_basis_vs_fill_replay": _max_abs_difference(stored_cost, replayed_cost_basis),
        "realized_pnl_vs_fill_replay": ledger["realized_pnl"] - replayed_realized,
        "equity_vs_cash_plus_market_value": ledger["equity"] - (ledger["cash"] + market_value),
        "market_value_vs_positions": ledger["market_value"] - market_value,
        "net_pnl_vs_equity": ledger["net_pnl"] - (ledger["equity"] - ledger["initial_cash"]),
        "fees_inside_net_pnl": ledger["net_pnl_excluding_explicit_fees"]
            - (ledger["net_pnl"] + ledger["fees_paid"]),
        "all_costs_inside_net_pnl": ledger["net_pnl_excluding_all_costs"]
            - (ledger["net_pnl"] + ledger["fees_paid"] + ledger["liquidation_fees_paid"]
               + ledger["funding_cost"]),
        "funding_cost_vs_payments": ledger["funding_cost"] + replayed_funding,
        "funding_payments_vs_cash_entries": replayed_funding - replayed_funding_entries,
        "liquidation_fees_vs_cash_entries": -ledger["liquidation_fees_paid"]
            - replayed_liquidation_fee_entries,
    }
    tolerance = 1e-9 * scale
    violations = sorted(name for name, residual in residuals.items() if abs(residual) > tolerance)
    negative_fees = sum(1 for fill in fills if fill["explicit_fee"] < 0)
    negative_cash_entries = sum(1 for entry in entries if entry["amount"] != entry["amount"]
                                or not math.isfinite(entry["amount"]))
    if negative_fees or negative_cash_entries:
        violations.append("fee_or_cash_sign")
    if liquidation_not_flat:
        violations.append("liquidation_did_not_flatten_position")
    return {"ok": not violations, "tolerance": tolerance, "scale": scale,
            "residuals": residuals, "violations": violations,
            "negative_fee_count": negative_fees,
            "non_finite_cash_entry_count": negative_cash_entries,
            "liquidation_not_flat_count": liquidation_not_flat}


def require_conservation(ledger):
    """Independently recompute conservation; the stored report is a receipt, not proof."""
    report = conservation_report(ledger)
    if not report["ok"]:
        raise LedgerConservationError(
            f"ledger conservation failed: {report['violations']} residuals={report['residuals']}")
    return report


def validate_status_history(order):
    """Verify the order actually followed the declared lifecycle."""
    history = order["status_history"]
    if not history or history[0]["status"] != ORDER_SUBMITTED:
        raise SimulationError("order status history must start with submitted")
    previous_ns = None
    for index, entry in enumerate(history):
        status = entry["status"]
        if status not in ORDER_STATUSES:
            raise SimulationError(f"unknown order status {status!r}")
        _timestamp(entry["ns"], "status ns")
        if previous_ns is not None and entry["ns"] < previous_ns:
            raise SimulationError("order status timestamps must be non-decreasing")
        previous_ns = entry["ns"]
        if index:
            allowed = ALLOWED_TRANSITIONS[history[index - 1]["status"]]
            if status not in allowed:
                raise SimulationError(
                    f"illegal order transition {history[index - 1]['status']} -> {status}")
    terminal = [entry["status"] for entry in history if entry["status"] in ORDER_TERMINAL_STATUSES]
    if [entry["status"] for entry in history][-1] not in ORDER_TERMINAL_STATUSES:
        raise SimulationError("order status history must end in a terminal status")
    if len(terminal) != 1:
        raise SimulationError("order must reach exactly one terminal status")
    return order


def marks_at(quotes_by_asset, cutoff_ns, *, price_source=PRICE_SOURCE_MARK):
    """Newest available price per asset at or before ``cutoff_ns``, cost-basis free.

    ``price`` is selected by ``price_source`` (``mark``/``last``/``mid``); the
    ``mark_price`` and ``last_price`` fields are always reported separately so a
    receipt shows both even when only one was used for valuation.
    """
    _choice(price_source, "price_source", PRICE_SOURCES)
    marks = {}
    for asset_id, quotes in quotes_by_asset.items():
        eligible = [quote for quote in quotes if quote.available_ns <= cutoff_ns]
        if not eligible:
            continue
        best = max(eligible, key=lambda quote: (quote.available_ns, quote.event_ns, quote.venue))
        marks[asset_id] = {"price": best.price_for(price_source), "ns": best.available_ns,
                           "event_ns": best.event_ns, "venue": best.venue,
                           "source": "synthetic_quote", "price_source": price_source,
                           "mark_price": best.declared_mark_price,
                           "last_price": best.declared_last_price,
                           "funding_rate": best.funding_rate}
    return marks


def _round_to_step(value, step, direction):
    """Round ``value`` onto a ``step`` grid; ``direction`` is +1 up, -1 down."""
    if step <= 0:
        return value
    ticks = value / step
    if direction > 0:
        return math.ceil(ticks - 1e-9) * step
    return math.floor(ticks + 1e-9) * step


def _floor_to_step(value, step):
    if step <= 0:
        return value
    return math.floor(value / step + 1e-9) * step



class ExecutionSimulator:
    """Deterministic event-time replay of synthetic decisions against synthetic quotes."""

    def __init__(self, policy=None, *, seed=0, initial_cash=1_000_000.0, risk_gate=None,
                 guard=None, reference_selector=None, clock_skew_ns=0, contracts=(),
                 trace_perp_curve=None):
        self.policy = policy or ExecutionPolicy()
        if type(seed) is not int:
            raise ValueError("seed requires an integer")
        self.seed = seed
        self.initial_cash = _nonnegative(initial_cash, "initial_cash")
        self.guard = guard or NoFutureInformationGuard()
        self.reference_selector = reference_selector or causal_reference_selector
        self.risk_gate = risk_gate
        self.clock_skew_ns = _timestamp(clock_skew_ns, "clock_skew_ns")
        self.contracts = {}
        for spec in contracts:
            if not isinstance(spec, ContractSpec):
                raise ValueError("contracts requires ContractSpec instances")
            if spec.key in self.contracts:
                raise ValueError(f"duplicate contract declaration for {spec.key}")
            if spec.margin_mode != self.policy.perps.margin_mode:
                raise ValueError(
                    f"contract {spec.key} declares margin_mode={spec.margin_mode!r} but the account "
                    f"policy declares {self.policy.perps.margin_mode!r}; mixed account-level margin "
                    "modes are not modelled")
            self.contracts[spec.key] = spec
        self.multipliers = {}
        for spec in self.contracts.values():
            existing = self.multipliers.get(spec.asset_id)
            if existing is not None and existing != spec.contract_multiplier:
                raise ValueError(f"{REASON_MULTIPLIER_CONFLICT}: {spec.asset_id}")
            self.multipliers[spec.asset_id] = spec.contract_multiplier
        self.trace_perp_curve = (self.policy.perps.trace_perp_curve
                                 if trace_perp_curve is None else bool(trace_perp_curve))
        self._curve = []
        self._clock_cursor = 0
        self._perp_obs_ns = ()
        self._liquidation_sequence = 0
        self._funding_sequence = 0

    # -- public API ---------------------------------------------------------
    def contract_for(self, asset_id, venue):
        return self.contracts.get((asset_id, venue))

    def position_key(self, asset_id, venue, leg, ledger=None):
        """Accounting bucket for one leg. ``one_way`` keys on the asset (venues aggregated)."""
        if self.policy.perps.position_mode == POSITION_HEDGE:
            return f"{asset_id}::{leg}"
        return asset_id

    def run(self, quotes, decisions, *, asof_ns, replay_id="replay"):
        asof = _timestamp(asof_ns, "asof_ns")
        ordered_quotes = sorted(quotes, key=lambda quote: (quote.asset_id, quote.venue,
                                                           quote.event_ns, quote.available_ns))
        by_asset = {}
        by_key = {}
        for quote in ordered_quotes:
            by_asset.setdefault(quote.asset_id, []).append(quote)
            by_key.setdefault((quote.asset_id, quote.venue), []).append(quote)
        ordered_decisions = sorted(decisions, key=lambda item: (item.decision_ns, item.decision_id))

        self._perp_obs_ns = tuple(sorted({quote.available_ns for quote in ordered_quotes
                                          if self._is_perp_asset(quote.asset_id, quote.venue)}))
        self._mark_index = self._build_mark_index(by_asset)
        self._curve = []
        self._liquidation_sequence = 0
        self._funding_sequence = 0
        starts = [asof] + [quote.available_ns for quote in ordered_quotes] \
            + [decision.decision_ns for decision in ordered_decisions]
        self._clock_cursor = max(0, min(starts) - 1) if starts else 0

        ledger = Ledger(self.initial_cash, multipliers=self.multipliers,
                        position_mode=self.policy.perps.position_mode)
        decision_log = []
        orders = []
        for order_index, decision in enumerate(ordered_decisions):
            self._advance_perp_clock(decision.decision_ns, ledger, by_asset)
            record, new_orders = self._process_decision(
                decision, by_key.get((decision.asset_id, decision.venue), []), by_asset, ledger,
                order_index)
            decision_log.append(record)
            orders.extend(new_orders)
        self._advance_perp_clock(asof, ledger, by_asset)

        marks = self._marks_upto(asof)
        marks_by_key = self._marks_by_position_key(marks, ledger)
        ledger_dict = ledger.to_dict(marks_by_key)
        require_conservation(ledger_dict)
        self._append_curve_point(asof, ledger, by_asset, final=True)

        counts = {status: sum(1 for order in orders if order["status"] == status)
                  for status in ORDER_TERMINAL_STATUSES}
        counts.update({"orders": len(orders),
                       "fills": ledger.fill_count,
                       "decisions": len(decision_log),
                       "partially_filled_orders": sum(
                           1 for order in orders
                           if 0.0 < order["filled_quantity"] < order["requested_quantity"] - 1e-12),
                       "unfilled_orders": sum(1 for order in orders if order["filled_quantity"] == 0.0),
                       "abstain_decisions": sum(1 for record in decision_log
                                                if record["action"] == ACTION_ABSTAIN),
                       "no_trade_decisions": sum(1 for record in decision_log
                                                 if record["action"] == ACTION_NO_TRADE),
                       "no_order_decisions": sum(1 for record in decision_log
                                                 if record["outcome"] == "no_order"
                                                 and record["action"] not in (ACTION_ABSTAIN,
                                                                              ACTION_NO_TRADE)),
                       "rejected_decisions": sum(1 for record in decision_log
                                                 if record["outcome"] == "rejected"),
                       "risk_blocked_decisions": sum(1 for record in decision_log
                                                     if record["outcome"] == "risk_blocked"),
                       "funding_payments": len(ledger.funding_payments),
                       "liquidations": len(ledger.liquidations)})
        # Every decision lands in exactly one bucket, so the counts always close; this
        # closes the auditability gap the P2 review raised about unexplained residuals.
        counts["ordered_decisions"] = sum(1 for record in decision_log
                                          if record["outcome"] == "order_submitted")
        counts["decision_closes"] = (
            counts["abstain_decisions"] + counts["no_trade_decisions"]
            + counts["risk_blocked_decisions"] + counts["rejected_decisions"]
            + counts["no_order_decisions"] + counts["ordered_decisions"]) == counts["decisions"]
        result = {
            "schema_version": SCHEMA,
            "scope": SCOPE,
            "perp_scope": PERP_SCOPE,
            "replay_id": replay_id,
            "seed": self.seed,
            "asof_ns": asof,
            "clock_skew_ns": self.clock_skew_ns,
            "initial_cash": self.initial_cash,
            "policy": self.policy.to_dict(),
            "contracts": [spec.to_dict() for spec in
                          sorted(self.contracts.values(), key=lambda item: item.key)],
            "guard": self.guard.snapshot(),
            "provisional_pending_r1": list(PROVISIONAL_POLICY_PARAMETERS),
            "counts": counts,
            "decisions": decision_log,
            "orders": orders,
            "marks": {key: {"price": mark["price"], "ns": mark["ns"], "source": mark["source"],
                            "venue": mark["venue"], "event_ns": mark["event_ns"],
                            "price_source": mark.get("price_source"),
                            "mark_price": mark.get("mark_price"),
                            "last_price": mark.get("last_price")}
                      for key, mark in marks_by_key.items()},
            "ledger": ledger_dict,
        }
        if self.trace_perp_curve:
            result["perp_curve"] = list(self._curve)
        result["ledger_sha256"] = sha256_bytes(canonical_json(ledger_dict).encode())
        result["replay_sha256"] = sha256_bytes(canonical_json(result).encode())
        return result

    # -- perp clock: funding and liquidation ---------------------------------
    def _is_perp_asset(self, asset_id, venue):
        spec = self.contract_for(asset_id, venue)
        return spec is not None and spec.is_perp

    def _build_mark_index(self, by_asset):
        """Per-asset observation index so a mark lookup is O(log n) not O(n)."""
        index = {}
        for asset_id, quotes in by_asset.items():
            ordered = sorted(quotes, key=lambda quote: (quote.available_ns, quote.event_ns,
                                                        quote.venue))
            index[asset_id] = ([(quote.available_ns, quote.event_ns, quote.venue)
                                for quote in ordered], ordered)
        return index

    def _mark_quote(self, asset_id, cutoff_ns):
        entry = self._mark_index.get(asset_id)
        if entry is None:
            return None
        keys, quotes = entry
        position = bisect_right(keys, (cutoff_ns, float("inf"), "\uffff"))
        return quotes[position - 1] if position else None

    def _marks_upto(self, cutoff_ns):
        marks = {}
        for asset_id in self._mark_index:
            quote = self._mark_quote(asset_id, cutoff_ns)
            if quote is None:
                continue
            marks[asset_id] = {
                "price": quote.price_for(self.policy.marks.price_source),
                "ns": quote.available_ns, "event_ns": quote.event_ns, "venue": quote.venue,
                "source": "synthetic_quote",
                "price_source": self.policy.marks.price_source,
                "mark_price": quote.declared_mark_price,
                "last_price": quote.declared_last_price,
                "funding_rate": quote.funding_rate,
            }
        return marks

    def _is_perp_position_key(self, key, ledger):
        asset_id = ledger.asset_of_key.get(key, key)
        return self._is_perp_asset(asset_id, ledger.venue_of_key.get(key, ""))

    def _marks_by_position_key(self, marks, ledger):
        by_key = {}
        for asset_id, mark in marks.items():
            if self.policy.perps.position_mode == POSITION_HEDGE:
                for leg in LEGS:
                    by_key.setdefault(f"{asset_id}::{leg}", mark)
            else:
                by_key.setdefault(asset_id, mark)
        for key, quantity in ledger.positions.items():
            if key not in by_key and quantity != 0.0:
                by_key[key] = self._fallback_mark(ledger, key)
        return by_key

    def _funding_boundaries_between(self, after_ns, up_to_ns):
        policy = self.policy.perps
        interval = policy.funding_interval_ns
        anchor = policy.funding_anchor_ns
        if interval <= 0 or up_to_ns <= after_ns:
            return []
        first = anchor + ((after_ns - anchor) // interval + 1) * interval
        if first > up_to_ns:
            return []
        count = (up_to_ns - first) // interval + 1
        if count > 100_000:
            raise SimulationError(
                f"funding interval {interval} would emit {count} payments in one window; "
                "declare a realistic funding interval")
        return [first + index * interval for index in range(count)]

    def _advance_perp_clock(self, up_to_ns, ledger, by_asset):
        """Replay funding boundaries and margin checks up to ``up_to_ns`` (exclusive of cursor).

        Every synthetic price observation in the window is visited exactly once and in
        ascending order, so a breach that later recovers is still detected and no
        observation after ``up_to_ns`` can influence the state at ``up_to_ns``.
        """
        if not self.contracts or up_to_ns <= self._clock_cursor:
            self._clock_cursor = max(self._clock_cursor, up_to_ns)
            return
        if not any(quantity != 0.0 for quantity in ledger.positions.values()):
            self._clock_cursor = up_to_ns
            return
        boundaries = set(self._funding_boundaries_between(self._clock_cursor, up_to_ns))
        low = bisect_right(self._perp_obs_ns, self._clock_cursor)
        high = bisect_right(self._perp_obs_ns, up_to_ns)
        events = sorted(boundaries | set(self._perp_obs_ns[low:high]))
        for ns in events:
            if ns in boundaries:
                self._apply_funding(ns, ledger, by_asset)
            self._check_liquidation(ns, ledger, by_asset)
            self._append_curve_point(ns, ledger, by_asset)
        self._clock_cursor = up_to_ns

    def _apply_funding(self, ns, ledger, by_asset):
        marks = self._marks_upto(ns)
        policy = self.policy.perps
        for key in sorted(ledger.positions):
            quantity = ledger.positions.get(key, 0.0)
            if quantity == 0.0:
                continue
            asset_id = ledger.asset_of_key.get(key, key)
            spec = self.contract_for(asset_id, ledger.venue_of_key.get(key, ""))
            mark = marks.get(asset_id)
            if mark is None:
                # No usable observation yet: funding cannot be computed and is not invented.
                continue
            quote = self._latest_quote(by_asset, asset_id, ns)
            rate, source = policy.funding_rate_at(asset_id, ledger.venue_of_key.get(key, ""), ns, quote)
            mark_price = mark["price"]
            self._funding_sequence += 1
            payment_id = f"funding::{key}::{ns}::{self._funding_sequence:05d}"
            ledger.apply_funding(payment_id=payment_id, asset_id=asset_id,
                                 venue=ledger.venue_of_key.get(key, ""), position_key=key, ns=ns,
                                 quantity=quantity,
                                 multiplier=spec.contract_multiplier if spec
                                 else ledger.multiplier(asset_id),
                                 mark_price=mark_price, funding_rate=rate, rate_source=source)

    def _latest_quote(self, by_asset, asset_id, cutoff_ns):
        return self._mark_quote(asset_id, cutoff_ns)

    def _check_liquidation(self, ns, ledger, by_asset):
        if not any(self._is_perp_position_key(key, ledger) for key in ledger.positions
                   if ledger.positions[key] != 0.0):
            return []
        snapshot = self._portfolio_snapshot(ledger, by_asset, ns)
        if not snapshot["positions"]:
            return []
        mode = self.policy.perps.margin_mode
        triggered = []
        tolerance = 1e-9 * max(1.0, snapshot["margin_balance"])
        if mode == MARGIN_CROSS:
            if snapshot["margin_balance"] <= snapshot["maintenance_margin_required"] + tolerance:
                triggered = sorted(key for key, quantity in snapshot["positions"].items()
                                   if quantity != 0.0
                                   and snapshot["position_detail"][key]["is_perp"])
        else:
            for key, detail in snapshot["position_detail"].items():
                if detail["quantity"] == 0.0 or not detail["is_perp"]:
                    continue
                leg_equity = (detail["allocated_margin"] + detail["unrealized_pnl"]
                              - detail["funding_cost_to_date"])
                if leg_equity <= detail["maintenance_margin"] + tolerance:
                    triggered.append(key)
        records = []
        for key in triggered:
            records.append(self._liquidate(ns, key, ledger, snapshot, by_asset))
        return records

    def _liquidate(self, ns, key, ledger, snapshot, by_asset):
        detail = snapshot["position_detail"][key]
        quantity = detail["quantity"]
        asset_id = detail["asset_id"]
        venue = ledger.venue_of_key.get(key, "")
        side = SIDE_SELL if quantity > 0 else SIDE_BUY
        quote = self._latest_quote(by_asset, asset_id, ns)
        if quote is None:
            return None
        self.guard.check_execution(quote.available_ns, quote.available_ns, ns, quote)
        spec = self.contract_for(asset_id, venue)
        mult = spec.contract_multiplier if spec else ledger.multiplier(asset_id)
        tick = spec.tick_size if spec else 0.0
        mark_price = _round_to_step(detail["mark_price"], tick, 1.0 if side == SIDE_BUY else -1.0)
        self._liquidation_sequence += 1
        liquidation_id = f"liquidation::{key}::{ns}::{self._liquidation_sequence:05d}"
        order_id = f"{liquidation_id}::order"
        fill = ledger.apply_fill(
            order_id=order_id, fill_index=0, asset_id=asset_id, venue=venue, side=side,
            quantity=abs(quantity), execution_ns=ns, quote=quote, mid=detail["mark_price"],
            execution_price=mark_price, spread_cost=0.0, slippage_cost=0.0, fee=0.0,
            limit_price=None, position_key=key, multiplier=mult, margin_rate=0.0,
            is_liquidation=True, liquidation_id=liquidation_id, order_effect=EFFECT_CLOSE)
        fee = self.policy.perps.liquidation_fee_bps / 10_000.0 * fill["notional"]
        ledger.apply_liquidation_fee(liquidation_id=liquidation_id, asset_id=asset_id, ns=ns, fee=fee)
        record = {
            "liquidation_id": liquidation_id, "ns": ns, "asset_id": asset_id, "venue": venue,
            "position_key": key, "margin_mode": self.policy.perps.margin_mode,
            "position_mode": self.policy.perps.position_mode,
            "position_quantity": quantity, "contract_multiplier": mult,
            "mark_price": detail["mark_price"], "last_price": detail["last_price"],
            "liquidation_price": detail["liquidation_price"],
            "margin_balance": snapshot["margin_balance"],
            "maintenance_margin": snapshot["maintenance_margin_required"],
            "margin_ratio": snapshot["margin_ratio"],
            "trigger_reason_code": REASON_LIQUIDATION,
            "close_side": side, "close_quantity": abs(quantity),
            "close_price": mark_price, "realized_pnl": fill["realized_pnl"],
            "liquidation_fee": fee, "position_after": fill["position_after"],
            "cash_after": ledger.cash, "fill_id": fill["fill_id"],
        }
        ledger.record_liquidation(record)
        return record

    def _append_curve_point(self, ns, ledger, by_asset, final=False):
        if not self.trace_perp_curve:
            return
        if self._curve and self._curve[-1]["ns"] == ns and not final:
            return
        if not final and not any(quantity != 0.0 for quantity in ledger.positions.values()):
            return
        snapshot = self._portfolio_snapshot(ledger, by_asset, ns)
        self._curve.append({
            "ns": ns, "cash": snapshot["cash"], "equity": snapshot["equity"],
            "margin_balance": snapshot["margin_balance"],
            "allocated_margin": snapshot["allocated_margin"],
            "maintenance_margin_required": snapshot["maintenance_margin_required"],
            "margin_ratio": snapshot["margin_ratio"],
            "liquidation_distance_fraction": snapshot["liquidation_distance_fraction"],
            "unrealized_pnl": snapshot["unrealized_pnl"],
            "gross_position_notional": snapshot["gross_exposure_notional"],
            "net_position_notional": snapshot["net_exposure_notional"],
            "funding_cost_to_date": ledger.funding_cost,
            "fees_to_date": ledger.fees_paid + ledger.liquidation_fees_paid,
            "turnover_to_date": ledger.gross_traded_notional,
            "fill_count": ledger.fill_count,
            "liquidation_count": len(ledger.liquidations),
            "positions": dict(sorted(snapshot["positions"].items())),
        })

    # -- internals ----------------------------------------------------------
    def _process_decision(self, decision, candidates, by_asset, ledger, order_index):
        record = {
            "decision_id": decision.decision_id, "decision_ns": decision.decision_ns,
            "asset_id": decision.asset_id, "venue": decision.venue, "action": decision.action,
            "requested_quantity": decision.quantity, "effective_quantity": 0.0,
            "order_id": None, "order_ids": [], "outcome": "no_order", "reason_codes": [],
            "reference_price": None, "reference_event_ns": None, "reference_available_ns": None,
            "risk_decision": None, "risk_reason_codes": [], "risk_allowed_quantity": None,
            "position_key": None, "resolved_side": None, "position_effect": None,
            "target_position_quantity": None, "leverage": None, "margin_required": None,
            "available_margin": None, "contract": None, "orders": 0,
        }
        if decision.action in (ACTION_ABSTAIN, ACTION_NO_TRADE):
            record["reason_codes"] = [REASON_ABSTAIN if decision.action == ACTION_ABSTAIN
                                      else REASON_NO_TRADE]
            if decision.reason:
                record["detail"] = decision.reason
            return record, []

        reference = self.reference_selector(candidates, decision)
        if reference is None:
            record["outcome"] = "rejected"
            record["reason_codes"] = [REASON_NO_REFERENCE_PRICE]
            return record, []
        self.guard.check_decision(decision.decision_ns, reference)
        record["reference_price"] = reference.mid
        record["reference_event_ns"] = reference.event_ns
        record["reference_available_ns"] = reference.available_ns

        contract = self.contract_for(decision.asset_id, decision.venue)
        record["contract"] = contract.to_dict() if contract is not None else None
        is_perp = contract is not None and contract.is_perp
        intents, reject_codes, notes = self._plan_orders(decision, ledger, contract, is_perp)
        record.update(notes)
        if not intents:
            record["outcome"] = "rejected" if reject_codes else "no_order"
            record["reason_codes"] = reject_codes or [REASON_TARGET_SATISFIED]
            return record, []

        new_orders = []
        reason_codes = []
        for intent_index, intent in enumerate(intents):
            snapshot = self._portfolio_snapshot(ledger, by_asset, decision.decision_ns)
            context = self._risk_context(decision, reference, snapshot, ledger, intent)
            risk = self._evaluate_risk(context)
            record["risk_decision"] = risk["decision"]
            record["risk_reason_codes"] = list(risk["reason_codes"])
            record["risk_allowed_quantity"] = risk["allowed_quantity"]
            record["available_margin"] = snapshot["available_margin"]
            if risk["decision"] == "block":
                if not new_orders:
                    record["outcome"] = "risk_blocked"
                    record["reason_codes"] = list(risk["reason_codes"]) or [REASON_RISK_BLOCK]
                    return record, []
                reason_codes.extend(risk["reason_codes"])
                continue
            effective = min(intent["quantity"], risk["allowed_quantity"])
            if effective < intent["quantity"] - 1e-12:
                reason_codes.append(REASON_RISK_REDUCE)
            side = intent["side"]
            if (side == SIDE_SELL and self.policy.fills.require_inventory_for_sell
                    and not (is_perp and self.policy.perps.allow_short)):
                held = ledger.asset_net_quantity(decision.asset_id)
                if held < effective - 1e-12:
                    if not new_orders:
                        record["outcome"] = "rejected"
                        record["reason_codes"] = reason_codes + [REASON_INSUFFICIENT_INVENTORY]
                        return record, []
                    reason_codes.append(REASON_INSUFFICIENT_INVENTORY)
                    continue
            effective, capacity_codes = self._apply_order_capacity(effective, reference.mid, contract)
            reason_codes.extend(capacity_codes)
            if effective <= 0:
                if not new_orders:
                    record["outcome"] = "rejected"
                    record["reason_codes"] = reason_codes + [REASON_CAPACITY_REJECT]
                    return record, []
                reason_codes.append(REASON_CAPACITY_REJECT)
                continue
            effective, sufficiency_codes, rejected = self._apply_sufficiency(
                effective, intent, ledger, reference, snapshot)
            reason_codes.extend(sufficiency_codes)
            if rejected:
                if not new_orders:
                    record["outcome"] = "rejected"
                    record["reason_codes"] = reason_codes
                    return record, []
                continue
            effective, rounding_codes, rejected = self._apply_contract_rounding(
                effective, intent, reference)
            reason_codes.extend(rounding_codes)
            if rejected:
                if not new_orders:
                    record["outcome"] = "rejected"
                    record["reason_codes"] = reason_codes
                    return record, []
                continue

            order = self._execute(decision, reference, candidates, effective, ledger,
                                  f"{order_index:05d}-{intent_index:02d}", reason_codes, intent)
            new_orders.append(order)
            record["order_ids"].append(order["order_id"])
            record["order_id"] = new_orders[0]["order_id"]
            record["effective_quantity"] += order["requested_quantity"]
            record["margin_required"] = order["required_initial_margin"]
            record["outcome"] = "order_submitted"
            record["reason_codes"] = reason_codes + [order["status"]]
            record["orders"] = len(new_orders)
            if self.risk_gate is not None:
                record_order = getattr(self.risk_gate, "record_order", None)
                if callable(record_order):
                    record_order(context)
            for fill in order["fills"]:
                record_fill = getattr(self.risk_gate, "record_fill", None)
                if callable(record_fill):
                    record_fill(context, fill["notional"])
        if not new_orders:
            record["outcome"] = "no_order"
            record["reason_codes"] = reason_codes or [REASON_TARGET_SATISFIED]
        return record, new_orders

    def _plan_orders(self, decision, ledger, contract, is_perp):
        """Translate one decision into explicit order intents.

        Returns ``(intents, reject_codes, notes)``. An empty intent list with no reject
        code means the declared target was already satisfied.
        """
        perps = self.policy.perps
        mode = perps.position_mode
        asset_id = decision.asset_id
        venue = decision.venue
        action = decision.action
        quantity = float(decision.quantity)
        leverage = decision.leverage if decision.leverage is not None else perps.default_leverage
        long_key = self.position_key(asset_id, venue, LEG_LONG)
        short_key = self.position_key(asset_id, venue, LEG_SHORT)
        notes = {"position_key": None, "resolved_side": None, "position_effect": None,
                 "target_position_quantity": None, "leverage": leverage}

        def intent(side, qty, effect, key, leg):
            return {"side": side, "quantity": qty, "effect": effect, "position_key": key, "leg": leg,
                    "is_perp": is_perp, "contract": contract, "leverage": leverage,
                    "target_position_quantity": notes["target_position_quantity"]}

        if action in TARGET_POSITION_ACTIONS and not is_perp:
            return [], [REASON_PERP_CONTRACT_NOT_DECLARED], notes
        if action in (ACTION_OPEN_LONG, ACTION_OPEN_SHORT, ACTION_CLOSE_LONG,
                      ACTION_CLOSE_SHORT) and not is_perp:
            return [], [REASON_PERP_CONTRACT_NOT_DECLARED], notes

        if action in TARGET_POSITION_ACTIONS:
            intents = []
            if action == ACTION_FLAT:
                intents = self._flatten_intents(ledger, asset_id, venue, long_key, short_key,
                                                intent, is_perp, notes)
            elif action == ACTION_LONG:
                notes["target_position_quantity"] = quantity
                if mode == POSITION_HEDGE:
                    held = ledger.key_quantity(long_key)
                    delta = quantity - max(0.0, held)
                    if delta > 1e-12:
                        intents.append(intent(SIDE_BUY, delta, EFFECT_AUTO, long_key, LEG_LONG))
                    elif delta < -1e-12:
                        intents.append(intent(SIDE_SELL, -delta, EFFECT_CLOSE, long_key, LEG_LONG))
                else:
                    held = ledger.asset_net_quantity(asset_id)
                    delta = quantity - held
                    if delta > 1e-12:
                        intents.append(intent(SIDE_BUY, delta, EFFECT_AUTO, long_key, LEG_LONG))
                    elif delta < -1e-12:
                        intents.append(intent(SIDE_SELL, -delta, EFFECT_CLOSE, long_key,
                                              LEG_LONG if held > 0 else LEG_SHORT))
            else:
                if not perps.allow_short:
                    return [], [REASON_SHORT_DISABLED], notes
                notes["target_position_quantity"] = -quantity
                if mode == POSITION_HEDGE:
                    held = ledger.key_quantity(short_key)
                    delta = -quantity - held
                    if delta < -1e-12:
                        intents.append(intent(SIDE_SELL, -delta, EFFECT_AUTO, short_key, LEG_SHORT))
                    elif delta > 1e-12:
                        intents.append(intent(SIDE_BUY, delta, EFFECT_CLOSE, short_key, LEG_SHORT))
                else:
                    held = ledger.asset_net_quantity(asset_id)
                    delta = -quantity - held
                    if delta < -1e-12:
                        intents.append(intent(SIDE_SELL, -delta, EFFECT_AUTO, short_key, LEG_SHORT))
                    elif delta > 1e-12:
                        intents.append(intent(SIDE_BUY, delta, EFFECT_CLOSE, long_key,
                                              LEG_LONG if held > 0 else LEG_SHORT))
            if intents:
                first = intents[0]
                notes.update(position_key=first["position_key"], resolved_side=first["side"],
                             position_effect=first["effect"])
            return intents, [], notes

        side = SIDE_BUY if action in (ACTION_BUY, ACTION_OPEN_LONG) else SIDE_SELL
        effect = decision.position_effect
        if action == ACTION_OPEN_LONG:
            effect = EFFECT_OPEN
        elif action == ACTION_OPEN_SHORT:
            effect = EFFECT_OPEN
        elif action == ACTION_CLOSE_LONG:
            effect = EFFECT_CLOSE
            side = SIDE_SELL
        elif action == ACTION_CLOSE_SHORT:
            effect = EFFECT_CLOSE
            side = SIDE_BUY
        leg = LEG_LONG if side == SIDE_BUY else LEG_SHORT
        if action in (ACTION_CLOSE_LONG,):
            leg = LEG_LONG
        if action in (ACTION_CLOSE_SHORT,):
            leg = LEG_SHORT
        key = long_key if leg == LEG_LONG else short_key
        candidate = intent(side, quantity, effect, key, leg)
        if is_perp and self._intent_opens_short(candidate, ledger) and not perps.allow_short:
            return [], [REASON_SHORT_DISABLED], notes
        if is_perp and effect == EFFECT_CLOSE:
            held = ledger.key_quantity(key)
            reducible = abs(held) if (held > 0) == (side == SIDE_SELL) else 0.0
            if reducible <= 1e-12:
                return [], [REASON_NO_POSITION_TO_CLOSE], notes
            candidate["quantity"] = min(quantity, reducible)
        if is_perp and effect == EFFECT_OPEN:
            held = ledger.key_quantity(key)
            direction = 1.0 if side == SIDE_BUY else -1.0
            if held != 0.0 and (held > 0.0) != (direction > 0.0) \
                    and mode == POSITION_ONE_WAY:
                return [], [REASON_OPEN_WOULD_FLIP], notes
        notes.update(position_key=key, resolved_side=side, position_effect=effect)
        return [candidate], [], notes

    def _flatten_intents(self, ledger, asset_id, venue, long_key, short_key, intent, is_perp, notes):
        intents = []
        notes["target_position_quantity"] = 0.0
        if self.policy.perps.position_mode == POSITION_HEDGE:
            keys = [(long_key, LEG_LONG), (short_key, LEG_SHORT)]
        else:
            keys = [(long_key, None)]
        for key, leg in keys:
            held = ledger.key_quantity(key)
            if held > 1e-12:
                intents.append(intent(SIDE_SELL, held, EFFECT_CLOSE, key, leg or LEG_LONG))
            elif held < -1e-12:
                intents.append(intent(SIDE_BUY, -held, EFFECT_CLOSE, key, leg or LEG_SHORT))
        return intents

    def _intent_opens_short(self, intent, ledger):
        held = ledger.key_quantity(intent["position_key"])
        direction = 1.0 if intent["side"] == SIDE_BUY else -1.0
        projected = held + direction * intent["quantity"]
        short_before = abs(min(held, 0.0))
        short_after = abs(min(projected, 0.0))
        return short_after > short_before + 1e-12

    def _exposure_split(self, held, direction, quantity):
        """Split a requested quantity into opening and closing contracts."""
        after = held + direction * quantity
        if held != 0.0 and (after == 0.0 or (after > 0.0) == (held > 0.0)):
            opening = max(0.0, abs(after) - abs(held))
        else:
            opening = abs(after)
        return opening, max(0.0, quantity - opening)

    def _apply_order_capacity(self, quantity, reference_price, contract=None):
        policy = self.policy.capacity
        mult = contract.contract_multiplier if contract is not None else 1.0
        capped = quantity
        codes = []
        if capped > policy.max_order_quantity:
            capped = policy.max_order_quantity
            codes.append(REASON_CAPACITY_CLAMP)
        if capped * reference_price * mult > policy.max_order_notional:
            capped = min(capped, policy.max_order_notional / (reference_price * mult))
            codes.append(REASON_CAPACITY_CLAMP)
        if codes and policy.on_order_exceeds_limit == "reject":
            return 0.0, [REASON_CAPACITY_REJECT]
        return max(0.0, capped), codes

    def _apply_sufficiency(self, quantity, intent, ledger, reference, snapshot):
        """Enforce margin sufficiency (perp) or cash sufficiency (spot).

        Returns ``(quantity, reason_codes, rejected)``. Margin is never assumed: the
        opening portion of the order must fit inside ``available_margin`` at the
        declared leverage, or the order is reduced and then rejected if nothing is
        affordable. Closes and reductions never require new margin.
        """
        contract = intent["contract"]
        asset_id = ledger.asset_of_key.get(intent["position_key"], intent["position_key"])
        mult = contract.contract_multiplier if contract is not None else ledger.multiplier(asset_id)
        price = reference.mid
        key = intent["position_key"]
        held = ledger.key_quantity(key)
        direction = 1.0 if intent["side"] == SIDE_BUY else -1.0
        opening, _closing = self._exposure_split(held, direction, quantity)
        codes = []
        if intent["is_perp"]:
            leverage = intent["leverage"]
            if leverage > contract.max_leverage + 1e-12:
                return 0.0, [REASON_LEVERAGE_ABOVE_CONTRACT_MAX], True
            unit_margin = price * mult / leverage
            available = max(0.0, snapshot["available_margin"])
            allowed_opening = available / unit_margin if unit_margin > 0 else opening
            if opening > allowed_opening + 1e-12:
                codes.append(REASON_MARGIN_INSUFFICIENT)
                quantity = max(0.0, quantity - (opening - allowed_opening))
                if quantity <= 1e-12:
                    return 0.0, codes, True
                codes.append(REASON_MARGIN_REDUCE)
            return quantity, codes, False

        if not self.policy.account.require_sufficient_cash or intent["side"] != SIDE_BUY:
            return quantity, codes, False
        execution_factor = (1.0 + (self.policy.spread.half_spread_bps
                                   + self.policy.slippage.fixed_bps
                                   + self.policy.slippage.impact_bps_at_full_capacity) / 10_000.0)
        unit_cost = ((price * execution_factor + self.policy.spread.fixed_half_spread_price) * mult
                     * (1.0 + self.policy.fees.fee_bps / 10_000.0))
        if unit_cost <= 0:
            return quantity, codes, False
        usable_cash = max(0.0, ledger.cash) * (1.0 - self.policy.account.cash_buffer_fraction)
        affordable = usable_cash / unit_cost
        if quantity > affordable + 1e-12:
            if self.policy.account.on_insufficient_cash == CASH_ACTION_REJECT:
                return 0.0, [REASON_INSUFFICIENT_CASH], True
            codes.append(REASON_INSUFFICIENT_CASH)
            quantity = max(0.0, affordable)
            if quantity <= 1e-12:
                return 0.0, codes, True
            codes.append(REASON_CASH_REDUCE)
        return quantity, codes, False

    def _apply_contract_rounding(self, quantity, intent, reference):
        """Floor to the declared lot size and enforce the contract minimum notional."""
        if not intent["is_perp"]:
            return quantity, [], False
        contract = intent["contract"]
        codes = []
        rounded = _floor_to_step(quantity, contract.lot_size)
        if rounded < quantity - 1e-12:
            codes.append(REASON_LOT_SIZE_ROUND)
        if rounded <= 1e-12:
            return 0.0, codes + [REASON_MIN_NOTIONAL], True
        notional = rounded * reference.mid * contract.contract_multiplier
        if notional < contract.min_notional - 1e-12:
            return 0.0, codes + [REASON_MIN_NOTIONAL], True
        return rounded, codes, False

    def _execute(self, decision, reference, candidates, quantity, ledger, order_suffix, reason_codes,
                 intent):
        latency = self.policy.latency
        side = intent["side"]
        contract = intent["contract"]
        is_perp = intent["is_perp"]
        mult = contract.contract_multiplier if contract is not None \
            else ledger.multiplier(decision.asset_id)
        tick = contract.tick_size if contract is not None else 0.0
        leverage = intent["leverage"] if is_perp else None
        margin_rate = (1.0 / leverage) if is_perp and leverage else 0.0
        submit_ns = decision.decision_ns + latency.decision_to_order_ns
        ack_ns = submit_ns + latency.order_to_ack_ns
        first_execution_ns = ack_ns + latency.ack_to_execution_ns
        expire_ns = decision.expire_ns if decision.expire_ns is not None \
            else decision.decision_ns + self.policy.fills.order_time_to_live_ns
        # An order cannot expire before it was submitted; clamping keeps the event
        # timeline monotonic for a caller that configured an impossible time to live.
        expire_ns = max(expire_ns, submit_ns)
        order_id = f"order::{decision.decision_id}::{order_suffix}"
        history = [{"status": ORDER_SUBMITTED, "ns": submit_ns, "reason_code": REASON_SUBMITTED}]
        fills = []
        remaining = quantity
        first_execution_seen = None
        last_execution_seen = None
        terminal_reason = REASON_EXPIRED_UNFILLED
        status = ORDER_SUBMITTED

        if expire_ns <= ack_ns:
            # The venue never acknowledged the order: it lapses straight from submitted.
            history.append({"status": ORDER_EXPIRED, "ns": expire_ns,
                            "reason_code": REASON_EXPIRED_UNFILLED})
            status = ORDER_EXPIRED
        elif uniform_draw(self.seed, "order-reject",
                          decision.decision_id) < self.policy.fills.reject_probability:
            history.append({"status": ORDER_REJECTED, "ns": ack_ns, "reason_code": REASON_SYNTHETIC_REJECT})
            status = ORDER_REJECTED
        else:
            history.append({"status": ORDER_ACCEPTED, "ns": ack_ns, "reason_code": REASON_ACCEPTED})
            status = ORDER_ACCEPTED
            for opportunity_index, quote in enumerate(candidates):
                if quote.available_ns < first_execution_ns:
                    continue
                if quote.available_ns > expire_ns:
                    break
                execution_ns = quote.available_ns
                self.guard.check_execution(decision.decision_ns, submit_ns, execution_ns, quote)
                capacity_quantity = quote.volume * self.policy.capacity.max_participation_fraction
                if capacity_quantity <= 0:
                    terminal_reason = REASON_ZERO_LIQUIDITY
                    continue
                draw = uniform_draw(self.seed, "fill", decision.decision_id, opportunity_index)
                if draw >= self.policy.fills.fill_probability:
                    terminal_reason = REASON_FILL_DRAW_MISS
                    continue
                fill_quantity = min(remaining, capacity_quantity)
                if fill_quantity < self.policy.fills.minimum_fill_quantity:
                    terminal_reason = REASON_BELOW_MINIMUM_FILL
                    continue
                mid = quote.mid
                participation_ratio = fill_quantity / capacity_quantity if capacity_quantity else 1.0
                half_spread = self.policy.spread.half_spread_price(mid)
                slippage_bps = self.policy.slippage.slippage_bps(participation_ratio)
                slippage_price = mid * slippage_bps / 10_000.0
                direction = 1.0 if side == SIDE_BUY else -1.0
                raw_price = mid + direction * (half_spread + slippage_price)
                execution_price = _round_to_step(raw_price, tick, direction)
                if execution_price < 0:
                    raise SimulationError("synthetic cost policy produced a negative execution price")
                if tick > 0 and execution_price != raw_price:
                    reason_codes.append(REASON_TICK_ROUND)
                if decision.limit_price is not None:
                    if side == SIDE_BUY and execution_price > decision.limit_price:
                        terminal_reason = REASON_LIMIT_NOT_REACHED
                        continue
                    if side == SIDE_SELL and execution_price < decision.limit_price:
                        terminal_reason = REASON_LIMIT_NOT_REACHED
                        continue
                notional = execution_price * fill_quantity * mult
                fee = self.policy.fees.explicit_fee(notional)
                fill = ledger.apply_fill(
                    order_id=order_id, fill_index=len(fills), asset_id=decision.asset_id,
                    venue=decision.venue, side=side, quantity=fill_quantity,
                    execution_ns=execution_ns, quote=quote, mid=mid,
                    execution_price=execution_price, spread_cost=half_spread * fill_quantity,
                    slippage_cost=slippage_price * fill_quantity, fee=fee,
                    limit_price=decision.limit_price, position_key=intent["position_key"],
                    multiplier=mult, margin_rate=margin_rate,
                    order_effect=intent["effect"])
                fills.append(fill)
                remaining -= fill_quantity
                if remaining <= 1e-9 * max(1.0, quantity):
                    remaining = 0.0
                first_execution_seen = execution_ns if first_execution_seen is None else first_execution_seen
                last_execution_seen = execution_ns
                if remaining == 0.0:
                    history.append({"status": ORDER_FILLED, "ns": execution_ns, "reason_code": REASON_FILLED})
                    status = ORDER_FILLED
                    break
                history.append({"status": ORDER_PARTIALLY_FILLED, "ns": execution_ns,
                                "reason_code": REASON_PARTIAL_FILL})
                status = ORDER_PARTIALLY_FILLED
                terminal_reason = REASON_PARTIAL_FILL
            if status != ORDER_FILLED:
                history.append({"status": ORDER_EXPIRED, "ns": expire_ns, "reason_code": terminal_reason})
                status = ORDER_EXPIRED

        filled_quantity = sum(fill["quantity"] for fill in fills)
        notional = sum(fill["notional"] for fill in fills)
        order = {
            "order_id": order_id, "decision_id": decision.decision_id, "asset_id": decision.asset_id,
            "venue": decision.venue, "side": side, "status": status,
            "position_key": intent["position_key"], "position_effect": intent["effect"],
            "contract_multiplier": mult, "is_perp": is_perp, "leverage": leverage,
            "required_initial_margin": filled_quantity * mult * reference.mid * margin_rate,
            "decision_ns": decision.decision_ns, "submit_ns": submit_ns, "ack_ns": ack_ns,
            "expire_ns": expire_ns, "first_execution_ns": first_execution_seen,
            "last_execution_ns": last_execution_seen,
            "requested_quantity": quantity, "original_requested_quantity": decision.quantity,
            "filled_quantity": filled_quantity, "remaining_quantity": remaining,
            "limit_price": decision.limit_price, "fill_count": len(fills),
            "average_fill_price": notional / (filled_quantity * mult) if filled_quantity else None,
            "traded_notional": notional,
            "explicit_fees": sum(fill["explicit_fee"] for fill in fills),
            "selection_reason_codes": list(reason_codes),
            "status_history": history, "fills": fills,
        }
        validate_status_history(order)
        return order

    def _fallback_mark(self, ledger, key):
        """Explicit, parameterized valuation fallback for an unmarkable position."""
        mode = self.policy.marks.fallback
        asset_id = ledger.asset_of_key.get(key, key)
        if mode == "zero":
            price = 0.0
        elif mode == "last_fill_price":
            price = next((fill["execution_price"] for fill in reversed(ledger.fills)
                          if fill.get("position_key", fill["asset_id"]) == key), 0.0)
        else:
            quantity = ledger.positions.get(key, 0.0)
            mult = ledger.multiplier(asset_id)
            price = ledger.cost_basis.get(key, 0.0) / (quantity * mult) if quantity else 0.0
        return {"price": price, "ns": None, "event_ns": None, "venue": "none",
                "source": f"mark_fallback::{mode}", "price_source": self.policy.marks.price_source,
                "mark_price": price, "last_price": price}

    def _position_detail(self, ledger, by_asset, cutoff_ns, marks):
        """Signed position, margin and analytic liquidation price for every open key."""
        detail = {}
        for key in sorted(ledger.positions):
            quantity = ledger.positions.get(key, 0.0)
            if quantity == 0.0:
                continue
            asset_id = ledger.asset_of_key.get(key, key)
            mult = ledger.multiplier(asset_id)
            spec = self.contract_for(asset_id, ledger.venue_of_key.get(key, ""))
            if asset_id in marks:
                mark = marks[asset_id]
            else:
                mark = self._fallback_mark(ledger, key)
            price = mark["price"]
            notional = abs(quantity) * mult * price
            unrealized = quantity * mult * price - ledger.cost_basis.get(key, 0.0)
            allocated = ledger.allocated_margin.get(key, 0.0)
            is_perp = spec is not None and spec.is_perp
            # A spot leg carries no maintenance requirement and is never liquidated by the
            # perpetual margin engine, even when perpetual contracts are declared elsewhere.
            mmr = spec.maintenance_margin_rate if is_perp else 0.0
            leverage = ledger.leverage_by_key.get(key, self.policy.perps.default_leverage)
            liquidation_price = None
            distance = None
            if is_perp and mult > 0 and price > 0 and mmr < 1.0:
                cost = ledger.cost_basis.get(key, 0.0)
                if quantity > 0:
                    denominator = quantity * mult * (1.0 - mmr)
                    liquidation_price = (cost - allocated) / denominator if denominator else None
                else:
                    denominator = abs(quantity) * mult * (1.0 + mmr)
                    liquidation_price = (cost + allocated) / denominator if denominator else None
                if liquidation_price is not None and liquidation_price > 0:
                    distance = (price - liquidation_price) / price if quantity > 0 \
                        else (liquidation_price - price) / price
            detail[key] = {
                "asset_id": asset_id, "venue": ledger.venue_of_key.get(key, ""),
                "is_perp": is_perp,
                "quantity": quantity, "contract_multiplier": mult,
                "mark_price": price, "last_price": mark.get("last_price", price),
                "notional": notional, "unrealized_pnl": unrealized,
                "cost_basis": ledger.cost_basis.get(key, 0.0),
                "allocated_margin": allocated, "leverage": leverage,
                "funding_cost_to_date": ledger.funding_by_key.get(key, 0.0),
                "maintenance_margin": notional * mmr,
                "maintenance_margin_rate": mmr,
                "liquidation_price": liquidation_price,
                "liquidation_distance_fraction": distance,
            }
        return detail

    def _portfolio_snapshot(self, ledger, by_asset, cutoff_ns):
        marks = self._marks_upto(cutoff_ns)
        detail = self._position_detail(ledger, by_asset, cutoff_ns, marks)
        gross = math.fsum(entry["notional"] for entry in detail.values())
        net = math.fsum(entry["quantity"] * entry["contract_multiplier"] * entry["mark_price"]
                        for entry in detail.values())
        market_value = net
        unrealized = math.fsum(entry["unrealized_pnl"] for entry in detail.values())
        equity = ledger.cash + market_value
        allocated = ledger.allocated_margin_total()
        maintenance = math.fsum(entry["maintenance_margin"] for entry in detail.values())
        margin_ratio = (maintenance / equity) if equity > 0 else None
        venue_notional = {}
        for entry in detail.values():
            venue = entry["venue"] or "undeclared"
            venue_notional[venue] = venue_notional.get(venue, 0.0) + entry["notional"]
        return {"positions": {key: entry["quantity"] for key, entry in detail.items()},
                "asset_net_quantities": {asset_id: ledger.asset_net_quantity(asset_id)
                                         for asset_id in sorted(ledger.asset_of_key.values())},
                "marks": marks, "position_detail": detail,
                "gross_exposure_notional": gross, "net_exposure_notional": net,
                "unrealized_pnl": unrealized, "equity": equity, "cash": ledger.cash,
                "net_pnl": equity - ledger.initial_cash,
                "margin_mode": self.policy.perps.margin_mode,
                "allocated_margin": allocated, "initial_margin_required": allocated,
                "maintenance_margin_required": maintenance,
                "margin_balance": equity, "available_margin": equity - allocated,
                "margin_ratio": margin_ratio,
                "liquidation_distance_fraction":
                    ((equity - maintenance) / equity) if equity > 0 else 0.0,
                "venue_position_notional": venue_notional,
                "funding_cost_to_date": ledger.funding_cost}

    def _risk_context(self, decision, reference, snapshot, ledger, intent=None):
        latest = reference.available_ns
        asset_id = decision.asset_id
        venue = decision.venue
        detail = snapshot["position_detail"]
        key = intent["position_key"] if intent is not None else None
        own = detail.get(key, {}) if key is not None else {}
        return {
            "decision_ns": decision.decision_ns,
            "local_ns": decision.decision_ns,
            "last_data_ns": latest,
            "feed_clock_ns": latest + self.clock_skew_ns,
            "asset_id": asset_id,
            "venue": venue,
            "side": intent["side"] if intent is not None else decision.order_side,
            "requested_quantity": intent["quantity"] if intent is not None else decision.quantity,
            "reference_price": reference.mid,
            "position_quantity": ledger.asset_net_quantity(asset_id),
            "gross_exposure_notional": snapshot["gross_exposure_notional"],
            "net_exposure_notional": snapshot["net_exposure_notional"],
            "realized_pnl": ledger.realized_pnl,
            "unrealized_pnl": snapshot["unrealized_pnl"],
            "fees_paid": ledger.fees_paid,
            "net_pnl": snapshot["net_pnl"],
            "venue_position_notional": snapshot["venue_position_notional"].get(venue, 0.0),
            "position_notional": own.get("notional", 0.0),
            "contract_multiplier": own.get("contract_multiplier"),
            "margin_mode": snapshot["margin_mode"],
            "margin_balance": snapshot["margin_balance"],
            "maintenance_margin_required": snapshot["maintenance_margin_required"],
            "initial_margin_required": snapshot["initial_margin_required"],
            "available_margin": snapshot["available_margin"],
            "margin_ratio": snapshot["margin_ratio"],
            "liquidation_distance_fraction": own.get("liquidation_distance_fraction")
            if own else snapshot["liquidation_distance_fraction"],
            "requested_leverage": intent["leverage"] if intent is not None else None,
            "contract_max_leverage": (intent["contract"].max_leverage
                                      if intent is not None and intent["contract"] else None),
            "funding_cost_to_date": ledger.funding_cost,
            "funding_cost_by_asset": ledger.funding_by_asset.get(asset_id, 0.0),
            "target_position_quantity": intent["target_position_quantity"]
            if intent is not None else None,
        }


    def _evaluate_risk(self, context):
        if self.risk_gate is None:
            return {"decision": "allow", "reason_codes": ["risk_gate_absent"],
                    "allowed_quantity": context["requested_quantity"]}
        result = self.risk_gate.evaluate(context)
        if hasattr(result, "decision"):
            return {"decision": result.decision, "reason_codes": list(result.reason_codes),
                    "allowed_quantity": float(result.allowed_quantity)}
        if isinstance(result, dict):
            return {"decision": result["decision"], "reason_codes": list(result.get("reason_codes", [])),
                    "allowed_quantity": float(result.get("allowed_quantity",
                                                         context["requested_quantity"]))}
        raise SimulationError("risk gate must return a decision object or mapping")


def replay(quotes, decisions, *, asof_ns, policy=None, seed=0, initial_cash=1_000_000.0,
           risk_gate=None, guard=None, reference_selector=None, clock_skew_ns=0, replay_id="replay",
           contracts=(), trace_perp_curve=None):
    simulator = ExecutionSimulator(policy=policy, seed=seed, initial_cash=initial_cash,
                                   risk_gate=risk_gate, guard=guard,
                                   reference_selector=reference_selector,
                                   clock_skew_ns=clock_skew_ns, contracts=contracts,
                                   trace_perp_curve=trace_perp_curve)
    return simulator.run(quotes, decisions, asof_ns=asof_ns, replay_id=replay_id)


def leaky_reference_selector(quotes, decision):
    """Deliberately non-causal selector used only to prove the guard fires.

    It returns the oldest quote whose availability is strictly after the decision,
    which is exactly the future-peeking behaviour the guard must reject.
    """
    future = [quote for quote in quotes if quote.available_ns > decision.decision_ns]
    if future:
        return min(future, key=lambda quote: (quote.available_ns, quote.event_ns))
    eligible = [quote for quote in quotes if quote.available_ns <= decision.decision_ns]
    return eligible[-1] if eligible else None


# --------------------------------------------------------------------------
# Synthetic stress suite (receipt generator). No real data is ever involved.
# --------------------------------------------------------------------------
ORIGIN_NS = 1_760_000_000_000_000_000


def build_synthetic_scenario(seed, *, assets=2, ticks=240, tick_ns=1_000_000,
                             feed_delay_ns=2_000_000, order_every=8, quantity=10.0):
    """Construct a deterministic synthetic quote/decision stream (no real data)."""
    quotes = []
    for asset_index in range(assets):
        asset_id = f"synthetic-asset-{asset_index}"
        price = 100.0 + 10.0 * asset_index
        for tick in range(ticks):
            event_ns = ORIGIN_NS + tick * tick_ns
            step = (uniform_draw(seed, "synthetic-walk", asset_id, tick) - 0.5) * 0.02
            price = max(1.0, price + step)
            half_spread = price * 0.0002
            volume = 500.0 + 500.0 * uniform_draw(seed, "synthetic-volume", asset_id, tick)
            quotes.append(Quote(asset_id=asset_id, venue="synthetic-venue", event_ns=event_ns,
                                available_ns=event_ns + feed_delay_ns, bid=price - half_spread,
                                ask=price + half_spread, volume=volume))
    decisions = []
    for slot, tick in enumerate(range(0, ticks, order_every)):
        for asset_index in range(assets):
            phase = (slot + asset_index) % 6
            action = (ACTION_ABSTAIN if phase == 4 else
                      ACTION_NO_TRADE if phase == 5 else
                      ACTION_BUY if phase in (0, 1) else ACTION_SELL)
            decisions.append(Decision(decision_id=f"synthetic-{slot}-{asset_index}",
                                      asset_id=f"synthetic-asset-{asset_index}",
                                      venue="synthetic-venue",
                                      decision_ns=ORIGIN_NS + tick * tick_ns,
                                      action=action, quantity=quantity))
    asof_ns = ORIGIN_NS + ticks * tick_ns + feed_delay_ns + 1
    return quotes, decisions, asof_ns


def run_stress_suite(seed=20_260_919):
    """Run the synthetic stress suite and return a machine-readable receipt."""
    import financial_risk_v1 as risk

    if not __debug__:
        raise SimulationError("verification requires Python assertions enabled")
    started = time.perf_counter()
    quotes, decisions, asof_ns = build_synthetic_scenario(seed)
    base_policy = ExecutionPolicy(fills=FillPolicy(reject_probability=0.0, fill_probability=1.0),
                                 capacity=CapacityPolicy(max_participation_fraction=0.05))

    baseline = replay(quotes, decisions, asof_ns=asof_ns, policy=base_policy, seed=seed,
                      risk_gate=risk.RiskEngine(), replay_id="baseline")
    repeat = replay(quotes, decisions, asof_ns=asof_ns, policy=base_policy, seed=seed,
                    risk_gate=risk.RiskEngine(), replay_id="baseline")
    if baseline["ledger_sha256"] != repeat["ledger_sha256"]:
        raise SimulationError("replay is not deterministic")

    rejection_policy = ExecutionPolicy(fills=FillPolicy(reject_probability=1.0),
                                       capacity=CapacityPolicy(max_participation_fraction=0.05))
    rejection = replay(quotes, decisions, asof_ns=asof_ns, policy=rejection_policy, seed=seed,
                       replay_id="rejection")

    partial_policy = ExecutionPolicy(fills=FillPolicy(reject_probability=0.0, fill_probability=1.0,
                                                      order_time_to_live_ns=20_000_000),
                                     capacity=CapacityPolicy(max_participation_fraction=0.0005))
    partial = replay(quotes, decisions, asof_ns=asof_ns, policy=partial_policy, seed=seed,
                     replay_id="partial")

    nofill_policy = ExecutionPolicy(fills=FillPolicy(reject_probability=0.0, fill_probability=0.0))
    nofill = replay(quotes, decisions, asof_ns=asof_ns, policy=nofill_policy, seed=seed,
                    replay_id="no_fill")

    fee_levels = (0.0, 0.5, 2.0, 10.0, 50.0)
    fee_series = []
    for fee_bps in fee_levels:
        policy = ExecutionPolicy(fees=FeePolicy(fee_bps=fee_bps),
                                 fills=FillPolicy(reject_probability=0.0, fill_probability=1.0),
                                 capacity=CapacityPolicy(max_participation_fraction=0.05))
        result = replay(quotes, decisions, asof_ns=asof_ns, policy=policy, seed=seed,
                        replay_id=f"fee-{fee_bps}")
        fee_series.append({"fee_bps": fee_bps, "net_pnl": result["ledger"]["net_pnl"],
                           "fees_paid": result["ledger"]["fees_paid"],
                           "fills": result["ledger"]["fill_count"]})
    violations = [fee_series[index]["fee_bps"] for index in range(1, len(fee_series))
                  if fee_series[index]["net_pnl"] > fee_series[index - 1]["net_pnl"] + 1e-9]
    if violations:
        raise SimulationError(f"net PnL increased when fees rose at {violations}")

    small_quotes = [Quote(asset_id="synthetic-asset-0", venue="synthetic-venue",
                          event_ns=ORIGIN_NS + index * 1_000_000,
                          available_ns=ORIGIN_NS + index * 1_000_000 + 1_000_000,
                          bid=99.99, ask=100.01, volume=1000.0) for index in range(12)]
    leaked_decision = Decision(decision_id="leaked", asset_id="synthetic-asset-0",
                               venue="synthetic-venue", decision_ns=ORIGIN_NS + 5_000_000,
                               action=ACTION_BUY, quantity=1.0)
    decision_leaks = 0
    for attempt in range(200):
        try:
            replay(small_quotes, [leaked_decision], asof_ns=ORIGIN_NS + 100_000_000,
                   reference_selector=leaky_reference_selector, seed=seed, replay_id=f"leak-{attempt}")
        except FutureInformationError:
            decision_leaks += 1
    if decision_leaks != 200:
        raise SimulationError("decision-side leak guard did not fire on every attempt")

    strict_guard = NoFutureInformationGuard(LeakGuardPolicy(forbid_post_decision_prices_in_executions=True))
    normal_decision = Decision(decision_id="strict", asset_id="synthetic-asset-0",
                               venue="synthetic-venue", decision_ns=ORIGIN_NS + 5_000_000,
                               action=ACTION_BUY, quantity=1.0)
    execution_leaks = 0
    try:
        replay(small_quotes, [normal_decision], asof_ns=ORIGIN_NS + 100_000_000, guard=strict_guard,
               seed=seed, replay_id="strict-leak")
    except FutureInformationError:
        execution_leaks = 1
    if execution_leaks != 1:
        raise SimulationError("strict post-decision execution guard did not fire")

    guard = NoFutureInformationGuard()
    future_quote = Quote(asset_id="synthetic-asset-0", venue="synthetic-venue",
                         event_ns=ORIGIN_NS + 20_000_000, available_ns=ORIGIN_NS + 21_000_000,
                         bid=99.0, ask=101.0, volume=10.0)
    direct_execution_leaks = 0
    try:
        guard.check_execution(ORIGIN_NS, ORIGIN_NS + 1_000_000, ORIGIN_NS + 5_000_000, future_quote)
    except FutureInformationError:
        direct_execution_leaks = 1
    if direct_execution_leaks != 1:
        raise SimulationError("execution-side guard did not reject a post-execution price")

    risk_receipts = _run_risk_guards(risk)
    perp_receipts = _run_perp_stress(seed)
    backtest_receipts = _run_backtest_stress(seed)

    receipt = {
        "schema_version": STRESS_SCHEMA,
        "status": "passed",
        "scope": SCOPE,
        "perp_scope": PERP_SCOPE,
        "seed": seed,
        "policy": base_policy.to_dict(),
        "guard_policy": asdict(LeakGuardPolicy()),
        "risk_limits": risk.RiskLimits().to_dict(),
        "provisional_pending_r1": list(PROVISIONAL_POLICY_PARAMETERS) + list(risk.PROVISIONAL_RISK_PARAMETERS),
        "determinism": {"runs": 2, "ledger_sha256": [baseline["ledger_sha256"], repeat["ledger_sha256"]],
                        "byte_identical": baseline["ledger_sha256"] == repeat["ledger_sha256"]},
        "conservation": baseline["ledger"]["conservation"],
        "counts": {"baseline": baseline["counts"], "rejection": rejection["counts"],
                   "partial": partial["counts"], "no_fill": nofill["counts"]},
        "order_statuses_observed": sorted({
            entry["status"] for scenario in (baseline, rejection, partial, nofill)
            for order in scenario["orders"] for entry in order["status_history"]}),
        "ledger_sha256": baseline["ledger_sha256"],
        "replay_sha256": baseline["replay_sha256"],
        "cost_monotonicity": {"fee_bps": [entry["fee_bps"] for entry in fee_series],
                              "net_pnl": [entry["net_pnl"] for entry in fee_series],
                              "fees_paid": [entry["fees_paid"] for entry in fee_series],
                              "violations": violations},
        "guard": {"decision_leaks_blocked": decision_leaks, "decision_leaks_attempted": 200,
                  "strict_execution_leaks_blocked": execution_leaks,
                  "post_execution_price_leaks_blocked": direct_execution_leaks,
                  "replay_guard": baseline["guard"]},
        "risk_guards": risk_receipts,
        "perpetual_contracts": perp_receipts,
        "backtest": backtest_receipts,
        "code": [file_identity(Path(__file__)),
                 file_identity(Path(__file__).with_name("financial_risk_v1.py"))],
        "deterministic_receipt": True,
        "limitations": [
            "Synthetic quotes and decisions only; no real instrument, venue, feed or broker.",
            "Fill, fee, spread, slippage, capacity and latency policies are explicit guesses provisional pending R1.",
            "Margin, funding, leverage and liquidation parameters are synthetic construction choices "
            "provisional pending R1; no venue rule is replicated.",
            "Paper trading means offline synthetic replay only; no order is submitted anywhere.",
            "Not evidence of profitability, calibration, fill authenticity, survivorship safety or live readiness.",
        ],
    }
    # Wall-clock timing is deliberately NOT part of the receipt: the receipt must be
    # byte-identical across runs. Timing is reported on the CLI stream instead.
    receipt["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
    return receipt


def _perp_stress_quotes(prices, *, asset="synthetic-perp", venue="binance", volume=1.0e6,
                        funding_rate=None, mark_price=None):
    return [Quote(asset_id=asset, venue=venue, event_ns=ORIGIN_NS + index * 1_000_000,
                  available_ns=ORIGIN_NS + index * 1_000_000,
                  bid=price * 0.9999, ask=price * 1.0001, volume=volume,
                  mark_price=mark_price, funding_rate=funding_rate)
            for index, price in enumerate(prices)]


def _perp_stress_decision(index, action, quantity, *, asset="synthetic-perp", venue="binance",
                          **kwargs):
    return Decision(decision_id=f"perp-stress-{index}-{action}", asset_id=asset, venue=venue,
                    decision_ns=ORIGIN_NS + index * 1_000_000, action=action, quantity=quantity,
                    **kwargs)


def _perp_stress_policy(*, margin_mode=MARGIN_CROSS, leverage=10.0, max_leverage=20.0,
                        funding_interval_ns=10**15, funding_rate=0.0,
                        funding_rate_source=FUNDING_SOURCE_POLICY_CONSTANT,
                        position_mode=POSITION_ONE_WAY):
    return ExecutionPolicy(
        fees=FeePolicy(fee_bps=0.0),
        spread=SpreadPolicy(half_spread_bps=0.0),
        slippage=SlippagePolicy(fixed_bps=0.0),
        capacity=CapacityPolicy(max_participation_fraction=1.0),
        fills=FillPolicy(reject_probability=0.0, fill_probability=1.0),
        latency=LatencyPolicy(decision_to_order_ns=0, order_to_ack_ns=0, ack_to_execution_ns=0),
        marks=MarkPolicy(fallback="cost_basis", price_source=PRICE_SOURCE_MARK),
        perps=PerpPolicy(allow_short=True, default_leverage=leverage, max_leverage=max_leverage,
                         margin_mode=margin_mode, position_mode=position_mode,
                         maintenance_margin_rate=0.005,
                         funding_interval_ns=funding_interval_ns, funding_anchor_ns=0,
                         funding_rate_source=funding_rate_source,
                         default_funding_rate=funding_rate))


def _perp_stress_contract(*, margin_mode=MARGIN_CROSS, max_leverage=20.0, multiplier=1.0):
    return ContractSpec(asset_id="synthetic-perp", venue="binance", margin_mode=margin_mode,
                        max_leverage=max_leverage, contract_multiplier=multiplier,
                        maintenance_margin_rate=0.005)


def _run_perp_stress(seed):
    """Exercise the perpetual-contract layer and record the observed behaviour."""
    del seed
    asset = "synthetic-perp"
    venue = "binance"
    contract = _perp_stress_contract()

    # 1. Short open -> favourable move -> flat, with conservation.
    prices = [100.0] * 3 + [90.0] * 3
    short_round_trip = replay(
        _perp_stress_quotes(prices),
        [_perp_stress_decision(0, ACTION_SHORT, 10.0), _perp_stress_decision(3, ACTION_FLAT, 0.0)],
        asof_ns=ORIGIN_NS + 10_000_000, policy=_perp_stress_policy(), seed=1,
        contracts=[contract], initial_cash=100_000.0, replay_id="perp-short")
    short_ledger = short_round_trip["ledger"]

    # 2. Funding-cost monotonicity in the declared funding rate.
    funding_series = []
    for rate in (0.0, 0.0001, 0.0005, 0.001):
        result = replay(
            _perp_stress_quotes([100.0] * 12),
            [_perp_stress_decision(0, ACTION_LONG, 10.0)],
            asof_ns=ORIGIN_NS + 12_000_000,
            policy=_perp_stress_policy(funding_interval_ns=2_000_000, funding_rate=rate),
            seed=1, contracts=[contract], initial_cash=100_000.0, replay_id=f"perp-funding-{rate}")
        funding_series.append({"funding_rate": rate, "funding_cost": result["ledger"]["funding_cost"],
                               "net_pnl": result["ledger"]["net_pnl"],
                               "payments": result["ledger"]["funding_payment_count"],
                               "conservation_ok": result["ledger"]["conservation"]["ok"]})
    funding_violations = [entry["funding_rate"] for index, entry in enumerate(funding_series)
                          if index and (entry["funding_cost"] <= funding_series[index - 1]["funding_cost"]
                                        or entry["net_pnl"] >= funding_series[index - 1]["net_pnl"])]

    # 3. Liquidation of a levered position on an adverse synthetic path.
    declining = [100.0 * (0.98 ** index) for index in range(40)]
    liquidation_run = replay(
        _perp_stress_quotes(declining),
        [_perp_stress_decision(0, ACTION_LONG, 50.0)],
        asof_ns=ORIGIN_NS + 41_000_000,
        policy=_perp_stress_policy(margin_mode=MARGIN_ISOLATED, leverage=10.0, max_leverage=10.0),
        seed=1, contracts=[_perp_stress_contract(margin_mode=MARGIN_ISOLATED, max_leverage=10.0)],
        initial_cash=10_000.0, replay_id="perp-liquidation")
    liquidation_ledger = liquidation_run["ledger"]
    liquidation_record = liquidation_ledger["liquidations"][0]

    # 4. The same path under cross margin: the whole wallet backs the position.
    cross_run = replay(
        _perp_stress_quotes(declining),
        [_perp_stress_decision(0, ACTION_LONG, 50.0)],
        asof_ns=ORIGIN_NS + 41_000_000,
        policy=_perp_stress_policy(margin_mode=MARGIN_CROSS, leverage=10.0, max_leverage=10.0),
        seed=1, contracts=[_perp_stress_contract(margin_mode=MARGIN_CROSS, max_leverage=10.0)],
        initial_cash=10_000.0, replay_id="perp-cross")
    cross_ledger = cross_run["ledger"]

    # 5. Leverage changes the margin requirement, not the marked PnL of a fixed size.
    leverage_pnls = []
    leverage_margins = []
    for leverage in (1.0, 2.0, 5.0, 10.0):
        result = replay(
            _perp_stress_quotes([100.0, 100.0, 105.0, 105.0]),
            [_perp_stress_decision(0, ACTION_LONG, 10.0), _perp_stress_decision(2, ACTION_FLAT, 0.0)],
            asof_ns=ORIGIN_NS + 6_000_000,
            policy=_perp_stress_policy(leverage=leverage, max_leverage=20.0), seed=1,
            contracts=[contract], initial_cash=100_000.0, replay_id=f"perp-leverage-{leverage}")
        leverage_pnls.append(result["ledger"]["net_pnl"])
        leverage_margins.append(result["orders"][0]["required_initial_margin"])
    leverage_spread = max(leverage_pnls) - min(leverage_pnls)

    # 6. Contract multiplier scaling (4x is exact in binary floating point).
    scaling_pnls = []
    for multiplier in (1.0, 4.0):
        result = replay(
            _perp_stress_quotes([100.0, 100.0, 110.0, 110.0]),
            [_perp_stress_decision(0, ACTION_LONG, 10.0), _perp_stress_decision(2, ACTION_FLAT, 0.0)],
            asof_ns=ORIGIN_NS + 6_000_000, policy=_perp_stress_policy(), seed=1,
            contracts=[_perp_stress_contract(multiplier=multiplier)], initial_cash=1_000_000.0,
            replay_id=f"perp-multiplier-{multiplier}")
        scaling_pnls.append(result["ledger"]["net_pnl"])
    multiplier_ratio_error = abs(scaling_pnls[1] / scaling_pnls[0] - 4.0) \
        if scaling_pnls[0] else None

    # 7. Margin sufficiency is enforced rather than assumed.
    sufficiency_run = replay(
        _perp_stress_quotes([100.0] * 8),
        [_perp_stress_decision(0, ACTION_LONG, 500.0)],
        asof_ns=ORIGIN_NS + 9_000_000,
        policy=_perp_stress_policy(leverage=10.0, max_leverage=10.0), seed=1,
        contracts=[_perp_stress_contract(max_leverage=10.0)], initial_cash=1_000.0,
        replay_id="perp-margin-sufficiency")
    sufficiency_record = sufficiency_run["decisions"][0]
    sufficiency_ledger = sufficiency_run["ledger"]
    margin_enforced = bool(
        sufficiency_ledger["allocated_margin_total"] <= 1_000.0 + 1e-6
        and REASON_MARGIN_REDUCE in sufficiency_record["reason_codes"]
        and sufficiency_ledger["cash"] <= sufficiency_ledger["initial_cash"] + 1e-9
        and all(fill["quantity"] * fill["execution_price"]
                <= 100.0 * 100.0 + 1e-6 for fill in sufficiency_ledger["fills"]))

    # 8. One-way netting versus hedge mode.
    netting_run = replay(
        _perp_stress_quotes([100.0] * 8),
        [_perp_stress_decision(0, ACTION_LONG, 5.0), _perp_stress_decision(2, ACTION_SELL, 2.0)],
        asof_ns=ORIGIN_NS + 9_000_000, policy=_perp_stress_policy(), seed=1,
        contracts=[contract], initial_cash=100_000.0, replay_id="perp-one-way")
    hedge_run = replay(
        _perp_stress_quotes([100.0] * 8),
        [_perp_stress_decision(0, ACTION_LONG, 5.0), _perp_stress_decision(2, ACTION_SELL, 2.0)],
        asof_ns=ORIGIN_NS + 9_000_000,
        policy=_perp_stress_policy(position_mode=POSITION_HEDGE), seed=1, contracts=[contract],
        initial_cash=100_000.0, replay_id="perp-hedge")

    return {
        "schema_version": "nanojev-financial-perp-stress-v1",
        "scope": PERP_SCOPE,
        "short_round_trip": {"net_pnl": short_ledger["net_pnl"],
                             "realized_pnl": short_ledger["realized_pnl"],
                             "final_quantity": short_ledger["positions"][asset]["quantity"],
                             "conservation_ok": short_ledger["conservation"]["ok"],
                             "fills": short_ledger["fill_count"]},
        "funding_monotonicity": {
            "funding_rate": [entry["funding_rate"] for entry in funding_series],
            "funding_cost": [entry["funding_cost"] for entry in funding_series],
            "net_pnl": [entry["net_pnl"] for entry in funding_series],
            "payments": [entry["payments"] for entry in funding_series],
            "violations": funding_violations,
            "conservation_ok": all(entry["conservation_ok"] for entry in funding_series)},
        "liquidation": {"liquidation_count": liquidation_ledger["liquidation_count"],
                        "position_flat": liquidation_ledger["positions"][asset]["quantity"] == 0.0,
                        "position_after": liquidation_record["position_after"],
                        "trigger_reason_code": liquidation_record["trigger_reason_code"],
                        "liquidation_price": liquidation_record["liquidation_price"],
                        "margin_ratio": liquidation_record["margin_ratio"],
                        "realized_pnl": liquidation_record["realized_pnl"],
                        "conservation_ok": liquidation_ledger["conservation"]["ok"]},
        "margin_mode_comparison": {
            "isolated_liquidations": liquidation_ledger["liquidation_count"],
            "cross_liquidations": cross_ledger["liquidation_count"],
            "isolated_final_equity": liquidation_ledger["equity"],
            "cross_final_equity": cross_ledger["equity"],
            "cross_conservation_ok": cross_ledger["conservation"]["ok"]},
        "leverage_effect": {"leverage": [1.0, 2.0, 5.0, 10.0], "net_pnl": leverage_pnls,
                            "required_initial_margin": leverage_margins,
                            "net_pnl_spread": leverage_spread},
        "multiplier_scaling": {"pnl": scaling_pnls, "pnl_ratio_error": multiplier_ratio_error},
        "margin_sufficiency": {"requested_quantity": sufficiency_record["requested_quantity"],
                                "effective_quantity": sufficiency_record["effective_quantity"],
                                "allocated_margin": sufficiency_ledger["allocated_margin_total"],
                                "reason_codes": sufficiency_record["reason_codes"],
                                "enforced": margin_enforced},
        "netting": {"one_way_keys": len(netting_run["ledger"]["positions"]),
                    "hedge_keys": len(hedge_run["ledger"]["positions"]),
                    "one_way_quantity": netting_run["ledger"]["positions"][asset]["quantity"],
                    "hedge_long": hedge_run["ledger"]["positions"][f"{asset}::long"]["quantity"],
                    "hedge_short": hedge_run["ledger"]["positions"][f"{asset}::short"]["quantity"],
                    "one_way_conservation_ok": netting_run["ledger"]["conservation"]["ok"],
                    "hedge_conservation_ok": hedge_run["ledger"]["conservation"]["ok"]},
        "venue": venue,
        "limitations": [
            "Synthetic perpetual-contract scenarios only; no venue rule is replicated.",
            "Linear quote-settled contracts only; inverse contracts, portfolio margin, insurance "
            "funds, ADL and bad-debt socialisation are not modelled.",
        ],
    }


def _run_backtest_stress(seed):
    """Run the paper-trading backtest harness twice and record its determinism summary."""
    import financial_backtest_v1 as backtest

    path = backtest.build_synthetic_perp_path(seed, ticks=240, tick_ns=3 * 60 * 1_000_000_000,
                                              regime_length_ticks=80)
    strategy = backtest.MovingAverageCrossoverStrategy(fast_ticks=4, slow_ticks=16,
                                                       target_quantity=2.0, rebalance_ticks=2)
    receipt = backtest.run_backtest(path, strategy=strategy, initial_cash=50_000.0,
                                    replay_id="stress-paper-backtest")
    tolerance = 1e-6 * max(1.0, receipt["initial_cash"])
    return {
        "schema_version": receipt["schema_version"],
        "scope": receipt["scope"],
        "backtest_sha256": receipt["backtest_sha256"],
        "byte_identical": receipt["determinism"]["byte_identical"],
        "equity_curve_identical": receipt["determinism"]["equity_curve_identical"],
        "funding_identical": receipt["determinism"]["funding_identical"],
        "fills": receipt["counts"]["fills"],
        "funding_payments": receipt["counts"]["funding_payments"],
        "liquidations": receipt["counts"]["liquidations"],
        "net_pnl": receipt["ledger"]["net_pnl"],
        "max_drawdown_fraction": receipt["drawdown"]["max_drawdown_fraction"],
        "turnover_notional": receipt["turnover"]["total_notional"],
        "instrument_residual": receipt["attribution"]["instrument_residual"],
        "regime_residual": receipt["attribution"]["regime_residual"],
        "attribution_closes": (abs(receipt["attribution"]["instrument_residual"]) <= tolerance
                               and abs(receipt["attribution"]["regime_residual"]) <= tolerance),
        "regime_labels": sorted(receipt["per_regime"]),
        "instrument_count": len(receipt["per_instrument"]),
        "equity_curve_points": len(receipt["equity_curve"]),
        "conservation_ok": receipt["ledger"]["conservation"]["ok"],
        "code": receipt["code"],
        "limitations": receipt["limitations"],
    }


def _run_risk_guards(risk):
    """Exercise every out-of-model guard and record the observed reason codes."""
    quotes, decisions, asof_ns = build_synthetic_scenario(20_260_919, assets=1, ticks=64)
    policy = ExecutionPolicy(fills=FillPolicy(reject_probability=0.0, fill_probability=1.0),
                             capacity=CapacityPolicy(max_participation_fraction=0.05))
    base = {"decision_ns": ORIGIN_NS, "local_ns": ORIGIN_NS, "last_data_ns": ORIGIN_NS - 1_000_000,
            "feed_clock_ns": ORIGIN_NS, "asset_id": "synthetic-asset-0", "venue": "synthetic-venue",
            "side": "buy", "requested_quantity": 5.0, "reference_price": 100.0,
            "position_quantity": 0.0, "gross_exposure_notional": 0.0, "net_exposure_notional": 0.0,
            "realized_pnl": 0.0, "unrealized_pnl": 0.0, "fees_paid": 0.0, "net_pnl": 0.0}
    cases = {}

    engine = risk.RiskEngine(limits=risk.RiskLimits(max_position_quantity=4.0))
    cases["position_reduce"] = engine.evaluate(dict(base, requested_quantity=5.0)).to_dict()
    engine = risk.RiskEngine(limits=risk.RiskLimits(max_position_quantity=0.0))
    cases["position_block"] = engine.evaluate(dict(base, requested_quantity=5.0)).to_dict()
    engine = risk.RiskEngine(limits=risk.RiskLimits(max_gross_exposure_notional=100.0))
    cases["gross_exposure_block"] = engine.evaluate(
        dict(base, gross_exposure_notional=100.0, net_exposure_notional=100.0)).to_dict()
    engine = risk.RiskEngine(limits=risk.RiskLimits(max_net_loss_notional=50.0))
    cases["net_loss_block"] = engine.evaluate(dict(base, net_pnl=-60.0)).to_dict()
    engine = risk.RiskEngine(limits=risk.RiskLimits(max_turnover_notional=250.0))
    cases["turnover_reduce"] = engine.evaluate(base).to_dict()
    engine = risk.RiskEngine(limits=risk.RiskLimits(max_orders_per_window=0))
    cases["order_rate_block"] = engine.evaluate(base).to_dict()
    engine = risk.RiskEngine(limits=risk.RiskLimits(max_data_staleness_ns=1_000))
    cases["stale_data_block"] = engine.evaluate(
        dict(base, local_ns=ORIGIN_NS + 10_000_000)).to_dict()
    engine = risk.RiskEngine(limits=risk.RiskLimits(max_clock_drift_ns=1_000))
    cases["clock_drift_block"] = engine.evaluate(dict(base, feed_clock_ns=ORIGIN_NS + 10_000_000)).to_dict()
    engine = risk.RiskEngine()
    engine.kill_switch.engage("synthetic_operator_halt", ORIGIN_NS)
    cases["kill_switch_block"] = engine.evaluate(base).to_dict()

    expected = {"position_reduce": "reduce", "position_block": "block",
                "gross_exposure_block": "block", "net_loss_block": "block",
                "turnover_reduce": "reduce", "order_rate_block": "block",
                "stale_data_block": "block", "clock_drift_block": "block",
                "kill_switch_block": "block"}
    receipts = {}
    for name, decision in cases.items():
        if decision["decision"] != expected[name]:
            raise SimulationError(f"risk guard {name} returned {decision['decision']}, "
                                  f"expected {expected[name]}")
        receipts[name] = {"decision": decision["decision"], "reason_codes": decision["reason_codes"],
                          "allowed_quantity": decision["allowed_quantity"]}

    risk_replay = replay(quotes, decisions, asof_ns=asof_ns, policy=policy, seed=7,
                         risk_gate=risk.RiskEngine(), replay_id="risk")
    blocked_engine = risk.RiskEngine(limits=risk.RiskLimits(max_order_notional=0.0))
    blocked_replay = replay(quotes, decisions, asof_ns=asof_ns, policy=policy, seed=7,
                            risk_gate=blocked_engine, replay_id="risk-blocked")
    receipts["replay_with_gate"] = {"risk_blocked_decisions": risk_replay["counts"]["risk_blocked_decisions"],
                                    "fills": risk_replay["counts"]["fills"]}
    receipts["replay_all_blocked"] = {"risk_blocked_decisions":
                                      blocked_replay["counts"]["risk_blocked_decisions"],
                                      "fills": blocked_replay["counts"]["fills"]}
    if blocked_replay["counts"]["fills"] != 0:
        raise SimulationError("blocked risk gate still produced fills")
    return receipts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stress", action="store_true",
                        help="run the synthetic stress suite and write a receipt")
    parser.add_argument("--output", type=Path, help="receipt path for --stress")
    parser.add_argument("--seed", type=int, default=20_260_919)
    args = parser.parse_args()
    if not args.stress:
        parser.print_help()
        raise SystemExit(0)
    if args.output is None:
        raise SystemExit("--stress requires --output")
    report = run_stress_suite(seed=args.seed)
    elapsed_ms = report.pop("elapsed_ms")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(canonical_json({"output": str(args.output), "status": report["status"],
                          "ledger_sha256": report["ledger_sha256"],
                          "fills": report["counts"]["baseline"]["fills"],
                          "elapsed_ms": elapsed_ms}))
