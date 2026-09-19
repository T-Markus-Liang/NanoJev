#!/usr/bin/env python3
"""Paper-trading backtest harness for the synthetic perpetual-contract simulator.

Scope and honesty statement
--------------------------
This harness replays a **constructed** market path against a **constructed** decision
stream (or a deterministic reference strategy) and reports a complete run: fills,
funding payments, fees, margin usage, liquidations, an equity curve, drawdown,
turnover and per-instrument / per-regime breakdowns, all inside a machine-readable
receipt.

It is a *paper-trading backtest*, which here means exactly:

* no order ever leaves this process; no venue, broker, exchange API or credential is
  contacted, and no network request is made;
* the market path is generated from a seeded SHA-256 draw function, not downloaded;
* every quantity, price, fee, funding rate and regime label is synthetic and is
  labelled as such in the receipt;
* a passing run proves internal consistency and determinism only. It is **not**
  evidence of profitability, fill authenticity, calibration, capacity realism or
  live readiness. No result may be quoted as a real return.

Determinism and replayability
-----------------------------
``run_backtest`` executes the replay twice and records both receipt hashes, so a
receipt is byte-identical for identical inputs, policy, strategy and seed. The
receipt identifies the exact policy, contracts, strategy parameters and code files
that produced it.

Usage::

    .venv/bin/python scripts/financial_backtest_v1.py --output /tmp/bt.json --seed 20260919
"""

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

from benchmark_nanojev_v2 import canonical_json, file_identity, sha256_bytes
from financial_simulator_v1 import (
    ACTION_FLAT, ACTION_LONG, ACTION_SHORT, INSTRUMENT_PERP, MARGIN_CROSS, MARGIN_ISOLATED,
    POSITION_HEDGE, POSITION_ONE_WAY, POLICY_VERSION, PRICE_SOURCE_MARK,
    AccountPolicy, CapacityPolicy, ContractSpec, Decision, ExecutionPolicy, FeePolicy, FillPolicy,
    LatencyPolicy, MarkPolicy, PerpPolicy, SlippagePolicy, SpreadPolicy,
    FUNDING_SOURCE_DECLARED_SCHEDULE,
    PROVISIONAL_POLICY_PARAMETERS, replay, uniform_draw,
)


SCHEMA = "nanojev-financial-backtest-v1"
BACKTEST_POLICY_VERSION = POLICY_VERSION
SCOPE = (
    "Synthetic paper-trading backtest of crypto secondary-market PERPETUAL CONTRACT decisions. "
    "Synthetic constructed market path and decisions only; no venue, exchange API or broker is "
    "contacted, no market data is downloaded, and no profitability, calibration, capacity or "
    "live-readiness claim is made."
)
REGIME_UNKNOWN = "unlabeled"

ORIGIN_NS = 1_760_000_000_000_000_000
MS = 1_000_000
SECOND = 1_000_000_000
MINUTE = 60 * SECOND
HOUR = 60 * MINUTE

# Defaults for the synthetic path: 480 three-minute ticks span 24 synthetic hours, so a
# declared 8-hour funding interval produces three funding boundaries. These are synthetic
# construction choices, not measurements of any venue's funding schedule.
DEFAULT_TICKS = 480
DEFAULT_TICK_NS = 3 * MINUTE
DEFAULT_FUNDING_INTERVAL_NS = 8 * HOUR

# Declared synthetic venue labels for the R1 scope (Binance / Bybit / Aster / Hyperliquid).
# Naming a venue does not replicate or claim any venue-specific rule.
DEFAULT_INSTRUMENTS = (("synthetic-perp-0", "binance"), ("synthetic-perp-1", "bybit"))
SYNTHETIC_REGIME_LABELS = ("synthetic-regime-up", "synthetic-regime-down", "synthetic-regime-chop")

PROVISIONAL_BACKTEST_PARAMETERS = (
    "path.ticks",
    "path.tick_ns",
    "path.feed_delay_ns",
    "path.regime_length_ticks",
    "path.initial_price",
    "path.drift_bps_per_tick",
    "path.noise_bps_per_tick",
    "path.volume",
    "path.mark_basis_fraction",
    "path.tick_size",
    "path.lot_size",
    "path.min_notional",
    "strategy.fast_ticks",
    "strategy.slow_ticks",
    "strategy.target_quantity",
    "strategy.rebalance_ticks",
    "strategy.threshold_fraction",
    "strategy.allow_short",
    "backtest.initial_cash",
)


class BacktestError(RuntimeError):
    """Base class for backtest contract violations."""


def _finite(value, field):
    if type(value) not in (int, float) or type(value) is bool or not math.isfinite(value):
        raise ValueError(f"{field} requires a finite number, not {value!r}")
    return float(value)


def _positive_int(value, field):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} requires a positive integer")
    return value


# --------------------------------------------------------------------------
# Synthetic market path (no real data, ever)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RegimeWindow:
    """Half-open ``[start_ns, end_ns)`` window carrying a synthetic regime label."""

    label: str
    start_ns: int
    end_ns: int

    def __post_init__(self):
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("regime label requires a nonempty string")
        if type(self.start_ns) is not int or type(self.end_ns) is not int:
            raise ValueError("regime windows require integer nanosecond bounds")
        if self.end_ns <= self.start_ns:
            raise ValueError("regime windows require end_ns > start_ns")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class SyntheticMarketPath:
    """A constructed quote path plus its synthetic regime schedule."""

    quotes: tuple
    regimes: tuple
    asof_ns: int
    instruments: tuple
    ticks: int
    tick_ns: int
    seed: int
    description: str = "synthetic generated perpetual-contract quote path"

    def __post_init__(self):
        if not self.quotes:
            raise ValueError("a market path requires at least one quote")
        for instrument in self.instruments:
            if len(instrument) != 2:
                raise ValueError("instruments require (asset_id, venue) pairs")
        _positive_int(self.ticks, "ticks")
        _positive_int(self.tick_ns, "tick_ns")
        if type(self.asof_ns) is not int or self.asof_ns < 0:
            raise ValueError("asof_ns requires a nonnegative UTC Unix nanosecond timestamp")

    def to_summary(self):
        return {"description": self.description, "synthetic": True, "seed": self.seed,
                "ticks": self.ticks, "tick_ns": self.tick_ns, "asof_ns": self.asof_ns,
                "instrument_count": len(self.instruments),
                "instruments": [{"asset_id": asset_id, "venue": venue}
                                for asset_id, venue in self.instruments],
                "quote_count": len(self.quotes),
                "regimes": [window.to_dict() for window in self.regimes]}


def build_regime_schedule(*, start_ns, ticks, tick_ns, regime_length_ticks):
    """Contiguous synthetic regime windows that drive the constructed drift."""
    _positive_int(regime_length_ticks, "regime_length_ticks")
    windows = []
    for start_tick in range(0, ticks, regime_length_ticks):
        end_tick = min(ticks, start_tick + regime_length_ticks)
        label = SYNTHETIC_REGIME_LABELS[(start_tick // regime_length_ticks)
                                        % len(SYNTHETIC_REGIME_LABELS)]
        windows.append(RegimeWindow(label=label, start_ns=start_ns + start_tick * tick_ns,
                                    end_ns=start_ns + end_tick * tick_ns))
    return tuple(windows)


def derive_funding_interval_ns(path, *, maximum=DEFAULT_FUNDING_INTERVAL_NS):
    """Declared funding interval for a path: at most ``maximum`` and at least three
    boundaries across the path, so a short synthetic tape still exercises funding.
    This is a declared construction choice, not a venue's real funding schedule."""
    span = path.ticks * path.tick_ns
    derived = max(1, span // 3)
    return int(min(maximum, derived))


def regime_at(regimes, ns):
    """Synthetic regime label covering ``ns``, or ``unlabeled`` outside every window."""
    for window in regimes:
        if window.start_ns <= ns < window.end_ns:
            return window.label
    return REGIME_UNKNOWN


def drift_for_regime(label, drift_bps_per_tick):
    if label == SYNTHETIC_REGIME_LABELS[0]:
        return drift_bps_per_tick
    if label == SYNTHETIC_REGIME_LABELS[1]:
        return -drift_bps_per_tick
    return 0.0


def build_synthetic_perp_path(seed, *, instruments=DEFAULT_INSTRUMENTS, ticks=DEFAULT_TICKS,
                              tick_ns=DEFAULT_TICK_NS,
                              feed_delay_ns=0, regime_length_ticks=120, initial_price=30_000.0,
                              drift_bps_per_tick=0.6, noise_bps_per_tick=6.0, volume=250.0,
                              mark_basis_fraction=0.0002, funding_rate=0.00001,
                              funding_rate_schedule=(),
                              origin_ns=ORIGIN_NS):
    """Construct a deterministic synthetic perp path. Nothing is downloaded."""
    from financial_simulator_v1 import Quote

    if type(seed) is not int:
        raise ValueError("seed requires an integer")
    _positive_int(ticks, "ticks")
    _positive_int(tick_ns, "tick_ns")
    if type(feed_delay_ns) is not int or feed_delay_ns < 0:
        raise ValueError("feed_delay_ns requires nonnegative integer nanoseconds")
    _finite(initial_price, "initial_price")
    price_scale = _finite(initial_price, "initial_price")
    if price_scale <= 0:
        raise ValueError("initial_price requires a positive price")
    for name, value in (("drift_bps_per_tick", drift_bps_per_tick),
                        ("noise_bps_per_tick", noise_bps_per_tick)):
        _finite(value, name)
    if _finite(volume, "volume") < 0:
        raise ValueError("volume requires a nonnegative synthetic size")
    _finite(mark_basis_fraction, "mark_basis_fraction")
    _finite(funding_rate, "funding_rate")

    schedule = {(asset_id, venue, ns): rate for asset_id, venue, ns, rate
                in funding_rate_schedule}
    regimes = build_regime_schedule(start_ns=origin_ns, ticks=ticks, tick_ns=tick_ns,
                                    regime_length_ticks=regime_length_ticks)
    quotes = []
    for instrument_index, (asset_id, venue) in enumerate(instruments):
        price = price_scale * (1.0 + 0.05 * instrument_index)
        for tick in range(ticks):
            event_ns = origin_ns + tick * tick_ns
            label = regime_at(regimes, event_ns)
            drift = drift_for_regime(label, drift_bps_per_tick)
            noise = (uniform_draw(seed, "synthetic-perp-walk", asset_id, venue, tick) - 0.5) \
                * 2.0 * noise_bps_per_tick
            price = max(1.0, price * (1.0 + (drift + noise) / 10_000.0))
            half_spread = price * 0.0002
            mark_price = price * (1.0 + mark_basis_fraction)
            quote_funding = schedule.get((asset_id, venue, event_ns), funding_rate)
            quotes.append(Quote(
                asset_id=asset_id, venue=venue, event_ns=event_ns,
                available_ns=event_ns + feed_delay_ns,
                bid=price - half_spread, ask=price + half_spread,
                volume=volume, source_id="synthetic-only",
                mark_price=mark_price, last_price=price, funding_rate=quote_funding))
    asof_ns = origin_ns + ticks * tick_ns + feed_delay_ns + 1
    return SyntheticMarketPath(quotes=tuple(quotes), regimes=regimes, asof_ns=asof_ns,
                               instruments=tuple(instruments), ticks=ticks, tick_ns=tick_ns,
                               seed=seed)


def default_contracts(instruments=DEFAULT_INSTRUMENTS, *, margin_mode=MARGIN_CROSS,
                      max_leverage=20.0, maintenance_margin_rate=0.005, tick_size=0.0,
                      lot_size=0.0, min_notional=0.0):
    return tuple(ContractSpec(asset_id=asset_id, venue=venue, instrument_type=INSTRUMENT_PERP,
                              contract_multiplier=1.0, tick_size=tick_size, lot_size=lot_size,
                              min_notional=min_notional, margin_mode=margin_mode,
                              max_leverage=max_leverage,
                              maintenance_margin_rate=maintenance_margin_rate)
                 for asset_id, venue in instruments)


# --------------------------------------------------------------------------
# Deterministic reference strategies
# --------------------------------------------------------------------------
class MovingAverageCrossoverStrategy:
    """Deterministic, causal reference strategy over the synthetic path.

    Only quotes with ``available_ns <= decision_ns`` enter the averages, so the
    strategy cannot see the future. It emits target-position actions
    (``long``/``short``/``flat``), which is what makes it two-sided.
    """

    name = "synthetic-moving-average-crossover"
    declared_applicability = ("Synthetic path only; not a validated signal and not evidence that "
                              "any moving-average rule has predictive value.")

    def __init__(self, *, fast_ticks=8, slow_ticks=32, target_quantity=1.0, rebalance_ticks=4,
                 threshold_fraction=0.0, allow_short=True):
        _positive_int(fast_ticks, "fast_ticks")
        _positive_int(slow_ticks, "slow_ticks")
        _positive_int(rebalance_ticks, "rebalance_ticks")
        if fast_ticks >= slow_ticks:
            raise ValueError("fast_ticks must be strictly less than slow_ticks")
        if _finite(target_quantity, "target_quantity") <= 0:
            raise ValueError("target_quantity requires a positive quantity")
        _finite(threshold_fraction, "threshold_fraction")
        if not 0.0 <= float(threshold_fraction) < 1.0:
            raise ValueError("threshold_fraction requires a fraction in [0, 1)")
        if type(allow_short) is not bool:
            raise ValueError("allow_short requires a boolean")
        self.fast_ticks = int(fast_ticks)
        self.slow_ticks = int(slow_ticks)
        self.target_quantity = float(target_quantity)
        self.rebalance_ticks = int(rebalance_ticks)
        self.threshold_fraction = float(threshold_fraction)
        self.allow_short = allow_short

    def describe(self):
        return {"name": self.name, "declared_applicability": self.declared_applicability,
                "parameters": {"fast_ticks": self.fast_ticks, "slow_ticks": self.slow_ticks,
                               "target_quantity": self.target_quantity,
                               "rebalance_ticks": self.rebalance_ticks,
                               "threshold_fraction": self.threshold_fraction,
                               "allow_short": self.allow_short}}

    @staticmethod
    def _average(history, window):
        values = [quote.mid for quote in history[-window:]]
        return math.fsum(values) / len(values) if values else None

    def decide(self, asset_id, venue, decision_ns, history):
        fast = self._average(history, self.fast_ticks)
        slow = self._average(history, self.slow_ticks)
        if fast is None or slow is None:
            return None
        decision_id = f"{self.name}::{asset_id}::{venue}::{decision_ns}"
        if fast > slow * (1.0 + self.threshold_fraction):
            return Decision(decision_id=decision_id, asset_id=asset_id, venue=venue,
                            decision_ns=decision_ns, action=ACTION_LONG,
                            quantity=self.target_quantity, reason="synthetic_crossover_long")
        if fast < slow * (1.0 - self.threshold_fraction):
            if self.allow_short:
                return Decision(decision_id=decision_id, asset_id=asset_id, venue=venue,
                                decision_ns=decision_ns, action=ACTION_SHORT,
                                quantity=self.target_quantity, reason="synthetic_crossover_short")
            return Decision(decision_id=decision_id, asset_id=asset_id, venue=venue,
                            decision_ns=decision_ns, action=ACTION_FLAT,
                            reason="synthetic_crossover_short_suppressed")
        return Decision(decision_id=decision_id, asset_id=asset_id, venue=venue,
                        decision_ns=decision_ns, action=ACTION_FLAT,
                        reason="synthetic_crossover_flat")


class NoOpStrategy:
    """Reference strategy that never trades, so a run can be checked against zero."""

    name = "synthetic-no-op"
    declared_applicability = "Control strategy: emits no orders at all."

    def describe(self):
        return {"name": self.name, "declared_applicability": self.declared_applicability,
                "parameters": {}}

    def decide(self, asset_id, venue, decision_ns, history):
        return None


def generate_decisions(path, strategy):
    """Deterministic decision stream from a strategy and a synthetic path."""
    if isinstance(strategy, NoOpStrategy):
        return []
    by_instrument = {}
    for quote in path.quotes:
        by_instrument.setdefault((quote.asset_id, quote.venue), []).append(quote)
    for quotes in by_instrument.values():
        quotes.sort(key=lambda quote: (quote.available_ns, quote.event_ns))
    decisions = []
    rebalance_ticks = getattr(strategy, "rebalance_ticks", 1)
    for instrument in path.instruments:
        history = by_instrument[instrument]
        for index in range(0, len(history), rebalance_ticks):
            decision_ns = history[index].available_ns
            visible = [quote for quote in history if quote.available_ns <= decision_ns]
            decision = strategy.decide(instrument[0], instrument[1], decision_ns, visible)
            if decision is not None:
                decisions.append(decision)
    return decisions


# --------------------------------------------------------------------------
# Default policies for the paper backtest
# --------------------------------------------------------------------------
def default_execution_policy(*, fee_bps=0.5, half_spread_bps=0.0, leverage=5.0, max_leverage=20.0,
                             margin_mode=MARGIN_CROSS, position_mode=POSITION_ONE_WAY,
                             maintenance_margin_rate=0.005,
                             funding_interval_ns=DEFAULT_FUNDING_INTERVAL_NS,
                             funding_anchor_ns=0,
                             funding_rate_source=FUNDING_SOURCE_DECLARED_SCHEDULE,
                             default_funding_rate=0.00001, funding_rates=(),
                             funding_rate_schedule=(), liquidation_fee_bps=0.0,
                             order_time_to_live_ns=None):
    # An order must live long enough to meet a later synthetic observation; the default is
    # 4 ms for millisecond-spaced paths and is raised by run_backtest for coarser paths.
    time_to_live = 4 * MS if order_time_to_live_ns is None else int(order_time_to_live_ns)
    return ExecutionPolicy(
        fees=FeePolicy(fee_bps=fee_bps),
        spread=SpreadPolicy(half_spread_bps=half_spread_bps),
        slippage=SlippagePolicy(fixed_bps=0.0, impact_bps_at_full_capacity=0.0),
        capacity=CapacityPolicy(max_participation_fraction=0.25),
        fills=FillPolicy(reject_probability=0.0, fill_probability=1.0,
                         require_inventory_for_sell=False,
                         order_time_to_live_ns=time_to_live),
        latency=LatencyPolicy(decision_to_order_ns=1 * MS, order_to_ack_ns=1 * MS,
                              ack_to_execution_ns=1 * MS),
        marks=MarkPolicy(fallback="cost_basis", price_source=PRICE_SOURCE_MARK),
        account=AccountPolicy(require_sufficient_cash=True, on_insufficient_cash="reduce"),
        perps=PerpPolicy(allow_short=True, default_leverage=leverage, max_leverage=max_leverage,
                         margin_mode=margin_mode, position_mode=position_mode,
                         maintenance_margin_rate=maintenance_margin_rate,
                         funding_interval_ns=funding_interval_ns,
                         funding_anchor_ns=funding_anchor_ns,
                         funding_rate_source=funding_rate_source,
                         default_funding_rate=default_funding_rate,
                         funding_rates=funding_rates,
                         funding_rate_schedule=funding_rate_schedule,
                         liquidation_fee_bps=liquidation_fee_bps,
                         trace_perp_curve=True))


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def _drawdown_metrics(points):
    """Peak-to-trough equity drawdown from a causal, observation-time equity curve."""
    if not points:
        return {"max_drawdown_notional": 0.0, "max_drawdown_fraction": 0.0, "peak_ns": None,
                "trough_ns": None, "recovery_ns": None, "longest_drawdown_ns": 0,
                "final_drawdown_notional": 0.0, "observation_count": 0}
    peak = points[0]["equity"]
    peak_ns = points[0]["ns"]
    worst = 0.0
    worst_fraction = 0.0
    worst_peak_ns = peak_ns
    worst_trough_ns = peak_ns
    drawdown_start_ns = None
    longest = 0
    for point in points:
        equity = point["equity"]
        if equity >= peak:
            if drawdown_start_ns is not None:
                longest = max(longest, point["ns"] - drawdown_start_ns)
                drawdown_start_ns = None
            peak = equity
            peak_ns = point["ns"]
            continue
        if drawdown_start_ns is None:
            drawdown_start_ns = peak_ns
        drop = peak - equity
        if drop > worst:
            worst = drop
            worst_fraction = drop / peak if peak > 0 else 0.0
            worst_peak_ns = peak_ns
            worst_trough_ns = point["ns"]
    recovery_ns = None
    if worst_trough_ns is not None and worst > 0:
        trough_index = next(index for index, point in enumerate(points)
                            if point["ns"] == worst_trough_ns)
        for point in points[trough_index:]:
            if point["equity"] >= points[trough_index]["equity"] + worst:
                recovery_ns = point["ns"]
                break
    if drawdown_start_ns is not None:
        longest = max(longest, points[-1]["ns"] - drawdown_start_ns)
    return {"max_drawdown_notional": worst, "max_drawdown_fraction": worst_fraction,
            "peak_ns": worst_peak_ns if worst > 0 else None,
            "trough_ns": worst_trough_ns if worst > 0 else None,
            "recovery_ns": recovery_ns, "longest_drawdown_ns": longest,
            "final_drawdown_notional": peak - points[-1]["equity"],
            "observation_count": len(points)}


def _margin_usage(points, ledger):
    if not points:
        return {"max_allocated_margin": 0.0, "max_margin_ratio": None,
                "max_initial_margin_utilization": 0.0,
                "min_liquidation_distance_fraction": None,
                "max_gross_position_notional": 0.0,
                "mean_allocated_margin": 0.0}
    allocated = [point["allocated_margin"] for point in points]
    ratios = [point["margin_ratio"] for point in points if point["margin_ratio"] is not None]
    distances = [point["liquidation_distance_fraction"] for point in points
                 if point["liquidation_distance_fraction"] is not None]
    utilizations = [point["allocated_margin"] / point["equity"]
                    for point in points if point["equity"] > 0]
    return {"max_allocated_margin": max(allocated),
            "max_margin_ratio": max(ratios) if ratios else None,
            "max_initial_margin_utilization": max(utilizations) if utilizations else 0.0,
            "min_liquidation_distance_fraction": min(distances) if distances else None,
            "max_gross_position_notional": max(point["gross_position_notional"]
                                               for point in points),
            "mean_allocated_margin": math.fsum(allocated) / len(allocated),
            "final_allocated_margin": ledger["allocated_margin_total"]}


def _per_instrument(ledger, liquidations):
    realized = ledger["realized_pnl_by_asset"]
    fees = ledger["fees_by_asset"]
    funding = ledger["funding_cost_by_asset"]
    turnover = ledger["turnover_by_asset"]
    unrealized = ledger["unrealized_pnl_by_asset"]
    liquidation_counts = {}
    liquidation_fees = {}
    for record in liquidations:
        liquidation_counts[record["asset_id"]] = liquidation_counts.get(record["asset_id"], 0) + 1
        liquidation_fees[record["asset_id"]] = liquidation_fees.get(record["asset_id"], 0.0) \
            + record["liquidation_fee"]
    assets = sorted(set(realized) | set(fees) | set(funding) | set(turnover) | set(unrealized)
                    | set(liquidation_counts))
    breakdown = {}
    for asset_id in assets:
        asset_fees = fees.get(asset_id, 0.0)
        asset_funding = funding.get(asset_id, 0.0)
        asset_liq_fees = liquidation_fees.get(asset_id, 0.0)
        breakdown[asset_id] = {
            "realized_pnl": realized.get(asset_id, 0.0),
            "unrealized_pnl": unrealized.get(asset_id, 0.0),
            "explicit_fees": asset_fees,
            "funding_cost": asset_funding,
            "liquidation_fees": asset_liq_fees,
            "turnover_notional": turnover.get(asset_id, 0.0),
            "liquidation_count": liquidation_counts.get(asset_id, 0),
            "net_pnl": (realized.get(asset_id, 0.0) + unrealized.get(asset_id, 0.0)
                        - asset_fees - asset_funding - asset_liq_fees),
        }
    return breakdown


def _per_regime(points, regimes):
    labels = sorted({regime_at(regimes, point["ns"]) for point in points})
    buckets = {label: {"regime": label, "observation_count": 0, "equity_change": 0.0,
                       "turnover_notional": 0.0, "funding_cost": 0.0, "fees_paid": 0.0,
                       "fills": 0, "liquidations": 0, "max_drawdown_notional": 0.0}
               for label in labels}
    for previous, current in zip(points, points[1:]):
        label = regime_at(regimes, current["ns"])
        bucket = buckets[label]
        bucket["observation_count"] += 1
        bucket["equity_change"] += current["equity"] - previous["equity"]
        bucket["turnover_notional"] += current["turnover_to_date"] - previous["turnover_to_date"]
        bucket["funding_cost"] += current["funding_cost_to_date"] - previous["funding_cost_to_date"]
        bucket["fees_paid"] += current["fees_to_date"] - previous["fees_to_date"]
        bucket["fills"] += current["fill_count"] - previous["fill_count"]
        bucket["liquidations"] += current["liquidation_count"] - previous["liquidation_count"]
    peak = points[0]["equity"]
    for point in points:
        label = regime_at(regimes, point["ns"])
        bucket = buckets[label]
        equity = point["equity"]
        if equity >= peak:
            peak = equity
        else:
            bucket["max_drawdown_notional"] = max(bucket["max_drawdown_notional"], peak - equity)
    for bucket in buckets.values():
        bucket["net_pnl_attribution"] = bucket["equity_change"]
    return buckets


# --------------------------------------------------------------------------
# The backtest run
# --------------------------------------------------------------------------
def _run_once(path, decisions, *, execution_policy, contracts, initial_cash, seed, risk_gate,
              replay_id):
    return replay(path.quotes, decisions, asof_ns=path.asof_ns, policy=execution_policy,
                  seed=seed, initial_cash=initial_cash, risk_gate=risk_gate,
                  contracts=contracts, replay_id=replay_id, trace_perp_curve=True)


def _build_receipt(path, decisions, result, *, execution_policy, contracts, initial_cash, seed,
                   strategy_identity, risk_summary, replay_ids):
    ledger = result["ledger"]
    curve = list(result.get("perp_curve", []))
    start_ns = min(quote.available_ns for quote in path.quotes)
    initial_point = {
        "ns": start_ns, "cash": initial_cash, "equity": initial_cash,
        "margin_balance": initial_cash, "allocated_margin": 0.0,
        "maintenance_margin_required": 0.0, "margin_ratio": None,
        "liquidation_distance_fraction": 1.0, "unrealized_pnl": 0.0,
        "gross_position_notional": 0.0, "net_position_notional": 0.0,
        "funding_cost_to_date": 0.0, "fees_to_date": 0.0, "turnover_to_date": 0.0,
        "fill_count": 0, "liquidation_count": 0, "positions": {},
    }
    points = [initial_point] + [point for point in curve if point["ns"] > start_ns]
    if not points or points[-1]["ns"] != path.asof_ns:
        points.append(dict(points[-1], ns=path.asof_ns))
    drawdown = _drawdown_metrics(points)
    margin_usage = _margin_usage(points, ledger)
    per_instrument = _per_instrument(ledger, ledger["liquidations"])
    per_regime = _per_regime(points, path.regimes)
    attributed = math.fsum(entry["net_pnl"] for entry in per_instrument.values())
    regime_attributed = math.fsum(entry["equity_change"] for entry in per_regime.values())
    turnover_total = math.fsum(entry["turnover_notional"] for entry in per_instrument.values())
    return {
        "schema_version": SCHEMA,
        "scope": SCOPE,
        "policy_version": BACKTEST_POLICY_VERSION,
        "replay_ids": list(replay_ids),
        "market_path": path.to_summary(),
        "strategy": strategy_identity,
        "seed": seed,
        "initial_cash": initial_cash,
        "execution_policy": execution_policy.to_dict(),
        "contracts": [spec.to_dict() for spec in sorted(contracts, key=lambda item: item.key)],
        "risk": risk_summary,
        "counts": result["counts"],
        "keys": {"equity_curve": {"fingerprint": sha256_bytes(
            canonical_json([[point["ns"], point["equity"]] for point in points]).encode()),
            "point_count": len(points)},
            "ledger_sha256": result["ledger_sha256"],
            "replay_sha256": result["replay_sha256"]},
        "equity_curve": points,
        "drawdown": drawdown,
        "margin_usage": margin_usage,
        "turnover": {"total_notional": turnover_total,
                     "by_asset": {asset_id: entry["turnover_notional"]
                                  for asset_id, entry in per_instrument.items()}},
        "costs": {"explicit_fees": ledger["fees_paid"],
                  "liquidation_fees": ledger["liquidation_fees_paid"],
                  "funding_cost": ledger["funding_cost"],
                  "total": ledger["fees_paid"] + ledger["liquidation_fees_paid"]
                  + ledger["funding_cost"]},
        "funding": {"payment_count": ledger["funding_payment_count"],
                    "cost": ledger["funding_cost"],
                    "by_asset": dict(ledger["funding_cost_by_asset"]),
                    "payments": list(ledger["funding_payments"])},
        "liquidations": list(ledger["liquidations"]),
        "fills": list(ledger["fills"]),
        "orders": result["orders"],
        "decisions": result["decisions"],
        "ledger": ledger,
        "per_instrument": per_instrument,
        "per_regime": per_regime,
        "attribution": {
            "instrument_net_pnl_sum": attributed,
            "regime_net_pnl_sum": regime_attributed,
            "ledger_net_pnl": ledger["net_pnl"],
            "instrument_residual": attributed - ledger["net_pnl"],
            "regime_residual": regime_attributed - ledger["net_pnl"],
        },
        "provisional_pending_r1": (list(PROVISIONAL_POLICY_PARAMETERS)
                                   + list(PROVISIONAL_BACKTEST_PARAMETERS)),
        "limitations": [
            "Synthetic constructed market path and synthetic decisions; no real instrument, "
            "venue, feed, broker or exchange is contacted.",
            "Paper trading here is offline replay only. No order is submitted anywhere.",
            "Fees, funding rates, spreads, latency, capacity and margin parameters are explicit "
            "guesses that remain provisional pending R1.",
            "Not evidence of profitability, calibration, fill authenticity, capacity realism, "
            "survivorship safety or live readiness.",
            "The equity curve is sampled at synthetic price-observation times, so intra-interval "
            "excursions between observations are not represented.",
        ],
    }


def run_backtest(path, decisions=None, *, strategy=None, execution_policy=None, contracts=None,
                 initial_cash=100_000.0, seed=None, risk_gate=None, verify_determinism=True,
                 replay_id="paper-backtest"):
    """Replay a synthetic path and return a complete, machine-readable backtest receipt."""
    if not isinstance(path, SyntheticMarketPath):
        raise ValueError("run_backtest requires a SyntheticMarketPath")
    if decisions is None:
        if strategy is None:
            strategy = MovingAverageCrossoverStrategy()
        decisions = generate_decisions(path, strategy)
    decisions = list(decisions)
    if strategy is None:
        strategy_identity = {"name": "caller-supplied-decision-stream",
                             "declared_applicability": "Caller-supplied synthetic decisions.",
                             "parameters": {}, "decision_count": len(decisions)}
    else:
        strategy_identity = strategy.describe()
        strategy_identity["decision_count"] = len(decisions)
    if execution_policy is None:
        execution_policy = default_execution_policy(
            funding_anchor_ns=min(quote.event_ns for quote in path.quotes),
            funding_interval_ns=derive_funding_interval_ns(path),
            order_time_to_live_ns=max(4 * MS, 3 * path.tick_ns))
    contracts = tuple(contracts) if contracts is not None \
        else default_contracts(path.instruments, margin_mode=execution_policy.perps.margin_mode)
    seed = path.seed if seed is None else seed
    if type(seed) is not int:
        raise ValueError("seed requires an integer")
    _finite(initial_cash, "initial_cash")

    first = _run_once(path, decisions, execution_policy=execution_policy, contracts=contracts,
                      initial_cash=float(initial_cash), seed=seed, risk_gate=risk_gate,
                      replay_id=replay_id)
    risk_summary = {"risk_gate": "absent" if risk_gate is None else "financial_risk_v1",
                    "risk_blocked_decisions": first["counts"]["risk_blocked_decisions"],
                    "limits": (risk_gate.limits.to_dict()
                               if risk_gate is not None and hasattr(risk_gate, "limits") else None)}
    receipt = _build_receipt(path, decisions, first, execution_policy=execution_policy,
                             contracts=contracts, initial_cash=float(initial_cash), seed=seed,
                             strategy_identity=strategy_identity, risk_summary=risk_summary,
                             replay_ids=[replay_id])
    receipt["determinism"] = {"runs": 1, "byte_identical": None,
                              "backtest_sha256": [None, None]}
    fingerprint = sha256_bytes(canonical_json(receipt).encode())
    if verify_determinism:
        second = _run_once(path, decisions, execution_policy=execution_policy, contracts=contracts,
                           initial_cash=float(initial_cash), seed=seed, risk_gate=risk_gate,
                           replay_id=replay_id)
        second_receipt = _build_receipt(path, decisions, second, execution_policy=execution_policy,
                                        contracts=contracts, initial_cash=float(initial_cash),
                                        seed=seed, strategy_identity=strategy_identity,
                                        risk_summary=risk_summary, replay_ids=[replay_id])
        second_receipt["determinism"] = {"runs": 1, "byte_identical": None,
                                         "backtest_sha256": [None, None]}
        second_fingerprint = sha256_bytes(canonical_json(second_receipt).encode())
        identical = fingerprint == second_fingerprint
        receipt["determinism"] = {"runs": 2, "byte_identical": identical,
                                  "backtest_sha256": [fingerprint, second_fingerprint],
                                  "equity_curve_identical":
                                      canonical_json(receipt["equity_curve"])
                                      == canonical_json(second_receipt["equity_curve"]),
                                  "fills_identical":
                                      canonical_json(receipt["fills"])
                                      == canonical_json(second_receipt["fills"]),
                                  "funding_identical":
                                      canonical_json(receipt["funding"]["payments"])
                                      == canonical_json(second_receipt["funding"]["payments"]),
                                  "liquidations_identical":
                                      canonical_json(receipt["liquidations"])
                                      == canonical_json(second_receipt["liquidations"])}
        fingerprint = sha256_bytes(canonical_json(receipt).encode())
    else:
        receipt["determinism"] = {"runs": 1, "byte_identical": None,
                                  "backtest_sha256": [fingerprint, None]}
    receipt["backtest_sha256"] = fingerprint
    receipt["code"] = [file_identity(Path(__file__)),
                       file_identity(Path(__file__).with_name("financial_simulator_v1.py")),
                       file_identity(Path(__file__).with_name("financial_risk_v1.py"))]
    require_backtest_invariants(receipt)
    return receipt


def require_backtest_invariants(receipt):
    """Independent consistency checks on the emitted receipt; raises on any violation."""
    ledger = receipt["ledger"]
    tolerance = 1e-6 * max(1.0, receipt["initial_cash"])
    attribution = receipt["attribution"]
    if abs(attribution["instrument_residual"]) > tolerance:
        raise BacktestError(f"per-instrument attribution does not close: {attribution}")
    if abs(attribution["regime_residual"]) > tolerance:
        raise BacktestError(f"per-regime attribution does not close: {attribution}")
    if not ledger["conservation"]["ok"]:
        raise BacktestError(f"ledger conservation failed: {ledger['conservation']['violations']}")
    curve = receipt["equity_curve"]
    if curve and abs(curve[0]["equity"] - receipt["initial_cash"]) > tolerance:
        raise BacktestError("equity curve does not start at the initial cash")
    if curve and abs(curve[-1]["equity"] - ledger["equity"]) > tolerance:
        raise BacktestError("equity curve does not end at the final ledger equity")
    if receipt["determinism"]["runs"] == 2 and not receipt["determinism"]["byte_identical"]:
        raise BacktestError("backtest is not byte-identical across runs")
    for point in curve:
        if point["equity"] != point["equity"] or not math.isfinite(point["equity"]):
            raise BacktestError("equity curve contains a non-finite value")
    return receipt


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_default_risk_gate(margin_mode=MARGIN_CROSS, max_leverage=20.0):
    """Perp-aware default guard set, kept out of the strategy entirely."""
    import financial_risk_v1 as risk

    return risk.RiskEngine(limits=risk.RiskLimits(
        max_leverage=max_leverage,
        max_position_quantity=1_000.0,
        min_position_quantity=-1_000.0,
        max_gross_exposure_notional=2_000_000.0,
        max_abs_net_exposure_notional=2_000_000.0,
        max_position_notional_per_venue=2_000_000.0,
        max_net_loss_notional=50_000.0,
        max_realized_loss_notional=50_000.0,
        max_turnover_notional=5_000_000.0,
        max_order_notional=500_000.0,
        max_funding_cost_notional=5_000.0,
        max_margin_ratio=0.8,
        min_available_margin_notional=0.0,
        min_liquidation_distance_fraction=0.05,
        max_initial_margin_utilization=0.95,
        require_margin_sufficiency=True,
        max_data_staleness_ns=60 * 10**9,
    ))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="receipt path")
    parser.add_argument("--seed", type=int, default=20_260_919)
    parser.add_argument("--ticks", type=int, default=DEFAULT_TICKS)
    parser.add_argument("--tick-ns", type=int, default=DEFAULT_TICK_NS)
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
    parser.add_argument("--leverage", type=float, default=5.0)
    parser.add_argument("--margin-mode", choices=(MARGIN_CROSS, MARGIN_ISOLATED),
                        default=MARGIN_CROSS)
    parser.add_argument("--position-mode", choices=(POSITION_ONE_WAY, POSITION_HEDGE),
                        default=POSITION_ONE_WAY)
    parser.add_argument("--funding-interval-ns", type=int, default=None,
                        help="declared funding interval; defaults to the path-derived value")
    parser.add_argument("--funding-rate", type=float, default=0.00001)
    parser.add_argument("--strategy", choices=("ma_crossover", "no_op"), default="ma_crossover")
    parser.add_argument("--with-risk-limits", action="store_true",
                        help="attach the financial_risk_v1 perp guard set")
    parser.add_argument("--no-determinism-check", action="store_true",
                        help="run once instead of twice (faster, weaker receipt)")
    args = parser.parse_args(argv)

    path = build_synthetic_perp_path(args.seed, ticks=args.ticks, tick_ns=args.tick_ns,
                                     funding_rate=args.funding_rate)
    strategy = (MovingAverageCrossoverStrategy()
                if args.strategy == "ma_crossover" else NoOpStrategy())
    contracts = default_contracts(path.instruments, margin_mode=args.margin_mode,
                                 max_leverage=max(20.0, args.leverage))
    policy = default_execution_policy(leverage=args.leverage, max_leverage=max(20.0, args.leverage),
                                      margin_mode=args.margin_mode,
                                      position_mode=args.position_mode,
                                      funding_interval_ns=(args.funding_interval_ns
                                                           or derive_funding_interval_ns(path)),
                                      funding_anchor_ns=min(quote.event_ns for quote in path.quotes),
                                      order_time_to_live_ns=max(4 * MS, 3 * path.tick_ns),
                                      default_funding_rate=args.funding_rate,
                                      funding_rates=tuple(
                                          (asset_id, venue, args.funding_rate)
                                          for asset_id, venue in path.instruments))
    gate = build_default_risk_gate(args.margin_mode, max(20.0, args.leverage)) \
        if args.with_risk_limits else None
    receipt = run_backtest(path, strategy=strategy, execution_policy=policy, contracts=contracts,
                           initial_cash=args.initial_cash, risk_gate=gate,
                           verify_determinism=not args.no_determinism_check,
                           replay_id=f"paper-backtest-{args.seed}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(canonical_json({
        "output": str(args.output), "schema_version": receipt["schema_version"],
        "backtest_sha256": receipt["backtest_sha256"],
        "byte_identical": receipt["determinism"]["byte_identical"],
        "fills": receipt["counts"]["fills"], "funding_payments": receipt["counts"]["funding_payments"],
        "liquidations": receipt["counts"]["liquidations"],
        "net_pnl": receipt["ledger"]["net_pnl"],
        "max_drawdown_fraction": receipt["drawdown"]["max_drawdown_fraction"],
        "synthetic_only": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
