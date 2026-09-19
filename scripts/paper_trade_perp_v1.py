#!/usr/bin/env python3
"""Paper-trade REAL Binance USDT-M perpetual daily bars through the P2b simulator.

What this is: an offline, simulated replay of a deterministic reference strategy over
REAL venue daily bars, executed by the perpetual-contract simulator, producing a complete
receipt (fills, funding, fees, margin, liquidations, equity curve, drawdown, turnover).

What this is NOT:
* No order leaves this process. No venue, broker, exchange API or credential is contacted
  and no network request is made by this script - it reads files already downloaded.
* The execution costs are DECLARED PROVISIONAL GUESSES, not measured venue costs. The
  archive has no order book, so bid/ask are set to the mark close and the entire spread
  cost is expressed through one declared policy parameter pending R1.
* The reference strategy is a mechanical moving-average crossover. It is an execution-path
  exerciser, NOT a claim of predictive value, edge or profitability.
* No result here may be quoted as a return.

Real inputs used: mark-price daily closes (the label/execution price convention), last-price
daily bars (volume, trade count, taker-buy flow), and settled funding rates. The archive is
a CURRENT SNAPSHOT of history, not an as-of vintage.
"""
import argparse
import json
import math
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from build_perp_pit_v1 import load_funding, load_klines  # noqa: E402
from financial_backtest_v1 import (  # noqa: E402
    RegimeWindow, _drawdown_metrics, _per_regime, regime_at,
)
from financial_simulator_v1 import (  # noqa: E402
    AccountPolicy, ContractSpec, Decision, ExecutionPolicy, FeePolicy, FillPolicy,
    LatencyPolicy, MarkPolicy, PerpPolicy, Quote, SlippagePolicy, SpreadPolicy, replay,
)

MS = 1_000_000
REGIME_NORMAL = "normal"
REGIME_VOL_HIGH = "vol_high"
REGIME_LIQUIDITY_LOW = "liquidity_low"
REGIME_FUNDING_EXTREME = "funding_extreme"
REGIME_BASIS_BLOWOUT = "basis_blowout"
# Declared priority order: the most specific stress label wins when several fire.
REGIME_PRIORITY = (REGIME_BASIS_BLOWOUT, REGIME_FUNDING_EXTREME, REGIME_VOL_HIGH,
                   REGIME_LIQUIDITY_LOW)


def quantile(sorted_values, fraction):
    """Nearest-rank quantile of an already-sorted list; no interpolation."""
    if not sorted_values:
        return None
    index = min(len(sorted_values) - 1, max(0, int(round(fraction * (len(sorted_values) - 1)))))
    return sorted_values[index]


def build_regime_windows(reference_rows, *, min_history=48):
    """Causal regime windows from trailing statistics of the reference instrument.

    Every threshold is an EXPANDING quantile computed only from observations at or before
    the bar being labelled, so a regime label never uses future data. ``reference_rows`` is
    an ordered list of ``(ns, realized_vol, log_quote_volume, abs_funding, abs_basis_bps)``.
    """
    volatility, liquidity, funding, basis = [], [], [], []
    minute_buckets = []
    for ns, vol, log_qv, abs_fund, abs_basis in reference_rows:
        label = REGIME_NORMAL
        if len(volatility) >= min_history:
            fired = {
                REGIME_VOL_HIGH: vol > (quantile(sorted(volatility), 0.75) or 0.0),
                REGIME_LIQUIDITY_LOW: log_qv < (quantile(sorted(liquidity), 0.25) or 0.0),
                REGIME_FUNDING_EXTREME: abs_fund > (quantile(sorted(funding), 0.90) or 0.0),
                REGIME_BASIS_BLOWOUT: abs_basis > (quantile(sorted(basis), 0.90) or 0.0),
            }
            for candidate in REGIME_PRIORITY:
                if fired.get(candidate):
                    label = candidate
                    break
        volatility.append(vol)
        liquidity.append(log_qv)
        funding.append(abs_fund)
        basis.append(abs_basis)
        minute_buckets.append((ns, label))
    return minute_buckets


def collapse_windows(minute_buckets):
    """Merge consecutive same-label observations into half-open RegimeWindows."""
    windows = []
    for index, (ns, label) in enumerate(minute_buckets):
        end = minute_buckets[index + 1][0] if index + 1 < len(minute_buckets) else ns + 1
        if end <= ns:
            end = ns + 1
        if windows and windows[-1].label == label and windows[-1].end_ns == ns:
            previous = windows[-1]
            windows[-1] = RegimeWindow(label=label, start_ns=previous.start_ns, end_ns=end)
        else:
            windows.append(RegimeWindow(label=label, start_ns=ns, end_ns=end))
    return tuple(windows)


def parse_spec(raw):
    """``SYMBOL:tick:lot:mult:maxlev:mmr`` -> ContractSpec for the ``binance_um`` venue."""
    parts = raw.split(":")
    if len(parts) != 6:
        raise ValueError("contract spec must be SYMBOL:tick:lot:mult:maxlev:mmr")
    symbol = parts[0]
    return symbol, ContractSpec(asset_id=f"{symbol}-PERP", venue="binance_um",
                                contract_multiplier=float(parts[3]), tick_size=float(parts[1]),
                                lot_size=float(parts[2]), max_leverage=float(parts[4]),
                                maintenance_margin_rate=float(parts[5]))


def load_bybit(root, symbol):
    """Read the Bybit v5 JSON fetch into the same shapes the Binance loader returns."""
    base = pathlib.Path(root) / "bybit"
    marks, lasts, indexes = {}, {}, {}
    for name, target in (("kline", lasts), ("mark", marks), ("index", indexes)):
        path = base / f"{symbol}.{name}.json"
        if not path.exists():
            raise SystemExit(f"missing bybit series {path}; run fetch_venue_perp_v1.py first")
        for row in json.loads(path.read_text()):
            start = int(row[0])
            target[start] = {"open": float(row[1]), "high": float(row[2]), "low": float(row[3]),
                             "close": float(row[4]), "close_time": start + 86_400_000 - 1,
                             "volume": float(row[5]) if len(row) > 5 else 0.0,
                             "quote_volume": float(row[6]) if len(row) > 6 else 0.0,
                             "count": 0, "taker_buy_volume": 0.0}
    funding_path = base / f"{symbol}.funding.json"
    funding = [{"calc_time": int(r["fundingRateTimestamp"]), "interval_hours": 8,
                "rate": float(r["fundingRate"])}
               for r in json.loads(funding_path.read_text())]
    funding.sort(key=lambda row: row["calc_time"])
    return marks, lasts, indexes, funding


def load_aster(root, symbol):
    """Read the Aster fetch (Binance-shaped arrays) into the common shapes."""
    base = pathlib.Path(root) / "aster"
    marks, lasts, indexes = {}, {}, {}
    for name, target in (("kline", lasts), ("mark", marks), ("index", indexes)):
        path = base / f"{symbol}.{name}.json"
        if not path.exists():
            raise SystemExit(f"missing aster series {path}; run fetch_venue_perp_v1.py first")
        for row in json.loads(path.read_text()):
            start = int(row[0])
            # Aster rows: [openTime, o, h, l, c, volume, closeTime, quoteVolume, count, ...]
            target[start] = {"open": float(row[1]), "high": float(row[2]), "low": float(row[3]),
                             "close": float(row[4]), "close_time": int(row[6]),
                             "volume": float(row[5]), "quote_volume": float(row[7]),
                             "count": int(row[8]), "taker_buy_volume": float(row[9])}
    funding_rows = json.loads((base / f"{symbol}.funding.json").read_text())
    # Aster funding rows carry no interval; fundingInfo declares it per symbol.
    info = json.loads((base / f"{symbol}.funding_info.json").read_text())
    interval = int(info[0].get("fundingIntervalHours", 8)) if info else 8
    funding = [{"calc_time": int(r["fundingTime"]), "interval_hours": interval,
                "rate": float(r["fundingRate"])} for r in funding_rows]
    funding.sort(key=lambda row: row["calc_time"])
    return marks, lasts, indexes, funding


def load_source(source, archive_root, venue_root, symbol):
    if source == "binance":
        return (load_klines(archive_root, "markPriceKlines", symbol),
                load_klines(archive_root, "klines", symbol),
                load_klines(archive_root, "indexPriceKlines", symbol),
                load_funding(archive_root, symbol))
    if source == "aster":
        return load_aster(venue_root, symbol)
    return load_bybit(venue_root, symbol)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive-root", type=pathlib.Path,
                        default=pathlib.Path("data/binance_vision_v1"))
    parser.add_argument("--venue-root", type=pathlib.Path,
                        default=pathlib.Path("data/venue_perp_v1"))
    parser.add_argument("--source", choices=("binance", "bybit", "aster"), default="binance",
                        help="which venue's data drives the quotes (cross-venue check)")
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path("results/paper_trade_perp_v1.json"))
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    parser.add_argument("--first-day", default="2024-01-01")
    parser.add_argument("--last-day", default="2026-08-31")
    parser.add_argument("--fast", type=int, default=20)
    parser.add_argument("--slow", type=int, default=60)
    parser.add_argument("--half-spread-bps", type=float, default=1.0)
    parser.add_argument("--fee-bps", type=float, default=5.0)
    parser.add_argument("--leverage", type=float, default=3.0)
    parser.add_argument("--margin-mode", choices=("cross", "isolated"), default="cross")
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
    parser.add_argument("--contract", action="append", default=None,
                        help="SYMBOL:tick:lot:mult:maxlev:mmr (repeatable)")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    specs = dict(parse_spec(item) for item in (args.contract or [
        "BTCUSDT:0.10:0.001:1:75:0.004", "ETHUSDT:0.01:0.01:1:75:0.005",
        "SOLUSDT:0.0010:0.1:1:50:0.01", "BNBUSDT:0.010:0.01:1:50:0.0065",
        "XRPUSDT:0.0001:1:1:50:0.01"]))

    policy = ExecutionPolicy(
        fees=FeePolicy(fee_bps=args.fee_bps), spread=SpreadPolicy(half_spread_bps=args.half_spread_bps),
        slippage=SlippagePolicy(fixed_bps=0.0),
        # Daily bars: an order must remain alive until the next observation, so the
        # 1-second default TTL cannot be used. Two days is a declared convention.
        fills=FillPolicy(fill_probability=1.0,
                         order_time_to_live_ns=2 * 86_400_000_000_000),
        latency=LatencyPolicy(), marks=MarkPolicy(),
        account=AccountPolicy(require_sufficient_cash=True),
        perps=PerpPolicy(allow_short=True, default_leverage=args.leverage,
                         margin_mode=args.margin_mode,
                         funding_rate_source="quote_funding_rate_field"),
    )
    contracts = tuple(specs[s] for s in symbols if s in specs)

    quotes, decisions, per_symbol = [], [], {}
    reference_rows = []
    regime_reference = symbols[0]
    for symbol in symbols:
        marks, lasts, indexes, funding = load_source(args.source, args.archive_root,
                                                     args.venue_root, symbol)
        days = sorted(set(marks) & set(lasts) & set(indexes))
        closes, taken = [], []
        quote_volume_logs, abs_funding, abs_basis = [], [], []
        cursor = 0
        last_rate = None
        for index, day in enumerate(days):
            mark = marks[day]["close"]
            if mark <= 0 or lasts[day]["volume"] <= 0 or indexes[day]["close"] <= 0:
                continue
            close_time = marks[day]["close_time"]
            while cursor < len(funding) and funding[cursor]["calc_time"] <= close_time:
                last_rate = funding[cursor]["rate"]
                cursor += 1
            quotes.append(Quote(asset_id=f"{symbol}-PERP", venue="binance_um",
                                event_ns=close_time * MS + 1, available_ns=close_time * MS + 2,
                                bid=mark, ask=mark, volume=lasts[day]["volume"],
                                source_id=f"{args.source}_mark_price_klines",
                                version="perp-1d-mark-price-close-v1",
                                mark_price=mark, last_price=lasts[day]["close"],
                                funding_rate=last_rate))
            closes.append(mark)
            taken.append(day)
            quote_volume_logs.append(lasts[day]["quote_volume"])
            abs_funding.append(abs(last_rate or 0.0))
            abs_basis.append(abs((mark - indexes[day]["close"]) / indexes[day]["close"] * 10000.0))
        # Causal MA crossover: uses only closes up to and including the decision bar.
        position = 0.0
        for index, day in enumerate(taken):
            if index + 1 < args.slow:
                continue
            fast = sum(closes[index + 1 - args.fast:index + 1]) / args.fast
            slow = sum(closes[index + 1 - args.slow:index + 1]) / args.slow
            target = 1.0 if fast > slow else -1.0
            action = "long" if target > 0 else "short"
            if target != position:
                quantity = max(1.0, round(0.25 * args.initial_cash * args.leverage / closes[index]))
                decisions.append(Decision(decision_id=f"{symbol}-{day}", asset_id=f"{symbol}-PERP",
                                          venue="binance_um",
                                          decision_ns=marks[day]["close_time"] * MS + 1,
                                          action=action, quantity=quantity,
                                          reason="reference_ma_crossover"))
                position = target
        if symbol == regime_reference:
            for index, day in enumerate(taken):
                window = closes[max(0, index - 24):index + 1]
                if len(window) < 25:
                    continue
                returns = [math.log(window[i + 1] / window[i]) for i in range(len(window) - 1)]
                reference_rows.append((
                    marks[day]["close_time"] * MS + 1, statistics.stdev(returns),
                    math.log(max(quote_volume_logs[index], 1e-9)), abs_funding[index],
                    abs_basis[index]))
        per_symbol[symbol] = {"quotes": len(quotes), "decisions": len(decisions)}
    regime_buckets = build_regime_windows(reference_rows)
    regimes = collapse_windows(regime_buckets)
    regime_labels = {}
    for _, label in regime_buckets:
        regime_labels[label] = regime_labels.get(label, 0) + 1

    if not quotes:
        raise SystemExit("no real quotes were built; check the archive root and range")
    asof_ns = max(q.event_ns for q in quotes) + 1
    result = replay(quotes, decisions, asof_ns=asof_ns, policy=policy, seed=20260919,
                    initial_cash=args.initial_cash, contracts=contracts, replay_id="paper-perp-v1",
                    trace_perp_curve=True)
    curve = result.get("perp_curve") or []
    drawdown = _drawdown_metrics(curve)
    per_regime = _per_regime(curve, regimes) if curve and regimes else {}

    counts = result["counts"]
    ledger = result.get("ledger", {})
    receipt = {
        "schema_version": "nanojev-paper-trade-perp-v1",
        "mode": "PAPER / SIMULATED. No order was placed and no venue was contacted.",
        "data": {"source": ("Binance public archive" if args.source == "binance"
                            else "Bybit public v5 market API (api.bybit.nl mirror)")
                         + " USDT-M perpetual daily bars",
                 "venue": "binance_um" if args.source == "binance" else "bybit_linear",
                 "symbols": symbols,
                 "first_day": args.first_day, "last_day": args.last_day,
                 "point_in_time": "current snapshot of history, NOT an as-of vintage"},
        "strategy": {"name": "reference_ma_crossover", "fast": args.fast, "slow": args.slow,
                     "declaration": "mechanical execution-path exerciser; NOT a claim of edge, "
                                    "predictive value or profitability"},
        "policy_provisional": {
            "half_spread_bps": args.half_spread_bps, "fee_bps": args.fee_bps,
            "leverage": args.leverage, "margin_mode": args.margin_mode,
            "note": "declared provisional guesses pending R1; the archive has no order book, so "
                    "bid/ask equal the mark close and all spread cost is one declared parameter",
        },
        "counts": counts,
        "ledger": {key: ledger.get(key) for key in (
            "initial_cash", "cash", "equity", "net_pnl", "net_pnl_excluding_explicit_fees",
            "net_pnl_excluding_all_costs", "realized_pnl", "unrealized_pnl", "fees_paid",
            "liquidation_fees_paid", "funding_cost", "funding_payment_count",
            "gross_traded_notional", "turnover_by_asset", "gross_position_notional",
            "allocated_margin_total", "fill_count", "liquidation_count")},
        "net_pnl_fraction_of_initial": (
            (ledger.get("net_pnl") / args.initial_cash) if ledger.get("net_pnl") is not None else None),
        "liquidations": ledger.get("liquidation_count"),
        "conservation": (ledger.get("conservation") or {}).get("ok"),
        "drawdown": drawdown,
        "regime_definition": {
            "reference_instrument": regime_reference,
            "method": "causal expanding quantiles over trailing observations only; nearest-rank, "
                      "no interpolation; minimum history 48 bars before any stress label may fire",
            "priority": list(REGIME_PRIORITY),
            "rules": {
                REGIME_VOL_HIGH: "trailing 24-bar realized vol above the expanding 75th percentile",
                REGIME_LIQUIDITY_LOW: "log quote volume below the expanding 25th percentile",
                REGIME_FUNDING_EXTREME: "|settled funding rate| above the expanding 90th percentile",
                REGIME_BASIS_BLOWOUT: "|mark-index basis bps| above the expanding 90th percentile",
                REGIME_NORMAL: "none of the above fired",
            },
            "observation_counts": regime_labels,
            "caveat": "one global schedule keyed off a single reference instrument; regimes are "
                      "descriptive buckets, not independent experiments",
        },
        "per_regime": per_regime,
        "curve_summary": {
            "point_count": len(curve),
            "first_equity": curve[0]["equity"] if curve else None,
            "last_equity": curve[-1]["equity"] if curve else None,
            "max_margin_ratio": max((p.get("margin_ratio") or 0.0) for p in curve) if curve else None,
            "min_liquidation_distance_fraction": min(
                (p.get("liquidation_distance_fraction") for p in curve
                 if p.get("liquidation_distance_fraction") is not None), default=None),
        },
        "replay_sha256": result.get("replay_sha256") or result.get("ledger_sha256"),
        "honesty": {
            "not_a_return": "costs are provisional guesses and the price convention is a gross "
                            "mark move; this figure is a simulation artefact, not a return",
            "no_profitability_claim": True,
            "not_live_ready": "no fill authenticity, queue position, capacity or venue-parameter "
                              "fidelity is established",
            "fill_convention": "a decision taken at the close of bar d fills at bar d's mark price "
                               "because that is the newest available observation; this is the "
                               "conventional decide-at-close convention and is mildly optimistic",
            "funding_convention": "each funding boundary charges the most recent SETTLED rate known "
                                  "at that instant; strictly the boundary should use the rate that "
                                  "settles at it. Declared approximation pending R1.",
            "equity_curve": "this driver reports ledger totals; the equity-curve, drawdown and "
                            "per-regime breakdown analytics live in financial_backtest_v1.py and "
                            "are not yet wired to real data",
            "recorded_caveat": "every quantity here is simulated; treat as an internal "
                               "consistency and plumbing check",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n",
                           encoding="utf-8")
    print(json.dumps({"output": str(args.output), "quotes": len(quotes),
                      "decisions": len(decisions), "counts": counts,
                      "net_pnl": receipt["ledger"]["net_pnl"],
                      "liquidations": receipt["liquidations"]}, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
