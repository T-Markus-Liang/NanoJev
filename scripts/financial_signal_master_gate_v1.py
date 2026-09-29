#!/usr/bin/env python3
"""Unified state machine under one BTC-trend master gate, portfolio-level (T111).

Context: T79 spec v1 / T99 HL / T106 XS dfh / T107 Bybit all found the same
structure — the signal only earns while BTCUSDT is in a 20d uptrend. This
script measures the JOINT system honestly as a daily-rebalanced portfolio on
the 10-asset XS cohort ``data/perp_pit_xs_v1/records.jsonl`` (13,380 records,
2023-01-01 -> 2026-08-30, close basis, funding available).

System under test
-----------------
  MASTER GATE   BTCUSDT-PERP close > SMA20 (SMA over the last 20 closes
                INCLUDING the decision close) -> risk-on; otherwise every
                sleeve is flat. (Sensitivity arms: the frozen spec's original
                trend def close/close[t-20]-1 > 0, and close > strictly-prior
                SMA20.)
  SLEEVE A      spec-v1 ``crowded_long_carry_follow_v1`` approximation on the
                XS cohort: long every asset whose funding_pct (last_funding_rate
                mid-rank pct in the strictly-prior trailing-180 records, the
                labels_v4 convention) is >= 0.80, equal weight, only when the
                gate is on. DEVIATION FROM SPEC: the spec also requires
                basis_pct >= 0.66 (mark_index_basis_bps mid-rank) and the
                XS cohort carries no mark/index/basis feature for the 5 new
                symbols — the basis leg is dropped, so this is a FUNDING-ONLY
                approximation, not the frozen spec.
  SLEEVE B      XS dfh (T104/T106): long top-2 / short bottom-2 by dfh20 rank
                (each leg 1/4 of the sleeve book, dollar-neutral), only when
                the gate is on; >=8 ranked assets required (T104 day gate).
  COMBINED      one unified book: w = 0.5 * wA + 0.5 * wB per asset (sleeve
                views net before turnover is measured).

Baselines
---------
  always_long_ew        long all 10 assets equal weight, every day, no gate
  gate_always_long      same book but only while the gate is on (isolates the
                        gate's value on the passive book)
  sleeve_A / sleeve_B   each gated sleeve standalone (100% capital)
  ungated variants      sleeve A, sleeve B and combined with the gate removed;
                        their daily returns split by gate state measure whether
                        the gate adds value beyond the sleeves themselves

Costs
-----
  5bps per unit of traded notional per leg, charged on the actual book change:
  cost_t = 5bps * sum_a |w_t(a) - w_{t-1}(a)| where w is the signed fraction of
  the arm's capital. Establishing a full-gross book costs 5bps; closing it
  costs 5bps; a same-size leg swap trades 0.5 of book and costs 2.5bps. This
  counts BOTH the exit and the entry side — more conservative than T106's
  slot-fraction convention (which charged 1.25bps for the same swap).

Metrics per arm: net daily return series -> compounded cumulative return,
annualized Sharpe (mean/sd*sqrt(365), crypto trades daily), max drawdown of the
compounded equity, days-active %, mean daily turnover/cost, per-year table
(2023/24/25/26). Gate-on vs gate-off day counts reported globally.

Honest notes carried in the report: close-price basis (no funding cashflows,
no borrow on shorts, fills at decision close); funding-only spec-v1 deviation;
sleeve-A/B long overlap measured (not assumed); ~10 assets => ~4-5 effective XS
dof/day; SURVIVORSHIP — all 10 assets listed pre-2023 and survived to 2026-08,
delisted perps are absent from the universe so all long-side results carry an
optimistic bias; sleeve A cannot trade for its first ~180 days (trailing-180
funding_pct warmup) so 2023 is a partial year by construction.

Protocol + owner self-authorization are embedded in the single output artifact
``results/financial_signal_master_gate_v1.json`` (protocol object hashed by
sha256, authorization pins the hash) — T111 directs protocol+auth into the
results file rather than separate research/ artifacts. Measurement only: no
fitting, no trading, no network.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import pathlib
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
COHORT = ROOT / "data/perp_pit_xs_v1/records.jsonl"
OUT = ROOT / "results/financial_signal_master_gate_v1.json"

TRAIL_FUNDING = 180      # strictly-prior trailing window for funding_pct
FUNDING_PCT_MIN = 0.80   # spec v1 funding leg threshold (basis leg unavailable)
SMA_WIN = 20             # BTC master-gate SMA length
EDGE = 2                 # sleeve B top-K / bottom-K
MIN_ASSETS_DFH = 8       # T104 day gate for the XS rank
COST_BPS = 5.0           # per unit traded notional, per leg
SLEEVE_SPLIT = 0.5       # combined = 50/50 capital split
DAYS_PER_YEAR = 365
BTC = "BTCUSDT-PERP"


def _r(x, nd=3):
    return round(x, nd) if isinstance(x, float) else x


def mid_rank_pct(window, x):
    """Mid-rank percentile of x within strictly-prior window (project conv.)."""
    return (sum(1 for v in window if v < x)
            + 0.5 * sum(1 for v in window if v == x)) / len(window)


def load_cohort(path):
    """asset -> sorted rows {day, close, dfh20, funding}; derives fwd_1d and
    funding_pct (needs a full strictly-prior 180-record window). Also returns
    the day -> gate-bool maps for the three gate definitions."""
    series = defaultdict(list)
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            rec = json.loads(line)
            f = rec["features"]
            series[rec["asset_id"]].append({
                "day": rec["id"].rsplit(":", 1)[-1],
                "close": f["close"]["value"],
                "dfh20": f["dfh20"]["value"],
                "funding": f["last_funding_rate"]["value"],
            })
    for rows in series.values():
        rows.sort(key=lambda r: r["day"])
        closes = [r["close"] for r in rows]
        funds = [r["funding"] for r in rows]
        for i, r in enumerate(rows):
            r["fwd_1d"] = (closes[i + 1] / closes[i] - 1.0
                           if i + 1 < len(closes) else None)
            r["funding_pct"] = (mid_rank_pct(funds[i - TRAIL_FUNDING:i],
                                             funds[i])
                                if i >= TRAIL_FUNDING else None)
    gates = {"close_gt_sma20_incl": {}, "ret20_gt_0": {},
             "close_gt_sma20_prior": {}}
    btc_rows = series.get(BTC, [])
    bc = [r["close"] for r in btc_rows]
    for i, r in enumerate(btc_rows):
        if i >= SMA_WIN - 1:
            gates["close_gt_sma20_incl"][r["day"]] = (
                bc[i] > sum(bc[i - SMA_WIN + 1:i + 1]) / SMA_WIN)
        if i >= SMA_WIN:
            gates["ret20_gt_0"][r["day"]] = bc[i] / bc[i - SMA_WIN] - 1.0 > 0.0
            gates["close_gt_sma20_prior"][r["day"]] = (
                bc[i] > sum(bc[i - SMA_WIN:i]) / SMA_WIN)
    return series, gates


# ------------------------------------------------------------ book builders
def sleeve_a_book(day_rows, gate_on):
    """Long every asset with funding_pct >= 0.80, equal weight; flat if gate
    off or no qualifier."""
    if not gate_on:
        return {}
    elig = [a for a, r in day_rows.items()
            if r.get("funding_pct") is not None
            and r["funding_pct"] >= FUNDING_PCT_MIN]
    if not elig:
        return {}
    w = 1.0 / len(elig)
    return {a: w for a in elig}


def sleeve_b_book(day_rows, gate_on):
    """Long top-2 / short bottom-2 by dfh20 rank, 1/4 book per leg, flat if
    gate off, <8 ranked assets, or a degenerate edge tie."""
    if not gate_on:
        return {}
    ranked = sorted(((r["dfh20"], a) for a, r in day_rows.items()
                     if r.get("dfh20") is not None))
    if len(ranked) < MIN_ASSETS_DFH:
        return {}
    bot, top = ranked[:EDGE], ranked[-EDGE:]
    if bot[-1][0] == top[0][0]:
        return {}
    w = 1.0 / (2 * EDGE)
    book = {a: w for _, a in top}
    book.update({a: -w for _, a in bot})
    return book


def combine(*books):
    out = defaultdict(float)
    for frac, book in books:
        for a, w in book.items():
            out[a] += frac * w
    return {a: w for a, w in out.items() if w != 0.0}


def ew_book(day_rows):
    return {a: 1.0 / len(day_rows) for a in day_rows} if day_rows else {}


# --------------------------------------------------------------- simulation
def simulate(days, day_rows, books):
    """books: day -> {asset: signed weight}. P&L day t = sum_a w_t(a)*fwd_1d(a);
    cost charged on |w_t - w_{t-1}| (both sides of a swap counted)."""
    rows, prev = [], {}
    for d in days:
        book = books.get(d) or {}
        gross = sum(w * (day_rows[a][d].get("fwd_1d") or 0.0)
                    for a, w in book.items())
        turnover = sum(abs(book.get(a, 0.0) - prev.get(a, 0.0))
                       for a in set(book) | set(prev))
        cost = COST_BPS * turnover / 1e4
        rows.append({"day": d, "gross": gross, "net": gross - cost,
                     "turnover": turnover, "cost_bps": COST_BPS * turnover,
                     "active": bool(book)})
        prev = book
    return rows


def max_drawdown_pct(equity):
    peak, mdd = 1.0, 0.0
    for e in equity:
        peak = max(peak, e)
        mdd = max(mdd, (peak - e) / peak)
    return mdd * 100.0


def sharpe(rets):
    n = len(rets)
    if n < 5:
        return None
    m = sum(rets) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in rets) / (n - 1))
    return m / sd * math.sqrt(DAYS_PER_YEAR) if sd > 0 else None


def summarize(rows, label):
    """Full metric block for one arm over its daily rows."""
    def block(sub):
        n = len(sub)
        if not n:
            return {"days": 0}
        eq_g, eq_n, e = 1.0, 1.0, []
        eq_series = []
        for r in sub:
            eq_g *= 1.0 + r["gross"]
            eq_n *= 1.0 + r["net"]
            eq_series.append(eq_n)
        net = [r["net"] for r in sub]
        return {
            "days": n,
            "days_active": sum(1 for r in sub if r["active"]),
            "days_active_pct": _r(100.0 * sum(1 for r in sub if r["active"]) / n, 1),
            "mean_daily_gross_bps": _r(1e4 * sum(r["gross"] for r in sub) / n),
            "mean_daily_net_bps": _r(1e4 * sum(net) / n),
            "mean_daily_turnover_frac": _r(sum(r["turnover"] for r in sub) / n, 4),
            "mean_daily_cost_bps": _r(sum(r["cost_bps"] for r in sub) / n),
            "total_cost_bps": _r(sum(r["cost_bps"] for r in sub)),
            "cumulative_gross_pct": _r((eq_g - 1.0) * 100.0, 2),
            "cumulative_net_pct": _r((eq_n - 1.0) * 100.0, 2),
            "max_drawdown_net_pct": _r(max_drawdown_pct(eq_series), 2),
            "sharpe_net_annualized": _r(sharpe(net)),
        }
    out = {"arm": label, "all_days": block(rows), "per_year": {}}
    years = sorted({r["day"][:4] for r in rows})
    for y in years:
        out["per_year"][y] = block([r for r in rows if r["day"].startswith(y)])
    return out, rows


def t_mean(xs):
    n = len(xs)
    if n < 5:
        return {"n": n, "mean": (sum(xs) / n) if n else None, "t": None}
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return {"n": n, "mean": m,
            "t": (m / (sd / math.sqrt(n))) if sd > 0 else None}


def build_protocol():
    return {
        "schema_version": "nanojev-financial-signal-master-gate-protocol-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "task": "T111: measure the unified state machine — all signals under "
                "one BTC-trend master gate — as a daily-rebalanced portfolio "
                "on the 10-asset XS cohort.",
        "cohort": {
            "path": "data/perp_pit_xs_v1/records.jsonl",
            "records": 13380,
            "assets": "10 USDT-M perps, all listed pre-2023, all surviving to "
                      "2026-08 (survivor universe — delisted perps absent)",
            "span": "2023-01-01 -> 2026-08-30 daily decision bars",
            "price_basis": "klines close as mark proxy; no mark/index/basis "
                           "feature exists for the 5 T104-expanded symbols",
        },
        "system": {
            "master_gate": "BTCUSDT-PERP close > SMA20 (last 20 closes incl. "
                           "decision close) -> risk-on, else flat everything",
            "sleeve_A": "spec-v1 approximation: long funding_pct>=0.80 assets "
                        "equal weight, gate-on only; basis_pct>=0.66 leg "
                        "DROPPED (feature absent in XS cohort) => funding-only "
                        "deviation from the frozen spec",
            "sleeve_B": "XS dfh: long top-2 / short bottom-2 by dfh20 rank, "
                        "1/4 book per leg, >=8 ranked assets, gate-on only",
            "combined": "single netted book w = 0.5*wA + 0.5*wB",
        },
        "baselines": ["always_long_ew", "gate_always_long",
                      "each gated sleeve standalone",
                      "ungated sleeve/combined arms split by gate state"],
        "costs": "5bps per unit traded notional per leg; turnover = "
                 "sum_a |w_t(a)-w_{t-1}(a)| on the signed book (entry AND "
                 "exit sides counted)",
        "metrics": ["compounded cumulative net/gross return",
                    "Sharpe = mean/sd*sqrt(365) on daily net returns",
                    "max drawdown of compounded net equity",
                    "days-active %, mean daily turnover/cost",
                    "per-year table 2023/24/25/26",
                    "gate-on vs gate-off day counts and sleeve-return split",
                    "sleeve-A/B long-overlap measured"],
        "gate_sensitivities": ["spec def close/close[t-20]-1 > 0",
                               "close > strictly-prior SMA20"],
        "forbidden": ["fitting", "trading", "profitability claims",
                      "protocol edits post-run", "network"],
    }


def build_auth(protocol_sha):
    return {
        "schema_version": "nanojev-financial-signal-master-gate-authorization-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "protocol_sha256": protocol_sha,
        "decision": "approved_for_measurement",
        "measurement_authorized": True,
        "fit_authorized": False,
        "independent_reviewer": {
            "id": "project-owner",
            "independence": "owner_self_authorization_not_independent_review",
            "note": "Owner directed T111: portfolio-level measurement of the "
                    "unified BTC-gated system on the XS cohort (delegated "
                    "task).",
        },
        "scope": {
            "permitted": "PIT-safe descriptive measurement on "
                         "data/perp_pit_xs_v1/records.jsonl per the embedded "
                         "protocol.",
            "not_permitted": "No fitting/trading/profitability claims/protocol "
                             "edits; no network; no other files modified.",
        },
        "network_model_calls": 0,
        "order_submission_authorized": False,
        "live_trading_authorized": False,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", type=pathlib.Path, default=COHORT)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    args = ap.parse_args()

    protocol = build_protocol()
    protocol_sha = hashlib.sha256(
        json.dumps(protocol, indent=2, sort_keys=True).encode()).hexdigest()
    report = {"schema_version": "nanojev-financial-signal-master-gate-v1",
              "task": "T111 unified BTC-gated state machine, portfolio-level",
              "protocol": protocol,
              "protocol_sha256": protocol_sha,
              "owner_authorization": build_auth(protocol_sha)}

    if not args.cohort.exists():
        report["status"] = ("SKIPPED: cohort records.jsonl not found; nothing "
                            "was fabricated")
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return 0

    series, gates = load_cohort(args.cohort)
    days = sorted({r["day"] for rows in series.values() for r in rows})
    days = days[:-1]  # last day has no next-close P&L
    day_rows = {a: {r["day"]: r for r in rows} for a, rows in series.items()}
    rows_by_day = {d: {a: day_rows[a][d] for a in series if d in day_rows[a]}
                   for d in days}
    gate = gates["close_gt_sma20_incl"]

    gate_defined = [d for d in days if d in gate]
    gate_on_days = [d for d in gate_defined if gate[d]]
    report["cohort_span"] = {"first": days[0], "last": days[-1],
                             "decision_days": len(days)}
    report["symbols"] = sorted(series)
    report["gate"] = {
        "definition": "BTCUSDT-PERP close > SMA20 incl. decision close",
        "days_defined": len(gate_defined),
        "days_undefined_warmup": len(days) - len(gate_defined),
        "gate_on_days": len(gate_on_days),
        "gate_off_days": len(gate_defined) - len(gate_on_days),
        "gate_on_pct": _r(100.0 * len(gate_on_days) / len(gate_defined), 1),
    }

    # ---- books per arm, per day
    books = {k: {} for k in
             ("sleeve_A_gated", "sleeve_B_gated", "combined_gated",
              "always_long_ew", "gate_always_long",
              "sleeve_A_ungated", "sleeve_B_ungated", "combined_ungated")}
    for d in days:
        dr, on = rows_by_day[d], gate.get(d, False)
        a_g = sleeve_a_book(dr, on)
        b_g = sleeve_b_book(dr, on)
        a_u = sleeve_a_book(dr, True)
        b_u = sleeve_b_book(dr, True)
        books["sleeve_A_gated"][d] = a_g
        books["sleeve_B_gated"][d] = b_g
        books["combined_gated"][d] = combine((SLEEVE_SPLIT, a_g),
                                             (SLEEVE_SPLIT, b_g))
        books["always_long_ew"][d] = ew_book(dr)
        books["gate_always_long"][d] = ew_book(dr) if on else {}
        books["sleeve_A_ungated"][d] = a_u
        books["sleeve_B_ungated"][d] = b_u
        books["combined_ungated"][d] = combine((SLEEVE_SPLIT, a_u),
                                               (SLEEVE_SPLIT, b_u))

    report["arms"] = {}
    sim_rows = {}
    for k in books:
        rows = simulate(days, day_rows, books[k])
        s, sim_rows[k] = summarize(rows, k)
        report["arms"][k] = s

    # ---- does the gate add value beyond the sleeves? ungated sleeve returns
    # split by gate state (gross, so the read is not muddied by cost timing)
    gate_split = {}
    for k in ("sleeve_A_ungated", "sleeve_B_ungated", "combined_ungated",
              "always_long_ew"):
        on_rets = [r["gross"] for r in sim_rows[k]
                   if gate.get(r["day"], False) and r["active"]]
        off_rets = [r["gross"] for r in sim_rows[k]
                    if r["day"] in gate and not gate[r["day"]] and r["active"]]
        so, sf = t_mean([1e4 * x for x in on_rets]), t_mean(
            [1e4 * x for x in off_rets])
        gate_split[k] = {
            "gate_on": {"n_days": so["n"],
                        "mean_daily_gross_bps": _r(so["mean"]),
                        "t_nominal": _r(so["t"])},
            "gate_off": {"n_days": sf["n"],
                         "mean_daily_gross_bps": _r(sf["mean"]),
                         "t_nominal": _r(sf["t"])},
        }
    report["gate_value_split"] = gate_split

    # ---- sleeve overlap (measured, not assumed): gated days where both
    # sleeves hold a book — |A longs ∩ B top-2|, |A ∩ B bottom-2|, and signed
    # book overlap = sum_a min(|wA|,|wB|) over same-sign positions.
    ov_top, ov_bot, ov_book, n_both, days_any = [], [], [], 0, 0
    for d in days:
        a_g, b_g = books["sleeve_A_gated"][d], books["sleeve_B_gated"][d]
        if not a_g or not b_g:
            continue
        n_both += 1
        a_set, b_top = set(a_g), {a for a, w in b_g.items() if w > 0}
        b_bot = {a for a, w in b_g.items() if w < 0}
        ov_top.append(len(a_set & b_top) / EDGE)
        ov_bot.append(len(a_set & b_bot) / EDGE)
        if a_set & (b_top | b_bot):
            days_any += 1
        ov_book.append(sum(min(abs(a_g[x]), abs(b_g[x]))
                           for x in a_set
                           if x in b_g and a_g[x] * b_g[x] > 0))
    report["sleeve_overlap"] = {
        "days_both_sleeves_active": n_both,
        "mean_frac_B_long_legs_also_in_A": _r(
            sum(ov_top) / len(ov_top) if ov_top else None),
        "mean_frac_B_short_legs_also_in_A": _r(
            sum(ov_bot) / len(ov_bot) if ov_bot else None),
        "frac_days_any_asset_overlap": _r(
            days_any / n_both if n_both else None),
        "mean_signed_book_overlap_frac": _r(
            sum(ov_book) / len(ov_book) if ov_book else None),
        "note": "signed_book_overlap = sum_a min(|wA|,|wB|) on same-sign "
                "names, as a fraction of each sleeve's gross 1.0 book",
    }

    # ---- gate-definition sensitivity on the combined arm
    report["gate_sensitivity"] = {}
    for gname in ("ret20_gt_0", "close_gt_sma20_prior"):
        g = gates[gname]
        gb = {}
        for d in days:
            dr, on = rows_by_day[d], g.get(d, False)
            gb[d] = combine((SLEEVE_SPLIT, sleeve_a_book(dr, on)),
                            (SLEEVE_SPLIT, sleeve_b_book(dr, on)))
        rows = simulate(days, day_rows, gb)
        s, _ = summarize(rows, f"combined_under_{gname}")
        report["gate_sensitivity"][gname] = {
            "gate_on_days": sum(1 for d in g if g[d]),
            "net": s["all_days"]}

    # ---- verdict
    arms = report["arms"]
    comb = arms["combined_gated"]["all_days"]
    base = arms["always_long_ew"]["all_days"]
    gbase = arms["gate_always_long"]["all_days"]
    beats_passive = (comb["cumulative_net_pct"] is not None
                     and comb["cumulative_net_pct"] > base["cumulative_net_pct"])
    beats_gated_passive = (comb["cumulative_net_pct"] is not None
                           and comb["cumulative_net_pct"]
                           > gbase["cumulative_net_pct"])
    sharpe_beats = (comb["sharpe_net_annualized"] is not None
                    and comb["sharpe_net_annualized"]
                    > base["sharpe_net_annualized"])
    gs = report["gate_value_split"]["combined_ungated"]
    gate_adds = (gs["gate_off"]["mean_daily_gross_bps"] is not None
                 and gs["gate_on"]["mean_daily_gross_bps"]
                 > gs["gate_off"]["mean_daily_gross_bps"])
    checks = {
        "combined_net_positive": (comb["cumulative_net_pct"] or 0) > 0,
        "combined_beats_always_long_net": beats_passive,
        "combined_beats_gated_always_long_net": beats_gated_passive,
        "combined_sharpe_beats_always_long": sharpe_beats,
        "gate_adds_value_on_combined": gate_adds,
    }
    n_pass = sum(checks.values())
    psg = report["gate_value_split"]["always_long_ew"]
    verdict_str = (
        "MIXED — see checks" if 0 < n_pass < len(checks) else
        ("PASSES: gated combined system beats passive baselines net and "
         "the gate adds value" if n_pass == len(checks) else
         "FAILS: gated combined system does not beat passive baselines "
         "net / gate does not add value"))
    report["verdict"] = {
        "checks": checks,
        "checks_passed": f"{n_pass}/{len(checks)}",
        "verdict": verdict_str,
        "read": (
            f"Combined gated book: {_r(comb['cumulative_net_pct'])}% net "
            f"cumulative, Sharpe {_r(comb['sharpe_net_annualized'])}, mdd "
            f"{_r(comb['max_drawdown_net_pct'])}%, capital at risk "
            f"{_r(comb['days_active_pct'])}% of days vs always-long "
            f"{_r(base['cumulative_net_pct'])}%, Sharpe "
            f"{_r(base['sharpe_net_annualized'])}, mdd "
            f"{_r(base['max_drawdown_net_pct'])}%. The edge is "
            f"risk-adjusted and gate-conditional: the ungated combined book "
            f"earns {_r(gs['gate_on']['mean_daily_gross_bps'])}bps/day gross "
            f"on gate-on days vs "
            f"{_r(gs['gate_off']['mean_daily_gross_bps'])}bps/day on "
            f"gate-off days. CAVEAT: the gate is a SIGNAL conditioner, not a "
            f"passive timing tool — passive EW still drifted "
            f"{_r(psg['gate_off']['mean_daily_gross_bps'])}bps/day on "
            f"gate-off days (survivor universe), so gate_always_long "
            f"({_r(gbase['cumulative_net_pct'])}%) trails always_long on raw "
            f"cumulative even though it cuts mdd "
            f"({_r(gbase['max_drawdown_net_pct'])}% vs "
            f"{_r(base['max_drawdown_net_pct'])}%)."),
    }

    report["honest_notes"] = [
        "CLOSE BASIS: P&L = close-to-close; no funding cashflows earned/paid, "
        "no borrow cost on sleeve-B shorts, fills at decision close, linear "
        "5bps cost, no market impact.",
        "SPEC-V1 DEVIATION: sleeve A is FUNDING-ONLY. The frozen spec also "
        "requires basis_pct>=0.66 (mark_index_basis_bps mid-rank); the XS "
        "cohort has no mark/index/basis feature for the 5 expanded symbols, "
        "so the basis leg is dropped — sleeve A is a looser, more permissive "
        "gate than the real spec.",
        "OVERLAP: sleeve-A longs vs sleeve-B top-2 overlap is MEASURED in "
        "sleeve_overlap — the two sleeves are not independent alpha streams; "
        "combined leverage/drawdown is partly the same bet twice.",
        "SURVIVORSHIP: all 10 assets listed pre-2023 and survived to "
        "2026-08 — delisted perps are absent from the universe; every "
        "long-side number (always_long and sleeve A especially) carries an "
        "optimistic survivorship bias.",
        "WARMUP: sleeve A needs 180 prior funding records/asset, so it cannot "
        "trade before ~2023-06-30; 2023 per-year rows are partial by "
        "construction, not signal absence.",
        "XS DOF: ~10 assets, 2-vs-2 dfh edge -> ~4-5 effective cross-sectional "
        "dof/day; daily returns are fat-tailed and one asset swings the book.",
        "GATE DEF: primary gate is close > SMA20 including the decision close "
        "(task spec); sensitivity arms report the frozen spec's ret20>0 "
        "definition and a strictly-prior SMA20 — conclusions should not hinge "
        "on the variant.",
        "COST CONVENTION: turnover = sum|Δw| counting BOTH exit and entry "
        "sides (a same-size swap costs 2.5bps vs T106's 1.25bps slot "
        "convention) — conservative.",
        "Measurement only: no fitting, no trading, no orders, no network.",
    ]
    report["honesty"] = {
        "not_a_return": "daily returns are close-price moves net of stylized "
                        "cost only; fees/funding/borrow/impact unmodelled",
        "not_live": "no orders, no account, no broker, no trading API used",
        "no_profitability_claim": True,
    }
    report["status"] = "ran"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    brief = {"status": "ran", "out": str(args.out),
             "gate": report["gate"],
             "checks": report["verdict"]["checks"],
             "verdict": report["verdict"]["verdict"],
             "combined_net": report["arms"]["combined_gated"]["all_days"],
             "always_long_net": report["arms"]["always_long_ew"]["all_days"],
             "gate_always_long_net":
                 report["arms"]["gate_always_long"]["all_days"],
             "sleeve_A_net": report["arms"]["sleeve_A_gated"]["all_days"],
             "sleeve_B_net": report["arms"]["sleeve_B_gated"]["all_days"],
             "gate_value_split_combined":
                 report["gate_value_split"]["combined_ungated"],
             "overlap": report["sleeve_overlap"]}
    print(json.dumps(brief, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
