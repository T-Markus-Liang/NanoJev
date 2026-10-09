#!/usr/bin/env python3
"""T134: the regime-switching MIRROR BOOK — monetization test of T132's
symmetric structure at mega scale.

T132 (``financial_signal_reversal_mega_v1``) found the mirror: daily
reversal lives gate-OFF (pooled rho -0.076 gate-off vs +0.006 gate-on),
while dfh/momentum lives gate-ON (+68.9bps XS decile spread gate-on).
Both halves were measured separately; nobody has combined them into one
switching book. This script runs the honest daily-rebalanced portfolio
simulation on ``data/perp_pit_mega_v1/records.jsonl`` (277 symbols,
2021-01 -> 2025-12, close-as-mark, fwd1d label) with the BTC ret20 gate
computed from ``data/rc_futures_v1/BTC/BTCUSDT_1d.csv``
(gate-on = BTC close/close[t-20]-1 > 0; T132 gate definition, external
BTC series rather than the in-cohort BTCUSDT-PERP rows).

Books (signed weights as a fraction of the arm's capital; LS books are
dollar-neutral with gross 1.0 = 0.5 long + 0.5 short):

  ARM M   momentum/dfh book, gate-ON days only: per decision day with
          >=30 ranked assets, rank by dfh20, long top-decile (nearest the
          20d high = strongest) / short bottom-decile, equal weight.
          Flat on gate-off and on gate-undefined warmup days.
  ARM R   reversal book, gate-OFF days only: rank by ret1d
          (close/prev-calendar-day close - 1, contiguous bars only), long
          bottom-decile yesterday-losers / short top-decile
          yesterday-winners — fade the move. Flat on gate-on/undefined.
  ARM MR  the mirror: M book when gate-on, R book when gate-off, flat
          while the gate is undefined. Regime switches unwind one book
          and establish the other — turnover charges capture the full
          cost of the flip.

Baselines:
  always_ls_dfh        the dfh decile LS book every day, no gate (what
                       the gate is compared against on the momentum side)
  always_reversal_ls   the reversal book every day, no gate (lets the
                       gate-off concentration be re-verified at book
                       level, and shows what gating removes)
  always_ew_long       EW long every asset with a fwd1 label, every day
  gate_ew_long         EW long on gate-on days, flat otherwise — the
                       simple "gate-on long universe / gate-off flat"
                       timing alternative MR must beat to justify the LS
                       machinery

Costs: 5bps per unit of traded notional per leg, charged on the signed
book change: cost_t = 5bps * sum_a |w_t(a) - w_{t-1}(a)| — exit AND entry
sides both counted (T111 convention; establishing or fully unwinding the
gross-1.0 book costs 5bps, a same-size leg swap costs 2.5bps).

Metrics per arm: net daily return series -> compounded cumulative
return, annualized Sharpe (mean/sd*sqrt(365), crypto trades daily), max
drawdown of the compounded net equity, occupancy (days-active %), mean
daily turnover/cost, per-year table 2021-2025.

Key comparisons carried in the report:
  * MR vs M-alone vs R-alone — does adding the gate-off reversal sleeve
    help net of its churn, or is R too weak after costs?
  * MR vs gate_ew_long — does the LS mirror beat trivial gate timing on
    the passive universe?
  * ungated-book gross splits by gate state — the T132 mirror re-verified
    at the book level (dfh spread gate-on vs gate-off; reversal spread
    gate-off vs gate-on).

Honesty: T132 showed the reversal rho is a TENDENCY, not a tail edge —
the ungated decile reversal spread was ~0 gross (mean +0.18bps/day) and
-3.9bps/day net of churn. The R book may well be flat-to-negative net
even in its own regime; the truthful verdict may be "mirror confirmed
descriptively, only the momentum side is monetizable". Survivorship:
early-stopped symbols are present (11 stop mid-sample) but assets listed
after 2021 enter late and the universe is built from symbols that have a
rc_futures archive at all. Close basis only: no funding cashflows, no
borrow on shorts, fills at decision close, linear cost, no market impact.

Protocol + owner self-authorization are embedded in the single output
artifact ``results/financial_signal_mirror_book_v1.json`` (protocol
object hashed by sha256, authorization pins the hash) — same convention
as T111. Measurement only: no fitting, no trading, no network.
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import pathlib
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
COHORT = ROOT / "data/perp_pit_mega_v1/records.jsonl"
BTC_CSV = ROOT / "data/rc_futures_v1/BTC/BTCUSDT_1d.csv"
OUT = ROOT / "results/financial_signal_mirror_book_v1.json"

MIN_ASSETS_PER_DAY = 30   # mega XS convention
DECILE = 10               # edge k = max(1, n//10) ranked assets per side
BTC_GATE_LOOKBACK = 20    # gate-on = BTC close/close[t-20]-1 > 0
COST_BPS = 5.0            # per unit traded notional, per leg (both sides)
DAYS_PER_YEAR = 365       # crypto trades daily


def _r(x, nd=3):
    return round(x, nd) if isinstance(x, float) else x


# ------------------------------------------------------------------ loading
def load_cohort(path):
    """asset -> sorted rows {day, ord, close, dfh20, ret1d, fwd1}. ret1d
    requires the previous record to be the previous CALENDAR day (no
    cross-gap returns, T132 convention). fwd1 is the strict next-day
    close-to-close label, converted from bps to a fraction."""
    series = defaultdict(list)
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            rec = json.loads(line)
            f = rec["features"]
            series[rec["asset_id"]].append({
                "day": rec["id"].rsplit(":", 1)[-1],
                "close": f["close"]["value"],
                "dfh20": f["dfh20"]["value"],
                "fwd1": (rec["label"]["forward_return_bps"] / 1e4
                         if rec["label"].get("forward_return_bps") is not None
                         else None),
            })
    for rows in series.values():
        rows.sort(key=lambda r: r["day"])
        closes = [r["close"] for r in rows]
        for i, r in enumerate(rows):
            r["ord"] = dt.date.fromisoformat(r["day"]).toordinal()
            ret = None
            if i > 0 and r["ord"] == rows[i - 1]["ord"] + 1 \
                    and closes[i - 1] and closes[i - 1] > 0 \
                    and r["close"] and r["close"] > 0:
                ret = r["close"] / closes[i - 1] - 1.0
            r["ret1d"] = ret
    return series


def load_btc_gate(path):
    """BTC ret20 gate from the external daily CSV: day -> ret20 (only when
    the row 20 bars back is exactly 20 calendar days back and positive),
    plus day -> gate_on bool (ret20 > 0). Undefined during warmup/gaps."""
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        for rec in csv.DictReader(stream):
            rows.append({"day": rec["timestamp"],
                         "ord": dt.date.fromisoformat(
                             rec["timestamp"]).toordinal(),
                         "close": float(rec["close"])})
    rows.sort(key=lambda r: r["day"])
    ret20, gate = {}, {}
    for i, r in enumerate(rows):
        j = i - BTC_GATE_LOOKBACK
        if j >= 0 and rows[j]["close"] > 0 \
                and r["ord"] - rows[j]["ord"] == BTC_GATE_LOOKBACK:
            ret20[r["day"]] = r["close"] / rows[j]["close"] - 1.0
            gate[r["day"]] = ret20[r["day"]] > 0.0
    return ret20, gate, rows


# ------------------------------------------------------------ book builders
def decile_book(rows_by_asset, key, long_top):
    """Per-day decile LS book on `key` rank. long_top=True: long top-decile
    / short bottom-decile (momentum on dfh20). long_top=False: long
    bottom-decile / short top-decile (reversal on ret1d). Dollar-neutral,
    gross 1.0, EW inside each leg. Requires >=MIN_ASSETS_PER_DAY ranked
    assets with a valid fwd1 label and a non-degenerate edge."""
    ranked = sorted((r[key], a) for a, r in rows_by_asset.items()
                    if r.get(key) is not None and r.get("fwd1") is not None)
    n = len(ranked)
    if n < MIN_ASSETS_PER_DAY:
        return {}
    k = max(1, n // DECILE)
    bot, top = ranked[:k], ranked[-k:]
    if bot[-1][0] == top[0][0]:
        return {}  # degenerate day: no dispersion at the ranked edges
    w = 0.5 / k
    if long_top:
        book = {a: w for _, a in top}
        book.update({a: -w for _, a in bot})
    else:
        book = {a: w for _, a in bot}
        book.update({a: -w for _, a in top})
    return book


def ew_book(rows_by_asset):
    elig = [a for a, r in rows_by_asset.items()
            if r.get("fwd1") is not None]
    return {a: 1.0 / len(elig) for a in elig} if elig else {}


# --------------------------------------------------------------- simulation
def simulate(days, day_rows, books):
    """books: day -> {asset: signed weight}. P&L day t =
    sum_a w_t(a)*fwd1(a); cost = COST_BPS * sum|w_t - w_{t-1}| (both the
    exit and entry side of every swap are charged)."""
    rows, prev = [], {}
    for d in days:
        book = books.get(d) or {}
        gross = sum(w * day_rows[a][d]["fwd1"] for a, w in book.items()
                    if day_rows[a][d].get("fwd1") is not None)
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


def t_mean(xs):
    n = len(xs)
    if n < 5:
        return {"n": n, "mean": (sum(xs) / n) if n else None, "t": None}
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return {"n": n, "mean": m,
            "t": (m / (sd / math.sqrt(n))) if sd > 0 else None}


def summarize(rows, label):
    """Full metric block for one arm over its daily rows (all decision
    days, including flat days — occupancy is part of the honest read)."""
    def block(sub):
        n = len(sub)
        if not n:
            return {"days": 0}
        eq_g, eq_n = 1.0, 1.0
        eq_series = []
        for r in sub:
            eq_g *= 1.0 + r["gross"]
            eq_n *= 1.0 + r["net"]
            eq_series.append(eq_n)
        net = [r["net"] for r in sub]
        gross = [r["gross"] for r in sub]
        active = [r for r in sub if r["active"]]
        return {
            "days": n,
            "days_active": len(active),
            "days_active_pct": _r(100.0 * len(active) / n, 1),
            "mean_daily_gross_bps": _r(1e4 * sum(gross) / n),
            "mean_daily_net_bps": _r(1e4 * sum(net) / n),
            "mean_active_day_gross_bps":
                _r(1e4 * sum(r["gross"] for r in active) / len(active))
                if active else None,
            "mean_active_day_net_bps":
                _r(1e4 * sum(r["net"] for r in active) / len(active))
                if active else None,
            "mean_daily_turnover_frac": _r(
                sum(r["turnover"] for r in sub) / n, 4),
            "mean_daily_cost_bps": _r(sum(r["cost_bps"] for r in sub) / n),
            "total_cost_bps": _r(sum(r["cost_bps"] for r in sub)),
            "cumulative_gross_pct": _r((eq_g - 1.0) * 100.0, 2),
            "cumulative_net_pct": _r((eq_n - 1.0) * 100.0, 2),
            "max_drawdown_net_pct": _r(max_drawdown_pct(eq_series), 2),
            "sharpe_gross_annualized": _r(sharpe(gross)),
            "sharpe_net_annualized": _r(sharpe(net)),
        }
    out = {"arm": label, "all_days": block(rows), "per_year": {}}
    for y in sorted({r["day"][:4] for r in rows}):
        out["per_year"][y] = block([r for r in rows
                                    if r["day"].startswith(y)])
    return out


def build_protocol():
    return {
        "schema_version": "nanojev-financial-signal-mirror-book-protocol-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "task": "T134: monetization test of T132's mirror structure — one "
                "daily-rebalanced book that holds a dfh decile LS on "
                "BTC-ret20 gate-on days and a ret1d reversal decile LS on "
                "gate-off days, vs standalone arms and passive baselines.",
        "cohort": {
            "path": "data/perp_pit_mega_v1/records.jsonl",
            "assets": "277+ USDT-M perp symbols (dirs_with_1d=301; assets "
                      "join/leave the ranked XS as their series allow; 11 "
                      "early-stopped symbols retained)",
            "span": "2021-01-01 -> 2025-12-31 daily decision bars; last "
                    "per-asset day has no fwd1 label",
            "price_basis": "klines close as mark proxy; label = strict "
                           "next-day close-to-close gross move",
        },
        "gate": {
            "source": "data/rc_futures_v1/BTC/BTCUSDT_1d.csv (external BTC "
                      "series, not in-cohort BTCUSDT-PERP)",
            "definition": "gate-on = BTC close/close[t-20]-1 > 0, requiring "
                          "the row 20 bars back to be exactly 20 calendar "
                          "days back; warmup/gap days are gate-UNDEFINED "
                          "and every gated arm is flat on them",
        },
        "arms": {
            "arm_M_momentum_gate_on": "dfh20 decile LS (long top-decile "
                                      "near-20d-high / short bottom-decile, "
                                      "EW, gross 1.0, >=30 ranked assets) "
                                      "on gate-on days only",
            "arm_R_reversal_gate_off": "ret1d decile LS (long bottom-decile "
                                       "yesterday-losers / short top-decile "
                                       "winners, EW, gross 1.0) on "
                                       "gate-off days only",
            "arm_MR_mirror": "M book on gate-on days, R book on gate-off "
                             "days; regime switches pay full unwind + "
                             "re-establish turnover",
            "always_ls_dfh": "dfh decile LS every day (ungated momentum)",
            "always_reversal_ls": "ret1d reversal decile LS every day "
                                  "(ungated; lets the gate-off reversal "
                                  "concentration be re-verified at book "
                                  "level)",
            "always_ew_long": "EW long all assets with a fwd1 label, daily",
            "gate_ew_long": "EW long universe on gate-on days, flat "
                            "otherwise — trivial gate-timing alternative",
        },
        "costs": "5bps per unit traded notional per leg; turnover = "
                 "sum_a |w_t(a)-w_{t-1}(a)| on the signed book (exit AND "
                 "entry sides both charged — a full regime switch costs "
                 "~10bps round trip)",
        "metrics": ["compounded cumulative net/gross return",
                    "Sharpe = mean/sd*sqrt(365) on daily net returns",
                    "max drawdown of compounded net equity",
                    "occupancy = days-active %, mean daily turnover/cost",
                    "per-year table 2021-2025",
                    "ungated-book gross splits by gate state",
                    "MR-minus-M daily-net difference (the isolated R "
                    "contribution)"],
        "priors_from_T132": "reversal pooled rho -0.076 gate-off vs +0.006 "
                            "gate-on; dfh XS decile spread +68.9bps gate-on; "
                            "ungated decile reversal ~0 gross / -3.9bps/day "
                            "net — R may be flat-to-negative net",
        "forbidden": ["fitting", "trading", "profitability claims",
                      "protocol edits post-run", "network"],
    }


def build_auth(protocol_sha):
    return {
        "schema_version":
            "nanojev-financial-signal-mirror-book-authorization-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "protocol_sha256": protocol_sha,
        "decision": "approved_for_measurement",
        "measurement_authorized": True,
        "fit_authorized": False,
        "independent_reviewer": {
            "id": "project-owner",
            "independence": "owner_self_authorization_not_independent_review",
            "note": "Owner directed T134: portfolio-level monetization test "
                    "of the T132 mirror structure on the mega cohort "
                    "(delegated task).",
        },
        "scope": {
            "permitted": "PIT-safe descriptive measurement on "
                         "data/perp_pit_mega_v1/records.jsonl with the BTC "
                         "ret20 gate from data/rc_futures_v1/BTC/"
                         "BTCUSDT_1d.csv per the embedded protocol.",
            "not_permitted": "No fitting/trading/profitability claims/"
                             "protocol edits; no network; no other files "
                             "modified.",
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
    ap.add_argument("--btc-csv", type=pathlib.Path, default=BTC_CSV)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    args = ap.parse_args()

    protocol = build_protocol()
    protocol_sha = hashlib.sha256(
        json.dumps(protocol, indent=2, sort_keys=True).encode()).hexdigest()
    report = {"schema_version": "nanojev-financial-signal-mirror-book-v1",
              "task": "T134 mirror book: momentum gate-on + reversal "
                      "gate-off, portfolio-level, mega cohort",
              "protocol": protocol,
              "protocol_sha256": protocol_sha,
              "owner_authorization": build_auth(protocol_sha)}

    if not args.cohort.exists() or not args.btc_csv.exists():
        report["status"] = ("SKIPPED: cohort records.jsonl or BTC csv not "
                            "found; nothing was fabricated")
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True)
                            + "\n", encoding="utf-8")
        return 0

    series = load_cohort(args.cohort)
    btc_ret20, gate, btc_rows = load_btc_gate(args.btc_csv)
    days = sorted({r["day"] for rows in series.values() for r in rows})
    days = days[:-1]  # last calendar day can have no complete fwd1 labels
    day_rows = {a: {r["day"]: r for r in rows} for a, rows in series.items()}
    rows_by_day = {d: {a: day_rows[a][d] for a in series if d in day_rows[a]}
                   for d in days}

    gate_defined = [d for d in days if d in gate]
    gate_on_days = [d for d in gate_defined if gate[d]]
    gate_off_days = [d for d in gate_defined if not gate[d]]
    report["cohort_span"] = {"first": days[0], "last": days[-1],
                             "decision_days": len(days),
                             "assets": len(series)}
    report["gate"] = {
        "definition": "BTC ret20 > 0 from "
                      "data/rc_futures_v1/BTC/BTCUSDT_1d.csv "
                      "(contiguous-20-calendar-day requirement)",
        "btc_csv_rows": len(btc_rows),
        "btc_first_day": btc_rows[0]["day"] if btc_rows else None,
        "btc_last_day": btc_rows[-1]["day"] if btc_rows else None,
        "days_defined": len(gate_defined),
        "days_undefined_warmup_or_gap": len(days) - len(gate_defined),
        "gate_on_days": len(gate_on_days),
        "gate_off_days": len(gate_off_days),
        "gate_on_pct_of_defined": _r(
            100.0 * len(gate_on_days) / len(gate_defined), 1)
            if gate_defined else None,
        "ret20_mean_gate_on": _r(
            sum(btc_ret20[d] for d in gate_on_days) / len(gate_on_days), 4)
            if gate_on_days else None,
        "ret20_mean_gate_off": _r(
            sum(btc_ret20[d] for d in gate_off_days) / len(gate_off_days), 4)
            if gate_off_days else None,
        "gate_switches": sum(
            1 for a, b in zip(gate_defined, gate_defined[1:])
            if gate[a] != gate[b]),
    }

    # ---- books per arm, per day (undefined gate -> flat for gated arms)
    arm_names = ("arm_M_momentum_gate_on", "arm_R_reversal_gate_off",
                 "arm_MR_mirror", "always_ls_dfh", "always_reversal_ls",
                 "always_ew_long", "gate_ew_long")
    books = {k: {} for k in arm_names}
    for d in days:
        dr = rows_by_day[d]
        state = gate.get(d)          # True on / False off / None undefined
        m_book = decile_book(dr, "dfh20", long_top=True) if state else {}
        r_book = (decile_book(dr, "ret1d", long_top=False)
                  if state is False else {})
        books["arm_M_momentum_gate_on"][d] = m_book
        books["arm_R_reversal_gate_off"][d] = r_book
        books["arm_MR_mirror"][d] = m_book if state else r_book
        books["always_ls_dfh"][d] = decile_book(dr, "dfh20", long_top=True)
        books["always_reversal_ls"][d] = decile_book(dr, "ret1d",
                                                    long_top=False)
        books["always_ew_long"][d] = ew_book(dr)
        books["gate_ew_long"][d] = ew_book(dr) if state else {}

    report["arms"] = {}
    sim_rows = {}
    for k in arm_names:
        rows = simulate(days, day_rows, books[k])
        sim_rows[k] = rows
        report["arms"][k] = summarize(rows, k)

    # ---- the mirror, re-verified at book level: ungated-book gross split
    # by gate state (gross so cost timing does not muddy the read)
    gate_split = {}
    for k in ("always_ls_dfh", "always_reversal_ls", "always_ew_long"):
        on = [r["gross"] for r in sim_rows[k]
              if gate.get(r["day"]) is True and r["active"]]
        off = [r["gross"] for r in sim_rows[k]
               if gate.get(r["day"]) is False and r["active"]]
        so, sf = t_mean([1e4 * x for x in on]), t_mean([1e4 * x for x in off])
        gate_split[k] = {
            "gate_on": {"n_days": so["n"],
                        "mean_daily_gross_bps": _r(so["mean"]),
                        "t_nominal": _r(so["t"])},
            "gate_off": {"n_days": sf["n"],
                         "mean_daily_gross_bps": _r(sf["mean"]),
                         "t_nominal": _r(sf["t"])},
        }
    report["book_level_mirror_check"] = {
        "definition": "ungated book gross returns split by gate state; "
                      "momentum should be gate-on, reversal gate-off "
                      "(T132 mirror at the traded-book level)",
        "split": gate_split,
    }

    # ---- key comparisons
    a = report["arms"]
    m, r_, mr = (a["arm_M_momentum_gate_on"]["all_days"],
                 a["arm_R_reversal_gate_off"]["all_days"],
                 a["arm_MR_mirror"]["all_days"])
    diff = [x["net"] - y["net"] for x, y in
            zip(sim_rows["arm_MR_mirror"], sim_rows["arm_M_momentum_gate_on"])]
    dm = t_mean([1e4 * x for x in diff])
    report["key_comparisons"] = {
        "mr_minus_m_daily_net": {
            "definition": "daily net(MR) - net(M); nonzero only on days "
                          "where MR holds the R book (or pays the switch) "
                          "while M is flat — the isolated gate-off "
                          "reversal contribution",
            "n_days": dm["n"],
            "mean_bps": _r(dm["mean"]),
            "t_nominal": _r(dm["t"]),
            "total_bps_added": _r(1e4 * sum(diff)),
        },
        "mr_vs_m": {
            "net_sharpe": {"M": _r(m["sharpe_net_annualized"]),
                           "MR": _r(mr["sharpe_net_annualized"])},
            "cumulative_net_pct": {"M": _r(m["cumulative_net_pct"]),
                                   "MR": _r(mr["cumulative_net_pct"])},
            "max_drawdown_net_pct": {"M": _r(m["max_drawdown_net_pct"]),
                                     "MR": _r(mr["max_drawdown_net_pct"])},
            "occupancy_pct": {"M": _r(m["days_active_pct"]),
                              "MR": _r(mr["days_active_pct"])},
        },
        "r_standalone": {
            "net_sharpe": _r(r_["sharpe_net_annualized"]),
            "cumulative_net_pct": _r(r_["cumulative_net_pct"]),
            "mean_active_day_net_bps": _r(r_["mean_active_day_net_bps"]),
            "mean_active_day_gross_bps": _r(r_["mean_active_day_gross_bps"]),
            "mean_daily_cost_bps": _r(r_["mean_daily_cost_bps"]),
        },
        "mr_vs_gate_ew_long": {
            "definition": "does the LS mirror beat trivial gate timing "
                          "(EW long universe on gate-on, flat off)?",
            "net_sharpe": {"MR": _r(mr["sharpe_net_annualized"]),
                           "gate_ew_long": _r(
                               a["gate_ew_long"]["all_days"]
                               ["sharpe_net_annualized"])},
            "cumulative_net_pct": {"MR": _r(mr["cumulative_net_pct"]),
                                   "gate_ew_long": _r(
                                       a["gate_ew_long"]["all_days"]
                                       ["cumulative_net_pct"])},
            "max_drawdown_net_pct": {"MR": _r(mr["max_drawdown_net_pct"]),
                                     "gate_ew_long": _r(
                                         a["gate_ew_long"]["all_days"]
                                         ["max_drawdown_net_pct"])},
        },
        "gating_vs_ungated": {
            "dfh_book": {"always": _r(a["always_ls_dfh"]["all_days"]
                                      ["sharpe_net_annualized"]),
                         "gated_M": _r(m["sharpe_net_annualized"])},
            "reversal_book": {"always": _r(
                a["always_reversal_ls"]["all_days"]["sharpe_net_annualized"]),
                "gated_R": _r(r_["sharpe_net_annualized"])},
            "note": "Sharpe of the same book ungated vs restricted to its "
                    "T132 regime — did the gate actually isolate the edge?",
        },
    }

    # ---- verdict
    def num(x):
        return x if isinstance(x, (int, float)) else float("-inf")
    checks = {
        "momentum_side_net_positive": num(m["cumulative_net_pct"]) > 0,
        "momentum_beats_ungated_dfh_sharpe":
            num(m["sharpe_net_annualized"])
            > num(a["always_ls_dfh"]["all_days"]["sharpe_net_annualized"]),
        "reversal_side_net_positive": num(r_["cumulative_net_pct"]) > 0,
        "mirror_beats_m_alone_net":
            num(mr["cumulative_net_pct"]) > num(m["cumulative_net_pct"]),
        "mirror_sharpe_ge_m":
            num(mr["sharpe_net_annualized"])
            >= num(m["sharpe_net_annualized"]),
        "mirror_beats_gate_ew_long_net":
            num(mr["cumulative_net_pct"])
            > num(a["gate_ew_long"]["all_days"]["cumulative_net_pct"]),
        "mirror_structure_descriptive": (
            gate_split["always_ls_dfh"]["gate_on"]["mean_daily_gross_bps"]
            is not None
            and gate_split["always_reversal_ls"]["gate_off"]
            ["mean_daily_gross_bps"] is not None
            and gate_split["always_ls_dfh"]["gate_on"]
            ["mean_daily_gross_bps"]
            > gate_split["always_ls_dfh"]["gate_off"]["mean_daily_gross_bps"]
            and gate_split["always_reversal_ls"]["gate_off"]
            ["mean_daily_gross_bps"]
            > gate_split["always_reversal_ls"]["gate_on"]
            ["mean_daily_gross_bps"]),
    }
    n_pass = sum(checks.values())
    if checks["reversal_side_net_positive"] \
            and checks["mirror_sharpe_ge_m"]:
        verdict_str = ("MIRROR MONETIZABLE: the gate-off reversal sleeve "
                       "adds net value to the gate-on momentum book")
    elif checks["momentum_side_net_positive"] \
            and checks["mirror_structure_descriptive"]:
        verdict_str = ("MIRROR CONFIRMED DESCRIPTIVELY, ONLY THE MOMENTUM "
                       "SIDE MONETIZABLE: reversal concentrates gate-off "
                       "but is too weak/expensive to add net value")
    elif checks["momentum_side_net_positive"]:
        verdict_str = ("ONLY THE MOMENTUM BOOK WORKS; the mirror structure "
                       "did not even replicate descriptively at book level")
    else:
        verdict_str = ("FAILS: neither side monetizes net under this "
                       "protocol")
    report["verdict"] = {
        "checks": checks,
        "checks_passed": f"{n_pass}/{len(checks)}",
        "verdict": verdict_str,
        "read": (
            f"M (gate-on dfh LS): {_r(m['cumulative_net_pct'])}% net cum, "
            f"Sharpe {_r(m['sharpe_net_annualized'])}, mdd "
            f"{_r(m['max_drawdown_net_pct'])}%, active "
            f"{_r(m['days_active_pct'])}% of days. R (gate-off reversal "
            f"LS): {_r(r_['cumulative_net_pct'])}% net cum, Sharpe "
            f"{_r(r_['sharpe_net_annualized'])}, mean active-day "
            f"{_r(r_['mean_active_day_net_bps'])}bps net. MR mirror: "
            f"{_r(mr['cumulative_net_pct'])}% net cum, Sharpe "
            f"{_r(mr['sharpe_net_annualized'])}, mdd "
            f"{_r(mr['max_drawdown_net_pct'])}%. Isolated R contribution "
            f"(MR-M daily net): {_r(dm['mean'])}bps/day, total "
            f"{_r(1e4 * sum(diff))}bps. gate_ew_long baseline: "
            f"{_r(a['gate_ew_long']['all_days']['cumulative_net_pct'])}% "
            f"net cum, Sharpe "
            f"{_r(a['gate_ew_long']['all_days']['sharpe_net_annualized'])}."),
    }

    report["honest_notes"] = [
        "CLOSE BASIS: P&L = close-to-close label returns; no funding "
        "cashflows earned/paid, no borrow cost on the short legs, fills at "
        "decision close, linear 5bps cost, no market impact.",
        "REVERSAL IS A TENDENCY, NOT A TAIL EDGE: T132 measured the gate-off "
        "pooled rho at -0.076 but the ungated decile reversal spread at "
        "~+0.18bps/day gross (-3.9bps/day net). The R book's job here is to "
        "test whether restricting to gate-off days rescues it — a negative "
        "result is the expected honest outcome, not a bug.",
        "SWITCH COSTS ARE REAL: MR pays a full unwind + re-establish "
        "(~10bps) on every gate flip; gate_switches is reported. A choppy "
        "gate taxes the mirror specifically.",
        "DECILE DEFINITION: k = max(1, n//10) of the day's ranked assets "
        "(n ranges ~30->250 as listings grow); edge size grows with the "
        "universe — not a fixed-K book.",
        "UNIVERSE DYNAMICS: symbols enter on listing and 11 early-stopped "
        "symbols leave mid-sample (renames/delistings per build_summary); "
        "the EW baselines and decile ranks use whatever is listed that day.",
        "WARMUP: BTC ret20 needs 20 contiguous prior BTC daily bars; the "
        "first ~20 cohort days are gate-undefined and all gated arms sit "
        "flat — reported as days_undefined_warmup_or_gap, not silently "
        "treated as gate-off.",
        "GATE SOURCE: the gate uses the external BTCUSDT 1d csv (same "
        "underlying as the cohort's BTCUSDT-PERP rows but independently "
        "loaded, per task spec); the two series can disagree by a day on "
        "gaps.",
        "OCCUPANCY IS PART OF THE READ: gated arms are flat most days; "
        "per-calendar-day mean bps and days_active_pct are both reported "
        "so idle-capital opportunity cost is visible.",
        "Measurement only: no fitting, no trading, no orders, no network.",
    ]
    report["honesty"] = {
        "not_a_return": "daily returns are close-price label moves net of "
                        "stylized 5bps/leg cost only; fees/funding/borrow/"
                        "impact unmodelled",
        "not_live": "no orders, no account, no broker, no trading API used",
        "no_profitability_claim": True,
    }
    report["status"] = "ran"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    brief = {"status": "ran", "out": str(args.out),
             "gate": report["gate"],
             "checks": checks,
             "verdict": report["verdict"]["verdict"],
             "read": report["verdict"]["read"],
             "M": m, "R": r_, "MR": mr,
             "always_ls_dfh": a["always_ls_dfh"]["all_days"],
             "always_reversal_ls": a["always_reversal_ls"]["all_days"],
             "always_ew_long": a["always_ew_long"]["all_days"],
             "gate_ew_long": a["gate_ew_long"]["all_days"],
             "book_level_mirror": gate_split,
             "key_comparisons": report["key_comparisons"]}
    print(json.dumps(brief, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
