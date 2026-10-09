#!/usr/bin/env python3
"""T118: EXIT STATES — when to stop rolling and wait — on the mega cohort.

Frame: the holder is LONG a near-high asset during a BTC uptrend. The held
state is ``BTC ret20 > 0`` (the T111/T114 master gate) AND the asset
``dfh20 >= -0.02`` (within 2% of its 20-bar high — "near-high"). On each
held asset-day we ask which OBSERVABLE trigger says the party is over:
the next-5d/10d forward return turns negative, or a drawdown is ahead.

Candidate exit triggers (all PIT — every input is available on the
decision day by construction of the mega cohort records / BTC 1d bars):

  E1 gate_off_cross      BTC ret20 crosses <= 0 (prev BTC day > 0). The
                         master gate itself — the baseline "ride it to
                         the end" exit. Fires on the first gate-off day;
                         the asset must still be near-high that day.
  E2 funding_euphoria    funding_pct (trailing-180 mid-rank of
                         last_funding_rate, MIN_WINDOW=20) >= 0.90 — top
                         decile crowding. State-days while held. Tests
                         whether extreme crowding tops out despite the
                         T66 funding-FOLLOW result (rho +0.042, top-dec
                         +190bps — crowding kept paying).
  E3 mom_exhaustion      mom20 crosses below 0 (prev cal-day >= 0) while
                         held — extended + stalling.
  E4 distribution        dfh20 crosses below -0.08 (prev cal-day >= -0.08)
                         after being >= -0.02 within the prior 5 days,
                         while still gate-on — a fast high-loss is top
                         distribution, not a dip.
  E5 vol_spike           vol_pct (trailing-180 mid-rank of vol20) crosses
                         >= 0.90 (prev cal-day < 0.90) — blow-off vol.
  E6 aged_trend          > 40 consecutive near-high days — time-in-state.
  E7 funding_flip_neg    last_funding_rate crosses < 0 (prev cal-day >= 0)
                         — 52-symbol funded subset only.
  E8 xs_rank_decay       cross-sectional dfh20 mid-rank falls out of the
                         top quintile (>= 0.80 -> < 0.80 on consecutive
                         cal-days; >= 30 ranked assets/day).

Outcomes (per asset-day, gross close-to-close, bps): fwd5 = the cohort
label ``forward_return_5d_bps`` (cross-checked against a recompute);
fwd10 and mae10 are computed from the reconstructed daily close series —
mae10 = min(close over the next <=10 calendar days)/close - 1, a
close-based max-adverse-excursion proxy (records carry no intraday low —
documented limitation).

Measurements
------------
1. Per trigger: n events, mean fwd5/fwd10 after trigger vs (a) ALL
   baseline held asset-days and (b) the composition-matched control —
   held days where the trigger was evaluable but did not fire — Welch t /
   normal-approx p; same for mae10 (a real exit trigger should fire
   BEFORE drawdowns, not just before flat returns).
2. Trigger independence: pairwise phi (Pearson on binary firing vectors)
   over the full decision frame, plus the E2-E8 matrix restricted to
   held days (E1/E4 cannot fire there by construction — E1 is disjoint
   BY DESIGN, E4 fires on the first sub-near-high day).
3. Combine: best single trigger vs "any-of-top-2" composite vs
   always-hold-until-gate-off (E1-only) -> net 5d-outcome distributions
   (net = -(fwd5 at exit) - 5bps taker exit cost, T67 convention:
   positive means exiting beat holding through the next 5d).
4. Per-year consistency of each trigger's fwd5 edge vs baseline.
5. Verdict: recommended exit rule set (primary trigger + catastrophe
   stop) for the campaign simulator.

Ranking note: triggers are ranked by Welch t of (event fwd5 - control
fwd5), most negative first — a good exit trigger marks days whose
forward path is worse than the average held day.

Artifacts follow the T114 convention: a frozen protocol object plus an
owner self-authorization pinning it by sha256 are embedded in the single
output artifact ``results/financial_signal_exit_state_v1.json``.
Measurement only: no fitting, no trading, no network, no other files
modified.
"""
import argparse
import bisect
import csv
import datetime as dt
import hashlib
import json
import math
import pathlib
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
COHORT = ROOT / "data/perp_pit_mega_v1/records.jsonl"
BTC_1D = ROOT / "data/rc_futures_v1/BTC/BTCUSDT_1d.csv"
OUT = ROOT / "results/financial_signal_exit_state_v1.json"

RET_LOOKBACK = 20          # master gate: BTC close/close[t-20]-1 > 0
NEAR_HIGH = -0.02          # dfh20 >= -0.02 = "near-high" (within 2%)
TRAIL = 180                # trailing window (rows ~= days) for *_pct
MIN_WINDOW = 20            # floor for a usable trailing pct (repo conv.)
FUNDING_EUPHORIA = 0.90    # E2: top-decile funding_pct
VOL_SPIKE = 0.90           # E5: vol_pct crossing threshold
AGED_DAYS = 40             # E6: consecutive near-high days
XS_TOP_Q = 0.80            # E8: top dfh20 quintile boundary
MIN_ASSETS_RANKED = 30     # E8 XS-rank day gate (mega convention)
E4_DROP = -0.08            # E4: dfh20 crosses below -0.08
E4_WINDOW = 5              # E4: near-high within the prior 5 days
EXIT_COST_BPS = 5.0        # taker exit cost (T67 net-backtest convention)
FWD_DAYS = (5, 10)
MAE_DAYS = 10
MIN_N = 10                 # Welch minimum per side (repo convention)
MIN_EVENTS_RANKED = 30     # floor for a trigger to enter the composite


def _r(x, nd=3):
    return round(x, nd) if isinstance(x, float) else x


def norm_p(t):
    """Two-sided normal-approx p from a t/z statistic (repo convention)."""
    if t is None:
        return None
    return 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))


def welch(xs, ys):
    """Welch two-sample t of mean(xs)-mean(ys), normal-approx p."""
    nx, ny = len(xs), len(ys)
    if nx < MIN_N or ny < MIN_N:
        return {"n_x": nx, "n_y": ny, "mean_x": (sum(xs) / nx if nx else None),
                "mean_y": (sum(ys) / ny if ny else None),
                "diff": None, "t": None, "p": None}
    mx, my = sum(xs) / nx, sum(ys) / ny
    vx = sum((x - mx) ** 2 for x in xs) / (nx - 1)
    vy = sum((y - my) ** 2 for y in ys) / (ny - 1)
    denom = vx / nx + vy / ny
    if denom <= 0:
        return {"n_x": nx, "n_y": ny, "mean_x": mx, "mean_y": my,
                "diff": mx - my, "t": None, "p": None}
    return {"n_x": nx, "n_y": ny, "mean_x": mx, "mean_y": my,
            "diff": mx - my, "t": (mx - my) / math.sqrt(denom),
            "p": norm_p((mx - my) / math.sqrt(denom))}


def dist(xs):
    """Mean/sd/median/p10/p90 summary of a list of floats."""
    n = len(xs)
    if not n:
        return {"n": 0}
    s = sorted(xs)
    mean = sum(s) / n
    sd = (sum((x - mean) ** 2 for x in s) / (n - 1)) ** 0.5 if n > 1 else 0.0
    def q(p):
        i = min(n - 1, max(0, int(round(p * (n - 1)))))
        return s[i]
    return {"n": n, "mean": _r(mean), "sd": _r(sd),
            "median": _r(q(0.5)), "p10": _r(q(0.1)), "p90": _r(q(0.9))}


# ------------------------------------------------------------------ loaders
def load_btc_ret20():
    """BTC 1d closes -> {day: ret20}, plus {day: prev-day ret20} for the
    crossing definition. None during the 20-bar warmup."""
    days, closes = [], []
    with BTC_1D.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            days.append(row["timestamp"][:10])
            closes.append(float(row["close"]))
    ret20, prev_ret20 = {}, {}
    for i, d in enumerate(days):
        v = (closes[i] / closes[i - RET_LOOKBACK] - 1.0
             if i >= RET_LOOKBACK and closes[i - RET_LOOKBACK] > 0 else None)
        ret20[d] = v
        if i > 0:
            prev_ret20[d] = (closes[i - 1] / closes[i - 1 - RET_LOOKBACK] - 1.0
                             if i - 1 >= RET_LOOKBACK
                             and closes[i - 1 - RET_LOOKBACK] > 0 else None)
    return ret20, prev_ret20, days


def load_cohort(path):
    """asset -> sorted rows {day, d, close, dfh20, mom20, vol20, funding,
    fwd5_label}."""
    series = defaultdict(list)
    n_rec = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            rec = json.loads(line)
            f = rec["features"]
            day = rec["id"].rsplit(":", 1)[-1]
            series[rec["asset_id"]].append({
                "day": day, "d": dt.date.fromisoformat(day),
                "asset": rec["asset_id"],
                "close": f["close"]["value"],
                "dfh20": f["dfh20"]["value"],
                "mom20": f["mom20"]["value"],
                "vol20": f["vol20"]["value"],
                "funding": f["last_funding_rate"]["value"],
                "fwd5": rec["label"].get("forward_return_5d_bps"),
            })
            n_rec += 1
    for rows in series.values():
        rows.sort(key=lambda r: r["day"])
    return series, n_rec


def add_trailing_pct(rows, key, out_key):
    """Per-asset mid-rank pct of row[key] vs its strictly-prior trailing-180
    non-None values, MIN_WINDOW floor (labels_v4/T112 convention)."""
    win, raw = [], []
    for r in rows:
        x = r.get(key)
        r[out_key] = None
        if x is not None and len(win) >= MIN_WINDOW:
            lt = bisect.bisect_left(win, x)
            eq = bisect.bisect_right(win, x) - lt
            r[out_key] = (lt + 0.5 * eq) / len(win)
        raw.append(x)
        if x is not None:
            bisect.insort(win, x)
        if len(raw) > TRAIL:
            old = raw.pop(0)
            if old is not None:
                win.pop(bisect.bisect_left(win, old))


def enrich_asset(rows):
    """In-place per-asset fields: near_high, streak, prev-cal-day feature
    echoes, recent-near-high flag, and computed fwd10 / mae10 / fwd5_calc.

    Crossing triggers require the previous CALENDAR day (gaps break the
    crossing — a stale 'previous bar' would fabricate a cross)."""
    by_day = {r["day"]: r for r in rows}
    streak = 0
    for i, r in enumerate(rows):
        d = r["d"]
        nh = r["dfh20"] is not None and r["dfh20"] >= NEAR_HIGH
        r["near_high"] = nh
        prev = rows[i - 1] if i > 0 else None
        prev_is_calday = (prev is not None
                          and (d - prev["d"]).days == 1)
        streak = streak + 1 if nh and prev_is_calday else (1 if nh else 0)
        r["streak"] = streak
        for k in ("dfh20", "mom20", "funding"):
            r[f"prev_{k}"] = prev[k] if prev_is_calday else None
        # prev-cal-day echoes of the trailing pcts: second pass below
        # (funding_pct / vol_pct are added before enrich_asset runs)
        # recent near-high: any dfh20 >= NEAR_HIGH on days d-5..d-1
        r["recent_nh5"] = False
        for k in range(1, E4_WINDOW + 1):
            q = by_day.get((d - dt.timedelta(days=k)).isoformat())
            if q is not None and q["dfh20"] is not None \
                    and q["dfh20"] >= NEAR_HIGH:
                r["recent_nh5"] = True
                break
        # computed forward outcomes from the close series (calendar days)
        c = r["close"]
        r["fwd5_calc"] = r["fwd10"] = r["mae10"] = None
        r["mae10_days"] = 0
        if c is not None and c > 0:
            f5 = by_day.get((d + dt.timedelta(days=5)).isoformat())
            if f5 is not None and f5["close"] is not None:
                r["fwd5_calc"] = (f5["close"] / c - 1.0) * 1e4
            f10 = by_day.get((d + dt.timedelta(days=10)).isoformat())
            if f10 is not None and f10["close"] is not None:
                r["fwd10"] = (f10["close"] / c - 1.0) * 1e4
            lo, n_fwd = None, 0
            for k in range(1, MAE_DAYS + 1):
                q = by_day.get((d + dt.timedelta(days=k)).isoformat())
                if q is not None and q["close"] is not None:
                    n_fwd += 1
                    lo = q["close"] if lo is None else min(lo, q["close"])
            r["mae10_days"] = n_fwd
            if lo is not None:
                r["mae10"] = (lo / c - 1.0) * 1e4
    # second pass: prev-cal-day echoes of the trailing pcts
    for i, r in enumerate(rows):
        prev = rows[i - 1] if i > 0 else None
        ok = prev is not None and (r["d"] - prev["d"]).days == 1
        r["prev_funding_pct"] = prev["funding_pct"] if ok else None
        r["prev_vol_pct"] = prev["vol_pct"] if ok else None


def add_xs_rank(series):
    """Per-day cross-sectional mid-rank pct of dfh20 across assets
    (n_ranked stored; E8 requires >= MIN_ASSETS_RANKED). Also adds
    prev-cal-day xs_rank echo per asset."""
    by_day = defaultdict(list)
    for rows in series.values():
        for r in rows:
            if r["dfh20"] is not None:
                by_day[r["day"]].append(r)
    for day, rs in by_day.items():
        vals = sorted(x["dfh20"] for x in rs)
        n = len(vals)
        for r in rs:
            lt = bisect.bisect_left(vals, r["dfh20"])
            eq = bisect.bisect_right(vals, r["dfh20"]) - lt
            r["xs_rank"] = (lt + 0.5 * eq) / n
            r["xs_n"] = n
    for rows in series.values():
        for i, r in enumerate(rows):
            r.setdefault("xs_rank", None)
            r.setdefault("xs_n", 0)
            prev = rows[i - 1] if i > 0 else None
            ok = prev is not None and (r["d"] - prev["d"]).days == 1
            r["prev_xs_rank"] = (prev.get("xs_rank") if ok else None)


# ------------------------------------------------------------------ triggers
def build_triggers():
    """name -> {desc, defined(row), fires(row)}.

    `defined` marks the composition-matched control: held asset-days where
    the trigger could have been evaluated. `fires` marks the exit event
    itself — for E1/E4 the event day sits just outside the held set by
    construction (gate-off / sub-near-high)."""
    def in_B(r):
        return r["gate_on"] and r["near_high"]

    def prev_day_field(r, k):
        return r.get(k)

    T = {}

    T["E1_gate_off_cross"] = {
        "desc": "BTC ret20 crosses <= 0 (prev BTC day > 0); asset still "
                "near-high on the crossing day. The baseline exit.",
        "defined": lambda r: r["near_high"] and r["ret20"] is not None
            and r["prev_btc_ret20"] is not None
            and (r["ret20"] > 0 or r["prev_btc_ret20"] > 0),
        "fires": lambda r: r["near_high"] and r["ret20"] is not None
            and r["ret20"] <= 0 and r["prev_btc_ret20"] is not None
            and r["prev_btc_ret20"] > 0,
    }
    T["E2_funding_euphoria"] = {
        "desc": "funding_pct (trailing-180 mid-rank, funded 52-sym subset) "
                ">= 0.90 while held",
        "defined": lambda r: in_B(r) and r["funding_pct"] is not None,
        "fires": lambda r: in_B(r) and r["funding_pct"] is not None
            and r["funding_pct"] >= FUNDING_EUPHORIA,
    }
    T["E3_mom_exhaustion"] = {
        "desc": "mom20 crosses < 0 (prev cal-day >= 0) while held",
        "defined": lambda r: in_B(r) and r["mom20"] is not None
            and prev_day_field(r, "prev_mom20") is not None,
        "fires": lambda r: in_B(r) and r["mom20"] is not None
            and r["mom20"] < 0 and r["prev_mom20"] is not None
            and r["prev_mom20"] >= 0,
    }
    T["E4_distribution"] = {
        "desc": "dfh20 crosses < -0.08 (prev cal-day >= -0.08) after "
                "near-high within prior 5d, still gate-on — fast "
                "high-loss = top distribution",
        "defined": lambda r: in_B(r),
        "fires": lambda r: r["gate_on"] and r["dfh20"] is not None
            and r["dfh20"] < E4_DROP and r["prev_dfh20"] is not None
            and r["prev_dfh20"] >= E4_DROP and r["recent_nh5"],
    }
    T["E5_vol_spike"] = {
        "desc": "vol_pct (trailing-180 mid-rank of vol20) crosses >= 0.90 "
                "(prev cal-day < 0.90) while held — blow-off",
        "defined": lambda r: in_B(r) and r["vol_pct"] is not None
            and r["prev_vol_pct"] is not None,
        "fires": lambda r: in_B(r) and r["vol_pct"] is not None
            and r["vol_pct"] >= VOL_SPIKE
            and r["prev_vol_pct"] is not None
            and r["prev_vol_pct"] < VOL_SPIKE,
    }
    T["E6_aged_trend"] = {
        "desc": f"> {AGED_DAYS} consecutive near-high days (time-in-state) "
                "while held",
        "defined": lambda r: in_B(r),
        "fires": lambda r: in_B(r) and r["streak"] > AGED_DAYS,
    }
    T["E7_funding_flip_neg"] = {
        "desc": "last_funding_rate crosses < 0 (prev cal-day >= 0) while "
                "held — funded 52-sym subset",
        "defined": lambda r: in_B(r) and r["funding"] is not None
            and r["prev_funding"] is not None,
        "fires": lambda r: in_B(r) and r["funding"] is not None
            and r["funding"] < 0 and r["prev_funding"] is not None
            and r["prev_funding"] >= 0,
    }
    T["E8_xs_rank_decay"] = {
        "desc": "XS dfh20 mid-rank falls out of the top quintile "
                "(>= 0.80 -> < 0.80, prev cal-day, >=30 ranked) while held",
        "defined": lambda r: in_B(r) and r.get("xs_rank") is not None
            and r.get("xs_n", 0) >= MIN_ASSETS_RANKED
            and r.get("prev_xs_rank") is not None,
        "fires": lambda r: in_B(r) and r.get("xs_rank") is not None
            and r.get("xs_n", 0) >= MIN_ASSETS_RANKED
            and r["xs_rank"] < XS_TOP_Q
            and r.get("prev_xs_rank") is not None
            and r["prev_xs_rank"] >= XS_TOP_Q,
    }
    return T


def phi(xs, ys):
    """Pearson correlation of two binary vectors (phi coefficient)."""
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    vx = mx * (1 - mx)
    vy = my * (1 - my)
    if vx <= 0 or vy <= 0:
        return None
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / n
    return cov / math.sqrt(vx * vy)


# ------------------------------------------------------------------ protocol
def build_protocol():
    return {
        "schema_version": "nanojev-financial-signal-exit-state-protocol-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "task": "T118: exit states — which observable PIT triggers tell a "
                "long-near-high-during-BTC-uptrend holder the party is "
                "over (negative fwd 5d/10d, or drawdown ahead).",
        "cohort": {
            "path": "data/perp_pit_mega_v1/records.jsonl",
            "assets": "277 USDT-M perps incl. delisted early-stoppers",
            "span": "2021-01 -> 2025-12 daily decision bars",
            "btc_bars": "data/rc_futures_v1/BTC/BTCUSDT_1d.csv",
            "price_basis": "csv close as mark proxy; gross close-to-close "
                           "outcomes; no funding cashflows/borrow/costs "
                           "except the stated 5bps exit-cost convention",
        },
        "held_state": "baseline B = asset-days with BTC ret20 > 0 AND "
                      "dfh20 >= -0.02 (gate-on near-high)",
        "triggers": {
            "E1_gate_off_cross": "BTC ret20 <= 0 after prev-BTC-day > 0; "
                                 "asset near-high on the crossing day",
            "E2_funding_euphoria": "funding_pct (trailing-180 mid-rank of "
                                   "last_funding_rate, min window 20) "
                                   ">= 0.90 while held",
            "E3_mom_exhaustion": "mom20 crosses < 0 (prev cal-day >= 0) "
                                 "while held",
            "E4_distribution": "dfh20 crosses < -0.08 (prev cal-day "
                               ">= -0.08) with a near-high day inside the "
                               "prior 5d, still gate-on",
            "E5_vol_spike": "vol_pct (trailing-180 mid-rank of vol20) "
                            "crosses >= 0.90 while held",
            "E6_aged_trend": ">40 consecutive near-high days while held",
            "E7_funding_flip_neg": "last_funding_rate crosses < 0 while "
                                   "held (52 funded symbols)",
            "E8_xs_rank_decay": "XS dfh20 mid-rank exits the top quintile "
                                "(>=0.80 -> <0.80 prev cal-day, >=30 "
                                "ranked assets/day) while held",
        },
        "outcomes": {
            "fwd5_bps": "cohort label forward_return_5d_bps (cross-checked "
                        "against close-series recompute)",
            "fwd10_bps": "close[t+10d]/close[t]-1 recomputed",
            "mae10_bps": "min close over next <=10 cal-days /close-1 "
                         "(close-based MAE proxy — records have no "
                         "intraday low)",
        },
        "statistics": {
            "per_trigger": "n events; Welch(events vs all-baseline) and "
                           "Welch(events vs defined-not-fired control) on "
                           "fwd5, fwd10, mae10; normal-approx p",
            "independence": "pairwise phi on firing vectors over the full "
                            "decision frame, plus the E2-E8 matrix on "
                            "held days only",
            "composite": "triggers ranked by fwd5 Welch t (most negative "
                         "first, n>=30); best-single vs union-of-top-2 vs "
                         "E1-only; net edge = -mean(fwd5 at exit) - 5bps",
            "per_year": "per-year n, mean fwd5 at events vs that-year "
                        "baseline mean, mean mae10",
        },
        "forbidden": ["fitting", "trading", "profitability claims",
                      "protocol edits post-run", "network"],
    }


def build_auth(protocol_sha):
    return {
        "schema_version":
            "nanojev-financial-signal-exit-state-authorization-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "protocol_sha256": protocol_sha,
        "decision": "approved_for_measurement",
        "measurement_authorized": True,
        "fit_authorized": False,
        "independent_reviewer": {
            "id": "project-owner",
            "independence": "owner_self_authorization_not_independent_review",
            "note": "Owner directed T118: exit-state trigger measurement "
                    "on the mega cohort (delegated task).",
        },
        "scope": {
            "permitted": "PIT-safe descriptive measurement on "
                         "data/perp_pit_mega_v1/records.jsonl and the BTC "
                         "1d bars per the embedded protocol.",
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
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    args = ap.parse_args()

    protocol = build_protocol()
    protocol_sha = hashlib.sha256(
        json.dumps(protocol, indent=2, sort_keys=True).encode()).hexdigest()
    report = {"schema_version": "nanojev-financial-signal-exit-state-v1",
              "task": "T118 exit states — when to stop rolling and wait, "
                      "mega cohort",
              "protocol": protocol,
              "protocol_sha256": protocol_sha,
              "owner_authorization": build_auth(protocol_sha)}

    if not args.cohort.exists() or not BTC_1D.exists():
        report["status"] = ("SKIPPED: cohort or BTC bars not found; nothing "
                            "was fabricated")
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return 0

    ret20, prev_ret20, btc_days = load_btc_ret20()
    series, n_rec = load_cohort(args.cohort)
    funded = sorted(a for a, rows in series.items()
                    if any(r["funding"] is not None for r in rows))

    # per-asset enrichment
    for rows in series.values():
        add_trailing_pct(rows, "funding", "funding_pct")
        add_trailing_pct(rows, "vol20", "vol_pct")
        enrich_asset(rows)
    add_xs_rank(series)

    # flat row list with gate state
    all_rows = [r for rows in series.values() for r in rows]
    for r in all_rows:
        r["ret20"] = ret20.get(r["day"])
        r["prev_btc_ret20"] = prev_ret20.get(r["day"])
        r["gate_on"] = r["ret20"] is not None and r["ret20"] > 0
        r["year"] = r["day"][:4]

    # fwd5 label cross-check vs close recompute
    chk = [(r["fwd5"], r["fwd5_calc"]) for r in all_rows
           if r["fwd5"] is not None and r["fwd5_calc"] is not None]
    max_abs = max((abs(a - b) for a, b in chk), default=None)
    mean_abs = (sum(abs(a - b) for a, b in chk) / len(chk)) if chk else None
    n_both = len(chk)
    n_label_only = sum(1 for r in all_rows
                       if r["fwd5"] is not None and r["fwd5_calc"] is None)
    n_calc_only = sum(1 for r in all_rows
                      if r["fwd5"] is None and r["fwd5_calc"] is not None)
    report["label_crosscheck"] = {
        "n_both": n_both,
        "fwd5_label_vs_calc_max_abs_bps": _r(max_abs, 4),
        "fwd5_label_vs_calc_mean_abs_bps": _r(mean_abs, 4),
        "label_only": n_label_only, "calc_only": n_calc_only,
        "note": "calc uses exact calendar +5d close; small diffs = label "
                "built on bar-index not calendar-day offsets"}

    cohort_days = sorted({r["day"] for r in all_rows})
    baseline = [r for r in all_rows if r["gate_on"] and r["near_high"]]
    report["cohort_span"] = {
        "records": n_rec, "symbols": len(series),
        "funded_symbols": len(funded),
        "first_day": cohort_days[0], "last_day": cohort_days[-1],
        "decision_days": len(cohort_days),
        "baseline_held_asset_days": len(baseline),
        "baseline_share_pct": _r(100.0 * len(baseline) / len(all_rows), 1),
        "gate_on_asset_days": sum(1 for r in all_rows if r["gate_on"]),
        "near_high_asset_days": sum(1 for r in all_rows if r["near_high"]),
    }

    def oc(rows, k):
        return [r[k] for r in rows if r[k] is not None]

    base_out = {k: oc(baseline, k) for k in ("fwd5", "fwd10", "mae10")}
    report["baseline_stats"] = {
        "n_held_asset_days": len(baseline),
        "fwd5_bps": dist(base_out["fwd5"]),
        "fwd10_bps": dist(base_out["fwd10"]),
        "mae10_bps": dist(base_out["mae10"]),
        "mae10_window_coverage": dist([r["mae10_days"] for r in baseline]),
    }
    base_year_mean = defaultdict(list)
    for r in baseline:
        if r["fwd5"] is not None:
            base_year_mean[r["year"]].append(r["fwd5"])
    base_year_mean = {y: sum(v) / len(v) for y, v in base_year_mean.items()}

    # ------------------------------------------------ section 1: triggers
    triggers = build_triggers()
    trig_events = {}   # name -> list of firing rows
    trig_defined = {}  # name -> list of evaluable rows
    for name, spec in triggers.items():
        ev = [r for r in all_rows if spec["fires"](r)]
        dv = [r for r in all_rows if spec["defined"](r)]
        ev_ids = {id(r) for r in ev}
        ctrl = [r for r in dv if id(r) not in ev_ids]
        trig_events[name] = ev
        trig_defined[name] = ctrl
        blk = {"desc": spec["desc"],
               "n_events": len(ev),
               "n_control_defined_not_fired": len(ctrl),
               "event_share_of_defined_pct": _r(
                   100.0 * len(ev) / len(dv), 2) if dv else None,
               "distinct_assets": len({r["asset"] for r in ev}),
               "distinct_days": len({r["day"] for r in ev})}
        for ok in ("fwd5", "fwd10", "mae10"):
            e = oc(ev, ok)
            blk[ok] = {
                "event_dist_bps": dist(e),
                "vs_all_baseline": {k: _r(v, 6) if k == "p" else _r(v)
                                    for k, v in welch(e, base_out[ok]).items()},
                "vs_defined_control": {
                    k: _r(v, 6) if k == "p" else _r(v)
                    for k, v in welch(e, oc(ctrl, ok)).items()},
            }
        report.setdefault("triggers", {})[name] = blk

    # --------------------------------------- section 2: independence (phi)
    e1_ids = {id(x) for x in trig_events["E1_gate_off_cross"]}
    e4_ids = {id(x) for x in trig_events["E4_distribution"]}
    frame = [r for r in all_rows
             if (r["gate_on"] and r["near_high"])
             or id(r) in e1_ids or id(r) in e4_ids]
    names = list(triggers)
    fire_vec = {n: [1 if spec["fires"](r) else 0 for r in frame]
                for n, spec in triggers.items()}
    phi_all, cofire = {}, {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            p = phi(fire_vec[a], fire_vec[b])
            both = sum(1 for x, y in zip(fire_vec[a], fire_vec[b])
                       if x and y)
            phi_all[f"{a}__{b}"] = _r(p, 4)
            cofire[f"{a}__{b}"] = {
                "both": both,
                "p_b_given_a": _r(both / sum(fire_vec[a]), 4)
                if sum(fire_vec[a]) else None,
                "p_a_given_b": _r(both / sum(fire_vec[b]), 4)
                if sum(fire_vec[b]) else None}
    held_names = [n for n in names if n not in
                  ("E1_gate_off_cross", "E4_distribution")]
    hv = {n: [1 if triggers[n]["fires"](r) else 0 for r in baseline]
          for n in held_names}
    phi_held = {}
    for i, a in enumerate(held_names):
        for b in held_names[i + 1:]:
            phi_held[f"{a}__{b}"] = _r(phi(hv[a], hv[b]), 4)
    report["trigger_independence"] = {
        "frame_asset_days": len(frame),
        "frame_note": "baseline held days + E1 + E4 event days; E1 is "
                      "disjoint from E2/E3/E5-E8 by construction (fires "
                      "only on the first gate-off day), E4 fires on the "
                      "first sub-near-high day",
        "phi_full_frame": phi_all,
        "phi_held_days_only_E2_E8": phi_held,
        "cofire_counts": cofire,
    }

    # ------------------------------------------- section 3: policy compare
    ranked = sorted(
        ((n, report["triggers"][n]["fwd5"]["vs_defined_control"]["t"],
          report["triggers"][n]["n_events"]) for n in names),
        key=lambda x: (x[1] is None, x[1] if x[1] is not None else 0))
    ranked_ok = [n for n, t, ne in ranked
                 if t is not None and ne >= MIN_EVENTS_RANKED]
    best = ranked_ok[0] if ranked_ok else None
    top2 = ranked_ok[:2]
    ev_id_sets = {n: {id(r) for r in ev} for n, ev in trig_events.items()}
    policies = {"always_hold_until_gate_off_E1":
                [r for r in all_rows if id(r) in
                 ev_id_sets["E1_gate_off_cross"]]}
    if best:
        policies[f"best_single_{best}"] = trig_events[best]
    if len(top2) == 2:
        u = ev_id_sets[top2[0]] | ev_id_sets[top2[1]]
        policies[f"any_of_top2_{'_'.join(top2)}"] = \
            [r for r in all_rows if id(r) in u]
    pol_out = {}
    for pname, ev in policies.items():
        f5 = oc(ev, "fwd5")
        pol_out[pname] = {
            "n_exit_events": len(ev),
            "exits_per_100_held_days": _r(
                100.0 * len(ev) / len(baseline), 2) if baseline else None,
            "fwd5_at_exit_bps": dist(f5),
            "fwd10_at_exit_bps": dist(oc(ev, "fwd10")),
            "mae10_at_exit_bps": dist(oc(ev, "mae10")),
            "welch_fwd5_vs_baseline": {
                k: _r(v, 6) if k == "p" else _r(v)
                for k, v in welch(f5, base_out["fwd5"]).items()},
            "net_exit_edge_bps_per_event": _r(
                (-sum(f5) / len(f5) - EXIT_COST_BPS) if f5 else None),
        }
    report["policy_compare"] = {
        "ranking_rule": "fwd5 Welch t vs defined control, most negative "
                        "first; n_events >= 30 required",
        "ranking": [{"trigger": n, "welch_t": _r(t), "n_events": ne}
                    for n, t, ne in ranked],
        "policies": pol_out,
        "exit_cost_bps": EXIT_COST_BPS,
        "net_edge_note": "positive = exiting at the trigger beat holding "
                         "the next 5d, net of the 5bps exit cost",
    }

    # ------------------------------------------------- section 4: per-year
    per_year = {}
    for name in names:
        ev = trig_events[name]
        by_y = defaultdict(list)
        for r in ev:
            if r["fwd5"] is not None:
                by_y[r["year"]].append(r["fwd5"])
        ytab, n_worse = {}, 0
        for y in sorted(by_y):
            m = sum(by_y[y]) / len(by_y[y])
            bm = base_year_mean.get(y)
            diff = m - bm if bm is not None else None
            if diff is not None and diff < 0:
                n_worse += 1
            ytab[y] = {"n_events": len(by_y[y]),
                       "mean_fwd5_bps": _r(m),
                       "baseline_mean_fwd5_bps": _r(bm),
                       "event_minus_baseline": _r(diff)}
        per_year[name] = {
            "years": ytab,
            "years_event_worse_than_baseline":
                f"{n_worse}/{len(ytab)}"}
    report["per_year"] = per_year

    # ------------------------------------------------------ section 5: verdict
    def t_of(n, ok="fwd5", ctrl="vs_defined_control"):
        return report["triggers"][n][ok][ctrl]["t"]

    fwd5_rank = [n for n in ranked_ok]
    mae_rank = sorted(
        (n for n in names
         if report["triggers"][n]["mae10"]["vs_defined_control"]["t"]
         is not None
         and report["triggers"][n]["n_events"] >= MIN_EVENTS_RANKED),
        key=lambda n: report["triggers"][n]["mae10"]
        ["vs_defined_control"]["t"])
    # a catastrophe stop must be an EARLY trigger: it fires while the
    # master gate is still on (E1 is the terminal exit itself)
    early_mae_rank = [n for n in mae_rank if n != "E1_gate_off_cross"]
    best_fwd = fwd5_rank[0] if fwd5_rank else None
    best_mae = mae_rank[0] if mae_rank else None
    best_early_mae = early_mae_rank[0] if early_mae_rank else None
    e1_t = t_of("E1_gate_off_cross")
    verdict_bits = {
        "ranked_by_fwd5_welch_t": fwd5_rank,
        "ranked_by_mae10_welch_t": mae_rank,
        "best_single_by_fwd5": best_fwd,
        "best_single_by_mae10": best_mae,
        "best_early_trigger_by_mae10": best_early_mae,
        "e1_fwd5_welch_t": _r(e1_t),
    }
    if best_fwd and best_fwd != "E1_gate_off_cross":
        primary = best_fwd
        cat = (best_early_mae if best_early_mae != primary
               else "E1_gate_off_cross")
        verdict = (f"EARLY EXIT ADDS: primary trigger {primary} (fwd5 "
                   f"Welch t {_r(t_of(primary))} vs defined control) "
                   f"exits before worse-than-held forward paths; "
                   f"catastrophe stop = {cat} (best early mae10 "
                   f"separation, t "
                   f"{_r(report['triggers'][cat]['mae10']['vs_defined_control']['t'])}) "
                   "with E1 gate-off as the terminal backstop.")
    elif e1_t is not None and e1_t < -2:
        verdict = ("HOLD UNTIL GATE-OFF: no early trigger separates held "
                   "days' forward paths better than the master gate "
                   f"itself (E1 fwd5 Welch t {_r(e1_t)}). Recommended "
                   "rule set: primary exit = E1 gate-off cross; "
                   f"catastrophe stop = {best_early_mae} (best early "
                   "mae10 separation — fires ahead of deep drawdowns "
                   "even where mean fwd returns stay flat; functions as "
                   "a risk reducer, not a return signal).")
    else:
        verdict = ("WEAK/NO SEPARATION: no trigger shows a robust negative "
                   "fwd5 separation vs held baseline; keep E1 gate-off as "
                   "the exit and treat E2-E8 as unproven.")
    verdict_bits["verdict"] = verdict
    report["verdict_block"] = verdict_bits

    report["honesty"] = {
        "not_a_return": "fwd5/fwd10/mae10 are gross close-price moves; "
                        "no funding cashflows, borrow or slippage; the "
                        "only cost modeled is the stated 5bps exit "
                        "convention in the net-edge line",
        "mae_is_close_based": "records carry no intraday low; mae10 "
                              "understates true intraday drawdown",
        "overlapping_labels": "5d/10d labels overlap on adjacent days; "
                              "events cluster in time and across assets; "
                              "all t/p nominal, no multiple-testing "
                              "correction claimed",
        "event_overlap": "n_events counts asset-days, not deduplicated "
                         "episodes; state triggers (E2, E6) can fire on "
                         "consecutive days of one episode",
        "funding_subset": "E2/E7 defined only on the 52 funded symbols; "
                          "their contrasts are subset-conditional",
        "not_an_asof_vintage": "second-hand archive copy (rc_futures_v1)",
        "not_live": "no orders, no account, no broker",
        "no_profitability_claim": True,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    brief = {"status": "ran", "out": str(args.out),
             "held_days": len(baseline),
             "fwd5_check_max_abs_bps": _r(max_abs, 4),
             "ranking_fwd5": report["policy_compare"]["ranking"],
             "verdict": verdict}
    for n in names:
        t = report["triggers"][n]
        brief[n] = {"n": t["n_events"],
                    "fwd5_mean": t["fwd5"]["event_dist_bps"].get("mean"),
                    "fwd5_t_ctrl": t["fwd5"]["vs_defined_control"]["t"],
                    "mae10_mean": t["mae10"]["event_dist_bps"].get("mean"),
                    "mae10_t_ctrl": t["mae10"]["vs_defined_control"]["t"]}
    print(json.dumps(brief, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
