#!/usr/bin/env python3
"""Principled-parameter rebuild of the rolling-campaign simulator (T123).

T119's composite arm produced final equity 34.37x vs BTC hold 3.02x, but
T122's 36-cell perturbation grid judged that parameter-LUCKY: median grid
cell 0.65x, the default cell ranked 2/36 (top-decile = overfit flag), and
the ex-2021 compounded equity was 0.22x. The three hand-tuned dimensions
were the pullback dfh20 band [-0.12,-0.03], the fixed 8% trailing stop,
and the funding_pct<0.8 entry filter (which T122 showed added nothing —
its funding=none row was the best cell).

This v2 replaces every arbitrary constant with a data-driven quantity
derived from per-asset volatility, and then asks the honest question: is
the conclusion FLAT across the immediate parameter neighbourhood
(robust) or PEAKED at the primary cell (lucky again)?

VOLATILITY BASIS (per asset, per day — all PIT, all from closes already
in the cohort):
  ret_i   = log(close_i / close_{i-1})
  vol20   = sample std (ddof=1) of the trailing 20 log returns
            (needs 21 closes; None during warmup)
  sigma20d = vol20 * sqrt(20)   — expected 20-bar dispersion, the same
            scale dfh20 (distance from the 20-bar high, as a fraction)
            is measured on.

PRINCIPLED REPLACEMENTS:
  trailing stop : fixed 8%  ->  close < max_close*(1 - k*vol20),
                  k=3 primary. The stop is k daily-sigmas under the
                  running max — it widens for high-vol assets and
                  tightens for low-vol ones instead of one number for
                  all 277 symbols. Safety clamps only (not tuning dims):
                  k*vol20 clipped to [0.02, 0.60]; if vol20 is missing
                  the trail simply does not fire that day.
  pullback band : fixed [-0.12,-0.03]  ->  dfh20 in
                  [-3*sigma20d, -0.5*sigma20d] — "pulled back between
                  half and three 20-bar sigmas off the high". Expressed
                  in each asset's own volatility units; requires both
                  dfh20 and vol20 non-None.
  funding filter: DROPPED at entry AND the funding_pct>=0.95 euphoria
                  exit is dropped — T122's grid showed the funding dims
                  contributed nothing (funding=none was the best cell;
                  |rho|=0.157 the weakest dim). Kept constant: BTC
                  ret20>0 gate, top-quintile dfh20 XS rank (>=30 ranked),
                  close>prev_close recovery day, +3% add trigger x2 at
                  50% margined, 30d time-stop, rank<median exit, data_end
                  and end_of_sample handling, 5bps/fill, 1 campaign at a
                  time, same-day re-entry allowed except into the asset
                  just exited.

ENTRY (v2 composite):  gate-on AND top dfh20 quintile AND
  close>prev_close AND dfh20 in [band_lo*sigma20d, band_hi*sigma20d].
EXITS (first to fire): data_end -> btc_regime_off -> vol_trail ->
  time_stop_30d -> rank_below_median (end_of_sample at tail).

ROBUSTNESS CELLS (3x3 = 9, the two perturbed dims only):
  trail multiplier k in {2, 3, 4}
  band shift s in {-1sd, 0, +1sd}: lo=-3+s, hi=-0.5+s
    s=-1 -> [-4sd, -1.5sd] (deeper pullbacks only)
    s=0  -> [-3sd, -0.5sd] (PRIMARY)
    s=+1 -> [-2sd, +0.5sd] (shallower, allows near-high entries)
  Every cell reports final equity AND ex-2021 compounded equity
  (prod(1+r) over campaigns exiting >=2022-01-01 — the T122 LOYO
  convention, THE headline honesty number). Flat = robust, peaked =
  lucky.

ARMS:
  primary     k=3, band shift 0 — full metrics: equity, maxDD, win%,
              n campaigns, per-year table, exit-reason histogram.
  no_filter   sibling baseline: entry = gate-on AND top-quintile only
              (T119 convention); identical v2 exits at k=3.
  btc_hold    buy-and-hold BTC over the sim calendar; also reported
              ex-2021 (close ratio from first 2022 day to last day) so
              the ex-2021 comparison is apples-to-apples.

ENGINE REUSE: loaders, day-view builders, metrics and shared constants
are imported unmodified from ``financial_campaign_sim_v1`` (the same
trick T122 used); only the state machine is re-implemented here as
``simulate_v2`` because the band/trail are now per-asset-per-day
quantities, not module constants.

Determinism: the full 9-cell grid plus the sibling are simulated twice;
serialized metrics must be identical. Artifacts follow convention: a
frozen protocol (``research/financial_campaign_v2_principled_protocol_
v1.json``) plus owner self-authorization
(``results/financial_campaign_v2_principled_authorization_v1.json``)
written BEFORE measurement. Measurement only: no fitting, no trading,
no network.
"""
import argparse
import bisect  # noqa: F401  (parity with engine imports)
import datetime as dt
import hashlib
import json
import math
import pathlib
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import financial_campaign_sim_v1 as engine  # noqa: E402  (T119 engine)

ROOT = pathlib.Path(__file__).resolve().parent.parent
COHORT = ROOT / "data/perp_pit_mega_v1/records.jsonl"
BTC_CSV = ROOT / "data/rc_futures_v1/BTC/BTCUSDT_1d.csv"
OUT = ROOT / "results/financial_campaign_v2_principled_v1.json"
PROTOCOL = ROOT / "research/financial_campaign_v2_principled_protocol_v1.json"
AUTH = ROOT / "results/financial_campaign_v2_principled_authorization_v1.json"

# ---- shared conventions (imported from T119, never re-tuned) ---------
FEE = engine.FEE                    # 5bps per fill
TOP_Q = engine.TOP_Q                # top XS dfh20 quintile
MEDIAN = engine.MEDIAN              # rank<median exit
TIME_STOP_DAYS = engine.TIME_STOP_DAYS
ADD_TRIGGER = engine.ADD_TRIGGER
ADD_FRAC = engine.ADD_FRAC
MAX_ADDS = engine.MAX_ADDS
MIN_RANKED = engine.MIN_RANKED
VOL_WINDOW = 20                     # trailing-20 log-return std
SQRT20 = math.sqrt(20.0)

# ---- primary (principled) parameters ----------------------------------
PRIMARY_K = 3               # trail = k daily sigmas under max close
PRIMARY_BAND_LO = -3.0      # dfh20 >= -3.0 * sigma20d
PRIMARY_BAND_HI = -0.5      # dfh20 <= -0.5 * sigma20d
TRAIL_MIN_FRAC = 0.02       # safety clamp: never tighter than 2%
TRAIL_MAX_FRAC = 0.60       # safety clamp: never looser than 60%

# ---- robustness cells --------------------------------------------------
K_LEVELS = (2, 3, 4)
BAND_SHIFTS = (-1, 0, 1)    # in units of sigma20d


def _r(x, nd=4):
    return round(x, nd) if isinstance(x, float) else x


def annotate_vol20(series):
    """Add r['vol20'] / r['sigma20d'] to every row in-place.

    vol20 = sample std (ddof=1) of the trailing 20 log returns; needs
    21 closes (first usable at index 20). sigma20d = vol20*sqrt(20) is
    the 20-bar dispersion, the same scale dfh20 is measured on.
    """
    n_with = 0
    for rows in series.values():
        closes = [r["close"] for r in rows]
        rets = [None] * len(rows)
        for i in range(1, len(rows)):
            if closes[i] and closes[i - 1] and closes[i - 1] > 0 \
                    and closes[i] > 0:
                rets[i] = math.log(closes[i] / closes[i - 1])
        for i, r in enumerate(rows):
            window = [x for x in rets[max(0, i - VOL_WINDOW + 1):i + 1]
                      if x is not None]
            if len(window) >= VOL_WINDOW:
                r["vol20"] = statistics.stdev(window)
                r["sigma20d"] = r["vol20"] * SQRT20
                n_with += 1
            else:
                r["vol20"] = r["sigma20d"] = None
    return n_with


def simulate_v2(calendar, rows_by_day, dfh_pct, n_ranked, btc, meta,
                trail_k=PRIMARY_K, band_lo_mult=PRIMARY_BAND_LO,
                band_hi_mult=PRIMARY_BAND_HI, use_band=True):
    """Vol-scaled state machine. Mirrors engine.simulate; returns
    {equity, campaigns, fills}.

    Differences vs T119: entry drops the funding filter and scales the
    pullback band by sigma20d; exits drop funding_euphoria and scale the
    trail by vol20. Everything else — fill-at-close, pyramid, exit
    order, re-entry rule — is identical.
    """
    cash, qty = 1.0, 0.0
    asset = None
    entry_day = entry_date = None
    cost_qty = 0.0
    max_close = 0.0
    last_close = None
    adds = 0
    entry_equity = None
    equity, campaigns, fills = {}, [], []

    def close_position(day, px, reason):
        nonlocal cash, qty, asset, cost_qty, max_close, last_close
        nonlocal adds, entry_equity
        proceeds = qty * px
        fee = proceeds * FEE
        cash += proceeds - fee
        fills.append({"day": day, "asset": asset, "side": "exit",
                      "qty": qty, "px": px, "fee": fee})
        ret = cash / entry_equity - 1.0
        campaigns.append({
            "asset": asset, "entry_day": entry_day, "exit_day": day,
            "days_held": (dt.date.fromisoformat(day)
                          - entry_date).days,
            "adds": adds, "return": ret, "exit_reason": reason,
            "listed": meta[asset]["listed"],
            "last_bar_date": meta[asset]["last_bar_date"],
        })
        qty = 0.0
        asset = None
        cost_qty = max_close = 0.0
        last_close = None
        adds = 0
        entry_equity = None
        return asset

    exited_today = set()
    for day in calendar:
        btc_r = btc.get(day, {}).get("ret20")
        gate_on = btc_r is not None and btc_r > 0
        ranked_ok = n_ranked.get(day, 0) >= MIN_RANKED
        pcts = dfh_pct.get(day, {})

        # ---------------- open campaign: exits, then pyramid ----------
        if asset is not None:
            row = rows_by_day[day].get(asset)
            reason = None
            px = None
            if row is None or row["close"] is None:
                reason, px = "data_end", last_close
            else:
                last_close = row["close"]
                max_close = max(max_close, last_close)
                pct = pcts.get(asset)
                held = (dt.date.fromisoformat(day) - entry_date).days
                vol = row["vol20"]
                trail_frac = None
                if vol is not None:
                    trail_frac = min(max(trail_k * vol,
                                         TRAIL_MIN_FRAC),
                                     TRAIL_MAX_FRAC)
                if btc_r is not None and btc_r <= 0:
                    reason = "btc_regime_off"
                elif (trail_frac is not None
                      and last_close < max_close * (1.0 - trail_frac)):
                    reason = "vol_trail"
                elif held >= TIME_STOP_DAYS:
                    reason = "time_stop_30d"
                elif pct is None or pct < MEDIAN:
                    reason = "rank_below_median"
                if reason:
                    px = last_close
            if reason:
                exited_today.add(asset)
                close_position(day, px, reason)
            elif row is not None:
                # pyramid: +3% vs vwap, gate on, still top quintile
                vwap = cost_qty / qty if qty else None
                unreal = (last_close / vwap - 1.0) if vwap else 0.0
                pct = pcts.get(asset)
                if (adds < MAX_ADDS and unreal >= ADD_TRIGGER and gate_on
                        and pct is not None and pct >= TOP_Q):
                    add_qty = qty * ADD_FRAC
                    notional = add_qty * last_close
                    fee = notional * FEE
                    cash -= notional + fee   # margined perp-style
                    qty += add_qty
                    cost_qty += notional
                    adds += 1
                    fills.append({"day": day, "asset": asset,
                                  "side": "add", "qty": add_qty,
                                  "px": last_close, "fee": fee})

        # ---------------- flat: entry scan ------------------------------
        if asset is None and gate_on and ranked_ok and cash > 0:
            cands = []
            for a, r in rows_by_day[day].items():
                if a in exited_today:
                    continue
                pct = pcts.get(a)
                if pct is None or pct < TOP_Q:
                    continue
                if use_band:
                    d = r["dfh20"]
                    sig = r["sigma20d"]
                    if d is None or sig is None or sig <= 0:
                        continue
                    if not (band_lo_mult * sig <= d
                            <= band_hi_mult * sig):
                        continue
                    if (r["prev_close"] is None
                            or r["close"] <= r["prev_close"]):
                        continue
                cands.append((pct, a))
            if cands:
                cands.sort(key=lambda x: (-x[0], x[1]))
                asset = cands[0][1]
                px = rows_by_day[day][asset]["close"]
                entry_equity = cash
                qty = cash / (px * (1.0 + FEE))
                fee = qty * px * FEE
                cash = 0.0
                fills.append({"day": day, "asset": asset,
                              "side": "entry", "qty": qty, "px": px,
                              "fee": fee})
                cost_qty = qty * px
                entry_day, entry_date = day, dt.date.fromisoformat(day)
                max_close = last_close = px
                adds = 0
        exited_today.clear()

        equity[day] = cash + qty * last_close if asset is not None \
            else cash

    if asset is not None:
        close_position(calendar[-1], last_close, "end_of_sample")
        equity[calendar[-1]] = cash
    return {"equity": equity, "campaigns": campaigns, "fills": fills}


def ex_2021_equity(campaigns):
    """prod(1+r) over campaigns exiting >=2022-01-01 — identical to
    T122's excl_2021 LOYO cell (dropping exit-year 2021)."""
    kept = [c for c in campaigns if c["exit_day"] >= "2022"]
    return (math.prod(1 + c["return"] for c in kept), len(kept))


def leave_one_year_out(campaigns):
    """Primary cell LOYO table (T122 convention)."""
    years = sorted({c["exit_day"][:4] for c in campaigns})
    table = {}
    for y in years:
        kept = [c for c in campaigns if c["exit_day"][:4] != y]
        table[f"excl_{y}"] = {
            "equity": _r(math.prod(1 + c["return"] for c in kept), 6),
            "n_campaigns_kept": len(kept),
            "n_campaigns_dropped": len(campaigns) - len(kept),
        }
    return table


def cell_metrics(simres, calendar):
    """Compact per-cell metrics incl. ex-2021 compounded equity."""
    m = engine.arm_metrics(simres, calendar)
    ex_eq, ex_n = ex_2021_equity(simres["campaigns"])
    return {
        "n_campaigns": m["n_campaigns"],
        "final_equity": _r(m["final_equity"], 6),
        "equity_ex_2021": _r(ex_eq, 6),
        "n_campaigns_ex_2021": ex_n,
        "max_drawdown_frac": _r(m["max_drawdown_frac"]),
        "win_rate": _r(m["win_rate"]),
        "time_in_market_frac": _r(m["time_in_market_frac"]),
        "exit_reason_histogram": m["exit_reason_histogram"],
    }


def cell_key(k, shift):
    tag = f"{shift:+d}sd".replace("+0sd", "+0sd")
    return f"k={k}|band={tag}"


def run_cells(calendar, rows_by_day, dfh_pct, n_ranked, btc, meta):
    """9-cell robustness grid: k in {2,3,4} x band shift {-1,0,+1}sd."""
    cells = {}
    for k in K_LEVELS:
        for s in BAND_SHIFTS:
            lo, hi = PRIMARY_BAND_LO + s, PRIMARY_BAND_HI + s
            simres = simulate_v2(calendar, rows_by_day, dfh_pct,
                                 n_ranked, btc, meta, trail_k=k,
                                 band_lo_mult=lo, band_hi_mult=hi,
                                 use_band=True)
            cells[cell_key(k, s)] = {
                "params": {"trail_k": k,
                           "band_lo_mult": lo,
                           "band_hi_mult": hi,
                           "band_shift_sd": s},
                "metrics": cell_metrics(simres, calendar),
                "_campaigns": simres["campaigns"],
                "_sim": simres,
            }
    return cells


def analyze_cells(cells, btc_hold_equity, btc_hold_ex_2021):
    """Is the conclusion flat (robust) or peaked (lucky)?"""
    keys = sorted(cells)
    eqs = {k: cells[k]["metrics"]["final_equity"] for k in keys}
    exs = {k: cells[k]["metrics"]["equity_ex_2021"] for k in keys}
    vals = sorted(eqs.values())
    n = len(vals)
    median_eq = (vals[n // 2] if n % 2
                 else (vals[n // 2 - 1] + vals[n // 2]) / 2.0)
    ex_vals = sorted(exs.values())
    median_ex = ex_vals[n // 2]
    frac_beat = sum(1 for v in vals if v > btc_hold_equity) / n
    frac_beat_ex = sum(1 for v in ex_vals
                       if v > btc_hold_ex_2021) / n
    primary = cell_key(PRIMARY_K, 0)
    rank_of = {k: r for k, r in
               zip(sorted(keys, key=lambda k: -eqs[k]),
                   range(1, n + 1))}
    primary_rank = rank_of[primary]
    spread = (max(vals) / min(vals)) if min(vals) > 0 else None
    # flat-vs-peaked judgment: the conclusion "X vs BTC hold" is robust
    # if most cells land on the SAME side as the primary, on BOTH the
    # full sample and the ex-2021 headline; a top-decile primary rank
    # (rank 1 of 9) is the peaked flag.
    primary_side = eqs[primary] > btc_hold_equity
    primary_side_ex = exs[primary] > btc_hold_ex_2021
    agree = sum(1 for k in keys
                if (eqs[k] > btc_hold_equity) == primary_side)
    agree_ex = sum(1 for k in keys
                   if (exs[k] > btc_hold_ex_2021) == primary_side_ex)
    label = ("flat" if primary_rank > 1 and agree >= 7
             and agree_ex >= 7 else "peaked")
    return {
        "n_cells": n,
        "btc_hold_equity": _r(btc_hold_equity, 6),
        "btc_hold_equity_ex_2021": _r(btc_hold_ex_2021, 6),
        "median_final_equity": _r(median_eq, 6),
        "median_equity_ex_2021": _r(median_ex, 6),
        "min_final_equity": _r(vals[0], 6),
        "max_final_equity": _r(vals[-1], 6),
        "max_min_spread": _r(spread, 3),
        "fraction_cells_beating_btc_hold": _r(frac_beat),
        "fraction_cells_beating_btc_hold_ex_2021": _r(frac_beat_ex),
        "cells_on_primary_side_full_sample": f"{agree}/{n}",
        "cells_on_primary_side_ex_2021": f"{agree_ex}/{n}",
        "primary_cell": {
            "key": primary,
            "final_equity": eqs[primary],
            "equity_ex_2021": exs[primary],
            "rank_of_n": f"{primary_rank}/{n}",
            "rank": primary_rank,
            "is_rank1_peaked_flag": primary_rank == 1,
        },
        "conclusion_shape": label,
        "shape_rule": ("flat = primary not rank-1 AND >=7/9 cells on the "
                       "primary's side of BTC hold on BOTH the full "
                       "sample and ex-2021; peaked otherwise"),
        "equity_ranking": [
            {"rank": i + 1, "key": k, "final_equity": eqs[k],
             "equity_ex_2021": exs[k]}
            for i, k in enumerate(
                sorted(keys, key=lambda k: -eqs[k]))],
    }


def write_protocol_and_auth():
    protocol = {
        "schema_version":
            "nanojev-financial-campaign-v2-principled-protocol-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "purpose": "T123: rebuild the T119 rolling-campaign simulator "
                   "with PRINCIPLED (vol-scaled, non-hand-tuned) "
                   "parameters after T122 judged the 34.4x composite "
                   "parameter-lucky (median grid cell 0.65x, default "
                   "rank 2/36, ex-2021 equity 0.22x). Check whether the "
                   "conclusion stabilizes: flat across the 9-cell "
                   "neighbourhood = robust, peaked = lucky.",
        "cohort": {
            "path": "data/perp_pit_mega_v1/records.jsonl",
            "assets": "277 USDT-M perps, 2021-01-01 -> 2025-12-30, "
                      "close basis; includes delisted/renamed "
                      "early-stoppers",
            "btc_gate_source": "data/rc_futures_v1/BTC/BTCUSDT_1d.csv "
                               "close/close[t-20]-1",
        },
        "volatility_basis": {
            "vol20": "per-asset sample std (ddof=1) of the trailing 20 "
                     "daily log returns; needs 21 closes; None in "
                     "warmup",
            "sigma20d": "vol20*sqrt(20) — 20-bar dispersion, the same "
                        "scale dfh20 (distance from 20-bar high, "
                        "fraction) is measured on",
        },
        "definitions": {
            "concurrency": "MAX_CONCURRENT=1 — one campaign at a time, "
                           "full equity deployed at entry",
            "entry": "BTC ret20>0 AND dfh20 XS mid-rank pct>=0.8 "
                     "(top quintile, >=30 ranked) AND close>prev_close "
                     "AND dfh20 in [band_lo*sigma20d, band_hi*sigma20d] "
                     "(primary [-3sd, -0.5sd]; requires dfh20 and vol20 "
                     "non-None). NO funding filter (T122 showed it adds "
                     "nothing). Highest rank wins; fill at signal-day "
                     "close.",
            "pyramid": "unrealized>=+3% vs vwap AND gate-on AND still "
                       "top-quintile -> add 50% of position, max 2 "
                       "adds, margined perp-style (implicit leverage "
                       "<= ~1.75x)",
            "exit_order": ["data_end (no bar -> last close)",
                           "btc_regime_off (BTC ret20<=0)",
                           "vol_trail (close < max_close*(1-clip(k*"
                           "vol20, 0.02, 0.60)); skipped when vol20 "
                           "missing)",
                           "time_stop_30d (>=30 calendar days held)",
                           "rank_below_median (dfh20 pct<0.5/unranked)",
                           "end_of_sample (force-close at tail)"],
            "dropped_vs_t119": ["fixed 8% trail -> k*vol20 (k=3)",
                                "fixed [-0.12,-0.03] band -> sigma20d "
                                "multiples",
                                "funding_pct<0.8 entry filter",
                                "funding_pct>=0.95 euphoria exit"],
            "costs": "5bps per fill; funding carry skipped (signal "
                     "dropped entirely in v2)",
            "fills": "all at day-t close on day-t PIT features",
        },
        "robustness_cells": {
            "shape": "3x3 = 9 cells: trail k in {2,3,4} x band shift "
                     "in {-1sd,0,+1sd} (lo=-3+s, hi=-0.5+s)",
            "primary": "k=3, shift 0 -> band [-3sd, -0.5sd]",
            "per_cell": ["final equity", "ex-2021 compounded equity "
                         "(prod(1+r) over campaigns exiting "
                         ">=2022-01-01, T122 LOYO convention)",
                         "n campaigns", "win rate", "maxDD",
                         "exit histogram"],
            "judgment": "flat = primary not rank-1 AND >=7/9 cells on "
                        "the primary's side of BTC hold on both full "
                        "sample and ex-2021; peaked otherwise",
        },
        "arms": {
            "primary": "k=3, band [-3sd,-0.5sd] — full metrics incl. "
                       "per-year table and LOYO",
            "no_filter": "sibling: entry = gate-on AND top-quintile "
                         "only (T119 convention); identical v2 exits "
                         "at k=3",
            "btc_hold": "buy-and-hold BTC over the sim calendar, also "
                        "normalized ex-2021 for the headline "
                        "comparison",
        },
        "headline": "ex-2021 equity — does the principled config beat "
                    "BTC hold once the 2021 mania is removed?",
        "caveats": [
            "in-sample 2021-2025 calendar again — a flat cell "
            "neighbourhood is still the same regime, not a held-out "
            "market",
            "close basis + daily bars: the vol-scaled trail can gap "
            "well beyond k*vol20 on a gap day",
            "vol20/sigma20d use the same close series as everything "
            "else — no new data, but a new derived column",
            "band units: a +1sd shift makes the band top +0.5sd, i.e. "
            "dfh20<=+0.5*sigma20d — since dfh20<=0 by construction "
            "this side is effectively unbounded for that cell",
            "clamps [0.02,0.60] on k*vol20 are safety rails, not "
            "tuning dims (they bind only at vol20<0.67% or >20% "
            "daily)",
        ],
        "forbidden": ["fitting", "trading", "profitability claims",
                      "protocol edits post-run", "network"],
    }
    PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(json.dumps(protocol, indent=2, sort_keys=True)
                        + "\n", encoding="utf-8")
    sha = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    auth = {
        "schema_version": "nanojev-financial-campaign-v2-principled-"
                          "authorization-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "protocol_sha256": sha,
        "protocol_path":
            "research/financial_campaign_v2_principled_protocol_v1.json",
        "decision": "approved_for_measurement",
        "measurement_authorized": True,
        "fit_authorized": False,
        "independent_reviewer": {
            "id": "project-owner",
            "independence":
                "owner_self_authorization_not_independent_review",
            "note": "Owner directed T123: rebuild the campaign sim "
                    "with principled vol-scaled parameters and check "
                    "whether the conclusion stabilizes across a coarse "
                    "robustness neighbourhood (delegated task).",
        },
        "scope": {
            "permitted": "PIT-safe event-driven daily simulation on "
                         "data/perp_pit_mega_v1/records.jsonl plus the "
                         "BTC 1d csv: 9-cell vol-scaled grid, no-filter "
                         "sibling, BTC buy-and-hold, per the pinned "
                         "protocol.",
            "not_permitted": "No fitting/trading/profitability claims/"
                             "protocol edits; no network; no other "
                             "files modified.",
        },
        "network_model_calls": 0,
        "order_submission_authorized": False,
        "live_trading_authorized": False,
    }
    AUTH.parent.mkdir(parents=True, exist_ok=True)
    AUTH.write_text(json.dumps(auth, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return sha


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohort", type=pathlib.Path, default=COHORT)
    parser.add_argument("--btc", type=pathlib.Path, default=BTC_CSV)
    parser.add_argument("--out", type=pathlib.Path, default=OUT)
    args = parser.parse_args()

    protocol_sha = write_protocol_and_auth()
    report = {
        "schema_version": "nanojev-financial-campaign-v2-principled-v1",
        "task": "T123 principled (vol-scaled) rebuild of the T119 "
                "campaign simulator + 9-cell robustness check",
        "protocol_path":
            "research/financial_campaign_v2_principled_protocol_v1.json",
        "protocol_sha256": protocol_sha,
        "authorization_path":
            "results/financial_campaign_v2_principled_authorization_v1"
            ".json",
        "engine": "financial_campaign_sim_v1 imported for loaders/"
                  "metrics/constants; simulate_v2 re-implements the "
                  "state machine with per-asset-per-day vol-scaled "
                  "band and trail",
        "parameters": {
            "trail": "close < max_close*(1 - clip(k*vol20, "
                     f"{TRAIL_MIN_FRAC}, {TRAIL_MAX_FRAC})), primary "
                     f"k={PRIMARY_K}",
            "pullback_band": f"dfh20 in [{PRIMARY_BAND_LO}*sigma20d, "
                             f"{PRIMARY_BAND_HI}*sigma20d] primary; "
                             "sigma20d = vol20*sqrt(20)",
            "vol20": "per-asset sample std of trailing 20 daily log "
                     "returns (ddof=1, needs 21 closes)",
            "funding": "dropped entirely (entry filter AND euphoria "
                       "exit) per T122",
            "kept_from_t119": {"fee_per_fill": FEE, "top_quintile":
                               TOP_Q, "median_exit": MEDIAN,
                               "time_stop_days": TIME_STOP_DAYS,
                               "add_trigger": ADD_TRIGGER,
                               "add_frac": ADD_FRAC,
                               "max_adds": MAX_ADDS,
                               "min_ranked_per_day": MIN_RANKED},
        },
    }
    if not args.cohort.exists() or not args.btc.exists():
        report["status"] = ("SKIPPED: cohort or BTC series missing; "
                            "nothing was fabricated")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True)
                            + "\n")
        print(json.dumps({"status": report["status"]}))
        return 0

    series, meta = engine.load_cohort(args.cohort)
    btc = engine.load_btc_ret20(args.btc)
    n_vol = annotate_vol20(series)
    calendar, rows_by_day, dfh_pct, n_ranked = engine.build_day_views(
        series)
    report["cohort_description"] = {
        "symbols": len(series),
        "rows_with_vol20": n_vol,
        "span": {"first": calendar[0], "last": calendar[-1],
                 "days": len(calendar)},
    }
    report["status"] = "ran"

    # BTC hold over the sim calendar + ex-2021 normalization
    first = next((d for d in calendar if d in btc), calendar[0])
    bh = {d: btc[d]["close"] / btc[first]["close"]
          for d in calendar if d in btc}
    btc_hold_equity = bh[max(bh)]
    first_2022 = next((d for d in sorted(bh) if d >= "2022"), None)
    btc_hold_ex_2021 = (btc[max(bh)]["close"] / btc[first_2022]["close"]
                        if first_2022 else None)

    # determinism: full grid + sibling twice, serialized must match
    def run_once():
        cells = run_cells(calendar, rows_by_day, dfh_pct, n_ranked,
                          btc, meta)
        sib = simulate_v2(calendar, rows_by_day, dfh_pct, n_ranked,
                          btc, meta, trail_k=PRIMARY_K,
                          use_band=False)
        return cells, sib

    cells1, sib1 = run_once()
    cells2, sib2 = run_once()
    det_cells = json.dumps(
        {k: v["metrics"] for k, v in cells1.items()},
        sort_keys=True) == json.dumps(
        {k: v["metrics"] for k, v in cells2.items()}, sort_keys=True)
    det_sib = json.dumps(sib1, sort_keys=True) == json.dumps(
        sib2, sort_keys=True)

    primary_key = cell_key(PRIMARY_K, 0)
    primary_sim = cells1[primary_key]["_sim"]
    primary_metrics = engine.arm_metrics(primary_sim, calendar)
    ex_eq, ex_n = ex_2021_equity(primary_sim["campaigns"])
    primary_metrics["equity_ex_2021"] = _r(ex_eq, 6)
    primary_metrics["n_campaigns_ex_2021"] = ex_n
    primary_metrics["leave_one_year_out"] = leave_one_year_out(
        primary_sim["campaigns"])

    sib_metrics = engine.arm_metrics(sib1, calendar)
    sib_ex, sib_ex_n = ex_2021_equity(sib1["campaigns"])
    sib_metrics["equity_ex_2021"] = _r(sib_ex, 6)
    sib_metrics["n_campaigns_ex_2021"] = sib_ex_n

    analysis = analyze_cells(cells1, btc_hold_equity, btc_hold_ex_2021)

    report["arms"] = {
        "primary": {
            "params": cells1[primary_key]["params"],
            "metrics": primary_metrics,
            "campaign_log": primary_sim["campaigns"],
            "n_fills": len(primary_sim["fills"]),
            "deterministic_double_run": det_cells,
        },
        "no_filter": {
            "params": {"entry": "gate-on AND top-quintile only; "
                                "identical v2 exits at k=3"},
            "metrics": sib_metrics,
            "n_fills": len(sib1["fills"]),
            "deterministic_double_run": det_sib,
        },
        "btc_hold": {
            "metrics": engine.equity_metrics(bh),
            "equity_ex_2021": _r(btc_hold_ex_2021, 6),
            "ex_2021_basis": f"close({max(bh)})/close({first_2022})",
        },
    }
    report["robustness_cells"] = {
        k: {"params": v["params"], "metrics": v["metrics"]}
        for k, v in sorted(cells1.items())}
    report["robustness_analysis"] = analysis

    pm, sm = primary_metrics, sib_metrics
    beats_hold = pm["final_equity"] > btc_hold_equity
    beats_hold_ex = (ex_eq > btc_hold_ex_2021
                     if btc_hold_ex_2021 else None)
    beats_sib = pm["final_equity"] > sm["final_equity"]
    report["verdict"] = {
        "headline_question":
            "does a principled config beat BTC hold ex-2021?",
        "primary_beats_btc_hold_full_sample": beats_hold,
        "primary_beats_btc_hold_ex_2021": beats_hold_ex,
        "primary_beats_no_filter_sibling": beats_sib,
        "conclusion_shape": analysis["conclusion_shape"],
        "summary": (
            f"primary(k=3,band[-3sd,-0.5sd]) final equity "
            f"{_r(pm['final_equity'])} (n={pm['n_campaigns']}, win "
            f"{_r(pm['win_rate'])}, mdd "
            f"{_r(pm['max_drawdown_frac'])}, tim "
            f"{_r(pm['time_in_market_frac'])}); EX-2021 equity "
            f"{_r(ex_eq)} (n={ex_n}) vs BTC hold ex-2021 "
            f"{_r(btc_hold_ex_2021)}; full-sample vs BTC hold "
            f"{_r(btc_hold_equity)} -> "
            f"{'BEATS' if beats_hold_ex else 'LOSES TO'} hold ex-2021, "
            f"{'BEATS' if beats_hold else 'LOSES TO'} hold full-sample, "
            f"{'BEATS' if beats_sib else 'LOSES TO'} sibling "
            f"({_r(sm['final_equity'])}); grid shape "
            f"{analysis['conclusion_shape']} (median cell "
            f"{analysis['median_final_equity']}, median ex-2021 "
            f"{analysis['median_equity_ex_2021']}, primary rank "
            f"{analysis['primary_cell']['rank_of_n']})"),
    }
    report["honesty"] = {
        "ex_2021_is_headline": ("prod(1+r) over campaigns exiting "
                                ">=2022-01-01 (T122 excl_2021 LOYO "
                                "convention); compared to BTC hold "
                                "rebased at the first 2022 day"),
        "close_basis": ("all fills at day close; the vol-scaled trail "
                        "can realize a much larger loss than k*vol20 "
                        "on a gap day"),
        "survivorship": ("mega cohort INCLUDES delisted/renamed "
                         "early-stoppers; campaigns on them retained"),
        "funding_carry": "funding dropped entirely in v2 (no signal, "
                         "no cashflow)",
        "not_an_asof_vintage": "second-hand archive copy "
                               "(rc_futures_v1)",
        "not_live": "no orders, no account, no broker, no trading API",
        "no_profitability_claim": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True)
                        + "\n", encoding="utf-8")
    brief = {"status": "ran", "out": str(args.out),
             "verdict": report["verdict"]["summary"],
             "grid_deterministic": det_cells,
             "sibling_deterministic": det_sib}
    print(json.dumps(brief, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
