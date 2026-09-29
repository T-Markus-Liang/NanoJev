#!/usr/bin/env python3
"""Parameter-robustness grid for the rolling-campaign simulator (T122).

T119's composite arm produced final equity 34.37x vs BTC hold 3.02x on
hand-picked defaults (pullback dfh20 band [-0.12,-0.03], funding_pct<0.8,
top-quintile XS rank, +3% add x2, 8% trail, 30d time-stop). This script
asks whether that conclusion survives parameter perturbation.

ENGINE REUSE: the T119 engine is imported unmodified —
``financial_campaign_sim_v1.simulate`` reads its tunables as module
globals at call time, so each grid cell injects its parameters via
``engine.PULLBACK_LO`` etc. (no edits to the T119 file, no copied
engine, identical state machine).

GRID (3 x 3 x 4 = 36 cells, all other params held at T119 defaults):
  pullback_band : none | [-0.12,-0.03] | [-0.08,-0.02]
                  ("none" = unbounded dfh20 band; the close>prev_close
                  recovery-day requirement is RETAINED — only the band
                  is perturbed)
  funding_max   : none | 0.8 | 0.66   ("none" = threshold +inf, funded
                  symbols always pass the stealth filter)
  trail_stop    : 0.05 | 0.08 | 0.12 | none
                  ("none" = trail multiplier 0.0, never fires on a
                  positive close)
  held constant : BTC ret20>0 gate, top-quintile dfh20 XS rank,
                  close>prev_close, funding_pct>=0.95 euphoria exit,
                  +3% add trigger x2 at 50%, 30d time-stop, rank<median
                  exit, 5bps/fill, 1 campaign at a time.

PER CELL: n campaigns, final equity, max drawdown, win rate, per-year
year-end equity vector. ANALYSIS: median equity across the grid vs BTC
hold; fraction of cells beating hold; Spearman rank correlation of each
perturbed parameter vs final equity plus encoding-free per-level median
equities; best/worst cells; rank of the DEFAULT cell (top-decile =
overfit flag, middle = robust); leave-one-year-out table for the
default cell (excl-2021 = the honest post-mania number).

Determinism: the full grid is simulated twice and the serialized cell
metrics must be identical. Artifacts: a frozen protocol
(``research/financial_campaign_grid_protocol_v1.json``) and owner
self-authorization (``results/financial_campaign_grid_authorization_v1.json``)
are written BEFORE measurement. Measurement only: no fitting, no
trading, no network.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import pathlib
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import financial_campaign_sim_v1 as engine  # noqa: E402  (T119 engine)

ROOT = pathlib.Path(__file__).resolve().parent.parent
COHORT = ROOT / "data/perp_pit_mega_v1/records.jsonl"
BTC_CSV = ROOT / "data/rc_futures_v1/BTC/BTCUSDT_1d.csv"
OUT = ROOT / "results/financial_campaign_grid_v1.json"
PROTOCOL = ROOT / "research/financial_campaign_grid_protocol_v1.json"
AUTH = ROOT / "results/financial_campaign_grid_authorization_v1.json"

INF = float("inf")

# ---- grid axes ---------------------------------------------------------
# pullback_band levels: label -> (PULLBACK_LO, PULLBACK_HI)
PULLBACK_LEVELS = {
    "none": (-INF, INF),            # band unbounded; recovery day kept
    "-0.12,-0.03": (-0.12, -0.03),  # T119 default
    "-0.08,-0.02": (-0.08, -0.02),  # tighter band
}
# funding_max levels: label -> FUND_ENTRY_MAX (inf = filter off)
FUNDING_LEVELS = {
    "none": INF,
    "0.8": 0.8,                     # T119 default
    "0.66": 0.66,
}
# trail_stop levels: label -> TRAIL multiplier (0.0 = never fires)
TRAIL_LEVELS = {
    "0.05": 0.95,
    "0.08": 0.92,                   # T119 default
    "0.12": 0.88,
    "none": 0.0,
}
DEFAULT_CELL = ("-0.12,-0.03", "0.8", "0.08")

# ordinal encodings for Spearman (monotone keys; per-level medians in the
# report are encoding-free)
PULLBACK_ORDER = {"-0.08,-0.02": 0, "-0.12,-0.03": 1, "none": 2}  # width
FUNDING_ORDER = {"0.66": 0, "0.8": 1, "none": 2}                  # thresh
TRAIL_ORDER = {"0.05": 0, "0.08": 1, "0.12": 2, "none": 3}        # loose


def _r(x, nd=4):
    return round(x, nd) if isinstance(x, float) else x


def apply_cell(pb_label, fund_label, trail_label):
    """Inject a grid cell's parameters into the T119 engine module."""
    lo, hi = PULLBACK_LEVELS[pb_label]
    engine.PULLBACK_LO, engine.PULLBACK_HI = lo, hi
    engine.FUND_ENTRY_MAX = FUNDING_LEVELS[fund_label]
    engine.TRAIL = TRAIL_LEVELS[trail_label]


def year_end_equity(equity):
    """{year: equity on last sim day of that year}."""
    out = {}
    for day in sorted(equity):
        out[day[:4]] = equity[day]
    return {y: _r(v, 6) for y, v in sorted(out.items())}


def cell_metrics(simres, calendar):
    """n campaigns, final equity, maxDD, win%, per-year equity vector."""
    m = engine.arm_metrics(simres, calendar)
    return {
        "n_campaigns": m["n_campaigns"],
        "final_equity": _r(m["final_equity"], 6),
        "max_drawdown_frac": _r(m["max_drawdown_frac"]),
        "win_rate": _r(m["win_rate"]),
        "time_in_market_frac": _r(m["time_in_market_frac"]),
        "year_end_equity": year_end_equity(simres["equity"]),
    }


def run_grid(calendar, rows_by_day, dfh_pct, n_ranked, btc, meta):
    """Simulate every grid cell once. Returns {cell_key: metrics}."""
    cells = {}
    for pb in PULLBACK_LEVELS:
        for fund in FUNDING_LEVELS:
            for trail in TRAIL_LEVELS:
                apply_cell(pb, fund, trail)
                simres = engine.simulate(
                    calendar, rows_by_day, dfh_pct, n_ranked, btc, meta,
                    composite_entry=True)
                key = f"pb={pb}|fund={fund}|trail={trail}"
                cells[key] = {"params": {"pullback_band": pb,
                                         "funding_max": fund,
                                         "trail_stop": trail},
                              "metrics": cell_metrics(simres, calendar),
                              "_campaigns": simres["campaigns"]}
    return cells


def ranks(xs):
    """Average ranks (1-based) handling ties."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return out


def spearman(xs, ys):
    """Spearman rho via Pearson on average ranks (None if degenerate)."""
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    rx, ry = ranks(xs), ranks(ys)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    return cov / math.sqrt(vx * vy) if vx and vy else None


def level_medians(cells, param):
    """Encoding-free per-level median final equity."""
    groups = defaultdict(list)
    for c in cells.values():
        groups[c["params"][param]].append(c["metrics"]["final_equity"])
    out = {}
    for lvl, vals in groups.items():
        vals = sorted(vals)
        n = len(vals)
        med = (vals[n // 2] if n % 2
               else (vals[n // 2 - 1] + vals[n // 2]) / 2.0)
        out[lvl] = {"n": n, "median_final_equity": _r(med, 6),
                    "min": _r(vals[0], 6), "max": _r(vals[-1], 6)}
    return dict(sorted(out.items(), key=lambda kv: str(kv[0])))


def leave_one_year_out(campaigns):
    """Default cell: compounded equity excluding each exit-year in turn
    (same convention as T119's excl-early-stopped sensitivity:
    prod(1+r) over kept campaigns)."""
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


def write_protocol_and_auth():
    protocol = {
        "schema_version": "nanojev-financial-campaign-grid-protocol-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "purpose": "T122: parameter-robustness grid for the T119 "
                   "rolling-campaign simulator — does the composite "
                   "34.4x vs BTC 3.0x result survive perturbation of "
                   "pullback_band x funding_max x trail_stop, or was it "
                   "parameter-lucky?",
        "engine": {
            "source": "scripts/financial_campaign_sim_v1.py imported "
                      "unmodified; simulate() reads tunables as module "
                      "globals at call time — each cell injects "
                      "PULLBACK_LO/HI, FUND_ENTRY_MAX, TRAIL. No copied "
                      "engine, no edits to the T119 file.",
            "state_machine": "identical to the pinned T119 protocol "
                             "(research/financial_campaign_sim_protocol"
                             "_v1.json): entry gate BTC ret20>0 + "
                             "top-quintile dfh20 XS rank + pullback "
                             "state + stealth funding; pyramid +3% x2 "
                             "margined; exits data_end/btc_off/"
                             "funding_euphoria/trail/time_stop/"
                             "rank_below_median; 5bps/fill; 1 campaign.",
        },
        "grid": {
            "pullback_band": {
                "levels": ["none", "[-0.12,-0.03]", "[-0.08,-0.02]"],
                "none_semantics": "dfh20 band unbounded "
                                  "(-inf,+inf); the close>prev_close "
                                  "recovery-day requirement is "
                                  "RETAINED — only the band varies",
            },
            "funding_max": {
                "levels": ["none", "0.8", "0.66"],
                "none_semantics": "FUND_ENTRY_MAX=+inf — funded symbols "
                                  "always pass the stealth filter",
            },
            "trail_stop": {
                "levels": ["0.05", "0.08", "0.12", "none"],
                "none_semantics": "TRAIL=0.0 — close < 0*max_close "
                                  "never fires on a positive close",
            },
            "held_constant": ["BTC ret20>0 gate", "top-quintile dfh20",
                              "close>prev_close recovery day",
                              "funding_pct>=0.95 euphoria exit",
                              "+3% add trigger, x2 adds at 50%, "
                              "margined", "30d time-stop",
                              "rank<median exit", "5bps/fill",
                              "MAX_CONCURRENT=1"],
            "n_cells": 36,
        },
        "metrics_per_cell": ["n campaigns", "final compounded equity",
                             "max drawdown (daily equity)", "win rate",
                             "per-year year-end equity vector"],
        "analysis": ["median equity across grid vs BTC hold",
                     "fraction of cells beating BTC hold",
                     "Spearman rank corr per perturbed param vs final "
                     "equity (monotone ordinal encodings) + "
                     "encoding-free per-level medians",
                     "best/worst cells",
                     "DEFAULT cell rank — top decile = overfit flag, "
                     "middle = robust",
                     "leave-one-year-out table for the default cell "
                     "(excl-2021 = post-mania number)"],
        "determinism": "full grid simulated twice; serialized cell "
                       "metrics must be identical",
        "caveats": [
            "in-sample perturbation on the same 2021-2025 calendar — "
            "a robust cell is still the same lucky regime, not a "
            "held-out market",
            "close basis + daily bars: coarse intra-day stops (a 5% "
            "trail can gap well beyond 5%)",
            "no_filter arm not re-gridded — it already collapsed to "
            "~0 under defaults; this grid probes the composite only",
            "36 cells is bounded by design; ungridded params (add "
            "trigger, time-stop, quintile cut) stay at T119 defaults",
        ],
        "forbidden": ["fitting", "trading", "profitability claims",
                      "protocol edits post-run", "network"],
    }
    PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(json.dumps(protocol, indent=2, sort_keys=True)
                        + "\n", encoding="utf-8")
    sha = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    auth = {
        "schema_version": "nanojev-financial-campaign-grid-"
                          "authorization-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "protocol_sha256": sha,
        "protocol_path":
            "research/financial_campaign_grid_protocol_v1.json",
        "decision": "approved_for_measurement",
        "measurement_authorized": True,
        "fit_authorized": False,
        "independent_reviewer": {
            "id": "project-owner",
            "independence":
                "owner_self_authorization_not_independent_review",
            "note": "Owner directed T122: grid the T119 campaign "
                    "simulator's entry/exit parameters to test whether "
                    "the 34.4x composite result is parameter-lucky "
                    "(delegated task).",
        },
        "scope": {
            "permitted": "36-cell parameter perturbation of the pinned "
                         "T119 engine on data/perp_pit_mega_v1/"
                         "records.jsonl + BTC 1d csv; per-cell metrics, "
                         "rank-correlation analysis, leave-one-year-out "
                         "on the default cell.",
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


def analyze(cells, btc_hold_equity):
    """Median vs hold, frac beating hold, Spearman, best/worst, default
    rank, per-level medians."""
    keys = sorted(cells)
    eqs = {k: cells[k]["metrics"]["final_equity"] for k in keys}
    vals = sorted(eqs.values())
    n = len(vals)
    median_eq = (vals[n // 2] if n % 2
                 else (vals[n // 2 - 1] + vals[n // 2]) / 2.0)
    frac_beat = sum(1 for v in vals if v > btc_hold_equity) / n

    rank_of = {k: r for k, r in
               zip(sorted(keys, key=lambda k: -eqs[k]),
                   range(1, n + 1))}
    ordered = sorted(keys, key=lambda k: -eqs[k])
    best, worst = ordered[0], ordered[-1]

    default_key = (f"pb={DEFAULT_CELL[0]}|fund={DEFAULT_CELL[1]}"
                   f"|trail={DEFAULT_CELL[2]}")
    default_rank = rank_of[default_key]
    top_decile_cut = max(1, int(0.1 * n))          # 36 -> top 3

    spearmans = {}
    for param, order in (("pullback_band", PULLBACK_ORDER),
                         ("funding_max", FUNDING_ORDER),
                         ("trail_stop", TRAIL_ORDER)):
        xs = [order[cells[k]["params"][param]] for k in keys]
        rho = spearman(xs, [eqs[k] for k in keys])
        spearmans[param] = {
            "spearman_rho_vs_final_equity": _r(rho) if rho is not None
            else None,
            "ordinal_encoding": {lvl: order[lvl] for lvl in
                                 sorted(order, key=order.get)},
            "per_level": level_medians(cells, param),
        }

    return {
        "n_cells": n,
        "btc_hold_equity": _r(btc_hold_equity, 6),
        "median_final_equity": _r(median_eq, 6),
        "median_beats_btc_hold": median_eq > btc_hold_equity,
        "fraction_of_cells_beating_btc_hold": _r(frac_beat),
        "best_cell": {"key": best,
                      "params": cells[best]["params"],
                      "final_equity": eqs[best]},
        "worst_cell": {"key": worst,
                       "params": cells[worst]["params"],
                       "final_equity": eqs[worst]},
        "default_cell": {
            "key": default_key,
            "final_equity": eqs[default_key],
            "rank_of_n": f"{default_rank}/{n}",
            "rank": default_rank,
            "top_decile_cutoff_rank": top_decile_cut,
            "in_top_decile": default_rank <= top_decile_cut,
            "in_top_quartile": default_rank <= int(0.25 * n),
            "interpretation": ("top-decile rank = overfit flag; "
                               "middle of grid = robust default"),
        },
        "rank_correlations": spearmans,
        "equity_ranking": [
            {"rank": i + 1, "key": k, "final_equity": eqs[k]}
            for i, k in enumerate(ordered)],
    }


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
        "schema_version": "nanojev-financial-campaign-grid-v1",
        "task": "T122 parameter-robustness grid for the T119 campaign "
                "simulator",
        "protocol_path":
            "research/financial_campaign_grid_protocol_v1.json",
        "protocol_sha256": protocol_sha,
        "authorization_path":
            "results/financial_campaign_grid_authorization_v1.json",
        "engine": "financial_campaign_sim_v1.simulate imported "
                  "unmodified; per-cell module-global injection",
        "grid": {
            "pullback_band": list(PULLBACK_LEVELS),
            "funding_max": list(FUNDING_LEVELS),
            "trail_stop": list(TRAIL_LEVELS),
            "n_cells": (len(PULLBACK_LEVELS) * len(FUNDING_LEVELS)
                        * len(TRAIL_LEVELS)),
            "default_cell": ("pb=%s|fund=%s|trail=%s" % DEFAULT_CELL),
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
    calendar, rows_by_day, dfh_pct, n_ranked = engine.build_day_views(
        series)
    report["cohort_description"] = {
        "symbols": len(series),
        "span": {"first": calendar[0], "last": calendar[-1],
                 "days": len(calendar)},
    }
    report["status"] = "ran"

    # BTC hold over the sim calendar (same basis as T119)
    first = next((d for d in calendar if d in btc), calendar[0])
    bh = {d: btc[d]["close"] / btc[first]["close"]
          for d in calendar if d in btc}
    btc_hold_equity = bh[max(bh)]

    # determinism: run the full grid twice, compare serialized metrics
    cells1 = run_grid(calendar, rows_by_day, dfh_pct, n_ranked, btc,
                      meta)
    cells2 = run_grid(calendar, rows_by_day, dfh_pct, n_ranked, btc,
                      meta)
    m1 = {k: c["metrics"] for k, c in cells1.items()}
    m2 = {k: c["metrics"] for k, c in cells2.items()}
    deterministic = json.dumps(m1, sort_keys=True) == json.dumps(
        m2, sort_keys=True)

    default_key = report["grid"]["default_cell"]
    loyo = leave_one_year_out(cells1[default_key]["_campaigns"])
    for c in cells1.values():
        del c["_campaigns"]

    report["btc_hold_equity"] = _r(btc_hold_equity, 6)
    report["cells"] = cells1
    report["deterministic_double_run"] = deterministic
    report["leave_one_year_out_default_cell"] = {
        "convention": "prod(1+campaign_return) over campaigns whose "
                      "exit_day year != the excluded year — same as "
                      "T119's excl-early-stopped sensitivity",
        "table": loyo,
        "equity_excl_2021": loyo.get("excl_2021", {}).get("equity"),
    }
    report["analysis"] = analyze(cells1, btc_hold_equity)

    an = report["analysis"]
    dc = an["default_cell"]
    frac = an["fraction_of_cells_beating_btc_hold"]
    if an["median_beats_btc_hold"] and not dc["in_top_decile"]:
        verdict = "robust"
    elif dc["in_top_decile"] and not an["median_beats_btc_hold"]:
        verdict = "parameter_lucky"
    else:
        verdict = "partially_robust"
    dims = sorted(an["rank_correlations"].items(),
                  key=lambda kv: -(abs(kv[1][
                      "spearman_rho_vs_final_equity"])
                      if kv[1]["spearman_rho_vs_final_equity"]
                      is not None else -1))
    report["verdict"] = {
        "label": verdict,
        "summary": (
            f"median grid equity {_r(an['median_final_equity'])} vs BTC "
            f"hold {_r(btc_hold_equity)}; {frac:.1%} of {an['n_cells']} "
            f"cells beat hold; default cell equity "
            f"{_r(dc['final_equity'])} ranks {dc['rank_of_n']} "
            f"({'TOP DECILE — overfit flag' if dc['in_top_decile'] else 'not top-decile'}); "
            f"excl-2021 default equity "
            f"{_r(report['leave_one_year_out_default_cell']['equity_excl_2021'])}; "
            f"param |rho| order: " + ", ".join(
                f"{k}={v['spearman_rho_vs_final_equity']}"
                for k, v in dims)),
        "dims_ranked_by_abs_spearman": [k for k, _ in dims],
    }
    report["honesty"] = {
        "in_sample_perturbation": ("all 36 cells share the 2021-2025 "
                                   "calendar — a robust median is still "
                                   "the same regime, not held-out "
                                   "evidence"),
        "close_basis": "daily-close fills; intra-day stops are coarse",
        "not_live": "no orders, no account, no broker, no trading API",
        "no_profitability_claim": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True)
                        + "\n", encoding="utf-8")
    brief = {"status": "ran", "out": str(args.out),
             "verdict": report["verdict"]["label"],
             "summary": report["verdict"]["summary"],
             "deterministic": deterministic}
    print(json.dumps(brief, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
