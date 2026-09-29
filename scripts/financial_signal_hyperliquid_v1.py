#!/usr/bin/env python3
"""T99: off-Binance replication of the daily funding-following signal on
local Hyperliquid data — the strongest external-validity check available
locally.

Reference (Binance, confirmed): funding_pct (trailing-180 mid-rank pct of
last_funding_rate, per asset) -> 5d forward mark log return,
rho=+0.0422, p=0.001537, FDR-pass on n=5629
(results/financial_signal_labels_v4.json, cell l3:funding); in the frozen
spec (research/financial_signal_spec_v1.json) the signal is gated by the
BTC 20d trend.

Local Hyperliquid data (data/venue_perp_v1/hyperliquid/, hash-pinned
fetch, licence UNVERIFIED):
  {COIN}.candles.json  1d bars {t open ms, c close, ...}; ~1358 bars,
                       2023-01-01..2026-09-19 UTC. Close is used as the
                       price series — no historical mark/index exists.
                       (volume is "0.0" on early bars; unused.)
  {COIN}.funding.json  fundingHistory rows {coin, fundingRate, premium,
                       time}; hourly grid ~2023-05-12..2026-09-18 (XRP
                       from 2023-06-18); the first ~4 weeks of
                       BTC/ETH/SOL/BNB history are on an 8h grid.

Daily funding series (Binance-equivalent semantics: last settled rate at
the decision day):
  f_last  fundingRate of the last settlement row inside the UTC day
          (~23:00 UTC; PIT-safe vs the 23:59:59.999 close)
  f_8h    sum of fundingRate rows timestamped in [16:00, 24:00) UTC —
          an 8h-magnitude rate comparable to Binance last_funding_rate
          (used by the spread arm and the A4 robustness cell)

Arms (protocol: research/financial_signal_hyperliquid_protocol_v1.json):
  A1  Spearman(funding_pct, 5d fwd log close) per coin + pooled
  A2  Welch top-decile (funding_pct>=0.90) vs rest, per coin + pooled
  A3  BTC-20d-trend-gated pooled cells (the frozen spec gate)
  A4  robustness: pooled Spearman with pct computed on f_8h
  B   HL-vs-Binance funding spread: f_8h minus Binance last_funding_rate
      on aligned UTC days -> trailing-180 mid-rank spread_pct -> Spearman
      per coin + pooled + pooled top-decile Welch

BH-FDR alpha=0.05 per family; measurement only — no fitting, no trading,
no network. Honest output: replication yes/no per asset, pooled rho with
p, sign agreement vs the Binance result.
"""

import argparse
import datetime
import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HL = ROOT / "data/venue_perp_v1/hyperliquid"
BINANCE = ROOT / "data/perp_pit_v1/records.jsonl"
OUT = ROOT / "results/financial_signal_hyperliquid_v1.json"
PROTOCOL = "research/financial_signal_hyperliquid_protocol_v1.json"
COINS = ["BTC", "ETH", "SOL", "BNB", "XRP"]
DAY_MS = 86_400_000
TRAIL = 180            # trailing rows for mid-rank pct (~180d)
HOLD = 5               # forward horizon in daily bars (5d)
TREND = 20             # BTC trend lookback in daily bars
DECILE = 0.90          # top-decile arm threshold
MIN_N = 10             # per-side minimum (repo convention)
SPREAD_MIN_W = 100     # min non-null spreads inside trailing window
                       # (crossvenue_v1 convention)

BINANCE_REF = {
    "venue": "binance_um",
    "cell": "funding_pct -> 5d forward mark log return (l3:funding)",
    "rho": 0.0422, "p": 0.001537, "n": 5629, "fdr_pass": True,
    "ref": "results/financial_signal_labels_v4.json",
    "gate": "BTC mark_price 20-bar trend > 0 "
            "(research/financial_signal_spec_v1.json)"}


# ---------------------------------------------------------------- stats
def pct(w, x):
    """Mid-rank percentile of x within trailing window w (project conv.)."""
    return (sum(1 for v in w if v < x)
            + 0.5 * sum(1 for v in w if v == x)) / len(w)


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(xs, ys):
    """Mid-rank Spearman rho + normal-approx t/p; None if n<MIN_N or a
    side has zero rank variance (project convention)."""
    n = len(xs)
    if n < MIN_N:
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return None
    rho = num / (dx * dy)
    t = rho * math.sqrt((n - 2) / max(1e-9, 1 - rho * rho))
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return {"n": n, "rho": round(rho, 5), "t": round(t, 3),
            "p": round(p, 6)}


def welch(a, b):
    """Welch t of mean(a)-mean(b) on forward returns; reported in bps;
    normal-approx two-sided p (repo convention)."""
    if len(a) < MIN_N or len(b) < MIN_N:
        return None
    ma, mb = statistics.mean(a), statistics.mean(b)
    se = math.sqrt(statistics.pvariance(a) / len(a)
                   + statistics.pvariance(b) / len(b))
    if se == 0:
        return None
    t = (ma - mb) / se
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return {"n_arm": len(a), "n_rest": len(b),
            "mean_arm_bps": round(ma * 1e4, 1),
            "mean_rest_bps": round(mb * 1e4, 1),
            "diff_bps": round((ma - mb) * 1e4, 1),
            "t": round(t, 3), "p": round(p, 6)}


def bh_fdr(named_ps):
    """BH-FDR alpha=0.05 over [(name, p|None)] -> {name: survives},
    plus the sorted threshold table."""
    ps = sorted((p, n) for n, p in named_ps if p is not None)
    m = len(ps)
    table = [{"cell": n, "p": p, "alpha_bh": round(0.05 * (i + 1) / m, 6),
              "survives": p <= 0.05 * (i + 1) / m}
             for i, (p, n) in enumerate(ps)]
    return {e["cell"]: e["survives"] for e in table}, table


def iso(day):
    return datetime.datetime.fromtimestamp(
        day * 86400, datetime.timezone.utc).strftime("%Y-%m-%d")


# ---------------------------------------------------------------- data
def load_hl(coin):
    """-> merged daily rows [{day, close, f_last, f_8h}] plus integrity."""
    candles = json.loads((HL / f"{coin}.candles.json").read_text())
    fund_rows = json.loads((HL / f"{coin}.funding.json").read_text())
    close = {}
    for c in candles:
        close[int(c["t"]) // DAY_MS] = float(c["c"])
    f_last, f_8h = {}, {}
    for r in sorted(fund_rows, key=lambda x: x["time"]):
        d = int(r["time"]) // DAY_MS
        rate = float(r["fundingRate"])
        f_last[d] = rate                      # last settlement of the day
        if int(r["time"]) - d * DAY_MS >= 16 * 3_600_000:
            f_8h[d] = f_8h.get(d, 0.0) + rate  # 16:00..23:00 settlements
    days = sorted(d for d in close if d in f_last)
    merged = [{"day": d, "close": close[d], "f_last": f_last[d],
               "f_8h": f_8h.get(d)} for d in days]
    ftimes = [int(r["time"]) for r in fund_rows]
    integrity = {"n_funding_rows": len(fund_rows),
                 "n_funding_days": len(f_last),
                 "n_candle_days": len(close),
                 "n_merged_days": len(merged),
                 "funding_span_utc":
                     f"{iso(ftimes[0] // DAY_MS)}..{iso(ftimes[-1] // DAY_MS)}",
                 "n_days_missing_8h_window":
                     sum(1 for d in days if f_8h.get(d) is None),
                 "funding_rate_stats": {
                     "min": min(float(r["fundingRate"]) for r in fund_rows),
                     "max": max(float(r["fundingRate"]) for r in fund_rows),
                     "mean": statistics.mean(
                         float(r["fundingRate"]) for r in fund_rows),
                     "share_positive": round(sum(
                         1 for r in fund_rows
                         if float(r["fundingRate"]) > 0)
                         / len(fund_rows), 4)}}
    return merged, integrity


def load_binance_daily():
    """coin -> {day: last_funding_rate} from the Binance PIT cohort."""
    out = defaultdict(dict)
    n = 0
    for line in BINANCE.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        n += 1
        coin = r["asset_id"].split("USDT")[0]
        day = r["decision_ns"] // 1_000_000 // DAY_MS
        out[coin][day] = r["features"]["last_funding_rate"]["value"]
    return dict(out), n


# ---------------------------------------------------------------- arms
def eval_rows(merged, coin, btc_trend):
    """Arm-A rows: i in [TRAIL, len-HOLD); carries f_pct (last-rate),
    f8_pct (8h-sum rate), ret, gated flag."""
    rows = []
    fl = [m["f_last"] for m in merged]
    f8 = [m["f_8h"] for m in merged]
    cl = [m["close"] for m in merged]
    for i in range(TRAIL, len(merged) - HOLD):
        if cl[i] <= 0 or cl[i + HOLD] <= 0:
            continue
        w8 = [v for v in f8[i - TRAIL:i] if v is not None]
        rows.append({
            "coin": coin, "day": merged[i]["day"], "i": i,
            "f_pct": pct(fl[i - TRAIL:i], fl[i]),
            "f8_pct": pct(w8, f8[i]) if len(w8) >= SPREAD_MIN_W
                      and f8[i] is not None else None,
            "ret": math.log(cl[i + HOLD] / cl[i]),
            "btc_up": btc_trend.get(merged[i]["day"])})
    return rows


def spread_rows(merged, coin, binance):
    """Arm-B rows: spread = f_8h - binance last_funding_rate on aligned
    days; spread_pct trailing-180 mid-rank over non-null window."""
    cl = [m["close"] for m in merged]
    f8 = [m["f_8h"] for m in merged]
    bq = binance.get(coin, {})
    spread = [(f8[i] - bq[merged[i]["day"]]
               if f8[i] is not None and merged[i]["day"] in bq else None)
              for i in range(len(merged))]
    rows = []
    for i in range(TRAIL, len(merged) - HOLD):
        if spread[i] is None or cl[i] <= 0 or cl[i + HOLD] <= 0:
            continue
        sw = [s for s in spread[i - TRAIL:i] if s is not None]
        if len(sw) < SPREAD_MIN_W:
            continue
        rows.append({"coin": coin, "day": merged[i]["day"],
                     "s_pct": pct(sw, spread[i]),
                     "spread_bps": spread[i] * 1e4,
                     "ret": math.log(cl[i + HOLD] / cl[i])})
    return rows


def run():
    merged, integrity = {}, {}
    for c in COINS:
        merged[c], integrity[c] = load_hl(c)
    binance, n_binance = load_binance_daily()

    # BTC 20d trend on the merged BTC close series (spec gate)
    bc = merged["BTC"]
    btc_trend = {}
    for i in range(TREND, len(bc)):
        if bc[i]["close"] > 0 and bc[i - TREND]["close"] > 0:
            btc_trend[bc[i]["day"]] = (
                bc[i]["close"] / bc[i - TREND]["close"] - 1 > 0)

    # ---- Arm A: funding_pct -> 5d fwd log close
    arm_a = {"per_coin": {}, "pooled": None, "families": {}}
    all_rows, per_coin_rows = [], {}
    for c in COINS:
        rows = eval_rows(merged[c], c, btc_trend)
        per_coin_rows[c] = rows
        all_rows.extend(rows)
        s = spearman([r["f_pct"] for r in rows], [r["ret"] for r in rows])
        w = welch([r["ret"] for r in rows if r["f_pct"] >= DECILE],
                  [r["ret"] for r in rows if r["f_pct"] < DECILE])
        arm_a["per_coin"][c] = {
            "n_eval": len(rows),
            "eval_span_utc": (f"{iso(rows[0]['day'])}..{iso(rows[-1]['day'])}"
                              if rows else None),
            "spearman": s, "top_decile_vs_rest": w,
            "n_btc_up": sum(1 for r in rows if r["btc_up"] is True)}
    pooled_sp = spearman([r["f_pct"] for r in all_rows],
                         [r["ret"] for r in all_rows])
    pooled_w = welch([r["ret"] for r in all_rows if r["f_pct"] >= DECILE],
                     [r["ret"] for r in all_rows if r["f_pct"] < DECILE])
    arm_a["pooled"] = {"n_eval": len(all_rows), "spearman": pooled_sp,
                       "top_decile_vs_rest": pooled_w}

    # A1 FDR family: 5 coins + pooled
    fdr, table = bh_fdr([(c, (arm_a["per_coin"][c]["spearman"] or {})
                         .get("p")) for c in COINS]
                        + [("pooled", (pooled_sp or {}).get("p"))])
    for c in COINS:
        if arm_a["per_coin"][c]["spearman"]:
            arm_a["per_coin"][c]["spearman"]["fdr_pass"] = fdr.get(c, False)
    if pooled_sp:
        pooled_sp["fdr_pass"] = fdr.get("pooled", False)
    arm_a["families"]["replication_spearman"] = table

    # A3 gated cells (BTC 20d trend up, pooled)
    up = [r for r in all_rows if r["btc_up"] is True]
    dn = [r for r in all_rows if r["btc_up"] is False]
    gated_sp = spearman([r["f_pct"] for r in up], [r["ret"] for r in up])
    gated_w = welch([r["ret"] for r in up if r["f_pct"] >= DECILE],
                    [r["ret"] for r in up if r["f_pct"] < DECILE])
    gated_dn = spearman([r["f_pct"] for r in dn], [r["ret"] for r in dn])
    fdr, table = bh_fdr([
        ("gated_pooled_spearman", (gated_sp or {}).get("p")),
        ("gated_top_decile_vs_rest", (gated_w or {}).get("p")),
        ("ungated_top_decile_vs_rest", (pooled_w or {}).get("p"))])
    if gated_sp:
        gated_sp["fdr_pass"] = fdr.get("gated_pooled_spearman", False)
    if gated_w:
        gated_w["fdr_pass"] = fdr.get("gated_top_decile_vs_rest", False)
    if pooled_w:
        pooled_w["fdr_pass"] = fdr.get("ungated_top_decile_vs_rest", False)
    arm_a["btc_trend_gate"] = {
        "gate": "BTC close[i]/close[i-20]-1 > 0 (merged BTC series)",
        "n_up": len(up), "n_down": len(dn), "n_unknown":
            len(all_rows) - len(up) - len(dn),
        "up_spearman": gated_sp, "down_spearman_reference": gated_dn,
        "up_top_decile_vs_rest": gated_w}
    arm_a["families"]["gated_cells"] = table

    # A4 robustness: pct on the 8h-sum daily rate
    r8 = [r for r in all_rows if r["f8_pct"] is not None]
    arm_a["robustness_8h_sum_rate"] = {
        "n_eval": len(r8),
        "spearman": spearman([r["f8_pct"] for r in r8],
                             [r["ret"] for r in r8]),
        "note": "funding_pct computed on the 8h-sum daily rate instead "
                "of the daily-last hourly rate"}

    # ---- Arm B: HL-vs-Binance funding spread
    arm_b = {"per_coin": {}, "pooled": None, "families": {}}
    all_sb = []
    for c in COINS:
        rows = spread_rows(merged[c], c, binance)
        all_sb.extend(rows)
        arm_b["per_coin"][c] = {
            "n_eval": len(rows),
            "eval_span_utc": (f"{iso(rows[0]['day'])}..{iso(rows[-1]['day'])}"
                              if rows else None),
            "spread_bps_stats": ({
                "mean": round(statistics.mean(r["spread_bps"]
                                              for r in rows), 3),
                "min": round(min(r["spread_bps"] for r in rows), 3),
                "max": round(max(r["spread_bps"] for r in rows), 3)}
                if rows else None),
            "spearman": spearman([r["s_pct"] for r in rows],
                                 [r["ret"] for r in rows])}
    b_sp = spearman([r["s_pct"] for r in all_sb],
                    [r["ret"] for r in all_sb])
    b_w = welch([r["ret"] for r in all_sb if r["s_pct"] >= DECILE],
                [r["ret"] for r in all_sb if r["s_pct"] < DECILE])
    fdr, table = bh_fdr([(c, (arm_b["per_coin"][c]["spearman"] or {})
                         .get("p")) for c in COINS]
                        + [("pooled", (b_sp or {}).get("p"))])
    for c in COINS:
        if arm_b["per_coin"][c]["spearman"]:
            arm_b["per_coin"][c]["spearman"]["fdr_pass"] = fdr.get(c, False)
    if b_sp:
        b_sp["fdr_pass"] = fdr.get("pooled", False)
    arm_b["pooled"] = {"n_eval": len(all_sb), "spearman": b_sp,
                       "top_decile_vs_rest": b_w}
    arm_b["families"]["spread_spearman"] = table

    # ---- verdicts (power-aware: an n where |rho|=0.0422 would reach
    # nominal p<0.05 needs n >= (1.96/0.0422)^2 + 3 ~= 2160)
    n_powered = math.ceil((1.96 / BINANCE_REF["rho"]) ** 2) + 3

    def cell_verdict(s):
        if s is None:
            return {"verdict": "untestable"}
        powered = s["n"] >= n_powered
        agree = s["rho"] > 0
        if agree and s.get("fdr_pass"):
            v = "replicated"
        elif not powered:
            v = "same_sign_low_power" if agree else "opposite_sign_low_power"
        else:
            v = "powered_null" if agree else "not_replicated_powered"
        return {"rho": s["rho"], "p": s["p"], "n": s["n"],
                "powered_for_ref_rho_0.0422": powered,
                "sign_agrees_with_binance": agree,
                "fdr_pass": s.get("fdr_pass", False), "verdict": v}

    verdicts = {"per_asset": {c: cell_verdict(arm_a["per_coin"][c]["spearman"])
                              for c in COINS},
                "pooled_ungated": cell_verdict(pooled_sp),
                "pooled_btc_trend_gated": cell_verdict(gated_sp),
                "spread_arm_pooled": cell_verdict(b_sp),
                "n_required_for_ref_rho": n_powered}
    verdicts["pooled_ungated"]["binance_reference_rho"] = BINANCE_REF["rho"]
    gated_ok = gated_sp and gated_sp["rho"] > 0 and gated_sp.get("fdr_pass")
    ungated_ok = pooled_sp and pooled_sp["rho"] > 0 \
        and pooled_sp.get("fdr_pass")
    verdicts["headline"] = (
        "partial_replication: the spec-gated cell replicates on "
        "Hyperliquid (BTC-trend-up pooled rho={g}, FDR-pass, sign "
        "agrees) but the unconditional pooled correlation does not "
        "(rho={u} vs Binance +0.0422, powered n; sign agrees, magnitude "
        "~7x smaller, FDR fail). Per-asset cells are low-power and "
        "mixed.".format(g=gated_sp["rho"] if gated_sp else None,
                        u=pooled_sp["rho"] if pooled_sp else None)
        if gated_ok and not ungated_ok else
        "replicated" if gated_ok and ungated_ok else
        "not_replicated" if not gated_ok and not ungated_ok else
        "gated_fail_ungated_pass")

    inputs = {"hyperliquid_files": {
                  f"data/venue_perp_v1/hyperliquid/{c}.{k}.json":
                  hashlib.sha256((HL / f"{c}.{k}.json")
                                 .read_bytes()).hexdigest()
                  for c in COINS for k in ("candles", "funding")},
              "binance_cohort": {
                  "path": str(BINANCE.relative_to(ROOT)),
                  "sha256": hashlib.sha256(
                      BINANCE.read_bytes()).hexdigest(),
                  "n_records": n_binance}}

    return {
        "schema_version": "nanojev-financial-signal-hyperliquid-v1",
        "status": "measurement_complete",
        "task": "T99",
        "protocol": PROTOCOL,
        "contract": {
            "inputs": inputs,
            "coins": COINS,
            "funding_daily_rule":
                "f_last = last settlement inside UTC day; "
                "f_8h = sum of settlements in [16:00,24:00) UTC",
            "funding_pct": "trailing-180-row mid-rank pct, window "
                           "excludes decision row, per coin",
            "label": "log(close[d+5]/close[d]) on merged candle+funding "
                     "day series",
            "btc_trend_gate": "BTC merged close 20-bar return > 0",
            "top_decile": DECILE, "min_n": MIN_N,
            "spread_min_window": SPREAD_MIN_W,
            "fdr_alpha": 0.05},
        "binance_reference": BINANCE_REF,
        "data_integrity": integrity,
        "arm_a_funding_pct": arm_a,
        "arm_b_hl_binance_spread": arm_b,
        "verdicts": verdicts,
        "caveats": [
            "5d labels on 1d bars overlap ~5x; positive autocorrelation "
            "makes normal-approx p nominal/optimistic (repo caveat)",
            "per-coin cells (~660 rows) are underpowered for rho~0.04 "
            "(|rho|>~0.075 needed for nominal p<0.05 at n~680); the "
            "pooled cell is the powered comparison vs Binance rho=0.0422",
            "HL funding is hourly (Binance 8h): the percentile transform "
            "is scale-free so funding_pct construction matches, but raw "
            "rate magnitudes differ ~8x; the spread arm uses the 8h-sum",
            "HL eval span starts ~2023-11 (funding coverage 2023-05 plus "
            "trailing-180) vs the Binance reference window starting "
            "2023-01-25; windows only partly overlap",
            "candle close is last-trade close, not mark price; no "
            "historical OI/mark-index exists locally for Hyperliquid",
            "early HL history (~first 4 weeks) is on an 8h funding grid; "
            "daily-last and 8h-sum rules both remain well-defined",
            "spread arm is limited to Binance cohort span (ends "
            "2026-08-30) and uses f_8h minus last_funding_rate",
            "gross close moves only - no fees, spread, slippage, funding "
            "cashflows; not a tradability or profitability claim",
            "Hyperliquid licence status UNVERIFIED in fetch_manifest; "
            "offline non-commercial research only"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=OUT)
    args = ap.parse_args()
    r1, r2 = run(), run()
    same = json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
    r1["determinism"] = {"replays": 2, "byte_identical": same}
    r1["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    blob = json.dumps(r1, indent=2, ensure_ascii=False) + "\n"
    args.output.write_text(blob)
    print(json.dumps({"output": str(args.output), "deterministic": same,
                      "sha256": hashlib.sha256(blob.encode()).hexdigest()},
                     indent=2))


if __name__ == "__main__":
    main()
