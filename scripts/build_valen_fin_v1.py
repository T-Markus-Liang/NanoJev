#!/usr/bin/env python3
"""Build a Valen-format decision dataset from the mega financial cohort (T143).

Bridges Track B measurements (``docs/TRACK_B_RESEARCH_STATE_V1.md``) into
trainable typed-decision records: reads
``data/perp_pit_mega_v1/records.jsonl`` (~287k rows, 301 USDT-M perps,
features dfh20/mom20/vol20/close/last_funding_rate + labels
forward_return_bps / forward_return_5d_bps) and emits
``data/valen_fin_v1/{train,eval}.jsonl`` + ``manifest.json`` in the Valen
schema (``external/valen/docs/data-format.md``, text-only, questions nested
under ``request`` as in ``external/valen/data/smoke/train.jsonl``).

*** THIS IS A LABELING-FROM-OUTCOMES DATASET ***
Targets are derived from REALIZED forward returns: each target is a hard
point-mass (1.0) on the bucket the realized outcome fell into. This is
outcome-supervised training data — it teaches the model to predict the
distribution of realized buckets, it is NOT oracle knowledge and NOT a
claim that the state determines the outcome. The realized bucket is
unknowable at decision time; treat these as noisy labels with the measured
signal levels (Track B: rho ~ 0.03-0.14 for the carried structure).

Sampling / split
----------------
* Decision dates: every 5th unique global decision date (stride 5). With a
  5d label horizon this also decorrelates consecutive sampled labels.
* Both BTC-regime states are kept (gate-on = btc_ret20 > 0 and gate-off);
  restricting to gate-on only would bias the regime question.
* Split: temporal per asset — the last 20% of each asset's sampled dates go
  to eval, the first 80% to train (no date range leakage between splits for
  a given asset; different assets still share calendar time, which is
  intended: the cross-section is the object of study).
* Caps: train ~20k, eval ~3k records. Within each split pool, records are
  ordered by sha256("<asset>|<date>") and the first N taken — a
  deterministic, reproducible selection ("hash-split by asset+date").
* Records are eligible only when close/dfh20/mom20/vol20/fwd_5d_bps are
  non-null AND the BTC 20d trend is defined for the date (drops the ~20-day
  cohort warmup).

Derived variables (repo conventions from financial_signal_xs_mega_v1.py)
-----------------------------------------------------------------------
* btc_ret20: BTCUSDT-PERP close/close-20d-ago - 1, contiguous days only
  (ordinal difference == 20); None during warmup.
* funding_pct: per-asset mid-rank percentile of last_funding_rate vs its
  trailing-180-day history (MIN_WINDOW=20); None for the ~249 symbols with
  no funding file.
* xs_med_fwd5[date]: cross-sectional median of forward_return_5d_bps over
  ALL assets in the cohort on that date (universe median, computed over the
  full cohort, not just the sampled subset).

State text (deterministic template; asset base name allowed, no row IDs)::

    BTC 20d trend +3.2% (up); asset SOLUSDT: dfh20=-2.1%, mom20=+5.4%,
    vol20=3.1%/d, funding_pct=0.83

Questions per record (all labeled, point-mass targets)
------------------------------------------------------
a. ``fwd5_bucket`` (choice): "Next 5 trading days, this asset will most
   likely:" — candidates {strong_up: ">+3%", up: "0..+3%",
   down: "-3%..0", strong_down: "<-3%"}; target = realized fwd_5d_bps
   bucket (>= +300bps / 0..+300 / -300..0 / <= -300; boundaries documented
   in the manifest).
b. ``xs_outperform_5d`` (noul): "This asset outperforms the universe median
   over the next 5 days" — target true iff realized fwd_5d_bps strictly
   exceeds the same-day cross-sectional median.
c. ``regime_gate`` (choice): "Current market regime favors:" —
   {momentum_follow, reversal, neutral}; target = SELF-SUPERVISED regime
   label derived from btc_ret20: > +1% -> momentum_follow, < -1% ->
   reversal, else neutral. Teaches the measured master gate (Track B:
   BTC 20d trend splits momentum-on vs reversal-off); the 1% deadband
   separates a flat regime from the two signed ones.

No network, no trading, no commits. Deterministic: no RNG is used anywhere;
the only "randomness" is a fixed sha256 ordering.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import pathlib
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
COHORT = ROOT / "data/perp_pit_mega_v1/records.jsonl"
OUT_DIR = ROOT / "data/valen_fin_v1"

SCHEMA_VERSION = "nanojev-valen-fin-v1"
DATE_STRIDE = 5            # every 5th global decision date
EVAL_TAIL_FRAC = 0.20      # last 20% of each asset's sampled dates -> eval
TRAIN_CAP = 20_000
EVAL_CAP = 3_000

FUND_LOOKBACK = 180        # trailing window for funding_pct mid-rank
MIN_WINDOW = 20            # floor for a usable trailing pct (repo convention)
BTC_GATE_LOOKBACK = 20     # btc_ret20 lookback in days

# fwd5 bucket boundaries in basis points of forward_return_5d_bps.
BUCKET_STRONG_BPS = 300.0  # |fwd5| >= 3% = "strong" leg

# regime_gate deadband: |btc_ret20| <= 1% -> neutral
REGIME_DEADBAND = 0.01

Q_FWD5 = "fwd5_bucket"
Q_XS = "xs_outperform_5d"
Q_REGIME = "regime_gate"

FWD5_CRITERIA = {
    "strong_up": "gain more than +3% over the next 5 trading days",
    "up": "gain between 0% and +3% over the next 5 trading days",
    "down": "lose between 0% and 3% over the next 5 trading days",
    "strong_down": "lose more than 3% over the next 5 trading days",
}
REGIME_CRITERIA = {
    "momentum_follow": "trend-following / momentum continuation",
    "reversal": "mean-reversion / fading recent moves",
    "neutral": "no directional regime edge",
}


def _hash_key(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def mid_rank_pct(window, x):
    """Mid-rank percentile of x vs trailing window (ties count half)."""
    if x is None or not window:
        return None
    return (sum(1 for w in window if w < x)
            + 0.5 * sum(1 for w in window if w == x)) / len(window)


def load_cohort(path):
    """asset -> sorted rows {day, close, dfh20, mom20, vol20, funding,
    fwd_1d_bps, fwd_5d_bps}; returns (series, sha256, n_rows)."""
    series = defaultdict(list)
    digest = hashlib.sha256()
    n_rows = 0
    with path.open("rb") as stream:
        for raw in stream:
            digest.update(raw)
            n_rows += 1
            record = json.loads(raw)
            features = record["features"]
            label = record.get("label") or {}
            series[record["asset_id"]].append({
                "day": record["id"].rsplit(":", 1)[-1],
                "close": features["close"]["value"],
                "dfh20": features["dfh20"]["value"],
                "mom20": features["mom20"]["value"],
                "vol20": features["vol20"]["value"],
                "funding": features["last_funding_rate"]["value"],
                "fwd_1d_bps": label.get("forward_return_bps"),
                "fwd_5d_bps": label.get("forward_return_5d_bps"),
            })
    for rows in series.values():
        rows.sort(key=lambda r: r["day"])
    return series, digest.hexdigest(), n_rows


def add_derived(series):
    """Attach funding_pct per row; build btc_ret20 and per-date XS median
    of fwd_5d_bps over the full cohort."""
    # funding_pct: trailing-180 mid-rank of last_funding_rate.
    for rows in series.values():
        fund = [r["funding"] for r in rows]
        for i, r in enumerate(rows):
            fwin = [x for x in fund[max(0, i - FUND_LOOKBACK):i]
                    if x is not None]
            r["funding_pct"] = (mid_rank_pct(fwin, r["funding"])
                                if r["funding"] is not None
                                and len(fwin) >= MIN_WINDOW else None)
    # btc_ret20: BTCUSDT-PERP close ratio over a contiguous 20d window.
    btc_ret20 = {}
    btc_rows = series.get("BTCUSDT-PERP", [])
    closes = [r["close"] for r in btc_rows]
    ords = [dt.date.fromisoformat(r["day"]).toordinal() for r in btc_rows]
    for i, r in enumerate(btc_rows):
        if (i >= BTC_GATE_LOOKBACK and closes[i - BTC_GATE_LOOKBACK]
                and closes[i - BTC_GATE_LOOKBACK] > 0
                and ords[i] - ords[i - BTC_GATE_LOOKBACK]
                == BTC_GATE_LOOKBACK):
            btc_ret20[r["day"]] = \
                closes[i] / closes[i - BTC_GATE_LOOKBACK] - 1.0
    # Per-date XS median of fwd5 over ALL cohort assets that day.
    by_day = defaultdict(list)
    for rows in series.values():
        for r in rows:
            if r["fwd_5d_bps"] is not None:
                by_day[r["day"]].append(r["fwd_5d_bps"])
    xs_med = {}
    for day, vals in by_day.items():
        vals.sort()
        n = len(vals)
        xs_med[day] = (vals[n // 2] if n % 2
                       else 0.5 * (vals[n // 2 - 1] + vals[n // 2]))
    return btc_ret20, xs_med


def fwd5_bucket(fwd_bps):
    """Realized fwd5 bucket. Boundaries: strong_up >= +300bps;
    up in [0, 300); down in (-300, 0); strong_down <= -300bps."""
    if fwd_bps >= BUCKET_STRONG_BPS:
        return "strong_up"
    if fwd_bps >= 0.0:
        return "up"
    if fwd_bps > -BUCKET_STRONG_BPS:
        return "down"
    return "strong_down"


def regime_label(btc_ret):
    """Self-supervised regime target from the BTC 20d trend sign."""
    if btc_ret > REGIME_DEADBAND:
        return "momentum_follow"
    if btc_ret < -REGIME_DEADBAND:
        return "reversal"
    return "neutral"


def regime_word(btc_ret):
    if btc_ret > REGIME_DEADBAND:
        return "up"
    if btc_ret < -REGIME_DEADBAND:
        return "down"
    return "flat"


def state_text(asset_base, btc_ret, row):
    fp = row["funding_pct"]
    fp_txt = f"{fp:.2f}" if fp is not None else "n/a"
    return (
        f"BTC 20d trend {btc_ret * 100:+.1f}% ({regime_word(btc_ret)}); "
        f"asset {asset_base}: dfh20={row['dfh20'] * 100:+.1f}%, "
        f"mom20={row['mom20'] * 100:+.1f}%, "
        f"vol20={row['vol20'] * 100:.1f}%/d, funding_pct={fp_txt}"
    )


def build_record(asset, row, btc_ret, xs_med):
    """One Valen record: 3 questions, all labeled (point-mass targets)."""
    day = row["day"]
    asset_base = asset[:-5] if asset.endswith("-PERP") else asset
    bucket = fwd5_bucket(row["fwd_5d_bps"])
    regime = regime_label(btc_ret)
    beat_median = row["fwd_5d_bps"] > xs_med[day]
    text = state_text(asset_base, btc_ret, row)
    group_id = f"fin-v1:{asset_base}:{day}"
    return {
        "group_id": group_id,
        "request": {
            "state": {"messages": [{"role": "user", "content": [
                {"type": "text", "text": text}]}]},
            "questions": {
                Q_FWD5: {
                    "type": "choice",
                    "instructions": "Next 5 trading days, this asset will "
                                    "most likely:",
                    "criteria": dict(FWD5_CRITERIA),
                },
                Q_XS: {
                    "type": "noul",
                    "instructions": "This asset outperforms the universe "
                                    "median over the next 5 days.",
                },
                Q_REGIME: {
                    "type": "choice",
                    "instructions": "Current market regime favors:",
                    "criteria": dict(REGIME_CRITERIA),
                },
            },
        },
        "targets": {
            Q_FWD5: {"probabilities": {
                k: 1.0 if k == bucket else 0.0 for k in FWD5_CRITERIA}},
            Q_XS: {"probabilities": {
                "true": 1.0 if beat_median else 0.0,
                "false": 0.0 if beat_median else 1.0}},
            Q_REGIME: {"probabilities": {
                k: 1.0 if k == regime else 0.0 for k in REGIME_CRITERIA}},
        },
        "assets": [],
        "meta": {
            "record_id": group_id,
            "domain": "crypto_perp_daily",
            "modality": "text",
            "language_bucket": "en",
            "asset_id": asset,
            "date": day,
            "btc_ret20": round(btc_ret, 6),
            "labeling": "outcome-supervised point-mass on realized bucket",
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--records", type=pathlib.Path, default=COHORT)
    ap.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    ap.add_argument("--train-cap", type=int, default=TRAIN_CAP)
    ap.add_argument("--eval-cap", type=int, default=EVAL_CAP)
    ap.add_argument("--date-stride", type=int, default=DATE_STRIDE)
    args = ap.parse_args()

    print(f"[load] {args.records}")
    series, src_sha, n_rows = load_cohort(args.records)
    print(f"[load] {n_rows} rows, {len(series)} assets, "
          f"sha256={src_sha[:16]}...")

    btc_ret20, xs_med = add_derived(series)
    print(f"[derived] btc_ret20 defined for {len(btc_ret20)} dates; "
          f"XS median for {len(xs_med)} dates")

    all_days = sorted(xs_med)
    sampled_days = {d for i, d in enumerate(all_days)
                    if i % args.date_stride == 0}
    print(f"[sample] {len(sampled_days)}/{len(all_days)} decision dates "
          f"(stride {args.date_stride})")

    # Per-asset temporal split over each asset's eligible sampled dates.
    train_pool, eval_pool = [], []
    n_skipped = Counter()
    for asset, rows in series.items():
        eligible = []
        for r in rows:
            if r["day"] not in sampled_days:
                continue
            if (r["close"] is None or r["dfh20"] is None
                    or r["mom20"] is None or r["vol20"] is None):
                n_skipped["null_features"] += 1
                continue
            if r["fwd_5d_bps"] is None:
                n_skipped["null_fwd5_label"] += 1
                continue
            if r["day"] not in btc_ret20:
                n_skipped["btc_gate_warmup"] += 1
                continue
            eligible.append(r)
        days = sorted({r["day"] for r in eligible})
        if not days:
            continue
        cutoff = int(math.floor((1.0 - EVAL_TAIL_FRAC) * len(days)))
        eval_days = set(days[cutoff:]) if cutoff < len(days) else set()
        for r in eligible:
            (eval_pool if r["day"] in eval_days
             else train_pool).append((asset, r))

    # Deterministic hash ordering by asset+date, then cap.
    def order(pool):
        return sorted(pool, key=lambda ar: _hash_key(
            f"{SCHEMA_VERSION}|{ar[0]}|{ar[1]['day']}"))

    train_sel = order(train_pool)[:args.train_cap]
    eval_sel = order(eval_pool)[:args.eval_cap]
    print(f"[pool] train {len(train_pool)} -> kept {len(train_sel)}; "
          f"eval {len(eval_pool)} -> kept {len(eval_sel)}; "
          f"skipped {dict(n_skipped)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    label_balance = {}
    for split, sel in (("train", train_sel), ("eval", eval_sel)):
        path = args.out_dir / f"{split}.jsonl"
        dates, assets = set(), set()
        balance = {Q_FWD5: Counter(), Q_XS: Counter(), Q_REGIME: Counter()}
        with path.open("w", encoding="utf-8") as out:
            for asset, r in sel:
                rec = build_record(asset, r, btc_ret20[r["day"]], xs_med)
                out.write(json.dumps(rec, ensure_ascii=False,
                                     separators=(",", ":")) + "\n")
                dates.add(r["day"])
                assets.add(asset)
                for q in balance:
                    probs = rec["targets"][q]["probabilities"]
                    balance[q][max(probs, key=probs.get)] += 1
        stats[split] = {
            "records": len(sel), "unique_assets": len(assets),
            "unique_dates": len(dates),
            "date_min": min(dates) if dates else None,
            "date_max": max(dates) if dates else None,
            "file": str(path.relative_to(ROOT)),
        }
        label_balance[split] = {q: dict(balance[q]) for q in balance}
        print(f"[write] {path} ({len(sel)} records, "
              f"{stats[split]['date_min']}..{stats[split]['date_max']})")
        for q, bal in balance.items():
            tot = sum(bal.values()) or 1
            print(f"        {q}: " + ", ".join(
                f"{k}={v} ({v / tot:.1%})"
                for k, v in sorted(bal.items())))

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "task": "T143: Valen-format decision dataset bridging Track B "
                "measurements to trainable decision tasks",
        "labeling_contract": {
            "kind": "outcome-supervised (labeling from outcomes)",
            "targets": "hard point-mass (1.0) on the REALIZED bucket",
            "not_oracle": "the realized bucket is unknowable at decision "
                          "time; these are noisy supervision labels, not "
                          "claims of determinism",
            "signal_context": "Track B measured structure is weak but "
                              "real (rho ~ 0.03-0.14, regime-conditional); "
                              "see docs/TRACK_B_RESEARCH_STATE_V1.md",
        },
        "source": {
            "records": str(args.records.relative_to(ROOT)),
            "records_sha256": src_sha,
            "records_rows": n_rows,
            "cohort_schema": "nanojev-financial-pit-mega-v1",
        },
        "sampling": {
            "date_stride": args.date_stride,
            "all_decision_dates": len(all_days),
            "sampled_decision_dates": len(sampled_days),
            "eligibility": "close/dfh20/mom20/vol20/fwd_5d_bps non-null AND "
                           "btc_ret20 defined (post-warmup); both BTC "
                           "regimes kept",
            "skipped": dict(n_skipped),
        },
        "split": {
            "rule": "temporal per asset: last 20% of each asset's sampled "
                    "dates -> eval, first 80% -> train",
            "eval_tail_frac": EVAL_TAIL_FRAC,
            "cap_order": "sha256('<schema>|<asset_id>|<date>') ascending, "
                         "first N kept (deterministic hash-split by "
                         "asset+date)",
            "train_cap": args.train_cap,
            "eval_cap": args.eval_cap,
            "pool_sizes": {"train": len(train_pool), "eval": len(eval_pool)},
        },
        "derived_variables": {
            "btc_ret20": "BTCUSDT-PERP close/close-20d-ago - 1, contiguous "
                         "days only; None during warmup (rows dropped)",
            "funding_pct": "per-asset mid-rank pct of last_funding_rate vs "
                           "trailing-180d (MIN_WINDOW=20); None on the "
                           "~249 no-funding symbols -> 'n/a' in state text",
            "xs_med_fwd5": "per-date cross-sectional median of "
                           "forward_return_5d_bps over ALL cohort assets",
        },
        "questions": {
            Q_FWD5: {
                "type": "choice",
                "candidates": FWD5_CRITERIA,
                "target": "realized forward_return_5d_bps bucket; "
                          "boundaries: strong_up >= +300bps, "
                          "up in [0,+300), down in (-300,0), "
                          "strong_down <= -300bps",
            },
            Q_XS: {
                "type": "noul",
                "target": "true iff realized fwd_5d_bps > same-day "
                          "universe median (strict)",
            },
            Q_REGIME: {
                "type": "choice",
                "candidates": REGIME_CRITERIA,
                "target": "SELF-SUPERVISED from btc_ret20: > +1% -> "
                          "momentum_follow, < -1% -> reversal, else "
                          "neutral (deadband teaches the measured master "
                          "gate; not a realized outcome)",
            },
        },
        "state_template": "BTC 20d trend {btc:+.1f}% ({up|down|flat}); "
                          "asset {BASE}: dfh20={:+.1f}%, mom20={:+.1f}%, "
                          "vol20={:.1f}%/d, funding_pct={0.00|n/a}",
        "splits": stats,
        "label_balance": label_balance,
        "honesty": {
            "no_profitability_claim": True,
            "no_network": True,
            "no_commits": True,
            "point_in_time": "source features carry their own PIT "
                             "available_ns metadata; labels are realized "
                             "future returns by construction",
        },
    }
    mpath = args.out_dir / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2) + "\n",
                     encoding="utf-8")
    print(f"[write] {mpath}")


if __name__ == "__main__":
    main()
