#!/usr/bin/env python3
"""T168 (next step): shippable isotonic calibration bundle for Winnow-12B.

`winnow_calibration_v1.py` showed the parametric layers saturate near
ECE ~0.08 while a PAVA isotonic map fitted on the same cal-fit rows reaches
~0.02 — the residual was functional-shape, not data. This script turns that
diagnostic into a checkpointed artifact: fit PAVA maps (probability ->
calibrated probability) on the cal-fit split ONLY, per source domain where
the rows carry a domain label, and emit a versioned bundle JSON with the
piecewise breakpoints plus before/after metrics on the group-isolated
cal-holdout and the frozen eval file.

Domain label: no explicit ``domain`` field exists in the scored rows, so the
domain key is the source-dataset prefix of ``group_id`` (text before the
first ``:``) — the same convention the v1 doc uses when it says the corpus
is "context_relevance_v1 + oracle_v1 sources". Rows whose domain was not
seen during fitting fall back to the global map.

Protocol: cal-fit rows are the ONLY fitting input. cal-holdout and eval are
measurement-only; eval rows never enter any fit. Advisory artifact — nothing
here touches serve_decisions.py, gateway thresholds, or production config.

Usage:

    .venv/bin/python scripts/winnow_isotonic_v1.py            # fit + bundle
    .venv/bin/python scripts/winnow_isotonic_v1.py --bundle-out /tmp/x.json
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from winnow_calibration_v1 import (
    EVAL_OUT, FIT_OUT, TRAIN_PATH,
    load_records, load_scored, metrics, sha256_file, split_groups)

ROOT = Path(__file__).resolve().parent.parent
BUNDLE_OUT = ROOT / "results" / "winnow_isotonic_bundle_v1.json"
SCHEMA_VERSION = "nanojev-winnow-isotonic-bundle-v1"
MIN_DOMAIN_FIT_ROWS = 100  # below this a domain falls back to the global map


def domain_of(row):
    """Source-dataset domain: group_id text before the first ':'."""
    gid = row.get("group_id") or ""
    return gid.split(":", 1)[0] if gid else ""


# ------------------------------- PAVA core -------------------------------

def fit_pava(probs, labels):
    """Pool-adjacent-violators isotonic fit y ~ p (non-decreasing), unit
    weights. Returns (knot_x, knot_y): step-function block boundaries —
    consecutive [xlo, val], [xhi, val] pairs per pooled block — suitable for
    np.interp. ~30 lines, no sklearn/scipy needed."""
    order = np.argsort(np.asarray(probs, dtype=float), kind="stable")
    x = np.asarray(probs, dtype=float)[order]
    t = np.asarray(labels, dtype=float)[order]
    blocks = []  # each: [label_sum, count, x_lo, x_hi]
    i = 0
    while i < len(x):  # collapse identical x into one block first
        j = i
        while j + 1 < len(x) and x[j + 1] == x[i]:
            j += 1
        blocks.append([float(t[i:j + 1].sum()), int(j - i + 1),
                       float(x[i]), float(x[i])])
        i = j + 1
        while (len(blocks) >= 2
               and blocks[-2][0] / blocks[-2][1]
               > blocks[-1][0] / blocks[-1][1]):
            tail = blocks.pop()
            head = blocks.pop()
            blocks.append([head[0] + tail[0], head[1] + tail[1],
                           head[2], tail[3]])
    kx, ky = [], []
    for total, count, x_lo, x_hi in blocks:
        val = total / count
        kx.append(x_lo)
        ky.append(val)
        if x_hi > x_lo:
            kx.append(x_hi)
            ky.append(val)
    kx, ky = np.asarray(kx), np.asarray(ky)
    if not (np.all(np.diff(kx) > 0) and np.all(np.diff(ky) >= -1e-12)):
        raise AssertionError("PAVA produced a non-monotone map")
    return kx, ky


def apply_map(knots, probs):
    """Piecewise-linear interpolation on the fitted breakpoints; constant
    (ky[0] / ky[-1]) outside the fitted range."""
    kx, ky = knots
    return np.interp(np.asarray(probs, dtype=float), kx, ky)


def fit_map(rows):
    probs = [r["noul_prob"] for r in rows]
    labels = [r["label"] for r in rows]
    kx, ky = fit_pava(probs, labels)
    return {"x": [float(v) for v in kx],
            "y": [float(v) for v in ky],
            "n_fit": len(rows)}


def map_knots(map_obj):
    return np.asarray(map_obj["x"]), np.asarray(map_obj["y"])


def predict(rows, maps, default="global"):
    """Per-domain apply with global fallback for unseen domains."""
    out = np.empty(len(rows))
    for i, row in enumerate(rows):
        dom = row.get("_domain") or domain_of(row)
        m = maps["per_domain"].get(dom, maps[default])
        out[i] = apply_map(map_knots(m), [row["noul_prob"]])[0]
    return out


# ------------------------------ build bundle ------------------------------

def split_of_fit_rows(fit_rows):
    """Authoritative cal-fit/cal-holdout assignment, cross-checked against
    the stored `split` tag (same rule as winnow_calibration_v1)."""
    train_records = load_records(TRAIN_PATH)
    splits = split_groups(train_records)
    for row in fit_rows:
        expected = splits[row["record_index"]]
        if row.get("split") and row["split"] != expected:
            raise ValueError(f"split mismatch at {row['record_index']}")
        row["split"] = expected
    return fit_rows


def build_bundle(fit_path=FIT_OUT, eval_path=EVAL_OUT):
    fit_rows = split_of_fit_rows(load_scored(fit_path))
    eval_rows = load_scored(eval_path)
    fit_set = [r for r in fit_rows if r["split"] == "cal-fit"]
    hold_set = [r for r in fit_rows if r["split"] == "cal-holdout"]
    if len(fit_set) < 100 or len(hold_set) < 50 or len(eval_rows) < 100:
        raise SystemExit("not enough scored rows "
                         f"(fit={len(fit_set)} hold={len(hold_set)} "
                         f"eval={len(eval_rows)})")

    for rows in (fit_set, hold_set, eval_rows):
        for row in rows:
            row["_domain"] = domain_of(row)

    maps = {"global": fit_map(fit_set), "per_domain": {}}
    domains = sorted({r["_domain"] for r in fit_set})
    skipped = []
    for dom in domains:
        dom_rows = [r for r in fit_set if r["_domain"] == dom]
        if len(dom_rows) >= MIN_DOMAIN_FIT_ROWS:
            maps["per_domain"][dom] = fit_map(dom_rows)
        else:
            skipped.append({"domain": dom, "n_fit": len(dom_rows)})

    split_metrics = {}
    for name, rows in (("cal_fit", fit_set), ("cal_holdout", hold_set),
                       ("eval", eval_rows)):
        raw = np.asarray([r["noul_prob"] for r in rows])
        glob = apply_map(map_knots(maps["global"]), raw)
        per = predict(rows, maps)
        split_metrics[name] = {
            "raw": metrics(rows, raw),
            "isotonic_global": metrics(rows, glob),
            "isotonic_per_domain": metrics(rows, per)}

    domain_breakdown = {}
    for name, rows in (("cal_holdout", hold_set), ("eval", eval_rows)):
        for dom in sorted({r["_domain"] for r in rows}):
            sub = [r for r in rows if r["_domain"] == dom]
            raw = np.asarray([r["noul_prob"] for r in sub])
            entry = {"n": len(sub),
                     "ece_raw": metrics(sub, raw)["ece"],
                     "ece_isotonic_global": metrics(
                         sub, apply_map(map_knots(maps["global"]), raw))["ece"]}
            if dom in maps["per_domain"]:
                entry["ece_isotonic_domain"] = metrics(
                    sub, apply_map(map_knots(maps["per_domain"][dom]),
                                   raw))["ece"]
            else:
                entry["ece_isotonic_domain"] = None
                entry["fallback"] = "global"
            domain_breakdown[f"{name}:{dom}"] = entry

    return {
        "schema_version": SCHEMA_VERSION,
        "task": "T168",
        "advisory_only": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scorer": {"model": "Winnow-12B", "question_qid": "irrelevant",
                   "semantics": "noul == P(true) == P(candidate certainly "
                                "irrelevant == drop)"},
        "domain_key": "group_id.split(':')[0] — source-dataset prefix "
                      "(context_relevance_v1 / "
                      "context_relevance_oracle_v1_seed20260919)",
        "apply_rule": "p_cal = np.interp(p_raw, map.x, map.y); domain from "
                      "group_id prefix; unseen or undersized domain -> "
                      "maps.global",
        "data": {
            "fit_rows": {"path": Path(fit_path).name,
                         "sha256": sha256_file(fit_path),
                         "n": len(fit_rows),
                         "cal_fit": len(fit_set),
                         "cal_holdout": len(hold_set)},
            "eval_rows": {"path": Path(eval_path).name,
                          "sha256": sha256_file(eval_path),
                          "n": len(eval_rows),
                          "usage": "measurement only; never fitted"},
            "fit_source": str(TRAIN_PATH.relative_to(ROOT)),
            "fit_source_sha256": sha256_file(TRAIN_PATH),
        },
        "maps": maps,
        "domains_skipped_to_global": skipped,
        "metrics": split_metrics,
        "domain_breakdown": domain_breakdown,
        "notes": [
            "fitted on cal-fit rows only (group-isolated 80% of train); "
            "cal-holdout and eval are measurement-only",
            "isotonic maps are non-parametric; consume as an advisory "
            "per-domain bundle keyed by (model, qid, domain), never as a "
            "global constant on other question types",
            "no production serving config, gateway threshold, or routing "
            "changed",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-out", default=str(BUNDLE_OUT))
    args = parser.parse_args()
    bundle = build_bundle()
    encoded = json.dumps(bundle, ensure_ascii=False, indent=2) + "\n"
    Path(args.bundle_out).write_text(encoded, encoding="utf-8")
    summary = {"bundle": args.bundle_out,
               "schema_version": bundle["schema_version"],
               "domains": sorted(bundle["maps"]["per_domain"]),
               "metrics": {k: {m: v["ece"] for m, v in split.items()}
                           for k, split in bundle["metrics"].items()}}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
