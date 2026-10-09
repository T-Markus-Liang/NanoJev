#!/usr/bin/env python3
"""T168: post-hoc calibration layer for Winnow-12B `irrelevant` noul probs.

Phase ``score`` posts each record's ``request`` verbatim to the local Winnow
``/v1/systemone`` endpoint (``http://127.0.0.1:8091``) and stores raw
(record_index, noul_prob, label) rows. Phase ``fit`` fits temperature scaling
and Platt/logistic scaling on the cal-fit split by NLL minimization (pure
numpy: golden-section for T, Newton/IRLS for the 2-param logistic) and
evaluates ECE(15-bin)/Brier/NLL/accuracy@0.5 on the cal-holdout split and the
frozen eval.jsonl (measurement only — never fitted).

Split rule: sort unique train group_ids by sha256 hex (same convention as the
dataset's own ``sha256(group_id) mod 100`` train/eval rule) — first 80% of
groups -> cal-fit, last 20% -> cal-holdout. Group-isolated, deterministic, and
keeps both source datasets represented in both splits.

Advisory layer only: nothing here touches the production service config,
gateway thresholds, or model weights.

Usage:

    .venv/bin/python scripts/winnow_calibration_v1.py score   # train + eval
    .venv/bin/python scripts/winnow_calibration_v1.py fit     # fit + evaluate

Scoring appends incrementally and resumes from existing output files.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import sys
import threading
import time
from datetime import datetime, timezone
from http.client import HTTPConnection
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "data" / "valen_nano_v1" / "train.jsonl"
EVAL_PATH = ROOT / "data" / "valen_nano_v1" / "eval.jsonl"
FIT_OUT = ROOT / "results" / "winnow_calib_fit_v1.jsonl"
EVAL_OUT = ROOT / "results" / "winnow_calib_eval_v1.jsonl"
RECEIPT_OUT = ROOT / "results" / "winnow_calibration_v1.json"
WINNOW_URL = "http://127.0.0.1:8091"
QID = "irrelevant"
FIT_FRACTION = 0.80
EPS = 1e-7

_write_lock = threading.Lock()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_records(path):
    return [json.loads(line)
            for line in Path(path).read_text("utf-8").splitlines()
            if line.strip()]


def label_of(record):
    """1.0 when target argmax is 'true' (certainly irrelevant == drop)."""
    probs = ((record.get("targets") or {}).get(QID) or {}).get("probabilities")
    if not isinstance(probs, dict):
        raise ValueError("record missing targets.%s.probabilities" % QID)
    return 1.0 if max(probs, key=probs.get) == "true" else 0.0


def split_groups(records):
    """sha256-ordered group split: first 80% of groups -> cal-fit."""
    seen, ordered = set(), []
    for record in records:
        gid = record["group_id"]
        if gid not in seen:
            seen.add(gid)
            ordered.append(gid)
    ranked = sorted(ordered,
                    key=lambda g: hashlib.sha256(g.encode()).hexdigest())
    cut = int(len(ranked) * FIT_FRACTION)
    fit_groups = set(ranked[:cut])
    return {i: ("cal-fit" if r["group_id"] in fit_groups else "cal-holdout")
            for i, r in enumerate(records)}


def post_noul(request, url, timeout=120.0, retries=2):
    """POST {model, state, questions} verbatim; return P(true) for `irrelevant`."""
    parts = urlsplit(url)
    body = json.dumps({"model": "Winnow-12B", **request})
    last = None
    for _ in range(retries + 1):
        connection = HTTPConnection(parts.hostname, parts.port, timeout=timeout)
        try:
            connection.request("POST", "/v1/systemone", body=body,
                               headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            data = json.loads(response.read(2_000_000))
            if response.status == 200:
                noul = data["answers"][QID]["noul"]
                if noul is not None:
                    return float(noul)
            last = f"status {response.status}: {str(data)[:200]}"
        except Exception as exc:  # transient loopback errors -> retry
            last = repr(exc)
        finally:
            connection.close()
        time.sleep(0.25)
    raise RuntimeError(f"winnow scoring failed: {last}")


def _done_indices(path):
    done = set()
    if Path(path).exists():
        for line in Path(path).read_text("utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["record_index"])
    return done


def score_file(records_path, out_path, url, workers, extra_fields=None):
    records = load_records(records_path)
    splits = split_groups(records) if extra_fields == "split" else {}
    done = _done_indices(out_path)
    todo = [i for i in range(len(records)) if i not in done]
    print(json.dumps({"event": "score_start", "input": str(records_path),
                      "output": str(out_path), "total": len(records),
                      "done": len(done), "todo": len(todo)}), flush=True)
    if not todo:
        return
    out = Path(out_path).open("a", encoding="utf-8")
    completed = [0]
    started = time.perf_counter()

    def work(index):
        record = records[index]
        noul = post_noul(record["request"], url)
        row = {"record_index": index,
               "record_id": (record.get("meta") or {}).get("record_id"),
               "group_id": record["group_id"],
               "noul_prob": noul,
               "label": label_of(record)}
        if splits:
            row["split"] = splits[index]
        with _write_lock:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            completed[0] += 1
            if completed[0] % 50 == 0 or completed[0] == len(todo):
                rate = completed[0] / (time.perf_counter() - started)
                print(json.dumps({"event": "progress",
                                  "file": out_path.name,
                                  "completed": completed[0],
                                  "todo": len(todo),
                                  "rate_per_s": round(rate, 2)}),
                      flush=True)

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(work, todo))
    finally:
        out.close()


# ----------------------------- fit / evaluate -----------------------------

def load_scored(path):
    rows = [json.loads(line)
            for line in Path(path).read_text("utf-8").splitlines()
            if line.strip()]
    rows.sort(key=lambda r: r["record_index"])
    return rows


def clip(p):
    return min(max(p, EPS), 1.0 - EPS)


def logit(p):
    p = clip(p)
    return math.log(p / (1.0 - p))


def metrics(rows, probs):
    """ECE(15) on top-label confidence (repo convention), Brier, NLL, acc@0.5."""
    import numpy as np
    p = np.asarray(probs, dtype=float)
    y = np.asarray([r["label"] for r in rows], dtype=float)
    conf = np.maximum(p, 1.0 - p)
    correct = ((p >= 0.5).astype(float) == y).astype(float)
    ece = 0.0
    for lo in range(15):
        hi = (lo + 1) / 15.0
        mask = (conf >= lo / 15.0) & ((conf < hi) if lo < 14
                                      else (conf <= hi))
        if not mask.any():
            continue
        ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    pc = np.clip(p, EPS, 1.0 - EPS)
    return {"n": len(rows),
            "ece": float(ece),
            "brier": float(np.mean((p - y) ** 2)),
            "nll": float(-np.mean(y * np.log(pc)
                                  + (1.0 - y) * np.log(1.0 - pc))),
            "accuracy_at_0.5": float(correct.mean())}


def fit_temperature(logits, labels):
    """Golden-section NLL minimization over T > 0. p = sigmoid(logit/T)."""
    import numpy as np
    z, y = np.asarray(logits), np.asarray(labels)

    def nll(t):
        p = np.clip(1.0 / (1.0 + np.exp(-z / t)), EPS, 1.0 - EPS)
        return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    lo, hi = 0.02, 50.0
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    x1, x2 = hi - gr * (hi - lo), lo + gr * (hi - lo)
    f1, f2 = nll(x1), nll(x2)
    for _ in range(200):
        if abs(hi - lo) < 1e-10:
            break
        if f1 > f2:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + gr * (hi - lo)
            f2 = nll(x2)
        else:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - gr * (hi - lo)
            f1 = nll(x1)
    t = (lo + hi) / 2.0
    return t, nll(t)


def fit_platt(logits, labels, init=(1.0, 0.0)):
    """IRLS/Newton fit of p = sigmoid(a*logit + b) by NLL. 2 params."""
    import numpy as np
    z = np.asarray(logits, dtype=float)
    y = np.asarray(labels, dtype=float)
    a, b = float(init[0]), float(init[1])
    final_nll = None
    for _ in range(200):
        p = np.clip(1.0 / (1.0 + np.exp(-(a * z + b))), EPS, 1.0 - EPS)
        w = p * (1.0 - p)
        r = y - p
        # dNLL/d(a,b) = -[sum(r*z), sum(r)]; Newton descent step solves
        # H @ step = -grad = [sum(r*z), sum(r)]
        descent = np.array([np.sum(r * z), np.sum(r)])
        h = np.array([[np.sum(w * z * z), np.sum(w * z)],
                      [np.sum(w * z), np.sum(w)]]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(h, descent)
        nll_prev = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        final_nll = nll_prev
        scale = 1.0
        for _ in range(50):  # step-halving backtracking
            a2, b2 = a + scale * step[0], b + scale * step[1]
            p2 = np.clip(1.0 / (1.0 + np.exp(-(a2 * z + b2))), EPS, 1.0 - EPS)
            nll2 = float(-np.mean(y * np.log(p2) + (1 - y) * np.log(1 - p2)))
            if nll2 < nll_prev - 1e-12:
                a, b, final_nll = a2, b2, nll2
                break
            scale *= 0.5
        else:
            break
        if scale * float(np.abs(step).max()) < 1e-10:
            break
    p = np.clip(1.0 / (1.0 + np.exp(-(a * z + b))), EPS, 1.0 - EPS)
    final_nll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return (a, b), final_nll


def fit_isotonic(probs, labels):
    """PAVA isotonic fit of y ~ p (nondecreasing). Returns knot arrays for
    np.interp. Diagnostic reference only — not a shippable parametric layer."""
    import numpy as np
    order = np.argsort(probs)
    x = np.asarray(probs, dtype=float)[order]
    t = np.asarray(labels, dtype=float)[order]
    stack = []
    for value in t:  # pool adjacent violators, unit weights
        stack.append([float(value), 1])
        while (len(stack) >= 2
               and stack[-2][0] / stack[-2][1] > stack[-1][0] / stack[-1][1]):
            tail = stack.pop()
            head = stack.pop()
            stack.append([head[0] + tail[0], head[1] + tail[1]])
    fitted = np.zeros(len(t))
    index = 0
    for total, count in stack:
        fitted[index:index + count] = total / count
        index += count
    return x, fitted


def apply_isotonic(knots, probs):
    import numpy as np
    kx, ky = knots
    return np.interp(np.asarray(probs, dtype=float), kx, ky)


def apply_probs(logits, mode, params):
    import numpy as np
    z = np.asarray(logits, dtype=float)
    if mode == "identity":
        return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
    if mode == "temperature":
        return np.clip(1.0 / (1.0 + np.exp(-z / params)), EPS, 1.0 - EPS)
    if mode == "platt":
        a, b = params
        return np.clip(1.0 / (1.0 + np.exp(-(a * z + b))), EPS, 1.0 - EPS)
    raise ValueError(mode)


def run_fit(args):
    fit_rows = [r for r in load_scored(FIT_OUT)]
    train_records = load_records(TRAIN_PATH)
    splits = split_groups(train_records)
    # authoritative split assignment from live data, cross-checked vs scored tag
    for row in fit_rows:
        expected = splits[row["record_index"]]
        if row.get("split") and row["split"] != expected:
            raise ValueError(f"split mismatch at {row['record_index']}")
        row["split"] = expected
    eval_rows = load_scored(EVAL_OUT)

    fit_set = [r for r in fit_rows if r["split"] == "cal-fit"]
    hold_set = [r for r in fit_rows if r["split"] == "cal-holdout"]
    if len(fit_set) < 100 or len(hold_set) < 50 or len(eval_rows) < 100:
        raise SystemExit("not enough scored rows; run `score` first "
                         f"(fit={len(fit_set)} hold={len(hold_set)} "
                         f"eval={len(eval_rows)})")

    fit_logits = [logit(r["noul_prob"]) for r in fit_set]
    fit_labels = [r["label"] for r in fit_set]
    temperature, temp_nll = fit_temperature(fit_logits, fit_labels)
    (platt_a, platt_b), platt_nll = fit_platt(fit_logits, fit_labels,
                                            init=(1.0 / temperature, 0.0))

    iso_knots = fit_isotonic([r["noul_prob"] for r in fit_set], fit_labels)

    modes = {"raw": ("identity", None),
             "temperature": ("temperature", temperature),
             "platt": ("platt", (platt_a, platt_b))}
    splits_out = {}
    for name, rows in (("cal_fit", fit_set), ("cal_holdout", hold_set),
                       ("eval", eval_rows)):
        logits = [logit(r["noul_prob"]) for r in rows]
        entry = {label: metrics(rows, apply_probs(logits, m, p))
                 for label, (m, p) in modes.items()}
        # non-parametric headroom reference (cal-fit only, not a shipped layer)
        entry["isotonic_reference"] = metrics(
            rows, apply_isotonic(iso_knots, [r["noul_prob"] for r in rows]))
        splits_out[name] = entry

    receipt = {
        "schema_version": "nanojev-winnow-calibration-v1",
        "task": "T168",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scorer": {"url": WINNOW_URL, "model": "Winnow-12B",
                   "endpoint": "/v1/systemone",
                   "question_qid": QID,
                   "semantics": "noul == P(true) == P(candidate certainly "
                                "irrelevant == drop)"},
        "data": {
            "train": {"path": str(TRAIN_PATH.relative_to(ROOT)),
                      "sha256": sha256_file(TRAIN_PATH),
                      "records": len(train_records)},
            "eval": {"path": str(EVAL_PATH.relative_to(ROOT)),
                     "sha256": sha256_file(EVAL_PATH),
                     "records": len(eval_rows),
                     "usage": "measurement only; never fitted"},
            "fit_rows": {"path": FIT_OUT.name, "sha256": sha256_file(FIT_OUT),
                         "n": len(fit_rows)},
            "eval_rows": {"path": EVAL_OUT.name,
                          "sha256": sha256_file(EVAL_OUT),
                          "n": len(eval_rows)},
            "split_rule": "sort unique train group_ids by sha256 hex; "
                          f"first {FIT_FRACTION:.0%} of groups -> cal-fit, "
                          "rest -> cal-holdout (group-isolated)",
            "groups": {"cal_fit": len({r['group_id'] for r in fit_set}),
                       "cal_holdout": len({r['group_id'] for r in hold_set})},
        },
        "fitted_params": {
            "temperature": {"T": temperature, "fit_nll": temp_nll},
            "platt": {"a": platt_a, "b": platt_b, "fit_nll": platt_nll},
            "isotonic_reference": {"knots": int(len(iso_knots[0])),
                                   "note": "PAVA fit on cal-fit only; "
                                           "headroom diagnostic, not a "
                                           "shipped parametric layer"},
        },
        "metrics": splits_out,
        "notes": [
            "fitted on NanoJev-owned CC0 train data only; eval rows never "
            "entered either fit",
            "params are domain-specific (context-filtering distribution); "
            "ship as a per-domain calibration bundle, not a global constant",
            "advisory layer: production service config and gateway thresholds "
            "unchanged",
        ],
    }
    encoded = json.dumps(receipt, ensure_ascii=False, indent=2,
                         sort_keys=False) + "\n"
    Path(RECEIPT_OUT).write_text(encoded, encoding="utf-8")
    print(encoded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["score", "fit", "all"])
    parser.add_argument("--scorer-url", default=WINNOW_URL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--train-only", action="store_true",
                        help="score phase: skip eval.jsonl")
    args = parser.parse_args()
    if args.phase in ("score", "all"):
        score_file(TRAIN_PATH, FIT_OUT, args.scorer_url, args.workers,
                   extra_fields="split")
        if not args.train_only:
            score_file(EVAL_PATH, EVAL_OUT, args.scorer_url, args.workers)
    if args.phase in ("fit", "all"):
        run_fit(args)


if __name__ == "__main__":
    main()
