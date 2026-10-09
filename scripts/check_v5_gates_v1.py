#!/usr/bin/env python3
"""v5 acceptance-gate checker — implements G1-G6 of docs/V5_DATA_DESIGN_V1.md
§9 over a results directory of scorer output files.

Inputs are read-only. Every gate emits {status, measured, criteria, detail}
with status in {"pass","fail","not_ready","tripwire"}; a gate whose inputs do
not exist yet (e.g. the v5 corpus / transfer holdout have not been built)
reports not_ready instead of crashing.

Result file convention (all inside --results-dir unless overridden with
--slot NAME=PATH):

    lora_v5_frozen_main.jsonl   v5 head scored on T1 main
                                (data/real_context_eval_v1/eval.jsonl)
    lora_v5_frozen_supp.jsonl   v5 head scored on T1 drop-supp
                                (data/real_context_eval_v1/drop_supp/eval.jsonl)
    winnow_frozen_main.jsonl    optional rescore; defaults to the incumbent
    winnow_frozen_supp.jsonl    winnow result files in --incumbent-dir
    lora_v5_synth_v4.jsonl      v5 head on data/valen_nano_v4/eval.jsonl (G4a)
    lora_v5_synth_v5.jsonl      v5 head on the context_relevance_v5 eval (G4b)
    lora_v5_seen.jsonl          v5 head on seen-source eval (G3)
    lora_v5_transfer.jsonl      v5 head on the T4 transfer holdout (G3)
    lora_v5_mined_ext.jsonl     optional: v5 head on T2 real-v5-ext (feeds G5)

Each results row follows scripts/eval_systemone_backend_v1.py:
    {"i": <index into the scored data file>, "ms": ..., "noul": <float|null>,
     "model": ..., "target": <float|null>, "error": ...}
Rows may also carry "record_id" directly (provence-style), which takes
precedence over index alignment.

Alignment: row "i" indexes the file the run was scored on (--scored-on-*
overrides; default = the tier eval file itself; the incumbent lora main file
was scored on data/real_context_eval_v1/candidates.jsonl). record_id joins
to the labeled eval file(s), so unlabeled/excluded candidates simply never
match a label and are skipped.

Usage:
    python3 scripts/check_v5_gates_v1.py --results-dir results/v5_acc_v1
    python3 scripts/check_v5_gates_v1.py --results-dir results/v5_acc_v1 \
        --out results/v5_acc_v1/gate_report.json
Exit: 0 all gates pass; 1 any fail/tripwire; 2 otherwise-not-ready.
"""

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# ------------------------------------------------------- thresholds (§9) --

INCUMBENT = {  # lora_v4 measured on the frozen 590, REAL_CONTEXT_EVAL_RESULTS_V1
    "acc": 0.9453, "fp": 11, "fn": 21, "drop_recall": 0.625,
    "consensus_recall_floor": 0.179,   # lora∧winnow AND-drop recall
    "confident_fp_band": (0.9, 1.0 + 1e-9),  # noul>=0.9 FP must be zero
}
G3_MAX_GAP = 0.10
G4_MIN_ACC = 0.99
TAIL_BAND = (0.55, 0.92)             # G6 keep-side FP tail band
ECE_BINS = 15
DEFAULT_ECE_MAX = 0.05               # used only when no incumbent baseline
PROTECTED_FAMILIES = {
    "f1_antishortcut", "anti_shortcut", "f4_outcome_positive",
    "outcome_positive",
}

RESULT_SLOTS = {  # name -> default filename inside --results-dir
    "frozen_main": "lora_v5_frozen_main.jsonl",
    "frozen_supp": "lora_v5_frozen_supp.jsonl",
    "winnow_main": "winnow_frozen_main.jsonl",
    "winnow_supp": "winnow_frozen_supp.jsonl",
    "synth_v4": "lora_v5_synth_v4.jsonl",
    "synth_v5": "lora_v5_synth_v5.jsonl",
    "seen": "lora_v5_seen.jsonl",
    "transfer": "lora_v5_transfer.jsonl",
    "mined_ext": "lora_v5_mined_ext.jsonl",
}
INCUMBENT_FILES = {  # inside --incumbent-dir
    "frozen_main": "lora_real_candidates_v1.jsonl",
    "frozen_supp": "lora_drop_supp_v1.jsonl",
    "winnow_main": "winnow_real_candidates_full_v1.jsonl",
    "winnow_supp": "winnow_drop_supp_v1.jsonl",
}


# ------------------------------------------------------------- loading ----

def _rid(rec):
    meta = rec.get("meta") or {}
    return meta.get("record_id") or rec.get("record_id")


def _target(rec):
    tgt = ((rec.get("targets") or {}).get("irrelevant") or {}).get(
        "probabilities") or {}
    return tgt.get("true")


def load_eval(path):
    """valen-format eval.jsonl -> {rid: {"target": float, "meta": dict}}."""
    out = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            rid = _rid(rec)
            if rid is None:
                rid = f"__idx_{len(out)}"
            out[rid] = {"target": _target(rec), "meta": rec.get("meta") or {},
                        "group_id": rec.get("group_id")}
    return out


def load_index(path):
    """File a results run was scored on -> {i: record_id}."""
    idx = {}
    with open(path) as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            idx[i] = _rid(json.loads(line))
    return idx


def align_results(results_path, index_path, label_maps):
    """Join a results jsonl to labels.

    label_maps: list of {rid: entry} dicts (eval files). Returns
    {"items": [...], "n_rows": int, "n_unlabeled": int, "n_null": int}.
    An item = {"rid", "noul", "target", "meta"} with target != None and
    noul != None.
    """
    items, n_rows, n_unlabeled, n_null = [], 0, 0, 0
    index = load_index(index_path) if index_path else {}
    labels = {}
    for lm in label_maps:
        labels.update(lm)
    with open(results_path) as f:
        for line in f:
            if not line.strip():
                continue
            n_rows += 1
            row = json.loads(line)
            rid = row.get("record_id") or index.get(row.get("i"))
            entry = labels.get(rid) if rid is not None else None
            target = entry["target"] if entry else row.get("target")
            meta = entry["meta"] if entry else {}
            if target is None:
                n_unlabeled += 1
                continue
            noul = row.get("noul")
            if noul is None:
                n_null += 1
                continue
            items.append({"rid": rid, "noul": float(noul),
                          "target": float(target), "meta": meta})
    return {"items": items, "n_rows": n_rows,
            "n_unlabeled": n_unlabeled, "n_null": n_null}


# ------------------------------------------------------------- metrics ----

def confusion(items, tau):
    tp = fp = fn = tn = 0
    for it in items:
        drop_hat = it["noul"] >= tau
        drop = it["target"] >= 0.5
        if drop_hat and drop:
            tp += 1
        elif drop_hat:
            fp += 1
        elif drop:
            fn += 1
        else:
            tn += 1
    n = tp + fp + fn + tn
    return {
        "n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "acc": (tp + tn) / n if n else None,
        "drop_recall": tp / (tp + fn) if (tp + fn) else None,
        "drop_precision": tp / (tp + fp) if (tp + fp) else None,
    }


def fp_at(items, lo, hi):
    return sum(1 for it in items
               if it["target"] < 0.5 and lo <= it["noul"] < hi)


def ece(items, bins=ECE_BINS):
    """Expected calibration error of noul as P(drop)."""
    n = len(items)
    if not n:
        return None
    total = 0.0
    per_bin = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        cell = [it for it in items
                if lo <= it["noul"] < hi or (b == bins - 1
                                             and it["noul"] == hi)]
        if not cell:
            continue
        mp = sum(it["noul"] for it in cell) / len(cell)
        mt = sum(1 if it["target"] >= 0.5 else 0 for it in cell) / len(cell)
        per_bin.append({"bin": b, "n": len(cell), "gap": abs(mp - mt)})
        total += len(cell) / n * abs(mp - mt)
    return {"ece": total, "bins": per_bin}


def paired_bootstrap_ci(items_a, items_b, stat, iters=2000, seed=20261201):
    """CI of stat(a) - stat(b) on rid-paired items. items_* are lists of
    dicts sharing 'rid'. Returns (lo, hi, delta) or None."""
    a = {it["rid"]: it for it in items_a}
    b = {it["rid"]: it for it in items_b}
    rids = sorted(set(a) & set(b))
    if len(rids) < 10:
        return None
    rng = random.Random(seed)
    delta = stat([a[r] for r in rids]) - stat([b[r] for r in rids])
    boots = []
    for _ in range(iters):
        samp = [rng.choice(rids) for _ in rids]
        boots.append(stat([a[r] for r in samp]) - stat([b[r] for r in samp]))
    boots.sort()
    return (boots[int(0.025 * iters)], boots[int(0.975 * iters)], delta)


# ------------------------------------------------------------- context ----

class Ctx:
    def __init__(self, args):
        self.args = args
        self.missing_evals = []
        self.labels = {}
        self.indexes = {}

    def eval_map(self, path):
        path = str(path)
        if path not in self.labels:
            if not Path(path).exists():
                self.missing_evals.append(path)
                self.labels[path] = None
            else:
                self.labels[path] = load_eval(path)
        return self.labels[path]

    def index_map(self, path):
        path = str(path)
        if path not in self.indexes:
            self.indexes[path] = load_index(path) if Path(path).exists() \
                else None
        return self.indexes[path]

    def slot(self, name):
        """Resolve a results slot to a path or None."""
        ov = self.args.slot.get(name)
        if ov:
            p = Path(ov)
            return p if p.exists() else p  # keep missing path for reporting
        p = Path(self.args.results_dir) / RESULT_SLOTS[name]
        if p.exists():
            return p
        # incumbent fallback applies ONLY to the winnow consensus partner —
        # the v5 slots must never silently bind to incumbent files
        if name.startswith("winnow_"):
            inc = INCUMBENT_FILES.get(name)
            if inc:
                ip = Path(self.args.incumbent_dir) / inc
                if ip.exists():
                    return ip
            return None
        return p

    def align(self, name, index_path, eval_paths):
        rp = self.slot(name)
        if rp is None or not Path(rp).exists():
            return None, f"results slot '{name}' missing ({RESULT_SLOTS[name]})"
        maps = []
        for ep in eval_paths:
            m = self.eval_map(ep)
            if m is None:
                return None, f"eval file missing: {ep}"
            maps.append(m)
        ip = index_path if index_path and Path(index_path).exists() else None
        if ip is None and index_path:
            # fall back to first eval file as the index
            ip = eval_paths[0]
        elif ip is None:
            ip = eval_paths[0]
        res = align_results(rp, ip, maps)
        res["results_file"] = str(rp)
        res["index_file"] = str(ip)
        return res, None


# ---------------------------------------------------------------- gates ---

def gate_g1(ctx):
    """G1 real-frozen superiority: beat lora_v4 on the frozen 590."""
    tau = ctx.args.tau
    parts, missing = [], []
    for slot, ev in (("frozen_main", [ctx.args.frozen_main]),
                     ("frozen_supp", [ctx.args.frozen_supp])):
        idx = (ctx.args.frozen_candidates if slot == "frozen_main"
               else None)
        res, err = ctx.align(slot, idx, ev)
        if err:
            missing.append(err)
        else:
            parts.append(res)
    if missing:
        return {"gate": "G1", "status": "not_ready", "missing": missing}
    items = [it for p in parts for it in p["items"]]
    m = confusion(items, tau)
    hi_fp = fp_at(items, INCUMBENT["confident_fp_band"][0],
                  INCUMBENT["confident_fp_band"][1])
    measured = {"tau": tau, **{k: m[k] for k in
                               ("n", "acc", "tp", "fp", "fn", "tn",
                                "drop_recall", "drop_precision")},
                "confident_fp_noul_ge_0_9": hi_fp}
    # incumbent reproduction + paired delta when files are present
    inc_items = []
    for name, idxf, evf in (
            ("frozen_main", ctx.args.frozen_candidates,
             [ctx.args.frozen_main]),
            ("frozen_supp", None, [ctx.args.frozen_supp])):
        ip = Path(ctx.args.incumbent_dir) / INCUMBENT_FILES[name]
        if not ip.exists():
            continue
        maps = [ctx.eval_map(e) for e in evf]
        if any(m is None for m in maps):
            continue
        idx_p = idxf if idxf and Path(idxf).exists() else evf[0]
        inc_items += align_results(ip, idx_p, maps)["items"]
    if inc_items:
        im = confusion(inc_items, tau)
        measured["incumbent"] = {"n": im["n"], "acc": im["acc"],
                                 "fp": im["fp"], "fn": im["fn"],
                                 "drop_recall": im["drop_recall"]}
        ci = paired_bootstrap_ci(
            items, inc_items,
            lambda xs: sum(1 for x in xs
                           if (x["noul"] >= tau) == (x["target"] >= 0.5))
            / len(xs))
        if ci:
            measured["acc_delta_vs_incumbent_ci95"] = ci
    checks = {
        "acc_gt_incumbent_0.9453": (m["acc"] or 0) > INCUMBENT["acc"],
        "drop_recall_gt_0.625": (m["drop_recall"] or 0)
                                > INCUMBENT["drop_recall"],
        "fp_lt_11": m["fp"] < INCUMBENT["fp"],
        "zero_confident_fp": hi_fp == 0,
    }
    return {"gate": "G1", "status": "pass" if all(checks.values())
            else "fail", "measured": measured, "checks": checks,
            "criteria": "acc>0.9453 AND drop_recall>62.5% AND fp<11 AND "
                        "fp@noul>=0.9==0 (§9)"}


def gate_g2(ctx):
    """G2 consensus preserved: v5 ∧ winnow AND-drop, FP=0, recall>=17.9%."""
    tau = ctx.args.tau
    v5, win, missing = {}, {}, []
    for side, slots in (("v5", ("frozen_main", "frozen_supp")),
                        ("winnow", ("winnow_main", "winnow_supp"))):
        for slot, sub in (("frozen_main", "main"), ("frozen_supp", "supp")):
            name = slot if side == "v5" else f"winnow_{sub}"
            idx = ctx.args.frozen_candidates if sub == "main" else None
            ev = ctx.args.frozen_main if sub == "main" \
                else ctx.args.frozen_supp
            res, err = ctx.align(name, idx, [ev])
            if err:
                missing.append(err)
            else:
                tgt = v5 if side == "v5" else win
                for it in res["items"]:
                    tgt[it["rid"]] = it
    if missing:
        return {"gate": "G2", "status": "not_ready", "missing": missing}
    both = [r for r in v5 if r in win]
    drops = [v5[r]["noul"] >= tau and win[r]["noul"] >= tau for r in both]
    labels = [v5[r]["target"] >= 0.5 for r in both]
    fp = sum(1 for d, l in zip(drops, labels) if d and not l)
    tp = sum(1 for d, l in zip(drops, labels) if d and l)
    ndrop = sum(labels)
    recall = tp / ndrop if ndrop else None
    prec = tp / (tp + fp) if (tp + fp) else (1.0 if fp == 0 else 0.0)
    checks = {"fp_zero": fp == 0, "precision_1": fp == 0,
              "recall_gte_0.179": (recall or 0)
                                  >= INCUMBENT["consensus_recall_floor"]}
    return {"gate": "G2", "status": "pass" if all(checks.values())
            else "fail",
            "measured": {"tau": tau, "double_covered": len(both),
                         "and_drop_fp": fp, "and_drop_tp": tp,
                         "drop_precision": prec, "drop_recall": recall},
            "checks": checks,
            "criteria": "lora_v5∧winnow AND-drop: FP=0, precision=1.000, "
                        "recall>=17.9% on double-covered intersect (§9)"}


def gate_g3(ctx):
    """G3 transfer gap: seen-source acc - transfer acc <= 10pp."""
    seen_res, err1 = ctx.align("seen", None, [ctx.args.seen_eval])
    tr_res, err2 = ctx.align("transfer", None, [ctx.args.transfer_eval])
    missing = [e for e in (err1, err2) if e]
    if missing:
        return {"gate": "G3", "status": "not_ready", "missing": missing,
                "note": "T4 transfer holdout is a placeholder until the v5 "
                        "train manifest fixes the withheld source family"}
    tau = ctx.args.tau
    sa = confusion(seen_res["items"], tau)
    ta = confusion(tr_res["items"], tau)
    gap = (sa["acc"] - ta["acc"]) if (sa["acc"] is not None
                                    and ta["acc"] is not None) else None
    ok = gap is not None and gap <= G3_MAX_GAP
    return {"gate": "G3", "status": "pass" if ok else "fail",
            "measured": {"tau": tau, "seen_acc": sa["acc"],
                         "seen_n": sa["n"], "transfer_acc": ta["acc"],
                         "transfer_n": ta["n"], "gap_pp":
                         None if gap is None else round(gap * 100, 3)},
            "checks": {"gap_le_10pp": ok},
            "criteria": "seen-source acc - transfer-holdout acc <= 10pp (§9)"}


def gate_g4(ctx):
    """G4 synthetic non-regression + anti-saturation tripwire."""
    tau = ctx.args.tau
    blo, bhi = ctx.args.borderline
    legs, missing, checks, tripwire = {}, [], {}, False

    res_a, err = ctx.align("synth_v4", None, [ctx.args.synth_v4_eval])
    if err:
        missing.append(f"leg_a: {err}")
    else:
        m = confusion(res_a["items"], tau)
        legs["v4_eval"] = {"n": m["n"], "acc": m["acc"]}
        checks["v4_acc_gte_0.99"] = (m["acc"] or 0) >= G4_MIN_ACC

    res_b, err = ctx.align("synth_v5", None, [ctx.args.synth_v5_eval])
    if err:
        missing.append(f"leg_b: {err}")
    else:
        m = confusion(res_b["items"], tau)
        border = sum(1 for it in res_b["items"] if blo < it["noul"] < bhi)
        legs["v5_synth_eval"] = {"n": m["n"], "acc": m["acc"],
                                 "borderline_predictions": border,
                                 "borderline_band": [blo, bhi]}
        checks["v5_acc_gte_0.99"] = (m["acc"] or 0) >= G4_MIN_ACC
        if (m["acc"] or 0) >= G4_MIN_ACC and border == 0:
            tripwire = True
        checks["no_saturation_tripwire"] = not tripwire

    if not legs:
        return {"gate": "G4", "status": "not_ready", "missing": missing}
    if tripwire:
        status = "tripwire"
    elif not all(checks.values()):
        status = "fail"
    elif missing:
        status = "not_ready"  # measured legs pass; other leg still pending
    else:
        status = "pass"
    out = {"gate": "G4", "status": status, "measured": legs,
           "checks": checks, "missing": missing,
           "criteria": "v4 eval acc>=0.99 (non-regression) AND v5 synth "
                       "acc>=0.99 AND >=1 borderline prediction — acc~1.0 "
                       "with zero borderline is a memorization tripwire: "
                       "extend the set, do not claim victory (§9)"}
    if tripwire:
        out["action"] = ("extend the context_relevance_v5 eval pool with "
                         "harder families before claiming victory")
    return out


def gate_g5(ctx):
    """G5 asymmetric posture: zero FP on protected-family keep rows and a
    safe-sided error structure (FP <= FN)."""
    tau = ctx.args.tau
    sets, missing = [], []
    slots = [("frozen_main", ctx.args.frozen_candidates,
              [ctx.args.frozen_main]),
             ("frozen_supp", None, [ctx.args.frozen_supp]),
             ("mined_ext", None, [ctx.args.mined_ext_eval]),
             ("synth_v5", None, [ctx.args.synth_v5_eval]),
             ("transfer", None, [ctx.args.transfer_eval]),
             ("seen", None, [ctx.args.seen_eval])]
    for name, idx, evs in slots:
        if any(not Path(e).exists() for e in evs):
            continue
        res, err = ctx.align(name, idx, evs)
        if err:
            continue
        if res["items"]:
            sets.append((name, res["items"]))
    if not sets:
        return {"gate": "G5", "status": "not_ready",
                "missing": ["no scored v5 result sets with labels found"]}

    def is_protected(meta):
        if meta.get("protected") is True:
            return True
        fam = (meta.get("family") or meta.get("sub_family") or "")
        return fam in PROTECTED_FAMILIES

    fp_total = fn_total = 0
    prot_rows = prot_fp = 0
    per_set = {}
    for name, items in sets:
        m = confusion(items, tau)
        p_rows = [it for it in items if it["target"] < 0.5
                  and is_protected(it["meta"])]
        p_fp = sum(1 for it in p_rows if it["noul"] >= tau)
        fp_total += m["fp"]
        fn_total += m["fn"]
        prot_rows += len(p_rows)
        prot_fp += p_fp
        per_set[name] = {"n": m["n"], "fp": m["fp"], "fn": m["fn"],
                         "protected_keep_rows": len(p_rows),
                         "protected_fp": p_fp}
    checks = {
        "protected_family_fp_zero": prot_fp == 0,
        "safe_sided_fp_le_fn": fp_total <= fn_total,
    }
    return {"gate": "G5",
            "status": "pass" if all(checks.values()) else "fail",
            "measured": {"tau": tau, "fp": fp_total, "fn": fn_total,
                         "protected_keep_rows": prot_rows,
                         "protected_fp": prot_fp, "per_set": per_set,
                         "protected_families": sorted(PROTECTED_FAMILIES)},
            "checks": checks,
            "criteria": "error structure stays safe-sided (FP<=FN) and zero "
                        "FP on F1/F4-type protected keep rows (V4-S0, §9)"}


def gate_g6(ctx):
    """G6 calibration: keep-side tail band must not grow; report ECE(15) +
    recall-at-low-p."""
    tau = ctx.args.tau
    lo, hi = ctx.args.tail_band
    res, err = ctx.align("frozen_main", ctx.args.frozen_candidates,
                         [ctx.args.frozen_main])
    res2, err2 = ctx.align("frozen_supp", None, [ctx.args.frozen_supp])
    if err and err2:
        return {"gate": "G6", "status": "not_ready", "missing": [err, err2]}
    items = (res["items"] if res else []) + (res2["items"] if res2 else [])
    tail_fp = fp_at(items, lo, hi + 1e-9)
    e = ece(items)
    drops = [it for it in items if it["target"] >= 0.5]
    rec_low = (sum(1 for it in drops if it["noul"] >= 0.1) / len(drops)
               if drops else None)

    # incumbent tail + ECE as the "must not grow" baseline
    inc_items = []
    for name, idxf, evf in (
            ("frozen_main", ctx.args.frozen_candidates,
             [ctx.args.frozen_main]),
            ("frozen_supp", None, [ctx.args.frozen_supp])):
        ip = Path(ctx.args.incumbent_dir) / INCUMBENT_FILES[name]
        maps_ok = all(Path(e).exists() for e in evf)
        if ip.exists() and maps_ok:
            idx_p = idxf if idxf and Path(idxf).exists() else evf[0]
            maps = [ctx.eval_map(e) for e in evf]
            inc_items += align_results(ip, idx_p, maps)["items"]
    inc_tail = fp_at(inc_items, lo, hi + 1e-9) if inc_items else 5
    inc_ece = ece(inc_items)["ece"] if inc_items else None
    ece_max = ctx.args.ece_max
    if ece_max is None:
        ece_max = (inc_ece * 1.05 + 0.005) if inc_ece is not None \
            else DEFAULT_ECE_MAX
    checks = {
        "tail_band_not_grown": tail_fp <= inc_tail,
        "ece_within_budget": (e["ece"] if e else 1.0) <= ece_max,
    }
    return {"gate": "G6",
            "status": "pass" if all(checks.values()) else "fail",
            "measured": {"n": len(items),
                         "tail_band": [lo, hi], "tail_fp": tail_fp,
                         "incumbent_tail_fp": inc_tail,
                         "ece_15bin": None if e is None else e["ece"],
                         "ece_budget": ece_max,
                         "incumbent_ece": inc_ece,
                         "drop_recall_at_p0.1": rec_low},
            "checks": checks,
            "criteria": "0.55-0.92 FP tail band must not grow vs incumbent "
                        "AND ECE(15) <= budget (§9)"}


GATES = {"G1": gate_g1, "G2": gate_g2, "G3": gate_g3,
         "G4": gate_g4, "G5": gate_g5, "G6": gate_g6}


def parse_pair(s):
    a, b = s.split(":")
    return float(a), float(b)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    ap.add_argument("--incumbent-dir", type=Path,
                    default=REPO_ROOT / "results")
    ap.add_argument("--frozen-main", type=Path,
                    default=REPO_ROOT / "data/real_context_eval_v1/eval.jsonl")
    ap.add_argument("--frozen-supp", type=Path, default=REPO_ROOT /
                    "data/real_context_eval_v1/drop_supp/eval.jsonl")
    ap.add_argument("--frozen-candidates", type=Path, default=REPO_ROOT /
                    "data/real_context_eval_v1/candidates.jsonl")
    ap.add_argument("--synth-v4-eval", type=Path,
                    default=REPO_ROOT / "data/valen_nano_v4/eval.jsonl")
    ap.add_argument("--synth-v5-eval", type=Path,
                    default=REPO_ROOT / "data/valen_nano_v5/eval.jsonl")
    ap.add_argument("--mined-ext-eval", type=Path, default=REPO_ROOT /
                    "data/real_context_eval_v5_ext/eval.jsonl")
    ap.add_argument("--seen-eval", type=Path, default=REPO_ROOT /
                    "data/v5_seen_source/eval.jsonl")
    ap.add_argument("--transfer-eval", type=Path, default=REPO_ROOT /
                    "data/v5_transfer_holdout/eval.jsonl")
    ap.add_argument("--slot", action="append", default=[],
                    help="NAME=PATH override for a results slot "
                         f"({', '.join(RESULT_SLOTS)})")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--tail-band", type=parse_pair, default=TAIL_BAND)
    ap.add_argument("--borderline", type=parse_pair, default=(0.05, 0.95))
    ap.add_argument("--ece-max", type=float, default=None,
                    help="absolute ECE budget; default = incumbent ECE * "
                         "1.05 + 0.005, or 0.05 when no incumbent")
    ap.add_argument("--only", default=None,
                    help="comma-separated gate subset, e.g. G1,G2")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    args.slot = dict(s.split("=", 1) for s in args.slot)
    ctx = Ctx(args)
    only = set(args.only.split(",")) if args.only else set(GATES)
    report = {"schema_version": "nanojev-v5-gates-v1",
              "results_dir": str(args.results_dir), "tau": args.tau,
              "gates": {}}
    for name in sorted(GATES):
        if name not in only:
            continue
        try:
            report["gates"][name] = GATES[name](ctx)
        except Exception as exc:  # never crash a gate run
            report["gates"][name] = {"gate": name, "status": "error",
                                     "error": f"{type(exc).__name__}: {exc}"}
    statuses = [g["status"] for g in report["gates"].values()]
    report["overall"] = ("fail" if any(s in ("fail", "tripwire", "error")
                                       for s in statuses)
                         else "not_ready" if any(s == "not_ready"
                                                 for s in statuses)
                         else "pass")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    for name in sorted(report["gates"]):
        g = report["gates"][name]
        print(f"{name}: {g['status'].upper()}")
        if g.get("measured") is not None:
            slim = {k: v for k, v in g["measured"].items()
                    if not isinstance(v, (list, dict))}
            print("   ", json.dumps(slim))
        for miss in g.get("missing") or []:
            print(f"    missing: {miss}")
    print(f"overall: {report['overall'].upper()}")
    if report["overall"] == "fail":
        return 1
    if report["overall"] == "not_ready":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
