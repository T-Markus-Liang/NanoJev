#!/usr/bin/env python3
"""Synthetic-fixture tests for scripts/check_v5_gates_v1.py.

Fabricates a tiny eval tree + results dirs under a temp dir and asserts each
gate (G1-G6) flips correctly: one passing case and at least one failing case
per gate, plus a not_ready case when result files are absent. No network, no
model calls, no real data touched — every path is overridden to the fixture.

Usage: python3 scripts/test_v5_gates_v1.py   (exit 0 = all assertions pass)
"""

import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve()
SPEC = importlib.util.spec_from_file_location(
    "check_v5_gates_v1", HERE.parent / "check_v5_gates_v1.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}  {detail}")


# ------------------------------------------------------------- fixtures ---

def rec(rid, drop, family=None):
    meta = {"record_id": rid}
    if family:
        meta["family"] = family
    return {"group_id": f"g:{rid.split(':')[0]}",
            "request": {"state": "{}", "questions": {}},
            "targets": {"irrelevant": {"probabilities":
                                     {"true": 1.0 if drop else 0.0,
                                      "false": 0.0 if drop else 1.0}}},
            "meta": meta}


def res(i, noul):
    return {"i": i, "ms": 1.0, "noul": noul, "model": "fixture"}


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def build_eval_tree(root):
    """Eval files shared by all cases.

    frozen: 120 records = 80 main (60 keep m00-m59 / 20 drop m60-m79)
            + 40 supp (30 keep s00-s29 / 10 drop s30-s39)
    synth_v4: 100 records (50 keep a / 50 drop a)
    synth_v5: 60 records (30 keep: 10 protected f1 + 20 plain / 30 drop)
    seen + transfer: 50 records each (25 keep / 25 drop)
    """
    ev = root / "eval"
    main = ([rec(f"m{i:02d}", False) for i in range(60)]
            + [rec(f"m{i:02d}", True) for i in range(60, 80)])
    write_jsonl(ev / "frozen_main.jsonl", main)
    # incumbent-era runs were scored on a candidates superset; here it is
    # identical to the eval file (index identity) plus 2 excluded rows
    write_jsonl(ev / "candidates.jsonl",
                main + [rec("mx0", False), rec("mx1", False)])
    supp = ([rec(f"s{i:02d}", False) for i in range(30)]
            + [rec(f"s{i:02d}", True) for i in range(30, 40)])
    write_jsonl(ev / "frozen_supp.jsonl", supp)
    write_jsonl(ev / "synth_v4.jsonl",
                [rec(f"a{i:02d}", False) for i in range(50)]
                + [rec(f"a{i:02d}", True) for i in range(50, 100)])
    write_jsonl(ev / "synth_v5.jsonl",
                [rec(f"b{i:02d}", False, "f1_antishortcut")
                 for i in range(10)]
                + [rec(f"b{i:02d}", False) for i in range(10, 30)]
                + [rec(f"b{i:02d}", True, "f2_hardneg")
                   for i in range(30, 60)])
    write_jsonl(ev / "seen.jsonl",
                [rec(f"e{i:02d}", False) for i in range(25)]
                + [rec(f"e{i:02d}", True) for i in range(25, 50)])
    write_jsonl(ev / "transfer.jsonl",
                [rec(f"t{i:02d}", False) for i in range(25)]
                + [rec(f"t{i:02d}", True) for i in range(25, 50)])
    return ev


def inc_fixture(root):
    """Incumbent (lora_v4 + winnow) result files scored on candidates/eval.

    lora_v4: fp=4 (m00@0.6, m01@0.7, m02@0.55, m03@0.93), fn=5
        -> acc=(120-9)/120, drop recall 25/30, tail-band(0.55-0.92) fp=3.
    winnow: fires noul>=0.9 on the first 6 true drops of each file only.
    """
    inc = root / "incumbent"
    lora_main, win_main = [], []
    for i in range(80):
        drop = i >= 60
        lp = 0.05
        if not drop and i in (0, 1, 2, 3):
            lp = (0.6, 0.7, 0.55, 0.93)[i]
        elif drop:
            lp = 0.9 if i < 75 else 0.1          # tp for m60-74, fn m75-79
        wp = 0.9 if (drop and i < 66) else 0.05  # winnow hits m60-65
        lora_main.append(res(i, lp))
        win_main.append(res(i, wp))
    lora_supp, win_supp = [], []
    for i in range(40):
        drop = i >= 30
        lp = 0.9 if drop and i < 37 else (0.4 if drop else 0.05)
        wp = 0.9 if (drop and i < 34) else 0.05
        lora_supp.append(res(i, lp))
        win_supp.append(res(i, wp))
    write_jsonl(inc / "lora_real_candidates_v1.jsonl", lora_main)
    write_jsonl(inc / "lora_drop_supp_v1.jsonl", lora_supp)
    write_jsonl(inc / "winnow_real_candidates_full_v1.jsonl", win_main)
    write_jsonl(inc / "winnow_drop_supp_v1.jsonl", win_supp)
    return inc


def v5_frozen(kind):
    """v5 results on the frozen pair.

    kind="pass": fp=2 (m10@0.7, s00@0.8), fn=3 -> acc=(120-5)/120=0.9583,
        drop recall 27/30=0.9, fp<11, no fp>=0.9.
    kind="confident_fp": same + m04@0.95 -> violates zero-confident-fp.
    kind="tail_grow": moves fp's into the tail band, 4 of them.
    """
    main, supp = [], []
    fp_idx = {"pass": {10: 0.7}, "confident_fp": {10: 0.7, 4: 0.95},
              "tail_grow": {10: 0.6, 11: 0.7, 12: 0.8}}[kind]
    for i in range(80):
        drop = i >= 60
        if not drop:
            p = fp_idx.get(i, 0.05)
        else:
            p = 0.9 if i < 78 else 0.3   # tp m60-77, fn m78-79
        main.append(res(i, p))
    for i in range(40):
        drop = i >= 30
        if not drop:
            p = 0.8 if (i == 0 and kind in ("pass", "confident_fp")) else (
                0.6 if (i == 0 and kind == "tail_grow") else 0.05)
        else:
            p = 0.9 if i < 39 else 0.3   # tp s30-38, fn s39
        supp.append(res(i, p))
    return main, supp


def run_case(root, ev, inc, v5files, extra=None):
    rdir = root / "res"
    if rdir.exists():
        shutil.rmtree(rdir)
    rdir.mkdir(parents=True)
    for name, rows in v5files.items():
        write_jsonl(rdir / name, rows)
    out = root / "report.json"
    argv = ["--results-dir", str(rdir),
            "--repo-root", str(root),
            "--incumbent-dir", str(inc),
            "--frozen-main", str(ev / "frozen_main.jsonl"),
            "--frozen-supp", str(ev / "frozen_supp.jsonl"),
            "--frozen-candidates", str(ev / "candidates.jsonl"),
            "--synth-v4-eval", str(ev / "synth_v4.jsonl"),
            "--synth-v5-eval", str(ev / "synth_v5.jsonl"),
            "--seen-eval", str(ev / "seen.jsonl"),
            "--transfer-eval", str(ev / "transfer.jsonl"),
            "--out", str(out)] + (extra or [])
    code = MOD.main(argv)
    return code, json.loads(out.read_text())


def synth_v4_rows(errors=0):
    rows = []
    err_done = 0
    for i in range(100):
        drop = i >= 50
        p = (0.9 if drop else 0.05)
        if err_done < errors and not drop:
            p = 0.9
            err_done += 1
        rows.append(res(i, p))
    return rows


def synth_v5_rows(borderline=True, protected_fp=False):
    rows = []
    for i in range(60):
        drop = i >= 30
        p = 0.98 if drop else 0.02
        if i == 40:                      # one mid-band prediction on a drop
            p = 0.6 if borderline else 0.98
        if protected_fp and i == 0:      # protected f1 keep dropped
            p = 0.9
        rows.append(res(i, p))
    return rows


def seen_rows(errs_fn=3):
    rows = []
    done = 0
    for i in range(50):
        drop = i >= 25
        p = 0.9 if drop else 0.05
        if drop and done < errs_fn:
            p = 0.2
            done += 1
        rows.append(res(i, p))
    return rows


def transfer_rows(errs_fn=5):
    rows = []
    done = 0
    for i in range(50):
        drop = i >= 25
        p = 0.9 if drop else 0.05
        if drop and done < errs_fn:
            p = 0.2
            done += 1
        rows.append(res(i, p))
    return rows


def main():
    root = Path(tempfile.mkdtemp(prefix="v5gates_test_"))
    try:
        ev = build_eval_tree(root)
        inc = inc_fixture(root)

        print("== G1 real-frozen superiority")
        m, s = v5_frozen("pass")
        code, rep = run_case(root, ev, inc, {
            "lora_v5_frozen_main.jsonl": m,
            "lora_v5_frozen_supp.jsonl": s},
            extra=["--only", "G1"])
        g = rep["gates"]["G1"]
        check("G1 pass case", g["status"] == "pass", json.dumps(g))
        check("G1 acc measured", abs(g["measured"]["acc"] - 115 / 120)
              < 1e-9, g["measured"].get("acc"))
        check("G1 incumbent reproduced",
              g["measured"]["incumbent"]["fp"] == 4)
        m, s = v5_frozen("confident_fp")
        code, rep = run_case(root, ev, inc, {
            "lora_v5_frozen_main.jsonl": m,
            "lora_v5_frozen_supp.jsonl": s}, extra=["--only", "G1"])
        g = rep["gates"]["G1"]
        check("G1 fail on confident fp", g["status"] == "fail"
              and not g["checks"]["zero_confident_fp"], json.dumps(g))
        code, rep = run_case(root, ev, inc, {}, extra=["--only", "G1"])
        check("G1 not_ready when absent",
              rep["gates"]["G1"]["status"] == "not_ready")

        print("== G2 consensus preserved")
        m, s = v5_frozen("pass")
        code, rep = run_case(root, ev, inc, {
            "lora_v5_frozen_main.jsonl": m,
            "lora_v5_frozen_supp.jsonl": s},
            extra=["--only", "G2"])
        g = rep["gates"]["G2"]
        # v5 fires on m60-77,s30-38; winnow(incumbent) fires m60-65,s30-33
        # -> AND tp=10, recall=10/30>0.179, fp=0
        check("G2 pass case", g["status"] == "pass", json.dumps(g))
        check("G2 recall", abs(g["measured"]["drop_recall"] - 10 / 30)
              < 1e-9, g["measured"])
        # failing: winnow+v5 both fire on keep m20
        win_m = [res(i, 0.05) for i in range(80)]
        for i in range(60, 66):
            win_m[i]["noul"] = 0.9
        win_m[20]["noul"] = 0.9           # FP on a keep
        win_s = [res(i, 0.05) for i in range(40)]
        for i in range(30, 34):
            win_s[i]["noul"] = 0.9
        m2 = [dict(r) for r in m]
        m2[20]["noul"] = 0.9              # v5 also fires -> AND-drop FP
        code, rep = run_case(root, ev, inc, {
            "lora_v5_frozen_main.jsonl": m2,
            "lora_v5_frozen_supp.jsonl": s,
            "winnow_frozen_main.jsonl": win_m,
            "winnow_frozen_supp.jsonl": win_s},
            extra=["--only", "G2"])
        g = rep["gates"]["G2"]
        check("G2 fail on consensus fp", g["status"] == "fail"
              and g["measured"]["and_drop_fp"] == 1, json.dumps(g))

        print("== G3 transfer gap")
        code, rep = run_case(root, ev, inc, {
            "lora_v5_seen.jsonl": seen_rows(3),
            "lora_v5_transfer.jsonl": transfer_rows(5)},
            extra=["--only", "G3"])
        g = rep["gates"]["G3"]
        check("G3 pass case (gap 4pp)", g["status"] == "pass"
              and abs(g["measured"]["transfer_acc"] - 0.9) < 1e-9,
              json.dumps(g))
        code, rep = run_case(root, ev, inc, {
            "lora_v5_seen.jsonl": seen_rows(0),
            "lora_v5_transfer.jsonl": transfer_rows(10)},
            extra=["--only", "G3"])
        g = rep["gates"]["G3"]
        check("G3 fail case (gap 20pp)", g["status"] == "fail"
              and g["measured"]["gap_pp"] > 10, json.dumps(g))
        code, rep = run_case(root, ev, inc, {}, extra=["--only", "G3"])
        check("G3 not_ready when absent",
              rep["gates"]["G3"]["status"] == "not_ready")

        print("== G4 synthetic non-regression + tripwire")
        code, rep = run_case(root, ev, inc, {
            "lora_v5_synth_v4.jsonl": synth_v4_rows(0),
            "lora_v5_synth_v5.jsonl": synth_v5_rows(borderline=True)},
            extra=["--only", "G4"])
        g = rep["gates"]["G4"]
        check("G4 pass case", g["status"] == "pass", json.dumps(g))
        code, rep = run_case(root, ev, inc, {
            "lora_v5_synth_v4.jsonl": synth_v4_rows(5),
            "lora_v5_synth_v5.jsonl": synth_v5_rows(borderline=True)},
            extra=["--only", "G4"])
        g = rep["gates"]["G4"]
        check("G4 fail on v4 regression", g["status"] == "fail"
              and not g["checks"]["v4_acc_gte_0.99"], json.dumps(g))
        code, rep = run_case(root, ev, inc, {
            "lora_v5_synth_v4.jsonl": synth_v4_rows(0),
            "lora_v5_synth_v5.jsonl": synth_v5_rows(borderline=False)},
            extra=["--only", "G4"])
        g = rep["gates"]["G4"]
        check("G4 tripwire on saturation", g["status"] == "tripwire",
              json.dumps(g))
        code, rep = run_case(root, ev, inc, {
            "lora_v5_synth_v4.jsonl": synth_v4_rows(0)},
            extra=["--only", "G4"])
        g = rep["gates"]["G4"]
        check("G4 not_ready with only leg A",
              g["status"] == "not_ready", json.dumps(g))

        print("== G5 asymmetric posture")
        m, s = v5_frozen("pass")
        code, rep = run_case(root, ev, inc, {
            "lora_v5_frozen_main.jsonl": m,
            "lora_v5_frozen_supp.jsonl": s,
            "lora_v5_synth_v5.jsonl": synth_v5_rows(borderline=True),
            "lora_v5_seen.jsonl": seen_rows(3),
            "lora_v5_transfer.jsonl": transfer_rows(5)},
            extra=["--only", "G5"])
        g = rep["gates"]["G5"]
        # fp = 2 (frozen) ; fn = 3+8 ; protected fp = 0 -> pass
        check("G5 pass case", g["status"] == "pass", json.dumps(g))
        check("G5 sees protected rows",
              g["measured"]["protected_keep_rows"] == 10,
              g["measured"].get("protected_keep_rows"))
        code, rep = run_case(root, ev, inc, {
            "lora_v5_synth_v5.jsonl": synth_v5_rows(protected_fp=True)},
            extra=["--only", "G5"])
        g = rep["gates"]["G5"]
        check("G5 fail on protected fp", g["status"] == "fail"
              and g["measured"]["protected_fp"] == 1, json.dumps(g))
        # unsafe-sided structure: fp>fn (only fp, no fn)
        m3 = [res(i, 0.9 if i in (10, 11, 12) else (
            0.95 if i >= 60 else 0.05)) for i in range(80)]
        code, rep = run_case(root, ev, inc, {
            "lora_v5_frozen_main.jsonl": m3}, extra=["--only", "G5"])
        g = rep["gates"]["G5"]
        check("G5 fail on unsafe-sided fp>fn", g["status"] == "fail"
              and not g["checks"]["safe_sided_fp_le_fn"], json.dumps(g))

        print("== G6 calibration")
        m, s = v5_frozen("pass")
        code, rep = run_case(root, ev, inc, {
            "lora_v5_frozen_main.jsonl": m,
            "lora_v5_frozen_supp.jsonl": s},
            extra=["--only", "G6", "--ece-max", "0.5"])
        g = rep["gates"]["G6"]
        check("G6 pass case", g["status"] == "pass", json.dumps(g))
        check("G6 incumbent tail measured",
              g["measured"]["incumbent_tail_fp"] == 3, g["measured"])
        m, s = v5_frozen("tail_grow")
        code, rep = run_case(root, ev, inc, {
            "lora_v5_frozen_main.jsonl": m,
            "lora_v5_frozen_supp.jsonl": s},
            extra=["--only", "G6", "--ece-max", "0.5"])
        g = rep["gates"]["G6"]
        check("G6 fail on tail growth", g["status"] == "fail"
              and g["measured"]["tail_fp"]
              > g["measured"]["incumbent_tail_fp"], json.dumps(g))
        code, rep = run_case(root, ev, inc, {
            "lora_v5_frozen_main.jsonl": v5_frozen("pass")[0]},
            extra=["--only", "G6", "--ece-max", "0.0001"])
        g = rep["gates"]["G6"]
        check("G6 fail on ece budget", g["status"] == "fail"
              and not g["checks"]["ece_within_budget"], json.dumps(g))

        print("== overall / not_ready")
        code, rep = run_case(root, ev, inc, {})
        check("empty results dir -> all not_ready",
              rep["overall"] == "not_ready" and code == 2,
              f"overall={rep['overall']} code={code}")

        print("== real-data smoke: incumbent reproduces the frozen-590 table")
        repo = HERE.parents[1]
        code, rep = run_case(root, ev, inc, {
            "lora_v5_frozen_main.jsonl": list(map(
                json.loads, open(repo / "results/lora_real_candidates_v1.jsonl"))),
            "lora_v5_frozen_supp.jsonl": list(map(
                json.loads, open(repo / "results/lora_drop_supp_v1.jsonl")))},
            extra=["--only", "G1",
                   "--frozen-main", str(repo /
                                        "data/real_context_eval_v1/eval.jsonl"),
                   "--frozen-supp", str(repo /
                                        "data/real_context_eval_v1/drop_supp/eval.jsonl"),
                   "--frozen-candidates", str(repo /
                                              "data/real_context_eval_v1/candidates.jsonl"),
                   "--incumbent-dir", str(repo / "results")])
        g = rep["gates"]["G1"]
        check("real incumbent acc=0.9453",
              abs(g["measured"]["acc"] - 0.945299) < 1e-4,
              g["measured"].get("acc"))
        check("real incumbent fp=11 fn=21 recall=0.625",
              g["measured"]["fp"] == 11 and g["measured"]["fn"] == 21
              and abs(g["measured"]["drop_recall"] - 0.625) < 1e-9,
              json.dumps({k: g["measured"][k] for k in ("fp", "fn",
                                                       "drop_recall")}))

    finally:
        shutil.rmtree(root, ignore_errors=True)

    print(f"\n{PASS} passed, {FAIL} failed")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
