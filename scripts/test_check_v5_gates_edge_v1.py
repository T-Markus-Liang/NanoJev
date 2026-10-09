#!/usr/bin/env python3
"""Edge-case tests for scripts/check_v5_gates_v1.py.

Covers empty/missing inputs, malformed JSONL lines, threshold boundary values exactly at gate cutoffs,
and NaN or missing metric fields. Follows the existing test style in test_v5_gates_v1.py.
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

def rec(rid, drop, target_val=None, meta_override=None):
    meta = {"record_id": rid}
    if meta_override:
        meta.update(meta_override)
    tgt = target_val if target_val is not None else (1.0 if drop else 0.0)
    return {"group_id": f"g:{rid.split(':')[0]}",
            "request": {"state": "{}", "questions": {}},
            "targets": {"irrelevant": {"probabilities":
                                     {"true": tgt,
                                      "false": 1.0 - tgt if isinstance(tgt, float) else tgt}}},
            "meta": meta}

def res(i, noul, rid=None):
    out = {"i": i, "ms": 1.0, "noul": noul, "model": "fixture"}
    if rid is not None:
        out["record_id"] = rid
    return out

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            if isinstance(r, str):
                f.write(r + "\n")
            else:
                f.write(json.dumps(r) + "\n")

def run_case(root, ev_files, inc_files, res_files, argv_extra=None):
    ev = root / "eval"
    for name, lines in ev_files.items():
        write_jsonl(ev / name, lines)

    inc = root / "incumbent"
    for name, lines in inc_files.items():
        write_jsonl(inc / name, lines)

    res_dir = root / "res"
    for name, lines in res_files.items():
        write_jsonl(res_dir / name, lines)

    out = root / "out.json"
    argv = [
        "--results-dir", str(res_dir),
        "--incumbent-dir", str(inc),
        "--frozen-main", str(ev / "frozen_main.jsonl"),
        "--frozen-supp", str(ev / "frozen_supp.jsonl"),
        "--frozen-candidates", str(ev / "candidates.jsonl"),
        "--synth-v4-eval", str(ev / "synth_v4.jsonl"),
        "--synth-v5-eval", str(ev / "synth_v5.jsonl"),
        "--mined-ext-eval", str(ev / "mined_ext.jsonl"),
        "--seen-eval", str(ev / "seen.jsonl"),
        "--transfer-eval", str(ev / "transfer.jsonl"),
        "--out", str(out)
    ] + (argv_extra or [])

    code = MOD.main(argv)

    if out.exists():
        return code, json.loads(out.read_text())
    return code, None

def test_missing_and_empty():
    root = Path(tempfile.mkdtemp(prefix="v5edge_"))
    try:
        code, out = run_case(root, {}, {}, {})
        check("empty directories yield NOT_READY overall", code == 2)
        if out:
            check("overall status is not_ready", out["overall"] == "not_ready")

        ev_files = {f: [] for f in [
            "frozen_main.jsonl", "frozen_supp.jsonl", "candidates.jsonl",
            "synth_v4.jsonl", "synth_v5.jsonl", "seen.jsonl", "transfer.jsonl", "mined_ext.jsonl"
        ]}
        res_files = {f: [] for f in [
            "lora_v5_frozen_main.jsonl", "lora_v5_frozen_supp.jsonl",
            "winnow_frozen_main.jsonl", "winnow_frozen_supp.jsonl",
            "lora_v5_synth_v4.jsonl", "lora_v5_synth_v5.jsonl",
            "lora_v5_seen.jsonl", "lora_v5_transfer.jsonl", "lora_v5_mined_ext.jsonl"
        ]}
        inc_files = {f: [] for f in [
            "lora_real_candidates_v1.jsonl", "lora_drop_supp_v1.jsonl",
            "winnow_real_candidates_full_v1.jsonl", "winnow_drop_supp_v1.jsonl"
        ]}

        code, out = run_case(root, ev_files, inc_files, res_files)
        check("empty files don't crash, returns a code", code in (0, 1, 2))
    finally:
        shutil.rmtree(root)

def test_malformed_lines():
    root = Path(tempfile.mkdtemp(prefix="v5edge_"))
    try:
        ev_files = {
            "frozen_main.jsonl": [rec("m1", False), "invalid json\n", rec("m2", True)]}
        res_files = {
            "lora_v5_frozen_main.jsonl": [res(0, 0.1, "m1"), "also invalid json\n", res(2, 0.9, "m2")]}
        inc_files = {}
        code, out = run_case(root, ev_files, inc_files, res_files)

        if out:
            check("malformed json causes error status in gates", any(g.get("status") == "error" for g in out["gates"].values()))
            check("overall is fail for malformed inputs", out["overall"] == "fail")
    finally:
        shutil.rmtree(root)

def test_missing_metrics():
    root = Path(tempfile.mkdtemp(prefix="v5edge_"))
    try:
        ev_files = {
            "frozen_main.jsonl": [
            rec("m1", False),
            {"meta": {"record_id": "m2"}, "targets": {"irrelevant": {}}},
            rec("m3", True)
        ],
        "frozen_supp.jsonl": [],
        "candidates.jsonl": [],
        "synth_v4.jsonl": [],
        "synth_v5.jsonl": [],
        "seen.jsonl": [],
        "transfer.jsonl": [],
        "mined_ext.jsonl": []
        }
        res_files = {
            "lora_v5_frozen_main.jsonl": [
            res(0, 0.1, "m1"),
            {"i": 1, "record_id": "m2", "noul": 0.5},
            {"i": 2, "record_id": "m3", "noul": None}
        ],
        "lora_v5_frozen_supp.jsonl": [],
        "winnow_frozen_main.jsonl": [],
        "winnow_frozen_supp.jsonl": [],
        "lora_v5_synth_v4.jsonl": [],
        "lora_v5_synth_v5.jsonl": [],
        "lora_v5_seen.jsonl": [],
        "lora_v5_transfer.jsonl": [],
        "lora_v5_mined_ext.jsonl": []
        }
        inc_files = {}
        code, out = run_case(root, ev_files, inc_files, res_files)

        if out:
            g1 = out["gates"].get("G1", {})
            check("missing metrics does not crash", g1.get("status") in ("pass", "fail", "not_ready"))
            if g1.get("measured"):
                check("missing metrics lines are excluded from n", g1["measured"]["n"] == 1)
    finally:
        shutil.rmtree(root)

def test_exact_boundaries():
    root = Path(tempfile.mkdtemp(prefix="v5edge_"))
    try:
        ev_files = {
            "frozen_main.jsonl": [
                rec("m1", False, target_val=0.499),
                rec("m2", True, target_val=0.500),
                rec("m3", False, target_val=0.0)
            ],
            "frozen_supp.jsonl": [],
            "candidates.jsonl": [],
            "synth_v4.jsonl": [],
            "synth_v5.jsonl": [],
            "seen.jsonl": [],
            "transfer.jsonl": [],
            "mined_ext.jsonl": []
        }
        res_files = {
            "lora_v5_frozen_main.jsonl": [
                res(0, 0.499, "m1"),
                res(1, 0.500, "m2"),
                res(2, 0.900, "m3")
            ],
            "lora_v5_frozen_supp.jsonl": [],
            "winnow_frozen_main.jsonl": [],
            "winnow_frozen_supp.jsonl": [],
            "lora_v5_synth_v4.jsonl": [],
            "lora_v5_synth_v5.jsonl": [],
            "lora_v5_seen.jsonl": [],
            "lora_v5_transfer.jsonl": [],
            "lora_v5_mined_ext.jsonl": []
        }
        inc_files = {}
        code, out = run_case(root, ev_files, inc_files, res_files)

        if out:
            g1 = out["gates"].get("G1", {})
            if g1.get("measured"):
                meas = g1["measured"]
                check("target=0.5 and noul=0.5 are counted as drops (TP)", meas["tp"] == 1)
                check("noul=0.9 is counted as confident fp", meas["confident_fp_noul_ge_0_9"] == 1)
    finally:
        shutil.rmtree(root)

def main():
    print("== Missing and Empty")
    test_missing_and_empty()
    print("== Malformed Lines")
    test_malformed_lines()
    print("== Missing Metrics / NaN")
    test_missing_metrics()
    print("== Exact Boundaries")
    test_exact_boundaries()

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)

if __name__ == "__main__":
    main()
