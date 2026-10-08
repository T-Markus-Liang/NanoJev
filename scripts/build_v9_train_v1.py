#!/usr/bin/env python3
"""build_v9_train_v1.py — assemble the v9 corpus.

v9 = conservative base
   + rule-audit verified drops (the same 151 as v7/v8, step-1 filter)
   + v9 supersession-injection drop arms
   + v9 keep arms whose record_id is NOT already in conservative
     (the mined f2 keeps are all already there → expected all deduped)

Usage: python3 scripts/build_v9_train_v1.py
"""
import json, os, shutil

CONS = "data/valen_nano_v5_ablation/conservative/train.jsonl"
INJ = "data/v9_injection/train.jsonl"
AUDIT = "data/v5_mining/label_audit.json"
CAND = ["data/v5_mining/f2_hard_negative_candidates.jsonl",
        "data/v5_mining/f4_outcome_positive_candidates.jsonl"]
OUT = "data/valen_nano_v9"


def drop_targets():
    return {"irrelevant": {"probabilities": {"true": 1.0, "false": 0.0}}}


def main():
    audit = {}
    for fam in ("f2", "f4"):
        for x in json.load(open(AUDIT)).get(fam, []):
            audit[x["id"]] = x

    cons = [json.loads(l) for l in open(CONS)]
    cons_ids = {r["meta"].get("record_id") for r in cons}
    rows = list(cons)
    counts = {"base": len(rows), "rule_verified": 0, "inj_drop": 0,
              "inj_keep_dedup": 0, "inj_keep_new": 0}

    cand_by_id = {}
    for path in CAND:
        for l in open(path):
            r = json.loads(l)
            cand_by_id[r["meta"].get("record_id")] = r

    # 1) rule-audit verified drops (same filter as v8 step 1)
    for rid, r in cand_by_id.items():
        if (r["meta"].get("proposed_label") == "yes"
                and audit.get(rid, {}).get("verdict") == "likely_correct"):
            rr = dict(r); rr["meta"] = dict(r["meta"])
            rr["targets"] = drop_targets()
            rr["meta"]["v9_source"] = "rule_verified"
            rows.append(rr); counts["rule_verified"] += 1

    # 2) v9 supersession-injection arms
    for l in open(INJ):
        r = json.loads(l)
        rid = r["meta"].get("record_id")
        if r["meta"].get("v9_arm") == "supersession_drop":
            rows.append(r); counts["inj_drop"] += 1
        else:
            if rid in cons_ids:
                counts["inj_keep_dedup"] += 1
            else:
                rows.append(r); counts["inj_keep_new"] += 1

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/train.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for s in ("eval.jsonl", "dev.jsonl"):
        shutil.copy(f"data/valen_nano_v5/{s}", f"{OUT}/{s}")
    print(f"v9 train: {len(rows)} rows  {counts}")


if __name__ == "__main__":
    main()
