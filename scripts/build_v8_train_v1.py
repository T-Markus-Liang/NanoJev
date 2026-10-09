#!/usr/bin/env python3
"""build_v8_train_v1.py — assemble the v8 corpus.

v8 = conservative base
   + rule-audit verified drops (the 151 already proven in v7)
   + foreign-injection drop arms (keep arms skipped when the same record_id
     already exists in the conservative mining keeps — they'd be exact dupes)
   + clef-adjudicated drops (verdict=drop rows from clef_verdicts.jsonl)

Usage: python3 scripts/build_v8_train_v1.py
"""
import json, os, shutil

CONS = "data/valen_nano_v5_ablation/conservative/train.jsonl"
INJ = "data/v8_injection/train.jsonl"
VERDICTS = "data/v8_adjudication/clef_verdicts.jsonl"
AUDIT = "data/v5_mining/label_audit.json"
CAND = ["data/v5_mining/f2_hard_negative_candidates.jsonl",
        "data/v5_mining/f4_outcome_positive_candidates.jsonl"]
OUT = "data/valen_nano_v8"


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
              "inj_keep_dedup": 0, "inj_keep_new": 0, "clef_verified": 0}

    verdicts = {}
    if os.path.exists(VERDICTS):
        verdicts = {json.loads(l)["record_id"]: json.loads(l)["verdict"]
                    for l in open(VERDICTS)}

    cand_by_id = {}
    for path in CAND:
        for l in open(path):
            r = json.loads(l)
            cand_by_id[r["meta"].get("record_id")] = r

    # 1) rule-audit verified drops (same as v7's 151)
    for rid, r in cand_by_id.items():
        if (r["meta"].get("proposed_label") == "yes"
                and audit.get(rid, {}).get("verdict") == "likely_correct"):
            rr = dict(r); rr["meta"] = dict(r["meta"])
            rr["targets"] = drop_targets()
            rr["meta"]["v8_source"] = "rule_verified"
            rows.append(rr); counts["rule_verified"] += 1

    # 2) injection arms
    for l in open(INJ):
        r = json.loads(l)
        rid = r["meta"].get("record_id")
        if r["meta"].get("v8_arm") == "foreign_injection":
            rows.append(r); counts["inj_drop"] += 1
        else:
            if rid in cons_ids:
                counts["inj_keep_dedup"] += 1
            else:
                rows.append(r); counts["inj_keep_new"] += 1

    # 3) clef-verified drops not already added
    added_ids = {r["meta"].get("record_id") for r in rows}
    for rid, v in verdicts.items():
        if v == "drop" and rid in cand_by_id and rid not in added_ids:
            rr = dict(cand_by_id[rid]); rr["meta"] = dict(rr["meta"])
            rr["targets"] = drop_targets()
            rr["meta"]["v8_source"] = "clef_verified"
            rows.append(rr); counts["clef_verified"] += 1

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/train.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for s in ("eval.jsonl", "dev.jsonl"):
        shutil.copy(f"data/valen_nano_v5/{s}", f"{OUT}/{s}")
    print(f"v8 train: {len(rows)} rows  {counts}")


if __name__ == "__main__":
    main()
