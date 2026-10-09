#!/usr/bin/env python3
"""build_v8_injection_v1.py — mechanically-correct drop labels via foreign
segment injection.

For each mined KEEP record (proposed_label=no, the protective hard-keeps):
  keep arm : the original record, label keep (noul false=1)
  drop arm : identical state except the candidate segment content is swapped
             for a same-role segment from a DIFFERENT transcript (group_id),
             label drop (noul true=1)

The injected content is certainly irrelevant by construction — it belongs to
an unrelated task. Contrastive pairs teach "belongs to this task" vs
"foreign content", which is the boundary v5's mined drops corrupted.

Usage: python3 scripts/build_v8_injection_v1.py --out data/v8_injection/train.jsonl
"""
import argparse, json, random, re
from collections import defaultdict


def parse_state(r):
    st = json.loads(r["request"]["state"])
    return st


def get_candidate(st):
    ptr = st.get("candidate_pointer")
    for seg in st.get("conversation", []):
        if seg.get("pointer") == ptr:
            return seg
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="data/v5_mining/f2_hard_negative_candidates.jsonl")
    ap.add_argument("--out", default="data/v8_injection/train.jsonl")
    ap.add_argument("--seed", type=int, default=20261006)
    ap.add_argument("--max-pairs", type=int, default=1500)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    keeps = []
    for l in open(args.candidates):
        r = json.loads(l)
        if r["meta"].get("proposed_label") != "no":
            continue
        st = parse_state(r)
        seg = get_candidate(st)
        if seg and seg.get("content"):
            keeps.append((r, st, seg))

    # foreign donor pool: candidate segments grouped by role, from different transcripts
    donors = defaultdict(list)  # role -> [(group_id, content)]
    for l in open(args.candidates):
        r = json.loads(l)
        st = parse_state(r)
        seg = get_candidate(st)
        if seg and seg.get("content") and len(seg["content"]) < 32000:
            donors[seg.get("role", "?")].append((r["group_id"], seg["content"]))

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n = 0
    with open(args.out, "w") as f:
        for r, st, seg in keeps[: args.max_pairs]:
            role = seg.get("role", "?")
            pool = [c for c in donors.get(role, []) if c[0] != r["group_id"]]
            if not pool:
                continue
            donor_gid, foreign = rng.choice(pool)
            # keep arm
            rk = dict(r); rk["meta"] = dict(r["meta"])
            rk["targets"] = {"irrelevant": {"probabilities": {"true": 0.0, "false": 1.0}}}
            rk["meta"]["v8_arm"] = "keep"
            f.write(json.dumps(rk, ensure_ascii=False) + "\n")
            # drop arm: same state, candidate content replaced by foreign segment
            st2 = json.loads(json.dumps(st))
            seg2 = get_candidate(st2)
            seg2["content"] = foreign
            rd = dict(r); rd["meta"] = dict(r["meta"])
            rd["request"] = dict(r["request"])
            rd["request"]["state"] = json.dumps(st2, ensure_ascii=False)
            rd["targets"] = {"irrelevant": {"probabilities": {"true": 1.0, "false": 0.0}}}
            rd["meta"]["v8_arm"] = "foreign_injection"
            rd["meta"]["donor_group"] = donor_gid
            rd["meta"]["pair_with"] = rk["meta"].get("record_id")
            f.write(json.dumps(rd, ensure_ascii=False) + "\n")
            n += 1
    print(f"injection pairs: {n} (kept+dropped each) -> {args.out}")


if __name__ == "__main__":
    main()
