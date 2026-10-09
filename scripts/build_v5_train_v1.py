#!/usr/bin/env python3
"""Assemble the v5 training corpus (docs/V5_DATA_DESIGN_V1.md).

v5 train = frozen v4 train + new families:
  F1 anti_shortcut         (synthetic, labeled, 1498)
  F3 boilerplate_breaker   (synthetic, labeled, 1500)
  F2 exit_hard_negative    (mined proposals -> labels, anti-shortcut veto on yes)
  F4 outcome_positive      (mined proposals -> labels, same veto)

Rules:
  - mined uncertain proposals are excluded from training AND eval
  - yes proposals whose candidate text looks like a negative-result /
    correction / error signal are vetoed (excluded) — F1 established that
    these surfaces must be KEEP; letting the miner label them drop would
    re-teach the exact shortcut we are fixing
  - F1/F3 split by group_id (pair members never cross splits):
    ~84% train / ~12% eval (v5 synth eval leg for G4) / ~4% dev
  - mined records are train-only (eval side already has its own pool)
  - dedup vs every prior split + intra-set, on request sha256
Outputs: data/valen_nano_v5/{train,eval,dev}.jsonl + manifest.json
"""
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261201

NEG_RESULT_RE = re.compile(
    r"(does not exist|not found|no matches|total 0|exit code [1-9]"
    r"|failed|error|cannot|can't|denied|permission|timed out"
    r"|important catch|actually,|更正|其实|不对)",
    re.IGNORECASE,
)


def load_jsonl(path):
    return [json.loads(l) for l in open(path)]


def req_hash(rec):
    blob = json.dumps(rec["request"], sort_keys=True,
                      separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def labeled(rec, keep_or_drop):
    """Attach a valen-style target: true==certainly irrelevant==drop."""
    rec["targets"] = {"irrelevant": {"probabilities": {
        "true": 1.0 if keep_or_drop == "drop" else 0.0,
        "false": 0.0 if keep_or_drop == "drop" else 1.0}}}
    return rec


def candidate_text(rec):
    st = rec["request"]["state"]
    st = json.loads(st) if isinstance(st, str) else st
    ptr = rec["meta"]["candidate_pointer"]
    for seg in st.get("conversation", []):
        if seg.get("pointer") == ptr:
            return str(seg.get("text") or seg.get("content") or "")
    return ""


def mined_label(rec):
    """proposed_label -> 'drop'|'keep'|None(excluded). Veto on negative-result yes."""
    pl = rec["meta"].get("proposed_label")
    if pl == "uncertain":
        return None
    if pl == "yes":
        if NEG_RESULT_RE.search(candidate_text(rec)):
            return None  # anti-shortcut veto
        return "drop"
    if pl == "no":
        return "keep"
    return None


def main():
    out = ROOT / "data" / "valen_nano_v5"
    out.mkdir(parents=True, exist_ok=True)
    stats = Counter()

    # prior-corpus reference hashes (dedup across v3/v4 + all eval surfaces)
    seen = set()
    prior_files = [
        "data/valen_nano_v4/eval.jsonl",
        "data/valen_nano_v4/dev.jsonl",
        "data/valen_nano_v3/eval.jsonl",
        "data/real_context_eval_v1/candidates.jsonl",
        "data/real_context_eval_v1/drop_supp/candidates_drop_supp_v1.jsonl",
        "data/real_context_eval_v5_ext/candidates.jsonl",
    ]
    for rel in prior_files:
        p = ROOT / rel
        if p.exists():
            for r in load_jsonl(p):
                seen.add(req_hash(r))
    stats["prior_reference_hashes"] = len(seen)

    train, evl, dev = [], [], []

    # --- frozen v4 train ----------------------------------------------------
    for r in load_jsonl(ROOT / "data/valen_nano_v4/train.jsonl"):
        h = req_hash(r)
        if h in seen:
            stats["v4_train_dup"] += 1
            continue
        seen.add(h)
        train.append(r)
    stats["v4_train_kept"] = len(train)

    # --- F1 + F3 synthetic: group split ------------------------------------
    synth = []
    for rel in ["data/v5_corpus/f1_antishortcut.jsonl",
                "data/v5_corpus/f3_boilerplate.jsonl"]:
        for r in load_jsonl(ROOT / rel):
            h = req_hash(r)
            if h in seen:
                stats["synth_dup"] += 1
                continue
            seen.add(h)
            synth.append(r)
    stats["synth_total"] = len(synth)

    groups = sorted({r["group_id"] for r in synth})
    # deterministic group -> split assignment; pairs share a group, so they
    # can never cross splits
    def gsplit(g):
        v = int(hashlib.sha256(f"{SEED}:{g}".encode()).hexdigest()[:8], 16)
        m = v % 100
        return "eval" if m < 12 else ("dev" if m < 16 else "train")

    pair_split = {}
    for r in synth:
        pid = r["meta"].get("pair_id")
        split = gsplit(r["group_id"])
        if pid:
            if pid in pair_split and pair_split[pid] != split:
                split = pair_split[pid]  # force pair members together
            pair_split[pid] = split
        (evl if split == "eval" else dev if split == "dev"
         else train).append(r)

    # --- F2 + F4 mined proposals -------------------------------------------
    mined_stats = Counter()
    for rel in ["data/v5_mining/f2_hard_negative_candidates.jsonl",
                "data/v5_mining/f4_outcome_positive_candidates.jsonl"]:
        for r in load_jsonl(ROOT / rel):
            lab = mined_label(r)
            if lab is None:
                mined_stats["excluded_uncertain_or_veto"] += 1
                continue
            h = req_hash(r)
            if h in seen:
                mined_stats["dup"] += 1
                continue
            seen.add(h)
            r = labeled(r, lab)
            r["meta"]["v5_label_source"] = "miner_proposal_vetted"
            train.append(r)
            mined_stats[lab] += 1
    stats.update({f"mined_{k}": v for k, v in mined_stats.items()})

    for name, rows in (("train", train), ("eval", evl), ("dev", dev)):
        with open(out / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "schema_version": "nanojev-valen-nano-v5",
        "builder": "scripts/build_v5_train_v1.py",
        "seed": SEED,
        "lineage": ("v4 train frozen reuse + F1/F3 synthetic + "
                    "F2/F4 mined proposals (anti-shortcut veto)"),
        "counts": {n: len(x) for n, x in
                   (("train", train), ("eval", evl), ("dev", dev))},
        "stats": dict(stats),
        "question": {"qid": "irrelevant", "type": "noul",
                     "semantics": "true==certainly irrelevant==drop"},
        "training_allowed": {"train": True, "eval": False, "dev": False},
        "evaluation_only": ["eval", "dev",
                            "data/real_context_eval_v1 (frozen)",
                            "data/real_context_eval_v5_ext"],
        "holdout_note": ("real-context evals never trained; T4 transfer "
                         "holdout = withheld source family per spec §7.2"),
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)

    print(json.dumps({"counts": manifest["counts"],
                      "stats": dict(stats)}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
