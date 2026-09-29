#!/usr/bin/env python3
"""X2 Arm B candidate-order check on engineering_heldout_v1 choice items.

Permutes the criteria dict of every heldout choice question (labels are assigned
in presentation order, so a permutation remaps each option's letter) and measures
argmax flips / max |dp| vs identity for the native-logit readout. This is the
T9a-style position-artifact test applied to the SemIf readout: a content-based
readout should return the same candidate id under permutation.

Evaluation only; no training. Usage:
  .venv/bin/python scripts/x2_permutation_check.py --source base \
      --output results/x2_semif_transfer/permutation_b_base.json
"""
import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import x2_semif_readout as x2  # noqa: E402

HELDOUT_ITEMS = ROOT / "research/engineering_heldout_v1/items.jsonl"
PERMS = ("identity", "reversed", "rot1", "shuffle_s11")


def permute_criteria(criteria, perm, seed_key):
    keys = list(criteria)
    if perm == "identity":
        order = keys
    elif perm == "reversed":
        order = keys[::-1]
    elif perm == "rot1":
        order = keys[1:] + keys[:1]
    elif perm.startswith("shuffle_"):
        order = list(keys)
        random.Random(f"{perm}:{seed_key}").shuffle(order)
    else:
        raise ValueError(f"unknown permutation: {perm}")
    return {k: criteria[k] for k in order}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=["base", "checkpoint"], required=True)
    p.add_argument("--checkpoint-dir")
    p.add_argument("--marker", default="Decision:")
    p.add_argument("--device", default="mps")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1")
    import torch
    rows = [json.loads(l) for l in HELDOUT_ITEMS.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    rows = [r for r in rows
            if any(q["type"] == "choice" for q in r["questions"].values())]
    gold = x2.load_gold_map(HELDOUT_ITEMS)
    model, tokenizer, meta = x2.load_native_model(args.source, args.checkpoint_dir,
                                                torch.device(args.device))
    answers = {}
    for perm in PERMS:
        permuted = []
        for r in rows:
            qs = {qid: ({**q, "criteria": permute_criteria(q["criteria"], perm, r["id"])}
                        if q["type"] == "choice" else q)
                  for qid, q in r["questions"].items()}
            permuted.append({**r, "questions": qs})
        res, _ = x2.run_readout(model, tokenizer, torch.device(args.device), permuted,
                                8, 2048, marker=args.marker)
        answers[perm] = {(r["id"], r["qid"]): r for r in res}

    identity = answers["identity"]
    items = sorted(identity)
    per_perm, flips = {}, {}
    for perm, ans in answers.items():
        n_correct = 0
        max_dp = 0.0
        changed = []
        for key in items:
            r = ans[key]
            g = x2.gold_key("choice", gold[r["id"]][r["qid"]])
            n_correct += int(r["selected_id"] == g)
            pi, pp = identity[key]["probabilities"], r["probabilities"]
            if set(pi) != set(pp):
                raise ValueError("candidate key set changed under permutation")
            max_dp = max(max_dp, max(abs(pi[c] - pp[c]) for c in pi))
            if r["selected_id"] != identity[key]["selected_id"]:
                changed.append(r["id"])
        per_perm[perm] = {"accuracy": n_correct / len(items), "max_abs_dp": max_dp}
        if perm != "identity":
            flips[perm] = {"n_flips": len(changed), "items": changed}
    stable = sum(int(len({answers[pm][k]["selected_id"] for pm in PERMS}) == 1)
                 for k in items)
    receipt = {
        "schema_version": "nanojev-x2-permutation-check-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": meta, "marker": args.marker,
        "data": {"path": str(HELDOUT_ITEMS.relative_to(ROOT)),
                 "sha256": hashlib.sha256(HELDOUT_ITEMS.read_bytes()).hexdigest()},
        "n_choice_items": len(items), "permutations": list(PERMS),
        "per_permutation": per_perm, "argmax_flips_vs_identity": flips,
        "items_stable_all_permutations": stable,
        "note": "labels are assigned in criteria presentation order, so permutation "
                "remaps each option's letter; a content-based readout keeps the same "
                "candidate id (argmax flips = position/label artifact)",
        "network_model_calls": 0,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(json.dumps({"output": str(out), "stable": stable, "n": len(items),
                      "identity_acc": per_perm["identity"]["accuracy"]}))


if __name__ == "__main__":
    main()
