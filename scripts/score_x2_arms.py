#!/usr/bin/env python3
"""X2 SemIf architecture-transfer ablation: assemble the three-arm comparison.

Arms (identical data: corpus v3 dev324 + engineering_heldout_v1 69 items):
  A  LoRA r16 seed18 + scalar/attention head  (existing results only; no retrain)
  B  SemIf-style direct native-logit readout  (scripts/x2_semif_readout.py metrics)
  C  LoRA r16 + minimal pointer head          (checkpoints/x2_pointer_seed*)

File reads only — no model calls. Emits results/x2_semif_transfer/receipt.json.
Dev accuracy for learned-head arms comes from each checkpoint's
predictions_dev.jsonl (student_probs vs gold_index); heldout from the frozen
predict receipts results/heldout_v1_post_*.json scored against items.jsonl gold.
"""
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

HELDOUT_ITEMS = ROOT / "research/engineering_heldout_v1/items.jsonl"
DEV_JSONL = ROOT / "research/engineering_judgment_corpus_v3/trainer_view/dev.jsonl"
OUT = ROOT / "results/x2_semif_transfer"

ABSTAIN = 0.90


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def gold_key(qtype, gold):
    if qtype == "boolean":
        return "true" if gold else "false"
    return str(gold)


def summarize(records):
    """records = [{'type','probs': {cid: p}, 'gold_id': cid}] -> metrics."""
    n = n_correct = answered = confident_wrong = 0
    confs = []
    by_type = {}
    for r in records:
        probs = r["probs"]
        pred = max(probs, key=probs.__getitem__)
        conf = probs[pred]
        correct = int(pred == r["gold_id"])
        n += 1
        n_correct += correct
        confs.append(conf)
        if conf >= ABSTAIN:
            answered += 1
            confident_wrong += (1 - correct)
        b = by_type.setdefault(r["type"], {"n": 0, "correct": 0, "answered": 0})
        b["n"] += 1
        b["correct"] += correct
        b["answered"] += int(conf >= ABSTAIN)
    return {"questions": n, "accuracy": n_correct / n if n else None,
            "cov_at_0.9": answered / n if n else None,
            "answered_count": answered, "confident_wrong_count": confident_wrong,
            "confidence_mean": statistics.fmean(confs) if confs else None,
            "by_type": {t: {"n": b["n"], "accuracy": b["correct"] / b["n"],
                            "answered_at_0.90": b["answered"],
                            "cov_at_0.9": b["answered"] / b["n"]}
                        for t, b in sorted(by_type.items())}}


def records_from_predictions_jsonl(path):
    """Checkpoint predictions_*.jsonl -> records (student_probs + gold_index)."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        k = len(r["candidate_ids"])
        probs = dict(zip(r["candidate_ids"], r["student_probs"][:k]))
        gold_id = r["candidate_ids"][r["gold_index"]] if r["gold_index"] is not None else None
        out.append({"type": r["type"], "probs": probs, "gold_id": gold_id})
    return out


def records_from_heldout_receipt(path, gold_by_id):
    """predict_toy_decisions output -> records scored against items.jsonl gold."""
    receipt = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for state in receipt["states"]:
        for qid, ans in state["answers"].items():
            out.append({"type": ans["type"], "probs": ans["probabilities"],
                        "gold_id": gold_key(ans["type"], gold_by_id[state["id"]][qid])})
    return out


def records_from_x2_rows(path, gold_by_id):
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out.append({"type": r["type"], "probs": r["probabilities"],
                    "gold_id": gold_key(r["type"], gold_by_id[r["id"]][r["qid"]])})
    return out


def heldout_gold():
    gold = {}
    for line in HELDOUT_ITEMS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            gold[row["id"]] = row["gold"]
    return gold


def dev_gold():
    gold = {}
    for line in DEV_JSONL.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            gold[row["id"]] = row["gold"]
    return gold


def main():
    gold_h, gold_d = heldout_gold(), dev_gold()
    arms = {}

    # ---- Arm A: LoRA r16 seed18 + scalar/attention head (frozen receipts) ----
    a_dir = ROOT / "checkpoints/domain_adaptation_v4_lora_seed18"
    arms["A_lora_r16_seed18_head"] = {
        "kind": "learned scalar + set-attention head on LoRA-merged backbone",
        "checkpoint": "checkpoints/domain_adaptation_v4_lora_seed18",
        "checkpoint_sha256": sha256(a_dir / "best.safetensors"),
        "dev": summarize(records_from_predictions_jsonl(a_dir / "predictions_dev.jsonl")),
        "heldout": summarize(records_from_heldout_receipt(
            ROOT / "results/heldout_v1_post_v4_lora_seed18.json", gold_h)),
        "sources": {"dev": str(a_dir / "predictions_dev.jsonl"),
                    "heldout": "results/heldout_v1_post_v4_lora_seed18.json",
                    "prior_summary": "results/t9d_v4_lora_summary.json"},
    }

    # ---- Arm B: native-logit readout metrics (already computed per run) ----
    for tag, label in (("b_base", "base Qwen3-0.6B, marker 'Decision:'"),
                       ("b_atomic17", "atomic seed17 backbone, marker 'Decision:'"),
                       ("b_lora18", "LoRA r16 seed18 backbone, marker 'Decision:'"),
                       ("b_base_answer", "base Qwen3-0.6B, marker 'Answer:' (diagnostic)"),
                       ("b_lora18_answer", "LoRA r16 seed18 backbone, marker 'Answer:' (diagnostic)")):
        arm = {"kind": "SemIf-style direct native-logit readout, no learned parameters",
               "readout": label}
        for split in ("dev", "heldout"):
            mpath = OUT / f"metrics_{tag}_{split}.json"
            if mpath.is_file():
                m = json.loads(mpath.read_text(encoding="utf-8"))
                arm[split] = {k: m[k] for k in
                              ("questions", "accuracy", "answered_at_0.90_rate",
                               "answered_count", "confident_wrong_count",
                               "confidence_mean", "label_vocabulary_mass_mean",
                               "by_type")}
                arm[split]["cov_at_0.9"] = m["answered_at_0.90_rate"]
                arm.setdefault("sources", {})[split] = str(mpath.relative_to(ROOT))
        arms[f"B_{tag}"] = arm

    # ---- Arm C: LoRA r16 + pointer head (whatever seeds finished) ----
    for seed_dir in sorted(ROOT.glob("checkpoints/x2_pointer_seed*")):
        seed = seed_dir.name.rsplit("seed", 1)[-1]
        name = f"C_lora_r16_pointer_seed{seed}"
        arm = {"kind": "LoRA r16 on pinned base + minimal pointer head "
                       "(anchor query x leaf keys, learned null key for boolean)",
               "checkpoint": str(seed_dir.relative_to(ROOT)),
               "checkpoint_sha256": sha256(seed_dir / "best.safetensors")
               if (seed_dir / "best.safetensors").is_file() else None}
        pdev = seed_dir / "predictions_dev.jsonl"
        if pdev.is_file():
            arm["dev"] = summarize(records_from_predictions_jsonl(pdev))
            summ = json.loads((seed_dir / "summary.json").read_text())
            arm["best_step"] = summ.get("best_step")
            arm["best_dev_target_ce"] = summ.get("best_dev_target_ce")
        heldout_receipt = ROOT / f"results/heldout_v1_post_x2_pointer_seed{seed}.json"
        if heldout_receipt.is_file():
            arm["heldout"] = summarize(records_from_heldout_receipt(heldout_receipt, gold_h))
            arm.setdefault("sources", {})["heldout"] = str(heldout_receipt.relative_to(ROOT))
        arms[name] = arm

    receipt = {
        "schema_version": "nanojev-x2-semif-transfer-v1",
        "task": "X2 SemIf-style architecture transfer ablation on Qwen3-0.6B",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evaluation_only": True, "network_model_calls": 0,
        "reference": {"repo": "github.com/TheoLeeCJ/SemIf", "license": "MIT",
                      "pinned": "ca3ba65f142967030ecb453346e94d6f476a69df"},
        "data": {"dev": {"path": str(DEV_JSONL.relative_to(ROOT)),
                          "sha256": sha256(DEV_JSONL), "questions": 324},
                 "heldout": {"path": str(HELDOUT_ITEMS.relative_to(ROOT)),
                             "sha256": sha256(HELDOUT_ITEMS), "questions": 69}},
        "endpoint_identities": {"confidence": "max(probabilities)",
                                 "answered": "confidence >= 0.90",
                                 "correct": "argmax(probabilities) == gold",
                                 "temperature": 1.0, "threshold_fitted": False},
        "arms": arms,
        "permutation_artifact": {
            "files": {"b_base": "results/x2_semif_transfer/permutation_b_base.json",
                      "b_base_answer": "results/x2_semif_transfer/permutation_b_base_answer.json",
                      "b_lora18": "results/x2_semif_transfer/permutation_b_lora18.json"},
            "summary": "Arm B label readout is strongly position-bound (T9a artifact "
                       "confirmed on heldout choice): 2/23 items stable for base "
                       "'Decision:', 7/23 for 'Answer:', 3/23 for the LoRA backbone; "
                       "canonical-order accuracy is inflated. Arms A/C are "
                       "permutation-invariant by construction.",
        },
        "conclusions": [
            "Arm B base-model native readout is competitive on canonical order "
            "(heldout 0.536 'Decision:' / 0.623 'Answer:' vs Arm A 0.580) but has "
            "near-zero label vocabulary mass (~0.001-0.005) and fails candidate-order "
            "stability — a diagnostic bound, not a deployable readout.",
            "Fine-tuning damages the native readout (J-D5 replicated): atomic seed17 "
            "0.261, LoRA seed18 0.275 heldout vs base 0.536. LoRA on the already-"
            "damaged atomic backbone does not repair it.",
            "Arm C (LoRA + minimal pointer head) is an honest negative: heldout "
            "0.507/0.362/0.377 (mean 0.415, worst 0.362 < baseline 0.464); score type "
            "collapses on 2/3 seeds; dev CE bottoms at step 62-112 then worsens. A "
            "pointer over isolated leaf vectors does not give the backbone "
            "cross-option comparison — the packed block-causal serialization is the "
            "load-bearing part of T9b Option B, not the pointer shape alone.",
            "Readout is not the binding constraint: zero-parameter base readout "
            "matches the trained head on heldout. Implies X4 (independently authored "
            "corpus growth + larger heldout) over more head redesign; any future "
            "native-logit arm must fix permutation stability first.",
            "cov@0.9 ~0 for all trained arms; abstention gate unchanged.",
        ],
        "verdict": "no promotion; Arm A (LoRA r16 + scalar/attention head) remains "
                   "the strongest reliable readout on heldout (0.580 best, 0.536 mean)",
    }
    path = OUT / "receipt.json"
    path.write_text(json.dumps(receipt, indent=1, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")
    # compact comparison table on stdout
    print(f"{'arm':<38}{'dev':>7}{'heldout':>9}{'cov@0.9':>9}  by-type heldout (b/c/s)")
    for name, arm in arms.items():
        dev = (arm.get("dev") or {}).get("accuracy")
        ho = arm.get("heldout") or {}
        bt = ho.get("by_type") or {}
        line = f"{name:<38}{(dev if dev is not None else float('nan')):>7.3f}" \
               f"{(ho.get('accuracy') if ho else float('nan')):>9.3f}" \
               f"{(ho.get('cov_at_0.9') if ho.get('cov_at_0.9') is not None else float('nan')):>9.3f}  "
        line += "/".join(f"{bt.get(t, {}).get('accuracy', float('nan')):.3f}"
                          for t in ("boolean", "choice", "score"))
        print(line)
    print(json.dumps({"receipt": str(path)}))


if __name__ == "__main__":
    main()
