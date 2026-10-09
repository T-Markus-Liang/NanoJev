#!/usr/bin/env python3
"""Score T9d-v5 LoRA run: heldout_v1 (69q) + heldout_v2 (312q) per seed.

Runs predict_toy_decisions.py once per (seed, heldout) pair, writes prediction
receipts under results/, scores argmax accuracy / cov@0.9 / confident-wrong
against items.jsonl gold, and emits results/t9d_v5_lora_summary.json.

Amendment: research/nanojev_v2_t9d_v5_lora_amendment_v2.json
Baselines: atomic 0.464 heldout_v1; v3 LoRA mean 0.536 / best 0.580.
Evaluation only: read-once per checkpoint per heldout.
"""
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv/bin/python"
PREDICT = ROOT / "scripts/predict_toy_decisions.py"
SEEDS = (17, 18, 19)
ABSTAIN = 0.90

HELDOUTS = {
    "heldout_v1": {
        "items": ROOT / "research/engineering_heldout_v1/items.jsonl",
        "predict_input": ROOT / "research/engineering_heldout_v1/predict_input.json",
    },
    "heldout_v2": {
        "items": ROOT / "research/engineering_heldout_v2/items.jsonl",
        "predict_input": ROOT / "research/engineering_heldout_v2/predict_input.json",
    },
}

BASELINES = {"atomic_heldout_v1": 0.463768115942029,
             "v3_lora_heldout_v1_mean": 0.536, "v3_lora_heldout_v1_best": 0.580}


def sha256(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gold_key(qtype, gold):
    if qtype == "boolean":
        return "true" if gold else "false"
    return str(gold)


def load_gold(items_path):
    gold = {}
    for line in items_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        gold[row["id"]] = row["gold"]
    return gold


def run_predict(ckpt_dir, predict_input, out_path):
    if out_path.is_file():
        return  # read-once already consumed; never re-run a heldout receipt
    cmd = [str(PY), str(PREDICT), "--checkpoint-dir", str(ckpt_dir),
           "--input", str(predict_input), "--output", str(out_path),
           "--batch-questions", "8", "--device", "mps", "--precision", "fp32"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"predict failed for {ckpt_dir}: {proc.stderr[-2000:]}")


def score_receipt(receipt_path, gold_by_id):
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    n = n_correct = answered = confident_wrong = 0
    confs, by_type = [], {}
    for state in receipt["states"]:
        for qid, ans in state["answers"].items():
            probs = ans["probabilities"]
            gid = gold_key(ans["type"], gold_by_id[state["id"]][qid])
            conf = max(probs.values())
            pred = max(probs.items(), key=lambda kv: kv[1])[0]
            correct = int(pred == gid)
            n += 1
            n_correct += correct
            answered += int(conf >= ABSTAIN)
            confident_wrong += int(conf >= ABSTAIN and not correct)
            confs.append(conf)
            b = by_type.setdefault(ans["type"], {"n": 0, "correct": 0, "answered": 0})
            b["n"] += 1
            b["correct"] += correct
            b["answered"] += int(conf >= ABSTAIN)
    return {"questions": n, "accuracy": n_correct / n if n else None,
            "cov_at_0.9": answered / n if n else None,
            "confident_wrong_at_0.9": confident_wrong,
            "confidence_mean": statistics.fmean(confs) if confs else None,
            "by_type": {t: {"n": b["n"], "accuracy": b["correct"] / b["n"],
                            "cov_at_0.9": b["answered"] / b["n"]}
                        for t, b in sorted(by_type.items())}}


def best_step(ckpt_dir):
    """Read the trainer's terminal receipt without touching heldout data."""
    log = ROOT / "logs" / f"t9d_v5_lora_s{ckpt_dir.name.rsplit('seed', 1)[-1]}.log"
    if log.is_file():
        for line in reversed(log.read_text(encoding="utf-8").splitlines()):
            if line.startswith('{"done"'):
                return json.loads(line)["best_step"]
    cfg = json.loads((ckpt_dir / "config.json").read_text(encoding="utf-8"))
    selection = cfg.get("selection")
    return selection.get("best_step") if isinstance(selection, dict) else None


def main():
    seeds = {}
    for seed in SEEDS:
        ckpt = ROOT / f"checkpoints/t9d_v5_lora_seed{seed}"
        entry = {"checkpoint": str(ckpt.relative_to(ROOT)),
                 "exists": ckpt.is_dir()}
        if not ckpt.is_dir():
            seeds[f"seed{seed}"] = entry
            continue
        entry["best_step"] = best_step(ckpt)
        entry["checkpoint_sha256"] = (sha256(ckpt / "best.safetensors")
                                    if (ckpt / "best.safetensors").is_file() else None)
        entry["dev_selected"] = None
        dev_log = ckpt / "initial_dev_metrics.json"
        if dev_log.is_file():
            entry["initial_dev_target_ce"] = json.loads(
                dev_log.read_text(encoding="utf-8"))["target_ce"]
        for name, spec in HELDOUTS.items():
            receipt = ROOT / f"results/heldout_v1_post_t9dv5_{name}_seed{seed}.json" \
                if name == "heldout_v1" else \
                ROOT / f"results/heldout_v2_post_t9dv5_seed{seed}.json"
            run_predict(ckpt, spec["predict_input"], receipt)
            entry[name] = score_receipt(receipt, load_gold(spec["items"]))
            entry[name]["receipt"] = str(receipt.relative_to(ROOT))
        seeds[f"seed{seed}"] = entry

    def agg(metric):
        vals = [e["heldout_v1"][metric] for e in seeds.values()
                if e.get("heldout_v1") and e["heldout_v1"][metric] is not None]
        return {"mean": statistics.fmean(vals) if vals else None,
                "min": min(vals) if vals else None,
                "values": vals}

    summary = {
        "schema_version": "nanojev-t9d-v5-run-summary-v1",
        "amendment": "research/nanojev_v2_t9d_v5_lora_amendment_v2.json",
        "baselines": BASELINES,
        "seeds": seeds,
        "heldout_v1_accuracy": agg("accuracy"),
        "heldout_v2_accuracy": {
            "values": [e["heldout_v2"]["accuracy"] for e in seeds.values()
                       if e.get("heldout_v2")],
            "mean": statistics.fmean(
                [e["heldout_v2"]["accuracy"] for e in seeds.values()
                 if e.get("heldout_v2")]) if any(e.get("heldout_v2") for e in seeds.values()) else None},
        "note": "heldout_v2 (312q) is the forward-looking independent cohort; "
                "heldout_v1 (69q) is reported for continuity with v3/v4 arms",
    }
    out = ROOT / "results/t9d_v5_lora_summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(json.dumps({"wrote": str(out), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
