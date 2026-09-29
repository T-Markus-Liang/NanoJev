#!/usr/bin/env python3
"""Head-free LM-readout baseline for the context-filter task (JEMM-style).

Instead of the trained decision head, this arm serializes each
``eval.jsonl`` record into a chat prompt and reads the *last-position*
LM-head logits over single-token candidate labels (``A`` = proposition
true, ``B`` = proposition false), then softmaxes. No trained parameters
beyond the stock backbone — the control arm that JEMM and Visual Jev
suggest may match a trained head.

    external/valen/.venv/bin/python scripts/lm_head_readout_eval_v1.py \
        --data data/valen_nano_v3/eval.jsonl \
        --output results/lm_readout_v3_fp32 \
        --device mps --dtype fp32

Output: ``predictions.jsonl`` (per-question rows) + ``metrics.json``
(overall acc/brier/nll/ECE-15bin + per-kind/domain splits), matching the
shape produced by ``valen.evaluate`` closely enough for side-by-side
comparison. Advisory measurement only.
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "external" / "valen"))

LABELS = ["A", "B"]
SYSTEM = ("Choose the best available candidate for the question using only "
          "the supplied state. Return exactly one candidate label.")
CANDIDATES = ("Candidates:\n"
              "A) true — the candidate context is certainly irrelevant to "
              "the current user request.\n"
              "B) false — the candidate context may contain required "
              "evidence, a user constraint, a correction, or information "
              "needed to interpret another segment.")


def render_state(state_str):
    s = json.loads(state_str)
    lines = []
    candidate_text = None
    for entry in s["conversation"]:
        text = str(entry["content"])
        lines.append(f'{entry["role"]}: {text}')
        if entry["pointer"] == s["candidate_pointer"]:
            candidate_text = text
    block = "\n".join(lines)
    if candidate_text is not None:
        block += f"\n\nCandidate context (under evaluation):\n{candidate_text}"
    return block


def build_prompt(processor, record):
    req = record["request"]
    q = req["questions"]["irrelevant"]
    state_block = render_state(req["state"])
    text = (f"State:\n{state_block}\n\n"
            f"Question: {q['instructions']}\n\n"
            f"{CANDIDATES}\n\n"
            "Answer with exactly one candidate label.")
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": [{"type": "text", "text": text}]}]
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", type=Path,
                    default=ROOT / "external/valen/models/Qwen3.5-0.8B")
    ap.add_argument("--lora-checkpoint", type=Path, default=None,
                    help="valen checkpoint dir whose LoRA deltas are merged "
                         "into the backbone before readout (JEMM-style arm: "
                         "trained backbone, head-free label-token readout)")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, dtype=dtype, device_map=args.device).eval()
    if args.lora_checkpoint:
        ckpt = torch.load(args.lora_checkpoint / "checkpoint.pt",
                          map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        scaling = cfg.get("lora_alpha", 64) / cfg.get("lora_rank", 32)
        merged = 0
        with torch.no_grad():
            for name, p in model.named_parameters():
                if not name.endswith(".weight"):
                    continue
                prefix = ("backbone.language_model.base_model.model."
                          + name.removeprefix("model.language_model.")
                          .removesuffix(".weight"))
                a = ckpt["weights"].get(prefix + ".lora_A.default.weight")
                b = ckpt["weights"].get(prefix + ".lora_B.default.weight")
                if a is not None and b is not None:
                    p.add_((b.float() @ a.float()) * scaling)
                    merged += 1
        if not merged:
            raise ValueError("no LoRA deltas matched the backbone "
                             "(check checkpoint/key mapping)")
        print(json.dumps({"event": "lora_merged", "modules": merged,
                          "scaling": scaling}), flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    tok = processor.tokenizer
    label_ids = [tok.encode(x, add_special_tokens=False)[0] for x in LABELS]

    records = [json.loads(l) for l in args.data.open()]
    if args.limit:
        records = records[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    started = time.perf_counter()
    for i, rec in enumerate(records):
        prompt = build_prompt(processor, rec)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            logits = model(**inputs, logits_to_keep=1).logits[0, -1]
        probs = torch.softmax(logits[label_ids].float(), dim=-1)
        p_true = probs[0].item()  # P(candidate is irrelevant)
        target = rec["targets"]["irrelevant"]["probabilities"]["true"]
        pred_drop = p_true > 0.5
        correct = pred_drop == bool(target)
        brier = (p_true - target) ** 2 + ((1 - p_true) - (1 - target)) ** 2
        nll = -__import__("math").log(max(
            p_true if target == 1.0 else 1 - p_true, 1e-12))
        row = {
            "record_index": i,
            "record_id": rec.get("meta", {}).get("record_id", str(i)),
            "group_id": rec.get("group_id"),
            "domain": rec.get("meta", {}).get("domain"),
            "target": {"true": target, "false": 1 - target},
            "probabilities": {"true": p_true, "false": 1 - p_true},
            "metrics": {"accuracy": float(correct), "brier": brier,
                        "nll": nll},
        }
        rows.append(row)
        if (i + 1) % 50 == 0:
            print(json.dumps({"event": "progress", "records": i + 1,
                              "elapsed": round(time.perf_counter() - started, 1)}),
                  flush=True)

    out = args.output / "predictions.jsonl"
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def summarize(sub):
        n = len(sub)
        acc = sum(r["metrics"]["accuracy"] for r in sub) / n
        brier = sum(r["metrics"]["brier"] for r in sub) / n
        nll = sum(r["metrics"]["nll"] for r in sub) / n
        # ECE-15 on top-label confidence
        ece, bins = 0.0, 15
        for lo in range(bins):
            bucket = [r for r in sub
                      if lo / bins <= max(r["probabilities"]["true"],
                                          r["probabilities"]["false"])
                      < (lo + 1) / bins
                      or (lo == bins - 1 and
                          max(r["probabilities"]["true"],
                              r["probabilities"]["false"]) == 1.0)]
            if bucket:
                conf = sum(max(r["probabilities"]["true"],
                               r["probabilities"]["false"])
                           for r in bucket) / len(bucket)
                acc_b = sum(r["metrics"]["accuracy"] for r in bucket) / len(bucket)
                ece += len(bucket) / n * abs(acc_b - conf)
        return {"questions": n, "accuracy": acc, "brier": brier,
                "nll": nll, "ece": ece}

    metrics = {"overall": summarize(rows)}
    for key, fn in (("kind", lambda r: r["record_id"].split(":")[-1]),
                    ("domain", lambda r: r["domain"])):
        groups = defaultdict(list)
        for r in rows:
            groups[fn(r)].append(r)
        for name, sub in groups.items():
            metrics[f"{key}/{name}"] = summarize(sub)
    metrics["device"] = f"{args.device}/{args.dtype}"
    metrics["elapsed_seconds"] = time.perf_counter() - started
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({"event": "evaluation_complete",
                      "overall": metrics["overall"]}))


if __name__ == "__main__":
    main()
