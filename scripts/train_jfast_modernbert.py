"""Train a J-FAST ModernBERT marker-head decision model on corpus v5 rows.

This is a deliberately small trainer for the J-FAST amendment. It consumes
trainer_view JSONL records, builds one marker sequence per question, trains a
linear marker head plus optional LoRA adapters, and saves a merged checkpoint.
No heldout or benchmark file is read by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoModel, AutoTokenizer

from jfast_modernbert_decision_v1 import prepare_payload
from predict_toy_decisions import resolve_runtime
from train_pipeline_decisions_v4 import (is_lora_param, lora_linear_cls,
                                        merged_state_dict)

JFAST_LORA_TARGETS = ("Wqkv", "Wo", "Wi")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def row_to_example(row, tokenizer, max_length, marker_placement="before"):
    if row.get("schema_version") != "nanojev-engineering-judgment-trainer-row-v1":
        raise ValueError(f"{row.get('id')}: unexpected schema {row.get('schema_version')!r}")
    qid, question = next(iter(row["questions"].items()))
    payload = {"states": [{"id": row["id"], "state": row["state"],
                           "questions": {qid: question}}]}
    ex = prepare_payload(payload, tokenizer, max_length, marker_placement)[0]
    gold = row["gold_probs"][qid]
    target = [float(gold.get(label, 0.0)) for label in ex["labels"]]
    if abs(sum(target) - 1.0) > 1e-6:
        raise ValueError(f"{row['id']}:{qid} gold_probs does not sum to 1")
    ex.update({"id": row["id"], "target": target})
    return ex


def prepare_rows(rows, tokenizer, max_length, marker_placement="before"):
    return [row_to_example(row, tokenizer, max_length, marker_placement)
            for row in rows]


def batch_examples(examples, pad_token_id, device):
    width = max(len(ex["input_ids"]) for ex in examples)
    input_ids = torch.full((len(examples), width), pad_token_id,
                           dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(examples), width), dtype=torch.long,
                                 device=device)
    positions = []
    for i, ex in enumerate(examples):
        input_ids[i, :len(ex["input_ids"])] = torch.tensor(
            ex["input_ids"], dtype=torch.long, device=device)
        attention_mask[i, :len(ex["input_ids"])] = 1
        positions.append(torch.tensor(ex["marker_positions"],
                                      dtype=torch.long, device=device))
    return input_ids, attention_mask, positions


class MarkerDecisionModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.marker_head = nn.Linear(backbone.config.hidden_size, 1)

    def forward(self, examples, pad_token_id):
        input_ids, attention_mask, positions = batch_examples(
            examples, pad_token_id, next(self.parameters()).device)
        hidden = self.backbone(input_ids=input_ids,
                               attention_mask=attention_mask).last_hidden_state
        logits = []
        for i, pos in enumerate(positions):
            logits.append(self.marker_head(hidden[i, pos]).squeeze(-1).float())
        return logits


def gold_ce(logits_list, examples):
    total = logits_list[0].new_zeros(())
    for logits, ex in zip(logits_list, examples):
        target = torch.tensor(ex["target"], dtype=torch.float32,
                              device=logits.device)
        total = total - (target * F.log_softmax(logits, dim=0)).sum()
    return total / len(examples)


def inject_jfast_lora(model, rank, alpha, dropout):
    cls = lora_linear_cls()
    wrapped = []
    for module_name, module in list(model.backbone.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name in JFAST_LORA_TARGETS and isinstance(child, nn.Linear):
                setattr(module, child_name, cls(child, rank, alpha, dropout))
                wrapped.append(f"{module_name}.{child_name}")
    for name, param in model.named_parameters():
        if name.startswith("backbone.") and not is_lora_param(name):
            param.requires_grad_(False)
    if not wrapped:
        raise ValueError("no ModernBERT LoRA target modules wrapped")
    return wrapped


def grouped_batches(examples, batch_questions, max_tokens):
    batches = []
    current = []
    current_max = 0
    for ex in examples:
        estimate = max(current_max, len(ex["input_ids"])) * (len(current) + 1)
        if current and (len(current) >= batch_questions or estimate > max_tokens):
            batches.append(current)
            current = []
            current_max = 0
        current.append(ex)
        current_max = max(current_max, len(ex["input_ids"]))
    if current:
        batches.append(current)
    return batches


def evaluate(model, examples, pad_token_id, batch_questions, max_tokens,
             device, precision):
    model.eval()
    losses = []
    correct = 0
    with torch.inference_mode():
        for batch in grouped_batches(examples, batch_questions, max_tokens):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda" and precision == "bf16"):
                logits = model(batch, pad_token_id)
            losses.append(float(gold_ce(logits, batch)))
            for logit, ex in zip(logits, batch):
                probs = torch.softmax(logit, dim=0)
                pred = ex["labels"][int(torch.argmax(probs))]
                gold = ex["labels"][int(torch.argmax(torch.tensor(ex["target"])))]
                correct += int(pred == gold)
    model.train()
    return {"ce": statistics.fmean(losses),
            "accuracy": correct / len(examples),
            "n": len(examples)}


def save_merged_checkpoint(model, tokenizer, out_dir: Path, config):
    out_dir.mkdir(parents=True, exist_ok=True)
    merged = merged_state_dict(model)
    backbone_state = {k[len("backbone."):]: v for k, v in merged.items()
                      if k.startswith("backbone.")}
    head_state = {k[len("marker_head."):]: v for k, v in merged.items()
                  if k.startswith("marker_head.")}
    save_file(backbone_state, str(out_dir / "backbone.safetensors"))
    save_file(head_state, str(out_dir / "marker_head.safetensors"))
    tokenizer.save_pretrained(out_dir / "tokenizer")
    (out_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def self_test():
    assert JFAST_LORA_TARGETS == ("Wqkv", "Wo", "Wi")
    fake = [{"input_ids": [1, 2, 3], "marker_positions": [1], "labels": ["a"],
             "target": [1.0]}]
    ids, mask, pos = batch_examples(fake, 0, torch.device("cpu"))
    assert ids.tolist() == [[1, 2, 3]] and mask.tolist() == [[1, 1, 1]]
    assert pos[0].tolist() == [1]
    logits = [torch.tensor([0.0, 1.0])]
    examples = [{"target": [0.0, 1.0]}]
    assert float(gold_ce(logits, examples)) > 0.0
    print(json.dumps({"self_test": "passed", "lora_targets": JFAST_LORA_TARGETS}))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--dev", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-id", default="answerdotai/ModernBERT-base")
    parser.add_argument("--revision", default="8949b909ec900327062f0ebf497f51aef5e6f0c8")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16"], default="auto")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--batch-questions", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--head-warmup", type=int, default=20)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--marker-placement", choices=["before", "after"],
                        default="before")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.train or not args.dev or not args.output_dir:
        raise ValueError("--train, --dev and --output-dir are required unless --self-test")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device, precision = resolve_runtime(torch, args.device, args.precision)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)
    train_rows = prepare_rows(load_rows(args.train), tokenizer, args.max_length,
                              args.marker_placement)
    dev_rows = prepare_rows(load_rows(args.dev), tokenizer, args.max_length,
                            args.marker_placement)
    backbone = AutoModel.from_pretrained(args.model_id, revision=args.revision).to(device)
    model = MarkerDecisionModel(backbone)
    wrapped = inject_jfast_lora(model, args.lora_rank, args.lora_alpha,
                                args.lora_dropout)
    model.to(device)
    head_params = list(model.marker_head.parameters())
    lora_params = [p for n, p in model.named_parameters() if is_lora_param(n)]
    optimizer = torch.optim.AdamW(
        [{"params": head_params, "lr": args.head_lr},
         {"params": lora_params, "lr": args.lr}], weight_decay=0.01)
    best = float("inf")
    step = 0
    pool = train_rows[:]
    logs = []
    model.train()
    while step < args.steps:
        if not pool:
            random.shuffle(pool := train_rows[:])
        batch = []
        max_len = 0
        while pool and len(batch) < args.batch_questions:
            ex = pool.pop()
            estimate = max(max_len, len(ex["input_ids"])) * (len(batch) + 1)
            if batch and estimate > args.max_tokens:
                pool.append(ex)
                break
            batch.append(ex)
            max_len = max(max_len, len(ex["input_ids"]))
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda" and precision == "bf16"):
            logits = model(batch, tokenizer.pad_token_id)
        loss = gold_ce(logits, batch)
        loss.backward()
        if step + 1 > args.head_warmup:
            optimizer.step()
        else:
            # Keep LoRA gradients out of warmup: retain only head grads.
            for p in lora_params:
                p.grad = None
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, dev_rows, tokenizer.pad_token_id,
                               args.batch_questions, args.max_tokens,
                               device, precision)
            item = {"step": step, "dev": metrics, "loss": float(loss)}
            logs.append(item)
            print(json.dumps(item), flush=True)
            if metrics["ce"] < best:
                best = metrics["ce"]
                save_merged_checkpoint(model, tokenizer, args.output_dir, {
                    "model": args.model_id, "revision": args.revision,
                    "set_head": "marker", "max_length": args.max_length,
                    "marker_placement": args.marker_placement,
                    "best_step": step, "dev_ce": best})
        elif step % 10 == 0:
            print(json.dumps({"step": step, "loss": float(loss)}), flush=True)
    (args.output_dir / "training_log.json").write_text(
        json.dumps({"wrapped_lora": wrapped, "logs": logs}, indent=2),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
