#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--max-memory-gb", type=float, default=32.0)
    args = parser.parse_args()

    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from predict_toy_decisions import prepare_examples
    from train_pipeline_decisions_v4 import inject_lora
    from train_toy_decisions import DecisionModel

    started = time.perf_counter()
    config = AutoConfig.from_pretrained(args.model, revision=args.revision, trust_remote_code=False)
    if not hasattr(config, "hidden_size"):
        raise ValueError("top-level config lacks hidden_size; a reviewed language-backbone adapter is required")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone = AutoModel.from_pretrained(
        args.model, revision=args.revision, dtype=torch.float32,
        trust_remote_code=False, low_cpu_mem_usage=True)
    model = DecisionModel(backbone, "attention").eval().to(args.device)
    payload = {"states": [{
        "id": "owned-compat-fixture",
        "state": "Service api-west has 3 healthy replicas, error rate 0.2%, and no active incident.",
        "questions": {
            "healthy": {"type": "boolean", "instructions": "Is the service healthy?"},
            "action": {"type": "choice", "instructions": "Choose the next action.",
                       "criteria": {"observe": "Continue observing", "rollback": "Roll back the release",
                                    "page": "Page the incident commander"}},
            "risk": {"type": "score", "instructions": "Score operational risk.",
                     "criteria": ["low", "medium", "high", "critical"]},
        },
    }]}
    examples = prepare_examples(payload, tokenizer, 512)
    if args.device == "mps":
        torch.mps.synchronize()
        before_memory = torch.mps.current_allocated_memory()
    elif args.device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before_memory = torch.cuda.memory_allocated()
    else:
        before_memory = 0
    with torch.inference_mode():
        logits_before, _ = model(examples, tokenizer.pad_token_id)
    if args.device == "mps":
        torch.mps.synchronize()
        after_forward_memory = torch.mps.current_allocated_memory()
    elif args.device == "cuda":
        torch.cuda.synchronize()
        after_forward_memory = torch.cuda.memory_allocated()
    else:
        after_forward_memory = 0
    wrapped = inject_lora(model, 16, 32.0, 0.05)
    model.to(args.device).eval()
    with torch.inference_mode():
        logits_after, _ = model(examples, tokenizer.pad_token_id)
    if args.device == "mps":
        torch.mps.synchronize()
        peak_observed = max(before_memory, after_forward_memory, torch.mps.current_allocated_memory())
    elif args.device == "cuda":
        torch.cuda.synchronize()
        peak_observed = max(torch.cuda.max_memory_allocated(), torch.cuda.memory_allocated())
    else:
        peak_observed = 0
    deltas = []
    finite = True
    for left, right, example in zip(logits_before, logits_after, examples):
        k = len(example["candidate_ids"])
        finite = finite and bool(torch.isfinite(left[:k]).all()) and bool(torch.isfinite(right[:k]).all())
        deltas.append(float((left[:k] - right[:k]).abs().max().cpu()))
    max_delta = max(deltas)
    receipt = {
        "schema_version": "nanojev-stronger-backbone-probe-v1",
        "status": "passed" if finite and max_delta <= 1e-5 and peak_observed <= args.max_memory_gb * 1e9 else "failed",
        "model": {"id": args.model, "revision": args.revision, "license": "Apache-2.0",
                  "config_class": type(config).__name__, "backbone_class": type(backbone).__name__,
                  "hidden_size": config.hidden_size, "layers": config.num_hidden_layers,
                  "parameter_count": sum(p.numel() for p in model.parameters())},
        "runtime": {"device": args.device, "precision": "fp32", "network_model_calls": 0,
                    "load_and_probe_seconds": time.perf_counter() - started,
                    "allocated_before_forward_gb": before_memory / 1e9,
                    "allocated_after_forward_gb": after_forward_memory / 1e9,
                    "peak_observed_gb": peak_observed / 1e9,
                    "declared_memory_budget_gb": args.max_memory_gb},
        "compatibility": {"questions": len(examples), "finite_logits": finite,
                          "lora_wrapped_modules": len(wrapped),
                          "zero_adapter_max_abs_logit_delta": max_delta,
                          "zero_adapter_tolerance": 1e-5},
        "fixture": {"owned": True, "contains_heldout_or_external_rows": False,
                    "payload_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()},
        "authorization": {"training_authorized": False, "deployment_authorized": False,
                          "jevbench_unlocked": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output), "receipt": receipt}, indent=2))
    if receipt["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
