"""Finalize a v4 LoRA run dir that crashed at the post-train merge verify step.

best.safetensors already holds the merged best-step weights (written by
save_best during training). This script reproduces the trainer's tail:
load merged weights into a plain DecisionModel, evaluate splits, write
predictions_{split}.jsonl / predictions.jsonl / summary.json.

Usage: finalize_v4_lora_run.py --run-dir <dir> --input <trainer_view_dir>
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from safetensors.torch import load_file

import train_pipeline_decisions_v4 as T
from train_toy_decisions import DecisionModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--best-step", type=int, required=True, help="best dev step recovered from train log")
    args = p.parse_args()
    out = Path(args.run_dir)
    config = json.loads((out / "config.json").read_text())
    best_step = args.best_step

    class A:  # minimal args namespace compatible with evaluate_pipeline
        pass
    a = A()
    for k in ("batch_questions", "microbatch_questions", "max_microbatch_tokens",
              "max_length", "device", "precision", "objective", "set_head"):
        setattr(a, k, config[k])
    device = torch.device(a.device)

    tokenizer = AutoTokenizer.from_pretrained(str(out / "tokenizer"))
    examples, _audit = T.load_training_examples(args.input, tokenizer, a.max_length)
    splits = {s: [e for e in examples if e["split"] == s] for s in T.SPLITS}

    bcfg = AutoConfig.from_pretrained(str(out / "backbone_config"), local_files_only=True)
    bcfg.use_cache = False
    backbone = AutoModel.from_config(bcfg, attn_implementation="sdpa").float()
    model = DecisionModel(backbone, a.set_head)
    model.load_state_dict(load_file(out / "best.safetensors"), strict=True)
    model.to(device).eval()

    final = {}
    for split in ("dev", "calibration", "test", "ood"):
        if splits[split]:
            final[split] = T.evaluate_pipeline(model, splits[split], tokenizer.pad_token_id,
                                               a, a.objective, out / f"predictions_{split}.jsonl")
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as h:
        for split in final:
            h.write((out / f"predictions_{split}.jsonl").read_text())
    dev = final["dev"]
    summary = {"best_step": best_step, "best_dev_target_ce": dev["target_ce"],
               "selected_on": "dev target CE (finalized post-hoc; see train_log.json for per-step dev)",
               "objective": a.objective, "metrics_by_split": final, "temperature": 1.0,
               "temperature_fitted": False, "runtime_device": str(device),
               "finalized_by": "scripts/finalize_v4_lora_run.py"}
    T.dump(out / "summary.json", summary)
    print(json.dumps({"finalized": str(out), "dev_ce": dev["target_ce"]}))


if __name__ == "__main__":
    main()
