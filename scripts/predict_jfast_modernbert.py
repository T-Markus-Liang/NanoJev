"""Predictor-compatible runtime for J-FAST ModernBERT marker-head checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from jfast_modernbert_decision_v1 import prepare_payload
from predict_toy_decisions import (answer_from_probabilities, resolve_runtime,
                                   validate_request)
from train_jfast_modernbert import MarkerDecisionModel, batch_examples


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class JFastDecisionPredictor:
    def __init__(self, checkpoint_dir, device_name="auto", precision="auto"):
        root = Path(checkpoint_dir).expanduser().resolve(strict=True)
        self.root = root
        self.run_config = json.loads((root / "config.json").read_text())
        self.tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer",
                                                       local_files_only=True)
        config = AutoConfig.from_pretrained(
            self.run_config["model"], revision=self.run_config["revision"],
            local_files_only=True)
        backbone = AutoModel.from_config(config)
        backbone.load_state_dict(load_file(root / "backbone.safetensors"),
                                 strict=True)
        self.model = MarkerDecisionModel(backbone)
        self.model.marker_head.load_state_dict(
            load_file(root / "marker_head.safetensors"), strict=True)
        self.device, self.precision = resolve_runtime(
            torch, device_name, precision)
        self.model.to(self.device)
        self.model.eval()
        self.limit = int(self.run_config["max_length"])
        self.inference_calls = 0

    def predict(self, payload, batch_questions=8, temperature=1.0):
        validate_request(payload)
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) \
                or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature 必须为有限正数")
        examples = prepare_payload(
            payload, self.tokenizer, self.run_config["max_length"],
            self.run_config.get("marker_placement", "before"))
        for ex in examples:
            ex["candidate_ids"] = ex["labels"]
            ex["type"] = ex["question_type"]
        groups = [examples[i:i + batch_questions]
                  for i in range(0, len(examples), batch_questions)] \
            if batch_questions else [examples]
        outputs = {state["id"]: {"id": state["id"], "answers": {}}
                   for state in payload["states"]}
        latencies = []
        forward_passes = 0
        self.inference_calls += 1
        with torch.inference_mode():
            for group in groups:
                start = time.perf_counter()
                logits = self.model(group, self.tokenizer.pad_token_id)
                if self.device.type == "mps":
                    torch.mps.synchronize()
                latency = time.perf_counter() - start
                latencies.extend([latency / len(group)] * len(group))
                forward_passes += 1
                for ex, scores in zip(group, logits):
                    if not torch.isfinite(scores).all():
                        raise ValueError("模型产生非有限logits，未返回部分预测")
                    probabilities = (scores / temperature).softmax(-1).cpu().tolist()
                    outputs[ex["state_id"]]["answers"][ex["question_id"]] = \
                        answer_from_probabilities(ex, probabilities)
        return {
            "schema_version": "openjev-toy-inference-v1",
            "checkpoint": {"directory": str(self.root), "base_model": self.run_config["model"],
                           "base_revision": self.run_config["revision"],
                           "set_head": self.run_config["set_head"]},
            "temperature": {"value": float(temperature),
                            "fitted_by_this_command": False,
                            "note": "显式应用给定标量；默认1不表示模型已校准。"},
            "execution": {"device": str(self.device),
                          "parameter_storage": "float32",
                          "precision": self.precision,
                          "forward_autocast": "disabled",
                          "states": len(payload["states"]),
                          "questions": len(examples),
                          "candidate_paths": sum(len(x["marker_positions"])
                                                 for x in examples),
                          "marker_packed_tokens": sum(len(x["input_ids"])
                                                      for x in examples),
                          "forward_passes": forward_passes,
                          "batch_questions_limit": batch_questions or "all",
                          "autoregressive_decode_steps": 0,
                          "prefix_sharing": True,
                          "max_length": self.run_config["max_length"],
                          "network_model_calls": 0,
                          "persistent_model_load_count": 1,
                          "inference_call_index": self.inference_calls},
            "timing": {"question_latency_s": {
                "min": min(latencies) if latencies else None,
                "max": max(latencies) if latencies else None,
                "mean": statistics.fmean(latencies) if latencies else None}},
            "states": list(outputs.values())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch-questions", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16"],
                        default="auto")
    args = parser.parse_args()
    engine = JFastDecisionPredictor(args.checkpoint_dir, args.device,
                                    args.precision)
    started = time.perf_counter()
    result = engine.predict(json.loads(Path(args.input).read_text()),
                            batch_questions=args.batch_questions,
                            temperature=args.temperature)
    result["timing"]["wall_s"] = time.perf_counter() - started
    text = json.dumps(result, ensure_ascii=False, indent=2,
                      allow_nan=False) + "\n"
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
