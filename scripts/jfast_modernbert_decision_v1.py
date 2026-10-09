"""J-FAST F1 probe: ModernBERT encoder + marker-head typed decisions.

Each question is encoded as one sequence with one [MASK] marker per output
label. A linear head scores the hidden state at each marker position and the
scores are softmaxed over that question's label set. This is an architecture
compatibility probe; with --zero-head it emits uniform probabilities and is not
a quality result.

Remote model files are disabled unless --allow-download is passed. Inputs over
--max-length fail closed instead of truncating.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def candidate_labels(question: dict):
    typ = question["type"]
    criteria = question.get("criteria")
    if typ == "boolean":
        criteria = criteria or {}
        return ["false", "true"], [
            criteria.get("false", "The proposition is false."),
            criteria.get("true", "The proposition is true."),
        ]
    if typ == "choice":
        labels = list(criteria)
        return labels, [criteria[k] for k in labels]
    if typ == "score":
        labels = [str(i) for i in range(len(criteria))]
        return labels, list(criteria)
    raise ValueError(f"unsupported question type {typ!r}")


def encode_question(tokenizer, state, question, max_length,
                    marker_placement="before"):
    if marker_placement not in {"before", "after"}:
        raise ValueError("marker_placement must be before or after")
    labels, texts = candidate_labels(question)
    segments = [
        "State:\n",
        state if isinstance(state, str) else json.dumps(state, ensure_ascii=False,
                                                       sort_keys=True),
        "\nQuestion type: ", question["type"],
        "\nQuestion: ", question["instructions"],
        "\nOptions:\n",
    ]
    marker_positions = []
    ids = tokenizer.encode("".join(segments), add_special_tokens=True)
    mask_id = tokenizer.mask_token_id
    if type(mask_id) is not int or mask_id < 0:
        raise ValueError("ModernBERT tokenizer must expose mask_token_id")
    for label, text in zip(labels, texts):
        if marker_placement == "after":
            ids.extend(tokenizer.encode(f" {label}: {text}\n",
                                        add_special_tokens=False))
            marker_positions.append(len(ids))
            ids.append(mask_id)
        else:
            marker_positions.append(len(ids))
            ids.append(mask_id)
            ids.extend(tokenizer.encode(f" {label}: {text}\n",
                                        add_special_tokens=False))
    ids.extend(tokenizer.encode("Decision:", add_special_tokens=False))
    if len(ids) > max_length:
        raise ValueError(f"marker sequence is {len(ids)} tokens, exceeds {max_length}")
    return {"input_ids": ids, "marker_positions": marker_positions,
            "labels": labels}


def prepare_payload(payload, tokenizer, max_length, marker_placement="before"):
    examples = []
    for state in payload["states"]:
        for qid, question in state["questions"].items():
            encoded = encode_question(tokenizer, state["state"], question,
                                      max_length, marker_placement)
            examples.append({"state_id": state["id"], "question_id": qid,
                             "question_type": question["type"],
                             "expected": question.get("expected"), **encoded})
    return examples


def percentile(values, q):
    if not values:
        return None
    vals = sorted(values)
    pos = (len(vals) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-json", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-id", default="answerdotai/ModernBERT-base")
    parser.add_argument("--revision", default="8949b909ec900327062f0ebf497f51aef5e6f0c8")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"],
                        default="auto")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path,
                        help="load backbone.safetensors + marker_head.safetensors")
    parser.add_argument("--marker-placement", choices=["before", "after"],
                        default=None)
    parser.add_argument("--zero-head", action="store_true",
                        help="keep marker head zero-initialized for compatibility probe")
    args = parser.parse_args()

    import torch
    from torch import nn
    from transformers import AutoModel, AutoTokenizer

    payload = json.loads(args.probe_json.read_text(encoding="utf-8"))
    checkpoint_config = {}
    if args.checkpoint_dir:
        checkpoint_config = json.loads(
            (args.checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    marker_placement = (args.marker_placement or
                        checkpoint_config.get("marker_placement", "before"))
    local_only = not args.allow_download
    tokenizer_source = (args.checkpoint_dir / "tokenizer"
                        if args.checkpoint_dir else args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        revision=None if args.checkpoint_dir else args.revision,
        local_files_only=local_only or args.checkpoint_dir is not None)
    device_name = ("mps" if args.device == "auto" and
                   torch.backends.mps.is_available() else
                   "cpu" if args.device == "auto" else args.device)
    device = torch.device(device_name)
    load_start = time.perf_counter()
    backbone = AutoModel.from_pretrained(
        args.model_id, revision=args.revision, local_files_only=local_only)
    head = nn.Linear(backbone.config.hidden_size, 1)
    if args.checkpoint_dir:
        from safetensors.torch import load_file
        backbone.load_state_dict(
            load_file(args.checkpoint_dir / "backbone.safetensors"), strict=True)
        head.load_state_dict(
            load_file(args.checkpoint_dir / "marker_head.safetensors"), strict=True)
    backbone = backbone.to(device)
    head = head.to(device)
    zero_head = args.zero_head or args.checkpoint_dir is None
    if zero_head:
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
    backbone.eval()
    head.eval()
    model_load_s = time.perf_counter() - load_start
    examples = prepare_payload(payload, tokenizer, args.max_length,
                               marker_placement)
    results = []
    latencies = []
    for ex in examples:
        tokens = torch.tensor([ex["input_ids"]], dtype=torch.long, device=device)
        positions = torch.tensor(ex["marker_positions"], dtype=torch.long,
                                 device=device)
        start = time.perf_counter()
        with torch.inference_mode():
            hidden = backbone(input_ids=tokens).last_hidden_state[0]
            marker_h = hidden[positions]
            logits = head(marker_h).squeeze(-1).float()
            probs = torch.softmax(logits, dim=0)
        if device.type == "mps":
            torch.mps.synchronize()
        latency = time.perf_counter() - start
        latencies.append(latency)
        prob_map = {label: float(p) for label, p in zip(ex["labels"], probs)}
        predicted = max(prob_map, key=prob_map.get)
        results.append({"state_id": ex["state_id"],
                        "question_id": ex["question_id"],
                        "question_type": ex["question_type"],
                        "input_tokens": len(ex["input_ids"]),
                        "marker_count": len(ex["marker_positions"]),
                        "probs": prob_map, "predicted": predicted,
                        "confidence": max(prob_map.values()),
                        "correct": (None if ex["expected"] is None else
                                    predicted == str(ex["expected"])),
                        "latency_s": latency,
                        "finite": bool(torch.isfinite(logits).all() and
                                       torch.isfinite(probs).all())})
    manifest = {"schema_version": "nanojev-jfast-marker-head-probe-v1",
                "status": "completed" if all(r["finite"] for r in results)
                          else "completed_with_nonfinite_outputs",
                "probe_json": str(args.probe_json),
                "probe_json_sha256": sha256_file(args.probe_json),
                "model_id": args.model_id, "revision": args.revision,
                "checkpoint_dir": str(args.checkpoint_dir) if args.checkpoint_dir else None,
                "device": device_name, "max_length": args.max_length,
                "zero_head": bool(zero_head),
                "marker_placement": marker_placement,
                "model_load_s": model_load_s,
                "questions": len(results),
                "latency_s": {"n": len(latencies),
                              "min": min(latencies) if latencies else None,
                              "max": max(latencies) if latencies else None,
                              "mean": statistics.fmean(latencies) if latencies else None,
                              "p50": percentile(latencies, 0.5),
                              "p95": percentile(latencies, 0.95)},
                "mps_allocated_bytes": (torch.mps.current_allocated_memory()
                                        if device.type == "mps" else None),
                "results": results}
    text = json.dumps(manifest, ensure_ascii=False, indent=2,
                      allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
