"""J-FAST owned compatibility probe for encoder-style decision backbones.

This probe is deliberately conservative:
  * --dry-run only inspects the owned payload and writes no model output;
  * remote model files are disabled unless --allow-download is passed;
  * inputs longer than --max-length are marked context_fail and are not
    truncated;
  * it measures tokenizer/model compatibility and latency, not task accuracy.

Example:
  PYTHONPATH=scripts .venv/bin/python scripts/probe_jfast_backbones_v1.py \
    --probe-json research/jfast_owned_probe_v1.json --dry-run
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


def question_text(state: str, question: dict) -> str:
    parts = [f"State:\n{state}", f"Question type: {question['type']}",
             f"Question: {question['instructions']}"]
    criteria = question.get("criteria")
    if isinstance(criteria, dict):
        for key, value in criteria.items():
            parts.append(f"Option {key}: {value}")
    elif isinstance(criteria, list):
        for index, value in enumerate(criteria):
            parts.append(f"Level {index}: {value}")
    parts.append("Decision:")
    return "\n".join(parts)


def iter_questions(payload: dict):
    for state in payload["states"]:
        for qid, question in state["questions"].items():
            yield state["id"], qid, state["state"], question


def percentile(values, q):
    if not values:
        return None
    vals = sorted(values)
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def mps_allocated_bytes(torch):
    if torch.backends.mps.is_available():
        try:
            return torch.mps.current_allocated_memory()
        except Exception:
            return None
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-json", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--backend", choices=["modernbert"], default="modernbert")
    parser.add_argument("--model-id", default="answerdotai/ModernBERT-base")
    parser.add_argument("--revision", default="8949b909ec900327062f0ebf497f51aef5e6f0c8")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--allow-download", action="store_true",
                        help="permit remote model files; otherwise local cache only")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    payload = json.loads(args.probe_json.read_text(encoding="utf-8"))
    rows = []
    for state_id, qid, state, question in iter_questions(payload):
        text = question_text(state, question)
        rows.append({"state_id": state_id, "question_id": qid,
                     "question_type": question["type"],
                     "chars": len(text), "text": text})

    manifest = {"schema_version": "nanojev-jfast-probe-v1",
                "status": "dry_run_validated" if args.dry_run else "pending",
                "probe_json": str(args.probe_json),
                "probe_json_sha256": sha256_file(args.probe_json),
                "backend": args.backend,
                "model_id": args.model_id,
                "revision": args.revision,
                "download_allowed": bool(args.allow_download),
                "max_length": args.max_length,
                "questions": len(rows),
                "char_stats": {"min": min(r["chars"] for r in rows),
                               "max": max(r["chars"] for r in rows),
                               "mean": statistics.fmean(r["chars"] for r in rows)},
                "results": []}
    if args.dry_run:
        manifest["results"] = [{k: v for k, v in row.items() if k != "text"}
                               for row in rows]
    else:
        import torch
        from transformers import AutoModel, AutoTokenizer
        local_only = not args.allow_download
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_id, revision=args.revision,
            local_files_only=local_only)
        device_name = ("mps" if args.device == "auto" and
                       torch.backends.mps.is_available() else
                       "cpu" if args.device == "auto" else args.device)
        device = torch.device(device_name)
        load_start = time.perf_counter()
        model = AutoModel.from_pretrained(
            args.model_id, revision=args.revision,
            local_files_only=local_only).to(device)
        model.eval()
        model_load_s = time.perf_counter() - load_start
        latencies = []
        context_failures = 0
        for row in rows:
            encoded = tokenizer(row["text"], add_special_tokens=True,
                                return_tensors="pt")
            token_count = int(encoded["input_ids"].shape[1])
            result = {k: v for k, v in row.items() if k != "text"}
            result["input_tokens"] = token_count
            if token_count > args.max_length:
                result["status"] = "context_fail"
                context_failures += 1
                manifest["results"].append(result)
                continue
            inputs = {k: v.to(device) for k, v in encoded.items()}
            start = time.perf_counter()
            with torch.inference_mode():
                output = model(**inputs)
            if device.type == "mps":
                torch.mps.synchronize()
            latency = time.perf_counter() - start
            latencies.append(latency)
            hidden = output.last_hidden_state
            result.update({"status": "ok", "latency_s": latency,
                           "hidden_shape": list(hidden.shape),
                           "finite_hidden": bool(torch.isfinite(hidden).all())})
            manifest["results"].append(result)
        manifest.update({
            "status": "completed" if all(r["status"] == "ok" for r in manifest["results"])
                      else "completed_with_context_failures",
            "device": device_name,
            "model_load_s": model_load_s,
            "context_failures": context_failures,
            "latency_s": {"n": len(latencies),
                          "min": min(latencies) if latencies else None,
                          "max": max(latencies) if latencies else None,
                          "mean": statistics.fmean(latencies) if latencies else None,
                          "p50": percentile(latencies, 0.5),
                          "p95": percentile(latencies, 0.95)},
            "mps_allocated_bytes": mps_allocated_bytes(torch),
        })
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
