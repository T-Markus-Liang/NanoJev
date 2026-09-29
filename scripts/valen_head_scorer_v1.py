#!/usr/bin/env python3
"""T156 prototype: nano_rlcd_v1 RLCD head (Qwen3.5-0.8B) as a scorer backend.

Mirrors the winnow scorer interface used by serve_decisions.py / nanojev-eval:
``score(record)`` accepts the same ``{"state", "questions"}`` request body that
is POSTed to ``/v1/systemone`` (or a full eval-format record with a ``request``
member) and returns keep/drop probabilities for the context-filter question
(``irrelevant``: ``noul`` = P(candidate certainly irrelevant) = drop).

Single-process, load-once design: the backbone + head and the compiler are
built exactly once per process via :func:`get_scorer`. NOT wired into
serve_decisions.py — see docs/VALEN_HEAD_SCORER_INTEGRATION_V1.md.

Run with the valen venv (transformers 5.4 / Qwen3.5 support):

    external/valen/.venv/bin/python scripts/valen_head_scorer_v1.py \
        --eval data/valen_nano_v1/eval.jsonl [--winnow-url http://127.0.0.1:8091]

Loopback only; no provider calls, no API keys anywhere.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
import time
from http.client import HTTPConnection

ROOT = Path(__file__).resolve().parent.parent
VALEN_ROOT = ROOT / "external" / "valen"
DEFAULT_CHECKPOINT = VALEN_ROOT / "output" / "nano_rlcd_v2" / "latest"
DEFAULT_EVAL = ROOT / "data" / "valen_nano_v1" / "eval.jsonl"
DEFAULT_WINNOW_URL = "http://127.0.0.1:8091"
MODEL_ID = "nano_rlcd_v2/valen-head@Qwen3.5-0.8B"

if str(VALEN_ROOT) not in sys.path:
    sys.path.insert(0, str(VALEN_ROOT))  # import valen without requiring install


def _resolve_model_path(config):
    """Checkpoint config stores model_path relative to the valen repo root."""
    path = Path(config["model_path"])
    if not path.is_absolute():
        path = (VALEN_ROOT / path).resolve()
    if not path.is_dir():
        raise ValueError(f"base model directory not found: {path}")
    config["model_path"] = str(path)
    return config


class ValenHeadScorer:
    """Load-once scorer wrapping the RLCD decision head on Qwen3.5-0.8B."""

    def __init__(self, checkpoint_dir=DEFAULT_CHECKPOINT, device="mps",
                 dtype="bf16", media_root="."):
        import torch
        from valen.modeling.factory import (build_model, build_compiler,
                                            get_backend, normalize_model_config)
        from valen.training.checkpoint import load_checkpoint
        self._torch = torch
        checkpoint_dir = Path(checkpoint_dir)
        config = json.loads((checkpoint_dir / "config.json").read_text("utf-8"))
        config = normalize_model_config(config)
        config = _resolve_model_path(config)
        config.update(device=device, dtype=dtype, gradient_checkpointing=False)
        self.config = config
        self.model = build_model(config)
        load_checkpoint(checkpoint_dir, self.model)
        self.model.eval()
        self.backend = get_backend(config["architecture"])
        self.compiler = build_compiler(config, media_root)
        self._lock = threading.Lock()  # one forward at a time, even for callers

    def _record(self, record):
        """Accept an eval-format record or a bare {state, questions} request."""
        if not isinstance(record, dict):
            raise ValueError("record must be a JSON object")
        if isinstance(record.get("request"), dict):
            return record
        if "state" in record and isinstance(record.get("questions"), dict):
            return {"request": {"state": record["state"],
                                "questions": record["questions"]}}
        raise ValueError("record needs a 'request' object or {state, questions}")

    def score(self, record, temperature=1.0):
        """Score one context-filter decision; returns keep/drop probabilities.

        Output mirrors the /v1/systemone shape (``answers`` map) plus a
        ``decision`` summary: ``drop`` = P(irrelevant=true), ``keep`` = 1-drop.
        """
        torch = self._torch
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        record = self._record(record)
        started = time.perf_counter()
        compiled = self.compiler.compile(record)
        with self._lock, torch.no_grad():
            decisions = [d for unit in self.backend.inference_units(self.model, compiled)
                         for d in unit]
        answers, probabilities = {}, {}
        for decision in decisions:
            question, logits = decision.question, decision.logits
            probs = (logits.float() / temperature).softmax(-1).cpu().tolist()
            total = sum(probs)
            probs = [p / total for p in probs]
            per_key = dict(zip(question.keys, probs))
            probabilities[question.qid] = per_key
            if question.kind == "noul":
                answers[question.qid] = {"type": "noul",
                                         "noul": per_key["true"],
                                         "probabilities": per_key}
            else:
                answers[question.qid] = {"type": question.kind,
                                         "probabilities": per_key}
        result = {"model": MODEL_ID,
                  "backend": "valen-head",
                  "answers": answers,
                  "usage": {"input_tokens": compiled.logical_tokens,
                            "output_tokens": 0},
                  "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)}
        # keep/drop summary: prefer the canonical 'irrelevant' question, else
        # the first noul question. noul semantics: true == certainly irrelevant.
        noul_qids = [qid for qid, a in answers.items() if a["type"] == "noul"]
        qid = "irrelevant" if "irrelevant" in noul_qids else (noul_qids[0] if noul_qids else None)
        if qid is not None:
            drop = answers[qid]["noul"]
            result["decision"] = {"qid": qid, "keep": 1.0 - drop, "drop": drop}
        return result


_SCORER = None


def get_scorer(**kwargs):
    """Process-wide singleton; loads the backbone + head exactly once."""
    global _SCORER
    if _SCORER is None:
        _SCORER = ValenHeadScorer(**kwargs)
    return _SCORER


def score(record, **kwargs):
    """Module-level convenience: score via the singleton scorer."""
    return get_scorer(**kwargs).score(record)


def _target_label(record, qid="irrelevant"):
    probs = ((record.get("targets") or {}).get(qid) or {}).get("probabilities")
    if not isinstance(probs, dict):
        return None
    return max(probs, key=probs.get)


def _ece(rows, bins=15):
    """Expected calibration error on top-label confidence vs correctness."""
    if not rows:
        return None
    error, total = 0.0, len(rows)
    for lo in range(bins):
        bucket = [r for r in rows
                  if lo / bins <= r["confidence"] < (lo + 1) / bins
                  or (lo == bins - 1 and r["confidence"] == 1.0)]
        if not bucket:
            continue
        conf = sum(r["confidence"] for r in bucket) / len(bucket)
        acc = sum(r["correct"] for r in bucket) / len(bucket)
        error += (len(bucket) / total) * abs(acc - conf)
    return error


def _winnow_score(request, url, timeout=120.0):
    from urllib.parse import urlsplit
    parts = urlsplit(url)
    body = json.dumps({"model": "Winnow-12B", **request})
    connection = HTTPConnection(parts.hostname, parts.port, timeout=timeout)
    started = time.perf_counter()
    try:
        connection.request("POST", "/v1/systemone", body=body,
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        data = json.loads(response.read(2_000_000))
        if response.status != 200:
            raise ValueError(f"winnow returned {response.status}")
        return (data["answers"]["irrelevant"]["noul"],
                round((time.perf_counter() - started) * 1000, 1))
    finally:
        connection.close()


def run_eval(args):
    records = [json.loads(line) for line in
               Path(args.eval).read_text("utf-8").splitlines() if line.strip()]
    if args.limit:
        records = records[: args.limit]
    scorer = get_scorer(checkpoint_dir=args.checkpoint, device=args.device,
                        dtype=args.dtype)
    valen_rows, winnow_rows, latencies, winnow_latencies = [], [], [], []
    disagreements = 0
    for index, record in enumerate(records):
        result = scorer.score(record)
        latency = result["elapsed_ms"] / 1000.0
        latencies.append(latency)
        noul = result["decision"]["drop"]
        target = _target_label(record)
        predicted = "true" if noul >= 0.5 else "false"
        valen_rows.append({"confidence": max(noul, 1.0 - noul),
                           "correct": float(predicted == target)})
        winnow_prediction = None
        if args.winnow_url:
            w_noul, w_ms = _winnow_score(record["request"], args.winnow_url)
            winnow_latencies.append(w_ms / 1000.0)
            w_pred = "true" if w_noul >= 0.5 else "false"
            winnow_rows.append({"confidence": max(w_noul, 1.0 - w_noul),
                                "correct": float(w_pred == target)})
            winnow_prediction = w_pred
            if w_pred != predicted:
                disagreements += 1
        if (index + 1) % 25 == 0 or index + 1 == len(records):
            print(json.dumps({"event": "progress", "records": index + 1,
                              "total": len(records)}), flush=True)
    def summarize(rows, times):
        accuracy = sum(r["correct"] for r in rows) / len(rows) if rows else None
        ordered = sorted(times)
        return {"n": len(rows), "accuracy": accuracy, "ece": _ece(rows),
                "latency_seconds": {
                    "mean": sum(times) / len(times) if times else None,
                    "p50": ordered[len(ordered) // 2] if ordered else None,
                    "p95": ordered[int(len(ordered) * 0.95)] if ordered else None}}
    report = {"schema_version": "nanojev-valen-head-scorer-v1",
              "checkpoint": str(args.checkpoint),
              "eval_data": str(args.eval),
              "eval_sha256": hashlib.sha256(
                  Path(args.eval).read_bytes()).hexdigest(),
              "device": args.device, "dtype": args.dtype,
              "valen_head": summarize(valen_rows, latencies),
              "winnow": summarize(winnow_rows, winnow_latencies) if winnow_rows else None,
              "disagreements": disagreements if winnow_rows else None}
    if winnow_rows:
        report["accuracy_delta"] = (report["valen_head"]["accuracy"]
                                    - report["winnow"]["accuracy"])
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(encoded, encoding="utf-8")
    print(encoded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL,
                        help="JSONL of eval-format records")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--winnow-url", default=None,
                        help="also score every record via this /v1/systemone scorer")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
