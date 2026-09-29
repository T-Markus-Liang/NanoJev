#!/usr/bin/env python3
"""Evaluate a local typed-decision scorer against the frozen A4 shadow fixtures.

This is a shadow-only diagnostic: every request is forwarded unchanged, and the
report records aggregate suggestions rather than applying any removal. It uses the
frozen tool-history fixture manifest as the only workload source.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_nanojev_v2 import dependency_versions, file_identity, hardware_descriptor, percentile
from build_tool_history_fixtures_v1 import (
    DEFAULT_MANIFEST, encode_request, load_manifest, substitute, synthetic_tag,
)
from context_gate_v1 import fingerprint, shadow_request
from scorer_adapters_v1 import DeadlineScorer, InProcessScorer, SystemOneHTTPScorer


SCHEMA_VERSION = "nanojev-context-shadow-systemone-v1"


class ReflexContextScorer:
    """In-process adapter from the gate payload to the Reflex engine contract."""

    def __init__(self, model, device="mps", revision=None, timeout=30.0):
        sys.path.insert(0, str(ROOT / "external" / "reflex" / "src"))
        import torch
        from huggingface_hub import snapshot_download
        from reflex import Engine, SystemOneRequest

        model_path = snapshot_download(model, revision=revision) if revision else model
        self.SystemOneRequest = SystemOneRequest
        self.engine = Engine.load(model_path, dtype=torch.float16, device=device,
                                  prompt_style="markdown", default_permutations=2)
        self.timeout = timeout

    def __call__(self, payload):
        states = []
        for state in payload.get("states", []):
            questions = {}
            for name, question in state.get("questions", {}).items():
                converted = dict(question)
                if converted.get("type") == "boolean":
                    converted["type"] = "noul"
                questions[name] = converted
            response = self.engine.answer(self.SystemOneRequest(
                state=state.get("state"), questions=questions))
            answers = {}
            for name, answer in response.answers.items():
                if getattr(answer, "noul", None) is not None:
                    probability = float(answer.noul)
                    answers[name] = {"type": "boolean", "probabilities": {
                        "false": 1.0 - probability, "true": probability}}
                else:
                    answers[name] = {"type": "boolean", "probabilities": answer.probabilities}
            states.append({"id": state["id"], "answers": answers})
        return {"checkpoint": {"adapter": "reflex-inprocess"}, "states": states}


class DeciderContextScorer:
    """In-process adapter from the gate payload to Mapika's Decider contract."""

    def __init__(self, model, device="mps", revision=None):
        import torch
        from huggingface_hub import snapshot_download

        model_path = snapshot_download(model, revision=revision) if revision else model
        sys.path.insert(0, str(model_path))
        from decider.infer import Decider

        self.decider = Decider(model_path, device=device, dtype=torch.float16,
                               use_graphs=False)

    def __call__(self, payload):
        states = []
        for state in payload.get("states", []):
            questions = {}
            for name, question in state.get("questions", {}).items():
                converted = dict(question)
                if converted.get("type") == "boolean":
                    converted["type"] = "noul"
                questions[name] = converted
            result = self.decider.system_one(
                state.get("state"), questions, independent=True)["answers"]
            answers = {}
            for name, answer in result.items():
                noul = answer.get("noul") if isinstance(answer, dict) else getattr(answer, "noul", None)
                if noul is not None:
                    probability = float(noul)
                    answers[name] = {"type": "boolean", "probabilities": {
                        "false": 1.0 - probability, "true": probability}}
                else:
                    probabilities = (answer.get("probabilities") if isinstance(answer, dict)
                                     else answer.probabilities)
                    answers[name] = {"type": "boolean", "probabilities": probabilities}
            states.append({"id": state["id"], "answers": answers})
        return {"checkpoint": {"adapter": "decider-inprocess"}, "states": states}


class SemIfContextScorer:
    """In-process adapter from the gate payload to SemIf's MLX scorer contract."""

    def __init__(self, model="Qwen/Qwen3.5-4B", revision=None, mlx_bits=4):
        sys.path.insert(0, str(ROOT / "external" / "semif" / "src"))
        from semif_phase1 import mlx_backend

        self.mlx_backend = mlx_backend
        self.model, self.tokenizer, self.metadata = mlx_backend.load_model(
            model, revision, mlx_bits)

    def __call__(self, payload):
        states = []
        for state in payload.get("states", []):
            answers = {}
            for name, question in state.get("questions", {}).items():
                if question.get("type") == "boolean":
                    options = [
                        {"id": "false", "description": "The proposition is false."},
                        {"id": "true", "description": "The proposition is true."},
                    ]
                else:
                    options = [{"id": key, "description": text}
                               for key, text in (question.get("criteria") or {}).items()]
                out = self.mlx_backend.score(
                    self.model, self.tokenizer,
                    {"id": state["id"], "state": state.get("state"),
                     "question": question.get("instructions"), "options": options},
                    self.metadata)
                probabilities = {option["id"]: float(probability)
                                 for option, probability in zip(options, out["probabilities"])}
                answers[name] = {"type": question.get("type"),
                                 "probabilities": probabilities}
            states.append({"id": state["id"], "answers": answers})
        return {"checkpoint": {"adapter": "semif-mlx-inprocess"}, "states": states}


class CountingScorer:
    def __init__(self, scorer):
        self.scorer = scorer
        self.calls = 0
        self.candidate_states = 0

    def __call__(self, payload):
        self.calls += 1
        self.candidate_states += len(payload.get("states", []))
        return self.scorer(payload)


def word_counter(text):
    return len(text.split()) if isinstance(text, str) else 0


def evaluate(manifest_path, scorer, threshold):
    manifest = load_manifest(manifest_path)
    seed = manifest.get("seed")
    rows, receipts, latencies = [], [], []
    counting = CountingScorer(scorer)
    for case in manifest.get("cases", []):
        tag = synthetic_tag(case["case_id"], seed)
        body = substitute(case["body"], tag)
        sidecar = substitute(case["sidecar"], tag)
        raw = encode_request(body)
        started = time.perf_counter()
        output, receipt = shadow_request(raw, case["wire_format"], sidecar, counting,
                                         threshold=threshold,
                                         token_counter=word_counter,
                                         tokenizer_id="whitespace-word-split-v1-not-a-provider-tokenizer")
        latencies.append((time.perf_counter() - started) * 1000)
        if output is not raw:
            raise RuntimeError("shadow_request did not return the original bytes object")
        declared = {segment["pointer"]: segment for segment in case.get("segments", [])}
        observed = {segment["pointer"]: segment for segment in receipt.get("segments", [])}
        case_rows = []
        for pointer, expected in declared.items():
            segment = observed.get(pointer, {})
            case_rows.append({
                "pointer": pointer,
                "expected_status": expected.get("expected_status"),
                "expected_suggestion": expected.get("expected_suggestion"),
                "suggestion": segment.get("suggestion"),
                "reason": segment.get("reason"),
                "p_irrelevant": segment.get("p_irrelevant"),
            })
        rows.append({
            "case_id": case["case_id"],
            "kind": case["kind"],
            "wire_format": case["wire_format"],
            "expected_gate": case.get("expected_gate"),
            "status": receipt.get("status"),
            "reason": receipt.get("reason"),
            "request_sha256": receipt.get("request_sha256"),
            "forwarded_sha256": receipt.get("forwarded_sha256"),
            "forwarded_unchanged": receipt.get("forwarded_unchanged"),
            "segments": case_rows,
            "token_counts": receipt.get("token_counts"),
        })
        receipts.append(receipt)
    return manifest, rows, receipts, latencies, counting


def summarize(rows, receipts, latencies, counting):
    segment_rows = [segment for row in rows for segment in row["segments"]]
    protected = [row for row in segment_rows if row["expected_status"] == "protected"]
    eligible = [row for row in segment_rows if row["expected_status"] == "eligible"]
    expected_drops = [row for row in eligible if row["expected_suggestion"] == "drop"]
    expected_retains = [row for row in eligible if row["expected_suggestion"] == "retain"]
    return {
        "cases": len(rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "reason_counts": dict(Counter(row["reason"] for row in rows)),
        "scored_cases": sum(row["status"] == "scored" for row in rows),
        "bypass_cases": sum(row["status"] == "bypass" for row in rows),
        "scorer_batches": counting.calls,
        "scorer_candidate_states": counting.candidate_states,
        "forwarded_unchanged": all(row["forwarded_unchanged"] for row in rows),
        "proposed_drops": sum(row["suggestion"] == "drop" for row in segment_rows),
        "protected_segments": len(protected),
        "protected_proposed_drops": sum(row["suggestion"] == "drop" for row in protected),
        "eligible_segments": len(eligible),
        "expected_drop_segments": len(expected_drops),
        "expected_drop_match": sum(row["suggestion"] == "drop" for row in expected_drops),
        "expected_retain_segments": len(expected_retains),
        "expected_retain_match": sum(row["suggestion"] == "retain" for row in expected_retains),
        "proposed_text_tokens_estimate": sum(
            row["token_counts"].get("proposed_text_tokens") or 0 for row in rows),
        "latency_ms": {"p50": percentile(latencies, 0.5), "p95": percentile(latencies, 0.95)},
        "all_receipts_content_free": all("raw_payload" not in receipt for receipt in receipts),
    }


def build_scorer(args):
    if args.scorer == "systemone":
        if not args.scorer_url:
            raise ValueError("--scorer-url is required for systemone")
        return DeadlineScorer(SystemOneHTTPScorer(
            args.scorer_url, endpoint=args.endpoint, model_id=args.model), args.timeout)
    if args.scorer == "reflex":
        if not args.model:
            raise ValueError("--model is required for reflex")
        return DeadlineScorer(InProcessScorer(ReflexContextScorer(
            args.model, device=args.device, revision=args.revision)), args.timeout)
    raise ValueError(f"unknown scorer {args.scorer!r}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--scorer", choices=("systemone", "reflex"), required=True)
    parser.add_argument("--scorer-url")
    parser.add_argument("--endpoint", default="/v1/systemone")
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--threshold", type=float, default=0.99)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    scorer = build_scorer(args)
    manifest, rows, receipts, latencies, counting = evaluate(
        args.manifest, scorer, args.threshold)
    report = {
        "schema_version": SCHEMA_VERSION,
        "mode": "shadow_only",
        "manifest": file_identity(args.manifest),
        "scorer": {"kind": args.scorer, "model": args.model, "endpoint": args.endpoint,
                   "timeout_s": args.timeout, "threshold": args.threshold},
        "summary": summarize(rows, receipts, latencies, counting),
        "cases": rows,
        "hardware": hardware_descriptor(args.device),
        "dependencies": dependency_versions(),
        "publication_scope": "aggregate metrics may be published; per-row provider/model receipts remain local",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                      allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": report["summary"]},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
