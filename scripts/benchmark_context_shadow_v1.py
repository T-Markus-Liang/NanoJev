#!/usr/bin/env python3
"""Deterministic guardrail stress and synthetic real-checkpoint shadow integration checks."""

import argparse
from collections import Counter
import json
from http.server import HTTPServer
from pathlib import Path
import random
import threading
import time

from benchmark_nanojev_v2 import dependency_versions, file_identity, hardware_descriptor, percentile
from context_gate_local import LoggedScorer, LoopbackPredictor, skill_helper
from context_gate_v1 import FORMATS, fingerprint, parse_segments, shadow_request
from predict_toy_decisions import DecisionPredictor
from serve_decisions import server_class


SCENARIOS = (
    ("test_command", "What is the test command for the current repository?",
     "The current repository uses uv run pytest tests/ for validation.", "An unrelated earlier task described a restaurant menu."),
    ("order_total", "What is the total for the current order?",
     "The current order contains two items priced at 35 each.", "An unrelated archived warehouse note discussed an old shipping address."),
    ("chinese", "请告诉我当前项目使用的端口。",
     "当前项目的本地端口是 8123。", "一份无关的历史记录描述了上周的天气。"),
    ("risk_limit", "Does the requested order of 120 units exceed the current position limit?",
     "The current maximum allowed position is 100 units.", "An unrelated historical report discussed a different account's old inventory."),
)


def envelope(wire, task, history):
    messages = [{"role": "assistant", "content": history}, {"role": "user", "content": task}]
    if wire == "openai_chat":
        body = {"model": "synthetic-main-model", "messages": [{"role": "system", "content": "Preserve all user constraints."}] + messages}
        pointer = "/messages/1/content"
    elif wire == "openai_responses":
        body = {"model": "synthetic-main-model", "instructions": "Preserve all user constraints.", "input": messages}
        pointer = "/input/0/content"
    else:
        body = {"model": "synthetic-main-model", "system": "Preserve all user constraints.", "messages": messages, "max_tokens": 64}
        pointer = "/messages/0/content"
    return body, pointer


def deterministic_scores(payload, value=0.999):
    return {"states": [{"id": s["id"], "answers": {"irrelevant": {"type": "boolean",
            "probabilities": {"false": 1-value, "true": value}}}} for s in payload["states"]]}


def stress(trials=1000, seed=17):
    if not __debug__:
        raise ValueError("verification requires Python assertions enabled")
    rng = random.Random(seed)
    counts, total = Counter(), 0
    for wire in sorted(FORMATS):
        for index in range(trials):
            secret = f"NEVER_LOG_{rng.getrandbits(128):032x}"
            body, pointer = envelope(wire, "Preserve this user's intent " + secret, "Optional unrelated context " + secret)
            sidecar = {"segments": {pointer: {"eligible": True}}}
            scorer = deterministic_scores
            case = index % 10
            if case == 1:
                sidecar["segments"][pointer]["safety"] = True
            elif case == 2:
                scorer = lambda p: deterministic_scores(p, 0.5)
            elif case == 3:
                def scorer(p):
                    raise TimeoutError(secret)
            elif case == 4:
                scorer = lambda p: {"states": []}
            elif case == 5:
                sidecar["segments"]["/does-not-exist"] = {"eligible": True}
            elif case == 6:
                key = "input" if wire == "openai_responses" else "messages"
                body[key][-1]["content"] = [{"type": "image_url", "image_url": {"url": secret}}]
            elif case == 8:
                key = "input" if wire == "openai_responses" else "messages"
                sidecar["segments"][f"/{key}/{len(body[key])-1}/content"] = {"depends_on": [pointer]}
            elif case == 9:
                body["unrecognized_semantic_field"] = secret
            raw = json.dumps(body, ensure_ascii=False, indent=index % 3).encode("utf-8")
            if case == 7:
                raw = b'{"messages":[],"messages":[]}'
            output, receipt = shadow_request(raw, wire, sidecar, scorer)
            assert output is raw and output == raw
            assert receipt["actual_removed_tokens"] == receipt["actual_removed_segments"] == 0
            assert secret not in json.dumps(receipt)
            drops = [segment for segment in receipt["segments"] if segment["suggestion"] == "drop"]
            assert [segment["pointer"] for segment in drops] == ([pointer] if case == 0 else [])
            assert all(segment["role"] == "assistant" for segment in drops)
            counts[f"{wire}/{case}/{receipt['reason']}"] += 1
            total += 1
    return {"checks": total, "seed": seed, "cases_per_format": trials,
            "failures": 0, "scope": "deterministic stub invariants, not learned relevance quality",
            "case_counts": dict(sorted(counts.items()))}


def real_model(checkpoint, usage_log, receipt_log):
    if usage_log.exists() or receipt_log.exists():
        raise ValueError("use fresh audit paths so this run has an unambiguous event count")
    before = time.perf_counter()
    engine = DecisionPredictor(checkpoint, device_name="mps", precision="fp32")
    load_ms = (time.perf_counter() - before) * 1000
    handler = server_class(engine, Path(__file__).resolve().parents[1] / "web")
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        predict = LoopbackPredictor(f"http://127.0.0.1:{server.server_port}", timeout=10)
        return real_trials(engine, predict, checkpoint, usage_log, receipt_log, load_ms)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        if thread.is_alive():
            raise RuntimeError("temporary inference server did not shut down")


def real_trials(engine, predict, checkpoint, usage_log, receipt_log, load_ms):
    scorer = LoggedScorer(predict, checkpoint, usage_log)
    fixtures, rows, latencies = [], [], []
    for wire in sorted(FORMATS):
        for name, task, required, irrelevant in SCENARIOS:
            for label, history in (("retain", required), ("drop", irrelevant)):
                body, pointer = envelope(wire, task, history)
                raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
                sidecar = {"segments": {pointer: {"eligible": True}}}
                fixtures.append({"format": wire, "body": body, "sidecar": sidecar, "expected": label})
                output, receipt = shadow_request(raw, wire, sidecar, scorer,
                    token_counter=lambda text: len(engine.tokenizer.encode(text, add_special_tokens=False)),
                    tokenizer_id=file_identity(checkpoint / "tokenizer/tokenizer.json"))
                assert output == raw
                assert receipt["actual_removed_tokens"] == 0
                skill_helper().append_event(receipt, receipt_log)
                segment = next(s for s in receipt["segments"] if s["pointer"] == pointer)
                rows.append({"format": wire, "scenario": name, "expected": label,
                             "suggestion": segment["suggestion"], "p_irrelevant": segment["p_irrelevant"],
                             "status": receipt["status"], "reason": receipt["reason"],
                             "event_id": receipt["event_id"], "scorer_event_id": receipt.get("scorer_event_id"),
                             "token_counts": receipt["token_counts"]})
                latencies.append(receipt["latency_ms"])
    events = [json.loads(line) for line in usage_log.read_text().splitlines()]
    assert len(events) == len(rows)
    assert all("raw_payload" not in event for event in events)
    assert {event["event_id"] for event in events} == {row["scorer_event_id"] for row in rows}
    assert all(row["p_irrelevant"] is not None for row in rows), "an integration failure is not model abstention"
    return {"scope": "24 constructed fixtures from 8 underlying scenarios; three dialects, one checkpoint, no main-model quality evaluation",
            "transport": "temporary loopback HTTP server /api/evaluate, using the real handler and checkpoint; no proxy inheritance",
            "rows": rows, "fixture_sha256": fingerprint(fixtures), "model_load_ms": load_ms,
            "latency_ms": {"p50": percentile(latencies, 0.5), "p95": percentile(latencies, 0.95)},
            "proposed_drops": sum(row["suggestion"] == "drop" for row in rows),
            "required_context_proposed_drops": sum(row["suggestion"] == "drop" and row["expected"] == "retain" for row in rows),
            "actual_saved_tokens": 0, "status_counts": dict(Counter(row["reason"] for row in rows)),
            "usage": file_identity(usage_log), "shadow_receipts": file_identity(receipt_log),
            "checkpoint_files": [file_identity(checkpoint / name) for name in (
                "config.json", "best.safetensors", "backbone_config/config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json")],
            "hardware": hardware_descriptor(engine.device), "dependencies": dependency_versions()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--usage-log", type=Path, required=True)
    parser.add_argument("--receipt-log", type=Path, required=True)
    parser.add_argument("--stress-trials", type=int, default=1000)
    args = parser.parse_args()
    if args.stress_trials < 10:
        parser.error("at least ten stress trials per format are required")
    report = {"schema_version": "nanojev-context-shadow-benchmark-v1", "stress": stress(args.stress_trials),
              "real_model": real_model(args.checkpoint, args.usage_log, args.receipt_log),
              "code": [file_identity(Path(__file__).with_name(name)) for name in (
                  "context_gate_v1.py", "context_gate_local.py", "benchmark_context_shadow_v1.py",
                  "predict_toy_decisions.py", "train_toy_decisions.py", "serve_decisions.py")],
              "helper": file_identity(Path(__file__).resolve().parents[1] / "integrations/codex-skill/nanojev-local-decider/scripts/nanojev_skill.py")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "stress": report["stress"]["checks"],
                      "real_requests": len(report["real_model"]["rows"]), "actual_saved_tokens": 0}))


if __name__ == "__main__":
    main()
