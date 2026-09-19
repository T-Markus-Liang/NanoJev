#!/usr/bin/env python3
"""Operate the local NanoJev service and append privacy-preserving usage events."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import socket
import urllib.error
import urllib.request
import uuid
from urllib.parse import urlsplit, urlunsplit


SKILL_DIR = Path(__file__).resolve().parents[1]
STAGES = ("development", "testing", "optimization", "deployment")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise RuntimeError("NanoJev local service redirects are forbidden")


def validate_local_url(url: str) -> None:
    parsed = urlsplit(url)
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "::1", "localhost"}
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment):
        raise ValueError("NanoJev requires an HTTP loopback URL without credentials/query/fragment")
    _ = parsed.port  # reject malformed/out-of-range ports before any network operation


def discover_project_root() -> Path:
    explicit = os.environ.get("NANOJEV_PROJECT_ROOT")
    if explicit:
        return Path(explicit)
    for candidate in (SKILL_DIR, *SKILL_DIR.parents):
        if (candidate / "scripts" / "serve_decisions.py").is_file():
            return candidate
    return Path.home() / "Documents" / "NanoJev"


DEFAULT_PROJECT_ROOT = discover_project_root()
DEFAULT_URL = os.environ.get("NANOJEV_URL", "http://127.0.0.1:8765").rstrip("/")
DEFAULT_CHECKPOINT = Path(os.environ.get(
    "NANOJEV_CHECKPOINT",
    str(DEFAULT_PROJECT_ROOT / "checkpoints/local_atomic_seed17/variants/local_atomic_seed17"),
))
DEFAULT_RUNTIME_DIR = Path(os.environ.get("NANOJEV_RUNTIME_DIR", str(Path.home() / ".codex/nanojev")))
DEFAULT_LOG = Path(os.environ.get("NANOJEV_LOG", str(DEFAULT_RUNTIME_DIR / "usage.jsonl")))


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def json_request(url: str, payload: object | None = None, timeout: float = 30.0) -> object:
    validate_local_url(url)
    # Local inference must not be affected by an unrelated HTTP proxy setting.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    if payload is None:
        request = urllib.request.Request(url, method="GET")
    else:
        request = urllib.request.Request(
            url, data=canonical_json(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
    with opener.open(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def health(url: str) -> dict:
    result = json_request(f"{url}/api/health", timeout=5)
    if not isinstance(result, dict) or result.get("ready") is not True or result.get("provider_calls") != 0:
        raise RuntimeError(f"NanoJev service is not ready: {result!r}")
    return result


def health_from_default_or_saved(url: str, runtime_dir: Path) -> dict:
    try:
        return health(url)
    except Exception as initial_error:
        saved_url = runtime_dir / "service.url"
        if saved_url.is_file():
            remembered = saved_url.read_text(encoding="utf-8").strip()
            if remembered:
                try:
                    result = health(remembered)
                    result["service_url"] = remembered
                    return result
                except Exception:
                    pass
        raise initial_error


def url_port(url: str) -> int:
    parsed = urlsplit(url)
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError("NanoJev URL must point to the local machine")
    return parsed.port or (443 if parsed.scheme == "https" else 80)


def port_is_open(url: str) -> bool:
    parsed = urlsplit(url)
    host = parsed.hostname or "127.0.0.1"
    try:
        with socket.create_connection((host, url_port(url)), timeout=0.2):
            return True
    except OSError:
        return False


def fallback_url(url: str) -> str:
    parsed = urlsplit(url)
    for port in range(8876, 8891):
        candidate = urlunsplit((parsed.scheme, f"{parsed.hostname or '127.0.0.1'}:{port}", parsed.path, parsed.query, parsed.fragment))
        if not port_is_open(candidate):
            return candidate
    raise RuntimeError("No free local NanoJev fallback port was found")


def service_command(project_root: Path, checkpoint: Path, url: str) -> list[str]:
    python = Path(os.environ.get("NANOJEV_PYTHON", str(project_root / ".venv/bin/python")))
    command = [
        str(python), str(project_root / "scripts/serve_decisions.py"),
        "--checkpoint-dir", str(checkpoint), "--web-root", str(project_root / "web"),
        "--host", "127.0.0.1", "--port", str(url_port(url)),
        "--device", os.environ.get("NANOJEV_DEVICE", "auto"),
        "--precision", os.environ.get("NANOJEV_PRECISION", "auto"),
    ]
    if os.environ.get("NANOJEV_DISABLE_NATIVE_TRITON", "") == "1":
        command.append("--disable-native-triton")
    return command


def ensure_service(url: str, project_root: Path, checkpoint: Path, runtime_dir: Path, startup_timeout: float = 180.0) -> dict:
    validate_local_url(url)
    try:
        return health(url)
    except Exception as initial_error:
        saved_url = runtime_dir / "service.url"
        if saved_url.is_file():
            remembered = saved_url.read_text(encoding="utf-8").strip()
            if remembered:
                try:
                    result = health(remembered)
                    result["service_url"] = remembered
                    return result
                except Exception:
                    pass
        if not (project_root / "scripts/serve_decisions.py").is_file():
            raise RuntimeError(f"NanoJev service is unavailable and project root is invalid: {project_root}") from initial_error
        if not checkpoint.is_dir():
            raise RuntimeError(f"NanoJev checkpoint does not exist: {checkpoint}") from initial_error
        service_url = fallback_url(url) if port_is_open(url) else url
        runtime_dir.mkdir(parents=True, exist_ok=True)
        log_path = Path(os.environ.get("NANOJEV_SERVICE_LOG", str(runtime_dir / "service.log")))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab") as log:
            process = subprocess.Popen(
                service_command(project_root, checkpoint, service_url), cwd=project_root,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                     "HF_HUB_DISABLE_TELEMETRY": "1"},
                start_new_session=True,
            )
        (runtime_dir / "service.pid").write_text(str(process.pid) + "\n", encoding="utf-8")
        (runtime_dir / "service.url").write_text(service_url + "\n", encoding="utf-8")
        deadline = time.monotonic() + startup_timeout
        last_error: Exception = initial_error
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"NanoJev service exited with code {process.returncode}; inspect {log_path}")
            try:
                result = health(service_url)
                result["service_url"] = service_url
                return result
            except Exception as error:
                last_error = error
                time.sleep(0.5)
        raise RuntimeError(f"NanoJev service did not become ready within {startup_timeout:.0f}s: {last_error}")


def load_payload(path: str | None, inline: str | None) -> dict:
    if inline is not None:
        raw = inline
    elif path in {None, "-"}:
        raw = sys.stdin.read()
    else:
        raw = Path(path).read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON input: {error}") from error
    if not isinstance(payload, dict) or set(payload) != {"states"}:
        raise ValueError('Input must be an object containing only a "states" array')
    return payload


def normalize_payload(payload: dict) -> tuple[dict, dict[tuple[str, str], dict]]:
    normalized = copy.deepcopy(payload)
    gates: dict[tuple[str, str], dict] = {}
    states = normalized.get("states")
    if not isinstance(states, list) or not states:
        raise ValueError("states must be a non-empty array")
    for state in states:
        if not isinstance(state, dict):
            raise ValueError("each state must be an object")
        state_id = state.get("id")
        questions = state.get("questions")
        if not isinstance(state_id, str) or not isinstance(questions, dict):
            raise ValueError("each state requires string id and object questions")
        for qid, question in questions.items():
            if not isinstance(question, dict):
                raise ValueError(f"{state_id}:{qid} must be an object")
            original_type = question.get("type")
            threshold = question.pop("abstain_below", None)
            if threshold is not None:
                if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
                    raise ValueError(f"{state_id}:{qid} abstain_below must be between 0 and 1")
            if original_type == "noul":
                question["type"] = "boolean"
            elif original_type not in {"boolean", "choice", "score"}:
                raise ValueError(f"{state_id}:{qid} has unsupported type: {original_type!r}")
            gates[(state_id, qid)] = {"type": original_type, "abstain_below": threshold}
    return normalized, gates


def confidence(answer: dict) -> float:
    probabilities = answer.get("probabilities", {})
    if not isinstance(probabilities, dict) or not probabilities:
        raise ValueError("Missing local probability distribution")
    values = list(probabilities.values())
    if (any(type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 1 for v in values)
            or not math.isclose(sum(values), 1.0, abs_tol=1e-5)):
        raise ValueError("Invalid local probability distribution")
    return max(values)


def apply_response_compatibility(result: dict, gates: dict[tuple[str, str], dict]) -> dict:
    output = copy.deepcopy(result)
    confidences: list[float] = []
    abstained = 0
    seen = set()
    for state in output.get("states", []):
        state_id = state.get("id")
        for qid, answer in state.get("answers", {}).items():
            key = (state_id, qid)
            if key in seen or key not in gates:
                raise ValueError("Unexpected/duplicate local answer")
            seen.add(key)
            gate = gates.get((state_id, qid), {})
            original_type = gate.get("type")
            if original_type == "noul":
                answer["type"] = "noul"
                answer["probability"] = answer.get("p_true")
            current_confidence = confidence(answer)
            confidences.append(current_confidence)
            threshold = gate.get("abstain_below")
            is_abstained = threshold is not None and current_confidence < threshold
            answer["confidence"] = current_confidence
            answer["abstained"] = is_abstained
            if is_abstained:
                answer["abstain_reason"] = "confidence_below_threshold"
                # Keep raw scores for analysis, but no executable selection survives abstention.
                answer["suggested_value"] = answer.get("value", answer.get("choice"))
                answer["value"] = None
                if "choice" in answer:
                    answer["choice"] = None
                abstained += 1
    if seen != set(gates):
        raise ValueError("Incomplete local response")
    output["decision_summary"] = {
        "questions": len(confidences),
        "confidence_min": min(confidences) if confidences else None,
        "confidence_max": max(confidences) if confidences else None,
        "confidence_mean": sum(confidences) / len(confidences) if confidences else None,
        "abstained": abstained,
    }
    return output


def checkpoint_identity(checkpoint: Path) -> dict:
    identity = {"directory": str(checkpoint.resolve())}
    config = checkpoint / "config.json"
    if config.is_file():
        identity["config_sha256"] = hashlib.sha256(config.read_bytes()).hexdigest()
        try:
            data = json.loads(config.read_text(encoding="utf-8"))
            identity.update({key: data.get(key) for key in ("model", "resolved_model_revision", "schema_version") if data.get(key) is not None})
        except (OSError, json.JSONDecodeError):
            pass
    return identity


def usage_record(payload: dict, result: dict, source: str, task_tag: str, checkpoint: Path, latency_ms: float, event_id: str) -> dict:
    questions = [q for state in payload["states"] for q in state["questions"].values()]
    candidate_counts = []
    question_types = []
    for question in questions:
        question_types.append(question["type"])
        candidate_counts.append(2 if question["type"] in {"boolean", "noul"} else len(question["criteria"]))
    summary = result.get("decision_summary", {})
    execution = result.get("execution", {})
    record = {
        "event_type": "decision", "event_id": event_id, "timestamp": utc_now(),
        "schema_version": "nanojev-usage-v1", "source": source, "task_tag": task_tag,
        "checkpoint": checkpoint_identity(checkpoint),
        "runtime": {"device": execution.get("device"), "precision": execution.get("precision")},
        "request": {
            "state_count": len(payload["states"]), "question_count": len(questions),
            "question_types": question_types,
            "candidate_count_min": min(candidate_counts) if candidate_counts else None,
            "candidate_count_max": max(candidate_counts) if candidate_counts else None,
            "candidate_count_mean": sum(candidate_counts) / len(candidate_counts) if candidate_counts else None,
            "input_sha256": hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest(),
        },
        "result": {
            "latency_ms": round(latency_ms, 3), "confidence_min": summary.get("confidence_min"),
            "confidence_max": summary.get("confidence_max"), "confidence_mean": summary.get("confidence_mean"),
            "abstained_count": summary.get("abstained", 0), "candidate_paths": execution.get("candidate_paths"),
            "forward_passes": execution.get("forward_passes"), "network_model_calls": execution.get("network_model_calls"),
            "inference_call_index": execution.get("inference_call_index"),
        },
    }
    if os.environ.get("NANOJEV_LOG_PAYLOADS") == "1":
        record["raw_payload"] = payload
    return record


def append_event(record: dict, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(canonical_json(record) + "\n")


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def summarize_log(log_path: Path) -> dict:
    decisions: list[dict] = []
    feedback: dict[str, list[str]] = {}
    if log_path.is_file():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event_type") == "decision":
                decisions.append(event)
            elif event.get("event_type") == "feedback":
                feedback.setdefault(event.get("decision_event_id"), []).append(event.get("label"))
    latencies = [float(e["result"]["latency_ms"]) for e in decisions if e.get("result", {}).get("latency_ms") is not None]
    confidences = [float(e["result"]["confidence_mean"]) for e in decisions if e.get("result", {}).get("confidence_mean") is not None]
    total_questions = sum(int(e.get("request", {}).get("question_count", 0)) for e in decisions)
    total_abstained = sum(int(e.get("result", {}).get("abstained_count", 0)) for e in decisions)
    feedback_counts: dict[str, int] = {}
    for labels in feedback.values():
        for label in labels:
            feedback_counts[label] = feedback_counts.get(label, 0) + 1
    by_type: dict[str, int] = {}
    for event in decisions:
        for typ in event.get("request", {}).get("question_types", []):
            by_type[typ] = by_type.get(typ, 0) + 1
    return {
        "schema_version": "nanojev-usage-summary-v1", "log": str(log_path),
        "decision_events": len(decisions), "question_count": total_questions,
        "question_types": by_type, "abstained_questions": total_abstained,
        "abstain_rate": total_abstained / total_questions if total_questions else None,
        "latency_ms": {"p50": percentile(latencies, 0.50), "p95": percentile(latencies, 0.95), "samples": len(latencies)},
        "confidence_mean": sum(confidences) / len(confidences) if confidences else None,
        "feedback_events": sum(len(labels) for labels in feedback.values()),
        "feedback_counts": feedback_counts,
        "feedback_coverage": sum(1 for event in decisions if event.get("event_id") in feedback) / len(decisions) if decisions else None,
    }


def command_health(args: argparse.Namespace) -> int:
    result = ensure_service(args.url, args.project_root, args.checkpoint, args.runtime_dir) if args.start else health_from_default_or_saved(args.url, args.runtime_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def command_decide(args: argparse.Namespace) -> int:
    payload = load_payload(args.input, args.json)
    normalized, gates = normalize_payload(payload)
    start = time.perf_counter()
    service = ensure_service(args.url, args.project_root, args.checkpoint, args.runtime_dir)
    service_url = service.get("service_url", args.url)
    result = json_request(f"{service_url}/api/evaluate", normalized, timeout=args.timeout)
    if not isinstance(result, dict):
        raise RuntimeError("NanoJev returned a non-object response")
    if result.get("execution", {}).get("network_model_calls") != 0:
        raise RuntimeError("Local-only inference evidence missing or remote model call reported")
    actual_checkpoint = result.get("checkpoint", {}).get("directory")
    if not actual_checkpoint or Path(actual_checkpoint).resolve() != args.checkpoint.resolve():
        raise RuntimeError("Running service checkpoint does not match requested checkpoint")
    result = apply_response_compatibility(result, gates)
    event_id = str(uuid.uuid4())
    elapsed_ms = (time.perf_counter() - start) * 1000
    record = usage_record(payload, result, args.source, args.task_tag, args.checkpoint, elapsed_ms, event_id)
    stage = getattr(args, "stage", None)
    if stage is not None:
        result["workflow"] = {"stage": stage, "advisory_only": True,
                              "requires_independent_verification": True, "authorizes_execution": False}
        record["workflow"] = result["workflow"]
    append_event(record, args.log)
    result["usage"] = {"event_id": event_id, "log": str(args.log), "service_url": service_url, "latency_ms": round(elapsed_ms, 3)}
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


def command_lifecycle(args: argparse.Namespace) -> int:
    candidates = json.loads(args.candidates)
    if (not isinstance(candidates, dict) or not 2 <= len(candidates) <= 32
            or any(not isinstance(k, str) or not k.strip() or not isinstance(v, str) or not v.strip()
                   for k, v in candidates.items())):
        raise ValueError("lifecycle candidates must be 2..32 named, nonempty descriptions")
    if not args.state.strip():
        raise ValueError("lifecycle state must contain observed facts")
    args.json = canonical_json({"states": [{"id": args.stage, "state": args.state, "questions": {
        "next_check": {"type": "choice", "instructions":
            "Select the most relevant next check based only on the observed facts. "
            "This is advisory, not permission to execute or a certification of correctness.",
            "criteria": candidates, "abstain_below": args.abstain_below}}}]})
    args.input, args.task_tag = None, args.stage
    return command_decide(args)


def command_feedback(args: argparse.Namespace) -> int:
    record = {
        "event_type": "feedback", "event_id": str(uuid.uuid4()), "decision_event_id": args.event_id,
        "timestamp": utc_now(), "schema_version": "nanojev-usage-v1", "label": args.label,
    }
    if args.note:
        record["note"] = " ".join(args.note.split())[:240]
    append_event(record, args.log)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    sub = parser.add_subparsers(dest="command", required=True)
    health_parser = sub.add_parser("health")
    health_parser.add_argument("--start", action="store_true")
    health_parser.set_defaults(handler=command_health)
    decide = sub.add_parser("decide")
    input_group = decide.add_mutually_exclusive_group()
    input_group.add_argument("--input", default="-", help="JSON file, or - for stdin")
    input_group.add_argument("--json", help="Inline JSON request")
    decide.add_argument("--source", default="codex")
    decide.add_argument("--task-tag", default="unspecified")
    decide.add_argument("--timeout", type=float, default=180.0)
    decide.set_defaults(handler=command_decide)
    lifecycle = sub.add_parser("lifecycle", help="Local advisory decision for each engineering phase")
    lifecycle.add_argument("--stage", choices=STAGES, required=True)
    lifecycle.add_argument("--state", required=True, help="Short observed facts, no secrets")
    lifecycle.add_argument("--candidates", required=True, help="JSON object of candidate IDs/descriptions")
    lifecycle.add_argument("--abstain-below", type=float, default=0.9)
    lifecycle.add_argument("--source", default="codex")
    lifecycle.add_argument("--timeout", type=float, default=30.0)
    lifecycle.set_defaults(handler=command_lifecycle)
    feedback = sub.add_parser("record-feedback")
    feedback.add_argument("--event-id", required=True)
    feedback.add_argument("--label", required=True, choices=["correct", "incorrect", "abstained", "fallback", "human_override"])
    feedback.add_argument("--note")
    feedback.set_defaults(handler=command_feedback)
    summary = sub.add_parser("summary")
    summary.set_defaults(handler=lambda args: (print(json.dumps(summarize_log(args.log), ensure_ascii=False, indent=2)), 0)[1])
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.url = args.url.rstrip("/")
    try:
        return args.handler(args)
    except (OSError, urllib.error.URLError, ValueError, RuntimeError) as error:
        print(f"nanojev-local-decider: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
