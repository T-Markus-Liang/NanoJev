#!/usr/bin/env python3
"""N1 contract-first benchmark harness for local NanoJev decision runs.

This harness is intentionally standard-library heavy. It imports the existing local
``DecisionPredictor`` lazily at runtime, so importing this module (for tests or for
argument inspection) never loads torch, weights, or the network.

Contract notes
--------------
* The numbers produced here are measurements of a local workload. The V3 roadmap
  targets (e.g. warm p95) are *targets*, not results this harness claims to meet.
* Any local token counts are local tokenizer counts, never provider billing. Only a
  provider's own usage report can bill.
* This harness does not enable production pruning. It only records evidence.
* ``network_model_calls`` must be exactly ``0``; anything else fails closed.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import sys
import time
from pathlib import Path


SCHEMA_VERSION = "nanojev-v3-benchmark-contract-v1"
RECEIPT_SCHEMA_VERSION = "nanojev-v3-receipt-v1"
CONTRACT_ID = "N1"

# Fail closed if any of these local checkpoint files is absent.
REQUIRED_CHECKPOINT_FILES = (
    "config.json",
    "best.safetensors",
    "backbone_config/config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
)

# Recorded when present, but not required for a valid checkpoint.
OPTIONAL_CHECKPOINT_FILES = (
    "tokenizer/chat_template.jinja",
    "tokenizer/special_tokens_map.json",
    "tokenizer/merges.txt",
    "tokenizer/vocab.json",
)

# Fields that are explicitly time-varying (or measurement noise) and are therefore
# excluded from deterministic report canonicalization.
TIME_VARYING_KEYS = frozenset({
    "generated_at",
    "timestamp_utc",
    "started_utc",
    "latency",
    "resources",
    "peak_rss_bytes",
    "load_ms",
    "total_elapsed_ms",
    "elapsed_ms",
    "questions_per_second",
    "tokens_per_second",
})

# Derived self-referential hashes that cannot be part of their own canonical form.
SELF_HASH_KEYS = frozenset({"canonical_sha256", "receipt_sha256"})


class ContractError(RuntimeError):
    """Raised when the N1 benchmark contract cannot be satisfied. Fails closed."""


# ---------------------------------------------------------------------------
# Canonicalization and hashing
# ---------------------------------------------------------------------------

def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def strip_time_varying(value):
    """Recursively drop explicitly time-varying keys from a report structure."""
    if isinstance(value, dict):
        return {key: strip_time_varying(item) for key, item in value.items()
                if key not in TIME_VARYING_KEYS}
    if isinstance(value, list):
        return [strip_time_varying(item) for item in value]
    return value


def canonical_report(report):
    """Return the deterministic canonical JSON string for a report."""
    projected = {key: value for key, value in report.items() if key not in SELF_HASH_KEYS}
    return canonical_json(strip_time_varying(projected))


def canonical_report_sha256(report):
    return sha256_bytes(canonical_report(report).encode("utf-8"))


# ---------------------------------------------------------------------------
# Latency statistics
# ---------------------------------------------------------------------------

def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize_latencies(values):
    values = list(values)
    if not values:
        return {"samples": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None,
                "min_ms": None, "max_ms": None, "mean_ms": None}
    return {
        "samples": len(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "min_ms": min(values),
        "max_ms": max(values),
        "mean_ms": math.fsum(values) / len(values),
    }


# ---------------------------------------------------------------------------
# Local artifact identities
# ---------------------------------------------------------------------------

def chat_template_identity(tokenizer_dir):
    """Content identity for the tokenizer chat template, or None when absent."""
    tokenizer_dir = Path(tokenizer_dir)
    jinja = tokenizer_dir / "chat_template.jinja"
    if jinja.is_file():
        raw = jinja.read_bytes()
        return {"source": "chat_template.jinja", "bytes": len(raw), "sha256": sha256_bytes(raw)}
    config_path = tokenizer_dir / "tokenizer_config.json"
    if not config_path.is_file():
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"tokenizer_config.json is not valid JSON: {exc}") from None
    template = config.get("chat_template") if isinstance(config, dict) else None
    if template is None:
        return None
    raw = (template if isinstance(template, str) else canonical_json(template)).encode("utf-8")
    return {"source": "tokenizer_config.json:chat_template",
            "bytes": len(raw), "sha256": sha256_bytes(raw)}


def checkpoint_identity(checkpoint_dir):
    """Record every required/optional checkpoint file plus the chat template."""
    root = Path(checkpoint_dir).expanduser().resolve()
    if not root.is_dir():
        raise ContractError(f"checkpoint directory does not exist: {root}")
    missing = [name for name in REQUIRED_CHECKPOINT_FILES if not (root / name).is_file()]
    if missing:
        raise ContractError(f"checkpoint is missing required files: {missing}")
    required = {name: file_identity(root / name) for name in REQUIRED_CHECKPOINT_FILES}
    optional = {name: file_identity(root / name) for name in OPTIONAL_CHECKPOINT_FILES
                if (root / name).is_file()}
    chat_template = chat_template_identity(root / "tokenizer")
    identity = {
        "directory": str(root),
        "required_files": required,
        "optional_files": optional,
        "tokenizer_chat_template": chat_template,
    }
    identity["sha256"] = sha256_bytes(canonical_json(identity).encode("utf-8"))
    return identity


def input_identity(input_path):
    """Record input jsonl file identities. Accepts a single file or a directory."""
    path = Path(input_path).expanduser().resolve()
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(candidate for candidate in path.glob("*.jsonl") if candidate.is_file())
        if not files:
            raise ContractError(f"input directory has no jsonl files: {path}")
    else:
        raise ContractError(f"input does not exist: {path}")
    identities = [file_identity(item) for item in files]
    identity = {
        "path": str(path),
        "files": identities,
        "sha256": sha256_bytes(canonical_json(identities).encode("utf-8")),
    }
    return identity


def load_input_rows(input_path):
    identity = input_identity(input_path)
    rows = []
    for entry in identity["files"]:
        for line in Path(entry["path"]).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(f"input row is not valid JSON: {exc}") from None
            if not isinstance(row, dict) or not {"id", "state", "questions"} <= set(row):
                raise ContractError("input rows must be objects with id, state and questions")
            rows.append(row)
    if not rows:
        raise ContractError("input contains no rows")
    return rows, identity


def build_payload(rows):
    return {"states": [{"id": row["id"], "state": row["state"], "questions": row["questions"]}
                       for row in rows]}


# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------

def validate_threads(threads):
    if threads is None:
        return None
    if type(threads) is not int or threads < 1:
        raise ContractError("threads must be a positive integer")
    return threads


def apply_thread_budget(threads):
    """Pin common math-library thread env vars before torch is imported."""
    threads = validate_threads(threads)
    if threads is None:
        return None
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                     "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = str(threads)
    return threads


def effective_thread_count():
    try:
        import torch  # noqa: PLC0415 - runtime-only import, keeps tests torch-free
    except ImportError:
        return None
    return int(torch.get_num_threads())


def validate_network_model_calls(execution, expected=0):
    """Fail closed unless the predictor reports exactly ``expected`` network calls."""
    if not isinstance(execution, dict):
        raise ContractError("predictor execution block is missing or malformed")
    value = execution.get("network_model_calls", "missing")
    if type(value) is not int or value != expected:
        raise ContractError(
            f"network_model_calls must be {expected} (got {value!r}); failing closed")
    return value


def load_predictor_class():
    """Import the existing local predictor only when a real run executes."""
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    module = importlib.import_module("predict_toy_decisions")
    return module.DecisionPredictor


# ---------------------------------------------------------------------------
# Environment reports
# ---------------------------------------------------------------------------

def physical_memory_bytes():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def dependency_versions(names=("torch", "transformers", "safetensors", "numpy")):
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def hardware_descriptor(device=None):
    descriptor = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "physical_memory_bytes": physical_memory_bytes(),
        "runtime_device": str(device) if device is not None else None,
    }
    descriptor["sha256"] = sha256_bytes(canonical_json(descriptor).encode("utf-8"))
    return descriptor


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------

def build_receipt(*, checkpoint_sha256, input_sha256, chat_template_sha256,
                  runtime, network_model_calls, output_digest, cold_samples, warm_samples):
    """Build a content-free receipt. It contains only hashes and configuration."""
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "checkpoint_sha256": checkpoint_sha256,
        "input_sha256": input_sha256,
        "tokenizer_chat_template_sha256": chat_template_sha256,
        "runtime": runtime,
        "network_model_calls": network_model_calls,
        "output_digest": output_digest,
        "cold_samples": cold_samples,
        "warm_samples": warm_samples,
        "goals_are_targets_not_results": True,
        "local_token_counts_are_not_provider_billing": True,
        "enables_production_pruning": False,
    }
    receipt["receipt_sha256"] = sha256_bytes(canonical_json(receipt).encode("utf-8"))
    return receipt


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(args):
    rows, input_manifest = load_input_rows(args.input)
    checkpoint_manifest = checkpoint_identity(args.checkpoint)

    if args.batch_states < 1:
        raise ContractError("batch-states must be a positive integer")
    if args.batch_questions < 0:
        raise ContractError("batch-questions must be nonnegative")
    if args.repeats < 1:
        raise ContractError("repeats must be a positive integer")
    if not isinstance(args.temperature, (int, float)) or isinstance(args.temperature, bool) \
            or not math.isfinite(args.temperature) or args.temperature <= 0:
        raise ContractError("temperature must be a finite positive number")

    apply_thread_budget(args.threads)
    predictor_class = load_predictor_class()
    load_started = time.perf_counter()
    engine = predictor_class(str(Path(args.checkpoint).expanduser().resolve()),
                             max_length=args.max_length, device_name=args.device,
                             precision=args.precision)
    load_ms = (time.perf_counter() - load_started) * 1000.0

    threads_effective = effective_thread_count()

    batches = [rows[index:index + args.batch_states]
               for index in range(0, len(rows), args.batch_states)]

    request_ms = []
    network_model_calls = []
    repeat_digests = []
    started = time.perf_counter()
    for _ in range(args.repeats):
        digest = hashlib.sha256()
        for group in batches:
            payload = build_payload(group)
            request_started = time.perf_counter()
            result = engine.predict(payload, batch_questions=args.batch_questions,
                                    temperature=args.temperature)
            request_ms.append((time.perf_counter() - request_started) * 1000.0)
            network_model_calls.append(
                validate_network_model_calls(result.get("execution")))
            digest.update(canonical_json(result["states"]).encode("utf-8"))
        repeat_digests.append(digest.hexdigest())
    total_elapsed_ms = (time.perf_counter() - started) * 1000.0

    cold_latency_ms = load_ms + request_ms[0]
    warm_request_ms = request_ms[1:]
    cold_summary = summarize_latencies([cold_latency_ms])
    warm_summary = summarize_latencies(warm_request_ms)

    output_digest = sha256_bytes("".join(repeat_digests).encode("utf-8"))
    deterministic = len(set(repeat_digests)) == 1

    runtime = {
        "device": str(engine.device),
        "precision": engine.precision,
        "threads_requested": args.threads,
        "threads_effective": threads_effective,
        "temperature": float(args.temperature),
        "batch_states": args.batch_states,
        "batch_questions": args.batch_questions,
        "max_length": args.max_length,
        "repeats": args.repeats,
    }

    dependencies = dependency_versions()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "input": input_manifest,
        "checkpoint": checkpoint_manifest,
        "benchmark_script": file_identity(Path(__file__).resolve()),
        "dependencies": {
            "versions": dependencies,
            "sha256": sha256_bytes(canonical_json(dependencies).encode("utf-8")),
        },
        "python": {"version": platform.python_version(),
                   "executable": str(Path(sys.executable).resolve())},
        "hardware": hardware_descriptor(engine.device),
    }
    manifest["manifest_sha256"] = sha256_bytes(canonical_json(manifest).encode("utf-8"))

    chat_template = checkpoint_manifest["tokenizer_chat_template"]
    receipt = build_receipt(
        checkpoint_sha256=checkpoint_manifest["sha256"],
        input_sha256=input_manifest["sha256"],
        chat_template_sha256=chat_template["sha256"] if chat_template else None,
        runtime=runtime,
        network_model_calls=sum(network_model_calls),
        output_digest=output_digest,
        cold_samples=cold_summary["samples"],
        warm_samples=warm_summary["samples"])

    report = {
        "schema_version": SCHEMA_VERSION,
        "contract": {
            "id": CONTRACT_ID,
            "network_model_calls_required": 0,
            "goals_are_targets_not_results": True,
            "local_token_counts_are_not_provider_billing": True,
            "enables_production_pruning": False,
        },
        "run": {"temperature": float(args.temperature), "repeats": args.repeats,
                "batch_states": args.batch_states, "batch_questions": args.batch_questions},
        "generated_at": int(time.time()),
        "manifest": manifest,
        "execution": runtime,
        "latency": {
            "load_ms": load_ms,
            "cold": cold_summary,
            "warm": warm_summary,
            "total_elapsed_ms": total_elapsed_ms,
        },
        "resources": {
            "peak_rss_bytes": peak_rss_bytes(),
            "physical_memory_bytes": manifest["hardware"]["physical_memory_bytes"],
            "energy_proxy": None,
        },
        "network_model_calls": sum(network_model_calls),
        "output_digest": output_digest,
        "determinism": {"repeats": repeat_digests, "stable": deterministic},
        "receipt": receipt,
        "receipt_sha256": receipt["receipt_sha256"],
        "gates": {
            "zero_network_model_calls": sum(network_model_calls) == 0,
            "manifest_complete": True,
            "warm_samples_present": warm_summary["samples"] >= 1,
            "deterministic_repeats": deterministic,
        },
    }
    report["canonical_sha256"] = canonical_report_sha256(report)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True,
                        help="jsonl file or directory of jsonl files")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps or cuda[:index]")
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16"), default="auto")
    parser.add_argument("--threads", type=int, default=None,
                        help="pin math-library threads (and torch) to this count")
    parser.add_argument("--batch-states", type=int, default=8)
    parser.add_argument("--batch-questions", type=int, default=0,
                        help="0 = all questions in one forward pass")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=2,
                        help="workload passes; first request is cold, the rest are warm")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run(args)
    except ContractError as exc:
        parser.error(str(exc))
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
