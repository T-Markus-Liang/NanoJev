#!/usr/bin/env python3
"""NanoJev -> JevBench adapter (v1), bound to the frozen adapter contract.

Implements roadmap task X1.  The adapter consumes canonical JevBench task
records (``jevbench/tasks.py`` at the pinned revision declared in
``research/jevbench_adapter_contract_v1.json``) and answers them with a local
NanoJev checkpoint through the pinned predictor
(``scripts/predict_toy_decisions.py:DecisionPredictor``), emitting one output
record per decision with the full probability distribution, per-decision
latency and a run manifest.

Hard rules enforced here (fail-closed):

* The contract file's ``contract_sha256`` is recomputed and checked before
  anything runs; a changed contract aborts the run.
* The adapter's own sha256 must match the contract's bound adapter hash.
* ``--mode synthetic`` accepts only records with ``split="synthetic"`` and
  ``provenance.synthetic=true`` — real benchmark rows cannot pass through it.
* ``--mode public_diagnostic`` accepts only already-downloaded
  ``split="public"`` task records and emits a diagnostic run receipt. It is
  not an official milestone and does not authorize promotion.
* ``--mode milestone`` and ``fetch_public_tasks`` remain the X5 code path and
  always raise in this revision: exactly one official public milestone run per
  frozen candidate is a separate authorization.
* The adapter itself performs no network access: callers provide a local JSONL.
  ``expected`` labels never enter the model-visible request.

Usage

    .venv/bin/python scripts/jevbench_adapter_v1.py \
        --contract research/jevbench_adapter_contract_v1.json \
        --checkpoint-dir checkpoints/domain_adaptation_v4_lora_seed18 \
        --input research/jevbench_fixtures_v1/items.jsonl \
        --mode synthetic \
        --output results/jevbench_adapter_synthetic_run_v1.json

    # validate only: contract hash + record schema + NanoJev payload mapping
    .venv/bin/python scripts/jevbench_adapter_v1.py \
        --contract research/jevbench_adapter_contract_v1.json \
        --input research/jevbench_fixtures_v1/items.jsonl \
        --mode synthetic --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_toy_decisions import (DecisionPredictor, prepare_examples,  # noqa: E402
                                   read_json, validate_request)

RUN_SCHEMA = "nanojev-jevbench-adapter-run-v1"
DECISION_SCHEMA = "nanojev-jevbench-decision-v1"
QUESTION_KEY = "decision"
QUESTION_TYPES = ("noul", "choice", "score")
SPLITS_SYNTHETIC = ("synthetic",)
SPLITS_PUBLIC_DIAGNOSTIC = ("public",)
SPLITS_MILESTONE = ("public", "private")
EXECUTION_MODES = ("serial", "bulk")
DEFAULT_BULK_MAX_QUESTIONS = 8
DEFAULT_BULK_MAX_LEAF_TOKENS = 32768
ABSTENTION_THRESHOLD = 0.9  # reported only; never suppresses a benchmark answer


class ContractBindingError(ValueError):
    """Contract hash, bound artifact hash, or contract flag violation."""


class BenchmarkFetchGated(RuntimeError):
    """The public benchmark fetch/milestone path is gated by roadmap X5."""


def digest_value(value):
    """Canonical-JSON sha256, same convention as the engineering contract."""
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_contract(contract_path):
    """Load the contract and verify its frozen hash.  Fail-closed."""
    contract_path = Path(contract_path).resolve(strict=True)
    contract = read_json(contract_path)
    if contract.get("schema_version") != "nanojev-jevbench-adapter-contract-v1":
        raise ContractBindingError(
            f"unexpected contract schema_version: "
            f"{contract.get('schema_version')!r}")
    declared = contract.get("contract_sha256")
    computed = digest_value({k: v for k, v in contract.items()
                             if k != "contract_sha256"})
    if not isinstance(declared, str) or declared != computed:
        raise ContractBindingError(
            f"contract_sha256 mismatch: declared {declared!r} "
            f"computed {computed!r} — contract file changed; refusing to run")
    quarantine = contract.get("data_quarantine", {})
    for flag in ("training_allowed", "calibration_fit_allowed",
                 "per_item_failure_tuning_allowed"):
        if quarantine.get(flag) is not False:
            raise ContractBindingError(
                f"contract data_quarantine.{flag} must be false")
    bound = contract.get("bound_artifacts", {}).get("adapter", {})
    expected_self = bound.get("sha256")
    if isinstance(expected_self, str):
        actual_self = file_hash(Path(__file__).resolve())
        if actual_self != expected_self:
            raise ContractBindingError(
                f"adapter self-hash mismatch: contract binds {expected_self} "
                f"but this file is {actual_self}")
    return contract, contract_path


def fetch_public_tasks(*_args, **_kwargs):
    """GATED — X5 only.  Downloading or reading public JevBench task rows
    requires a frozen candidate, the recorded contract hash and milestone
    authorization.  This revision never fetches."""
    raise BenchmarkFetchGated(
        "public JevBench task rows are gated: the adapter contract hash must "
        "be recorded first, the candidate must be frozen by the internal "
        "selection gate (J-E1), and exactly one public milestone run is "
        "allowed per frozen candidate (roadmap X5). This revision performs "
        "no network access and reads no benchmark rows.")


def validate_task_record(record, allowed_splits):
    """Canonical JevBench record validation (mirrors jevbench.tasks at the
    pinned revision) plus the contract's label-consistency rules.  Raises on
    any violation — malformed input fails closed."""
    if not isinstance(record, dict):
        raise ValueError("task record must be an object")
    required = {"id", "family", "state", "question", "labels", "split"}
    missing = required - set(record)
    if missing:
        raise ValueError(f"task record missing keys: {sorted(missing)}")
    if not isinstance(record["id"], str) or not record["id"].strip():
        raise ValueError("task id must be a nonempty string")
    if not isinstance(record["state"], (str, dict, list)) or not record["state"]:
        raise ValueError(f"{record['id']}: state must be nonempty "
                         "string/object/array")
    q = record["question"]
    if not isinstance(q, dict) or q.get("type") not in QUESTION_TYPES:
        raise ValueError(f"{record['id']}: bad question type")
    if not isinstance(q.get("instructions"), str) or not q["instructions"].strip():
        raise ValueError(f"{record['id']}: instructions must be nonempty")
    if set(q) - {"type", "instructions", "criteria"}:
        raise ValueError(f"{record['id']}: unsupported question fields")
    if record["split"] not in allowed_splits:
        raise ValueError(f"{record['id']}: split {record['split']!r} not "
                         f"allowed in this mode {sorted(allowed_splits)}")
    labels = record["labels"]
    if not isinstance(labels, list) or not labels \
            or not all(isinstance(x, str) and x for x in labels):
        raise ValueError(f"{record['id']}: labels must be a nonempty list "
                         "of strings")
    if len(set(labels)) != len(labels):
        raise ValueError(f"{record['id']}: duplicate labels")
    criteria = q.get("criteria")
    if q["type"] == "noul":
        if labels != ["no", "yes"]:
            raise ValueError(f"{record['id']}: noul labels must be "
                             "['no','yes']")
        if criteria is not None and (
                not isinstance(criteria, dict)
                or not set(criteria) <= {"false", "true"}
                or not all(isinstance(v, str) and v.strip()
                           for v in criteria.values())):
            raise ValueError(f"{record['id']}: noul criteria must be an "
                             "object with a subset of keys false/true")
    elif q["type"] == "choice":
        if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 255:
            raise ValueError(f"{record['id']}: choice criteria must be a "
                             "2-255 entry object")
        if set(labels) != set(criteria):
            raise ValueError(f"{record['id']}: choice labels must equal "
                             "criteria keys")
    else:  # score
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
            raise ValueError(f"{record['id']}: score criteria must be a "
                             "2-10 entry ordered list")
        if labels != [str(i) for i in range(len(criteria))]:
            raise ValueError(f"{record['id']}: score labels must be level "
                             "index strings 0..k-1")
    expected = record.get("expected")
    if expected is not None:
        if q["type"] == "score":
            if not isinstance(expected, int) or isinstance(expected, bool) \
                    or str(expected) not in labels:
                raise ValueError(f"{record['id']}: score expected must be an "
                                 "int level index present in labels")
        elif expected not in labels:
            raise ValueError(f"{record['id']}: expected {expected!r} not in "
                             "labels")
    if isinstance(record["state"], dict):
        for banned in ("expected", "label", "ground_truth", "answer_key"):
            if banned in record["state"]:
                raise ValueError(f"{record['id']}: state contains banned key "
                                 f"{banned!r}")
    if record["split"] == "synthetic":
        prov = record.get("provenance")
        if not isinstance(prov, dict) or prov.get("synthetic") is not True:
            raise ValueError(f"{record['id']}: synthetic split requires "
                             "provenance.synthetic=true")
    return record


def task_to_nanojev_payload(task):
    """Map one JevBench task record to a NanoJev predict request.

    noul   -> NanoJev boolean (criteria subset of false/true passed through)
    choice -> NanoJev choice  (criteria dict passed through verbatim)
    score  -> NanoJev score   (criteria list passed through verbatim)

    Only {id, state, questions} enter the request: labels/expected/group/
    provenance never reach the model."""
    q = task["question"]
    nq = {"type": {"noul": "boolean"}.get(q["type"], q["type"]),
          "instructions": q["instructions"]}
    if q.get("criteria") is not None:
        nq["criteria"] = q["criteria"]
    payload = {"states": [{"id": task["id"], "state": task["state"],
                           "questions": {QUESTION_KEY: nq}}]}
    validate_request(payload)  # pinned NanoJev request contract, fail-closed
    return payload


def map_probs(task, answer):
    """Project a NanoJev answer onto the task's exact label set."""
    nanojev_probs = answer.get("probabilities")
    if not isinstance(nanojev_probs, dict):
        raise ValueError("NanoJev answer missing probabilities")
    if task["question"]["type"] == "noul":
        mapped = {"no": nanojev_probs.get("false"),
                  "yes": nanojev_probs.get("true")}
    else:
        mapped = {label: nanojev_probs.get(label) for label in task["labels"]}
    probs = {}
    total = 0.0
    for key, value in mapped.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) \
                or not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"{task['id']}: invalid probability for label "
                             f"{key!r}")
        probs[str(key)] = float(value)
        total += float(value)
    if abs(total - 1.0) > 1e-5:
        raise ValueError(f"{task['id']}: mapped probabilities sum to {total}")
    if set(probs) != set(task["labels"]):
        raise ValueError(f"{task['id']}: mapped probs do not cover the exact "
                         "label set")
    return probs


def argmax_label(probs):
    best, best_p = None, -1.0
    for key in sorted(probs):
        if probs[key] > best_p:
            best, best_p = key, probs[key]
    return best


def percentile(sorted_vals, q):
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = q * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (rank - lo)


def load_tasks(input_path, allowed_splits):
    path = Path(input_path).resolve(strict=True)
    tasks, seen = [], set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line, object_pairs_hook=_unique_object,
                                parse_constant=_reject_nonfinite)
            validate_task_record(record, allowed_splits)
        except (ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path.name}:{lineno}: {exc}") from None
        if record["id"] in seen:
            raise ValueError(f"{path.name}:{lineno}: duplicate task id "
                             f"{record['id']!r}")
        seen.add(record["id"])
        tasks.append(record)
    if not tasks:
        raise ValueError(f"{path}: no task records")
    return tasks, path


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"duplicate JSON key: {key}")
        obj[key] = value
    return obj


def _reject_nonfinite(value):
    raise ValueError(f"non-finite JSON number: {value}")


def marker_task_groups(tasks, payloads, tokenizer, max_length,
                       marker_placement="before",
                       max_questions=DEFAULT_BULK_MAX_QUESTIONS,
                       max_leaf_tokens=DEFAULT_BULK_MAX_LEAF_TOKENS):
    from jfast_modernbert_decision_v1 import prepare_payload
    items = []
    for task in tasks:
        ex = prepare_payload(payloads[task["id"]], tokenizer, max_length,
                             marker_placement)[0]
        items.append({"task": task, "leaf_tokens": len(ex["input_ids"]),
                      "leaf_count": len(ex["labels"]),
                      "max_leaf_tokens": len(ex["input_ids"])})
    items.sort(key=lambda x: (x["max_leaf_tokens"], x["task"]["id"]))
    groups = []
    current = []
    width = 0
    for item in items:
        projected = max(width, item["max_leaf_tokens"]) * (len(current) + 1)
        if current and (len(current) >= max_questions or
                        projected > max_leaf_tokens):
            groups.append(current)
            current = []
            width = 0
        current.append(item)
        width = max(width, item["max_leaf_tokens"])
    if current:
        groups.append(current)
    stats = {"groups": len(groups), "questions": len(items),
             "candidate_paths": sum(item["leaf_count"] for item in items),
             "candidate_leaf_tokens": sum(item["leaf_tokens"] for item in items),
             "max_questions": max_questions,
             "max_leaf_tokens_budget": max_leaf_tokens}
    return groups, stats


def bulk_task_groups(tasks, payloads, tokenizer, max_length,
                     max_questions=DEFAULT_BULK_MAX_QUESTIONS,
                     max_leaf_tokens=DEFAULT_BULK_MAX_LEAF_TOKENS):
    """Group tasks by similar padded leaf width for bulk accuracy diagnostics.

    Groups are formed after sorting by each task's largest candidate-leaf length.
    The budget approximates padded transformer work as
    ``max_leaf_tokens_in_group * total_candidate_leaves``. This is deliberately
    simple and deterministic; it is an execution strategy, not a model change.
    """
    if type(max_questions) is not int or max_questions <= 0:
        raise ValueError("bulk max_questions must be a positive int")
    if type(max_leaf_tokens) is not int or max_leaf_tokens <= 0:
        raise ValueError("bulk max_leaf_tokens must be a positive int")
    indexed = []
    for index, task in enumerate(tasks):
        examples = prepare_examples(payloads[task["id"]], tokenizer, max_length)
        leaf_count = sum(len(ex["leaf_tokens"]) for ex in examples)
        leaf_tokens = sum(len(leaf) for ex in examples for leaf in ex["leaf_tokens"])
        max_leaf = max(len(leaf) for ex in examples for leaf in ex["leaf_tokens"])
        indexed.append({"index": index, "task": task, "leaf_count": leaf_count,
                        "leaf_tokens": leaf_tokens, "max_leaf": max_leaf})
    indexed.sort(key=lambda item: (item["max_leaf"], item["leaf_tokens"], item["index"]))
    groups = []
    current = []
    current_max = 0
    current_leaves = 0
    for item in indexed:
        estimated = max(current_max, item["max_leaf"]) * (current_leaves + item["leaf_count"])
        if current and (len(current) >= max_questions or estimated > max_leaf_tokens):
            groups.append(current)
            current = []
            current_max = 0
            current_leaves = 0
        current.append(item)
        current_max = max(current_max, item["max_leaf"])
        current_leaves += item["leaf_count"]
    if current:
        groups.append(current)
    stats = {"groups": len(groups),
             "questions": len(tasks),
             "candidate_paths": sum(item["leaf_count"] for item in indexed),
             "candidate_leaf_tokens": sum(item["leaf_tokens"] for item in indexed),
             "max_questions": max_questions,
             "max_leaf_tokens_budget": max_leaf_tokens}
    return groups, stats


def _decision_record(task, leaf_tokens, leaf_count, answer, latency_s, latency_scope):
    probs = map_probs(task, answer)
    record = {"schema_version": DECISION_SCHEMA, "task_id": task["id"],
              "family": task["family"], "split": task["split"],
              "group": task.get("group"),
              "question_type": task["question"]["type"], "ok": True,
              "probs_source": "native", "model": "nanojev-local",
              "expected": task.get("expected"), "nanojev_type": answer["type"],
              "probs": probs, "predicted": argmax_label(probs),
              "confidence": max(probs.values()),
              "correct": (None if task.get("expected") is None else
                          argmax_label(probs) == str(task["expected"])),
              "latency_s": latency_s,
              "latency_scope": latency_scope,
              "abstention": {"threshold": ABSTENTION_THRESHOLD,
                             "would_abstain": max(probs.values())
                             < ABSTENTION_THRESHOLD},
              "usage": {"input_tokens": leaf_tokens,
                        "output_tokens": 0,
                        "candidate_paths": leaf_count,
                        "accounting": "candidate-leaf tokens actually scored; "
                                      "local weights, no provider tariff"}}
    return record


def run(input_path, checkpoint_dir=None, contract_path=None, mode="synthetic",
        dry_run=False, device="auto", precision="auto", max_length=None,
        temperature=1.0, execution="serial", bulk_max_questions=DEFAULT_BULK_MAX_QUESTIONS,
        bulk_max_leaf_tokens=DEFAULT_BULK_MAX_LEAF_TOKENS,
        shared_prefix=False, shared_prefix_row_tokens=None,
        shared_prefix_max_rows=16, shared_prefix_max_attn_positions=4_000_000):
    """Run the adapter.  Returns the run-manifest dict (the receipt)."""
    if contract_path is None:
        raise ContractBindingError("a --contract file is required")
    contract, resolved_contract = load_contract(contract_path)
    contract_sha = contract["contract_sha256"]
    if execution not in EXECUTION_MODES:
        raise ValueError(f"unknown execution mode {execution!r}")

    if mode == "synthetic":
        allowed_splits = SPLITS_SYNTHETIC
    elif mode == "public_diagnostic":
        allowed_splits = SPLITS_PUBLIC_DIAGNOSTIC
    elif mode == "milestone":
        raise BenchmarkFetchGated(
            "milestone mode is the X5 path: it requires a frozen candidate "
            "selected by the internal gate and exactly-once public-run "
            "authorization. Not permitted by this revision.")
    else:
        raise ValueError(f"unknown mode {mode!r}")

    tasks, resolved_input = load_tasks(input_path, allowed_splits)
    input_sha = file_hash(resolved_input)

    manifest = {
        "schema_version": RUN_SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": mode,
        "dry_run": bool(dry_run),
        "contract": {"path": str(resolved_contract),
                     "contract_sha256": contract_sha,
                     "jevbench": contract.get("jevbench_source", {})},
        "input": {"path": str(resolved_input), "sha256": input_sha,
                  "tasks": len(tasks)},
        "quarantine": {"evaluation_only": True, "training_allowed": False,
                       "calibration_fit_allowed": False,
                       "per_item_failure_tuning_allowed": False,
                       "benchmark_rows_consumed": len(tasks)
                       if mode == "public_diagnostic" else 0,
                       "network_calls": 0},
        "abstention": {"threshold": ABSTENTION_THRESHOLD,
                       "applied": False,
                       "note": "full distribution always emitted; abstention "
                               "is reported per decision but never suppresses "
                               "a benchmark answer"},
    }

    # Structural pass over every task before any model load: mapping and the
    # pinned NanoJev request validator must accept each record.
    payloads = {}
    for task in tasks:
        payloads[task["id"]] = task_to_nanojev_payload(task)

    if dry_run:
        manifest["status"] = "dry_run_validated"
        manifest["decisions"] = [
            {"task_id": t["id"], "family": t["family"], "split": t["split"],
             "question_type": t["question"]["type"],
             "nanojev_type": next(iter(
                 payloads[t["id"]]["states"][0]["questions"].values()))["type"],
             "labels": t["labels"], "ok": None,
             "note": "validated only; no inference performed"}
            for t in tasks]
        return manifest

    if checkpoint_dir is None:
        raise ValueError("--checkpoint-dir is required unless --dry-run")

    checkpoint_config = json.loads(
        (Path(checkpoint_dir) / "config.json").read_text(encoding="utf-8"))
    is_marker_head = checkpoint_config.get("set_head") == "marker"
    load_start = time.perf_counter()
    if is_marker_head:
        from predict_jfast_modernbert import JFastDecisionPredictor
        predictor = JFastDecisionPredictor(checkpoint_dir, device_name=device,
                                           precision=precision)
    else:
        predictor = DecisionPredictor(checkpoint_dir, max_length=max_length,
                                      device_name=device, precision=precision)
    model_load_s = time.perf_counter() - load_start

    decisions_by_id = {}
    run_start = time.perf_counter()
    forward_passes = 0
    bulk_stats = None
    if execution == "serial":
        for task in tasks:
            payload = payloads[task["id"]]
            # Token accounting uses the pinned renderer on the same payload; it
            # counts the candidate-leaf tokens actually scored.
            if is_marker_head:
                from jfast_modernbert_decision_v1 import prepare_payload
                marker_ex = prepare_payload(
                    payload, predictor.tokenizer, predictor.limit,
                    predictor.run_config.get("marker_placement", "before"))[0]
                leaf_tokens = len(marker_ex["input_ids"])
                leaf_count = len(marker_ex["labels"])
            else:
                examples = prepare_examples(payload, predictor.tokenizer,
                                            predictor.limit)
                leaf_tokens = sum(len(leaf) for ex in examples
                                  for leaf in ex["leaf_tokens"])
                leaf_count = sum(len(ex["leaf_tokens"]) for ex in examples)
            t0 = time.perf_counter()
            record = {"schema_version": DECISION_SCHEMA, "task_id": task["id"],
                      "family": task["family"], "split": task["split"],
                      "group": task.get("group"),
                      "question_type": task["question"]["type"], "ok": False,
                      "probs_source": "native", "model": "nanojev-local",
                      "expected": task.get("expected")}
            try:
                if is_marker_head:
                    out = predictor.predict(payload, batch_questions=0,
                                            temperature=temperature)
                else:
                    out = predictor.predict(
                        payload, temperature=temperature,
                        shared_prefix=shared_prefix,
                        shared_prefix_row_tokens=shared_prefix_row_tokens
                        or predictor.limit,
                        shared_prefix_max_rows=shared_prefix_max_rows,
                        shared_prefix_max_attn_positions=
                        shared_prefix_max_attn_positions)
                forward_passes += out["execution"]["forward_passes"]
                answer = out["states"][0]["answers"][QUESTION_KEY]
                record = _decision_record(task, leaf_tokens, leaf_count,
                                          answer, time.perf_counter() - t0,
                                          "serial_decision")
            except Exception as exc:  # noqa: BLE001 - failed decision is a record
                record["latency_s"] = time.perf_counter() - t0
                record["latency_scope"] = "serial_decision"
                record["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            decisions_by_id[task["id"]] = record
    else:
        if is_marker_head:
            groups, bulk_stats = marker_task_groups(
                tasks, payloads, predictor.tokenizer, predictor.limit,
                marker_placement=predictor.run_config.get(
                    "marker_placement", "before"),
                max_questions=bulk_max_questions,
                max_leaf_tokens=bulk_max_leaf_tokens)
        else:
            groups, bulk_stats = bulk_task_groups(
                tasks, payloads, predictor.tokenizer, predictor.limit,
                max_questions=bulk_max_questions,
                max_leaf_tokens=bulk_max_leaf_tokens)
        for group_index, group in enumerate(groups):
            group_payload = {"states": [payloads[item["task"]["id"]]["states"][0]
                                        for item in group]}
            t0 = time.perf_counter()
            try:
                if is_marker_head:
                    out = predictor.predict(group_payload, batch_questions=0,
                                            temperature=temperature)
                else:
                    out = predictor.predict(
                        group_payload, batch_questions=0,
                        temperature=temperature,
                        shared_prefix=shared_prefix,
                        shared_prefix_row_tokens=shared_prefix_row_tokens
                        or predictor.limit,
                        shared_prefix_max_rows=shared_prefix_max_rows,
                        shared_prefix_max_attn_positions=
                        shared_prefix_max_attn_positions)
                group_wall_s = time.perf_counter() - t0
                forward_passes += out["execution"]["forward_passes"]
                answers = {state["id"]: state["answers"][QUESTION_KEY]
                           for state in out["states"]}
                for item in group:
                    task = item["task"]
                    if task["id"] not in answers:
                        raise ValueError(f"missing bulk answer for {task['id']}")
                    record = _decision_record(task, item["leaf_tokens"],
                                              item["leaf_count"],
                                              answers[task["id"]], None,
                                              "bulk_group")
                    record["bulk"] = {"group_index": group_index,
                                      "group_questions": len(group),
                                      "group_wall_s": group_wall_s,
                                      "group_forward_passes":
                                          out["execution"]["forward_passes"]}
                    decisions_by_id[task["id"]] = record
            except Exception as exc:  # noqa: BLE001 - failed group is a record
                for item in group:
                    task = item["task"]
                    decisions_by_id[task["id"]] = {
                        "schema_version": DECISION_SCHEMA,
                        "task_id": task["id"], "family": task["family"],
                        "split": task["split"], "group": task.get("group"),
                        "question_type": task["question"]["type"],
                        "ok": False, "probs_source": "native",
                        "model": "nanojev-local", "expected": task.get("expected"),
                        "latency_s": None, "latency_scope": "bulk_group",
                        "bulk": {"group_index": group_index,
                                 "group_questions": len(group)},
                        "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    decisions = [decisions_by_id[task["id"]] for task in tasks]
    timed_wall_s = time.perf_counter() - run_start

    ok_n = sum(1 for d in decisions if d["ok"])
    measured = [d for d in decisions if d["ok"] and d["expected"] is not None]
    if execution == "serial":
        latencies = sorted(d["latency_s"] for d in decisions)
        timing = {"scope": "per-decision latency_s covers request build, "
                           "tokenization, forward pass(es) and softmax; "
                           "checkpoint/tokenizer load and fixture IO are "
                           "excluded and reported separately",
                  "decisions": len(decisions),
                  "wall_s": timed_wall_s,
                  "latency_s": {"min": latencies[0], "max": latencies[-1],
                                "mean": statistics.fmean(latencies),
                                "p50": percentile(latencies, 0.5),
                                "p95": percentile(latencies, 0.95)}}
    else:
        timing = {"scope": "bulk accuracy diagnostic; groups are scored in one "
                           "forward each and per-decision latency is not claimed",
                  "decisions": len(decisions),
                  "wall_s": timed_wall_s,
                  "latency_s": None,
                  "bulk": bulk_stats,
                  "forward_passes": forward_passes}
    manifest.update({
        "status": "completed" if ok_n == len(decisions) else
                  "completed_with_failures",
        "checkpoint": {"directory": str(predictor.root),
                       "config_sha256": file_hash(
                           predictor.root / "config.json"),
                       "base_model": predictor.run_config.get("model"),
                       "set_head": predictor.run_config.get("set_head")},
        "execution": {"device": str(predictor.device),
                      "precision": predictor.precision,
                      "parameter_storage": "float32",
                      "max_length": predictor.limit,
                      "temperature": float(temperature),
                      "mode": execution,
                      "forward_passes": forward_passes,
                      "shared_prefix": True if is_marker_head else bool(shared_prefix),
                      "shared_prefix_row_tokens":
                          predictor.limit if is_marker_head else
                          (shared_prefix_row_tokens or predictor.limit),
                      "shared_prefix_max_rows": shared_prefix_max_rows,
                      "shared_prefix_max_attn_positions":
                          shared_prefix_max_attn_positions,
                      "model_load_s": model_load_s,
                      "model_load_counted_in_latency": False},
        "timing": timing,
        "summary": {"decisions": len(decisions), "ok": ok_n,
                    "failed": len(decisions) - ok_n,
                    "measured": len(measured),
                    "correct": sum(1 for d in measured if d["correct"]),
                    "accuracy_on_measured": (
                        sum(1 for d in measured if d["correct"])
                        / len(measured) if measured else None),
                    "would_abstain_at_0_9": sum(
                        1 for d in decisions
                        if d["ok"] and d["abstention"]["would_abstain"])},
        "decisions": decisions,
    })
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--input", type=Path, required=True,
                        help="JSONL of JevBench-schema task records")
    parser.add_argument("--mode",
                        choices=["synthetic", "public_diagnostic", "milestone"],
                        default="synthetic")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate contract, records and mapping only; "
                             "no model load, no inference")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16"],
                        default="auto")
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--execution", choices=list(EXECUTION_MODES),
                        default="serial",
                        help="serial preserves per-decision latency; bulk groups tasks for accuracy diagnostics")
    parser.add_argument("--bulk-max-questions", type=int,
                        default=DEFAULT_BULK_MAX_QUESTIONS)
    parser.add_argument("--bulk-max-leaf-tokens", type=int,
                        default=DEFAULT_BULK_MAX_LEAF_TOKENS)
    parser.add_argument("--shared-prefix", action="store_true",
                        help="pack candidate suffixes behind a shared prefix after owned-parity validation")
    parser.add_argument("--shared-prefix-row-tokens", type=int)
    parser.add_argument("--shared-prefix-max-rows", type=int, default=16)
    parser.add_argument("--shared-prefix-max-attn-positions", type=int,
                        default=4_000_000)
    args = parser.parse_args()
    try:
        manifest = run(args.input, checkpoint_dir=args.checkpoint_dir,
                       contract_path=args.contract, mode=args.mode,
                       dry_run=args.dry_run, device=args.device,
                       precision=args.precision, max_length=args.max_length,
                       temperature=args.temperature, execution=args.execution,
                       bulk_max_questions=args.bulk_max_questions,
                       bulk_max_leaf_tokens=args.bulk_max_leaf_tokens,
                       shared_prefix=args.shared_prefix,
                       shared_prefix_row_tokens=args.shared_prefix_row_tokens,
                       shared_prefix_max_rows=args.shared_prefix_max_rows,
                       shared_prefix_max_attn_positions=
                       args.shared_prefix_max_attn_positions)
        text = json.dumps(manifest, ensure_ascii=False, indent=2,
                          allow_nan=False) + "\n"
        if args.output:
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("x", encoding="utf-8") as stream:
                stream.write(text)
            print(json.dumps({"status": manifest["status"],
                              "output": str(destination),
                              "decisions": manifest["input"]["tasks"]},
                             ensure_ascii=False))
        else:
            print(text, end="")
        return 0
    except (ContractBindingError, BenchmarkFetchGated, ValueError, OSError,
            KeyError, TypeError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "error", "error": type(exc).__name__,
                          "message": str(exc)}, ensure_ascii=False),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
