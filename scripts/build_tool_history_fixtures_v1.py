#!/usr/bin/env python3
"""Deterministic builder for the A4 tool-history shadow fixtures.

Materializes synthetic tool transcripts declared in the frozen manifest
``research/tool_history_fixture_manifest_v1.json`` and validates the declared
protected/eligible contract against the existing byte-preserving shadow core
``scripts/context_gate_v1.py``. The core is imported read-only and is never
modified or forked.

Shadow-only boundaries enforced here:

* no active context deletion, no provider proxy, no network access;
* no production request path is touched; output goes only to a caller-chosen
  directory that must be empty;
* fixtures are self-authored synthetic transcripts (no real user data, no real
  credentials, no real tool output);
* tools are never re-executed; the dependency contract only labels what may be
  proposed for removal in a shadow receipt.

``--self-test`` validates manifest integrity without writing any output.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from context_gate_v1 import Bypass, FORMATS, parse_segments, shadow_request, serialized


MANIFEST_SCHEMA = "nanojev-tool-history-fixture-manifest-v1"
CASE_SCHEMA = "nanojev-tool-history-fixture-case-v1"
INDEX_SCHEMA = "nanojev-tool-history-fixture-index-v1"
SOURCE_GROUP_SCHEMA = "nanojev-tool-history-source-group-v1"
TAG_PLACEHOLDER = "@SYNTH_TAG@"
DEFAULT_SEED = 20260919
DEFAULT_MANIFEST = Path(__file__).resolve().parent.parent / "research" / "tool_history_fixture_manifest_v1.json"

REQUIRED_KINDS = ("success", "error", "parallel_calls", "duplicate_ids", "orphan_result", "pending_call",
                  "non_repeatable_result", "mutable_file_read", "stale_output", "user_correction",
                  "secrets_credentials")

KEEP_BOTH = "keep_both"
KEEP_BOUNDED = "keep_call_bounded_result"
REMOVE_PAIR = "remove_pair_atomically"
OUTCOMES = (KEEP_BOTH, KEEP_BOUNDED, REMOVE_PAIR)

# Reasons that mean "retained for trust, not for a score".
TRUST_REASONS = {"protected_structure", "caller_protected", "required_dependency", "not_explicitly_eligible"}
# Reasons a scoring candidate can carry in a receipt.
MODEL_REASONS = {"model_retain", "high_irrelevance_score"}
# Flags on a declared call/result pair that forbid a narrower outcome.
BLOCKING_FLAGS = ("non_repeatable", "credential_bearing", "contains_error", "stale_evidence",
                  "correction_dependent", "volatile_source")
# Flags that only forbid removal of the whole pair, not a bounded result.
WEAK_BLOCKING_FLAGS = ("volatile_source",)

STUB_CHECKPOINT = {"model": "synthetic-tool-history-stub", "revision": "tool-history-fixture-v1"}


def deterministic_scorer(payload):
    """Fixed high-irrelevance stub. It never inspects or replays any tool."""
    return {"checkpoint": dict(STUB_CHECKPOINT), "states": [
        {"id": state["id"], "answers": {"irrelevant": {"type": "boolean",
                                                       "probabilities": {"false": 0.001, "true": 0.999}}}}
        for state in payload["states"]]}


def source_group_id(kind, source_world):
    payload = {"schema": SOURCE_GROUP_SCHEMA, "kind": kind, "source_world": source_world}
    return hashlib.sha256(serialized(payload).encode("utf-8")).hexdigest()


def synthetic_tag(case_id, seed):
    material = f"nanojev-tool-history:{seed}:{case_id}".encode("utf-8")
    return "tg-" + hashlib.sha256(material).hexdigest()[:12]


def substitute(value, tag):
    """Replace the frozen placeholder with the seed-derived synthetic tag."""
    if isinstance(value, str):
        return value.replace(TAG_PLACEHOLDER, tag)
    if isinstance(value, list):
        return [substitute(item, tag) for item in value]
    if isinstance(value, dict):
        return {key: substitute(item, tag) for key, item in value.items()}
    return value


def encode_request(body):
    """Realistic non-canonical JSON bytes; byte preservation is asserted later."""
    return ("\n" + json.dumps(body, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def pretty(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def load_manifest(path=DEFAULT_MANIFEST):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def apply_probe(body, probe):
    """Apply the declared minimal ambiguity repair used only for verification."""
    repaired = deepcopy(body)
    for operation in probe["ops"]:
        target, path = repaired, operation["path"]
        for key in path[:-1]:
            target = target[key]
        if operation["op"] == "set":
            target[path[-1]] = deepcopy(operation["value"])
        elif operation["op"] == "append":
            target[path[-1]].append(deepcopy(operation["value"]))
        else:
            raise ValueError(f"unknown probe op {operation['op']!r}")
    return repaired


def linkage_pointers(segments):
    """Structural tool linkage, independent of the declared manifest labels."""
    calls, results = set(), set()
    for segment in segments:
        value, pointer = segment.value, segment.pointer
        if isinstance(value, dict) and value.get("type") == "tool_use":
            calls.add(pointer)
        elif isinstance(value, dict) and value.get("type") == "tool_result":
            results.add(pointer)
        elif pointer.endswith("/tool_calls"):
            calls.add(pointer)
        elif pointer.endswith("/tool_call_id"):
            results.add(pointer)
        elif segment.role == "tool":
            results.add(pointer)
    return calls, results


def _message_index(pointer):
    parts = pointer.strip("/").split("/")
    for position, part in enumerate(parts[:-1]):
        if part in {"messages", "input"} and parts[position + 1].isdigit():
            return int(parts[position + 1])
    return None


def validate_manifest(manifest, seed=None):
    """Return a list of integrity errors; an empty list means the manifest is sound."""
    errors = []
    seed = manifest.get("seed", DEFAULT_SEED) if seed is None else seed

    def fail(message):
        errors.append(message)

    if not isinstance(manifest, dict):
        return ["manifest must be a JSON object"]
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        fail(f"schema_version must be {MANIFEST_SCHEMA!r}")
    if type(manifest.get("seed")) is not int:
        fail("seed must be an integer")

    contract = manifest.get("gate_contract", {})
    if contract.get("core") != "scripts/context_gate_v1.py":
        fail("gate_contract.core must name scripts/context_gate_v1.py")
    if contract.get("entrypoint") != "shadow_request":
        fail("gate_contract.entrypoint must be shadow_request")
    if contract.get("threshold") != 0.99:
        fail("gate_contract.threshold must stay frozen at 0.99")
    if contract.get("threshold_frozen") is not True:
        fail("gate_contract.threshold_frozen must be true")
    if contract.get("active_filtering") is not False:
        fail("gate_contract.active_filtering must be false")
    if tuple(contract.get("outcomes", ())) != OUTCOMES:
        fail(f"gate_contract.outcomes must be {list(OUTCOMES)}")

    provenance = manifest.get("provenance", {})
    for flag in ("contains_real_user_data", "contains_real_credentials", "contains_real_tool_output",
                 "reuses_frozen_relevance_corpus"):
        if provenance.get(flag) is not False:
            fail(f"provenance.{flag} must be false")
    if provenance.get("source") != "self_authored_synthetic_tool_transcript":
        fail("provenance.source must declare self-authored synthetic transcripts")

    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        return errors + ["cases must be a non-empty list"]
    kinds = [case.get("kind") for case in cases if isinstance(case, dict)]
    if sorted(kinds) != sorted(REQUIRED_KINDS):
        fail(f"case kinds must cover exactly {list(REQUIRED_KINDS)}")
    if len(set(kinds)) != len(kinds):
        fail("case kinds must be unique")
    if len(cases) != len(REQUIRED_KINDS):
        fail("manifest must contain exactly one case per required kind")

    groups = manifest.get("source_groups")
    declared_groups = []
    if not isinstance(groups, list) or len(groups) != len(cases):
        fail("source_groups must declare exactly one group per case")
        groups = []
    else:
        for group in groups:
            declared_groups.append(group.get("group_id"))
            if group.get("isolation_unit") != "single_fixture":
                fail(f"source group {group.get('fixture_id')!r} must be isolated to a single fixture")
    entity_tokens = []
    seen_case_ids = set()

    for case in cases:
        if not isinstance(case, dict):
            fail("each case must be an object")
            continue
        case_id, kind = case.get("case_id"), case.get("kind")
        label = f"case {case_id!r}"
        if case_id != kind:
            fail(f"{label}: case_id must equal kind")
        if case_id in seen_case_ids:
            fail(f"{label}: duplicate case_id")
        seen_case_ids.add(case_id)
        for key in ("description", "wire_format", "source_world", "source_group_id", "pairing_resolved",
                    "body", "sidecar", "segments", "pairing", "expected_gate", "notes", "leak_tokens"):
            if key not in case:
                fail(f"{label}: missing key {key!r}")
        if not isinstance(case.get("leak_tokens"), list) or not case["leak_tokens"] or any(
                not isinstance(token, str) or not token for token in case.get("leak_tokens", [])):
            fail(f"{label}: leak_tokens must be a non-empty list of non-empty strings")
        if case.get("wire_format") not in FORMATS:
            fail(f"{label}: wire_format must be one of {sorted(FORMATS)}")

        world = case.get("source_world")
        if not isinstance(world, dict) or not isinstance(world.get("entity"), str) or not world["entity"]:
            fail(f"{label}: source_world.entity must be a non-empty string")
        else:
            entity_tokens.extend((case_id, token) for key, token in sorted(world.items())
                                 if key.startswith("entity") and isinstance(token, str) and token)
        if case.get("source_group_id") != source_group_id(kind, world):
            fail(f"{label}: source_group_id must derive from the declared synthetic world")

        body, sidecar = case.get("body"), case.get("sidecar")
        if not isinstance(body, dict):
            fail(f"{label}: body must be an object")
            body = {}
        if not isinstance(sidecar, dict) or set(sidecar) - {"segments"}:
            fail(f"{label}: sidecar may only contain 'segments'")
            sidecar = {"segments": {}}

        segments = case.get("segments")
        if not isinstance(segments, list) or not segments:
            fail(f"{label}: segments must declare every segment")
            segments = []
        pointers, declared = [], {}
        for segment in segments:
            if not isinstance(segment, dict):
                fail(f"{label}: segment entries must be objects")
                continue
            pointer = segment.get("pointer")
            if not isinstance(pointer, str) or not pointer.startswith("/"):
                fail(f"{label}: invalid segment pointer {pointer!r}")
                continue
            if pointer in declared:
                fail(f"{label}: duplicate declared pointer {pointer!r}")
            declared[pointer] = segment
            pointers.append(pointer)
            status, suggestion = segment.get("expected_status"), segment.get("expected_suggestion")
            reason = segment.get("expected_reason_code")
            if status not in {"protected", "eligible"}:
                fail(f"{label}: {pointer} expected_status must be protected|eligible")
            if suggestion not in {"retain", "drop"}:
                fail(f"{label}: {pointer} expected_suggestion must be retain|drop")
            if status == "protected":
                if suggestion != "retain":
                    fail(f"{label}: {pointer} protected segments must retain")
                if reason not in TRUST_REASONS | {"model_retain"}:
                    fail(f"{label}: {pointer} protected reason {reason!r} is not a trust reason")
            elif status == "eligible" and reason not in MODEL_REASONS:
                fail(f"{label}: {pointer} eligible reason {reason!r} is not a scoring reason")
            if not isinstance(segment.get("role"), str) or not segment["role"]:
                fail(f"{label}: {pointer} must declare a role")

        gate = case.get("expected_gate")
        if not isinstance(gate, dict):
            fail(f"{label}: expected_gate must be an object")
            gate = {}
        if gate.get("status") not in {"scored", "bypass"}:
            fail(f"{label}: expected_gate.status must be scored|bypass")
        if not isinstance(gate.get("reason"), str) or not gate["reason"]:
            fail(f"{label}: expected_gate.reason must be a non-empty string")
        if not isinstance(gate.get("segments_observed"), bool):
            fail(f"{label}: expected_gate.segments_observed must be a boolean")
        resolved = case.get("pairing_resolved")
        if not isinstance(resolved, bool):
            fail(f"{label}: pairing_resolved must be a boolean")
        if resolved and gate.get("status") != "scored":
            fail(f"{label}: resolved pairing must be scored")
        if not resolved and gate.get("status") != "bypass":
            fail(f"{label}: unresolved pairing must bypass analysis")
        if not resolved and not isinstance(case.get("ambiguity_probe"), dict):
            fail(f"{label}: unresolved pairing needs an ambiguity_probe")
        if gate.get("all_retain") is not (gate.get("status") == "bypass"):
            fail(f"{label}: expected_gate.all_retain must match the gate status")
        if gate.get("status") == "scored" and not any(s.get("expected_suggestion") == "drop" for s in segments):
            fail(f"{label}: a scored case must declare at least one droppable candidate")
        if gate.get("status") == "bypass" and any(s.get("expected_suggestion") == "drop" for s in segments):
            fail(f"{label}: a bypass case cannot propose a drop")

        pairing = case.get("pairing")
        if not isinstance(pairing, list):
            fail(f"{label}: pairing must be a list")
            pairing = []
        pair_calls, pair_results, pair_ids = set(), set(), set()
        for pair in pairing:
            if not isinstance(pair, dict):
                fail(f"{label}: pairing entries must be objects")
                continue
            pair_id = pair.get("pair_id")
            if not isinstance(pair_id, str) or not pair_id or pair_id in pair_ids:
                fail(f"{label}: pair_id must be a unique non-empty string")
            pair_ids.add(pair_id)
            call, results = pair.get("call_segment"), pair.get("result_segments")
            if call not in declared:
                fail(f"{label}: pair {pair_id!r} call_segment must be a declared segment")
            elif call in pair_calls:
                fail(f"{label}: call segment {call!r} appears in two pairs")
            else:
                pair_calls.add(call)
            if not isinstance(results, list) or any(result not in declared for result in results):
                fail(f"{label}: pair {pair_id!r} result_segments must be declared segments")
                results = []
            for result in results:
                if result in pair_results:
                    fail(f"{label}: result segment {result!r} appears in two pairs")
                pair_results.add(result)
            if pair.get("atomic") is not True:
                fail(f"{label}: pair {pair_id!r} must be atomic")
            allowed = pair.get("allowed_outcomes")
            if not isinstance(allowed, list) or set(allowed) - set(OUTCOMES):
                fail(f"{label}: pair {pair_id!r} allowed_outcomes must be a subset of {list(OUTCOMES)}")
                allowed = []
            if KEEP_BOTH not in allowed:
                fail(f"{label}: pair {pair_id!r} must always allow {KEEP_BOTH}")
            if not resolved and list(allowed) != [KEEP_BOTH]:
                fail(f"{label}: unresolved pair {pair_id!r} must allow only {KEEP_BOTH}")
            for flag in BLOCKING_FLAGS:
                if type(pair.get(flag)) is not bool:
                    fail(f"{label}: pair {pair_id!r} flag {flag!r} must be a boolean")
            hard = [flag for flag in BLOCKING_FLAGS if flag not in WEAK_BLOCKING_FLAGS and pair.get(flag)]
            if hard and KEEP_BOUNDED in allowed:
                fail(f"{label}: pair {pair_id!r} forbids a bounded result under {hard}")
            if hard and REMOVE_PAIR in allowed:
                fail(f"{label}: pair {pair_id!r} forbids atomic removal under {hard}")
            if pair.get("volatile_source") and REMOVE_PAIR in allowed:
                fail(f"{label}: pair {pair_id!r} must not remove a volatile-source pair")
            if not pair.get("repeatable", False) and REMOVE_PAIR in allowed:
                fail(f"{label}: pair {pair_id!r} must be repeatable to allow removal")

        try:
            if not _verify_case_against_core(case, declared, gate, resolved, pairing, seed, fail):
                continue
        except Exception as error:  # defensive: a malformed case must not crash the validator
            fail(f"{label}: core verification raised {type(error).__name__}")
            continue

    if len(set(kinds)) == len(kinds) and len(groups) == len(cases):
        for group, case in zip(groups, cases):
            if group.get("fixture_id") != case.get("case_id"):
                fail(f"source group order must match case order ({group.get('fixture_id')!r} != {case.get('case_id')!r})")
            world = case.get("source_world", {})
            expected_tokens = [world[key] for key in sorted(world)
                               if key.startswith("entity") and isinstance(world[key], str)]
            if group.get("entity_tokens") != expected_tokens:
                fail(f"source group for {case.get('case_id')!r} must declare the case entity tokens")
            if group.get("group_id") != case.get("source_group_id"):
                fail(f"source group id for {case.get('case_id')!r} must match the case")
    if declared_groups and len(set(declared_groups)) != len(declared_groups):
        fail("source groups must be unique per fixture")
    for index, (case_id, token) in enumerate(entity_tokens):
        for other_id, other in entity_tokens[index + 1:]:
            if token == other:
                fail(f"entity token {token!r} is shared by {case_id!r} and {other_id!r}")
            elif token in other or other in token:
                fail(f"entity token {token!r} overlaps {other!r}; byte-level isolation would be ambiguous")
    return errors


def _verify_case_against_core(case, declared, gate, resolved, pairing, seed, fail):
    """Cross-check the declared contract against the real shadow core."""
    label = f"case {case.get('case_id')!r}"
    try:
        tag = synthetic_tag(case["case_id"], seed)
        body = substitute(case["body"], tag)
        sidecar = substitute(case["sidecar"], tag)
        raw = encode_request(body)
        parsed = parse_segments(raw, case["wire_format"])
        parsed_error = None
    except Bypass as error:
        parsed, parsed_error = [], str(error)
    except (KeyError, TypeError, ValueError) as error:
        fail(f"{label}: fixture could not be encoded ({error})")
        return False

    pointers = set(declared)
    if gate.get("segments_observed"):
        if parsed_error is not None:
            fail(f"{label}: declared observable segments but the core bypassed with {parsed_error!r}")
            return False
        observed = parsed
        if {segment.pointer for segment in observed} != pointers:
            fail(f"{label}: declared segment pointers do not match parse_segments output")
            return False
        for segment in observed:
            if declared[segment.pointer].get("role") != segment.role:
                fail(f"{label}: declared role for {segment.pointer!r} does not match the core")
            structural = declared[segment.pointer].get("expected_reason_code") == "protected_structure"
            if segment.protected != structural:
                fail(f"{label}: {segment.pointer!r} protected flag disagrees with the declared reason")
    else:
        if parsed_error != gate.get("reason"):
            fail(f"{label}: expected bypass reason {gate.get('reason')!r} but the core returned {parsed_error!r}")
            return False
        probe = case.get("ambiguity_probe")
        try:
            repaired = apply_probe(body, substitute(probe, tag))
            observed = parse_segments(encode_request(repaired), case["wire_format"])
        except (Bypass, KeyError, TypeError, ValueError) as error:
            fail(f"{label}: ambiguity probe did not resolve linkage ({error})")
            return False
        observed_pointers = {segment.pointer for segment in observed}
        missing = pointers - observed_pointers
        if missing:
            fail(f"{label}: ambiguity probe lost declared segments {sorted(missing)}")
        for segment in observed:
            if segment.pointer in declared and declared[segment.pointer].get("role") != segment.role:
                fail(f"{label}: declared role for {segment.pointer!r} does not match the probed core output")
        original_length = len(case["body"].get("messages", case["body"].get("input", [])))
        for pointer in observed_pointers - pointers:
            index = _message_index(pointer)
            if index is None or index < original_length:
                fail(f"{label}: ambiguity probe added an unexpected segment {pointer!r}")

    calls, results = linkage_pointers(observed)
    observed_pointers = {segment.pointer for segment in observed}
    calls &= pointers
    results &= pointers
    pair_calls = {pair["call_segment"] for pair in pairing}
    pair_results = {result for pair in pairing for result in pair["result_segments"]}
    if calls != pair_calls:
        fail(f"{label}: declared call segments {sorted(pair_calls)} do not cover {sorted(calls)}")
    if results != pair_results:
        fail(f"{label}: declared result segments {sorted(pair_results)} do not cover {sorted(results)}")
    if not pair_calls <= observed_pointers or not pair_results <= observed_pointers:
        fail(f"{label}: pairing references segments the core never enumerates")

    output, receipt = shadow_request(raw, case["wire_format"], sidecar, deterministic_scorer)
    if output is not raw:
        fail(f"{label}: shadow_request must return the identical bytes object")
    if receipt.get("forwarded_sha256") != receipt.get("request_sha256"):
        fail(f"{label}: forwarded bytes must be byte-identical")
    if receipt.get("status") != gate.get("status"):
        fail(f"{label}: gate status {receipt.get('status')!r} != declared {gate.get('status')!r}")
    if receipt.get("reason") != gate.get("reason"):
        fail(f"{label}: gate reason {receipt.get('reason')!r} != declared {gate.get('reason')!r}")
    if receipt.get("actual_removed_segments") != 0 or receipt.get("actual_removed_tokens") != 0:
        fail(f"{label}: shadow receipts must remove nothing")
    if receipt.get("policy_threshold") != 0.99:
        fail(f"{label}: shadow receipt must report the frozen 0.99 threshold")
    if gate.get("segments_observed"):
        seen = {segment["pointer"] for segment in receipt.get("segments", [])}
        if seen != pointers:
            fail(f"{label}: receipt segments {sorted(seen)} != declared {sorted(pointers)}")
        for segment in receipt.get("segments", []):
            declared_segment = declared[segment["pointer"]]
            if segment.get("suggestion") != declared_segment.get("expected_suggestion"):
                fail(f"{label}: {segment['pointer']} suggestion {segment.get('suggestion')!r} != declared "
                     f"{declared_segment.get('expected_suggestion')!r}")
            if segment.get("reason") != declared_segment.get("expected_reason_code"):
                fail(f"{label}: {segment['pointer']} reason {segment.get('reason')!r} != declared "
                     f"{declared_segment.get('expected_reason_code')!r}")
            if segment.get("applied") is not False:
                fail(f"{label}: {segment['pointer']} must never be applied")
    elif receipt.get("segments"):
        fail(f"{label}: a bypassed request must not produce a shadow plan")

    rendered = json.dumps(receipt, ensure_ascii=False)
    if tag in rendered:
        fail(f"{label}: receipt leaked the synthetic tag")
    for token in case["leak_tokens"]:
        if token in rendered:
            fail(f"{label}: receipt leaked declared token {token!r}")
    return True


def materialize_case(case, seed):
    """Return the exact bytes for one fixture directory."""
    tag = synthetic_tag(case["case_id"], seed)
    expected = {
        "schema_version": CASE_SCHEMA,
        "case_id": case["case_id"],
        "kind": case["kind"],
        "wire_format": case["wire_format"],
        "description": case["description"],
        "source_group_id": case["source_group_id"],
        "source_world": case["source_world"],
        "synthetic_tag": tag,
        "pairing_resolved": case["pairing_resolved"],
        "pairing": case["pairing"],
        "segments": case["segments"],
        "expected_gate": case["expected_gate"],
        "leak_tokens": case["leak_tokens"],
        "notes": case["notes"],
        "provenance": {
            "source": "self_authored_synthetic_tool_transcript",
            "license": "CC0-1.0",
            "contains_real_user_data": False,
            "contains_real_credentials": False,
            "contains_real_tool_output": False,
        },
        "shadow_only": True,
    }
    return {
        "request.json": encode_request(substitute(case["body"], tag)),
        "sidecar.json": pretty(substitute(case["sidecar"], tag)),
        "expected.json": pretty(substitute(expected, tag)),
    }


def build(output_dir, manifest_path=DEFAULT_MANIFEST, seed=None):
    """Validate the manifest and materialize fixtures; refuse a non-empty target."""
    manifest = load_manifest(manifest_path)
    seed = manifest.get("seed", DEFAULT_SEED) if seed is None else seed
    errors = validate_manifest(manifest, seed)
    if errors:
        raise ValueError("manifest validation failed: " + "; ".join(errors))
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    cases, index_entries = manifest["cases"], []
    for case in cases:
        files = materialize_case(case, seed)
        directory = output / case["case_id"]
        directory.mkdir()
        digests = {}
        for name, content in files.items():
            (directory / name).write_bytes(content)
            digests[name] = hashlib.sha256(content).hexdigest()
        index_entries.append({
            "case_id": case["case_id"],
            "kind": case["kind"],
            "wire_format": case["wire_format"],
            "source_group_id": case["source_group_id"],
            "synthetic_tag": synthetic_tag(case["case_id"], seed),
            "pairing_resolved": case["pairing_resolved"],
            "request_bytes": len(files["request.json"]),
            "request_sha256": digests["request.json"],
            "sidecar_sha256": digests["sidecar.json"],
            "expected_sha256": digests["expected.json"],
        })
    index = {
        "schema_version": INDEX_SCHEMA,
        "manifest": Path(manifest_path).name,
        "manifest_sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": seed,
        "case_count": len(cases),
        "source_group_count": len(cases),
        "cases": index_entries,
        "shadow_only": True,
    }
    (output / "fixture_index.json").write_bytes(pretty(index))
    return index


def self_test(manifest_path=DEFAULT_MANIFEST, seed=None):
    """Validate manifest integrity without writing any output."""
    manifest = load_manifest(manifest_path)
    seed = manifest.get("seed", DEFAULT_SEED) if seed is None else seed
    errors = validate_manifest(manifest, seed)
    cases = manifest.get("cases", [])
    return {
        "schema_version": MANIFEST_SCHEMA,
        "mode": "self-test",
        "status": "failed" if errors else "ok",
        "seed": seed,
        "manifest": Path(manifest_path).name,
        "cases": len(cases),
        "source_groups": len(manifest.get("source_groups", [])),
        "scored_cases": sum(1 for case in cases if case["expected_gate"]["status"] == "scored"),
        "bypass_cases": sum(1 for case in cases if case["expected_gate"]["status"] == "bypass"),
        "required_kinds": list(REQUIRED_KINDS),
        "errors": errors,
        "wrote_output": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help="frozen fixture manifest (default: research/tool_history_fixture_manifest_v1.json)")
    parser.add_argument("--output-dir", type=Path, help="empty directory to materialize fixtures into")
    parser.add_argument("--seed", type=int, help="override the manifest seed for synthetic tags")
    parser.add_argument("--self-test", action="store_true", help="validate manifest integrity without writing output")
    args = parser.parse_args(argv)
    if args.self_test and args.output_dir is not None:
        parser.error("--self-test and --output-dir are mutually exclusive")
    if not args.self_test and args.output_dir is None:
        parser.error("either --self-test or --output-dir is required")
    if args.self_test:
        report = self_test(args.manifest, args.seed)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] == "ok" else 1
    try:
        index = build(args.output_dir, args.manifest, args.seed)
    except ValueError as error:
        print(json.dumps({"status": "failed", "error": str(error)}, indent=2))
        return 2
    print(json.dumps(index, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
