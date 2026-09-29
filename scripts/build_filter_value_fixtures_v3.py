#!/usr/bin/env python3
"""Build the held-out context-filter value/stress fixture manifest V3.

V3 intentionally contains only new self-authored cases. V2 remains the development
suite used to tune the diagnostic protocol; V3 is a local held-out replay cohort.
"""

import argparse
import hashlib
import json
from pathlib import Path

from build_filter_value_fixtures_v2 import basic_segments, openai_case, seg

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "research" / "context_filter_value_fixture_manifest_v2.json"
OUT = ROOT / "research" / "context_filter_value_fixture_manifest_v3.json"


def make_cases():
    cases = []

    cases.append(openai_case(
        "heldout_clear_archive_note", "heldout_clear_irrelevant",
        "A current queue depth is required; an unrelated archive note is eligible.",
        [
            {"role": "system", "content": "Answer only from recorded values."},
            {"role": "assistant", "content": "The latest queue depth is 42 jobs."},
            {"role": "assistant", "content": "Archived note about picnic blankets, reusable bottles, and weather forecasts unrelated to queues or jobs."},
            {"role": "user", "content": "What is the latest queue depth?"},
        ],
        sidecar={"segments": {"/messages/2/content": {"eligible": True}}},
        segments=basic_segments("/messages/2/content", 2, 4),
        downstream={"required_strings": ["queue depth", "42"], "expected_answer": "42 jobs"},
    ))

    cases.append(openai_case(
        "heldout_ambiguous_status_history", "heldout_ambiguous_history",
        "A stale warning may still explain the current state, so it should be retained.",
        [
            {"role": "system", "content": "Prefer current state but preserve conflicting evidence."},
            {"role": "assistant", "content": "Two days ago the scheduler reported quota exhaustion."},
            {"role": "assistant", "content": "The current scheduler state is running normally."},
            {"role": "user", "content": "What is the current scheduler state?"},
        ],
        sidecar={"segments": {"/messages/1/content": {"eligible": True}}},
        segments=[
            seg("/messages/0/content", "system", "protected", "retain", "protected_structure"),
            seg("/messages/1/content", "assistant", "eligible", "retain", "model_retain"),
            seg("/messages/2/content", "assistant", "protected", "retain", "not_explicitly_eligible"),
            seg("/messages/3/content", "user", "protected", "retain", "protected_structure"),
        ],
        expected_gate={"status": "scored", "reason": "shadow_only", "all_retain": True,
                       "segments_observed": True},
        downstream={"required_strings": ["current scheduler", "running normally"],
                    "expected_answer": "running normally"},
    ))

    cases.append(openai_case(
        "heldout_correction_dependency", "heldout_correction_dependency",
        "A stale endpoint note is dependency-linked to a later correction and must stay.",
        [
            {"role": "system", "content": "Preserve correction chains."},
            {"role": "assistant", "content": "Old endpoint note: metrics path was /metrics/old."},
            {"role": "assistant", "content": "Correction: metrics path is now /metrics/v2."},
            {"role": "user", "content": "Which metrics path is current?"},
        ],
        sidecar={"segments": {"/messages/1/content": {"eligible": True},
                              "/messages/2/content": {"depends_on": ["/messages/1/content"]}}},
        segments=[
            seg("/messages/0/content", "system", "protected", "retain", "protected_structure"),
            seg("/messages/1/content", "assistant", "eligible", "retain", "required_dependency"),
            seg("/messages/2/content", "assistant", "protected", "retain", "not_explicitly_eligible"),
            seg("/messages/3/content", "user", "protected", "retain", "protected_structure"),
        ],
        expected_gate={"status": "scored", "reason": "shadow_only", "all_retain": True,
                       "segments_observed": True},
        downstream={"required_strings": ["/metrics/old", "Correction", "/metrics/v2"],
                    "expected_answer": "/metrics/v2"},
    ))

    cases.append(openai_case(
        "heldout_tool_result_noise", "heldout_tool_result_noise",
        "A valid tool result is protected while a long unrelated assistant note is eligible.",
        [
            {"role": "system", "content": "Preserve tool calls and results."},
            {"role": "assistant", "tool_calls": [{"id": "call-heldout-cache", "type": "function", "function": {"name": "get_queue", "arguments": "{}"}}]},
            {"role": "tool", "content": "queue=ingest; backlog=913; state=draining", "tool_call_id": "call-heldout-cache"},
            {"role": "assistant", "content": "Long unrelated archive note about calendars, key cards, chair height, meeting snacks, label makers, drawer organizers, spare chargers, wall hooks, whiteboards, hallway signs, and cleaning schedules. It does not mention ingest, backlog, queues, or draining."},
            {"role": "user", "content": "What backlog did the queue tool report?"},
        ],
        sidecar={"segments": {"/messages/3/content": {"eligible": True}}},
        segments=[
            seg("/messages/0/content", "system", "protected", "retain", "protected_structure"),
            seg("/messages/1/tool_calls", "control", "protected", "retain", "protected_structure"),
            seg("/messages/2/tool_call_id", "control", "protected", "retain", "protected_structure"),
            seg("/messages/2/content", "tool", "protected", "retain", "protected_structure"),
            seg("/messages/3/content", "assistant", "eligible", "drop", "high_irrelevance_score"),
            seg("/messages/4/content", "user", "protected", "retain", "protected_structure"),
        ],
        downstream={"required_strings": ["queue=ingest", "backlog=913", "state=draining"],
                    "expected_answer": "913"},
    ))

    cases.append(openai_case(
        "heldout_mixed_part_drop", "heldout_mixed_message_part",
        "A cited config value is protected; an unrelated text part in the same message is eligible.",
        [
            {"role": "system", "content": "Use the cited configuration."},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Config source says retry_budget=4 and mode=strict.", "citations": [{"id": "cfg"}]},
                {"type": "text", "text": "A separate old note discusses lunch orders and badge colors unrelated to configuration."},
            ]},
            {"role": "user", "content": "What retry budget is configured?"},
        ],
        sidecar={"segments": {"/messages/1/content/1/text": {"eligible": True}}},
        segments=[
            seg("/messages/0/content", "system", "protected", "retain", "protected_structure"),
            seg("/messages/1/content/0/text", "assistant", "protected", "retain", "protected_structure"),
            seg("/messages/1/content/0/citations", "control", "protected", "retain", "protected_structure"),
            seg("/messages/1/content/1/text", "assistant", "eligible", "drop", "high_irrelevance_score"),
            seg("/messages/2/content", "user", "protected", "retain", "protected_structure"),
        ],
        downstream={"required_strings": ["retry_budget=4", "mode=strict"],
                    "expected_answer": "4"},
    ))

    cases.append(openai_case(
        "heldout_spanish_clear_irrelevant", "heldout_multilingual",
        "A Spanish user asks for a recorded deadline; unrelated English archive text is eligible.",
        [
            {"role": "system", "content": "Responde solo con los valores registrados."},
            {"role": "assistant", "content": "La fecha límite registrada es 2026-10-15."},
            {"role": "assistant", "content": "Archived note about team jerseys, picnic chairs, and office snacks unrelated to deadlines."},
            {"role": "user", "content": "¿Cuál es la fecha límite registrada?"},
        ],
        sidecar={"segments": {"/messages/2/content": {"eligible": True}}},
        segments=basic_segments("/messages/2/content", 2, 4),
        downstream={"required_strings": ["2026-10-15"], "expected_answer": "2026-10-15"},
    ))

    cases.append(openai_case(
        "heldout_exact_duplicate", "heldout_exact_duplicate",
        "The later duplicate copy is eligible while the first copy remains protected.",
        [
            {"role": "system", "content": "Preserve unique evidence."},
            {"role": "assistant", "content": "The backup window is Sunday 03:00 UTC."},
            {"role": "assistant", "content": "The backup window is Sunday 03:00 UTC."},
            {"role": "user", "content": "When is the backup window?"},
        ],
        sidecar={"segments": {"/messages/2/content": {"eligible": True}}},
        segments=[
            seg("/messages/0/content", "system", "protected", "retain", "protected_structure"),
            seg("/messages/1/content", "assistant", "protected", "retain", "not_explicitly_eligible"),
            seg("/messages/2/content", "assistant", "eligible", "drop", "exact_duplicate"),
            seg("/messages/3/content", "user", "protected", "retain", "protected_structure"),
        ],
        downstream={"required_strings": ["Sunday", "03:00 UTC"], "expected_answer": "Sunday 03:00 UTC"},
    ))

    cases.append(openai_case(
        "heldout_required_marked_eligible", "heldout_required_marked_eligible",
        "A required value is marked eligible in the sidecar, but expected behavior is retain.",
        [
            {"role": "system", "content": "Preserve required values even when marked eligible."},
            {"role": "assistant", "content": "Release freeze starts at 2026-10-01T18:00Z."},
            {"role": "user", "content": "When does the release freeze start?"},
        ],
        sidecar={"segments": {"/messages/1/content": {"eligible": True}}},
        segments=[
            seg("/messages/0/content", "system", "protected", "retain", "protected_structure"),
            seg("/messages/1/content", "assistant", "eligible", "retain", "model_retain"),
            seg("/messages/2/content", "user", "protected", "retain", "protected_structure"),
        ],
        expected_gate={"status": "scored", "reason": "shadow_only", "all_retain": True,
                       "segments_observed": True},
        downstream={"required_strings": ["2026-10-01T18:00Z"], "expected_answer": "2026-10-01T18:00Z"},
    ))

    cases.append(openai_case(
        "heldout_zero_drop_protected", "heldout_protected_only",
        "A request with only protected segments must remain unchanged even at diagnostic thresholds.",
        [
            {"role": "system", "content": "Preserve all evidence."},
            {"role": "assistant", "content": "The approved namespace is nanojev-shadow."},
            {"role": "user", "content": "Which namespace is approved?"},
        ],
        segments=[
            seg("/messages/0/content", "system", "protected", "retain", "protected_structure"),
            seg("/messages/1/content", "assistant", "protected", "retain", "not_explicitly_eligible"),
            seg("/messages/2/content", "user", "protected", "retain", "protected_structure"),
        ],
        expected_gate={"status": "scored", "reason": "shadow_only", "all_retain": True,
                       "segments_observed": True},
        downstream={"required_strings": ["nanojev-shadow"], "expected_answer": "nanojev-shadow"},
    ))

    cases.append(openai_case(
        "heldout_long_tail_after_answer", "heldout_long_tail",
        "A long unrelated tail follows the evidence and should be removable.",
        [
            {"role": "system", "content": "Use only the recorded value."},
            {"role": "assistant", "content": "The incident report says error_rate=2.4% and window=5m."},
            {"role": "assistant", "content": "Unrelated long tail about desk plants, cable ties, sticky labels, lamp stands, hall passes, shelf labels, charger bricks, whiteboard pens, floor mats, visitor stickers, spare keyboards, window blinds, and cleaning carts. It never mentions error rate or the five minute window."},
            {"role": "user", "content": "What error rate was reported?"},
        ],
        sidecar={"segments": {"/messages/2/content": {"eligible": True}}},
        segments=basic_segments("/messages/2/content", 2, 4),
        downstream={"required_strings": ["error_rate=2.4%", "window=5m"], "expected_answer": "2.4%"},
    ))

    cases.append(openai_case(
        "heldout_pending_tool_bypass", "heldout_pending_tool_bypass",
        "A tool call without a result remains unresolved and must bypass.",
        [
            {"role": "system", "content": "Preserve unresolved tool linkage."},
            {"role": "assistant", "tool_calls": [{"id": "call-heldout-pending", "type": "function", "function": {"name": "fetch_report", "arguments": "{}"}}]},
            {"role": "user", "content": "What did fetch_report return?"},
        ],
        expected_gate={"status": "bypass", "reason": "unresolved_tool_link", "all_retain": True,
                       "segments_observed": False},
        downstream={"required_strings": ["call-heldout-pending", "fetch_report"],
                    "expected_answer": "unknown"},
    ))

    cases.append(openai_case(
        "heldout_unknown_envelope_bypass", "heldout_unsupported_envelope",
        "An unknown envelope extension must bypass rather than be reduced.",
        [
            {"role": "system", "content": "Preserve unknown request fields."},
            {"role": "user", "content": "Use the vendor extension if present."},
        ],
        body_extra={"vendor_extension_blob": "synthetic"},
        expected_gate={"status": "bypass", "reason": "unsupported_envelope_fields",
                       "all_retain": True, "segments_observed": False},
        downstream={"required_strings": ["vendor_extension_blob", "synthetic"],
                    "expected_answer": "unsupported"},
    ))

    return cases


def build_manifest():
    base = json.loads(V2.read_text(encoding="utf-8"))
    cases = make_cases()
    manifest = {
        "schema_version": "nanojev-context-filter-value-fixture-manifest-v3",
        "status": "synthetic_shadow_only_v3_heldout",
        "purpose": "Held-out synthetic replay cohort for context filtering. Cases are new and do not reuse V2 development examples.",
        "seed": 20260924,
        "base_manifest": {"path": "research/context_filter_value_fixture_manifest_v2.json",
                          "sha256": hashlib.sha256(V2.read_bytes()).hexdigest()},
        "provenance": {
            "source": "self_authored_synthetic_conversation",
            "license": "CC0-1.0",
            "contains_real_user_data": False,
            "contains_real_credentials": False,
            "contains_real_tool_output": False,
            "contains_benchmark_rows": False,
            "contains_provider_outputs": False,
            "development_cohort": "research/context_filter_value_fixture_manifest_v2.json",
            "split": "heldout_replay",
        },
        "gate_contract": dict(base["gate_contract"]),
        "case_count": len(cases),
        "families": [case["kind"] for case in cases],
        "cases": cases,
    }
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--check", action="store_true", help="verify output matches builder")
    args = parser.parse_args()
    manifest = build_manifest()
    encoded = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.check:
        existing = args.output.read_text(encoding="utf-8")
        if existing != encoded:
            raise SystemExit(f"manifest does not match builder output: {args.output}")
        print(json.dumps({"status": "ok", "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
                          "case_count": manifest["case_count"]}, indent=2))
        return
    if args.output.exists() and args.output.read_bytes():
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")
    args.output.write_text(encoded, encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
                      "case_count": manifest["case_count"]}, indent=2))


if __name__ == "__main__":
    main()
