#!/usr/bin/env python3
"""Build the expanded context-filter value/stress fixture manifest V2.

All cases are self-authored synthetic conversations. The builder reuses the V1 cases
verbatim and appends broader provenance/format/adversarial coverage.
"""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V1 = ROOT / "research" / "context_filter_value_fixture_manifest_v1.json"
OUT = ROOT / "research" / "context_filter_value_fixture_manifest_v2.json"


def seg(pointer, role, status, suggestion, reason):
    return {"pointer": pointer, "role": role, "expected_status": status,
            "expected_suggestion": suggestion, "expected_reason_code": reason}


def openai_case(case_id, kind, description, messages, sidecar=None, segments=None,
                expected_gate=None, downstream=None, body_extra=None):
    body = {"model": "synthetic", "messages": messages}
    if body_extra:
        body.update(body_extra)
    return {
        "case_id": case_id,
        "kind": kind,
        "wire_format": "openai_chat",
        "description": description,
        "body": body,
        "sidecar": sidecar or {"segments": {}},
        "segments": segments or [],
        "expected_gate": expected_gate or {"status": "scored", "reason": "shadow_only",
                                           "all_retain": False, "segments_observed": True},
        "downstream": downstream or {},
    }


def basic_segments(drop_pointer, drop_index, total_messages):
    items = [seg("/messages/0/content", "system", "protected", "retain",
                 "protected_structure")]
    for index in range(1, total_messages - 1):
        pointer = f"/messages/{index}/content"
        if pointer == drop_pointer:
            items.append(seg(pointer, "assistant", "eligible", "drop",
                             "high_irrelevance_score"))
        else:
            items.append(seg(pointer, "assistant", "protected", "retain",
                             "not_explicitly_eligible"))
    items.append(seg(f"/messages/{total_messages - 1}/content", "user", "protected",
                     "retain", "protected_structure"))
    return items


def make_cases():
    cases = []

    cases.append(openai_case(
        "multilingual_clear_irrelevant", "multilingual_clear_irrelevant",
        "Chinese user asks for the recorded Nova port; an unrelated English archive note is eligible.",
        [
            {"role": "system", "content": "保留所有用户约束并只根据记录回答。"},
            {"role": "assistant", "content": "Project Nova currently uses local port 8123."},
            {"role": "assistant", "content": "Archived note about office plants, badge lanyards, and cafeteria snacks. It does not mention Nova, software, ports, or deployment."},
            {"role": "user", "content": "项目 Nova 现在使用哪个本地端口？"},
        ],
        sidecar={"segments": {"/messages/2/content": {"eligible": True}}},
        segments=basic_segments("/messages/2/content", 2, 4),
        downstream={"required_strings": ["8123", "Project Nova"], "expected_answer": "8123"},
    ))

    cases.append(openai_case(
        "multilingual_ambiguous_retain", "multilingual_ambiguous_retain",
        "An older Chinese note may still be relevant to the current inventory question, so it should be retained.",
        [
            {"role": "system", "content": "Prefer current evidence but preserve uncertain context."},
            {"role": "assistant", "content": "上个月记录显示 SKU 123 缺货，等待供应商确认。"},
            {"role": "assistant", "content": "今天的库存记录显示 SKU 123 in stock，数量为 18。"},
            {"role": "user", "content": "SKU 123 现在是否有货？"},
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
        downstream={"required_strings": ["今天", "in stock", "18"], "expected_answer": "in stock"},
    ))

    cases.append(openai_case(
        "policy_rotation_window", "policy_and_credential_placeholder",
        "The policy rotation window is required; a placeholder credential remains protected and an unrelated lunch note is eligible.",
        [
            {"role": "system", "content": "Never print credentials or placeholder secrets. Preserve policy constraints."},
            {"role": "assistant", "content": "Policy note: credential rotation window is 02:00-03:00 UTC. Placeholder secret id is PLACEHOLDER_ONLY_NOT_A_SECRET."},
            {"role": "assistant", "content": "Archived lunch note: soup, sandwiches, and a fruit list unrelated to policy or rotation."},
            {"role": "user", "content": "When is the credential rotation window?"},
        ],
        sidecar={"segments": {"/messages/2/content": {"eligible": True}}},
        segments=basic_segments("/messages/2/content", 2, 4),
        downstream={"required_strings": ["rotation window", "02:00-03:00"], "expected_answer": "02:00-03:00 UTC"},
    ))

    cases.append(openai_case(
        "correction_placeholder_dependency", "correction_placeholder_dependency",
        "A rotated placeholder depends on the stale placeholder note it corrects; dependency closure retains both.",
        [
            {"role": "system", "content": "Preserve correction chains and never print real credentials."},
            {"role": "assistant", "content": "Old placeholder marker OLD_PLACEHOLDER_DO_NOT_USE was assigned to service alpha."},
            {"role": "assistant", "content": "Correction: service alpha now uses ROTATED_PLACEHOLDER_ONLY."},
            {"role": "user", "content": "Which placeholder marker is current for service alpha?"},
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
        downstream={"required_strings": ["OLD_PLACEHOLDER_DO_NOT_USE", "Correction", "ROTATED_PLACEHOLDER_ONLY"],
                    "expected_answer": "ROTATED_PLACEHOLDER_ONLY"},
    ))

    cases.append(openai_case(
        "nonrepeatable_export_tool_result", "nonrepeatable_tool_result",
        "A non-repeatable export ID in a tool result is required; unrelated archive text is eligible.",
        [
            {"role": "system", "content": "Use recorded tool results and preserve one-time identifiers."},
            {"role": "assistant", "tool_calls": [{"id": "call-export-1", "type": "function", "function": {"name": "create_export", "arguments": "{\"dataset\":\"ops\"}"}}]},
            {"role": "tool", "content": "export_id=exp-778; status=ready; one_time_download=true", "tool_call_id": "call-export-1"},
            {"role": "assistant", "content": "Old note about choosing a café table unrelated to exports, tools, datasets, or identifiers."},
            {"role": "user", "content": "What is the export ID and download property?"},
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
        downstream={"required_strings": ["exp-778", "one_time_download=true"], "expected_answer": "exp-778 one_time_download=true"},
    ))

    cases.append(openai_case(
        "mutable_file_read_state", "mutable_file_read_state",
        "A mutable file-read tool result must remain with its timestamp; unrelated notes can be removed.",
        [
            {"role": "system", "content": "Treat file reads as time-bound evidence."},
            {"role": "assistant", "tool_calls": [{"id": "call-read-config", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\":\"config.yaml\"}"}}]},
            {"role": "tool", "content": "path=config.yaml; read_at=2026-09-23T10:00Z; mode=safe", "tool_call_id": "call-read-config"},
            {"role": "assistant", "content": "Unrelated note about chair colors, office lighting, and a whiteboard marker order."},
            {"role": "user", "content": "What mode did the recorded config read report?"},
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
        downstream={"required_strings": ["config.yaml", "read_at=2026-09-23T10:00Z", "mode=safe"], "expected_answer": "safe"},
    ))

    cases.append(openai_case(
        "long_tool_trace_noise", "long_tool_trace_noise",
        "A valid tool pair plus a long unrelated operations note exercises measurable removal around protected tool linkage.",
        [
            {"role": "system", "content": "Preserve tool calls and results."},
            {"role": "assistant", "tool_calls": [{"id": "call-cache", "type": "function", "function": {"name": "get_cache_status", "arguments": "{}"}}]},
            {"role": "tool", "content": "cache=warm; hit_rate=0.83", "tool_call_id": "call-cache"},
            {"role": "assistant", "content": "Long unrelated operations log about moving desks, replacing mugs, ordering markers, checking parking passes, sorting cables, labeling shelves, watering plants, printing visitor stickers, changing whiteboard erasers, arranging hallway bins, and selecting a brighter lobby sign. It never mentions cache, hit rate, warmth, tools, service state, or metrics."},
            {"role": "user", "content": "What cache hit rate was recorded?"},
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
        downstream={"required_strings": ["cache=warm", "hit_rate=0.83"], "expected_answer": "0.83"},
    ))

    cases.append(openai_case(
        "retrieval_distractor_cited_evidence", "retrieval_distractor_cited_evidence",
        "A cited evidence part is protected; a separate uncited distractor text part is eligible.",
        [
            {"role": "system", "content": "Use cited evidence when available."},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Doc A says the migration freeze starts on Friday at 18:00 UTC.", "citations": [{"id": "docA"}]},
                {"type": "text", "text": "Doc B discusses weekend picnic recipes unrelated to migration or freeze windows."},
            ]},
            {"role": "user", "content": "When does the migration freeze start?"},
        ],
        sidecar={"segments": {"/messages/1/content/1/text": {"eligible": True}}},
        segments=[
            seg("/messages/0/content", "system", "protected", "retain", "protected_structure"),
            seg("/messages/1/content/0/text", "assistant", "protected", "retain", "protected_structure"),
            seg("/messages/1/content/0/citations", "control", "protected", "retain", "protected_structure"),
            seg("/messages/1/content/1/text", "assistant", "eligible", "drop", "high_irrelevance_score"),
            seg("/messages/2/content", "user", "protected", "retain", "protected_structure"),
        ],
        downstream={"required_strings": ["Doc A", "Friday", "18:00 UTC"], "expected_answer": "Friday at 18:00 UTC"},
    ))

    cases.append(openai_case(
        "current_user_instruction_retained", "current_user_instruction_retained",
        "The current user instruction is protected while an old unrelated assistant note is eligible.",
        [
            {"role": "system", "content": "Answer the current user request."},
            {"role": "assistant", "content": "The audit flag is enabled in the staging workspace."},
            {"role": "assistant", "content": "Archived note about travel packing, rain jackets, and hotel breakfast unrelated to audits or staging."},
            {"role": "user", "content": "Is the audit flag enabled in staging?"},
        ],
        sidecar={"segments": {"/messages/2/content": {"eligible": True}}},
        segments=basic_segments("/messages/2/content", 2, 4),
        downstream={"required_strings": ["audit flag", "enabled", "staging"], "expected_answer": "enabled"},
    ))

    cases.append(openai_case(
        "code_review_config_evidence", "code_review_config_evidence",
        "A code-review configuration value is required; unrelated design chatter is eligible.",
        [
            {"role": "system", "content": "Preserve review configuration."},
            {"role": "assistant", "content": "Review config: max_comments=7 and severity_threshold=high."},
            {"role": "assistant", "content": "Archived design note about icon spacing, logo color, and slide transitions unrelated to review configuration."},
            {"role": "user", "content": "What is the configured maximum comment count?"},
        ],
        sidecar={"segments": {"/messages/2/content": {"eligible": True}}},
        segments=basic_segments("/messages/2/content", 2, 4),
        downstream={"required_strings": ["max_comments=7", "severity_threshold=high"], "expected_answer": "7"},
    ))

    cases.append(openai_case(
        "temporal_stale_status_ambiguous", "temporal_stale_status_ambiguous",
        "A stale status could conflict with the current one, so it remains ambiguous and should be retained.",
        [
            {"role": "system", "content": "Prefer current status but retain conflicting evidence."},
            {"role": "assistant", "content": "Yesterday the queue status was degraded while workers restarted."},
            {"role": "assistant", "content": "The current status check says the queue is healthy."},
            {"role": "user", "content": "What is the current queue status?"},
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
        downstream={"required_strings": ["current status", "healthy"], "expected_answer": "healthy"},
    ))

    cases.append(openai_case(
        "three_message_long_tail", "three_message_long_tail",
        "A long irrelevant tail between evidence and question should be removable without touching evidence.",
        [
            {"role": "system", "content": "Use only the recorded value."},
            {"role": "assistant", "content": "The report contains metric=latency_p95 and value=183ms."},
            {"role": "assistant", "content": "Long unrelated tail: notebooks, chargers, tote bags, mug lids, sticky notes, desk lamps, cable clips, extension cords, monitor risers, plant soil, reusable cups, hallway carts, printer labels, spare adapters, cleaning wipes, visitor badges, shelf brackets, keyboard foam, chair wheels, office clocks, and meeting room magnets."},
            {"role": "user", "content": "What latency_p95 value is recorded?"},
        ],
        sidecar={"segments": {"/messages/2/content": {"eligible": True}}},
        segments=basic_segments("/messages/2/content", 2, 4),
        downstream={"required_strings": ["latency_p95", "183ms"], "expected_answer": "183ms"},
    ))

    cases.append(openai_case(
        "pending_tool_call_bypass", "pending_tool_call_bypass",
        "A pending tool call has no result; linkage is unresolved so the request must bypass.",
        [
            {"role": "system", "content": "Preserve tool linkage."},
            {"role": "assistant", "tool_calls": [{"id": "call-pending", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
            {"role": "user", "content": "What did lookup return?"},
        ],
        expected_gate={"status": "bypass", "reason": "unresolved_tool_link", "all_retain": True,
                       "segments_observed": False},
        downstream={"required_strings": ["call-pending", "lookup"], "expected_answer": "unknown"},
    ))

    cases.append(openai_case(
        "orphan_tool_result_bypass", "orphan_tool_result_bypass",
        "A tool result without a matching pending call must bypass rather than guess attribution.",
        [
            {"role": "system", "content": "Preserve tool linkage."},
            {"role": "tool", "content": "status=ready", "tool_call_id": "call-missing"},
            {"role": "user", "content": "What status did the tool report?"},
        ],
        expected_gate={"status": "bypass", "reason": "unresolved_tool_link", "all_retain": True,
                       "segments_observed": False},
        downstream={"required_strings": ["status=ready", "call-missing"], "expected_answer": "ready"},
    ))

    cases.append(openai_case(
        "unsupported_image_part_bypass", "unsupported_image_part_bypass",
        "An image-like content part is unsupported and must bypass rather than be reduced.",
        [
            {"role": "system", "content": "Preserve unsupported modalities."},
            {"role": "assistant", "content": [{"type": "image_url", "image_url": {"url": "data:synthetic;base64,AAAA"}}]},
            {"role": "user", "content": "Can you inspect the synthetic image?"},
        ],
        expected_gate={"status": "bypass", "reason": "unsupported_content", "all_retain": True,
                       "segments_observed": False},
        downstream={"required_strings": ["image_url", "AAAA"], "expected_answer": "unsupported"},
    ))

    cases.append(openai_case(
        "unsupported_server_context_bypass", "unsupported_server_context_bypass",
        "Server-side previous_response context cannot be safely reconstructed locally and must bypass.",
        [
            {"role": "system", "content": "Preserve unresolved server context."},
            {"role": "user", "content": "Continue the previous request."},
        ],
        body_extra={"previous_response_id": "resp_synthetic_previous"},
        expected_gate={"status": "bypass", "reason": "unresolved_server_context", "all_retain": True,
                       "segments_observed": False},
        downstream={"required_strings": ["previous_response_id", "resp_synthetic_previous"], "expected_answer": "unavailable"},
    ))

    cases.append(openai_case(
        "unsupported_envelope_field_bypass", "unsupported_envelope_field_bypass",
        "An unknown top-level envelope field is unsupported and must bypass.",
        [
            {"role": "system", "content": "Preserve unknown envelopes."},
            {"role": "user", "content": "Use the experimental field if present."},
        ],
        body_extra={"experimental_opaque_payload": "AAAA"},
        expected_gate={"status": "bypass", "reason": "unsupported_envelope_fields", "all_retain": True,
                       "segments_observed": False},
        downstream={"required_strings": ["experimental_opaque_payload", "AAAA"], "expected_answer": "unsupported"},
    ))

    return cases


def build_manifest():
    base = json.loads(V1.read_text(encoding="utf-8"))
    cases = list(base["cases"]) + make_cases()
    manifest = {
        "schema_version": "nanojev-context-filter-value-fixture-manifest-v2",
        "status": "synthetic_shadow_only_v2",
        "purpose": "Expanded value/stress fixtures for context filtering: multilingual, policy/credential placeholders, corrections, tool linkage, non-repeatable/mutable results, cited retrieval distractors, unsupported inputs, and protected-family cases.",
        "seed": 20260923,
        "base_manifest": {"path": "research/context_filter_value_fixture_manifest_v1.json",
                          "sha256": hashlib.sha256(V1.read_bytes()).hexdigest()},
        "provenance": {
            "source": "self_authored_synthetic_conversation",
            "license": "CC0-1.0",
            "contains_real_user_data": False,
            "contains_real_credentials": False,
            "contains_real_tool_output": False,
            "contains_benchmark_rows": False,
            "contains_provider_outputs": False,
        },
        "gate_contract": dict(base["gate_contract"]),
        "case_count": len(cases),
        "families": [
            "clear_irrelevant_history", "required_evidence_marked_eligible",
            "dependency_retains_stale_note", "tool_result_plus_irrelevant_note",
            "mixed_message_part", "exact_duplicate", "long_tail_noise",
            "ambiguous_history", "multilingual_clear_irrelevant",
            "multilingual_ambiguous_retain", "policy_and_credential_placeholder",
            "correction_placeholder_dependency", "nonrepeatable_tool_result",
            "mutable_file_read_state", "long_tool_trace_noise",
            "retrieval_distractor_cited_evidence", "current_user_instruction_retained",
            "code_review_config_evidence", "temporal_stale_status_ambiguous",
            "three_message_long_tail", "pending_tool_call_bypass",
            "orphan_tool_result_bypass", "unsupported_image_part_bypass",
            "unsupported_server_context_bypass", "unsupported_envelope_field_bypass",
        ],
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
