#!/usr/bin/env python3
"""Reversible-filtering tests: content-free manifests and byte-exact restore.

Coverage:

* byte-identical round-trip for every supported wire format, for whole-message removal and
  for partial-content (text part) removal, including a message emptied by removing all of
  its parts;
* a no-op (nothing removed) round-trips trivially;
* the manifest is content-free: no raw prompt text, and only the recorded safe vocabulary;
* a tampered manifest or tampered removed segment is detected, never silently accepted;
* malformed, duplicated, out-of-range, wrong-wire, and out-of-grammar pointers are
  rejected at both build time and ingestion time.

Run with::

    .venv/bin/python -m unittest discover -s scripts -p 'test_context_restore*.py'
"""

import hashlib
import json
from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from context_restore_v1 import (  # noqa: E402
    SCHEMA_VERSION, RestoreError, build_restore_manifest, canonical_bytes,
    normalize_removed_segments, restore_request, serialized, verify_round_trip,
)
from main_model_gateway_v1 import build_reduced_request  # noqa: E402


SYSTEM_SENTINEL = "SYSTEM_SENTINEL_KEEP_ALL_CONSTRAINTS_4a1c"
USER_SENTINEL = "USER_SENTINEL_FIND_THE_WAREHOUSE_9b2e"
HISTORY_SENTINEL = "HISTORY_SENTINEL_ARCHIVED_WEATHER_NOTE_7f31"
SECOND_SENTINEL = "SECOND_SENTINEL_ARCHIVED_LUNCH_ASIDE_c0d5"
TOOL_SENTINEL = "TOOL_SENTINEL_RESULT_ROW_8812"


def openai_body():
    return {"model": "synthetic-main-model", "messages": [
        {"role": "system", "content": SYSTEM_SENTINEL},
        {"role": "assistant", "content": HISTORY_SENTINEL},
        {"role": "user", "content": USER_SENTINEL},
    ]}


def openai_two_history_body():
    return {"model": "synthetic-main-model", "messages": [
        {"role": "system", "content": SYSTEM_SENTINEL},
        {"role": "assistant", "content": HISTORY_SENTINEL},
        {"role": "assistant", "content": SECOND_SENTINEL},
        {"role": "user", "content": USER_SENTINEL},
    ]}


def openai_partial_body():
    return {"model": "synthetic-main-model", "messages": [
        {"role": "system", "content": SYSTEM_SENTINEL},
        {"role": "assistant", "content": [
            {"type": "text", "text": HISTORY_SENTINEL},
            {"type": "text", "text": SECOND_SENTINEL},
            {"type": "text", "text": TOOL_SENTINEL},
        ]},
        {"role": "user", "content": USER_SENTINEL},
    ]}


def anthropic_body():
    return {"model": "synthetic-main-model", "system": SYSTEM_SENTINEL, "max_tokens": 64,
            "messages": [
                {"role": "assistant", "content": HISTORY_SENTINEL},
                {"role": "user", "content": USER_SENTINEL},
            ]}


def anthropic_partial_body():
    return {"model": "synthetic-main-model", "system": SYSTEM_SENTINEL, "max_tokens": 64,
            "messages": [
                {"role": "assistant", "content": [{"type": "text", "text": HISTORY_SENTINEL}]},
                {"role": "user", "content": USER_SENTINEL},
            ]}


def responses_body():
    return {"model": "synthetic-main-model", "instructions": SYSTEM_SENTINEL, "input": [
        {"role": "assistant", "content": HISTORY_SENTINEL},
        {"role": "user", "content": USER_SENTINEL},
    ]}


def raw_of(body):
    return canonical_bytes(body)


def removed_values(body):
    """The segment values a caller would have retained, keyed by their pointers."""
    parsed = json.loads(serialized(body))
    key = "input" if "input" in parsed else "messages"
    values = {}
    for index, message in enumerate(parsed[key]):
        content = message.get("content")
        if isinstance(content, str):
            values[f"/{key}/{index}/content"] = content
        elif isinstance(content, list):
            for part_index, part in enumerate(content):
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    values[f"/{key}/{index}/content/{part_index}/text"] = part["text"]
    return values


def manifest_of(body, pointers):
    raw = raw_of(body)
    manifest = build_restore_manifest(raw, wire_format_for(body), pointers)
    return raw, manifest


def wire_format_for(body):
    if "input" in body:
        return "openai_responses"
    if "system" in body:
        return "anthropic_messages"
    return "openai_chat"


def manifest_dict(manifest):
    return json.loads(manifest.to_json())


def all_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, entry in value.items():
            yield key
            yield from all_strings(entry)
    elif isinstance(value, (list, tuple)):
        for entry in value:
            yield from all_strings(entry)


class WholeMessageRemoveTest(unittest.TestCase):
    """Byte-exact round-trip for every wire format, whole-message removal."""

    def test_openai_chat_whole_message_is_byte_identical(self):
        body = openai_body()
        raw, manifest = manifest_of(body, ["/messages/1/content"])
        reduced = canonical_bytes({**body, "messages": [body["messages"][0], body["messages"][2]]})
        self.assertEqual(manifest.reduced_request_sha256, hashlib.sha256(reduced).hexdigest())
        restored = restore_request(reduced, manifest, {"/messages/1/content": HISTORY_SENTINEL})
        self.assertEqual(restored, raw)
        self.assertEqual(restored, canonical_bytes(body))
        self.assertTrue(manifest.applied)
        self.assertEqual(manifest.dropped_segment_count, 1)
        self.assertTrue(manifest.canonical_json)

    def test_anthropic_whole_message_is_byte_identical(self):
        body = {"model": "synthetic-main-model", "system": SYSTEM_SENTINEL, "max_tokens": 64,
                "messages": [{"role": "assistant", "content": HISTORY_SENTINEL},
                             {"role": "user", "content": USER_SENTINEL}]}
        raw, manifest = manifest_of(body, ["/messages/0/content"])
        reduced = canonical_bytes({**body, "messages": [body["messages"][1]]})
        restored = restore_request(reduced, manifest, {"/messages/0/content": HISTORY_SENTINEL})
        self.assertEqual(restored, raw)
        self.assertEqual(manifest.wire_format, "anthropic_messages")

    def test_anthropic_list_content_whole_message_drop_is_refused(self):
        # A mixed/list-content message is not a single text segment, so the reduction is
        # not reversible and must never be presented as one.
        body = {"model": "synthetic-main-model", "system": SYSTEM_SENTINEL, "max_tokens": 64,
                "messages": [{"role": "assistant", "content": [{"type": "text", "text": HISTORY_SENTINEL}]},
                             {"role": "user", "content": USER_SENTINEL}]}
        with self.assertRaises(RestoreError) as caught:
            manifest_of(body, ["/messages/0/content"])
        self.assertIn("whole_message_drop_requires_string_content", str(caught.exception))

    def test_openai_responses_whole_message_is_byte_identical(self):
        body = responses_body()
        raw, manifest = manifest_of(body, ["/input/0/content"])
        reduced = canonical_bytes({**body, "input": [body["input"][1]]})
        restored = restore_request(reduced, manifest, {"/input/0/content": HISTORY_SENTINEL})
        self.assertEqual(restored, raw)
        self.assertEqual(manifest.wire_format, "openai_responses")
        self.assertEqual(manifest.message_key, "input")

    def test_multiple_whole_messages_round_trip_in_original_order(self):
        body = openai_two_history_body()
        raw, manifest = manifest_of(body, ["/messages/1/content", "/messages/2/content"])
        reduced = canonical_bytes({**body, "messages": [body["messages"][0], body["messages"][3]]})
        # Descending removal order must not be assumed: positions are inserted by index.
        supplied = {"/messages/2/content": SECOND_SENTINEL, "/messages/1/content": HISTORY_SENTINEL}
        restored = restore_request(reduced, manifest, supplied)
        self.assertEqual(restored, raw)

    def test_whole_message_manifest_record_shape(self):
        _, manifest = manifest_of(openai_body(), ["/messages/1/content"])
        record = manifest_dict(manifest)["records"][0]
        self.assertEqual(record["pointer"], "/messages/1/content")
        self.assertEqual(record["kind"], "message")
        self.assertEqual(record["role"], "assistant")
        self.assertEqual(record["message_key"], "messages")
        self.assertEqual(record["message_index"], 1)
        self.assertIsNone(record["part_index"])
        self.assertEqual(len(record["segment_sha256"]), 64)
        self.assertEqual(manifest_dict(manifest)["schema_version"], SCHEMA_VERSION)


class PartialContentRemoveTest(unittest.TestCase):
    """Byte-exact round-trip for partial (text-part) content removal."""

    def test_partial_text_part_is_byte_identical(self):
        body = openai_partial_body()
        raw, manifest = manifest_of(body, ["/messages/1/content/1/text"])
        kept = [body["messages"][1]["content"][0], body["messages"][1]["content"][2]]
        reduced = canonical_bytes({**body, "messages": [
            body["messages"][0], {**body["messages"][1], "content": kept}, body["messages"][2]]})
        restored = restore_request(reduced, manifest, {"/messages/1/content/1/text": SECOND_SENTINEL})
        self.assertEqual(restored, raw)
        record = manifest_dict(manifest)["records"][0]
        self.assertEqual(record["kind"], "text_part")
        self.assertEqual(record["part_index"], 1)

    def test_anthropic_partial_text_part_is_byte_identical(self):
        body = {"model": "synthetic-main-model", "system": SYSTEM_SENTINEL, "max_tokens": 64,
                "messages": [
                    {"role": "assistant", "content": [
                        {"type": "text", "text": HISTORY_SENTINEL},
                        {"type": "text", "text": TOOL_SENTINEL}]},
                    {"role": "user", "content": USER_SENTINEL}]}
        raw, manifest = manifest_of(body, ["/messages/0/content/0/text"])
        reduced = canonical_bytes({**body, "messages": [
            {**body["messages"][0], "content": [body["messages"][0]["content"][1]]},
            body["messages"][1]]})
        restored = restore_request(reduced, manifest, {"/messages/0/content/0/text": HISTORY_SENTINEL})
        self.assertEqual(restored, raw)

    def test_multiple_parts_from_one_message_round_trip(self):
        body = openai_partial_body()
        raw, manifest = manifest_of(body, ["/messages/1/content/0/text", "/messages/1/content/2/text"])
        reduced = canonical_bytes({**body, "messages": [
            body["messages"][0],
            {**body["messages"][1], "content": [body["messages"][1]["content"][1]]},
            body["messages"][2]]})
        restored = restore_request(reduced, manifest, {
            "/messages/1/content/0/text": HISTORY_SENTINEL,
            "/messages/1/content/2/text": TOOL_SENTINEL})
        self.assertEqual(restored, raw)
        self.assertEqual(manifest.dropped_segment_count, 2)

    def test_removing_every_part_removes_the_message_and_restores_it(self):
        body = {"model": "synthetic-main-model", "messages": [
            {"role": "system", "content": SYSTEM_SENTINEL},
            {"role": "assistant", "content": [
                {"type": "text", "text": HISTORY_SENTINEL},
                {"type": "text", "text": SECOND_SENTINEL}]},
            {"role": "user", "content": USER_SENTINEL}]}
        raw, manifest = manifest_of(body, ["/messages/1/content/0/text", "/messages/1/content/1/text"])
        # An emptied message is dropped whole, exactly as the gateway applicator does.
        reduced = canonical_bytes({**body, "messages": [body["messages"][0], body["messages"][2]]})
        restored = restore_request(reduced, manifest, {
            "/messages/1/content/0/text": HISTORY_SENTINEL,
            "/messages/1/content/1/text": SECOND_SENTINEL})
        self.assertEqual(restored, raw)
        self.assertEqual(manifest.records[0].kind, "text_part")

    def test_partial_removal_keeps_the_message_when_a_part_remains(self):
        body = openai_partial_body()
        raw, manifest = manifest_of(body, ["/messages/1/content/0/text", "/messages/1/content/1/text"])
        reduced = canonical_bytes({**body, "messages": [
            body["messages"][0],
            {**body["messages"][1], "content": [body["messages"][1]["content"][2]]},
            body["messages"][2]]})
        self.assertNotEqual(reduced, canonical_bytes({**body, "messages": [
            body["messages"][0], body["messages"][2]]}))
        restored = restore_request(reduced, manifest, {
            "/messages/1/content/0/text": HISTORY_SENTINEL,
            "/messages/1/content/1/text": SECOND_SENTINEL})
        self.assertEqual(restored, raw)

    def test_mixed_whole_message_and_partial_removal_round_trips(self):
        body = openai_partial_body()
        body["messages"].insert(2, {"role": "assistant", "content": SECOND_SENTINEL})
        raw, manifest = manifest_of(body, ["/messages/1/content/0/text", "/messages/2/content"])
        reduced = canonical_bytes({**body, "messages": [
            body["messages"][0],
            {**body["messages"][1], "content": body["messages"][1]["content"][1:]},
            body["messages"][3]]})
        restored = restore_request(reduced, manifest, {
            "/messages/1/content/0/text": HISTORY_SENTINEL,
            "/messages/2/content": SECOND_SENTINEL})
        self.assertEqual(restored, raw)


class NoOpTest(unittest.TestCase):
    """Nothing removed must round-trip trivially."""

    def test_empty_plan_is_a_no_op_round_trip(self):
        body = openai_body()
        raw = raw_of(body)
        manifest = build_restore_manifest(raw, "openai_chat", [])
        self.assertFalse(manifest.applied)
        self.assertEqual(manifest.dropped_segment_count, 0)
        self.assertEqual(manifest.pointers, ())
        self.assertEqual(manifest.original_request_sha256, manifest.reduced_request_sha256)
        self.assertEqual(manifest_dict(manifest)["records"], [])
        self.assertEqual(restore_request(raw, manifest, {}), raw)
        result = verify_round_trip(raw, raw, manifest, {})
        self.assertTrue(result["restored_bytes_identical"])
        self.assertEqual(result["dropped_segment_count"], 0)
        self.assertFalse(result["applied"])

    def test_no_op_manifest_rejects_supplied_segments(self):
        raw = raw_of(openai_body())
        manifest = build_restore_manifest(raw, "openai_chat", [])
        with self.assertRaises(RestoreError):
            restore_request(raw, manifest, {"/messages/1/content": HISTORY_SENTINEL})

    def test_no_op_manifest_rejects_different_bytes(self):
        body = openai_body()
        raw = raw_of(body)
        manifest = build_restore_manifest(raw, "openai_chat", [])
        other = canonical_bytes({**body, "max_tokens": 12})
        with self.assertRaises(RestoreError):
            restore_request(other, manifest, {})

    def test_verify_round_trip_without_segments_never_claims_reconstruction(self):
        body = openai_body()
        raw, manifest = manifest_of(body, ["/messages/1/content"])
        reduced = canonical_bytes({**body, "messages": [body["messages"][0], body["messages"][2]]})
        result = verify_round_trip(raw, reduced, manifest)
        self.assertFalse(result["reconstruction_performed"])
        self.assertIsNone(result["restored_bytes_identical"])
        self.assertTrue(result["original_request_sha256_matches"])
        self.assertTrue(result["reduced_request_sha256_matches"])


class ContentFreeManifestTest(unittest.TestCase):
    """A manifest must never contain raw prompt text."""

    SAFE_VOCABULARY = {
        SCHEMA_VERSION, "openai_chat", "openai_responses", "anthropic_messages",
        "messages", "input", "message", "text_part", "assistant", "system", "user",
        "developer", "tool", "control",
        # Structural field names and enum-ish keys that carry no request content.
        "schema_version", "wire_format", "message_key", "canonical_json",
        "original_request_sha256", "restored_bytes_sha256", "reduced_request_sha256", "applied",
        "dropped_segment_count", "records", "pointer", "kind", "role",
        "message_index", "part_index", "segment_sha256",
    }

    def sample_manifests(self):
        builders = [
            ("openai_chat", openai_body(), ["/messages/1/content"]),
            ("openai_chat", openai_partial_body(), ["/messages/1/content/1/text"]),
            ("openai_chat", openai_two_history_body(), ["/messages/1/content", "/messages/2/content"]),
            ("anthropic_messages", anthropic_body(), ["/messages/0/content"]),
            ("openai_responses", responses_body(), ["/input/0/content"]),
        ]
        for wire_format, body, pointers in builders:
            yield wire_format, body, build_restore_manifest(canonical_bytes(body), wire_format, pointers)

    def test_no_raw_prompt_text_appears_anywhere_in_the_manifest(self):
        sentinels = (SYSTEM_SENTINEL, USER_SENTINEL, HISTORY_SENTINEL, SECOND_SENTINEL, TOOL_SENTINEL)
        for wire_format, body, manifest in self.sample_manifests():
            with self.subTest(wire_format=wire_format, pointers=manifest.pointers):
                payload = manifest.to_json()
                for sentinel in sentinels:
                    self.assertNotIn(sentinel, payload)
                    self.assertNotIn(sentinel.lower(), payload.lower())
                for text in all_strings(manifest_dict(manifest)):
                    for sentinel in sentinels:
                        self.assertNotIn(sentinel, text)

    def test_every_manifest_string_is_recorded_safe_vocabulary_or_a_pointer(self):
        pointer_re = re.compile(r"^/(messages|input)/\d+/content(?:/\d+/text)?$")
        hex_re = re.compile(r"^[0-9a-f]{64}$")
        for wire_format, body, manifest in self.sample_manifests():
            with self.subTest(wire_format=wire_format, pointers=manifest.pointers):
                for text in all_strings(manifest_dict(manifest)):
                    if pointer_re.match(text) or hex_re.match(text):
                        continue
                    self.assertIn(text, self.SAFE_VOCABULARY,
                                  msg=f"unexpected manifest string {text!r} could carry content")

    def test_manifest_hashes_are_not_the_raw_prompt_and_are_hex(self):
        import re
        _, _, manifest = next(iter(self.sample_manifests()))
        payload = manifest_dict(manifest)
        for name in ("original_request_sha256", "reduced_request_sha256"):
            self.assertRegex(payload[name], r"^[0-9a-f]{64}$")
        for record in payload["records"]:
            self.assertRegex(record["segment_sha256"], r"^[0-9a-f]{64}$")
        self.assertNotIn(HISTORY_SENTINEL, serialized(payload))

    def test_manifest_round_trips_through_json_and_the_dict_form(self):
        body = openai_body()
        raw, manifest = manifest_of(body, ["/messages/1/content"])
        reduced = canonical_bytes({**body, "messages": [body["messages"][0], body["messages"][2]]})
        for encoded in (manifest.to_json(), manifest.as_dict(), json.loads(manifest.to_json())):
            with self.subTest(form=type(encoded).__name__):
                self.assertEqual(restore_request(reduced, encoded,
                                                 {"/messages/1/content": HISTORY_SENTINEL}), raw)

    def test_removed_values_helper_matches_manifest_pointers(self):
        body = openai_partial_body()
        _, manifest = manifest_of(body, ["/messages/1/content/1/text"])
        supplied = removed_values(body)
        self.assertEqual(set(manifest.pointers), {"/messages/1/content/1/text"})
        self.assertEqual(supplied["/messages/1/content/1/text"], SECOND_SENTINEL)
        for pointer in manifest.pointers:
            self.assertIn(pointer, supplied)


class TamperDetectionTest(unittest.TestCase):
    """A tampered manifest or segment must never restore silently."""

    def setUp(self):
        self.body = openai_body()
        self.raw, self.manifest = manifest_of(self.body, ["/messages/1/content"])
        self.reduced = canonical_bytes({**self.body, "messages": [
            self.body["messages"][0], self.body["messages"][2]]})
        self.supplied = {"/messages/1/content": HISTORY_SENTINEL}

    def test_removing_a_record_from_the_manifest_is_detected(self):
        payload = manifest_dict(self.manifest)
        payload["records"] = []
        payload["dropped_segment_count"] = 0
        payload["applied"] = False
        with self.assertRaises(RestoreError):
            restore_request(self.reduced, payload, self.supplied)

    def test_swapping_a_record_pointer_is_detected(self):
        payload = manifest_dict(self.manifest)
        payload["records"][0]["pointer"] = "/messages/0/content"
        with self.assertRaises(RestoreError):
            restore_request(self.reduced, payload,
                            {"/messages/0/content": SYSTEM_SENTINEL})

    def test_unknown_record_field_is_detected(self):
        payload = manifest_dict(self.manifest)
        payload["records"][0]["removed_text"] = HISTORY_SENTINEL
        with self.assertRaises(RestoreError):
            restore_request(self.reduced, payload, self.supplied)

    def test_tampered_segment_text_is_detected(self):
        with self.assertRaises(RestoreError) as caught:
            restore_request(self.reduced, self.manifest,
                            {"/messages/1/content": "A DIFFERENT ARCHIVED NOTE"})
        self.assertIn("removed_segment_hash_mismatch", str(caught.exception))

    def test_tampered_segment_role_is_detected(self):
        tampered = {**self.body["messages"][1], "role": "user"}
        with self.assertRaises(RestoreError):
            restore_request(self.reduced, self.manifest, {"/messages/1/content": tampered})

    def test_tampered_segment_extra_field_is_detected(self):
        tampered = {**self.body["messages"][1], "name": "injected"}
        with self.assertRaises(RestoreError):
            restore_request(self.reduced, self.manifest, {"/messages/1/content": tampered})

    def test_tampered_original_hash_is_detected(self):
        payload = manifest_dict(self.manifest)
        payload["original_request_sha256"] = "0" * 64
        with self.assertRaises(RestoreError) as caught:
            verify_round_trip(self.raw, self.reduced, payload, self.supplied)
        self.assertIn("original_request_sha256_mismatch", str(caught.exception))

    def test_tampered_restored_bytes_hash_is_detected(self):
        payload = manifest_dict(self.manifest)
        payload["restored_bytes_sha256"] = "0" * 64
        with self.assertRaises(RestoreError) as caught:
            restore_request(self.reduced, payload, self.supplied)
        self.assertIn("restored_request_hash_mismatch", str(caught.exception))

    def test_tampered_reduced_hash_is_detected(self):
        payload = manifest_dict(self.manifest)
        payload["reduced_request_sha256"] = "f" * 64
        with self.assertRaises(RestoreError) as caught:
            restore_request(self.reduced, payload, self.supplied)
        self.assertIn("reduced_bytes_do_not_match_manifest", str(caught.exception))

    def test_tampered_segment_hash_is_detected(self):
        payload = manifest_dict(self.manifest)
        payload["records"][0]["segment_sha256"] = "a" * 64
        with self.assertRaises(RestoreError) as caught:
            restore_request(self.reduced, payload, self.supplied)
        self.assertIn("removed_segment_hash_mismatch", str(caught.exception))

    def test_tampered_position_is_detected(self):
        payload = manifest_dict(self.manifest)
        payload["records"][0]["pointer"] = "/messages/0/content"
        payload["records"][0]["message_index"] = 0
        with self.assertRaises(RestoreError):
            restore_request(self.reduced, payload, self.supplied)

    def test_tampered_count_is_detected(self):
        payload = manifest_dict(self.manifest)
        payload["dropped_segment_count"] = 2
        with self.assertRaises(RestoreError) as caught:
            restore_request(self.reduced, payload, self.supplied)
        self.assertIn("manifest_count_mismatch", str(caught.exception))

    def test_tampered_applied_flag_is_detected(self):
        payload = manifest_dict(self.manifest)
        payload["applied"] = False
        with self.assertRaises(RestoreError):
            restore_request(self.reduced, payload, self.supplied)

    def test_tampered_schema_version_is_detected(self):
        payload = manifest_dict(self.manifest)
        payload["schema_version"] = "nanojev-context-restore-v2"
        with self.assertRaises(RestoreError) as caught:
            restore_request(self.reduced, payload, self.supplied)
        self.assertIn("unsupported_manifest_schema_version", str(caught.exception))

    def test_unknown_manifest_field_is_detected(self):
        payload = manifest_dict(self.manifest)
        payload["prompt_text"] = HISTORY_SENTINEL  # an injected extra field is refused
        with self.assertRaises(RestoreError):
            restore_request(self.reduced, payload, self.supplied)

    def test_missing_manifest_field_is_detected(self):
        payload = manifest_dict(self.manifest)
        payload.pop("reduced_request_sha256")
        with self.assertRaises(RestoreError):
            restore_request(self.reduced, payload, self.supplied)

    def test_malformed_manifest_json_is_refused(self):
        with self.assertRaises(RestoreError) as caught:
            restore_request(self.reduced, "{not json", self.supplied)
        self.assertIn("manifest_is_not_valid_json", str(caught.exception))

    def test_reduced_bytes_from_another_request_are_refused(self):
        other = openai_body()
        other["messages"][2]["content"] = "A COMPLETELY DIFFERENT USER REQUEST"
        wrong_reduced = canonical_bytes({**other, "messages": [
            other["messages"][0], other["messages"][2]]})
        with self.assertRaises(RestoreError):
            restore_request(wrong_reduced, self.manifest, self.supplied)

    def test_reduced_bytes_that_keep_the_dropped_message_are_refused(self):
        # The caller claims a reduction that never happened; no reconstruction is allowed.
        unreduced = canonical_bytes(self.body)
        with self.assertRaises(RestoreError):
            restore_request(unreduced, self.manifest, self.supplied)

    def test_verify_round_trip_raises_on_mismatched_raw(self):
        other = canonical_bytes({**self.body, "max_tokens": 7})
        with self.assertRaises(RestoreError) as caught:
            verify_round_trip(other, self.reduced, self.manifest, self.supplied)
        self.assertIn("original_request_sha256_mismatch", str(caught.exception))

    def test_verify_round_trip_raises_on_mismatched_reduced(self):
        with self.assertRaises(RestoreError) as caught:
            verify_round_trip(self.raw, canonical_bytes(self.body), self.manifest, self.supplied)
        self.assertIn("reduced_request_sha256_mismatch", str(caught.exception))


class MalformedPointerTest(unittest.TestCase):
    """Malformed / duplicate / out-of-range / out-of-grammar pointers are rejected."""

    def setUp(self):
        self.body = openai_body()
        self.raw = raw_of(self.body)

    def build(self, pointers, wire_format="openai_chat", body=None):
        return build_restore_manifest(raw_of(body or self.body), wire_format, pointers)

    def test_pointer_outside_the_allowed_grammar_is_rejected(self):
        for pointer in ("/messages/1/role", "/messages/1", "/messages/1/content/0",
                        "/messages/1/content/0/text/extra", "/messages/1/content/-1/text",
                        "/messages/1/content/0/ text", "messages/1/content",
                        "/messages/1/content/0/text/", "", "/model", "/messages/x/content"):
            with self.subTest(pointer=pointer):
                with self.assertRaises(RestoreError):
                    self.build([pointer])

    def test_message_index_out_of_range_is_rejected(self):
        with self.assertRaises(RestoreError) as caught:
            self.build(["/messages/9/content"])
        self.assertIn("drop_pointer_out_of_range", str(caught.exception))

    def test_part_index_out_of_range_is_rejected(self):
        body = openai_partial_body()
        with self.assertRaises(RestoreError) as caught:
            self.build(["/messages/1/content/7/text"], body=body)
        self.assertIn("drop_pointer_out_of_range", str(caught.exception))

    def test_part_pointer_at_a_string_content_segment_is_rejected(self):
        with self.assertRaises(RestoreError):
            self.build(["/messages/1/content/0/text"])

    def test_whole_pointer_at_mixed_content_segment_is_rejected(self):
        # A whole-message drop is only reversible for a single string text segment, so a
        # mixed (list) content message is refused rather than half-restored.
        body = openai_partial_body()
        with self.assertRaises(RestoreError) as caught:
            self.build(["/messages/1/content"], body=body)
        self.assertIn("whole_message_drop_requires_string_content", str(caught.exception))

    def test_whole_and_part_records_for_the_same_message_conflict(self):
        body = openai_body()
        body["messages"][1] = {"role": "assistant", "content": [
            {"type": "text", "text": HISTORY_SENTINEL}]}
        with self.assertRaises(RestoreError) as caught:
            self.build(["/messages/1/content", "/messages/1/content/0/text"], body=body)
        self.assertIn("conflicting_drop_positions", str(caught.exception))

    def test_two_part_records_on_the_same_message_are_allowed(self):
        body = openai_partial_body()
        manifest = self.build(["/messages/1/content/0/text", "/messages/1/content/2/text"], body=body)
        self.assertEqual(manifest.dropped_segment_count, 2)

    def test_duplicate_pointer_is_rejected(self):
        with self.assertRaises(RestoreError) as caught:
            self.build(["/messages/1/content", "/messages/1/content"])
        self.assertIn("duplicate_drop_pointer", str(caught.exception))

    def test_conflicting_part_positions_are_rejected(self):
        body = openai_partial_body()
        with self.assertRaises(RestoreError) as caught:
            self.build(["/messages/1/content/0/text", "/messages/1/content/0/text"], body=body)
        self.assertIn("duplicate_drop_pointer", str(caught.exception))

    def test_wrong_wire_key_is_rejected(self):
        with self.assertRaises(RestoreError) as caught:
            self.build(["/input/1/content"], wire_format="openai_chat")
        self.assertIn("drop_pointer_wrong_wire_key", str(caught.exception))
        with self.assertRaises(RestoreError):
            self.build(["/messages/1/content"], wire_format="openai_responses",
                       body=responses_body())

    def test_unsupported_wire_format_and_type_errors_are_rejected(self):
        with self.assertRaises(RestoreError):
            self.build(["/messages/1/content"], wire_format="gemini")
        with self.assertRaises(RestoreError):
            build_restore_manifest("not bytes", "openai_chat", [])
        with self.assertRaises(RestoreError):
            build_restore_manifest(self.raw, "openai_chat", "/messages/1/content")
        with self.assertRaises(RestoreError):
            build_restore_manifest(self.raw, "openai_chat", None)

    def test_duplicate_json_keys_and_nonfinite_values_are_rejected(self):
        duplicated = b'{"model":"m","model":"m2","messages":[{"role":"assistant","content":"x"},{"role":"user","content":"u"}]}'
        with self.assertRaises(RestoreError):
            build_restore_manifest(duplicated, "openai_chat", ["/messages/0/content"])
        nonfinite = (b'{"model":"m","temperature":NaN,"messages":'
                     b'[{"role":"assistant","content":"x"},{"role":"user","content":"u"}]}')
        with self.assertRaises(RestoreError):
            build_restore_manifest(nonfinite, "openai_chat", ["/messages/0/content"])

    def test_restore_rejects_unknown_missing_and_extra_pointers(self):
        raw, manifest = manifest_of(openai_body(), ["/messages/1/content"])
        reduced = canonical_bytes({**openai_body(), "messages": [
            openai_body()["messages"][0], openai_body()["messages"][2]]})
        with self.assertRaises(RestoreError) as caught:
            restore_request(reduced, manifest, {})
        self.assertIn("removed_segments_do_not_match_manifest", str(caught.exception))
        with self.assertRaises(RestoreError):
            restore_request(reduced, manifest, {"/messages/1/content": HISTORY_SENTINEL,
                                                "/messages/0/content": SYSTEM_SENTINEL})
        with self.assertRaises(RestoreError):
            restore_request(reduced, manifest, {"/messages/0/content": SYSTEM_SENTINEL})
        self.assertEqual(raw, canonical_bytes(openai_body()))

    def test_restore_rejects_non_mapping_and_non_string_keys(self):
        raw, manifest = manifest_of(openai_body(), ["/messages/1/content"])
        reduced = canonical_bytes({**openai_body(), "messages": [
            openai_body()["messages"][0], openai_body()["messages"][2]]})
        with self.assertRaises(RestoreError):
            restore_request(reduced, manifest, ["/messages/1/content"])
        with self.assertRaises(RestoreError):
            restore_request(reduced, manifest, {1: HISTORY_SENTINEL})

    def test_restore_rejects_a_manifest_with_a_poisoned_record_position(self):
        payload = manifest_dict(manifest_of(openai_body(), ["/messages/1/content"])[1])
        payload["records"][0]["message_index"] = 99
        reduced = canonical_bytes({**openai_body(), "messages": [
            openai_body()["messages"][0], openai_body()["messages"][2]]})
        with self.assertRaises(RestoreError) as caught:
            restore_request(reduced, payload, {"/messages/1/content": HISTORY_SENTINEL})
        self.assertIn("manifest_record_position_mismatch", str(caught.exception))


class WrapperAndHelperTest(unittest.TestCase):
    def test_normalize_removed_segments_keeps_values_and_rejects_bad_keys(self):
        self.assertEqual(normalize_removed_segments({"a": "x"}), {"a": "x"})
        message = {"role": "assistant", "content": "x"}
        self.assertEqual(normalize_removed_segments({"a": message}), {"a": message})
        with self.assertRaises(RestoreError):
            normalize_removed_segments({1: "x"})
        with self.assertRaises(RestoreError):
            normalize_removed_segments(["a"])

    def test_whole_message_object_form_restores_identically(self):
        body = openai_body()
        raw, manifest = manifest_of(body, ["/messages/1/content"])
        reduced = canonical_bytes({**body, "messages": [body["messages"][0], body["messages"][2]]})
        # Passing the retained original message object is accepted for a whole-message drop.
        self.assertEqual(restore_request(reduced, manifest,
                                         {"/messages/1/content": body["messages"][1]}), raw)

    def test_whole_message_object_form_with_extra_fields_is_refused(self):
        body = openai_body()
        raw, manifest = manifest_of(body, ["/messages/1/content"])
        reduced = canonical_bytes({**body, "messages": [body["messages"][0], body["messages"][2]]})
        with self.assertRaises(RestoreError):
            restore_request(reduced, manifest,
                            {"/messages/1/content": {**body["messages"][1], "name": "trailing"}})

    def test_whole_message_drop_rejects_non_text_segment(self):
        body = openai_body()
        manifest = build_restore_manifest(raw_of(body), "openai_chat", ["/messages/1/content"])
        reduced = raw_of({**body, "messages": [body["messages"][0], body["messages"][2]]})
        with self.assertRaises(RestoreError):
            restore_request(reduced, manifest, {"/messages/1/content": 7})

    def test_text_value_form_restores_identically(self):
        body = openai_body()
        raw, manifest = manifest_of(body, ["/messages/1/content"])
        reduced = canonical_bytes({**body, "messages": [body["messages"][0], body["messages"][2]]})
        self.assertEqual(restore_request(reduced, manifest,
                                         {"/messages/1/content": HISTORY_SENTINEL}), raw)

    def test_non_canonical_original_is_flagged_and_restores_canonically(self):
        body = openai_body()
        pretty = json.dumps(body, ensure_ascii=False, indent=2).encode("utf-8")
        manifest = build_restore_manifest(pretty, "openai_chat", ["/messages/1/content"])
        self.assertFalse(manifest.canonical_json)
        # The manifest identifies the received bytes and the restoration target separately.
        self.assertEqual(manifest.original_request_sha256, hashlib.sha256(pretty).hexdigest())
        self.assertEqual(manifest.restored_bytes_sha256,
                         hashlib.sha256(canonical_bytes(body)).hexdigest())
        self.assertNotEqual(manifest.original_request_sha256, manifest.restored_bytes_sha256)
        reduced = canonical_bytes({**body, "messages": [body["messages"][0], body["messages"][2]]})
        restored = restore_request(reduced, manifest, {"/messages/1/content": HISTORY_SENTINEL})
        self.assertEqual(restored, canonical_bytes(body))
        # verify_round_trip reports the truth instead of claiming byte-identity.
        result = verify_round_trip(pretty, reduced, manifest,
                                   {"/messages/1/content": HISTORY_SENTINEL})
        self.assertFalse(result["restored_bytes_identical"])
        self.assertFalse(result["canonical_original"])
        self.assertTrue(result["original_request_sha256_matches"])
        self.assertTrue(result["reduced_request_sha256_matches"])
        # A canonical original is byte-identical, and the manifest says so.
        canonical_manifest = build_restore_manifest(canonical_bytes(body), "openai_chat",
                                                    ["/messages/1/content"])
        self.assertTrue(canonical_manifest.canonical_json)
        identical = verify_round_trip(canonical_bytes(body), reduced, canonical_manifest,
                                      {"/messages/1/content": HISTORY_SENTINEL})
        self.assertTrue(identical["restored_bytes_identical"])


class LastMessageEmptiedRegressionTest(unittest.TestCase):
    """Regression: an emptied message at the END of the list must still round-trip.

    ``_dropped_message_indexes`` computed the group's position in the reduced list and
    raised whenever that position was past the end. But being exactly at the end is the
    signature of the FINAL message collapsing off the list after every one of its parts
    was dropped, so a reduction the caller could not restore was accepted as valid and
    forwarded. Found by the reviewer scanning message position x content shape without
    reusing the author's tests.
    """

    @staticmethod
    def _anthropic_body(roles):
        messages = []
        for index, role in enumerate(roles):
            if role == "assistant":
                messages.append({"role": "assistant",
                                 "content": [{"type": "text", "text": f"HIST{index}"}]})
            else:
                messages.append({"role": "user", "content": f"USER{index}"})
        return {"model": "synthetic-main-model", "messages": messages}

    def test_emptied_message_at_every_position_round_trips(self):
        cases = [(["assistant", "user"], 0),
                 (["user", "assistant"], 1),
                 (["user", "assistant", "user"], 1),
                 (["user", "user", "assistant"], 2),
                 (["user", "assistant", "user", "assistant"], 3)]
        for roles, drop_index in cases:
            with self.subTest(roles=roles):
                body = self._anthropic_body(roles)
                raw = canonical_bytes(body)
                pointer = f"/messages/{drop_index}/content/0/text"
                manifest = build_restore_manifest(raw, "anthropic_messages", [pointer])
                reduced = build_reduced_request(raw, "anthropic_messages", [pointer])
                restored = restore_request(reduced, manifest, {pointer: f"HIST{drop_index}"})
                self.assertEqual(restored, raw)

    def test_partial_drop_leaving_the_message_alive_still_round_trips(self):
        body = {"model": "synthetic-main-model", "messages": [
            {"role": "user", "content": "USER0"},
            {"role": "assistant", "content": [{"type": "text", "text": "HIST1"},
                                              {"type": "text", "text": "HIST1b"}]}]}
        raw = canonical_bytes(body)
        pointer = "/messages/1/content/0/text"
        manifest = build_restore_manifest(raw, "anthropic_messages", [pointer])
        reduced = build_reduced_request(raw, "anthropic_messages", [pointer])
        self.assertEqual(restore_request(reduced, manifest, {pointer: "HIST1"}), raw)


if __name__ == "__main__":
    unittest.main()
