#!/usr/bin/env python3
"""Reversible half of the decision/compression layer: content-free restore manifests.

The shadow core and the main-model gateway may *reduce* a request in active mode: an
eligible assistant text segment suggested for removal by the scored plan is omitted from
the bytes that go to the provider. The paired **remove-then-restore** operation is
byte-reversible, and this module is the whole of that machinery.

Ownership split (the load-bearing privacy rule):

* **The caller holds content.** Only the caller keeps the removed segment values (raw
  prompt text). It is the only party able to reconstruct the original request.
* **The gateway holds only hashes and pointers.** A manifest records the wire format, the
  original/reduced request hashes, and per-pointer records with the removed segment's hash
  and its exact position. It contains no raw prompt text, and it is not encryption: a
  hash of a short or guessable segment can be dictionary-attacked, so a manifest is
  exactly as sensitive as a digest of the request.

Restoring is therefore a *caller-driven* operation: the caller passes back the removed
segments it retained, and :func:`restore_request` places each one at the recorded position
and proves the result hashes to the recorded original request. Nothing here ever receives,
persists, logs, or transmits original prompt text on the gateway's side.

Byte-exactness contract: restore reconstructs the original request object and returns its
canonical serialization (UTF-8 JSON, ``ensure_ascii=False``, ``separators=(",", ":")``, key
order preserved, non-finite constants rejected). Two hashes make the guarantee precise
instead of assumed:

* ``original_request_sha256`` is the hash of the exact bytes the caller sent. It identifies
  the request and lets a verifier confirm the manifest belongs to it.
* ``restored_bytes_sha256`` is the hash of the canonical serialization of that same request
  -- the exact bytes :func:`restore_request` returns. A reconstruction is only accepted when
  it hashes to this value.

When the original request was itself canonical (``canonical_json`` is ``true``), the two
hashes are equal and the round-trip is byte-identical to the original. When it was not (for
example an indented client body), the reconstruction is byte-identical to the canonical
form of the original, which is reported as ``restored_bytes_identical: false`` by
:func:`verify_round_trip` rather than being claimed as identity. Nothing is lost or
reordered either way: the reconstructed object re-serializes to ``restored_bytes_sha256``,
and re-applying the recorded reduction to it reproduces the reduced bytes exactly.

Malformed, out-of-range, duplicated, unknown, or tampered manifests/segments always raise
:class:`RestoreError`; this module never silently returns a wrong reconstruction.
Standard library only.
"""

from dataclasses import dataclass, field
import hashlib
import json
import re

from predict_toy_decisions import reject_nonfinite, unique_object


SCHEMA_VERSION = "nanojev-context-restore-v1"

WIRE_FORMATS = ("openai_chat", "openai_responses", "anthropic_messages")

# The exact pointer grammar the reduction applicator can produce. Anything else is
# rejected: a manifest is never allowed to point at a shape this module cannot restore.
DROP_POINTER = re.compile(r"^/(messages|input)/(\d+)/content(?:/(\d+)/text)?$")

MESSAGE_KEY_BY_FORMAT = {
    "openai_chat": "messages",
    "anthropic_messages": "messages",
    "openai_responses": "input",
}

MAX_SEGMENT_INDEX = 4096

KIND_MESSAGE = "message"
KIND_TEXT_PART = "text_part"

_HEX64 = re.compile(r"^[0-9a-f]{64}$")

_MANIFEST_FIELDS = {"schema_version", "wire_format", "message_key", "canonical_json",
                    "original_request_sha256", "restored_bytes_sha256",
                    "reduced_request_sha256", "applied", "dropped_segment_count", "records"}
_RECORD_FIELDS = {"pointer", "kind", "role", "message_key", "message_index", "part_index",
                  "segment_sha256"}


class RestoreError(ValueError):
    """The manifest or the caller-supplied segments cannot restore the original request.

    Subclasses :class:`ValueError` so callers may catch either. Every failure mode in this
    module raises this type; a wrong reconstruction is never returned silently.
    """


@dataclass(frozen=True)
class DropRecord:
    """One removed segment: a pointer plus the hash and position needed to put it back."""

    pointer: str
    kind: str
    role: str
    message_key: str
    message_index: int
    part_index: int | None
    segment_sha256: str

    def as_dict(self):
        return {
            "pointer": self.pointer,
            "kind": self.kind,
            "role": self.role,
            "message_key": self.message_key,
            "message_index": self.message_index,
            "part_index": self.part_index,
            "segment_sha256": self.segment_sha256,
        }


@dataclass(frozen=True)
class RestoreManifest:
    """Content-free description of what was removed. Carries pointers and hashes only."""

    schema_version: str
    wire_format: str
    message_key: str
    canonical_json: bool
    original_request_sha256: str
    restored_bytes_sha256: str
    reduced_request_sha256: str
    applied: bool
    dropped_segment_count: int
    records: tuple = field(default_factory=tuple)

    def as_dict(self):
        return {
            "schema_version": self.schema_version,
            "wire_format": self.wire_format,
            "message_key": self.message_key,
            "canonical_json": self.canonical_json,
            "original_request_sha256": self.original_request_sha256,
            "restored_bytes_sha256": self.restored_bytes_sha256,
            "reduced_request_sha256": self.reduced_request_sha256,
            "applied": self.applied,
            "dropped_segment_count": self.dropped_segment_count,
            "records": [record.as_dict() for record in self.records],
        }

    def to_json(self):
        """Compact JSON with no raw prompt text; safe to place in a response header."""
        return serialized(self.as_dict())

    @property
    def pointers(self):
        return tuple(record.pointer for record in self.records)


def serialized(value):
    """Canonical content-free JSON encoding, identical to the gateway's encoding."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def canonical_bytes(value):
    return serialized(value).encode("utf-8")


def _hash_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _hash_value(value):
    return _hash_bytes(canonical_bytes(value))


def _hash_text(text):
    """SHA-256 of the removed segment's UTF-8 text; the only content-derived value kept."""
    return _hash_bytes(text.encode("utf-8"))


def _parse_json(raw, what):
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object,
                          parse_constant=reject_nonfinite)
    except (ValueError, UnicodeError, AttributeError, TypeError):
        raise RestoreError(f"{what}_is_not_valid_json") from None


def _is_hex64(value):
    return isinstance(value, str) and _HEX64.match(value) is not None


def _parse_pointer(pointer, wire_format):
    """Return ``(kind, message_key, message_index, part_index)`` or raise."""
    if not isinstance(pointer, str):
        raise RestoreError("pointer_not_a_string")
    match = DROP_POINTER.match(pointer)
    if match is None:
        raise RestoreError("unsupported_drop_pointer")
    message_key = match.group(1)
    if message_key != MESSAGE_KEY_BY_FORMAT[wire_format]:
        raise RestoreError("drop_pointer_wrong_wire_key")
    message_index = int(match.group(2))
    raw_part = match.group(3)
    if message_index > MAX_SEGMENT_INDEX:
        raise RestoreError("drop_pointer_out_of_range")
    if raw_part is None:
        return KIND_MESSAGE, message_key, message_index, None
    part_index = int(raw_part)
    if part_index > MAX_SEGMENT_INDEX:
        raise RestoreError("drop_pointer_out_of_range")
    return KIND_TEXT_PART, message_key, message_index, part_index


def _message_list(body, message_key):
    messages = body.get(message_key)
    if not isinstance(messages, list) or not messages:
        raise RestoreError("missing_or_empty_message_list")
    return messages


def _apply_drops(body, message_key, records):
    """Rebuild the reduced envelope exactly as the gateway's applicator does."""
    items = _message_list(body, message_key)
    whole_messages = {record.message_index for record in records if record.kind == KIND_MESSAGE}
    part_indexes = {}
    for record in records:
        if record.kind == KIND_TEXT_PART:
            part_indexes.setdefault(record.message_index, set()).add(record.part_index)
    reduced = []
    for index, item in enumerate(items):
        if index in whole_messages:
            continue
        dropped_parts = part_indexes.get(index)
        if dropped_parts:
            if not isinstance(item, dict) or not isinstance(item.get("content"), list):
                raise RestoreError("drop_pointer_target_not_a_list")
            kept = [part for position, part in enumerate(item["content"]) if position not in dropped_parts]
            if not kept:
                continue  # an emptied message is removed whole
            item = {**item, "content": kept}
        reduced.append(item)
    if not reduced:
        raise RestoreError("reduced_request_would_be_empty")
    if not any(isinstance(item, dict) and item.get("role") == "user" for item in reduced):
        raise RestoreError("reduced_request_would_lose_user_intent")
    return {**body, message_key: reduced}


# --------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------

def build_restore_manifest(raw, wire_format, drop_pointers):
    """Build a content-free manifest for ``drop_pointers`` applied to ``raw``.

    ``raw`` is the ORIGINAL request bytes; ``drop_pointers`` are the pointers actually
    removed from it (as produced by the gateway's removal-plan applicator). The manifest
    records what the caller must retain, plus the hash of the exact reduced body the same
    plan produces, so the caller's later reconstruction can be verified.

    The returned manifest contains no raw prompt text: it never receives or stores the
    removed segment values, only their hashes and positions. Raises :class:`RestoreError`
    for a malformed request, an unsupported wire format, or a pointer that is duplicated,
    out of range, of the wrong message key, or outside the allowed grammar.
    """
    if not isinstance(raw, bytes):
        raise RestoreError("raw_request_must_be_bytes")
    if wire_format not in WIRE_FORMATS:
        raise RestoreError("unsupported_wire_format")
    if drop_pointers is None:
        raise RestoreError("drop_pointers_required")
    if isinstance(drop_pointers, (str, bytes)) or not isinstance(drop_pointers, (list, tuple)):
        raise RestoreError("drop_pointers_must_be_a_sequence")

    body = _parse_json(raw, "raw_request")
    if not isinstance(body, dict):
        raise RestoreError("invalid_request_envelope")
    message_key = MESSAGE_KEY_BY_FORMAT[wire_format]
    items = _message_list(body, message_key)

    records = []
    seen_pointers = set()
    seen_positions = set()
    whole_indexes = set()
    part_indexes = set()
    parsed_pointers = []
    for pointer in drop_pointers:
        if not isinstance(pointer, str):
            raise RestoreError("pointer_not_a_string")
        kind, pointer_key, message_index, part_index = _parse_pointer(pointer, wire_format)
        if pointer in seen_pointers:
            raise RestoreError("duplicate_drop_pointer")
        seen_pointers.add(pointer)
        if message_index >= len(items):
            raise RestoreError("drop_pointer_out_of_range")
        if kind == KIND_MESSAGE:
            whole_indexes.add(message_index)
        else:
            part_indexes.add(message_index)
        parsed_pointers.append((pointer, kind, pointer_key, message_index, part_index))
    # Structural conflicts are decided before any content validation, so a malformed plan
    # always receives the pointer-level reason regardless of pointer order.
    if whole_indexes & part_indexes:
        raise RestoreError("conflicting_drop_positions")
    for pointer, kind, pointer_key, message_index, part_index in parsed_pointers:
        position = (message_index, part_index)
        if position in seen_positions:
            raise RestoreError("conflicting_drop_positions")
        seen_positions.add(position)
        message = items[message_index]
        if not isinstance(message, dict):
            raise RestoreError("drop_pointer_target_not_an_object")
        role = message.get("role")
        if not isinstance(role, str) or not role:
            raise RestoreError("drop_pointer_target_has_no_role")
        content = message.get("content")
        if kind == KIND_MESSAGE:
            if not isinstance(content, str):
                # Whole-message removal is only reversible when the removed segment is a
                # single text segment. V1 never proposes a mixed-content message.
                raise RestoreError("whole_message_drop_requires_string_content")
            segment_text = content
        else:
            if not isinstance(content, list) or part_index >= len(content):
                raise RestoreError("drop_pointer_out_of_range")
            part = content[part_index]
            if not isinstance(part, dict) or not isinstance(part.get("text"), str):
                raise RestoreError("drop_pointer_target_is_not_a_text_part")
            segment_text = part["text"]
        records.append(DropRecord(
            pointer=pointer, kind=kind, role=role, message_key=pointer_key,
            message_index=message_index, part_index=part_index,
            segment_sha256=_hash_text(segment_text)))

    reduced_body = _apply_drops(body, message_key, records)
    return RestoreManifest(
        schema_version=SCHEMA_VERSION,
        wire_format=wire_format,
        message_key=message_key,
        canonical_json=(canonical_bytes(body) == raw),
        original_request_sha256=_hash_bytes(raw),
        restored_bytes_sha256=_hash_bytes(canonical_bytes(body)),
        reduced_request_sha256=_hash_value(reduced_body),
        applied=bool(records),
        dropped_segment_count=len(records),
        records=tuple(records),
    )


# --------------------------------------------------------------------------------------
# Manifest ingestion
# --------------------------------------------------------------------------------------

def _coerce_manifest(manifest):
    """Accept a :class:`RestoreManifest`, or its ``as_dict`` / JSON form, strictly."""
    if isinstance(manifest, RestoreManifest):
        return manifest
    if isinstance(manifest, (bytes, bytearray)):
        manifest = _parse_json(bytes(manifest), "manifest")
    elif isinstance(manifest, str):
        manifest = _parse_json(manifest.encode("utf-8"), "manifest")
    if not isinstance(manifest, dict):
        raise RestoreError("manifest_must_be_an_object")
    return _manifest_from_dict(manifest)


def _manifest_from_dict(data):
    if set(data) != _MANIFEST_FIELDS:
        raise RestoreError("unknown_or_missing_manifest_fields")
    if data["schema_version"] != SCHEMA_VERSION:
        raise RestoreError("unsupported_manifest_schema_version")
    wire_format = data["wire_format"]
    if wire_format not in WIRE_FORMATS:
        raise RestoreError("unsupported_wire_format")
    if data["message_key"] != MESSAGE_KEY_BY_FORMAT[wire_format]:
        raise RestoreError("manifest_message_key_mismatch")
    if type(data["canonical_json"]) is not bool or type(data["applied"]) is not bool:
        raise RestoreError("invalid_manifest_flag")
    for name in ("original_request_sha256", "restored_bytes_sha256", "reduced_request_sha256"):
        if not _is_hex64(data[name]):
            raise RestoreError("invalid_manifest_hash")
    if type(data["dropped_segment_count"]) is not int or data["dropped_segment_count"] < 0:
        raise RestoreError("invalid_manifest_count")
    raw_records = data["records"]
    if not isinstance(raw_records, list):
        raise RestoreError("manifest_records_must_be_a_list")
    if len(raw_records) != data["dropped_segment_count"]:
        raise RestoreError("manifest_count_mismatch")
    if data["applied"] != bool(raw_records):
        raise RestoreError("manifest_applied_flag_mismatch")

    records = []
    seen_pointers = set()
    seen_positions = set()
    for raw_record in raw_records:
        if not isinstance(raw_record, dict):
            raise RestoreError("malformed_manifest_record")
        if set(raw_record) != _RECORD_FIELDS:
            raise RestoreError("unknown_or_missing_record_fields")
        kind, message_key, message_index, part_index = _parse_pointer(raw_record["pointer"], wire_format)
        if raw_record["kind"] != kind:
            raise RestoreError("manifest_record_kind_mismatch")
        if raw_record["message_key"] != message_key:
            raise RestoreError("manifest_record_message_key_mismatch")
        if type(raw_record["message_index"]) is not int or raw_record["message_index"] != message_index:
            raise RestoreError("manifest_record_position_mismatch")
        if raw_record["part_index"] != part_index:
            raise RestoreError("manifest_record_position_mismatch")
        if not isinstance(raw_record["role"], str) or not raw_record["role"]:
            raise RestoreError("invalid_manifest_role")
        if not _is_hex64(raw_record["segment_sha256"]):
            raise RestoreError("invalid_manifest_hash")
        pointer = raw_record["pointer"]
        if pointer in seen_pointers:
            raise RestoreError("duplicate_drop_pointer")
        seen_pointers.add(pointer)
        position = (message_index, part_index)
        if position in seen_positions:
            raise RestoreError("conflicting_drop_positions")
        seen_positions.add(position)
        records.append(DropRecord(
            pointer=pointer, kind=kind, role=raw_record["role"], message_key=message_key,
            message_index=message_index, part_index=part_index,
            segment_sha256=raw_record["segment_sha256"]))
    return RestoreManifest(
        schema_version=SCHEMA_VERSION, wire_format=wire_format, message_key=data["message_key"],
        canonical_json=data["canonical_json"],
        original_request_sha256=data["original_request_sha256"],
        restored_bytes_sha256=data["restored_bytes_sha256"],
        reduced_request_sha256=data["reduced_request_sha256"],
        applied=data["applied"], dropped_segment_count=data["dropped_segment_count"],
        records=tuple(records))


# --------------------------------------------------------------------------------------
# Restore
# --------------------------------------------------------------------------------------

def normalize_removed_segments(removed_segments):
    """Validate the caller's retained segments without copying or transforming them.

    Expects ``{pointer: removed_segment}`` where the value is the removed segment's text,
    or (for a whole-message removal) the original message object. Every key must be a
    string pointer; no value is interpreted here.
    """
    if not isinstance(removed_segments, dict):
        raise RestoreError("removed_segments_must_be_a_mapping")
    normalized = {}
    for pointer, entry in removed_segments.items():
        if not isinstance(pointer, str):
            raise RestoreError("removed_segment_pointer_not_a_string")
        normalized[pointer] = entry
    return normalized


def normalized_segment_texts(parsed, supplied):
    """Map each pointer to the removed text, enforcing the value's shape and role.

    The caller may supply the removed text directly, or the original single-text message
    object for a whole-message removal. A message object must carry exactly ``role`` and
    ``content``, its role must match the recorded role, and its content must be the
    recorded text. Nothing is copied into the manifest.
    """
    texts = {}
    for record in parsed.records:
        value = supplied[record.pointer]
        if isinstance(value, str):
            texts[record.pointer] = value
            continue
        if isinstance(value, dict):
            if set(value) != {"role", "content"} or not isinstance(value.get("content"), str):
                raise RestoreError(f"removed_segment_has_no_string_content:{record.pointer}")
            if value["role"] != record.role:
                raise RestoreError(f"removed_segment_role_mismatch:{record.pointer}")
            texts[record.pointer] = value["content"]
            continue
        raise RestoreError(f"removed_segment_is_not_text:{record.pointer}")
    return texts


def _merge_content(content, part_records, supplied):
    """Reinsert dropped text parts at their recorded indexes, keeping the rest in order.

    The kept content is walked in order and interleaved with the retained parts; a recorded
    index that cannot be reached exactly this way is rejected instead of producing a
    shifted (silently wrong) reconstruction.
    """
    merged = []
    kept_position = 0
    for position in range(len(content) + len(part_records)):
        record = part_records.get(position)
        if record is not None:
            merged.append({"type": "text", "text": supplied[record.pointer]})
            continue
        if kept_position >= len(content):
            raise RestoreError("drop_pointer_out_of_range")
        merged.append(content[kept_position])
        kept_position += 1
    if kept_position != len(content):
        raise RestoreError("drop_pointer_out_of_range")
    return merged


def _dropped_message_indexes(parsed, reduced_body):
    """Exactly which message indexes the reduction removed from the original list.

    Derived from the manifest's record set alone, not guessed from content shape:

    * every whole-message drop removes one message;
    * each part group whose every original part was dropped leaves an empty message, which
      the reduction removes whole. The original part count is ``kept + dropped``, where
      ``kept`` is the number of parts that survive in the caller's reduced message.
    """
    message_key = parsed.message_key
    reduced_items = _message_list(reduced_body, message_key)
    whole = {record.message_index for record in parsed.records if record.kind == KIND_MESSAGE}
    parts = {}
    for record in parsed.records:
        if record.kind == KIND_TEXT_PART:
            parts.setdefault(record.message_index, {})[record.part_index] = record
    dropped = set(whole)
    for index in sorted(parts):
        # Position of this message in the caller's reduced list: it follows every message
        # the reduction already removed at a lower original index.
        position = index - sum(1 for dropped_index in dropped if dropped_index < index)
        if position > len(reduced_items):
            raise RestoreError("reduced_bytes_do_not_match_manifest")
        if position == len(reduced_items):
            # The group sits exactly past the end of the reduced list, so this message has
            # no surviving counterpart and the reduction removed it whole. This is the
            # last-message case: when the final message's every part is dropped, it empties
            # and collapses off the end, leaving nothing at its computed position to
            # inspect. Treating it as an inconsistency was a bug that made a reduction
            # the caller could not restore look valid.
            dropped.add(index)
            continue
        item = reduced_items[position]
        content = item.get("content") if isinstance(item, dict) else None
        if not isinstance(content, list):
            # This original message has no surviving list-content counterpart, so the
            # reduction removed it whole after every one of its parts was dropped.
            dropped.add(index)
    return dropped


def _rebuild_original(parsed, reduced_body, supplied):
    """Insert the caller's retained segments at their recorded positions.

    ``supplied`` maps each pointer to the removed segment's text.

    The original message list is reconstructed by walking original indexes and consuming
    the caller's reduced messages in order. Dropping a whole message collapses indexes to
    the left, so the reduced list is deliberately NOT indexed by original position.
    """
    message_key = parsed.message_key
    reduced_items = _message_list(reduced_body, message_key)
    whole = {record.message_index: record for record in parsed.records
             if record.kind == KIND_MESSAGE}
    parts = {}
    for record in parsed.records:
        if record.kind == KIND_TEXT_PART:
            parts.setdefault(record.message_index, {})[record.part_index] = record
    if set(parts) & set(whole):
        raise RestoreError("conflicting_drop_positions")

    dropped = _dropped_message_indexes(parsed, reduced_body)
    if not dropped <= (set(whole) | set(parts)):
        raise RestoreError("reduced_bytes_do_not_match_manifest")
    total_original = len(reduced_items) + len(dropped)

    rebuilt = []
    kept_cursor = 0
    for original_index in range(total_original):
        if original_index in dropped:
            whole_record = whole.get(original_index)
            if whole_record is not None:
                rebuilt.append({"role": whole_record.role,
                                "content": supplied[whole_record.pointer]})
                continue
            # Emptied message: every original part was dropped and retained by the caller.
            part_records = parts[original_index]
            merged = [{"type": "text", "text": supplied[record.pointer]}
                      for _, record in sorted(part_records.items())]
            role = next(iter(part_records.values())).role
            rebuilt.append({"role": role, "content": merged})
            continue
        if kept_cursor >= len(reduced_items):
            raise RestoreError("reduced_bytes_do_not_match_manifest")
        item = reduced_items[kept_cursor]
        part_records = parts.get(original_index)
        if not part_records:
            rebuilt.append(item)
        else:
            if not isinstance(item, dict):
                raise RestoreError("drop_pointer_target_not_an_object")
            content = item.get("content")
            if not isinstance(content, list) or not content:
                raise RestoreError("drop_pointer_target_not_a_list")
            rebuilt.append({**item, "content": _merge_content(content, part_records, supplied)})
        kept_cursor += 1
    if kept_cursor != len(reduced_items):
        raise RestoreError("reduced_bytes_do_not_match_manifest")
    return {**reduced_body, message_key: rebuilt}


def restore_request(reduced, manifest, removed_segments):
    """Reconstruct the ORIGINAL request bytes from the reduced bytes and retained segments.

    The caller supplies the removed segment content it always retained; the gateway never
    persists it. ``removed_segments`` maps each manifest pointer to the removed segment's
    text, or to the original single-text message object for a whole-message removal. Every
    value is verified against the manifest's recorded hash and role, inserted at the
    recorded position, re-validated with the core parser, and the reconstruction is proven
    to hash to the manifest's restoration target before it is returned.

    Raises :class:`RestoreError` for a malformed or tampered manifest, reduced bytes that
    are not the exact reduced request the manifest describes, an unknown, missing, or extra
    pointer, an out-of-range position, a segment that does not match its recorded hash or
    role, or a reconstruction that does not reproduce the recorded hashes.

    Byte-identity note: the returned bytes are the canonical serialization of the original
    request, proven against ``manifest.restored_bytes_sha256``. They are byte-identical to
    the bytes the caller originally sent when ``manifest.canonical_json`` is true; otherwise
    they are the canonical equivalent of that request (see :func:`verify_round_trip`).
    """
    parsed = _coerce_manifest(manifest)
    if not isinstance(reduced, bytes):
        raise RestoreError("reduced_request_must_be_bytes")
    reduced_body = _parse_json(reduced, "reduced_request")
    if not isinstance(reduced_body, dict):
        raise RestoreError("invalid_request_envelope")

    supplied = normalize_removed_segments(removed_segments)
    expected_pointers = set(parsed.pointers)
    if set(supplied) != expected_pointers:
        missing = sorted(expected_pointers - set(supplied))
        extra = sorted(set(supplied) - expected_pointers)
        raise RestoreError(f"removed_segments_do_not_match_manifest:{missing}:{extra}")
    if not parsed.applied:
        if supplied:
            raise RestoreError("no_op_manifest_expects_no_removed_segments")
        if reduced != canonical_bytes(reduced_body):
            raise RestoreError("reduced_request_is_not_canonical")
        if _hash_bytes(reduced) != parsed.restored_bytes_sha256:
            raise RestoreError("restored_request_hash_mismatch")
        return reduced

    # The caller's reduced bytes must be the exact reduced request this manifest describes.
    if _hash_bytes(reduced) != parsed.reduced_request_sha256:
        raise RestoreError("reduced_bytes_do_not_match_manifest")

    texts = normalized_segment_texts(parsed, supplied)
    for record in parsed.records:
        if _hash_bytes(texts[record.pointer].encode("utf-8")) != record.segment_sha256:
            raise RestoreError(f"removed_segment_hash_mismatch:{record.pointer}")

    rebuilt = _rebuild_original(parsed, reduced_body, texts)
    rebuilt_bytes = canonical_bytes(rebuilt)
    # The recorded reduction applied to the reconstruction must reproduce the caller's
    # reduced bytes byte-for-byte. This is what proves the supplied segments belong to
    # this pair, that no already-present segment was duplicated, and that an emptied
    # message was restored the same way the gateway would have removed it.
    try:
        re_reduced = canonical_bytes(_apply_drops(rebuilt, parsed.message_key, parsed.records))
    except RestoreError:
        raise RestoreError("reduced_bytes_do_not_match_manifest") from None
    if re_reduced != reduced:
        raise RestoreError("reduced_bytes_do_not_match_manifest")
    # The core parser is the authority on structural validity; a reconstruction the gate
    # itself could not have parsed is refused rather than returned.
    try:
        from context_gate_v1 import parse_segments
        parse_segments(rebuilt_bytes, parsed.wire_format)
    except Exception:  # noqa: BLE001 - any core refusal means the reconstruction is unusable
        raise RestoreError("restored_request_failed_core_validation") from None
    if _hash_bytes(rebuilt_bytes) != parsed.restored_bytes_sha256:
        raise RestoreError("restored_request_hash_mismatch")
    return rebuilt_bytes


def verify_round_trip(raw, reduced, manifest, removed_segments=None):
    """Explicitly verify a remove/restore round-trip and return a structured result.

    * With ``removed_segments`` the reconstruction is performed and the result is checked
      against the recorded hashes; byte-identity to ``raw`` is reported truthfully (it holds
      when ``raw`` was canonical, which ``canonical_original`` states).
    * Without ``removed_segments`` the manifest's original/reduced hashes are checked
      against ``raw``/``reduced`` only; no reconstruction claim is made.

    Raises :class:`RestoreError` on tamper or mismatch. Returns a result dict on success.
    """
    if not isinstance(raw, bytes) or not isinstance(reduced, bytes):
        raise RestoreError("round_trip_requires_bytes")
    parsed = _coerce_manifest(manifest)
    result = {
        "schema_version": SCHEMA_VERSION,
        "wire_format": parsed.wire_format,
        "applied": parsed.applied,
        "dropped_segment_count": parsed.dropped_segment_count,
        "canonical_original": parsed.canonical_json,
        "original_request_sha256_matches": _hash_bytes(raw) == parsed.original_request_sha256,
        "reduced_request_sha256_matches": _hash_bytes(reduced) == parsed.reduced_request_sha256,
        "reconstruction_performed": removed_segments is not None,
        "reconstructed_sha256": None,
        "restored_bytes_identical": None,
    }
    if not result["original_request_sha256_matches"]:
        raise RestoreError("original_request_sha256_mismatch")
    if not result["reduced_request_sha256_matches"]:
        raise RestoreError("reduced_request_sha256_mismatch")
    if parsed.applied != bool(parsed.records):
        raise RestoreError("manifest_applied_flag_mismatch")
    if removed_segments is None:
        return result
    if not parsed.applied:
        # Nothing was removed: the reduced bytes ARE the original bytes.
        if reduced != raw:
            raise RestoreError("no_op_round_trip_requires_identical_bytes")
        reconstructed = reduced
    else:
        reconstructed = restore_request(reduced, parsed, removed_segments)
    result["reconstructed_sha256"] = _hash_bytes(reconstructed)
    result["restored_bytes_identical"] = reconstructed == raw
    return result


__all__ = [
    "DropRecord", "RestoreError", "RestoreManifest", "SCHEMA_VERSION", "WIRE_FORMATS",
    "build_restore_manifest", "canonical_bytes", "normalize_removed_segments",
    "restore_request", "serialized", "verify_round_trip",
]
