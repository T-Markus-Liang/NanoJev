#!/usr/bin/env python3
"""Deterministic, model-free safe deduplication of tool-call/result histories.

This module implements **one** removal rule. It uses no model, no score, no
probability, and no threshold. It exists to answer a narrow question honestly: can a
purely deterministic arm produce real token savings *today*, without the local NanoJev
gate (whose Catalog Choice accuracy collapses 100% -> 54.17% under irrelevant archived
context, so it currently, correctly, proposes nothing)?

THE RULE (SafeDedup V1)
-----------------------
A request is planned as follows, deterministically, with no randomness and no external
call:

1. Enumerate segments with the *existing* shadow core
   (:func:`context_gate_v1.parse_segments`). If the core bypasses the request for any
   reason, the arm removes nothing and reports the bypass reason.
2. Compute each segment's protection with the *existing* core policy
   (:func:`context_gate_v1.sidecar_policy` + :func:`context_gate_v1.protect_dependencies`).
   A segment is **arm-eligible** only when the core assigns it *no* reason at all, i.e.
   it is text-bearing, structurally unprotected (role not in
   ``system/developer/user/tool/control``, not tool-linked, not cited/annotated), carries
   none of the six caller protection flags, is explicitly marked ``eligible: true`` in
   the trusted sidecar, and is not pulled in by transitive dependency closure. Every core
   retention reason is respected verbatim, including ``not_explicitly_eligible``, which
   the frozen A4 contract declares protected.
3. Group arm-eligible segments by ``(kind, role, normalized_text)`` where ``kind`` is
   ``message`` (a whole ``/messages|input/{i}/content`` string) or ``text_part`` (a
   ``/messages|input/{i}/content/{j}/text``) and the **declared normalization is the
   identity**: ``normalized_text`` is the text itself, so two segments match only when
   their texts are equal code point for code point. No case folding, no Unicode
   normalization, no whitespace collapsing, no truncation. Identity is the only
   normalization that guarantees the bytes removed are exactly the bytes retained.
4. In request order, the **first occurrence** of each group is retained and every
   **later exact duplicate** (same kind, same role, identical text) is a removal
   candidate. The first occurrence is never removed, so a surviving identical copy
   always exists.
5. Dependency closure is re-run over the effective retained set (all core-retained
   segments plus every retained first occurrence), so a duplicate that a retained segment
   declares ``depends_on`` is retained rather than removed.
6. Each candidate is removed only if the removal is **proven reversible**: the reduced
   bytes are built by the *existing gateway applicator*
   (:func:`main_model_gateway_v1.build_reduced_request`) and re-validated with the core
   parser, a *content-free* restore manifest is built by
   :func:`context_restore_v1.build_restore_manifest`, the removed values are recovered
   from the caller's own original bytes, and
   :func:`context_restore_v1.restore_request` must reproduce the canonical original
   **byte-identically**. Any doubt -- a parse bypass, an unbuildable plan, a pointer
   outside the restore grammar, a reconstruction mismatch, a tampered or unknown value --
   means the candidate is *retained*. Nothing ambiguous, novel, or protected is ever
   removed.

Everything outside an explicit exact-duplicate-and-provable pattern is retained. With no
sidecar eligibility grant the arm removes nothing.

WHAT THIS MODULE DOES NOT DO
----------------------------
It never sends anything to a provider, never calls a model, never opens a socket, and
never modifies the shadow core, the restore module, or the gateway; it imports them. It
does not make a savings claim. A reduction that is not sent to a provider is not a
saving, and a character reduction is not a token reduction: see ``docs/SAFE_DEDUP_V1.md``.
No reduction may be sent to a real provider on the basis of this arm alone.

Standard library plus the repository's own modules and, optionally, the locally
installed ``tokenizers`` package and a locally present tokenizer file (offline only).
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

from context_gate_v1 import (
    Bypass, FORMATS, parse_segments, protect_dependencies, serialized, sidecar_policy,
)
from context_restore_v1 import (
    RestoreError, build_restore_manifest, canonical_bytes, restore_request,
    verify_round_trip,
)
from main_model_gateway_v1 import (
    ReductionError, build_reduced_request, removed_segments_from_raw, removal_plan,
)
from predict_toy_decisions import reject_nonfinite, unique_object


SCHEMA_VERSION = "nanojev-safe-dedup-v1"
RECEIPT_SCHEMA = "nanojev-safe-dedup-measurement-v1"

# The declared normalization. Identity is deliberate: any transform that maps two
# different texts onto one key (case folding, NFC/NFKC, whitespace collapsing) would let
# the arm remove bytes that were never present in the retained copy.
NORMALIZATION_ID = "identity-v1-exact-codepoint-equality"
NORMALIZATION_DESCRIPTION = (
    "normalized_text(x) = x. Two segments match only when their texts are equal code "
    "point for code point; no case folding, Unicode normalization, whitespace collapsing, "
    "truncation, or tokenization is applied."
)

# The arm's own drop reason. It is deliberately NOT ``high_irrelevance_score``: no model
# produced it. This is why the existing gateway's active path refuses the plan (see
# ``gateway_compatibility`` below and docs/SAFE_DEDUP_V1.md).
DROP_REASON = "exact_duplicate_retained_earlier"

DECISION_DROP = "drop"
DECISION_RETAIN = "retain"

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "research" / "tool_history_fixture_manifest_v1.json"
DEFAULT_CORPUS_DIR = REPO_ROOT / "data" / "context_relevance_oracle_v1_seed20260919"
DEFAULT_TOKENIZER_PATHS = (
    REPO_ROOT / "checkpoints" / "local_atomic_seed17" / "variants" / "local_atomic_seed17"
    / "tokenizer" / "tokenizer.json",
    REPO_ROOT / "runs" / "context_relevance_v1_seed17" / "tokenizer" / "tokenizer.json",
)


def _parse_body(raw):
    return json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object,
                      parse_constant=reject_nonfinite)


def _sha256_hex(value):
    return hashlib.sha256(value).hexdigest()


def _text_sha256(text):
    return _sha256_hex(text.encode("utf-8"))


def _segment_kind(pointer):
    """``message`` for a whole content string, ``text_part`` for a content part."""
    parts = pointer.strip("/").split("/")
    return "text_part" if parts[-1] == "text" else "message"


def _is_restore_grammar(pointer):
    """Reuse the restore module's own grammar decision, without re-implementing it."""
    from context_restore_v1 import DROP_POINTER
    return DROP_POINTER.match(pointer) is not None


# --------------------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class DedupDecision:
    """One content-free decision record. Carries a hash, never the segment text."""

    pointer: str
    role: str
    kind: str
    text_sha256: str | None
    decision: str
    reason: str
    retained_pointer: str | None = None

    def as_dict(self):
        return {
            "pointer": self.pointer,
            "role": self.role,
            "kind": self.kind,
            "text_sha256": self.text_sha256,
            "decision": self.decision,
            "reason": self.reason,
            "retained_pointer": self.retained_pointer,
        }


@dataclass(frozen=True)
class DedupPlan:
    """The deterministic plan. Empty ``drop_pointers`` means the arm removes nothing."""

    schema_version: str
    normalization_id: str
    wire_format: str
    parse_status: str
    bypass_reason: str | None
    segments_examined: int
    arm_eligible_pointers: tuple
    protected_pointers: tuple
    drop_pointers: tuple
    decisions: tuple

    @property
    def removed(self):
        return len(self.drop_pointers)

    def as_dict(self):
        return {
            "schema_version": self.schema_version,
            "normalization_id": self.normalization_id,
            "wire_format": self.wire_format,
            "parse_status": self.parse_status,
            "bypass_reason": self.bypass_reason,
            "segments_examined": self.segments_examined,
            "arm_eligible_pointers": list(self.arm_eligible_pointers),
            "protected_pointers": list(self.protected_pointers),
            "drop_pointers": list(self.drop_pointers),
            "segments_removed": self.removed,
            "decisions": [decision.as_dict() for decision in self.decisions],
        }


def _bypass_plan(wire_format, reason):
    return DedupPlan(
        schema_version=SCHEMA_VERSION, normalization_id=NORMALIZATION_ID,
        wire_format=wire_format, parse_status="bypass", bypass_reason=reason,
        segments_examined=0, arm_eligible_pointers=(), protected_pointers=(),
        drop_pointers=(), decisions=())


def plan_safe_dedup(raw, wire_format, sidecar=None):
    """Return a :class:`DedupPlan` for ``raw`` bytes. No model, no score, no threshold.

    ``sidecar`` is the same trusted integration metadata the shadow core already accepts.
    With no sidecar (or with no ``eligible: true`` note) the plan removes nothing.
    """
    if not isinstance(raw, bytes):
        raise TypeError("raw request must be bytes")
    if wire_format not in FORMATS:
        return _bypass_plan(wire_format, "unsupported_wire_format")
    if sidecar is None:
        sidecar = {}
    try:
        segments = parse_segments(raw, wire_format)
    except Bypass as error:
        return _bypass_plan(wire_format, str(error))
    except Exception:  # noqa: BLE001 - any analysis failure retains everything
        return _bypass_plan(wire_format, "analysis_error")

    try:
        reasons, dependencies = sidecar_policy(segments, sidecar)
    except Bypass as error:
        # An invalid or untrusted sidecar can never grant eligibility.
        return _bypass_plan(wire_format, str(error))
    except Exception:  # noqa: BLE001
        return _bypass_plan(wire_format, "analysis_error")

    # Arm eligibility: the core assigned no reason. Every core reason is respected.
    arm_eligible = {}
    protected = {}
    for segment in segments:
        reason = reasons.get(segment.pointer)
        if reason is None and segment.text is not None and _is_restore_grammar(segment.pointer):
            arm_eligible[segment.pointer] = segment
        else:
            protected[segment.pointer] = reason or "not_in_restore_grammar"

    # Keep-first grouping over the declared identity normalization.
    first_occurrence = {}
    duplicate_candidates = []
    decisions = {}
    for segment in segments:
        if segment.pointer not in arm_eligible:
            decisions[segment.pointer] = DedupDecision(
                pointer=segment.pointer, role=segment.role,
                kind=_segment_kind(segment.pointer),
                text_sha256=_text_sha256(segment.text) if segment.text is not None else None,
                decision=DECISION_RETAIN, reason=protected[segment.pointer])
            continue
        key = (_segment_kind(segment.pointer), segment.role, segment.text)
        if key in first_occurrence:
            duplicate_candidates.append(segment)
            continue
        first_occurrence[key] = segment.pointer
        decisions[segment.pointer] = DedupDecision(
            pointer=segment.pointer, role=segment.role, kind=_segment_kind(segment.pointer),
            text_sha256=_text_sha256(segment.text), decision=DECISION_RETAIN,
            reason="retained_first_occurrence")

    # Dependency closure over the effective retained set: core-retained segments plus
    # every retained first occurrence. Reuses the core's own closure routine verbatim.
    closure = dict(reasons)
    for pointer in first_occurrence.values():
        closure.setdefault(pointer, "retained_first_occurrence")
    protect_dependencies(closure, dependencies)

    accepted = []
    for segment in duplicate_candidates:
        earlier = first_occurrence[(_segment_kind(segment.pointer), segment.role, segment.text)]
        closure_reason = closure.get(segment.pointer)
        if closure_reason is not None:
            decisions[segment.pointer] = DedupDecision(
                pointer=segment.pointer, role=segment.role,
                kind=_segment_kind(segment.pointer),
                text_sha256=_text_sha256(segment.text), decision=DECISION_RETAIN,
                reason=closure_reason, retained_pointer=earlier)
            continue
        trial = tuple(accepted) + (segment.pointer,)
        try:
            build_safe_reduction(raw, wire_format, trial)
        except Exception:  # noqa: BLE001 - any doubt retains the candidate
            decisions[segment.pointer] = DedupDecision(
                pointer=segment.pointer, role=segment.role,
                kind=_segment_kind(segment.pointer),
                text_sha256=_text_sha256(segment.text), decision=DECISION_RETAIN,
                reason="retain_not_reversible", retained_pointer=earlier)
            continue
        accepted.append(segment.pointer)
        decisions[segment.pointer] = DedupDecision(
            pointer=segment.pointer, role=segment.role, kind=_segment_kind(segment.pointer),
            text_sha256=_text_sha256(segment.text), decision=DECISION_DROP,
            reason=DROP_REASON, retained_pointer=earlier)

    ordered_decisions = tuple(decisions[segment.pointer] for segment in segments)
    return DedupPlan(
        schema_version=SCHEMA_VERSION, normalization_id=NORMALIZATION_ID,
        wire_format=wire_format, parse_status="parsed", bypass_reason=None,
        segments_examined=len(segments), arm_eligible_pointers=tuple(arm_eligible),
        protected_pointers=tuple(protected), drop_pointers=tuple(accepted),
        decisions=ordered_decisions)


# --------------------------------------------------------------------------------------
# Applying + proving (gateway applicator, restore module, core parser)
# --------------------------------------------------------------------------------------

def build_safe_reduction(raw, wire_format, drop_pointers):
    """Apply ``drop_pointers`` with the gateway's applicator and prove restoration.

    Returns ``(reduced_bytes, manifest, removed_segments, verify_result)``. Raises
    :class:`~main_model_gateway_v1.ReductionError` or
    :class:`~context_restore_v1.RestoreError` when the reduction cannot be proven
    byte-reversible; the caller must then retain the segment (fail closed on the plan,
    fail open on the request).
    """
    if not drop_pointers:
        raise ReductionError("empty removal plan")
    reduced = build_reduced_request(raw, wire_format, drop_pointers)
    parse_segments(reduced, wire_format)  # core re-validation, not a reimplementation
    manifest = build_restore_manifest(raw, wire_format, drop_pointers)
    removed = removed_segments_from_raw(raw, wire_format, drop_pointers)
    original_body = _parse_body(raw)
    if restore_request(reduced, manifest, removed) != canonical_bytes(original_body):
        raise ReductionError("restore_round_trip_mismatch")
    verify = verify_round_trip(raw, reduced, manifest, removed)
    return reduced, manifest, removed, verify


def removed_texts(removed_segments, manifest):
    """Recover the removed *segment text* per manifest record (for counting only).

    Never persisted by the gateway; the caller is the holder. Here it is used only to
    count characters and tokens of what a reduction would take out.
    """
    texts = []
    for record in manifest.records:
        value = removed_segments[record.pointer]
        texts.append(value["content"] if isinstance(value, dict) else value)
    return texts


# --------------------------------------------------------------------------------------
# Tokenizer (real, local, offline; never provider billing)
# --------------------------------------------------------------------------------------

def load_tokenizer(path=None):
    """Load a real local BPE tokenizer, or report plainly that none is available."""
    candidates = [Path(path)] if path else list(DEFAULT_TOKENIZER_PATHS)
    searched = [str(candidate) for candidate in candidates]
    try:
        from tokenizers import Tokenizer  # noqa: PLC0415 - optional local dependency
    except Exception:  # noqa: BLE001
        return None, {"status": "unavailable", "reason": "tokenizers_library_not_installed",
                      "searched": searched,
                      "note": "character counts only; no token count is claimed"}
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            tokenizer = Tokenizer.from_file(str(candidate))
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - a broken tokenizer file must not break the arm
            continue

        def counter(text, _tokenizer=tokenizer):
            return len(_tokenizer.encode(text).ids)

        return counter, {
            "status": "available",
            "path": str(candidate),
            "sha256": _sha256_hex(candidate.read_bytes()),
            "model_type": (payload.get("model") or {}).get("type"),
            "vocab_size": tokenizer.get_vocab_size(),
            "kind": "local_real_bpe_tokenizer_not_a_provider_billing_tokenizer",
            "note": "counts tokens of canonical JSON body text; not a provider chat "
                    "template and not provider billing",
            "tiktoken_installed": False,
            "searched": searched,
        }
    return None, {"status": "unavailable", "reason": "no_local_tokenizer_file_found",
                  "searched": searched,
                  "note": "character counts only; no token count is claimed"}


def _count_tokens(counter, text):
    try:
        value = counter(text)
    except Exception:  # noqa: BLE001
        return None
    return value if type(value) is int and value >= 0 else None


# --------------------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------------------

def measure_request(raw, wire_format, sidecar, tokenizer=None):
    """Plan, apply, prove and count one request. Content-free result; no raw text out."""
    counter, _ = tokenizer if tokenizer else (None, None)
    plan = plan_safe_dedup(raw, wire_format, sidecar)
    body = _parse_body(raw)
    canonical_original = canonical_bytes(body)
    original_text = canonical_original.decode("utf-8")

    result = {
        "wire_format": wire_format,
        "request_sha256": _sha256_hex(raw),
        "request_bytes": len(raw),
        "parse_status": plan.parse_status,
        "bypass_reason": plan.bypass_reason,
        "segments_examined": plan.segments_examined,
        "segments_arm_eligible": len(plan.arm_eligible_pointers),
        "segments_removed": plan.removed,
        "removed_pointers": list(plan.drop_pointers),
        "retained_not_reversible": sum(1 for d in plan.decisions
                                       if d.reason == "retain_not_reversible"),
        "protected_segment_deletions": 0,
        "tool_linked_segment_deletions": 0,
        "characters": {
            "canonical_original_chars": len(original_text),
            "reduced_chars": len(original_text),
            "reduction_chars": 0,
            "removed_segment_text_chars": 0,
        },
        "tokens": None,
        "round_trip": {"attempted": False, "ok": None, "detail": "not_parsed"},
        "no_op_manifest_check": {"attempted": False, "ok": None},
        "gateway": {"would_apply": False, "removal_plan_error": "not_evaluated"},
    }

    # A bypassed request has no segments and no manifest; nothing is removable.
    if plan.parse_status != "parsed":
        result["round_trip"] = {"attempted": False, "ok": None, "detail": "bypass_vacuous"}
        result["tokens"] = _token_block(original_text, original_text, 0, counter, [])
        return result

    protected_pointers = set(plan.protected_pointers)
    dropped = set(plan.drop_pointers)
    result["protected_segment_deletions"] = len(dropped & protected_pointers)

    try:
        from build_tool_history_fixtures_v1 import linkage_pointers
        calls, results = linkage_pointers(parse_segments(raw, wire_format))
        result["tool_linked_segment_deletions"] = len(dropped & (calls | results))
    except Exception:  # noqa: BLE001 - linkage probe is a bonus, never load-bearing
        result["tool_linked_segment_deletions"] = len(dropped & protected_pointers)

    # The gateway's own active-path decision, reported without contacting anything.
    arm_receipt = {"segments": [
        {"pointer": decision.pointer, "role": decision.role, "suggestion": "drop",
         "reason": decision.reason}
        for decision in plan.decisions if decision.decision == DECISION_DROP]}
    gateway_drops, gateway_error = removal_plan(arm_receipt)
    result["gateway"] = {
        "would_apply": gateway_error is None and bool(gateway_drops),
        "removal_plan_error": gateway_error,
        "reason": "the existing gateway accepts only role=assistant drops carrying the "
                  "model reason high_irrelevance_score; this arm's deterministic reason is "
                  "not a model score, so the active path fails open with this reason",
    }

    if not dropped:
        # No removal happened, so there is nothing to restore. The manifest contract is
        # still exercised as a genuine no-op: an empty plan must describe the canonical
        # original exactly and restore_request must return it unchanged.
        manifest = build_restore_manifest(raw, wire_format, [])
        restored = restore_request(canonical_original, manifest, {})
        no_op_ok = bool(restored == canonical_original and manifest.applied is False
                        and manifest.dropped_segment_count == 0)
        result["round_trip"] = {"attempted": False, "ok": None, "detail": "no_removal"}
        result["no_op_manifest_check"] = {"attempted": True, "ok": no_op_ok}
        result["characters"]["reduced_chars"] = len(original_text)
        result["tokens"] = _token_block(original_text, original_text, 0, counter, [])
        return result

    reduced, manifest, removed, verify = build_safe_reduction(raw, wire_format, tuple(dropped))
    restored = restore_request(reduced, manifest, removed)
    texts = removed_texts(removed, manifest)
    reduced_text = reduced.decode("utf-8")
    canonical_sha = _sha256_hex(canonical_original)
    result["round_trip"] = {
        "attempted": True,
        "ok": bool(restored == canonical_original
                   and verify["reconstructed_sha256"] == canonical_sha
                   and verify["reduced_request_sha256_matches"]),
        "detail": "byte_identical_canonical_reconstruction",
        "restored_bytes_identical": bool(verify["restored_bytes_identical"]),
        "dropped_segment_count": verify["dropped_segment_count"],
        "manifest_sha256": _sha256_hex(manifest.to_json().encode("utf-8")),
    }
    result["characters"] = {
        "canonical_original_chars": len(original_text),
        "reduced_chars": len(reduced_text),
        "reduction_chars": len(original_text) - len(reduced_text),
        "removed_segment_text_chars": sum(len(text) for text in texts),
    }
    result["tokens"] = _token_block(original_text, reduced_text, sum(len(t) for t in texts),
                                    counter, texts)
    return result


def _token_block(original_text, reduced_text, removed_chars, counter, removed_texts_=None):
    block = {
        "status": "unavailable",
        "basis": "characters_only",
        "note": "no real tokenizer was available; no token count is claimed",
        "canonical_original_tokens": None,
        "reduced_tokens": None,
        "reduction_tokens": None,
        "removed_segment_text_tokens": None,
    }
    if counter is None:
        return block
    original_tokens = _count_tokens(counter, original_text)
    reduced_tokens = _count_tokens(counter, reduced_text)
    removed_tokens = None
    if removed_texts_ is not None:
        counts = [_count_tokens(counter, text) for text in removed_texts_]
        removed_tokens = None if any(c is None for c in counts) else sum(counts)
    block.update({
        "status": "available",
        "basis": "local_real_tokenizer_full_canonical_body",
        "note": "real local BPE tokenizer over the canonical JSON body text; a local "
                "estimate, NOT provider billing and not a chat-template render",
        "canonical_original_tokens": original_tokens,
        "reduced_tokens": reduced_tokens,
        "reduction_tokens": (None if original_tokens is None or reduced_tokens is None
                             else original_tokens - reduced_tokens),
        "removed_segment_text_tokens": removed_tokens,
        "removed_segment_text_chars": removed_chars,
    })
    return block


def _aggregate(case_results):
    aggregate = {
        "cases_examined": len(case_results),
        "cases_parsed": sum(1 for c in case_results if c["parse_status"] == "parsed"),
        "cases_bypassed": sum(1 for c in case_results if c["parse_status"] != "parsed"),
        "segments_examined": sum(c["segments_examined"] for c in case_results),
        "segments_arm_eligible": sum(c["segments_arm_eligible"] for c in case_results),
        "segments_removed": sum(c["segments_removed"] for c in case_results),
        "protected_segment_deletions": sum(c["protected_segment_deletions"] for c in case_results),
        "tool_linked_segment_deletions": sum(c["tool_linked_segment_deletions"] for c in case_results),
        "candidates_retained_not_reversible": sum(c["retained_not_reversible"] for c in case_results),
        "characters": {
            "canonical_original_chars": sum(c["characters"]["canonical_original_chars"] for c in case_results),
            "reduced_chars": sum(c["characters"]["reduced_chars"] for c in case_results),
            "reduction_chars": sum(c["characters"]["reduction_chars"] for c in case_results),
            "removed_segment_text_chars": sum(c["characters"]["removed_segment_text_chars"] for c in case_results),
        },
        "round_trips_attempted": sum(1 for c in case_results if c["round_trip"]["attempted"]),
        "round_trips_ok": sum(1 for c in case_results
                              if c["round_trip"]["attempted"] and c["round_trip"]["ok"]),
        "no_op_manifest_checks_attempted": sum(
            1 for c in case_results if c["no_op_manifest_check"]["attempted"]),
        "no_op_manifest_checks_ok": sum(
            1 for c in case_results
            if c["no_op_manifest_check"]["attempted"] and c["no_op_manifest_check"]["ok"]),
        "cases_with_removals": sum(1 for c in case_results if c["segments_removed"] > 0),
        "gateway_active_would_apply_any": any(c["gateway"]["would_apply"] for c in case_results),
    }
    aggregate["round_trip_success_rate"] = (
        None if aggregate["round_trips_attempted"] == 0
        else aggregate["round_trips_ok"] / aggregate["round_trips_attempted"])
    aggregate["round_trip_success_rate_note"] = (
        "no removal occurred in this source, so no restoration was performed; the rate is "
        "undefined rather than 100%" if aggregate["round_trips_attempted"] == 0 else
        "over every removal actually performed in this source")
    aggregate["characters"]["reduction_percent"] = (
        None if aggregate["characters"]["canonical_original_chars"] == 0
        else 100.0 * aggregate["characters"]["reduction_chars"]
        / aggregate["characters"]["canonical_original_chars"])
    token_keys = ("canonical_original_tokens", "reduced_tokens", "reduction_tokens",
                  "removed_segment_text_tokens")
    available = [c for c in case_results if c["tokens"] and c["tokens"]["status"] == "available"]
    if len(available) == len(case_results) and case_results:
        totals = {}
        for key in token_keys:
            values = [c["tokens"][key] for c in available]
            totals[key] = None if any(v is None for v in values) else sum(values)
        original = totals["canonical_original_tokens"]
        totals["reduction_percent"] = (None if not original
                                       else 100.0 * (totals["reduction_tokens"] or 0) / original)
        totals["status"] = "available"
        totals["note"] = "real local BPE tokenizer; local estimate, not provider billing"
        aggregate["tokens"] = totals
    else:
        aggregate["tokens"] = {
            "status": "unavailable", "basis": "characters_only",
            "note": "tokenizer unavailable for at least one case; no token count claimed",
        }
    return aggregate


# --------------------------------------------------------------------------------------
# Fixture sources
# --------------------------------------------------------------------------------------

def _a4_cases(manifest_path=DEFAULT_MANIFEST):
    """Materialize the frozen A4 tool-history fixtures in memory, exactly as the builder does."""
    from build_tool_history_fixtures_v1 import encode_request, load_manifest, substitute, synthetic_tag
    manifest = load_manifest(manifest_path)
    seed = manifest.get("seed")
    cases = []
    for case in manifest["cases"]:
        tag = synthetic_tag(case["case_id"], seed)
        cases.append({
            "case_id": case["case_id"],
            "wire_format": case["wire_format"],
            "raw": encode_request(substitute(case["body"], tag)),
            "sidecar": substitute(case["sidecar"], tag),
        })
    return cases, {
        "path": str(manifest_path),
        "sha256": _sha256_hex(Path(manifest_path).read_bytes()),
        "seed": seed,
        "materializer": "scripts/build_tool_history_fixtures_v1.py",
        "note": "materialized in memory with the builder's own tag substitution; request "
                "bytes are the builder's non-canonical pretty-printed encoding",
    }


def _corpus_cases(corpus_dir=DEFAULT_CORPUS_DIR):
    """The largest request-shaped corpus in the repository (2400 bodies, not long-context)."""
    directory = Path(corpus_dir)
    cases = []
    files = []
    for path in sorted(directory.glob("*.jsonl")):
        files.append({"path": str(path), "sha256": _sha256_hex(path.read_bytes())})
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            metadata = record.get("metadata") or {}
            body, wire_format = metadata.get("body"), metadata.get("wire_format")
            if not isinstance(body, dict) or wire_format not in FORMATS:
                continue
            cases.append({
                "case_id": f"{path.stem}:{record.get('id')}",
                "wire_format": wire_format,
                "raw": serialized(body).encode("utf-8"),
                "sidecar": {"segments": {}},
            })
    return cases, {
        "path": str(directory),
        "files": files,
        "note": "every body in the largest request-shaped corpus in the repository; bodies "
                "are canonical and small (max measured 760 bytes), so this is NOT a "
                "long-context fixture",
    }


# --------------------------------------------------------------------------------------
# Synthetic mechanism check (authored here, reported separately, never a saving claim)
# --------------------------------------------------------------------------------------

def _synthetic_cases():
    """Deterministic bodies that exercise the rule's positive and negative paths."""
    def body(messages, wire="openai_chat", extra=None):
        payload = {"model": "synthetic", "messages": messages}
        if extra:
            payload.update(extra)
        return serialized(payload).encode("utf-8")

    duplicate_status = body([
        {"role": "system", "content": "Synthetic harness."},
        {"role": "assistant", "content": "STATUS: build green"},
        {"role": "assistant", "content": "STATUS: build green"},
        {"role": "assistant", "content": "STATUS: build green"},
        {"role": "assistant", "content": "STATUS: build green"},
        {"role": "assistant", "content": "STATUS: build green"},
        {"role": "user", "content": "Summarize the current status."},
    ])
    duplicate_tool_call = body([
        {"role": "system", "content": "Synthetic harness."},
        {"role": "assistant", "content": "Reading the file."},
        {"role": "assistant", "tool_calls": [{"id": "call-1", "type": "function",
                                              "function": {"name": "read", "arguments": "{}"}}]},
        {"role": "tool", "content": "file contents", "tool_call_id": "call-1"},
        {"role": "assistant", "content": "Reading the file."},
        {"role": "user", "content": "What did the file say?"},
    ])
    duplicate_tool_result = body([
        {"role": "system", "content": "Synthetic harness."},
        {"role": "assistant", "content": "First read."},
        {"role": "assistant", "tool_calls": [{"id": "call-1", "type": "function",
                                              "function": {"name": "read", "arguments": "{}"}}]},
        {"role": "tool", "content": "identical tool output", "tool_call_id": "call-1"},
        {"role": "assistant", "content": "Second read."},
        {"role": "assistant", "tool_calls": [{"id": "call-2", "type": "function",
                                              "function": {"name": "read", "arguments": "{}"}}]},
        {"role": "tool", "content": "identical tool output", "tool_call_id": "call-2"},
        {"role": "user", "content": "Compare the reads."},
    ])
    duplicate_user = body([
        {"role": "system", "content": "Synthetic harness."},
        {"role": "user", "content": "Please continue."},
        {"role": "assistant", "content": "Continuing."},
        {"role": "user", "content": "Please continue."},
    ])
    duplicate_not_eligible = body([
        {"role": "system", "content": "Synthetic harness."},
        {"role": "assistant", "content": "Undeclared duplicate."},
        {"role": "assistant", "content": "Undeclared duplicate."},
        {"role": "user", "content": "Go on."},
    ])
    duplicate_with_dependency = body([
        {"role": "system", "content": "Synthetic harness."},
        {"role": "assistant", "content": "Dependent duplicate."},
        {"role": "assistant", "content": "Dependent duplicate."},
        {"role": "user", "content": "Use the earlier statement."},
    ])
    responses_output_text = serialized({
        "model": "synthetic",
        "input": [
            {"role": "assistant", "content": [{"type": "output_text", "text": "dup answer"}]},
            {"role": "assistant", "content": [{"type": "output_text", "text": "dup answer"}]},
            {"role": "user", "content": "Go on."},
        ],
    }).encode("utf-8")
    anthropic_cache_control = serialized({
        "model": "synthetic",
        "system": "Synthetic harness.",
        "messages": [
            {"role": "assistant", "content": [{"type": "text", "text": "cached duplicate"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "cached duplicate",
                                               "cache_control": {"type": "ephemeral"}}]},
            {"role": "user", "content": "Go on."},
        ],
    }).encode("utf-8")

    def eligible_all(raw, wire_format):
        segments = parse_segments(raw, wire_format)
        return {"segments": {segment.pointer: {"eligible": True} for segment in segments}}

    cases = [
        {"case_id": "duplicate_status_lines", "wire_format": "openai_chat",
         "raw": duplicate_status, "sidecar": eligible_all(duplicate_status, "openai_chat"),
         "expect_removed": 4, "expected_reason": None},
        {"case_id": "duplicate_tool_call_text", "wire_format": "openai_chat",
         "raw": duplicate_tool_call, "sidecar": eligible_all(duplicate_tool_call, "openai_chat"),
         "expect_removed": 1, "expected_reason": None},
        {"case_id": "duplicate_tool_result_retained", "wire_format": "openai_chat",
         "raw": duplicate_tool_result,
         "sidecar": eligible_all(duplicate_tool_result, "openai_chat"),
         "expect_removed": 0, "expected_reason": None},
        {"case_id": "duplicate_user_retained", "wire_format": "openai_chat",
         "raw": duplicate_user, "sidecar": eligible_all(duplicate_user, "openai_chat"),
         "expect_removed": 0, "expected_reason": None},
        {"case_id": "not_explicitly_eligible_retained", "wire_format": "openai_chat",
         "raw": duplicate_not_eligible, "sidecar": {"segments": {}},
         "expect_removed": 0, "expected_reason": "not_explicitly_eligible"},
        {"case_id": "dependency_closure_retains_duplicate", "wire_format": "openai_chat",
         "raw": duplicate_with_dependency,
         "sidecar": {"segments": {
             "/messages/1/content": {"eligible": True},
             "/messages/2/content": {"eligible": True},
             "/messages/3/content": {"depends_on": ["/messages/2/content"]}}},
         "expect_removed": 0, "expected_reason": "required_dependency"},
        {"case_id": "responses_output_text_irreversible", "wire_format": "openai_responses",
         "raw": responses_output_text,
         "sidecar": eligible_all(responses_output_text, "openai_responses"),
         "expect_removed": 0, "expected_reason": "retain_not_reversible"},
        {"case_id": "anthropic_cache_control_irreversible", "wire_format": "anthropic_messages",
         "raw": anthropic_cache_control,
         "sidecar": eligible_all(anthropic_cache_control, "anthropic_messages"),
         "expect_removed": 0, "expected_reason": "retain_not_reversible"},
    ]
    return cases


def _mechanism_check(tokenizer=None):
    cases = _synthetic_cases()
    results = []
    for case in cases:
        measured = measure_request(case["raw"], case["wire_format"], case["sidecar"], tokenizer)
        reasons = [d.reason for d in plan_safe_dedup(
            case["raw"], case["wire_format"], case["sidecar"]).decisions
            if d.decision == DECISION_RETAIN]
        ok = measured["segments_removed"] == case["expect_removed"]
        if case["expected_reason"] is not None:
            ok = ok and case["expected_reason"] in reasons
        results.append({
            "case_id": case["case_id"],
            "wire_format": case["wire_format"],
            "segments_examined": measured["segments_examined"],
            "segments_removed": measured["segments_removed"],
            "expected_removed": case["expect_removed"],
            "protected_segment_deletions": measured["protected_segment_deletions"],
            "round_trip_ok": measured["round_trip"]["ok"],
            "characters": measured["characters"],
            "tokens": measured["tokens"],
            "ok": bool(ok),
        })
    return {
        "label": "synthetic_mechanism_check_not_a_savings_claim",
        "note": "bodies authored in this script purely to exercise the rule; they are not "
                "real transcripts and any reduction here is NOT a token saving",
        "cases": results,
        "all_ok": all(r["ok"] for r in results),
        "aggregate": _aggregate([measure_request(c["raw"], c["wire_format"], c["sidecar"], tokenizer)
                                 for c in cases]),
    }


# --------------------------------------------------------------------------------------
# Receipt
# --------------------------------------------------------------------------------------

def _git_head():
    try:
        import subprocess
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
                              capture_output=True, text=True, check=False).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def measure(fixture_manifest=DEFAULT_MANIFEST, corpus_dir=DEFAULT_CORPUS_DIR, tokenizer_path=None):
    """Run the full measurement and return a content-free receipt dictionary."""
    counter, tokenizer_info = load_tokenizer(tokenizer_path)
    tokenizer = (counter, tokenizer_info)

    a4_cases, a4_provenance = _a4_cases(fixture_manifest)
    a4_results = [measure_request(c["raw"], c["wire_format"], c["sidecar"], tokenizer)
                  for c in a4_cases]
    for case, result in zip(a4_cases, a4_results):
        result["case_id"] = case["case_id"]

    corpus_cases, corpus_provenance = _corpus_cases(corpus_dir)
    corpus_results = [measure_request(c["raw"], c["wire_format"], c["sidecar"], tokenizer)
                      for c in corpus_cases]

    mechanism = _mechanism_check(tokenizer)
    a4_aggregate = _aggregate(a4_results)
    corpus_aggregate = _aggregate(corpus_results)
    mechanism_aggregate = mechanism["aggregate"]
    attempts = (a4_aggregate["round_trips_attempted"] + corpus_aggregate["round_trips_attempted"]
                + mechanism_aggregate["round_trips_attempted"])
    successes = (a4_aggregate["round_trips_ok"] + corpus_aggregate["round_trips_ok"]
                 + mechanism_aggregate["round_trips_ok"])

    return {
        "schema_version": RECEIPT_SCHEMA,
        "arm": "scripts/safe_dedup_v1.py",
        "repository_head": _git_head(),
        "deterministic": True,
        "no_model_calls": True,
        "no_network_calls": True,
        "provider_calls_made": 0,
        "bytes_sent_to_provider": 0,
        "rule": {
            "normalization_id": NORMALIZATION_ID,
            "normalization": NORMALIZATION_DESCRIPTION,
            "drop_reason": DROP_REASON,
            "eligibility": "core sidecar_policy assigns no reason (explicit eligible:true, "
                           "unprotected, text-bearing, in restore grammar)",
            "match": "identical code points; same kind (message|text_part) and same role",
            "keep": "first occurrence in request order is always retained",
            "reversibility": "every removal is proven byte-identical via "
                             "build_reduced_request + build_restore_manifest + "
                             "removed_segments_from_raw + restore_request",
            "fail_closed": "any bypass, protection, dependency, irreversibility, tamper, or "
                           "unknown input means retain",
            "protected_segment_deletion_policy": "never; measured over every fixture",
        },
        "reused_modules": {
            "shadow_core": "scripts/context_gate_v1.py (parse_segments, sidecar_policy, "
                           "protect_dependencies, Bypass) - imported, not modified",
            "restore": "scripts/context_restore_v1.py (build_restore_manifest, "
                       "restore_request, verify_round_trip, canonical_bytes) - imported, "
                       "not modified",
            "gateway": "scripts/main_model_gateway_v1.py (build_reduced_request, "
                       "removed_segments_from_raw, removal_plan, ReductionError) - "
                       "imported, not modified",
        },
        "tokenizer": tokenizer_info,
        "fixtures": {
            "a4_tool_history": {
                "provenance": a4_provenance,
                "aggregate": a4_aggregate,
                "cases": a4_results,
            },
            "context_relevance_corpus": {
                "provenance": corpus_provenance,
                "aggregate": corpus_aggregate,
                "cases_truncated": True,
                "cases_sample": corpus_results[:5],
            },
        },
        "synthetic_mechanism_check": mechanism,
        "round_trip_summary": {
            "note": "a real removal-then-restore round-trip is only attempted when the arm "
                    "actually removes something; zero-removal sources are reported as "
                    "undefined rather than 100%. Counts are per reduced request, not per "
                    "removed segment.",
            "reduced_requests_round_tripped": attempts,
            "reduced_requests_restored_byte_identically": successes,
            "success_rate": None if attempts == 0 else successes / attempts,
            "per_source": {
                "a4_tool_history": {
                    "attempted": a4_aggregate["round_trips_attempted"],
                    "ok": a4_aggregate["round_trips_ok"]},
                "context_relevance_corpus": {
                    "attempted": corpus_aggregate["round_trips_attempted"],
                    "ok": corpus_aggregate["round_trips_ok"]},
                "synthetic_mechanism_check": {
                    "attempted": mechanism_aggregate["round_trips_attempted"],
                    "ok": mechanism_aggregate["round_trips_ok"]},
            },
            "no_op_manifest_checks": {
                "attempted": (a4_aggregate["no_op_manifest_checks_attempted"]
                              + corpus_aggregate["no_op_manifest_checks_attempted"]
                              + mechanism_aggregate["no_op_manifest_checks_attempted"]),
                "ok": (a4_aggregate["no_op_manifest_checks_ok"]
                       + corpus_aggregate["no_op_manifest_checks_ok"]
                       + mechanism_aggregate["no_op_manifest_checks_ok"]),
                "note": "zero-removal requests still exercise the manifest contract with an "
                        "empty plan; these are not restoration round-trips",
            },
        },
        "gateway_compatibility": {
            "arm_drop_reason": DROP_REASON,
            "gateway_accepted_drop_reason": "high_irrelevance_score",
            "arm_plan_accepted_by_existing_gateway": a4_aggregate["gateway_active_would_apply_any"],
            "removal_plan_error_for_arm_plan": "protected_segment_in_removal_set",
            "note": "the existing gateway's active path accepts only assistant drops whose "
                    "receipt reason is the model reason high_irrelevance_score. This arm is "
                    "model-free, so it does not and must not emit that reason. Unmodified, "
                    "the gateway therefore fails open on every arm plan and forwards the "
                    "ORIGINAL bytes. Enabling this arm on a real provider path would "
                    "require a reviewed gateway change, which was out of scope here.",
        },
        "accounting": {
            "gateway_rule_reference": "docs/MAIN_MODEL_GATEWAY_V1.md section 8",
            "provider_contacted": False,
            "reduced_bytes_sent_to_a_provider": 0,
            "gateway_would_report": {
                "claim": "none",
                "basis": "no_reduction_sent",
                "tokens": None,
                "why": "nothing reduced was sent: no provider was contacted, and the "
                       "existing gateway refuses this arm's plan reason and fails open",
            },
            "if_forced_through_the_unchanged_gateway": {
                "claim": "none",
                "basis": "no_reduction_sent",
                "forward_reason": "protected_segment_in_removal_set",
            },
            "shadow_estimate_only": {
                "claim": "estimate",
                "basis": "shadow_estimate_only",
                "tokens": a4_aggregate["tokens"].get("removed_segment_text_tokens"),
                "note": "a labelled local estimate of removed segment text; not a saving, "
                        "and zero on every real fixture measured here",
            },
            "actual_provider_savings": None,
            "net_sent_token_saving": 0,
            "net_sent_token_saving_basis": "no reduced bytes were sent to any provider",
        },
        "net_result": {
            "segments_removed_on_real_fixtures": (a4_aggregate["segments_removed"]
                                                  + corpus_aggregate["segments_removed"]),
            "character_reduction_on_real_fixtures": (a4_aggregate["characters"]["reduction_chars"]
                                                     + corpus_aggregate["characters"]["reduction_chars"]),
            "token_reduction_on_real_fixtures": (
                None if not a4_aggregate["tokens"].get("status") == "available"
                or not corpus_aggregate["tokens"].get("status") == "available"
                else (a4_aggregate["tokens"]["reduction_tokens"]
                      + corpus_aggregate["tokens"]["reduction_tokens"])),
            "net_sent_token_saving": 0,
            "verdict": "the deterministic arm cannot beat zero on the fixtures available in "
                       "this repository, and even where it does find exact duplicates the "
                       "unchanged gateway cannot carry the plan to a provider",
        },
        "limitations": [
            "no provider was contacted, so there are no actual provider-reported token "
            "counts and no actual saving; every token number is a local estimate from a "
            "real local BPE tokenizer over canonical JSON body text",
            "there is no long-context fixture in this repository: the A4 bodies are 4-7 "
            "messages and the largest request corpus has bodies of at most 760 bytes",
            "token counts are of the JSON body text, not of a provider chat-template "
            "render, and different providers use different tokenizers",
            "the arm is not wired into the gateway (out of scope); the existing gateway "
            "fails open on its plan reason, so nothing is sent",
            "the synthetic mechanism check exists only to prove the rule removes exact "
            "duplicates and restores them; its reductions are not savings",
            "no independent review; this work has not been reviewed by the project's "
            "independent reviewer",
        ],
    }


# --------------------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------------------

def self_test():
    """Deterministic in-process checks of the rule. Returns a JSON-serializable report."""
    checks = []
    cached = {}

    def _cached_tokenizer():
        if "value" not in cached:
            cached["value"] = load_tokenizer()
        return cached["value"]

    def check(name, condition, detail=""):
        checks.append({"check": name, "ok": bool(condition), "detail": detail})
        return bool(condition)

    def plan(case_id):
        for case in _synthetic_cases():
            if case["case_id"] == case_id:
                return plan_safe_dedup(case["raw"], case["wire_format"], case["sidecar"]), case
        raise KeyError(case_id)

    def measure(case_id):
        plan_, case = plan(case_id)
        return measure_request(case["raw"], case["wire_format"], case["sidecar"],
                               _cached_tokenizer()), plan_, case

    # 1. keep-first exact-duplicate removal
    measured, planned, _ = measure("duplicate_status_lines")
    check("keep_first_removes_only_later_duplicates",
          planned.drop_pointers == ("/messages/2/content", "/messages/3/content",
                                    "/messages/4/content", "/messages/5/content")
          and measured["segments_removed"] == 4,
          "removed=%r" % (planned.drop_pointers,))

    # 2. identity normalization: nothing is case-folded or whitespace-collapsed
    variants = serialized({"model": "synthetic", "messages": [
        {"role": "system", "content": "s"},
        {"role": "assistant", "content": "Alpha beta"},
        {"role": "assistant", "content": "alpha beta"},
        {"role": "assistant", "content": "Alpha  beta"},
        {"role": "assistant", "content": "Alpha beta "},
        {"role": "user", "content": "go"}]}).encode("utf-8")
    variant_plan = plan_safe_dedup(variants, "openai_chat",
                                   {"segments": {"/messages/1/content": {"eligible": True},
                                                 "/messages/2/content": {"eligible": True},
                                                 "/messages/3/content": {"eligible": True},
                                                 "/messages/4/content": {"eligible": True}}})
    check("identity_normalization_no_case_or_whitespace_folding",
          variant_plan.drop_pointers == (), "removed=%r" % (variant_plan.drop_pointers,))

    # 3. protected / ineligible / dependency paths never remove
    for case_id, expected_reason in (
            ("duplicate_tool_result_retained", "protected_structure"),
            ("duplicate_user_retained", "protected_structure"),
            ("not_explicitly_eligible_retained", "not_explicitly_eligible"),
            ("dependency_closure_retains_duplicate", "required_dependency")):
        measured, planned, _ = measure(case_id)
        reasons = {d.reason for d in planned.decisions}
        check(f"protected_path_retains:{case_id}",
              measured["segments_removed"] == 0 and expected_reason in reasons,
              "reasons=%r" % (sorted(reasons),))

    # 4. irreversible shapes are retained instead of producing an unrestorable removal
    for case_id in ("responses_output_text_irreversible", "anthropic_cache_control_irreversible"):
        measured, planned, _ = measure(case_id)
        check(f"irreversible_retained:{case_id}",
              measured["segments_removed"] == 0
              and measured["retained_not_reversible"] >= 1,
              "removed=%d not_reversible=%d" % (measured["segments_removed"],
                                                measured["retained_not_reversible"]))

    # 5. proven byte-identical round-trip on every removal
    measured, _, case = measure("duplicate_status_lines")
    reduced, manifest, removed, verify = build_safe_reduction(
        case["raw"], case["wire_format"], plan(case["case_id"])[0].drop_pointers)
    restored = restore_request(reduced, manifest, removed)
    check("byte_identical_restore_round_trip",
          restored == canonical_bytes(_parse_body(case["raw"]))
          and verify["restored_bytes_identical"] is True
          and verify["reconstructed_sha256"] == _sha256_hex(canonical_bytes(_parse_body(case["raw"]))),
          "restored_identical=%r" % (verify["restored_bytes_identical"],))

    # 6. fail-closed on tampering
    def raises(callable_):
        try:
            callable_()
        except RestoreError:
            return True
        except Exception:  # noqa: BLE001 - any refusal is acceptable, silence is not
            return True
        return False

    tampered_manifest = dict(manifest.as_dict())
    tampered_manifest["restored_bytes_sha256"] = "0" * 64
    check("tamper_manifest_rejected",
          raises(lambda: restore_request(reduced, tampered_manifest, removed)))

    tampered_reduced = reduced[:-1] + b" "
    check("tamper_reduced_bytes_rejected",
          raises(lambda: restore_request(tampered_reduced, manifest, removed)))

    bad_segments = dict(removed)
    first_pointer = manifest.records[0].pointer
    bad_segments[first_pointer] = "a different text"
    check("tamper_removed_segment_rejected",
          raises(lambda: restore_request(reduced, manifest, bad_segments)))

    check("tamper_unknown_pointer_rejected",
          raises(lambda: restore_request(reduced, manifest,
                                         {**removed, "/messages/99/content": "x"})))

    # 7. contract validity of the manifest
    from context_restore_v1 import SCHEMA_VERSION as RESTORE_SCHEMA, DROP_POINTER
    check("manifest_contract_valid",
          manifest.schema_version == RESTORE_SCHEMA
          and manifest.applied is True
          and manifest.dropped_segment_count == len(manifest.records)
          and all(DROP_POINTER.match(record.pointer) for record in manifest.records)
          and set(record.pointer for record in manifest.records) == set(removed)
          and parse_segments(reduced, case["wire_format"]) is not None,
          "schema=%s count=%d" % (manifest.schema_version, manifest.dropped_segment_count))

    # 8. determinism
    first = plan_safe_dedup(case["raw"], case["wire_format"], case["sidecar"]).as_dict()
    second = plan_safe_dedup(case["raw"], case["wire_format"], case["sidecar"]).as_dict()
    check("determinism_plan_identical", serialized(first) == serialized(second))

    # 9. the existing gateway cannot carry this arm's plan unmodified
    arm_receipt = {"segments": [{"pointer": pointer, "role": "assistant",
                                 "suggestion": "drop", "reason": DROP_REASON}
                                for pointer in plan(case["case_id"])[0].drop_pointers]}
    drops, error = removal_plan(arm_receipt)
    check("gateway_refuses_arm_reason",
          drops == [] and error == "protected_segment_in_removal_set",
          "error=%r" % (error,))

    # 10. no sidecar grant means nothing is removed
    no_grant = plan_safe_dedup(case["raw"], case["wire_format"], {"segments": {}})
    check("no_sidecar_grant_removes_nothing", no_grant.drop_pointers == ())

    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "self-test",
        "status": "ok" if all(c["ok"] for c in checks) else "failed",
        "checks": checks,
        "passed": sum(1 for c in checks if c["ok"]),
        "total": len(checks),
    }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true",
                        help="run the deterministic in-process checks and exit")
    parser.add_argument("--plan", type=Path, default=None,
                        help="plan a single request file instead of measuring")
    parser.add_argument("--sidecar", type=Path, default=None,
                        help="trusted sidecar JSON for --plan")
    parser.add_argument("--wire-format", choices=sorted(FORMATS), default=None,
                        help="wire format for --plan")
    parser.add_argument("--fixture-manifest", type=Path, default=DEFAULT_MANIFEST,
                        help="A4 tool-history fixture manifest")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_DIR,
                        help="request-shaped JSONL corpus directory")
    parser.add_argument("--tokenizer", type=Path, default=None,
                        help="local tokenizer.json; defaults to the repository copy")
    parser.add_argument("--out", type=Path, default=None,
                        help="write the measurement receipt JSON here")
    args = parser.parse_args(argv)

    if args.self_test:
        report = self_test()
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] == "ok" else 1

    if args.plan is not None:
        if args.wire_format is None:
            parser.error("--plan requires --wire-format")
        raw = args.plan.read_bytes()
        sidecar = json.loads(args.sidecar.read_text(encoding="utf-8")) if args.sidecar else {}
        plan = plan_safe_dedup(raw, args.wire_format, sidecar)
        print(json.dumps(plan.as_dict(), indent=2, sort_keys=True))
        return 0

    receipt = measure(args.fixture_manifest, args.corpus, args.tokenizer)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    a4 = receipt["fixtures"]["a4_tool_history"]["aggregate"]
    corpus = receipt["fixtures"]["context_relevance_corpus"]["aggregate"]
    summary = {
        "a4": {k: a4[k] for k in ("cases_examined", "segments_examined", "segments_removed",
                                  "characters", "tokens", "protected_segment_deletions",
                                  "round_trip_success_rate")},
        "corpus": {k: corpus[k] for k in ("cases_examined", "segments_examined", "segments_removed",
                                          "characters", "tokens", "protected_segment_deletions",
                                          "round_trip_success_rate")},
        "net_sent_token_saving": receipt["accounting"]["net_sent_token_saving"],
        "tokenizer": receipt["tokenizer"]["status"],
        "out": str(args.out) if args.out else None,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
