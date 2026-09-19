#!/usr/bin/env python3
"""Deterministic contrastive-pair builder for the NanoJev context gate (V1).

This is the *data* half of the pre-registered Track A contrastive-curation
experiment described in ``docs/GATE_CONTRASTIVE_PROTOCOL_V1.md`` and in the
community reference "Track A recipe and serving: Bespoke Nimble"
(``docs/JEV_COMMUNITY_REFERENCES.md``). It builds training-shaped contrastive
pairs; it does not train anything, does not touch a checkpoint, does not call a
provider, and does not enable active filtering.

Contrastive rule (frozen as ``CONTRASTIVE_RULE``)
-------------------------------------------------

A *pair* is two request bodies that differ in exactly **one declared fact leaf
of exactly one record**. The removal oracle (a deterministic last-value-wins
lookup over the record set, the same independent oracle style used by the
existing bounded record-dependency curriculum) must return ``keep`` for the
load-bearing member and ``remove`` for the other member. The flip is *verified*
against the real byte-preserving core ``scripts/context_gate_v1.py`` through a
fixed synthetic oracle stub scorer -- never assumed. The builder never reads,
imports, or derives from the existing relevance test/OOD corpus or the
workflow V2 evaluation cohort; those are evaluation-only and are refused both
by path and by reserved content token.

Boundaries enforced here
------------------------

* no training, no checkpoint read or write, no network access, no provider call;
* the byte-preserving shadow core is imported read-only and never forked;
* outputs go only to a caller-chosen directory that must be empty, and never to
  an excluded evaluation location;
* every pair records what changed (path, field, values, direction) and every
  item records a content hash;
* the manifest is re-derivable: ``validate_manifest`` re-runs the frozen rule
  and the core, so a hand edit that contradicts them fails loudly;
* ``--self-test`` validates without writing any output.

Usage
-----

    .venv/bin/python scripts/build_gate_contrastive_v1.py --self-test
    .venv/bin/python scripts/build_gate_contrastive_v1.py --output-dir /tmp/gate-contrastive
    .venv/bin/python scripts/build_gate_contrastive_v1.py --output-dir /tmp/gate-contrastive --seed 20260920
    .venv/bin/python -m unittest discover -s scripts -p 'test_gate_contrastive*.py'
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

from context_gate_v1 import Bypass, FORMATS, parse_segments, shadow_request, serialized


BASE_SCHEMA = "nanojev-gate-contrastive-base-v1"
MANIFEST_SCHEMA = "nanojev-gate-contrastive-manifest-v1"
ITEM_SCHEMA = "nanojev-gate-contrastive-item-v1"
SOURCE_GROUP_SCHEMA = "nanojev-gate-contrastive-source-group-v1"
TAG_PLACEHOLDER = "@SYNTH_TAG@"
MANIFEST_STATUS = "frozen_before_training_requires_independent_review"
MANIFEST_NAME = "contrastive_manifest.json"
DEFAULT_SEED = 20260919

#: The only mutable fact slots. Every pair mutates exactly one leaf of one record.
MECHANISMS = ("supersession", "entity_binding", "field_binding")
WIRES = ("openai_chat", "openai_responses", "anthropic_messages")
WORLDS_PER_MECHANISM = 4
PAIR_COUNT = len(MECHANISMS) * WORLDS_PER_MECHANISM

#: Split skeleton. Source-group separation is what matters: a pair's two members
#: always share one source group and one split, and split names never define a
#: source hash. No mechanism is held out of train; test/dev/calibration each see
#: at least two mechanisms so a split is not a mechanism proxy.
SPLIT_TABLE = {
    "supersession": ("train", "train", "dev", "calibration"),
    "entity_binding": ("train", "train", "calibration", "test"),
    "field_binding": ("train", "train", "dev", "test"),
}
SPLITS = ("train", "dev", "calibration", "test")

#: Field names are drawn from this pool; it deliberately shares no token with any
#: existing evaluation cohort.
FIELD_POOL = ("capacity", "calibration_hz", "tolerance", "interval_rate", "batch_size",
              "hold_seconds", "drift_ratio", "seal_width")

INSTRUCTIONS = (f"Session tag: {TAG_PLACEHOLDER}. Records are historical entries in "
                "chronological order; the last value for an item and field wins. Use only "
                "the records to answer; do not invent missing values.")

#: Evaluation material this builder must never read, cite, or derive from.
EXCLUDED_DIR_PARTS = ("data", "results", "research")
EXCLUDED_PATH_MARKERS = ("relevance", "context_relevance", "fast_jev", "challenge", "ood")
RESERVED_TOKENS = (
    # Context-relevance V1 corpus: split names, kinds, family field tokens, OOD renderings.
    "required_field", "two_fields", "latest_correction", "limit_check", "wrong_entity",
    "wrong_field", "superseded", "unrelated", "unit_price", "position_limit", "order_limit",
    "refund_days", "response_hours", "speed_limit", "distance_limit", "robotics",
    "multilingual", "\u65f6\u95f4", "\u5730\u70b9\u7f16\u53f7",
    # Workflow V2 evaluation cohort: family ids and OOD renderings.
    "catalog_lookup", "smart_home", "known_chance", "\u5e93\u5b58\u8bb0\u5f55",
    "\u968f\u673a\u62bd\u6837\u5b9e\u9a8c", "\u5bb6\u5ead\u63a7\u5236\u8bb0\u5f55",
    # Tool-history shadow fixture corpus.
    "tool_history_fixture", "non_repeatable_result", "mutable_file_read", "secrets_credentials",
)

GENERATION_RULE = ("frozen mechanism x index skeleton: for every mechanism in "
                   "MECHANISMS and every index in range(WORLDS_PER_MECHANISM), draw five "
                   "distinct integer values and three distinct field names from the "
                   "seed-derived stream, derive two synthetic entity ids from the stream "
                   "digest, and place the pair at SPLIT_TABLE[mechanism][index].")

CONTRASTIVE_RULE = {
    "version": "gate-contrastive-rule-v1",
    "statement": ("A pair is two request bodies that differ in exactly one declared fact leaf "
                  "of exactly one record. The deterministic removal oracle must return keep "
                  "for one member and remove for the other."),
    "mechanisms": {
        "supersession": "Mutate the field label of a later record so the candidate record either stops being the last value for the queried field or starts being it, and the candidate therefore flips between load-bearing and redundant.",
        "entity_binding": "Mutate the entity of the candidate record so the record stops being about the queried item or starts being about it.",
        "field_binding": "Mutate the field of the candidate record so the record stops carrying the queried field or starts carrying it.",
    },
    "invariants": [
        "exactly one fact leaf changes between the two members; every other byte that is not the declared fact is identical",
        "the two members share one source group and one split; the pair is never split",
        "the decision flip is recomputed by the removal oracle and re-verified through the real shadow core on every validation",
        "the corpus is self-authored synthetic data; no evaluation cohort is read or derived from",
        "no training, checkpoint change, network call, or active filtering is performed by this builder",
    ],
    "oracle": ("last-value-wins lookup: the answer is the chronologically last record of the "
               "queried item and field; the candidate is load-bearing (keep) exactly when "
               "removing it changes that answer, otherwise it is irrelevant (remove)"),
    "mutation_directions": ["keep_to_remove", "remove_to_keep"],
}

PROVENANCE = {
    "source": "self_authored_synthetic_contrastive_worlds",
    "license": "CC0-1.0",
    "contains_real_user_data": False,
    "contains_real_credentials": False,
    "contains_real_tool_output": False,
    "reuses_frozen_relevance_corpus": False,
    "reuses_workflow_v2_evaluation_cohort": False,
    "uses_relevance_test_or_ood_as_training_source": False,
    "derived_from_any_existing_evaluation_record": False,
    "training_performed": False,
    "checkpoint_modified": False,
    "network_access": False,
}

CORE_CONTRACT = {
    "core": "scripts/context_gate_v1.py",
    "entrypoint": "shadow_request",
    "threshold": 0.99,
    "threshold_frozen": True,
    "active_filtering": False,
    "scorer_kind": "synthetic_contrastive_oracle_stub",
}

STUB_CHECKPOINT = {"model": "synthetic-contrastive-oracle-stub", "revision": "gate-contrastive-v1"}


# --------------------------------------------------------------------------------------
# deterministic primitives
# --------------------------------------------------------------------------------------

def digest_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def digest_value(value):
    """Canonical content hash: key order never changes the digest.

    The manifest is validated after a JSON round-trip (where ``pretty`` has sorted
    every key), so a digest that depended on insertion order would not be stable.
    """
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                           allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def synthetic_tag(pair_id, seed):
    material = f"nanojev-gate-contrastive:{seed}:{pair_id}".encode("utf-8")
    return "gc-" + hashlib.sha256(material).hexdigest()[:12]


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
    """Realistic non-canonical JSON bytes, exactly as a client might send them."""
    return ("\n" + json.dumps(body, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def pretty(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def reserved_token_hits(text):
    return sorted({token for token in RESERVED_TOKENS if token in text})


def excluded_path_reason(path):
    """Return a fixed reason string when a path belongs to an evaluation location."""
    for part in Path(path).resolve().parts:
        lowered = part.lower()
        if lowered in EXCLUDED_DIR_PARTS:
            return f"path is under the excluded directory {lowered!r}"
        for marker in EXCLUDED_PATH_MARKERS:
            if marker == "ood":
                if lowered == "ood" or lowered.startswith("ood.") or lowered.endswith("_ood") or "_ood." in lowered:
                    return f"path part {part!r} names an out-of-distribution cohort"
            elif marker in lowered:
                return f"path part {part!r} names an evaluation cohort"
    return None


def guard_text(text, label):
    hits = reserved_token_hits(text)
    if hits:
        raise ValueError(f"{label} contains reserved evaluation-corpus tokens: {hits}")


# --------------------------------------------------------------------------------------
# frozen base-fixture set
# --------------------------------------------------------------------------------------

def worlds_for(seed):
    """Derive the frozen world skeleton from the seed. Deterministic and isolated."""
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    worlds = []
    for mechanism_index, mechanism in enumerate(MECHANISMS):
        for index in range(WORLDS_PER_MECHANISM):
            material = f"nanojev-gate-contrastive-v1:{seed}:{mechanism}:{index}"
            rng = random.Random(material)
            values = dict(zip(("candidate", "anchor", "filler", "other", "tail"),
                              rng.sample(range(10, 1000), 5)))
            fields = dict(zip(("primary", "secondary", "tertiary"),
                              rng.sample(FIELD_POOL, 3)))
            stream = hashlib.sha256(material.encode("utf-8")).hexdigest()
            worlds.append({
                "world_id": f"{mechanism}-{index:02d}",
                "mechanism": mechanism,
                "index": index,
                "wire_format": WIRES[(mechanism_index + index) % len(WIRES)],
                "split": SPLIT_TABLE[mechanism][index],
                "base_member": "keep" if index % 2 == 0 else "remove",
                "entity_a": "item-" + stream[:12],
                "entity_b": "item-" + stream[12:24],
                "fields": fields,
                "values": values,
            })
    return worlds


def default_base_fixture(seed):
    return {
        "schema_version": BASE_SCHEMA,
        "provenance": dict(PROVENANCE),
        "generation_rule": GENERATION_RULE,
        "seed": seed,
        "worlds": worlds_for(seed),
    }


def load_base_fixture(path):
    """Load a caller-supplied base fixture set, refusing evaluation sources."""
    resolved = Path(path).resolve()
    reason = excluded_path_reason(resolved)
    if reason:
        raise ValueError(f"base fixture refused: {reason}")
    if not resolved.is_file():
        raise ValueError(f"base fixture {resolved.name!r} is not a readable file")
    text = resolved.read_text(encoding="utf-8")
    guard_text(text, f"base fixture {resolved.name!r}")
    try:
        base = json.loads(text)
    except ValueError as error:
        raise ValueError(f"base fixture is not valid JSON: {type(error).__name__}") from None
    if not isinstance(base, dict):
        raise ValueError("base fixture must be a JSON object")
    if base.get("schema_version") != BASE_SCHEMA:
        raise ValueError(f"base fixture schema_version must be {BASE_SCHEMA!r}")
    return base


def world_signature(world):
    """The semantic identity of a world; split and direction labels are excluded."""
    return {key: world[key] for key in sorted(world) if key not in ("split", "base_member")}


def source_group_id(world):
    payload = {"schema": SOURCE_GROUP_SCHEMA, "mechanism": world["mechanism"],
               "world": world_signature(world)}
    return digest_value(payload)


def validate_base_fixture(base):
    """Return integrity errors for a base fixture set; empty means sound."""
    errors = []

    def fail(message):
        errors.append(message)

    if not isinstance(base, dict):
        return ["base fixture must be a JSON object"]
    if base.get("schema_version") != BASE_SCHEMA:
        fail(f"base fixture schema_version must be {BASE_SCHEMA!r}")
    worlds = base.get("worlds")
    if not isinstance(worlds, list) or not worlds:
        return errors + ["base fixture worlds must be a non-empty list"]

    seen_world_ids, seen_entities, mechanisms = set(), set(), {}
    for position, world in enumerate(worlds):
        label = f"world #{position}"
        if not isinstance(world, dict):
            fail(f"{label}: must be an object")
            continue
        world_id = world.get("world_id")
        if not isinstance(world_id, str) or not world_id or world_id in seen_world_ids:
            fail(f"{label}: world_id must be a unique non-empty string")
        seen_world_ids.add(world_id)
        label = f"world {world_id!r}"
        mechanism = world.get("mechanism")
        if mechanism not in MECHANISMS:
            fail(f"{label}: mechanism must be one of {list(MECHANISMS)}")
        if type(world.get("index")) is not int or world["index"] < 0:
            fail(f"{label}: index must be a non-negative integer")
        if world.get("wire_format") not in FORMATS:
            fail(f"{label}: wire_format must be one of {sorted(FORMATS)}")
        if world.get("split") not in SPLITS:
            fail(f"{label}: split must be one of {list(SPLITS)}")
        if world.get("base_member") not in ("keep", "remove"):
            fail(f"{label}: base_member must be keep|remove")
        mechanisms.setdefault(mechanism, set()).add(world.get("base_member"))
        for key in ("entity_a", "entity_b"):
            value = world.get(key)
            if not isinstance(value, str) or not value:
                fail(f"{label}: {key} must be a non-empty string")
            elif value in seen_entities:
                fail(f"{label}: {key} {value!r} is not isolated from the other worlds")
            else:
                seen_entities.add(value)
        if world.get("entity_a") == world.get("entity_b"):
            fail(f"{label}: entity_a and entity_b must differ")
        fields = world.get("fields")
        if not isinstance(fields, dict) or set(fields) != {"primary", "secondary", "tertiary"}:
            fail(f"{label}: fields must declare exactly primary/secondary/tertiary")
        elif len(set(fields.values())) != 3 or any(name not in FIELD_POOL for name in fields.values()):
            fail(f"{label}: field names must be three distinct names from the frozen pool")
        values = world.get("values")
        if not isinstance(values, dict) or set(values) != {"candidate", "anchor", "filler", "other", "tail"}:
            fail(f"{label}: values must declare exactly candidate/anchor/filler/other/tail")
        elif any(type(value) is not int for value in values.values()) or len(set(values.values())) != 5:
            fail(f"{label}: values must be five distinct integers")
    for mechanism, members in mechanisms.items():
        if members != {"keep", "remove"}:
            fail(f"mechanism {mechanism!r} must contain both mutation directions (keep and remove bases)")
    try:
        guard_text(serialized(base), "base fixture")
    except ValueError as error:
        fail(str(error))
    return errors


# --------------------------------------------------------------------------------------
# construction rule: layout, oracle, rendering, core verification
# --------------------------------------------------------------------------------------

def _layout(world):
    """Return (facts_keep, facts_remove, pivot_index, pivot_field, pivot_values, candidate_index).

    The remove member is *constructed* from the keep member by writing exactly one
    leaf of exactly one record, so the one-fact invariant holds by construction.
    """
    mechanism = world["mechanism"]
    entity_a, entity_b = world["entity_a"], world["entity_b"]
    primary, secondary, tertiary = world["fields"]["primary"], world["fields"]["secondary"], world["fields"]["tertiary"]
    values = world["values"]
    if mechanism == "supersession":
        facts_keep = [
            {"entity": entity_a, "field": tertiary, "value": values["filler"]},
            {"entity": entity_b, "field": secondary, "value": values["other"]},
            {"entity": entity_a, "field": primary, "value": values["candidate"]},
            {"entity": entity_a, "field": secondary, "value": values["tail"]},
        ]
        pivot_index, pivot_field = 3, "field"
        pivot_values = {"keep": secondary, "remove": primary}
        candidate_index = 2
    elif mechanism == "entity_binding":
        facts_keep = [
            {"entity": entity_a, "field": tertiary, "value": values["filler"]},
            {"entity": entity_a, "field": primary, "value": values["anchor"]},
            {"entity": entity_a, "field": primary, "value": values["candidate"]},
            {"entity": entity_b, "field": secondary, "value": values["other"]},
        ]
        pivot_index, pivot_field = 2, "entity"
        pivot_values = {"keep": entity_a, "remove": entity_b}
        candidate_index = 2
    elif mechanism == "field_binding":
        facts_keep = [
            {"entity": entity_a, "field": tertiary, "value": values["filler"]},
            {"entity": entity_a, "field": primary, "value": values["anchor"]},
            {"entity": entity_a, "field": primary, "value": values["candidate"]},
            {"entity": entity_b, "field": secondary, "value": values["other"]},
        ]
        pivot_index, pivot_field = 2, "field"
        pivot_values = {"keep": primary, "remove": secondary}
        candidate_index = 2
    else:
        raise ValueError(f"unknown mechanism {mechanism!r}")
    facts_remove = deepcopy(facts_keep)
    facts_remove[pivot_index][pivot_field] = pivot_values["remove"]
    return facts_keep, facts_remove, pivot_index, pivot_field, pivot_values, candidate_index


def facts_for(world, member):
    keep, remove, _, _, _, _ = _layout(world)
    return keep if member == "keep" else remove


def query_spec(world):
    return {"entity": world["entity_a"], "fields": [world["fields"]["primary"]]}


def query_text(world):
    return (f"Using the records in chronological order, return the "
            f"{world['fields']['primary']} of {world['entity_a']}.")


def oracle_value(facts, entity, fields):
    wanted = set(fields)
    values = {fact["field"]: fact["value"] for fact in facts
              if fact["entity"] == entity and fact["field"] in wanted}
    if set(values) != wanted:
        return None
    return [values[field] for field in fields]


def oracle_decision(facts, entity, fields, candidate_index):
    """keep when removing the candidate record changes the answer, else remove."""
    reference = oracle_value(facts, entity, fields)
    if reference is None:
        raise ValueError("the queried item and field must be answerable before removal")
    dropped = oracle_value(facts[:candidate_index] + facts[candidate_index + 1:], entity, fields)
    return "keep" if reference != dropped else "remove"


def fact_pointer(wire_format, fact_index):
    """Message pointer of a record.

    ``openai_chat`` carries a system message at index 0; ``anthropic_messages``
    puts the system prompt in a top-level field, so records start at 0 there;
    ``openai_responses`` uses the ``input`` array.
    """
    if wire_format == "openai_responses":
        return f"/input/{fact_index}/content"
    if wire_format == "anthropic_messages":
        return f"/messages/{fact_index}/content"
    return f"/messages/{fact_index + 1}/content"


def render_body(wire_format, instructions, facts, question):
    records = [{"role": "assistant", "content": serialized(fact)} for fact in facts]
    user = {"role": "user", "content": question}
    if wire_format == "openai_chat":
        return {"model": "synthetic",
                "messages": [{"role": "system", "content": instructions}, *records, user]}
    if wire_format == "openai_responses":
        return {"model": "synthetic", "instructions": instructions, "input": [*records, user]}
    if wire_format == "anthropic_messages":
        return {"model": "synthetic", "system": instructions,
                "messages": [*records, user], "max_tokens": 64}
    raise ValueError(f"unsupported wire format {wire_format!r}")


def oracle_scorer(facts, entity, fields, pointer_to_index):
    """A fixed deterministic stub scorer standing in for the gate checkpoint.

    It answers only from the declared oracle over the records it is shown; it does
    not call a model, does not read the filesystem, and does not see gold labels.
    """
    def score(payload):
        states = []
        for state in payload["states"]:
            pointer = json.loads(state["state"])["candidate_pointer"]
            index = pointer_to_index.get(pointer)
            if index is None:
                raise KeyError(f"unscored candidate pointer {pointer!r}")
            decision = oracle_decision(facts, entity, fields, index)
            probability = 0.0 if decision == "keep" else 1.0
            states.append({
                "id": state["id"],
                "answers": {"irrelevant": {"type": "boolean", "probabilities": {
                    "false": 1.0 - probability, "true": probability}}},
            })
        return {"checkpoint": dict(STUB_CHECKPOINT), "states": states}
    return score


def verify_member(world, member, tag):
    """Render one member and prove its gate decision through the real shadow core."""
    wire = world["wire_format"]
    facts = facts_for(world, member)
    _, _, pivot_index, pivot_field, pivot_values, candidate_index = _layout(world)
    if facts[pivot_index][pivot_field] != pivot_values[member]:
        raise ValueError(f"{world['world_id']}: layout disagrees with the declared mutation")
    entity, fields = world["entity_a"], (world["fields"]["primary"],)
    body = render_body(wire, substitute(INSTRUCTIONS, tag), facts, query_text(world))
    raw = encode_request(body)
    try:
        segments = parse_segments(raw, wire)
    except Bypass as error:
        raise ValueError(f"{world['world_id']}/{member}: core bypassed with {error}") from None
    pointer_to_index = {fact_pointer(wire, index): index for index in range(len(facts))}
    declared_pointers = {segment.pointer for segment in segments}
    for pointer in pointer_to_index:
        if pointer not in declared_pointers:
            raise ValueError(f"{world['world_id']}/{member}: core did not enumerate {pointer}")
    focus_pointer = fact_pointer(wire, candidate_index)
    sidecar = {"segments": {pointer: {"eligible": True} for pointer in pointer_to_index}}
    output, receipt = shadow_request(raw, wire, sidecar,
                                     oracle_scorer(facts, entity, fields, pointer_to_index),
                                     threshold=0.99)
    if output is not raw:
        raise ValueError(f"{world['world_id']}/{member}: shadow core must return identical bytes")
    if receipt.get("status") != "scored":
        raise ValueError(f"{world['world_id']}/{member}: core did not score ({receipt.get('reason')})")
    focus = next(segment for segment in receipt["segments"] if segment["pointer"] == focus_pointer)
    decision = oracle_decision(facts, entity, fields, candidate_index)
    expected_suggestion = "retain" if decision == "keep" else "drop"
    if focus["suggestion"] != expected_suggestion:
        raise ValueError(f"{world['world_id']}/{member}: gate suggested {focus['suggestion']!r}, "
                         f"oracle requires {expected_suggestion!r}")
    if decision != member:
        raise ValueError(f"{world['world_id']}: member {member!r} is not the {member} member ({decision})")
    expected = {
        "schema_version": ITEM_SCHEMA,
        "pair_id": world["world_id"],
        "world_id": world["world_id"],
        "member": member,
        "decision": decision,
        "mechanism": world["mechanism"],
        "split": world["split"],
        "wire_format": wire,
        "source_group_id": source_group_id(world),
        "synthetic_tag": tag,
        "candidate_pointer": focus_pointer,
        "query": {"entity": entity, "fields": list(fields)},
        "oracle": {
            "reference": oracle_value(facts, entity, fields),
            "after_removal": oracle_value(facts[:candidate_index] + facts[candidate_index + 1:],
                                          entity, fields),
            "load_bearing": decision == "keep",
        },
        "gate": {
            "status": receipt["status"],
            "suggestion": focus["suggestion"],
            "reason": focus["reason"],
            "p_irrelevant": focus["p_irrelevant"],
            "policy_threshold": receipt["policy_threshold"],
            "probabilities_calibrated": receipt["probabilities_calibrated"],
            "applied": focus["applied"],
        },
        "provenance": dict(PROVENANCE),
        "training_authorized": False,
    }
    files = {
        "request.json": raw,
        "sidecar.json": pretty(sidecar),
        "expected.json": pretty(expected),
    }
    return {
        "member": member,
        "decision": decision,
        "candidate_pointer": focus_pointer,
        "oracle": dict(expected["oracle"]),
        "gate": dict(expected["gate"]),
        "body": body,
        "sidecar": sidecar,
        "expected": expected,
        "files": files,
    }


def derive_pair(world, seed):
    pair_id = world["world_id"]
    tag = synthetic_tag(pair_id, seed)
    members = {member: verify_member(world, member, tag) for member in ("keep", "remove")}
    decisions = {member: members[member]["decision"] for member in members}
    if decisions != {"keep": "keep", "remove": "remove"}:
        raise ValueError(f"{pair_id}: the one-fact mutation did not flip the gate decision ({decisions})")
    base_member = world["base_member"]
    other_member = "remove" if base_member == "keep" else "keep"
    _, _, pivot_index, pivot_field, pivot_values, candidate_index = _layout(world)
    mutation = {
        "fact_index": pivot_index,
        "fact_path": fact_pointer(world["wire_format"], pivot_index),
        "fact_field": pivot_field,
        "from_member": base_member,
        "to_member": other_member,
        "value_before": pivot_values[base_member],
        "value_after": pivot_values[other_member],
    }
    pair = {
        "pair_id": pair_id,
        "world_id": pair_id,
        "mechanism": world["mechanism"],
        "split": world["split"],
        "wire_format": world["wire_format"],
        "source_group_id": source_group_id(world),
        "synthetic_tag": tag,
        "direction": f"{base_member}_to_{other_member}",
        "decision_before": decisions[base_member],
        "decision_after": decisions[other_member],
        "candidate_pointer": fact_pointer(world["wire_format"], candidate_index),
        "mutation": mutation,
        "diff": {"changed_paths": [mutation["fact_path"]], "changed_fact_keys": [pivot_field]},
        "members": {
            member: {
                "decision": members[member]["decision"],
                "oracle": members[member]["oracle"],
                "gate": members[member]["gate"],
                "files": {name: {"bytes": len(content), "sha256": digest_bytes(content)}
                          for name, content in sorted(members[member]["files"].items())},
                "item_content_sha256": digest_value({
                    "body": members[member]["body"],
                    "sidecar": members[member]["sidecar"],
                    "expected": members[member]["expected"],
                }),
            }
            for member in ("keep", "remove")
        },
        "notes": ("This pair is training-shaped synthetic data. It is not evidence of model "
                  "behaviour, token savings, or production readiness."),
    }
    files = {member: members[member]["files"] for member in members}
    return pair, files


def _item_record(pair, member):
    record = pair["members"][member]
    mutation = pair["mutation"]
    return {
        "item_id": f"{pair['pair_id']}:{member}",
        "pair_id": pair["pair_id"],
        "member": member,
        "decision": record["decision"],
        "mechanism": pair["mechanism"],
        "split": pair["split"],
        "wire_format": pair["wire_format"],
        "source_group_id": pair["source_group_id"],
        "synthetic_tag": pair["synthetic_tag"],
        "candidate_pointer": pair["candidate_pointer"],
        "mutation": dict(mutation),
        "gate": dict(record["gate"]),
        "files": dict(record["files"]),
        "item_content_sha256": record["item_content_sha256"],
    }


def derive_manifest(base, seed):
    """Build the manifest (hashes only) and the materialized file bytes."""
    pairs, files = [], {}
    for world in base["worlds"]:
        pair, pair_files = derive_pair(world, seed)
        pairs.append(pair)
        for member, member_files in pair_files.items():
            for name, content in member_files.items():
                files[(pair["pair_id"], member, name)] = content
    items = [_item_record(pair, member) for pair in pairs for member in ("keep", "remove")]
    split_geometry = {}
    for split in SPLITS:
        split_pairs = [pair for pair in pairs if pair["split"] == split]
        split_geometry[split] = {
            "pairs": [pair["pair_id"] for pair in split_pairs],
            "source_groups": sorted({pair["source_group_id"] for pair in split_pairs}),
            "mechanisms": sorted({pair["mechanism"] for pair in split_pairs}),
        }
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": MANIFEST_STATUS,
        "created_by": "scripts/build_gate_contrastive_v1.py",
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": seed,
        "core": dict(CORE_CONTRACT),
        "contrastive_rule": deepcopy(CONTRASTIVE_RULE),
        "provenance": dict(PROVENANCE),
        "exclusions": {
            "policy": ("The builder refuses to read a base fixture from, or write output to, an "
                       "excluded evaluation location, and refuses any base fixture or emitted "
                       "file containing a reserved evaluation-corpus token. Evaluation cohorts "
                       "are never training sources."),
            "excluded_directory_parts": list(EXCLUDED_DIR_PARTS),
            "excluded_path_markers": list(EXCLUDED_PATH_MARKERS),
            "reserved_token_count": len(RESERVED_TOKENS),
            "reserved_tokens_sha256": digest_value(list(RESERVED_TOKENS)),
            "uses_relevance_test_or_ood_as_training_source": False,
        },
        "base_fixture": {
            "schema_version": base["schema_version"],
            "generation_rule": base.get("generation_rule", GENERATION_RULE),
            "seed": seed,
            "world_count": len(base["worlds"]),
            "worlds": deepcopy(base["worlds"]),
            "worlds_sha256": digest_value(base["worlds"]),
        },
        "split_geometry": split_geometry,
        "pair_count": len(pairs),
        "item_count": len(items),
        "pairs": pairs,
        "items": items,
        "pair_digest": digest_value(pairs),
        "item_digest": digest_value(items),
    }
    return manifest, files


# --------------------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------------------

def _first_difference(left, right, path=""):
    if type(left) is not type(right):
        return f"{path or '/'}: {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, dict):
        if set(left) != set(right):
            return f"{path or '/'}: keys differ ({sorted(set(left) ^ set(right))})"
        for key in sorted(left):
            difference = _first_difference(left[key], right[key], f"{path}/{key}")
            if difference:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path or '/'}: length {len(left)} != {len(right)}"
        for index, (first, second) in enumerate(zip(left, right)):
            difference = _first_difference(first, second, f"{path}/{index}")
            if difference:
                return difference
        return None
    if left != right:
        return f"{path or '/'}: {left!r} != {right!r}"
    return None


def validate_manifest(manifest):
    """Return integrity errors; empty means the manifest matches the frozen rule."""
    errors = []

    def fail(message):
        errors.append(message)

    if not isinstance(manifest, dict):
        return ["manifest must be a JSON object"]
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        fail(f"schema_version must be {MANIFEST_SCHEMA!r}")
    if type(manifest.get("seed")) is not int:
        fail("seed must be an integer")
    if manifest.get("status") != MANIFEST_STATUS:
        fail(f"status must be {MANIFEST_STATUS!r}")
    if manifest.get("created_by") != "scripts/build_gate_contrastive_v1.py":
        fail("created_by must name this builder")
    core = manifest.get("core", {})
    if core.get("core") != "scripts/context_gate_v1.py":
        fail("core.core must name scripts/context_gate_v1.py")
    if core.get("entrypoint") != "shadow_request":
        fail("core.entrypoint must be shadow_request")
    if core.get("threshold") != 0.99:
        fail("core.threshold must stay frozen at 0.99")
    if core.get("threshold_frozen") is not True:
        fail("core.threshold_frozen must be true")
    if core.get("active_filtering") is not False:
        fail("core.active_filtering must be false")
    if manifest.get("contrastive_rule") != CONTRASTIVE_RULE:
        fail("contrastive_rule must be the frozen rule")
    for flag, value in PROVENANCE.items():
        if not isinstance(value, bool):
            continue
        if manifest.get("provenance", {}).get(flag) is not value:
            fail(f"provenance.{flag} must be {value}")
    exclusions = manifest.get("exclusions", {})
    for key in ("policy", "excluded_directory_parts", "excluded_path_markers",
                "reserved_token_count", "reserved_tokens_sha256"):
        if key not in exclusions:
            fail(f"exclusions.{key} is required")
    if exclusions.get("reserved_token_count") != len(RESERVED_TOKENS):
        fail("exclusions.reserved_token_count must match the builder")
    if exclusions.get("reserved_tokens_sha256") != digest_value(list(RESERVED_TOKENS)):
        fail("exclusions.reserved_tokens_sha256 must match the builder")

    base = manifest.get("base_fixture")
    if not isinstance(base, dict):
        fail("base_fixture must be embedded so the manifest stays re-derivable")
        return errors
    for message in validate_base_fixture(base):
        fail(f"base_fixture: {message}")

    try:
        guard_text(serialized(manifest), "manifest")
    except ValueError as error:
        fail(str(error))

    try:
        rederived, _ = derive_manifest(base, manifest["seed"])
    except Exception as error:  # defensive: a corrupt manifest must not crash validation
        fail(f"re-derivation raised {type(error).__name__}: {error}")
        return errors

    difference = _first_difference(manifest.get("pairs"), rederived["pairs"], "/pairs")
    if difference:
        fail(f"declared pairs do not match the frozen construction rule: {difference}")
    difference = _first_difference(manifest.get("items"), rederived["items"], "/items")
    if difference:
        fail(f"declared items do not match the frozen construction rule: {difference}")
    if manifest.get("split_geometry") != rederived["split_geometry"]:
        fail("split_geometry must be re-derivable from the pairs")
    if manifest.get("pair_count") != len(manifest.get("pairs", [])):
        fail("pair_count must equal the declared pair count")
    if manifest.get("item_count") != len(manifest.get("items", [])):
        fail("item_count must equal the declared item count")
    if manifest.get("pair_digest") != digest_value(manifest.get("pairs")):
        fail("pair_digest does not match the declared pairs")
    if manifest.get("item_digest") != digest_value(manifest.get("items")):
        fail("item_digest does not match the declared items")
    if manifest.get("base_fixture", {}).get("worlds_sha256") != digest_value(
            manifest.get("base_fixture", {}).get("worlds")):
        fail("base_fixture.worlds_sha256 does not match the embedded worlds")

    pairs = manifest.get("pairs", [])
    if not isinstance(pairs, list) or not pairs:
        return errors + ["pairs must be a non-empty list"]
    groups, seen_pair_ids = {}, set()
    for pair in pairs:
        pair_id = pair.get("pair_id")
        if pair_id in seen_pair_ids:
            fail(f"pair_id {pair_id!r} is duplicated")
        seen_pair_ids.add(pair_id)
        group, split = pair.get("source_group_id"), pair.get("split")
        if group in groups and groups[group] != split:
            fail(f"source group {group!r} crosses splits")
        groups[group] = split
        if pair.get("decision_before") == pair.get("decision_after"):
            fail(f"pair {pair_id!r} must flip the decision")
        if {pair.get("decision_before"), pair.get("decision_after")} != {"keep", "remove"}:
            fail(f"pair {pair_id!r} must contain exactly one keep and one remove decision")
        expected_direction = f"{pair.get('mutation', {}).get('from_member')}_to_{pair.get('mutation', {}).get('to_member')}"
        if pair.get("direction") != expected_direction:
            fail(f"pair {pair_id!r} direction must match the recorded mutation")
        if pair.get("diff", {}).get("changed_paths") != [pair.get("mutation", {}).get("fact_path")]:
            fail(f"pair {pair_id!r} must declare exactly the mutated fact path")
    if len(set(groups)) != len(pairs):
        fail("every pair must own a unique source group")
    for split in SPLITS:
        geometry = manifest.get("split_geometry", {}).get(split, {})
        if geometry.get("pairs") and not geometry.get("source_groups"):
            fail(f"split {split!r} must declare its source groups")
    return errors


def guard_materialized(manifest, files):
    guard_text(serialized(manifest), "manifest")
    for (pair_id, member, name), content in sorted(files.items()):
        guard_text(content.decode("utf-8"), f"{pair_id}/{member}/{name}")


# --------------------------------------------------------------------------------------
# build / self-test / CLI
# --------------------------------------------------------------------------------------

def build(output_dir, seed=DEFAULT_SEED, base_fixture_path=None):
    """Validate, derive, and materialize the contrastive cohort; refuse unsafe targets."""
    reason = excluded_path_reason(output_dir)
    if reason:
        raise ValueError(f"output directory refused: {reason}")
    base = default_base_fixture(seed) if base_fixture_path is None else load_base_fixture(base_fixture_path)
    errors = validate_base_fixture(base)
    if errors:
        raise ValueError("base fixture validation failed: " + "; ".join(errors))
    manifest, files = derive_manifest(base, seed)
    guard_materialized(manifest, files)
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("derived manifest failed validation: " + "; ".join(errors))
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    for (pair_id, member, name), content in sorted(files.items()):
        directory = output / "pairs" / pair_id / member
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).write_bytes(content)
    (output / MANIFEST_NAME).write_bytes(pretty(manifest))
    return manifest


def self_test(seed=DEFAULT_SEED, base_fixture_path=None):
    """Validate the frozen rule end to end without writing any output."""
    try:
        base = default_base_fixture(seed) if base_fixture_path is None else load_base_fixture(base_fixture_path)
        errors = validate_base_fixture(base)
        manifest, files = (None, {}) if errors else derive_manifest(base, seed)
        if not errors:
            guard_materialized(manifest, files)
            errors = validate_manifest(manifest)
    except Exception as error:
        manifest, errors = None, [f"{type(error).__name__}: {error}"]
    pairs = [] if not manifest else manifest["pairs"]
    return {
        "schema_version": MANIFEST_SCHEMA,
        "mode": "self-test",
        "status": "failed" if errors else "ok",
        "seed": seed,
        "builder": "scripts/build_gate_contrastive_v1.py",
        "pairs": len(pairs),
        "items": len(pairs) * 2,
        "mechanisms": sorted({pair["mechanism"] for pair in pairs}),
        "splits": {split: len(manifest["split_geometry"][split]["pairs"])
                   for split in SPLITS} if pairs else {},
        "directions": sorted({pair["direction"] for pair in pairs}),
        "training_performed": False,
        "wrote_output": False,
        "errors": errors,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, help="empty directory to materialize the cohort into")
    parser.add_argument("--base-fixture", type=Path,
                        help="optional base fixture set JSON (must not be an evaluation corpus)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="deterministic generation seed")
    parser.add_argument("--self-test", action="store_true", help="validate the frozen rule without writing output")
    args = parser.parse_args(argv)
    if args.self_test and args.output_dir is not None:
        parser.error("--self-test and --output-dir are mutually exclusive")
    if not args.self_test and args.output_dir is None:
        parser.error("either --self-test or --output-dir is required")
    if args.self_test:
        report = self_test(args.seed, args.base_fixture)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] == "ok" else 1
    try:
        manifest = build(args.output_dir, args.seed, args.base_fixture)
    except ValueError as error:
        print(json.dumps({"status": "failed", "error": str(error)}, indent=2, sort_keys=True))
        return 2
    print(json.dumps({
        "status": "ok",
        "output_dir": str(args.output_dir),
        "seed": manifest["seed"],
        "pairs": manifest["pair_count"],
        "items": manifest["item_count"],
        "splits": {split: len(manifest["split_geometry"][split]["pairs"]) for split in SPLITS},
        "pair_digest": manifest["pair_digest"],
        "item_digest": manifest["item_digest"],
        "training_performed": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
