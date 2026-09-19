#!/usr/bin/env python3
"""Bounded record-dependency curriculum with oracle deletion labels and grouped splits."""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

from context_gate_v1 import parse_segments, scoring_payload, serialized
from train_pipeline_decisions import validate_training_row


SPLITS = {"train": 1200, "dev": 240, "calibration": 240, "test": 360, "ood": 360}
FAMILIES = {"code": ("port", "workers"), "order": ("unit_price", "quantity"),
            "risk": ("position_limit", "order_limit"), "support": ("refund_days", "response_hours"),
            "robotics": ("speed_limit", "distance_limit"), "multilingual": ("时间", "地点编号")}
KINDS = ("required_field", "two_fields", "latest_correction", "limit_check",
         "wrong_entity", "wrong_field", "superseded", "unrelated")
WIRES = ("openai_chat", "openai_responses", "anthropic_messages")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def oracle(facts, entity, fields, quantity=None):
    values = {}
    for fact in facts:
        if fact["entity"] == entity and fact["field"] in fields:
            values[fact["field"]] = fact["value"]
    if set(values) != set(fields):
        return None
    if quantity is not None:
        return quantity <= values[fields[0]]
    return tuple(values[field] for field in fields)


def make_record(family, world, kind, split, wire, rng):
    if family not in FAMILIES or kind not in KINDS or wire not in WIRES:
        raise ValueError("unknown family/kind/wire")
    a, b, c, quantity = world
    if len(set(world)) != 4:
        raise ValueError("world values must be distinct")
    group = digest({"world": world})  # Never include split names or incidental record IDs.
    subject, other = f"object-{group[:8]}", f"object-{group[8:16]}"
    field, second = FAMILIES[family]
    fields, limit = [field], None
    candidate = {"entity": subject, "field": field, "value": a}
    facts, candidate_index = [candidate], 0
    if kind == "two_fields":
        fields = [field, second]
        facts.append({"entity": subject, "field": second, "value": b})
    elif kind == "latest_correction":
        facts.insert(0, {"entity": subject, "field": field, "value": b})
        candidate_index = 1
    elif kind == "limit_check":
        limit = quantity
    elif kind == "wrong_entity":
        candidate["entity"] = other
        facts.append({"entity": subject, "field": field, "value": b})
    elif kind == "wrong_field":
        candidate["field"] = second
        facts.append({"entity": subject, "field": field, "value": b})
    elif kind == "superseded":
        facts.append({"entity": subject, "field": field, "value": b})
    elif kind == "unrelated":
        candidate.update(entity=other, field=second)
        facts.append({"entity": subject, "field": field, "value": b})
    # A third non-target field varies placement in every arm, independently of the label.
    position = rng.randrange(len(facts) + 1)
    facts.insert(position, {"entity": subject, "field": "revision", "value": c})
    candidate_index += position <= candidate_index
    reference = oracle(facts, subject, fields, limit)
    dropped = oracle(facts[:candidate_index] + facts[candidate_index+1:], subject, fields, limit)
    irrelevant = reference == dropped
    expected = kind in KINDS[4:]
    if reference is None or irrelevant != expected:
        raise ValueError("gold oracle and declared case disagree")
    if family == "multilingual":
        task = f"按时间顺序读取记录，后面的同一对象同一字段覆盖前面的值。请给出 {subject} 的 {', '.join(fields)}。"
    else:
        task = f"Read records in chronological order; the last value for an entity and field wins. Return {', '.join(fields)} for {subject}."
    if limit is not None:
        task += f" Instead of the value, decide whether the requested quantity {limit} is <= that limit."
    instructions = "Use only the records to answer. Do not invent missing values."
    messages = [{"role": "assistant", "content": serialized(fact)} for fact in facts]
    messages.append({"role": "user", "content": task})
    if wire == "openai_chat":
        messages.insert(0, {"role": "system", "content": instructions})
        body = {"model": "synthetic", "messages": messages}
        pointer = f"/messages/{candidate_index+1}/content"
    elif wire == "openai_responses":
        body = {"model": "synthetic", "instructions": instructions, "input": messages}
        pointer = f"/input/{candidate_index}/content"
    else:
        body = {"model": "synthetic", "system": instructions, "messages": messages, "max_tokens": 32}
        pointer = f"/messages/{candidate_index}/content"
    segments = parse_segments(serialized(body).encode(), wire)
    candidate_segment = next(segment for segment in segments if segment.pointer == pointer)
    prepared = scoring_payload(segments, [candidate_segment])["states"][0]
    uid = f"context:{group}:{kind}"
    row = {"id": uid, "state_id": uid, "family_id": f"context_{family}", "split": split,
           "state": prepared["state"], "questions": prepared["questions"], "gold": {"irrelevant": irrelevant},
           "gold_probs": {"irrelevant": {"false": float(not irrelevant), "true": float(irrelevant)}},
           "gold_probs_kind": "deterministic_truth", "gold_label_kind": "deterministic_truth",
           "metadata": {"source_group_id": group, "world": list(world), "scenario_family": family,
                        "candidate_kind": kind, "wire_format": wire, "source": "self_authored_oracle_lookup",
                        "license": "CC0-1.0", "body": body, "candidate_pointer": pointer,
                        "facts": deepcopy(facts), "candidate_index": candidate_index,
                        "query": {"entity": subject, "fields": fields, "quantity": limit},
                        "oracle_before": reference, "oracle_after": dropped}}
    validate_training_row(row)
    return row


def validate_cohort(rows):
    groups, inputs = {}, {}
    for row in rows:
        metadata = row["metadata"]
        group = digest({"world": tuple(metadata["world"])})
        if group != metadata["source_group_id"]:
            raise ValueError("source group must depend on actual facts, never split identity")
        if group in groups and groups[group] != row["split"]:
            raise ValueError("semantic world crosses splits")
        groups[group] = row["split"]
        key = digest({"state": row["state"], "questions": row["questions"]})
        if key in inputs:
            raise ValueError("duplicate or contradictory model input")
        inputs[key] = row["gold"]
        facts, index, query = metadata["facts"], metadata["candidate_index"], metadata["query"]
        expected = oracle(facts, **query) == oracle(facts[:index] + facts[index+1:], **query)
        if row["gold"]["irrelevant"] != expected:
            raise ValueError("gold does not match deletion oracle")


def build(output, seed=20260919):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty")
    rng = random.Random(seed)
    worlds, seen = [], set()
    count = sum(SPLITS.values()) // len(KINDS)
    while len(worlds) < count:
        world = tuple(rng.sample(range(10, 1000), 4))
        if world not in seen:
            seen.add(world); worlds.append(world)
    rng.shuffle(worlds)
    rows, offset = [], 0
    for split, size in SPLITS.items():
        families = tuple(FAMILIES)[:4] if split != "ood" else tuple(FAMILIES)[4:]
        for group_index in range(size // len(KINDS)):
            world = worlds[offset]; offset += 1
            family = families[group_index % len(families)]
            for kind_index, kind in enumerate(KINDS):
                rows.append(make_record(family, world, kind, split, WIRES[(group_index+kind_index) % 3], rng))
    validate_cohort(rows)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": "nanojev-context-relevance-oracle-v1", "seed": seed,
                "source": "self_authored_programmatic", "license": "CC0-1.0",
                "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "limitations": ["Synthetic latest-record dependency labels, not general semantic relevance or production evidence.",
                                "All eight variants share a semantic world and remain in one split; split names never define source hashes.",
                                "OOD holds out robotics and Chinese domains; shares oracle and rendering structure, not an unseen reasoning task.",
                                "No head selection or threshold tuning on calibration/test/OOD during training."], "splits": {}}
    for split in SPLITS:
        items = [row for row in rows if row["split"] == split]
        content = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in items)
        (output / f"{split}.jsonl").write_text(content, encoding="utf-8")
        manifest["splits"][split] = {"records": len(items), "source_groups": len({r["metadata"]["source_group_id"] for r in items}),
                                     "families": dict(Counter(r["family_id"] for r in items)),
                                     "kinds": dict(Counter(r["metadata"]["candidate_kind"] for r in items)),
                                     "labels": dict(Counter(str(r["gold"]["irrelevant"]) for r in items)),
                                     "sha256": hashlib.sha256(content.encode()).hexdigest()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()
    print(json.dumps(build(args.output_dir, args.seed), indent=2))
