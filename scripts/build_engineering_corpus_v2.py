#!/usr/bin/env python3
"""Engineering-judgment corpus V2: split assignment over connected components.

Repairs the two T8g audit findings without touching V1 (``engineering_judgment_corpus_v1``
stays preserved as evidence):

1. **Canonical-input leakage.** V1 assigned splits per pair by catalog position, so two
   pairs whose members render identical model-visible state+question text could land in
   different splits (27 cross-split groups, 63 records). V2 first builds the union-find
   components over (a) both members of a contrastive pair and (b) any members sharing an
   identical canonical visible input, then assigns one split per component. A canonical
   input can never cross a split boundary by construction.
2. **Evaluation-derived provenance.** Three rules whose declared source aliases the
   pre-registered abstention survey are removed outright:
   ``lifecycle-data-missing``, ``routing-budget-exceeded``,
   ``checkpoint-invalid-outputs``. Rewording does not create independent evidence, so
   the pairs are dropped, not relabelled. The corpus shrinks honestly rather than being
   padded back to a target count.

Additional v2 contract changes:

* ``source_group_id`` is now the SHA-256 of the component's sorted canonical-input
  fingerprints (semantic content), not of ``seed+family+pair_id``. Splits are still
  assigned by deterministic per-family component position, never by content value.
* ``derived_from_evaluation_corpus`` remains ``false`` and is now true by construction:
  every surviving rule's declared source passed the normalized-alias check that the V1
  builder's underscore-only path markers missed.
* No training is authorized by this corpus; the T9d protocol review and the merged-corpus
  adapter review remain separate gates.

Usage

    .venv/bin/python scripts/build_engineering_corpus_v2.py --self-test
    .venv/bin/python scripts/build_engineering_corpus_v2.py --output-dir research/engineering_judgment_corpus_v2
    .venv/bin/python scripts/build_engineering_corpus_v2.py --check research/engineering_judgment_corpus_v2
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import build_engineering_corpus_v1 as v1
from build_gate_contrastive_v1 import digest_value

SCHEMA_VERSION = "nanojev-engineering-judgment-corpus-v2"
MANIFEST_SCHEMA = "nanojev-engineering-judgment-manifest-v2"
SOURCE_GROUP_SCHEMA = "nanojev-engineering-source-group-v2"
CATALOG_VERSION = "engineering-judgment-catalog-v1-componented"  # same rules, new grouping
ITEM_DIR, TRAINER_DIR, MANIFEST_NAME = v1.ITEM_DIR, v1.TRAINER_DIR, v1.MANIFEST_NAME
SPLITS, SPLIT_CYCLE, QUESTION_TYPES, FAMILIES = (v1.SPLITS, v1.SPLIT_CYCLE,
                                               v1.QUESTION_TYPES, v1.FAMILIES)
DEFAULT_SEED = v1.DEFAULT_SEED

EVALUATION_SOURCE_MARKERS = ("abstention survey", "workflow challenge",
                             "workflow v2 evaluation", "context relevance test",
                             "context relevance ood")


def evaluation_derived(source_id):
    normalized = re.sub(r"[-_\s]+", " ", str(source_id).lower())
    return any(marker in normalized for marker in EVALUATION_SOURCE_MARKERS)


def visible_fingerprint(item):
    """SHA-256 of exactly the bytes the model sees: state text + question body."""
    state = item["request"]["states"][0]
    question = state["questions"][item["qid"]]
    return digest_value({"state": state["state"], "type": question["type"],
                         "instructions": question["instructions"],
                         "criteria": question.get("criteria", {})})


class _Components:
    def __init__(self):
        self.parent = {}

    def find(self, node):
        parent = self.parent
        while parent.setdefault(node, node) != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def derive_manifest(seed=DEFAULT_SEED):
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    v1._validate_tables()
    pairs = v1.pairs_for(seed)

    dropped_rules = sorted({pair["rule_id"] for pair in pairs
                            if evaluation_derived(pair["rule"]["source_id"])})
    pairs = [pair for pair in pairs if pair["rule_id"] not in dropped_rules]

    # Materialize items with placeholder split/group, then group by visible content.
    members = {}   # (pair_id, member) -> items
    components = _Components()
    fingerprint_nodes = {}
    for pair in pairs:
        pair["split"], pair["source_group_id"] = "train", "pending"
        nodes = []
        for member in ("base", "variant"):
            items = v1.make_items(pair, member)
            node = (pair["pair_id"], member)
            members[node] = items
            nodes.append(node)
            for item in items:
                fp = visible_fingerprint(item)
                if fp in fingerprint_nodes:
                    components.union(node, fingerprint_nodes[fp])
                else:
                    fingerprint_nodes[fp] = node
        components.union(*nodes)

    grouped = {}
    for node in members:
        grouped.setdefault(components.find(node), []).append(node)

    # Deterministic, content-independent order: catalog position of the earliest pair.
    catalog_order = {pair["pair_id"]: index for index, pair in enumerate(pairs)}
    per_family = {}
    for comp_nodes in grouped.values():
        first = min(catalog_order[node[0]] for node in comp_nodes)
        family = next(pair["family"] for pair in pairs if pair["pair_id"] == comp_nodes[0][0])
        per_family.setdefault(family, []).append((first, comp_nodes))

    for family, comps in per_family.items():
        for index, (_, comp_nodes) in enumerate(sorted(comps)):
            split = SPLIT_CYCLE[(index + abs(seed)) % len(SPLIT_CYCLE)]
            fps = sorted({visible_fingerprint(item) for node in comp_nodes
                          for item in members[node]})
            group_id = "ejc-v2-" + digest_value(
                {"schema": SOURCE_GROUP_SCHEMA, "canonical_inputs": fps})[:32]
            for node in comp_nodes:
                for item in members[node]:
                    item["split"] = split
                    item["source_group_id"] = group_id
                    item["provenance"]["component_size_pairs"] = len({n[0] for n in comp_nodes})
                    item.pop("item_content_sha256", None)
                    item["item_content_sha256"] = digest_value(item)
            for pair in pairs:
                if any(node[0] == pair["pair_id"] for node in comp_nodes):
                    pair["split"] = split
                    pair["source_group_id"] = group_id

    items = [item for node in members for item in members[node]]
    for item in items:
        v1.guard_source(item["provenance"]["source_id"], item, f"item {item['item_id']}")

    pair_records = [{
        "pair_id": pair["pair_id"], "rule_id": pair["rule_id"], "state_id": pair["state_id"],
        "family": pair["family"], "feature": pair["feature"], "split": pair["split"],
        "source_group_id": pair["source_group_id"], "mutated_fact": pair["mutated_fact"],
        "value_before": pair["value_before"], "value_after": pair["value_after"],
        "flip_question_types": list(pair["flip_question_types"]),
        "gold_before": pair["gold_before"], "gold_after": pair["gold_after"],
        "item_ids": [item["item_id"] for item in items if item["pair_id"] == pair["pair_id"]],
        "declared_flip": list(pair["declared_flip"]),
    } for pair in pairs]

    split_geometry = {}
    for split in SPLITS:
        split_items = [item for item in items if item["split"] == split]
        split_geometry[split] = {
            "items": len(split_items),
            "pairs": sum(1 for pair in pairs if pair["split"] == split),
            "source_groups": sorted({item["source_group_id"] for item in split_items}),
            "families": sorted({item["family"] for item in split_items}),
            "question_types": sorted({item["question_type"] for item in split_items}),
        }

    provenance = dict(v1.PROVENANCE)
    construction = deepcopy(v1.CONSTRUCTION_RULE)
    construction["version"] = "engineering-judgment-rule-v2"
    construction["split_unit"] = ("canonical-input connected component: both members of a "
                                  "contrastive pair plus every member sharing an identical "
                                  "visible state+question always share one split")
    construction["split_rule"] = ("deterministic per-family component position + seed offset; "
                                  "content never selects a split")
    construction["removed_evaluation_derived_rules"] = dropped_rules

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "training_data_prerequisite_no_training_authorized",
        "corpus_role": "training_data_prerequisite_only",
        "created_by": "scripts/build_engineering_corpus_v2.py",
        "catalog_version": CATALOG_VERSION,
        "repairs": {
            "v1_audit": "docs/T8G_ADAPTER_REVIEW_V1.md",
            "canonical_input_components": len(grouped),
            "removed_evaluation_derived_rules": dropped_rules,
            "v1_corpus_preserved": "research/engineering_judgment_corpus_v1 unchanged",
        },
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": seed,
        "contract": {
            "validator": "scripts/predict_toy_decisions.py:validate_request",
            "shape": ('{"states":[{"id","state","questions":'
                      '{qid:{"type","instructions","criteria"}}}]}'),
            "trainer_row_contract": f"{v1.TRAINER}:validate_training_row",
            "views": {"item": f"{ITEM_DIR}/<split>.jsonl",
                      "trainer": f"{TRAINER_DIR}/<split>.jsonl"},
            "validated": True,
        },
        "construction_rule": construction,
        "provenance": provenance,
        "source_group_count": len({item["source_group_id"] for item in items}),
        "pair_count": len(pairs),
        "item_count": len(items),
        "counts_by_family": v1._family_counts(items, pairs),
        "counts_by_split": {split: split_geometry[split]["items"] for split in SPLITS},
        "counts_by_question_type": {qtype: sum(1 for item in items
                                               if item["question_type"] == qtype)
                                    for qtype in QUESTION_TYPES},
        "split_geometry": split_geometry,
        "pairs": pair_records,
        "items": items,
        "item_digest": digest_value(items),
        "pair_digest": digest_value(pair_records),
        "content_sha256": None,
    }
    manifest["exclusions"] = {
        "policy": ("V2 additionally refuses evaluation-derived declared sources by normalized "
                   "alias, and removes the three rules that carried them. Evaluation cohorts "
                   "are never a training, validation, calibration or selection source."),
        "removed_rules": dropped_rules,
        "uses_evaluation_corpus_as_source": False,
    }
    manifest["content_sha256"] = digest_value({key: value for key, value in manifest.items()
                                               if key not in ("content_sha256", "builder_sha256")})
    return manifest


def trainer_rows(manifest):
    return [v1.trainer_row(item) for item in manifest["items"]]


def output_files(manifest):
    files = {f"{ITEM_DIR}/{v1.SPLIT_FILES[split]}":
             v1._jsonl([item for item in manifest["items"] if item["split"] == split])
             for split in SPLITS}
    rows = trainer_rows(manifest)
    files.update({f"{TRAINER_DIR}/{v1.SPLIT_FILES[split]}":
                  v1._jsonl([row for row in rows if row["split"] == split])
                  for split in SPLITS})
    files[MANIFEST_NAME] = v1.pretty(manifest)
    return files


def validate_manifest(manifest):
    """Reuse the V1 structural validators; v2 items keep the v1 item schema fields."""
    errors = []
    for item in manifest["items"]:
        errors.extend(v1.validate_item(item))
    try:
        for row in trainer_rows(manifest):
            v1.validate_trainer_row(row)
    except Exception as error:  # noqa: BLE001 - surfaced as a validation error, not a crash
        errors.append(f"trainer row validation failed: {type(error).__name__}: {error}")
    # v2-specific: no canonical input may appear in two splits.
    seen = {}
    for item in manifest["items"]:
        fp = visible_fingerprint(item)
        if fp in seen and seen[fp] != item["split"]:
            errors.append(f"canonical input crosses splits: {item['item_id']} "
                          f"({seen[fp]} vs {item['split']})")
        seen[fp] = item["split"]
    return errors


def build(output_dir, seed=DEFAULT_SEED):
    manifest = derive_manifest(seed)
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("derived corpus failed validation: " + "; ".join(errors))
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    for name, content in sorted(output_files(manifest).items()):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return manifest


def check(output_dir):
    try:
        manifest = json.loads((Path(output_dir) / MANIFEST_NAME).read_text(encoding="utf-8"))
    except Exception as error:
        return [f"cannot read {MANIFEST_NAME}: {type(error).__name__}: {error}"]
    errors = validate_manifest(manifest)
    try:
        expected = output_files(manifest)
    except Exception as error:
        return errors + [f"cannot re-derive the views from this manifest: "
                         f"{type(error).__name__}: {error}"]
    for name, content in sorted(expected.items()):
        path = Path(output_dir) / name
        if not path.is_file():
            errors.append(f"{name} is missing")
        elif path.read_bytes() != content:
            errors.append(f"{name} does not match the re-derived corpus")
    return errors


def self_test(seed=DEFAULT_SEED):
    try:
        manifest = derive_manifest(seed)
        errors = validate_manifest(manifest)
        difference = v1._first_difference(manifest, derive_manifest(seed), "/")
        if difference:
            errors.append(f"a second in-process derivation differs: {difference}")
        if output_files(manifest)[MANIFEST_NAME] != v1.pretty(manifest):
            errors.append("the manifest view is not stable")
    except Exception as error:
        manifest, errors = None, [f"{type(error).__name__}: {error}"]
    return {
        "schema_version": MANIFEST_SCHEMA, "mode": "self-test",
        "status": "failed" if errors else "ok", "seed": seed,
        "builder": "scripts/build_engineering_corpus_v2.py",
        "components": manifest["repairs"]["canonical_input_components"] if manifest else 0,
        "removed_rules": manifest["repairs"]["removed_evaluation_derived_rules"] if manifest else [],
        "source_groups": manifest["source_group_count"] if manifest else 0,
        "pairs": manifest["pair_count"] if manifest else 0,
        "items": manifest["item_count"] if manifest else 0,
        "items_by_split": dict(manifest["counts_by_split"]) if manifest else {},
        "training_performed": False, "wrote_output": False, "errors": errors,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, help="empty directory for the corpus")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--check", type=Path, help="re-derive and verify an existing corpus")
    args = parser.parse_args(argv)
    if args.self_test and (args.output_dir is not None or args.check is not None):
        parser.error("--self-test is mutually exclusive with --output-dir/--check")
    if not args.self_test and args.output_dir is None and args.check is None:
        parser.error("one of --self-test, --output-dir or --check is required")
    if args.self_test:
        report = self_test(args.seed)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] == "ok" else 1
    if args.check is not None:
        errors = check(args.check)
        print(json.dumps({"status": "ok" if not errors else "failed",
                          "checked": str(args.check), "errors": errors,
                          "training_performed": False}, indent=2, sort_keys=True))
        return 0 if not errors else 1
    try:
        manifest = build(args.output_dir, args.seed)
    except ValueError as error:
        print(json.dumps({"status": "failed", "error": str(error)}, indent=2, sort_keys=True))
        return 2
    print(json.dumps({
        "status": "ok", "output_dir": str(args.output_dir), "seed": manifest["seed"],
        "components": manifest["repairs"]["canonical_input_components"],
        "removed_rules": manifest["repairs"]["removed_evaluation_derived_rules"],
        "source_groups": manifest["source_group_count"], "pairs": manifest["pair_count"],
        "items": manifest["item_count"],
        "items_by_split": dict(manifest["counts_by_split"]),
        "content_sha256": manifest["content_sha256"], "training_performed": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
