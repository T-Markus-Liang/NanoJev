#!/usr/bin/env python3
"""Tests for the deterministic engineering-judgment corpus builder (V1).

These tests actually execute ``scripts/build_engineering_corpus_v1.py``: they build
the corpus twice, diff every contrastive pair leaf by leaf, recompute each declared
gold answer independently from the item's own fact basis, re-run the real served
request validator and the real trainer row validator, and try to feed the builder the
evaluation corpora it must refuse.  Nothing here trains anything or touches a
checkpoint.
"""

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from build_engineering_corpus_v1 import (
    ALLOWED_QUESTION_KEYS, CATALOG_VERSION, DEFAULT_SEED, FAMILIES, FORBIDDEN_PATH_MARKERS,
    ITEM_DIR, MANIFEST_NAME, PROVENANCE, QUESTION_TYPES, RESERVED_TOKENS, SPLITS, TRAINER_DIR,
    build, check, digest_value, fact_delta, forbidden_path_reason, gold_for, reserved_hits,
    self_test, validate_item, validate_manifest, validate_trainer_row,
)


BUILDER = Path(__file__).resolve().parent / "build_engineering_corpus_v1.py"
CORPUS_DIR = Path(__file__).resolve().parent.parent / "research" / "engineering_judgment_corpus_v1"

#: Evaluation corpora this builder must refuse, with a concrete representative path
#: for each family.  The last entry is the pre-registered abstention survey.
REFUSED_SOURCES = (
    "dataset/games_v4/data/local_maze_v1/test.jsonl",
    "dataset/games_v4/data/local_maze_v1/ood.jsonl",
    "dataset/games_v4/data/local_maze_v1/ood-far.jsonl",
    "research/context_relevance_v1/test.jsonl",
    "research/context_relevance_v1/ood.jsonl",
    "research/workflow_manifest_v2.json",
    "research/workflow_challenge_v2/test.jsonl",
    "research/tool_history_fixture_manifest_v1.json",
    "research/skill_abstention_survey_v1.json",
    "research/skill_abstention_survey_gated_run_v1.json",
    "results/nanojev_v2_baseline_seed17.json",
)


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def files_tree(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in sorted(Path(root).rglob("*")) if path.is_file()}


def leaf_diff(left, right, path=""):
    """Every differing JSON leaf, as (path, left, right)."""
    changes = []
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            child = f"{path}/{key}"
            if key not in left or key not in right:
                changes.append((child, left.get(key), right.get(key)))
            else:
                changes.extend(leaf_diff(left[key], right[key], child))
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            changes.append((path, left, right))
        else:
            for index, (first, second) in enumerate(zip(left, right)):
                changes.extend(leaf_diff(first, second, f"{path}/{index}"))
    elif left != right:
        changes.append((path, left, right))
    return changes


def option_keys(question):
    if question["type"] == "boolean":
        return ["false", "true"]
    if question["type"] == "choice":
        return list(question["criteria"])
    return [str(index) for index in range(len(question["criteria"]))]


class EngineeringCorpusBuilderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.output = cls.root / "corpus"
        cls.manifest = build(cls.output, DEFAULT_SEED)
        cls.items = cls.manifest["items"]
        cls.by_pair = {}
        for item in cls.items:
            cls.by_pair.setdefault(item["pair_id"], []).append(item)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    # ------------------------------------------------------------------ determinism

    def test_two_runs_are_byte_identical(self):
        first, second = self.root / "run-a", self.root / "run-b"
        first_manifest = build(first, DEFAULT_SEED)
        second_manifest = build(second, DEFAULT_SEED)
        self.assertEqual(first_manifest, second_manifest)
        self.assertEqual(files_tree(first), files_tree(second))
        self.assertEqual(files_tree(first), files_tree(self.output))
        self.assertEqual(first_manifest["content_sha256"], self.manifest["content_sha256"])

    def test_a_different_seed_changes_source_groups_but_not_the_skeleton(self):
        other = build(self.root / "seed-other", DEFAULT_SEED + 1)
        self.assertEqual([pair["pair_id"] for pair in other["pairs"]],
                         [pair["pair_id"] for pair in self.manifest["pairs"]])
        self.assertEqual({pair["source_group_id"] for pair in other["pairs"]}
                         & {pair["source_group_id"] for pair in self.manifest["pairs"]}, set())
        self.assertEqual(other["pair_count"], self.manifest["pair_count"])
        self.assertEqual(other["item_count"], self.manifest["item_count"])

    # ------------------------------------------------------------------ flip proof

    def test_every_pair_differs_in_exactly_one_fact_leaf(self):
        for pair_id, members in sorted(self.by_pair.items()):
            with self.subTest(pair=pair_id):
                bases = [item for item in members if item["member"] == "base"]
                variants = [item for item in members if item["member"] == "variant"]
                self.assertEqual(len(bases), len(QUESTION_TYPES))
                self.assertEqual(len(variants), len(QUESTION_TYPES))
                for base in bases:
                    variant = next(item for item in variants
                                   if item["question_type"] == base["question_type"])
                    delta = leaf_diff(base["provenance"]["fact_basis"],
                                      variant["provenance"]["fact_basis"])
                    self.assertEqual(len(delta), 1, delta)
                    self.assertEqual(delta[0][0].lstrip("/"),
                                     base["contrastive"]["mutated_fact"])
                    self.assertEqual(delta[0][1], base["contrastive"]["value_before"])
                    self.assertEqual(delta[0][2], base["contrastive"]["value_after"])
                    self.assertNotEqual(delta[0][1], delta[0][2])
                    # everything else about the two members is identical
                    self.assertEqual(base["request"]["states"][0]["id"],
                                     variant["request"]["states"][0]["id"])
                    self.assertEqual(base["qid"], variant["qid"])
                    self.assertEqual(base["split"], variant["split"])
                    self.assertEqual(base["source_group_id"], variant["source_group_id"])

    def test_declared_flip_actually_flips_and_gold_is_independently_recomputable(self):
        for pair_id, members in sorted(self.by_pair.items()):
            for item in members:
                with self.subTest(item=item["item_id"]):
                    # Recompute the expected answer from the item's own fact basis with the
                    # builder's rule (an independent call, not the stored value).
                    qid, gold = gold_for(item["family"], item["provenance"]["fact_basis"],
                                         item["question_type"])
                    self.assertEqual(qid, item["qid"])
                    self.assertEqual(gold, item["expected"]["gold"])
            bases = {item["question_type"]: item for item in members if item["member"] == "base"}
            variants = {item["question_type"]: item
                        for item in members if item["member"] == "variant"}
            declared = {qtype for qtype, item in bases.items()
                        if item["contrastive"]["is_flip_question"]}
            self.assertTrue(declared, pair_id)
            for qtype in QUESTION_TYPES:
                flipped = (bases[qtype]["expected"]["gold"] != variants[qtype]["expected"]["gold"])
                if qtype in declared:
                    with self.subTest(pair=pair_id, qtype=qtype):
                        self.assertTrue(flipped, f"{pair_id}:{qtype} declared a flip but did not "
                                                 f"flip ({bases[qtype]['expected']['gold']!r} -> "
                                                 f"{variants[qtype]['expected']['gold']!r})")
                self.assertEqual(flipped, qtype in declared)

    def test_pair_metadata_matches_the_materialized_members(self):
        for record in self.manifest["pairs"]:
            members = self.by_pair[record["pair_id"]]
            with self.subTest(pair=record["pair_id"]):
                self.assertEqual(record["split"], members[0]["split"])
                self.assertEqual(record["source_group_id"], members[0]["source_group_id"])
                self.assertEqual(sorted(record["flip_question_types"]),
                                 sorted({item["question_type"] for item in members
                                         if item["contrastive"]["is_flip_question"]}))
                self.assertEqual(record["mutated_fact"], members[0]["contrastive"]["mutated_fact"])
                self.assertEqual(sorted(record["item_ids"]),
                                 sorted(item["item_id"] for item in members))

    # ------------------------------------------------------------------ split geometry

    def test_splits_are_separated_by_source_group_and_state_id(self):
        groups, states = {}, {}
        for item in self.items:
            self.assertIn(item["split"], SPLITS)
            group, state, split = item["source_group_id"], item["state_id"], item["split"]
            if group in groups:
                self.assertEqual(groups[group], split, f"source group {group} crosses splits")
            groups[group] = split
            if state in states:
                self.assertEqual(states[state], split, f"state {state} crosses splits")
                self.assertEqual(states[state], split)
            states[state] = split
        self.assertEqual(len(groups), self.manifest["source_group_count"])
        for pair_id, members in self.by_pair.items():
            self.assertEqual(len({item["split"] for item in members}), 1, pair_id)
            self.assertEqual(len({item["source_group_id"] for item in members}), 1, pair_id)
        geometry = self.manifest["split_geometry"]
        self.assertEqual(sorted(geometry), sorted(SPLITS))
        self.assertEqual(sum(geometry[split]["items"] for split in SPLITS), len(self.items))
        test_groups = set(geometry["test"]["source_groups"])
        train_groups = set(geometry["train"]["source_groups"])
        self.assertEqual(test_groups & train_groups, set())
        self.assertEqual(set(geometry["test"]["families"]), set(FAMILIES) - set(
            family for family in FAMILIES
            if family not in geometry["test"]["families"]))
        for split in SPLITS:
            self.assertTrue(geometry[split]["items"] > 0, split)
            self.assertTrue(geometry[split]["families"], split)
        self.assertEqual(sorted(geometry["train"]["question_types"]), sorted(QUESTION_TYPES))

    # ------------------------------------------------------------------ contract

    def test_every_item_request_passes_the_real_served_validator(self):
        from predict_toy_decisions import prepare_examples, validate_request
        for item in self.items:
            with self.subTest(item=item["item_id"]):
                self.assertEqual(validate_request(item["request"]), item["request"]["states"])
                state = item["request"]["states"][0]
                self.assertEqual(set(state), {"id", "state", "questions"})
                self.assertEqual(set(item["request"]), {"states"})
                question = state["questions"][item["qid"]]
                self.assertEqual(set(question) - ALLOWED_QUESTION_KEYS, set())
                self.assertEqual(question["type"], item["question_type"])
                self.assertEqual(set(question["criteria"]) if isinstance(question["criteria"], dict)
                                 else set(), set(question["criteria"])
                                 if isinstance(question["criteria"], dict) else set())
                self.assertEqual(item["expected"]["candidate_keys"], option_keys(question))
                self.assertEqual(item["expected"]["distribution_kind"], "hard_label")
                self.assertFalse(item["expected"]["calibrated"])
                self.assertEqual(sum(item["expected"]["distribution"].values()), 1.0)
                self.assertEqual(item["expected"]["distribution"][
                    item["expected"]["candidate_keys"][item["expected"]["gold_index"]]], 1.0)
                state_questions = state["questions"]
                if item["question_type"] == "choice":
                    self.assertLessEqual(len(state_questions[item["qid"]]["criteria"]), 255)
                if item["question_type"] == "score":
                    self.assertLessEqual(len(state_questions[item["qid"]]["criteria"]), 10)
        # the same validator also accepts every emitted split file as one request set
        for split in SPLITS:
            items = [item for item in self.items if item["split"] == split]
            states, seen = [], set()
            for item in items:
                state = item["request"]["states"][0]
                if state["id"] in seen:
                    continue
                seen.add(state["id"])
                states.append(state)
            with self.subTest(split=split):
                self.assertEqual(len(states), len({item["state_id"] for item in items}))
                self.assertEqual(validate_request({"states": states}), states)

    def test_every_trainer_row_passes_the_real_trainer_contract(self):
        rows = [json.loads(line) for line in
                (self.output / TRAINER_DIR / "train.jsonl").read_text(
                    encoding="utf-8").splitlines() if line.strip()]
        self.assertTrue(rows)
        for row in rows:
            with self.subTest(row=row["id"]):
                validate_trainer_row(row)
                self.assertEqual(set(row) >= {"id", "state_id", "family_id", "split", "state",
                                              "questions", "gold", "gold_probs",
                                              "gold_probs_kind", "gold_label_kind"}, True)
                qid = next(iter(row["questions"]))
                self.assertEqual(row["gold_probs_kind"][qid], "deterministic_truth")
                self.assertEqual(row["gold_label_kind"][qid], "deterministic_truth")
                self.assertEqual(sum(row["gold_probs"][qid].values()), 1.0)
                gold = row["gold"][qid]
                keys = option_keys(row["questions"][qid])
                index = (1 if gold is True else 0) if row["questions"][qid]["type"] == "boolean" \
                    else (keys.index(gold) if row["questions"][qid]["type"] == "choice" else int(gold))
                self.assertEqual(row["gold_probs"][qid][keys[index]], 1.0)
                self.assertEqual(row["metadata"]["source_group_id"],
                                 next(item["source_group_id"] for item in self.items
                                      if item["item_id"] == row["id"]))
                self.assertFalse(row["metadata"]["derived_from_evaluation_corpus"])
                self.assertFalse(row["metadata"]["training_authorized_by_this_corpus"])

    def test_trainer_view_matches_the_item_view_one_to_one(self):
        items = self.items
        rows = []
        for split in SPLITS:
            rows.extend(jsonl(self.output / TRAINER_DIR / f"{split}.jsonl"))
        self.assertEqual(len(rows), len(items))
        self.assertEqual({row["id"] for row in rows}, {item["item_id"] for item in items})
        by_id = {item["item_id"]: item for item in items}
        for row in rows:
            item = by_id[row["id"]]
            self.assertEqual(row["id"], item["item_id"])
            self.assertEqual(row["state_id"], item["state_id"])
            self.assertEqual(row["family_id"], item["family"])
            self.assertEqual(row["split"], item["split"])
            self.assertEqual(row["state"], item["request"]["states"][0]["state"])
            self.assertEqual(row["questions"], {item["qid"]:
                                                item["request"]["states"][0]["questions"][item["qid"]]})
            self.assertEqual(row["gold"], {item["qid"]: item["expected"]["gold"]})

    def test_question_type_balance_and_family_counts_are_declared(self):
        counts = self.manifest["counts_by_family"]
        self.assertEqual(sorted(counts), sorted(FAMILIES))
        for family, record in counts.items():
            with self.subTest(family=family):
                self.assertEqual(record["items"],
                                 record["boolean"] + record["choice"] + record["score"])
                self.assertEqual(record["items"], sum(record["splits"].values()))
                self.assertGreater(record["items"], 0)
                self.assertGreater(record["pairs"], 0)
        self.assertEqual(sum(record["items"] for record in counts.values()), len(self.items))
        for qtype in QUESTION_TYPES:
            self.assertEqual(self.manifest["counts_by_question_type"][qtype],
                             sum(1 for item in self.items if item["question_type"] == qtype))
        self.assertEqual(sum(self.manifest["counts_by_split"].values()), len(self.items))

    # ------------------------------------------------------------------ provenance

    def test_every_item_records_stable_id_and_provenance(self):
        for item in self.items:
            with self.subTest(item=item["item_id"]):
                self.assertEqual(item["schema_version"],
                                 "nanojev-engineering-judgment-item-v1")
                self.assertTrue(item["item_id"])
                self.assertTrue(item["source_group_id"])
                self.assertTrue(item["pair_id"])
                self.assertIn(item["member"], ("base", "variant"))
                provenance = item["provenance"]
                self.assertTrue(provenance["source_id"])
                self.assertTrue(provenance["rule_id"])
                self.assertTrue(provenance["rule"])
                self.assertTrue(provenance["why_correct"])
                self.assertIn("authoring", provenance)
                self.assertIn(provenance["authoring"],
                              ("hand_authored_state_and_rule_with_programmatic_rendering",
                               "programmatic_one_fact_mutation_of_a_hand_authored_state"))
                self.assertIs(provenance["human_reviewed"], False)
                self.assertIs(provenance["derived_from_evaluation_corpus"], False)
                self.assertIs(provenance["training_authorized"], False)
                self.assertEqual(sorted(provenance["fact_keys"]), sorted(provenance["fact_basis"]))

    def test_no_item_or_manifest_leaks_a_reserved_token_or_refused_source(self):
        for item in self.items:
            with self.subTest(item=item["item_id"]):
                self.assertEqual(reserved_hits(json.dumps(item, ensure_ascii=False)), [])
                self.assertIsNone(forbidden_path_reason(item["provenance"]["source_id"]))
        manifest_text = (self.output / MANIFEST_NAME).read_text(encoding="utf-8")
        manifest_json = json.loads(manifest_text)
        self.assertEqual(reserved_hits(manifest_text), [])
        for name, content in files_tree(self.output).items():
            self.assertEqual(reserved_hits(content.decode("utf-8")), [], name)
        # the excluded-corpus names may only appear as refused-source *policy* text, never
        # as data: the reserved-token scan above is the tripwire, and no item may reference
        # an evaluation source at all (checked per item above).
        self.assertEqual(manifest_json["exclusions"]["uses_evaluation_corpus_as_source"], False)
        self.assertGreater(manifest_json["exclusions"]["refused_path_marker_count"], 0)
        self.assertGreater(manifest_json["exclusions"]["reserved_token_count"], 0)

    def test_evaluation_sources_are_refused(self):
        for source in REFUSED_SOURCES:
            with self.subTest(source=source):
                self.assertIsNotNone(forbidden_path_reason(source))
        for marker in FORBIDDEN_PATH_MARKERS:
            self.assertTrue(marker)
        # a legitimate source is not refused
        self.assertIsNone(forbidden_path_reason("docs/GATE_CONTRASTIVE_PROTOCOL_V1.md"))
        self.assertIsNone(forbidden_path_reason("scripts/build_engineering_corpus_v1.py"))
        # an evaluation corpus token in content is refused
        self.assertTrue(reserved_hits("speed_limit is not a field here"))

    def test_refusal_actually_fires_on_a_poisoned_source(self):
        import build_engineering_corpus_v1 as module
        original = module.CATALOG
        poisoned = dict(original[0])
        poisoned["source_id"] = "dataset/games_v4/data/local_maze_v1/test.jsonl"
        try:
            module.CATALOG = (poisoned,) + tuple(original[1:])
            with self.assertRaisesRegex(ValueError, "refused"):
                module.derive_manifest(DEFAULT_SEED)
        finally:
            module.CATALOG = original
        self.assertEqual(module.derive_manifest(DEFAULT_SEED)["pair_count"],
                         self.manifest["pair_count"])

    def test_refusal_actually_fires_on_reserved_content(self):
        import build_engineering_corpus_v1 as module
        original = module.RESERVED_TOKENS
        try:
            module.RESERVED_TOKENS = tuple(original) + ("trained-invariance",)
            module._RESERVED_PATTERNS = tuple(
                (token, __import__("re").compile(r"(?<![0-9A-Za-z])" + __import__("re").escape(token)
                                                 + r"(?![0-9A-Za-z])"))
                for token in module.RESERVED_TOKENS)
            with self.assertRaisesRegex(ValueError, "reserved"):
                module.derive_manifest(DEFAULT_SEED)
        finally:
            module.RESERVED_TOKENS = original
            module._RESERVED_PATTERNS = tuple(
                (token, __import__("re").compile(r"(?<![0-9A-Za-z])" + __import__("re").escape(token)
                                                 + r"(?![0-9A-Za-z])"))
                for token in original)

    # ------------------------------------------------------------------ hashes

    def test_manifest_hashes_are_stable_and_per_item(self):
        on_disk = json.loads((self.output / MANIFEST_NAME).read_text(encoding="utf-8"))
        self.assertEqual(on_disk["item_digest"], digest_value(on_disk["items"]))
        self.assertEqual(on_disk["pair_digest"], digest_value(on_disk["pairs"]))
        for item in on_disk["items"]:
            with self.subTest(item=item["item_id"]):
                recomputed = digest_value({key: value for key, value in item.items()
                                           if key != "item_content_sha256"})
                self.assertEqual(item["item_content_sha256"], recomputed)
        content = {key: value for key, value in on_disk.items()
                   if key not in ("content_sha256", "builder_sha256")}
        self.assertEqual(on_disk["content_sha256"], digest_value(content))
        self.assertEqual(on_disk["exclusions"]["reserved_tokens_sha256"],
                         digest_value(list(RESERVED_TOKENS)))
        rebuild = self.root / "hash-rebuild"
        build(rebuild, DEFAULT_SEED)
        self.assertEqual((rebuild / MANIFEST_NAME).read_bytes(),
                         (self.output / MANIFEST_NAME).read_bytes())

    # ------------------------------------------------------------------ tamper detection

    def test_validation_rejects_hand_edits(self):
        on_disk = json.loads((self.output / MANIFEST_NAME).read_text(encoding="utf-8"))
        mutations = (
            lambda m: m["items"][0]["expected"].__setitem__("gold", not m["items"][0]["expected"]["gold"]),
            lambda m: m["items"][0]["provenance"]["fact_basis"].__setitem__(
                m["items"][0]["contrastive"]["mutated_fact"], "tampered"),
            lambda m: m["items"][0].__setitem__("split", "not_a_split"),
            lambda m: m["items"][0].__setitem__("split",
                                                "dev" if m["items"][0]["split"] != "dev" else "test"),
            lambda m: m["items"][0].__setitem__("item_id", "ejc-tampered"),
            lambda m: m["pairs"][0].__setitem__("mutated_fact", "not_a_fact"),
            lambda m: m["provenance"].__setitem__("training_performed", True),
            lambda m: m["provenance"].__setitem__("training_authorized_by_this_corpus", True),
            lambda m: m.__setitem__("status", "training_authorized"),
            lambda m: m.__setitem__("corpus_role", "training_corpus"),
            lambda m: m["contract"].__setitem__("validator", "some_other_validator"),
            lambda m: m.__setitem__("content_sha256", "0" * 64),
            lambda m: m.__setitem__("item_digest", "0" * 64),
            lambda m: m["exclusions"].__setitem__("uses_evaluation_corpus_as_source", True),
            lambda m: m["construction_rule"].__setitem__("version", "tampered"),
        )
        for index, mutate in enumerate(mutations):
            broken = json.loads(json.dumps(on_disk))
            mutate(broken)
            with self.subTest(mutation=index):
                self.assertTrue(validate_manifest(broken))

    def test_validate_item_rejects_tampered_items(self):
        item = json.loads(json.dumps(self.items[0]))
        self.assertEqual(validate_item(item), [])
        for tamper in (
            lambda i: i["expected"].__setitem__("gold", not i["expected"]["gold"]),
            lambda i: i["expected"].__setitem__("calibrated", True),
            lambda i: i["expected"].__setitem__("distribution_kind", "model_probability"),
            lambda i: i["provenance"].__setitem__("training_authorized", True),
            lambda i: i["provenance"].__setitem__("derived_from_evaluation_corpus", True),
            lambda i: i["provenance"].__setitem__("fact_basis",
                                                  {**i["provenance"]["fact_basis"],
                                                   i["contrastive"]["mutated_fact"]: "other"}),
            lambda i: i["request"]["states"][0]["questions"].__setitem__(
                i["qid"], {"type": i["question_type"], "instructions": "x", "criteria": {}}),
            lambda i: i.__setitem__("question_type", "score"),
            lambda i: i.__setitem__("split", "nowhere"),
        ):
            broken = json.loads(json.dumps(item))
            tamper(broken)
            with self.subTest():
                self.assertTrue(validate_item(broken))

    def test_check_handles_a_malformed_manifest_without_crashing(self):
        manifest_path = self.output / MANIFEST_NAME
        backup = manifest_path.read_bytes()
        try:
            manifest_path.write_bytes(b'{"schema_version": "nope"}\n')
            try:
                errors = check(self.output)
            except Exception as error:  # pragma: no cover - surfaced as a failure below
                self.fail(f"check() crashed on a malformed manifest: {type(error).__name__}: {error}")
            self.assertTrue(errors)
        finally:
            manifest_path.write_bytes(backup)
        self.assertEqual(check(self.output), [])

    def test_check_rejects_a_modified_tampered_file(self):
        self.assertEqual(check(self.output), [])
        target = self.output / ITEM_DIR / "train.jsonl"
        original = target.read_bytes()
        try:
            target.write_bytes(original + b'{"tampered": true}\n')
            errors = check(self.output)
            self.assertTrue(any("does not match" in error for error in errors), errors)
        finally:
            target.write_bytes(original)
        self.assertEqual(check(self.output), [])
        missing = self.output / TRAINER_DIR / "dev.jsonl"
        content = missing.read_bytes()
        try:
            missing.unlink()
            self.assertTrue(any("missing" in error for error in check(self.output)))
        finally:
            missing.write_bytes(content)
        self.assertEqual(check(self.output), [])

    # ------------------------------------------------------------------ CLI / safety

    def test_self_test_writes_nothing_and_reports_the_counts(self):
        report = self_test(DEFAULT_SEED)
        self.assertEqual(report["status"], "ok", report["errors"])
        self.assertFalse(report["wrote_output"])
        self.assertFalse(report["training_performed"])
        self.assertEqual(report["pairs"], self.manifest["pair_count"])
        self.assertEqual(report["items"], self.manifest["item_count"])
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run([sys.executable, str(BUILDER), "--self-test"], cwd=temporary,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('"status": "ok"', result.stdout)
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_cli_build_and_check(self):
        out = self.root / "cli-corpus"
        built = subprocess.run([sys.executable, str(BUILDER), "--output-dir", str(out),
                                "--seed", str(DEFAULT_SEED)], capture_output=True, text=True)
        self.assertEqual(built.returncode, 0, built.stderr)
        self.assertIn('"status": "ok"', built.stdout)
        self.assertIn('"training_performed": false', built.stdout)
        self.assertEqual(files_tree(out), files_tree(self.output))
        checked = subprocess.run([sys.executable, str(BUILDER), "--check", str(out)],
                                 capture_output=True, text=True)
        self.assertEqual(checked.returncode, 0, checked.stderr)
        self.assertIn('"status": "ok"', checked.stdout)
        help_result = subprocess.run([sys.executable, str(BUILDER), "--help"],
                                     capture_output=True, text=True)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        clash = subprocess.run([sys.executable, str(BUILDER), "--self-test", "--output-dir",
                                str(self.root / "never")], capture_output=True, text=True)
        self.assertEqual(clash.returncode, 2)
        self.assertFalse((self.root / "never").exists())
        neither = subprocess.run([sys.executable, str(BUILDER)], capture_output=True, text=True)
        self.assertEqual(neither.returncode, 2)

    def test_refuses_a_nonempty_output_directory(self):
        target = self.root / "occupied"
        build(target, DEFAULT_SEED)
        marker = target / "keep.txt"
        marker.write_text("sentinel", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "empty"):
            build(target, DEFAULT_SEED)
        self.assertEqual(marker.read_text(encoding="utf-8"), "sentinel")

    def test_builder_has_no_network_training_or_checkpoint_dependency(self):
        import ast
        source = BUILDER.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertEqual(imported & {"requests", "urllib", "socket", "torch", "openai",
                                     "safetensors", "subprocess", "transformers"}, set())
        # the only runtime imports are the stdlib, the real validators and the digest helper
        self.assertTrue(imported <= {"argparse", "copy", "hashlib", "json", "pathlib", "random",
                                     "re", "ast", "importlib", "sys", "__future__",
                                     "build_gate_contrastive_v1",
                                     "predict_toy_decisions"}, imported)
        self.assertNotIn("subprocess", source)
        self.assertNotIn("http://", source)
        self.assertNotIn("https://", source)
        self.assertIn("training_performed", source)
        self.assertIn("training_authorized_by_this_corpus", source)
        self.assertIn("checkpoint_read", source)

    def test_committed_corpus_on_disk_still_validates(self):
        if not (CORPUS_DIR / MANIFEST_NAME).is_file():
            self.skipTest("the corpus has not been materialized in this checkout")
        errors = check(CORPUS_DIR)
        self.assertEqual(errors, [], errors)
        on_disk = json.loads((CORPUS_DIR / MANIFEST_NAME).read_text(encoding="utf-8"))
        self.assertEqual(on_disk["catalog_version"], CATALOG_VERSION)
        self.assertIs(on_disk["provenance"]["training_authorized_by_this_corpus"], False)
        self.assertIs(on_disk["provenance"]["training_performed"], False)
        self.assertIs(on_disk["provenance"]["network_access"], False)
        self.assertEqual(on_disk["status"],
                         "training_data_prerequisite_no_training_authorized")
        self.assertEqual(on_disk["corpus_role"], "training_data_prerequisite_only")
        self.assertGreater(on_disk["pair_count"], 0)
        self.assertGreater(on_disk["item_count"], 0)
        for split in SPLITS:
            self.assertTrue((CORPUS_DIR / ITEM_DIR / f"{split}.jsonl").is_file())
            self.assertTrue((CORPUS_DIR / TRAINER_DIR / f"{split}.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
