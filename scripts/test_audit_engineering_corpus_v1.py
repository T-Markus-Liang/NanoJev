#!/usr/bin/env python3
"""Bounded stdlib unit tests for audit_engineering_corpus_v1.py.

The tests fabricate minimal in-memory trainer rows so the read-only preflight can be
exercised without building, merging or training on any real corpus. No network, model,
agent or partition is used.
"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import audit_engineering_corpus_v1 as audit


def question(qid, qtype="choice", instructions=None, criteria=None):
    """A minimal question body accepted by the served and trainer validators."""
    instructions = instructions or f"instructions for {qid}"
    if qtype == "choice":
        criteria = criteria if criteria is not None else {"a": f"A {qid}", "b": f"B {qid}"}
    elif qtype == "boolean":
        criteria = criteria if criteria is not None else {"false": f"F {qid}", "true": f"T {qid}"}
    elif qtype == "score":
        criteria = criteria if criteria is not None else ["low", "mid", "high"]
    else:
        raise ValueError("unsupported question type")
    return {"type": qtype, "instructions": instructions, "criteria": criteria}


def answer_for(q):
    """A one-hot deterministic target matching the offered candidates."""
    if q["type"] == "boolean":
        return True, {"false": 0.0, "true": 1.0}
    if q["type"] == "choice":
        keys = list(q["criteria"])
        return keys[0], {key: (1.0 if key == keys[0] else 0.0) for key in keys}
    ids = [str(i) for i in range(len(q["criteria"]))]
    return 0, {cid: (1.0 if cid == "0" else 0.0) for cid in ids}


def make_row(rid="r1", split="train", state="evidence alpha", questions=None,
             state_id=None, source_group_id="src-1",
             provenance_source_id="fact_rule_1", declared_derived=None,
             gold_override=None, gold_probs_override=None, metadata_extra=None):
    """One minimal row in the train_pipeline_decisions record contract."""
    questions = questions if questions is not None else {"q1": question("q1")}
    gold, gold_probs, kinds = {}, {}, {}
    for qid, q in questions.items():
        g, p = answer_for(q)
        gold[qid] = g
        gold_probs[qid] = p
        kinds[qid] = "deterministic_truth"
    if gold_override:
        gold.update(gold_override)
    if gold_probs_override:
        gold_probs.update(gold_probs_override)
    metadata = {"source_group_id": source_group_id,
                "provenance_source_id": provenance_source_id}
    if declared_derived is not None:
        metadata["derived_from_evaluation_corpus"] = declared_derived
    if metadata_extra:
        metadata.update(metadata_extra)
    return {"id": rid, "state_id": state_id or f"state-{rid}",
            "family_id": "family-1", "split": split, "state": state,
            "questions": questions, "gold": gold, "gold_probs": gold_probs,
            "gold_probs_kind": kinds, "gold_label_kind": dict(kinds),
            "metadata": metadata}


def six_question_row(rid="multi", split="train", source_group_id="src-multi"):
    questions = {
        f"q{i}": question(f"q{i}", instructions=f"instructions number {i}",
                          criteria={"a": f"A{i}", "b": f"B{i}"})
        for i in range(6)
    }
    return make_row(rid=rid, split=split, questions=questions,
                    source_group_id=source_group_id)


class CanonicalInputTests(unittest.TestCase):
    def test_ids_qids_state_ids_sources_and_gold_are_excluded(self):
        r1 = make_row(rid="r1", state_id="st1", source_group_id="s1",
                      provenance_source_id="p1", state="evidence same",
                      questions={"q1": question("q1", instructions="pick one",
                                                criteria={"a": "A", "b": "B"})})
        r2 = make_row(rid="r2", state_id="st2", source_group_id="s2",
                      provenance_source_id="p2", state="evidence same",
                      questions={"q2": question("q2", instructions="pick one",
                                                criteria={"a": "A", "b": "B"})},
                      gold_override={"q2": "b"},
                      gold_probs_override={"q2": {"a": 0.0, "b": 1.0}})
        self.assertNotEqual(r1["id"], r2["id"])
        self.assertNotEqual(r1["state_id"], r2["state_id"])
        self.assertNotEqual(r1["gold"], r2["gold"])
        buckets = audit.references([r1, r2])
        self.assertEqual(len(buckets), 1, "IDs/qids/gold must not change the input digest")

    def test_visible_input_is_only_state_type_instructions_criteria(self):
        row = make_row(rid="r1")
        visible = audit.visible_input(row, row["questions"]["q1"])
        self.assertEqual(set(visible), {"state", "type", "instructions", "criteria"})
        for excluded in ("id", "qid", "source_group_id", "gold", "target"):
            self.assertNotIn(excluded, visible)

    def test_changed_evidence_produces_a_distinct_input(self):
        r1 = make_row(rid="r1", state="evidence alpha")
        r2 = make_row(rid="r2", state="evidence beta")
        buckets = audit.references([r1, r2])
        self.assertEqual(len(buckets), 2, "different state evidence must not collide")

    def test_choice_dict_order_normalized_but_score_list_order_retained(self):
        choice_a = make_row(rid="ca", questions={"q1": question(
            "q1", "choice", criteria={"a": "A", "b": "B", "c": "C"})})
        choice_b = make_row(rid="cb", questions={"q1": question(
            "q1", "choice", criteria={"c": "C", "a": "A", "b": "B"})})
        self.assertEqual(len(audit.references([choice_a, choice_b])), 1,
                         "choice criteria dict order must be normalized")

        score_a = make_row(rid="sa", questions={"q1": question(
            "q1", "score", criteria=["low", "mid", "high"])})
        score_b = make_row(rid="sb", questions={"q1": question(
            "q1", "score", criteria=["high", "mid", "low"])})
        self.assertEqual(len(audit.references([score_a, score_b])), 2,
                         "score criteria list order is ordinal and must be retained")


class CrossSplitIsolationTests(unittest.TestCase):
    def test_identical_inputs_across_splits_detected_despite_new_ids(self):
        r1 = make_row(rid="r1", split="train", state_id="st1", source_group_id="s1")
        r2 = make_row(rid="r2", split="test", state_id="st2", source_group_id="s2")
        report = audit.audit_rows([r1, r2])
        self.assertEqual(report["canonical_input_cross_split_groups"], 1)
        self.assertEqual(report["affected_records"], 2)
        self.assertEqual(report["cross_split_combinations"], {"test|train": 1})
        self.assertIn("canonical_model_input_crosses_splits", report["block_reasons"])

    def test_same_split_duplicates_are_not_called_cross_split(self):
        r1 = make_row(rid="r1", split="train", state_id="st1", source_group_id="s1")
        r2 = make_row(rid="r2", split="train", state_id="st2", source_group_id="s2")
        report = audit.audit_rows([r1, r2])
        self.assertEqual(report["canonical_input_cross_split_groups"], 0)
        self.assertEqual(report["conflicts"], [])
        self.assertNotIn("canonical_model_input_crosses_splits", report["block_reasons"])

    def test_state_identity_crossing_is_detected(self):
        r1 = make_row(rid="r1", split="train", state="evidence one",
                      state_id="shared-state", source_group_id="s1")
        r2 = make_row(rid="r2", split="test", state="evidence two",
                      state_id="shared-state", source_group_id="s2")
        report = audit.audit_rows([r1, r2])
        self.assertEqual(report["declared_identity_conflicts"]["state_id"],
                         {"shared-state": ["test", "train"]})
        self.assertIn("declared_identity_crosses_splits", report["block_reasons"])

    def test_source_group_identity_crossing_is_detected(self):
        r1 = make_row(rid="r1", split="train", state="evidence one",
                      state_id="st1", source_group_id="shared-source")
        r2 = make_row(rid="r2", split="test", state="evidence two",
                      state_id="st2", source_group_id="shared-source")
        report = audit.audit_rows([r1, r2])
        self.assertEqual(report["declared_identity_conflicts"]["source_group_id"],
                         {"shared-source": ["test", "train"]})
        self.assertIn("declared_identity_crosses_splits", report["block_reasons"])


class RegistryAndProvenanceTests(unittest.TestCase):
    def test_missing_source_group_id_is_flagged(self):
        row = make_row(rid="m1", state="evidence m1", source_group_id=None)
        report = audit.audit_rows([row])
        self.assertEqual(report["missing_source_group_ids"], ["m1"])
        self.assertIn("missing_source_group_id", report["block_reasons"])

    def test_duplicate_record_ids_are_flagged(self):
        r1 = make_row(rid="dup", split="train", state="evidence one",
                      source_group_id="s1")
        r2 = make_row(rid="dup", split="train", state="evidence two",
                      source_group_id="s1")
        report = audit.audit_rows([r1, r2])
        self.assertEqual(report["duplicate_record_ids"], ["dup"])
        self.assertIn("duplicate_record_id", report["block_reasons"])

    def test_survey_provenance_aliases_are_recognized(self):
        aliases = [
            "abstention-survey", "abstention_survey", "Abstention Survey",
            "workflow challenge", "workflow-v2_evaluation",
            "context relevance test", "CONTEXT-RELEVANCE-OOD",
        ]
        for alias in aliases:
            self.assertTrue(audit.provenance_is_evaluation_derived(alias), alias)
        self.assertFalse(audit.provenance_is_evaluation_derived("family_fact_rule_17"))

    def test_false_flag_still_flagged_when_source_alias_matches(self):
        row = make_row(rid="pf", provenance_source_id="abstention-survey",
                       declared_derived=False)
        report = audit.audit_rows([row])
        self.assertEqual(len(report["evaluation_provenance_hits"]), 1)
        self.assertIs(report["evaluation_provenance_hits"][0]["declared_derived_flag"], False)
        self.assertIn("evaluation_derived_provenance_requires_review", report["block_reasons"])

    def test_true_flag_is_flagged_even_without_alias(self):
        row = make_row(rid="tf", provenance_source_id="family_fact_rule_17",
                       declared_derived=True)
        report = audit.audit_rows([row])
        self.assertEqual(len(report["evaluation_provenance_hits"]), 1)
        self.assertIs(report["evaluation_provenance_hits"][0]["declared_derived_flag"], True)


class MetricsTests(unittest.TestCase):
    def test_six_question_row_metrics(self):
        row = six_question_row(rid="multi", split="train", source_group_id="src-multi")
        report = audit.audit_rows([row])
        self.assertEqual(report["record_count"], 1)
        self.assertEqual(report["question_count"], 6)
        self.assertEqual(report["counts_by_split"], {"train": 1})
        self.assertEqual(report["declared_source_groups"], 1)

    def test_multiquestion_rows_aggregate_question_count(self):
        first = six_question_row(rid="m1", split="train", source_group_id="src-a")
        second = six_question_row(rid="m2", split="dev", source_group_id="src-b")
        report = audit.audit_rows([first, second])
        self.assertEqual(report["record_count"], 2)
        self.assertEqual(report["question_count"], 12)
        self.assertEqual(report["counts_by_split"], {"dev": 1, "train": 1})


class CrossCohortTests(unittest.TestCase):
    def test_same_inputs_across_cohorts_with_distinct_source_groups(self):
        left = [make_row(rid="l1", split="train", state="evidence shared",
                         source_group_id="orig-src")]
        right = [make_row(rid="r2", split="dev", state="evidence shared",
                          source_group_id="comp-src")]
        comparison = audit.compare_cohorts(left, right)
        self.assertEqual(comparison["common_canonical_inputs"], 1)
        self.assertEqual(comparison["declared_source_id_intersection"], 0)
        self.assertEqual(comparison["left_affected_records"], 1)
        self.assertEqual(comparison["right_affected_records"], 1)

    def test_shared_source_group_without_shared_inputs(self):
        left = [make_row(rid="l1", split="train", state="evidence left",
                         source_group_id="shared-src")]
        right = [make_row(rid="r2", split="dev", state="evidence right",
                          source_group_id="shared-src")]
        comparison = audit.compare_cohorts(left, right)
        self.assertEqual(comparison["common_canonical_inputs"], 0)
        self.assertEqual(comparison["declared_source_id_intersection"], 1)


class CrossSeedRegressionTests(unittest.TestCase):
    def test_seeded_id_only_isolation_is_not_sufficient(self):
        """Different declared IDs/sources/seeds can still share canonical inputs."""
        original = [make_row(rid="seed1-row", split="train",
                             source_group_id="seed1-src")]
        comparison = [make_row(rid="seed2-row", split="dev",
                               source_group_id="seed2-src")]
        overlap = audit.compare_cohorts(original, comparison)
        self.assertEqual(overlap["common_canonical_inputs"], 1)
        self.assertEqual(overlap["declared_source_id_intersection"], 0,
                         "ID/source isolation alone does not prove semantic holdout")

    def test_audit_corpus_cross_seed_overlap_appends_blocker(self):
        rows = [make_row(rid="orig-1", split="train", source_group_id="orig-src")]
        other = [make_row(rid="other-1", split="dev", source_group_id="other-src")]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            patches = [
                mock.patch.object(audit, "check", return_value=[]),
                mock.patch.object(audit, "read_json",
                                  return_value={"seed": 1, "pair_count": 0, "items": []}),
                mock.patch.object(audit, "read_training_records",
                                  return_value=(rows, [])),
                mock.patch.object(audit, "file_hash", return_value="deadbeef"),
                mock.patch.object(audit, "derive_manifest", return_value={"seed": 2}),
                mock.patch.object(audit, "trainer_rows", return_value=other),
            ]
            for patcher in patches:
                patcher.start()
                self.addCleanup(patcher.stop)
            report = audit.audit_corpus(root, compare_seed=2)
        self.assertFalse(report["cross_seed"]["materialized"])
        self.assertEqual(report["cross_seed"]["common_canonical_inputs"], 1)
        self.assertEqual(report["cross_seed"]["declared_source_id_intersection"], 0)
        self.assertIn("seed_change_does_not_prove_semantic_holdout", report["block_reasons"])
        self.assertEqual(report["status"], "blocked")
        self.assertFalse(report["training_authorized"])
        self.assertFalse(report["training_performed"])


class FrozenCorpusRegressionTests(unittest.TestCase):
    def test_frozen_v1_is_schema_valid_but_not_isolated_and_stays_unchanged(self):
        root = Path(__file__).resolve().parent.parent / "research/engineering_judgment_corpus_v1"
        if not (root / "manifest.json").is_file():
            self.skipTest("Frozen engineering V1 corpus is not present")
        paths = [root / "manifest.json"] + [root / view / f"{split}.jsonl"
                 for view in ("items", "trainer_view") for split in audit.SPLITS]
        before = {str(path.relative_to(root)): audit.file_hash(path) for path in paths}
        report = audit.audit_corpus(root, compare_seed=20260920)
        self.assertEqual(report["builder_integrity_errors"], [])
        self.assertEqual(report["trainer_schema_and_declared_split_checks"], "passed")
        self.assertEqual(report["audit"]["record_count"], 246)
        self.assertEqual(report["audit"]["canonical_input_cross_split_groups"], 27)
        self.assertEqual(report["audit"]["affected_records"], 63)
        self.assertEqual(len(report["audit"]["evaluation_provenance_hits"]), 18)
        self.assertEqual(report["cross_seed"]["common_canonical_inputs"], 129)
        self.assertEqual(report["cross_seed"]["declared_source_id_intersection"], 0)
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["merged_rows_written"], 0)
        self.assertFalse(report["training_performed"])
        self.assertEqual(before, report["source_hashes"])
        self.assertEqual(before, {str(path.relative_to(root)): audit.file_hash(path) for path in paths})


class CLITests(unittest.TestCase):
    def run_main(self, argv):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            code = audit.main(argv)
        return code, stream.getvalue()

    def test_blocked_report_exits_two(self):
        report = {"schema_version": audit.SCHEMA, "status": "blocked",
                  "block_reasons": ["canonical_model_input_crosses_splits"]}
        with mock.patch.object(audit, "audit_corpus", return_value=report):
            code, out = self.run_main(["--corpus", "/tmp/fake-corpus"])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(out), report)

    def test_clean_report_exits_zero(self):
        report = {"schema_version": audit.SCHEMA,
                  "status": "preflight_passed_not_training_authorized",
                  "block_reasons": []}
        with mock.patch.object(audit, "audit_corpus", return_value=report):
            code, out = self.run_main(["--corpus", "/tmp/fake-corpus"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out), report)

    def test_output_is_exclusive_and_never_overwritten(self):
        report = {"schema_version": audit.SCHEMA,
                  "status": "preflight_passed_not_training_authorized",
                  "block_reasons": []}
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "corpus"
            corpus.mkdir()
            output = Path(tmp) / "receipt.json"
            with mock.patch.object(audit, "audit_corpus", return_value=report):
                code, out = self.run_main(["--corpus", str(corpus),
                                           "--output", str(output)])
            self.assertEqual(code, 0)
            written = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(written, report)
            self.assertIn("output", json.loads(out))

            original = output.read_text(encoding="utf-8")
            with mock.patch.object(audit, "audit_corpus", return_value=report):
                code, out = self.run_main(["--corpus", str(corpus),
                                           "--output", str(output)])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(out)["status"], "error")
            self.assertEqual(output.read_text(encoding="utf-8"), original,
                             "an existing audit receipt must not be overwritten")

    def test_output_under_corpus_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "corpus"
            corpus.mkdir()
            output = corpus / "receipt.json"
            with mock.patch.object(audit, "audit_corpus") as mocked:
                with self.assertRaises(SystemExit) as caught:
                    with contextlib.redirect_stderr(io.StringIO()):
                        audit.main(["--corpus", str(corpus), "--output", str(output)])
            self.assertEqual(caught.exception.code, 2)
            mocked.assert_not_called()


if __name__ == "__main__":
    unittest.main()
