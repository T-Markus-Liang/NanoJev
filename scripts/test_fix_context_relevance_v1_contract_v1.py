import json
from pathlib import Path
import tempfile
import unittest

from fix_context_relevance_v1_contract_v1 import (
    AFFECTED_KINDS, CURRENT_VALUE_IN_STATE, SPLITS, contract_label, fix_row, build)
from validate_valen_label_consistency_v1 import check_dir, normalized_state


def row(family="risk", kind="correction", irrelevant=False):
    return {"id": "context:train:0", "state_id": "context:train:0",
            "family_id": "context_relevance_v1", "split": "train",
            "state": {"conversation": [
                {"pointer": "user", "role": "user",
                 "content": "Can account risk-0002 open a position of 42 units under its current limit?"},
                {"pointer": "candidate", "role": "assistant",
                 "content": "A previous risk snapshot for risk-0002 had a limit of 47 units."}],
                "candidate_pointer": "candidate", "scenario_id": "risk-0002"},
            "questions": {"irrelevant": {"type": "boolean", "instructions": "..."}},
            "gold": {"irrelevant": irrelevant},
            "gold_probs": {"irrelevant": {"false": float(not irrelevant), "true": float(irrelevant)}},
            "gold_probs_kind": {"irrelevant": "deterministic_truth"},
            "gold_label_kind": {"irrelevant": "deterministic_truth"},
            "metadata": {"source_group_id": "g0", "scenario_family": family,
                         "candidate_kind": kind, "source": "self_authored_context_relevance",
                         "license": "CC0-1.0"}}


def valen_record(state, p_true, kind="correction", record_id="r0"):
    return {"group_id": "g",
            "request": {"state": json.dumps(state, ensure_ascii=False),
                        "questions": {"irrelevant": {"type": "noul", "instructions": "..."}}},
            "targets": {"irrelevant": {"probabilities": {"true": p_true, "false": 1.0 - p_true}}},
            "meta": {"record_id": record_id, "source_dataset": "context_relevance_v1",
                     "candidate_kind": kind}}


class ContractLabelTest(unittest.TestCase):
    def test_contract_matches_stale_fact_rule(self):
        # current value present in the user request -> drop (true)
        self.assertTrue(contract_label("code", "correction"))
        self.assertTrue(contract_label("order", "overlap_distractor"))
        # candidate is the sole evidence -> keep (false)
        for family in ("risk", "support", "robotics", "multilingual"):
            for kind in AFFECTED_KINDS:
                self.assertFalse(contract_label(family, kind), (family, kind))
        # unaffected kinds are untouched
        self.assertIsNone(contract_label("risk", "required_evidence"))
        with self.assertRaises(ValueError):
            contract_label("unknown_family", "correction")

    def test_fix_row_flips_and_annotates(self):
        corrected = row("code", "correction", irrelevant=False)
        _, changed = fix_row(corrected)
        self.assertTrue(changed)
        self.assertTrue(corrected["gold"]["irrelevant"])
        self.assertEqual(corrected["gold_probs"]["irrelevant"], {"true": 1.0, "false": 0.0})
        self.assertEqual(corrected["metadata"]["contract_fix"]["previous_irrelevant"], False)
        # already-correct labels pass through without annotation
        stable = row("risk", "overlap_distractor", irrelevant=True)
        _, changed = fix_row(stable)
        self.assertTrue(changed)  # mislabelled drop on sole-evidence family flips to keep
        self.assertFalse(stable["gold"]["irrelevant"])
        ok = row("risk", "correction", irrelevant=False)
        _, changed = fix_row(ok)
        self.assertFalse(changed)
        self.assertNotIn("contract_fix", ok["metadata"])

    def test_build_relabels_and_writes_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "src", Path(tmp) / "dst"
            src.mkdir()
            for split in SPLITS:
                rows = [row("code", "correction"), row("risk", "overlap_distractor", True),
                        row("code", "unrelated", True)]
                (src / f"{split}.jsonl").write_text(
                    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
            manifest = build(src, dst)
            self.assertEqual(manifest["schema_version"], "nanojev-context-relevance-v2")
            self.assertIn("contract_fix", manifest)
            out = [json.loads(line) for line in (dst / "train.jsonl").read_text().splitlines()]
            self.assertEqual([r["gold"]["irrelevant"] for r in out], [True, False, True])
            self.assertEqual(manifest["splits"]["train"]["relabelled"],
                             {"code:correction": 1, "risk:overlap_distractor": 1})


class LabelConsistencyCheckTest(unittest.TestCase):
    def state(self, entity="risk-0002", stale_limit="47"):
        return {"conversation": [
            {"pointer": "user", "role": "user",
             "content": f"Can account {entity} open a position of 42 units under its current limit?"},
            {"pointer": "candidate", "role": "assistant",
             "content": f"A previous risk snapshot for {entity} had a limit of {stale_limit} units."}],
            "candidate_pointer": "candidate", "scenario_id": entity}

    def test_normalized_state_ignores_ids_but_keeps_pointers(self):
        a, b = self.state("risk-0002"), self.state("risk-9999", "61")
        self.assertEqual(normalized_state(json.dumps(a)), normalized_state(json.dumps(b)))
        b["candidate_pointer"] = "other"
        self.assertNotEqual(normalized_state(json.dumps(a)), normalized_state(json.dumps(b)))

    def test_contradictory_labels_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = [valen_record(self.state("risk-0002"), 0.0, "correction", "a"),
                       valen_record(self.state("risk-9999"), 1.0, "overlap_distractor", "b")]
            (Path(tmp) / "train.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
            failures, report = check_dir(tmp)
            self.assertEqual(report["violations"], 2)
            self.assertTrue(failures)

    def test_consistent_labels_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = [valen_record(self.state("risk-0002"), 0.0, "correction", "a"),
                       valen_record(self.state("risk-9999"), 0.0, "overlap_distractor", "b")]
            (Path(tmp) / "train.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
            failures, report = check_dir(tmp)
            self.assertFalse(failures)
            self.assertEqual(report["violations"], 0)


if __name__ == "__main__":
    unittest.main()
