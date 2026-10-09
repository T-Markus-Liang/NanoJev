"""Unit tests for scripts/gen_context_relevance_v4_v1.py.

Covers: byte determinism, seed sensitivity, valen eval.jsonl schema (state is a
JSON string, candidate_pointer resolvable, noul instructions verbatim from the
v3 eval, probabilities sum to 1 and argmax matches the implied label), the
constant-label-per-kind contract, per-family semantic rules (pending tool
result kept / resolved-pair stale dropped, sole-evidence keep vs restated
drop), global normalized-state label consistency (validator approach extended
to dev AND train), fresh group_id lineage disjoint from v3, the new train
split (group-level train/eval/dev disjointness), and the consolidated
valen_nano_v4 merge (v3 rows byte-preserved, merged counts, 0 violations).

Run: python3 -m unittest scripts/test_gen_context_relevance_v4_v1.py -v
     (or python3 scripts/test_gen_context_relevance_v4_v1.py)
"""

import json
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from gen_context_relevance_v4_v1 import (  # noqa: E402
    FAMILIES, INSTRUCTIONS, KIND_LABEL, LINEAGE, SOURCE_DATASET,
    TRAIN_GIDX_OFFSET, build_consolidated, generate)
from validate_valen_label_consistency_v1 import (  # noqa: E402
    argmax_label, check_dir, normalized_state)

SEED = 20261115
SCALE = 0.1  # small cohort for tests
V3 = ROOT / "data" / "valen_nano_v3"


def load(out):
    rows = {"train": [], "eval": [], "dev": []}
    for split in rows:
        path = Path(out) / f"{split}.jsonl"
        if path.exists():
            rows[split] = [json.loads(l) for l in
                           path.read_text(encoding="utf-8").splitlines()
                           if l.strip()]
    return rows


class DeterminismTest(unittest.TestCase):
    def test_same_seed_byte_identical(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            generate(seed=SEED, scale=SCALE, out=a)
            generate(seed=SEED, scale=SCALE, out=b)
            for name in ("train", "eval", "dev"):
                self.assertEqual((Path(a) / f"{name}.jsonl").read_bytes(),
                                 (Path(b) / f"{name}.jsonl").read_bytes())

    def test_different_seed_changes_output(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            generate(seed=SEED, scale=SCALE, out=a)
            generate(seed=SEED + 1, scale=SCALE, out=b)
            self.assertNotEqual((Path(a) / "eval.jsonl").read_bytes(),
                                (Path(b) / "eval.jsonl").read_bytes())

    def test_label_determinism(self):
        """Record ids are stable and their labels reproduce run to run."""
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            generate(seed=SEED, scale=SCALE, out=a)
            generate(seed=SEED, scale=SCALE, out=b)
            la = {r["meta"]["record_id"]: argmax_label(r)
                  for s in ("train", "eval", "dev") for r in load(a)[s]}
            lb = {r["meta"]["record_id"]: argmax_label(r)
                  for s in ("train", "eval", "dev") for r in load(b)[s]}
            self.assertEqual(la, lb)


class SchemaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls._tmp.name)
        cls.manifest = generate(seed=SEED, scale=SCALE, out=cls.out)
        cls.rows = load(cls.out)
        cls.all_rows = cls.rows["train"] + cls.rows["eval"] + cls.rows["dev"]

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_valen_record_schema(self):
        required_meta = {"record_id", "domain", "modality", "language_bucket",
                         "source_dataset", "source_split", "candidate_kind",
                         "candidate_pointer", "hard_negative", "eval_family"}
        for row in self.all_rows:
            self.assertTrue({"group_id", "request", "targets", "meta"}
                            <= set(row), row["meta"].get("record_id"))
            self.assertTrue(row["group_id"].startswith(f"{LINEAGE}:"))
            q = row["request"]["questions"]["irrelevant"]
            self.assertEqual(q["type"], "noul")
            self.assertEqual(q["instructions"], INSTRUCTIONS)
            state = json.loads(row["request"]["state"])  # must be a JSON string
            pointers = [m["pointer"] for m in state["conversation"]]
            self.assertIn(state["candidate_pointer"], pointers)
            probs = row["targets"]["irrelevant"]["probabilities"]
            self.assertAlmostEqual(probs["true"] + probs["false"], 1.0)
            meta = row["meta"]
            self.assertTrue(required_meta <= set(meta), meta["record_id"])
            self.assertEqual(meta["source_dataset"], SOURCE_DATASET)
            self.assertIn(meta["eval_family"], FAMILIES)
            self.assertIn(meta["candidate_kind"], KIND_LABEL)
            self.assertEqual(meta["candidate_pointer"], state["candidate_pointer"])

    def test_train_split_exists_and_balanced(self):
        """v4 is consolidated: train rows exist for all five new families,
        roughly balanced across families and candidate kinds."""
        train = self.rows["train"]
        self.assertTrue(train)
        fams = Counter(r["meta"]["eval_family"] for r in train)
        self.assertEqual(set(fams), set(FAMILIES))
        lo, hi = min(fams.values()), max(fams.values())
        self.assertLessEqual(hi - lo, max(2, int(0.1 * hi)),
                             f"unbalanced train families: {fams}")
        # every kind appears in train
        self.assertEqual({r["meta"]["candidate_kind"] for r in train},
                         set(KIND_LABEL))
        for row in train:
            self.assertEqual(row["meta"]["source_split"], "train")

    def test_eval_only_flags_on_splits(self):
        for split, rows in self.rows.items():
            for row in rows:
                self.assertEqual(row["meta"]["source_split"], split)

    def test_family_and_kind_coverage(self):
        fams = Counter(r["meta"]["eval_family"] for r in self.all_rows)
        kinds = Counter(r["meta"]["candidate_kind"] for r in self.all_rows)
        self.assertEqual(set(fams), set(FAMILIES))
        self.assertEqual(set(kinds), set(KIND_LABEL))

    def test_manifest_declares_eval_only(self):
        self.assertEqual(self.manifest["training_allowed"],
                         {"train": True, "eval": False, "dev": False})
        self.assertEqual(sorted(self.manifest["evaluation_only"]),
                         ["dev", "eval"])
        self.assertEqual(self.manifest["label_consistency"]["violations"], 0)
        self.assertTrue(all(self.manifest["group_disjointness"].values()))


class ContractTest(unittest.TestCase):
    """Labels are a pure function of state content."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls._tmp.name)
        generate(seed=SEED, scale=SCALE, out=cls.out)
        rows = load(cls.out)
        cls.all_rows = rows["train"] + rows["eval"] + rows["dev"]

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _state(self, row):
        return json.loads(row["request"]["state"])

    def test_constant_label_per_kind(self):
        by_kind = defaultdict(set)
        for row in self.all_rows:
            by_kind[row["meta"]["candidate_kind"]].add(argmax_label(row))
        for kind, labels in by_kind.items():
            self.assertEqual(len(labels), 1, kind)
            self.assertEqual(labels.pop(),
                             "true" if KIND_LABEL[kind] else "false", kind)

    def test_tool_pairing_semantics(self):
        """Stale result drops only with a resolved successor; pending keeps."""
        for row in self.all_rows:
            kind = row["meta"]["candidate_kind"]
            if not kind.startswith("tool_"):
                continue
            conv = self._state(row)["conversation"]
            resolved = any('"status":"resolved"' in m["content"]
                           and '"re-verified"' in m["content"]
                           for m in conv)
            if kind == "tool_stale_resolved":
                self.assertTrue(resolved)
                self.assertEqual(argmax_label(row), "true")
            elif kind == "tool_stale_pending":
                self.assertFalse(resolved)
                self.assertEqual(argmax_label(row), "false")

    def test_cross_pointer_semantics(self):
        """Sole-evidence pointer keeps; restated/corrected pointer drops."""
        for row in self.all_rows:
            kind = row["meta"]["candidate_kind"]
            if not kind.startswith("xptr_"):
                continue
            conv = self._state(row)["conversation"]
            has_confirm = any("Confirmed record" in m["content"] for m in conv)
            has_corr = any(m["content"].startswith("Correction:") for m in conv)
            if kind == "xptr_sole_evidence":
                self.assertFalse(has_confirm or has_corr)
                self.assertEqual(argmax_label(row), "false")
            elif kind in ("xptr_restated_stale", "xptr_corrected_stale"):
                self.assertTrue(has_confirm or has_corr)
                self.assertEqual(argmax_label(row), "true")

    def test_no_normalized_label_collision(self):
        """Stricter than the official check: eval + dev pooled."""
        labels = defaultdict(set)
        for row in self.all_rows:
            labels[normalized_state(row["request"]["state"])].add(argmax_label(row))
        bad = {k: v for k, v in labels.items() if len(v) > 1}
        self.assertEqual(bad, {})

    def test_official_validator_zero_violations(self):
        failures, report = check_dir(self.out)
        self.assertEqual(failures, [])
        self.assertEqual(report["violations"], 0)

    def test_no_duplicate_record_ids(self):
        ids = [r["meta"]["record_id"] for r in self.all_rows]
        self.assertEqual(len(ids), len(set(ids)))

    def test_split_group_disjointness(self):
        """train/eval/dev share no group_id (the core honesty rule: v4-new
        eval rows must never share a group instance with v4-new train rows)."""
        rows = load(self.out)
        gsets = {s: {r["group_id"] for r in rows[s]} for s in rows}
        self.assertTrue(gsets["train"])
        self.assertTrue(gsets["eval"])
        for a in gsets:
            for b in gsets:
                if a < b:
                    self.assertTrue(gsets[a].isdisjoint(gsets[b]),
                                    f"{a} overlaps {b}")


class ConsolidatedBuildTest(unittest.TestCase):
    """Merged data/valen_nano_v4: frozen v3 rows + v4 new-family rows."""

    @classmethod
    def setUpClass(cls):
        if not V3.exists():
            raise unittest.SkipTest("data/valen_nano_v3 not present")
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        cls.raw = tmp / "raw"
        generate(seed=SEED, scale=SCALE, out=cls.raw)
        cls.out = tmp / "valen_nano_v4"
        cls.manifest = build_consolidated(cls.raw, v3_dir=V3, out=cls.out)
        cls.merged = load(cls.out)
        cls.v3_train_lines = {l for l in (V3 / "train.jsonl")
                              .read_text(encoding="utf-8").splitlines()
                              if l.strip()}
        cls.v3_eval_lines = {l for l in (V3 / "eval.jsonl")
                             .read_text(encoding="utf-8").splitlines()
                             if l.strip()}
        cls.raw_rows = load(cls.raw)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_merged_counts(self):
        self.assertEqual(len(self.merged["train"]),
                         len(self.v3_train_lines) + len(self.raw_rows["train"]))
        self.assertEqual(len(self.merged["eval"]),
                         len(self.v3_eval_lines) + len(self.raw_rows["eval"]))
        self.assertEqual(len(self.merged["dev"]), len(self.raw_rows["dev"]))

    def test_v3_rows_byte_preserved(self):
        """Every frozen v3 line appears verbatim in the merged output."""
        merged_train = set((self.out / "train.jsonl")
                           .read_text(encoding="utf-8").splitlines())
        merged_eval = set((self.out / "eval.jsonl")
                          .read_text(encoding="utf-8").splitlines())
        self.assertTrue(self.v3_train_lines <= merged_train)
        self.assertTrue(self.v3_eval_lines <= merged_eval)

    def test_merged_group_disjointness(self):
        v4 = {s: {r["group_id"] for r in self.raw_rows[s]}
              for s in self.raw_rows}
        self.assertTrue(v4["train"].isdisjoint(v4["eval"]))
        self.assertTrue(v4["train"].isdisjoint(v4["dev"]))
        self.assertTrue(v4["eval"].isdisjoint(v4["dev"]))
        v3_groups = {r["group_id"]
                     for s in ("train", "eval") for r in self.merged[s]
                     if r["meta"]["source_dataset"] != SOURCE_DATASET}
        self.assertTrue(
            set().union(*v4.values()).isdisjoint(v3_groups))
        self.assertTrue(all(self.manifest["group_disjointness"][k]
                            for k in ("v4_train_vs_eval", "v4_train_vs_dev",
                                      "v4_eval_vs_dev", "v4_all_vs_v3_all")))

    def test_merged_validator_zero_violations(self):
        failures, report = check_dir(self.out)
        self.assertEqual(failures, [])
        self.assertEqual(report["violations"], 0)

    def test_merged_training_allowed_flags(self):
        self.assertTrue(self.manifest["training_allowed"]["train"])
        self.assertFalse(self.manifest["training_allowed"]["eval"])
        self.assertFalse(self.manifest["training_allowed"]["dev"])


class NamespaceTest(unittest.TestCase):
    def test_group_ids_disjoint_from_v3(self):
        v3_eval = ROOT / "data" / "valen_nano_v3" / "eval.jsonl"
        if not v3_eval.exists():
            self.skipTest("valen_nano_v3 not present")
        v3_groups = {json.loads(l)["group_id"]
                     for l in v3_eval.read_text(encoding="utf-8").splitlines()
                     if l.strip()}
        with tempfile.TemporaryDirectory() as d:
            generate(seed=SEED, scale=SCALE, out=d)
            rows = load(d)
            groups = {r["group_id"] for s in rows for r in rows[s]}
        self.assertTrue(groups)
        self.assertEqual(groups & v3_groups, set())
        for g in groups:
            self.assertTrue(g.startswith(f"{LINEAGE}:"))


class LongContextTest(unittest.TestCase):
    def test_segment_band_and_dilution(self):
        with tempfile.TemporaryDirectory() as d:
            generate(seed=SEED, scale=SCALE, out=d)
            loaded = load(d)
            rows = [r for s in ("train", "eval", "dev") for r in loaded[s]
                    if r["meta"]["eval_family"] == "long_context_dilution"]
            self.assertTrue(rows)
            for row in rows:
                n = len(json.loads(row["request"]["state"])["conversation"])
                self.assertTrue(30 <= n <= 120, n)
                self.assertEqual(row["meta"].get("n_segments"), n)


if __name__ == "__main__":
    unittest.main()
