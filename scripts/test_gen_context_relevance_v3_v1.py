"""Unit tests for scripts/gen_context_relevance_v3_v1.py and the v3 merge build.

Covers: byte determinism, seed sensitivity, source schema validity, the
v2 label contract (stale candidate drops only when the current value is in the
state), global normalized-state label consistency, strict group isolation on the
valen build, the hard_negative flag and eval coverage, and zh/en mixing in the
multilingual family.

Run: python3 -m unittest scripts/test_gen_context_relevance_v3_v1.py -v
     (or python3 scripts/test_gen_context_relevance_v3_v1.py)
"""

import hashlib
import json
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "external" / "valen" / "scripts"))

from gen_context_relevance_v3_v1 import (  # noqa: E402
    FAMILIES, HARD_KINDS, KINDS, generate)
from validate_valen_label_consistency_v1 import normalized_state  # noqa: E402
import build_nanojev_valen_v3  # noqa: E402

SEED = 20261005
GROUPS = 6  # small cohort for tests


def load_source(out):
    rows = []
    for split in ("train", "dev", "calibration"):
        for line in (Path(out) / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


class DeterminismTest(unittest.TestCase):
    def test_same_seed_byte_identical(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            generate(seed=SEED, groups_per_family=GROUPS, out=a)
            generate(seed=SEED, groups_per_family=GROUPS, out=b)
            for name in ("train", "dev", "calibration"):
                self.assertEqual((Path(a) / f"{name}.jsonl").read_bytes(),
                                 (Path(b) / f"{name}.jsonl").read_bytes())

    def test_different_seed_changes_output(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            generate(seed=SEED, groups_per_family=GROUPS, out=a)
            generate(seed=SEED + 1, groups_per_family=GROUPS, out=b)
            self.assertNotEqual((Path(a) / "train.jsonl").read_bytes(),
                                (Path(b) / "train.jsonl").read_bytes())


class SchemaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls._tmp.name)
        generate(seed=SEED, groups_per_family=GROUPS, out=cls.out)
        cls.rows = load_source(cls.out)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_record_schema(self):
        required = {"id", "state_id", "family_id", "split", "state", "questions",
                    "gold", "gold_probs", "gold_probs_kind", "gold_label_kind", "metadata"}
        for row in self.rows:
            self.assertTrue(required <= set(row), row["id"])
            self.assertEqual(row["family_id"], "context_relevance_v3")
            state = row["state"]
            pointers = [m["pointer"] for m in state["conversation"]]
            self.assertIn(state["candidate_pointer"], pointers)
            probs = row["gold_probs"]["irrelevant"]
            self.assertAlmostEqual(probs["true"] + probs["false"], 1.0)
            self.assertEqual(bool(probs["true"] > probs["false"]),
                             row["gold"]["irrelevant"])
            meta = row["metadata"]
            for key in ("source_group_id", "scenario_family", "candidate_kind",
                        "variant", "hard_negative", "source", "license"):
                self.assertIn(key, meta)
            self.assertIn(meta["candidate_kind"], KINDS)
            self.assertIn(meta["scenario_family"], FAMILIES)

    def test_full_kind_and_family_coverage(self):
        kinds = Counter(r["metadata"]["candidate_kind"] for r in self.rows)
        families = Counter(r["metadata"]["scenario_family"] for r in self.rows)
        self.assertEqual(set(kinds), set(KINDS))
        self.assertEqual(set(families), set(FAMILIES))
        self.assertIn("near_duplicate_evidence", kinds)
        self.assertIn("correction_confirmed", kinds)
        for kind in KINDS:
            self.assertEqual(kinds[kind], GROUPS * len(FAMILIES))

    def test_valen_record_shape_loads(self):
        """Each generated source record converts through the valen path."""
        for row in self.rows:
            valen, side = build_nanojev_valen_v3.to_valen_v3(
                row, "context_relevance_v3", "context_relevance_v3", row["split"])
            self.assertIn(side, ("train", "eval"))
            req = valen["request"]
            self.assertEqual(set(req["questions"]), {"irrelevant"})
            self.assertEqual(req["questions"]["irrelevant"]["type"], "noul")
            json.loads(req["state"])  # state is a JSON string
            probs = valen["targets"]["irrelevant"]["probabilities"]
            self.assertAlmostEqual(probs["true"] + probs["false"], 1.0)
            meta = valen["meta"]
            for key in ("record_id", "domain", "modality", "language_bucket",
                        "source_dataset", "source_split", "candidate_kind",
                        "candidate_pointer", "hard_negative"):
                self.assertIn(key, meta)


class ContractTest(unittest.TestCase):
    """The label must be a pure function of state content under the v2 contract:
    drop(true) for a stale candidate only when the current value is in state."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls._tmp.name)
        generate(seed=SEED, groups_per_family=GROUPS, out=cls.out)
        cls.rows = load_source(cls.out)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    # Marker proving the user request states the current value, per family.
    VALUE_MARKERS = {
        "code": lambda t: "use port " in t,
        "order": lambda t: "units at " in t,
        "risk": lambda t: "current limit of " in t,
        "support": lambda t: "refund deadline is " in t,
        "robotics": lambda t: "speed limit of " in t,
        # en plain request says "user's last confirmed time" — marker must use
        # the withval phrasing "user confirmed time" / zh "已确认时间为".
        "multilingual": lambda t: "已确认时间为" in t or "user confirmed time" in t,
    }

    def _user(self, row):
        return next(m for m in row["state"]["conversation"]
                    if m["pointer"] == "user")["content"]

    def test_variant_label_rule(self):
        for row in self.rows:
            meta = row["metadata"]
            marker = self.VALUE_MARKERS[meta["scenario_family"]]
            if meta["variant"] == "current_in_state":
                self.assertTrue(row["gold"]["irrelevant"], row["id"])
                self.assertTrue(marker(self._user(row)),
                                f"stale candidate dropped without a stated "
                                f"current value: {row['id']}")
            elif meta["variant"] == "sole_evidence":
                self.assertFalse(row["gold"]["irrelevant"], row["id"])
                self.assertFalse(marker(self._user(row)),
                                 f"sole-evidence record but request already "
                                 f"states the current value: {row['id']}")

    def test_fixed_drop_kinds_state_current_value(self):
        """superseded / near_duplicate_evidence are always drop, so their
        request must always carry the current value (contract precondition)."""
        drops = [r for r in self.rows
                 if r["metadata"]["candidate_kind"]
                 in ("superseded", "near_duplicate_evidence")]
        self.assertTrue(drops)
        for row in drops:
            self.assertTrue(row["gold"]["irrelevant"], row["id"])
            marker = self.VALUE_MARKERS[row["metadata"]["scenario_family"]]
            self.assertTrue(marker(self._user(row)),
                            f"drop label without current value in state: {row['id']}")

    def test_global_label_consistency(self):
        """Stricter than the official validator: one label per normalized state
        across ALL splits (the official check is per-split)."""
        labels = defaultdict(set)
        for row in self.rows:
            state_text = json.dumps(row["state"], ensure_ascii=False,
                                    separators=(",", ":"))
            labels[normalized_state(state_text)].add(
                str(row["gold"]["irrelevant"]).lower())
        bad = {k: v for k, v in labels.items() if len(v) > 1}
        self.assertEqual(bad, {})

    def test_no_duplicate_record_ids(self):
        ids = [r["id"] for r in self.rows]
        self.assertEqual(len(ids), len(set(ids)))


class BuildMergeTest(unittest.TestCase):
    """Group isolation and hard-negative coverage on the merged valen build."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        cls.v3_src = tmp / "context_relevance_v3"
        generate(seed=SEED, groups_per_family=GROUPS, out=cls.v3_src)
        # Synthetic repo root: real v2 sources symlinked, generated v3 linked.
        cls.repo = tmp / "repo"
        data = cls.repo / "data"
        data.mkdir(parents=True)
        for name in ("context_relevance_v2_seed20260919",
                     "context_relevance_oracle_v1_seed20260919"):
            (data / name).symlink_to(ROOT / "data" / name, target_is_directory=True)
        (data / "context_relevance_v3").symlink_to(cls.v3_src.resolve(),
                                                 target_is_directory=True)
        cls.out = tmp / "valen_nano_v3"
        cls.manifest = build_nanojev_valen_v3.build(repo_root=cls.repo, out=cls.out)
        cls.train = [json.loads(l) for l in
                     (cls.out / "train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        cls.eval = [json.loads(l) for l in
                    (cls.out / "eval.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_group_isolation_strict(self):
        train_groups = {r["group_id"] for r in self.train}
        eval_groups = {r["group_id"] for r in self.eval}
        self.assertEqual(train_groups & eval_groups, set())

    def test_v3_group_namespace(self):
        v3 = [r for r in self.train + self.eval
              if r["meta"]["source_dataset"] == "context_relevance_v3"]
        self.assertTrue(v3)
        for row in v3:
            self.assertTrue(row["group_id"].startswith("context_relevance_v3:"))

    def test_split_rule(self):
        eval_keys = {(r["group_id"], r["meta"]["record_id"]) for r in self.eval}
        for row in self.train + self.eval:
            side = ("eval" if int(hashlib.sha256(row["group_id"].encode()).hexdigest(), 16)
                    % 100 < 20 else "train")
            actual = ("eval" if (row["group_id"], row["meta"]["record_id"]) in eval_keys
                      else "train")
            self.assertEqual(side, actual)

    def test_hard_negative_flag(self):
        v3 = [r for r in self.train + self.eval
              if r["meta"]["source_dataset"] == "context_relevance_v3"]
        for row in v3:
            self.assertIn("hard_negative", row["meta"])
            kind = row["meta"]["candidate_kind"]
            self.assertEqual(row["meta"]["hard_negative"], kind in HARD_KINDS)

    def test_eval_hard_negative_floor(self):
        """v3 eval records are ~50% hard-negative by construction (8/16 kinds);
        assert >=30% so the merged eval stays discriminative. The full-size
        build measured 38.5% hard on the merged eval set."""
        v3_eval = [r for r in self.eval
                   if r["meta"]["source_dataset"] == "context_relevance_v3"]
        self.assertTrue(v3_eval)
        frac = sum(1 for r in v3_eval if r["meta"]["hard_negative"]) / len(v3_eval)
        self.assertGreaterEqual(frac, 0.30)

    def test_manifest_counts(self):
        self.assertEqual(self.manifest["records"]["total"],
                         len(self.train) + len(self.eval))
        self.assertEqual(self.manifest["schema_version"], "nanojev-valen-nano-v3")


class MultilingualMixTest(unittest.TestCase):
    def test_zh_en_mixed_records_exist(self):
        with tempfile.TemporaryDirectory() as d:
            generate(seed=SEED, groups_per_family=GROUPS, out=d)
            rows = [r for r in load_source(d)
                    if r["metadata"]["scenario_family"] == "multilingual"]
            mixed = [r for r in rows if r["metadata"].get("mixed_language")]
            self.assertTrue(len(mixed) >= GROUPS,
                            "expected zh/en mixed multilingual records")
            en_requests = [r for r in rows
                           if not any("一" <= c <= "鿿"
                                      for c in r["state"]["conversation"][1]["content"])]
            self.assertTrue(en_requests, "expected some English requests")


if __name__ == "__main__":
    unittest.main()
