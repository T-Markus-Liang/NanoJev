#!/usr/bin/env python3
"""X3 shared-prefix path tests: plan/mask invariants (pure python), chunking at
255 candidates, flag on/off contract, and real-checkpoint numerical parity."""
import unittest
from pathlib import Path

from predict_toy_decisions import (chunk_packed_rows, common_prefix_length,
                                   shared_prefix_row_plans)


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoints" / "domain_adaptation_v4_lora_seed18"
PARITY_TOL = 1e-5  # pre-registered: max |Δp| allowed between the two paths


def visible_tokens(row, last_idx):
    """Tokens a query at last_idx may attend to, per the packed mask rule."""
    seg = row["segment_ids"]
    return [tok for u, tok in enumerate(row["tokens"])
            if u <= last_idx and (seg[u] == 0 or seg[u] == seg[last_idx])]


def visible_positions(row, last_idx):
    seg = row["segment_ids"]
    return [pos for u, pos in enumerate(row["positions"])
            if u <= last_idx and (seg[u] == 0 or seg[u] == seg[last_idx])]


def fake_example(leaves, anchor=None, k=None, typ="choice"):
    ex = {"id": "t:q", "state_id": "t", "qid": "q", "type": typ,
          "candidate_ids": [str(i) for i in range(k if k is not None else len(leaves))],
          "leaf_tokens": leaves}
    if anchor:
        ex["anchor_tokens"] = anchor
    return ex


class CommonPrefixTest(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(common_prefix_length([[1, 2, 3], [1, 2, 4], [1, 2]]), 2)

    def test_single_sequence(self):
        self.assertEqual(common_prefix_length([[5, 6, 7]]), 3)

    def test_no_common(self):
        self.assertEqual(common_prefix_length([[1, 2], [3, 4]]), 0)

    def test_empty(self):
        self.assertEqual(common_prefix_length([]), 0)
        self.assertEqual(common_prefix_length([[]]), 0)


class RowPlanTest(unittest.TestCase):
    def _example(self, prefix, suffixes):
        return fake_example([prefix + s for s in suffixes])

    def test_leaf_visibility_reconstructs_original_path(self):
        """For every leaf, mask-visible tokens at its last position must equal
        the original independent leaf token sequence, in the same positions."""
        prefix = [10, 11, 12, 13, 14]
        ex = self._example(prefix, [[20, 21], [20, 22, 23], [24]])
        rows, _ = shared_prefix_row_plans([ex], need_anchor=False, row_tokens=512)
        seen = {}
        for row in rows:
            for leaf_idx, last in row["leaf_lasts"]:
                self.assertNotIn(leaf_idx, seen)
                seen[leaf_idx] = (visible_tokens(row, last), row, last)
        for li, leaf in enumerate(ex["leaf_tokens"]):
            tokens, row, last = seen[li]
            self.assertEqual(tokens, leaf, f"leaf {li}: visible tokens differ")
            # positions of visible tokens must be 0..len(leaf)-1 in order
            self.assertEqual(visible_positions(row, last), list(range(len(leaf))))

    def test_suffix_tokens_cannot_see_other_suffixes(self):
        prefix = [1, 2, 3]
        ex = self._example(prefix, [[7, 7], [8, 8]])
        rows, _ = shared_prefix_row_plans([ex], need_anchor=False, row_tokens=512)
        row = rows[0]
        seg = row["segment_ids"]
        for t, s_t in enumerate(seg):
            for u, s_u in enumerate(seg):
                allowed = u <= t and (s_u == 0 or s_u == s_t)
                if s_u != 0 and s_u != s_t:
                    self.assertFalse(allowed, "cross-suffix visibility leaks")
                if s_t == 0 and s_u != 0:
                    self.assertFalse(allowed, "prefix token attends to a suffix token")

    def test_anchor_packed_in_first_row_and_reconstructs(self):
        prefix = [5, 6, 7, 8]
        ex = fake_example([prefix + [9, 9], prefix + [9, 8]], anchor=prefix + [4, 0])
        rows, _ = shared_prefix_row_plans([ex], need_anchor=True, row_tokens=512)
        anchor_seen = {}
        for row in rows:
            for ex_idx, last in row["anchor_lasts"]:
                anchor_seen[ex_idx] = (visible_tokens(row, last), row, last)
        tokens, row, last = anchor_seen[0]
        self.assertEqual(tokens, ex["anchor_tokens"])
        self.assertEqual(visible_positions(row, last), list(range(len(ex["anchor_tokens"]))))

    def test_single_leaf_degenerates_to_full_path(self):
        leaf = [3, 4, 5, 6, 7]
        ex = fake_example([leaf], k=2, typ="boolean")
        rows, _ = shared_prefix_row_plans([ex], need_anchor=False, row_tokens=512)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["tokens"], leaf)
        self.assertEqual(rows[0]["leaf_lasts"], [(0, len(leaf) - 1)])

    def test_chunking_splits_leaves_and_repeats_prefix(self):
        prefix = [1] * 40
        ex = self._example(prefix, [[100 + i] * 10 for i in range(10)])
        rows, _ = shared_prefix_row_plans([ex], need_anchor=False, row_tokens=70)
        # row budget 70, prefix 40, suffix width 10 -> per_row = 3
        self.assertEqual(len(rows), 4)  # ceil(10/3)
        for row in rows:
            self.assertEqual(row["tokens"][:40], prefix)
        covered = sorted(li for row in rows for li, _ in row["leaf_lasts"])
        self.assertEqual(covered, list(range(10)))

    def test_prefix_longer_than_budget_still_emits_one_suffix_per_row(self):
        prefix = [7] * 100
        ex = self._example(prefix, [[200 + i] * 5 for i in range(3)])
        rows, _ = shared_prefix_row_plans([ex], need_anchor=False, row_tokens=50)
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertEqual(len(row["leaf_lasts"]), 1)

    def test_inconsistent_paths_rejected(self):
        ex = fake_example([[1, 2, 3], [4, 5, 6]])
        with self.assertRaisesRegex(ValueError, "前缀"):
            shared_prefix_row_plans([ex], need_anchor=False)

    def test_255_leaves_planned(self):
        prefix = list(range(60))
        ex = self._example(prefix, [[500 + i] * 15 for i in range(255)])
        rows, stats = shared_prefix_row_plans([ex], need_anchor=False, row_tokens=512)
        covered = sorted(li for row in rows for li, _ in row["leaf_lasts"])
        self.assertEqual(covered, list(range(255)))
        self.assertEqual(stats["leaves"], 255)
        self.assertGreater(stats["packed_tokens"], 0)


class ChunkPackedRowsTest(unittest.TestCase):
    def _rows(self, lens):
        return [{"tokens": [0] * n, "positions": [0] * n, "segment_ids": [0] * n,
                 "leaf_lasts": [], "anchor_lasts": []} for n in lens]

    def test_max_rows_bound(self):
        chunks = chunk_packed_rows(self._rows([10] * 40), max_rows=8,
                                   max_attn_positions=10**12)
        self.assertEqual(len(chunks), 5)
        self.assertTrue(all(len(c) <= 8 for c in chunks))

    def test_attention_area_bound(self):
        # width 100 -> each row contributes ~100*100 = 10k attention positions
        chunks = chunk_packed_rows(self._rows([100] * 10), max_rows=64,
                                   max_attn_positions=30_000)
        self.assertTrue(all(len(c) * 100 * 100 <= 30_000 for c in chunks))
        self.assertEqual(sum(len(c) for c in chunks), 10)

    def test_oversized_row_gets_own_chunk(self):
        chunks = chunk_packed_rows(self._rows([5000, 10, 10]), max_rows=8,
                                   max_attn_positions=1_000_000)
        self.assertEqual(len(chunks[0]), 1)
        self.assertEqual(len(chunks[0][0]["tokens"]), 5000)
        self.assertEqual(sum(len(c) for c in chunks), 3)


@unittest.skipUnless(CHECKPOINT.is_dir(), "checkpoint not present")
class RealModelParityTest(unittest.TestCase):
    """Loads the real checkpoint once; parity gate mirrors the X3 receipt."""

    @classmethod
    def setUpClass(cls):
        from predict_toy_decisions import DecisionPredictor
        cls.engine = DecisionPredictor(str(CHECKPOINT), device_name="mps", precision="fp32")

    def _payload(self):
        return {"states": [{
            "id": "x3t", "state": "Runbook: promote replica only if lag<10s. "
                                  "pg-02 lag is 41s. pg-01 reachable.",
            "questions": {
                "b": {"type": "boolean", "instructions": "May pg-02 be promoted now?",
                      "criteria": {"false": "no", "true": "yes"}},
                "c": {"type": "choice", "instructions": "Best action?",
                      "criteria": {"a": "promote", "b": "wait", "c": "freeze", "d": "rollback"}},
                "s": {"type": "score", "instructions": "Severity?",
                      "criteria": ["none", "minor", "major", "critical"]}}}]}

    def test_flag_off_matches_default_predict(self):
        ref = self.engine.predict(self._payload())
        default = self.engine.predict(self._payload(), shared_prefix=False)
        self.assertEqual(ref["states"], default["states"])
        self.assertFalse(ref["execution"]["prefix_sharing"])

    def test_parity_within_gate_tolerance(self):
        ref = self.engine.predict(self._payload())
        fast = self.engine.predict(self._payload(), shared_prefix=True)
        for state_r, state_f in zip(ref["states"], fast["states"]):
            self.assertEqual(set(state_r["answers"]), set(state_f["answers"]))
            for qid, a in state_r["answers"].items():
                b = state_f["answers"][qid]
                self.assertEqual(set(a["probabilities"]), set(b["probabilities"]))
                for cid in a["probabilities"]:
                    self.assertLessEqual(abs(a["probabilities"][cid] - b["probabilities"][cid]),
                                         PARITY_TOL, f"{qid}/{cid}")
                am = max(a["probabilities"], key=a["probabilities"].get)
                bm = max(b["probabilities"], key=b["probabilities"].get)
                self.assertEqual(am, bm, f"{qid}: argmax flip")
        self.assertTrue(fast["execution"]["prefix_sharing"])

    def test_contract_schema_unchanged(self):
        ref = self.engine.predict(self._payload())
        fast = self.engine.predict(self._payload(), shared_prefix=True)
        self.assertEqual(set(ref), set(fast))
        self.assertEqual(ref["schema_version"], fast["schema_version"])
        self.assertEqual(set(ref["checkpoint"]), set(fast["checkpoint"]))
        for s_r, s_f in zip(ref["states"], fast["states"]):
            self.assertEqual(set(s_r), set(s_f))
            for qid in s_r["answers"]:
                self.assertEqual(set(s_r["answers"][qid]), set(s_f["answers"][qid]))

    def test_255_candidate_parity(self):
        criteria = {f"o{i:03d}": f"option {i} description text" for i in range(255)}
        payload = {"states": [{"id": "big", "state": "state text for a large fan-out",
                               "questions": {"q": {"type": "choice", "instructions": "pick one",
                                                   "criteria": criteria}}}]}
        ref = self.engine.predict(payload)
        fast = self.engine.predict(payload, shared_prefix=True)
        a = ref["states"][0]["answers"]["q"]
        b = fast["states"][0]["answers"]["q"]
        dp = max(abs(a["probabilities"][c] - b["probabilities"][c]) for c in a["probabilities"])
        self.assertLessEqual(dp, PARITY_TOL)
        self.assertEqual(a["choice"], b["choice"])


if __name__ == "__main__":
    unittest.main()
