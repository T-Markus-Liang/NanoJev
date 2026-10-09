#!/usr/bin/env python3
"""ConsensusScorer (AND-drop) contract tests.

Semantics: combined P(irrelevant) = min(member P) so a drop fires only when
every member independently clears the threshold — the FP=0 ensemble measured
on the real-context eval (docs/REAL_CONTEXT_EVAL_RESULTS_V1.md §ensemble).
Member failure propagates as ScorerError so the gate fails open (retain).
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scorer_adapters_v1 import ConsensusScorer, ScorerError


def _stub(prob_by_qid):
    def score(payload):
        states = []
        for state in payload["states"]:
            answers = {}
            for qid in state.get("questions", {}):
                true_p = prob_by_qid.get(qid, prob_by_qid.get("*", 0.5))
                answers[qid] = {"type": "boolean",
                                "probabilities": {"false": 1.0 - true_p, "true": true_p}}
            states.append({"id": state["id"], "answers": answers})
        return {"checkpoint": {"adapter": "stub"}, "states": states}
    return score


def _payload(n=2, questions=("irrelevant_0",)):
    return {"states": [{"id": f"s{i}", "state": "x",
                        "questions": {q: {"type": "boolean", "instructions": "q?"}
                                      for q in questions}}
                       for i in range(n)]}


class ConsensusScorerTest(unittest.TestCase):
    def test_min_semantics(self):
        scorer = ConsensusScorer([_stub({"*": 0.9}), _stub({"*": 0.3})])
        result = scorer(_payload(1))
        self.assertAlmostEqual(
            result["states"][0]["answers"]["irrelevant_0"]["probabilities"]["true"],
            0.3)

    def test_both_high_stays_high(self):
        scorer = ConsensusScorer([_stub({"*": 0.9}), _stub({"*": 0.8})])
        p = scorer(_payload(1))["states"][0]["answers"]["irrelevant_0"]["probabilities"]["true"]
        self.assertAlmostEqual(p, 0.8)

    def test_per_question_independence(self):
        a = _stub({"irrelevant_0": 0.9, "irrelevant_1": 0.1})
        b = _stub({"irrelevant_0": 0.7, "irrelevant_1": 0.6})
        result = ConsensusScorer([a, b])({"states": [{"id": "b", "state": "x", "questions": {
            "irrelevant_0": {"type": "boolean", "instructions": "?"},
            "irrelevant_1": {"type": "boolean", "instructions": "?"}}}]})
        answers = result["states"][0]["answers"]
        self.assertAlmostEqual(answers["irrelevant_0"]["probabilities"]["true"], 0.7)
        self.assertAlmostEqual(answers["irrelevant_1"]["probabilities"]["true"], 0.1)

    def test_member_failure_propagates(self):
        def boom(_payload):
            raise RuntimeError("member down")
        with self.assertRaises(RuntimeError):
            ConsensusScorer([_stub({"*": 0.9}), boom])(_payload(1))

    def test_member_missing_state_fails(self):
        def short(_payload):
            return {"states": []}
        with self.assertRaises(ScorerError):
            ConsensusScorer([_stub({"*": 0.9}), short])(_payload(1))

    def test_invalid_probability_fails(self):
        def bad(_payload):
            return {"states": [{"id": "s0", "answers": {
                "irrelevant_0": {"type": "boolean",
                                 "probabilities": {"true": "NaN"}}}}]}
        with self.assertRaises(ScorerError):
            ConsensusScorer([_stub({"*": 0.9}), bad])(_payload(1))

    def test_requires_two_members(self):
        with self.assertRaises(ValueError):
            ConsensusScorer([_stub({"*": 0.9})])
        with self.assertRaises(TypeError):
            ConsensusScorer([_stub({"*": 0.9}), "not-callable"])

    def test_requires_states_list(self):
        with self.assertRaises(ScorerError):
            ConsensusScorer([_stub({}), _stub({})])({"states": "bad"})


if __name__ == "__main__":
    unittest.main()
