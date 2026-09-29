import json
import unittest
from unittest.mock import patch

import jev_route as router


class JevRouteTest(unittest.TestCase):
    def test_bucket_is_deterministic(self):
        payload = {"state": {"a": 1}, "questions": {"q": {"type": "noul"}}}
        self.assertEqual(router.route_bucket(payload),
                         router.route_bucket(payload))
        self.assertEqual(router.route_bucket(payload),
                         router.route_bucket({"questions": {"q": {"type": "noul"}},
                                              "state": {"a": 1}}))  # key order

    def test_bucket_range_and_ratio_edges(self):
        payload = {"state": {}, "questions": {}}
        bucket = router.route_bucket(payload)
        self.assertTrue(0 <= bucket < 1000)
        # ratio 0 -> never local; ratio 1 -> always local
        self.assertFalse(bucket < 0.0 * 1000)
        self.assertTrue(bucket < 1.0 * 1000)

    def test_ratio_split_is_approximately_correct(self):
        # 200 distinct payloads at ratio 0.3 should land near 60 local.
        local = sum(
            router.route_bucket({"state": {"i": i}, "questions": {"q": {}}}) < 300
            for i in range(200))
        self.assertTrue(30 <= local <= 95, f"local count {local} outside bounds")

    def test_answer_confidence(self):
        self.assertEqual(router.answer_confidence(
            {"answers": {"q": {"noul": 0.99}}}), 0.99)
        self.assertAlmostEqual(router.answer_confidence(
            {"answers": {"q": {"noul": 0.01}}}), 0.99)
        self.assertEqual(router.answer_confidence(
            {"answers": {"q": {"probabilities": {"a": 0.7, "b": 0.3}}}}), 0.7)
        self.assertIsNone(router.answer_confidence({"answers": {}}))
        self.assertIsNone(router.answer_confidence({"answers": {"q": {}}}))

    def test_quality_mode_escalation(self):
        calls = []

        def fake_cli(binary, payload):
            calls.append(binary)
            if "nanojev" in binary:
                return {"answers": {"q": {"noul": 0.6}}, "elapsed_ms": 1}
            return {"answers": {"q": {"noul": 0.99}}, "elapsed_ms": 1}

        with patch.object(router, "run_cli", fake_cli):
            result, route = router.run_quality({}, 0.95)
        self.assertEqual(route["actual"], "official")
        self.assertTrue(route["escalated"])
        self.assertEqual(route["local_confidence"], 0.6)
        self.assertEqual(len(calls), 2)
        # Confident local stays local — no official call.
        calls.clear()

        def confident(binary, payload):
            calls.append(binary)
            return {"answers": {"q": {"noul": 0.99}}, "elapsed_ms": 1}

        with patch.object(router, "run_cli", confident):
            result, route = router.run_quality({}, 0.95)
        self.assertEqual(route["actual"], "local")
        self.assertFalse(route["escalated"])
        self.assertEqual(len(calls), 1)
        # Local down → official fallback.
        calls.clear()

        def down(binary, payload):
            calls.append(binary)
            if "nanojev" in binary:
                raise RuntimeError("down")
            return {"answers": {"q": {"noul": 0.99}}, "elapsed_ms": 1}

        with patch.object(router, "run_cli", down):
            result, route = router.run_quality({}, 0.95)
        self.assertEqual(route["actual"], "official")
        self.assertTrue(route["fallback"])
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
