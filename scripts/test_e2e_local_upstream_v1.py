import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_e2e_local_upstream_v1 as e2e


class FakeGateway:
    def __init__(self, *a, **k):
        self.server_address = ("127.0.0.1", 9999)

    def serve_forever(self):
        pass

    def shutdown(self):
        pass

    def server_close(self):
        pass


def fake_chat(body, port, sidecar=None):
    saved = 10 if sidecar else 0
    return {"status": 200, "prompt_tokens": 100 - saved,
            "completion_tokens": 5, "text": "the answer is ok"}


class E2ERunnerTest(unittest.TestCase):
    def _manifest(self, tmp, cases):
        path = Path(tmp) / "m.json"
        path.write_text(json.dumps({"cases": cases}))
        return path

    def test_counts_savings_and_regressions(self):
        with tempfile.TemporaryDirectory() as tmp:
            cases = [
                {"case_id": "a", "wire_format": "openai_chat",
                 "body": {"messages": []},
                 "sidecar": {"segments": {}},
                 "downstream": {"required_strings": ["ok"]}},
                {"case_id": "b", "wire_format": "openai_chat",
                 "body": {"messages": []},
                 "downstream": {"required_strings": ["missing"]}},
            ]
            with patch.object(e2e, "chat", fake_chat), \
                 patch.object(e2e, "make_server", lambda *a, **k: FakeGateway()), \
                 patch.object(e2e, "SystemOneHTTPScorer", lambda *a, **k: object()):
                receipt = e2e.run_e2e(
                    [self._manifest(tmp, cases).relative_to(e2e.ROOT)
                     if False else str(self._manifest(tmp, cases))])
            self.assertEqual(receipt["totals"]["cases"], 2)
            self.assertEqual(receipt["totals"]["prompt_tokens_saved"], 10)
            # "b" lacks sidecar → same answer both ways; baseline also fails
            # required strings → no regression flagged
            self.assertEqual(receipt["totals"]["answer_regressions"], 0)

    def test_upstream_unsupported_skips_case(self):
        def error_chat(body, port, sidecar=None):
            return {"status": 500, "prompt_tokens": None,
                    "completion_tokens": None, "text": ""}

        with tempfile.TemporaryDirectory() as tmp:
            cases = [{"case_id": "img", "wire_format": "openai_chat",
                      "body": {"messages": []}, "sidecar": {"segments": {}}}]
            manifest = self._manifest(tmp, cases)
            with patch.object(e2e, "chat", error_chat), \
                 patch.object(e2e, "make_server", lambda *a, **k: FakeGateway()), \
                 patch.object(e2e, "SystemOneHTTPScorer", lambda *a, **k: object()):
                receipt = e2e.run_e2e([str(manifest)])
            self.assertTrue(receipt["ok"])
            self.assertEqual(receipt["cases"][0]["skipped"],
                             "upstream_unsupported")
            self.assertEqual(receipt["failures"], [])


if __name__ == "__main__":
    unittest.main()
