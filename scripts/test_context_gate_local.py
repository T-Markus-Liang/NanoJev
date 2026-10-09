from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from context_gate_local import LoggedScorer, LoopbackPredictor
from context_gate_v1 import shadow_request
from test_context_gate_v1 import encoded, request, score_result


class LocalContextGateTest(unittest.TestCase):
    def test_reject_remote_credentials_redirect_url_and_invalid_timeout(self):
        for url in ("https://127.0.0.1", "http://example.org", "http://localhost", "http://127.0.0.1/path",
                    "http://user:password@127.0.0.1", "http://127.0.0.1?token=secret", "http://127.0.0.1#fragment"):
            with self.assertRaises(ValueError):
                LoopbackPredictor(url)
        for timeout in (0, -1, 31, float("nan"), True):
            with self.assertRaises(ValueError):
                LoopbackPredictor("http://127.0.0.1", timeout)

    def test_proxy_is_ignored_and_redirect_not_followed(self):
        class Handler(BaseHTTPRequestHandler):
            redirect = False
            requests = 0

            def do_POST(self):
                Handler.requests += 1
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if Handler.redirect:
                    self.send_response(302)
                    self.send_header("Location", "http://must-not-be-contacted.invalid")
                    self.end_headers()
                else:
                    result = json.dumps(score_result(body)).encode()
                    self.send_response(200); self.send_header("Content-Length", str(len(result))); self.end_headers()
                    self.wfile.write(result)

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch.dict(os.environ, {"HTTP_PROXY": "http://127.0.0.1:1", "http_proxy": "http://127.0.0.1:1"}):
                scorer = LoopbackPredictor(f"http://127.0.0.1:{server.server_port}", timeout=1)
                payload = {"states": [{"id": "segment_0"}]}
                self.assertEqual(len(scorer(payload)["states"]), 1)
                Handler.redirect = True
                with self.assertRaisesRegex(ValueError, "failed"):
                    scorer(payload)
                self.assertEqual(Handler.requests, 2)
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=2)

    def test_existing_usage_schema_and_no_raw_debug_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "usage.jsonl"
            with patch.dict(os.environ, {"NANOJEV_LOG_PAYLOADS": "1"}):
                scorer = LoggedScorer(score_result, Path(tmp), log)
                raw = encoded(request(history="UNIQUE_PRIVATE_TEXT"))
                _, receipt = shadow_request(raw, "openai_chat", {"segments": {"/messages/1/content": {"eligible": True}}}, scorer)
            self.assertEqual(receipt["status"], "scored")
            event = json.loads(log.read_text())
            self.assertEqual(event["schema_version"], "nanojev-usage-v1")
            self.assertNotIn("raw_payload", event)
            self.assertNotIn("UNIQUE_PRIVATE_TEXT", log.read_text())
            self.assertEqual(scorer.helper.summarize_log(log)["decision_events"], 1)


if __name__ == "__main__":
    unittest.main()
