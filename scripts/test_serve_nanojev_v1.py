import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

import serve_decisions as service


class FakeSystemOneServer:
    def __init__(self, noul=0.99, fail=False):
        owner = self
        self.calls = []
        self.noul = noul
        self.fail = fail

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0) or 0)
                self.rfile.read(length)
                owner.calls.append(self.path)
                if owner.fail:
                    self.send_response(500)
                    body = b"{}"
                else:
                    body = json.dumps({
                        "answers": {"q": {"type": "noul", "noul": owner.noul}},
                        "usage": {"input_tokens": 1, "output_tokens": 0},
                    }).encode()
                    self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                body = b'{"status":"ok"}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
                return

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class UnifiedServiceTest(unittest.TestCase):
    def test_confident_gate(self):
        self.assertTrue(service._confident({"q": {"noul": 0.99}}))
        self.assertTrue(service._confident({"q": {"noul": 0.01}}))
        self.assertFalse(service._confident({"q": {"noul": 0.7}}))
        self.assertFalse(service._confident({}))
        self.assertFalse(service._confident({"q": {"probabilities": {"a": 0.4, "b": 0.6}}}, threshold=0.9))
        self.assertTrue(service._confident({"q": {"probabilities": {"a": 0.05, "b": 0.95}}}, threshold=0.9))

    def test_systemone_route_direct_and_unknown(self):
        backend = FakeSystemOneServer(noul=0.5)
        self.addCleanup(backend.close)
        with patch.dict(service.SCORER_BACKENDS,
                        {"winnow": ("127.0.0.1", backend.port, "fake")}):
            status, body = service.systemone_route(b'{"x":1}', "winnow")
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)["answers"]["q"]["noul"], 0.5)
        status, body = service.systemone_route(b"{}", "nope")
        self.assertEqual(status, 400)

    def test_systemone_route_cascade_fallback(self):
        fast = FakeSystemOneServer(noul=0.6)      # not confident -> strong
        strong = FakeSystemOneServer(noul=0.99)
        self.addCleanup(fast.close)
        self.addCleanup(strong.close)
        backends = {"kev": ("127.0.0.1", fast.port, "kev"),
                    "winnow": ("127.0.0.1", strong.port, "win")}
        with patch.dict(service.SCORER_BACKENDS, backends):
            status, body = service.systemone_route(b'{"x":1}', "cascade")
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)["answers"]["q"]["noul"], 0.99)
            self.assertEqual(len(fast.calls), 1)
            self.assertEqual(len(strong.calls), 1)
            # Confident fast answer stays on fast path.
            fast.noul = 0.99
            status, body = service.systemone_route(b'{"x":1}', "cascade")
            self.assertEqual(json.loads(body)["answers"]["q"]["noul"], 0.99)
            self.assertEqual(len(fast.calls), 2)
            self.assertEqual(len(strong.calls), 1)  # no extra strong call

    def test_context_gate_eval_with_injected_scorer(self):
        def stub(payload):
            return {"states": [{"id": s["id"], "answers": {
                name: {"type": "boolean",
                       "probabilities": {"false": 0.01, "true": 0.99}}
                for name in (s.get("questions") or {})}}
                for s in payload["states"]]}
        payload = {"request": {"model": "t", "messages": [
            {"role": "system", "content": "keep"},
            {"role": "assistant", "content": "old note"},
            {"role": "user", "content": "current ask"}]},
            "wire_format": "openai_chat", "threshold": 0.9,
            "sidecar": {"segments": {"/messages/1/content": {"eligible": True}}}}
        status, receipt = service.context_gate_eval(payload, scorer=stub)
        self.assertEqual(status, 200)
        self.assertEqual(receipt["status"], "scored")
        drops = [s for s in receipt["segments"] if s["suggestion"] == "drop"]
        self.assertEqual([s["pointer"] for s in drops], ["/messages/1/content"])
        self.assertTrue(receipt["forwarded_unchanged"])
        # Bad payloads and unknown backends fail with fixed errors.
        self.assertEqual(service.context_gate_eval({})[0], 400)
        self.assertEqual(service.context_gate_eval(
            payload | {"backend": "nope"})[0], 400)


if __name__ == "__main__":
    unittest.main()
