import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

import check_local_services_v1 as check


class FakeHealthServer:
    def __init__(self, ready=True, evaluate_ok=True):
        owner = self
        self.ready = ready
        self.evaluate_ok = evaluate_ok

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/api/health" and owner.ready:
                    body = json.dumps({"ready": True, "provider_calls": 0}).encode()
                    self.send_response(200)
                else:
                    body = b"{}"
                    self.send_response(404)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0) or 0)
                self.rfile.read(length)
                if self.path == "/api/evaluate" and owner.evaluate_ok:
                    body = json.dumps({"states": []}).encode()
                    self.send_response(200)
                else:
                    body = b"{}"
                    self.send_response(404)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
                return

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


class ProbeTest(unittest.TestCase):
    def test_probe_url_up_and_down(self):
        server = FakeHealthServer()
        self.addCleanup(server.close)
        up = check.probe_url(f"http://127.0.0.1:{server.port}/api/health")
        self.assertTrue(up["up"])
        self.assertEqual(up["status_code"], 200)
        down = check.probe_url(f"http://127.0.0.1:{server.port}/nope")
        self.assertFalse(down["up"])
        unreachable = check.probe_url("http://127.0.0.1:1/api/health", timeout=0.5)
        self.assertFalse(unreachable["up"])

    def test_smoke_evaluate(self):
        server = FakeHealthServer()
        self.addCleanup(server.close)
        result = check.smoke_evaluate(server.port)
        self.assertTrue(result["ok"])

    def test_collect_marks_required_down(self):
        # Ports in NANOJEV_PORTS may or may not be up on a dev machine; force the
        # check logic by stubbing find_nanojev.
        original = check.find_nanojev
        check.find_nanojev = lambda timeout=2.0: (None, None)
        try:
            receipt = check.collect(timeout=0.2)
            self.assertFalse(receipt["ok"])
            required = [s for s in receipt["services"] if s["required"]]
            self.assertTrue(all(s["status"] == "down" for s in required))
            receipt = check.collect(timeout=0.2, smoke=True)
            self.assertFalse(receipt["ok"])
            self.assertFalse(receipt["smoke"]["ok"])
        finally:
            check.find_nanojev = original


if __name__ == "__main__":
    unittest.main()
