import io
import json
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

import nanojev_eval as cli


class FakeService:
    def __init__(self):
        owner = self
        self.requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                body = b'{"ready":true}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0) or 0)
                body = self.rfile.read(length)
                owner.requests.append({"path": self.path, "body": json.loads(body)})
                out = json.dumps({"model": "fake", "answers": {"q": {"noul": 0.9}},
                                  "usage": {"input_tokens": 1}}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, *_args):
                return

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class NanojevEvalTest(unittest.TestCase):
    def test_backend_arg_parsing(self):
        self.assertEqual(cli.backend_arg(["--backend", "kev"]), "kev")
        self.assertEqual(cli.backend_arg([]), "winnow")
        self.assertEqual(cli.backend_arg(["--backend", "cascade", "f.json"]),
                         "cascade")

    def test_read_input_skips_flag_values(self):
        # `--backend cascade` must not treat "cascade" as the input file.
        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False) as handle:
            handle.write('{"state":{}, "questions":{}}')
            path = handle.name
        self.addCleanup(Path(path).unlink)
        data = cli.read_input(["--backend", "cascade", path])
        self.assertEqual(data["state"], {})

    def test_systemone_call_posts_to_service(self):
        service = FakeService()
        self.addCleanup(service.close)
        payload = {"state": {"t": 1}, "questions": {"q": {"type": "noul",
                                                        "instructions": "i"}}}
        stdout = io.StringIO()
        with patch.object(sys, "argv",
                          ["nanojev_eval.py", "--backend", "kev"]), \
             patch.object(cli, "DEFAULT_URL", service.url), \
             patch("sys.stdin", io.StringIO(json.dumps(payload))), \
             redirect_stdout(stdout):
            cli.main()
        out = json.loads(stdout.getvalue())
        self.assertEqual(out["answers"]["q"]["noul"], 0.9)
        self.assertEqual(out["backend"], "kev")
        self.assertEqual(out["source"], "nanojev-local")
        self.assertTrue(service.requests[0]["path"].endswith("backend=kev"))


if __name__ == "__main__":
    unittest.main()
