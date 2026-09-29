#!/usr/bin/env python3
"""T156 sidecar: loopback HTTP wrapper around the RLCD decision head.

serve_decisions.py is single-threaded, so the ~0.9s MPS forward cannot run
inside it — this sidecar owns the model in a separate process on
``127.0.0.1:8093`` and speaks the same wire shape as the winnow/kev
backends:

    POST /v1/systemone   {"state", "questions"} -> answers + decision
    GET  /v1/models      llama.cpp-style model list (backend_up probe)
    GET  /health         plain 200 (health probes)

Run with the valen venv (transformers 5.4 / Qwen3.5 support):

    external/valen/.venv/bin/python scripts/valen_head_server_v1.py

Loopback only; no provider calls, no API keys. Advisory scorer — the
service's shadow/fail-open semantics are unchanged.
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import valen_head_scorer_v1 as scorer_mod  # noqa: E402

MODEL_ID = scorer_mod.MODEL_ID


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, status, obj):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/health", "/v1/models"):
            self._send(200, {"data": [{"id": MODEL_ID}], "status": "ok"})
        else:
            self._send(404, {"error": "unknown path"})

    def do_POST(self):
        if self.path != "/v1/systemone":
            self._send(404, {"error": "unknown path"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            result = scorer_mod.get_scorer().score(body)
            self._send(200, result)
        except Exception as exc:
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8093)
    parser.add_argument("--checkpoint", type=Path,
                        default=scorer_mod.DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="bf16")
    args = parser.parse_args()
    if args.checkpoint != scorer_mod.DEFAULT_CHECKPOINT:
        global MODEL_ID
        MODEL_ID = scorer_mod.MODEL_ID = (
            f"{args.checkpoint.parent.name}/valen-head@Qwen3.5-0.8B")
    scorer_mod.get_scorer(checkpoint_dir=args.checkpoint,
                          device=args.device, dtype=args.dtype)
    print(f"valen-head sidecar on http://{args.host}:{args.port} "
          f"(model {MODEL_ID})", flush=True)
    HTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
