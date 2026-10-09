#!/usr/bin/env python3
"""CLEF-Flash local /v1/systemone server — drop-in for parallel_eval_v1.py.

Cloudflare/clef-flash ships joint_schema_model.py with a `systemone()` helper
that accepts a Jev/SystemOne request body and returns the same response shape.
This wraps it in the same HTTP protocol as valen_head_server_v1.py:

    POST /v1/systemone   {"state","questions"} -> {"answers","model","usage"}
    GET  /health, /v1/models -> liveness

Usage:
    python scripts/clef_server_v1.py --port 8100 [--device mps|cpu]
"""
import argparse, gc, json, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer

MODEL = None
PROCESSOR = None
SYSTEMONE = None
RSS_LIMIT = None


def load(path, device):
    global MODEL, PROCESSOR, SYSTEMONE
    sys.path.insert(0, path)
    from joint_schema_model import load_release_model, systemone
    MODEL, PROCESSOR = load_release_model(path, device=device, attn_implementation="eager")
    SYSTEMONE = systemone
    print(f"clef loaded on {device}", flush=True)


def rss_gb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)
    except Exception:
        return 0.0


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/health", "/v1/models"):
            self._send(200, {"status": "ok", "model": "clef-flash"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/systemone":
            self._send(404, {"error": "not found"})
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            body.setdefault("model", "clef-flash")
            t0 = time.perf_counter()
            resp = SYSTEMONE(MODEL, PROCESSOR, body)
            ms = (time.perf_counter() - t0) * 1000
            resp.setdefault("model", "clef-flash")
            resp["ms"] = round(ms, 1)
            self._send(200, resp)
        except Exception as e:
            self._send(500, {"error": str(e)})
        finally:
            gc.collect()
            try:
                import torch
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception:
                pass
            if RSS_LIMIT and rss_gb() > RSS_LIMIT:
                print("rss limit hit, exiting", flush=True)
                import os, signal
                os.kill(os.getpid(), signal.SIGTERM)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model-path", required=True, help="snapshot_download dir")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--rss-restart-gb", type=float, default=0)
    args = ap.parse_args()
    global RSS_LIMIT
    RSS_LIMIT = args.rss_restart_gb
    load(args.model_path, args.device)
    HTTPServer((args.host, args.port), H).serve_forever()


if __name__ == "__main__":
    main()
