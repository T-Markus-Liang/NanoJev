#!/usr/bin/env python3
"""Unified nanojev local service: checkpoint decisions + scorer proxy + context-gate.

One port, one process. Weights load once; no provider calls. Backends
(Winnow/Kev scorer servers) are probed, never started, by this process.
"""
import argparse
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import mimetypes
from pathlib import Path
import time
from urllib.parse import unquote, urlparse, parse_qs

SCORER_BACKENDS = {
    "winnow": ("127.0.0.1", 8091, "Winnow-12B"),
    "kev": ("127.0.0.1", 8092, "kev-latest"),
    "valen": ("127.0.0.1", 8093, "nano_sft_v2/valen-head@Qwen3.5-0.8B"),
    "valen_lora": ("127.0.0.1", 8094, "nano_sft_text_v4_fp32/valen-head@Qwen3.5-0.8B"),
}
DEFAULT_SCORER_BACKEND = "winnow"
CONSENSUS_MEMBERS = ("valen_lora", "winnow")
CASCADE_THRESHOLD = 0.95
MAX_CONTEXT_GATE_BYTES = 2_000_000


def backend_up(host, port, timeout=0.5):
    try:
        connection = HTTPConnection(host, port, timeout=timeout)
        connection.request("GET", "/health" if port == 8091 else "/v1/models")
        status = connection.getresponse().status
        connection.close()
        return status == 200
    except Exception:
        return False


def post_systemone(host, port, body, timeout=60.0):
    """Forward a raw /v1/systemone JSON body to a scorer backend."""
    connection = HTTPConnection(host, port, timeout=timeout)
    try:
        connection.request("POST", "/v1/systemone", body=body,
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, response.read(2_000_000)
    finally:
        connection.close()


def _confident(answers, threshold=CASCADE_THRESHOLD):
    """A /v1/systemone answer map is confident when every answer's top
    probability reaches the threshold."""
    if not isinstance(answers, dict) or not answers:
        return False
    for answer in answers.values():
        if not isinstance(answer, dict):
            return False
        if "noul" in answer:
            value = answer["noul"]
            if not isinstance(value, (int, float)) or max(value, 1 - value) < threshold:
                return False
        elif isinstance(answer.get("probabilities"), dict) and answer["probabilities"]:
            if max(answer["probabilities"].values()) < threshold:
                return False
        else:
            return False
    return True


def systemone_route(body, backend=DEFAULT_SCORER_BACKEND):
    """Route one /v1/systemone request to a backend.

    Returns (status, response_bytes). ``backend=cascade`` sends to the fast
    backend first and falls through to winnow when the fast answer is not
    confident or the fast backend is unreachable.
    """
    if backend == "cascade":
        fast_host, fast_port, _ = SCORER_BACKENDS["kev"]
        try:
            status, content = post_systemone(fast_host, fast_port, body)
            if status == 200 and _confident(
                    (json.loads(content) or {}).get("answers")):
                return status, content
        except Exception:
            pass
        backend = "winnow"
    if backend not in SCORER_BACKENDS:
        return 400, json.dumps({"error": f"unknown backend {backend!r}"}).encode()
    host, port, _model = SCORER_BACKENDS[backend]
    try:
        return post_systemone(host, port, body)
    except Exception:
        return 502, json.dumps({"error": "scorer backend unreachable"}).encode()


def context_gate_eval(payload, scorer_timeout=30.0, scorer=None):
    """Run the context gate in shadow mode on a posted request object.

    Input: {"request": <wire body>, "wire_format"?, "sidecar"?, "backend"?,
            "threshold"?, "max_scorer_payload_bytes"?}.
    Returns (status, receipt_dict). The receipt is content-free by
    construction (hashes, pointers, reason codes, probabilities).
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("request"), dict):
        return 400, {"error": "payload requires a 'request' object"}
    wire = payload.get("wire_format") or "openai_chat"
    backend = payload.get("backend") or DEFAULT_SCORER_BACKEND
    from context_gate_v1 import serialized, shadow_request
    from scorer_adapters_v1 import CascadeScorer, ConsensusScorer, SystemOneHTTPScorer
    if scorer is not None:
        pass  # injected scorer (tests)
    elif backend == "consensus":
        scorer = ConsensusScorer([
            SystemOneHTTPScorer(
                f"http://{SCORER_BACKENDS[name][0]}:{SCORER_BACKENDS[name][1]}",
                timeout=scorer_timeout, model_id=SCORER_BACKENDS[name][2],
                # eval-proven contract: heads consume the serialized state
                # string verbatim (compiler wraps it as one user message)
                parse_state=False)
            for name in CONSENSUS_MEMBERS])
    elif backend == "cascade":
        scorer = CascadeScorer(
            SystemOneHTTPScorer(
                f"http://{SCORER_BACKENDS['kev'][0]}:{SCORER_BACKENDS['kev'][1]}",
                timeout=scorer_timeout, model_id=SCORER_BACKENDS["kev"][2]),
            SystemOneHTTPScorer(
                f"http://{SCORER_BACKENDS['winnow'][0]}:{SCORER_BACKENDS['winnow'][1]}",
                timeout=scorer_timeout, model_id=SCORER_BACKENDS["winnow"][2]))
    elif backend in SCORER_BACKENDS:
        host, port, model = SCORER_BACKENDS[backend]
        scorer = SystemOneHTTPScorer(f"http://{host}:{port}",
                                     timeout=scorer_timeout, model_id=model)
    else:
        return 400, {"error": f"unknown backend {backend!r}"}
    raw = serialized(payload["request"]).encode("utf-8")
    threshold = payload.get("threshold", 0.99)
    max_payload = payload.get("max_scorer_payload_bytes")
    try:
        _, receipt = shadow_request(raw, wire,
                                    sidecar=payload.get("sidecar") or {},
                                    scorer=scorer, threshold=threshold,
                                    max_scorer_payload_bytes=max_payload)
    except Exception:
        return 500, {"error": "context gate evaluation failed"}
    return 200, receipt


def server_class(engine, web_root):
    class Handler(BaseHTTPRequestHandler):
        def send(self, code, content, mime='application/json; charset=utf-8'):
            self.send_response(code);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(content)))
            self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(content)

        def send_json(self, code, data):
            self.send(code,json.dumps(data,ensure_ascii=False,allow_nan=False).encode())

        def do_GET(self):
            route=urlparse(self.path).path
            if route=='/api/health':
                backends={name: backend_up(host, port)
                          for name,(host,port,_) in SCORER_BACKENDS.items()}
                self.send_json(200,{'ready':True,'model_loaded_once':True,'provider_calls':0,
                                    'backends':backends});return
            relative=unquote(route).lstrip('/') or 'index.html'
            target=(web_root/relative).resolve()
            if not target.is_relative_to(web_root) or not target.is_file():
                self.send_json(404,{'error':'File not found'});return
            self.send(200,target.read_bytes(),mimetypes.guess_type(str(target))[0] or 'application/octet-stream')

        def do_POST(self):
            route=urlparse(self.path)
            if route.path in ('/v1/systemone','/v1/context-gate'):
                try:
                    length=int(self.headers.get('Content-Length','0'))
                    if not 0 < length <= MAX_CONTEXT_GATE_BYTES:raise ValueError('Request must contain 1..2000000 bytes')
                    body=self.rfile.read(length)
                    if route.path=='/v1/systemone':
                        backend=(parse_qs(route.query).get('backend') or [DEFAULT_SCORER_BACKEND])[0]
                        status,content=systemone_route(body,backend)
                        self.send(status,content)
                    else:
                        from predict_toy_decisions import unique_object,reject_nonfinite
                        payload=json.loads(body,object_pairs_hook=unique_object,parse_constant=reject_nonfinite)
                        status,result=context_gate_eval(payload)
                        self.send_json(status,result)
                except (ValueError,TypeError) as exc:
                    self.send_json(400,{'error':str(exc)})
                except Exception:
                    self.send_json(500,{'error':'nanojev service request failed'})
                return
            if route.path!='/api/evaluate':
                self.send_json(404,{'error':'Unknown endpoint'});return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0 < length <= 2_000_000:raise ValueError('Request must contain 1..2000000 bytes')
                origin=self.headers.get('Origin')
                if origin and urlparse(origin).netloc != self.headers.get('Host'):raise ValueError('Cross-origin requests are disabled')
                from predict_toy_decisions import unique_object,reject_nonfinite,validate_request
                payload=json.loads(self.rfile.read(length),object_pairs_hook=unique_object,parse_constant=reject_nonfinite)
                states=validate_request(payload)
                questions=[q for s in states for q in s['questions'].values()]
                paths=sum(1 if q['type']=='boolean' else len(q['criteria']) for q in questions)
                if len(states)>32 or len(questions)>96 or paths>256:raise ValueError('Local demo limit:32 states,96 questions,256 candidate paths per request')
                before=time.perf_counter();result=engine.predict(payload)
                result['execution']['server_evaluation_seconds']=time.perf_counter()-before
                self.send_json(200,result)
            except (ValueError,TypeError,KeyError) as exc:
                self.send_json(400,{'error':str(exc)})
            except Exception:
                self.send_json(500,{'error':'Local model inference failed; inspect the server process. No teacher fallback was used.'})
                raise

        def log_message(self, fmt, *args):
            # HTTP method/path/status only; request states and credentials are not logged.
            print(fmt % args,flush=True)
    return Handler


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint-dir',required=True)
    p.add_argument('--web-root',default='web');p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8765)
    p.add_argument('--device',default='auto',help='auto, cpu, mps, or cuda[:index]')
    p.add_argument('--precision',choices=['auto','fp32','bf16'],default='auto');p.add_argument('--disable-native-triton',action='store_true')
    a=p.parse_args()
    from predict_toy_decisions import DecisionPredictor
    engine=DecisionPredictor(a.checkpoint_dir,device_name=a.device,precision=a.precision,
                             disable_native_triton=a.disable_native_triton)
    root=Path(a.web_root).resolve()
    if not (root/'index.html').is_file():raise ValueError('web-root must contain index.html')
    server=HTTPServer((a.host,a.port),server_class(engine,root))
    print(json.dumps({'url':f'http://{a.host}:{a.port}','ready':True,'provider_calls':0}),flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
