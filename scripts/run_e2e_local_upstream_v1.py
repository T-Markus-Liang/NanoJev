#!/usr/bin/env python3
"""End-to-end real-upstream measurement, fully local.

Sends fixture requests twice — once directly to the local Winnow-12B chat
upstream, once through the gateway in ACTIVE mode (the only authorized use
of active mode: loopback test upstreams). Compares real `prompt_tokens`
from upstream usage accounting and whether downstream answers still satisfy
the fixture's required_strings contract.

Scope: loopback only, fixture traffic only, no provider calls, no real user
data. This is the local substitute for Phase1/Phase2 real-provider
measurement — same metrics, zero external egress.
"""

import argparse
import base64
import hashlib
import json
import tempfile
import threading
import time
from http.client import HTTPConnection
from pathlib import Path

from main_model_gateway_v1 import (
    ContextGateGateway, GatewayConfig, SIDECAR_HEADER, make_server,
)
from scorer_adapters_v1 import SystemOneHTTPScorer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "e2e_local_upstream_v1.json"
UPSTREAM = "http://127.0.0.1:8091"
MODEL = "Winnow-12B"
MAX_TOKENS = 64


def sha256_text(data):
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def post_chat(url, body, timeout=120.0, headers=None):
    parsed_host, parsed_port = "127.0.0.1", int(url.rsplit(":", 1)[1])
    connection = HTTPConnection(parsed_host, parsed_port, timeout=timeout)
    all_headers = {"Content-Type": "application/json"}
    all_headers.update(headers or {})
    connection.request("POST", "/v1/chat/completions",
                       body=json.dumps(body), headers=all_headers)
    response = connection.getresponse()
    status, payload = response.status, response.read(8_000_000)
    connection.close()
    return status, payload


def chat(body, port, sidecar=None):
    body = dict(body, model=MODEL, max_tokens=MAX_TOKENS, temperature=0)
    headers = {}
    if sidecar:
        headers[SIDECAR_HEADER] = base64.b64encode(
            json.dumps(sidecar).encode("utf-8")).decode("ascii")
    status, raw = post_chat(f"http://127.0.0.1:{port}", body,
                            headers=headers)
    result = json.loads(raw or b"{}")
    usage = result.get("usage") or {}
    text = ""
    choices = result.get("choices") or []
    if choices:
        text = (choices[0].get("message") or {}).get("content") or ""
    return {"status": status, "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "text": text}


def judge_equivalence(question, baseline_text, filtered_text, timeout=30.0):
    """Advisory proxy only: local Winnow noul on answer equivalence."""
    connection = HTTPConnection("127.0.0.1", 8876, timeout=timeout)
    body = {"state": {"question": question,
                      "answer_a": baseline_text, "answer_b": filtered_text},
            "questions": {"same": {"type": "noul", "instructions":
                                   "Do answer_a and answer_b give the same "
                                   "answer to the question?"}}}
    connection.request("POST", "/v1/systemone", body=json.dumps(body),
                       headers={"Content-Type": "application/json"})
    response = connection.getresponse()
    payload = json.loads(response.read() or b"{}")
    connection.close()
    answer = ((payload.get("answers") or {}).get("same") or {})
    return answer.get("noul")


def run_e2e(manifest_paths, judge=False):
    failures, cases = [], []
    scorer = SystemOneHTTPScorer(UPSTREAM, timeout=30.0, model_id=MODEL)
    with tempfile.TemporaryDirectory() as tempdir:
        config = GatewayConfig(
            upstream_base_url=UPSTREAM, listen_host="127.0.0.1",
            listen_port=0, mode="active",
            receipt_log=Path(tempdir) / "receipts.jsonl",
            scorer=scorer, threshold=0.90)
        server = make_server(ContextGateGateway(config))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        port = server.server_address[1]
        try:
            for manifest_path in manifest_paths:
                manifest = json.loads(
                    (ROOT / manifest_path).read_text(encoding="utf-8"))
                for case in manifest.get("cases") or []:
                    if case.get("wire_format") != "openai_chat":
                        continue  # chat-completions upstream only speaks OpenAI wire
                    body = case["body"]
                    sidecar = case.get("sidecar")
                    started = time.perf_counter()
                    baseline = chat(body, port=8091)
                    filtered = chat(body, port=port, sidecar=sidecar)
                    elapsed = round((time.perf_counter() - started) * 1000, 1)
                    required = ((case.get("downstream") or {})
                                .get("required_strings") or [])
                    baseline_ok = all(s.lower() in baseline["text"].lower()
                                      for s in required)
                    filtered_ok = all(s.lower() in filtered["text"].lower()
                                      for s in required)
                    saved = None
                    if baseline["prompt_tokens"] and filtered["prompt_tokens"]:
                        saved = baseline["prompt_tokens"] - filtered["prompt_tokens"]
                    record = {
                        "case_id": case["case_id"],
                        "baseline_prompt_tokens": baseline["prompt_tokens"],
                        "filtered_prompt_tokens": filtered["prompt_tokens"],
                        "prompt_tokens_saved": saved,
                        "baseline_status": baseline["status"],
                        "filtered_status": filtered["status"],
                        "required_strings_total": len(required),
                        "baseline_required_ok": baseline_ok,
                        "filtered_required_ok": filtered_ok,
                        "answer_regression": baseline_ok and not filtered_ok,
                        "elapsed_ms": elapsed,
                    }
                    if baseline["status"] != 200 and \
                            baseline["status"] == filtered["status"]:
                        record["skipped"] = "upstream_unsupported"
                        cases.append(record)
                        continue
                    if not required and judge and \
                            baseline["status"] == 200 and \
                            filtered["status"] == 200:
                        question = (body.get("messages") or [{}])[-1].get(
                            "content", "")[:500]
                        agreement = judge_equivalence(
                            question, baseline["text"], filtered["text"])
                        record["answer_agreement_p"] = agreement
                        record["answer_disagreement"] = \
                            agreement is not None and agreement < 0.5
                    cases.append(record)
                    if record["baseline_status"] == 200 and \
                            record["filtered_status"] != 200:
                        failures.append(f"{case['case_id']}: filtered-only "
                                        "upstream error")
                    if record["answer_regression"]:
                        failures.append(
                            f"{case['case_id']}: filtered answer lost "
                            "required strings")
        finally:
            server.shutdown()
            server.server_close()
    totals = {
        "cases": len(cases),
        "prompt_tokens_baseline": sum(c["baseline_prompt_tokens"] or 0
                                      for c in cases),
        "prompt_tokens_filtered": sum(c["filtered_prompt_tokens"] or 0
                                      for c in cases),
        "prompt_tokens_saved": sum(c["prompt_tokens_saved"] or 0
                                   for c in cases),
        "answer_regressions": sum(1 for c in cases if c["answer_regression"]),
        "answer_disagreements": sum(1 for c in cases
                                    if c.get("answer_disagreement")),
        "judged_cases": sum(1 for c in cases
                            if c.get("answer_agreement_p") is not None),
        "skipped_cases": sum(1 for c in cases if c.get("skipped")),
        "required_string_cases": sum(1 for c in cases
                                     if c["required_strings_total"]),
    }
    if totals["prompt_tokens_baseline"]:
        totals["savings_pct"] = round(
            100 * totals["prompt_tokens_saved"]
            / totals["prompt_tokens_baseline"], 2)
    return {
        "schema_version": "nanojev-e2e-local-upstream-v1",
        "status": "e2e_pass" if not failures else "e2e_fail",
        "scope": "loopback upstream only; fixture traffic; active mode on "
                 "loopback; no provider calls; no real user data",
        "upstream": UPSTREAM, "upstream_model": MODEL,
        "provider_calls": 0,
        "totals": totals,
        "cases": cases,
        "failures": failures,
        "ok": not failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", action="store_true",
                        help="for cases without required_strings, ask the "
                             "local service whether answers agree (advisory "
                             "proxy, not ground truth)")
    parser.add_argument("--manifests", nargs="+",
                        default=["research/context_filter_value_fixture_manifest_v2.json",
                                 "research/context_filter_value_fixture_manifest_v3.json"])
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    receipt = run_e2e(args.manifests, judge=args.judge)
    encoded = json.dumps(receipt, ensure_ascii=False, indent=2,
                         sort_keys=True) + "\n"
    args.output.write_text(encoded, encoding="utf-8")
    print(json.dumps({"output": str(args.output),
                      "sha256": sha256_text(encoded), "ok": receipt["ok"],
                      "status": receipt["status"],
                      "totals": receipt["totals"]}, indent=2))
    if not receipt["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
