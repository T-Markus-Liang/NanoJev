#!/usr/bin/env python3
"""Score JevBench public rows against a local /v1/systemone backend.

Each JevBench task record (``{id, family, state, question, labels,
expected, split, ...}``) is posted verbatim as a flat systemone request
``{state, questions: {"decision": {type, instructions, criteria}}}`` — the
``expected`` label, ``labels`` list, ``group`` and ``provenance`` never
reach the model.  The answer's probability distribution is mapped onto the
task's exact label set (noul -> {no: 1-p, yes: p}; choice/score -> the
returned ``probabilities`` dict) and argmax gives the prediction.

    python3 scripts/eval_jevbench_local_v1.py \
        --url http://127.0.0.1:8091/v1/systemone \
        --data data/jevbench_offline_bundle_v1/official_jevbench_v1.2.4_public/public_231.jsonl \
        --output results/jevbench_local_231_winnow_v1.jsonl

Evaluation-only.  Per-item probabilities are local model outputs; the
``results/`` tree is gitignored, and publishable notes carry aggregates
only (AGENTS.md publication discipline).
"""
import argparse
import json
import math
import time
from http.client import HTTPConnection, HTTPSConnection
from pathlib import Path
from urllib.parse import urlparse

QUESTION_KEY = "decision"


def post(url, body, headers, timeout):
    parts = urlparse(url)
    conn_cls = HTTPSConnection if parts.scheme == "https" else HTTPConnection
    conn = conn_cls(parts.hostname, parts.port, timeout=timeout)
    path = parts.path + (f"?{parts.query}" if parts.query else "")
    started = time.perf_counter()
    try:
        conn.request("POST", path, body=body,
                     headers={"Content-Type": "application/json", **headers})
        resp = conn.getresponse()
        data = json.loads(resp.read(8_000_000))
        ms = (time.perf_counter() - started) * 1000
        return resp.status, data, ms
    finally:
        conn.close()


def build_request(record, stringify_state=False):
    """Map one JevBench task record to a flat systemone request body."""
    q = record["question"]
    nq = {"type": q["type"], "instructions": q["instructions"]}
    if q.get("criteria") is not None:
        nq["criteria"] = q["criteria"]
    state = record["state"]
    if stringify_state and not isinstance(state, str):
        # Some backends (e.g. the valen-head sidecar) accept only text or a
        # {"messages": [...]} object state.  Serialize other object/array
        # states to canonical JSON text so every scorer sees the same
        # information; disclosed in the run report.
        state = json.dumps(state, ensure_ascii=False)
    return {"state": state, "questions": {QUESTION_KEY: nq}}


def map_probs(record, answer):
    """Project a systemone answer onto the task's exact label set."""
    qtype = record["question"]["type"]
    if qtype == "noul":
        noul = answer.get("noul")
        mapped = {"no": None if noul is None else 1.0 - noul,
                  "yes": noul}
    else:
        probs = answer.get("probabilities")
        if not isinstance(probs, dict):
            raise ValueError("answer missing probabilities")
        mapped = {label: probs.get(label) for label in record["labels"]}
    out = {}
    for key, value in mapped.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) \
                or not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"invalid probability for label {key!r}")
        out[str(key)] = float(value)
    return out


def argmax_label(probs):
    best, best_p = None, -1.0
    for key in sorted(probs):
        if probs[key] > best_p:
            best, best_p = key, probs[key]
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--retry", type=int, default=3)
    ap.add_argument("--stringify-state", action="store_true",
                    help="json.dumps non-string state before POSTing "
                         "(for backends that only accept text/messages states)")
    args = ap.parse_args()

    records = [json.loads(l) for l in args.data.open() if l.strip()]
    if args.limit:
        records = records[: args.limit]

    done = set()
    if args.output.exists():
        # Rows carrying an "error" are re-attempted on resume (they were
        # transient/backend failures, not scored predictions).
        for l in args.output.open():
            row = json.loads(l)
            if "error" not in row:
                done.add(row["i"])
        mode = "a"
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        mode = "w"

    out = args.output.open(mode)
    n_done = errors = 0
    for i, rec in enumerate(records):
        if i in done:
            n_done += 1
            continue
        body = json.dumps(build_request(rec, args.stringify_state))
        status, data, ms = 0, None, 0.0
        for attempt in range(args.retry):
            try:
                status, data, ms = post(args.url, body, {}, args.timeout)
                if status == 200:
                    break
            except Exception:
                status = 0
            time.sleep(5 * (attempt + 1))
        row = {"i": i, "id": rec["id"], "family": rec.get("family"),
               "group": rec.get("group"),
               "question_type": rec["question"]["type"],
               "expected": rec.get("expected"), "ms": round(ms, 1)}
        try:
            if status != 200 or not data or "answers" not in data:
                raise ValueError(f"status {status}")
            answer = data["answers"].get(QUESTION_KEY) or {}
            probs = map_probs(rec, answer)
            predicted = argmax_label(probs)
            row.update({
                "probs": probs,
                "noul": answer.get("noul"),
                "predicted": predicted,
                "correct": (None if rec.get("expected") is None else
                            predicted == str(rec["expected"])),
                "model": data.get("model"),
            })
        except (ValueError, KeyError, TypeError) as exc:
            row.update({"probs": None, "noul": None, "predicted": None,
                        "correct": None, "error": str(exc)})
            errors += 1
        out.write(json.dumps(row) + "\n")
        out.flush()
        if (i + 1) % 25 == 0:
            print(json.dumps({"event": "progress", "records": i + 1,
                              "errors": errors}), flush=True)
    print(json.dumps({"event": "complete", "errors": errors,
                      "skipped_resumed": n_done}))


if __name__ == "__main__":
    main()
