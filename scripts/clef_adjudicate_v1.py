#!/usr/bin/env python3
"""clef_adjudicate_v1.py — adjudicate mined drop candidates with clef-flash.

Reads mining candidate files (each record carries a SystemOne `request`), posts
to a local clef server, and writes adjudicated labels:

  noul >= --tau      -> verified drop (targets injected true=1)
  otherwise          -> rejected (kept for audit, not used as drop label)

Resumable: skips record_ids already present in --output.
Usage:
  python3 scripts/clef_adjudicate_v1.py --port 8099 --tau 0.7 \
      --candidates data/v5_mining/f2_hard_negative_candidates.jsonl \
                   data/v5_mining/f4_outcome_positive_candidates.jsonl \
      --output data/v8_adjudication/clef_verdicts.jsonl
"""
import argparse, json, os, time, urllib.request


def post(port, body, timeout):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/systemone",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode()), (time.perf_counter() - t0) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    seen = set()
    if os.path.exists(args.output):
        for l in open(args.output):
            seen.add(json.loads(l)["record_id"])

    todo = []
    for path in args.candidates:
        for l in open(path):
            r = json.loads(l)
            rid = r["meta"].get("record_id")
            if r["meta"].get("proposed_label") == "yes" and rid not in seen:
                todo.append(r)
    print(f"to adjudicate: {len(todo)} (already done: {len(seen)})")

    verified = rejected = errors = 0
    with open(args.output, "a") as out:
        for n, r in enumerate(todo, 1):
            rid = r["meta"].get("record_id")
            try:
                resp, ms = post(args.port, r["request"], args.timeout)
                noul = (resp.get("answers", {}).get("irrelevant") or {}).get("noul")
            except Exception as e:
                noul, ms, resp = None, 0, {"error": str(e)}
            verdict = None
            if noul is not None:
                verdict = "drop" if noul >= args.tau else "reject"
                if verdict == "drop":
                    verified += 1
                else:
                    rejected += 1
            else:
                errors += 1
            out.write(json.dumps({"record_id": rid, "noul": noul,
                "verdict": verdict, "ms": round(ms, 1),
                "kind": r["meta"].get("candidate_kind"),
                "proposed": r["meta"].get("proposed_label")},
                ensure_ascii=False) + "\n")
            out.flush()
            if n % 20 == 0:
                print(f"{n}/{len(todo)} verified={verified} rejected={rejected} err={errors}", flush=True)
    print(f"done: verified={verified} rejected={rejected} errors={errors}")


if __name__ == "__main__":
    main()
