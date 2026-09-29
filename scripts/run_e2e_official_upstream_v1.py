#!/usr/bin/env python3
"""Official TypeSafe Jev as the real billed upstream (owner-authorized test).

For each open-corpus case:

1. Local Winnow scorer (via context-gate logic) proposes which eligible
   history segments are certainly irrelevant.
2. Official Jev receives the SAME conversation twice — full state and
   filtered state — and we compare its billed `usage.input_tokens`.
   This is the real-provider accounting Phase1 was designed to measure,
   without any customer data: open-corpus traffic only.
3. For every segment the local scorer dropped, official Jev independently
   answers the gate's own question ("certainly irrelevant?") — a direct
   scorer-agreement measurement on real data.

Budget: --max-cases bounds calls; every call logs usage + reported model
version (per AGENTS.md official-Jev channel rule).
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_gate_v1 import shadow_request  # noqa: E402
from scorer_adapters_v1 import SystemOneHTTPScorer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
JEV_EVAL = str(Path.home() / ".local/bin/jev-eval")
OUT = ROOT / "results" / "e2e_official_upstream_v1.json"

DEP_QUESTION = ("Does the final user message rely on information established "
                "earlier in this conversation to be answered correctly?")
DROP_QUESTION = ("Is the candidate context certainly irrelevant to fulfilling "
                 "the current user request? Answer false if uncertain.")


def sha256_text(data):
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def official_call(state, questions, timeout=60.0):
    started = time.perf_counter()
    proc = subprocess.run([JEV_EVAL], input=json.dumps(
        {"state": state, "questions": questions}), capture_output=True,
        text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"jev-eval failed: {(proc.stdout or proc.stderr)[:200]}")
    result = json.loads(proc.stdout)
    return {"answers": result.get("answers"),
            "input_tokens": (result.get("usage") or {}).get("inputTokens"),
            "model": result.get("model"),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)}


def resolve_pointer(messages, pointer):
    """Resolve '/messages/i/content[/j/text]' → the targeted text or None."""
    parts = [p for p in pointer.split("/") if p]
    node = messages
    try:
        for part in parts[1:]:
            node = node[int(part)] if isinstance(node, list) else node[part]
    except (IndexError, KeyError, TypeError, ValueError):
        return None
    return node if isinstance(node, str) else None


def apply_drops(messages, drop_pointers):
    """Remove only the targeted text; part pointers blank that part."""
    messages = json.loads(json.dumps(messages))  # deep copy
    whole = set()
    for pointer in drop_pointers:
        parts = [p for p in pointer.split("/") if p]
        if len(parts) == 3:                       # /messages/i/content
            whole.add(int(parts[1]))
        elif len(parts) >= 5:                     # /messages/i/content/j/text
            try:
                i, j = int(parts[1]), int(parts[3])
                messages[i]["content"][j]["text"] = ""
            except (IndexError, KeyError, TypeError, ValueError):
                continue
    return [m for i, m in enumerate(messages) if i not in whole]


def run(corpus_path, max_cases, threshold=0.90, offset=0):
    manifest = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
    scorer = SystemOneHTTPScorer("http://127.0.0.1:8091", timeout=30.0,
                               model_id="Winnow-12B")
    cases, failures, official_calls, input_tokens = [], [], 0, 0
    official_models = set()
    for case in (manifest.get("cases") or [])[offset:offset + max_cases]:
        if case.get("wire_format") != "openai_chat":
            continue
        messages = case["body"]["messages"]
        # Step 1: local filter decision (shadow semantics — advisory only)
        _, receipt = shadow_request(
            json.dumps(case["body"]).encode("utf-8"), "openai_chat",
            case.get("sidecar") or {}, scorer, threshold=threshold)
        drops = [s["pointer"] for s in receipt.get("segments", [])
                 if s.get("suggestion") == "drop"]
        filtered_messages = apply_drops(messages, drops)
        record = {"case_id": case["case_id"], "local_drops": len(drops)}
        try:
            # Step 2: official Jev on full vs filtered state (real billed tokens)
            questions = {"context_dependent":
                         {"type": "noul", "instructions": DEP_QUESTION}}
            base = official_call({"conversation": messages}, questions)
            official_calls += 1
            input_tokens += base["input_tokens"] or 0
            official_models.add(base.get("model"))
            record.update({
                "baseline_input_tokens": base["input_tokens"],
                "baseline_answer": ((base["answers"] or {})
                                    .get("context_dependent") or {}).get("noul"),
            })
            if drops:
                filt = official_call({"conversation": filtered_messages},
                                     questions)
                official_calls += 1
                input_tokens += filt["input_tokens"] or 0
                record.update({
                    "filtered_input_tokens": filt["input_tokens"],
                    "input_tokens_saved": (base["input_tokens"] or 0)
                        - (filt["input_tokens"] or 0),
                    "filtered_answer": ((filt["answers"] or {})
                                        .get("context_dependent") or {}).get("noul"),
                })
                # Step 3: official Jev judges each local drop decision
                agreements = []
                for pointer in drops:
                    candidate_text = resolve_pointer(messages, pointer)
                    if candidate_text is None:
                        record.setdefault("drop_verdict_skipped", []).append(
                            pointer)
                        continue
                    verdict = official_call({
                        "current_user_request": messages[-1]["content"],
                        "candidate_context": candidate_text,
                    }, {"irrelevant": {"type": "noul",
                                       "instructions": DROP_QUESTION}})
                    official_calls += 1
                    input_tokens += verdict["input_tokens"] or 0
                    p = ((verdict["answers"] or {}).get("irrelevant") or {}) \
                        .get("noul")
                    agreements.append({"pointer": pointer,
                                       "official_p_irrelevant": p})
                record["drop_verdicts"] = agreements
                record["official_agrees_with_drops"] = all(
                    (a["official_p_irrelevant"] or 0) >= threshold
                    for a in agreements) if agreements else None
            cases.append(record)
        except Exception as exc:
            failures.append(f"{case['case_id']}: {exc}")
    proposed = [c for c in cases if c["local_drops"]]
    totals = {
        "cases": len(cases),
        "cases_with_drops": len(proposed),
        "official_calls": official_calls,
        "official_input_tokens": input_tokens,
        "baseline_input_tokens": sum(c.get("baseline_input_tokens") or 0
                                     for c in proposed),
        "filtered_input_tokens": sum(c.get("filtered_input_tokens") or 0
                                     for c in proposed),
        "input_tokens_saved": sum(c.get("input_tokens_saved") or 0
                                  for c in proposed),
        "drops_official_agreed": sum(1 for c in proposed
                                     if c.get("official_agrees_with_drops")),
        "drops_official_disagreed": sum(1 for c in proposed
                                        if c.get("official_agrees_with_drops")
                                        is False),
    }
    base_tokens = totals["baseline_input_tokens"]
    if base_tokens:
        totals["savings_pct_on_dropped_cases"] = round(
            100 * totals["input_tokens_saved"] / base_tokens, 2)
    return {
        "schema_version": "nanojev-e2e-official-upstream-v1",
        "status": "pass" if not failures else "fail",
        "scope": "official TypeSafe Jev upstream on public open-corpus data "
                 "only; no private data; bounded by max-cases",
        "official_models": sorted(m for m in official_models if m),
        "local_scorer": "winnow @127.0.0.1:8091", "threshold": threshold,
        "totals": totals, "cases": cases, "failures": failures,
        "ok": not failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path,
                        default=ROOT / "data/open_corpus_v1/manifest.json")
    parser.add_argument("--max-cases", type=int, default=30)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    receipt = run(args.corpus, args.max_cases, offset=args.offset)
    encoded = json.dumps(receipt, ensure_ascii=False, indent=2,
                         sort_keys=True) + "\n"
    args.output.write_text(encoded, encoding="utf-8")
    print(json.dumps({"output": str(args.output),
                      "sha256": sha256_text(encoded), "ok": receipt["ok"],
                      "totals": receipt["totals"]}, indent=2))
    if not receipt["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
