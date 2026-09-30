#!/usr/bin/env python3
"""Pure-python validation for the v5 F2 hard-negative mining output
(data/v5_mining/f2_hard_negative_candidates.jsonl + f2_report.json).

No model calls, no network, no tokenizer — stdlib + the
build_real_context_eval_v1 helpers only (credential scan / hashing / state
normalization). Covers:

  1. schema validity   — candidates.jsonl-compatible record shape, verbatim
                         instructions, targets null, v5f2 record_ids
  2. pointer resolution — every candidate_pointer resolves inside the emitted
                         conversation; candidate content non-empty, <=32KB;
                         anchor is the LAST user turn of the emitted state
  3. pool disjointness — no mined transcript is one of the 60 eval-v1
                         transcripts, none is in the v5-eval reserve, and the
                         report's three pool ledgers are pairwise disjoint
  4. credential scan   — no SECRET_PATTERNS hit in any emitted conversation
                         segment or user message (component-level scan; the
                         serialized blob is NOT scanned because JSON escaping
                         can fabricate hits, e.g. "<tab>@x.y" -> "\\t@x.y")
  5. pair integrity    — pair_id members are minimal-difference
                         opposite-label records (anchor_flip: same candidate,
                         different anchors; same_anchor: same anchor,
                         opposite labels)
  6. report consistency — counts in f2_report.json match the emitted file

Usage: python3 scripts/test_mine_v5_v1.py  (exit 0 = all checks pass)
"""

import hashlib
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "build_real_context_eval_v1", ROOT / "scripts" / "build_real_context_eval_v1.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

CANDIDATES = ROOT / "data" / "v5_mining" / "f2_hard_negative_candidates.jsonl"
REPORT = ROOT / "data" / "v5_mining" / "f2_report.json"
EVAL_V1 = ROOT / "data" / "real_context_eval_v1" / "candidates.jsonl"
MAX_CANDIDATE_BYTES = 32 * 1024
STATE_CHAR_HARD = 14_000
VALID_LABELS = {"yes", "no", "uncertain"}
VALID_STRATEGIES = {"cross_task_reanchor", "stale_superseded",
                    "same_anchor_tail"}

failures = []
checks = Counter()


def check(ok, label, detail=""):
    checks[label] += 1
    if not ok:
        failures.append(f"[{label}] {detail}")


def load_jsonl(path):
    return [json.loads(l) for l in
            Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    records = load_jsonl(CANDIDATES)
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    eval_hashes = {json.loads(l)["group_id"].split(":", 1)[1]
                   for l in EVAL_V1.read_text(encoding="utf-8").splitlines()
                   if l.strip()}
    instructions = builder.load_verbatim_instructions(
        ROOT / "data" / "valen_nano_v3")
    instr_sha = hashlib.sha256(instructions.encode("utf-8")).hexdigest()

    reserved_hashes = {e["transcript_hash"] for e in
                       report["pool_split"]["v5_eval_reserved"]["transcripts"]}
    train_hashes = {e["transcript_hash"] for e in
                    report["pool_split"]["train_pool"]["transcripts"]}
    frozen_hashes = set(report["pool_split"]["eval_v1_frozen"]
                        ["transcript_hashes"])

    # --- pool ledger disjointness -------------------------------------------
    check(eval_hashes == frozen_hashes, "pool_ledger",
          f"eval-v1 frozen ledger mismatch: {len(eval_hashes)} vs "
          f"{len(frozen_hashes)}")
    check(not (reserved_hashes & frozen_hashes), "pool_ledger",
          "reserved pool intersects eval-v1 frozen")
    check(not (reserved_hashes & train_hashes), "pool_ledger",
          "reserved pool intersects train pool")
    check(not (train_hashes & frozen_hashes), "pool_ledger",
          "train pool intersects eval-v1 frozen")

    seen_ids, seen_requests, seen_states = set(), set(), set()
    pair_members = defaultdict(list)
    per_strategy, per_source = Counter(), Counter()

    for r in records:
        meta = r.get("meta", {})
        rid = meta.get("record_id", "?")

        # --- schema -----------------------------------------------------------
        check(set(r.keys()) >= {"group_id", "request", "targets", "meta"},
              "schema", f"{rid}: missing top-level keys")
        check(isinstance(r.get("group_id"), str)
              and r["group_id"].startswith("real_context_v5:"),
              "schema", f"{rid}: bad group_id {r.get('group_id')}")
        check(r.get("targets") is None, "schema",
              f"{rid}: targets must be null (proposals only)")
        check(rid.startswith("v5f2:"), "schema",
              f"{rid}: bad record_id prefix")
        check(rid not in seen_ids, "schema", f"{rid}: duplicate record_id")
        seen_ids.add(rid)
        check(meta.get("supp_source") == "mining", "schema",
              f"{rid}: supp_source={meta.get('supp_source')}")
        check(meta.get("proposed_label") in VALID_LABELS, "schema",
              f"{rid}: bad proposed_label {meta.get('proposed_label')}")
        check(isinstance(meta.get("proposal_note"), str)
              and meta["proposal_note"], "schema",
              f"{rid}: missing proposal_note")
        check(meta.get("strategy") in VALID_STRATEGIES, "schema",
              f"{rid}: bad strategy {meta.get('strategy')}")
        check(meta.get("mining_side") in ("keep", "drop"), "schema",
              f"{rid}: bad mining_side {meta.get('mining_side')}")
        check("pair_id" in meta, "schema", f"{rid}: missing pair_id field")
        check(meta.get("state_chars", 0) <= STATE_CHAR_HARD, "schema",
              f"{rid}: state_chars over hard cap")
        per_strategy[meta.get("strategy")] += 1
        per_source[meta.get("source_root")] += 1

        req = r.get("request", {})
        q = (req.get("questions") or {}).get("irrelevant") or {}
        check(q.get("type") == "noul", "schema", f"{rid}: q type {q.get('type')}")
        check(hashlib.sha256(q.get("instructions", "").encode()).hexdigest()
              == instr_sha, "schema", f"{rid}: instructions not verbatim")

        try:
            state = json.loads(req["state"])
        except (ValueError, KeyError, TypeError):
            check(False, "schema", f"{rid}: unparseable request.state")
            continue
        conv = state.get("conversation") or []
        users = state.get("user_messages_in_order") or []
        ptr = state.get("candidate_pointer")

        # --- dedup --------------------------------------------------------------
        rhash = builder.sha256_bytes(
            builder.serialized(req).encode("utf-8"))
        shash = builder.normalized_state_hash(r)
        check(rhash not in seen_requests, "dedup", f"{rid}: dup request hash")
        check(shash not in seen_states, "dedup", f"{rid}: dup state hash")
        seen_requests.add(rhash)
        seen_states.add(shash)

        # --- pointer resolution --------------------------------------------------
        ptrs = [s.get("pointer") for s in conv]
        check(ptr in ptrs, "pointer", f"{rid}: {ptr} not in conversation")
        check(len(ptrs) == len(set(ptrs)), "pointer",
              f"{rid}: duplicate conv pointers")
        cand = next((s for s in conv if s.get("pointer") == ptr), None)
        if cand is not None:
            check(bool(cand.get("content", "").strip()), "pointer",
                  f"{rid}: empty candidate content")
            check(len(cand["content"].encode("utf-8"))
                  <= MAX_CANDIDATE_BYTES, "pointer",
                  f"{rid}: candidate over 32KB")
            check(cand.get("role") != "control", "pointer",
                  f"{rid}: candidate is a control segment")
        check(ptr == meta.get("candidate_pointer"), "pointer",
              f"{rid}: state pointer {ptr} != meta {meta.get('candidate_pointer')}")

        # --- anchor is the last user turn --------------------------------------
        check(bool(users), "anchor", f"{rid}: empty user_messages_in_order")
        user_segs = [s for s in conv if s.get("role") == "user"]
        check(bool(user_segs), "anchor", f"{rid}: no user segment in conv")
        if user_segs and users:
            check(user_segs[-1].get("pointer")
                  == meta.get("anchor_pointer"), "anchor",
                  f"{rid}: last user seg {user_segs[-1].get('pointer')} != "
                  f"anchor {meta.get('anchor_pointer')}")
            check(user_segs[-1].get("content") == users[-1], "anchor",
                  f"{rid}: last user seg content != user_messages[-1]")

        # --- credential scan (component level) -----------------------------------
        for s in conv:
            hit = builder.credential_scan(s.get("content", ""))
            check(hit is None, "credential",
                  f"{rid}: {hit} in conv {s.get('pointer')}")
        for u in users:
            hit = builder.credential_scan(u)
            check(hit is None, "credential",
                  f"{rid}: {hit} in user_messages_in_order")

        # --- pool disjointness ----------------------------------------------------
        th = meta.get("transcript_hash")
        check(th not in eval_hashes, "pool_disjoint",
              f"{rid}: transcript in eval-v1 frozen pool")
        check(th not in reserved_hashes, "pool_disjoint",
              f"{rid}: transcript in v5-eval reserved pool")
        check(th in train_hashes, "pool_disjoint",
              f"{rid}: transcript not in reported train pool")
        check(meta.get("pool") == "train", "pool_disjoint",
              f"{rid}: meta.pool={meta.get('pool')}")
        check(r["group_id"] == f"real_context_v5:{th}", "pool_disjoint",
              f"{rid}: group_id/transcript_hash mismatch")

        if meta.get("pair_id"):
            pair_members[meta["pair_id"]].append((meta, rid))

    # --- pair integrity ---------------------------------------------------------
    for pid, members in sorted(pair_members.items()):
        kinds = {m[0].get("pair_kind") for m in members}
        check(len(kinds) == 1, "pairs", f"{pid}: mixed pair_kind {kinds}")
        labels = {m[0]["proposed_label"] for m in members}
        check(labels <= {"yes", "no"} and len(labels) == 2, "pairs",
              f"{pid}: labels {sorted(labels)} not opposite yes/no")
        if kinds == {"anchor_flip"}:
            check(len({m[0]["candidate_pointer"] for m in members}) == 1,
                  "pairs", f"{pid}: flip pair with different candidates")
            check(len({m[0]["anchor_pointer"] for m in members}) >= 2,
                  "pairs", f"{pid}: flip pair with same anchor")
        elif kinds == {"same_anchor"}:
            check(len({m[0]["anchor_pointer"] for m in members}) == 1,
                  "pairs", f"{pid}: same-anchor pair with different anchors")

    # --- report consistency -------------------------------------------------------
    c = report["counts"]
    check(c["records"] == len(records), "report",
          f"report records {c['records']} != {len(records)}")
    check(dict(per_strategy) == c["by_strategy"], "report",
          f"strategy counts mismatch: {dict(per_strategy)} vs {c['by_strategy']}")
    check(dict(per_source) == c["by_source"], "report",
          f"source counts mismatch")
    check(report["self_check"]["status"] == "pass", "report",
          "miner self_check did not pass")
    check(report["seed"] == 20261201, "report", "wrong seed in report")

    # --- summary --------------------------------------------------------------------
    total = sum(checks.values())
    print(f"{total} checks, {len(failures)} failures "
          f"({len(records)} records, {len(pair_members)} pairs)")
    for f in failures[:25]:
        print("  FAIL", f)
    if failures:
        print(f"... {len(failures)} total failures")
        return 1
    print("all checks pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
