#!/usr/bin/env python3
"""Pure-python validation for the v5 F4 outcome-positive miner
(scripts/mine_v5_outcome_positive_v1.py ->
data/v5_mining/f4_outcome_positive_candidates.jsonl + f4_report.json +
f4_review.md + pool_split.json).

Two layers, no model calls, no network, no tokenizer:

  A) Synthetic-fixture linker tests — planted link structures in fabricated
     transcripts (same segment dict shape the builder extractors emit):
       * file_read_edit fires on Read(P) result -> Edit(P) in the same task
         span, and does NOT fire across a user-turn boundary or beyond
         LINK_MAX_GAP;
       * command_output_echo fires only when a >=40-char distinctive needle
         recurs verbatim in a LATER assistant-authored segment — not on
         identical boilerplate (dedup df guard), not on sub-40-char lines,
         not on earlier-only occurrences (directionality);
       * test_failure_fix fires on failing verification output followed by
         a same-span fix edit of a path named in the output;
       * unused_drop_control proposes 'yes' only when the reach-back
         battery is empty AND the task moved on; a referenced result never
         becomes a control.
  B) Emitted-file validation (skipped gracefully if outputs are absent):
     schema shape, v5f4 record ids, supp_source=outcome_linker,
     evidence_basis dict, pointer resolution, anchor = last user turn,
     component-level credential scan, intra-set dedup, report consistency,
     and pool disjointness vs the shared ledger / eval-v1's 60.

Usage: python3 scripts/test_mine_v5_outcome_v1.py  (exit 0 = all checks pass)
"""

import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "build_real_context_eval_v1", ROOT / "scripts" / "build_real_context_eval_v1.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
spec2 = importlib.util.spec_from_file_location(
    "mine_v5_outcome_positive_v1",
    ROOT / "scripts" / "mine_v5_outcome_positive_v1.py")
mine = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(mine)

CANDIDATES = ROOT / "data" / "v5_mining" / "f4_outcome_positive_candidates.jsonl"
REPORT = ROOT / "data" / "v5_mining" / "f4_report.json"
LEDGER = ROOT / "data" / "v5_mining" / "pool_split.json"
F2_REPORT = ROOT / "data" / "v5_mining" / "f2_report.json"
EVAL_V1 = ROOT / "data" / "real_context_eval_v1" / "candidates.jsonl"
MAX_CANDIDATE_BYTES = 32 * 1024
VALID_LABELS = {"yes", "no", "uncertain"}
VALID_STRATEGIES = {"file_read_edit", "test_failure_fix",
                    "command_output_echo", "unused_drop_control"}

failures = []
checks = Counter()


def check(ok, label, detail=""):
    checks[label] += 1
    if not ok:
        failures.append(f"[{label}] {detail}")


def seg(role, kind, text):
    return {"role": role, "kind": kind, "text": text}


def tu(name, inp):
    return seg("assistant", "tool_use",
               json.dumps({"tool_use": {"name": name, "input": inp}}))


def mk_transcript(segments):
    t = SimpleNamespace(segments=[], users=[], text_bytes=0)
    for s in segments:
        t.segments.append(s)
        if s["kind"] == "user_turn":
            t.users.append(s["text"])
        t.text_bytes += len(s["text"].encode("utf-8", errors="ignore"))
    return t


def mk_item(t, th):
    flagged = {i for i, s in enumerate(t.segments)
               if builder.credential_scan(s["text"])}
    flagged_users = {j for j, u in enumerate(t.users)
                     if builder.credential_scan(u)}
    return {"transcript_hash": th, "source": "claude_projects",
            "path": "<synthetic>", "transcript": t,
            "nseg": len(t.segments), "nusers": len(t.users),
            "flagged": flagged,
            "first_flagged_user": (min(flagged_users) if flagged_users
                                   else len(t.users))}


def props_for(segments, th):
    item = mk_item(mk_transcript(segments), th)
    skips = Counter()
    return mine.mine_transcript(item, skips), skips


def by_seg(props):
    return {p["seg"]: p for p in props}


# --- A) synthetic-fixture linker tests -------------------------------------------

def test_file_read_edit():
    # Read(src/alpha.py) result -> Edit(src/alpha.py) same task span: keep.
    props, _ = props_for([
        seg("user", "user_turn", "please update src/alpha.py to add logging"),
        tu("Read", {"file_path": "src/alpha.py"}),
        seg("tool", "tool_result",
            "def alpha():\n    return 1  # current body of the module"),
        seg("assistant", "assistant_text",
            "I see src/alpha.py needs a logging import"),
        tu("Edit", {"file_path": "src/alpha.py",
                    "old_string": "return 1", "new_string": "return 2"}),
        seg("tool", "tool_result", "File src/alpha.py updated"),
        seg("user", "user_turn", "thanks, now do something else"),
    ], "a" * 64)
    p = by_seg(props).get(2)
    check(p is not None and p["strategy"] == "file_read_edit"
          and p["label"] == "no" and p["usage_seg"] == 4,
          "linker_file_edit", f"expected keep file_read_edit on seg2, got {p}")

    # Same but the Edit happens AFTER a new user task -> must not link.
    props, _ = props_for([
        seg("user", "user_turn", "task one: read src/beta.py please"),
        tu("Read", {"file_path": "src/beta.py"}),
        seg("tool", "tool_result", "beta contents: small module body here"),
        seg("user", "user_turn", "task two: rewrite beta from scratch now"),
        tu("Edit", {"file_path": "src/beta.py",
                    "old_string": "x", "new_string": "y"}),
        seg("tool", "tool_result", "File src/beta.py updated"),
    ], "b" * 64)
    p = by_seg(props).get(2)
    check(p is None or p["strategy"] != "file_read_edit",
          "linker_no_cross_task",
          f"cross-task edit must not link, got {p}")
    check(p is None or p["strategy"] != "unused_drop_control",
          "linker_no_cross_task",
          "cross-task-referenced result must not be a drop control")

    # Same but the Edit is beyond LINK_MAX_GAP -> must not link.
    filler = [seg("assistant", "assistant_text",
                  f"working note number {i} with no special content")
              for i in range(mine.LINK_MAX_GAP + 4)]
    props, _ = props_for(
        [seg("user", "user_turn", "read src/gamma.py then long work"),
         tu("Read", {"file_path": "src/gamma.py"}),
         seg("tool", "tool_result", "gamma body here")]
        + filler
        + [tu("Edit", {"file_path": "src/gamma.py",
                       "old_string": "x", "new_string": "y"}),
           seg("tool", "tool_result", "File src/gamma.py updated")],
        "c" * 64)
    p = by_seg(props).get(2)
    check(p is None or p["strategy"] != "file_read_edit",
          "linker_gap_bound",
          f"edit beyond LINK_MAX_GAP must not link, got {p}")


def test_echo():
    line = ("Processed 1287 rows from dataset/train_split_v3.parquet "
            "in 42.1 seconds")
    # unique distinctive >=40-char line quoted verbatim later -> keep.
    props, _ = props_for([
        seg("user", "user_turn", "run the pipeline please"),
        tu("Bash", "python run_pipeline.py --dataset x"),
        seg("tool", "tool_result",
            line + "\nstatus: all rows validated ok"),
        seg("assistant", "assistant_text",
            f"Done: {line} — results are clean."),
    ], "d" * 64)
    p = by_seg(props).get(2)
    check(p is not None and p["strategy"] == "command_output_echo"
          and p["label"] == "no" and p["usage_seg"] == 3,
          "linker_echo", f"expected echo keep on seg2, got {p}")

    # identical boilerplate: the same >=40 distinctive line in 3 tool
    # results -> df guard makes the needle inadmissible even though a later
    # assistant message quotes it.
    boiler = ("status checkpoint worker-7 heartbeat seq 0042 all systems "
              "nominal and fully operational")
    props, _ = props_for([
        seg("user", "user_turn", "poll the workers three times"),
        tu("Bash", "check worker-7"),
        seg("tool", "tool_result", boiler),
        tu("Bash", "check worker-7 again"),
        seg("tool", "tool_result", boiler),
        tu("Bash", "final worker-7 check"),
        seg("tool", "tool_result", boiler),
        seg("assistant", "assistant_text",
            f"last poll still says {boiler} — wrapping up now"),
    ], "e" * 64)
    echoes = [p for p in props if p["strategy"] == "command_output_echo"]
    check(not echoes, "linker_boilerplate_df",
          f"identical boilerplate must not link, got {echoes}")

    # a 39-char distinctive line does not meet the >=40 threshold.
    short39 = "job#77123 build warnings 0 errors 0 okk"  # len == 39
    assert len(short39) == 39, len(short39)
    props, _ = props_for([
        seg("user", "user_turn", "check the build output"),
        tu("Bash", "make build"),
        seg("tool", "tool_result", short39),
        seg("assistant", "assistant_text",
            f"build summary: {short39} — nothing to fix"),
    ], "f" * 64)
    strong = [p for p in props if p["label"] == "no"]
    check(not strong, "linker_threshold_40",
          f"sub-40-char echo must not produce keep, got {strong}")

    # directionality: the needle appears only BEFORE the result, never
    # after -> no link.
    earlier = ("quotas_refresh_interval_seconds=1800 default_scope="
               "team_wide_region")
    props, _ = props_for([
        seg("user", "user_turn", "show me the config values"),
        seg("assistant", "assistant_text",
            f"I recall the line: {earlier}"),
        tu("Bash", "cat app.conf"),
        seg("tool", "tool_result",
            earlier + "\nlog_level=verbose\nregion=east1"),
    ], "9" * 64)
    echoes = [p for p in props if p["strategy"] == "command_output_echo"
              and p["label"] == "no"]
    check(not echoes, "linker_directionality",
          f"earlier-only occurrence must not link, got {echoes}")


def test_failure_fix():
    props, _ = props_for([
        seg("user", "user_turn", "run the tests and fix any failure"),
        tu("Bash", "pytest tests/test_alpha.py -x"),
        seg("tool", "tool_result",
            "FAILED tests/test_alpha.py::test_counter - assert 2 == 3\n"
            "  File \"src/alpha.py\", line 42, in increment\n"
            "1 failed, 0 passed in 0.42s"),
        seg("assistant", "assistant_text",
            "The failure is in the counter increment; fixing src/alpha.py"),
        tu("Edit", {"file_path": "src/alpha.py",
                    "old_string": "n += 2", "new_string": "n += 1"}),
        seg("tool", "tool_result", "File src/alpha.py updated"),
        tu("Bash", "pytest tests/test_alpha.py -x"),
        seg("tool", "tool_result", "1 passed in 0.10s"),
    ], "8" * 64)
    p = by_seg(props).get(2)
    check(p is not None and p["strategy"] == "test_failure_fix"
          and p["label"] == "no",
          "linker_test_fix", f"expected test_failure_fix keep, got {p}")
    if p is not None:
        check(p["evidence"].get("link") == "fix_edit_after_failure",
              "linker_test_fix", f"evidence link {p['evidence']}")


def test_drop_control():
    listing = ("omega_prd_bkup_7741.tmp\nquartz_jrn_old_3392.log\n"
               "zebra_cfg_stale_6610.bak size 3392 bytes modified today\n"
               "umbra_dat_legacy_5502.csv\ntally_arc_retired_8820.tgz\n"
               "fennel_idx_dead_4417.db extra padding here")
    assert len(" ".join(listing.split())) >= 120
    # unused substantive result + later user turn -> control 'yes'.
    props, _ = props_for([
        seg("user", "user_turn", "list the scratch dir for me"),
        tu("Bash", "ls /tmp/scratch_8842"),
        seg("tool", "tool_result", listing),
        seg("assistant", "assistant_text",
            "scratch dir contents noted; nothing needed from it"),
        seg("user", "user_turn", "ok next: write the weekly report"),
        seg("assistant", "assistant_text", "drafting the report now"),
    ], "7" * 64)
    p = by_seg(props).get(2)
    check(p is not None and p["strategy"] == "unused_drop_control"
          and p["label"] == "yes",
          "linker_drop_control", f"expected drop control, got {p}")
    if p is not None:
        eb = p["evidence"]
        check(eb.get("signal") == "unused_drop_control"
              and not eb.get("path_rereferenced")
              and not eb.get("signature_echoed")
              and not eb.get("distinctive_echo"),
              "linker_drop_control", f"evidence_basis {eb}")

    # same fixture but a later assistant quotes a distinctive line ->
    # referenced -> no control (the echo keep fires instead).
    echo_line = "zebra_cfg_stale_6610.bak size 3392 bytes modified today"
    props, _ = props_for([
        seg("user", "user_turn", "list the scratch dir for me"),
        tu("Bash", "ls /tmp/scratch_8842"),
        seg("tool", "tool_result", listing),
        seg("assistant", "assistant_text",
            f"interesting entry: {echo_line} — flagging it"),
        seg("user", "user_turn", "ok next: write the weekly report"),
        seg("assistant", "assistant_text", "drafting the report now"),
    ], "6" * 64)
    p = by_seg(props).get(2)
    check(p is None or p["strategy"] != "unused_drop_control",
          "linker_no_false_drop",
          f"referenced result must not be a drop control, got {p}")
    check(p is not None and p["label"] == "no",
          "linker_no_false_drop",
          f"echo-referenced result should be a keep, got {p}")


def test_emit_schema():
    instructions = builder.load_verbatim_instructions(
        ROOT / "data" / "valen_nano_v3")
    line = ("Processed 1287 rows from dataset/train_split_v3.parquet "
            "in 42.1 seconds")
    t = mk_transcript([
        seg("user", "user_turn", "run the pipeline please"),
        tu("Bash", "python run_pipeline.py --dataset x"),
        seg("tool", "tool_result", line + "\nstatus: validated ok"),
        seg("assistant", "assistant_text", f"Done: {line}"),
    ])
    item = mk_item(t, "abcd" * 16)
    props = mine.mine_transcript(item, Counter())
    check(len(props) >= 1, "emit_fixture", f"expected a prop, got {props}")
    if not props:
        return
    rec = mine.emit(item, props[0], instructions, {}, set(), Counter())
    check(rec is not None, "emit_fixture", "emit returned None")
    if rec is None:
        return
    meta = rec["meta"]
    check(meta["record_id"].startswith("v5f4:"), "emit_fixture",
          meta["record_id"])
    check(meta["supp_source"] == "outcome_linker", "emit_fixture",
          meta["supp_source"])
    check(rec["targets"] is None, "emit_fixture", "targets not null")
    check(isinstance(meta["evidence_basis"], dict)
          and meta["evidence_basis"].get("signal"), "emit_fixture",
          "evidence_basis missing")
    state = json.loads(rec["request"]["state"])
    ptrs = {s["pointer"] for s in state["conversation"]}
    check(state["candidate_pointer"] in ptrs, "emit_fixture",
          "candidate pointer unresolved")
    q = rec["request"]["questions"]["irrelevant"]
    check(q["type"] == "noul", "emit_fixture", q["type"])
    check(hashlib.sha256(q["instructions"].encode()).hexdigest()
          == hashlib.sha256(instructions.encode()).hexdigest(),
          "emit_fixture", "instructions not verbatim")


# --- B) emitted-file validation ---------------------------------------------------

def test_emitted_files():
    if not (CANDIDATES.is_file() and REPORT.is_file()):
        check(True, "file_validation", "outputs absent — skipped")
        return
    records = [json.loads(l) for l in
               CANDIDATES.read_text(encoding="utf-8").splitlines()
               if l.strip()]
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    instructions = builder.load_verbatim_instructions(
        ROOT / "data" / "valen_nano_v3")
    instr_sha = hashlib.sha256(instructions.encode("utf-8")).hexdigest()

    eval_hashes = {json.loads(l)["group_id"].split(":", 1)[1]
                   for l in EVAL_V1.read_text(encoding="utf-8").splitlines()
                   if l.strip()}
    if LEDGER.is_file():
        led = json.loads(LEDGER.read_text(encoding="utf-8"))
        train_hashes = set(led["train_pool_hashes"])
        reserved_hashes = set(led["eval_pool_hashes"])
        check(not (train_hashes & reserved_hashes), "pool_ledger",
              "canonical ledger train/eval overlap")
    else:
        rep = json.loads(F2_REPORT.read_text(encoding="utf-8"))
        ps = rep["pool_split"]
        train_hashes = {e["transcript_hash"]
                        for e in ps["train_pool"]["transcripts"]}
        reserved_hashes = {e["transcript_hash"]
                           for e in ps["v5_eval_reserved"]["transcripts"]}

    seen_ids, seen_states = set(), set()
    per_strategy, per_source, per_label = Counter(), Counter(), Counter()
    for r in records:
        meta = r.get("meta", {})
        rid = meta.get("record_id", "?")
        check(set(r.keys()) >= {"group_id", "request", "targets", "meta"},
              "schema", f"{rid}: missing keys")
        check(r.get("targets") is None, "schema", f"{rid}: targets")
        check(rid.startswith("v5f4:"), "schema", f"{rid}: prefix")
        check(rid not in seen_ids, "schema", f"{rid}: dup id")
        seen_ids.add(rid)
        check(meta.get("supp_source") == "outcome_linker", "schema",
              f"{rid}: supp_source")
        check(meta.get("proposed_label") in VALID_LABELS, "schema",
              f"{rid}: label {meta.get('proposed_label')}")
        check(meta.get("strategy") in VALID_STRATEGIES, "schema",
              f"{rid}: strategy {meta.get('strategy')}")
        check(meta.get("mining_side") in ("keep", "drop"), "schema",
              f"{rid}: side")
        check(isinstance(meta.get("evidence_basis"), dict)
              and meta["evidence_basis"].get("signal"), "schema",
              f"{rid}: evidence_basis")
        check("pair_id" in meta, "schema", f"{rid}: pair_id field")
        check(meta.get("state_chars", 0) <= 14_000, "schema",
              f"{rid}: state_chars")
        check(meta.get("pool") == "train", "pool", f"{rid}: pool")
        per_strategy[meta.get("strategy")] += 1
        per_source[meta.get("source_root")] += 1
        per_label[meta.get("proposed_label")] += 1
        if meta.get("mining_side") == "keep":
            check(isinstance(meta.get("usage_seg"), int)
                  and meta["usage_seg"] > meta.get("candidate_seg", -1),
                  "link", f"{rid}: usage_seg not after candidate")
            check(meta["proposed_label"] != "yes", "link",
                  f"{rid}: keep-side 'yes'")

        q = (r["request"].get("questions") or {}).get("irrelevant") or {}
        check(q.get("type") == "noul", "schema", f"{rid}: q type")
        check(hashlib.sha256(
            q.get("instructions", "").encode()).hexdigest() == instr_sha,
            "schema", f"{rid}: instructions")
        try:
            state = json.loads(r["request"]["state"])
        except (ValueError, KeyError, TypeError):
            check(False, "schema", f"{rid}: unparseable state")
            continue
        conv = state.get("conversation") or []
        users = state.get("user_messages_in_order") or []
        ptr = state.get("candidate_pointer")
        shash = builder.normalized_state_hash(r)
        check(shash not in seen_states, "dedup", f"{rid}: dup state")
        seen_states.add(shash)
        ptrs = [s.get("pointer") for s in conv]
        check(ptr in ptrs, "pointer", f"{rid}: unresolved ptr")
        check(len(ptrs) == len(set(ptrs)), "pointer", f"{rid}: dup ptrs")
        cand = next((s for s in conv if s.get("pointer") == ptr), None)
        if cand is not None:
            check(bool(cand.get("content", "").strip()), "pointer",
                  f"{rid}: empty candidate")
            check(len(cand["content"].encode("utf-8"))
                  <= MAX_CANDIDATE_BYTES, "pointer", f"{rid}: over 32KB")
            check(cand.get("role") != "control", "pointer",
                  f"{rid}: control candidate")
        user_segs = [s for s in conv if s.get("role") == "user"]
        check(bool(users) and bool(user_segs), "anchor",
              f"{rid}: no users")
        if user_segs and users:
            check(user_segs[-1].get("pointer")
                  == meta.get("anchor_pointer"), "anchor",
                  f"{rid}: anchor mismatch")
            check(user_segs[-1].get("content") == users[-1], "anchor",
                  f"{rid}: anchor content mismatch")
        for s in conv:
            hit = builder.credential_scan(s.get("content", ""))
            check(hit is None, "credential", f"{rid}: {hit}")
        for u in users:
            hit = builder.credential_scan(u)
            check(hit is None, "credential", f"{rid}: user {hit}")
        th = meta.get("transcript_hash")
        check(th not in eval_hashes, "pool", f"{rid}: eval-v1 transcript")
        check(th not in reserved_hashes, "pool",
              f"{rid}: reserved transcript")
        check(th in train_hashes, "pool", f"{rid}: not in train ledger")
        check(r["group_id"] == f"real_context_v5:{th}", "pool",
              f"{rid}: group_id mismatch")

    c = report["counts"]
    check(c["records"] == len(records), "report", "record count")
    check(dict(per_strategy) == c["by_signal"], "report",
          f"signal counts {dict(per_strategy)} vs {c['by_signal']}")
    check(dict(per_source) == c["by_source"], "report", "source counts")
    check(dict(sorted(per_label.items())) == c["proposed_labels"],
          "report", "label counts")
    check(report["self_check"]["status"] == "pass", "report",
          "self_check")
    check(report["seed"] == 20261201, "report", "seed")
    check((ROOT / "data" / "v5_mining" / "f4_review.md").is_file(),
          "report", "f4_review.md missing")
    # keep-side dominance: F4 is a keep family; drop rows must be controls
    drop_rows = [r for r in records
                 if r["meta"]["proposed_label"] == "yes"]
    check(all(r["meta"]["strategy"] == "unused_drop_control"
              for r in drop_rows), "report",
          "'yes' rows from non-control strategy")


def main():
    test_file_read_edit()
    test_echo()
    test_failure_fix()
    test_drop_control()
    test_emit_schema()
    test_emitted_files()
    total = sum(checks.values())
    print(f"{total} checks, {len(failures)} failures")
    for f in failures[:30]:
        print("  FAIL", f)
    if failures:
        print(f"... {len(failures)} total failures")
        return 1
    print("all checks pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
