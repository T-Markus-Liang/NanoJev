#!/usr/bin/env python3
"""Mine supplemental DROP-positive candidates for real_context_eval_v1 from the
EXISTING transcripts' own structure (no synthetic generation, no new sources).

Rationale (docs/REAL_CONTEXT_EVAL_RESULTS_V1.md): the v1 set has only 2 drop
labels / 525 scored, so drop-side discrimination cannot be measured. This miner
produces *proposals* — `targets` stay null and each record carries
meta.proposed_label + meta.proposal_note for the owner/agent label pass.
Nothing is written into owner_labels.json / eval.jsonl.

Mining strategies (all local, all inside already-admitted transcripts):

  A) cross_task_reanchor — a transcript contains sequential tasks: an earlier
     task's segments are re-anchored against a LATER user message that starts a
     new task. user_messages_in_order is truncated at that anchor and the
     conversation is cut just before the next user turn, so the emitted state
     is the prefix a context filter would actually have seen mid-new-task.
  B) same_anchor_tail — foreign-task blocks embedded inside a transcript that
     are judged against the transcript's existing final anchor (the two
     existing drop labels seg0025/seg0027 in 133b2c56542e are this pattern).
  C) boilerplate / superseded — repeated identical wait-poll outputs, empty
     search stubs, and stale reads superseded by fresher results.

Windowing mirrors build_real_context_eval_v1: keep candidate + request anchor
(the last user message of the re-anchored user list), then add segments nearest
the candidate while the serialized state fits a conservative character budget
(~7000 packed tokens at the observed worst 0.55 tokens/char); dropped ranges
become {"pointer":"/elided/N","role":"control"} markers, same as the builder.

Usage: python3 scripts/mine_drop_supp_v1.py
Output: data/real_context_eval_v1/candidates_drop_supp_v1.jsonl
        data/real_context_eval_v1/drop_supp_report.json
"""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "build_real_context_eval_v1", ROOT / "scripts" / "build_real_context_eval_v1.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

CANDIDATES = ROOT / "data" / "real_context_eval_v1" / "candidates.jsonl"
OUT_JSONL = ROOT / "data" / "real_context_eval_v1" / "candidates_drop_supp_v1.jsonl"
OUT_REPORT = ROOT / "data" / "real_context_eval_v1" / "drop_supp_report.json"

# conservative stand-in for the builder's 7000 packed-token budget
# (observed worst ratio in candidates.jsonl: 0.545 tokens/char)
STATE_CHAR_BUDGET = 12_000

# ---------------------------------------------------------------------------
# Mining plan: (transcript_hash_prefix, mode, anchor_user_index|None, segs,
#               per-seg overrides {seg: (label, note)}, default_label, default_note)
# anchor_user_index None -> final user message (existing anchor semantics).
# label: True = certainly irrelevant (drop), False = keep, "uncertain".
# ---------------------------------------------------------------------------
Y, N, U = "yes", "no", "uncertain"  # proposed_label per task spec

PLAN = [
    # A foreign camera/orbit-controls audit thread sits inside the cabinet-photo
    # transcript tail; seg25/27 were already labeled drop in v1. Same final
    # anchor (u1: cabinet height/photo fill).
    ("133b2c56542e", "same_anchor_tail", None,
     [19, 30, 31], {},  # segs 21/23/29 are >32KB tool dumps; over state budget
     Y, "mujoco camera/orbit audit thread unrelated to cabinet-photo anchor; extends v1 drops seg25/27"),

    # User explicitly abandons the port-15732/cc-switch investigation ("ok算了")
    # and pivots to autoware planning. All task-A segments judged vs u2 anchor.
    ("987eca2e79b9", "cross_task_reanchor", 2,
     [0, 2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 24, 25, 26], {},
     Y, "port/cc-switch investigation explicitly abandoned by user ('ok算了'); anchor is autoware trajectory-debug task"),

    # World-model project audit (segs 0-49) -> pi0-VLA investigation (u2@50).
    ("61587fdcfb74", "cross_task_reanchor", 2,
     [0, 1, 2, 6, 7, 8, 9, 14, 15, 24, 26, 35, 38, 45, 48, 49],
     {
         1:  (U, "old-task user directive; task B continues inside the same "
               "research project and seg60 reaches back to its findings"),
         6:  (N, "registry read reach-back referenced by seg60 ('比上一轮我看到的状态更新了')"),
         9:  (N, "assistant 'not yet registered' claim is the stale value seg60 compares against"),
         49: (N, "audit report contains the pre-registration claim seg60's comparison needs"),
     },
     Y, "closed world-model audit segments; pi0-VLA anchor does not reach back"),

    # Chit-chat + capability Q&A (segs 0-5) -> mujoco install task (u3@6).
    ("52e3793428b1", "cross_task_reanchor", 3,
     [0, 1, 2, 3, 4, 5], {},
     Y, "greeting/web-capability chit-chat; post-shift tool calls carry the capability fact themselves"),

    # Response-speed chit-chat + model pick (segs 0-7) -> Atlas OS frontend
    # review task (u4@8). u6/a7 form a user decision + commit-ack pair (R2/R6).
    ("2995f1bff456", "cross_task_reanchor", 4,
     [0, 1, 2, 3, 4, 5, 6, 7],
     {
         6: (N, "user turn settling on the faster model — a decision that may carry forward (R2 caution)"),
         7: (N, "assistant acknowledgement committing to the model choice — R6 commit-ack exception"),
     },
     Y, "pre-task chit-chat (greeting/response-speed probe); frontend-review anchor never reaches back"),

    # Visual-servo doc task (segs 0-5) -> AGX Orin LLM deployment task (u3@6,
    # final user; full conversation visible).
    ("8444684933cc", "cross_task_reanchor", None,
     [0, 1, 2, 3, 4, 5], {},
     Y, "closed visual-servo doc-update task; anchor switches to AGX Orin deployment doc, no reach-back"),

    # Same pair structure as 8444684933cc but anchor is u4@8 (docker constraint).
    ("89ac9dc92a80", "cross_task_reanchor", None,
     [0, 1, 2, 3, 4, 5], {},
     Y, "closed visual-servo doc-update task; anchor is AGX deployment + docker constraint"),

    # opencode quota investigation (segs 0-58, mostly empty WebSearch stubs)
    # -> "find mimo free opencode plugin" (u3@59, final user). Final answer
    # (seg80) integrates quota findings, so substantive evidence stays keep;
    # the content-free stubs are drop-eligible.
    ("fc3e19d63e48", "same_anchor_tail", None,
     [3, 5, 13, 19, 30, 34, 46, 56],
     {
         46: (N, "github repo JSON result feeds the seg80 quota/plugin summary"),
         56: (U, "rejected-tool-use control result; interruption semantics unverifiable"),
     },
     # v1 precedent: identical empty stubs (segs 3/5/30) were adjudicated
     # keep under this same anchor -> honest proposal here is keep.
     N, "empty WebSearch boilerplate stub; v1 precedent kept identical stubs under this anchor"),

    # Identical wait-poll boilerplate ("Script running with cell ID 4") —
    # same pattern v1 finalized as uncertain.
    ("0e9a892b6ab9", "same_anchor_tail", None,
     [13, 15, 17], {},
     U, "identical async wait-poll boilerplate; v1 adjudicated this pattern uncertain"),

    # Aborted broad-scope branch (kimi subagent discovery) narrowed by u1 to a
    # single wording fix; empty polls + stale plan ack.
    ("840be7780742", "same_anchor_tail", None,
     [14, 24, 32, 34],
     {
         14: (N, "update_plan ack; v1 precedent kept this segment"),
         24: (N, "provider --help output informed the successful retry path"),
         32: (N, "empty write_stdin poll; v1 precedent kept this segment"),
         34: (N, "empty write_stdin poll; v1 precedent kept identical seg32"),
     },
     U, "aborted-branch boilerplate"),

    # 22 identical empty write_stdin polls of session 77053 in the NDE task.
    ("40b024a75133", "same_anchor_tail", None,
     [113, 115, 117], {},
     U, "identical empty session-poll result; v1 adjudicated wait-poll pattern uncertain"),

]


def norm(text):
    return " ".join(str(text).split())


def render_window(segs, kept):
    conv, dropped, markers = [], 0, 0
    for i, seg in enumerate(segs):
        if i in kept:
            if dropped:
                conv.append({"pointer": f"/elided/{markers}", "role": "control",
                             "content": f"<elided {dropped} earlier segments>"})
                markers += 1
                dropped = 0
            conv.append({"pointer": f"/messages/{i}/content",
                         "role": seg["role"], "content": seg["text"]})
        else:
            dropped += 1
    if dropped:
        conv.append({"pointer": f"/elided/{markers}", "role": "control",
                     "content": f"<elided {dropped} earlier segments>"})
    return conv


def windowed_conversation(segs, candidate_idx, anchor_idx, user_list):
    """Prefix window: keep candidate+anchor, expand nearest-to-candidate while
    the serialized state stays under STATE_CHAR_BUDGET."""
    def state_size(kept):
        conv = render_window(segs, kept)
        return len(builder.serialized({
            "conversation": conv,
            "candidate_pointer": f"/messages/{candidate_idx}/content",
            "user_messages_in_order": user_list}).encode("utf-8"))

    kept = {candidate_idx, anchor_idx}
    if state_size(kept | set(range(len(segs)))) <= STATE_CHAR_BUDGET:
        return render_window(segs, set(range(len(segs)))), False
    estimate = state_size(kept)
    optional = sorted((i for i in range(len(segs)) if i not in kept),
                      key=lambda i: (abs(i - candidate_idx), i))
    # greedy add in distance order; re-check size incrementally on candidates
    for i in optional:
        trial = kept | {i}
        if state_size(trial) <= STATE_CHAR_BUDGET:
            kept = trial
    return render_window(segs, kept), kept != set(range(len(segs)))


def main():
    # load targets + instructions
    existing = [json.loads(l) for l in open(CANDIDATES)]
    want = {r["meta"]["transcript_hash"]: r["meta"]["source_root"] for r in existing}
    existing_ptrs = {(r["meta"]["transcript_hash"], r["meta"]["candidate_pointer"])
                     for r in existing}
    instructions = builder.load_verbatim_instructions(ROOT / "data" / "valen_nano_v3")

    # locate source transcripts by sha256
    paths = {}
    for root, src in [(Path.home() / ".codex" / "sessions", "codex_sessions"),
                      (Path.home() / ".claude" / "projects", "claude_projects")]:
        for p in sorted(root.glob("**/*.jsonl")):
            try:
                h = hashlib.sha256(p.read_bytes()).hexdigest()
            except OSError:
                continue
            if h in want:
                paths[h] = (p, src)
    missing = {h[:12] for h in want} - {h[:12] for h in paths}
    if missing:
        print("missing source transcripts:", sorted(missing))

    records, report_rows, skips = [], [], Counter()
    used_ids = set()
    for hprefix, mode, anchor_uidx, cand_segs, overrides, dlabel, dnote in PLAN:
        th = next((h for h in paths if h.startswith(hprefix)), None)
        if th is None:
            skips["no_source"] += len(cand_segs)
            continue
        path, src = paths[th]
        t = (builder.extract_codex if src == "codex_sessions"
             else builder.extract_claude)(path)
        segs = t.segments
        user_seg_idx = [i for i, s in enumerate(segs) if s["kind"] == "user_turn"]
        if anchor_uidx is None:
            anchor_uidx_eff = len(t.users) - 1
            cut = len(segs)
        else:
            anchor_uidx_eff = anchor_uidx
            if anchor_uidx + 1 < len(user_seg_idx):
                cut = user_seg_idx[anchor_uidx + 1]
            else:
                cut = len(segs)
        if anchor_uidx_eff >= len(t.users):
            skips["bad_anchor"] += len(cand_segs)
            continue
        anchor_seg = user_seg_idx[anchor_uidx_eff]
        user_list = t.users[:anchor_uidx_eff + 1]
        prefix = segs[:cut]

        for idx in cand_segs:
            seg = segs[idx]
            if anchor_uidx is None:
                # existing final anchor: candidate may sit anywhere except the
                # anchor itself (tail blocks come after the last user turn).
                ok = idx != anchor_seg and idx < cut
            else:
                # re-anchored: candidate must be a pre-shift segment.
                ok = idx < anchor_seg
            if not ok:
                skips["candidate_out_of_scope"] += 1
                continue
            if seg["kind"] == "tool_use" or seg["role"] in ("control", "system"):
                skips["bad_kind"] += 1
                continue
            ptr = f"/messages/{idx}/content"
            # Same-anchor records duplicate an existing v1 question verbatim;
            # re-anchored records ask a different question (different
            # user_messages_in_order/state) even at a shared pointer.
            if mode == "same_anchor_tail" and (th, ptr) in existing_ptrs:
                skips["already_in_v1"] += 1
                continue
            conv, windowed = windowed_conversation(prefix, idx, anchor_seg, user_list)
            state = {"conversation": conv,
                     "candidate_pointer": ptr,
                     "user_messages_in_order": user_list}
            # hard guard: drop states that could exceed the compiler's
            # 8192 packed-token limit (~0.55 tok/char worst observed + suffix)
            if len(builder.serialized(state).encode("utf-8")) > 14_000:
                skips["exceeds_budget"] += 1
                continue
            label, note = overrides.get(idx, (dlabel, dnote))
            rid = f"realdrop:{th[:12]}:seg{idx:04d}"
            if rid in used_ids:
                rid = f"{rid}-u{anchor_uidx_eff}"
            used_ids.add(rid)
            records.append({
                "group_id": f"{builder.GROUP_LINEAGE}:{th}",
                "request": {"state": builder.serialized(state),
                            "questions": {"irrelevant": {"type": "noul",
                                                       "instructions": instructions}}},
                "targets": None,
                "meta": {
                    "record_id": rid,
                    "domain": "real",
                    "modality": "text",
                    "language_bucket": builder.language_bucket(t),
                    "source": "real_transcript",
                    "source_dataset": builder.GROUP_LINEAGE,
                    "source_root": src,
                    "transcript_hash": th,
                    "candidate_kind": seg["kind"],
                    "candidate_pointer": ptr,
                    "length_band": builder.length_band(len(segs)),
                    "selection_method": f"mining:{mode}",
                    "supp_source": "mining",
                    "mine_mode": mode,
                    "anchor_user_index": anchor_uidx_eff,
                    "anchor_pointer": f"/messages/{anchor_seg}/content",
                    "windowed": windowed,
                    "state_chars": len(builder.serialized(state).encode("utf-8")),
                    "proposed_label": label,
                    "proposal_note": note,
                },
            })
            report_rows.append({
                "record_id": rid, "transcript": hprefix, "mode": mode,
                "seg": idx, "kind": seg["kind"], "proposed_label": label,
                "note": note,
                "candidate_snippet": norm(seg["text"])[:140],
                "anchor_snippet": norm(t.users[anchor_uidx_eff])[:140],
            })

    label_counts = Counter(str(r["meta"]["proposed_label"]) for r in records)
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in records:
            f.write(builder.serialized(r) + "\n")
    report = {
        "schema_version": "nanojev-real-context-drop-supp-v1",
        "builder": "scripts/mine_drop_supp_v1.py",
        "created_from": "existing candidates.jsonl transcript states / source "
                        "transcripts under ~/.codex/sessions + ~/.claude/projects",
        "note": ("PROPOSALS ONLY: targets stay null; meta.proposed_label + "
                 "proposal_note are the miner's suggested label pending the "
                 "owner/agent label pass. owner_labels.json and eval.jsonl "
                 "were not modified. Labels follow "
                 "docs/REAL_CONTEXT_LABELING_GUIDE_V1.md asymmetric semantics "
                 "(yes == certainly irrelevant == drop)."),
        "mining_strategies": {
            "cross_task_reanchor": "earlier-task segments re-anchored at the "
                "first user message of the next task; conversation cut before "
                "the following user turn; user_messages_in_order truncated at "
                "the anchor",
            "same_anchor_tail": "foreign-task/boilerplate blocks judged under "
                "the transcript's existing final anchor",
        },
        "counts": {
            "records": len(records),
            "proposed_labels": dict(label_counts),
            "by_transcript": dict(Counter(r["meta"]["transcript_hash"][:12]
                                          for r in records)),
            "by_kind": dict(Counter(r["meta"]["candidate_kind"] for r in records)),
            "windowed": sum(1 for r in records if r["meta"]["windowed"]),
            "skips": dict(skips),
        },
        "rows": report_rows,
    }
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"wrote {len(records)} records -> {OUT_JSONL}")
    print("labels:", dict(label_counts), "skips:", dict(skips))
    print(f"report -> {OUT_REPORT}")


if __name__ == "__main__":
    sys.exit(main())
