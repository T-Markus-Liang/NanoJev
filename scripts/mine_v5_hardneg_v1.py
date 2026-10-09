#!/usr/bin/env python3
"""V5 F2 hard-negative miner: generalize mine_drop_supp_v1.py from a
hand-written PLAN to an automatic miner over the whole local transcript pool.

Implements docs/V5_DATA_DESIGN_V1.md §3 (family F2, exit_hard_negative):

  * Sources: the SAME roots as build_real_context_eval_v1.py
    (~/.claude/projects + ~/.codex/sessions). All transcripts meeting the
    builder's minimum quality thresholds are mined — min_segments,
    min_user_messages, byte cap, credential scan, duplicate-source — with NO
    admission quota and NO >=9-candidates requirement. The 60 eval-admitted
    transcripts are also mined (task directive: eval-adjacent transcripts'
    OTHER segments are usable) but the 590 frozen record_ids are protected at
    the (transcript_hash, candidate_pointer, anchor_pointer) pair level —
    colliding pairs are skipped and every emitted row records
    meta.holdout_collision_check. NOTE: this deviates from the doc's §1.6
    blanket ban on the 60 eval transcripts for train rows; the task directive
    replaces it with pair-level exclusion. Flagged in the report caveats.

  * Strategies (all deterministic, no model/service calls):
      - cross_task_reanchor: every inter-user span (s_{a-1}, s_a) is reanchored
        against user turn `a` (the shift boundary), plus a deep-reanchor pass
        judging earlier spans against the final user request. Conversation is
        cut before the next user turn and user_messages_in_order truncated at
        the anchor — same state shape as v1.
      - stale_read_edit: Read/file-listing tool_result superseded by a later
        Edit + re-Read of the same path inside one task span (R4 stale rule:
        droppable only when superseded AND unreferenced post-shift). Anchored
        at the next user turn after the task span; last-task chains get the
        final anchor but are forced uncertain (post-shift is unverifiable).
      - same_anchor_tail: foreign-task/boilerplate under the final anchor —
        digit-normalized duplicate boilerplate groups and coherent foreign
        thread runs in the post-final-user tail.

  * Proposed labels via deterministic rules mirroring the v1 adjudication
    outcome (docs/REAL_CONTEXT_LABELING_GUIDE_V1.md R2-R7):
      - explicit task-abandon markers (算了/换/先不/never mind/switch to/...)
        in the anchor strengthen cross_task "yes";
      - subject-vocabulary continuity (shared file paths / identifiers
        between the candidate and anchor-or-post-anchor content) downgrades
        to "uncertain" — same-repo continuity cannot prove "no reach-back";
      - a post-anchor segment quoting the candidate's content verbatim is a
        reach-back pointer -> "no" (keep);
      - boilerplate duplicates -> "uncertain" (v1 adjudicated the identical
        wait-poll/empty-stub pattern uncertain / keep);
      - coherent foreign tail threads (>=2 consecutive segments sharing
        vocabulary disjoint from anchor + immediate response) -> "yes".

  * Output shape mirrors candidates_drop_supp_v1.jsonl:
      {group_id, request:{state, questions.irrelevant}, targets:null,
       meta:{record_id=v5f2:<hash>:segNNNN, ..., proposed_label,
             proposal_note, strategy, holdout_collision_check}} plus the
      top-level record_id/strategy/proposed_label/proposal_note fields the
      v5 spec asks for. targets stay null — proposals only, pending the
      owner/agent adjudication pass.

Determinism: fixed seed, sorted file/transcript iteration, pure local rules,
no network, no service calls, no tokenizer (character-budget windowing like
the v1 miner).

Usage: python3 scripts/mine_v5_hardneg_v1.py
Output: data/valen_nano_v5/f2_mined_hardneg.jsonl
        data/valen_nano_v5/f2_report.json
"""

import hashlib
import importlib.util
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "build_real_context_eval_v1", ROOT / "scripts" / "build_real_context_eval_v1.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

OUT_DIR = ROOT / "data" / "valen_nano_v5"
OUT_JSONL = OUT_DIR / "f2_mined_hardneg.jsonl"
OUT_REPORT = OUT_DIR / "f2_report.json"

# Frozen holdout: the 590 labeled record_ids + the 27 excluded (uncertain/
# adjudicated) rows — all adjudicated candidate+anchor pairs stay eval-only.
HOLDOUT_FILES = [
    ROOT / "data" / "real_context_eval_v1" / "eval.jsonl",
    ROOT / "data" / "real_context_eval_v1" / "drop_supp" / "eval.jsonl",
    ROOT / "data" / "real_context_eval_v1" / "excluded.jsonl",
    ROOT / "data" / "real_context_eval_v1" / "drop_supp" / "excluded.jsonl",
]

SEED = 20261201            # v5 reserved seed (V5_DATA_DESIGN_V1 §1.3)
STATE_CHAR_BUDGET = 12_000  # v1 miner conservative stand-in for 7000 packed tok
STATE_CHAR_HARD = 14_000    # ~8192 packed-token compiler limit with headroom
GROUP_LINEAGE = "real_context_v5"

Y, N, U = "yes", "no", "uncertain"  # proposed_label values, v1 convention

# caps (all deterministic)
MAX_PER_BOUNDARY = 8        # cross_task candidates per shift boundary
MAX_DEEP_REANCHOR = 8       # deep earlier-span candidates vs final anchor
MAX_STALE_PER_TRANSCRIPT = 6
MAX_DUP_GROUPS = 4          # boilerplate dup groups per transcript
MAX_DUP_PER_GROUP = 3       # occurrences per dup group
MAX_FOREIGN_TAIL = 6        # foreign tail segments per transcript
MAX_PER_TRANSCRIPT = 30     # total mined rows per transcript

# --- deterministic text rules ------------------------------------------------

# Explicit task-abandon / pivot markers (zh + en). Presence in the anchor
# strengthens the cross_task "yes" proposal.
ABANDON_RE = re.compile(
    r"(?:算了|不用了|先不[要做弄看用]|别管了?|不管了|取消|放弃|换个|换一种|换一个|"
    r"换方向|先跳过|先放一放?|先放|不弄了|不做了|先停|停一下|重来|重新来|改主意|"
    r"改为|改成|先别急|回头再)"
    r"|(?:never\s*mind|forget\s+(?:it|about\s+that)|let'?s\s+move\s+on|"
    r"move\s+on\s+to|switch\s+to|instead\b|actually\s+let'?s|new\s+task|"
    r"different\s+task|separate\s+(?:task|question)|unrelated\s+question|"
    r"drop\s+that|skip\s+that|abandon|scrap\s+that|leave\s+that|"
    r"change\s+of\s+plan|another\s+thing)",
    re.IGNORECASE)

# A user turn that is only an ack/continuation is not a task-shift boundary.
CONTINUATION_RE = re.compile(
    r"^\W*(?:ok(?:ay)?|yes|yep|yeah|y|sure|go(?:\s*on)?|continue|proceed|next|"
    r"keep\s+going|do\s+it|go\s+ahead|thanks?|thank\s*you|thx|"
    r"好|好的|好吧|嗯|嗯嗯|是|是的|对|对的|行|可以|继续|继续吧|接着|接着来|"
    r"谢谢|收到|就这样|就这个|弄吧|做吧|来吧|开始吧?|往下)\W*$",
    re.IGNORECASE)

# Path-ish tokens (contain a directory separator) and long identifiers.
PATH_TOKEN_RE = re.compile(r"(?<![\w./-])(?:[A-Za-z0-9_.~-]+/)+[A-Za-z0-9_./~-]+")
IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{5,}\b")
PATCH_PATH_RE = re.compile(r"\*\*\*\s*(?:Update|Add|Delete|Move)\s+File:\s*(\S+)")
DIGIT_RUN_RE = re.compile(r"\d+")

GENERIC_TOKENS = {
    "true", "false", "null", "none", "stdout", "stderr", "output", "result",
    "results", "return", "import", "function", "export", "const", "string",
    "number", "object", "print", "error", "errors", "warning", "failed",
    "passed", "content", "message", "system", "assistant", "python", "python3",
    "json", "http", "https", "localhost", "command", "commands", "script",
    "scripts", "index", "test", "tests", "main", "utils", "readme", "config",
    "users", "documents", "server", "client", "value", "values", "params",
    "arguments", "options", "status", "running", "completed", "output.txt",
}

# Pure-acknowledgement heuristic (R6): short, no digits/paths, ack-shaped.
ACK_RE = re.compile(
    r"^\W*(?:ok(?:ay)?|sure|got it|on it|understood|wilco|roger|let me|"
    r"i'?ll|i will|looking|working on it|one (?:sec|moment)|give me|"
    r"好|好的|收到|明白|了解|我来|我看下|我看看|稍等|马上|这就|行|嗯|"
    r"看一下|让我|我来处理|马上来)",
    re.IGNORECASE)

READ_NAMES = {"read", "view", "open_file", "cat", "read_file"}
EDIT_NAMES = {"edit", "multiedit", "write", "notebookedit", "str_replace_editor",
              "apply_patch", "patch", "update_file", "insert_edit_into_file",
              "create_file"}
LIST_NAMES = {"list_dir", "list_directory", "glob", "ls"}
SHELL_NAMES = {"bash", "shell", "exec", "exec_command", "local_shell", "cmd",
               "run_terminal_cmd", "js", "node", "write_stdin", "powershell",
               "terminal", "container.exec", "sh", "zsh", "run_command",
               "process", "container"}
IGNORED_TOOL_NAMES = {"grep", "search", "web_search", "send_message", "wait",
                      "agent", "task", "todowrite", "update_plan", "think",
                      "image_generation", "notebook"}
READ_CMD_RE = re.compile(
    r"\b(?:sed\s+-n|cat|head|tail|less|more|bat|awk|nl)\b|"
    r"open\(|read_text\(|Path\(")
LIST_CMD_RE = re.compile(r"\b(?:ls|dir|find|tree|fd)\b|--files\b")


def norm(text):
    return " ".join(str(text).split())


def subject_tokens(text):
    """Strong tokens = full path-ish strings; weak = basenames + long
    identifiers (snake_case / camelCase / len>=10)."""
    strong, weak = set(), set()
    for m in PATH_TOKEN_RE.finditer(text):
        tok = m.group(0).rstrip("/.").lower()
        if len(tok) < 4:
            continue
        strong.add(tok)
        base = tok.rsplit("/", 1)[-1]
        if len(base) >= 4 and base not in GENERIC_TOKENS:
            weak.add(base)
    for m in IDENT_RE.finditer(text):
        w = m.group(0)
        wl = w.lower()
        if wl in GENERIC_TOKENS:
            continue
        if "_" in w or re.search(r"[a-z][A-Z]", w) or len(w) >= 10:
            weak.add(wl)
    return strong, weak


def signatures(norm_text):
    """Up to 3 fixed-position 60-char chunks for verbatim reach-back checks."""
    if len(norm_text) < 40:
        return []
    chunks = []
    for pos in (0, max(0, len(norm_text) // 2 - 30), max(0, len(norm_text) - 60)):
        chunks.append(norm_text[pos:pos + 60])
    return [c for c in dict.fromkeys(chunks) if len(c) >= 40]


def ack_like(text):
    n = norm(text)
    if len(n) > 100 or DIGIT_RUN_RE.search(n) or PATH_TOKEN_RE.search(n):
        return False
    return bool(ACK_RE.search(n))


def is_continuation_anchor(text):
    n = norm(text)
    return len(n) <= 24 and bool(CONTINUATION_RE.match(n))


def boilerplate_key(text):
    """Digit-normalized dedup key: identical outputs that differ only in
    volatile numbers (cell ids, counts, pids) group together."""
    n = norm(text).lower()
    n = DIGIT_RUN_RE.sub("#", n)
    return n


def tool_call_ops(text):
    """Parse a tool_use segment -> list of (op, [paths]); op in
    read|edit|list. Deterministic, both claude dict-input and codex
    command-string dialects."""
    try:
        payload = json.loads(text)["tool_use"]
    except (ValueError, KeyError, TypeError):
        return []
    name = (payload.get("name") or "").lower()
    inp = payload.get("input")
    paths = []
    if isinstance(inp, dict):
        for key in ("file_path", "path", "filename", "filepath", "target_file",
                    "notebook_path"):
            v = inp.get(key)
            if isinstance(v, str) and v.strip():
                paths.append(v.strip())
    cmd_text = inp if isinstance(inp, str) else (
        json.dumps(inp, ensure_ascii=False) if inp is not None else "")

    def npaths(ps):
        return [p.lstrip("./") or p for p in ps]

    if name in READ_NAMES:
        return [("read", npaths(paths))]
    if name in EDIT_NAMES:
        return [("edit", npaths(paths))]
    if name in LIST_NAMES:
        return [("list", npaths(paths or [m.group(0) for m in
                                          PATH_TOKEN_RE.finditer(cmd_text)]))]
    if name in IGNORED_TOOL_NAMES:
        return []
    if name in SHELL_NAMES or not name:
        ops = []
        low = cmd_text.lower()
        patch_paths = PATCH_PATH_RE.findall(cmd_text)
        if "apply_patch" in low or "*** begin patch" in low:
            ops.append(("edit", npaths(patch_paths or
                                       [m.group(0) for m in
                                        PATH_TOKEN_RE.finditer(cmd_text)])))
        if READ_CMD_RE.search(cmd_text):
            # paths in the command that are not the patch targets
            found = [m.group(0) for m in PATH_TOKEN_RE.finditer(cmd_text)]
            if patch_paths:
                pset = set(patch_paths)
                found = [p for p in found if p not in pset]
            if found:
                ops.append(("read", npaths(found)))
        if LIST_CMD_RE.search(cmd_text):
            found = [m.group(0) for m in PATH_TOKEN_RE.finditer(cmd_text)]
            ops.append(("list", npaths(found)))
        return ops
    return []


# --- windowing (verbatim port of mine_drop_supp_v1.py) ------------------------

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
    optional = sorted((i for i in range(len(segs)) if i not in kept),
                      key=lambda i: (abs(i - candidate_idx), i))
    for i in optional:
        trial = kept | {i}
        if state_size(trial) <= STATE_CHAR_BUDGET:
            kept = trial
    return render_window(segs, kept), kept != set(range(len(segs)))


# --- holdout -----------------------------------------------------------------

def _anchor_pointer_from_state(record):
    """Anchor pointer for a main-eval record: pointer of the last role=='user'
    segment in the serialized windowed conversation (the builder always keeps
    the last user message as the request anchor)."""
    try:
        state = json.loads(record["request"]["state"])
        ptr = None
        for seg in state.get("conversation", []):
            if seg.get("role") == "user":
                ptr = seg.get("pointer")
        return ptr
    except (ValueError, KeyError, TypeError):
        return None


def load_holdout():
    """Frozen adjudicated pairs. Returns (pairs, pointers, transcripts):
    pairs: {(th, candidate_pointer, anchor_pointer)} — never re-emitted;
    pointers: {th: {candidate_pointers}} — informational; transcripts: {th}
    of the 60 eval transcripts."""
    pairs, pointers, transcripts = set(), defaultdict(set), set()
    for path in HOLDOUT_FILES:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            meta = rec.get("meta", {})
            th, cp = meta.get("transcript_hash"), meta.get("candidate_pointer")
            if not th or not cp:
                continue
            anchor = meta.get("anchor_pointer") or _anchor_pointer_from_state(rec)
            transcripts.add(th)
            pointers[th].add(cp)
            if anchor:
                pairs.add((th, cp, anchor))
    return pairs, pointers, transcripts


# --- transcript scan (builder thresholds, no admission quota) -----------------

def scan_transcripts():
    """All source transcripts meeting the builder's minimum quality
    thresholds. Returns (usable, drops, scanned)."""
    files = builder.iter_transcript_files(builder.DEFAULT_CLAUDE_ROOT,
                                          builder.DEFAULT_CODEX_ROOT)
    drops, seen = Counter(), set()
    usable = []
    for source, path in files:
        try:
            extract = (builder.extract_claude if source == "claude_projects"
                       else builder.extract_codex)
            t = extract(path)
        except (builder.Drop, OSError) as drop:
            drops[getattr(drop, "reason", "unreadable")] += 1
            continue
        if len(t.segments) < builder.MIN_SEGMENTS:
            drops["min_segments"] += 1
            continue
        if len(t.users) < builder.MIN_USER_MESSAGES:
            drops["no_user_messages"] += 1
            continue
        if not builder.eligible_candidate_indices(t.segments):
            drops["no_candidates"] += 1
            continue
        hit = builder.credential_scan("\n".join(s["text"] for s in t.segments))
        if hit is not None:
            drops[f"credential_scan:{hit}"] += 1
            continue
        try:
            th = builder.sha256_file(path)
        except OSError:
            drops["unreadable"] += 1
            continue
        if th in seen:
            drops["duplicate_source_file"] += 1
            continue
        seen.add(th)
        usable.append({"transcript_hash": th, "source": source,
                       "path": str(path), "transcript": t})
    usable.sort(key=lambda u: (u["source"], u["transcript_hash"]))
    return usable, drops, len(files)


# --- mining -------------------------------------------------------------------

def eligible_mine_kinds(seg):
    return seg["kind"] in ("assistant_text", "tool_result", "user_turn") \
        and seg["role"] not in ("control", "system")


def evenly_spaced(indices, cap):
    """Deterministic even spread of <=cap indices from a sorted list."""
    indices = sorted(indices)
    if len(indices) <= cap:
        return indices
    if cap <= 1:
        return indices[:cap]
    step = (len(indices) - 1) / (cap - 1)
    return sorted({indices[round(k * step)] for k in range(cap)})


def find_stale_chains(t, span_lo, span_hi):
    """Read/list tool_result superseded by Edit + re-Read of the same path
    inside [span_lo, span_hi). Returns [(result_idx, path, reread_use_idx)]."""
    segs = t.segments
    ops = []  # (seg_idx, op, path)
    for i in range(span_lo, span_hi):
        if segs[i]["kind"] != "tool_use":
            continue
        for op, paths in tool_call_ops(segs[i]["text"]):
            for p in paths:
                ops.append((i, op, p))
    chains = []
    for oi, (use_i, op_i, p) in enumerate(ops):
        if op_i not in ("read", "list"):
            continue
        edit_j = next(((u, p2) for (u, o2, p2) in ops[oi + 1:]
                       if o2 == "edit" and (p2 == p or
                                            (op_i == "list" and
                                             p2.startswith(p.rstrip("/") + "/")))),
                      None)
        if edit_j is None:
            continue
        reread_k = next(((u, p2) for (u, o2, p2) in ops[oi + 1:]
                         if u > edit_j[0] and o2 in ("read", "list") and
                         (p2 == p or (op_i == "list" and
                                      p2.startswith(p.rstrip("/") + "/")))),
                        None)
        if reread_k is None:
            continue  # spec: superseded means Edit AND a re-Read/re-list
        # candidate = first tool_result after the read/list call
        res = next((k for k in range(use_i + 1, min(span_hi, edit_j[0] + 1))
                    if segs[k]["kind"] == "tool_result"), None)
        if res is not None:
            chains.append((res, p, reread_k[0] if reread_k else None))
    return chains


def mine_transcript(item, skips):
    """Collect mined proposals for one transcript. Returns list of proposal
    dicts {seg, anchor_seg, anchor_uidx, cut, strategy, label, note, extra}."""
    th, t = item["transcript_hash"], item["transcript"]
    segs = t.segments
    n_users = len(t.users)
    user_seg = [i for i, s in enumerate(segs) if s["kind"] == "user_turn"]
    last_user_seg = user_seg[-1] if user_seg else None
    norm_segs = [norm(s["text"]).lower() for s in segs]
    subj = [subject_tokens(s["text"]) for s in segs]

    # post-text tables for reach-back checks: joined normalized text after
    # each index (computed lazily per boundary to keep memory sane).
    def post_text(start):
        return " ".join(norm_segs[start:])

    def post_subj(start):
        strong, weak = set(), set()
        for st, wk in subj[start:]:
            strong |= st
            weak |= wk
        return strong, weak

    proposals = {}  # (seg_idx, anchor_seg) -> proposal, first strategy wins

    # --- 1) stale_read_edit ---------------------------------------------------
    stale_count = 0
    for j in range(n_users):
        span_lo = user_seg[j] + 1
        span_hi = user_seg[j + 1] if j + 1 < n_users else len(segs)
        chains = find_stale_chains(t, span_lo, span_hi)
        for res_idx, path, reread_use in chains:
            if stale_count >= MAX_STALE_PER_TRANSCRIPT:
                break
            if j + 1 < n_users:
                a_uid, a_seg = j + 1, user_seg[j + 1]
                post_strong, post_weak = post_subj(a_seg + 1)
                cand_strong, _ = subj[res_idx]
                pl = path.lower()
                if any(pl in tok or tok in pl for tok in post_strong) or \
                        post_text(a_seg + 1).find(pl) >= 0:
                    label = U
                    note = (f"stale read/list of {path} superseded by "
                            f"Edit+re-Read in task span, but the path is "
                            f"referenced post-shift; R4 unreferenced check "
                            f"fails")
                elif any(sig in post_text(a_seg + 1)
                         for sig in signatures(norm(segs[res_idx]["text"]).lower())):
                    label = N
                    note = ("superseded read of %s is quoted/referenced "
                            "post-shift; reach-back pointer (R5.2)" % path)
                else:
                    label = Y
                    note = (f"read of {path} superseded by subsequent "
                            f"Edit+re-Read of the same path inside one task; "
                            f"no post-shift reference (R4 stale rule)")
            else:
                a_uid, a_seg = n_users - 1, last_user_seg
                label = U
                note = (f"read/list of {path} superseded by Edit+re-Read "
                        f"inside the FINAL task; R4 'unreferenced post-shift' "
                        f"is unverifiable without a later task")
            key = (res_idx, a_seg)
            if key not in proposals:
                proposals[key] = {
                    "seg": res_idx, "anchor_seg": a_seg,
                    "anchor_uidx": a_uid,
                    "cut": user_seg[a_uid + 1] if a_uid + 1 < n_users
                    else len(segs),
                    "strategy": "stale_read_edit", "label": label,
                    "note": note,
                    "extra": {"superseded_path": path,
                              "reread_pointer": (
                                  f"/messages/{reread_use}/content"
                                  if reread_use is not None else None)}}
                stale_count += 1

    # --- 2) cross_task_reanchor ------------------------------------------------
    def cross_label(cand_idx, a_uid):
        cand_strong, cand_weak = subj[cand_idx]
        post_strong, post_weak = post_subj(user_seg[a_uid])
        anchor_strong, anchor_weak = subject_tokens(t.users[a_uid])
        ptext = post_text(user_seg[a_uid] + 1)
        marker = bool(ABANDON_RE.search(t.users[a_uid]))
        # reach-back: post-anchor text quotes the candidate verbatim
        if any(sig in ptext
               for sig in signatures(norm_segs[cand_idx])):
            return N, ("post-shift text quotes the candidate's content; "
                       "explicit reach-back pointer (R5.2)")
        if ack_like(segs[cand_idx]["text"]):
            return Y, ("pure acknowledgement / low-content assistant text "
                       "pre-shift; carries no evidence (R6)")
        strong_hits = cand_strong & (post_strong | anchor_strong)
        weak_hits = cand_weak & (post_weak | anchor_weak)
        if strong_hits or len(weak_hits) >= 2:
            hits = sorted(strong_hits | weak_hits)[:4]
            base = (f"candidate shares subject tokens {hits} with "
                    f"anchor/post-shift content; same-repo continuity, "
                    f"reach-back unverifiable (R5.2)")
            if marker:
                return U, (base + "; explicit abandon marker in anchor keeps "
                           "this a candidate but cannot prove no-reach-back")
            return U, base
        if marker:
            return Y, ("explicit task-abandon/pivot marker in anchor "
                       "(mirrors v1 'ok算了' adjudication); disjoint subject "
                       "vocabulary, no post-shift reference")
        return Y, ("pre-shift segment reanchored at next user task; disjoint "
                   "subject vocabulary, no post-shift reference detected "
                   "(auto R5 approximation — needs adjudication)")

    for a in range(1, n_users):
        a_seg = user_seg[a]
        prev_seg = user_seg[a - 1]
        if a_seg - prev_seg <= 1:
            skips["degenerate_boundary"] += 1
            continue
        if is_continuation_anchor(t.users[a]):
            skips["continuation_anchor"] += 1
            continue
        span = [i for i in range(prev_seg + 1, a_seg)
                if eligible_mine_kinds(segs[i])]
        for i in evenly_spaced(span, MAX_PER_BOUNDARY):
            label, note = cross_label(i, a)
            key = (i, a_seg)
            if key in proposals:
                continue
            proposals[key] = {
                "seg": i, "anchor_seg": a_seg, "anchor_uidx": a,
                "cut": user_seg[a + 1] if a + 1 < n_users else len(segs),
                "strategy": "cross_task_reanchor", "label": label,
                "note": note, "extra": {}}

    # deep reanchor: earlier spans judged vs the FINAL user request (v1's
    # "all pre-shift vs u_last" pattern); only when >=3 user turns.
    if n_users >= 3 and last_user_seg is not None:
        a = n_users - 1
        deep_lo = user_seg[a - 1]  # earlier = before the last task span
        deep = [i for i in range(0, deep_lo) if eligible_mine_kinds(segs[i])]
        for i in evenly_spaced(deep, MAX_DEEP_REANCHOR):
            if (i, last_user_seg) in proposals:
                continue
            label, note = cross_label(i, a)
            if segs[i]["kind"] == "user_turn":
                label, note = U, ("pre-shift user turn judged vs final "
                                  "anchor; may carry a persistent "
                                  "constraint/correction (R2/R3)")
            proposals[(i, last_user_seg)] = {
                "seg": i, "anchor_seg": last_user_seg, "anchor_uidx": a,
                "cut": len(segs), "strategy": "cross_task_reanchor",
                "label": label, "note": "deep reanchor: " + note,
                "extra": {"deep_reanchor": True}}

    # --- 3) same_anchor_tail ---------------------------------------------------
    if last_user_seg is not None:
        # (a) digit-normalized duplicate boilerplate groups
        groups = defaultdict(list)
        for i, s in enumerate(segs):
            if s["kind"] in ("tool_result", "assistant_text") and i != last_user_seg:
                key = boilerplate_key(s["text"])
                if 4 <= len(key) <= 400:
                    groups[key].append(i)
        dup_groups = sorted((g for g in groups.values() if len(g) >= 2),
                            key=lambda g: (g[0]))
        for g in dup_groups[:MAX_DUP_GROUPS]:
            for i in g[:MAX_DUP_PER_GROUP]:
                if (i, last_user_seg) in proposals:
                    continue
                proposals[(i, last_user_seg)] = {
                    "seg": i, "anchor_seg": last_user_seg,
                    "anchor_uidx": n_users - 1, "cut": len(segs),
                    "strategy": "same_anchor_tail", "label": U,
                    "note": (f"identical (digit-normalized) boilerplate "
                             f"repeated x{len(g)} under the final anchor; v1 "
                             f"adjudicated this pattern uncertain"),
                    "extra": {"dup_group_size": len(g)}}
        # (b) foreign coherent thread in the post-final-user tail
        anchor_strong, anchor_weak = subject_tokens(t.users[-1])
        resp_strong, resp_weak = set(), set()
        for i in range(last_user_seg + 1, min(len(segs), last_user_seg + 5)):
            resp_strong |= subj[i][0]
            resp_weak |= subj[i][1]
        tail = [i for i in range(last_user_seg + 1, len(segs))
                if eligible_mine_kinds(segs[i])]
        foreign = []
        for i in tail:
            st, wk = subj[i]
            if not (st or wk):
                continue
            if st & (anchor_strong | resp_strong) or \
                    wk & (anchor_weak | resp_weak):
                continue
            foreign.append(i)
        # coherent run: a foreign segment adjacent (within 2 segs) to another
        run_ok = set()
        for i in foreign:
            if any(abs(i - j) <= 2 for j in foreign if j != i):
                run_ok.add(i)
        for i in foreign[:MAX_FOREIGN_TAIL]:
            if (i, last_user_seg) in proposals:
                continue
            if i in run_ok:
                label = Y
                note = ("foreign-task thread in the post-final-user tail: "
                        "subject vocabulary disjoint from the anchor request "
                        "and its immediate response (auto-detected, mirrors "
                        "v1 drop labels seg0025/seg0027 pattern)")
            else:
                label = U
                note = ("isolated foreign-looking tail segment under the "
                        "final anchor; no coherent thread detected")
            proposals[(i, last_user_seg)] = {
                "seg": i, "anchor_seg": last_user_seg,
                "anchor_uidx": n_users - 1, "cut": len(segs),
                "strategy": "same_anchor_tail", "label": label, "note": note,
                "extra": {"foreign_tail": True}}

    # --- per-transcript cap: strategy priority then seg ------------------------
    prio = {"stale_read_edit": 0, "cross_task_reanchor": 1,
            "same_anchor_tail": 2}
    ordered = sorted(proposals.values(),
                     key=lambda p: (prio[p["strategy"]], p["seg"]))
    if len(ordered) > MAX_PER_TRANSCRIPT:
        skips["per_transcript_cap"] += len(ordered) - MAX_PER_TRANSCRIPT
        ordered = ordered[:MAX_PER_TRANSCRIPT]
    return ordered


def emit(item, prop, instructions, holdout_pairs, holdout_ptrs,
         eval_transcripts, used_ids, skips):
    th, t, src = item["transcript_hash"], item["transcript"], item["source"]
    segs = t.segments
    idx, a_seg, a_uid = prop["seg"], prop["anchor_seg"], prop["anchor_uidx"]
    if idx == a_seg or idx >= prop["cut"]:
        skips["candidate_out_of_scope"] += 1
        return None
    seg = segs[idx]
    if not eligible_mine_kinds(seg):
        skips["bad_kind"] += 1
        return None
    ptr = f"/messages/{idx}/content"
    anchor_ptr = f"/messages/{a_seg}/content"
    check = {
        "transcript_in_eval60": th in eval_transcripts,
        "candidate_pointer_in_holdout": ptr in holdout_ptrs.get(th, set()),
        "pair_in_holdout": (th, ptr, anchor_ptr) in holdout_pairs,
    }
    if check["pair_in_holdout"]:
        skips["holdout_pair_collision"] += 1
        return None
    prefix = segs[:prop["cut"]]
    user_list = t.users[:a_uid + 1]
    conv, windowed = windowed_conversation(prefix, idx, a_seg, user_list)
    state = {"conversation": conv, "candidate_pointer": ptr,
             "user_messages_in_order": user_list}
    state_str = builder.serialized(state)
    if len(state_str.encode("utf-8")) > STATE_CHAR_HARD:
        skips["exceeds_budget"] += 1
        return None
    rid = f"v5f2:{th[:12]}:seg{idx:04d}"
    if rid in used_ids:
        rid = f"{rid}-u{a_uid}"
    if rid in used_ids:
        rid = f"{rid}-{prop['strategy'][:4]}"
    used_ids.add(rid)
    meta = {
        "record_id": rid,
        "domain": "real",
        "modality": "text",
        "language_bucket": builder.language_bucket(t),
        "source": "real_transcript",
        "source_dataset": GROUP_LINEAGE,
        "source_root": src,
        "transcript_hash": th,
        "candidate_kind": seg["kind"],
        "candidate_pointer": ptr,
        "length_band": builder.length_band(len(segs)),
        "selection_method": f"mining:{prop['strategy']}",
        "supp_source": "mining",
        "mine_mode": prop["strategy"],
        "strategy": prop["strategy"],
        "anchor_user_index": a_uid,
        "anchor_pointer": anchor_ptr,
        "windowed": windowed,
        "state_chars": len(state_str.encode("utf-8")),
        "proposed_label": prop["label"],
        "proposal_note": prop["note"],
        "holdout_collision_check": check,
        **prop.get("extra", {}),
    }
    return {
        "record_id": rid,
        "group_id": f"{GROUP_LINEAGE}:{th}",
        "request": {"state": state_str,
                    "questions": {"irrelevant": {"type": "noul",
                                               "instructions": instructions}}},
        "targets": None,
        "strategy": prop["strategy"],
        "proposed_label": prop["label"],
        "proposal_note": prop["note"],
        "holdout_collision_check": check,
        "meta": meta,
    }


def main():
    rng = random.Random(SEED)
    instructions = builder.load_verbatim_instructions(ROOT / "data" / "valen_nano_v3")
    holdout_pairs, holdout_ptrs, eval_transcripts = load_holdout()
    usable, drops, scanned = scan_transcripts()
    eval_usable = sum(1 for u in usable if u["transcript_hash"] in eval_transcripts)

    records, report_rows, skips = [], [], Counter()
    used_ids = set()
    for item in usable:
        props = mine_transcript(item, skips)
        for prop in props:
            rec = emit(item, prop, instructions, holdout_pairs, holdout_ptrs,
                       eval_transcripts, used_ids, skips)
            if rec is None:
                continue
            records.append(rec)
            report_rows.append({
                "record_id": rec["record_id"],
                "transcript": item["transcript_hash"][:12],
                "source": item["source"],
                "strategy": prop["strategy"],
                "seg": prop["seg"],
                "anchor_seg": prop["anchor_seg"],
                "kind": rec["meta"]["candidate_kind"],
                "proposed_label": prop["label"],
                "note": prop["note"],
                "candidate_snippet": norm(
                    item["transcript"].segments[prop["seg"]]["text"])[:140],
                "anchor_snippet": norm(
                    item["transcript"].users[prop["anchor_uidx"]])[:140],
            })

    records.sort(key=lambda r: (r["meta"]["transcript_hash"], r["record_id"]))
    label_counts = Counter(r["proposed_label"] for r in records)
    strat_counts = Counter(r["strategy"] for r in records)
    strat_label = Counter((r["strategy"], r["proposed_label"]) for r in records)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in records:
            f.write(builder.serialized(r) + "\n")

    # deterministic seeded sample of up to 20 proposals for spot review
    sample_idx = sorted(rng.sample(range(len(report_rows)),
                                   min(20, len(report_rows))))
    top20 = [report_rows[i] for i in sample_idx]

    report = {
        "schema_version": "nanojev-v5-f2-hardneg-mined-v1",
        "builder": "scripts/mine_v5_hardneg_v1.py",
        "seed": SEED,
        "created_from": ("all quality-passing transcripts under "
                         "~/.claude/projects + ~/.codex/sessions "
                         "(same roots/thresholds as "
                         "build_real_context_eval_v1.py, no admission quota)"),
        "note": ("PROPOSALS ONLY: targets stay null; proposed_label + "
                 "proposal_note are deterministic-rule suggestions pending "
                 "the owner/agent adjudication pass (R8 evidence_basis = "
                 "proposal_note). Labels follow "
                 "docs/REAL_CONTEXT_LABELING_GUIDE_V1.md asymmetric semantics "
                 "(yes == certainly irrelevant == drop). No model/service "
                 "calls."),
        "mining_strategies": {
            "cross_task_reanchor": ("every non-degenerate inter-user span "
                                    "reanchored at the next user turn (+ deep "
                                    "reanchor of earlier spans vs the final "
                                    "request); continuation/ack anchors "
                                    "skipped; subject-vocabulary continuity "
                                    "downgrades to uncertain, verbatim "
                                    "reach-back to no, abandon markers "
                                    "strengthen yes"),
            "stale_read_edit": ("read/list tool_result superseded by a later "
                                "Edit + re-Read of the same path inside one "
                                "task span; anchored at the next user turn "
                                "(final-task chains forced uncertain)"),
            "same_anchor_tail": ("digit-normalized duplicate boilerplate "
                                 "groups (uncertain, v1 precedent) + coherent "
                                 "foreign tail threads under the final "
                                 "anchor"),
        },
        "holdout": {
            "frozen_pairs_loaded": len(holdout_pairs),
            "files": [str(p.relative_to(ROOT)) for p in HOLDOUT_FILES
                      if p.is_file()],
            "eval_transcripts": len(eval_transcripts),
            "eval_transcripts_mined": eval_usable,
            "rule": ("a mined record is skipped iff its "
                     "(transcript_hash, candidate_pointer, anchor_pointer) "
                     "triple matches a frozen pair; sharing a transcript_hash "
                     "or a candidate_pointer alone is allowed and recorded "
                     "in meta.holdout_collision_check"),
        },
        "counts": {
            "files_scanned": scanned,
            "usable_transcripts": len(usable),
            "unusable_drops": dict(sorted(drops.items())),
            "records": len(records),
            "proposed_labels": dict(label_counts),
            "by_strategy": dict(strat_counts),
            "by_strategy_label": {f"{k[0]}:{k[1]}": v
                                  for k, v in sorted(strat_label.items())},
            "by_transcript": dict(Counter(r["meta"]["transcript_hash"][:12]
                                          for r in records)),
            "by_kind": dict(Counter(r["meta"]["candidate_kind"]
                                    for r in records)),
            "by_source": dict(Counter(r["meta"]["source_root"]
                                      for r in records)),
            "windowed": sum(1 for r in records if r["meta"]["windowed"]),
            "from_eval60_transcripts": sum(
                1 for r in records
                if r["meta"]["holdout_collision_check"]["transcript_in_eval60"]),
            "skips": dict(sorted(skips.items())),
        },
        "spot_review_sample_20": top20,
        "rows": report_rows,
        "caveats": [
            ("pool size: %d transcripts pass the builder's minimum quality "
             "thresholds (min_segments>=12, >=1 user msg, byte cap, credential "
             "scan), far fewer than the ~290 'unadmitted' estimate — that "
             "figure conflated files scanned (351) with usable transcripts; "
             "the v1 manifest's own drop table implies ~70" % len(usable)),
            ("task directive mines eval-adjacent transcripts including the 60 "
             "admitted ones under pair-level holdout; this deviates from "
             "V5_DATA_DESIGN_V1 §1.6's blanket train-row ban on the 60 — the "
             "pair-level rule is what the task specified"),
            ("proposals are deterministic heuristics, not labels: cross_task "
             "'yes' rows assume disjoint vocabulary implies no reach-back; "
             "the human/agent adjudication pass must still verify R5's four "
             "checks before any training use"),
            ("codex read/edit detection parses shell command strings "
             "(sed/cat/apply_patch) — approximate; claude Read/Edit tool_use "
             "dict-inputs are exact"),
            ("foreign-tail detection is a weak proxy (vocabulary disjointness "
             "from anchor + immediate response); isolated cases are kept "
             "uncertain"),
        ],
    }
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"scanned {scanned} files -> {len(usable)} usable transcripts "
          f"({eval_usable} of them eval-admitted)")
    print(f"wrote {len(records)} records -> {OUT_JSONL}")
    print("labels:", dict(label_counts), "| strategies:", dict(strat_counts))
    print("skips:", dict(sorted(skips.items())), "| drops:", dict(sorted(drops.items())))
    print(f"report -> {OUT_REPORT}")


if __name__ == "__main__":
    sys.exit(main())
