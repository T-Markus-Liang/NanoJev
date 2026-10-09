#!/usr/bin/env python3
"""V5 F2 hard-negative miner v1: EXIT-style drop/keep proposals mined
automatically from the *remaining* local real transcripts (the transcripts NOT
admitted to real_context_eval_v1).

Implements docs/V5_DATA_DESIGN_V1.md §3 (family F2, exit_hard_negative) with
the v1-task refinements:

  * Pool split FIRST (this is the load-bearing difference from
    mine_v5_hardneg_v1.py):
      - the 60 eval-v1 transcripts (group_ids in
        data/real_context_eval_v1/candidates.jsonl) are NEVER mined — §1.6
        forbids them for train rows and the v5 test asserts pool
        disjointness;
      - of the remaining quality-passing transcripts, a deterministic
        reserve (prefer longer / more task switches, hash tie-break) is set
        aside as the v5-EVAL mining pool — its hashes are recorded in the
        report ledger so they stay out of train;
      - everything else is the TRAIN mining pool and is the only pool mined
        here.
      NOTE: the residual pool is much smaller than the doc's ~290 estimate
      (see report caveats); the reserve is capped at ~1/3 of the pool and the
      shortfall vs the 15–20 target is reported, not widened.

  * Mining strategies (deterministic, no model/service calls):
      - cross_task_reanchor: earlier-task segment judged under a later-task
        anchor (per boundary: the immediately preceding span plus an evenly
        spaced deep sample of still-earlier segments; plus a deep-reanchor
        pass against the final request). user_messages_in_order truncated at
        the anchor; conversation cut before the next user turn — same state
        shape as v1.
      - stale_superseded: Read/cat/list tool_result superseded by a later
        Read of the same path OR an Edit of that path inside one task span;
        anchored at the next user turn (final-task chains forced uncertain).
      - same_anchor_tail: foreign-task/boilerplate blocks under the
        transcript's existing final anchor — digit-normalized duplicate
        boilerplate groups plus coherent foreign-thread runs.
      - KEEP-SIDE complement for every strategy: segments that ARE reached
        back to (evidence: path re-referenced downstream, or output quoted
        verbatim in later assistant text) are emitted with proposed_label
        "no" (keep) — hard negatives for the wrong-direction prior. Emitted
        rows carry meta.mining_side in {drop, keep} and meta.evidence_basis.

  * Contrastive pairs (doc §1.7): meta.pair_id groups minimal-difference
    opposite-label records —
      pair_kind=anchor_flip: same candidate_pointer under different anchors
        with opposite labels (only user_messages_in_order/window differ);
      pair_kind=same_anchor: same anchor_pointer + opposite labels under one
        boundary (keep-side vs drop-side complement pair).

  * Guards: same credential scan PATTERNS as v1, enforced at the granularity
    windowed mining actually needs — a flagged segment is never a candidate
    and never enters a context window (it is elided); a flagged user message
    disqualifies every anchor at/after it (user_messages_in_order cannot
    elide); the emitted state of every record is scanned again in
    self_check. This replaces v1's whole-transcript drop, which exists only
    because eval-v1 emitted whole-transcript states — mining emits small
    windowed states, so transcript-level drops would destroy the pool
    (85/351 files flagged incl. nearly all long multi-task sessions) for no
    privacy gain. Also: <=32KB candidate segment cap, character-budget
    windowing (~7000 packed tokens stand-in, hard 14_000 bytes),
    request/state hash dedup vs valen_nano_v2/v3/v4 + real_context_eval_v1
    candidate files and intra-set. Determinism: seed 20261201, sorted
    iteration everywhere.

  * Output shape matches data/real_context_eval_v1/candidates.jsonl:
    {group_id, request:{state, questions.irrelevant}, targets:null,
     meta:{record_id=v5f2:..., supp_source, proposed_label, proposal_note,
           pair_id, ...}}. PROPOSALS ONLY — pending the owner/agent
    adjudication pass.

Usage: python3 scripts/mine_v5_hard_negatives_v1.py
Output: data/v5_mining/f2_hard_negative_candidates.jsonl
        data/v5_mining/f2_report.json
Self-check + tests: scripts/test_mine_v5_v1.py
"""

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

OUT_DIR = ROOT / "data" / "v5_mining"
OUT_JSONL = OUT_DIR / "f2_hard_negative_candidates.jsonl"
OUT_REPORT = OUT_DIR / "f2_report.json"
EVAL_V1_CANDIDATES = ROOT / "data" / "real_context_eval_v1" / "candidates.jsonl"
EVAL_V1_SUPP = ROOT / "data" / "real_context_eval_v1" / "candidates_drop_supp_v1.jsonl"

SEED = 20261201            # v5 reserved seed (V5_DATA_DESIGN_V1 §1.3)
STATE_CHAR_BUDGET = 12_000  # v1 miner conservative stand-in for 7000 packed tok
STATE_CHAR_HARD = 14_000    # ~8192 packed-token compiler limit with headroom
MAX_CANDIDATE_BYTES = 32 * 1024  # <=32KB candidate segment cap (v1 convention)
GROUP_LINEAGE = "real_context_v5"

# v1's 1MB MAX_STATE_BYTES exists because eval-v1 emitted the *whole*
# transcript as one state; mining emits windowed per-candidate states, so the
# extraction-time byte cap is lifted (segment hard cap + credential scan
# unchanged). Documented in the report caveats.
MINING_MAX_STATE_BYTES = 12_000_000

MIN_SEGMENTS = builder.MIN_SEGMENTS          # 12 — v1 admission convention
MIN_USER_MESSAGES = builder.MIN_USER_MESSAGES  # 1

EVAL_POOL_TARGET = 18      # spec: ~15–20 transcripts reserved for v5-eval mining
EVAL_POOL_MAX_FRACTION = 1 / 3  # never take more than a third of a scarce pool

Y, N, U = "yes", "no", "uncertain"  # proposed_label values, v1 convention

# caps (all deterministic)
MAX_PER_BOUNDARY = 8        # immediate-span candidates per shift boundary
MAX_DEEP_PER_BOUNDARY = 3   # earlier-than-preceding-span candidates per boundary
MAX_DEEP_REANCHOR = 10      # earlier-span candidates vs the final request
MAX_STALE_PER_TRANSCRIPT = 6
MAX_DUP_GROUPS = 4          # boilerplate dup groups per transcript
MAX_DUP_PER_GROUP = 3       # occurrences per dup group
MAX_FOREIGN_TAIL = 8        # foreign-tail segments per transcript
MAX_KEEP_SIDE = 12          # extra keep-side reach-back proposals per transcript
MAX_PER_TRANSCRIPT = 50     # total mined rows per transcript

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


# --- windowing ----------------------------------------------------------------
# Same policy as mine_drop_supp_v1.py (keep candidate+anchor, expand
# nearest-to-candidate, elision markers for dropped ranges), extended with a
# `blocked` set: credential-flagged segments are never kept (they count
# toward elided ranges). Sizes are estimated from per-entry serialized
# lengths (exact entry length + a conservative marker allowance) so the
# greedy pass is O(n) per proposal; the emitted conversation is then
# serialized exactly and must pass the hard byte cap.

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


def _entry_bytes(i, seg):
    return len(builder.serialized(
        {"pointer": f"/messages/{i}/content", "role": seg["role"],
         "content": seg["text"]}).encode("utf-8"))


MARKER_BYTES_EST = 110  # '{"pointer":"/elided/0","role":"control","content":"<elided 0000 earlier segments>"}' + comma headroom


def windowed_conversation(segs, candidate_idx, anchor_idx, user_list,
                          blocked=frozenset()):
    """Prefix window: keep candidate+anchor, expand nearest-to-candidate while
    the estimated serialized state stays under STATE_CHAR_BUDGET. `blocked`
    segments are never kept. Returns (conversation, windowed)."""
    n = len(segs)
    fixed = len(builder.serialized({
        "conversation": [],
        "candidate_pointer": f"/messages/{candidate_idx}/content",
        "user_messages_in_order": user_list}).encode("utf-8"))
    costs = [_entry_bytes(i, s) if i not in blocked else 0
             for i, s in enumerate(segs)]
    keepable = {i for i in range(n) if i not in blocked}

    def est(kept):
        # one marker per kept-run boundary is a safe upper bound
        return fixed + sum(costs[i] for i in kept) + \
            (len(kept) + 1) * MARKER_BYTES_EST

    if est(keepable) <= STATE_CHAR_BUDGET:
        return render_window(segs, keepable), len(keepable) != n
    kept = {candidate_idx, anchor_idx}
    optional = sorted((i for i in keepable if i not in kept),
                      key=lambda i: (abs(i - candidate_idx), i))
    estimate = est(kept)
    for i in optional:
        add = costs[i] + MARKER_BYTES_EST
        if estimate + add <= STATE_CHAR_BUDGET:
            kept.add(i)
            estimate += add
    conv = render_window(segs, kept)
    # exact check; shed farthest non-must segments until the real
    # serialized size fits the hard cap
    def exact(k):
        c = render_window(segs, k)
        return c, len(builder.serialized({
            "conversation": c,
            "candidate_pointer": f"/messages/{candidate_idx}/content",
            "user_messages_in_order": user_list}).encode("utf-8"))
    conv, size = exact(kept)
    must = {candidate_idx, anchor_idx}
    while size > STATE_CHAR_BUDGET:
        farthest = max((i for i in kept - must),
                       key=lambda i: (abs(i - candidate_idx), i), default=None)
        if farthest is None:
            break
        kept.discard(farthest)
        conv, size = exact(kept)
    return conv, kept != set(range(n))


# --- eval-v1 frozen pool ------------------------------------------------------

def load_eval_v1_hashes():
    """Transcript hashes admitted to real_context_eval_v1 (the 60 frozen
    group_ids). These transcripts are never mined."""
    hashes = set()
    for line in EVAL_V1_CANDIDATES.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        gid = rec.get("group_id", "")
        if ":" in gid:
            hashes.add(gid.split(":", 1)[1])
    return hashes


# --- dedup reference hashes ---------------------------------------------------

def load_reference_hashes():
    """request_sha256 + normalized-state hashes from prior corpora, per the
    v1 dedup convention extended to v4 + real_context_eval_v1 candidates."""
    request_hashes, state_hashes = set(), set()
    dirs = [ROOT / "data" / "valen_nano_v2", ROOT / "data" / "valen_nano_v3",
            ROOT / "data" / "valen_nano_v4"]
    names = ("train.jsonl", "eval.jsonl", "dev.jsonl")
    files = []
    for d in dirs:
        for name in names:
            p = d / name
            if p.is_file():
                files.append(p)
    for p in (EVAL_V1_CANDIDATES, EVAL_V1_SUPP):
        if p.is_file():
            files.append(p)
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if "request" not in record:
                continue
            request_hashes.add(
                builder.sha256_bytes(
                    builder.serialized(record["request"]).encode("utf-8")))
            shash = builder.normalized_state_hash(record)
            if shash:
                state_hashes.add(shash)
    return request_hashes, state_hashes


# --- transcript scan (builder thresholds, lifted extraction byte cap) ---------

def scan_transcripts():
    """All source transcripts meeting the builder's minimum quality
    thresholds. Returns (usable, drops, scanned). The 1MB full-state byte cap
    is lifted to MINING_MAX_STATE_BYTES because mining emits windowed
    per-candidate states; every other v1 admission rule is unchanged."""
    builder.MAX_STATE_BYTES = MINING_MAX_STATE_BYTES
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
        if len(t.segments) < MIN_SEGMENTS:
            drops["min_segments"] += 1
            continue
        if len(t.users) < MIN_USER_MESSAGES:
            drops["no_user_messages"] += 1
            continue
        if not builder.eligible_candidate_indices(t.segments):
            drops["no_candidates"] += 1
            continue
        # Same credential scan PATTERNS as v1, enforced at segment/user
        # granularity (windowed mining): flagged segments are never
        # candidates and never enter a context window; a flagged user
        # message disqualifies every anchor at/after it because
        # user_messages_in_order cannot elide. The emitted state of every
        # record is scanned again in self_check.
        flagged = {i for i, s in enumerate(t.segments)
                   if builder.credential_scan(s["text"])}
        flagged_users = {j for j, u in enumerate(t.users)
                         if builder.credential_scan(u)}
        first_flagged_user = (min(flagged_users) if flagged_users
                              else len(t.users))
        if first_flagged_user == 0:
            drops["credential_scan:first_user_message"] += 1
            continue
        clean_eligible = [i for i in
                          builder.eligible_candidate_indices(t.segments)
                          if i not in flagged]
        if not clean_eligible:
            drops["credential_scan:no_clean_candidates"] += 1
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
                       "path": str(path), "transcript": t,
                       "nseg": len(t.segments), "nusers": len(t.users),
                       "flagged": flagged,
                       "first_flagged_user": first_flagged_user})
    usable.sort(key=lambda u: (u["source"], u["transcript_hash"]))
    return usable, drops, len(files)


def split_pools(usable, eval_hashes):
    """Deterministic three-way ledger:
      eval_v1_frozen — the 60 transcripts eval v1 admitted (never mined);
      v5_eval_reserved — hash-sorted preference for longer / more task
        switches (score = n_users, then n_segments, then hash);
      train_pool — everything else; the only pool mined here.
    """
    frozen = [u for u in usable if u["transcript_hash"] in eval_hashes]
    remaining = [u for u in usable if u["transcript_hash"] not in eval_hashes]
    scored = sorted(remaining,
                    key=lambda u: (-u["nusers"], -u["nseg"],
                                   u["transcript_hash"]))
    if remaining:
        k = min(EVAL_POOL_TARGET,
                max(1, int(len(remaining) * EVAL_POOL_MAX_FRACTION)))
    else:
        k = 0
    reserved = scored[:k]
    reserved_hashes = {u["transcript_hash"] for u in reserved}
    train = sorted((u for u in remaining
                    if u["transcript_hash"] not in reserved_hashes),
                   key=lambda u: u["transcript_hash"])
    return {"eval_v1_frozen": frozen, "v5_eval_reserved": reserved,
            "train_pool": train}


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


def find_superseded_chains(t, span_lo, span_hi):
    """Read/cat/list tool_result superseded by a LATER read of the same path
    OR an edit of that path inside [span_lo, span_hi).
    Returns [(result_idx, path, supersede_use_idx, supersede_op)]."""
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
        sup = next(((u, o2) for (u, o2, p2) in ops[oi + 1:]
                    if o2 in ("read", "list", "edit")
                    and (p2 == p or (op_i == "list" and
                                     p2.startswith(p.rstrip("/") + "/")))),
                   None)
        if sup is None:
            continue
        res = next((k for k in range(use_i + 1, min(span_hi, sup[0] + 1))
                    if segs[k]["kind"] == "tool_result"), None)
        if res is not None:
            chains.append((res, p, sup[0], sup[1]))
    return chains


def mine_transcript(item, skips):
    """Collect mined proposals for one transcript. Returns list of proposal
    dicts {seg, anchor_seg, anchor_uidx, cut, strategy, side, label, note,
    evidence, extra}."""
    t = item["transcript"]
    segs = t.segments
    n_users = len(t.users)
    flagged = item["flagged"]
    ffu = item["first_flagged_user"]  # anchors allowed only for a < ffu
    user_seg = [i for i, s in enumerate(segs) if s["kind"] == "user_turn"]
    if not user_seg:
        skips["no_user_turns"] += 1
        return []
    last_user_seg = user_seg[-1]

    def ok(i):
        return i not in flagged and eligible_mine_kinds(segs[i])
    norm_segs = [norm(s["text"]).lower() for s in segs]
    subj = [subject_tokens(s["text"]) for s in segs]
    n = len(segs)

    # Suffix unions of subject tokens: suf_strong[k] = union over segs >= k.
    suf_strong, suf_weak = [set() for _ in range(n + 1)], \
                           [set() for _ in range(n + 1)]
    st_acc, wk_acc = set(), set()
    for k in range(n - 1, -1, -1):
        st_acc = st_acc | subj[k][0]
        wk_acc = wk_acc | subj[k][1]
        suf_strong[k], suf_weak[k] = st_acc, wk_acc

    # Joined normalized text with per-index char offsets, so "does text from
    # index `start` onward contain X" is a single str.find from the offset.
    # off[k] = join length of all matching segs with index < k.
    assistant_parts, assistant_off, pos = [], [], 0
    all_parts, all_off, apos = [], [], 0
    for k, s in enumerate(segs):
        assistant_off.append(pos)
        if s["kind"] == "assistant_text":
            assistant_parts.append(norm_segs[k])
            pos += len(norm_segs[k]) + 1
        all_off.append(apos)
        all_parts.append(norm_segs[k])
        apos += len(norm_segs[k]) + 1
    assistant_join = " ".join(assistant_parts)
    all_join = " ".join(all_parts)
    assistant_off.append(pos)
    all_off.append(apos)

    def post_subj(start):
        return suf_strong[min(start, n)], suf_weak[min(start, n)]

    def post_all_text_contains(needle, start):
        return all_join.find(needle, all_off[min(start, n)]) >= 0

    def reach_back(i, start):
        """Evidence that segment i is reached back to at/after index start.
        Returns (evidence_kind, detail) or None."""
        start = min(start, n)
        sigs = signatures(norm_segs[i])
        if sigs and any(assistant_join.find(sig, assistant_off[start]) >= 0
                        for sig in sigs):
            return ("output_quoted_in_later_assistant", sigs[0][:60])
        st, wk = subj[i]
        strong_hits = st & suf_strong[start]
        if strong_hits:
            return ("path_rereferenced_downstream", sorted(strong_hits)[0])
        weak_hits = wk & suf_weak[start]
        if len(weak_hits) >= 4:
            return ("identifiers_rereferenced_downstream",
                    ",".join(sorted(weak_hits)[:4]))
        return None

    proposals = {}  # (seg_idx, anchor_seg) -> proposal, first strategy wins

    def put(i, a_uid, strategy, side, label, note, evidence=None, extra=None):
        a_seg = user_seg[a_uid]
        if i == a_seg:
            return
        key = (i, a_seg)
        if key in proposals:
            return
        cut = user_seg[a_uid + 1] if a_uid + 1 < n_users else len(segs)
        proposals[key] = {
            "seg": i, "anchor_seg": a_seg, "anchor_uidx": a_uid, "cut": cut,
            "strategy": strategy, "side": side, "label": label, "note": note,
            "evidence": evidence, "extra": extra or {}}

    # --- 1) stale_superseded ---------------------------------------------------
    stale_count = 0
    for j in range(n_users):
        span_lo = user_seg[j] + 1
        span_hi = user_seg[j + 1] if j + 1 < n_users else len(segs)
        for res_idx, path, sup_idx, sup_op in \
                find_superseded_chains(t, span_lo, span_hi):
            if stale_count >= MAX_STALE_PER_TRANSCRIPT:
                break
            if j + 1 < n_users:
                a_uid = j + 1
            else:
                a_uid = n_users - 1
            if a_uid >= ffu:
                skips["cred_anchor_blocked"] += 1
                continue
            if res_idx in flagged:
                skips["cred_candidate_blocked"] += 1
                continue
            a_seg = user_seg[a_uid]
            pl = path.lower()
            # keep-side complement: superseded-looking output that is still
            # quoted/referenced after the superseding op or after the anchor.
            rb = reach_back(res_idx, min(sup_idx + 1, len(segs)))
            if rb is not None:
                label, side = N, "keep"
                note = (f"stale-looking read of {path} superseded by a later "
                        f"{sup_op} of the same path, but reach-back evidence "
                        f"({rb[0]}: {rb[1]}) shows it is still referenced; "
                        f"hard keep for the wrong-direction prior")
                ev = {"reach_back": rb[0], "reach_back_detail": rb[1]}
            elif j + 1 >= n_users:
                label, side = U, "drop"
                note = (f"read/list of {path} superseded by a later {sup_op} "
                        f"of the same path inside the FINAL task; R4 "
                        f"'unreferenced post-shift' is unverifiable without "
                        f"a later task")
                ev = None
            elif post_all_text_contains(pl, a_seg + 1) or any(
                    pl in tok or tok in pl for tok in post_subj(a_seg + 1)[0]):
                label, side = U, "drop"
                note = (f"stale read/list of {path} superseded by a later "
                        f"{sup_op} in task span, but the path is referenced "
                        f"post-shift; R4 unreferenced check fails")
                ev = None
            else:
                label, side = Y, "drop"
                note = (f"read/list of {path} superseded by a later {sup_op} "
                        f"of the same path inside one task; no post-shift "
                        f"reference (R4 stale rule)")
                ev = None
            put(res_idx, a_uid, "stale_superseded", side, label, note,
                evidence=ev,
                extra={"superseded_path": path,
                       "superseded_by": sup_op,
                       "supersede_pointer": f"/messages/{sup_idx}/content"})
            stale_count += 1

    # --- 2) cross_task_reanchor -------------------------------------------------
    def cross_label(i, a_uid):
        a_seg = user_seg[a_uid]
        rb = reach_back(i, a_seg + 1)
        if rb is not None:
            return (N, "keep",
                    (f"post-shift reach-back evidence ({rb[0]}: {rb[1]}); the "
                     f"earlier-task segment is still used after the task "
                     f"switch — keep-side hard negative"),
                    {"reach_back": rb[0], "reach_back_detail": rb[1]})
        if segs[i]["kind"] == "user_turn":
            return (U, "drop",
                    "pre-shift user turn reanchored at a later task; may "
                    "carry a persistent constraint/correction (R2/R3 "
                    "caution — adjudication decides)", None)
        if ack_like(segs[i]["text"]):
            return (Y, "drop",
                    "pure acknowledgement / low-content assistant text "
                    "pre-shift; carries no evidence (R6)", None)
        cand_strong, cand_weak = subj[i]
        post_strong, post_weak = post_subj(a_seg)
        anchor_strong, anchor_weak = subject_tokens(t.users[a_uid])
        marker = bool(ABANDON_RE.search(t.users[a_uid]))
        strong_hits = cand_strong & (post_strong | anchor_strong)
        weak_hits = cand_weak & (post_weak | anchor_weak)
        if strong_hits or len(weak_hits) >= 2:
            hits = sorted(strong_hits | weak_hits)[:4]
            base = (f"candidate shares subject tokens {hits} with "
                    f"anchor/post-shift content; same-repo continuity, "
                    f"reach-back unverifiable (R5.2)")
            if marker:
                base += "; explicit abandon marker in anchor keeps this a " \
                        "candidate but cannot prove no-reach-back"
            return (U, "drop", base, None)
        if marker:
            return (Y, "drop",
                    "explicit task-abandon/pivot marker in anchor (mirrors "
                    "v1 'ok算了' adjudication); disjoint subject vocabulary, "
                    "no post-shift reference", None)
        return (Y, "drop",
                "pre-shift segment reanchored at a later user task; disjoint "
                "subject vocabulary, no post-shift reference detected "
                "(auto R5 approximation — needs adjudication)", None)

    for a in range(1, min(n_users, ffu)):
        a_seg = user_seg[a]
        prev_seg = user_seg[a - 1]
        if a_seg - prev_seg <= 1:
            skips["degenerate_boundary"] += 1
            continue
        if is_continuation_anchor(t.users[a]):
            skips["continuation_anchor"] += 1
            continue
        span = [i for i in range(prev_seg + 1, a_seg) if ok(i)]
        deep = [i for i in range(0, prev_seg + 1) if ok(i)]
        cand_idx = (evenly_spaced(span, MAX_PER_BOUNDARY)
                    + evenly_spaced(deep, MAX_DEEP_PER_BOUNDARY))
        for i in sorted(set(cand_idx)):
            label, side, note, ev = cross_label(i, a)
            put(i, a, "cross_task_reanchor", side, label, note,
                evidence=ev)

    # deep reanchor: earlier spans judged vs the FINAL user request (v1's
    # "all pre-shift vs u_last" pattern); only when >=3 user turns and every
    # user message is credential-clean (final anchor includes all users).
    if n_users >= 3 and ffu == n_users:
        a = n_users - 1
        deep = [i for i in range(0, user_seg[a - 1]) if ok(i)]
        for i in evenly_spaced(deep, MAX_DEEP_REANCHOR):
            label, side, note, ev = cross_label(i, a)
            if segs[i]["kind"] == "user_turn":
                label, side, ev = U, "drop", None
                note = ("pre-shift user turn judged vs final anchor; may "
                        "carry a persistent constraint/correction (R2/R3)")
            put(i, a, "cross_task_reanchor", side, label,
                "deep reanchor: " + note, evidence=ev,
                extra={"deep_reanchor": True})

    # --- 3) same_anchor_tail ----------------------------------------------------
    # The final anchor's user_messages_in_order contains ALL user turns, so
    # it is only usable when no user message is credential-flagged.
    if ffu == n_users:
        a_last = n_users - 1
        anchor_strong, anchor_weak = subject_tokens(t.users[-1])
        resp_strong, resp_weak = set(), set()
        for i in range(last_user_seg + 1, min(len(segs), last_user_seg + 6)):
            resp_strong |= subj[i][0]
            resp_weak |= subj[i][1]

        # (a) digit-normalized duplicate boilerplate groups
        groups = defaultdict(list)
        for i, s in enumerate(segs):
            if s["kind"] in ("tool_result", "assistant_text") \
                    and i != last_user_seg and i not in flagged:
                key = boilerplate_key(s["text"])
                if 4 <= len(key) <= 400:
                    groups[key].append(i)
        dup_groups = sorted((g for g in groups.values() if len(g) >= 2),
                            key=lambda g: g[0])
        for g in dup_groups[:MAX_DUP_GROUPS]:
            for i in g[:MAX_DUP_PER_GROUP]:
                rb = reach_back(i, i + 1)
                if rb is not None:
                    put(i, a_last, "same_anchor_tail", "keep", N,
                        (f"repeated boilerplate occurrence, but reach-back "
                         f"evidence ({rb[0]}: {rb[1]}) shows this occurrence is "
                         f"still referenced — keep-side complement"),
                        evidence={"reach_back": rb[0], "reach_back_detail": rb[1]},
                        extra={"dup_group_size": len(g)})
                else:
                    put(i, a_last, "same_anchor_tail", "drop", U,
                        (f"identical (digit-normalized) boilerplate repeated "
                         f"x{len(g)} under the final anchor; v1 adjudicated "
                         f"this pattern uncertain"),
                        extra={"dup_group_size": len(g)})

        # (b) foreign coherent threads anywhere under the final anchor +
        #     keep-side on-task tail segments
        foreign = []
        for i, s in enumerate(segs):
            if i == last_user_seg or not ok(i):
                continue
            st, wk = subj[i]
            if not (st or wk):
                continue
            if st & (anchor_strong | resp_strong) or \
                    wk & (anchor_weak | resp_weak):
                continue
            foreign.append(i)
        run_ok = {i for i in foreign
                  if any(abs(i - j) <= 2 for j in foreign if j != i)}
        for i in foreign[:MAX_FOREIGN_TAIL]:
            rb = reach_back(i, i + 1)
            if rb is not None:
                put(i, a_last, "same_anchor_tail", "keep", N,
                    (f"foreign-looking segment under the final anchor, but "
                     f"reach-back evidence ({rb[0]}: {rb[1]}) — keep-side "
                     f"complement"),
                    evidence={"reach_back": rb[0], "reach_back_detail": rb[1]},
                    extra={"foreign_thread": True})
            elif i in run_ok:
                put(i, a_last, "same_anchor_tail", "drop", Y,
                    ("foreign-task thread under the final anchor: subject "
                     "vocabulary disjoint from the anchor request and its "
                     "immediate response, coherent run (auto-detected, mirrors "
                     "v1 drop labels seg0025/seg0027 pattern)"),
                    extra={"foreign_thread": True})
            else:
                put(i, a_last, "same_anchor_tail", "drop", U,
                    ("isolated foreign-looking segment under the final anchor; "
                     "no coherent thread detected"),
                    extra={"foreign_thread": True})

        # (c) on-task tail segments that look boilerplate but share anchor
        #     vocabulary — explicit keep-side tail complement
        tail = [i for i in range(last_user_seg + 1, len(segs)) if ok(i)]
        on_task = 0
        for i in tail:
            if on_task >= 4:
                break
            st, wk = subj[i]
            if st & (anchor_strong | resp_strong):
                put(i, a_last, "same_anchor_tail", "keep", N,
                    ("post-anchor tail segment shares strong subject tokens with "
                     "the anchor request; boilerplate-shaped but on-task — "
                     "keep-side complement"),
                    evidence={"reach_back": "shares_anchor_path_tokens",
                              "reach_back_detail":
                                  ",".join(sorted(st & anchor_strong)[:3])},
                    extra={"on_task_tail": True})
                on_task += 1

    # --- 4) dedicated keep-side complement pass --------------------------------
    # Segments reached back to (quoted in later assistant text / path
    # re-referenced downstream) that no strategy already proposed. Anchor =
    # the next user turn after the segment when one exists (cross_task
    # frame, evidence must be POST-anchor: carry-over across the boundary),
    # else the final user turn for tail segments (same_anchor frame,
    # any later reference counts).
    keep_added = 0
    for i, s in enumerate(segs):
        if keep_added >= MAX_KEEP_SIDE:
            break
        if not ok(i) or i == last_user_seg:
            continue
        a_uid = next((k for k in range(n_users) if user_seg[k] > i), None)
        if a_uid is not None and a_uid >= 1 and \
                not (n_users >= 2 and a_uid == n_users - 1
                     and i > user_seg[n_users - 2]):
            # earlier-task segment under a later-task anchor: require
            # post-anchor reach-back (the cross-boundary carry-over claim).
            if a_uid >= ffu:
                continue
            rb = reach_back(i, user_seg[a_uid] + 1)
            strategy = "cross_task_reanchor"
        elif a_uid is not None:
            # inside the last task's own span (or before the first user):
            # same-anchor frame; the final/first anchor needs clean users.
            if a_uid == n_users - 1 and ffu != n_users:
                continue
            rb = reach_back(i, i + 1)
            strategy = "same_anchor_tail"
        elif i > last_user_seg:
            # tail segment under the existing final anchor
            if ffu != n_users:
                continue
            a_uid = n_users - 1
            rb = reach_back(i, i + 1)
            strategy = "same_anchor_tail"
        else:
            continue
        if rb is None:
            continue
        if (i, user_seg[a_uid]) in proposals:
            continue
        put(i, a_uid, strategy, "keep", N,
            (f"keep-side complement: reach-back evidence ({rb[0]}: {rb[1]}) — "
             f"the segment is quoted/referenced downstream of the anchor"),
            evidence={"reach_back": rb[0], "reach_back_detail": rb[1]})
        keep_added += 1

    # --- per-transcript cap: strategy priority, keep-side first, then seg -------
    prio = {"stale_superseded": 0, "cross_task_reanchor": 1,
            "same_anchor_tail": 2}
    ordered = sorted(proposals.values(),
                     key=lambda p: (prio[p["strategy"]],
                                    0 if p["side"] == "keep" else 1,
                                    p["seg"], p["anchor_seg"]))
    if len(ordered) > MAX_PER_TRANSCRIPT:
        skips["per_transcript_cap"] += len(ordered) - MAX_PER_TRANSCRIPT
        ordered = ordered[:MAX_PER_TRANSCRIPT]
    return ordered


def assign_pairs(emitted, th):
    """meta.pair_id assignment on EMITTED records (doc §1.7). Deterministic.
    Pairs are only valid if all members survive emit/dedup, so this runs
    post-emit. `emitted` = list of (record_dict, prop_dict).

    anchor_flip: same candidate_pointer under >=2 different anchors, with at
      least one yes and one no label -> pair_id v5f2pair:<h>:flip:segNNNN.
    same_anchor: same anchor_pointer with both yes and no members -> pair_id
      v5f2pair:<h>:same:aNNNN (only for members not already in a flip pair).
    """
    by_seg = defaultdict(list)
    for rec, prop in emitted:
        by_seg[prop["seg"]].append((rec, prop))
    for seg, members in sorted(by_seg.items()):
        anchors = {p["anchor_seg"] for _, p in members}
        labels = {p["label"] for _, p in members}
        if len(anchors) >= 2 and Y in labels and N in labels:
            pid = f"v5f2pair:{th[:12]}:flip:seg{seg:04d}"
            for rec, p in members:
                if p["label"] in (Y, N):
                    rec["meta"]["pair_id"] = pid
                    rec["meta"]["pair_kind"] = "anchor_flip"
    by_anchor = defaultdict(list)
    for rec, prop in emitted:
        by_anchor[prop["anchor_seg"]].append((rec, prop))
    for a_seg, members in sorted(by_anchor.items()):
        free = [(rec, p) for rec, p in members
                if not rec["meta"]["pair_id"] and p["label"] in (Y, N)]
        if len(free) >= 2 and {p["label"] for _, p in free} == {Y, N}:
            pid = f"v5f2pair:{th[:12]}:same:a{a_seg:04d}"
            for rec, p in free:
                rec["meta"]["pair_id"] = pid
                rec["meta"]["pair_kind"] = "same_anchor"


# --- emit ---------------------------------------------------------------------

def emit(item, prop, instructions, used_ids, skips):
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
    if idx in item["flagged"]:
        skips["cred_candidate_blocked"] += 1
        return None
    if len(seg["text"].encode("utf-8", errors="ignore")) > MAX_CANDIDATE_BYTES:
        skips["candidate_over_32kb"] += 1
        return None
    ptr = f"/messages/{idx}/content"
    anchor_ptr = f"/messages/{a_seg}/content"
    prefix = segs[:prop["cut"]]
    user_list = t.users[:a_uid + 1]
    conv, windowed = windowed_conversation(prefix, idx, a_seg, user_list,
                                           blocked=item["flagged"])
    state = {"conversation": conv, "candidate_pointer": ptr,
             "user_messages_in_order": user_list}
    state_str = builder.serialized(state)
    if len(state_str.encode("utf-8")) > STATE_CHAR_HARD:
        skips["exceeds_budget"] += 1
        return None
    # emit-level credential gate: scan every content component (NOT the
    # serialized blob — JSON escaping can fabricate pattern hits, e.g.
    # "<tab>@x.y" serializes to "\\t@x.y" and the email regex matches
    # "t@x.y"). self_check re-verifies with the same component scan.
    cred_parts = [s["content"] for s in conv] + list(user_list)
    if any(builder.credential_scan(part) for part in cred_parts):
        skips["cred_state_blocked"] += 1
        return None
    rid = f"v5f2:{th[:12]}:seg{idx:04d}:a{a_seg:04d}"
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
        "selection_method": f"mining:{prop['strategy']}:{prop['side']}",
        "supp_source": "mining",
        "mine_mode": prop["strategy"],
        "strategy": prop["strategy"],
        "mining_side": prop["side"],
        "anchor_user_index": a_uid,
        "anchor_pointer": anchor_ptr,
        "windowed": windowed,
        "state_chars": len(state_str.encode("utf-8")),
        "proposed_label": prop["label"],
        "proposal_note": prop["note"],
        "evidence_basis": prop["note"],
        "pair_id": None,   # assigned post-emit by assign_pairs
        "pool": "train",
    }
    if prop.get("evidence"):
        meta.update(prop["evidence"])
    meta.update(prop.get("extra", {}))
    return {
        "group_id": f"{GROUP_LINEAGE}:{th}",
        "request": {"state": state_str,
                    "questions": {"irrelevant": {"type": "noul",
                                               "instructions": instructions}}},
        "targets": None,
        "meta": meta,
    }


# --- self-check -----------------------------------------------------------------

def self_check(records, eval_hashes, reserved_hashes):
    """Hard validation of every emitted record. Raises AssertionError."""
    errors = []
    seen_ids = set()
    for r in records:
        rid = r["meta"]["record_id"]
        if rid in seen_ids:
            errors.append(f"{rid}: duplicate record_id")
        seen_ids.add(rid)
        try:
            state = json.loads(r["request"]["state"])
        except ValueError:
            errors.append(f"{rid}: unparseable state")
            continue
        ptr = state.get("candidate_pointer")
        conv = state.get("conversation", [])
        ptrs = {s.get("pointer") for s in conv}
        if ptr not in ptrs:
            errors.append(f"{rid}: candidate_pointer {ptr} not in conv")
        cand = next((s for s in conv if s.get("pointer") == ptr), None)
        if cand is not None and not cand.get("content", "").strip():
            errors.append(f"{rid}: empty candidate content")
        if cand is not None and \
                len(cand["content"].encode("utf-8")) > MAX_CANDIDATE_BYTES:
            errors.append(f"{rid}: candidate over 32KB")
        users = state.get("user_messages_in_order") or []
        if not users:
            errors.append(f"{rid}: empty user_messages_in_order")
            continue
        user_segs = [s for s in conv if s.get("role") == "user"]
        if not user_segs:
            errors.append(f"{rid}: no user seg in conv")
            continue
        last_user = user_segs[-1]
        if last_user.get("pointer") != r["meta"].get("anchor_pointer"):
            errors.append(f"{rid}: last user conv seg "
                          f"{last_user.get('pointer')} != anchor "
                          f"{r['meta'].get('anchor_pointer')}")
        if last_user.get("content") != users[-1]:
            errors.append(f"{rid}: last user seg content != "
                          f"user_messages_in_order[-1]")
        # component-level scan (serialization can fabricate pattern hits —
        # a literal "<tab>@x.y" becomes "\\t@x.y" inside the JSON string)
        for s in conv:
            hit = builder.credential_scan(s.get("content", ""))
            if hit is not None:
                errors.append(f"{rid}: credential_scan:{hit} in conv "
                              f"{s.get('pointer')}")
                break
        for u in users:
            hit = builder.credential_scan(u)
            if hit is not None:
                errors.append(f"{rid}: credential_scan:{hit} in "
                              f"user_messages_in_order")
                break
        th = r["meta"]["transcript_hash"]
        if th in eval_hashes:
            errors.append(f"{rid}: transcript in eval_v1 pool")
        if th in reserved_hashes:
            errors.append(f"{rid}: transcript in v5_eval_reserved pool")
        if not rid.startswith("v5f2:"):
            errors.append(f"{rid}: bad record_id prefix")
        if r["targets"] is not None:
            errors.append(f"{rid}: targets must stay null (proposals only)")
    if errors:
        raise AssertionError(f"self_check failed ({len(errors)}): "
                             + "; ".join(errors[:10]))
    return {"records_checked": len(records), "status": "pass"}


# --- main -----------------------------------------------------------------------

def main():
    rng = random.Random(SEED)
    instructions = builder.load_verbatim_instructions(ROOT / "data" / "valen_nano_v3")
    eval_hashes = load_eval_v1_hashes()
    ref_requests, ref_states = load_reference_hashes()
    usable, drops, scanned = scan_transcripts()
    pools = split_pools(usable, eval_hashes)
    reserved_hashes = {u["transcript_hash"] for u in pools["v5_eval_reserved"]}

    records, report_rows, skips = [], [], Counter()
    used_ids = set()
    seen_requests, seen_states = set(), set()
    dedup = Counter()
    for item in pools["train_pool"]:
        props = mine_transcript(item, skips)
        emitted = []
        for prop in props:
            rec = emit(item, prop, instructions, used_ids, skips)
            if rec is None:
                continue
            rhash = builder.sha256_bytes(
                builder.serialized(rec["request"]).encode("utf-8"))
            shash = builder.normalized_state_hash(rec)
            if rhash in ref_requests or shash in ref_states:
                dedup["vs_prior_corpora"] += 1
                continue
            if rhash in seen_requests or shash in seen_states:
                dedup["intra_set"] += 1
                continue
            seen_requests.add(rhash)
            seen_states.add(shash)
            emitted.append((rec, prop))
        # contrastive pair assignment on surviving records only (doc §1.7)
        assign_pairs(emitted, item["transcript_hash"])
        for rec, prop in emitted:
            records.append(rec)
            report_rows.append({
                "record_id": rec["meta"]["record_id"],
                "transcript": item["transcript_hash"][:12],
                "source": item["source"],
                "strategy": prop["strategy"],
                "side": prop["side"],
                "seg": prop["seg"],
                "anchor_seg": prop["anchor_seg"],
                "kind": rec["meta"]["candidate_kind"],
                "proposed_label": prop["label"],
                "pair_id": rec["meta"]["pair_id"],
                "note": prop["note"],
                "candidate_snippet": norm(
                    item["transcript"].segments[prop["seg"]]["text"])[:140],
                "anchor_snippet": norm(
                    item["transcript"].users[prop["anchor_uidx"]])[:140],
            })

    records.sort(key=lambda r: (r["meta"]["transcript_hash"],
                                r["meta"]["record_id"]))

    # mandatory self-check before anything is written
    check = self_check(records, eval_hashes, reserved_hashes)

    label_counts = Counter(r["meta"]["proposed_label"] for r in records)
    strat_counts = Counter(r["meta"]["strategy"] for r in records)
    strat_side = Counter((r["meta"]["strategy"], r["meta"]["mining_side"])
                         for r in records)
    strat_label = Counter((r["meta"]["strategy"], r["meta"]["proposed_label"])
                          for r in records)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in records:
            f.write(builder.serialized(r) + "\n")

    sample_idx = sorted(rng.sample(range(len(report_rows)),
                                   min(20, len(report_rows))))
    top20 = [report_rows[i] for i in sample_idx]

    def ledger_entry(u):
        return {"transcript_hash": u["transcript_hash"],
                "source": u["source"], "segment_count": u["nseg"],
                "user_message_count": u["nusers"],
                "cred_flagged_segments": len(u["flagged"])}

    report = {
        "schema_version": "nanojev-v5-f2-hardneg-v1",
        "builder": "scripts/mine_v5_hard_negatives_v1.py",
        "seed": SEED,
        "created_from": ("remaining real transcripts under ~/.claude/projects "
                         "+ ~/.codex/sessions after pool split (same roots/"
                         "thresholds as build_real_context_eval_v1.py; "
                         "extraction byte cap lifted to 12MB — records are "
                         "windowed)"),
        "note": ("PROPOSALS ONLY: targets stay null; proposed_label + "
                 "proposal_note + evidence_basis are deterministic-rule "
                 "suggestions pending the owner/agent adjudication pass. "
                 "Labels follow docs/REAL_CONTEXT_LABELING_GUIDE_V1.md "
                 "asymmetric semantics (yes == certainly irrelevant == "
                 "drop). No model/service calls."),
        "pool_split": {
            "rule": ("eval_v1 transcripts never mined; v5_eval_reserved = "
                     "top-K by (user turns, segments, hash) preferring "
                     "longer/more-task-switches, capped at "
                     f"min({EVAL_POOL_TARGET}, floor(pool/3)); train_pool = "
                     "remainder"),
            "requested_eval_reserve": "15-20",
            "actual_eval_reserve": len(pools["v5_eval_reserved"]),
            "eval_v1_frozen": {
                "count": len(eval_hashes),
                "transcript_hashes": sorted(eval_hashes)},
            "v5_eval_reserved": {
                "count": len(pools["v5_eval_reserved"]),
                "transcripts": [ledger_entry(u)
                                for u in pools["v5_eval_reserved"]]},
            "train_pool": {
                "count": len(pools["train_pool"]),
                "transcripts": [ledger_entry(u)
                                for u in pools["train_pool"]]},
            "eval_transcripts_seen_but_not_mined":
                len(pools["eval_v1_frozen"]),
        },
        "mining_strategies": {
            "cross_task_reanchor": ("earlier-task segment under a later-task "
                                    "anchor (per boundary: preceding span + "
                                    "deep sample; plus deep reanchor vs "
                                    "final request)"),
            "stale_superseded": ("read/cat/list tool_result superseded by a "
                                 "later read of the same path OR an edit "
                                 "inside one task span; drop when stale and "
                                 "unreferenced post-shift"),
            "same_anchor_tail": ("foreign-task boilerplate under the "
                                 "existing final anchor (dup groups + "
                                 "coherent foreign threads)"),
            "keep_side_complement": ("same strategies, mining segments that "
                                     "ARE reached back to (path "
                                     "re-referenced / output quoted in "
                                     "later assistant text) -> "
                                     "proposed_label=no"),
        },
        "self_check": check,
        "dedup": {
            "reference_sets": ["data/valen_nano_v2", "data/valen_nano_v3",
                               "data/valen_nano_v4",
                               "data/real_context_eval_v1/candidates*.jsonl"],
            **dict(sorted(dedup.items())),
        },
        "counts": {
            "files_scanned": scanned,
            "usable_transcripts": len(usable),
            "unusable_drops": dict(sorted(drops.items())),
            "records": len(records),
            "proposed_labels": dict(sorted(label_counts.items())),
            "by_strategy": dict(sorted(strat_counts.items())),
            "by_strategy_side": {f"{k[0]}:{k[1]}": v
                                 for k, v in sorted(strat_side.items())},
            "by_strategy_label": {f"{k[0]}:{k[1]}": v
                                  for k, v in sorted(strat_label.items())},
            "by_kind": dict(Counter(r["meta"]["candidate_kind"]
                                    for r in records)),
            "by_source": dict(Counter(r["meta"]["source_root"]
                                      for r in records)),
            "windowed": sum(1 for r in records if r["meta"]["windowed"]),
            "paired_records": sum(1 for r in records
                                  if r["meta"]["pair_id"]),
            "pairs": len({r["meta"]["pair_id"] for r in records
                          if r["meta"]["pair_id"]}),
            "skips": dict(sorted(skips.items())),
        },
        "spot_review_sample_20": top20,
        "rows": report_rows,
        "caveats": [
            (f"pool split: {len(usable)} usable transcripts, "
             f"{len(eval_hashes)} of them the frozen eval-v1 set; "
             f"reserved {len(pools['v5_eval_reserved'])} for v5-eval mining "
             f"(target 15-20, capped at floor(pool/3)); "
             f"{len(pools['train_pool'])} train-pool transcripts mined."),
            ("credential-scan enforcement adapted to windowed mining: the "
             "same SECRET_PATTERNS run per segment and per user message; "
             "flagged segments are never candidates and never enter a "
             "context window; a flagged user message disables every anchor "
             "at/after it (user_messages_in_order cannot elide); every "
             "emitted state is re-scanned in self_check. This replaces v1's "
             "whole-transcript drop, which exists only because eval-v1 "
             "emitted whole-transcript states — transcript-level drops "
             "would flag 85/351 files including nearly all long multi-task "
             "sessions for no privacy gain in the emitted records."),
            ("extraction byte cap lifted from v1's 1MB to 12MB: eval-v1 "
             "emitted whole-transcript states; mining emits windowed "
             "per-candidate states so the per-transcript cap is "
             "inapplicable. Segment hard cap (4000) and all other admission "
             "rules unchanged."),
            ("unlike mine_v5_hardneg_v1.py (pair-level holdout over the 60 "
             "eval transcripts), this miner excludes eval-v1 transcripts "
             "entirely, per V5_DATA_DESIGN_V1 §1.6 and the pool-disjointness "
             "test requirement."),
            ("proposals are deterministic heuristics, not labels: the "
             "adjudication pass must still verify R5's four checks before "
             "any training use; keep-side rows cite reach-back evidence as "
             "their evidence_basis."),
            ("codex read/edit detection parses shell command strings "
             "(sed/cat/apply_patch) — approximate; claude Read/Edit dict "
             "inputs are exact."),
            ("~1500-3000 raw-candidate target depends on pool size; with a "
             "single-digit train pool the yield is bounded — the shortfall "
             "is reported rather than widening scope (project norm). "
             "~/.codex/archived_sessions holds ~107 additional rollout "
             "files outside the sanctioned roots if the owner later "
             "authorizes a wider pool."),
        ],
    }
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"scanned {scanned} files -> {len(usable)} usable transcripts")
    print(f"pools: frozen_eval={len(eval_hashes)} "
          f"reserved={len(pools['v5_eval_reserved'])} "
          f"train={len(pools['train_pool'])}")
    print(f"wrote {len(records)} records -> {OUT_JSONL}")
    print("labels:", dict(sorted(label_counts.items())),
          "| strategies:", dict(sorted(strat_counts.items())))
    print("skips:", dict(sorted(skips.items())),
          "| drops:", dict(sorted(drops.items())),
          "| dedup:", dict(sorted(dedup.items())))
    print(f"report -> {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
