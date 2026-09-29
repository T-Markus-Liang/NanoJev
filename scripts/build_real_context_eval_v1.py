#!/usr/bin/env python3
"""Collect unlabeled keep/drop CANDIDATE records from real local agent transcripts.

Implements the collection+redaction half of docs/REAL_CONTEXT_EVAL_V1.md:

  - Sources: ~/.claude/projects/**/*.jsonl (Claude Code session events) and
    ~/.codex/sessions/**/*.jsonl (Codex rollout events), when present.
  - Isolation: one transcript file = one group_id
    ("real_context_eval_v1:<sha256(file bytes)>"); a transcript is never split.
  - Segment policy (declared): each source event is flattened into ordered
    segments with pointers /messages/<i>/content.
      * Claude Code: user string/text events (skipping isMeta /
        isCompactSummary) -> role "user"; assistant text blocks ->
        "assistant"; tool_use blocks -> "assistant" tool_use segments;
        tool_result blocks -> "tool". Thinking blocks, attachments,
        file-history, mode and other bookkeeping events are skipped.
      * Codex rollouts: only top-level type "response_item" is read
        (event_msg duplicates the same payloads). message role=user ->
        "user", role=assistant -> "assistant"; function_call /
        custom_tool_call / web_search_call / image_generation_call ->
        tool_use segments; function_call_output / custom_tool_call_output
        -> "tool". Encrypted reasoning is never extracted.
      * Harness-injected scaffolding is NOT conversation and is excluded at
        extraction (never becomes a segment): codex developer/system
        boilerplate (<permissions instructions>, <app-context>) and
        machine-wrapped user parts (<environment_context>,
        <codex_internal_context>, <turn_aborted>, <subagent_notification>,
        <recommended_plugins>, <in-app-browser-context>, image placeholders,
        '# AGENTS.md instructions', '# Files mentioned', '# In app
        browser'); claude machine-wrapped user text (<command-name>,
        <local-command-*>, <task-notification>, <bash-*>, ide tags).
  - Candidate pointers: segments of kind assistant_text, tool_result, and
    non-final user_turn (the final user message anchors the request and is
    never a candidate). tool_use/control/system segments are context only.
    Cap of <=25 candidates per transcript, stratified across early/mid/late
    position thirds, deterministic.
  - Stratification across conversation-length bands
    (s=12-49, m=50-149, l=150-399, xl>=400 segments): bands are filled
    round-robin so the admitted transcript mix covers all bands.
  - Redaction BEFORE anything is written: a deterministic credential scan
    (API keys, bearer/private-key/JWT blobs, emails, absolute paths naming
    secret-bearing files, long base64/hex blobs) over the extracted segment
    text; any hit drops the ENTIRE transcript. Transcripts under the minimum
    segment count or whose serialized state exceeds the byte cap are dropped.
  - Near-duplicate screening: canonical request_sha256 and a normalized-text
    state hash are checked against data/valen_nano_v2/v3 (train+eval) and
    intra-set; collisions are rejected.
  - Output: data/real_context_eval_v1/candidates.jsonl (targets: null —
    unlabeled, for the manual labeling pass) plus a content-free
    manifest.json (counts, sha256s, segment stats, drop counts only;
    data/* is gitignored).

No provider calls, no scorer calls, no active filtering. Raw text stays
under the gitignored data root; the manifest carries hashes and counts only.

Usage: python3 scripts/build_real_context_eval_v1.py [--limit N] [--print-examples K]
"""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLAUDE_ROOT = Path.home() / ".claude" / "projects"
DEFAULT_CODEX_ROOT = Path.home() / ".codex" / "sessions"
DEFAULT_VALEN_V3 = ROOT / "data" / "valen_nano_v3"
DEFAULT_VALEN_V2 = ROOT / "data" / "valen_nano_v2"
DEFAULT_OUT = ROOT / "data" / "real_context_eval_v1"

MANIFEST_SCHEMA = "nanojev-real-context-eval-v1"
GROUP_LINEAGE = "real_context_eval_v1"

# --- collection policy (declared, deterministic) -----------------------------
MIN_SEGMENTS = 12          # drop transcripts with fewer extracted segments
MIN_USER_MESSAGES = 1      # need at least one real user turn to anchor a request
MAX_STATE_BYTES = 1_000_000  # serialized conversation byte cap -> drop transcript
MAX_SEGMENTS_HARD = 4_000  # safety valve while streaming huge transcripts
MAX_CANDIDATES_PER_TRANSCRIPT = 25
MAX_TRANSCRIPTS = 60
TARGET_TOTAL_RECORDS = 600
LENGTH_BANDS = (("s", 12, 49), ("m", 50, 149), ("l", 150, 399), ("xl", 400, 10**9))
BAND_ORDER = ("xl", "l", "m", "s")  # round-robin order; largest first for coverage

# --- token budget (valen compiler: valen/data/compilers/qwen.py) -------------
# The compiler packs chat-template(state string) + fixed noul suffix and raises
# when the total exceeds max_length=8192. We target a lower bound so estimates
# have headroom.
DEFAULT_TOKENIZER = ROOT / "external" / "valen" / "models" / "Qwen3.5-0.8B"
TOKEN_BUDGET = 7_000     # packed-token target per emitted record
MAX_PACKED_TOKENS = 8_192  # compiler hard limit (never exceeded)

CANDIDATE_KINDS = ("assistant_text", "tool_result", "user_turn")

# Machine-wrapped pseudo-user text (Claude Code / Codex injects these as user
# role, but they are not real user requests). Matched on the stripped prefix.
CLAUDE_CONTROL_PREFIXES = (
    "<local-command-caveat", "<local-command-stdout", "<local-command-stderr",
    "<command-name", "<command-message", "<command-args", "<command-contents",
    "<task-notification", "<bash-input", "<bash-stdout", "<bash-stderr",
    "<ide_opened_file", "<ide_selection",
)
CODEX_CONTROL_PREFIXES = (
    "<codex_internal_context", "<environment_context", "<turn_aborted",
    "<recommended_plugins", "<subagent_notification", "<in-app-browser-context",
    "<image", "</image", "<permissions", "<app-context", "<collaboration",
    "# AGENTS.md instructions", "# Files mentioned", "# In app browser",
)

# --- deterministic credential scan (extends real_context_holdout_manifest_v1)
SECRET_PATTERNS = [
    ("apikey", re.compile(r"apikey_[A-Za-z0-9_-]{8,}")),
    ("openai_style_key", re.compile(r"\bsk-[A-Za-z0-9_-]{12,}")),
    ("bearer", re.compile(r"Bearer\s+[A-Za-z0-9._-]{12,}")),
    ("private_key_block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("github_token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr|gho|ghu)_[A-Za-z0-9]{16,}|github_pat_[A-Za-z0-9_]{16,}")),
    ("slack_token", re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}")),
    ("google_api_key", re.compile(r"AIza[0-9A-Za-z_-]{35}")),
    ("jwt", re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")),
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("secret_path", re.compile(
        r"(?<![\w./-])(?:~|/[A-Za-z0-9._~-]+)?(?:/[A-Za-z0-9._~-]+)*"
        r"/(?:\.ssh|\.aws|\.gnupg|\.config/jev-eval)(?:/[A-Za-z0-9._~-]+)*"
        r"|(?<![\w./-])(?:~|/(?:[A-Za-z0-9._~-]+/)+)"
        r"(?:id_rsa|id_ed25519|id_dsa|id_ecdsa|\.env(?:\.[A-Za-z0-9._-]+)?|"
        r"\.netrc|credentials|apikey|api_key|secrets?|[A-Za-z0-9._-]*\.pem|"
        r"[A-Za-z0-9._-]*\.key)(?![\w.-])")),
    ("long_base64", re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{160,}={0,2}(?![A-Za-z0-9+/=])")),
    ("long_hex", re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{128,}(?![0-9a-fA-F])")),
]


class Drop(Exception):
    """Transcript-level rejection with a fixed reason code."""

    def __init__(self, reason, detail=None):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


def serialized(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(text):
    return " ".join(str(text).lower().split())


# ---------------------------------------------------------------------------
# Packed-token accounting (mirrors valen/data/compilers/qwen.py)
# ---------------------------------------------------------------------------

class PackedTokenCounter:
    """Estimates the compiler's packed length: chat-template overhead +
    tokens(state string) + fixed question suffix. ``encode`` is any
    ``str -> list`` callable, so tests can inject a fake."""

    def __init__(self, encode, instructions, template_overhead=5, reserved=()):
        self.encode = encode
        self.template_overhead = template_overhead
        self.reserved = tuple(reserved)
        # The compiler appends these pieces separately; replicate the split.
        pieces = [
            "<|im_start|>user\n",
            "Task: noul\nQuestion: " + instructions + "\nCandidates:\n",
            "true: True / 是：满足问题中的条件。", "\n",
            "false: False / 否：不满足问题中的条件。", "\n",
            "Decision:",
        ]
        self.suffix = sum(len(encode(p)) for p in pieces)
        # JSON-entry overheads inside the serialized state string.
        self.entry_overhead = len(encode(
            '{"pointer":"/messages/0000/content","role":"assistant","content":""},'))
        self.user_entry_overhead = len(encode('"",'))
        self.state_overhead = len(encode(
            '{"conversation":[],"candidate_pointer":"/messages/0000/content",'
            '"user_messages_in_order":[]}'))
        self.marker_tokens = len(encode(
            '{"pointer":"/elided/0","role":"control",'
            '"content":"<elided 0000 earlier segments>"},'))

    @classmethod
    def from_pretrained(cls, model_dir, instructions):
        from transformers import AutoTokenizer  # external/valen/.venv
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
        base = tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}], tokenize=True,
            add_generation_prompt=False)
        overhead = len(base["input_ids"] if isinstance(base, dict) else base)
        return cls(encode, instructions, template_overhead=overhead,
                   reserved=tokenizer.all_special_tokens)

    def count(self, text):
        return len(self.encode(text))

    def packed(self, state_str):
        """Packed tokens the compiler would produce for this state string."""
        return self.template_overhead + len(self.encode(state_str)) + self.suffix

    def has_reserved(self, text):
        """Compiler rejects states containing tokenizer control-token text."""
        return any(token in text for token in self.reserved)


def _window_indices(segments, candidate_index, seg_costs, fixed_cost, counter, budget):
    """Return the kept segment index set, or None when even the minimal window
    (control/system + candidate + last user) exceeds the budget.

    Policy: keep all control/system segments, the candidate, and the last user
    message; then add remaining segments nearest the candidate, expanding
    outward, while the estimated packed size fits. Segments containing reserved
    tokenizer control-token text are never eligible (the compiler rejects
    them); if a must-keep segment or a user message has one, the candidate is
    unscoreable -> returns the string "reserved_token"."""
    last_user = max((i for i, s in enumerate(segments) if s["kind"] == "user_turn"),
                    default=candidate_index)
    must = {i for i, s in enumerate(segments) if s["role"] in ("control", "system")}
    must.add(candidate_index)
    must.add(last_user)
    if any(counter.has_reserved(segments[i]["text"]) for i in must):
        return "reserved_token"
    keepable = {i for i in range(len(segments))
                if not counter.has_reserved(segments[i]["text"])}
    # Fast path: everything that may be kept fits.
    if fixed_cost + sum(seg_costs[i] for i in keepable) <= budget:
        return keepable
    kept = set(must)
    estimate = fixed_cost + sum(seg_costs[i] for i in must)
    if estimate > budget:
        return None
    optional = sorted(keepable - must,
                      key=lambda i: (abs(i - candidate_index), i))
    for i in optional:
        if estimate + seg_costs[i] + counter.marker_tokens <= budget:
            kept.add(i)
            estimate += seg_costs[i]
    return kept


def _render_window(segments, kept):
    """Ordered conversation for the kept set with elision-marker segments at
    every gap: {"role":"control","content":"<elided N earlier segments>"}."""
    conversation, dropped, markers = [], 0, 0
    for i, seg in enumerate(segments):
        if i in kept:
            if dropped:
                conversation.append({"pointer": f"/elided/{markers}",
                                     "role": "control",
                                     "content": f"<elided {dropped} earlier segments>"})
                markers += 1
                dropped = 0
            conversation.append({"pointer": f"/messages/{i}/content",
                                 "role": seg["role"], "content": seg["text"]})
        else:
            dropped += 1
    if dropped:
        conversation.append({"pointer": f"/elided/{markers}",
                             "role": "control",
                             "content": f"<elided {dropped} earlier segments>"})
    return conversation


# ---------------------------------------------------------------------------
# Segment extraction
# ---------------------------------------------------------------------------

def _claude_control(text):
    stripped = text.lstrip()
    return stripped.startswith(CLAUDE_CONTROL_PREFIXES)


def _codex_control(text):
    stripped = text.lstrip()
    return stripped.startswith(CODEX_CONTROL_PREFIXES)


def _tool_result_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [block.get("text", "") for block in content
                 if isinstance(block, dict) and block.get("type") == "text"]
        if parts:
            return "\n".join(parts)
        return serialized(content)
    return serialized(content)


class _Transcript:
    def __init__(self):
        self.segments = []   # dicts: role, kind, text
        self.users = []      # real user message texts, in order
        self.text_bytes = 0

    def add(self, role, kind, text):
        if not isinstance(text, str):
            text = serialized(text)
        if not text.strip():
            return
        if len(self.segments) >= MAX_SEGMENTS_HARD:
            raise Drop("byte_cap", "segment_count_hard_cap")
        self.segments.append({"role": role, "kind": kind, "text": text})
        if kind == "user_turn":
            self.users.append(text)
        self.text_bytes += len(text.encode("utf-8", errors="ignore"))
        if self.text_bytes > MAX_STATE_BYTES:
            raise Drop("byte_cap", "extracted_text_over_cap")


def extract_claude(path):
    """Flatten a Claude Code session jsonl into ordered segments."""
    t = _Transcript()
    with open(path, encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            if '"type"' not in line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            etype = event.get("type")
            if etype == "assistant":
                if event.get("isCompactSummary"):
                    continue
                content = (event.get("message") or {}).get("content")
                if not isinstance(content, list):
                    continue
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type")
                    if btype == "text":
                        t.add("assistant", "assistant_text", block.get("text", ""))
                    elif btype == "tool_use":
                        t.add("assistant", "tool_use",
                              serialized({"tool_use": {"name": block.get("name"),
                                                       "input": block.get("input")}}))
            elif etype == "user":
                if event.get("isMeta") or event.get("isCompactSummary"):
                    continue
                if event.get("promptSource") == "system":
                    continue
                content = (event.get("message") or {}).get("content")
                if isinstance(content, str):
                    if not _claude_control(content):
                        t.add("user", "user_turn", content)
                elif isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        btype = block.get("type")
                        if btype == "tool_result":
                            t.add("tool", "tool_result",
                                  _tool_result_text(block.get("content", "")))
                        elif btype == "text":
                            text = block.get("text", "")
                            if not _claude_control(text):
                                t.add("user", "user_turn", text)
                        # image/tool_reference/other blocks are not text context
    return t


def _codex_message_texts(payload):
    """Split a codex message payload into (real_text, control_text)."""
    real, control = [], []
    for part in payload.get("content") or []:
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        if part.get("type") in ("input_text", "output_text", "text"):
            (control if _codex_control(text) else real).append(text)
        else:
            control.append(text)
    return "\n\n".join(real), "\n\n".join(control)


def extract_codex(path):
    """Flatten a Codex rollout jsonl (response_item events only) into segments."""
    t = _Transcript()
    with open(path, encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            if '"response_item"' not in line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("type") != "response_item":
                continue
            payload = event.get("payload")
            if not isinstance(payload, dict):
                continue
            ptype = payload.get("type")
            if ptype == "message":
                real, control = _codex_message_texts(payload)
                role = payload.get("role")
                if role == "user":
                    if real:
                        t.add("user", "user_turn", real)
                    # control parts are harness scaffolding; not segments
                elif role == "assistant":
                    t.add("assistant", "assistant_text", real or control)
                # developer/system boilerplate is harness scaffolding; skipped
            elif ptype in ("function_call", "custom_tool_call"):
                call = {"tool_use": {"name": payload.get("name"),
                                     "input": payload.get("arguments",
                                                          payload.get("input"))}}
                t.add("assistant", "tool_use", serialized(call))
            elif ptype in ("function_call_output", "custom_tool_call_output"):
                output = payload.get("output", "")
                t.add("tool", "tool_result",
                      output if isinstance(output, str) else serialized(output))
            elif ptype == "web_search_call":
                t.add("assistant", "tool_use",
                      serialized({"tool_use": {"name": "web_search",
                                               "input": payload.get("action")}}))
            elif ptype == "image_generation_call":
                t.add("assistant", "tool_use",
                      serialized({"tool_use": {"name": "image_generation",
                                               "input": payload.get("revised_prompt")}}))
            # reasoning is encrypted/unreadable; never extracted
    return t


# ---------------------------------------------------------------------------
# Candidate enumeration + stratified selection
# ---------------------------------------------------------------------------

def length_band(n_segments):
    for name, low, high in LENGTH_BANDS:
        if low <= n_segments <= high:
            return name
    return "s"


def eligible_candidate_indices(segments):
    """Declared segment policy: assistant text, tool results, and non-final
    user turns. tool_use/control/system segments and the final user message
    (the request anchor) are never candidates."""
    last_user = max((i for i, s in enumerate(segments) if s["kind"] == "user_turn"),
                    default=None)
    eligible = []
    for i, seg in enumerate(segments):
        if seg["kind"] in ("assistant_text", "tool_result"):
            eligible.append(i)
        elif seg["kind"] == "user_turn" and i != last_user:
            eligible.append(i)
    return eligible


def select_candidates(segments, cap=MAX_CANDIDATES_PER_TRANSCRIPT):
    """Deterministic stratified pick: round-robin over early/mid/late position
    thirds, then guarantee one of each present kind where possible."""
    eligible = eligible_candidate_indices(segments)
    if len(eligible) <= cap:
        return eligible
    n = len(segments)
    thirds = [[], [], []]
    for i in eligible:
        thirds[min(2, (3 * i) // max(1, n))].append(i)
    chosen = []
    while len(chosen) < cap and any(thirds):
        for bucket in thirds:
            if bucket and len(chosen) < cap:
                chosen.append(bucket.pop(0))
    chosen.sort()
    # guarantee each candidate kind present in the transcript appears at least once
    present_kinds = {segments[i]["kind"] for i in eligible}
    chosen_kinds = {segments[i]["kind"] for i in chosen}
    for kind in sorted(present_kinds - chosen_kinds):
        swap = next((i for i in eligible if segments[i]["kind"] == kind and i not in chosen),
                    None)
        if swap is not None:
            chosen[-1] = swap
            chosen.sort()
    return chosen


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

def credential_scan(text):
    """Return the matching pattern name, or None. A hit drops the transcript."""
    for name, pattern in SECRET_PATTERNS:
        if pattern.search(text):
            return name
    return None


# ---------------------------------------------------------------------------
# Record construction + dedup
# ---------------------------------------------------------------------------

def load_verbatim_instructions(valen_v3):
    path = Path(valen_v3) / "eval.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        return record["request"]["questions"]["irrelevant"]["instructions"]
    raise Drop("no_valen_instructions")


def load_reference_hashes(valen_dirs):
    """Exact request hashes + normalized state hashes from prior valen sets."""
    request_hashes, state_hashes = set(), set()
    for directory in valen_dirs:
        directory = Path(directory)
        if not directory.is_dir():
            continue
        for name in ("train.jsonl", "eval.jsonl"):
            path = directory / name
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                request_hashes.add(sha256_bytes(serialized(record.get("request")).encode("utf-8")))
                shash = _record_state_hash(record)
                if shash:
                    state_hashes.add(shash)
    return request_hashes, state_hashes


def language_bucket(transcript):
    sample = "\n".join(transcript.users[:5])
    return "multi" if re.search(r"[一-鿿぀-ヿ가-힯]", sample) else "en"


def build_records(transcript, transcript_hash, source, instructions,
                  cap=MAX_CANDIDATES_PER_TRANSCRIPT, counter=None,
                  token_budget=TOKEN_BUDGET):
    segments = transcript.segments
    full = [{"pointer": f"/messages/{i}/content",
             "role": seg["role"], "content": seg["text"]}
            for i, seg in enumerate(segments)]
    if len(serialized(full).encode("utf-8")) > MAX_STATE_BYTES:
        raise Drop("byte_cap", "serialized_state_over_cap")
    users = list(transcript.users)
    candidate_indices = select_candidates(segments, cap=cap)
    group_id = f"{GROUP_LINEAGE}:{transcript_hash}"
    band = length_band(len(segments))
    lang = language_bucket(transcript)

    # Precompute per-segment token costs once per transcript.
    seg_costs = user_cost = None
    if counter is not None:
        seg_costs = [counter.count(seg["text"]) + counter.entry_overhead
                     for seg in segments]
        user_cost = sum(counter.count(u) + counter.user_entry_overhead
                        for u in users)
    reserved_in_users = (counter is not None
                         and any(counter.has_reserved(u) for u in users))

    records, candidate_drops = [], Counter()
    last_user = max((i for i, s in enumerate(segments) if s["kind"] == "user_turn"),
                    default=None)
    anchors = {i for i, s in enumerate(segments)
               if s["role"] in ("control", "system")}
    if last_user is not None:
        anchors.add(last_user)
    for index in candidate_indices:
        seg = segments[index]
        windowed = False
        packed = None
        if counter is None:
            conversation = full
        else:
            if reserved_in_users:
                candidate_drops["reserved_token"] += 1
                continue
            fixed = counter.state_overhead + user_cost
            kept = _window_indices(segments, index, seg_costs, fixed,
                                   counter, token_budget)
            if kept is None:
                candidate_drops["exceeds_budget"] += 1
                continue
            if kept == "reserved_token":
                candidate_drops["reserved_token"] += 1
                continue
            windowed = kept != set(range(len(segments)))
            conversation = _render_window(segments, kept)
            state_probe = {"conversation": conversation,
                           "candidate_pointer": f"/messages/{index}/content",
                           "user_messages_in_order": users}
            packed = counter.packed(serialized(state_probe))
            while packed > token_budget:
                # Exact count disagreed with the estimate; shed the kept
                # segment farthest from the candidate (never an anchor or the
                # candidate itself) and recount.
                farthest = max(kept - anchors - {index},
                               key=lambda i: (abs(i - index), i), default=None)
                if farthest is None:
                    break
                kept.discard(farthest)
                conversation = _render_window(segments, kept)
                state_probe["conversation"] = conversation
                packed = counter.packed(serialized(state_probe))
            if packed > token_budget:
                candidate_drops["exceeds_budget"] += 1
                continue
            windowed = kept != set(range(len(segments)))
        state = {"conversation": conversation,
                 "candidate_pointer": f"/messages/{index}/content",
                 "user_messages_in_order": users}
        request = {"state": serialized(state),
                   "questions": {"irrelevant": {"type": "noul",
                                              "instructions": instructions}}}
        records.append({
            "group_id": group_id,
            "request": request,
            "targets": None,  # unlabeled: scoring targets assigned by the labeling pass
            "meta": {
                "record_id": f"realctx:{transcript_hash[:12]}:seg{index:04d}",
                "domain": "real",
                "modality": "text",
                "language_bucket": lang,
                "source": "real_transcript",
                "source_dataset": GROUP_LINEAGE,
                "source_root": source,
                "transcript_hash": transcript_hash,
                "candidate_kind": seg["kind"],
                "candidate_pointer": f"/messages/{index}/content",
                "length_band": band,
                "selection_method": "deterministic",
                "windowed": windowed,
                "packed_tokens": packed,
            },
        })
    return records, candidate_drops


def request_hash(record):
    return sha256_bytes(serialized(record["request"]).encode("utf-8"))


def _record_state_hash(record):
    """Normalized near-duplicate key: candidate pointer + normalized segment
    contents, so formatting-only variants of the same record collide but
    distinct candidates of one transcript do not."""
    try:
        state = json.loads(record["request"]["state"])
        contents = [normalize_text(s.get("content", ""))
                    for s in state.get("conversation", [])]
        key = "\x1f".join([str(state.get("candidate_pointer")), *contents])
        return sha256_bytes(key.encode("utf-8"))
    except (ValueError, KeyError, TypeError):
        return None


def normalized_state_hash(record):
    return _record_state_hash(record)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def iter_transcript_files(claude_root, codex_root):
    files = []
    claude_root, codex_root = Path(claude_root), Path(codex_root)
    if claude_root.is_dir():
        files += [("claude_projects", p) for p in sorted(claude_root.glob("**/*.jsonl"))]
    if codex_root.is_dir():
        files += [("codex_sessions", p) for p in sorted(codex_root.glob("**/*.jsonl"))]
    return files


def collect(claude_root=DEFAULT_CLAUDE_ROOT, codex_root=DEFAULT_CODEX_ROOT,
            valen_dirs=(DEFAULT_VALEN_V2, DEFAULT_VALEN_V3),
            max_transcripts=MAX_TRANSCRIPTS, target_records=TARGET_TOTAL_RECORDS,
            limit=None, tokenizer_path=DEFAULT_TOKENIZER,
            token_budget=TOKEN_BUDGET, counter=None):
    instructions = load_verbatim_instructions(valen_dirs[-1] if valen_dirs else DEFAULT_VALEN_V3)
    if counter is None and tokenizer_path is not None:
        counter = PackedTokenCounter.from_pretrained(tokenizer_path, instructions)
    ref_requests, ref_states = load_reference_hashes(valen_dirs)
    drops = Counter()
    admitted = []  # (band, transcript_hash, source, transcript, records)
    seen_transcripts = set()
    files = iter_transcript_files(claude_root, codex_root)
    if limit:
        files = files[:limit]
    stats = {"files_scanned": 0}
    for source, path in files:
        stats["files_scanned"] += 1
        try:
            extract = extract_claude if source == "claude_projects" else extract_codex
            transcript = extract(path)
        except (Drop, OSError) as drop:
            drops[getattr(drop, "reason", "unreadable")] += 1
            continue
        if len(transcript.segments) < MIN_SEGMENTS:
            drops["min_segments"] += 1
            continue
        if len(transcript.users) < MIN_USER_MESSAGES:
            drops["no_user_messages"] += 1
            continue
        if not eligible_candidate_indices(transcript.segments):
            drops["no_candidates"] += 1
            continue
        # Redaction before anything is written: a credential hit drops the
        # entire transcript, not just the offending segment.
        hit = credential_scan("\n".join(seg["text"] for seg in transcript.segments))
        if hit is not None:
            drops[f"credential_scan:{hit}"] += 1
            continue
        try:
            raw_hash = sha256_file(path)
        except OSError:
            drops["unreadable"] += 1
            continue
        if raw_hash in seen_transcripts:
            drops["duplicate_source_file"] += 1
            continue
        seen_transcripts.add(raw_hash)
        admitted.append({"band": length_band(len(transcript.segments)),
                         "transcript_hash": raw_hash, "source": source,
                         "segment_count": len(transcript.segments),
                         "user_message_count": len(transcript.users),
                         "extracted_bytes": transcript.text_bytes,
                         "eligible_candidates": len(
                             eligible_candidate_indices(transcript.segments)),
                         "transcript": transcript})

    # Stratified transcript admission: round-robin over length bands.
    by_band = {name: sorted((a for a in admitted if a["band"] == name),
                            key=lambda a: a["transcript_hash"])
               for name, *_ in LENGTH_BANDS}
    selected = []
    while len(selected) < max_transcripts:
        progressed = False
        for band in BAND_ORDER:
            if by_band.get(band) and len(selected) < max_transcripts:
                selected.append(by_band[band].pop(0))
                progressed = True
        if not progressed:
            break
    for band_name, rest in by_band.items():
        drops["over_transcript_quota"] += len(rest)

    # Adaptive per-transcript candidate cap (never above the 25 policy cap):
    # spread the record budget across as many distinct transcripts as possible.
    cap = min(MAX_CANDIDATES_PER_TRANSCRIPT,
              max(1, math.ceil(target_records / max(1, len(selected)))))
    candidate_drops = Counter()
    for item in selected:
        records, cdrops = build_records(
            item.pop("transcript"), item["transcript_hash"], item["source"],
            instructions, cap=cap, counter=counter, token_budget=token_budget)
        item["candidate_count"] = len(records)
        item["records"] = records
        candidate_drops.update(cdrops)

    # Near-duplicate / exact-hash screening vs prior valen sets and intra-set.
    dedup = Counter()
    kept, seen_requests, seen_states = [], set(), set()
    for item in selected:
        had_records = bool(item["records"])
        remaining = []
        for record in item["records"]:
            rhash, shash = request_hash(record), normalized_state_hash(record)
            if rhash in ref_requests or shash in ref_states:
                dedup["vs_prior_valen"] += 1
            elif rhash in seen_requests or shash in seen_states:
                dedup["intra_set"] += 1
            else:
                seen_requests.add(rhash)
                seen_states.add(shash)
                remaining.append(record)
        item["records"] = remaining
        item["candidate_count"] = len(remaining)
        if remaining:
            kept.append(item)
        elif had_records:
            dedup["transcripts_fully_deduplicated"] += 1
        else:
            drops["all_candidates_dropped"] += 1

    records = sorted((r for item in kept for r in item["records"]),
                     key=lambda r: (r["group_id"], r["meta"]["record_id"]))
    return {"records": records, "transcripts": kept, "drops": drops,
            "dedup": dedup, "stats": stats, "instructions": instructions,
            "candidate_cap": cap, "files_scanned": len(files),
            "candidate_drops": dict(sorted(candidate_drops.items())),
            "token_budget": token_budget if counter is not None else None}


def build_manifest(result, out_dir):
    records, transcripts = result["records"], result["transcripts"]
    kind_counts = Counter(r["meta"]["candidate_kind"] for r in records)
    band_counts = Counter(t["band"] for t in transcripts)
    source_counts = Counter(t["source"] for t in transcripts)
    seg_counts = sorted(t["segment_count"] for t in transcripts)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "builder": "scripts/build_real_context_eval_v1.py",
        "output_dir": str(out_dir),
        "content_free": True,
        "training_allowed": False,
        "evaluation_only": True,
        "collection": {
            "phase": "candidate_collection_unlabeled",
            "provider_calls_allowed": False,
            "active_filtering_allowed": False,
            "labels": "targets are null; manual labeling pass required before scoring",
        },
        "group_id": f"{GROUP_LINEAGE}:<sha256(source transcript file bytes)>",
        "isolation": "one transcript = one group_id; a transcript is never split",
        "segment_policy": {
            "candidate_kinds": list(CANDIDATE_KINDS),
            "excluded": ["tool_use", "control", "system", "final_user_message"],
            "scaffolding_excluded_at_extraction": (
                "harness-injected blocks (codex developer/system boilerplate, "
                "environment/internal context, command wrappers, task "
                "notifications, image placeholders) are not conversation "
                "segments"),
            "max_candidates_per_transcript": MAX_CANDIDATES_PER_TRANSCRIPT,
            "candidate_cap_applied": result["candidate_cap"],
            "candidate_selection": "deterministic round-robin over early/mid/late position thirds",
            "length_bands": {name: [low, high if high < 10**9 else None]
                             for name, low, high in LENGTH_BANDS},
        },
        "thresholds": {
            "min_segments": MIN_SEGMENTS,
            "min_user_messages": MIN_USER_MESSAGES,
            "max_state_bytes": MAX_STATE_BYTES,
            "max_transcripts": MAX_TRANSCRIPTS,
            "target_total_records": TARGET_TOTAL_RECORDS,
        },
        "records": {"candidates": len(records)},
        "groups": {"transcripts": len(transcripts)},
        "token_budget": result["token_budget"],
        "packed_token_limit": MAX_PACKED_TOKENS,
        "windowed_count": sum(1 for r in records if r["meta"].get("windowed")),
        "dropped_exceeds_budget": result["candidate_drops"].get("exceeds_budget", 0),
        "dropped_reserved_token": result["candidate_drops"].get("reserved_token", 0),
        "label_counts": {"unlabeled": len(records)},
        "question": {
            "qid": "irrelevant", "type": "noul",
            "semantics": "true == candidate segment certainly irrelevant == drop; false == keep",
            "instructions_preserved_verbatim": True,
            "instructions_sha256": sha256_bytes(result["instructions"].encode("utf-8")),
        },
        "sources": dict(source_counts),
        "files_scanned": result["stats"]["files_scanned"],
        "candidate_kinds": dict(kind_counts),
        "length_bands_admitted": dict(band_counts),
        "segment_stats": {
            "min": seg_counts[0] if seg_counts else 0,
            "median": seg_counts[len(seg_counts) // 2] if seg_counts else 0,
            "max": seg_counts[-1] if seg_counts else 0,
        },
        "drops": dict(sorted(result["drops"].items())),
        "dedup": {
            "reference_sets": ["data/valen_nano_v2", "data/valen_nano_v3"],
            **dict(sorted(result["dedup"].items())),
        },
        "excluded": {
            "data/jevbench_offline_bundle_v1": "bundle manifest: evaluation_only, training_allowed=false",
            "*_test.jsonl,*_ood.jsonl": "frozen splits reserved for NanoJev gates",
            "results/e2e_*,context_shadow receipts": "contain provider probabilities (never enter training)",
            "provider outputs": "never enter training, calibration, or eval labels",
        },
        "transcripts": [
            {"transcript_hash": t["transcript_hash"], "source": t["source"],
             "length_band": t["band"], "segment_count": t["segment_count"],
             "user_message_count": t["user_message_count"],
             "candidate_count": t["candidate_count"],
             "extracted_bytes": t["extracted_bytes"]}
            for t in sorted(transcripts, key=lambda t: t["transcript_hash"])
        ],
    }
    return manifest


def write_output(result, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "candidates.jsonl"
    with path.open("w", encoding="utf-8") as stream:
        for record in result["records"]:
            stream.write(serialized(record) + "\n")
    manifest = build_manifest(result, out_dir)
    data = path.read_bytes()
    manifest["sha256"] = {"candidates": sha256_bytes(data)}
    manifest["bytes"] = {"candidates": len(data)}
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claude-root", type=Path, default=DEFAULT_CLAUDE_ROOT)
    parser.add_argument("--codex-root", type=Path, default=DEFAULT_CODEX_ROOT)
    parser.add_argument("--valen-v3", type=Path, default=DEFAULT_VALEN_V3)
    parser.add_argument("--valen-v2", type=Path, default=DEFAULT_VALEN_V2)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-transcripts", type=int, default=MAX_TRANSCRIPTS)
    parser.add_argument("--target-records", type=int, default=TARGET_TOTAL_RECORDS)
    parser.add_argument("--limit", type=int, default=None,
                        help="scan only the first N transcript files (debug)")
    parser.add_argument("--print-examples", type=int, default=0,
                        help="print K truncated candidate renderings to stdout")
    parser.add_argument("--dry-run", action="store_true",
                        help="collect and report without writing any output")
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER,
                        help="Qwen tokenizer dir used for packed-token budgets")
    parser.add_argument("--no-tokenizer", action="store_true",
                        help="skip token budgeting (debug/unittest only)")
    parser.add_argument("--token-budget", type=int, default=TOKEN_BUDGET,
                        help="packed-token target per record (hard limit 8192)")
    args = parser.parse_args(argv)

    result = collect(args.claude_root, args.codex_root,
                     valen_dirs=(args.valen_v2, args.valen_v3),
                     max_transcripts=args.max_transcripts,
                     target_records=args.target_records,
                     limit=args.limit,
                     tokenizer_path=None if args.no_tokenizer else args.tokenizer,
                     token_budget=args.token_budget)
    report = {
        "files_scanned": result["stats"]["files_scanned"],
        "transcripts_admitted": len(result["transcripts"]),
        "candidates": len(result["records"]),
        "drops": dict(sorted(result["drops"].items())),
        "dedup": dict(sorted(result["dedup"].items())),
        "candidate_drops": result["candidate_drops"],
        "windowed": sum(1 for r in result["records"] if r["meta"].get("windowed")),
        "token_budget": result["token_budget"],
        "candidate_kinds": dict(Counter(r["meta"]["candidate_kind"]
                                        for r in result["records"])),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.print_examples:
        for record in result["records"][:args.print_examples]:
            state = json.loads(record["request"]["state"])
            candidate = next(s for s in state["conversation"]
                             if s["pointer"] == state["candidate_pointer"])
            print("=" * 72)
            print("record_id:", record["meta"]["record_id"],
                  "| kind:", record["meta"]["candidate_kind"],
                  "| band:", record["meta"]["length_band"])
            print("candidate:", candidate["role"],
                  repr(candidate["content"][:200]))
            print("last user msg:", repr((state["user_messages_in_order"] or [""])[-1][:200]))
    if args.dry_run:
        return 0
    manifest = write_output(result, args.out)
    print(json.dumps({"candidates_jsonl": str(args.out / "candidates.jsonl"),
                      "sha256": manifest["sha256"]["candidates"],
                      "manifest": str(args.out / "manifest.json")}, indent=2))
    if len(result["records"]) < 400 or len(result["transcripts"]) < 40:
        print("WARNING: below design targets (>=400 candidates across >=40 "
              "transcripts); report the shortfall rather than widening scope.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
