#!/usr/bin/env python3
"""V5 F4 outcome-positive miner (pool-split edition): keep-labeled candidates
where a segment was provably USED downstream — "the agent actually consumed
this" labels, no labeler needed for the positive claim.

Implements docs/V5_DATA_DESIGN_V1.md §5 (family F4, outcome_positive), source
(b) "local unused transcripts". Source (a) public resolved-task trajectories
(SWE-Gym/SWE-chat) needs downloads + license screening — out of scope (no
network); the yield gap is reported honestly in f4_report.json.

Pool coordination (task requirement):
  * Reads the pool ledger from data/v5_mining/f2_report.json["pool_split"]
    when present (it is: F2 reserved 18 transcripts for v5-eval mining,
    train_pool = 62) and mines ONLY the train pool — eval-v1's 60 frozen
    transcripts and the v5-eval reserve are never mined.
  * If that ledger were absent, the fallback rule is deterministic:
    eval_pool = first 18 usable transcripts by sorted transcript_hash not in
    eval-v1's 60; train_pool = rest. In both cases the materialized canonical
    ledger data/v5_mining/pool_split.json is written (when missing) so future
    miners honor one file; an existing pool_split.json takes precedence.

Linker signals (all deterministic; every row records evidence_basis):
  * file_read_edit — a tool_result holding the content of path P (paired
    Read/cat tool_call names P, or P literally in the result text) followed,
    WITHIN THE SAME TASK SPAN and <= LINK_MAX_GAP segments, by an
    Edit/Write/apply_patch of P -> keep. Dir-listing -> child-edit is the
    weak form -> uncertain.
  * command_output_echo — a tool_result whose distinctive >=40-char substring
    (normalized line or fixed-position chunk that carries a path / URL /
    >=3-digit run / long identifier), URL, or digit-bearing hex hash appears
    verbatim in a LATER assistant-authored segment (assistant_text or
    tool_use input) -> keep. Dedup guard: the needle must occur in at most
    ONE tool_result of the transcript (itself) — repeated boilerplate can
    never attribute an echo to a specific result. Echo only in a later
    user_turn, or only a rare long identifier recurring -> uncertain.
  * test_failure_fix — a tool_result bearing failure signals (FAILED/assert/
    traceback/nonzero exit/中文等价) in a test/verification context (paired
    call is a test runner command, or the output carries a test summary)
    followed, same span + <= LINK_MAX_GAP, by an Edit/patch, a git fix-up
    commit, or an assistant message that references the failure (echoes an
    error line or names a failing identifier + fix vocabulary) -> keep.
  * unused_drop_control — NEGATIVE CONTROL, emitted as a separate proposal
    (mining_side="drop"): a substantive tool_result (>=120 norm chars) that
    passes the full reach-back battery with NO hit (no path re-reference, no
    quoted signature, no shared identifiers, no admissible echo, no later
    read/edit of the same path) AND a later user turn exists (the task moved
    on) -> proposed_label "yes". The label is proposed only after the
    reach-back check returns empty; rows whose F2 miner twin already
    proposed the same pointer 'no' are skipped (label conflict).

Emission mirrors scripts/mine_v5_hard_negatives_v1.py (the F2 miner):
  {group_id: real_context_v5:<hash>, request{state, questions.irrelevant
  verbatim noul}, targets: null, meta{record_id v5f4:<h>:segNNNN[:aNNNN],
  supp_source="outcome_linker", proposed_label, proposal_note,
  evidence_basis (dict), pair_id, pool="train", ...}}.
Anchor/cut for keep rows: state = transcript prefix ending right after the
proving usage segment (cut = usage_seg+1), anchor = last user turn at/before
the usage — the request context a filter would see when the segment is used.
For drop controls: anchor = the next user turn (the "task moved on"
boundary); cut = end of that anchor's own span — same re-anchor shape as F2.
Contrastive pairing (doc §1.7): a usage-proven keep and a reach-back-clean
drop control under the SAME anchor form pair_id
v5f4pair:<h>:use:aNNNN (pair_kind="same_anchor").

Guards identical to the F2 miner: same SECRET_PATTERNS credential scan at
segment/user granularity (flagged segments are never candidates and never
enter a window; a flagged user message disables every anchor at/after it),
<=32KB candidate cap, character-budget windowing (12_000 target / 14_000
hard), request/state hash dedup vs valen_nano_v2/v3/v4 + real_context_eval_v1
+ the F2 candidates file, and a hard self_check before anything is written.
Determinism: seed 20261201, sorted iteration, pure local rules — no model,
service, network, or tokenizer calls.

Usage: python3 scripts/mine_v5_outcome_positive_v1.py
Output: data/v5_mining/f4_outcome_positive_candidates.jsonl
        data/v5_mining/f4_report.json
        data/v5_mining/f4_review.md   (20-row link-precision spot render)
        data/v5_mining/pool_split.json (only if absent — canonical ledger)
Self-check + tests: scripts/test_mine_v5_outcome_v1.py
"""

import bisect
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
spec2 = importlib.util.spec_from_file_location(
    "mine_v5_hard_negatives_v1",
    ROOT / "scripts" / "mine_v5_hard_negatives_v1.py")
f2m = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(f2m)

OUT_DIR = ROOT / "data" / "v5_mining"
OUT_JSONL = OUT_DIR / "f4_outcome_positive_candidates.jsonl"
OUT_REPORT = OUT_DIR / "f4_report.json"
OUT_REVIEW = OUT_DIR / "f4_review.md"
F2_JSONL = OUT_DIR / "f2_hard_negative_candidates.jsonl"
F2_REPORT = OUT_DIR / "f2_report.json"
POOL_LEDGER = OUT_DIR / "pool_split.json"
EVAL_V1_CANDIDATES = ROOT / "data" / "real_context_eval_v1" / "candidates.jsonl"

SEED = 20261201            # v5 reserved seed (V5_DATA_DESIGN_V1 §1.3)
STATE_CHAR_HARD = f2m.STATE_CHAR_HARD          # 14_000
MAX_CANDIDATE_BYTES = f2m.MAX_CANDIDATE_BYTES  # 32KB
GROUP_LINEAGE = f2m.GROUP_LINEAGE              # "real_context_v5"

Y, N, U = "yes", "no", "uncertain"  # proposed_label values, v1 convention

# --- linker bounds / caps (all deterministic) --------------------------------
LINK_MAX_GAP = 48        # "within same task, <=N turns" — segment-step bound
ECHO_MIN_CHARS = 40      # distinctive-substring length floor (task spec)
ECHO_DF_MAX = 1          # needle must occur in <=1 tool_result (itself)
CTRL_MIN_CHARS = 120     # drop controls must be substantive results
FALLBACK_EVAL_RESERVE = 18  # only used if the F2 ledger is absent

MAX_FILE_PER_TRANSCRIPT = 6
MAX_ECHO_PER_TRANSCRIPT = 6
MAX_TEST_PER_TRANSCRIPT = 4
MAX_CTRL_PER_TRANSCRIPT = 4
MAX_UNCERTAIN_PER_TRANSCRIPT = 3
MAX_PER_TRANSCRIPT = 16
MAX_UNITS_PER_RESULT = 30
MAX_NEEDLES_PER_TRANSCRIPT = 2_000

PATH_TOKEN_RE = f2m.PATH_TOKEN_RE
IDENT_RE = f2m.IDENT_RE
DIGIT_RUN_RE = f2m.DIGIT_RUN_RE
GENERIC_TOKENS = f2m.GENERIC_TOKENS

# Evidence-unit regexes for command_output_echo.
URL_RE = re.compile(r"https?://[^\s'\"<>\)\]]{8,}")
HASH_RE = re.compile(r"\b(?=[0-9a-f]*\d)[0-9a-f]{7,40}\b")

# test_failure_fix detection: failure surface + verification context.
FAILURE_RE = re.compile(
    r"(?:\bFAILED\b|\bfailed\b|\bfailure\b|\bfailing\b|assert\w*\s*error|"
    r"AssertionError|Traceback|\bpanic\b|\bexception\b|"
    r"exit(?:ed)?(?:\s+with)?(?:\s+code|\s+status)?\s*[1-9]\d*\b|"
    r"non-?zero\b|not ok\b|\bERR!\b|✗|✘|失败|错误|报错|未通过|不通过|断言)",
    re.IGNORECASE)
VERIFY_CMD_RE = re.compile(
    r"(pytest|py\.test|unittest|nose2?|jest|vitest|mocha|ava\b|cargo\s+test|"
    r"go\s+test|npm\s+(?:run\s+)?test|yarn\s+(?:run\s+)?test|pnpm\s+test|"
    r"mvn\s+(?:test|verify)|gradle\w*\s+(?:test|check)|ctest|tox|nox|rspec|"
    r"phpunit|dotnet\s+test|swift\s+test|make\s+(?:check|test)|bin/test|"
    r"check_|verify_|断言|测试|校验)",
    re.IGNORECASE)
TEST_SUMMARY_RE = re.compile(
    r"(\d+\s+(?:failed|passed|errors?)\b|\bFAILED\b|failures=\d|"
    r"tests?\s+ran|test result:|passed,\s*\d+|====.*(?:passed|failed))",
    re.IGNORECASE)
GIT_FIX_RE = re.compile(r"\bgit\s+(?:commit|apply\b|add\b)")
FIX_TALK_RE = re.compile(
    r"(fix(?:ed|es|ing)?|root cause|resolved|resolving|patch(?:ed|ing)?|"
    r"workaround|the (?:bug|issue|error|failure|problem)\s+(?:is|was)|"
    r"修复|已修|原因|问题在|报错|已解决|定位到|改成|改为)",
    re.IGNORECASE)
FAIL_IDENT_RE = re.compile(
    r"(?:test_[A-Za-z0-9_]{3,}|[A-Za-z_][A-Za-z0-9_.]*::[A-Za-z0-9_.:]+|"
    r"[A-Za-z_][A-Za-z0-9_]{7,})")

# "the agent consumed it" = assistant-authored usage; a later user message
# pasting output back is weaker evidence (human carried it) -> uncertain.
ASSISTANT_USAGE_KINDS = ("assistant_text", "tool_use")
USER_USAGE_KINDS = ("user_turn",)


def norm_path(p):
    p = str(p).strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p


def clean_op_path(p):
    """Sanitize a path extracted from a (possibly shell-string) tool call:
    cut at the first literal backslash — apply_patch `*** File: <path>`
    parsing can absorb trailing `\\n@@` escape fragments, which would
    poison path_in_text matching and evidence strings."""
    p = str(p).split("\\", 1)[0].strip()
    return norm_path(p).rstrip("/.,;:'\"")


def path_match(a, b):
    """Normalized exact or abs-vs-rel suffix match."""
    a, b = norm_path(a), norm_path(b)
    if a == b:
        return "exact"
    if len(a) >= 6 and len(b) >= 6 and (
            a.endswith("/" + b) or b.endswith("/" + a)):
        return "suffix"
    return None


def path_in_text(path, norm_text):
    np = norm_path(path)
    if len(np) >= 6 and np in norm_text:
        return np
    parts = np.split("/")
    for k in (2, 3):
        if len(parts) >= k:
            suffix = "/".join(parts[-k:])
            if len(suffix) >= 8 and suffix in norm_text:
                return suffix
    return None


def pair_results(segs):
    """FIFO pairing of tool_use -> tool_result segments."""
    pending, pairs = [], []
    for i, s in enumerate(segs):
        if s["kind"] == "tool_use":
            pending.append(i)
        elif s["kind"] == "tool_result":
            pairs.append((i, pending.pop(0) if pending else None))
    return pairs


def distinctive_line(n):
    """A normalized line worth echo-checking: carries a path, a >=3-digit
    run, a URL, or a long non-generic identifier."""
    if PATH_TOKEN_RE.search(n) or URL_RE.search(n) or re.search(r"\d{3,}", n):
        return True
    for m in IDENT_RE.finditer(n):
        w = m.group(0)
        if len(w) >= 8 and w.lower() not in GENERIC_TOKENS:
            return True
    return False


def echo_units(text):
    """(unit_kind, strength, needle) from a tool_result's raw text.

    needles are whitespace-normalized, matched verbatim (case preserved)
    against normalized later-segment text. The >=40-char rule applies to
    line/chunk units; URLs and digit-bearing hashes are inherently
    distinctive evidence units.
    """
    units, seen = [], set()

    def add(kind, strength, needle):
        if needle and needle not in seen:
            seen.add(needle)
            units.append((kind, strength, needle))

    for m in URL_RE.finditer(text):
        add("url", "strong", m.group(0).rstrip(".,;"))
    for m in HASH_RE.finditer(text):
        add("hash", "strong" if len(m.group(0)) >= 10 else "medium",
            m.group(0))
    lines = [f2m.norm(ln) for ln in text.splitlines()]
    ntext = f2m.norm(text)
    if len(lines) <= 2 and len(ntext) >= ECHO_MIN_CHARS:
        # single-block output: fixed-position chunks like f2m.signatures()
        for pos in (0, max(0, len(ntext) // 2 - 30), max(0, len(ntext) - 60)):
            chunk = ntext[pos:pos + 60]
            if len(chunk) >= ECHO_MIN_CHARS and distinctive_line(chunk):
                add("chunk", "strong", chunk)
    else:
        for ln in lines:
            if len(units) >= MAX_UNITS_PER_RESULT:
                break
            if len(ln) >= ECHO_MIN_CHARS and len(ln) <= 400 \
                    and distinctive_line(ln):
                add("line", "strong", ln)
    return units[:MAX_UNITS_PER_RESULT]


def weak_idents(text, df):
    """Rare identifier-shaped tokens for the weak-link fallback (same rule
    as the earlier F4 prototype): len 8-40, identifier-like, occurring in at
    most 3 segments of the transcript."""
    out = []
    for m in IDENT_RE.finditer(text):
        w = m.group(0)
        wl = w.lower()
        if not (8 <= len(w) <= 40) or wl in GENERIC_TOKENS:
            continue
        if not ("_" in w or any(c.isdigit() for c in w)
                or re.search(r"[a-z][A-Z]", w) or len(w) >= 12):
            continue
        if df.get(wl, 0) > 3:
            continue
        out.append(w)
    return list(dict.fromkeys(out))[:15]


class JoinIndex:
    """Joined normalized text for a subset of segments, with per-segment
    char offsets — a single str.find answers 'does a segment >= start
    contain needle' and maps a hit offset back to the owning segment
    (verified against the segment's own text so a needle spanning the
    join's space separator can't fabricate a match)."""

    def __init__(self, norm_segs, seg_kinds, kinds):
        self.norm_segs = norm_segs
        self.idxs = [i for i, k in enumerate(seg_kinds)
                     if kinds is None or k in kinds]
        parts, offs, pos = [], [], 0
        for i in self.idxs:
            offs.append(pos)
            parts.append(norm_segs[i])
            pos += len(norm_segs[i]) + 1
        self.offs = offs
        self.join = " ".join(parts)

    def find(self, needle, start):
        """First segment index >= start (among this join's kinds) containing
        needle verbatim, or None."""
        lo = bisect.bisect_left(self.idxs, start)
        pos = self.offs[lo] if lo < len(self.offs) else len(self.join) + 1
        while True:
            p = self.join.find(needle, pos)
            if p < 0:
                return None
            k = bisect.bisect_right(self.offs, p) - 1
            if k < 0 or k >= len(self.idxs):
                return None
            idx = self.idxs[k]
            if needle in self.norm_segs[idx]:
                return idx
            pos = p + 1


def joins_for(segs):
    norm_segs = [f2m.norm(s["text"]) for s in segs]
    seg_kinds = [s["kind"] for s in segs]
    return norm_segs, {
        "assistant": JoinIndex(norm_segs, seg_kinds, ASSISTANT_USAGE_KINDS),
        "user": JoinIndex(norm_segs, seg_kinds, USER_USAGE_KINDS),
        "all": JoinIndex(norm_segs, seg_kinds, None),
    }


def find_later(joins, norm_segs, seg_kinds, start, needles):
    """Best later usage of any needle: assistant-authored kinds prove
    consumption (strong/medium); a user_turn echo is weak. Returns
    (strength, seg_idx, unit_kind, needle, usage_kind) or None."""
    rank = {"strong": 0, "medium": 1}
    best_a, best_u = None, None
    for unit_kind, strength, needle in needles:
        ja = joins["assistant"].find(needle, start)
        if ja is not None:
            cand = (strength, ja, unit_kind, needle,
                    seg_kinds[ja])
            if best_a is None or (rank[strength], ja) < (rank[best_a[0]],
                                                        best_a[1]):
                best_a = cand
        ju = joins["user"].find(needle, start)
        if ju is not None and best_u is None:
            best_u = ("weak", ju, unit_kind, needle, "user_turn")
    return best_a or best_u


def span_of(user_seg, idx):
    """(span_lo, span_hi) = segment bounds of the task span containing idx:
    after the previous user turn through the next user turn (exclusive)."""
    lo = 0
    for u in user_seg:
        if u < idx:
            lo = u + 1
        else:
            return lo, u
    return lo, 10 ** 9


def same_span_bounded(user_seg, r, k):
    """usage k is a legal same-task usage of candidate r: no user turn
    strictly between them and gap <= LINK_MAX_GAP."""
    if k - r > LINK_MAX_GAP:
        return False
    return not any(r < u <= k for u in user_seg)


# --- pool ledger --------------------------------------------------------------

def load_pool_ledger(usable, eval_hashes, skips):
    """Resolve (train_hashes, reserved_hashes, frozen_hashes, provenance).

    Order: an existing canonical data/v5_mining/pool_split.json wins; else
    the f2_report.json pool_split (present for this build); else compute the
    fallback rule (eval_pool = first 18 usable by sorted hash not in
    eval-v1's 60; train = rest). Writes the canonical ledger when absent.
    """
    if POOL_LEDGER.is_file():
        led = json.loads(POOL_LEDGER.read_text(encoding="utf-8"))
        train = set(led.get("train_pool_hashes", []))
        reserved = set(led.get("eval_pool_hashes", []))
        frozen = set(led.get("eval_v1_frozen_hashes", [])) or eval_hashes
        return train, reserved, frozen, {
            "canonical_path": str(POOL_LEDGER.relative_to(ROOT)),
            "source": led.get("source", "pool_split.json"),
            "note": "pre-existing canonical ledger honored as-is",
        }

    ledger_entries = lambda us: [
        {"transcript_hash": u["transcript_hash"], "source": u["source"],
         "segment_count": u["nseg"], "user_message_count": u["nusers"]}
        for u in us]

    if F2_REPORT.is_file():
        rep = json.loads(F2_REPORT.read_text(encoding="utf-8"))
        ps = rep.get("pool_split") or {}
        train_e = (ps.get("train_pool") or {}).get("transcripts") or []
        res_e = (ps.get("v5_eval_reserved") or {}).get("transcripts") or []
        train = {e["transcript_hash"] for e in train_e}
        reserved = {e["transcript_hash"] for e in res_e}
        frozen = set((ps.get("eval_v1_frozen") or {})
                     .get("transcript_hashes") or []) or eval_hashes
        prov = {"canonical_path": str(POOL_LEDGER.relative_to(ROOT)),
                "source": "data/v5_mining/f2_report.json:pool_split",
                "rule": ps.get("rule"),
                "note": ("F2 ledger honored verbatim; canonical copy "
                         "materialized at pool_split.json for future "
                         "miners")}
    else:
        remaining = sorted((u for u in usable
                            if u["transcript_hash"] not in eval_hashes),
                           key=lambda u: u["transcript_hash"])
        res_us = remaining[:FALLBACK_EVAL_RESERVE]
        train_us = remaining[FALLBACK_EVAL_RESERVE:]
        train = {u["transcript_hash"] for u in train_us}
        reserved = {u["transcript_hash"] for u in res_us}
        frozen = set(eval_hashes)
        train_e = ledger_entries(train_us)
        res_e = ledger_entries(res_us)
        prov = {"canonical_path": str(POOL_LEDGER.relative_to(ROOT)),
                "source": "computed_by_mine_v5_outcome_positive_v1",
                "rule": ("eval_pool = first 18 usable transcripts by sorted "
                         "transcript_hash not in eval-v1's 60; train_pool = "
                         "rest"),
                "note": ("F2 ledger was absent — split computed here and "
                         "written to the canonical path for both miners")}
    # sanity: ledger must be disjoint and cover the same usable universe
    usable_hashes = {u["transcript_hash"] for u in usable}
    overlap = (train & reserved) or (train & frozen) or (reserved & frozen)
    if overlap:
        skips["pool_ledger_overlap"] += len(overlap)
        raise SystemExit(f"pool ledger overlap: {sorted(overlap)[:4]}")
    missing = usable_hashes - train - reserved - frozen
    if missing:
        skips["pool_ledger_unassigned"] += len(missing)
        # transcripts that entered the pool after the ledger was written
        # default to eval-reserve (conservative: never mine unledgered)
        reserved |= missing
        prov["unassigned_added_to_reserve"] = len(missing)
    if not POOL_LEDGER.is_file():
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "nanojev-v5-pool-split-v1",
            "written_by": "scripts/mine_v5_outcome_positive_v1.py",
            "seed": SEED,
            "source": prov["source"],
            "rule": prov.get("rule"),
            "eval_v1_frozen_hashes": sorted(frozen),
            "eval_pool_hashes": sorted(reserved),
            "train_pool_hashes": sorted(train),
            "counts": {"eval_v1_frozen": len(frozen),
                       "eval_pool": len(reserved), "train_pool": len(train)},
            "note": ("canonical v5 mining pool ledger; honor this file (or "
                     "f2_report.json pool_split, same content) in every v5 "
                     "miner. eval_pool transcripts are reserved for v5-eval "
                     "mining and must stay out of train."),
        }
        POOL_LEDGER.write_text(json.dumps(payload, ensure_ascii=False,
                                          indent=2) + "\n",
                               encoding="utf-8")
    return train, reserved, frozen, prov


def load_f2_pointer_labels():
    """(transcript_hash, candidate_pointer) -> proposed_label from the F2
    candidates file (label-conflict coordination)."""
    m = {}
    if not F2_JSONL.is_file():
        return m
    for line in F2_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        meta = rec.get("meta", {})
        th, cp = meta.get("transcript_hash"), meta.get("candidate_pointer")
        if th and cp:
            m[(th, cp)] = meta.get("proposed_label")
    return m


# --- mining -------------------------------------------------------------------

def mine_transcript(item, skips):
    """Collect outcome-linker proposals for one transcript. Item dict shape
    matches mine_v5_hard_negatives_v1.scan_transcripts output."""
    th, t = item["transcript_hash"], item["transcript"]
    segs = t.segments
    n = len(segs)
    flagged = item["flagged"]
    ffu = item["first_flagged_user"]
    n_users = len(t.users)
    user_seg = [i for i, s in enumerate(segs) if s["kind"] == "user_turn"]
    user_pos = {s_: i for i, s_ in enumerate(user_seg)}
    seg_kinds = [s["kind"] for s in segs]
    norm_segs, joins = joins_for(segs)
    norm_lower = [x.lower() for x in norm_segs]

    def anchor_for_usage(usage_idx):
        """Anchor = last user turn at/before the usage segment."""
        prev = [s_ for s_ in user_seg if s_ <= usage_idx]
        if not prev:
            return None, None
        a_seg = prev[-1]
        return a_seg, user_pos[a_seg]

    def anchor_ok(a_uid):
        return a_uid is not None and a_uid < ffu

    # identifier document frequency (weak-link rareness test)
    df_ident = Counter()
    for x in norm_lower:
        df_ident.update({w for w in IDENT_RE.findall(x)})

    pairs = pair_results(segs)
    res_use = {r: u for r, u in pairs}
    res_sorted = sorted(res_use)
    ops_by_use = {i: [(op, [clean_op_path(p) for p in ps])
                      for op, ps in f2m.tool_call_ops(s["text"])]
                  for i, s in enumerate(segs) if s["kind"] == "tool_use"}

    # echo-unit extraction once per unique (digit-normalized) result text;
    # unit_df counts how many DISTINCT tool_results yield each needle —
    # repeated boilerplate reaches df > ECHO_DF_MAX and is inadmissible.
    # (A needle surviving this prefilter is re-verified at claim time with a
    # true text-membership count over all tool_results via res_df().)
    key_results = defaultdict(list)
    for r in res_sorted:
        key_results[f2m.boilerplate_key(segs[r]["text"])].append(r)
    result_units, unit_df, n_needles = {}, Counter(), 0
    for key in sorted(key_results):
        rs = key_results[key]
        if n_needles > MAX_NEEDLES_PER_TRANSCRIPT:
            units = []
        else:
            units = echo_units(segs[rs[0]]["text"])
            n_needles += len(units)
        for _, _, nd in units:
            unit_df[nd] += len(rs)
        for r in rs:
            result_units[r] = units
    res_df_cache = {}

    def res_df(needle):
        """True membership count of needle over tool_result texts (the
        strict boilerplate dedup guard, computed lazily for winning units
        only)."""
        if needle not in res_df_cache:
            res_df_cache[needle] = sum(
                1 for r in res_sorted if needle in norm_segs[r])
        return res_df_cache[needle]

    props = []  # {seg, usage_seg|None, anchor_seg, anchor_uidx, cut, ...}

    def claim(i, usage, strategy, side, label, strength, note, evidence):
        a_seg, a_uid = anchor_for_usage(usage if usage is not None else i)
        if a_seg is None or a_seg == i or not anchor_ok(a_uid):
            skips["anchor_blocked"] += 1
            return None
        return {"seg": i, "usage_seg": usage, "anchor_seg": a_seg,
                "anchor_uidx": a_uid,
                "cut": (usage + 1) if usage is not None else a_seg,
                "strategy": strategy, "side": side, "label": label,
                "strength": strength, "note": note, "evidence": evidence}

    # --- 1) file_read_edit -----------------------------------------------------
    edit_ops = [(i, p) for i, ops in ops_by_use.items()
                for op, ps in ops for p in ps if op == "edit"]
    strong_hits, weak_hits = {}, {}
    for k, p in edit_ops:
        np_ = norm_path(p)
        hit = None
        for r in reversed(res_sorted):
            if r >= k:
                continue
            if not same_span_bounded(user_seg, r, k):
                continue
            u_ = res_use[r]
            ops = ops_by_use.get(u_, []) if u_ is not None else []
            if any(op == "read" and path_match(np_, q)
                   for op, ps in ops for q in ps):
                hit = (r, "read_call_same_path")
                break
            if path_in_text(np_, norm_lower[r]):
                if any(op == "edit" and path_match(np_, q)
                       for op, ps in ops for q in ps):
                    # the result is an edit-confirmation, not read content
                    weak_hits.setdefault(r, (k, "edit_confirm_then_reedit", p))
                else:
                    hit = (r, "path_in_result_text")
                break
        if hit is not None:
            r, link = hit
            if r not in strong_hits or k < strong_hits[r][0]:
                strong_hits[r] = (k, link, p)
            continue
        base = np_.rsplit("/", 1)[-1]
        for r in reversed(res_sorted):
            if r >= k or not same_span_bounded(user_seg, r, k):
                continue
            u_ = res_use[r]
            if u_ is not None and any(
                    op == "list" and
                    np_.startswith(norm_path(d).rstrip("/") + "/")
                    for op, ps in ops_by_use.get(u_, []) for d in ps):
                weak_hits.setdefault(r, (k, "list_dir_then_child_edit", p))
                break
            if len(base) >= 6 and base in norm_lower[r] and u_ is not None \
                    and any(op == "list" for op, _ in ops_by_use.get(u_, [])):
                weak_hits.setdefault(r, (k, "listed_file_later_edited", p))
                break

    file_cands = []
    for r, (k, link, p) in sorted(strong_hits.items()):
        file_cands.append((r, k, link, p, N, "strong",
            f"tool_result holding content of {p} precedes a later "
            f"Edit/Write/apply_patch on the same path ({link}), same task "
            f"span, gap {k - r} segs <= {LINK_MAX_GAP}; the read was "
            f"provably used by the edit"))
    for r, (k, link, p) in sorted(weak_hits.items()):
        if r in strong_hits:
            continue
        file_cands.append((r, k, link, p, U, "weak",
            f"directory-listing/edit-confirmation result precedes an Edit "
            f"inside the listed dir ({p}); weaker link — the result informed "
            f"but did not carry the file content"))
    file_cands.sort(key=lambda c: (0 if c[4] == N else 1, c[1] - c[0], c[0]))
    for r, k, link, p, label, strength, note in \
            file_cands[:MAX_FILE_PER_TRANSCRIPT]:
        pr = claim(r, k, "file_read_edit", "keep", label, strength, note,
                   {"signal": "file_read_edit", "link": link,
                    "path": norm_path(p), "usage_seg": k,
                    "usage_pointer": f"/messages/{k}/content",
                    "link_gap_segments": k - r,
                    "same_task_span": True})
        if pr is not None:
            props.append(pr)

    # --- 2) test_failure_fix ----------------------------------------------------
    test_cands = []
    for r in res_sorted:
        text = segs[r]["text"]
        if not FAILURE_RE.search(text):
            continue
        u_ = res_use[r]
        cmd = segs[u_]["text"] if u_ is not None else ""
        if not (VERIFY_CMD_RE.search(cmd) or TEST_SUMMARY_RE.search(text)):
            continue
        # failure identifiers (test names, ns::paths, long idents) on
        # failure-bearing lines
        fail_idents = set()
        for ln in text.splitlines():
            if FAILURE_RE.search(ln):
                for m in FAIL_IDENT_RE.finditer(ln):
                    wl = m.group(0).lower()
                    if wl not in GENERIC_TOKENS:
                        fail_idents.add(wl)
        link, k, detail, strength = None, None, None, "strong"
        # (a) a later edit/patch in the same span = the fix; strongest when
        # the edit path is named in the failure output, else it is just the
        # next step and the link is weaker.
        nxt_edit_strong = next(
            ((ek, p) for ek, p in edit_ops
             if ek > r and same_span_bounded(user_seg, r, ek)
             and path_in_text(norm_path(p), norm_lower[r])), None)
        nxt_edit_any = next(
            ((ek, p) for ek, p in edit_ops
             if ek > r and same_span_bounded(user_seg, r, ek)), None)
        # (b) a git fix-up commit in the same span
        nxt_git = None
        for j in range(r + 1, min(n, r + LINK_MAX_GAP + 1)):
            if any(r < uu <= j for uu in user_seg):
                break
            if segs[j]["kind"] == "tool_use" and \
                    GIT_FIX_RE.search(segs[j]["text"]) and \
                    "commit" in segs[j]["text"]:
                nxt_git = j
                break
        if nxt_edit_strong is not None and (nxt_git is None
                                          or nxt_edit_strong[0] <= nxt_git):
            link, k = "fix_edit_after_failure", nxt_edit_strong[0]
            detail = {"fixed_path": norm_path(nxt_edit_strong[1])}
        elif nxt_git is not None:
            link, k = "fix_commit_after_failure", nxt_git
            detail = {}
        else:
            # (c) later assistant message references the failure
            for j in range(r + 1, min(n, r + LINK_MAX_GAP + 1)):
                if any(r < uu <= j for uu in user_seg):
                    break
                if segs[j]["kind"] != "assistant_text":
                    continue
                low = norm_lower[j]
                hits = [fi for fi in fail_idents if fi in low]
                echo_hit = next(
                    (nd for _, _, nd in result_units.get(r, [])
                     if unit_df[nd] <= ECHO_DF_MAX and nd in norm_segs[j]),
                    None)
                if echo_hit or (hits and FIX_TALK_RE.search(norm_segs[j])):
                    link, k = "failure_referenced", j
                    detail = {"failing_idents": sorted(hits)[:4]}
                    if echo_hit:
                        detail["echoed"] = echo_hit[:160]
                    break
            if link is None and nxt_edit_any is not None:
                # next-step edit on a path NOT named in the failure output:
                # plausible fix sequence but the link is inferential.
                link, k = "next_step_edit_after_failure", nxt_edit_any[0]
                detail = {"fixed_path": norm_path(nxt_edit_any[1])}
                strength = "medium"
        if link is None:
            continue
        label = N if strength == "strong" else U
        test_cands.append((r, k, link, detail, strength, label,
            f"failing test/verification output followed by {link} within "
            f"the same task span (gap {k - r} segs); the failure output "
            f"provably drove the next action"
            if strength == "strong" else
            f"failing test/verification output followed by a next-step edit "
            f"({detail.get('fixed_path')}) not named in the output; "
            f"inferential link — uncertain"))
    test_cands.sort(key=lambda c: (0 if c[5] == N else 1, c[0]))
    for r, k, link, detail, strength, label, note in \
            test_cands[:MAX_TEST_PER_TRANSCRIPT]:
        ev = {"signal": "test_failure_fix", "link": link, "usage_seg": k,
              "usage_pointer": f"/messages/{k}/content",
              "link_gap_segments": k - r, "same_task_span": True}
        ev.update(detail)
        pr = claim(r, k, "test_failure_fix", "keep", label, strength,
                   note, ev)
        if pr is not None:
            props.append(pr)

    # --- 3) command_output_echo -------------------------------------------------
    echo_cands = []
    for r in res_sorted:
        units = [(uk, st, nd) for uk, st, nd in result_units.get(r, [])
                 if unit_df[nd] <= ECHO_DF_MAX]
        best = None
        # verify the winning needle with the strict true-df guard; a
        # boilerplate needle that slipped the prefilter is removed and the
        # next-best unit is tried.
        while units:
            best = find_later(joins, norm_segs, seg_kinds, r + 1, units)
            if best is None:
                break
            if res_df(best[3]) <= ECHO_DF_MAX:
                break
            skips["echo_boilerplate_df"] += 1
            units = [u_ for u_ in units if u_[2] != best[3]]
            best = None
        if best is None:
            # weak fallback: a rare long identifier echoed later
            for w in weak_idents(segs[r]["text"], df_ident):
                wl = " " + w.lower() + " "
                ja = joins["assistant"].find(" " + wl + " ", r + 1)
                if ja is not None:
                    best = ("weak", ja, "rare_ident", w, seg_kinds[ja])
                    break
                ju = joins["user"].find(" " + wl + " ", r + 1)
                if ju is not None:
                    best = ("weak", ju, "rare_ident", w, "user_turn")
                    break
        if best is None:
            continue
        strength, j, unit_kind, needle, usage_kind = best
        label = U if strength == "weak" else N
        note = (
            f"tool_result {unit_kind} unit echoed verbatim by segment {j} "
            f"({usage_kind}); the output was provably incorporated "
            f"downstream"
            if label == N else
            f"only a {'user-turn echo' if usage_kind == 'user_turn' else 'rare identifier'} "
            f"({needle[:60]}) from the tool_result recurs at segment {j}; "
            f"weak link — marked uncertain")
        echo_cands.append((strength, r, j, unit_kind, needle, usage_kind,
                           label, note))
    rank = {"strong": 0, "medium": 1, "weak": 2}
    echo_cands.sort(key=lambda c: (rank[c[0]], c[1]))
    for strength, r, j, unit_kind, needle, usage_kind, label, note in \
            echo_cands[:MAX_ECHO_PER_TRANSCRIPT]:
        _, span_hi = span_of(user_seg, r)
        pr = claim(r, j, "command_output_echo", "keep", label, strength,
                   note,
                   {"signal": "command_output_echo", "link": "verbatim_echo",
                    "unit_kind": unit_kind, "matched": needle[:160],
                    "usage_seg": j,
                    "usage_pointer": f"/messages/{j}/content",
                    "usage_kind": usage_kind,
                    "link_gap_segments": j - r,
                    "same_task_span": j < span_hi,
                    "needle_result_df": (res_df(needle)
                                         if unit_kind != "rare_ident"
                                         else None)})
        if pr is not None:
            props.append(pr)

    # --- 4) unused_drop_control (negative control) -------------------------------
    claimed = {p["seg"] for p in props}
    subj = [f2m.subject_tokens(s["text"]) for s in segs]
    suf_strong, suf_weak = [set() for _ in range(n + 1)], \
                           [set() for _ in range(n + 1)]
    st_acc, wk_acc = set(), set()
    for k in range(n - 1, -1, -1):
        st_acc = st_acc | subj[k][0]   # new set each step — suf_* must not
        wk_acc = wk_acc | subj[k][1]   # alias one mutating accumulator
        suf_strong[k], suf_weak[k] = st_acc, wk_acc

    def negative_check(r):
        """All reach-back probes must return empty for the drop proposal."""
        checks = {}
        # (a) path/subject re-reference anywhere later; identifiers count
        # only when rare in the transcript (df<=4) — common idents can't
        # attribute a mention to this result.
        st, wk = subj[r]
        checks["path_rereferenced"] = sorted(st & suf_strong[min(r + 1, n)])
        id_hits = sorted(wk & suf_weak[min(r + 1, n)])
        checks["idents_rereferenced"] = [
            w for w in id_hits if df_ident.get(w, 0) <= 4][:6]
        # (b) verbatim signature quoted later (f2 signatures)
        sigs = f2m.signatures(norm_segs[r])
        checks["signature_echoed"] = any(
            joins["assistant"].find(sig, r + 1) is not None
            or joins["user"].find(sig, r + 1) is not None
            for sig in sigs)
        # (c) any extracted echo unit reappears later — for the drop claim
        # we deliberately skip the df filter: even boilerplate reappearing
        # counts against 'never referenced again' (conservative).
        units = result_units.get(r, [])
        checks["distinctive_echo"] = bool(units) and (
            find_later(joins, norm_segs, seg_kinds, r + 1, units) is not None)
        # (d) a later op touches a path this result's call read/listed
        u_ = res_use[r]
        paths = [p for op, ps in (ops_by_use.get(u_, []) if u_ is not None
                                  else []) for p in ps]
        touched = False
        for j, ops in ops_by_use.items():
            if j <= r:
                continue
            for op, ps in ops:
                for p in paths:
                    if any(path_match(p, q) for q in ps):
                        touched = True
        checks["path_retouched_by_later_op"] = touched
        clean = not (checks["path_rereferenced"]
                     or checks["idents_rereferenced"]
                     or checks["signature_echoed"]
                     or checks["distinctive_echo"]
                     or checks["path_retouched_by_later_op"])
        return clean, checks

    ctrl_pool = []
    for r in res_sorted:
        if r in claimed or r in flagged:
            continue
        if len(norm_segs[r]) < CTRL_MIN_CHARS:
            continue
        # task must have moved on: a later user turn exists
        a_uid = next((k for k in range(n_users) if user_seg[k] > r), None)
        if a_uid is None or a_uid >= ffu:
            continue
        clean, checks = negative_check(r)
        if not clean:
            skips["control_reachback_hit"] += 1
            continue
        ctrl_pool.append((r, a_uid, checks))
    ctrl_by_seg = {c[0]: c for c in ctrl_pool}
    for r in f2m.evenly_spaced(sorted(ctrl_by_seg), MAX_CTRL_PER_TRANSCRIPT):
        _, a_uid, checks = ctrl_by_seg[r]
        a_seg = user_seg[a_uid]
        cut = user_seg[a_uid + 1] if a_uid + 1 < n_users else n
        checks = dict(checks)
        checks["signal"] = "unused_drop_control"
        checks["moved_on_anchor_seg"] = a_seg
        props.append({
            "seg": r, "usage_seg": None, "anchor_seg": a_seg,
            "anchor_uidx": a_uid, "cut": cut,
            "strategy": "unused_drop_control", "side": "drop",
            "label": Y, "strength": "control",
            "note": ("negative control: substantive tool_result with no "
                     "downstream usage link — no path re-reference, no "
                     "quoted signature/echo, no later op on its paths — and "
                     f"the task moved on at user turn {a_uid}; eligible "
                     "drop candidate pending adjudication"),
            "evidence": checks})

    # --- merge: one row per segment, priority, caps ------------------------------
    # test_failure_fix outranks file_read_edit for the same result: a
    # traceback merely *names* the file — the verified claim is that the
    # failure output drove the fix, not that file content was read.
    prio = {"test_failure_fix": 0, "file_read_edit": 1,
            "command_output_echo": 2, "unused_drop_control": 3}
    props.sort(key=lambda p: (prio[p["strategy"]],
                              0 if p["label"] == N else 1, p["seg"]))
    seen_seg, out, n_unc = set(), [], 0
    for p in props:
        if p["seg"] in seen_seg:
            skips["dup_segment"] += 1
            continue
        if p["label"] == U:
            if n_unc >= MAX_UNCERTAIN_PER_TRANSCRIPT:
                skips["uncertain_cap"] += 1
                continue
            n_unc += 1
        if len(out) >= MAX_PER_TRANSCRIPT:
            skips["per_transcript_cap"] += 1
            break
        seen_seg.add(p["seg"])
        out.append(p)
    return out


def assign_pairs(emitted, th):
    """Contrastive pairs (doc §1.7): a usage-proven keep and a
    reach-back-clean drop control under the SAME anchor ->
    v5f4pair:<h>:use:aNNNN, pair_kind='same_anchor'."""
    by_anchor = defaultdict(list)
    for rec, prop in emitted:
        by_anchor[prop["anchor_seg"]].append((rec, prop))
    for a_seg, members in sorted(by_anchor.items()):
        keeps = [(rc, p) for rc, p in members if p["label"] == N]
        drops = [(rc, p) for rc, p in members if p["label"] == Y]
        if keeps and drops:
            pid = f"v5f4pair:{th[:12]}:use:a{a_seg:04d}"
            for rc, p in keeps[:1] + drops[:1]:
                if not rc["meta"]["pair_id"]:
                    rc["meta"]["pair_id"] = pid
                    rc["meta"]["pair_kind"] = "same_anchor"


# --- emit ----------------------------------------------------------------------

def emit(item, prop, instructions, f2_map, used_ids, skips):
    th, t, src = item["transcript_hash"], item["transcript"], item["source"]
    segs = t.segments
    idx, a_seg, a_uid = prop["seg"], prop["anchor_seg"], prop["anchor_uidx"]
    usage = prop["usage_seg"]
    if idx == a_seg or idx >= prop["cut"]:
        skips["candidate_out_of_scope"] += 1
        return None
    if usage is not None and (usage <= idx or usage >= prop["cut"]):
        skips["usage_out_of_scope"] += 1
        return None
    seg = segs[idx]
    want_kind = "tool_result"  # every F4 signal candidates a tool_result
    if seg["kind"] != want_kind or seg["role"] in ("control", "system"):
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
    usage_ptr = (f"/messages/{usage}/content" if usage is not None else None)
    f2_label = f2_map.get((th, ptr))
    if f2_label == Y and prop["label"] == N:
        skips["f2_label_conflict"] += 1
        return None
    if f2_label == N and prop["label"] == Y:
        skips["f2_label_conflict"] += 1
        return None
    prefix = segs[:prop["cut"]]
    user_list = t.users[:a_uid + 1]
    conv, windowed = f2m.windowed_conversation(prefix, idx, a_seg, user_list,
                                               blocked=item["flagged"])
    state = {"conversation": conv, "candidate_pointer": ptr,
             "user_messages_in_order": user_list}
    state_str = builder.serialized(state)
    if len(state_str.encode("utf-8")) > STATE_CHAR_HARD:
        skips["exceeds_budget"] += 1
        return None
    cred_parts = [s["content"] for s in conv] + list(user_list)
    if any(builder.credential_scan(part) for part in cred_parts):
        skips["cred_state_blocked"] += 1
        return None
    rid = f"v5f4:{th[:12]}:seg{idx:04d}:a{a_seg:04d}"
    if rid in used_ids:
        rid = f"{rid}-{prop['strategy'][:4]}"
    used_ids.add(rid)
    evidence = dict(prop["evidence"])
    evidence.setdefault("signal", prop["strategy"])
    if usage is not None:
        evidence.setdefault("usage_seg", usage)
        evidence.setdefault("usage_pointer", usage_ptr)
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
        "candidate_seg": idx,
        "length_band": builder.length_band(len(segs)),
        "selection_method": f"outcome_linker:{prop['strategy']}",
        "supp_source": "outcome_linker",
        "mine_mode": prop["strategy"],
        "strategy": prop["strategy"],
        "mining_side": prop["side"],
        "anchor_user_index": a_uid,
        "anchor_pointer": anchor_ptr,
        "usage_seg": usage,
        "usage_pointer": usage_ptr,
        "evidence_strength": prop["strength"],
        "evidence_basis": evidence,
        "windowed": windowed,
        "state_chars": len(state_str.encode("utf-8")),
        "proposed_label": prop["label"],
        "proposal_note": prop["note"],
        "pair_id": None,   # assigned post-emit by assign_pairs
        "pool": "train",
    }
    if f2_label is not None:
        meta["f2_same_pointer"] = f2_label
    return {
        "group_id": f"{GROUP_LINEAGE}:{th}",
        "request": {"state": state_str,
                    "questions": {"irrelevant": {"type": "noul",
                                               "instructions": instructions}}},
        "targets": None,
        "meta": meta,
    }


# --- self-check -----------------------------------------------------------------

VALID_STRATEGIES = {"file_read_edit", "test_failure_fix",
                    "command_output_echo", "unused_drop_control"}


def self_check(records, eval_hashes, reserved_hashes, train_hashes):
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
        meta = r["meta"]
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
        if user_segs[-1].get("pointer") != meta.get("anchor_pointer"):
            errors.append(f"{rid}: last user conv seg != anchor_pointer")
        if user_segs[-1].get("content") != users[-1]:
            errors.append(f"{rid}: last user seg != user_messages[-1]")
        for s in conv:
            hit = builder.credential_scan(s.get("content", ""))
            if hit is not None:
                errors.append(f"{rid}: credential_scan:{hit} in conv")
                break
        for u in users:
            hit = builder.credential_scan(u)
            if hit is not None:
                errors.append(f"{rid}: credential_scan:{hit} in users")
                break
        th = meta["transcript_hash"]
        if th in eval_hashes:
            errors.append(f"{rid}: transcript in eval_v1 pool")
        if th in reserved_hashes:
            errors.append(f"{rid}: transcript in v5_eval_reserved pool")
        if th not in train_hashes:
            errors.append(f"{rid}: transcript not in train pool ledger")
        if not rid.startswith("v5f4:"):
            errors.append(f"{rid}: bad record_id prefix")
        if r["targets"] is not None:
            errors.append(f"{rid}: targets must stay null")
        if meta.get("supp_source") != "outcome_linker":
            errors.append(f"{rid}: bad supp_source")
        if meta.get("strategy") not in VALID_STRATEGIES:
            errors.append(f"{rid}: bad strategy")
        eb = meta.get("evidence_basis")
        if not isinstance(eb, dict) or not eb.get("signal"):
            errors.append(f"{rid}: evidence_basis must be a dict w/ signal")
        if meta.get("mining_side") == "keep":
            if meta["proposed_label"] == Y:
                errors.append(f"{rid}: keep-side row proposed 'yes'")
            us = meta.get("usage_seg")
            if not isinstance(us, int) or us <= meta["candidate_seg"]:
                errors.append(f"{rid}: keep usage_seg not after candidate")
        if meta.get("mining_side") == "drop" \
                and meta["proposed_label"] not in (Y, U):
            errors.append(f"{rid}: drop-side row proposed 'no'")
    if errors:
        raise AssertionError(f"self_check failed ({len(errors)}): "
                             + "; ".join(errors[:10]))
    return {"records_checked": len(records), "status": "pass"}


# --- review render ---------------------------------------------------------------

def render_review(rows, item_by_hash, path, total):
    """Render the 20-row link-precision spot sample as markdown."""
    lines = [
        "# F4 `outcome_positive` — link-precision spot review",
        "",
        f"Deterministic seeded sample (seed {SEED}) of {len(rows)} rows out "
        f"of {total} emitted candidates. Each keep row cites the downstream "
        "usage segment that proves the claim (`evidence_basis`); verify the "
        "usage text genuinely consumes the candidate output. Drop rows are "
        "negative controls: the reach-back battery returned empty AND the "
        "task moved on at the anchor user turn.",
        "",
        "Labels: `no` = proposed keep, `yes` = proposed drop, "
        "`uncertain` = weak link (excluded from scored targets per spec §6).",
        "",
    ]
    for k, row in enumerate(rows, 1):
        lines.append(f"## {k}. `{row['record_id']}`")
        lines.append("")
        lines.append(f"- strategy `{row['strategy']}` · side "
                     f"`{row['side']}` · proposed_label "
                     f"**{row['proposed_label']}** · strength "
                     f"`{row.get('strength')}`")
        lines.append(f"- evidence_basis: `{json.dumps(row['evidence_basis'], ensure_ascii=False)[:400]}`")
        lines.append(f"- anchor (user turn): {row['anchor_snippet']}")
        lines.append(f"- candidate seg{row['seg']:04d} snippet:")
        lines.append("")
        lines.append("  > " + row["candidate_snippet"].replace("\n", " ")[:500])
        lines.append("")
        if row.get("usage_snippet"):
            lines.append(f"- usage seg{row['usage_seg']:04d} snippet:")
            lines.append("")
            lines.append("  > "
                         + row["usage_snippet"].replace("\n", " ")[:500])
            lines.append("")
        lines.append(f"- note: {row['note']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# --- main -------------------------------------------------------------------------

def main():
    rng = random.Random(SEED)
    instructions = builder.load_verbatim_instructions(
        ROOT / "data" / "valen_nano_v3")
    eval_hashes = f2m.load_eval_v1_hashes()
    ref_requests, ref_states = f2m.load_reference_hashes()
    # also dedup against the F2 candidates file (same pool, same lineage)
    for p in (F2_JSONL,):
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if "request" not in rec:
                continue
            ref_requests.add(builder.sha256_bytes(
                builder.serialized(rec["request"]).encode("utf-8")))
            sh = builder.normalized_state_hash(rec)
            if sh:
                ref_states.add(sh)
    f2_map = load_f2_pointer_labels()

    skips = Counter()
    usable, drops, scanned = f2m.scan_transcripts()
    train_hashes, reserved_hashes, frozen_hashes, pool_prov = \
        load_pool_ledger(usable, eval_hashes, skips)
    eval_hashes = eval_hashes | frozen_hashes
    train_pool = [u for u in usable if u["transcript_hash"] in train_hashes]

    records, report_rows, used_ids = [], [], set()
    seen_requests, seen_states, dedup = set(), set(), Counter()
    item_by_hash = {}
    for item in sorted(train_pool, key=lambda u: u["transcript_hash"]):
        item_by_hash[item["transcript_hash"]] = item
        props = mine_transcript(item, skips)
        emitted = []
        for prop in props:
            rec = emit(item, prop, instructions, f2_map, used_ids, skips)
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
        assign_pairs(emitted, item["transcript_hash"])
        for rec, prop in emitted:
            records.append(rec)
            segs = item["transcript"].segments
            u_seg = prop["usage_seg"]
            report_rows.append({
                "record_id": rec["meta"]["record_id"],
                "transcript": item["transcript_hash"][:12],
                "source": item["source"],
                "strategy": prop["strategy"],
                "side": prop["side"],
                "seg": prop["seg"],
                "usage_seg": u_seg,
                "anchor_seg": prop["anchor_seg"],
                "kind": rec["meta"]["candidate_kind"],
                "strength": prop["strength"],
                "proposed_label": prop["label"],
                "pair_id": rec["meta"]["pair_id"],
                "note": prop["note"],
                "evidence_basis": rec["meta"]["evidence_basis"],
                "candidate_snippet": f2m.norm(
                    segs[prop["seg"]]["text"])[:140],
                "usage_snippet": (f2m.norm(segs[u_seg]["text"])[:140]
                                  if u_seg is not None else None),
                "anchor_snippet": f2m.norm(
                    item["transcript"].users[prop["anchor_uidx"]])[:140],
            })

    records.sort(key=lambda r: (r["meta"]["transcript_hash"],
                                r["meta"]["record_id"]))

    check = self_check(records, eval_hashes, reserved_hashes, train_hashes)

    label_counts = Counter(r["meta"]["proposed_label"] for r in records)
    strat_counts = Counter(r["meta"]["strategy"] for r in records)
    strat_label = Counter((r["meta"]["strategy"],
                           r["meta"]["proposed_label"]) for r in records)
    strength_counts = Counter(r["meta"]["evidence_strength"]
                              for r in records)
    strat_strength = Counter((r["meta"]["strategy"],
                              r["meta"]["evidence_strength"])
                             for r in records)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in records:
            f.write(builder.serialized(r) + "\n")

    sample_idx = sorted(rng.sample(range(len(report_rows)),
                                   min(20, len(report_rows))))
    top20 = [report_rows[i] for i in sample_idx]
    render_review(top20, item_by_hash, OUT_REVIEW, len(report_rows))

    report = {
        "schema_version": "nanojev-v5-f4-outcome-positive-v1",
        "builder": "scripts/mine_v5_outcome_positive_v1.py",
        "seed": SEED,
        "created_from": ("v5 TRAIN-pool transcripts under ~/.claude/projects "
                         "+ ~/.codex/sessions (same roots/thresholds as "
                         "build_real_context_eval_v1.py; extraction byte cap "
                         "lifted to 12MB — records are windowed). Pool "
                         "membership comes from the shared ledger, NOT from "
                         "a fresh split."),
        "note": ("PROPOSALS ONLY: targets stay null; proposed_label + "
                 "proposal_note + evidence_basis are deterministic-linker "
                 "outputs pending the owner/agent adjudication spot pass "
                 "(F4 spec: >=10% of rows adjudicated to estimate linker "
                 "precision). Asymmetric semantics: yes == certainly "
                 "irrelevant == drop; this family is keep-side 'no' plus a "
                 "weak-link 'uncertain' band and the separate "
                 "unused_drop_control 'yes' proposals. No model/service "
                 "calls, no network."),
        "pool_ledger": {
            **pool_prov,
            "counts": {"eval_v1_frozen": len(eval_hashes),
                       "v5_eval_reserved": len(reserved_hashes),
                       "train_pool": len(train_hashes)},
            "train_pool_mined": len(train_pool),
        },
        "linker_signals": {
            "file_read_edit": ("tool_result holding content of path P "
                               "(paired read call or literal path in text) "
                               "+ later Edit/Write/apply_patch of P within "
                               f"the same task span and <= {LINK_MAX_GAP} "
                               "segments -> keep; dir-listing -> child-edit "
                               "is the weak uncertain form"),
            "test_failure_fix": ("failing test/verification output + "
                                 "same-span bounded follow-up (edit/patch, "
                                 "git fix commit, or assistant message "
                                 "referencing the failure) -> keep"),
            "command_output_echo": ("distinctive result unit (URL / "
                                    "digit-bearing hex hash / >=40-char "
                                    "distinctive line or chunk) echoed "
                                    "verbatim in a later assistant-authored "
                                    "segment -> keep; needle must occur in "
                                    f"<= {ECHO_DF_MAX} tool_result "
                                    "(boilerplate dedup guard); user-turn "
                                    "echo or rare-ident-only -> uncertain"),
            "unused_drop_control": ("NEGATIVE CONTROL: substantive "
                                    f"(>= {CTRL_MIN_CHARS} chars) "
                                    "tool_result with a clean reach-back "
                                    "battery (no path re-reference, no "
                                    "quoted signature, no admissible echo, "
                                    "no later op on its paths) AND a later "
                                    "user turn (task moved on) -> proposed "
                                    "'yes' as a separate proposal"),
        },
        "link_semantics": {
            "cut": ("keep rows: state = transcript prefix ending right "
                    "after the proving usage segment; drop controls: "
                    "re-anchor at the next user turn (task moved on), cut "
                    "at that anchor's span end"),
            "anchor": "last user turn at/before the usage segment "
                      "(keep); next user turn after the candidate (control)",
            "label_rule": ("strong/medium evidence -> 'no' (keep); "
                           "weak-only link -> 'uncertain'; clean control -> "
                           "'yes'"),
            "bounds": {"LINK_MAX_GAP_segments": LINK_MAX_GAP,
                       "ECHO_MIN_CHARS": ECHO_MIN_CHARS,
                       "ECHO_DF_MAX_per_tool_result": ECHO_DF_MAX,
                       "CTRL_MIN_CHARS": CTRL_MIN_CHARS},
        },
        "self_check": check,
        "dedup": {
            "reference_sets": ["data/valen_nano_v2", "data/valen_nano_v3",
                               "data/valen_nano_v4",
                               "data/real_context_eval_v1/candidates*.jsonl",
                               "data/v5_mining/f2_hard_negative_candidates.jsonl"],
            **dict(sorted(dedup.items())),
        },
        "counts": {
            "files_scanned": scanned,
            "usable_transcripts": len(usable),
            "train_pool_transcripts": len(train_pool),
            "unusable_drops": dict(sorted(drops.items())),
            "records": len(records),
            "proposed_labels": dict(sorted(label_counts.items())),
            "by_signal": dict(sorted(strat_counts.items())),
            "by_signal_label": {f"{k[0]}:{k[1]}": v
                                for k, v in sorted(strat_label.items())},
            "strength_distribution": dict(sorted(strength_counts.items())),
            "by_signal_strength": {f"{k[0]}:{k[1]}": v
                                   for k, v in sorted(strat_strength.items())},
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
        "link_precision_review": {
            "sampled": len(top20),
            "rendered_to": str(OUT_REVIEW.relative_to(ROOT)),
            "sample": top20,
        },
        "rows": report_rows,
        "caveats": [
            ("pool coordination: ledger honored from "
             f"{pool_prov.get('source')}; the F2 miner's identical pool "
             "view means F4 train rows and F2 train rows share the same 62 "
             "transcripts (overlap on the same candidate_pointer is "
             "recorded via meta.f2_same_pointer; direct label conflicts are "
             "skipped)."),
            ("only spec source (b) is covered: public resolved-task "
             "trajectories (SWE-Gym/SWE-chat) need downloads + license "
             "screening and were out of scope (no network). This supersedes "
             "scripts/mine_v5_outcome_v1.py, which mined all usable "
             "transcripts including 93 rows on eval-v1's 60 — its "
             "data/valen_nano_v5/f4_* outputs predate the pool-split "
             "protocol and are kept untouched for reference only."),
            ("tool_use->tool_result pairing is FIFO heuristic: rejected/"
             "resultless calls can misalign ops; the path-in-result-text "
             "link covers those cases content-side."),
            ("codex read/edit detection parses shell command strings "
             "(cat/sed/apply_patch) — approximate; claude Read/Edit "
             "dict inputs are exact."),
            ("verbatim echo is whitespace-normalized case-sensitive; "
             "paraphrased downstream use is not detected (linker recall, "
             "not precision, is the cost)."),
            (f"bounded signals: file_read_edit and test_failure_fix require "
             f"same-task span AND <= {LINK_MAX_GAP} segment gap; echo "
             f"claims are unbounded but require the {ECHO_MIN_CHARS}-char / "
             f"df<= {ECHO_DF_MAX} distinctiveness guard."),
            ("unused_drop_control proposals are the family's negative "
             "control: emitted only when the reach-back battery is empty "
             "AND a later user turn exists; they remain proposals pending "
             "adjudication like every other row. They share this file but "
             "are fenced by strategy='unused_drop_control' + "
             "mining_side='drop' so a merge can exclude them cleanly."),
            ("candidates are tool_result segments only: all four spec'd "
             "linker signals are result-anchored (Read/cat results, "
             "command output, test/verification output). Assistant-text "
             "candidates quoted downstream are a possible later extension, "
             "not in this v1."),
            ("'uncertain' rows are the deliberately weak band (user-turn "
             "echo / rare-ident / dir-listing links); excluded from scored "
             "targets per spec §6."),
        ],
    }
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"scanned {scanned} files -> {len(usable)} usable transcripts; "
          f"train pool {len(train_pool)}")
    print(f"wrote {len(records)} records -> {OUT_JSONL}")
    print("labels:", dict(sorted(label_counts.items())),
          "| signals:", dict(sorted(strat_counts.items())),
          "| strength:", dict(sorted(strength_counts.items())))
    print("skips:", dict(sorted(skips.items())),
          "| drops:", dict(sorted(drops.items())),
          "| dedup:", dict(sorted(dedup.items())))
    print(f"report -> {OUT_REPORT}")
    print(f"review -> {OUT_REVIEW}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
