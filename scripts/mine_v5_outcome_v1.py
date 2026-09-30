#!/usr/bin/env python3
"""V5 F4 outcome-positive miner: keep-labeled candidates where a segment was
provably USED downstream — hard keeps nobody can dispute.

Implements docs/V5_DATA_DESIGN_V1.md §5 (family F4, outcome_positive), source
(b) "local unused transcripts". Source (a) public resolved-task trajectories
requires downloads and is out of scope here (deterministic, no network); the
yield caveat is reported honestly in f4_report.json.

Downstream-usage linker (all deterministic, zero labeler cost for the
positive claim):

  * file_read_edit — a tool_result holding the content of path P (either its
    paired Read/cat tool_call names P, or the result text literally contains
    P) followed by a later Edit/Write/apply_patch tool call on P in the SAME
    transcript -> outcome-positive keep. Directory-listing results whose
    listed dir is an ancestor of an edited path are the weak-link form ->
    uncertain.
  * command_output_echo — a tool_result whose output is quoted verbatim in a
    LATER assistant_text / user_turn / tool_use segment. Evidence units:
    URLs, hex hashes (>=7 chars, digit-bearing), and distinctive normalized
    output lines (>=24 chars containing a path / >=3-digit run / long
    identifier). strong|medium units -> keep ('no'); a weak fallback (a rare
    long identifier echoed with no stronger unit) -> 'uncertain' — this is
    the "quoted only a generic word" weak-link class the task asks to flag.
  * user_cited_constraint — a non-final user_turn whose distinctive phrase
    (>=6 consecutive words, or a >=6-char CJK run, dedup'd) is echoed by a
    later assistant action / user message / the final request -> keep.
    Weak form: only a rare long identifier from the user text recurs ->
    'uncertain'.

Emission shape mirrors mine_v5_hardneg_v1.py (F2): record_id
"v5f4:<hash>:segNNNN", request{state, questions.irrelevant} with the
verbatim noul instructions, targets:null (proposals pending the
owner/agent adjudication pass), proposed_label mostly 'no' (= keep; the
valen asymmetric semantics has true/yes == certainly irrelevant == drop).

meta.evidence_basis is REQUIRED on every row: a dict carrying the
downstream pointer/seg-index that proves the keep (labeling-guide R8 style,
auditable without re-reading text).

Anchor/cut: the emitted state is the transcript prefix ending immediately
AFTER the proving usage segment (cut = usage_seg + 1), anchored at the last
user turn at/before the usage — the request context a filter would see at
the moment the segment gets used. user_messages_in_order is truncated at
that anchor, same as the F2 re-anchor shape.

Holdout collision rule (same as F2): a record is skipped iff its
(transcript_hash, candidate_pointer, anchor_pointer) triple matches a
frozen adjudicated pair (the 590 labeled rows + excluded rows); sharing a
transcript or a pointer alone is allowed and recorded in
meta.holdout_collision_check. Rows whose (transcript, candidate_pointer)
was already proposed 'yes' by the F2 miner are additionally skipped
(direct label conflict — the F4 usage proof would contradict the drop
proposal); other F2 overlaps are emitted with meta.f2_same_pointer set.

Determinism: fixed seed, sorted iteration, pure local rules, no model/
service calls, no network, no tokenizer (character-budget windowing like
the F1/F2 miners).

Usage: python3 scripts/mine_v5_outcome_v1.py
Output: data/valen_nano_v5/f4_mined_outcome.jsonl
        data/valen_nano_v5/f4_report.json
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
    "mine_v5_hardneg_v1", ROOT / "scripts" / "mine_v5_hardneg_v1.py")
f2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f2)
builder = f2.builder

OUT_DIR = ROOT / "data" / "valen_nano_v5"
OUT_JSONL = OUT_DIR / "f4_mined_outcome.jsonl"
OUT_REPORT = OUT_DIR / "f4_report.json"
F2_JSONL = OUT_DIR / "f2_mined_hardneg.jsonl"
HOLDOUT_FILES = f2.HOLDOUT_FILES

SEED = 20261201            # v5 reserved seed (V5_DATA_DESIGN_V1 §1.3)
STATE_CHAR_HARD = 14_000   # ~8192 packed-token compiler limit with headroom
GROUP_LINEAGE = "real_context_v5"

Y, N, U = "yes", "no", "uncertain"  # proposed_label values, v1 convention

# caps (all deterministic)
MAX_PER_TRANSCRIPT = 14
MAX_FILE_PER_TRANSCRIPT = 6
MAX_ECHO_PER_TRANSCRIPT = 6
MAX_USER_PER_TRANSCRIPT = 4
MAX_UNCERTAIN_PER_TRANSCRIPT = 3

PATH_TOKEN_RE = f2.PATH_TOKEN_RE
IDENT_RE = f2.IDENT_RE
DIGIT_RUN_RE = f2.DIGIT_RUN_RE
GENERIC_TOKENS = f2.GENERIC_TOKENS

# Evidence-unit regexes for command_output_echo.
URL_RE = re.compile(r"https?://[^\s'\"<>\)\]]{8,}")
HASH_RE = re.compile(r"\b(?=[0-9a-f]*\d)[0-9a-f]{7,40}\b")
CJK_RUN_RE = re.compile(r"[一-鿿぀-ヿ가-힯]{6,}")

STOPWORDS = GENERIC_TOKENS | {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "it", "its", "this", "that",
    "these", "those", "you", "your", "we", "our", "i", "me", "my", "not",
    "no", "yes", "do", "does", "did", "can", "could", "should", "would",
    "will", "just", "please", "than", "then", "when", "what", "how", "why",
    "which", "there", "here", "from", "into", "about", "also", "run",
    "make", "made", "use", "used", "using", "need", "needs", "want",
    "file", "files", "code", "data", "line", "lines", "text", "check",
    "see", "get", "set", "new", "now", "all", "any", "each", "same",
    "only", "first", "last", "完成", "一下", "这个", "那个", "已经",
}

# Segment kinds that may carry the *downstream usage* of an earlier result:
# assistant prose quoting output, a later user message pasting it back, or a
# tool_call whose input incorporates it (e.g. a commit message, an edited URL).
USAGE_KINDS = ("assistant_text", "user_turn", "tool_use")


def norm_path(p):
    p = str(p).strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p


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
    """Path literally visible in (normalized) result text; also tolerates
    abs-vs-rel via 2/3-component suffixes."""
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
    """FIFO pairing of tool_use -> tool_result segments. Returns
    [(result_idx, use_idx|None)] in segment order."""
    pending, pairs = [], []
    for i, s in enumerate(segs):
        if s["kind"] == "tool_use":
            pending.append(i)
        elif s["kind"] == "tool_result":
            pairs.append((i, pending.pop(0) if pending else None))
    return pairs


def distinctive_line(n):
    """A normalized output line worth echo-checking: carries a path, a
    >=3-digit run, or a long non-generic identifier."""
    if PATH_TOKEN_RE.search(n) or re.search(r"\d{3,}", n):
        return True
    for m in IDENT_RE.finditer(n):
        w = m.group(0)
        if len(w) >= 8 and w.lower() not in GENERIC_TOKENS:
            return True
    return False


def echo_units(text):
    """(unit_kind, strength, needle) extracted from a tool_result's raw text.
    needle is whitespace-normalized and matched verbatim (case preserved)
    against normalized later-segment text."""
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
    for line in text.splitlines():
        if len(units) >= 40:
            break
        n = f2.norm(line)
        if 24 <= len(n) <= 220 and distinctive_line(n):
            add("line", "strong" if len(n) >= 40 else "medium", n)
    return units


def weak_idents(text, df):
    """Rare identifier-shaped tokens for the weak-link fallback: len>=8,
    non-generic, identifier-like (snake_case / digit-bearing / camelCase /
    len>=12 — plain English words do not qualify), appearing in at most 3
    segments of the whole transcript."""
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


def user_phrase_units(text):
    """Distinctive phrases of a user turn: sliding 6-word shingles (dedup'd,
    must contain a non-stop word len>=5) plus CJK runs (>=12 chars ->
    sliding 10-char shingles; 6-11 chars -> whole run)."""
    n = f2.norm(text)
    units, seen = [], set()

    def add(kind, needle):
        if needle and needle not in seen:
            seen.add(needle)
            units.append((kind, "strong", needle))

    words = n.lower().split()
    for k in range(0, max(0, len(words) - 5)):
        sh = " ".join(words[k:k + 6])
        swords = set(sh.split())
        if not any(len(w) >= 5 and w not in STOPWORDS for w in swords):
            continue
        add("shingle", sh)
        if len(units) >= 30:
            break
    for m in CJK_RUN_RE.finditer(n):
        run = m.group(0)
        if len(run) >= 12:
            for k in range(0, len(run) - 9, 6):
                add("cjk", run[k:k + 10])
                if len(units) >= 50:
                    break
        else:
            add("cjk", run)
        if len(units) >= 50:
            break
    return units


def find_echo(segs, norm_segs, start, needles):
    """First later USAGE_KINDS segment containing each needle. Returns the
    best match (strength first, then earliest usage) or None."""
    rank = {"strong": 0, "medium": 1, "weak": 2}
    matches = []
    for unit_kind, strength, needle in needles:
        for j in range(start, len(segs)):
            if segs[j]["kind"] in USAGE_KINDS and needle in norm_segs[j]:
                matches.append((strength, j, unit_kind, needle))
                break
    if not matches:
        return None
    return min(matches, key=lambda m: (rank[m[0]], m[1]))


# --- mining -----------------------------------------------------------------

def mine_transcript(item, skips):
    """Collect downstream-usage keep proposals for one transcript."""
    th, t = item["transcript_hash"], item["transcript"]
    segs = t.segments
    n = len(segs)
    user_seg = [i for i, s in enumerate(segs) if s["kind"] == "user_turn"]
    user_pos = {s: i for i, s in enumerate(user_seg)}
    norm_segs = [f2.norm(s["text"]) for s in segs]
    norm_lower = [x.lower() for x in norm_segs]
    # ident document frequency over the whole transcript (rareness for weak
    # link tests; also keeps "generic word" echoes out of the keep class).
    df = Counter()
    for x in norm_lower:
        df.update({w for w in IDENT_RE.findall(x)})

    def anchor_for(usage_idx):
        """Anchor = last user turn at/before the usage segment; cut ends the
        state right after the usage so the proof is in-window when the
        window budget allows."""
        prev = [s for s in user_seg if s <= usage_idx]
        if not prev:
            return None, None
        return prev[-1], user_pos[prev[-1]]

    props = []  # dicts: seg, usage_seg, strategy, label, strength, note, evidence

    # --- 1) file_read_edit ----------------------------------------------------
    pairs = pair_results(segs)
    res_use = {r: u for r, u in pairs}
    res_sorted = sorted(res_use)
    ops_by_use = {i: f2.tool_call_ops(s["text"])
                  for i, s in enumerate(segs) if s["kind"] == "tool_use"}
    edit_ops = [(i, p) for i, ops in ops_by_use.items()
                for op, ps in ops for p in ps if op == "edit"]
    strong_hits = {}   # result_idx -> (usage_seg, link, path)
    weak_hits = {}     # result_idx -> (usage_seg, link, path)
    for k, p in edit_ops:
        np = norm_path(p)
        # nearest preceding result linked to this path wins the usage credit
        hit = None
        for r in reversed(res_sorted):
            if r >= k:
                continue
            u = res_use[r]
            ops = ops_by_use.get(u, []) if u is not None else []
            if any(op == "read" and path_match(np, q)
                   for op, ps in ops for q in ps):
                hit = (r, "read_call_same_path")
                break
            if path_in_text(np, norm_segs[r]):
                # an Edit's own confirmation result ("file X updated") is not
                # file content being read — a later Edit on X is a weak link
                if any(op == "edit" and path_match(np, q)
                       for op, ps in ops for q in ps):
                    weak_hits.setdefault(
                        r, (k, "edit_confirm_then_reedit", p))
                else:
                    hit = (r, "path_in_result_text")
                break
        if hit is not None:
            r, link = hit
            if r not in strong_hits or k < strong_hits[r][0]:
                strong_hits[r] = (k, link, p)
            continue
        # weak form: a directory-listing result whose listed dir (or listed
        # basename) precedes an edit inside that dir
        base = np.rsplit("/", 1)[-1]
        for r in reversed(res_sorted):
            if r >= k:
                continue
            u = res_use[r]
            if u is not None and any(
                    op == "list" and
                    np.startswith(norm_path(d).rstrip("/") + "/")
                    for op, ps in ops_by_use.get(u, []) for d in ps):
                weak_hits.setdefault(r, (k, "list_dir_then_child_edit", p))
                break
            if len(base) >= 6 and base in norm_lower[r] and u is not None \
                    and any(op == "list" for op, _ in ops_by_use.get(u, [])):
                weak_hits.setdefault(r, (k, "listed_file_later_edited", p))
                break

    file_cands = []
    for r, (k, link, p) in sorted(strong_hits.items()):
        a_seg, a_uid = anchor_for(k)
        if a_seg is None or a_seg == r:
            skips["anchor_is_candidate"] += 1
            continue
        file_cands.append({
            "seg": r, "usage_seg": k, "anchor_seg": a_seg,
            "anchor_uidx": a_uid, "cut": k + 1,
            "strategy": "file_read_edit", "label": N, "strength": "strong",
            "note": (f"tool_result holding content of {p} precedes a later "
                     f"Edit/Write/apply_patch on the same path ({link}); "
                     f"the read was provably used by the edit"),
            "evidence": {"link": link, "path": norm_path(p),
                         "usage_seg": k,
                         "usage_pointer": f"/messages/{k}/content",
                         "link_gap_segments": k - r},
        })
    for r, (k, link, p) in sorted(weak_hits.items()):
        if r in strong_hits:
            continue
        a_seg, a_uid = anchor_for(k)
        if a_seg is None or a_seg == r:
            continue
        file_cands.append({
            "seg": r, "usage_seg": k, "anchor_seg": a_seg,
            "anchor_uidx": a_uid, "cut": k + 1,
            "strategy": "file_read_edit", "label": U, "strength": "weak",
            "note": (f"directory-listing result precedes an Edit inside the "
                     f"listed dir ({p}); weaker link — the listing informed "
                     f"but did not carry the file content"),
            "evidence": {"link": link, "path": norm_path(p),
                         "usage_seg": k,
                         "usage_pointer": f"/messages/{k}/content",
                         "link_gap_segments": k - r},
        })
    file_cands.sort(key=lambda c: (0 if c["label"] == N else 1,
                                   c["usage_seg"] - c["seg"], c["seg"]))
    props.extend(file_cands[:MAX_FILE_PER_TRANSCRIPT])

    # --- 2) command_output_echo -----------------------------------------------
    echo_cands = []
    for r in res_sorted:
        units = echo_units(segs[r]["text"])
        best = find_echo(segs, norm_segs, r + 1, units) if units else None
        if best is None:
            # weak fallback: a rare long identifier echoed later
            for w in weak_idents(segs[r]["text"], df):
                wl = " " + w.lower() + " "
                for j in range(r + 1, n):
                    if segs[j]["kind"] in USAGE_KINDS and \
                            wl in (" " + norm_lower[j] + " "):
                        best = ("weak", j, "rare_ident", w)
                        break
                if best:
                    break
        if best is None:
            continue
        strength, j, unit_kind, needle = best
        a_seg, a_uid = anchor_for(j)
        if a_seg is None or a_seg == r:
            skips["anchor_is_candidate"] += 1
            continue
        label = U if strength == "weak" else N
        note = (
            f"tool_result {unit_kind} unit echoed verbatim by segment {j} "
            f"({segs[j]['kind']}); the output was provably incorporated "
            f"downstream"
            if label == N else
            f"only a rare identifier ({needle}) from the tool_result recurs "
            f"in segment {j}; weak link — marked uncertain")
        echo_cands.append({
            "seg": r, "usage_seg": j, "anchor_seg": a_seg,
            "anchor_uidx": a_uid, "cut": j + 1,
            "strategy": "command_output_echo", "label": label,
            "strength": strength,
            "note": note,
            "evidence": {"link": "verbatim_echo", "unit_kind": unit_kind,
                         "matched": needle[:160],
                         "usage_seg": j,
                         "usage_pointer": f"/messages/{j}/content",
                         "usage_kind": segs[j]["kind"],
                         "link_gap_segments": j - r},
        })
    echo_cands.sort(key=lambda c: ({"strong": 0, "medium": 1, "weak": 2}
                                   [c["strength"]], c["seg"]))
    props.extend(echo_cands[:MAX_ECHO_PER_TRANSCRIPT])

    # --- 3) user_cited_constraint ----------------------------------------------
    user_cands = []
    for i, s in enumerate(segs):
        if s["kind"] != "user_turn":
            continue
        if f2.is_continuation_anchor(s["text"]):
            continue
        units = user_phrase_units(s["text"])
        best = None
        if units:
            lower_needles = [(k_, st, sh.lower()) for k_, st, sh in units]
            best = find_echo(segs, norm_lower, i + 1, lower_needles)
            if best is not None:
                st, j, k_, sh_low = best
                # recover original-case phrase for the evidence snippet
                orig = next(u[2] for u in units if u[2].lower() == sh_low)
                best = (st, j, k_, orig)
        if best is None:
            for w in weak_idents(s["text"], df):
                wl = " " + w.lower() + " "
                for j in range(i + 1, n):
                    if segs[j]["kind"] in USAGE_KINDS and \
                            wl in (" " + norm_lower[j] + " "):
                        best = ("weak", j, "rare_ident", w)
                        break
                if best:
                    break
        if best is None:
            continue
        strength, j, unit_kind, needle = best
        a_seg, a_uid = anchor_for(j)
        if a_seg is None or a_seg == i:
            skips["anchor_is_candidate"] += 1
            continue
        label = U if strength == "weak" else N
        note = (
            f"distinctive phrase from this user turn is echoed by segment "
            f"{j} ({segs[j]['kind']}); the constraint was provably carried "
            f"into a later request/action"
            if label == N else
            f"only a rare identifier ({needle}) from this user turn recurs "
            f"in segment {j}; weak link — marked uncertain")
        user_cands.append({
            "seg": i, "usage_seg": j, "anchor_seg": a_seg,
            "anchor_uidx": a_uid, "cut": j + 1,
            "strategy": "user_cited_constraint", "label": label,
            "strength": strength,
            "note": note,
            "evidence": {"link": "phrase_echo" if unit_kind != "rare_ident"
                         else "rare_ident_echo",
                         "unit_kind": unit_kind,
                         "matched": needle[:160],
                         "usage_seg": j,
                         "usage_pointer": f"/messages/{j}/content",
                         "usage_kind": segs[j]["kind"],
                         "link_gap_segments": j - i},
        })
    user_cands.sort(key=lambda c: (0 if c["label"] == N else 1, c["seg"]))
    props.extend(user_cands[:MAX_USER_PER_TRANSCRIPT])

    # --- merge: one row per segment, strategy priority, per-transcript cap -----
    prio = {"file_read_edit": 0, "command_output_echo": 1,
            "user_cited_constraint": 2}
    props.sort(key=lambda p: (prio[p["strategy"]],
                              0 if p["label"] == N else 1, p["seg"]))
    seen_seg, out, n_uncertain = set(), [], 0
    for p in props:
        if p["seg"] in seen_seg:
            skips["dup_segment"] += 1
            continue
        if p["label"] == U:
            if n_uncertain >= MAX_UNCERTAIN_PER_TRANSCRIPT:
                skips["uncertain_cap"] += 1
                continue
            n_uncertain += 1
        if len(out) >= MAX_PER_TRANSCRIPT:
            skips["per_transcript_cap"] += 1
            break
        seen_seg.add(p["seg"])
        out.append(p)
    return out


def emit(item, prop, instructions, holdout_pairs, holdout_ptrs,
         eval_transcripts, f2_map, used_ids, skips):
    th, t, src = item["transcript_hash"], item["transcript"], item["source"]
    segs = t.segments
    idx, a_seg, a_uid = prop["seg"], prop["anchor_seg"], prop["anchor_uidx"]
    usage = prop["usage_seg"]
    if idx == a_seg or idx >= prop["cut"] or usage >= prop["cut"]:
        skips["candidate_out_of_scope"] += 1
        return None
    seg = segs[idx]
    want_kind = ("user_turn" if prop["strategy"] == "user_cited_constraint"
                 else "tool_result")
    if seg["kind"] != want_kind or seg["role"] in ("control", "system"):
        skips["bad_kind"] += 1
        return None
    ptr = f"/messages/{idx}/content"
    anchor_ptr = f"/messages/{a_seg}/content"
    usage_ptr = f"/messages/{usage}/content"
    check = {
        "transcript_in_eval60": th in eval_transcripts,
        "candidate_pointer_in_holdout": ptr in holdout_ptrs.get(th, set()),
        "pair_in_holdout": (th, ptr, anchor_ptr) in holdout_pairs,
    }
    if check["pair_in_holdout"]:
        skips["holdout_pair_collision"] += 1
        return None
    f2_label = f2_map.get((th, ptr))
    if f2_label == Y:
        skips["f2_label_conflict"] += 1
        return None
    prefix = segs[:prop["cut"]]
    user_list = t.users[:a_uid + 1]
    conv, windowed = f2.windowed_conversation(prefix, idx, a_seg, user_list)
    state = {"conversation": conv, "candidate_pointer": ptr,
             "user_messages_in_order": user_list}
    state_str = builder.serialized(state)
    if len(state_str.encode("utf-8")) > STATE_CHAR_HARD:
        skips["exceeds_budget"] += 1
        return None
    rid = f"v5f4:{th[:12]}:seg{idx:04d}"
    if rid in used_ids:
        rid = f"{rid}-u{a_uid}"
    if rid in used_ids:
        rid = f"{rid}-{prop['strategy'][:4]}"
    used_ids.add(rid)
    evidence = dict(prop["evidence"])
    evidence.setdefault("usage_pointer", usage_ptr)
    evidence.setdefault("usage_seg", usage)
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
        "usage_seg": usage,
        "usage_pointer": usage_ptr,
        "evidence_strength": prop["strength"],
        "evidence_basis": evidence,
        "windowed": windowed,
        "state_chars": len(state_str.encode("utf-8")),
        "proposed_label": prop["label"],
        "proposal_note": prop["note"],
        "holdout_collision_check": check,
        "f2_same_pointer": f2_label,
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
        "evidence_basis": evidence,
        "meta": meta,
    }


def load_f2_rows():
    """(transcript_hash, candidate_pointer) -> proposed_label for F2's mined
    rows; 'yes' collisions are skipped as direct label conflicts."""
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


def main():
    rng = random.Random(SEED)
    instructions = builder.load_verbatim_instructions(ROOT / "data" / "valen_nano_v3")
    holdout_pairs, holdout_ptrs, eval_transcripts = f2.load_holdout()
    f2_map = load_f2_rows()
    usable, drops, scanned = f2.scan_transcripts()
    eval_usable = sum(1 for u in usable
                      if u["transcript_hash"] in eval_transcripts)

    records, report_rows, skips = [], [], Counter()
    used_ids = set()
    for item in usable:
        props = mine_transcript(item, skips)
        for prop in props:
            rec = emit(item, prop, instructions, holdout_pairs, holdout_ptrs,
                       eval_transcripts, f2_map, used_ids, skips)
            if rec is None:
                continue
            records.append(rec)
            report_rows.append({
                "record_id": rec["record_id"],
                "transcript": item["transcript_hash"][:12],
                "source": item["source"],
                "strategy": prop["strategy"],
                "seg": prop["seg"],
                "usage_seg": prop["usage_seg"],
                "anchor_seg": prop["anchor_seg"],
                "kind": rec["meta"]["candidate_kind"],
                "strength": prop["strength"],
                "proposed_label": prop["label"],
                "note": prop["note"],
                "evidence_basis": rec["evidence_basis"],
                "candidate_snippet": f2.norm(
                    item["transcript"].segments[prop["seg"]]["text"])[:140],
                "anchor_snippet": f2.norm(
                    item["transcript"].users[prop["anchor_uidx"]])[:140],
            })

    records.sort(key=lambda r: (r["meta"]["transcript_hash"], r["record_id"]))
    label_counts = Counter(r["proposed_label"] for r in records)
    strat_counts = Counter(r["strategy"] for r in records)
    strat_label = Counter((r["strategy"], r["proposed_label"]) for r in records)
    strength_counts = Counter(r["meta"]["evidence_strength"] for r in records)
    strat_strength = Counter((r["strategy"], r["meta"]["evidence_strength"])
                             for r in records)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSONL, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(builder.serialized(r) + "\n")

    # deterministic seeded sample of up to 15 proposals for spot review
    sample_idx = sorted(rng.sample(range(len(report_rows)),
                                   min(15, len(report_rows))))
    top15 = [report_rows[i] for i in sample_idx]

    report = {
        "schema_version": "nanojev-v5-f4-outcome-mined-v1",
        "builder": "scripts/mine_v5_outcome_v1.py",
        "seed": SEED,
        "created_from": ("all quality-passing transcripts under "
                         "~/.claude/projects + ~/.codex/sessions "
                         "(same roots/thresholds as "
                         "build_real_context_eval_v1.py, no admission quota; "
                         "sha256-matched like mine_drop_supp_v1.py via the "
                         "F2 scan)"),
        "note": ("PROPOSALS ONLY: targets stay null; proposed_label + "
                 "evidence_basis are deterministic-linker outputs pending "
                 "the owner/agent adjudication spot pass (F4 spec: sampled "
                 "rows >=10% still go through adjudication to estimate "
                 "linker precision). Labels follow "
                 "docs/REAL_CONTEXT_LABELING_GUIDE_V1.md asymmetric "
                 "semantics (yes == certainly irrelevant == drop; this "
                 "family emits keep-side 'no' plus a weak-link 'uncertain' "
                 "band). No model/service calls, no network."),
        "mining_strategies": {
            "file_read_edit": ("tool_result holding content of path P "
                               "(paired read tool_call or literal path in "
                               "result text) precedes a same-transcript "
                               "Edit/Write/apply_patch on P -> keep; "
                               "dir-listing -> child-edit is the weak "
                               "uncertain form"),
            "command_output_echo": ("tool_result output unit (URL / "
                                    "digit-bearing hex hash / distinctive "
                                    ">=24-char line) quoted verbatim in a "
                                    "later assistant/user/tool_use segment "
                                    "-> keep; rare-ident-only echo -> "
                                    "uncertain"),
            "user_cited_constraint": ("non-final user turn whose dedup'd "
                                      ">=6-word shingle or >=6-char CJK "
                                      "run is echoed by a later assistant "
                                      "action/user request -> keep; "
                                      "rare-ident-only echo -> uncertain"),
        },
        "link_semantics": {
            "cut": ("state = transcript prefix ending immediately after "
                    "the proving usage segment (usage_seg+1); the evidence "
                    "is in-window unless elided by the char budget"),
            "anchor": ("last user turn at/before the usage segment; "
                       "user_messages_in_order truncated there"),
            "label_rule": ("strong/medium evidence -> 'no' (keep); "
                           "weak-only link -> 'uncertain'"),
        },
        "holdout": {
            "frozen_pairs_loaded": len(holdout_pairs),
            "files": [str(p.relative_to(ROOT)) for p in HOLDOUT_FILES
                      if p.is_file()],
            "eval_transcripts": len(eval_transcripts),
            "eval_transcripts_mined": eval_usable,
            "rule": ("a mined record is skipped iff its "
                     "(transcript_hash, candidate_pointer, anchor_pointer) "
                     "triple matches a frozen pair; sharing a "
                     "transcript_hash or a candidate_pointer alone is "
                     "allowed and recorded in meta.holdout_collision_check"),
            "f2_conflict_rule": ("rows whose (transcript, candidate_pointer) "
                                 "already carries an F2 'yes' proposal are "
                                 "skipped (f2_label_conflict); other F2 "
                                 "overlaps keep meta.f2_same_pointer"),
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
            "strength_distribution": dict(strength_counts),
            "by_strategy_strength": {f"{k[0]}:{k[1]}": v
                                     for k, v in
                                     sorted(strat_strength.items())},
            "by_transcript": dict(Counter(r["meta"]["transcript_hash"][:12]
                                          for r in records)),
            "by_kind": dict(Counter(r["meta"]["candidate_kind"]
                                    for r in records)),
            "by_source": dict(Counter(r["meta"]["source_root"]
                                      for r in records)),
            "windowed": sum(1 for r in records if r["meta"]["windowed"]),
            "from_eval60_transcripts": sum(
                1 for r in records
                if r["meta"]["holdout_collision_check"]
                ["transcript_in_eval60"]),
            "skips": dict(sorted(skips.items())),
        },
        "spot_check_sample_15": top15,
        "rows": report_rows,
        "caveats": [
            ("pool size: %d transcripts pass the builder's minimum quality "
             "thresholds (min_segments>=12, >=1 user msg, byte cap, "
             "credential scan) — the ~290 'unadmitted' figure in the design "
             "doc conflated files scanned (351) with usable transcripts; "
             "local yield is bounded by that pool" % len(usable)),
            ("only spec source (b) is covered: public resolved-task "
             "trajectories (SWE-Gym/SWE-chat) need downloads/license "
             "screening and were out of scope (no network)"),
            ("tool_use->tool_result pairing is FIFO heuristic: rejected/"
             "resultless calls can misalign ops; the path-in-result-text "
             "link covers those cases content-side"),
            ("codex read/edit detection parses shell command strings "
             "(cat/sed/apply_patch) — approximate; claude Read/Edit "
             "tool_use dict-inputs are exact"),
            ("verbatim echo is whitespace-normalized case-sensitive; "
             "paraphrased downstream use is not detected (linker recall, "
             "not precision, is the cost)"),
            ("'uncertain' rows are the deliberately weak band (rare-ident / "
             "dir-listing links); they are excluded from scored targets per "
             "spec §6 like all uncertain proposals"),
        ],
    }
    with open(OUT_REPORT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"scanned {scanned} files -> {len(usable)} usable transcripts "
          f"({eval_usable} of them eval-admitted)")
    print(f"wrote {len(records)} records -> {OUT_JSONL}")
    print("labels:", dict(label_counts), "| strategies:", dict(strat_counts),
          "| strength:", dict(strength_counts))
    print("skips:", dict(sorted(skips.items())),
          "| drops:", dict(sorted(drops.items())))
    print(f"report -> {OUT_REPORT}")


if __name__ == "__main__":
    sys.exit(main())
