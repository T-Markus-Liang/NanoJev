#!/usr/bin/env python3
"""build_v9_injection_v1.py — Path B of docs/DROP_LABEL_DILEMMA_V1.md:
mechanically-correct SUBTLE drop labels via supersession injection.

v8 foreign-injection taught "obviously alien content = drop" and did not
transfer. v9 instead keeps the candidate verbatim and appends a realistic
supersession segment AFTER the candidate (and before the state's end) that
explicitly replaces/corrects the candidate's core content (path, value, or
quoted passage). The candidate now refers to obsolete information, so
label drop (true=1) is mechanically justified — while the surface pattern
(same domain, plausible content) stays subtle.

For each mined KEEP record (proposed_label=no) we emit a contrastive pair:
  keep arm : original record, targets keep        (meta.v9_arm="keep")
  drop arm : same state + 1 injected supersession segment,
             targets drop                         (meta.v9_arm="supersession_drop")
Both arms share meta.pair_id="v9pair:<record_id>"; the drop arm also carries
meta.pair_with and meta.v9_supersede details.

Insertion realism constraints:
- the new segment uses a "/messages/N/content" pointer with N taken from a
  numeric gap between neighbouring message pointers (windowed transcripts
  already skip numbers), so pointers stay unique and monotonically ordered;
- the segment sits strictly after the candidate and strictly before the
  conversation's last segment;
- we prefer positions at or before the last user segment so the injected
  turn lands "before user_messages_in_order ends"; a user-role supersession
  is only emitted in that region and its text is inserted into
  user_messages_in_order at the matching ordinal position;
- injected segments are assistant/tool/user only; the candidate_pointer and
  the candidate segment content are left untouched.

Records with no legal gap-bearing position after the candidate are skipped.

Usage: python3 scripts/build_v9_injection_v1.py \
           --out data/v9_injection/train.jsonl \
           --report data/v9_injection/report.json
"""
import argparse, json, os, random, re
from collections import Counter

MSG_PTR = re.compile(r"^/messages/(\d+)/")
CJK = re.compile(r"[一-鿿]")
# absolute-ish path or dotted relative path with an extension
PATH_RE = re.compile(
    r"(?:[~.]?/[\w.\-~]+)+/[\w.\-~]+\.\w{1,10}"          # /a/b/file.ext
    r"|(?:/[\w.\-~]+){2,}"                              # /a/b/c (no ext)
    r"|\b[\w.\-]+(?:/[\w.\-]+)+\.\w{1,10}\b"            # a/b/file.ext
)
NUM_RES = [
    ("pct_float", re.compile(r"\b\d+\.\d+%")),
    ("version",   re.compile(r"\bv\d+(?:\.\d+)*\b")),
    ("decimal",   re.compile(r"\b\d+\.\d+\b")),
    ("pct_int",   re.compile(r"\b\d{1,3}%")),
    ("bigint",    re.compile(r"\b\d{3,}\b")),
    ("int",       re.compile(r"\b\d+\b")),
]

KEEP_T = {"irrelevant": {"probabilities": {"true": 0.0, "false": 1.0}}}
DROP_T = {"irrelevant": {"probabilities": {"true": 1.0, "false": 0.0}}}


def parse_state(r):
    return json.loads(r["request"]["state"])


def seg_index(conv, ptr):
    for i, s in enumerate(conv):
        if s.get("pointer") == ptr:
            return i
    return None


def msg_num(seg):
    m = MSG_PTR.match(seg.get("pointer", ""))
    return int(m.group(1)) if m else None


def is_zh(text):
    return bool(CJK.search(text))


# ---------------- entity extraction ----------------

def find_path(text, rng):
    cands = [m.group(0) for m in PATH_RE.finditer(text)]
    cands = [c for c in cands if len(c) >= 6 and "…" not in c]
    if not cands:
        return None
    with_ext = [c for c in cands if re.search(r"\.[A-Za-z0-9]{1,10}$", c)]
    pool = with_ext or cands
    return max(pool, key=len)


def new_path(old, rng):
    d, _, base = old.rpartition("/")
    stem, dot, ext = base.rpartition(".")
    if not dot:
        stem, ext = base, ""
    m = re.search(r"_v(\d+)$", stem)
    if m:
        nstem = stem[: m.start()] + "_v" + str(int(m.group(1)) + 1)
    elif re.search(r"v(\d+)$", stem):
        n = int(re.search(r"v(\d+)$", stem).group(1))
        nstem = re.sub(r"v\d+$", "v" + str(n + 1), stem)
    else:
        nstem = stem + rng.choice(["_v2", "_new", "_final", "_updated"])
    nbase = nstem + ("." + ext if ext else "")
    return (d + "/" + nbase) if d else nbase


def find_number(text):
    for kind, rx in NUM_RES:
        m = rx.search(text)
        if m:
            return kind, m.group(0)
    return None, None


def new_number(kind, old, rng):
    if kind == "pct_float":
        v = float(old[:-1])
        nv = max(0.1, min(99.9, v + rng.uniform(-15, 15)))
        s = f"{nv:.{len(old.split('.')[1]) - 1}f}%"
        return s if s != old else f"{min(99.9, v + 3):.2f}%"
    if kind == "version":
        n = int(re.match(r"v(\d+)", old).group(1))
        return "v" + str(n + 1) + old[1 + len(str(n)):]
    if kind == "decimal":
        dec = len(old.split(".")[1])
        v = float(old)
        nv = v + rng.choice([-1, 1]) * rng.uniform(0.5, 5.0)
        s = f"{nv:.{dec}f}"
        return s if s != old else f"{v + 1:.{dec}f}"
    if kind == "pct_int":
        v = int(old[:-1])
        nv = max(1, min(99, v + rng.randint(-20, 20)))
        return str(nv if nv != v else v + 3) + "%"
    v = int(old)
    return str(v + rng.randint(max(1, v // 10), max(2, v // 3)) + 1)


def snippet(text):
    for line in text.splitlines():
        line = line.strip().strip("|#>*` ")
        if len(line) >= 8:
            return line[:60]
    return text.strip()[:60]


# ---------------- supersession segment text ----------------

def corr_text(entity, old, new, zh, rng):
    if entity == "path":
        zh_t = [
            f"更正：`{old}` 已被取代——实际改动落在 `{new}`（刚写入完成）。"
            f"前文对 `{old}` 的引用已过时，以 `{new}` 为准。",
            f"更新：`{old}` 不再适用，实际文件为 `{new}`；"
            f"请忽略前面关于 `{old}` 的内容。",
        ]
        en_t = [
            f"Correction: `{old}` has been superseded — the actual change landed "
            f"in `{new}` (just written). Earlier references to `{old}` are stale; "
            f"use `{new}`.",
            f"Update: `{old}` is no longer current; the live file is `{new}`. "
            f"Disregard the earlier `{old}` reference.",
        ]
    elif entity == "value":
        zh_t = [
            f"更正：前面给出的 `{old}` 已作废，实际值为 `{new}`（以最新结果为准）。",
            f"更新：上文所述数值 `{old}` 已被修正为 `{new}`，旧值不再适用。",
        ]
        en_t = [
            f"Correction: the earlier `{old}` figure is void — the actual value "
            f"is `{new}` (latest result supersedes it).",
            f"Update: the previously stated `{old}` has been corrected to "
            f"`{new}`; the old value no longer applies.",
        ]
    else:  # snippet
        zh_t = [
            f"更新：前文中「{old}」一段所述内容已被取代，不再适用，"
            f"请忽略该段并以最新说明为准。",
        ]
        en_t = [
            f"Update: the earlier passage \"{old}\" has been superseded and no "
            f"longer applies — disregard it.",
        ]
    return rng.choice(zh_t if zh else en_t)


def corr_user(entity, old, new, zh, rng):
    if zh:
        t = [
            f"更正一下：{old} 已经过时了，实际是 {new}，忽略之前那段。",
            f"等一下，前面说的 {old} 不对——以 {new} 为准，旧的作废。",
        ]
    else:
        t = [
            f"Correction — the earlier {old} is outdated; it's actually {new}. "
            f"Ignore the previous note.",
            f"Wait, {old} above is stale — the correct one is {new}; treat the "
            f"old value as superseded.",
        ]
    return rng.choice(t)


def corr_tool_result(entity, old, new, rng):
    if entity == "path":
        body = f"{new}\n# {old} superseded by {new}; earlier content under {old} is obsolete"
    else:
        body = f"updated_value={new}  # supersedes {old}; {old} obsolete"
    style = rng.random()
    if style < 0.5:
        tok = len(body) // 4 + 1
        return (f"Chunk ID: {rng.randbytes(3).hex()}\nWall time: 0.0000 seconds\n"
                f"Process exited with code 0\nOriginal token count: {tok}\n"
                f"Output:\n{body}")
    return json.dumps(
        [{"type": "input_text",
          "text": f"Script completed\nWall time 0.{rng.randint(1, 9)} seconds\nOutput:\n"},
         {"type": "input_text", "text": body}],
        ensure_ascii=False)


def corr_tool_use_path(old, new, rng):
    cmd = rng.choice([
        f"sed -n '1,60p' '{new}'  # {old} superseded -> {new}",
        f"git diff -- '{new}'; # NOTE: {old} replaced by {new}",
        f"cat '{new}'  # new canonical file, {old} obsolete",
    ])
    inner = {"name": "exec",
             "input": f"text(await tools.exec_command({{cmd:{json.dumps(cmd)}}}));\n"}
    return json.dumps({"tool_use": inner}, ensure_ascii=False)


def build_supersession(cand_content, rng):
    """Return (entity_type, old, new). None when nothing extractable."""
    p = find_path(cand_content, rng)
    if p:
        return "path", p, new_path(p, rng)
    kind, num = find_number(cand_content)
    if num:
        return "value", num, new_number(kind, num, rng)
    sn = snippet(cand_content)
    if sn:
        return "snippet", sn, None
    return None


# ---------------- insertion machinery ----------------

def valid_positions(conv, cand_idx):
    """Indices i in (cand_idx, len) where inserting at i keeps the segment
    before the conversation's last element and a /messages/N pointer gap
    exists between neighbours."""
    nums = [msg_num(s) for s in conv]
    out = []
    for i in range(cand_idx + 1, len(conv)):
        prev = next((n for n in reversed(nums[:i]) if n is not None), None)
        nxt = next((n for n in nums[i:] if n is not None), None)
        if prev is not None and nxt is not None and nxt - prev >= 2:
            out.append((i, prev, nxt))
    return out


def make_pair(r, rng):
    """Return (keep_record, drop_record, diff_info) or (None, None, reason)."""
    st = parse_state(r)
    conv = st.get("conversation", [])
    cptr = st.get("candidate_pointer")
    ci = seg_index(conv, cptr)
    if ci is None or not conv[ci].get("content"):
        return None, None, "no_candidate"
    cand_seg = conv[ci]
    zh = is_zh(cand_seg["content"])

    ent = build_supersession(cand_seg["content"], rng)
    if ent is None:
        return None, None, "no_entity"
    etype, old, new = ent

    spots = valid_positions(conv, ci)
    if not spots:
        return None, None, "no_position"
    last_user = max((i for i, s in enumerate(conv)
                     if s.get("role") == "user"), default=-1)
    pre_user = [s for s in spots if s[0] <= last_user]

    # choose role/template
    role = "assistant"
    content = None
    want_user = rng.random() < 0.15
    if want_user and pre_user:
        role = "user"
        content = corr_user(etype, old, new if new else "the updated version", zh, rng)
    else:
        roll = rng.random()
        if etype == "path" and roll < 0.20:
            role = "assistant"; content = corr_tool_use_path(old, new, rng)
        elif etype in ("path", "value") and roll < 0.40:
            role = "tool"; content = corr_tool_result(etype, old, new, rng)
        else:
            role = "assistant"; content = corr_text(etype, old, new, zh, rng)

    if role == "user":
        i, prev, nxt = rng.choice(pre_user)
    else:
        i, prev, nxt = rng.choice(pre_user or spots)
    new_n = rng.randint(prev + 1, nxt - 1)
    inj = {"pointer": f"/messages/{new_n}/content", "role": role,
           "content": content}

    # ---- keep arm ----
    rk = dict(r); rk["meta"] = dict(r["meta"])
    rk["targets"] = KEEP_T
    rk["meta"]["v9_arm"] = "keep"
    rk["meta"]["pair_id"] = f"v9pair:{r['meta'].get('record_id')}"

    # ---- drop arm ----
    st2 = json.loads(json.dumps(st))          # deep copy
    conv2 = st2["conversation"]
    conv2.insert(i, dict(inj))
    if role == "user":
        # user_messages_in_order also contains user messages elided from the
        # window, so the ordinal position is NOT "count of visible user segs".
        # The injected correction sits right before the next visible user
        # message → insert immediately before that message's uio entry.
        uio = st2.setdefault("user_messages_in_order", [])
        next_user = next(s["content"] for s in conv[i:]
                         if s.get("role") == "user")
        prev_users = [s["content"] for s in conv[:i]
                      if s.get("role") == "user"]
        lo = (max(j for j, t in enumerate(uio) if t == prev_users[-1]) + 1
              if prev_users and prev_users[-1] in uio else 0)
        j = next((k for k in range(lo, len(uio)) if uio[k] == next_user),
                 len(uio))
        uio.insert(j, content)
    rd = dict(r); rd["meta"] = dict(r["meta"])
    rd["request"] = dict(r["request"])
    rd["request"]["state"] = json.dumps(st2, ensure_ascii=False)
    rd["targets"] = DROP_T
    rd["meta"]["v9_arm"] = "supersession_drop"
    rd["meta"]["pair_id"] = rk["meta"]["pair_id"]
    rd["meta"]["pair_with"] = r["meta"].get("record_id")
    rd["meta"]["v9_supersede"] = {
        "entity_type": etype, "old": old, "new": new,
        "insert_index": i, "pointer": inj["pointer"], "role": role,
    }

    # ---- validation ----
    validate(st, st2, inj["pointer"], cptr, ci, i, role, content)

    diff = {
        "pair_id": rk["meta"]["pair_id"],
        "record_id": r["meta"].get("record_id"),
        "entity_type": etype, "old": old, "new": new,
        "insert_index": i, "injected_pointer": inj["pointer"],
        "neighbour_pointers": [conv[i - 1].get("pointer"),
                               conv[i].get("pointer")],
        "injected_segment": inj,
    }
    return rk, rd, diff


def validate(st_keep, st_drop, inj_ptr, cand_ptr, cand_idx, ins_idx, role,
             content):
    dk = {k: v for k, v in st_keep.items() if k != "conversation"}
    dd = {k: v for k, v in st_drop.items() if k != "conversation"}
    if role != "user":
        assert dk == dd, "non-conversation state changed"
    else:
        uk, ud = dk.pop("user_messages_in_order", []), \
                 dd.pop("user_messages_in_order", [])
        assert dk == dd, "non-conversation state changed"
        assert content in ud and len(ud) == len(uk) + 1, "uio not updated"
        # injected user message must sit before uio's end, and every visible
        # user message must still appear in-order inside uio
        assert ud.index(content) < len(ud) - 1
        pos = -1
        for t in [s["content"] for s in st_keep["conversation"]
                  if s.get("role") == "user"]:
            pos = next((k for k in range(pos + 1, len(ud)) if ud[k] == t),
                       None)
            assert pos is not None, "uio lost a visible user message"
    ck, cd = st_keep["conversation"], st_drop["conversation"]
    assert len(cd) == len(ck) + 1, "expected exactly one appended segment"
    assert cd[ins_idx]["pointer"] == inj_ptr
    rest = cd[:ins_idx] + cd[ins_idx + 1:]
    assert rest == ck, "state differs beyond the appended segment"
    assert ins_idx > cand_idx, "supersession not after candidate"
    assert ins_idx < len(cd) - 1, "supersession not before state end"
    assert st_keep["candidate_pointer"] == st_drop["candidate_pointer"] == cand_ptr
    assert ck[cand_idx]["content"] == cd[seg_index(cd, cand_ptr)]["content"], \
        "candidate content changed"
    ptrs = [s["pointer"] for s in cd]
    assert len(set(ptrs)) == len(ptrs), "duplicate pointer introduced"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates",
                    default="data/v5_mining/f2_hard_negative_candidates.jsonl")
    ap.add_argument("--out", default="data/v9_injection/train.jsonl")
    ap.add_argument("--report", default="data/v9_injection/report.json")
    ap.add_argument("--seed", type=int, default=20261007)
    ap.add_argument("--max-pairs", type=int, default=900)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    keeps = []
    for l in open(args.candidates):
        r = json.loads(l)
        if r["meta"].get("proposed_label") == "no":
            keeps.append(r)
    rng.shuffle(keeps)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    skips = Counter(); etypes = Counter(); roles = Counter()
    diffs = []
    n_pairs = n_rows = 0
    with open(args.out, "w") as f:
        for r in keeps:
            if n_pairs >= args.max_pairs:
                break
            rk, rd, info = make_pair(r, rng)
            if rk is None:
                skips[info] += 1
                continue
            f.write(json.dumps(rk, ensure_ascii=False) + "\n")
            f.write(json.dumps(rd, ensure_ascii=False) + "\n")
            n_pairs += 1; n_rows += 2
            etypes[info["entity_type"]] += 1
            roles[info["injected_segment"]["role"]] += 1
            diffs.append(info)

    report = {
        "script": "build_v9_injection_v1.py",
        "seed": args.seed,
        "source": args.candidates,
        "keeps_considered": len(keeps),
        "pairs_emitted": n_pairs,
        "rows_emitted": n_rows,
        "skipped": dict(skips),
        "entity_types": dict(etypes),
        "injected_roles": dict(roles),
        "sample_diffs": diffs[:2],
    }
    with open(args.report, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"v9 supersession pairs: {n_pairs} (rows {n_rows}) -> {args.out}")
    print(f"skipped: {dict(skips)}  entity: {dict(etypes)}  roles: {dict(roles)}")


if __name__ == "__main__":
    main()
