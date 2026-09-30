#!/usr/bin/env python3
"""V5 family F3 — boilerplate_prior_breaker corpus builder
(data/v5_corpus/f3_boilerplate.jsonl, ~1500 LABELED records, 1:1 keep:drop).

Implements docs/V5_DATA_DESIGN_V1.md §4 (family F3) targeting the kev
false-positive pattern (docs/KEV_FP_PATTERN_V1.md §clusters A-G): the scorer
learned "generic harness envelope -> irrelevant" instead of judging
payload-vs-anchor. Fix = contrastive pairs where an identical envelope shape
appears once with task evidence (keep) and once with task-foreign noise
(drop), plus anchor-flip pairs where the SAME payload is keep under a
relevant anchor and drop under a foreign one (kills payload->label
shortcuts).

Both harness dialects are mandatory (kev cluster A/C/D evidence):
  - codex exec envelopes: "Chunk ID: <hex> / Wall time / Process exited with
    code 0 / Original token count / Output:" heads, "Script completed"
    variants, serialized input_text-list exec outputs, empty polls
    ("(Bash completed with no output)", {"output":""});
  - claude control/search shells: "Web search results for query:" + REMINDER
    tails, domain-verify blocks, tool-use-rejected records,
    "[Request interrupted by user ...]";
  - shared shapes: line-numbered file dumps, URL-probe lists, git-status
    lists;
  - low-content conversational forms (kev clusters E/F/G): micro user
    openers/directives, assistant scope-ack micro turns, completion reports.

Envelope skeletons + foreign-payload bodies are harvested from the vetted
v5 mining pool (data/v5_mining/f2_hard_negative_candidates.jsonl — train
pool only; eval-v1/reserved transcripts never enter it); doc-canonical
forms fill thin slots. Payloads are synthesized per task domain (>=5
domains) or re-contextualized harvested bodies.

Pair kinds (meta.pair_id shared, opposite labels):
  - envelope_payload_flip: one conversation holds two same-envelope
    segments — one payload is the anchor's evidence (keep), one is foreign
    noise (drop); records differ only in candidate_pointer;
  - anchor_flip: identical envelope+payload candidate; keep-anchor record
    and foreign-anchor record live in DIFFERENT groups (doc §4).

Record schema is byte-compatible with data/valen_nano_v4/eval.jsonl.
Self-check enforces schema, pointer resolution, component-level credential/
placeholder scan, pair integrity, and the v4 label-consistency contract.
Deterministic seed 20261201; no network, no model calls.

Usage: python3 scripts/build_v5_boilerplate_breaker_v1.py
Output: data/v5_corpus/f3_boilerplate.jsonl
        data/v5_corpus/build_report.json  (merged per-family manifest)
"""

import argparse
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
spec2 = importlib.util.spec_from_file_location(
    "mine_v5_hard_negatives_v1",
    ROOT / "scripts" / "mine_v5_hard_negatives_v1.py")
miner = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(miner)
spec3 = importlib.util.spec_from_file_location(
    "build_v5_antishortcut_v1",
    ROOT / "scripts" / "build_v5_antishortcut_v1.py")
f1 = importlib.util.module_from_spec(spec3)
spec3.loader.exec_module(f1)

OUT_DIR = ROOT / "data" / "v5_corpus"
OUT_JSONL = OUT_DIR / "f3_boilerplate.jsonl"
REPORT = OUT_DIR / "build_report.json"
HARVEST_SOURCE = f1.HARVEST_SOURCE

SEED = f1.SEED                  # 20261201
GROUP_LINEAGE = f1.GROUP_LINEAGE
FAMILY = "f3_boilerplate"
SCHEMA = f1.SCHEMA
SYSTEM_TEXT = f1.SYSTEM_TEXT
INSTRUCTIONS = None
MAX_CANDIDATE_BYTES = f1.MAX_CANDIDATE_BYTES
SLOT_RE = f1.SLOT_RE
WINDOWED_FRACTION = 0.30

# pair counts per envelope sub-family -> ~750 pairs => ~1500 records
PLAN = {
    "chunk_exec": 130,
    "script_completed": 80,
    "input_text_wrap": 30,
    "file_dump": 90,
    "url_probe": 55,
    "git_status": 45,
    "web_search_shell": 80,
    "domain_verify": 40,
    "tool_rejected": 40,
    "request_interrupted": 30,
    "empty_poll": 40,
    "user_micro": 40,
    "assistant_micro": 30,
    "completion_report": 20,
}
ANCHOR_FLIP_SHARE = 0.45       # share of pairs emitted cross-group

# ---------------------------------------------------------------------------
# harvest: envelope skeletons + payload bodies + micro turns
# ---------------------------------------------------------------------------

CHUNK_HEAD_RE = re.compile(
    r"^Chunk ID: [0-9a-f]+\nWall time: [0-9.]+ seconds\n"
    r"Process exited with code \d+\nOriginal token count: \d+\nOutput:\n",
    re.MULTILINE)
SCRIPT_HEAD_RE = re.compile(
    r"^Script completed[^\n]*\n(?:Wall time [0-9.]+ seconds\n)?Output:\n?")
REMINDER_RE = re.compile(r"REMINDER:[^\n]*(?:\n[^\n]*){0,3}")
WEBSEARCH_HEAD_RE = re.compile(r'^Web search results for query: "[^"]*"')
REJECTED_RE = re.compile(r"doesn't want to proceed with this tool use", re.I)
INTERRUPTED_RE = re.compile(r"\[Request interrupted")
VERIFY_RE = re.compile(r"Unable to verify (?:if|the) domain", re.I)
FILEDUMP_RE = re.compile(r"(?:^|\n)\s*\d+\t")
GITSTATUS_RE = re.compile(
    r"^On branch |^Changes not staged|^(?: M|\?\?|A ) ", re.M)
URLPROBE_RE = re.compile(r"https?://\S+", re.M)
EMPTY_POLL_RE = re.compile(
    r"\(Bash completed with no output\)|\"output\"\s*:\s*\"\"")

HARVEST_CAPS = {"chunk_head": 60, "script_head": 60, "websearch_tail": 60,
                "rejected": 10, "interrupted": 10, "domain_verify": 30,
                "payload_body": 260, "user_micro": 120,
                "assistant_micro": 120, "report": 80}


def harvest_pool(path=HARVEST_SOURCE):
    """Return (inv, refmap, stats). inv slots:
      chunk_head/script_head: real exec-envelope prefixes (payload excluded);
      websearch_tail: REMINDER tail paragraphs;
      rejected/interrupted/domain_verify: real control-surface strings;
      payload_body: real tool-result bodies (drop-side foreign payloads);
      user_micro/assistant_micro/report: low-content conversational forms.
    """
    inv = {k: [] for k in HARVEST_CAPS}
    refmap = {k: {} for k in HARVEST_CAPS}
    seen = set()
    stats = Counter()
    if not Path(path).is_file():
        return inv, refmap, {"source_missing": 1}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        rid = rec.get("meta", {}).get("record_id", "?")
        try:
            state = json.loads(rec["request"]["state"])
        except (ValueError, KeyError, TypeError):
            continue
        for segm in state.get("conversation", []):
            text = segm.get("content", "")
            role = segm.get("role", "")
            if not text.strip():
                continue

            def take(slot, t):
                key = builder.normalize_text(t)[:400]
                if key in seen or len(inv[slot]) >= HARVEST_CAPS[slot]:
                    return
                if builder.credential_scan(t) or SLOT_RE.search(t):
                    stats["harvest_skip"] += 1
                    return
                seen.add(key)
                inv[slot].append(t)
                refmap[slot][t] = rid
                stats[f"harvest_{slot}"] += 1

            # control-surface forms can appear as tool results OR user-role
            # blocks depending on the harness — scan both
            if REJECTED_RE.search(text):
                take("rejected", text[:600])
            if INTERRUPTED_RE.search(text):
                take("interrupted", text[:300])
            if VERIFY_RE.search(text):
                take("domain_verify", text[:500])
            if REMINDER_RE.search(text):
                take("websearch_tail",
                     "\n".join(REMINDER_RE.findall(text))[:400])
            if role == "tool":
                m = CHUNK_HEAD_RE.match(text)
                if m:
                    take("chunk_head", text[:m.end()])
                else:
                    m2 = SCRIPT_HEAD_RE.match(text)
                    if m2:
                        take("script_head", text[:m2.end()])
                # payload bodies: strip known envelope heads, keep the body
                body = text
                for pat in (CHUNK_HEAD_RE, SCRIPT_HEAD_RE):
                    mm = pat.match(body)
                    if mm:
                        body = body[mm.end():]
                        break
                body = body.strip()
                if 60 <= len(body) <= 900 and not EMPTY_POLL_RE.search(body):
                    take("payload_body", body[:700])
            elif role == "user" and len(text) <= 80:
                take("user_micro", text)
            elif role == "assistant":
                if len(text) <= 120 and miner.ack_like(text):
                    take("assistant_micro", text)
                elif 120 <= len(text) <= 700 and re.search(
                        r"完成|报告|verified|summary|deploy|修复|结论", text):
                    take("report", text)
    return inv, refmap, dict(stats)


# ---------------------------------------------------------------------------
# domain pools + envelope renderers
# ---------------------------------------------------------------------------

DOMAINS = f1.DOMAINS
DIRS = f1.DIRS
FILLER_ASSISTANT = f1.FILLER_ASSISTANT
fill = f1.fill

HOSTS = ["hf-mirror.com", "huggingface.co", "api.github.com",
         "packages.internal.dev", "mirrors.aliyun.com", "registry.npmjs.org"]
REPO_SLUGS = ["Qwen/Qwen3.5-0.8B", "unitreerobotics/g1_sdk",
              "org/firmware-utils", "team/deploy-manifests",
              "vendor/dexhand-driver", "lab/world-model-notes"]
CODE_LINES = [
    "def compute_quota(user):", "    limit = config.get('daily_cap', 500)",
    "    if user.tier == 'gold':", "        limit *= 2",
    "    return min(limit, MAX_QUOTA)", "import os, sys",
    "def main():", "    for path in pending:",
    "        process(path)", "class RewardModel:",
    "def forward(self, x):", "    return self.head(self.enc(x))",
    "def load_config(path):", "    return yaml.safe_load(open(path))",
    "conn = sqlite3.connect(DB)", "cur.execute(SELECT_SQL)",
]
ZH_NOTE_LINES = [
    "已确认配置生效", "校验通过", "无异常输出", "结果为空，符合预期",
]


def hex6(rng):
    return f"{rng.randrange(0x100000, 0xFFFFFF):x}"


def envelope_chunk(rng, payload, inv, refmap):
    head = rng.choice(inv["chunk_head"]) if inv["chunk_head"] else \
        ("Chunk ID: <HEX>\nWall time: <T> seconds\nProcess exited with "
         "code 0\nOriginal token count: <N>\nOutput:\n")
    head = fill(head, rng, HEX=hex6(rng), T=f"{rng.random():.4f}")
    return head + payload


def envelope_script(rng, payload, inv, refmap):
    head = rng.choice(inv["script_head"]) if inv["script_head"] else \
        "Script completed\nWall time <T> seconds\nOutput:\n"
    head = fill(head, rng, T=f"{rng.random():.4f}")
    return head + payload


def envelope_wrap(rng, payload, inv, refmap):
    return builder.serialized(
        [{"type": "input_text",
          "text": f"Script completed\nWall time {rng.random():.1f} seconds\n"
                  "Output:\n"},
         {"type": "input_text", "text": payload}])


def envelope_filedump(rng, payload_lines, inv, refmap):
    return "\n".join(f"{i + 1:6d}\t{ln}" for i, ln in enumerate(payload_lines))


def envelope_urlprobe(rng, rows, inv, refmap):
    return "\n".join(rows)


def envelope_gitstatus(rng, rows, inv, refmap):
    return "On branch main\nChanges not staged for commit:\n" + \
        "\n".join(rows)


def envelope_websearch(rng, query, body, inv, refmap):
    tail = rng.choice(inv["websearch_tail"]) if inv["websearch_tail"] else \
        ("REMINDER: You MUST include the sources above in your response to "
         "the user using markdown hyperlinks.")
    return (f'Web search results for query: "{query}"\n\n{body}\n\n\n{tail}')


def envelope_verify(rng, domain_name, inv, refmap):
    if inv["domain_verify"]:
        t = rng.choice(inv["domain_verify"])
        return t if "<D>" not in t else t.replace("<D>", domain_name)
    return (f"Unable to verify if domain {domain_name} is safe to fetch. "
            "This may be due to a corporate proxy or security policy. The "
            "user can add the domain to the allowlist to proceed.")


def envelope_rejected(rng, inv, refmap):
    return rng.choice(inv["rejected"]) if inv["rejected"] else (
        "The user doesn't want to proceed with this tool use. The tool use "
        "was rejected (eg. if it was a file edit, the new_string was NOT "
        "written to the file). STOP what you are doing and wait for the "
        "user to tell you how to proceed.")


def envelope_interrupted(rng, inv, refmap):
    return rng.choice(inv["interrupted"]) if inv["interrupted"] else \
        "[Request interrupted by user for tool use]"


def envelope_empty_poll(rng, inv, refmap):
    return rng.choice(["(Bash completed with no output)",
                       '{"output":""}',
                       "Script completed\nWall time 0.0 seconds\nOutput:\n"])


# ---------------------------------------------------------------------------
# payload pools: keep = task evidence for the anchor domain; drop = foreign
# ---------------------------------------------------------------------------

def keep_payload(rng, domain_key, kind):
    """A payload that IS the anchor task's evidence."""
    D = DOMAINS[domain_key]
    e = rng.choice(D["entities"])
    f = rng.choice(D["files"])
    if kind == "file_dump":
        lines = ([f"# {f}"] + rng.sample(CODE_LINES, 6)
                 + [f"# end of {f}"])
        return lines, (e, f)
    if kind == "url_probe":
        rows = [f"https://{rng.choice(HOSTS)}/{rng.choice(REPO_SLUGS)}"
                f" -> {rng.choice([200, 200, 301])} (len={rng.randrange(200, 9000)})"
                for _ in range(3)]
        return "\n".join(rows), (e, f)
    if kind == "git_status":
        rows = [f" M {f}", f" M {rng.choice(D['files'])}",
                f"?? {rng.choice(DIRS)}tmp_{rng.randrange(10, 99)}.log"]
        return "\n".join(rows), (e, f)
    if kind == "web_search_shell":
        items = [f"- {e} release notes ({rng.choice(HOSTS)})",
                 f"- {e} configuration reference",
                 f"- {e} troubleshooting thread"]
        return "\n".join(items), (e, f)
    if kind == "completion_report":
        return (f"完成报告：{e} 已按约束处理并通过验证。改动范围 {f}，"
                "校验输出为空（无异常）。操作说明见上。", ), (e, f)
    # exec-style payloads
    body = rng.choice([
        f"running checks for {e}\n{rng.randrange(4, 30)} passed, 0 failed",
        f"{e}: verification ok\ndiff check passed for {f}",
        f"{rng.choice(ZH_NOTE_LINES)}：{e}",
        f"{e}\n{e}-aux",
        f"total {rng.randrange(1, 9)}\n{f}",
    ])
    return body, (e, f)


def drop_payload(rng, domain_key, kind, inv):
    """A payload foreign to the anchor: prefer a harvested real body (which
    is off-domain by construction), else synthesize a different domain's."""
    if inv["payload_body"] and rng.random() < 0.6:
        return rng.choice(inv["payload_body"]), ("", "")
    others = [d for d in DOMAINS if d != domain_key]
    return keep_payload(rng, rng.choice(others), kind)


# anchors ---------------------------------------------------------------------

K_ANCHOR = [
    "Use the check results above for <E>: if they are clean, proceed with <ACTION>.",
    "基于上面 <E> 的检查结果继续 <ACTION>。",
    "The output above is the evidence — apply <ACTION> for <E> accordingly.",
    "用上面的输出作为依据，继续处理 <E>。",
]
D_ANCHOR = [
    "那个检查到此为止，不用管了。新任务：<FTASK>。",
    "Never mind the earlier run — it's closed. New task: <FTASK>.",
    "先放下这个，换个任务：<FTASK>。",
]
SETUP = [
    "Prepare <ACTION> for <E>; first run the checks.",
    "任务：处理 <E>，先跑一遍检查，再 <ACTION>。",
    "Work on <E>: gather the evidence, then <ACTION>.",
]
# anchors for control-surface kinds (rejected / interrupted / verify /
# empty_poll): the control record is the trajectory-boundary evidence.
CTL_K_ANCHOR = [
    "The step above was refused/interrupted — that's why it stopped; "
    "continue <ACTION> for <E> read-only instead.",
    "上面那步被拒/中断了——这就是停下的原因；改用只读方式继续 <ACTION>。",
    "That fetch/check came back blocked or empty — record it as the "
    "blocker and proceed with <ACTION> using local files only.",
]
CTL_KINDS = {"tool_rejected", "request_interrupted", "domain_verify",
             "empty_poll"}
TAIL_FILLER = [
    ("assistant", "Logged the outcome."),
    ("assistant", "Moving to the next step."),
    ("assistant", rng_s := "先记录一下，继续往下。"),
]


# ---------------------------------------------------------------------------
# emit
# ---------------------------------------------------------------------------

def emit_record(records, gidx, segs, cand_idx, drop, kind, sub_family,
                pair_id, pair_kind, evidence, domain, rng, harvest_refs,
                elide=None):
    protect = {cand_idx}
    protect.update(i for i, (role, _t) in enumerate(segs) if role == "user")
    protect.add(0)
    if elide is None:
        elide = f1.pick_elision(segs, protect, rng)
    conv = f1.render_conv(segs, elide)
    users = [t for role, t in segs if role == "user"]
    ptr = f"/messages/{cand_idx}/content"
    state = {"conversation": conv, "candidate_pointer": ptr,
             "user_messages_in_order": users}
    state_str = builder.serialized(state)
    role = segs[cand_idx][0]
    kind_map = {"user": "user_turn", "assistant": "assistant_text",
                "tool": "tool_result"}
    meta = {
        "record_id": f"v5f3:{gidx:06d}:{sub_family[:14]}:"
                     f"{'d' if drop else 'k'}",
        "domain": domain,
        "modality": "text",
        "language_bucket": f1.lang_bucket(users),
        "source_dataset": GROUP_LINEAGE,
        "source_split": "train",
        "candidate_kind": kind_map.get(role, role),
        "candidate_pointer": ptr,
        "family": FAMILY,
        "sub_family": sub_family,
        "envelope_dialect": ("claude" if sub_family in
                             ("web_search_shell", "domain_verify",
                              "tool_rejected", "request_interrupted")
                             else "codex" if sub_family in
                             ("chunk_exec", "script_completed",
                              "input_text_wrap", "empty_poll")
                             else "shared"),
        "pair_id": pair_id,
        "pair_kind": pair_kind,
        "length_band": builder.length_band(len(conv)),
        "hard_negative": True,
        "windowed": bool(elide),
        "state_chars": len(state_str.encode("utf-8")),
        "seed": SEED,
        "evidence_basis": evidence,
        "harvest_refs": sorted(set(harvest_refs))[:8],
    }
    records.append({
        "group_id": f"{GROUP_LINEAGE}:"
                    f"{f1.group_digest(FAMILY, gidx)}",
        "request": {"state": state_str,
                    "questions": {"irrelevant": {
                        "type": "noul",
                        "instructions": INSTRUCTIONS}}},
        "targets": {"irrelevant": {"probabilities": {
            "true": 1.0 if drop else 0.0,
            "false": 0.0 if drop else 1.0}}},
        "meta": meta,
    })


def _filler(rng, n=2):
    return [("assistant", rng.choice(FILLER_ASSISTANT)) for _ in range(n)]


# --- pair emitters -------------------------------------------------------------

def emit_pair_payload_flip(gidx, records, rng, inv, refmap, kind, dialect_role):
    """One conversation, two same-envelope tool results at different
    positions: payload_keep is the anchor's evidence, payload_drop belongs
    to a closed side errand. Records differ only in candidate_pointer."""
    dk = rng.choice(list(DOMAINS))
    e = rng.choice(DOMAINS[dk]["entities"])
    action = rng.choice(DOMAINS[dk]["actions"])
    kw = {"E": e, "ACTION": action}
    kp, _ = keep_payload(rng, dk, kind)
    dp, _ = drop_payload(rng, dk, kind, inv)
    kp_text = kp if isinstance(kp, str) else "\n".join(kp)
    dp_text = dp if isinstance(dp, str) else "\n".join(dp)
    env = _render_envelope(rng, kind, inv, refmap, kp_text)
    env_d = _render_envelope(rng, kind, inv, refmap, dp_text)
    refs = []
    foreign_setup = rng.choice(f1.A_FOREIGN_SETUP)
    segs = [
        ("system", SYSTEM_TEXT),
        ("user", fill(foreign_setup, rng,
                      FTHING=rng.choice(DOMAINS[dk]["things"]))),
        ("assistant", "Checking that first."),
        ("assistant", builder.serialized({"tool_use": {
            "name": "shell", "input": {"command": "check side task"}}})),
        ("tool", env_d),                               # idx 4 = DROP
        ("assistant", "That is settled — closing it."),
        ("user", fill(rng.choice([
            "好，那个到此为止。现在主线：<TASK>",
            "Side errand closed. Main task: <TASK>"]), rng,
            TASK=fill(rng.choice(SETUP), rng, **kw))),
        ("assistant", "Running the real checks."),
        ("assistant", builder.serialized({"tool_use": {
            "name": "shell", "input": {"command": f"check {e}"}}})),
        ("tool", env),                                 # idx 9 = KEEP
        ("assistant", "Got the result."),
        ("user", fill(rng.choice(CTL_K_ANCHOR if kind in CTL_KINDS
                                 else K_ANCHOR), rng, **kw)),
    ]
    drop_idx = segs.index(("tool", env_d))
    # constant envelopes (rejected/interrupted/empty_poll/verify) can be
    # byte-identical between the two positions — take the LAST occurrence
    keep_idx = len(segs) - 1 - segs[::-1].index(("tool", env))
    elide = f1.pick_elision(segs, {drop_idx, keep_idx}, rng)
    pid = f"v5f3pair:{gidx:06d}:pflip"
    emit_record(records, gidx, segs, keep_idx, False, dialect_role, kind,
                pid, "envelope_payload_flip",
                "envelope is harness boilerplate; payload is the anchor's "
                "required evidence", dk, rng, refs, elide=elide)
    emit_record(records, gidx, segs, drop_idx, True, dialect_role, kind,
                pid, "envelope_payload_flip",
                "identical envelope shape; payload belongs to the explicitly "
                "closed side errand -> certainly irrelevant",
                dk, rng, refs, elide=elide)


def emit_pair_anchor_flip(gidx, gidx2, records, rng, inv, refmap, kind,
                          dialect_role):
    """Identical envelope+payload candidate; keep-anchor record in group
    gidx, drop-anchor record in a DIFFERENT group gidx2 (doc §4 — the same
    payload appears once keep / once drop across groups)."""
    dk = rng.choice(list(DOMAINS))
    e = rng.choice(DOMAINS[dk]["entities"])
    action = rng.choice(DOMAINS[dk]["actions"])
    kw = {"E": e, "ACTION": action}
    kp, _ = keep_payload(rng, dk, kind)
    kp_text = kp if isinstance(kp, str) else "\n".join(kp)
    env = _render_envelope(rng, kind, inv, refmap, kp_text)
    head = [
        ("system", SYSTEM_TEXT),
        ("user", fill(rng.choice(SETUP), rng, **kw)),
        ("assistant", "Running the checks."),
        ("assistant", builder.serialized({"tool_use": {
            "name": "shell", "input": {"command": f"check {e}"}}})),
        ("tool", env),                                 # idx 4 = candidate
        ("assistant", "Result captured."),
    ]
    conv_k = head + _filler(rng, 1) + [
        ("user", fill(rng.choice(CTL_K_ANCHOR if kind in CTL_KINDS
                                 else K_ANCHOR), rng, **kw))]
    conv_d = head + _filler(rng, 1) + [
        ("user", fill(rng.choice(D_ANCHOR), rng, **kw))]
    elide = f1.shared_elision(conv_k, rng, 4)
    pid = f"v5f3pair:{gidx:06d}:aflip"
    emit_record(records, gidx, conv_k, 4, False, dialect_role, kind,
                pid, "anchor_flip",
                "anchor makes the envelope's payload the operative evidence",
                dk, rng, [], elide=elide)
    emit_record(records, gidx2, conv_d, 4, True, dialect_role, kind,
                pid, "anchor_flip",
                "identical candidate, anchor explicitly closed the check "
                "and switched foreign -> certainly irrelevant",
                dk, rng, [], elide=elide)


def emit_pair_lowcontent(gidx, gidx2, records, rng, inv, refmap, kind):
    """Low-content conversational forms (kev clusters E/F/G): micro user
    openers, assistant scope-acks, completion reports. anchor_flip only —
    same text is a task-relevant commitment under one anchor, foreign noise
    under another."""
    dk = rng.choice(list(DOMAINS))
    e = rng.choice(DOMAINS[dk]["entities"])
    action = rng.choice(DOMAINS[dk]["actions"])
    kw = {"E": e, "ACTION": action}
    if kind == "user_micro":
        cand_role = "user"
        cand_text = rng.choice(
            inv["user_micro"] or ["先把链路跑通再录", "研究一下",
                                  "run the checks first"])
        cand_idx = 1
    elif kind == "assistant_micro":
        cand_role = "assistant"
        cand_text = rng.choice(
            inv["assistant_micro"] or [
                "我先只读查看，不改文件。",
                "I'll keep this read-only and inspect first.",
                "先确认范围再动手。"])
        cand_idx = 2
    else:  # completion_report
        cand_role = "assistant"
        cand_text = rng.choice(
            inv["report"] or [
                f"完成：{e} 验证通过，改动仅落在约定文件，输出为空即无异常。"])
        cand_idx = 6
    slot = {"user_micro": "user_micro", "assistant_micro": "assistant_micro",
            "completion_report": "report"}[kind]
    refs = f1.ref_for(refmap, slot, cand_text)
    if cand_role == "assistant" and kind == "completion_report":
        head = [
            ("system", SYSTEM_TEXT),
            ("user", fill(rng.choice(SETUP), rng, **kw)),
            ("assistant", "On it."),
            ("assistant", builder.serialized({"tool_use": {
                "name": "shell", "input": {"command": f"verify {e}"}}})),
            ("tool", "checks done"),
            ("assistant", "Wrapping up."),
            ("assistant", cand_text),
        ]
    elif cand_role == "assistant":
        head = [
            ("system", SYSTEM_TEXT),
            ("user", fill(rng.choice(SETUP), rng, **kw)),
            ("assistant", cand_text),
            ("assistant", builder.serialized({"tool_use": {
                "name": "shell", "input": {"command": f"inspect {e}"}}})),
            ("tool", "inspection output"),
        ]
    else:
        head = [
            ("system", SYSTEM_TEXT),
            ("user", cand_text),
            ("assistant", "Acknowledged — working on it."),
            ("assistant", builder.serialized({"tool_use": {
                "name": "shell", "input": {"command": f"work {e}"}}})),
            ("tool", "progress"),
        ]
    conv_k = head + _filler(rng, 1) + [
        ("user", fill(rng.choice(K_ANCHOR), rng, **kw))]
    conv_d = head + _filler(rng, 1) + [
        ("user", fill(rng.choice(D_ANCHOR), rng, **kw))]
    elide = f1.shared_elision(conv_k, rng, cand_idx)
    pid = f"v5f3pair:{gidx:06d}:aflip"
    emit_record(records, gidx, conv_k, cand_idx, False, kind, kind,
                pid, "anchor_flip",
                "low-content turn carries the task's scope/commitment the "
                "anchor relies on", dk, rng, refs, elide=elide)
    emit_record(records, gidx2, conv_d, cand_idx, True, kind, kind,
                pid, "anchor_flip",
                "identical low-content turn under a foreign anchor after "
                "explicit task closure -> certainly irrelevant",
                dk, rng, refs, elide=elide)


def _render_envelope(rng, kind, inv, refmap, payload):
    if kind == "chunk_exec":
        return envelope_chunk(rng, payload, inv, refmap)
    if kind == "script_completed":
        return envelope_script(rng, payload, inv, refmap)
    if kind == "input_text_wrap":
        return envelope_wrap(rng, payload, inv, refmap)
    if kind == "file_dump":
        lines = payload if isinstance(payload, list) else payload.split("\n")
        return envelope_filedump(rng, lines, inv, refmap)
    if kind == "url_probe":
        return envelope_urlprobe(rng, payload, inv, refmap)
    if kind == "git_status":
        return envelope_gitstatus(rng, payload, inv, refmap)
    if kind == "web_search_shell":
        q = payload.split("\n", 1)[0].lstrip("-•* ").strip()[:60]
        return envelope_websearch(rng, q, payload, inv, refmap)
    if kind == "domain_verify":
        return envelope_verify(rng, rng.choice(HOSTS), inv, refmap)
    if kind == "tool_rejected":
        return envelope_rejected(rng, inv, refmap)
    if kind == "request_interrupted":
        return envelope_interrupted(rng, inv, refmap)
    if kind == "empty_poll":
        return envelope_empty_poll(rng, inv, refmap)
    return payload


ENVELOPE_KINDS = ("chunk_exec", "script_completed", "input_text_wrap",
                  "file_dump", "url_probe", "git_status", "web_search_shell",
                  "domain_verify", "tool_rejected", "request_interrupted",
                  "empty_poll")
LOW_KINDS = ("user_micro", "assistant_micro", "completion_report")


# ---------------------------------------------------------------------------
# self-check / report (F1 conventions)
# ---------------------------------------------------------------------------

argmax_label = f1.argmax_label


def self_check(records):
    errors = []
    seen_ids = set()
    DIGITS = re.compile(r"\d+(\.\d+)?")

    def norm_state(state_text):
        state = json.loads(state_text)
        for m in state.get("conversation", []):
            if "content" in m:
                m["content"] = DIGITS.sub("<N>", m["content"])
        return json.dumps(state, sort_keys=True, ensure_ascii=False)

    norm_groups = defaultdict(set)
    pair_members = defaultdict(list)
    for r in records:
        meta = r["meta"]
        rid = meta["record_id"]
        if rid in seen_ids:
            errors.append(f"{rid}: duplicate record_id")
        seen_ids.add(rid)
        if not rid.startswith("v5f3:"):
            errors.append(f"{rid}: bad prefix")
        norm_groups[norm_state(r["request"]["state"])].add(argmax_label(r))
        try:
            state = json.loads(r["request"]["state"])
        except ValueError:
            errors.append(f"{rid}: unparseable state")
            continue
        ptr = state["candidate_pointer"]
        conv = state["conversation"]
        ptrs = [s["pointer"] for s in conv]
        if ptr not in ptrs:
            errors.append(f"{rid}: candidate pointer missing")
        cand = next((s for s in conv if s["pointer"] == ptr), None)
        if cand is None or not cand["content"].strip():
            errors.append(f"{rid}: empty candidate")
        if cand and len(cand["content"].encode()) > MAX_CANDIDATE_BYTES:
            errors.append(f"{rid}: candidate over 32KB")
        users = state.get("user_messages_in_order") or []
        user_segs = [s for s in conv if s.get("role") == "user"]
        if not users or not user_segs:
            errors.append(f"{rid}: missing user anchor")
        elif user_segs[-1]["content"] != users[-1]:
            errors.append(f"{rid}: anchor mismatch")
        for s in conv:
            hit = builder.credential_scan(s.get("content", ""))
            if hit:
                errors.append(f"{rid}: credential_scan:{hit}")
                break
            if SLOT_RE.search(s.get("content", "")):
                errors.append(f"{rid}: unfilled slot")
                break
        for u in users:
            if builder.credential_scan(u):
                errors.append(f"{rid}: credential in user message")
                break
            if SLOT_RE.search(u):
                errors.append(f"{rid}: unfilled slot in user message")
                break
        if meta.get("pair_id"):
            pair_members[meta["pair_id"]].append(r)

    for state_key, labels in norm_groups.items():
        if len(labels) > 1:
            errors.append(f"label-consistency: normalized state has {labels}")
    for pid, members in pair_members.items():
        labels = {argmax_label(m) for m in members}
        if labels != {"true", "false"}:
            errors.append(f"{pid}: pair labels {sorted(labels)}")
        kinds = {m["meta"]["pair_kind"] for m in members}
        if len(kinds) != 1:
            errors.append(f"{pid}: mixed pair_kind")
        elif kinds == {"anchor_flip"}:
            cand_texts, groups = set(), set()
            for m in members:
                st = json.loads(m["request"]["state"])
                cp = st["candidate_pointer"]
                cand_texts.add(next(
                    s["content"] for s in st["conversation"]
                    if s["pointer"] == cp))
                groups.add(m["group_id"])
            if len(cand_texts) != 1:
                errors.append(f"{pid}: anchor_flip pair different text")
            if len(groups) != 2:
                errors.append(f"{pid}: anchor_flip pair should span 2 groups")
        elif kinds == {"envelope_payload_flip"}:
            if len({m["meta"]["candidate_pointer"] for m in members}) != 2:
                errors.append(f"{pid}: payload_flip needs 2 pointers")
            if len({m["group_id"] for m in members}) != 1:
                errors.append(f"{pid}: payload_flip must share group")
    if errors:
        raise AssertionError(f"self_check failed ({len(errors)}): "
                             + "; ".join(errors[:12]))
    return {"records_checked": len(records), "status": "pass"}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_JSONL))
    ap.add_argument("--report", default=str(REPORT))
    args = ap.parse_args()
    out_jsonl, report_path = Path(args.out), Path(args.report)

    rng = random.Random(SEED)
    instructions = builder.load_verbatim_instructions(
        ROOT / "data" / "valen_nano_v3")
    instr_sha = hashlib.sha256(instructions.encode("utf-8")).hexdigest()
    assert instr_sha == f1.CANONICAL_INSTR_SHA, \
        f"instructions drifted: {instr_sha}"
    global INSTRUCTIONS
    INSTRUCTIONS = instructions
    f1.INSTRUCTIONS = instructions

    inv, refmap, harvest_stats = harvest_pool()
    ref_requests, ref_states = miner.load_reference_hashes()

    records = []
    pair_seq = 0
    kinds = [k for k in ENVELOPE_KINDS if PLAN.get(k)]
    # deterministic interleave across envelope kinds
    remaining = {k: PLAN[k] for k in PLAN}
    order = list(PLAN)
    while any(remaining.values()):
        for kind in order:
            if remaining[kind] <= 0:
                continue
            remaining[kind] -= 1
            if kind in LOW_KINDS:
                emit_pair_lowcontent(pair_seq, pair_seq + 10 ** 6,
                                     records, rng, inv, refmap, kind)
            elif rng.random() < ANCHOR_FLIP_SHARE:
                emit_pair_anchor_flip(pair_seq, pair_seq + 10 ** 6,
                                      records, rng, inv, refmap, kind,
                                      "tool")
            else:
                emit_pair_payload_flip(pair_seq, records, rng, inv,
                                       refmap, kind, "tool")
            pair_seq += 1

    # unique record_ids
    for i, r in enumerate(records):
        r["meta"]["record_id"] = f"{r['meta']['record_id']}:{i:05d}"

    # dedup vs prior corpora + intra-set + vs f1 output if present
    dedup = Counter()
    f1_path = OUT_DIR / "f1_antishortcut.jsonl"
    if f1_path.is_file():
        for line in f1_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            ref_requests.add(builder.sha256_bytes(
                builder.serialized(rec["request"]).encode("utf-8")))
            sh = builder.normalized_state_hash(rec)
            if sh:
                ref_states.add(sh)
    seen_requests, seen_states = set(), set()
    kept = []
    for r in records:
        rhash = builder.sha256_bytes(
            builder.serialized(r["request"]).encode("utf-8"))
        shash = builder.normalized_state_hash(r)
        if rhash in ref_requests or shash in ref_states:
            dedup["vs_prior_corpora"] += 1
            continue
        if rhash in seen_requests or shash in seen_states:
            dedup["intra_set"] += 1
            continue
        seen_requests.add(rhash)
        seen_states.add(shash)
        kept.append(r)
    # drop orphan pair members whose partner was deduplicated
    pid_counts = Counter(r["meta"]["pair_id"] for r in kept)
    before = len(kept)
    kept = [r for r in kept
            if not r["meta"]["pair_id"] or pid_counts[r["meta"]["pair_id"]] >= 2]
    dedup["orphan_pair_dropped"] = before - len(kept)
    records = kept

    check = self_check(records)

    label_counts = Counter(argmax_label(r) for r in records)
    sub_counts = Counter(r["meta"]["sub_family"] for r in records)
    dialect_counts = Counter(r["meta"]["envelope_dialect"] for r in records)
    domain_counts = Counter(r["meta"]["domain"] for r in records)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(builder.serialized(r) + "\n")

    section = {
        "builder": "scripts/build_v5_boilerplate_breaker_v1.py",
        "file": str(out_jsonl.relative_to(ROOT)),
        "records": len(records),
        "sha256": builder.sha256_file(out_jsonl),
        "bytes": out_jsonl.stat().st_size,
        "labels": dict(sorted(label_counts.items())),
        "keep_drop_ratio": round(
            label_counts["false"] / max(1, label_counts["true"]), 3),
        "sub_families": dict(sorted(sub_counts.items())),
        "dialects": dict(sorted(dialect_counts.items())),
        "domains": dict(sorted(domain_counts.items())),
        "pairs": len({r["meta"]["pair_id"] for r in records
                      if r["meta"]["pair_id"]}),
        "paired_records": sum(1 for r in records if r["meta"]["pair_id"]),
        "windowed": sum(1 for r in records if r["meta"]["windowed"]),
        "groups": len({r["group_id"] for r in records}),
        "self_check": check,
    }
    report = {}
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except ValueError:
            report = {}
    report.setdefault("schema_version", SCHEMA)
    report["seed"] = SEED
    report["content_free"] = True
    report["training_allowed"] = True
    report["instructions_sha256"] = instr_sha
    report["instructions_preserved_verbatim"] = True
    report.setdefault("families", {})[FAMILY] = section
    report.setdefault("harvest", {})[FAMILY] = {
        "source": str(HARVEST_SOURCE.relative_to(ROOT)),
        "note": ("envelope skeletons + foreign payload bodies harvested "
                 "from the v5 mining candidate pool (train-pool transcripts "
                 "only); every string re-scanned; both codex and claude "
                 "dialects emitted"),
        "counts": {k: len(v) for k, v in inv.items()},
        "stats": harvest_stats,
    }
    report.setdefault("dedup", {})[FAMILY] = {
        "reference_sets": ["data/valen_nano_v2", "data/valen_nano_v3",
                           "data/valen_nano_v4",
                           "data/real_context_eval_v1/candidates*.jsonl",
                           "data/v5_corpus/f1_antishortcut.jsonl"],
        **dict(sorted(dedup.items())),
    }
    report.setdefault("caveats", {})
    report["caveats"][FAMILY] = [
        "contrastive pairs only; keep:drop = 1:1 by construction",
        "anchor_flip pair members live in different groups so the same "
        "payload cannot leak payload->label",
        "envelope_payload_flip pairs share one conversation and differ "
        "only in candidate_pointer (envelope identical, payload relevant "
        "vs foreign)",
        "drop payloads are harvested real bodies (foreign by construction) "
        "or foreign-domain syntheses",
    ]
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print(f"wrote {len(records)} records -> {out_jsonl}")
    print("labels:", dict(sorted(label_counts.items())),
          "| dialects:", dict(sorted(dialect_counts.items())))
    print("subs:", dict(sorted(sub_counts.items())))
    print("pairs:", section["pairs"], "| dedup:", dict(sorted(dedup.items())))
    print(f"report -> {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
