#!/usr/bin/env python3
"""V5 family F1 — anti_shortcut corpus builder
(data/v5_corpus/f1_antishortcut.jsonl, ~1500 LABELED records).

Implements docs/V5_DATA_DESIGN_V1.md §2 (family F1) as the synthetic/hybrid
emitter half: template-derived skeletons x real transcript content slots.
The failure being fixed (docs/REAL_CONTEXT_EVAL_RESULTS_V1.md lora-FP-pattern):
the v4 head learned "no content -> no meaning" and confidently dropped
empty/negative results that WERE the answer (noul=0.919), assistant
correction segments (0.615), and low-content user turns carrying constraints.

Sub-families (all candidates short/empty/negative/boilerplate-looking):
  a) empty_result_is_answer — "total 0" / "no matches" / "file not found"
     results where the anchor task needs exactly that fact.
  b) correction_keep — "Important catch —" / "Actually, X" / retraction
     segments mid-task.
  c) short_ack_with_constraint — micro user turns carrying constraints
     ("别动其他文件", "只用 local") vs micro turns carrying none.
Overall keep:drop ~= 2:1 (keep-side protection family; drop members are real
short/empty outputs that are truly irrelevant, preventing the reverse
shortcut "empty -> keep").

Contrastive pairs (doc §1.7): every pair shares meta.pair_id and carries
pair_kind:
  - same_surface:   identical candidate surface text, same conversation,
                    different candidate_pointer, opposite labels;
  - anchor_flip:    identical candidate pointer + text, different final
                    anchor (explicit task closure -> foreign task), opposite
                    labels.

Content slots are harvested from the ALREADY-VETTED v5 mining pool
(data/v5_mining/f2_hard_negative_candidates.jsonl — train-pool transcripts
only, eval-v1 frozen and v5-eval-reserved transcripts are never in it,
component-level credential scan already applied; every harvested string is
re-scanned anyway). Where the pool is thin, doc-described canonical surface
forms fill in (KEV_FP_PATTERN_V1.md §2 clusters B/E/F and the lora FP table).

Record schema is byte-compatible with data/valen_nano_v4/eval.jsonl:
{group_id, request:{state: JSON-string {conversation, candidate_pointer,
user_messages_in_order}, questions:{irrelevant:{type:"noul",
instructions:<verbatim canonical>}}}, targets, meta{record_id:v5f1:*,
pair_id, family, length_band, ...}}. Labels are a pure function of state
content; self_check enforces schema, pointer resolution, credential/
placeholder scan (component level), pair integrity, and the v4
label-consistency contract (no normalized-identical states with opposite
labels). Deterministic: seed 20261201, sorted iteration, no network, no
model calls.

Usage: python3 scripts/build_v5_antishortcut_v1.py
Output: data/v5_corpus/f1_antishortcut.jsonl
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

OUT_DIR = ROOT / "data" / "v5_corpus"
OUT_JSONL = OUT_DIR / "f1_antishortcut.jsonl"
REPORT = OUT_DIR / "build_report.json"
HARVEST_SOURCE = ROOT / "data" / "v5_mining" / "f2_hard_negative_candidates.jsonl"

SEED = 20261201                 # v5 reserved seed (V5_DATA_DESIGN_V1 §1.3)
GROUP_LINEAGE = "context_relevance_v5"
FAMILY = "f1_antishortcut"
SCHEMA = "nanojev-v5-corpus-v1"
INSTRUCTIONS = None             # set in main() from the canonical valen record
CANONICAL_INSTR_SHA = "4659848727be93d683a792b8566b97e82191f114b9aa1944a74c6df05f222b90"
SYSTEM_TEXT = "Preserve current user constraints and required evidence."

# record-count plan (keep:drop ~2:1 overall -> ~990 keep / ~510 drop)
PLAN = {
    "empty_result_is_answer": {"pairs_same_surface": 120, "pairs_anchor_flip": 55,
                               "solo_keep": 200, "solo_drop": 0},
    "correction_keep":        {"pairs_same_surface": 60, "pairs_anchor_flip": 75,
                               "solo_keep": 200, "solo_drop": 0},
    "short_ack_with_constraint": {"pairs_same_surface": 0, "pairs_anchor_flip": 90,
                                  "solo_keep": 190, "solo_drop": 110},
}
WINDOWED_FRACTION = 0.30       # share of groups rendered with elision markers
MAX_HARVEST = 400              # per harvested slot inventory
MAX_CANDIDATE_BYTES = 32 * 1024

# ---------------------------------------------------------------------------
# Harvest: real content slots from the v5 mining candidate pool.
# ---------------------------------------------------------------------------

EMPTY_RE = re.compile(
    r"(?:^|\b)total\s+0\b|no matches|not found|no such file|0 hits|"
    r"no output|\"output\"\s*:\s*\"\"|0 results|nothing found|"
    r"cannot access|does not exist|0 rows|未找到|无匹配|没有.{0,6}结果|"
    r"找不到|为空", re.IGNORECASE)
CORRECT_RE = re.compile(
    r"important catch|actually[, —]|correction:|i was wrong|let me correct|"
    r"scratch that|on second thought|walk.*back|i need to correct|"
    r"更正|纠正|说错|搞错|弄错|应为|其实是|撤回|改正", re.IGNORECASE)
CONSTRAINT_RE = re.compile(
    r"别|不要|先别|勿|只|仅|最多|至少|限制|约束|不能|"
    r"keep it|only |don'?t |without |read[- ]?only|no new|stay inside|"
    r"under \d|must not|do not|never ", re.IGNORECASE)
SLOT_RE = re.compile(r"<[A-Z][A-Z0-9_]*>")   # generation slot markers

HARVEST_CAPS = {"empty": 320, "correction": 200, "constraint": 200,
                "ack": 120, "filler_tool": 160, "filler_user": 120}


def _norm_key(text):
    return builder.normalize_text(text)[:400]


def harvest_pool(path=HARVEST_SOURCE):
    """Scan the v5 mining candidate file; collect real surface strings for
    content slots. Returns {slot: [text,...]} plus source refs."""
    inv = {k: [] for k in HARVEST_CAPS}
    refmap = {k: {} for k in HARVEST_CAPS}   # slot -> {text: source record_id}
    seen = set()
    if not Path(path).is_file():
        return inv, refmap, {"source_missing": 1}
    stats = Counter()
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
            ntext = _norm_key(text)
            if not ntext or ntext in seen:
                continue
            if builder.credential_scan(text):
                stats["harvest_cred_skip"] += 1
                continue

            def take(slot, t):
                if len(inv[slot]) >= HARVEST_CAPS[slot]:
                    return
                if SLOT_RE.search(t):
                    stats["harvest_slotmarker_skip"] += 1
                    return
                inv[slot].append(t)
                refmap[slot][t] = rid
                seen.add(ntext)
                stats[f"harvest_{slot}"] += 1

            if role == "tool" and len(text) <= 420 and EMPTY_RE.search(text):
                take("empty", text)
            elif role == "assistant" and CORRECT_RE.search(text):
                # full segment if short, else the marker-bearing first line
                if len(text) <= 480:
                    take("correction", text)
                else:
                    first = text.split("\n", 1)[0]
                    if 20 <= len(first) <= 300 and CORRECT_RE.search(first):
                        take("correction", first)
            elif role == "user" and len(text) <= 140:
                if miner.is_continuation_anchor(text) or len(ntext) <= 40:
                    take("ack", text)
                elif CONSTRAINT_RE.search(text) \
                        and not text.rstrip().endswith(("?", "？", "吗")):
                    take("constraint", text)
            elif role == "tool" and 30 <= len(text) <= 800:
                take("filler_tool", text)
            elif role == "user" and 8 <= len(text) <= 220:
                take("filler_user", text)
    return inv, refmap, dict(stats)


def ref_for(refmap, slot, text):
    """Provenance: the mined-pool record a harvested slot string came from."""
    r = refmap.get(slot, {}).get(text)
    return [r] if r else []


# Fallback inventories (doc-described canonical surface forms; used to pad
# thin harvest slots so every family keeps coverage).
EMPTY_FALLBACK = [
    "total 0",
    "no matches found",
    "grep: no matches",
    "find: <PATH>: No such file or directory",
    "ls: cannot access '<PATH>': No such file or directory",
    "(Bash completed with no output)",
    '{"output":""}',
    "0 hits",
    "No results.",
    "zsh:1: no matches found: <PAT>",
    "process not found",
    "0 rows returned",
    "未找到匹配项",
    "没有匹配的结果",
]
CORRECTION_FALLBACK = [
    "Important catch — <STATEMENT>. Adjusting before we continue.",
    "Actually, wait — <STATEMENT>.",
    "Correction: <STATEMENT>.",
    "更正一下：<STATEMENT_ZH>。",
    "刚才说错了，<STATEMENT_ZH>。",
    "Scratch that — <STATEMENT>.",
    "On second thought, <STATEMENT>.",
    "I need to walk that back: <STATEMENT>.",
    "等一下，前面那个判断不对——<STATEMENT_ZH>。",
]
CONSTRAINT_FALLBACK = [
    "别动其他文件",
    "只改 <FILE>，其他别动",
    "只用 local",
    "只读模式就行，不要写",
    "keep it read-only",
    "no new dependencies",
    "don't touch anything else",
    "keep it under <N> lines",
    "only the docs, nothing else",
    "先别提交",
    "不要联网",
    "用最小改动",
    "stay inside <DIR>",
    "tests only, no source edits",
]
ACK_FALLBACK = [
    "好的", "继续", "ok", "go ahead", "嗯", "可以", "继续吧", "looks good",
    "thanks", "收到", "行", "接着来", "yep", "proceed",
]

# ---------------------------------------------------------------------------
# Scenario pools — >=5 task domains, zh/en mix, paraphrase slots.
# Slot syntax: <E> entity, <F> file, <DIR>, <THING> absent-thing,
# <FTHING> foreign thing, <ACTION>, <FCMD>/<CMD>, <PAT>, <V1>/<V2>,
# <PARAM>/<PARAM2>, <FTASK> foreign task, <N>, <Q>, <STATEMENT>.
# ---------------------------------------------------------------------------

DOMAINS = {
    "code": {
        "entities": ["payment-service", "auth-module", "ingest-worker",
                     "cache-layer", "router-core", "session-store"],
        "files": ["src/auth/session.py", "src/payments/charge.py",
                  "internal/cache/store.go", "web/router/index.ts",
                  "scripts/deploy.sh", "config/prod.yaml"],
        "things": ["stale lock files", "orphan temp tables",
                   "deprecated config keys", "leftover debug flags",
                   "TODO markers", "残留进程", "遗留的测试文件",
                   "未清理的临时表"],
        "actions": ["proceed with the rollout", "继续发布", "run the migration",
                    "执行清理脚本", "start the refactor"],
        "ftasks": ["draft the release announcement", "整理会议纪要",
                   "update the onboarding guide", "审阅预算表",
                   "summarize the support queue"],
    },
    "devops": {
        "entities": ["staging-cluster", "prod-deploy", "ci-pipeline",
                     "edge-cache", "备份任务", "监控告警"],
        "files": ["deploy/helm/values.yaml", "ci/workflows/main.yml",
                  "infra/nginx.conf", "deploy/scripts/rollback.sh"],
        "things": ["orphan cron entries", "stale containers",
                   "failed jobs", "残留的 pod", "未释放的锁"],
        "actions": ["重启服务", "continue the deploy", "apply the manifest",
                    "推进上线"],
        "ftasks": ["write the incident report", "整理巡检记录",
                   "draft the SLA summary"],
    },
    "robotics": {
        "entities": ["g1-simulation", "dexterous-hand", "locomotion-stack",
                     "rviz-plugin", "抓取模块"],
        "files": ["launch/bringup.launch.xml", "src/control/locomotion.h",
                  "urdf/g1.urdf", "config/gazebo.yaml"],
        "things": ["stale gazebo processes", "leftover bag files",
                   "残留仿真进程", "未保存的标定文件"],
        "actions": ["rerun the fall test", "继续跌倒分析", "重启仿真"],
        "ftasks": ["整理演示视频清单", "write the hardware checklist",
                   "更新实验记录"],
    },
    "research": {
        "entities": ["world-model-survey", "pi0-paper", "flm-report",
                     "benchmark-suite"],
        "files": ["notes/literature.md", "refs/biblio.bib",
                  "reports/comparison.md"],
        "things": ["duplicate citations", "unresolved references",
                   "重复条目", "缺失的引用"],
        "actions": ["finalize the survey", "继续综述整理", "export the notes"],
        "ftasks": ["draft the seminar agenda", "整理读书会记录",
                   "update the reading list"],
    },
    "docs": {
        "entities": ["user-manual", "api-reference", "onboarding-guide",
                     "faq-page"],
        "files": ["docs/api/reference.md", "docs/guide/quickstart.md",
                  "docs/faq.md", "docs/README.md"],
        "things": ["broken links", "outdated screenshots", "失效链接",
                   "过时的命令"],
        "actions": ["publish the docs", "继续文档更新", "merge the PR"],
        "ftasks": ["draft the changelog", "整理反馈清单",
                   "plan the workshop"],
    },
    "data": {
        "entities": ["sales-pipeline", "metrics-table", "ingest-job",
                     "对账任务"],
        "files": ["pipelines/daily_ingest.py", "sql/metrics.sql",
                  "data/exports/q3.csv"],
        "things": ["null rows", "duplicate keys", "空结果集",
                   "缺失的分区"],
        "actions": ["run the backfill", "继续对账", "export the report"],
        "ftasks": ["draft the weekly summary", "整理数据字典",
                   "update the dashboard notes"],
    },
}

CHECK_VERBS = ["grep -rn", "find", "rg", "ls", "ps aux | grep",
               "cat", "fd", "ack"]

FILLER_ASSISTANT = [
    "Working through the checklist now.",
    "Routine heartbeat log: all workers nominal.",
    "I will keep the changes minimal.",
    "Reading the surrounding context first.",
    "先看一下现状再动手。",
    "按步骤来，先确认现状。",
    "An archived meeting note mentions parking logistics.",
    "Cached stylesheet reference stored for later.",
]

# paraphrase banks ----------------------------------------------------------------

A_SETUP = [
    "Before <ACTION>, verify that <E> really has no <THING>. If nothing turns up, tell me so.",
    "先确认 <E> 里没有 <THING>，如果没有就告诉我，然后 <ACTION>。",
    "I need proof that <E> is clean of <THING>. Empty result = good. Then <ACTION>.",
    "确认一下 <E> 是不是已经没有 <THING> 了，空结果就是答案，之后 <ACTION>。",
    "Check whether <E> still contains any <THING>; a no-hit result is what we want before we <ACTION>.",
]
A_FOREIGN_SETUP = [
    "Quick side errand first: see if any <FTHING> are left over.",
    "顺手查一下 <FTHING> 还有没有残留。",
    "Also, quickly check for leftover <FTHING> — separate small thing.",
]
A_FOREIGN_ACK = [
    "Nothing there — noted.",
    "空的，收到。",
    "Right, that's clear. Moving on.",
    "没有发现，这个先放下。",
]
A_MAIN_TASK = [
    "OK, that errand is closed. Now the real check: confirm <E> has no <THING>.",
    "好，那个到此为止。现在确认 <E> 是否真的没有 <THING>。",
    "Done with that. Now — verify <E> is free of <THING>.",
    "那个先到这里。主线任务：查 <E> 有没有 <THING>。",
]
A_ANCHOR_KEEP = [
    "If the search came back empty, that confirms no <THING> — proceed with <ACTION> on that basis.",
    "结果是空的就证明没有 <THING>，按这个结论继续 <ACTION>。",
    "So — nothing found means we're clear. Apply <ACTION> now.",
    "空结果就是我们需要的证据，继续 <ACTION>。",
    "Empty output = answer. Go ahead with <ACTION>.",
]
A_ANCHOR_DROP = [
    "那个检查到此为止，不用管了。新任务：<FTASK>。",
    "Never mind that check — it's closed. New task: <FTASK>.",
    "先不查这个了，换方向：<FTASK>。",
    "Forget the earlier search entirely. New task now: <FTASK>.",
]
B_SETUP = [
    "Update <F>: set <PARAM> to <V1> and verify it took effect.",
    "把 <F> 里的 <PARAM> 改成 <V1>，改完验证一下。",
    "Apply <PARAM>=<V1> in <F> and confirm.",
    "调整 <F> 的配置：<PARAM> 设为 <V1>，然后确认生效。",
]
B_ANCHOR_KEEP = [
    "Verify the corrected setting is in place and summarize what was applied.",
    "确认修正后的配置已生效，并汇总实际写入的值。",
    "Now double-check <F> reflects the corrected value.",
    "核对一下 <F> 最终落地的值，给我一个结论。",
]
B_ANCHOR_DROP = A_ANCHOR_DROP
C_SETUP = [
    "Refactor <F> and run the related tests.",
    "重构 <F>，顺手把相关测试跑一遍。",
    "Clean up <F> and verify nothing breaks.",
    "整理 <F> 的实现，确认行为不变。",
]
C_ANCHOR_KEEP = [
    "Continue — run the verification now, within those limits.",
    "继续，按刚才的限制做验证。",
    "Go ahead with the check, respecting what I said.",
    "按我说的约束继续执行。",
]
C_ANCHOR_DROP = A_ANCHOR_DROP

STATEMENTS_EN = [
    "the field is <PARAM2>, not <PARAM> — the earlier name is deprecated",
    "the value must be <V2>, not <V1>; the first read was stale",
    "the flag lives under <PARAM2>, so the previous edit targeted the wrong key",
    "the limit is actually <V2>; my earlier figure was wrong",
]
STATEMENTS_ZH = [
    "字段名是 <PARAM2> 不是 <PARAM>，之前的名字已经废弃",
    "值应该是 <V2> 不是 <V1>，刚才读到的是旧缓存",
    "上限其实是 <V2>，之前的数字不对",
    "要改的是 <PARAM2>，刚才那次编辑改错了地方",
]
PARAMS = ["timeout_ms", "max_retries", "cache_ttl", "rate_limit",
          "batch_size", "queue_depth", "deploy_window", "refresh_interval"]
FILES_FLAT = [f for d in DOMAINS.values() for f in d["files"]]
DIRS = ["src/", "config/", "deploy/", "docs/", "scripts/", "infra/"]


# ---------------------------------------------------------------------------
# generation helpers
# ---------------------------------------------------------------------------

def fill(text, rng, **kw):
    """Replace <KEY> slots. Keys not in kw are drawn from built-in pools."""
    defaults = {
        "N": lambda: str(rng.randrange(50, 400)),
        "V1": lambda: str(rng.randrange(100, 900)),
        "V2": lambda: str(rng.randrange(100, 900)),
        "PATH": lambda: rng.choice(FILES_FLAT),
        "FILE": lambda: rng.choice(FILES_FLAT),
        "DIR": lambda: rng.choice(DIRS),
        "PAT": lambda: rng.choice(["*.lock", "*.tmp", "debug_*", "*.bak"]),
        "PARAM": lambda: rng.choice(PARAMS),
        "PARAM2": lambda: rng.choice(PARAMS),
        "STATEMENT": lambda: rng.choice(STATEMENTS_EN),
        "STATEMENT_ZH": lambda: rng.choice(STATEMENTS_ZH),
        "FTASK": lambda: rng.choice(
            [t for d in DOMAINS.values() for t in d["ftasks"]]),
    }
    def rep(m):
        key = m.group(0)[1:-1]
        if key in kw:
            return str(kw[key])
        if key in defaults:
            return defaults[key]()
        return m.group(0)
    prev = None
    while prev != text:
        prev = text
        text = SLOT_RE.sub(rep, text)
    return text


def tool_use(text):
    return ("assistant", builder.serialized({"tool_use": text}))


def render_conv(segs, elide):
    """segs: list of (role, text). elide: set of indices to drop.
    Emits /elided markers exactly like the v1 windowing renderer."""
    conv, dropped, markers = [], 0, 0
    for i, (role, text) in enumerate(segs):
        if i in elide:
            dropped += 1
            continue
        if dropped:
            conv.append({"pointer": f"/elided/{markers}", "role": "control",
                         "content": f"<elided {dropped} earlier segments>"})
            markers += 1
            dropped = 0
        conv.append({"pointer": f"/messages/{i}/content", "role": role,
                     "content": text})
    if dropped:
        conv.append({"pointer": f"/elided/{markers}", "role": "control",
                     "content": f"<elided {dropped} earlier segments>"})
    return conv


def pick_elision(segs, protect, rng):
    """Choose filler segments to elide (never candidate, never any user turn,
    never system). Returns a set, possibly empty."""
    if rng.random() >= WINDOWED_FRACTION:
        return set()
    candidates = [i for i, (role, _t) in enumerate(segs)
                  if i not in protect and role in ("assistant", "tool")]
    rng.shuffle(candidates)
    return set(candidates[:rng.randrange(1, min(4, len(candidates) + 1))])


def shared_elision(segs, rng, candidate_idx=None):
    """One elision set reused by every member of an anchor_flip pair so the
    pair differs only in the anchor text."""
    protect = {i for i, (role, _t) in enumerate(segs)
               if role in ("user", "system", "control")}
    if candidate_idx is not None:
        protect.add(candidate_idx)
    return pick_elision(segs, protect, rng)


def group_digest(family, gidx):
    return hashlib.sha256(
        f"{GROUP_LINEAGE}:{SEED}:{family}:{gidx}".encode()).hexdigest()


def lang_bucket(users):
    return "multi" if re.search(r"[一-鿿぀-ヿ가-힯]", "\n".join(users[:5])) \
        else "en"


def emit_record(records, gidx, segs, cand_idx, drop, kind, sub_family,
                pair_id, pair_kind, evidence, domain, rng, harvest_refs,
                elide=None):
    """Append one valen-schema record. segs: list of (role, text)."""
    protect = {cand_idx}
    protect.update(i for i, (role, _t) in enumerate(segs) if role == "user")
    protect.add(0)  # system
    if elide is None:
        elide = pick_elision(segs, protect, rng)
    conv = render_conv(segs, elide)
    users = [t for role, t in segs if role == "user"]
    ptr = f"/messages/{cand_idx}/content"
    state = {"conversation": conv, "candidate_pointer": ptr,
             "user_messages_in_order": users}
    state_str = builder.serialized(state)
    role = segs[cand_idx][0]
    kind_map = {"user": "user_turn", "assistant": "assistant_text",
                "tool": "tool_result"}
    rid = (f"v5f1:{gidx:06d}:{sub_family[:12]}:"
           f"{'d' if drop else 'k'}")
    # uniqueness handled by caller counter
    meta = {
        "record_id": rid,
        "domain": domain,
        "modality": "text",
        "language_bucket": lang_bucket(users),
        "source_dataset": GROUP_LINEAGE,
        "source_split": "train",
        "candidate_kind": kind_map.get(role, role),
        "candidate_pointer": ptr,
        "family": FAMILY,
        "sub_family": sub_family,
        "pair_id": pair_id,
        "pair_kind": pair_kind,
        "length_band": builder.length_band(len(conv)),
        "hard_negative": pair_id is not None,
        "windowed": bool(elide),
        "state_chars": len(state_str.encode("utf-8")),
        "seed": SEED,
        "evidence_basis": evidence,
        "harvest_refs": sorted(set(harvest_refs))[:8],
    }
    records.append({
        "group_id": f"{GROUP_LINEAGE}:{group_digest(FAMILY, gidx)}",
        "request": {"state": state_str,
                    "questions": {"irrelevant": {
                        "type": "noul",
                        "instructions": INSTRUCTIONS}}},
        "targets": {"irrelevant": {"probabilities": {
            "true": 1.0 if drop else 0.0,
            "false": 0.0 if drop else 1.0}}},
        "meta": meta,
    })


# ---------------------------------------------------------------------------
# sub-family emitters — each appends records; returns harvest refs used
# ---------------------------------------------------------------------------

def pick_empty(rng, inv):
    pool = inv["empty"] or EMPTY_FALLBACK
    return rng.choice(pool)


def fill_empty(surface, rng, e=None, dir_=None, fthing=None):
    """Fill placeholders inside a harvested/fallback empty-result string."""
    text = surface
    if "<PATH>" in text:
        text = text.replace("<PATH>", rng.choice(FILES_FLAT))
    if "<PAT>" in text:
        text = text.replace("<PAT>", rng.choice(["*.lock", "*.tmp",
                                                 "debug_*", "*.bak"]))
    return text


def emit_a_pair_same_surface(gidx, records, rng, inv, refmap):
    """One conversation; two identical-surface empty results: the relevant
    one is the answer (keep), the closed side-errand one is foreign (drop)."""
    d = rng.choice(list(DOMAINS))
    D = DOMAINS[d]
    e = rng.choice(D["entities"])
    thing = rng.choice(D["things"])
    action = rng.choice(D["actions"])
    fthing = rng.choice(D["things"])
    while fthing == thing:
        fthing = rng.choice(D["things"])
    surface = pick_empty(rng, inv)
    refs = ref_for(refmap, "empty", surface)
    empty = fill_empty(surface, rng)
    kw = {"E": e, "THING": thing, "ACTION": action, "FTHING": fthing}
    segs = [
        ("system", SYSTEM_TEXT),
        ("user", fill(rng.choice(A_FOREIGN_SETUP), rng, **kw)),
        ("assistant", rng.choice(A_FOREIGN_ACK[:2])),
        ("assistant", builder.serialized({"tool_use": {
            "name": "shell",
            "input": {"command": f"{rng.choice(CHECK_VERBS)} {fthing}"}}})),
        ("tool", empty),                              # idx 4 = DROP candidate
        ("assistant", rng.choice(A_FOREIGN_ACK)),
        ("user", fill(rng.choice(A_MAIN_TASK), rng, **kw)),
        ("assistant", fill(rng.choice([
            "Running the check on <E> now.",
            "我来查 <E> 的 <THING>。",
            "Checking <E> for <THING>."]), rng, **kw)),
        ("assistant", builder.serialized({"tool_use": {
            "name": "shell",
            "input": {"command": f"{rng.choice(CHECK_VERBS)} {thing} {e}"}}})),
        ("tool", empty),                              # idx 9 = KEEP candidate
        ("assistant", rng.choice([
            "The check returned empty.",
            "结果是空的。",
            "Empty result confirmed."])),
        ("user", fill(rng.choice(A_ANCHOR_KEEP), rng, **kw)),
    ]
    # position variation: optionally insert a filler before the foreign errand
    if rng.random() < 0.5:
        segs.insert(1, ("assistant", rng.choice(FILLER_ASSISTANT)))
    drop_idx = segs.index(("tool", empty), 0)
    keep_idx = len(segs) - 1 - [s for s in segs[::-1]].index(("tool", empty))
    elide = pick_elision(segs, {drop_idx, keep_idx}, rng)
    pid = f"v5f1pair:{gidx:06d}:surf"
    emit_record(records, gidx, segs, keep_idx, False, "tool_result",
                "empty_result_is_answer", pid, "same_surface",
                ("empty/negative tool result IS the answer the anchor asks "
                 "for (empty proves absence of %s)" % thing),
                d, rng, refs, elide=elide)
    emit_record(records, gidx, segs, drop_idx, True, "tool_result",
                "empty_result_is_answer", pid, "same_surface",
                ("same empty surface but it belongs to an explicitly closed "
                 "side errand disjoint from the anchor (closed-task moot)"),
                d, rng, refs, elide=elide)


def emit_a_pair_anchor_flip(gidx, records, rng, inv, refmap):
    d = rng.choice(list(DOMAINS))
    D = DOMAINS[d]
    e = rng.choice(D["entities"])
    thing = rng.choice(D["things"])
    action = rng.choice(D["actions"])
    surface = pick_empty(rng, inv)
    refs = ref_for(refmap, "empty", surface)
    empty = fill_empty(surface, rng)
    kw = {"E": e, "THING": thing, "ACTION": action}
    head = [
        ("system", SYSTEM_TEXT),
        ("user", fill(rng.choice(A_SETUP), rng, **kw)),
        ("assistant", rng.choice(FILLER_ASSISTANT)),
        ("assistant", builder.serialized({"tool_use": {
            "name": "shell",
            "input": {"command": f"{rng.choice(CHECK_VERBS)} {thing} {e}"}}})),
        ("tool", empty),                              # idx 4 = candidate
        ("assistant", rng.choice([
            "The check returned empty.", "结果是空的。"])),
    ]
    conv_k = head + [("user", fill(rng.choice(A_ANCHOR_KEEP), rng, **kw))]
    conv_d = head + [("user", fill(rng.choice(A_ANCHOR_DROP), rng, **kw))]
    elide = shared_elision(conv_k, rng, 4)
    pid = f"v5f1pair:{gidx:06d}:flip"
    emit_record(records, gidx, conv_k, 4, False, "tool_result",
                "empty_result_is_answer", pid, "anchor_flip",
                ("anchor makes the empty result the operative evidence "
                 "(%s absent -> proceed)" % thing),
                d, rng, refs, elide=elide)
    emit_record(records, gidx, conv_d, 4, True, "tool_result",
                "empty_result_is_answer", pid, "anchor_flip",
                ("identical candidate, anchor explicitly abandons the check "
                 "and switches to a foreign task -> certainly irrelevant"),
                d, rng, refs, elide=elide)


def emit_a_solo_keep(gidx, records, rng, inv, refmap):
    """Unpaired keep: file-not-found / no-matches as the required answer."""
    d = rng.choice(list(DOMAINS))
    D = DOMAINS[d]
    e = rng.choice(D["entities"])
    thing = rng.choice(D["things"])
    action = rng.choice(D["actions"])
    surface = pick_empty(rng, inv)
    refs = ref_for(refmap, "empty", surface)
    empty = fill_empty(surface, rng)
    kw = {"E": e, "THING": thing, "ACTION": action}
    segs = [
        ("system", SYSTEM_TEXT),
        ("user", fill(rng.choice(A_SETUP), rng, **kw)),
        ("assistant", rng.choice(FILLER_ASSISTANT)),
        ("assistant", builder.serialized({"tool_use": {
            "name": "shell",
            "input": {"command": f"{rng.choice(CHECK_VERBS)} {thing} {e}"}}})),
        ("tool", empty),
        ("assistant", rng.choice([
            "Empty — that settles it.", "空结果就是结论。"])),
        ("user", fill(rng.choice(A_ANCHOR_KEEP), rng, **kw)),
    ]
    emit_record(records, gidx, segs, 4, False, "tool_result",
                "empty_result_is_answer", None, None,
                ("empty/negative result is the sole evidence for the "
                 "anchor's absence-claim"),
                d, rng, refs)


def emit_b_pair_same_surface(gidx, records, rng, inv, refmap):
    """Two identical correction-shaped assistant segments in one
    conversation: the on-task correction is keep; the byte-identical one
    about an explicitly closed earlier task is drop (same surface form,
    opposite label)."""
    d = rng.choice(list(DOMAINS))
    D = DOMAINS[d]
    f = rng.choice(D["files"])
    kw = {"F": f}
    corr = rng.choice(inv["correction"] or CORRECTION_FALLBACK)
    refs = ref_for(refmap, "correction", corr)
    ftask = rng.choice(D["ftasks"])
    segs = [
        ("system", SYSTEM_TEXT),
        ("user", f"First, <FTASK>.".replace("<FTASK>", ftask)),
        ("assistant", fill(corr, rng, **kw)),        # idx 2 correction, task0
        ("assistant", "Done with that part."),
        ("user", fill(rng.choice([
            "好，那件事结了。现在：<TASK>",
            "That one's closed. Now: <TASK>"]), rng,
            **kw, TASK=fill(rng.choice(B_SETUP), rng, **kw))),
        ("assistant", "On it."),
        ("assistant", builder.serialized({"tool_use": {
            "name": "read", "input": {"file_path": f}}})),
        ("tool", f"     1\t{PARAMS[0]}: {rng.randrange(100, 900)}"),
        ("assistant", fill(corr, rng, **kw)),        # idx 8 = KEEP correction
        ("assistant", builder.serialized({"tool_use": {
            "name": "edit", "input": {"file_path": f}}})),
        ("tool", "Edit applied."),
        ("assistant", "Applied the corrected value."),
        ("user", fill(rng.choice(B_ANCHOR_KEEP), rng, **kw)),
    ]
    drop_idx, keep_idx = 2, 8
    elide = pick_elision(segs, {drop_idx, keep_idx}, rng)
    pid = f"v5f1pair:{gidx:06d}:surf"
    emit_record(records, gidx, segs, keep_idx, False, "assistant_text",
                "correction_keep", pid, "same_surface",
                "self-correction segment corrects the fact the anchor verifies",
                d, rng, refs, elide=elide)
    emit_record(records, gidx, segs, drop_idx, True, "assistant_text",
                "correction_keep", pid, "same_surface",
                ("correction-shaped segment but it belongs to the explicitly "
                 "closed earlier task -> certainly irrelevant"),
                d, rng, refs, elide=elide)


def emit_b_pair_anchor_flip(gidx, records, rng, inv, refmap):
    d = rng.choice(list(DOMAINS))
    D = DOMAINS[d]
    f = rng.choice(D["files"])
    corr = rng.choice(inv["correction"] or CORRECTION_FALLBACK)
    refs = ref_for(refmap, "correction", corr)
    kw = {"F": f}
    head = [
        ("system", SYSTEM_TEXT),
        ("user", fill(rng.choice(B_SETUP), rng, **kw)),
        ("assistant", "On it."),
        ("assistant", builder.serialized({"tool_use": {
            "name": "read", "input": {"file_path": f}}})),
        ("tool", f"     1\t{PARAMS[0]}: {rng.randrange(100, 900)}"),
        ("assistant", fill(corr, rng, **kw)),        # idx 5 = candidate
        ("assistant", builder.serialized({"tool_use": {
            "name": "edit", "input": {"file_path": f}}})),
        ("tool", "Edit applied."),
    ]
    conv_k = head + [("user", fill(rng.choice(B_ANCHOR_KEEP), rng, **kw))]
    conv_d = head + [("user", fill(rng.choice(B_ANCHOR_DROP), rng, **kw))]
    elide = shared_elision(conv_k, rng, 5)
    pid = f"v5f1pair:{gidx:06d}:flip"
    emit_record(records, gidx, conv_k, 5, False, "assistant_text",
                "correction_keep", pid, "anchor_flip",
                "correction is the fact the anchor asks to verify",
                d, rng, refs, elide=elide)
    emit_record(records, gidx, conv_d, 5, True, "assistant_text",
                "correction_keep", pid, "anchor_flip",
                ("identical correction, but anchor closed the config task "
                 "and moved to a foreign task -> certainly irrelevant"),
                d, rng, refs, elide=elide)


def emit_b_solo_keep(gidx, records, rng, inv, refmap):
    d = rng.choice(list(DOMAINS))
    D = DOMAINS[d]
    f = rng.choice(D["files"])
    corr = rng.choice(inv["correction"] or CORRECTION_FALLBACK)
    refs = ref_for(refmap, "correction", corr)
    kw = {"F": f}
    segs = [
        ("system", SYSTEM_TEXT),
        ("user", fill(rng.choice(B_SETUP), rng, **kw)),
        ("assistant", rng.choice(FILLER_ASSISTANT)),
        ("assistant", builder.serialized({"tool_use": {
            "name": "read", "input": {"file_path": f}}})),
        ("tool", f"     1\t{PARAMS[0]}: {rng.randrange(100, 900)}"),
        ("assistant", fill(corr, rng, **kw)),
        ("assistant", builder.serialized({"tool_use": {
            "name": "edit", "input": {"file_path": f}}})),
        ("tool", "Edit applied."),
        ("user", fill(rng.choice(B_ANCHOR_KEEP), rng, **kw)),
    ]
    emit_record(records, gidx, segs, 5, False, "assistant_text",
                "correction_keep", None, None,
                "mid-task correction/retraction is required evidence",
                d, rng, refs)


def emit_c_pair_anchor_flip(gidx, records, rng, inv, refmap):
    """Constraint micro-turn: keep under the task it binds, drop after the
    task is explicitly closed and the anchor is foreign."""
    d = rng.choice(list(DOMAINS))
    D = DOMAINS[d]
    f = rng.choice(D["files"])
    turn = rng.choice(inv["constraint"] or CONSTRAINT_FALLBACK)
    refs = ref_for(refmap, "constraint", turn)
    kw = {"F": f, "FILE": f, "DIR": rng.choice(DIRS)}
    head = [
        ("system", SYSTEM_TEXT),
        ("user", fill(rng.choice(C_SETUP), rng, **kw)),
        ("assistant", "Planning the change."),
        ("user", fill(turn, rng, **kw)),            # idx 3 = candidate
        ("assistant", "Understood — staying inside that limit."),
        ("assistant", builder.serialized({"tool_use": {
            "name": "edit", "input": {"file_path": f}}})),
        ("tool", "Edit applied."),
    ]
    conv_k = head + [("user", fill(rng.choice(C_ANCHOR_KEEP), rng, **kw))]
    conv_d = head + [("user", fill(rng.choice(C_ANCHOR_DROP), rng, **kw))]
    elide = shared_elision(conv_k, rng, 3)
    pid = f"v5f1pair:{gidx:06d}:flip"
    emit_record(records, gidx, conv_k, 3, False, "user_turn",
                "short_ack_with_constraint", pid, "anchor_flip",
                "micro user turn carries a binding constraint for the anchor task",
                d, rng, refs, elide=elide)
    emit_record(records, gidx, conv_d, 3, True, "user_turn",
                "short_ack_with_constraint", pid, "anchor_flip",
                ("identical micro turn, but the task it constrained is "
                 "explicitly closed and the anchor is foreign -> drop"),
                d, rng, refs, elide=elide)


def emit_c_solo_keep(gidx, records, rng, inv, refmap):
    d = rng.choice(list(DOMAINS))
    D = DOMAINS[d]
    f = rng.choice(D["files"])
    turn = rng.choice(inv["constraint"] or CONSTRAINT_FALLBACK)
    refs = ref_for(refmap, "constraint", turn)
    kw = {"F": f, "FILE": f, "DIR": rng.choice(DIRS)}
    segs = [
        ("system", SYSTEM_TEXT),
        ("user", fill(rng.choice(C_SETUP), rng, **kw)),
        ("assistant", "Planning."),
        ("user", fill(turn, rng, **kw)),
        ("assistant", "Got it — honoring that constraint."),
        ("assistant", builder.serialized({"tool_use": {
            "name": "edit", "input": {"file_path": f}}})),
        ("tool", "Edit applied within the constraint."),
        ("user", fill(rng.choice(C_ANCHOR_KEEP), rng, **kw)),
    ]
    emit_record(records, gidx, segs, 3, False, "user_turn",
                "short_ack_with_constraint", None, None,
                "short user turn carries a constraint the anchor relies on",
                d, rng, refs)


def emit_c_solo_drop(gidx, records, rng, inv, refmap):
    """Pure-ack micro user turn: low content AND no constraint -> drop.
    Prevents the reverse shortcut (short user turn -> keep)."""
    d = rng.choice(list(DOMAINS))
    D = DOMAINS[d]
    f = rng.choice(D["files"])
    ack = rng.choice(inv["ack"] or ACK_FALLBACK)
    refs = ref_for(refmap, "ack", ack)
    kw = {"F": f, "FILE": f}
    segs = [
        ("system", SYSTEM_TEXT),
        ("user", fill(rng.choice(C_SETUP), rng, **kw)),
        ("assistant", "Planning the change."),
        ("user", ack),                               # idx 3 = DROP candidate
        ("assistant", builder.serialized({"tool_use": {
            "name": "edit", "input": {"file_path": f}}})),
        ("tool", "Edit applied."),
        ("user", fill(rng.choice(C_ANCHOR_KEEP), rng, **kw)),
    ]
    emit_record(records, gidx, segs, 3, True, "user_turn",
                "short_ack_with_constraint", None, None,
                ("pure acknowledgement micro-turn carries no constraint and "
                 "no evidence -> certainly irrelevant"),
                d, rng, refs)


# ---------------------------------------------------------------------------
# self-check (fail hard before writing anything)
# ---------------------------------------------------------------------------

def argmax_label(record):
    p = record["targets"]["irrelevant"]["probabilities"]
    return "true" if p["true"] >= p["false"] else "false"


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
        if not rid.startswith("v5f1:"):
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
        elif kinds == {"same_surface"}:
            cand_texts = set()
            for m in members:
                st = json.loads(m["request"]["state"])
                cp = st["candidate_pointer"]
                cand_texts.add(next(
                    s["content"] for s in st["conversation"]
                    if s["pointer"] == cp))
            if len(cand_texts) != 1:
                errors.append(f"{pid}: same_surface pair with different text")
        elif kinds == {"anchor_flip"}:
            if len({m["meta"]["candidate_pointer"] for m in members}) != 1:
                errors.append(f"{pid}: flip pair different pointers")
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
    assert instr_sha == CANONICAL_INSTR_SHA, \
        f"instructions drifted: {instr_sha}"
    global INSTRUCTIONS
    INSTRUCTIONS = instructions

    inv, refmap, harvest_stats = harvest_pool()

    ref_requests, ref_states = miner.load_reference_hashes()
    records = []
    skips = Counter()
    gidx = 0
    counters = {"same_surface": PLAN["empty_result_is_answer"]["pairs_same_surface"],
                "a_flip": PLAN["empty_result_is_answer"]["pairs_anchor_flip"],
                "a_solo": PLAN["empty_result_is_answer"]["solo_keep"],
                "b_surf": PLAN["correction_keep"]["pairs_same_surface"],
                "b_flip": PLAN["correction_keep"]["pairs_anchor_flip"],
                "b_solo": PLAN["correction_keep"]["solo_keep"],
                "c_flip": PLAN["short_ack_with_constraint"]["pairs_anchor_flip"],
                "c_solo_k": PLAN["short_ack_with_constraint"]["solo_keep"],
                "c_solo_d": PLAN["short_ack_with_constraint"]["solo_drop"]}
    # deterministic interleave: iterate emitters round-robin until exhausted
    emitters = [
        ("a_surf", counters["same_surface"],
         lambda g: emit_a_pair_same_surface(g, records, rng, inv, refmap)),
        ("a_flip", counters["a_flip"],
         lambda g: emit_a_pair_anchor_flip(g, records, rng, inv, refmap)),
        ("a_solo", counters["a_solo"],
         lambda g: emit_a_solo_keep(g, records, rng, inv, refmap)),
        ("b_surf", counters["b_surf"],
         lambda g: emit_b_pair_same_surface(g, records, rng, inv, refmap)),
        ("b_flip", counters["b_flip"],
         lambda g: emit_b_pair_anchor_flip(g, records, rng, inv, refmap)),
        ("b_solo", counters["b_solo"],
         lambda g: emit_b_solo_keep(g, records, rng, inv, refmap)),
        ("c_flip", counters["c_flip"],
         lambda g: emit_c_pair_anchor_flip(g, records, rng, inv, refmap)),
        ("c_solo_k", counters["c_solo_k"],
         lambda g: emit_c_solo_keep(g, records, rng, inv, refmap)),
        ("c_solo_d", counters["c_solo_d"],
         lambda g: emit_c_solo_drop(g, records, rng, inv, refmap)),
    ]
    remaining = {name: n for name, n, _ in emitters}
    order = [name for name, _, _ in emitters]
    while any(remaining.values()):
        for name in order:
            if remaining[name] <= 0:
                continue
            fn = next(f for n2, _c, f in emitters if n2 == name)
            fn(gidx)
            gidx += 1
            remaining[name] -= 1

    # unique record_ids (emitters share prefixes)
    seen_ids = set()
    for i, r in enumerate(records):
        base = r["meta"]["record_id"]
        rid = f"{base}:{i:05d}"
        r["meta"]["record_id"] = rid
        seen_ids.add(rid)

    # dedup vs prior corpora + intra-set
    dedup = Counter()
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
    records = kept

    check = self_check(records)

    label_counts = Counter(argmax_label(r) for r in records)
    sub_counts = Counter(r["meta"]["sub_family"] for r in records)
    sub_labels = Counter((r["meta"]["sub_family"], argmax_label(r))
                         for r in records)
    domain_counts = Counter(r["meta"]["domain"] for r in records)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(builder.serialized(r) + "\n")

    file_bytes = out_jsonl.stat().st_size
    file_sha = builder.sha256_file(out_jsonl)
    section = {
        "builder": "scripts/build_v5_antishortcut_v1.py",
        "file": str(out_jsonl.relative_to(ROOT)),
        "records": len(records),
        "sha256": file_sha,
        "bytes": file_bytes,
        "labels": dict(sorted(label_counts.items())),
        "keep_drop_ratio": round(
            label_counts["false"] / max(1, label_counts["true"]), 3),
        "sub_families": dict(sorted(sub_counts.items())),
        "sub_family_labels": {f"{k[0]}:{k[1]}": v
                              for k, v in sorted(sub_labels.items())},
        "domains": dict(sorted(domain_counts.items())),
        "pairs": len({r["meta"]["pair_id"] for r in records
                      if r["meta"]["pair_id"]}),
        "paired_records": sum(1 for r in records if r["meta"]["pair_id"]),
        "hard_negative": sum(1 for r in records
                             if r["meta"]["hard_negative"]),
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
        "note": ("surface strings harvested from the v5 mining candidate "
                 "pool (train-pool transcripts only; eval-v1 + v5-eval "
                 "reserved never enter it); every string re-scanned"),
        "counts": {k: len(v) for k, v in inv.items()},
        "stats": harvest_stats,
    }
    report.setdefault("dedup", {})[FAMILY] = {
        "reference_sets": ["data/valen_nano_v2", "data/valen_nano_v3",
                           "data/valen_nano_v4",
                           "data/real_context_eval_v1/candidates*.jsonl"],
        **dict(sorted(dedup.items())),
    }
    report.setdefault("caveats", {})
    report["caveats"][FAMILY] = [
        "synthetic skeletons x harvested real surface strings; the harvest "
        "pool is data/v5_mining/f2_hard_negative_candidates.jsonl which "
        "already excludes eval-v1 and v5-eval-reserved transcripts",
        "labels are rule-derived per the v4 contract (label = pure function "
        "of state content); same_surface pairs share byte-identical "
        "candidate text and differ only in candidate_pointer",
        "overall keep:drop ~= 2:1 by design (keep-side protection family)",
    ]
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print(f"wrote {len(records)} records -> {out_jsonl}")
    print("labels:", dict(sorted(label_counts.items())),
          "| sub:", dict(sorted(sub_counts.items())))
    print("domains:", dict(sorted(domain_counts.items())),
          "| pairs:", section["pairs"], "| dedup:", dict(sorted(dedup.items())))
    print(f"report -> {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
