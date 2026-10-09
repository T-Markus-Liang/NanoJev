#!/usr/bin/env python3
"""Generate data/context_relevance_v4_seed<N>/ — synthetic-harder eval families.

Per docs/REAL_CONTEXT_EVAL_V1.md §3.2: a new generator namespace aimed at the
failure modes the v3 template space cannot express. Records are emitted
DIRECTLY in the valen eval.jsonl schema (byte-compatible with
data/valen_nano_v3/eval.jsonl) — no intermediate source format, no builder
step — under a fresh lineage:

  - output root:  data/context_relevance_v4_seed<seed>/  (seed suffix keeps
    the namespace disjoint from context_relevance_v1/v2/v3 outputs);
  - group_id:     "context_relevance_v4:<sha256 lineage digest>" — never
    reuses v1/v2/v3/oracle group ids (checked against valen_nano_v3 when
    present);
  - splits:       v4 is the consolidated next-generation dataset (owner
    directive — there is no separate v5), so the raw emit now carries all
    three splits. The eval pool (gidx < GROUPS_PER_FAMILY) keeps the
    original assignment — sha256(group_id) mod 100 < 12 -> dev, else eval —
    so eval/dev rows are byte-identical to the first emit. Train groups are
    drawn from a DISJOINT gidx range ([TRAIN_GIDX_OFFSET, ...)); different
    digest inputs -> different group_ids, so train/eval/dev group sets are
    disjoint by construction and the pairwise disjointness is asserted in
    code (equivalent disjoint scheme to "mod<8 dev, mod<25 eval, else
    train"; a single mod-100 rule cannot reproduce the frozen eval=1301 /
    dev=199 counts).
  - consolidated: build_consolidated() merges the raw emit with the frozen
    valen_nano_v3 rows into data/valen_nano_v4/{train,eval,dev}.jsonl —
    train = v3 train + v4 train (training_allowed=true), eval = v3 eval +
    v4 eval, dev = v4 dev. v3 rows are copied byte-for-byte (raw lines).
  - instructions: questions.irrelevant.noul copied verbatim from the v3 eval
    records (same string as gen_context_relevance_v3_v1.INSTRUCTIONS).

Families (§3.2 bullet list), each emitting a fixed candidate-kind roster with
a CONSTANT label per kind:

  tool_pairing            — multi-call tool histories; a stale single tool
                            result is droppable ONLY if a resolved successor
                            exists in the state; a stale result with a lone
                            pending successor (or none) must be kept.
  cross_pointer_dependency— early assistant value that is sole evidence for a
                            late user question (keep), versus the same pointer
                            made redundant by a later confirmation or
                            superseded by a later correction (drop).
  long_context_dilution   — 30..120 segment conversations; one authoritative
                            fact (keep) among near-duplicate "outdated cache"
                            distractors (drop) and fillers (drop).
  topic_shift             — mid-conversation task change; pre-shift segments
                            are irrelevant EXCEPT declared carry-over
                            constraints/corrections; task-scoped declarations
                            marked "only" do NOT carry (hard negative).
  adversarial_restatement — extends near_duplicate_evidence: an exact
                            restatement of the operative value drops, while a
                            restatement differing by one token (word-swap or
                            off-by-one) is a potential correction -> keep.

Label contract (same discipline as the v2/v3 contract): the label is a pure
function of state content. Structurally this is guaranteed by two rules:
(1) every candidate_kind has a constant label, and (2) any two kinds with
different labels always differ in non-digit surface tokens or in message
structure/count, so validate_valen_label_consistency_v1.normalized_state
(digits masked) can never map opposite labels onto one normalized state. The
generator runs the official validator on its own output and asserts 0
violations; it additionally checks dev.jsonl rows (the official check covers
train/eval only).

Usage: python3 scripts/gen_context_relevance_v4_v1.py [--seed N] [--scale F]
       [--out DIR]
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from validate_valen_label_consistency_v1 import check_dir, normalized_state  # noqa: E402

# Seeds already used: v1/v2/oracle = 20260919, v3 = 20261005. v4 must differ.
DEFAULT_SEED = 20261115
SCHEMA = "nanojev-context-relevance-v4"
SOURCE_DATASET = "context_relevance_v4"
LINEAGE = "context_relevance_v4"
DEV_LT = 12  # sha256(group_id) mod 100 < DEV_LT -> dev

# Verbatim from data/valen_nano_v3/eval.jsonl records with
# source_dataset == "context_relevance_v3" (identical to the v3 generator's
# INSTRUCTIONS constant). Do not edit.
INSTRUCTIONS = ("Is the candidate context certainly irrelevant to fulfilling the "
                "current user request? Answer false when it is required evidence, "
                "a user constraint, a correction, a tool dependency, or when "
                "uncertainty remains.")
SYSTEM_TEXT = "Preserve current user constraints and required evidence."

FAMILIES = ("tool_pairing", "cross_pointer_dependency", "long_context_dilution",
            "topic_shift", "adversarial_restatement")

# Groups per family at scale=1.0 -> 5 + 6 + 4 + 5 + 5 candidates/group
# = 275 + 300 + 300 + 300 + 325 = ~1500 records total.
GROUPS_PER_FAMILY = {
    "tool_pairing": 55,
    "cross_pointer_dependency": 50,
    "long_context_dilution": 75,
    "topic_shift": 60,
    "adversarial_restatement": 65,
}

# Train pool: disjoint gidx range [TRAIN_GIDX_OFFSET, ...) so train group
# digests can never collide with the eval pool. Sizes chosen for ~1000 train
# records per family (kinds emitted per group: 5/6/4/5/5) -> ~5002 total.
TRAIN_GROUPS_PER_FAMILY = {
    "tool_pairing": 200,
    "cross_pointer_dependency": 167,
    "long_context_dilution": 250,
    "topic_shift": 200,
    "adversarial_restatement": 200,
}
TRAIN_GIDX_OFFSET = 1_000_000

VALEN_V3_DIR = ROOT / "data" / "valen_nano_v3"
VALEN_V4_DIR = ROOT / "data" / "valen_nano_v4"

# Constant label per kind: True = certainly irrelevant = drop.
KIND_LABEL = {
    # tool_pairing
    "tool_stale_resolved": True,      # stale result; resolved successor exists
    "tool_stale_pending": False,      # stale result; successor only pending
    "tool_resolved_result": False,    # the resolved successor itself
    "tool_pending_successor": False,  # in-flight re-check: tool dependency
    "tool_unrelated": True,           # resolved result for another entity
    # cross_pointer_dependency
    "xptr_sole_evidence": False,      # early value, sole evidence for late ask
    "xptr_restated_stale": True,      # same early value, later confirmation in state
    "xptr_corrected_stale": True,     # early value superseded by later correction
    "xptr_confirmation": False,       # the confirming restatement (operative)
    "xptr_correction": False,         # the correction segment (operative)
    "xptr_filler": True,              # mid-conversation filler
    # long_context_dilution
    "dilution_required_fact": False,  # the single authoritative fact
    "dilution_near_dup": True,        # outdated-cache distractor, true value in state
    "dilution_filler": True,          # unrelated filler segment
    # topic_shift
    "shift_moot_preshift": True,      # pre-shift content about the closed task
    "shift_carryover": False,         # declared standing constraint (carries)
    "shift_taskscoped": True,         # declared "for this task only" constraint
    "shift_postshift_evidence": False,
    "shift_filler": True,
    # adversarial_restatement
    "adv_operative": False,           # the operative value statement
    "adv_exact_restatement": True,    # verbatim-value restatement: redundant
    "adv_one_token_off": False,       # restatement differing by one token
    "adv_off_value_estimate": False,  # tool estimate differing from operative
    "adv_filler": True,
}
KINDS = tuple(KIND_LABEL)

HARD_KINDS = frozenset({
    "tool_stale_resolved", "tool_stale_pending", "tool_pending_successor",
    "xptr_sole_evidence", "xptr_restated_stale", "xptr_corrected_stale",
    "dilution_required_fact", "dilution_near_dup",
    "shift_carryover", "shift_taskscoped",
    "adv_exact_restatement", "adv_one_token_off", "adv_off_value_estimate",
})

DOMAINS = {
    "tool_pairing": ("code", "robotics"),
    "cross_pointer_dependency": ("support", "order", "code"),
    "long_context_dilution": ("code", "risk", "support"),
    "topic_shift": ("support", "order", "robotics"),
    "adversarial_restatement": ("code", "order", "risk"),
}

CODENAMES = ("atlas", "birch", "cinder", "delta", "ember", "fjord", "garnet",
             "harbor", "iris", "juniper", "kestrel", "lumen", "mica", "north",
             "onyx", "pico", "quartz", "rowan", "sable", "tarn", "umber",
             "vesper", "willow", "xenia")
FIELDS = ("refresh_interval", "batch_size", "deploy_window", "rate_limit",
          "cache_ttl", "max_retries", "queue_depth", "timeout_budget")
TOOLS = ("check_config", "read_policy", "probe_endpoint", "fetch_quote",
         "inspect_route")
WORD_PAIRS = (("phase two", "phase three"), ("tier gold", "tier silver"),
              ("tier gold", "tier bronze"), ("Friday", "Thursday"),
              ("Monday", "Saturday"), ("blue channel", "green channel"),
              ("room alpha", "room beta"), ("mode strict", "mode relaxed"))
FILLERS = (
    ("assistant", "Routine heartbeat log: all workers nominal."),
    ("assistant", "A retired benchmark note lists unrelated scores."),
    ("assistant", "The previous retro document covers process only."),
    ("user", "Noted, please continue."),
    ("assistant", "A cached stylesheet reference is stored for later."),
    ("assistant", "Log rotation completed overnight without errors."),
    ("assistant", "An archived meeting transcript mentions parking logistics."),
    ("user", "Thanks — carry on with the audit."),
)
ZH_FILLERS = (
    ("assistant", "例行心跳日志：所有节点正常，与当前任务无关。"),
    ("assistant", "归档的周报只包含流程记录，不包含配置值。"),
)


def serialized(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def seg(i):
    return f"/messages/{i}/content"


def msg(pointer, role, content):
    return {"pointer": pointer, "role": role, "content": content}


def group_digest(seed, family, gidx):
    return hashlib.sha256(
        f"context_relevance_v4:{seed}:{family}:{gidx}".encode()).hexdigest()


# ---------------------------------------------------------------------------
# Family emitters.
#
# Each returns a list of dicts:
#   {"kind": str, "conversation": [msg...], "candidate_pointer": str}
# Labels are looked up from KIND_LABEL (constant per kind).
# ---------------------------------------------------------------------------

def emit_tool_pairing(p, rng):
    e, e2, tool = p["e"], p["e2"], p["tool"]
    head = [
        msg("/system", "system",
            SYSTEM_TEXT + " A tool result marked resolved supersedes earlier "
            "attempts for the same call; pending results do not resolve them."),
        msg(seg(0), "user",
            f"Configure service {e} using the verified {tool} reading; a stale "
            "attempt may be ignored only when a resolved re-verification exists."),
        msg(seg(1), "assistant",
            f"I will call {tool} for {e} and re-verify the reading before applying it."),
        msg(seg(2), "tool",
            serialized({"tool": tool, "entity": e, "attempt": "first",
                        "status": "stale", "value": p["old"]})),
        msg(seg(3), "assistant",
            f"The first {tool} reading for {e} looked inconsistent; re-running the check."),
    ]
    unrelated = msg(seg(5), "tool",
                    serialized({"tool": "list_backups", "entity": e2,
                                "status": "resolved", "count": p["n"]}))
    tail = msg(seg(6), "user", f"Apply the verified {tool} value for {e} now.")
    conv_resolved = head + [
        msg(seg(4), "tool",
            serialized({"tool": tool, "entity": e, "attempt": "re-verified",
                        "status": "resolved", "value": p["new"]})),
        unrelated, tail]
    conv_pending = head + [
        msg(seg(4), "tool",
            serialized({"tool": tool, "entity": e, "attempt": "re-run",
                        "status": "pending"})),
        unrelated, tail]
    return [
        {"kind": "tool_stale_resolved", "conversation": conv_resolved,
         "candidate_pointer": seg(2)},
        {"kind": "tool_resolved_result", "conversation": conv_resolved,
         "candidate_pointer": seg(4)},
        {"kind": "tool_unrelated", "conversation": conv_resolved,
         "candidate_pointer": seg(5)},
        {"kind": "tool_stale_pending", "conversation": conv_pending,
         "candidate_pointer": seg(2)},
        {"kind": "tool_pending_successor", "conversation": conv_pending,
         "candidate_pointer": seg(4)},
    ]


def emit_cross_pointer(p, rng):
    e, f = p["e"], p["f"]
    v1, v2 = p["v1"], p["v2"]
    head = [
        msg("/system", "system", SYSTEM_TEXT),
        msg(seg(0), "user",
            f"Start work on project {e}: record the {f} we agree on, then continue setup."),
        msg(seg(1), "assistant",
            f"Setup note: project {e} uses {f} = {v1}, recorded at intake."),
        msg(seg(2), "assistant",
            f"Workspace scaffolding for {e} is complete; caches are warm."),
    ]
    late_q_a = msg(seg(4), "user",
                   f"Quick check: what was the {f} we recorded for project {e}?")
    late_q_b = msg(seg(4), "user",
                   f"Quick check: what is the current {f} for project {e}?")
    conv_sole = head + [
        msg(seg(3), "assistant",
            f"Documentation draft for {e} covers naming only."),
        late_q_a]
    conv_restated = head + [
        msg(seg(3), "assistant",
            f"Confirmed record: project {e} {f} = {v1}, re-verified just now."),
        late_q_a]
    conv_corrected = head + [
        msg(seg(3), "assistant",
            f"Correction: the {f} for project {e} is now {v2}, superseding the intake note."),
        late_q_b]
    return [
        {"kind": "xptr_sole_evidence", "conversation": conv_sole,
         "candidate_pointer": seg(1)},
        {"kind": "xptr_filler", "conversation": conv_sole,
         "candidate_pointer": seg(3)},
        {"kind": "xptr_restated_stale", "conversation": conv_restated,
         "candidate_pointer": seg(1)},
        {"kind": "xptr_confirmation", "conversation": conv_restated,
         "candidate_pointer": seg(3)},
        {"kind": "xptr_corrected_stale", "conversation": conv_corrected,
         "candidate_pointer": seg(1)},
        {"kind": "xptr_correction", "conversation": conv_corrected,
         "candidate_pointer": seg(3)},
    ]


def emit_dilution(p, rng):
    e, f, v = p["e"], p["f"], p["v"]
    n_seg = p["n_seg"]
    pool = list(FILLERS) + (list(ZH_FILLERS) if p["zh"] else [])
    fact_idx = rng.randrange(3, n_seg - 3)
    distractors = set()
    while len(distractors) < p["n_dist"]:
        pos = rng.randrange(1, n_seg - 1)
        if pos != fact_idx:
            distractors.add(pos)
    conv = [
        msg("/system", "system", SYSTEM_TEXT),
        msg(seg(0), "user",
            f"Audit workspace {e}: many segments follow; apply the "
            "authoritative configuration only."),
    ]
    for i in range(1, n_seg - 1):
        if i == fact_idx:
            conv.append(msg(seg(i), "assistant",
                            f"authoritative configuration record: {e} {f} = {v}"))
        elif i in distractors:
            conv.append(msg(seg(i), "assistant",
                            f"outdated cache entry: {e} {f} = {rng.randrange(100, 999)}"))
        else:
            role, text = pool[rng.randrange(len(pool))]
            conv.append(msg(seg(i), role, text))
    conv.append(msg(seg(n_seg - 1), "user",
                    f"Now apply the {f} for {e} using the authoritative configuration."))
    dist_sorted = sorted(distractors)
    filler_idx = next(i for i in range(1, n_seg - 1)
                      if i != fact_idx and i not in distractors)
    return [
        {"kind": "dilution_required_fact", "conversation": conv,
         "candidate_pointer": seg(fact_idx)},
        {"kind": "dilution_near_dup", "conversation": conv,
         "candidate_pointer": seg(dist_sorted[0])},
        {"kind": "dilution_near_dup", "conversation": conv,
         "candidate_pointer": seg(dist_sorted[-1])},
        {"kind": "dilution_filler", "conversation": conv,
         "candidate_pointer": seg(filler_idx)},
    ]


def emit_topic_shift(p, rng):
    o1, o2 = p["o1"], p["o2"]
    if p["carry"]:
        constraint_kind = "shift_carryover"
        constraint_text = ("Standing constraint: regardless of task, always cite "
                           f"the {p['tag']} revision tag in the final answer.")
    else:
        constraint_kind = "shift_taskscoped"
        constraint_text = (f"Constraint for the {o1} summary only: keep it under "
                           f"{p['n']} words.")
    conv = [
        msg("/system", "system", SYSTEM_TEXT),
        msg(seg(0), "user",
            f"First task: draft the migration summary for system {o1}."),
        msg(seg(1), "assistant",
            f"Migration notes for {o1}: phase plan sketched and owners listed."),
        msg(seg(2), "user", constraint_text),
        msg(seg(3), "assistant",
            f"Archived metric for {o1}: a stale throughput reading from last quarter."),
        msg(seg(4), "user",
            f"Stop — switch tasks. The new task is the rollout checklist for "
            f"release {o2}; the {o1} summary work is closed."),
        msg(seg(5), "assistant",
            f"Rollout checklist for {o2}: owner {p['owner']}, required gate {p['gate']}."),
        msg(seg(6), "assistant",
            "The cafeteria menu for the week is posted on the intranet."),
    ]
    return [
        {"kind": "shift_moot_preshift", "conversation": conv,
         "candidate_pointer": seg(1)},
        {"kind": constraint_kind, "conversation": conv,
         "candidate_pointer": seg(2)},
        {"kind": "shift_moot_preshift", "conversation": conv,
         "candidate_pointer": seg(3)},
        {"kind": "shift_postshift_evidence", "conversation": conv,
         "candidate_pointer": seg(5)},
        {"kind": "shift_filler", "conversation": conv,
         "candidate_pointer": seg(6)},
    ]


def emit_adversarial(p, rng):
    e, f = p["e"], p["f"]
    v, voff = p["v"], p["voff"]
    conv = [
        msg("/system", "system", SYSTEM_TEXT),
        msg(seg(0), "user",
            f"Confirm the operative {f} for {e} and apply it to the plan."),
        msg(seg(1), "assistant", f"Operative {f} for {e}: {v}."),
        msg(seg(2), "assistant",
            f"Restating the confirmed record: the {e} {f} is {v}."),
        msg(seg(3), "assistant",
            f"A side note claims the {e} {f} is {voff}."),
        msg(seg(4), "tool",
            serialized({"source": "estimate", "entity": e, "field": f,
                        "estimated_value": p["est"]})),
        msg(seg(5), "assistant",
            "The parking-lot resurfacing schedule was announced yesterday."),
    ]
    return [
        {"kind": "adv_operative", "conversation": conv,
         "candidate_pointer": seg(1)},
        {"kind": "adv_exact_restatement", "conversation": conv,
         "candidate_pointer": seg(2)},
        {"kind": "adv_one_token_off", "conversation": conv,
         "candidate_pointer": seg(3)},
        {"kind": "adv_off_value_estimate", "conversation": conv,
         "candidate_pointer": seg(4)},
        {"kind": "adv_filler", "conversation": conv,
         "candidate_pointer": seg(5)},
    ]


EMITTERS = {
    "tool_pairing": emit_tool_pairing,
    "cross_pointer_dependency": emit_cross_pointer,
    "long_context_dilution": emit_dilution,
    "topic_shift": emit_topic_shift,
    "adversarial_restatement": emit_adversarial,
}


def family_params(fam, gidx, rng):
    p = {"e": f"{rng.choice(CODENAMES)}-{gidx:04d}",
         "e2": f"{rng.choice(CODENAMES)}-{gidx + 7000:04d}",
         "f": rng.choice(FIELDS),
         "domain": rng.choice(DOMAINS[fam])}
    if fam == "tool_pairing":
        p.update(tool=rng.choice(TOOLS), old=rng.randrange(100, 500),
                 new=rng.randrange(600, 900), n=rng.randrange(2, 9))
    elif fam == "cross_pointer_dependency":
        p.update(v1=rng.randrange(10, 90), v2=rng.randrange(100, 200))
    elif fam == "long_context_dilution":
        p.update(v=rng.randrange(10, 99), n_seg=rng.randrange(30, 121),
                 n_dist=rng.randrange(3, 8), zh=rng.random() < 0.2)
    elif fam == "topic_shift":
        p.update(o1=p["e"], o2=p["e2"], carry=rng.random() < 0.5,
                 tag=rng.choice(("v2", "rev-B", "signed")),
                 n=rng.randrange(80, 300),
                 owner=rng.choice(CODENAMES),
                 gate=rng.choice(("security", "legal", "perf")))
    elif fam == "adversarial_restatement":
        if rng.random() < 0.5:
            v, voff = rng.choice(WORD_PAIRS)
        else:  # numeric one-token-off flavor
            v = str(rng.randrange(100, 900))
            voff = str(int(v) + rng.choice((-1, 1)))
        p.update(v=v, voff=voff, est=rng.randrange(1000, 9999))
    return p


def has_cjk(text):
    return any("一" <= ch <= "鿿" for ch in text)


def make_record(fam, gidx, split, group_id, domain, spec):
    kind = spec["kind"]
    label = KIND_LABEL[kind]
    ptr = spec["candidate_pointer"]
    ptr_id = ptr.strip("/").replace("/", "_")
    rid = f"context_v4:{split}:{fam}:{gidx:04d}:{kind}:{ptr_id}"
    conv = spec["conversation"]
    state = {
        "conversation": conv,
        "candidate_pointer": ptr,
        "user_messages_in_order": [m["content"] for m in conv
                                   if m["role"] == "user"],
        "family": fam,
        "wire_format": "synthetic_canonical",
        "scenario_id": f"{fam}-{gidx:04d}",
    }
    meta = {
        "record_id": rid,
        "domain": domain,
        "eval_family": fam,
        "modality": "text",
        "language_bucket": "multi",
        "source_dataset": SOURCE_DATASET,
        "source_split": split,
        "candidate_kind": kind,
        "candidate_pointer": ptr,
        "hard_negative": kind in HARD_KINDS,
        "label_confidence": "high",
        "labeler": "generator_deterministic",
        "selection_method": "deterministic",
    }
    if any(has_cjk(m["content"]) for m in conv):
        meta["mixed_language"] = True
    if fam == "long_context_dilution":
        meta["n_segments"] = len(conv)
    return {
        "group_id": group_id,
        "request": {
            "state": serialized(state),
            "questions": {"irrelevant": {"type": "noul",
                                       "instructions": INSTRUCTIONS}},
        },
        "targets": {"irrelevant": {"probabilities":
                                   {"true": float(label),
                                    "false": float(not label)}}},
        "meta": meta,
    }


def argmax_label(record):
    probs = record["targets"]["irrelevant"]["probabilities"]
    return max(probs.items(), key=lambda kv: kv[1])[0]


def assert_internal_consistency(records):
    """The official validator covers train/eval; we additionally check dev so
    no normalized state in the whole output carries two labels."""
    groups = defaultdict(set)
    for r in records:
        groups[normalized_state(r["request"]["state"])].add(argmax_label(r))
    bad = {k: v for k, v in groups.items() if len(v) > 1}
    assert not bad, (f"label-collision bug: {len(bad)} normalized states carry "
                     f"opposite labels: {list(bad.items())[:3]}")
    return len(groups)


def assert_disjoint_from_v3(group_ids):
    """v4 group ids must never collide with the frozen v3 lineage."""
    v3 = ROOT / "data" / "valen_nano_v3"
    if not v3.exists():
        return
    seen = set()
    for name in ("train", "eval"):
        path = v3 / f"{name}.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    seen.add(json.loads(line)["group_id"])
    overlap = group_ids & seen
    assert not overlap, f"group_id collision with valen_nano_v3: {sorted(overlap)[:3]}"


def generate(seed=DEFAULT_SEED, scale=1.0, out=None):
    out = Path(out) if out else ROOT / "data" / f"context_relevance_v4_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    splits = {"train": [], "eval": [], "dev": []}
    all_group_ids = set()
    # Eval pool first for ALL families: identical rng draw order to the
    # original eval-only emit, so eval.jsonl/dev.jsonl stay byte-identical
    # regardless of the train pool size.
    for fam in FAMILIES:
        n_groups = max(1, round(GROUPS_PER_FAMILY[fam] * scale))
        for gidx in range(n_groups):
            p = family_params(fam, gidx, rng)
            digest = group_digest(seed, fam, gidx)
            group_id = f"{LINEAGE}:{digest}"
            all_group_ids.add(group_id)
            split = "dev" if int(digest, 16) % 100 < DEV_LT else "eval"
            for spec in EMITTERS[fam](p, rng):
                splits[split].append(
                    make_record(fam, gidx, split, group_id, p["domain"], spec))
    # Train pool: disjoint gidx range -> disjoint group_id digests.
    for fam in FAMILIES:
        n_train = max(1, round(TRAIN_GROUPS_PER_FAMILY[fam] * scale))
        for t in range(n_train):
            gidx = TRAIN_GIDX_OFFSET + t
            p = family_params(fam, gidx, rng)
            digest = group_digest(seed, fam, gidx)
            group_id = f"{LINEAGE}:{digest}"
            all_group_ids.add(group_id)
            for spec in EMITTERS[fam](p, rng):
                splits["train"].append(
                    make_record(fam, gidx, "train", group_id, p["domain"], spec))
    key = lambda r: (r["group_id"], r["meta"]["record_id"])
    for rows in splits.values():
        rows.sort(key=key)
    all_records = splits["train"] + splits["eval"] + splits["dev"]
    n_states = assert_internal_consistency(all_records)
    assert_disjoint_from_v3(all_group_ids)
    # Group-level split disjointness: train/eval/dev share no group_id.
    group_sets = {s: {r["group_id"] for r in rows} for s, rows in splits.items()}
    disjoint = {f"{a}_{b}": group_sets[a].isdisjoint(group_sets[b])
                for a in group_sets for b in group_sets if a < b}
    assert all(disjoint.values()), f"split group overlap: {disjoint}"
    for name, rows in splits.items():
        text = "".join(serialized(r) + "\n" for r in rows)
        (out / f"{name}.jsonl").write_text(text, encoding="utf-8")
    # Official validator pass on the written output (covers train+eval; the
    # internal pass above additionally covers dev).
    failures, report = check_dir(out)
    assert not failures, f"validator violations: {failures[:3]}"
    manifest = {
        "schema_version": SCHEMA,
        "builder": "scripts/gen_context_relevance_v4_v1.py",
        "output_dir": str(out),
        "seed": seed,
        "scale": scale,
        "evaluation_only": ["eval", "dev"],
        "training_allowed": {"train": True, "eval": False, "dev": False},
        "split_rule": (f"eval pool gidx < per-family pool: sha256(group_id) mod 100 "
                       f"< {DEV_LT} -> dev, else eval; train pool gidx in "
                       f"[{TRAIN_GIDX_OFFSET}, ...) all -> train (disjoint digest "
                       f"range; pairwise group disjointness asserted)"),
        "group_id": f"{LINEAGE}:<sha256('context_relevance_v4:<seed>:<family>:<gidx>')>",
        "group_disjointness": disjoint,
        "lineage_note": ("fresh namespace; disjoint seed from v1/v2/oracle "
                         "(20260919) and v3 (20261005); group ids asserted "
                         "disjoint from data/valen_nano_v3"),
        "families": list(FAMILIES),
        "kind_label": KIND_LABEL,
        "hard_negative_kinds": sorted(HARD_KINDS),
        "contract": ("label is a pure function of state content; constant label "
                     "per candidate_kind; differing-label kinds always differ in "
                     "non-digit tokens or message structure so digit-masked "
                     "normalization cannot merge them"),
        "question": {"qid": "irrelevant", "type": "noul",
                     "semantics": "true == candidate segment certainly irrelevant == drop; false == keep",
                     "instructions_preserved_verbatim": True},
        "label_consistency": {**report, "internal_normalized_states": n_states,
                              "violations": 0},
        "excluded": {
            "data/jevbench_offline_bundle_v1": "evaluation_only bundle",
            "*_test.jsonl,*_ood.jsonl": "frozen splits reserved for NanoJev gates",
            "provider/scorer outputs": "never enter labels, training, or calibration",
        },
        "splits": {},
    }
    for name, rows in splits.items():
        data = (out / f"{name}.jsonl").read_bytes()
        manifest["splits"][name] = {
            "records": len(rows),
            "groups": len({r["group_id"] for r in rows}),
            "families": dict(Counter(r["meta"]["eval_family"] for r in rows)),
            "kinds": dict(Counter(r["meta"]["candidate_kind"] for r in rows)),
            "labels": dict(Counter(argmax_label(r) for r in rows)),
            "hard_negative": sum(1 for r in rows if r["meta"]["hard_negative"]),
            "mixed_language": sum(1 for r in rows
                                  if r["meta"].get("mixed_language")),
            "training_allowed": name == "train",
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({s: manifest["splits"][s]["records"] for s in splits}))
    print(json.dumps({"label_consistency": manifest["label_consistency"]}))
    return manifest


def _read_lines(path):
    return [l for l in Path(path).read_text(encoding="utf-8").splitlines()
            if l.strip()]


def build_consolidated(raw_dir, v3_dir=VALEN_V3_DIR, out=VALEN_V4_DIR):
    """Merge the raw v4 emit with frozen valen_nano_v3 into data/valen_nano_v4.

    train.jsonl = all v3 train rows (byte-for-byte) + v4 train rows
    eval.jsonl  = all v3 eval rows (byte-for-byte) + v4 eval rows
    dev.jsonl   = v4 dev rows only
    Group-level disjointness (v4 train/eval/dev pairwise, and v4 vs v3) is
    asserted before writing; the official label-consistency validator runs on
    the merged dir and must report 0 violations.
    """
    raw_dir, v3_dir, out = Path(raw_dir), Path(v3_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    v3_lines = {s: _read_lines(v3_dir / f"{s}.jsonl") for s in ("train", "eval")}
    v4 = {s: [json.loads(l) for l in _read_lines(raw_dir / f"{s}.jsonl")]
          for s in ("train", "eval", "dev")}
    v4_groups = {s: {r["group_id"] for r in rows} for s, rows in v4.items()}
    v3_groups = {json.loads(l)["group_id"]
                 for lines in v3_lines.values() for l in lines}
    disjoint = {
        "v4_train_vs_eval": v4_groups["train"].isdisjoint(v4_groups["eval"]),
        "v4_train_vs_dev": v4_groups["train"].isdisjoint(v4_groups["dev"]),
        "v4_eval_vs_dev": v4_groups["eval"].isdisjoint(v4_groups["dev"]),
        "v4_all_vs_v3_all": set().union(*v4_groups.values()).isdisjoint(v3_groups),
    }
    assert all(disjoint.values()), f"group disjointness failed: {disjoint}"
    merged_lines = {
        "train": v3_lines["train"] + [serialized(r) for r in v4["train"]],
        "eval": v3_lines["eval"] + [serialized(r) for r in v4["eval"]],
        "dev": [serialized(r) for r in v4["dev"]],
    }
    # Deterministic order, same convention as the v3 builder: sort by
    # (group_id, record_id); v3 rows keep their original line bytes.
    def sort_key(line):
        r = json.loads(line)
        return (r["group_id"], r["meta"]["record_id"])
    for name, lines in merged_lines.items():
        text = "".join(l + "\n" for l in sorted(lines, key=sort_key))
        (out / f"{name}.jsonl").write_text(text, encoding="utf-8")
    failures, report = check_dir(out)
    assert not failures, f"merged validator violations: {failures[:3]}"
    merged_rows = {s: [json.loads(l) for l in _read_lines(out / f"{s}.jsonl")]
                   for s in merged_lines}
    manifest = {
        "schema_version": "nanojev-valen-nano-v4",
        "builder": "scripts/gen_context_relevance_v4_v1.py::build_consolidated",
        "output_dir": str(out),
        "lineage": ("v4 IS the consolidated next-generation dataset (owner "
                    "directive; no separate v5): frozen valen_nano_v3 rows + "
                    "context_relevance_v4 new-family rows"),
        "inputs": {
            "data/valen_nano_v3/train.jsonl": {
                "records": len(v3_lines["train"]),
                "sha256": hashlib.sha256(
                    (v3_dir / "train.jsonl").read_bytes()).hexdigest()},
            "data/valen_nano_v3/eval.jsonl": {
                "records": len(v3_lines["eval"]),
                "sha256": hashlib.sha256(
                    (v3_dir / "eval.jsonl").read_bytes()).hexdigest()},
            **{f"{raw_dir.name}/{s}.jsonl": {
                "records": len(v4[s]),
                "sha256": hashlib.sha256(
                    (raw_dir / f"{s}.jsonl").read_bytes()).hexdigest()}
               for s in ("train", "eval", "dev")},
        },
        "group_disjointness": {
            **disjoint,
            "v4_groups": {s: len(g) for s, g in v4_groups.items()},
            "v3_groups": len(v3_groups),
            "rule": "v4 eval pool gidx < pool size; v4 train pool gidx in "
                    "[1000000, ...); v4 lineage prefix cannot collide with "
                    "v1/v2/v3/oracle group ids",
        },
        "training_allowed": {"train": True, "eval": False, "dev": False},
        "evaluation_only": ["eval", "dev"],
        "question": {"qid": "irrelevant", "type": "noul",
                     "semantics": "true == candidate segment certainly irrelevant == drop; false == keep",
                     "instructions_preserved_verbatim": True},
        "label_consistency": {**report, "violations": 0},
        "excluded": {
            "data/jevbench_offline_bundle_v1": "bundle manifest: evaluation_only, training_allowed=false",
            "*_test.jsonl,*_ood.jsonl": "frozen splits reserved for NanoJev gates",
            "results/e2e_*,context_shadow receipts": "contain provider probabilities (never enter training)",
        },
        "splits": {},
    }
    for name, rows in merged_rows.items():
        data = (out / f"{name}.jsonl").read_bytes()
        src = Counter(r["meta"]["source_dataset"] for r in rows)
        manifest["splits"][name] = {
            "records": len(rows),
            "groups": len({r["group_id"] for r in rows}),
            "per_source": dict(src),
            "labels": dict(Counter(argmax_label(r) for r in rows)),
            "hard_negative": sum(1 for r in rows
                                 if r["meta"].get("hard_negative")),
            "families_v4": dict(Counter(r["meta"].get("eval_family")
                                        for r in rows
                                        if r["meta"].get("eval_family"))),
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({s: manifest["splits"][s]["records"]
                      for s in manifest["splits"]}))
    print(json.dumps({"merged_label_consistency": manifest["label_consistency"],
                      "group_disjointness": disjoint}))
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiplier on default groups-per-family")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: data/context_relevance_v4_seed<seed>")
    ap.add_argument("--raw-only", action="store_true",
                    help="skip the consolidated valen_nano_v4 merge")
    ap.add_argument("--v3", type=Path, default=VALEN_V3_DIR)
    ap.add_argument("--consolidated-out", type=Path, default=VALEN_V4_DIR)
    args = ap.parse_args()
    raw = Path(args.out) if args.out else \
        ROOT / "data" / f"context_relevance_v4_seed{args.seed}"
    generate(seed=args.seed, scale=args.scale, out=raw)
    if not args.raw_only:
        build_consolidated(raw, v3_dir=args.v3, out=args.consolidated_out)


if __name__ == "__main__":
    main()
