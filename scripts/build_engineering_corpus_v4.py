#!/usr/bin/env python3
"""Engineering-judgment corpus V4: roadmap task X4 scale-out.

Why this file exists
--------------------

X2 (``docs/NANOJEV_V2_ROADMAP.md``) concluded that the readout is not the
binding constraint — corpus scale and diversity are.  V4 therefore rebuilds the
catalog at >2x the V3 scale while keeping every V3 safety property, and widens
family coverage along the failure axes observed in the J-D diagnosis:

* **36 distinct families.**  The 18 V3 family *rules* (domains + frozen
  oracles) are reused as code — imported, not copied — and re-rendered under a
  new render namespace, seed and id namespace, so every V4 state is a fresh
  surface that shares no canonical model-visible input with V3.  18 additional
  families are declared here: multi-entity state binding (per-slot /
  per-plan / per-window facts the label must bind to the right entity),
  scheduling/resource-allocation/code-review-flavoured domains, rule-compliance
  cascades, negation-phrased boolean questions, high-cardinality choice
  (8/32/255 candidates — 255 exercises the full served-contract range), and two
  *analytic-chance* families whose expected answer is a conditional
  distribution (``programmatic_conditional_distribution``), matching the
  ``known_chance`` cohort mechanics of the workflow corpus without reusing any
  of its content.  ("known_chance" itself is a V1 reserved evaluation token and
  never appears in emitted content.)
* **Programmatic contrastive pairs (unchanged semantics).**  Each family
  declares an enumerated/sampled fact domain and a frozen oracle; each pair's
  variant differs in exactly one fact leaf; declared flip question types are
  *computed* from both fact sets — a pair with no real flip is never emitted.
  For chance families a "flip" is a changed distribution.
* **Component-connected split isolation (unchanged).**  Both members of a pair
  and every member sharing an identical canonical visible input are unioned
  into one component; one split per component by deterministic per-family
  position plus a seed offset.  V4 alternates two 20-slot cycles by family
  parity so dev and test each receive ~17.5% overall (cycle A: 11/3/2/4,
  cycle B: 11/4/2/3 train/dev/calibration/test per 20 components).
* **Label semantics preserved + extended.**  Deterministic items keep
  ``deterministic_truth`` one-hot gold.  Chance items carry the analytic
  conditional distribution with ``gold``/``gold_index`` null and
  ``gold_label_kind`` ``unobserved`` — never an invented hard outcome.
* **No evaluation-derived sources, ever.**  V1 reserved-token/path tripwires
  are reused verbatim via ``v1.guard_source``; ``derived_from_evaluation_corpus``
  is false on every item.

Boundaries
----------

V1, V2, V3 and ``engineering_heldout_v1``/``v2`` are read-only inputs for the
audit mode; this builder never modifies them.  Passing this build authorizes
nothing — training remains behind its own protocol review.

Usage
-----

    .venv/bin/python scripts/build_engineering_corpus_v4.py --self-test
    .venv/bin/python scripts/build_engineering_corpus_v4.py \
        --output-dir research/engineering_judgment_corpus_v4
    .venv/bin/python scripts/build_engineering_corpus_v4.py \
        --check research/engineering_judgment_corpus_v4
    .venv/bin/python scripts/build_engineering_corpus_v4.py --audit \
        --corpus research/engineering_judgment_corpus_v4 \
        --heldout research/engineering_heldout_v1/items.jsonl \
        --heldout research/engineering_heldout_v2/items.jsonl \
        --compare-corpus research/engineering_judgment_corpus_v2 \
        --compare-corpus research/engineering_judgment_corpus_v3 \
        --output results/engineering_corpus_v4_audit_receipt.json
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_gate_contrastive_v1 import digest_value  # noqa: E402
import build_engineering_corpus_v1 as v1  # noqa: E402
import build_engineering_corpus_v2 as v2  # noqa: E402
import build_engineering_corpus_v3 as v3  # noqa: E402
from predict_toy_decisions import validate_request  # noqa: E402

SCHEMA_VERSION = "nanojev-engineering-judgment-corpus-v4"
MANIFEST_SCHEMA = "nanojev-engineering-judgment-manifest-v4"
ITEM_SCHEMA = "nanojev-engineering-judgment-item-v4"
SOURCE_GROUP_SCHEMA = "nanojev-engineering-source-group-v4"
AUDIT_SCHEMA = "nanojev-engineering-corpus-v4-audit-v1"
CATALOG_VERSION = "engineering-judgment-catalog-v4-scaled"
TRAINER_ROW_SCHEMA = v1.TRAINER_ROW_SCHEMA
ITEM_DIR, TRAINER_DIR, MANIFEST_NAME = v1.ITEM_DIR, v1.TRAINER_DIR, v1.MANIFEST_NAME
SPLITS, SPLIT_FILES, QUESTION_TYPES = v1.SPLITS, v1.SPLIT_FILES, v1.QUESTION_TYPES
DEFAULT_SEED = 20260922
SOURCE_ID = "scripts/build_engineering_corpus_v4.py"

#: Cycle A is the V3 20-slot cycle verbatim: 11 train / 3 dev / 2 calibration /
#: 4 test.  Cycle B swaps the dev and test slots.  Alternating A/B by family
#: parity yields overall ~55/17.5/10/17.5 while every family still receives a
#: deterministic per-position assignment.
SPLIT_CYCLE_A = v3.SPLIT_CYCLE
SPLIT_CYCLE_B = tuple("dev" if slot == "test" else
                      "test" if slot == "dev" else slot
                      for slot in SPLIT_CYCLE_A)
SPLIT_CYCLES = (SPLIT_CYCLE_A, SPLIT_CYCLE_B)

PAIRS_PER_FAMILY = 20
MIN_PAIRS_PER_FAMILY = 20
SAMPLE_CAP = 6000           # sampled candidates per family when the domain is large
ENUMERATE_CAP = 20000       # domains at/below this size are enumerated exhaustively
CHANCE_KIND = "programmatic_conditional_distribution"

# T9c grouped-calibration minimums (research/nanojev_v2_t9c_grouped_calibration_protocol_v1.json):
# a (question_type, option_count) group is estimable with >=32 calibration rows
# over >=8 distinct states.  V4 requires >=8 estimable groups in calibration.
T9C_MIN_ROWS = 32
T9C_MIN_STATES = 8
T9C_MIN_GROUPS = 8

ALLOWED_QUESTION_KEYS = v1.ALLOWED_QUESTION_KEYS
_evaluation_derived = v2.evaluation_derived
_visible_fingerprint = v2.visible_fingerprint

BOOL = v3.BOOL
INT = v3.INT
_field = v3._field
domain_values = v3.domain_values


# --------------------------------------------------------------------------------------
# family declarations: V3's 18 rule sets reused as code + 18 new families
# --------------------------------------------------------------------------------------

FAMILY_DEFS = list(v3.FAMILY_DEFS)
V3_FAMILY_NAMES = tuple(spec["family"] for spec in v3.FAMILY_DEFS)


def _fam(family, feature, headings, services, tickets, fields, qids,
         instructions, boolean_criteria, choice_criteria, score_criteria,
         oracle, why, kind="deterministic", sampler=None,
         choice_coverage_min=None):
    FAMILY_DEFS.append({
        "family": family, "feature": feature, "headings": tuple(headings),
        "services": tuple(services), "tickets": tuple(tickets),
        "fields": tuple(fields), "qids": dict(qids),
        "instructions": {qt: tuple(v) for qt, v in instructions.items()},
        "boolean_criteria": tuple(boolean_criteria),
        "choice_criteria": dict(choice_criteria),
        "score_criteria": list(score_criteria),
        "oracle": oracle, "why": why,
        "kind": kind, "sampler": sampler,
        "choice_coverage_min": choice_coverage_min,
    })


def _levels(*texts):
    return list(texts)


# ---------------------------------------------------------------- slot_fleet_binding
# Multi-entity state binding: eight slots each carry a load fact; the gold binds
# the per-slot value to the right candidate id.

def _slot_fleet_oracle(f):
    loads = [f[f"slot_{k}_load"] for k in range(8)]
    top = max(range(8), key=lambda k: (loads[k], -k))
    over = sum(1 for load in loads if load > f["limit"])
    return {"boolean": over > 0,
            "choice": f"slot_{top:02d}",
            "score": min(over, 5)}


_fam(
    "slot_fleet_binding", "per-slot-load-binding",
    ("Fleet load board for {service} ({ticket}):",
     "Dispatch review — {service} — {ticket}:",
     "Slot utilization report, {service} [{ticket}]:"),
    ("render-farm", "build-cache", "asset-pipe", "mail-relay",
     "geo-tile-svc", "doc-indexer", "voice-gw", "meter-reader"),
    ("FLEET", "SLOT", "DISP"),
    tuple([_field(f"slot_{k}_load", f"slot {k:02d} current load, percent",
                  INT(0, 100, 5)) for k in range(8)]
          + [_field("limit", "per-slot overload limit, percent", INT(50, 95, 5))]),
    {"boolean": "any_slot_over_limit", "choice": "busiest_slot",
     "score": "slots_over_limit"},
    {"boolean": ("Is any slot above the overload limit?",
                 "Does at least one slot exceed its load limit?",
                 "Is there an overloaded slot in this fleet?"),
     "choice": ("Which slot is carrying the heaviest load?",
                "Pick the busiest slot in this fleet.",
                "Which slot id has the highest load?"),
     "score": ("Score how many slots exceed the limit, capped at 5, 0 to 5.",
               "Rate fleet congestion by overloaded slot count, 0 to 5.",
               "How many slots breach the limit (5+ caps), 0 to 5?")},
    ({}, {"false": "no slot is above the overload limit",
          "true": "at least one slot is above the limit"}),
    {f"slot_{k:02d}": f"Slot {k:02d} carries the highest load" for k in range(8)},
    _levels("no slot over the limit", "one slot over", "two slots over",
            "three slots over", "four slots over", "five or more slots over"),
    _slot_fleet_oracle,
    "the busiest slot is the highest per-slot load (ties resolve to the lower "
    "slot id); the boolean is whether any load exceeds the stated limit and the "
    "score counts breaching slots capped at five",)


# ---------------------------------------------------------------- wide_dispatch_32
# 32-candidate choice: the gold is the first free dispatch slot.

def _wide_dispatch_oracle(f):
    first = f["first_free_slot"]
    congestion = 0 if first <= 7 else (1 if first <= 23 else 2)
    return {"boolean": first == 0,
            "choice": f"slot_{first:02d}",
            "score": congestion}


def _wide_dispatch_sample(rng):
    # bias toward slot 0 so the boolean (slot_00 free) sees both labels
    first = 0 if rng.random() < 0.30 else rng.randrange(32)
    return {"first_free_slot": first,
            "reserved_pool": rng.randrange(4),
            "priority_lane_open": rng.random() < 0.5}


_fam(
    "wide_dispatch_32", "wide-dispatch-target-selection",
    ("Dispatch table for {service} ({ticket}):",
     "Worker pool status — {service} — {ticket}:",
     "Slot occupancy board, {service} [{ticket}]:"),
    ("batch-runner", "packager", "scan-worker", "encode-farm",
     "sync-agent", "pdf-render", "thumb-gen", "etl-slot-pool"),
    ("DISP", "WRK", "POOL"),
    (_field("first_free_slot", "slots s000 up to this index are occupied; slots "
            "from this index onward are free", INT(0, 31)),
     _field("reserved_pool", "reserved standby slots not in the dispatch pool",
            INT(0, 3)),
     _field("priority_lane_open", "the priority lane is accepting jobs", BOOL)),
    {"boolean": "slot_zero_free", "choice": "dispatch_target",
     "score": "congestion_band"},
    {"boolean": ("Is dispatch slot s000 free right now?",
                 "Can slot s000 take the next job?",
                 "Is the lowest slot available for dispatch?"),
     "choice": ("Which slot should receive the next dispatched job?",
                "Pick the correct dispatch target slot.",
                "Which slot id takes the next job under the first-free rule?"),
     "score": ("Score dispatch congestion, 0 to 2: low (s000-s007 free range), "
               "medium, high.",
               "Rate how congested the dispatch pool is, 0 to 2.",
               "How congested is the pool, 0 to 2?")},
    ({}, {"false": "slot s000 is occupied",
          "true": "slot s000 is free"}),
    {f"slot_{k:02d}": f"Dispatch the next job to slot s{k:03d}"
     for k in range(32)},
    _levels("low congestion: a slot in s000-s007 is first free",
            "medium congestion: first free slot is s008-s023",
            "high congestion: first free slot is s024-s031"),
    _wide_dispatch_oracle,
    "the pool dispatches to the lowest-indexed free slot: slots before "
    "first_free_slot are occupied, so the target is exactly that index; "
    "congestion bands are 0-7 low, 8-23 medium, 24-31 high",
    sampler=_wide_dispatch_sample, choice_coverage_min=20)


# ---------------------------------------------------------------- batch_grid_255
# 255-candidate choice: exercises the full served-contract candidate range.

def _batch_grid_oracle(f):
    start = f["window_start"]
    size = f["window_size"]
    depth = 0 if size <= 2 else (1 if size <= 6 else (2 if size <= 16 else
            (3 if size <= 40 else (4 if size <= 100 else 5))))
    return {"boolean": size >= 32,
            "choice": f"batch_{start:03d}",
            "score": depth}


_BATCH_BANDS = ((1, 2), (3, 6), (7, 16), (17, 40), (41, 100), (101, 199))


def _batch_grid_sample(rng):
    # draw the size band uniformly so all six score levels are reachable early
    lo, hi = _BATCH_BANDS[rng.randrange(6)]
    return {"window_start": rng.randrange(255),
            "window_size": rng.randrange(lo, hi + 1),
            "priority_claim": rng.random() < 0.5}


_fam(
    "batch_grid_255", "full-range-candidate-selection",
    ("Reconciliation window for {service} ({ticket}):",
     "Batch eligibility board — {service} — {ticket}:",
     "Claim window report, {service} [{ticket}]:"),
    ("ledger-sync", "settlement-run", "recon-engine", "clearing-svc",
     "statement-gen", "audit-export", "fx-netting", "gl-posting"),
    ("RECON", "BATCH", "SETTLE"),
    (_field("window_start", "the first eligible batch id in the claim window "
            "(batches are numbered 000 to 254)", INT(0, 254)),
     _field("window_size", "how many consecutive batch ids are eligible",
            INT(1, 199)),
     _field("priority_claim", "the window was opened under a priority claim",
            BOOL)),
    {"boolean": "wide_window", "choice": "first_claim",
     "score": "window_depth"},
    {"boolean": ("Does the eligible window cover at least 32 batches?",
                 "Is the claim window at least 32 batches wide?",
                 "Is this a wide (32+) eligibility window?"),
     "choice": ("Which batch id should the reconciler claim first?",
                "Pick the first batch id the reconciler must claim.",
                "Under the lowest-id-first rule, which batch is claimed first?"),
     "score": ("Score the window depth, 0 to 5.",
               "Rate how deep the eligibility window is, 0 to 5.",
               "How deep is this claim window, 0 to 5?")},
    ({}, {"false": "fewer than 32 batches are eligible",
          "true": "32 or more batches are eligible"}),
    {f"batch_{k:03d}": f"Claim batch {k:03d} first" for k in range(255)},
    _levels("depth 1-2 batches", "depth 3-6", "depth 7-16",
            "depth 17-40", "depth 41-100", "depth over 100"),
    _batch_grid_oracle,
    "the reconciler always claims the lowest eligible batch id, which is exactly "
    "window_start; the boolean marks windows of at least 32 batches and the "
    "score buckets the window size 1-2/3-6/7-16/17-40/41-100/100+",
    sampler=_batch_grid_sample, choice_coverage_min=24)


# ------------------------------------------------------- weighted_queue_lottery
# Analytic-chance family: the served answer is the conditional distribution of a
# weighted lottery; no realized outcome exists, so gold stays unobserved.

def _queue_lottery_oracle(f):
    weights = [f[f"w_{k}"] for k in range(8)]
    total = sum(weights)
    probs = [(w / total) if total else 1.0 / 8 for w in weights]
    focus = f["focus_queue"]
    group_p = [sum(probs[g * 2:(g + 1) * 2]) for g in range(4)]
    return {"boolean": {"false": 1.0 - probs[focus], "true": probs[focus]},
            "choice": {f"queue_{k}": probs[k] for k in range(8)},
            "score": {str(g): group_p[g] for g in range(4)}}


_fam(
    "weighted_queue_lottery", "analytic-queue-lottery",
    ("Routing lottery for {service} ({ticket}):",
     "Weighted dispatch record — {service} — {ticket}:",
     "Queue lottery setup, {service} [{ticket}]:"),
    ("job-router", "task-broker", "flow-dispatch", "work-shed",
     "ingest-router", "ops-queue", "sync-broker", "sched-core"),
    ("LOTR", "ROUTE", "QUEUE"),
    tuple([_field(f"w_{k}", f"routing weight of queue_{k} in the lottery",
                  INT(0, 9)) for k in range(8)]
          + [_field("focus_queue", "the queue the boolean question asks about",
                    INT(0, 7))]),
    {"boolean": "focus_queue_hit", "choice": "queue_draw",
     "score": "group_draw"},
    {"boolean": ("Will the next routed job land in the focus queue?",
                 "Does the next job go to the focus queue?"),
     "choice": ("Predict the unobserved routing draw. Return the conditional "
                "probability of each queue given the weights.",
                "Give the conditional probability that the next job lands in "
                "each queue."),
     "score": ("Predict which pair-group the next job's queue belongs to. "
               "Probabilities follow the routing weights.",
               "Return the conditional probability of each queue pair-group "
               "for the next draw.")},
    ({}, {}),
    {f"queue_{k}": f"The next job is routed to queue_{k}" for k in range(8)},
    _levels("the drawn queue is one of: queue_0, queue_1",
            "the drawn queue is one of: queue_2, queue_3",
            "the drawn queue is one of: queue_4, queue_5",
            "the drawn queue is one of: queue_6, queue_7"),
    _queue_lottery_oracle,
    "the next job is routed by a weighted lottery over the eight queues; if "
    "every weight is zero the draw is uniform; no draw has been observed so "
    "the target is the analytic conditional distribution, not an outcome",
    kind="chance")


# ------------------------------------------------------- traffic_split_lottery
# Second analytic-chance family: canary traffic split over five tracks.

def _traffic_lottery_oracle(f):
    weights = [f[f"w_{k}"] for k in range(5)]
    total = sum(weights)
    probs = [(w / total) if total else 1.0 / 5 for w in weights]
    focus = f["focus_track"]
    group_p = [probs[0] + probs[1], probs[2] + probs[3], probs[4]]
    return {"boolean": {"false": 1.0 - probs[focus], "true": probs[focus]},
            "choice": {f"track_{k}": probs[k] for k in range(5)},
            "score": {str(g): group_p[g] for g in range(3)}}


_fam(
    "traffic_split_lottery", "analytic-traffic-split",
    ("Traffic split for {service} ({ticket}):",
     "Rollout weights — {service} — {ticket}:",
     "Split configuration, {service} [{ticket}]:"),
    ("edge-pop", "web-tier", "api-edge", "static-cdn",
     "grpc-gw", "ws-hub", "auth-edge", "media-gw"),
    ("SPLIT", "TRAFFIC", "POP"),
    tuple([_field(f"w_{k}", f"traffic weight of track_{k}", INT(0, 20))
           for k in range(5)]
          + [_field("focus_track", "the track the boolean question asks about",
                    INT(0, 4))]),
    {"boolean": "focus_track_hit", "choice": "next_request_track",
     "score": "track_group"},
    {"boolean": ("Will the next request be served by the focus track?",
                 "Does the next request hit the focus track?"),
     "choice": ("Predict which track serves the next request. Return the "
                "conditional probability of each track given the weights.",
                "Give the conditional probability of each track for one "
                "request."),
     "score": ("Predict which track-group serves the next request. "
               "Probabilities follow the split weights.",
               "Return the conditional probability of each track-group.")},
    ({}, {}),
    {f"track_{k}": f"The next request is served by track_{k}" for k in range(5)},
    _levels("the serving track is one of: track_0, track_1",
            "the serving track is one of: track_2, track_3",
            "the serving track is: track_4"),
    _traffic_lottery_oracle,
    "one incoming request is assigned by the stated split weights; if all "
    "weights are zero the assignment is uniform; the request has not arrived "
    "so the target is the analytic conditional distribution",
    kind="chance")


# ---------------------------------------------------------------- export_compliance
# Rule-compliance cascade: embargo, classification, approval and encryption.

def _export_oracle(f):
    cls, ok = f["class_level"], f["approved"]
    if f["embargoed_region"] and cls >= 1:
        choice = "deny_export"
    elif f["embargoed_region"]:
        choice = "route_legal"
    elif cls == 3 and not ok:
        choice = "deny_export"
    elif cls == 3 and f["region_risk"] == 2:
        choice = "route_dpo"
    elif cls >= 2 and not ok:
        choice = "route_approval"
    elif cls >= 2 and not f["encrypted"]:
        choice = "encrypt_first"
    elif f["region_risk"] == 2:
        choice = "route_review"
    elif f["volume_gb"] > 400:
        choice = "chunk_export"
    else:
        choice = "allow"
    severity = 0 if choice == "allow" else (2 if choice == "deny_export" else 1)
    return {"boolean": choice == "allow", "choice": choice, "score": severity}


_fam(
    "export_compliance", "export-rule-compliance",
    ("Export request review for {service} ({ticket}):",
     "Compliance check — {service} — {ticket}:",
     "Data export board, {service} [{ticket}]:"),
    ("reporting-svc", "crm-export", "billing-extract", "telemetry-hub",
     "hr-sync", "order-archive", "partner-feed", "usage-export"),
    ("COMP", "EXPT", "REG"),
    (_field("class_level", "data classification level 0-3 (0 public, 3 restricted)",
            INT(0, 3)),
     _field("embargoed_region", "the destination region is under embargo", BOOL),
     _field("approved", "a compliance approval ticket is on file", BOOL),
     _field("encrypted", "the export payload is encrypted at rest", BOOL),
     _field("region_risk", "destination risk tier 0-2", INT(0, 2)),
     _field("volume_gb", "export volume in GB", INT(1, 500))),
    {"boolean": "export_allowed_as_is", "choice": "export_verdict",
     "score": "restriction_level"},
    {"boolean": ("May this export proceed exactly as requested?",
                 "Is the export allowed without any extra step?",
                 "Can this export ship as-is under policy?"),
     "choice": ("What is the correct verdict for this export request?",
                "Pick the compliant outcome for this export.",
                "Which verdict applies under the export rules?"),
     "score": ("Score the restriction level, 0 to 2.",
               "Rate how restricted this export is, 0 to 2.",
               "How restricted is this export, 0 to 2?")},
    ({}, {"false": "an extra step or denial applies",
          "true": "the export is allowed as requested"}),
    {"allow": "Allow the export as requested",
     "chunk_export": "Allow only after splitting the payload below 400 GB",
     "route_review": "Send to standard review for a high-risk region",
     "encrypt_first": "Encrypt the payload before export",
     "route_approval": "Collect a compliance approval first",
     "route_legal": "Embargoed region with public data: route to legal",
     "deny_export": "Deny the export outright",
     "route_dpo": "Route to the data protection officer"},
    _levels("unrestricted", "conditionally restricted", "denied or embargoed"),
    _export_oracle,
    "embargo plus any classification denies outright, embargo on public data "
    "routes to legal; restricted class without approval is denied, restricted "
    "class approved into a high-risk region goes to the data protection "
    "officer, confidential without approval routes for approval, confidential "
    "unencrypted encrypts first, other high-risk regions review, oversized "
    "payloads chunk, else allow",)


# ---------------------------------------------------------------- review_merge_gate
# Code-review-flavoured merge verdict with an ordered gate cascade.

def _merge_gate_oracle(f):
    if not f["ci_green"]:
        choice = "fix_ci"
    elif f["changes_requested"]:
        choice = "resolve_requests"
    elif f["touches_auth"] and not f["security_signoff"]:
        choice = "security_review"
    elif f["author_is_committer"] and f["approvals"] == 0:
        choice = "needs_external_review"
    elif f["approvals"] < f["required_approvals"]:
        choice = "await_approvals"
    elif f["diff_lines"] > 3000:
        choice = "split_diff"
    elif f["flaky_seen"]:
        choice = "quarantine_flaky"
    else:
        choice = "merge_now"
    gates = sum([not f["ci_green"], bool(f["changes_requested"]),
                 f["touches_auth"] and not f["security_signoff"],
                 f["author_is_committer"] and f["approvals"] == 0,
                 f["approvals"] < f["required_approvals"],
                 f["diff_lines"] > 3000 or f["flaky_seen"]])
    return {"boolean": choice == "merge_now", "choice": choice,
            "score": min(gates, 5)}


_fam(
    "review_merge_gate", "review-gate-cascade",
    ("Merge review for {service} ({ticket}):",
     "PR gate board — {service} — {ticket}:",
     "Change review checklist, {service} [{ticket}]:"),
    ("svc-payments", "svc-search", "svc-users", "svc-orders",
     "svc-notify", "svc-catalog", "svc-geo", "svc-media"),
    ("PR", "REV", "MERGE"),
    (_field("ci_green", "the CI pipeline is green on the head commit", BOOL),
     _field("changes_requested", "a reviewer has open change requests", BOOL),
     _field("touches_auth", "the diff touches the auth module", BOOL),
     _field("security_signoff", "a security reviewer has signed off", BOOL),
     _field("author_is_committer", "the author holds commit rights", BOOL),
     _field("approvals", "recorded approvals", INT(0, 4)),
     _field("required_approvals", "approvals the policy requires", INT(1, 3)),
     _field("diff_lines", "lines changed in the diff", INT(10, 5000, 10)),
     _field("flaky_seen", "a flaky test failure was seen on this branch", BOOL)),
    {"boolean": "mergeable_now", "choice": "merge_verdict",
     "score": "open_gate_count"},
    {"boolean": ("May this change merge right now?",
                 "Is the change mergeable at this moment?",
                 "Can the PR be merged as it stands?"),
     "choice": ("What is the correct merge verdict?",
                "Pick the gate outcome for this change.",
                "Which verdict applies to this PR?"),
     "score": ("Score the number of unmet merge gates, capped at 5, 0 to 5.",
               "Rate how many gates remain unmet, 0 to 5.",
               "How many merge gates are still open, 0 to 5?")},
    ({}, {"false": "at least one merge gate is unmet",
          "true": "every gate is satisfied; merge now"}),
    {"merge_now": "All gates pass; merge the change",
     "await_approvals": "Wait for the remaining required approvals",
     "security_review": "Auth-touched diff needs a security sign-off",
     "fix_ci": "CI is red; fix the pipeline first",
     "resolve_requests": "Resolve the open change requests",
     "split_diff": "Split the oversized diff before merging",
     "needs_external_review": "A committer-authored diff needs a non-author approval",
     "quarantine_flaky": "Quarantine the flaky test before merging"},
    _levels("no gates unmet", "one gate unmet", "two gates unmet",
            "three gates unmet", "four gates unmet", "five or more unmet"),
    _merge_gate_oracle,
    "gates run in a fixed order: red CI first, then open change requests, then "
    "auth diffs missing security sign-off, then committer-authored diffs with "
    "zero approvals, then missing required approvals, then oversized diffs, "
    "then flaky evidence; the score counts unmet gate conditions capped at 5",)


# ---------------------------------------------------------------- shift_coverage
# Scheduling domain: coverage arithmetic over weekday/weekend/holiday demand.

def _shift_oracle(f):
    available = int(f["primary_free"]) + int(f["backup_free"]) + f["volunteers"]
    required = f["min_coverage"] + int(f["weekend"] or f["holiday"])
    if available == 0:
        choice = "page_manager"
    elif f["holiday"] and not (f["primary_free"] or f["backup_free"]):
        choice = "activate_holiday_rota"
    elif available < required:
        choice = "split_shift"
    elif required >= 3:
        choice = "assign_all_available"
    else:
        choice = "assign_primary"
    strain = 0 if available >= required else (2 if available == 0 else 1)
    return {"boolean": available >= required, "choice": choice, "score": strain}


_fam(
    "shift_coverage", "shift-coverage-scheduling",
    ("Coverage plan for {service} ({ticket}):",
     "Shift roster — {service} — {ticket}:",
     "Staffing board, {service} [{ticket}]:"),
    ("store-north", "store-south", "depot-east", "depot-west",
     "kiosk-mall", "hub-airport", "lab-downtown", "site-harbor"),
    ("ROTA", "SHIFT", "STAFF"),
    (_field("primary_free", "the primary on-shift engineer is available", BOOL),
     _field("backup_free", "the backup engineer is available", BOOL),
     _field("volunteers", "extra volunteers signed up", INT(0, 4)),
     _field("min_coverage", "minimum staffed seats required", INT(1, 3)),
     _field("weekend", "the shift falls on a weekend", BOOL),
     _field("holiday", "the shift falls on a public holiday", BOOL)),
    {"boolean": "coverage_met", "choice": "coverage_plan",
     "score": "staffing_strain"},
    {"boolean": ("Does the roster meet the required coverage?",
                 "Is the required coverage reached by this roster?",
                 "Are enough people available for this shift?"),
     "choice": ("What is the correct coverage plan?",
                "Pick the staffing action for this shift.",
                "Which coverage decision applies?"),
     "score": ("Score the staffing strain, 0 to 2.",
               "Rate how strained coverage is, 0 to 2.",
               "How strained is this roster, 0 to 2?")},
    ({}, {"false": "available headcount is below the requirement",
          "true": "coverage is met"}),
    {"assign_primary": "Assign the primary engineer; coverage is met",
     "assign_all_available": "Assign everyone; the requirement is elevated",
     "split_shift": "Split the shift; available staff cannot cover it alone",
     "activate_holiday_rota": "Use the holiday rota; no regular staff on a holiday",
     "page_manager": "Nobody is available; page the manager"},
    _levels("covered", "strained: short of the requirement",
            "critical: nobody available"),
    _shift_oracle,
    "weekends and holidays raise the requirement by one; nobody available pages "
    "the manager, a holiday without regular staff uses the holiday rota, "
    "understaffed shifts split, elevated requirements assign everyone, "
    "otherwise the primary covers",)


# ---------------------------------------------------------------- budget_portfolio
# Resource allocation: pick the best-value project inside a stated budget.

def _portfolio_oracle(f):
    costs = [f[f"proj_{k}_cost"] for k in range(8)]
    rois = [f[f"proj_{k}_roi"] for k in range(8)]
    budget = f["budget"]
    affordable = [k for k in range(8) if costs[k] <= budget]
    if affordable:
        pick = max(affordable, key=lambda k: (rois[k], -k))
    else:
        pick = min(range(8), key=lambda k: (costs[k], k))
    return {"boolean": bool(affordable),
            "choice": f"proj_{pick}",
            "score": min(len(affordable), 5)}


_fam(
    "budget_portfolio", "budget-constrained-selection",
    ("Portfolio review for {service} ({ticket}):",
     "Quarterly budget board — {service} — {ticket}:",
     "Investment shortlist, {service} [{ticket}]:"),
    ("eng-dept", "growth-team", "platform-eng", "data-org",
     "field-ops", "research-lab", "it-services", "logistics"),
    ("BUDG", "PORT", "INV"),
    tuple([_field(f"proj_{k}_cost", f"project {k} cost in kUSD", INT(5, 60, 5))
           for k in range(8)]
          + [_field(f"proj_{k}_roi", f"project {k} expected ROI score 1-10",
                    INT(1, 10)) for k in range(8)]
          + [_field("budget", "remaining discretionary budget in kUSD",
                    INT(10, 100, 5))]),
    {"boolean": "any_affordable", "choice": "fund_project",
     "score": "affordable_count"},
    {"boolean": ("Is at least one project inside the remaining budget?",
                 "Can the budget fund any listed project?",
                 "Does any project fit the budget?"),
     "choice": ("Which project should be funded?",
                "Pick the project the budget rule selects.",
                "Which project gets the funding under the stated rule?"),
     "score": ("Score how many projects are affordable, capped at 5, 0 to 5.",
               "Rate portfolio fit by affordable count, 0 to 5.",
               "How many projects fit the budget (5+ caps), 0 to 5?")},
    ({}, {"false": "nothing fits the remaining budget",
          "true": "at least one project is affordable"}),
    {f"proj_{k}": f"Fund project {k}" for k in range(8)},
    _levels("nothing affordable", "one project affordable",
            "two affordable", "three affordable", "four affordable",
            "five or more affordable"),
    _portfolio_oracle,
    "fund the highest-ROI project whose cost fits the remaining budget (ties to "
    "the lower project number); when nothing fits, fund the cheapest project "
    "as the documented fallback; the score counts affordable projects",)


# ---------------------------------------------------------------- negated_change_policy
# Negation handling: fields and boolean phrasing carry explicit negations.

def _negated_change_oracle(f):
    # a freeze blocks unless an exception is on file or it is a security fix
    blocked = f["freeze_active"] and f["no_exception_on_file"] \
        and not f["security_fix"]
    if blocked:
        choice = "hold_change"
    elif f["security_fix"] and f["freeze_active"] and f["no_exception_on_file"]:
        choice = "expedite_security_path"
    elif f["approvals"] < f["required_approvals"]:
        choice = "collect_approvals"
    elif not f["no_exception_on_file"] and f["freeze_active"]:
        choice = "proceed_under_exception"
    else:
        choice = "proceed"
    friction = 0 if choice == "proceed" else (2 if choice == "hold_change" else 1)
    return {"boolean": blocked, "choice": choice, "score": friction}


_fam(
    "negated_change_policy", "negated-policy-conditions",
    ("Change advisory for {service} ({ticket}):",
     "Freeze-window review — {service} — {ticket}:",
     "Change policy board, {service} [{ticket}]:"),
    ("core-banking", "payroll-svc", "trade-ledger", "custody-svc",
     "tax-engine", "claims-svc", "policy-admin", "fraud-ml"),
    ("CAB", "FRZ", "CHG"),
    (_field("freeze_active", "a change freeze window is active", BOOL),
     _field("no_exception_on_file", "no freeze exception is on file", BOOL),
     _field("security_fix", "the change carries a security fix", BOOL),
     _field("approvals", "recorded change approvals", INT(0, 3)),
     _field("required_approvals", "approvals the change policy requires",
            INT(1, 3))),
    {"boolean": "change_not_permitted", "choice": "change_verdict",
     "score": "process_friction"},
    {"boolean": ("Is it the case that this change may NOT proceed right now?",
                 "Is the change currently blocked from proceeding?",
                 "Is it true that no path currently permits this change?"),
     "choice": ("What is the correct verdict for this change?",
                "Pick the outcome the change policy dictates.",
                "Which verdict applies under the freeze rules?"),
     "score": ("Score the process friction, 0 to 2.",
               "Rate how much friction this change faces, 0 to 2.",
               "How much process friction applies, 0 to 2?")},
    ({}, {"false": "a permitted path exists right now",
          "true": "no path currently permits the change"}),
    {"proceed": "Proceed on the normal path",
     "proceed_under_exception": "Proceed under the filed freeze exception",
     "collect_approvals": "Collect the missing approvals first",
     "expedite_security_path": "Use the expedited security-fix path",
     "hold_change": "Hold the change until the freeze lifts"},
    _levels("no friction: proceed normally",
            "intermediate step required",
            "fully blocked by the freeze"),
    _negated_change_oracle,
    "an active freeze blocks the change unless an exception is on file or the "
    "change carries a security fix; a security fix under freeze without an "
    "exception takes the expedited path; missing approvals are collected next; "
    "a filed exception permits proceeding under exception; otherwise proceed",)


# ---------------------------------------------------------------- plan_constraint_pick
# Multi-constraint tradeoff: eight plans each carry a violation count.

def _plan_pick_oracle(f):
    viol = [f[f"plan_{k}_violations"] for k in range(8)]
    best = min(range(8), key=lambda k: (viol[k], k))
    compliant = sum(1 for v in viol if v == 0)
    return {"boolean": compliant > 0,
            "choice": f"plan_{best}",
            "score": min(compliant, 5)}


_fam(
    "plan_constraint_pick", "constraint-satisfaction-pick",
    ("Migration plan review for {service} ({ticket}):",
     "Constraint board — {service} — {ticket}:",
     "Plan comparison, {service} [{ticket}]:"),
    ("datacenter-m", "region-x", "zone-eu1", "zone-us2",
     "cell-apac", "site-north", "site-south", "colo-east"),
    ("PLAN", "MIGR", "CONST"),
    tuple(_field(f"plan_{k}_violations",
                 f"hard-constraint violations counted in plan {k}", INT(0, 4))
          for k in range(8)),
    {"boolean": "compliant_plan_exists", "choice": "selected_plan",
     "score": "noncompliant_count"},
    {"boolean": ("Does at least one plan violate zero constraints?",
                 "Is there a fully compliant plan on the board?",
                 "Does any plan satisfy every constraint?"),
     "choice": ("Which plan should be selected?",
                "Pick the plan the selection rule chooses.",
                "Which plan id wins under the fewest-violations rule?"),
     "score": ("Score how many plans are fully compliant, capped at 5, 0 to 5.",
               "Rate plan quality by compliant count, 0 to 5.",
               "How many plans violate zero constraints (5+ caps), 0 to 5?")},
    ({}, {"false": "every plan violates at least one constraint",
          "true": "a fully compliant plan exists"}),
    {f"plan_{k}": f"Select plan {k}" for k in range(8)},
    _levels("no compliant plan", "one compliant plan",
            "two compliant", "three compliant", "four compliant",
            "five or more compliant"),
    _plan_pick_oracle,
    "select the plan with the fewest hard-constraint violations (ties to the "
    "lower plan number); the boolean asks whether a zero-violation plan exists "
    "and the score counts fully compliant plans capped at five",)


# ---------------------------------------------------------------- tenant_isolation_review
# Rule compliance: isolation tiers vs deployed controls.

def _tenant_oracle(f):
    tier = f["isolation_tier"]
    if f["cross_tenant_queries"] and not f["row_level_security"]:
        choice = "block_deploy"
    elif f["shared_schema"] and not f["row_level_security"] and tier == 2:
        choice = "block_deploy"
    elif not f["dedicated_keys"] and tier == 2:
        choice = "add_dedicated_keys"
    elif f["shared_schema"] and not f["row_level_security"]:
        choice = "require_rls"
    elif tier == 0:
        choice = "accept_relaxed"
    else:
        choice = "accept"
    gaps = sum([f["cross_tenant_queries"] and not f["row_level_security"],
                f["shared_schema"] and not f["row_level_security"],
                not f["dedicated_keys"] and tier >= 1])
    return {"boolean": choice in ("accept", "accept_relaxed"),
            "choice": choice, "score": min(gaps, 2)}


_fam(
    "tenant_isolation_review", "tenant-isolation-compliance",
    ("Isolation review for {service} ({ticket}):",
     "Tenancy board — {service} — {ticket}:",
     "Isolation checklist, {service} [{ticket}]:"),
    ("crm-tenants", "saas-portal", "agency-suite", "clinic-erp",
     "edu-platform", "franchise-pos", "gov-suite", "fintech-core"),
    ("TEN", "ISO", "RLS"),
    (_field("shared_schema", "tenants share one database schema", BOOL),
     _field("row_level_security", "row-level security policies are enforced",
            BOOL),
     _field("dedicated_keys", "each tenant has dedicated encryption keys", BOOL),
     _field("cross_tenant_queries", "application code issues cross-tenant "
            "queries", BOOL),
     _field("isolation_tier", "contracted isolation tier 0-2 (2 strictest)",
            INT(0, 2))),
    {"boolean": "deployable_as_is", "choice": "isolation_verdict",
     "score": "isolation_gap"},
    {"boolean": ("Is this deployment acceptable exactly as configured?",
                 "May the tenant configuration ship as-is?",
                 "Is the posture deployable without changes?"),
     "choice": ("What is the correct isolation verdict?",
                "Pick the verdict the tenancy policy requires.",
                "Which isolation outcome applies?"),
     "score": ("Score the isolation gap, 0 to 2.",
               "Rate the isolation gap, 0 to 2.",
               "How large is the isolation gap, 0 to 2?")},
    ({}, {"false": "a control change is required first",
          "true": "deployable as configured"}),
    {"accept": "Tier requirements met; accept the configuration",
     "accept_relaxed": "Tier 0 accepts the relaxed posture",
     "require_rls": "Shared schema requires row-level security first",
     "add_dedicated_keys": "Tier 2 requires dedicated encryption keys",
     "block_deploy": "Block: cross-tenant exposure without enforcement"},
    _levels("no isolation gap", "one control gap", "two or more control gaps"),
    _tenant_oracle,
    "cross-tenant queries without row-level security always block; strict tier "
    "with shared schema and no RLS also blocks; strict tier needs dedicated "
    "keys; shared schema without RLS must add it; tier 0 accepts the relaxed "
    "posture; otherwise accept",)


# ---------------------------------------------------------------- maintenance_window_pick
# Scheduling: eight candidate windows, blocked flags and load facts bind the pick.

def _window_oracle(f):
    blocked = [f[f"win_{k}_blocked"] for k in range(8)]
    loads = [f[f"win_{k}_load"] for k in range(8)]
    open_windows = [k for k in range(8) if not blocked[k]]
    raw_min = min(range(8), key=lambda k: (loads[k], k))
    if open_windows:
        pick = min(open_windows, key=lambda k: (loads[k], k))
    else:
        pick = 0
    n_blocked = sum(blocked)
    band = 0 if n_blocked <= 2 else (1 if n_blocked <= 5 else 2)
    return {"boolean": bool(blocked[raw_min]),
            "choice": f"win_{pick}",
            "score": band}


_fam(
    "maintenance_window_pick", "window-selection-binding",
    ("Maintenance calendar for {service} ({ticket}):",
     "Window planner — {service} — {ticket}:",
     "Downtime board, {service} [{ticket}]:"),
    ("region-alpha", "region-beta", "cell-gamma", "cell-delta",
     "zone-epsilon", "zone-zeta", "site-eta", "site-theta"),
    ("MAINT", "WIN", "CAL"),
    tuple([_field(f"win_{k}_blocked", f"window {k} is blocked by a freeze or "
                  f"conflict", BOOL) for k in range(8)]
          + [_field(f"win_{k}_load", f"expected customer load in window {k}, "
                    f"percent", INT(0, 100, 5)) for k in range(8)]),
    {"boolean": "lightest_window_blocked", "choice": "selected_window",
     "score": "blocked_band"},
    {"boolean": ("Is the lightest-load window blocked?",
                 "Is the cheapest window (by load) unavailable?",
                 "Does a block sit on the lowest-load window?"),
     "choice": ("Which maintenance window should be taken?",
                "Pick the window the scheduling rule selects.",
                "Which window id does the policy choose?"),
     "score": ("Score how many windows are blocked, 0 to 2 (0-2, 3-5, 6-8).",
               "Rate calendar congestion by blocked count, 0 to 2.",
               "How congested is the calendar, 0 to 2?")},
    ({}, {"false": "the lightest window is available",
          "true": "the lightest window is blocked"}),
    {f"win_{k}": f"Take maintenance window {k}" for k in range(8)},
    _levels("0-2 windows blocked", "3-5 windows blocked",
            "6-8 windows blocked"),
    _window_oracle,
    "take the unblocked window with the lowest expected load (ties to the lower "
    "window number; if every window is blocked the documented fallback is "
    "window 0); the boolean asks whether the globally lightest window is "
    "blocked and the score bands the blocked count",)


# ---------------------------------------------------------------- queue_backpressure
# Ops cascade on a filling queue.

def _backpressure_oracle(f):
    depth, cap, lag = f["depth"], f["capacity"], f["consumer_lag_sec"]
    if f["dlq_growing"]:
        choice = "inspect_dlq"
    elif f["paused"] and depth < cap * 0.2:
        choice = "resume_producers"
    elif depth > cap * 0.8 or lag > 300:
        choice = "throttle_producers"
    else:
        choice = "keep_draining"
    severity = 0 if choice == "keep_draining" else (
        2 if choice in ("inspect_dlq", "throttle_producers") else 1)
    healthy = depth <= cap * 0.5 and not f["dlq_growing"]
    return {"boolean": healthy, "choice": choice, "score": severity}


_fam(
    "queue_backpressure", "queue-backpressure-policy",
    ("Queue status for {service} ({ticket}):",
     "Backpressure review — {service} — {ticket}:",
     "Broker board, {service} [{ticket}]:"),
    ("event-bus", "task-queue", "log-broker", "metric-pipe",
     "mail-queue", "iot-ingest", "audit-pipe", "sync-queue"),
    ("QUEUE", "BROKER", "LAG"),
    (_field("depth", "current queue depth in messages", INT(0, 10000, 50)),
     _field("capacity", "configured queue capacity in messages",
            INT(2000, 10000, 500)),
     _field("consumer_lag_sec", "consumer lag in seconds", INT(0, 600, 10)),
     _field("dlq_growing", "the dead-letter queue is growing", BOOL),
     _field("paused", "producers are currently paused", BOOL)),
    {"boolean": "queue_healthy", "choice": "backpressure_action",
     "score": "pressure_level"},
    {"boolean": ("Is this queue in a healthy state?",
                 "Is the queue inside its healthy envelope?",
                 "Is the queue healthy right now?"),
     "choice": ("What is the correct backpressure action?",
                "Pick the action the queue policy dictates.",
                "Which backpressure response applies?"),
     "score": ("Score the pressure level, 0 to 2.",
               "Rate queue pressure, 0 to 2.",
               "How pressured is this queue, 0 to 2?")},
    ({}, {"false": "the queue is outside its healthy envelope",
          "true": "the queue is healthy"}),
    {"keep_draining": "No intervention; keep draining normally",
     "resume_producers": "Depth recovered; resume paused producers",
     "throttle_producers": "Throttle producers; the queue is near capacity",
     "inspect_dlq": "Inspect the dead-letter queue before anything else"},
    _levels("nominal", "elevated", "critical"),
    _backpressure_oracle,
    "a growing DLQ is inspected first; a paused queue below 20% depth resumes "
    "producers; depth above 80% of capacity or lag over 300s throttles "
    "producers; otherwise keep draining",)


# ---------------------------------------------------------------- restore_point_pick
# Binding + negation + freshness bound over eight restore points.

def _restore_oracle(f):
    ages = [f[f"rp_{k}_age_days"] for k in range(8)]
    intact = [f[f"rp_{k}_intact"] for k in range(8)]
    bound = f["max_age_days"]
    eligible = [k for k in range(8) if intact[k] and ages[k] <= bound]
    if eligible:
        pick = min(eligible, key=lambda k: (ages[k], k))
    elif any(intact):
        pick = max((k for k in range(8) if intact[k]), key=lambda k: ages[k])
    else:
        pick = 7
    freshest = min(range(8), key=lambda k: (ages[k], k))
    return {"boolean": freshest in eligible,
            "choice": f"rp_{pick:02d}",
            "score": min(len(eligible), 5)}


_fam(
    "restore_point_pick", "restore-point-selection",
    ("Restore plan for {service} ({ticket}):",
     "Snapshot board — {service} — {ticket}:",
     "Recovery picker, {service} [{ticket}]:"),
    ("vault-store", "snap-repo", "archive-tier", "backup-pool",
     "dr-site", "replica-vault", "cold-store", "tape-bridge"),
    ("REST", "SNAP", "RCV"),
    tuple([_field(f"rp_{k}_age_days", f"restore point {k} age in days",
                  INT(1, 30)) for k in range(8)]
          + [_field(f"rp_{k}_intact", f"restore point {k} passed verification",
                    BOOL) for k in range(8)]
          + [_field("max_age_days", "the freshest acceptable age in days",
                    INT(3, 21))]),
    {"boolean": "frestest_eligible", "choice": "selected_restore",
     "score": "eligible_count"},
    {"boolean": ("Is the freshest restore point eligible under the policy?",
                 "Can the most recent snapshot be used?",
                 "Is the newest restore point both intact and inside the age bound?"),
     "choice": ("Which restore point should be used?",
                "Pick the restore point the recovery rule selects.",
                "Which restore point id applies?"),
     "score": ("Score how many restore points are eligible, capped at 5, 0 to 5.",
               "Rate recovery depth by eligible count, 0 to 5.",
               "How many restore points qualify (5+ caps), 0 to 5?")},
    ({}, {"false": "the freshest point is corrupted or too old",
          "true": "the freshest point is eligible"}),
    {f"rp_{k:02d}": f"Restore from point {k:02d}" for k in range(8)},
    _levels("no eligible restore point", "one eligible", "two eligible",
            "three eligible", "four eligible", "five or more eligible"),
    _restore_oracle,
    "eligible means intact and no older than the stated bound; pick the "
    "freshest eligible point; if none is eligible fall back to the oldest "
    "intact point, and if none is intact use point 07 as the documented "
    "deepest fallback",)


# ---------------------------------------------------------------- rollout_ring_choice
# Ring rollout: pick the next healthy ring after the current one.

def _ring_oracle(f):
    health = [f[f"ring_{k}_health"] for k in range(8)]
    current = f["current_ring"]
    healthy = [k for k in range(8) if health[k] >= 90]
    later = [k for k in healthy if k > current]
    pick = min(later) if later else current
    return {"boolean": bool(later),
            "choice": f"ring_{pick:02d}",
            "score": min(len(healthy), 5)}


_fam(
    "rollout_ring_choice", "ring-rollout-sequencing",
    ("Ring rollout for {service} ({ticket}):",
     "Staged deploy board — {service} — {ticket}:",
     "Ring status report, {service} [{ticket}]:"),
    ("desktop-agent", "mobile-fleet", "edge-nodes", "iot-fleet",
     "browser-ext", "pos-terminals", "kiosk-fleet", "gateway-mesh"),
    ("RING", "RAMP", "FLEET"),
    tuple([_field(f"ring_{k}_health", f"ring {k} health score 0-100",
                  INT(0, 100)) for k in range(8)]
          + [_field("current_ring", "the ring currently receiving the rollout",
                    INT(0, 7))]),
    {"boolean": "next_ring_ready", "choice": "next_ring",
     "score": "healthy_ring_count"},
    {"boolean": ("Is a later ring healthy enough to receive the rollout?",
                 "Does a healthy ring exist beyond the current one?",
                 "Can the rollout advance past the current ring?"),
     "choice": ("Which ring receives the rollout next?",
                "Pick the next ring under the rollout rule.",
                "Which ring id should the rollout move to?"),
     "score": ("Score how many rings are healthy (>=90), capped at 5, 0 to 5.",
               "Rate fleet health by healthy ring count, 0 to 5.",
               "How many rings are healthy (5+ caps), 0 to 5?")},
    ({}, {"false": "no later ring is healthy; hold at the current ring",
          "true": "a later healthy ring exists"}),
    {f"ring_{k:02d}": (f"Advance the rollout to ring {k:02d}" if k else
                       "Advance to ring 00 / stay at ring 00")
     for k in range(8)},
    _levels("no ring healthy", "one ring healthy", "two healthy",
            "three healthy", "four healthy", "five or more healthy"),
    _ring_oracle,
    "advance to the lowest-indexed ring after the current one whose health is "
    "at least 90; if none qualifies the rollout stays on the current ring; "
    "the score counts rings at or above 90 capped at five",)


# ---------------------------------------------------------------- cache_rescue_policy
# Ops cascade on cache degradation.

def _cache_oracle(f):
    degraded = f["hit_rate"] < f["baseline_hit"] - 15
    if f["stampede"] and not f["origin_ok"]:
        choice = "shed_to_stale"
    elif f["stampede"]:
        choice = "enable_coalescing"
    elif degraded and not f["jitter_applied"]:
        choice = "add_ttl_jitter"
    elif degraded:
        choice = "investigate_keys"
    else:
        choice = "no_action"
    severity = (int(degraded) + int(f["stampede"])
                + int(f["stampede"] and not f["origin_ok"])
                + int(degraded and not f["jitter_applied"]))
    return {"boolean": not degraded, "choice": choice,
            "score": min(severity, 3)}


_fam(
    "cache_rescue_policy", "cache-degradation-response",
    ("Cache review for {service} ({ticket}):",
     "Hit-rate board — {service} — {ticket}:",
     "Cache health check, {service} [{ticket}]:"),
    ("page-cache", "api-cache", "cdn-shield", "session-cache",
     "query-cache", "asset-cache", "token-cache", "geo-cache"),
    ("CACHE", "HIT", "TTL"),
    (_field("hit_rate", "current hit rate, percent", INT(0, 100)),
     _field("baseline_hit", "30-day baseline hit rate, percent", INT(30, 99)),
     _field("stampede", "a request stampede is in progress", BOOL),
     _field("jitter_applied", "TTL jitter is already applied", BOOL),
     _field("origin_ok", "the origin is healthy enough to absorb misses", BOOL)),
    {"boolean": "hit_rate_healthy", "choice": "cache_action",
     "score": "degradation_severity"},
    {"boolean": ("Is the hit rate inside the healthy band?",
                 "Is the cache hit rate healthy versus baseline?",
                 "Is the cache performing within 15 points of baseline?"),
     "choice": ("What is the correct cache action?",
                "Pick the response the cache policy dictates.",
                "Which cache intervention applies?"),
     "score": ("Score the degradation severity, 0 to 3.",
               "Rate how degraded the cache is, 0 to 3.",
               "How severe is the cache degradation, 0 to 3?")},
    ({}, {"false": "hit rate is more than 15 points below baseline",
          "true": "hit rate is within 15 points of baseline"}),
    {"no_action": "Cache is healthy; no action",
     "add_ttl_jitter": "Add TTL jitter to spread expirations",
     "investigate_keys": "Investigate the shifted key mix",
     "enable_coalescing": "Enable request coalescing against the stampede",
     "shed_to_stale": "Origin is unhealthy; serve stale and shed load"},
    _levels("nominal", "mild degradation", "serious degradation",
            "severe: stampede with an unhealthy origin"),
    _cache_oracle,
    "a stampede with an unhealthy origin sheds to stale; a stampede with a "
    "healthy origin coalesces requests; a degraded hit rate without jitter "
    "adds TTL jitter, with jitter it investigates the key mix; otherwise no "
    "action",)


# ---------------------------------------------------------------- audit_trail_gap
# Compliance: audit-log gaps versus stated tolerance.

def _audit_gap_oracle(f):
    if not f["gap_explained"] and f["writes_continued"] and f["regulated_data"]:
        choice = "halt_and_escalate"
    elif not f["gap_explained"] and f["writes_continued"]:
        choice = "pause_writes"
    elif not f["gap_explained"]:
        choice = "open_investigation"
    elif f["gap_minutes"] > f["max_gap"] * 4:
        choice = "open_investigation"
    else:
        choice = "annotate_and_close"
    severity = (int(f["gap_minutes"] > f["max_gap"]) + int(not f["gap_explained"])
                + int(f["writes_continued"]) + int(f["regulated_data"])
                + int(f["gap_minutes"] > f["max_gap"] * 4))
    return {"boolean": f["gap_minutes"] <= f["max_gap"],
            "choice": choice, "score": min(severity, 4)}


_fam(
    "audit_trail_gap", "audit-gap-triage",
    ("Audit gap review for {service} ({ticket}):",
     "Log continuity board — {service} — {ticket}:",
     "Audit trail check, {service} [{ticket}]:"),
    ("pay-ledger", "order-ledger", "access-log", "trade-blotter",
     "claims-log", "rx-audit", "vote-tally", "tax-ledger"),
    ("AUDIT", "GAP", "LOG"),
    (_field("gap_minutes", "unexplained audit gap in minutes", INT(0, 720, 10)),
     _field("max_gap", "the tolerated gap in minutes", INT(15, 120, 5)),
     _field("gap_explained", "the gap has a documented explanation", BOOL),
     _field("writes_continued", "writes continued during the gap", BOOL),
     _field("regulated_data", "the stream carries regulated records", BOOL)),
    {"boolean": "within_tolerance", "choice": "gap_disposition",
     "score": "gap_severity"},
    {"boolean": ("Is the gap inside the tolerated bound?",
                 "Is this audit gap within tolerance?",
                 "Does the gap stay under the stated maximum?"),
     "choice": ("What is the correct disposition for this gap?",
                "Pick the audit-gap outcome the policy dictates.",
                "Which disposition applies?"),
     "score": ("Score the gap severity, 0 to 4.",
               "Rate how severe this audit gap is, 0 to 4.",
               "How severe is the gap, 0 to 4?")},
    ({}, {"false": "the gap exceeds the tolerated bound",
          "true": "the gap is within tolerance"}),
    {"annotate_and_close": "Explained and bounded; annotate and close",
     "open_investigation": "Unexplained or extreme gap; open an investigation",
     "pause_writes": "Unexplained gap with ongoing writes; pause writes",
     "halt_and_escalate": "Regulated data, unexplained, still writing; halt "
     "and escalate"},
    _levels("trivial", "minor", "moderate", "serious",
            "severe: regulated unexplained write-through gap"),
    _audit_gap_oracle,
    "an unexplained gap on a regulated stream with ongoing writes halts and "
    "escalates; unexplained with writes pauses them; unexplained or beyond "
    "four times the bound opens an investigation; otherwise annotate and close",)


NEW_FAMILY_NAMES = tuple(spec["family"] for spec in FAMILY_DEFS
                         if spec["family"] not in V3_FAMILY_NAMES)
FAMILIES = tuple(spec["family"] for spec in FAMILY_DEFS)
FAMILY_MAP = {spec["family"]: spec for spec in FAMILY_DEFS}
QIDS = {qtype: {spec["family"]: spec["qids"][qtype] for spec in FAMILY_DEFS}
        for qtype in QUESTION_TYPES}

assert len(FAMILIES) == len(set(FAMILIES)) == 36, "v4 expects exactly 36 families"
assert len(NEW_FAMILY_NAMES) == 18


# --------------------------------------------------------------------------------------
# oracle validation, deterministic fact sampling, coverage selection
# --------------------------------------------------------------------------------------

def _fact_key(facts):
    return json.dumps(facts, sort_keys=True, separators=(",", ":"))


def _candidate_keys(spec, qtype):
    if qtype == "boolean":
        return ["false", "true"]
    if qtype == "choice":
        return list(spec["choice_criteria"])
    return [str(i) for i in range(len(spec["score_criteria"]))]


def _check_distribution(spec, qtype, dist, label):
    keys = _candidate_keys(spec, qtype)
    if not isinstance(dist, dict) or set(dist) != set(keys):
        raise ValueError(f"{label}: distribution keys must be exactly the "
                         f"{qtype} candidates")
    total = 0.0
    for key, p in dist.items():
        if isinstance(p, bool) or not isinstance(p, (int, float)) \
                or not 0.0 <= p <= 1.0:
            raise ValueError(f"{label}: probability for {key!r} is not in [0,1]")
        total += float(p)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"{label}: probabilities sum to {total!r}, not 1")


def _oracle_checked(spec, facts):
    """Run the family oracle and enforce the answer contract on every fact set."""
    gold = spec["oracle"](facts)
    if spec.get("kind") == "chance":
        for qtype in QUESTION_TYPES:
            _check_distribution(spec, qtype, gold.get(qtype),
                                f"{spec['family']}:{qtype}")
        return gold
    if type(gold.get("boolean")) is not bool:
        raise ValueError(f"{spec['family']}: oracle boolean is not a JSON bool")
    if gold.get("choice") not in spec["choice_criteria"]:
        raise ValueError(f"{spec['family']}: oracle choice {gold.get('choice')!r} "
                         "is not a declared candidate")
    score = gold.get("score")
    if type(score) is not int or not 0 <= score < len(spec["score_criteria"]):
        raise ValueError(f"{spec['family']}: oracle score {score!r} is outside "
                         "the declared rubric")
    return gold


def all_gold(family, facts):
    return _oracle_checked(FAMILY_MAP[family], facts)


def gold_for(family, facts, qtype):
    return QIDS[qtype][family], all_gold(family, facts)[qtype]


def fact_candidates(spec, seed):
    """Deterministic candidate base states for one family.

    Families may declare a ``sampler(rng)`` hook that draws fact sets directly
    (used to spread high-cardinality gold labels); otherwise the domain is
    enumerated exhaustively when small, or uniformly sampled when large.
    """
    rng = random.Random(f"nanojev-engineering-catalog-v4:{seed}:{spec['family']}")
    combos, seen = [], set()
    if spec.get("sampler") is not None:
        attempts = 0
        while len(combos) < SAMPLE_CAP and attempts < SAMPLE_CAP * 4:
            attempts += 1
            facts = dict(spec["sampler"](rng))
            tag = _fact_key(facts)
            if tag in seen:
                continue
            seen.add(tag)
            combos.append(facts)
        return [{"facts": facts, "gold": _oracle_checked(spec, facts)}
                for facts in combos]
    keys = [f["key"] for f in spec["fields"]]
    values = [domain_values(f["domain"]) for f in spec["fields"]]
    total = 1
    for v in values:
        total *= len(v)
    if total <= ENUMERATE_CAP:
        combos = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
        rng.shuffle(combos)
    else:
        attempts = 0
        while len(combos) < SAMPLE_CAP and attempts < SAMPLE_CAP * 4:
            attempts += 1
            facts = {key: rng.choice(domain) for key, domain in zip(keys, values)}
            tag = _fact_key(facts)
            if tag in seen:
                continue
            seen.add(tag)
            combos.append(facts)
    return [{"facts": facts, "gold": _oracle_checked(spec, facts)}
            for facts in combos]


def _choice_label(gold):
    """Hashable per-family label used for coverage/balancing passes."""
    choice = gold["choice"]
    if isinstance(choice, dict):
        return max(sorted(choice), key=lambda k: choice[k])
    return choice


def select_for_coverage(spec, candidates, target):
    """Order candidates so labels are covered early, then balance choice labels."""
    pool = list(candidates)
    selected, picked_ids = [], set()

    def take(pred):
        for cand in pool:
            if id(cand) in picked_ids or not pred(cand["gold"]):
                continue
            picked_ids.add(id(cand))
            selected.append(cand)
            return True
        return False

    chance = spec.get("kind") == "chance"
    if not chance:
        # score/boolean coverage first: these are <=12 takes and must land inside
        # the emitted prefix even when a family declares hundreds of candidates
        for level in range(len(spec["score_criteria"])):
            take(lambda g, lv=level: g["score"] == lv)
        for value in (False, True):
            take(lambda g, v=value: g["boolean"] is v)
    coverage_min = spec.get("choice_coverage_min")
    keys = list(spec["choice_criteria"])
    limit = coverage_min if coverage_min is not None else len(keys)
    covered = 0
    for key in keys:
        if covered >= limit:
            break
        if take(lambda g, k=key: _choice_label(g) == k):
            covered += 1
    # Balance the choice label histogram while filling to the soft cap.
    buckets = defaultdict(list)
    for cand in pool:
        if id(cand) not in picked_ids:
            buckets[_choice_label(cand["gold"])].append(cand)
    soft_cap = min(len(pool), max(target * 3, target + len(keys)))
    while len(selected) < soft_cap and buckets:
        smallest = min(buckets, key=lambda k: sum(
            1 for c in selected if _choice_label(c["gold"]) == k))
        if not buckets[smallest]:
            del buckets[smallest]
            continue
        cand = buckets[smallest].pop(0)
        picked_ids.add(id(cand))
        selected.append(cand)
    selected.extend(c for c in pool if id(c) not in picked_ids)
    return selected


def flipping_mutations(spec, base_facts, base_gold):
    """Every one-fact change that flips at least one question type, with its flips."""
    out = []
    for field in spec["fields"]:
        key = field["key"]
        for value in domain_values(field["domain"]):
            if value == base_facts[key]:
                continue
            variant = deepcopy(base_facts)
            variant[key] = value
            variant_gold = _oracle_checked(spec, variant)
            flips = tuple(qt for qt in QUESTION_TYPES
                          if base_gold[qt] != variant_gold[qt])
            if flips:
                out.append({"key": key, "value": value, "flips": flips,
                            "variant_facts": variant,
                            "variant_gold": variant_gold})
    out.sort(key=lambda m: (-len(m["flips"]), m["key"], str(m["value"])))
    return out


# --------------------------------------------------------------------------------------
# state rendering (V3 templates) with a V4 render namespace and wider ticket range
# --------------------------------------------------------------------------------------

def make_render(spec, seed, ordinal, attempt=0):
    rng = random.Random(
        f"nanojev-engineering-v4-render:{seed}:{spec['family']}:{ordinal}:{attempt}")
    order = [f["key"] for f in spec["fields"]]
    rng.shuffle(order)
    return {"template": rng.randrange(5),
            "heading": spec["headings"][rng.randrange(len(spec["headings"]))],
            "service": spec["services"][rng.randrange(len(spec["services"]))],
            "ticket": f"{spec['tickets'][rng.randrange(len(spec['tickets']))]}"
                      f"-{rng.randrange(1000, 99999)}",
            "field_order": order,
            "text_variant": rng.randrange(64)}


# --------------------------------------------------------------------------------------
# pair generation with computed flip proofs
# --------------------------------------------------------------------------------------

def _validate_specs():
    seen = set()
    for spec in FAMILY_DEFS:
        family = spec["family"]
        if family in seen:
            raise ValueError(f"duplicate family {family!r}")
        seen.add(family)
        keys = [f["key"] for f in spec["fields"]]
        if len(keys) != len(set(keys)):
            raise ValueError(f"{family}: duplicate fact keys")
        for qtype in QUESTION_TYPES:
            if qtype not in spec["qids"] or qtype not in spec["instructions"]:
                raise ValueError(f"{family}: missing qid/instructions for {qtype}")
            if not spec["instructions"][qtype]:
                raise ValueError(f"{family}: empty instruction variants for {qtype}")
        if not 2 <= len(spec["choice_criteria"]) <= 255:
            raise ValueError(f"{family}: choice criteria out of contract range")
        if not 2 <= len(spec["score_criteria"]) <= 10:
            raise ValueError(f"{family}: score criteria out of contract range")
        for criteria in spec["boolean_criteria"]:
            if set(criteria) - {"false", "true"}:
                raise ValueError(f"{family}: bad boolean criteria keys")
        if spec.get("choice_coverage_min") is not None and not (
                1 <= spec["choice_coverage_min"] <= len(spec["choice_criteria"])):
            raise ValueError(f"{family}: choice_coverage_min out of range")
        if spec.get("kind") == "chance" and spec.get("sampler") is None \
                and spec.get("choice_coverage_min") is not None:
            raise ValueError(f"{family}: coverage_min is meaningless for chance "
                             "families")
    return len(FAMILY_DEFS)


def _coverage_errors(spec, golds):
    """Label/distribution coverage requirements for one family's pair members."""
    if spec.get("kind") == "chance":
        distinct = {qt: len({json.dumps(g[qt], sort_keys=True) for g in golds})
                    for qt in QUESTION_TYPES}
        need = {"boolean": 2, "choice": 4, "score": 2}
        short = {qt: (distinct[qt], need[qt]) for qt in QUESTION_TYPES
                 if distinct[qt] < need[qt]}
        if short:
            return f"chance distribution coverage too thin: {short}"
        return None
    required = (spec.get("choice_coverage_min")
                if spec.get("choice_coverage_min") is not None
                else len(spec["choice_criteria"]))
    distinct_choice = {g["choice"] for g in golds}
    missing_choice = len(spec["choice_criteria"]) - len(distinct_choice)
    if len(distinct_choice) < required:
        return (f"choice coverage {len(distinct_choice)} below required "
                f"{required} (uncovered {missing_choice})")
    missing_scores = sorted(set(range(len(spec["score_criteria"])))
                            - {g["score"] for g in golds})
    missing_bool = sorted({False, True} - {g["boolean"] for g in golds})
    if missing_scores or missing_bool:
        return (f"label coverage gaps: score {missing_scores}, "
                f"boolean {missing_bool}")
    return None


def pairs_for(seed=DEFAULT_SEED):
    """Generate base/variant pairs; every declared flip is recomputed and asserted."""
    _validate_specs()
    pairs = []
    used_states = set()
    for spec in FAMILY_DEFS:
        family = spec["family"]
        candidates = select_for_coverage(
            spec, fact_candidates(spec, seed), PAIRS_PER_FAMILY)
        made = 0
        for ordinal, cand in enumerate(candidates):
            if made >= PAIRS_PER_FAMILY:
                break
            base_facts, base_gold = cand["facts"], cand["gold"]
            render, base_state = None, None
            for attempt in range(8):
                trial = make_render(spec, seed, ordinal, attempt)
                text = v3.render_state(base_facts, spec, trial)
                if text not in used_states:
                    render, base_state = trial, text
                    break
            if render is None:
                continue
            mutations = flipping_mutations(spec, base_facts, base_gold)
            if not mutations:
                continue
            start = (ordinal * 5 + abs(seed)) % len(mutations)
            picked = None
            for offset in range(len(mutations)):
                mutation = mutations[(start + offset) % len(mutations)]
                variant_state = v3.render_state(mutation["variant_facts"],
                                              spec, render)
                if variant_state not in used_states:
                    picked = (mutation, variant_state)
                    break
            if picked is None:
                continue
            mutation, variant_state = picked
            delta = v1.fact_delta(base_facts, mutation["variant_facts"])
            if len(delta) != 1 or delta[0][0] != mutation["key"]:
                raise ValueError(f"{family}#{ordinal}: mutation must change exactly "
                                 "the named fact")
            if delta[0][1] == delta[0][2]:
                raise ValueError(f"{family}#{ordinal}: mutation does not change "
                                 "the value")
            used_states.add(base_state)
            used_states.add(variant_state)
            rule_id = f"{family}-v4r{made:03d}"
            pairs.append({
                "pair_id": f"ej4-{family}-p{made:03d}",
                "rule_id": rule_id, "state_id": rule_id,
                "family": family, "feature": spec["feature"],
                "mutated_fact": mutation["key"],
                "value_before": delta[0][1], "value_after": delta[0][2],
                "flip_question_types": mutation["flips"],
                "declared_flip": list(mutation["flips"]),
                "gold_before": base_gold, "gold_after": mutation["variant_gold"],
                "base_facts": base_facts, "variant_facts": mutation["variant_facts"],
                "render": render, "rule": spec,
            })
            made += 1
        if made < MIN_PAIRS_PER_FAMILY:
            raise ValueError(f"{family}: only {made} pairs could be built "
                             f"(minimum {MIN_PAIRS_PER_FAMILY})")
        family_golds = [pair["gold_before"] for pair in pairs
                        if pair["family"] == family]
        family_golds += [pair["gold_after"] for pair in pairs
                         if pair["family"] == family]
        coverage = _coverage_errors(spec, family_golds)
        if coverage:
            raise ValueError(f"{family}: {coverage}")
    return pairs


# --------------------------------------------------------------------------------------
# items, components, splits
# --------------------------------------------------------------------------------------

def make_items(pair, member):
    family = pair["family"]
    spec = pair["rule"]
    facts = pair[f"{member}_facts"]
    request = v3.build_request(spec, facts, pair["render"], pair["state_id"])
    chance = spec.get("kind") == "chance"
    items = []
    for qtype in QUESTION_TYPES:
        qid, gold = gold_for(family, facts, qtype)
        criteria_values = request["states"][0]["questions"][qid].get("criteria", {})
        if chance:
            keys = _candidate_keys(spec, qtype)
            expected = {
                "answer_type": qtype,
                "candidate_keys": keys,
                "gold": None,
                "gold_index": None,
                "distribution": dict(gold),
                "distribution_kind": CHANCE_KIND,
                "calibrated": False,
            }
        else:
            vector = v1.probability_vector(qtype, criteria_values, gold)
            expected = {
                "answer_type": qtype,
                "candidate_keys": vector["keys"],
                "gold": gold,
                "gold_index": vector["gold_index"],
                "distribution": vector["probabilities"],
                "distribution_kind": "hard_label",
                "calibrated": False,
            }
        item_id = f"{pair['pair_id']}-{qtype}" + ("-v" if member == "variant" else "")
        items.append({
            "schema_version": ITEM_SCHEMA,
            "item_id": item_id,
            "family": family,
            "feature": pair["feature"],
            "question_type": qtype,
            "state_id": pair["state_id"],
            "qid": qid,
            "split": pair["split"],
            "source_group_id": pair["source_group_id"],
            "pair_id": pair["pair_id"],
            "member": member,
            "request": request,
            "expected": expected,
            "provenance": {
                "source_id": SOURCE_ID,
                "rule_id": pair["rule_id"],
                "catalog_version": CATALOG_VERSION,
                "rule": spec["why"],
                "why_correct": (f"under the frozen {family} rule the facts in this "
                                f"state fix {qid}={gold!r}" if not chance else
                                f"under the frozen {family} lottery the facts in "
                                f"this state fix the conditional distribution of "
                                f"{qid}; no outcome was observed"),
                "fact_basis": {key: facts[key] for key in sorted(facts)},
                "fact_keys": sorted(facts),
                "surface_template": pair["render"]["template"],
                "authoring": ("programmatic_base_state_from_family_domain"
                              if member == "base" else
                              "programmatic_one_fact_mutation_of_a_sampled_state"),
                "label_kind": ("analytic_conditional_distribution" if chance
                               else "deterministic_truth"),
                "human_reviewed": False,
                "derived_from_evaluation_corpus": False,
                "training_authorized": False,
            },
            "contrastive": {
                "pair_id": pair["pair_id"],
                "member": member,
                "variant_of": (f"{pair['pair_id']}-{qtype}-base"
                               if member == "variant" else None),
                "mutated_fact": pair["mutated_fact"],
                "value_before": pair["value_before"],
                "value_after": pair["value_after"],
                "flip_question_types": list(pair["flip_question_types"]),
                "is_flip_question": qtype in pair["flip_question_types"],
            },
        })
    for item in items:
        item["item_content_sha256"] = digest_value(item)
    return items


def derive_manifest(seed=DEFAULT_SEED):
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    _validate_specs()
    pairs = pairs_for(seed)

    dropped = sorted({pair["rule_id"] for pair in pairs
                      if _evaluation_derived(pair["rule"]["family"])
                      or _evaluation_derived(SOURCE_ID)})
    pairs = [pair for pair in pairs if pair["rule_id"] not in dropped]

    members = {}
    components = v2._Components()
    fingerprint_nodes = {}
    for pair in pairs:
        pair["split"], pair["source_group_id"] = "train", "pending"
        nodes = []
        for member in ("base", "variant"):
            items = make_items(pair, member)
            node = (pair["pair_id"], member)
            members[node] = items
            nodes.append(node)
            for item in items:
                fp = _visible_fingerprint(item)
                if fp in fingerprint_nodes:
                    components.union(node, fingerprint_nodes[fp])
                else:
                    fingerprint_nodes[fp] = node
        components.union(*nodes)

    grouped = {}
    for node in members:
        grouped.setdefault(components.find(node), []).append(node)

    catalog_order = {pair["pair_id"]: index for index, pair in enumerate(pairs)}
    per_family = {}
    for comp_nodes in grouped.values():
        first = min(catalog_order[node[0]] for node in comp_nodes)
        family = pairs[catalog_order[comp_nodes[0][0]]]["family"]
        per_family.setdefault(family, []).append((first, comp_nodes))

    family_parity = {family: index for index, family in enumerate(FAMILIES)}
    for family, comps in per_family.items():
        cycle = SPLIT_CYCLES[family_parity[family] % 2]
        for index, (_, comp_nodes) in enumerate(sorted(comps)):
            split = cycle[(index + abs(seed)) % len(cycle)]
            fps = sorted({_visible_fingerprint(item) for node in comp_nodes
                          for item in members[node]})
            group_id = "ejc-v4-" + digest_value(
                {"schema": SOURCE_GROUP_SCHEMA, "canonical_inputs": fps})[:32]
            for node in comp_nodes:
                for item in members[node]:
                    item["split"] = split
                    item["source_group_id"] = group_id
                    item["provenance"]["component_size_pairs"] = len(
                        {n[0] for n in comp_nodes})
                    item.pop("item_content_sha256", None)
                    item["item_content_sha256"] = digest_value(item)
            for pair in pairs:
                if any(node[0] == pair["pair_id"] for node in comp_nodes):
                    pair["split"] = split
                    pair["source_group_id"] = group_id

    items = [item for node in members for item in members[node]]
    for item in items:
        v1.guard_source(item["provenance"]["source_id"], item,
                        f"item {item['item_id']}")

    pair_records = [{
        "pair_id": pair["pair_id"], "rule_id": pair["rule_id"],
        "state_id": pair["state_id"], "family": pair["family"],
        "feature": pair["feature"], "split": pair["split"],
        "source_group_id": pair["source_group_id"],
        "mutated_fact": pair["mutated_fact"],
        "value_before": pair["value_before"], "value_after": pair["value_after"],
        "flip_question_types": list(pair["flip_question_types"]),
        "gold_before": pair["gold_before"], "gold_after": pair["gold_after"],
        "item_ids": [item["item_id"] for item in items
                     if item["pair_id"] == pair["pair_id"]],
        "declared_flip": list(pair["declared_flip"]),
    } for pair in pairs]

    split_geometry = {}
    for split in SPLITS:
        split_items = [item for item in items if item["split"] == split]
        split_geometry[split] = {
            "items": len(split_items),
            "pairs": sum(1 for pair in pairs if pair["split"] == split),
            "source_groups": sorted({item["source_group_id"]
                                     for item in split_items}),
            "families": sorted({item["family"] for item in split_items}),
            "question_types": sorted({item["question_type"]
                                      for item in split_items}),
        }

    construction = {
        "version": "engineering-judgment-rule-v4",
        "statement": ("A pair is two served-contract request bodies whose states "
                      "differ in exactly one fact leaf of the family's declared "
                      "domain; the expected answers (hard gold for deterministic "
                      "families, analytic conditional distributions for chance "
                      "families) of both members are computed from their own "
                      "facts with the family's frozen oracle, and the declared "
                      "flip question types are required to differ. A pair whose "
                      "declared flip does not hold is never emitted."),
        "base_states": ("programmatically sampled/enumerated fact sets rendered "
                        "through the five V3 surface templates under a V4 render "
                        "namespace (fresh ticket range 1000-99999), per-pair "
                        "entities, headings, field order and phrasing variants; "
                        "high-cardinality families may declare a sampler that "
                        "spreads gold labels across the candidate range"),
        "split_unit": ("canonical-input connected component: both members of a "
                       "contrastive pair plus every member sharing an identical "
                       "visible state+question always share one split"),
        "split_rule": ("deterministic per-family component position + seed "
                       "offset over two alternating 20-slot cycles: even-indexed "
                       "families use the V3 cycle (11 train/3 dev/2 calibration/"
                       "4 test), odd-indexed families swap dev and test slots, "
                       "yielding ~55/17.5/10/17.5 overall; content never "
                       "selects a split"),
        "question_types": list(QUESTION_TYPES),
        "families": list(FAMILIES),
        "v3_families_reused_as_rules": list(V3_FAMILY_NAMES),
        "new_families": list(NEW_FAMILY_NAMES),
        "family_count_basis": ("36 distinct families in v4 = the 18 v3 rule sets "
                               "re-rendered under v4 ids/seeds/surfaces + 18 "
                               "newly declared families"),
        "label_kinds": {
            "deterministic": ("one-hot deterministic_truth gold; every family "
                              "must cover all score levels and both boolean "
                              "values plus either all choice candidates or its "
                              "declared choice_coverage_min"),
            "chance": ("analytic conditional distribution "
                       "(programmatic_conditional_distribution); no observed "
                       "outcome exists so gold/gold_index are null and "
                       "gold_label_kind is unobserved; coverage requires >=4 "
                       "distinct choice distributions across pair members"),
        },
        "views": {
            "items/<split>.jsonl": ("audit view: one object per (member x question "
                                    "type) with the full request, declared gold/"
                                    "distribution, contrast and provenance"),
            "trainer_view/<split>.jsonl": ("row view in the "
                                           "scripts/train_pipeline_decisions.py "
                                           "contract, generated from the same items"),
        },
        "provenance_field": ("every item records source_id, rule_id, the family "
                             "rule, why_correct, the full fact basis, the surface "
                             "template and the authoring mode"),
    }

    provenance = dict(v1.PROVENANCE)

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "training_data_prerequisite_no_training_authorized",
        "corpus_role": "training_data_prerequisite_only",
        "created_by": "scripts/build_engineering_corpus_v4.py",
        "catalog_version": CATALOG_VERSION,
        "scale_up": {
            "motivation": ("X2 ablation (W47): readout is not the binding "
                           "constraint; corpus scale/diversity is the "
                           "higher-value axis (task X4)"),
            "canonical_input_components": len(grouped),
            "families": len(FAMILIES),
            "new_families": len(NEW_FAMILY_NAMES),
            "surface_templates": 5,
            "v1_corpus_preserved": "research/engineering_judgment_corpus_v1 unchanged",
            "v2_corpus_preserved": "research/engineering_judgment_corpus_v2 unchanged",
            "v3_corpus_preserved": "research/engineering_judgment_corpus_v3 unchanged",
            "heldout_v1_preserved": "research/engineering_heldout_v1 unchanged",
        },
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": seed,
        "contract": {
            "validator": "scripts/predict_toy_decisions.py:validate_request",
            "shape": ('{"states":[{"id","state","questions":'
                      '{qid:{"type","instructions","criteria"}}}]}'),
            "trainer_row_contract": f"{v1.TRAINER}:validate_training_row",
            "views": {"item": f"{ITEM_DIR}/<split>.jsonl",
                      "trainer": f"{TRAINER_DIR}/<split>.jsonl"},
            "validated": True,
        },
        "construction_rule": construction,
        "provenance": provenance,
        "exclusions": {
            "policy": ("V4 keeps the V1/V2/V3 refusal wall: no source path naming "
                       "an evaluation corpus or the pre-registered abstention "
                       "survey, no reserved evaluation-corpus token in emitted "
                       "content, and no evaluation-derived declared source. "
                       "Evaluation cohorts and external model outputs are never "
                       "a training, validation, calibration or selection source."),
            "refused_path_marker_count": len(v1.FORBIDDEN_PATH_MARKERS),
            "refused_path_markers_sha256": digest_value(list(v1.FORBIDDEN_PATH_MARKERS)),
            "reserved_token_count": len(v1.RESERVED_TOKENS),
            "reserved_tokens_sha256": digest_value(list(v1.RESERVED_TOKENS)),
            "evaluation_source_markers": list(v2.EVALUATION_SOURCE_MARKERS),
            "removed_rules": dropped,
            "uses_evaluation_corpus_as_source": False,
        },
        "source_group_count": len({item["source_group_id"] for item in items}),
        "pair_count": len(pairs),
        "item_count": len(items),
        "counts_by_family": _family_counts(items, pairs),
        "counts_by_split": {split: split_geometry[split]["items"]
                            for split in SPLITS},
        "counts_by_question_type": {qtype: sum(1 for item in items
                                             if item["question_type"] == qtype)
                                    for qtype in QUESTION_TYPES},
        "split_geometry": split_geometry,
        "pairs": pair_records,
        "items": items,
        "item_digest": digest_value(items),
        "pair_digest": digest_value(pair_records),
        "content_sha256": None,
    }
    manifest["content_sha256"] = digest_value(
        {key: value for key, value in manifest.items()
         if key not in ("content_sha256", "builder_sha256")})
    return manifest


def _family_counts(items, pairs):
    counts = {}
    for family in FAMILIES:
        family_items = [item for item in items if item["family"] == family]
        family_pairs = [pair for pair in pairs if pair["family"] == family]
        counts[family] = {
            "items": len(family_items),
            "pairs": len(family_pairs),
            "kind": FAMILY_MAP[family].get("kind", "deterministic"),
            "choice_candidates": len(FAMILY_MAP[family]["choice_criteria"]),
            "boolean": sum(1 for item in family_items
                           if item["question_type"] == "boolean"),
            "choice": sum(1 for item in family_items
                          if item["question_type"] == "choice"),
            "score": sum(1 for item in family_items
                         if item["question_type"] == "score"),
            "splits": {split: sum(1 for item in family_items
                                  if item["split"] == split)
                       for split in SPLITS},
        }
    return counts


# --------------------------------------------------------------------------------------
# trainer view (train_pipeline_decisions row contract + provenance block)
# --------------------------------------------------------------------------------------

def trainer_row(item):
    state = item["request"]["states"][0]
    qid = item["qid"]
    chance = item["expected"]["distribution_kind"] == CHANCE_KIND
    row = {
        "id": item["item_id"],
        "state_id": item["state_id"],
        "family_id": item["family"],
        "split": item["split"],
        "state": state["state"],
        "questions": {qid: state["questions"][qid]},
        "gold": {} if chance else {qid: item["expected"]["gold"]},
        "gold_probs": {qid: dict(item["expected"]["distribution"])},
        "gold_probs_kind": {qid: (CHANCE_KIND if chance
                                  else "deterministic_truth")},
        "gold_label_kind": {qid: ("unobserved" if chance
                                  else "deterministic_truth")},
        "schema_version": TRAINER_ROW_SCHEMA,
        "metadata": {
            "source_group_id": item["source_group_id"],
            "pair_id": item["pair_id"],
            "member": item["member"],
            "variant_of": item["contrastive"]["variant_of"],
            "question_type": item["question_type"],
            "feature": item["feature"],
            "mutated_fact": item["contrastive"]["mutated_fact"],
            "is_flip_question": item["contrastive"]["is_flip_question"],
            "provenance_source_id": item["provenance"]["source_id"],
            "provenance_rule_id": item["provenance"]["rule_id"],
            "why_correct": item["provenance"]["why_correct"],
            "authoring": item["provenance"]["authoring"],
            "derived_from_evaluation_corpus": False,
            "training_authorized_by_this_corpus": False,
        },
        "provenance": {
            "authoring": item["provenance"]["authoring"],
            "cohort": "engineering_judgment_corpus_v4",
            "catalog_version": CATALOG_VERSION,
            "derived_from_evaluation_corpus": False,
            "human_reviewed": False,
            "license": "CC0-1.0",
            "training_authorized": False,
        },
    }
    return row


def trainer_rows(manifest):
    return [trainer_row(item) for item in manifest["items"]]


# --------------------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------------------

def validate_item(item):
    errors = []
    item_id = item.get("item_id")
    try:
        validate_request(item["request"])
    except Exception as error:
        return [f"{item_id}: request rejected by the served validator: {error}"]
    state = item["request"]["states"][0]
    if state["id"] != item["state_id"]:
        errors.append(f"{item_id}: state id mismatch")
    qid = item["qid"]
    if qid not in state["questions"]:
        return errors + [f"{item_id}: qid {qid!r} missing from the request"]
    body = state["questions"][qid]
    if body["type"] != item["question_type"]:
        errors.append(f"{item_id}: question type mismatch")
    if set(body) - ALLOWED_QUESTION_KEYS:
        errors.append(f"{item_id}: unsupported question keys")
    if item["family"] not in FAMILIES:
        errors.append(f"{item_id}: unknown family {item['family']!r}")
    if item["split"] not in SPLITS:
        errors.append(f"{item_id}: split must be one of {list(SPLITS)}")
    expected = item["expected"]
    chance = expected.get("distribution_kind") == CHANCE_KIND
    if set(expected["distribution"]) != set(expected["candidate_keys"]):
        errors.append(f"{item_id}: distribution keys must equal candidate keys")
    elif chance:
        if expected["gold"] is not None or expected["gold_index"] is not None:
            errors.append(f"{item_id}: chance items must not invent a hard gold")
        try:
            _check_distribution(FAMILY_MAP[item["family"]],
                                item["question_type"], expected["distribution"],
                                item_id)
        except Exception as error:
            errors.append(f"{item_id}: {error}")
    else:
        if expected["distribution_kind"] != "hard_label":
            errors.append(f"{item_id}: deterministic items must be hard_label")
        if not 0 <= expected["gold_index"] < len(expected["candidate_keys"]):
            errors.append(f"{item_id}: gold_index is outside the candidate list")
        elif item["question_type"] == "boolean":
            if expected["candidate_keys"][expected["gold_index"]] != \
                    ("true" if expected["gold"] is True else "false"):
                errors.append(f"{item_id}: gold_index does not address the gold")
        elif item["question_type"] == "choice":
            if expected["candidate_keys"][expected["gold_index"]] != expected["gold"]:
                errors.append(f"{item_id}: gold_index does not address the gold")
        elif expected["gold_index"] != expected["gold"]:
            errors.append(f"{item_id}: gold_index does not address the gold level")
        if sum(1 for p in expected["distribution"].values() if p == 1.0) != 1:
            errors.append(f"{item_id}: deterministic distribution must be one-hot")
    if expected["calibrated"] is not False:
        errors.append(f"{item_id}: distribution must stay uncalibrated")
    if item["family"] in FAMILIES:
        recomputed = gold_for(item["family"], item["provenance"]["fact_basis"],
                              item["question_type"])
        if recomputed[0] != qid:
            errors.append(f"{item_id}: qid mismatch with the family oracle")
        elif chance:
            if recomputed[1] != expected["distribution"]:
                errors.append(f"{item_id}: the declared distribution is not what "
                              "the family oracle computes")
        elif recomputed[1] != expected["gold"]:
            errors.append(f"{item_id}: the declared gold is not what the family "
                          f"oracle computes ({recomputed[0]}={recomputed[1]!r})")
    hits = v1.reserved_hits(json.dumps(item, ensure_ascii=False, sort_keys=True))
    if hits:
        errors.append(f"{item_id}: reserved evaluation tokens {hits}")
    if v1.forbidden_path_reason(item["provenance"]["source_id"]):
        errors.append(f"{item_id}: refused evaluation source")
    if item["provenance"]["derived_from_evaluation_corpus"] is not False:
        errors.append(f"{item_id}: derived_from_evaluation_corpus must be false")
    if item["provenance"]["training_authorized"] is not False:
        errors.append(f"{item_id}: training_authorized must be false")
    if item["item_content_sha256"] != digest_value(
            {key: value for key, value in item.items()
             if key != "item_content_sha256"}):
        errors.append(f"{item_id}: item_content_sha256 does not match the item")
    return errors


def _answer_payload(item):
    expected = item["expected"]
    return (json.dumps(expected["gold"], sort_keys=True),
            json.dumps(expected["distribution"], sort_keys=True))


def validate_manifest(manifest):
    """Re-derive the corpus from the seed and fail on any hand edit."""
    errors = []
    if not isinstance(manifest, dict):
        return ["manifest must be a JSON object"]
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        errors.append(f"schema_version must be {MANIFEST_SCHEMA!r}")
    if manifest.get("created_by") != "scripts/build_engineering_corpus_v4.py":
        errors.append("created_by must name this builder")
    if manifest.get("status") != "training_data_prerequisite_no_training_authorized":
        errors.append("status must record that no training is authorized")
    if manifest.get("corpus_role") != "training_data_prerequisite_only":
        errors.append("corpus_role must be training_data_prerequisite_only")
    if manifest.get("catalog_version") != CATALOG_VERSION:
        errors.append("catalog_version must match the builder")
    for flag, value in v1.PROVENANCE.items():
        if isinstance(value, bool) and \
                manifest.get("provenance", {}).get(flag) is not value:
            errors.append(f"provenance.{flag} must be {value}")
    if manifest.get("construction_rule", {}).get("split_unit") is None:
        errors.append("construction_rule must record the component split unit")
    if manifest.get("contract", {}).get("validator") != \
            "scripts/predict_toy_decisions.py:validate_request":
        errors.append("contract.validator must name the real served validator")
    exclusions = manifest.get("exclusions", {})
    if exclusions.get("uses_evaluation_corpus_as_source") is not False:
        errors.append("exclusions.uses_evaluation_corpus_as_source must be false")
    if type(manifest.get("seed")) is not int:
        errors.append("seed must be an integer")
        return errors

    try:
        rederived = derive_manifest(manifest["seed"])
    except Exception as error:
        errors.append(f"re-derivation raised {type(error).__name__}: {error}")
        return errors

    difference = v1._first_difference(manifest.get("items"), rederived["items"], "/items")
    if difference:
        errors.append(f"declared items do not match the frozen rule: {difference}")
    difference = v1._first_difference(manifest.get("pairs"), rederived["pairs"], "/pairs")
    if difference:
        errors.append(f"declared pairs do not match the frozen rule: {difference}")
    for key in ("source_group_count", "pair_count", "item_count", "counts_by_family",
                "counts_by_split", "counts_by_question_type", "split_geometry",
                "item_digest", "pair_digest", "content_sha256"):
        if manifest.get(key) != rederived[key]:
            errors.append(f"{key} must be re-derivable from the seed")
    if manifest.get("item_digest") != digest_value(manifest.get("items")):
        errors.append("item_digest does not match the declared items")
    if manifest.get("pair_digest") != digest_value(manifest.get("pairs")):
        errors.append("pair_digest does not match the declared pairs")
    content = {key: value for key, value in manifest.items()
               if key not in ("content_sha256", "builder_sha256")}
    if manifest.get("content_sha256") != digest_value(content):
        errors.append("content_sha256 does not match the declared manifest content")

    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        return errors + ["items must be a non-empty list"]
    seen_ids, groups, states, fingerprints = set(), {}, {}, {}
    for item in items:
        if item.get("item_id") in seen_ids:
            errors.append(f"item_id {item.get('item_id')!r} is duplicated")
        seen_ids.add(item.get("item_id"))
        group, split = item.get("source_group_id"), item.get("split")
        if group in groups and groups[group] != split:
            errors.append(f"source group {group!r} crosses splits")
        groups[group] = split
        state = item.get("state_id")
        if state in states and states[state] != split:
            errors.append(f"state id {state!r} crosses splits")
        states[state] = split
        fp = _visible_fingerprint(item)
        if fp in fingerprints and fingerprints[fp] != split:
            errors.append(f"canonical input crosses splits: {item['item_id']}")
        fingerprints[fp] = split
        errors.extend(validate_item(item))
    pair_ids = {item["pair_id"] for item in items}
    if len(pair_ids) != manifest.get("pair_count"):
        errors.append("pair_count must equal the number of distinct pair ids")
    for pair_id in sorted(pair_ids):
        pair_items = [item for item in items if item["pair_id"] == pair_id]
        bases = [item for item in pair_items if item["member"] == "base"]
        variants = [item for item in pair_items if item["member"] == "variant"]
        if len(bases) != len(QUESTION_TYPES) or len(variants) != len(QUESTION_TYPES):
            errors.append(f"pair {pair_id!r} must carry all {len(QUESTION_TYPES)} "
                          "question types for both members")
            continue
        if len({item["source_group_id"] for item in pair_items}) != 1:
            errors.append(f"pair {pair_id!r} must share exactly one source group")
        if len({item["split"] for item in pair_items}) != 1:
            errors.append(f"pair {pair_id!r} must not cross splits")
        delta = v1.fact_delta(bases[0]["provenance"]["fact_basis"],
                              variants[0]["provenance"]["fact_basis"])
        if len(delta) != 1:
            errors.append(f"pair {pair_id!r} differs in {len(delta)} fact leaves, "
                          "not 1")
        elif delta[0][0] != bases[0]["contrastive"]["mutated_fact"]:
            errors.append(f"pair {pair_id!r} mutated fact does not match the "
                          "observed diff")
        flipped, declared = set(), set()
        for base in bases:
            variant = next(item for item in variants
                           if item["question_type"] == base["question_type"])
            if _answer_payload(base) != _answer_payload(variant):
                flipped.add(base["question_type"])
            if base["contrastive"]["is_flip_question"]:
                declared.add(base["question_type"])
        if not declared:
            errors.append(f"pair {pair_id!r} declares no flip question type")
        if declared - flipped:
            errors.append(f"pair {pair_id!r} declares a flip that does not hold "
                          f"for {sorted(declared - flipped)}")
    hits = v1.reserved_hits(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    if hits:
        errors.append(f"manifest contains reserved evaluation tokens {hits}")
    return errors


# --------------------------------------------------------------------------------------
# build / check / self-test
# --------------------------------------------------------------------------------------

def output_files(manifest):
    files = {f"{ITEM_DIR}/{SPLIT_FILES[split]}":
             v1._jsonl([item for item in manifest["items"]
                        if item["split"] == split])
             for split in SPLITS}
    rows = trainer_rows(manifest)
    files.update({f"{TRAINER_DIR}/{SPLIT_FILES[split]}":
                  v1._jsonl([row for row in rows if row["split"] == split])
                  for split in SPLITS})
    files[MANIFEST_NAME] = v1.pretty(manifest)
    return files


def build(output_dir, seed=DEFAULT_SEED):
    manifest = derive_manifest(seed)
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("derived corpus failed validation: " + "; ".join(errors))
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    for name, content in sorted(output_files(manifest).items()):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return manifest


def check(output_dir):
    try:
        manifest = json.loads((Path(output_dir) / MANIFEST_NAME)
                              .read_text(encoding="utf-8"))
    except Exception as error:
        return [f"cannot read {MANIFEST_NAME}: {type(error).__name__}: {error}"]
    errors = validate_manifest(manifest)
    try:
        expected = output_files(manifest)
    except Exception as error:
        return errors + [f"cannot re-derive the views from this manifest: "
                         f"{type(error).__name__}: {error}"]
    for name, content in sorted(expected.items()):
        path = Path(output_dir) / name
        if not path.is_file():
            errors.append(f"{name} is missing")
        elif path.read_bytes() != content:
            errors.append(f"{name} does not match the re-derived corpus")
    return errors


def self_test(seed=DEFAULT_SEED):
    try:
        manifest = derive_manifest(seed)
        errors = validate_manifest(manifest)
        difference = v1._first_difference(manifest, derive_manifest(seed), "/")
        if difference:
            errors.append(f"a second in-process derivation differs: {difference}")
        if output_files(manifest)[MANIFEST_NAME] != v1.pretty(manifest):
            errors.append("the manifest view is not stable")
    except Exception as error:
        manifest, errors = None, [f"{type(error).__name__}: {error}"]
    return {
        "schema_version": MANIFEST_SCHEMA, "mode": "self-test",
        "status": "failed" if errors else "ok", "seed": seed,
        "builder": "scripts/build_engineering_corpus_v4.py",
        "components": (manifest["scale_up"]["canonical_input_components"]
                       if manifest else 0),
        "families": len(FAMILIES),
        "new_families": len(NEW_FAMILY_NAMES),
        "source_groups": manifest["source_group_count"] if manifest else 0,
        "pairs": manifest["pair_count"] if manifest else 0,
        "items": manifest["item_count"] if manifest else 0,
        "items_by_family": ({family: manifest["counts_by_family"][family]["items"]
                             for family in FAMILIES} if manifest else {}),
        "items_by_split": dict(manifest["counts_by_split"]) if manifest else {},
        "training_performed": False, "wrote_output": False, "errors": errors,
    }


# --------------------------------------------------------------------------------------
# T9c grouped-calibration readiness check on the calibration split
# --------------------------------------------------------------------------------------

def t9c_group_stats(rows):
    """Group calibration rows by (question_type, option_count); report T9c minimums."""
    groups = defaultdict(lambda: {"rows": 0, "states": set()})
    for row in rows:
        for qid, question in row["questions"].items():
            qtype = question["type"]
            if qtype == "boolean":
                options = 2
            elif qtype == "choice":
                options = len(question["criteria"])
            else:
                options = len(question["criteria"])
            key = f"{qtype}:{options}"
            groups[key]["rows"] += 1
            groups[key]["states"].add(row["state_id"])
    report = {}
    for key in sorted(groups):
        stat = groups[key]
        report[key] = {
            "rows": stat["rows"], "states": len(stat["states"]),
            "estimable": (stat["rows"] >= T9C_MIN_ROWS
                          and len(stat["states"]) >= T9C_MIN_STATES),
        }
    return report


# --------------------------------------------------------------------------------------
# read-only audit receipt (v3 audit conventions, extended to N heldouts/corpora)
# --------------------------------------------------------------------------------------

def _cohort_fingerprints(rows):
    from audit_engineering_corpus_v1 import references
    return references(rows)


def audit_report(corpus_dir, heldout_paths=(), compare_dirs=()):
    """Read-only audit: builder integrity + split isolation + cross-corpus overlap."""
    from audit_engineering_corpus_v1 import audit_rows, file_hash
    from train_pipeline_decisions import read_training_records

    corpus_dir = Path(corpus_dir).resolve(strict=True)
    heldout_paths = [Path(p).resolve(strict=True) for p in heldout_paths]
    compare_dirs = [Path(p).resolve(strict=True) for p in compare_dirs]
    inputs = [corpus_dir / MANIFEST_NAME]
    inputs += [corpus_dir / TRAINER_DIR / f"{s}.jsonl" for s in SPLITS]
    inputs += [corpus_dir / ITEM_DIR / f"{s}.jsonl" for s in SPLITS]
    inputs += list(heldout_paths)
    for compare_dir in compare_dirs:
        inputs.append(compare_dir / MANIFEST_NAME)
        inputs += [compare_dir / TRAINER_DIR / f"{s}.jsonl" for s in SPLITS]
    before = {str(p): file_hash(p) for p in inputs}

    integrity_errors = check(corpus_dir)
    manifest = json.loads((corpus_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    rows, _ = read_training_records(corpus_dir / TRAINER_DIR)
    audit = audit_rows(rows)

    corpus_refs = _cohort_fingerprints(rows)
    item_fps = {_visible_fingerprint(item) for item in manifest["items"]}
    block_reasons = list(audit["block_reasons"])
    if integrity_errors:
        block_reasons.append("builder_integrity_check_failed")

    heldout_reports = {}
    for heldout_path in heldout_paths:
        heldout_rows, _ = read_training_records(heldout_path)
        heldout_refs = _cohort_fingerprints(heldout_rows)
        common = sorted(set(corpus_refs) & set(heldout_refs))
        per_split = {}
        for split in SPLITS:
            split_rows = [r for r in rows if r["split"] == split]
            shared = sorted(set(_cohort_fingerprints(split_rows))
                            & set(heldout_refs))
            per_split[split] = {"shared_canonical_inputs": len(shared),
                                "shared_input_sha256": shared}
        heldout_item_overlap = sorted(item_fps & set(heldout_refs))
        identity = {}
        for key in ("id", "state_id", "family_id"):
            identity[key] = sorted({r[key] for r in rows}
                                   & {r[key] for r in heldout_rows})
        identity["source_group_id"] = sorted(
            {r.get("metadata", {}).get("source_group_id") for r in rows}
            & {r.get("metadata", {}).get("source_group_id")
               for r in heldout_rows} - {None})
        name = heldout_path.parent.name + "/" + heldout_path.name
        heldout_reports[name] = {
            "heldout": str(heldout_path), "heldout_items": len(heldout_rows),
            "heldout_source_groups": len(
                {r.get("metadata", {}).get("source_group_id")
                 for r in heldout_rows}),
            "common_canonical_inputs_total": len(common),
            "common_canonical_inputs_by_split": per_split,
            "item_view_overlap": len(heldout_item_overlap),
            "identity_overlap": identity,
            "overlap_details": [{"input_sha256": k,
                                 "references": corpus_refs[k] + heldout_refs[k]}
                                for k in common],
        }
        if common or heldout_item_overlap:
            block_reasons.append(f"heldout_shares_canonical_inputs_with_corpus_v4:"
                                 f"{heldout_path.parent.name}")
        if any(identity.values()):
            block_reasons.append(f"heldout_shares_declared_identities_with_corpus_v4:"
                                 f"{heldout_path.parent.name}")

    compare_reports = {}
    for compare_dir in compare_dirs:
        compare_rows, _ = read_training_records(compare_dir / TRAINER_DIR)
        compare_refs = _cohort_fingerprints(compare_rows)
        per_split = {}
        for split in SPLITS:
            split_rows = [r for r in compare_rows if r["split"] == split]
            shared = sorted(set(_cohort_fingerprints(split_rows))
                            & set(corpus_refs))
            per_split[split] = {"shared_canonical_inputs": len(shared),
                                "shared_input_sha256": shared}
        common_all = sorted(set(compare_refs) & set(corpus_refs))
        compare_manifest = json.loads(
            (compare_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
        compare_item_fps = {_visible_fingerprint(item)
                            for item in compare_manifest.get("items", [])}
        compare_reports[compare_dir.name] = {
            "compare_corpus": str(compare_dir),
            "policy": ("zero canonical-input overlap is required between corpus "
                       "v4 and every earlier corpus version: they remain "
                       "separately versioned cohorts and any shared visible "
                       "input would blur the isolation evidence"),
            "common_canonical_inputs_total": len(common_all),
            "common_canonical_inputs_by_compare_split": per_split,
            "item_view_overlap": len(compare_item_fps & item_fps),
            "overlap_details": [
                {"input_sha256": k,
                 "references": corpus_refs[k] + compare_refs[k]}
                for k in common_all],
        }
        if common_all or (compare_item_fps & item_fps):
            block_reasons.append(
                f"corpus_v4_shares_canonical_inputs_with_{compare_dir.name}")

    # Cross-cohort checks beyond the corpus itself: heldout_v2 must also be
    # disjoint from heldout_v1 and from every earlier corpus version.
    heldout_refs_map = {}
    for heldout_path in heldout_paths:
        heldout_rows, _ = read_training_records(heldout_path)
        heldout_refs_map[str(heldout_path)] = _cohort_fingerprints(heldout_rows)
    pairwise_extra = {}
    heldout_keys = sorted(heldout_refs_map)
    for i, left in enumerate(heldout_keys):
        for right in heldout_keys[i + 1:]:
            shared = sorted(set(heldout_refs_map[left]) & set(heldout_refs_map[right]))
            name = f"{Path(left).parent.name}~{Path(right).parent.name}"
            pairwise_extra[name] = {"left": left, "right": right,
                                   "shared_canonical_inputs": len(shared),
                                   "shared_input_sha256": shared}
            if shared:
                block_reasons.append(f"heldouts_share_canonical_inputs:{name}")
    for heldout_path, refs in heldout_refs_map.items():
        for compare_dir in compare_dirs:
            compare_rows, _ = read_training_records(compare_dir / TRAINER_DIR)
            compare_refs = _cohort_fingerprints(compare_rows)
            shared = sorted(set(refs) & set(compare_refs))
            name = f"{Path(heldout_path).parent.name}~{compare_dir.name}"
            pairwise_extra[name] = {"left": heldout_path,
                                    "right": str(compare_dir),
                                    "shared_canonical_inputs": len(shared),
                                    "shared_input_sha256": shared}
            if shared:
                block_reasons.append(
                    f"heldout_shares_canonical_inputs_with_corpus:{name}")

    comp_sizes = Counter()
    comp_splits = {}
    for item in manifest["items"]:
        gid = item["source_group_id"]
        comp_sizes[gid] += 1
        comp_splits.setdefault(gid, set()).add(item["split"])
    cross_split_groups = sorted(g for g, s in comp_splits.items() if len(s) > 1)
    if cross_split_groups:
        block_reasons.append("source_group_crosses_splits")

    cal_rows = [r for r in rows if r["split"] == "calibration"]
    t9c = t9c_group_stats(cal_rows)
    estimable = sorted(k for k, v in t9c.items() if v["estimable"])

    if manifest["item_count"] < 4000:
        block_reasons.append("below_scale_target_4000_items")
    if len(manifest["counts_by_family"]) < 30:
        block_reasons.append("below_family_target_30")
    if len(estimable) < T9C_MIN_GROUPS:
        block_reasons.append(
            f"calibration_below_t9c_groups:{len(estimable)}<{T9C_MIN_GROUPS}")

    after = {str(p): file_hash(p) for p in inputs}
    inputs_changed = before != after
    if inputs_changed:
        block_reasons.append("input_files_changed_during_audit")

    return {
        "schema_version": AUDIT_SCHEMA,
        "corpus": str(corpus_dir),
        "audit_subject": ("V4 scaled corpus (task X4); V1/V2/V3 corpora and "
                          "engineering_heldout_v1/v2 preserved as immutable "
                          "evidence"),
        "source_hashes": before,
        "source_hashes_after": after,
        "input_files_changed": inputs_changed,
        "builder_integrity_errors": integrity_errors,
        "builder": "scripts/build_engineering_corpus_v4.py",
        "builder_sha256": manifest.get("builder_sha256"),
        "seed": manifest.get("seed"),
        "audit": audit,
        "component_isolation": {
            "components": manifest["scale_up"]["canonical_input_components"],
            "source_groups": manifest["source_group_count"],
            "component_item_count_histogram": dict(sorted(
                Counter(comp_sizes.values()).items())),
            "source_groups_crossing_splits": cross_split_groups,
            "canonical_input_cross_split_groups":
                audit["canonical_input_cross_split_groups"],
        },
        "heldout_isolation": heldout_reports,
        "corpus_overlap": compare_reports,
        "cross_cohort_overlap": pairwise_extra,
        "t9c_calibration_groups": {
            "grouping": "question_type + option_count recomputed from the row",
            "min_rows": T9C_MIN_ROWS, "min_states": T9C_MIN_STATES,
            "min_estimable_groups": T9C_MIN_GROUPS,
            "groups": t9c, "estimable_groups": estimable,
            "estimable_group_count": len(estimable),
        },
        "scale_targets": {"items_min": 4000, "families_min": 30,
                          "items_actual": manifest["item_count"],
                          "families_actual": len(manifest["counts_by_family"]),
                          "new_families": len(manifest["construction_rule"]
                                              .get("new_families", [])),
                          "pairs_actual": manifest["pair_count"]},
        "counts_by_family": manifest["counts_by_family"],
        "counts_by_split": manifest["counts_by_split"],
        "counts_by_question_type": manifest["counts_by_question_type"],
        "pair_count": manifest["pair_count"],
        "item_count": manifest["item_count"],
        "block_reasons": sorted(set(block_reasons)),
        "status": ("preflight_passed_not_training_authorized"
                   if not block_reasons else "blocked_isolation_plan_only"),
        "limitations": [
            "exact-match canonicalization is a lower bound; paraphrase-level "
            "leakage is addressed by disjoint rule families for the heldouts, "
            "fresh surfaces/entities for v4 items and declared provenance",
            "v4 reuses v3 family *rules* (oracles/domains) with fresh surfaces; "
            "a v3-trained model evaluated on v4 shares rule semantics but no "
            "visible input",
            "analytic-chance families carry programmatic conditional "
            "distributions, not observed outcomes",
            "passing isolation does not authorize training, merging or deployment",
        ],
        "training_authorized": False,
        "training_performed": False,
        "measurement_authorized": False,
        "measurement_performed": False,
        "deployment_authorized": False,
        "deployment_performed": False,
        "merged_rows_written": 0,
        "review_requirements": [
            "engineering_heldout_v2 is the independent heldout going forward; "
            "heldout_v1 remains preserved evidence; never train or select on "
            "either",
            "T9d-scale training on v4 requires its own protocol review; this "
            "corpus authorizes nothing",
            "V1/V2/V3 corpora remain immutable evidence; do not merge without "
            "a reviewed adapter",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, help="empty directory for the corpus")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--check", type=Path,
                        help="re-derive and verify an existing corpus")
    parser.add_argument("--audit", action="store_true",
                        help="emit the read-only audit receipt for --corpus")
    parser.add_argument("--corpus", type=Path, help="built corpus directory (audit)")
    parser.add_argument("--heldout", type=Path, action="append", default=[],
                        help="heldout items.jsonl for the overlap check (repeatable)")
    parser.add_argument("--compare-corpus", type=Path, action="append", default=[],
                        help="earlier corpus dir for the overlap check (repeatable)")
    parser.add_argument("--output", type=Path, help="receipt path (audit, write-once)")
    parser.add_argument("--print-families", action="store_true")
    args = parser.parse_args(argv)

    if args.print_families:
        print(json.dumps({"families": list(FAMILIES), "count": len(FAMILIES),
                          "new_families": list(NEW_FAMILY_NAMES),
                          "pairs_per_family_target": PAIRS_PER_FAMILY},
                         indent=2, sort_keys=True))
        return 0
    if args.self_test and (args.output_dir is not None or args.check is not None
                           or args.audit):
        parser.error("--self-test is mutually exclusive with build/check/audit")
    if args.self_test:
        report = self_test(args.seed)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] == "ok" else 1
    if args.check is not None:
        errors = check(args.check)
        print(json.dumps({"schema_version": MANIFEST_SCHEMA, "mode": "check",
                          "status": "ok" if not errors else "failed",
                          "errors": errors}, indent=2, sort_keys=True))
        return 0 if not errors else 1
    if args.audit:
        if args.corpus is None or args.output is None:
            parser.error("--audit requires --corpus and --output")
        report = audit_report(args.corpus, heldout_paths=args.heldout,
                              compare_dirs=args.compare_corpus)
        path = Path(args.output)
        if path.exists():
            print(json.dumps({"status": "refused", "reason": "receipt exists; "
                              "audit output is write-once"}, indent=2))
            return 2
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
        print(json.dumps({"status": report["status"], "receipt": str(path),
                          "block_reasons": report["block_reasons"]}, indent=2,
                         sort_keys=True))
        return 0 if report["status"] == "preflight_passed_not_training_authorized" else 1
    if args.output_dir is None:
        parser.error("one of --output-dir, --self-test, --check or --audit is required")
    manifest = build(args.output_dir, args.seed)
    print(json.dumps({"status": "ok", "output_dir": str(args.output_dir),
                      "items": manifest["item_count"],
                      "pairs": manifest["pair_count"],
                      "families": len(manifest["counts_by_family"]),
                      "splits": manifest["counts_by_split"],
                      "content_sha256": manifest["content_sha256"],
                      "training_performed": False}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
