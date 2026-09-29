#!/usr/bin/env python3
"""Engineering-judgment corpus V5: v4 base plus independently authored repair arm.

V5 keeps the frozen V4 corpus byte-identical as its base and adds repair families
for the J-D7 diagnosis: missing choice cardinalities k=6/k=12, richer ordinal
score semantics, and explicit Boolean risk-boundary reasoning.  The repair items
are programmatically generated from new family rules under a V5 render/id
namespace; no heldout rows, labels, family names, or model outputs are sources.

This builder authorizes nothing.  It emits a training-data prerequisite corpus
with item and trainer views, a re-derivable manifest, and a read-only audit.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_gate_contrastive_v1 import digest_value  # noqa: E402
import build_engineering_corpus_v1 as v1  # noqa: E402
import build_engineering_corpus_v2 as v2  # noqa: E402
import build_engineering_corpus_v3 as v3  # noqa: E402
import build_engineering_corpus_v4 as v4  # noqa: E402
from predict_toy_decisions import validate_request  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "research/engineering_judgment_corpus_v4"
BASE_MANIFEST_SHA256 = (
    "66ad3da85db54998cd1944e31639321db2046819830849c601db48284352fa98")

SCHEMA_VERSION = "nanojev-engineering-judgment-corpus-v5"
MANIFEST_SCHEMA = "nanojev-engineering-judgment-manifest-v5"
ITEM_SCHEMA = "nanojev-engineering-judgment-item-v5"
SOURCE_GROUP_SCHEMA = "nanojev-engineering-source-group-v5"
AUDIT_SCHEMA = "nanojev-engineering-corpus-v5-audit-v1"
CATALOG_VERSION = "engineering-judgment-catalog-v5-repair"
SOURCE_ID = "scripts/build_engineering_corpus_v5.py"
DEFAULT_SEED = 20260923
REPAIR_PAIRS_PER_FAMILY = 20
REPAIR_MIN_PAIRS_PER_FAMILY = 20

TRAINER_ROW_SCHEMA = v1.TRAINER_ROW_SCHEMA
ITEM_DIR, TRAINER_DIR, MANIFEST_NAME = v1.ITEM_DIR, v1.TRAINER_DIR, v1.MANIFEST_NAME
SPLITS, SPLIT_FILES, QUESTION_TYPES = v1.SPLITS, v1.SPLIT_FILES, v1.QUESTION_TYPES
BOOL, INT = v3.BOOL, v3.INT
_field = v3._field
_levels = v4._levels

REPAIR_FAMILY_DEFS = []


def _fam(family, feature, headings, services, tickets, fields, qids,
         instructions, boolean_criteria, choice_criteria, score_criteria,
         oracle, why, sampler=None, choice_coverage_min=None):
    REPAIR_FAMILY_DEFS.append({
        "family": family, "feature": feature, "headings": tuple(headings),
        "services": tuple(services), "tickets": tuple(tickets),
        "fields": tuple(fields), "qids": dict(qids),
        "instructions": {qt: tuple(v) for qt, v in instructions.items()},
        "boolean_criteria": tuple(boolean_criteria),
        "choice_criteria": dict(choice_criteria),
        "score_criteria": list(score_criteria),
        "oracle": oracle, "why": why, "kind": "deterministic",
        "sampler": sampler, "choice_coverage_min": choice_coverage_min,
    })


def _six_queue_oracle(f):
    loads = [f[f"queue_{k}_load"] for k in range(6)]
    top = max(range(6), key=lambda k: (loads[k], -k))
    over = sum(load >= f["shed_limit"] for load in loads)
    return {"boolean": over > 0,
            "choice": f"queue_{top}",
            "score": min(over, 4)}


_fam(
    "six_queue_route", "six-candidate-load-routing",
    ("Queue shed board for {service} ({ticket}):",
     "Six-queue dispatch review — {service} — {ticket}:",
     "Queue pressure report, {service} [{ticket}]:"),
    ("ingest-pool", "mail-fanout", "thumb-queue", "ledger-worker",
     "geo-sync", "doc-queue", "notify-pool", "scan-router"),
    ("QSHED", "ROUTE6", "LOAD6"),
    tuple([_field(f"queue_{k}_load", f"current load of queue_{k}, percent",
                  INT(0, 100, 5)) for k in range(6)]
          + [_field("shed_limit", "queue load considered saturated, percent",
                    INT(55, 95, 5))]),
    {"boolean": "any_queue_saturated", "choice": "hottest_queue",
     "score": "saturated_queue_count"},
    {"boolean": ("Is any queue at or above the shed limit?",
                 "Does at least one queue reach the saturation threshold?",
                 "Is the queue set overloaded anywhere?"),
     "choice": ("Which queue is carrying the largest load?",
                "Pick the hottest queue by load.",
                "Which queue id has the highest load?"),
     "score": ("Score the number of saturated queues, capped at 4, 0 to 4.",
               "Rate queue saturation by count, 0 to 4.",
               "How many queues are saturated (4+ caps), 0 to 4?")},
    ({}, {"false": "no queue reaches the shed limit",
          "true": "at least one queue reaches the shed limit"}),
    {f"queue_{k}": f"Queue {k} has the highest load" for k in range(6)},
    _levels("no saturated queues", "one saturated queue",
            "two saturated queues", "three saturated queues",
            "four or more saturated queues"),
    _six_queue_oracle,
    "the choice is the queue with maximum load, ties resolve to the lower queue "
    "id; boolean marks any queue at or above the shed limit; score counts "
    "saturated queues capped at four")


def _failover6_oracle(f):
    if not f["primary_healthy"]:
        choice = "hold_rollout"
    elif f["data_lag_minutes"] > 20 or f["error_budget_pct"] < 10:
        choice = "rollback_now"
    elif not f["canary_passed"]:
        choice = "pause_canary"
    elif f["secondary_healthy"] and f["rollback_ready"]:
        choice = "promote_secondary"
    elif f["data_lag_minutes"] > 5 or f["error_budget_pct"] < 30:
        choice = "extend_watch"
    else:
        choice = "continue_rollout"
    severity = (0 if f["primary_healthy"] else 2) + \
        (0 if f["canary_passed"] else 1) + \
        (1 if f["data_lag_minutes"] > 20 else 0) + \
        (1 if f["error_budget_pct"] < 10 else 0)
    return {"boolean": choice == "continue_rollout",
            "choice": choice, "score": min(severity, 4)}


_fam(
    "six_candidate_failover", "six-action-failover-rule",
    ("Failover checklist for {service} ({ticket}):",
     "Release failover board — {service} — {ticket}:",
     "Promotion gate review, {service} [{ticket}]:"),
    ("orders-api", "billing-svc", "identity-gw", "search-index",
     "stock-worker", "payments-api", "profile-db", "edge-cache"),
    ("FAIL6", "GATE6", "PROMO"),
    (_field("primary_healthy", "the primary path is healthy", BOOL),
     _field("secondary_healthy", "the secondary path is healthy", BOOL),
     _field("canary_passed", "the latest canary passed", BOOL),
     _field("rollback_ready", "rollback artifacts are ready", BOOL),
     _field("data_lag_minutes", "replication lag in minutes", INT(0, 60, 5)),
     _field("error_budget_pct", "remaining error budget, percent", INT(0, 100, 5))),
    {"boolean": "safe_to_continue", "choice": "failover_action",
     "score": "rollback_severity"},
    {"boolean": ("Is it safe to continue the rollout?",
                 "Can the rollout proceed without a gate?",
                 "Does the checklist allow continuing?"),
     "choice": ("Which failover action should be taken?",
                "Pick the correct rollout gate action.",
                "What is the next failover decision?"),
     "score": ("Score rollback severity, 0 to 4.",
               "Rate the severity of the failover state, 0 to 4.",
               "How severe is this failover condition, 0 to 4?")},
    ({}, {"false": "the rollout is not safe to continue",
          "true": "the rollout may continue"}),
    {"hold_rollout": "Hold the rollout until the primary path is healthy",
     "rollback_now": "Rollback immediately because lag or budget is critical",
     "pause_canary": "Pause and repair the failed canary",
     "promote_secondary": "Promote the healthy secondary with rollback ready",
     "extend_watch": "Extend monitoring for moderate lag or low budget",
     "continue_rollout": "Continue the rollout because every gate is clear"},
    _levels("no failover risk", "minor failover risk",
            "moderate failover risk", "high failover risk",
            "critical failover risk"),
    _failover6_oracle,
    "unhealthy primary holds; critical lag or budget rolls back; failed canary "
    "pauses; healthy secondary with rollback promotes; moderate lag or low "
    "budget extends watch; otherwise continue")


def _twelve_lane_oracle(f):
    loads = [f[f"lane_{k:02d}_load"] for k in range(12)]
    best = min(range(12), key=lambda k: (loads[k], k))
    busy = sum(load >= f["busy_limit"] for load in loads)
    return {"boolean": busy > 0,
            "choice": f"lane_{best:02d}",
            "score": 0 if busy == 0 else (1 if busy <= 3 else
                     (2 if busy <= 7 else 3))}


_fam(
    "twelve_lane_dispatch", "twelve-candidate-dispatch",
    ("Lane dispatch table for {service} ({ticket}):",
     "Twelve-lane scheduler — {service} — {ticket}:",
     "Worker lane report, {service} [{ticket}]:"),
    ("batch-sink", "image-pipe", "report-gen", "crawl-worker",
     "ml-export", "tile-render", "mail-batch", "audit-pipe"),
    ("LANE12", "DISP12", "SCHED"),
    tuple([_field(f"lane_{k:02d}_load", f"current load of lane {k:02d}, percent",
                  INT(0, 100, 5)) for k in range(12)]
          + [_field("busy_limit", "lane load considered busy, percent",
                    INT(50, 95, 5))]),
    {"boolean": "any_lane_busy", "choice": "dispatch_lane",
     "score": "busy_lane_band"},
    {"boolean": ("Is any lane at or above the busy limit?",
                 "Does at least one lane reach the busy threshold?",
                 "Is there a busy lane in this dispatch table?"),
     "choice": ("Which lane should receive the next job?",
                "Pick the least-loaded dispatch lane.",
                "Which lane id has the lowest load?"),
     "score": ("Score busy-lane breadth, 0 to 3.",
               "Rate dispatch congestion by number of busy lanes, 0 to 3.",
               "How broad is lane congestion, 0 to 3?")},
    ({}, {"false": "no lane is at or above the busy limit",
          "true": "at least one lane is busy"}),
    {f"lane_{k:02d}": f"Dispatch the next job to lane {k:02d}"
     for k in range(12)},
    _levels("no busy lanes", "one to three busy lanes",
            "four to seven busy lanes", "eight or more busy lanes"),
    _twelve_lane_oracle,
    "the dispatch target is the lowest-load lane, ties resolve to the lower id; "
    "the boolean marks any busy lane and the score bands the count of busy lanes")


def _region12_oracle(f):
    health = [f[f"region_{k:02d}_health"] for k in range(12)]
    best = max(range(12), key=lambda k: (health[k], -k))
    healthy = sum(score >= f["required_health"] for score in health)
    return {"boolean": healthy > 0,
            "choice": f"region_{best:02d}",
            "score": 0 if healthy == 0 else (1 if healthy <= 4 else
                     (2 if healthy <= 8 else 3))}


_fam(
    "twelve_region_weight_pick", "twelve-region-health-pick",
    ("Regional health board for {service} ({ticket}):",
     "Twelve-region failover review — {service} — {ticket}:",
     "Region readiness table, {service} [{ticket}]:"),
    ("cdn-edge", "auth-region", "media-store", "metric-hub",
     "queue-pop", "search-pop", "billing-pop", "stream-edge"),
    ("REG12", "HEALTH", "FOVER"),
    tuple([_field(f"region_{k:02d}_health", f"health score of region {k:02d}, "
                  "0 to 100", INT(0, 100, 5)) for k in range(12)]
          + [_field("required_health", "minimum region health considered usable",
                    INT(40, 90, 5))]),
    {"boolean": "any_region_usable", "choice": "best_region",
     "score": "usable_region_band"},
    {"boolean": ("Is any region at or above the required health score?",
                 "Does at least one region meet the health requirement?",
                 "Is there a usable failover region?"),
     "choice": ("Which region should receive the failover?",
                "Pick the healthiest region for failover.",
                "Which region id has the highest health score?"),
     "score": ("Score usable-region breadth, 0 to 3.",
               "Rate how many regions meet the health floor, 0 to 3.",
               "How broad is usable regional capacity, 0 to 3?")},
    ({}, {"false": "no region meets the required health score",
          "true": "at least one region is usable"}),
    {f"region_{k:02d}": f"Fail over to region {k:02d}" for k in range(12)},
    _levels("no usable regions", "one to four usable regions",
            "five to eight usable regions", "nine or more usable regions"),
    _region12_oracle,
    "the choice is the highest-health region with lower-id tie break; boolean "
    "marks whether any region meets the floor; score bands the usable count")


def _score_depth_oracle(f):
    depth = min(5, f["missing_checks"] + f["critical_findings"]
                + (0 if f["docs_complete"] else 1))
    if depth == 0:
        choice = "ship_now"
    elif depth <= 2:
        choice = "fix_minor"
    elif depth <= 4:
        choice = "block_review"
    else:
        choice = "escalate_owner"
    return {"boolean": depth >= 3, "choice": choice, "score": depth}


_fam(
    "score_depth_ladder", "ordinal-depth-score-rubric",
    ("Review depth card for {service} ({ticket}):",
     "Readiness depth review — {service} — {ticket}:",
     "Checklist gap report, {service} [{ticket}]:"),
    ("release-train", "sdk-publish", "infra-change", "model-rollout",
     "schema-update", "cache-change", "api-launch", "data-pipe"),
    ("DEPTH", "READY", "REVW"),
    (_field("missing_checks", "required checks not completed", INT(0, 5)),
     _field("critical_findings", "open critical review findings", INT(0, 3)),
     _field("docs_complete", "required documentation is complete", BOOL)),
    {"boolean": "deep_review_needed", "choice": "review_action",
     "score": "readiness_depth"},
    {"boolean": ("Does this change require a blocking review?",
                 "Is the checklist depth high enough to block release?",
                 "Should this change be held for deep review?"),
     "choice": ("Which review action applies?",
                "Pick the release-review action.",
                "What should the release gate do?"),
     "score": ("Score checklist depth, 0 to 5.",
               "Rate readiness depth from clear to critical, 0 to 5.",
               "How deep is the release-review gap, 0 to 5?")},
    ({}, {"false": "the change can proceed without a blocking review",
          "true": "the change requires a blocking review"}),
    {"ship_now": "Ship now; the checklist is clear",
     "fix_minor": "Fix minor checklist gaps before shipping",
     "block_review": "Block release pending a deeper review",
     "escalate_owner": "Escalate to the owning reviewer for critical gaps"},
    _levels("depth 0: checklist clear", "depth 1: minor gap",
            "depth 2: bounded gap", "depth 3: blocking gap",
            "depth 4: severe gap", "depth 5: critical gap"),
    _score_depth_oracle,
    "score is missing checks plus critical findings plus one for incomplete "
    "docs, capped at five; depth zero ships, 1-2 fixes minor gaps, 3-4 blocks, "
    "and five escalates")


def _risk_gate_oracle(f):
    high = f["affected_users_pct"] >= 20
    slow = f["rollback_time_min"] > 30
    dangerous = f["irreversible"] and not f["approval_present"]
    if dangerous or (high and slow):
        choice = "block_change"
    elif high or slow or f["irreversible"]:
        choice = "require_review"
    elif f["affected_users_pct"] >= 5 or f["rollback_time_min"] > 10:
        choice = "staged_rollout"
    else:
        choice = "auto_proceed"
    risk = int(high) + int(slow) + int(dangerous)
    return {"boolean": dangerous or (high and slow),
            "choice": choice, "score": min(risk, 3)}


_fam(
    "risk_boundary_gate", "boolean-risk-boundary",
    ("Risk boundary card for {service} ({ticket}):",
     "Change risk gate — {service} — {ticket}:",
     "Rollout safety boundary, {service} [{ticket}]:"),
    ("config-push", "worker-deploy", "schema-change", "traffic-shift",
     "cache-purge", "api-rollout", "batch-update", "secret-rotate"),
    ("RISK", "BOUND", "GATE"),
    (_field("affected_users_pct", "estimated affected users, percent",
            INT(0, 100, 5)),
     _field("rollback_time_min", "expected rollback time, minutes", INT(0, 120, 5)),
     _field("irreversible", "the change cannot be fully reversed", BOOL),
     _field("approval_present", "required human approval is present", BOOL)),
    {"boolean": "mandatory_block", "choice": "risk_action",
     "score": "risk_boundary_level"},
    {"boolean": ("Does this change require a mandatory block?",
                 "Is this change beyond the automatic-release boundary?",
                 "Must the gate block this change?"),
     "choice": ("Which risk-gate action applies?",
                "Pick the correct action at the risk boundary.",
                "What should the release gate do with this change?"),
     "score": ("Score boundary risk, 0 to 3.",
               "Rate release-boundary risk, 0 to 3.",
               "How risky is this change boundary, 0 to 3?")},
    ({}, {"false": "the change does not require a mandatory block",
          "true": "the change must be blocked"}),
    {"auto_proceed": "Proceed automatically within the safe boundary",
     "staged_rollout": "Use a staged rollout for moderate blast radius",
     "require_review": "Require human review before proceeding",
     "block_change": "Block the change because the boundary is unsafe"},
    _levels("safe boundary", "single risk trigger",
            "two risk triggers", "critical blocking boundary"),
    _risk_gate_oracle,
    "dangerous irreversible changes without approval, or high blast radius plus "
    "slow rollback, block; high blast radius, slow rollback, or irreversibility "
    "alone require review; moderate values stage; otherwise proceed")


def _deploy_score_oracle(f):
    if not f["tests_passed"] or f["open_blockers"] >= 2:
        score = 0
    elif not f["docs_updated"] or not f["owner_signed"]:
        score = 1
    elif f["open_blockers"] == 1:
        score = 2
    else:
        score = 3
    choice = {0: "stop_deploy", 1: "finish_requirements",
              2: "deploy_after_fix", 3: "deploy_now"}[score]
    return {"boolean": score >= 2, "choice": choice, "score": score}


_fam(
    "score_semantics_deploy", "semantic-readiness-score",
    ("Deploy readiness card for {service} ({ticket}):",
     "Release readiness review — {service} — {ticket}:",
     "Deploy evidence board, {service} [{ticket}]:"),
    ("api-svc", "worker-pool", "web-shell", "mobile-api",
     "ingest-api", "report-svc", "search-svc", "event-pipe"),
    ("DEPLOY", "READY4", "SHIP"),
    (_field("tests_passed", "required tests all passed", BOOL),
     _field("docs_updated", "required docs are updated", BOOL),
     _field("owner_signed", "the owning team signed off", BOOL),
     _field("open_blockers", "open release blockers", INT(0, 3))),
    {"boolean": "deploy_allowed", "choice": "deploy_action",
     "score": "readiness_score"},
    {"boolean": ("Is the deploy allowed to proceed?",
                 "Does this release meet the proceed boundary?",
                 "Can deployment continue under the readiness rule?"),
     "choice": ("Which deploy action should be taken?",
                "Pick the readiness-gated deploy action.",
                "What is the correct deploy decision?"),
     "score": ("Score deploy readiness, 0 to 3.",
               "Rate deploy readiness from stopped to ready, 0 to 3.",
               "How ready is this deploy, 0 to 3?")},
    ({}, {"false": "the deploy is not ready to proceed",
          "true": "the deploy may proceed"}),
    {"stop_deploy": "Stop the deploy because required evidence is missing",
     "finish_requirements": "Finish docs or owner signoff first",
     "deploy_after_fix": "Fix the single blocker, then deploy",
     "deploy_now": "Deploy now; all readiness evidence is present"},
    _levels("stopped: tests fail or multiple blockers",
            "incomplete requirements", "one remaining blocker",
            "ready to deploy"),
    _deploy_score_oracle,
    "score is ordinal readiness: failed tests or two-plus blockers stop; missing "
    "docs/signoff scores one; one blocker scores two; all evidence scores three")


def _triage12_oracle(f):
    if f["sev"] == 3 and f["data_risk"]:
        choice = "freeze_all_changes"
    elif f["sev"] == 3:
        choice = "page_exec"
    elif f["data_risk"] and not f["rollback_ready"]:
        choice = "halt_writes"
    elif f["data_risk"]:
        choice = "prepare_rollback"
    elif f["customer_impact"] and not f["owner_available"]:
        choice = "page_owner"
    elif f["customer_impact"] and f["traffic_high"]:
        choice = "shift_traffic"
    elif f["customer_impact"]:
        choice = "open_bridge"
    elif not f["rollback_ready"]:
        choice = "hold_release"
    elif f["traffic_high"]:
        choice = "schedule_offpeak"
    elif f["owner_available"]:
        choice = "assign_owner"
    elif f["sev"] == 1:
        choice = "open_ticket"
    else:
        choice = "monitor_only"
    urgency = min(5, f["sev"] + int(f["customer_impact"]) +
                  int(f["data_risk"]))
    return {"boolean": f["data_risk"] or f["sev"] >= 2,
            "choice": choice, "score": urgency}


_fam(
    "twelve_action_triage", "twelve-semantic-action-triage",
    ("Incident action board for {service} ({ticket}):",
     "Triage action matrix — {service} — {ticket}:",
     "Operational decision card, {service} [{ticket}]:"),
    ("commerce-api", "media-pipe", "iot-hub", "risk-engine",
     "ml-serve", "identity-svc", "ledger-api", "chat-gateway"),
    ("TRI12", "ACT12", "OPS"),
    (_field("sev", "incident severity, 0 to 3", INT(0, 3)),
     _field("customer_impact", "customers are visibly impacted", BOOL),
     _field("data_risk", "data integrity may be at risk", BOOL),
     _field("rollback_ready", "a rollback path is ready", BOOL),
     _field("owner_available", "the owning engineer is available", BOOL),
     _field("traffic_high", "traffic is currently high", BOOL)),
    {"boolean": "high_risk_state", "choice": "triage_action",
     "score": "urgency_score"},
    {"boolean": ("Is this a high-risk operational state?",
                 "Does this state cross the high-risk boundary?",
                 "Should this be treated as high risk?"),
     "choice": ("Which triage action should be selected?",
                "Pick the correct operational action.",
                "What action should the operator take?"),
     "score": ("Score urgency, 0 to 5.",
               "Rate operational urgency, 0 to 5.",
               "How urgent is this state, 0 to 5?")},
    ({}, {"false": "the state is below the high-risk boundary",
          "true": "the state is high risk"}),
    {"freeze_all_changes": "Freeze all changes because sev-3 includes data risk",
     "page_exec": "Page executive response for sev-3 without data risk",
     "halt_writes": "Halt writes until a rollback path exists",
     "prepare_rollback": "Prepare rollback for a data-risk state",
     "page_owner": "Page the owner for customer impact",
     "shift_traffic": "Shift traffic for customer impact during high traffic",
     "open_bridge": "Open an incident bridge for customer impact",
     "hold_release": "Hold release until rollback is ready",
     "schedule_offpeak": "Schedule the change for off-peak traffic",
     "assign_owner": "Assign the available owner",
     "open_ticket": "Open a standard ticket for low severity",
     "monitor_only": "Monitor only; no immediate action"},
    _levels("no urgency", "low urgency", "moderate urgency",
            "high urgency", "very high urgency", "critical urgency"),
    _triage12_oracle,
    "the rule chain selects one of twelve operational actions; boolean marks "
    "data risk or severity two-plus; score is severity plus customer/data flags "
    "capped at five")


REPAIR_FAMILIES = tuple(spec["family"] for spec in REPAIR_FAMILY_DEFS)
REPAIR_FAMILY_MAP = {spec["family"]: spec for spec in REPAIR_FAMILY_DEFS}
ALL_FAMILY_DEFS = list(v4.FAMILY_DEFS) + REPAIR_FAMILY_DEFS
ALL_FAMILIES = tuple(spec["family"] for spec in ALL_FAMILY_DEFS)
ALL_FAMILY_MAP = {spec["family"]: spec for spec in ALL_FAMILY_DEFS}
ALL_QIDS = {qtype: {spec["family"]: spec["qids"][qtype]
                    for spec in ALL_FAMILY_DEFS} for qtype in QUESTION_TYPES}
assert len(REPAIR_FAMILIES) == len(set(REPAIR_FAMILIES)) == 8
assert len(ALL_FAMILIES) == len(set(ALL_FAMILIES)) == 44


@contextmanager
def _v5_namespace():
    names = ("FAMILY_DEFS", "FAMILIES", "FAMILY_MAP", "QIDS",
             "V3_FAMILY_NAMES", "NEW_FAMILY_NAMES", "DEFAULT_SEED", "SOURCE_ID",
             "CATALOG_VERSION", "ITEM_SCHEMA", "MANIFEST_SCHEMA",
             "SOURCE_GROUP_SCHEMA", "AUDIT_SCHEMA")
    saved = {name: getattr(v4, name) for name in names}
    try:
        v4.FAMILY_DEFS = ALL_FAMILY_DEFS
        v4.FAMILIES = ALL_FAMILIES
        v4.FAMILY_MAP = ALL_FAMILY_MAP
        v4.QIDS = ALL_QIDS
        v4.V3_FAMILY_NAMES = tuple(spec["family"] for spec in v3.FAMILY_DEFS)
        v4.NEW_FAMILY_NAMES = REPAIR_FAMILIES
        v4.DEFAULT_SEED = DEFAULT_SEED
        v4.SOURCE_ID = SOURCE_ID
        v4.CATALOG_VERSION = CATALOG_VERSION
        v4.ITEM_SCHEMA = ITEM_SCHEMA
        v4.MANIFEST_SCHEMA = MANIFEST_SCHEMA
        v4.SOURCE_GROUP_SCHEMA = SOURCE_GROUP_SCHEMA
        v4.AUDIT_SCHEMA = AUDIT_SCHEMA
        yield
    finally:
        for name, value in saved.items():
            setattr(v4, name, value)


def _base_manifest():
    path = BASE_DIR / MANIFEST_NAME
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != BASE_MANIFEST_SHA256:
        raise ValueError("v4 base manifest hash does not match the frozen base")
    return json.loads(path.read_text(encoding="utf-8"))


def _make_render(spec, seed, ordinal, attempt=0):
    rng = random.Random(
        f"nanojev-engineering-v5-render:{seed}:{spec['family']}:{ordinal}:{attempt}")
    order = [f["key"] for f in spec["fields"]]
    rng.shuffle(order)
    return {"template": rng.randrange(5),
            "heading": spec["headings"][rng.randrange(len(spec["headings"]))],
            "service": spec["services"][rng.randrange(len(spec["services"]))],
            "ticket": f"{spec['tickets'][rng.randrange(len(spec['tickets']))]}"
                      f"-{rng.randrange(100000, 999999)}",
            "field_order": order,
            "text_variant": rng.randrange(64)}


def _repair_pairs(seed, reserved_states):
    with _v5_namespace():
        v4._validate_specs()
    pairs = []
    used_states = set(reserved_states)
    for spec in REPAIR_FAMILY_DEFS:
        family = spec["family"]
        candidates = v4.select_for_coverage(
            spec, v4.fact_candidates(spec, seed), REPAIR_PAIRS_PER_FAMILY)
        made = 0
        for ordinal, cand in enumerate(candidates):
            if made >= REPAIR_PAIRS_PER_FAMILY:
                break
            base_facts, base_gold = cand["facts"], cand["gold"]
            render, base_state = None, None
            for attempt in range(12):
                trial = _make_render(spec, seed, ordinal, attempt)
                text = v3.render_state(base_facts, spec, trial)
                if text not in used_states:
                    render, base_state = trial, text
                    break
            if render is None:
                continue
            mutations = v4.flipping_mutations(spec, base_facts, base_gold)
            if not mutations:
                continue
            start = (ordinal * 7 + abs(seed)) % len(mutations)
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
            used_states.add(base_state)
            used_states.add(variant_state)
            rule_id = f"{family}-v5r{made:03d}"
            pairs.append({
                "pair_id": f"ej5-{family}-p{made:03d}",
                "rule_id": rule_id, "state_id": rule_id,
                "family": family, "feature": spec["feature"],
                "mutated_fact": mutation["key"],
                "value_before": delta[0][1], "value_after": delta[0][2],
                "flip_question_types": mutation["flips"],
                "declared_flip": list(mutation["flips"]),
                "gold_before": base_gold, "gold_after": mutation["variant_gold"],
                "base_facts": base_facts, "variant_facts": mutation["variant_facts"],
                "render": render, "rule": spec,
                "split": "train", "source_group_id": "pending",
            })
            made += 1
        if made < REPAIR_MIN_PAIRS_PER_FAMILY:
            raise ValueError(f"{family}: only {made} repair pairs could be built")
        golds = [pair["gold_before"] for pair in pairs
                 if pair["family"] == family]
        golds += [pair["gold_after"] for pair in pairs
                  if pair["family"] == family]
        coverage = v4._coverage_errors(spec, golds)
        if coverage:
            raise ValueError(f"{family}: {coverage}")
    return pairs


def _assign_repair_splits(seed, pairs, base_items):
    members = {}
    components = v2._Components()
    fingerprint_nodes = {}
    base_fp_splits = {}
    for item in base_items:
        base_fp_splits.setdefault(v4._visible_fingerprint(item), item["split"])
    for pair in pairs:
        nodes = []
        for member in ("base", "variant"):
            with _v5_namespace():
                items = v4.make_items(pair, member)
            node = (pair["pair_id"], member)
            members[node] = items
            nodes.append(node)
            for item in items:
                fp = v4._visible_fingerprint(item)
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
    forced = {}
    for comp_nodes in grouped.values():
        first = min(catalog_order[node[0]] for node in comp_nodes)
        family = pairs[catalog_order[comp_nodes[0][0]]]["family"]
        per_family.setdefault(family, []).append((first, comp_nodes))
        fps = {v4._visible_fingerprint(item) for node in comp_nodes
               for item in members[node]}
        splits = {base_fp_splits[fp] for fp in fps if fp in base_fp_splits}
        if len(splits) > 1:
            raise ValueError("repair component matches base items in multiple splits")
        if splits:
            forced[tuple(node[0] for node in comp_nodes)] = splits.pop()
    family_parity = {family: index for index, family in enumerate(REPAIR_FAMILIES)}
    for family, comps in per_family.items():
        cycle = v4.SPLIT_CYCLES[family_parity[family] % 2]
        for index, (_, comp_nodes) in enumerate(sorted(comps)):
            key = tuple(node[0] for node in comp_nodes)
            split = forced.get(key, cycle[(index + abs(seed)) % len(cycle)])
            fps = sorted({v4._visible_fingerprint(item) for node in comp_nodes
                          for item in members[node]})
            group_id = "ejc-v5-" + digest_value(
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
    return [item for node in members for item in members[node]]


def _pair_records(pairs, items):
    return [{
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


def _split_geometry(items, pairs):
    out = {}
    for split in SPLITS:
        split_items = [item for item in items if item["split"] == split]
        out[split] = {
            "items": len(split_items),
            "pairs": sum(1 for pair in pairs if pair["split"] == split),
            "source_groups": sorted({item["source_group_id"]
                                     for item in split_items}),
            "families": sorted({item["family"] for item in split_items}),
            "question_types": sorted({item["question_type"]
                                      for item in split_items}),
        }
    return out


def _counts_by_family(items, pairs):
    counts = dict()
    for family in ALL_FAMILIES:
        family_items = [item for item in items if item["family"] == family]
        family_pairs = [pair for pair in pairs if pair["family"] == family]
        counts[family] = {
            "items": len(family_items), "pairs": len(family_pairs),
            "kind": ALL_FAMILY_MAP[family].get("kind", "deterministic"),
            "choice_candidates": len(ALL_FAMILY_MAP[family]["choice_criteria"]),
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


def derive_manifest(seed=DEFAULT_SEED):
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    base = _base_manifest()
    base_items = list(base["items"])
    base_states = {item["request"]["states"][0]["state"] for item in base_items}
    repair_pairs = _repair_pairs(seed, base_states)
    repair_items = _assign_repair_splits(seed, repair_pairs, base_items)
    items = base_items + repair_items
    pairs = list(base["pairs"]) + _pair_records(repair_pairs, repair_items)
    for item in repair_items:
        v1.guard_source(item["provenance"]["source_id"], item,
                        f"item {item['item_id']}")
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "training_data_prerequisite_no_training_authorized",
        "corpus_role": "training_data_prerequisite_only",
        "created_by": "scripts/build_engineering_corpus_v5.py",
        "catalog_version": CATALOG_VERSION,
        "seed": seed,
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "base_corpus": {
            "path": "research/engineering_judgment_corpus_v4",
            "manifest_sha256": BASE_MANIFEST_SHA256,
            "content_sha256": base["content_sha256"],
            "items": base["item_count"], "pairs": base["pair_count"],
            "included_byte_identical": True,
        },
        "repair_arm": {
            "motivation": ("J-D7 MiniCPM v6 diagnosis: capacity improved but "
                           "choice k=6/12, score semantics, and Boolean risk "
                           "boundaries remain weak"),
            "families": list(REPAIR_FAMILIES),
            "pairs": len(repair_pairs), "items": len(repair_items),
            "namespace": "ej5 item ids, ejc-v5 source groups, v5 render seed",
            "heldout_sources_used": False,
            "heldout_labels_used": False,
        },
        "contract": {
            "validator": "scripts/predict_toy_decisions.py:validate_request",
            "shape": ('{"states":[{"id","state","questions":'
                      '{qid:{"type","instructions","criteria"}}}]}'),
            "trainer_row_contract": "scripts/train_pipeline_decisions.py:validate_training_row",
            "views": {"item": f"{ITEM_DIR}/<split>.jsonl",
                      "trainer": f"{TRAINER_DIR}/<split>.jsonl"},
            "validated": True,
        },
        "construction_rule": {
            "version": "engineering-judgment-rule-v5-repair",
            "base_rule": "v4 rows are included unchanged under a pinned manifest hash",
            "repair_rule": ("each new pair is a base state plus a one-fact mutation "
                            "under a V5-only family rule; declared flips are "
                            "recomputed from the family oracle"),
            "split_unit": "canonical-input connected component, same as v4",
            "split_rule": ("v4 rows keep their frozen split; repair components use "
                           "the same alternating 20-slot cycles by repair-family "
                           "parity and seed offset"),
            "question_types": list(QUESTION_TYPES),
            "families": list(ALL_FAMILIES),
            "base_families": [spec["family"] for spec in v4.FAMILY_DEFS],
            "repair_families": list(REPAIR_FAMILIES),
            "views": {
                "items/<split>.jsonl": "audit view with full request and provenance",
                "trainer_view/<split>.jsonl": "train_pipeline_decisions row contract",
            },
            "provenance_field": ("every repair item records source_id, rule_id, "
                                 "why_correct, full fact basis, surface template, "
                                 "and authoring mode"),
        },
        "provenance": dict(v1.PROVENANCE),
        "exclusions": {
            "policy": ("V5 preserves the V1/V2/V3/V4 refusal wall: no evaluation "
                       "source path, no reserved evaluation token, no external "
                       "model output, and no heldout-derived source."),
            "refused_path_marker_count": len(v1.FORBIDDEN_PATH_MARKERS),
            "refused_path_markers_sha256": digest_value(list(v1.FORBIDDEN_PATH_MARKERS)),
            "reserved_token_count": len(v1.RESERVED_TOKENS),
            "reserved_tokens_sha256": digest_value(list(v1.RESERVED_TOKENS)),
            "evaluation_source_markers": list(v2.EVALUATION_SOURCE_MARKERS),
            "uses_evaluation_corpus_as_source": False,
            "uses_heldout_items_or_labels": False,
            "uses_external_model_outputs": False,
        },
        "source_group_count": len({item["source_group_id"] for item in items}),
        "pair_count": len(pairs),
        "item_count": len(items),
        "counts_by_family": _counts_by_family(items, pairs),
        "counts_by_split": {split: sum(1 for item in items
                                     if item["split"] == split)
                            for split in SPLITS},
        "counts_by_question_type": {qtype: sum(1 for item in items
                                             if item["question_type"] == qtype)
                                    for qtype in QUESTION_TYPES},
        "split_geometry": _split_geometry(items, pairs),
        "pairs": pairs,
        "items": items,
        "item_digest": digest_value(items),
        "pair_digest": digest_value(pairs),
        "content_sha256": None,
    }
    manifest["content_sha256"] = digest_value(
        {key: value for key, value in manifest.items()
         if key not in ("content_sha256", "builder_sha256")})
    return manifest


def trainer_row(item):
    if item["provenance"]["source_id"] == v4.SOURCE_ID:
        return v4.trainer_row(item)
    with _v5_namespace():
        row = v4.trainer_row(item)
    row["provenance"]["cohort"] = "engineering_judgment_corpus_v5_repair"
    return row


def trainer_rows(manifest):
    return [trainer_row(item) for item in manifest["items"]]


def validate_item(item):
    if item.get("schema_version") not in (v4.ITEM_SCHEMA, ITEM_SCHEMA):
        return [f"{item.get('item_id')}: unexpected item schema"]
    with _v5_namespace():
        return v4.validate_item(item)


def validate_manifest(manifest):
    errors = []
    if not isinstance(manifest, dict):
        return ["manifest must be a JSON object"]
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        errors.append(f"schema_version must be {MANIFEST_SCHEMA!r}")
    if manifest.get("created_by") != "scripts/build_engineering_corpus_v5.py":
        errors.append("created_by must name this builder")
    if manifest.get("status") != "training_data_prerequisite_no_training_authorized":
        errors.append("status must record that no training is authorized")
    if manifest.get("corpus_role") != "training_data_prerequisite_only":
        errors.append("corpus_role must be training_data_prerequisite_only")
    if manifest.get("catalog_version") != CATALOG_VERSION:
        errors.append("catalog_version must match the builder")
    if manifest.get("base_corpus", {}).get("manifest_sha256") != BASE_MANIFEST_SHA256:
        errors.append("base_corpus must pin the frozen v4 manifest hash")
    if type(manifest.get("seed")) is not int:
        errors.append("seed must be an integer")
        return errors
    try:
        base = _base_manifest()
        rederived = derive_manifest(manifest["seed"])
    except Exception as error:
        errors.append(f"re-derivation raised {type(error).__name__}: {error}")
        return errors
    base_ids = {item["item_id"] for item in base["items"]}
    declared_base = [item for item in manifest.get("items", [])
                     if item["item_id"] in base_ids]
    if declared_base != base["items"]:
        errors.append("v4 base items must be included byte-identically")
    if manifest.get("items") != rederived["items"]:
        difference = v1._first_difference(manifest.get("items"),
                                          rederived["items"], "/items")
        errors.append(f"declared items do not match the frozen rule: {difference}")
    if manifest.get("pairs") != rederived["pairs"]:
        difference = v1._first_difference(manifest.get("pairs"),
                                          rederived["pairs"], "/pairs")
        errors.append(f"declared pairs do not match the frozen rule: {difference}")
    for key in ("source_group_count", "pair_count", "item_count", "counts_by_family",
                "counts_by_split", "counts_by_question_type", "split_geometry",
                "item_digest", "pair_digest", "content_sha256"):
        if manifest.get(key) != rederived[key]:
            errors.append(f"{key} must be re-derivable from the seed")
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
        fp = v4._visible_fingerprint(item)
        if fp in fingerprints and fingerprints[fp] != split:
            errors.append(f"canonical input crosses splits: {item['item_id']}")
        fingerprints[fp] = split
        errors.extend(validate_item(item))
    for pair_id in {item["pair_id"] for item in items}:
        pair_items = [item for item in items if item["pair_id"] == pair_id]
        bases = [item for item in pair_items if item["member"] == "base"]
        variants = [item for item in pair_items if item["member"] == "variant"]
        if len(bases) != len(QUESTION_TYPES) or len(variants) != len(QUESTION_TYPES):
            errors.append(f"pair {pair_id!r} must carry all question types")
            continue
        if len({item["source_group_id"] for item in pair_items}) != 1:
            errors.append(f"pair {pair_id!r} must share one source group")
        if len({item["split"] for item in pair_items}) != 1:
            errors.append(f"pair {pair_id!r} must not cross splits")
        delta = v1.fact_delta(bases[0]["provenance"]["fact_basis"],
                              variants[0]["provenance"]["fact_basis"])
        if len(delta) != 1:
            errors.append(f"pair {pair_id!r} differs in {len(delta)} fact leaves")
        flipped, declared = set(), set()
        for base_item in bases:
            variant = next(item for item in variants
                           if item["question_type"] == base_item["question_type"])
            if v4._answer_payload(base_item) != v4._answer_payload(variant):
                flipped.add(base_item["question_type"])
            if base_item["contrastive"]["is_flip_question"]:
                declared.add(base_item["question_type"])
        if not declared:
            errors.append(f"pair {pair_id!r} declares no flip question type")
        if declared - flipped:
            errors.append(f"pair {pair_id!r} declares a flip that does not hold")
    hits = v1.reserved_hits(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    if hits:
        errors.append(f"manifest contains reserved evaluation tokens {hits}")
    return errors


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
        return errors + ["cannot re-derive the views from this manifest: "
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
        "builder": "scripts/build_engineering_corpus_v5.py",
        "families": len(ALL_FAMILIES),
        "base_families": len(v4.FAMILY_DEFS),
        "repair_families": len(REPAIR_FAMILIES),
        "source_groups": manifest["source_group_count"] if manifest else 0,
        "pairs": manifest["pair_count"] if manifest else 0,
        "items": manifest["item_count"] if manifest else 0,
        "items_by_split": dict(manifest["counts_by_split"]) if manifest else {},
        "training_performed": False, "wrote_output": False, "errors": errors,
    }


def t9c_group_stats(rows):
    groups = defaultdict(lambda: {"rows": 0, "states": set()})
    for row in rows:
        for question in row["questions"].values():
            qtype = question["type"]
            options = 2 if qtype == "boolean" else len(question["criteria"])
            key = f"{qtype}:{options}"
            groups[key]["rows"] += 1
            groups[key]["states"].add(row["state_id"])
    return {key: {"rows": stat["rows"], "states": len(stat["states"]),
                  "estimable": stat["rows"] >= v4.T9C_MIN_ROWS
                  and len(stat["states"]) >= v4.T9C_MIN_STATES}
            for key, stat in sorted(groups.items())}


def audit_report(corpus_dir, heldout_paths=(), compare_dirs=()):
    from audit_engineering_corpus_v1 import audit_rows, file_hash, references
    from train_pipeline_decisions import read_training_records

    corpus_dir = Path(corpus_dir).resolve(strict=True)
    heldout_paths = [Path(p).resolve(strict=True) for p in heldout_paths]
    compare_dirs = [Path(p).resolve(strict=True) for p in compare_dirs]
    inputs = [corpus_dir / MANIFEST_NAME]
    inputs += [corpus_dir / view / f"{split}.jsonl"
               for view in (ITEM_DIR, TRAINER_DIR) for split in SPLITS]
    inputs += list(heldout_paths)
    for compare_dir in compare_dirs:
        inputs.append(compare_dir / MANIFEST_NAME)
        inputs += [compare_dir / TRAINER_DIR / f"{split}.jsonl"
                   for split in SPLITS]
    before = {str(p): file_hash(p) for p in inputs}
    integrity_errors = check(corpus_dir)
    manifest = json.loads((corpus_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    rows, _ = read_training_records(corpus_dir / TRAINER_DIR)
    audit = audit_rows(rows)
    block_reasons = list(audit["block_reasons"])
    if integrity_errors:
        block_reasons.append("builder_integrity_check_failed")

    repair_rows = [row for row in rows
                   if row.get("metadata", {}).get("provenance_source_id") == SOURCE_ID]
    base_rows = [row for row in rows
                 if row.get("metadata", {}).get("provenance_source_id") != SOURCE_ID]
    repair_refs = references(repair_rows, "v5_repair")
    all_refs = references(rows, "v5_all")
    heldout_reports = {}
    for heldout_path in heldout_paths:
        heldout_rows, _ = read_training_records(heldout_path)
        heldout_refs = references(heldout_rows, heldout_path.parent.name)
        shared_all = sorted(set(all_refs) & set(heldout_refs))
        shared_repair = sorted(set(repair_refs) & set(heldout_refs))
        heldout_reports[heldout_path.parent.name] = {
            "heldout": str(heldout_path), "heldout_items": len(heldout_rows),
            "shared_canonical_inputs_all": len(shared_all),
            "shared_canonical_inputs_repair": len(shared_repair),
            "repair_overlap": [{"input_sha256": key,
                                "references": repair_refs[key] + heldout_refs[key]}
                               for key in shared_repair],
        }
        if shared_all or shared_repair:
            block_reasons.append(
                f"heldout_shares_canonical_inputs_with_corpus_v5:{heldout_path.parent.name}")

    compare_reports = {}
    for compare_dir in compare_dirs:
        compare_rows, _ = read_training_records(compare_dir / TRAINER_DIR)
        compare_refs = references(compare_rows, compare_dir.name)
        shared_all = sorted(set(all_refs) & set(compare_refs))
        shared_repair = sorted(set(repair_refs) & set(compare_refs))
        expected_base_overlap = compare_dir.name == "engineering_judgment_corpus_v4"
        compare_reports[compare_dir.name] = {
            "compare_corpus": str(compare_dir),
            "shared_canonical_inputs_all": len(shared_all),
            "shared_canonical_inputs_repair": len(shared_repair),
            "expected_v4_base_overlap": expected_base_overlap,
            "unexpected_repair_overlap": bool(shared_repair),
        }
        if shared_repair or (shared_all and not expected_base_overlap):
            block_reasons.append(
                f"corpus_v5_shares_unexpected_inputs_with_{compare_dir.name}")

    cal_rows = [row for row in rows if row["split"] == "calibration"]
    t9c = t9c_group_stats(cal_rows)
    estimable = sorted(key for key, value in t9c.items() if value["estimable"])
    if manifest["item_count"] < 4800:
        block_reasons.append("below_v5_scale_target_4800_items")
    if len(manifest["counts_by_family"]) < 40:
        block_reasons.append("below_v5_family_target_40")
    if len(estimable) < v4.T9C_MIN_GROUPS:
        block_reasons.append("calibration_below_t9c_groups")
    after = {str(p): file_hash(p) for p in inputs}
    inputs_changed = before != after
    if inputs_changed:
        block_reasons.append("input_files_changed_during_audit")
    return {
        "schema_version": AUDIT_SCHEMA,
        "corpus": str(corpus_dir),
        "audit_subject": ("V5 repair corpus: v4 base included byte-identically "
                          "plus independently authored repair families"),
        "source_hashes": before, "source_hashes_after": after,
        "input_files_changed": inputs_changed,
        "builder_integrity_errors": integrity_errors,
        "builder": "scripts/build_engineering_corpus_v5.py",
        "builder_sha256": manifest.get("builder_sha256"),
        "seed": manifest.get("seed"),
        "audit": audit,
        "heldout_isolation": heldout_reports,
        "corpus_overlap": compare_reports,
        "t9c_calibration_groups": {
            "min_rows": v4.T9C_MIN_ROWS, "min_states": v4.T9C_MIN_STATES,
            "min_estimable_groups": v4.T9C_MIN_GROUPS,
            "groups": t9c, "estimable_groups": estimable,
            "estimable_group_count": len(estimable),
        },
        "scale_targets": {"items_min": 4800, "families_min": 40,
                          "items_actual": manifest["item_count"],
                          "families_actual": len(manifest["counts_by_family"]),
                          "repair_families": len(REPAIR_FAMILIES),
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
            "v4 base overlap with corpus_v4 is expected because v4 is the pinned base",
            "new repair families are checked for zero canonical overlap with heldouts "
            "and every earlier corpus",
            "canonical equality is a lower bound and does not prove semantic disjointness",
            "passing isolation does not authorize training, promotion, or deployment",
        ],
        "training_authorized": False,
        "training_performed": False,
        "deployment_authorized": False,
        "candidate_promotion_authorized": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, help="empty directory for the corpus")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--check", type=Path,
                        help="re-derive and verify an existing corpus")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--corpus", type=Path, help="built corpus directory")
    parser.add_argument("--heldout", type=Path, action="append", default=[])
    parser.add_argument("--compare-corpus", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, help="receipt path (audit, write-once)")
    parser.add_argument("--print-families", action="store_true")
    args = parser.parse_args(argv)

    if args.print_families:
        print(json.dumps({"families": list(ALL_FAMILIES), "count": len(ALL_FAMILIES),
                          "base_families": len(v4.FAMILY_DEFS),
                          "repair_families": list(REPAIR_FAMILIES),
                          "repair_pairs_per_family": REPAIR_PAIRS_PER_FAMILY},
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
            print(json.dumps({"status": "refused",
                              "reason": "receipt exists; audit output is write-once"},
                             indent=2))
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
