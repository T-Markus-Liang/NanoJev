#!/usr/bin/env python3
"""Engineering-judgment corpus V3: scaled catalog over new families.

Why this file exists
--------------------

W42 evidence (``docs/NANOJEV_V2_ROADMAP.md`` task J-T3): the T9d full-backbone
training run on ``engineering_judgment_corpus_v2`` (228 items, 102 train
questions, 26 canonical-input components) overfit instantly — dev selected
step 0 for 2/3 seeds and heldout accuracy regressed (0.362 vs the 0.464 atomic
baseline).  Data scale and diversity are the binding constraint, so V3 rebuilds
the *catalog* at ~8x scale while keeping every V2 safety property:

* **All-new families.**  V3 defines 18 question families that share no name,
  fact set, instruction string or candidate vocabulary with the 7 V1/V2
  families or with ``engineering_heldout_v1``.  State text is rendered through
  five surface templates (report bullets, ticket excerpt, metric table,
  checklist, config block) with per-pair headings, entity names, ticket ids and
  shuffled field order, plus per-pair instruction/criteria phrasing variants,
  so items are not near-duplicates of each other or of earlier corpora.
* **Programmatic contrastive pairs.**  Instead of 33 hand-authored rules, each
  family declares an enumerated fact domain and a frozen oracle.  Base states
  are sampled/enumerated deterministically from the domain and selected for
  label coverage; each pair's variant differs in exactly one fact leaf and the
  declared flip question types are *computed* from both fact sets — a pair with
  no real flip is never emitted.
* **Component-connected split isolation (unchanged).**  Both members of a pair
  and every member sharing an identical canonical visible input (state +
  question type + instructions + criteria, the same fingerprint the auditor and
  ``heldout_v1_isolation_check.py`` use) are unioned into one component; one
  split is assigned per component by deterministic per-family position plus a
  seed offset — content never selects a split.
* **Label semantics preserved.**  ``deterministic_truth`` gold + one-hot
  ``gold_probs``; item view carries the full request, expected distribution,
  contrast metadata and provenance; trainer view uses the frozen
  ``nanojev-engineering-judgment-trainer-row-v1`` contract (id, state_id,
  family_id, split, state, questions, gold, gold_probs, gold_probs_kind,
  gold_label_kind, metadata, provenance).
* **No evaluation-derived sources, ever.**  The V1 reserved-token/path
  tripwires are reused verbatim via ``v1.guard_source``; every item's
  provenance declares ``derived_from_evaluation_corpus: false``.
* **No training.**  Nothing here reads or writes a checkpoint, optimizer or
  model; the corpus is a training-data prerequisite only.

Boundaries
----------

V1, V2 and ``engineering_heldout_v1`` are read-only inputs for the audit mode;
this builder never modifies them.  Passing this build authorizes nothing —
T9d-scale training remains behind its own protocol review.

Usage
-----

    .venv/bin/python scripts/build_engineering_corpus_v3.py --self-test
    .venv/bin/python scripts/build_engineering_corpus_v3.py \
        --output-dir research/engineering_judgment_corpus_v3
    .venv/bin/python scripts/build_engineering_corpus_v3.py \
        --check research/engineering_judgment_corpus_v3
    .venv/bin/python scripts/build_engineering_corpus_v3.py --audit \
        --corpus research/engineering_judgment_corpus_v3 \
        --heldout research/engineering_heldout_v1/items.jsonl \
        --compare-corpus research/engineering_judgment_corpus_v2 \
        --output results/engineering_corpus_v3_audit_receipt.json
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
from predict_toy_decisions import validate_request  # noqa: E402

SCHEMA_VERSION = "nanojev-engineering-judgment-corpus-v3"
MANIFEST_SCHEMA = "nanojev-engineering-judgment-manifest-v3"
ITEM_SCHEMA = "nanojev-engineering-judgment-item-v3"
SOURCE_GROUP_SCHEMA = "nanojev-engineering-source-group-v3"
AUDIT_SCHEMA = "nanojev-engineering-corpus-v3-audit-v1"
CATALOG_VERSION = "engineering-judgment-catalog-v3-scaled"
TRAINER_ROW_SCHEMA = v1.TRAINER_ROW_SCHEMA
ITEM_DIR, TRAINER_DIR, MANIFEST_NAME = v1.ITEM_DIR, v1.TRAINER_DIR, v1.MANIFEST_NAME
SPLITS, SPLIT_FILES, QUESTION_TYPES = v1.SPLITS, v1.SPLIT_FILES, v1.QUESTION_TYPES
DEFAULT_SEED = 20260921
SOURCE_ID = "scripts/build_engineering_corpus_v3.py"

#: ~55/15/10/20 train/dev/calibration/test over a 20-slot deterministic cycle.
SPLIT_CYCLE = ("train", "dev", "train", "train", "test",
               "train", "calibration", "train", "dev", "train",
               "train", "test", "train", "train", "dev",
               "train", "test", "calibration", "train", "test")

PAIRS_PER_FAMILY = 18
MIN_PAIRS_PER_FAMILY = 12
SAMPLE_CAP = 4000          # sampled candidates per family when the domain is large
ENUMERATE_CAP = 20000      # domains at/below this size are enumerated exhaustively

ALLOWED_QUESTION_KEYS = v1.ALLOWED_QUESTION_KEYS
_evaluation_derived = v2.evaluation_derived
_visible_fingerprint = v2.visible_fingerprint


# --------------------------------------------------------------------------------------
# fact domains and per-family specifications
# --------------------------------------------------------------------------------------

BOOL = ("bool",)


def INT(lo, hi, step=1):
    return ("int", lo, hi, step)


def domain_values(domain):
    if domain[0] == "bool":
        return [False, True]
    return list(range(domain[1], domain[2] + 1, domain[3]))


def _field(key, label, domain):
    return {"key": key, "label": label, "domain": domain}


FAMILY_DEFS = []


def _fam(family, feature, headings, services, tickets, fields, qids,
         instructions, boolean_criteria, choice_criteria, score_criteria,
         oracle, why):
    FAMILY_DEFS.append({
        "family": family, "feature": feature, "headings": tuple(headings),
        "services": tuple(services), "tickets": tuple(tickets),
        "fields": tuple(fields), "qids": dict(qids),
        "instructions": {qt: tuple(v) for qt, v in instructions.items()},
        "boolean_criteria": tuple(boolean_criteria),
        "choice_criteria": dict(choice_criteria),
        "score_criteria": list(score_criteria),
        "oracle": oracle, "why": why,
    })


# ---------------------------------------------------------------- incident_response

def _incident_oracle(f):
    if not f["comms_sent"]:
        choice = "send_comms"
    elif not f["mitigation_applied"]:
        choice = "apply_mitigation"
    elif not f["root_cause_found"]:
        choice = "diagnose_root_cause"
    elif not f["monitoring_green"]:
        choice = "watch_dashboards"
    else:
        choice = "close_incident"
    score = (int(f["comms_sent"]) + int(f["mitigation_applied"])
             + int(f["root_cause_found"]) + int(f["monitoring_green"]))
    return {"boolean": bool(f["mitigation_applied"] and f["monitoring_green"]),
            "choice": choice, "score": score}


_fam(
    "incident_response", "incident-response-sequencing",
    ("Incident timeline for {service} ({ticket}):",
     "Sev review board — {service} — {ticket}:",
     "Ongoing incident record, {service} [{ticket}]:"),
    ("checkout-api", "search-api", "auth-gateway", "billing-worker",
     "notify-svc", "inventory-api", "orders-db", "edge-proxy"),
    ("INC", "OPS", "ALRT"),
    (_field("sev1_confirmed", "sev-1 impact is confirmed", BOOL),
     _field("comms_sent", "stakeholder comms have been sent", BOOL),
     _field("mitigation_applied", "a mitigation is applied and holding", BOOL),
     _field("root_cause_found", "root cause is identified", BOOL),
     _field("monitoring_green", "dashboards green for the last 30 minutes", BOOL),
     _field("elapsed_minutes", "minutes since the page", INT(5, 300, 5))),
    {"boolean": "incident_stable", "choice": "next_step", "score": "milestones_cleared"},
    {"boolean": ("Is the incident stabilized (mitigated with green monitoring)?",
                 "Has the incident reached a stable, mitigated state?",
                 "Can this incident be called stabilized right now?"),
     "choice": ("Which response step should happen next?",
                "What is the correct next step in this incident?",
                "Pick the next action the responder should take."),
     "score": ("Score how many response milestones are cleared, 0 to 4.",
               "Rate incident response progress, 0 to 4.",
               "How far along is this response, 0 to 4?")},
    ({}, {"false": "the incident is not yet stabilized",
          "true": "the incident is stabilized"}),
    {"send_comms": "Send stakeholder communications before anything else",
     "apply_mitigation": "Apply or verify the mitigation",
     "diagnose_root_cause": "Work the root-cause diagnosis",
     "watch_dashboards": "Monitor dashboards for regression",
     "close_incident": "All milestones cleared; close the incident"},
    ("no response milestones cleared", "one milestone cleared",
     "two milestones cleared", "three milestones cleared",
     "all milestones cleared; the incident can be closed"),
    _incident_oracle,
    "the response sequence is fixed: comms first, then mitigation, then root cause, "
    "then a green-monitoring watch, then close; the score counts cleared milestones"),

# ---------------------------------------------------------------- capacity_planning

def _capacity_oracle(f):
    breach_month = f["projected_weeks"] <= 4 or f["headroom_pct"] < 15
    if f["projected_weeks"] <= 2 or f["headroom_pct"] < 10:
        choice = "provision_now"
    elif breach_month or (f["peak_season_soon"] and f["headroom_pct"] < 40):
        choice = "schedule_expansion"
    elif f["weekly_growth_pct"] >= 20 and not f["spare_budget"]:
        choice = "reduce_growth"
    else:
        choice = "monitor"
    score = 0
    if f["projected_weeks"] <= 10 or f["headroom_pct"] < 40:
        score = 1
    if breach_month or (f["peak_season_soon"] and f["headroom_pct"] < 40):
        score = 2
    if f["projected_weeks"] <= 2 or f["headroom_pct"] < 10:
        score = 3
    return {"boolean": breach_month, "choice": choice, "score": score}


_fam(
    "capacity_planning", "capacity-planning-judgment",
    ("Capacity review for {service} ({ticket}):",
     "Quarterly capacity board — {service} — {ticket}:",
     "Headroom assessment, {service} [{ticket}]:"),
    ("checkout-api", "search-api", "video-encode", "ml-serve", "orders-db",
     "log-ingest", "billing-worker", "cdn-origin"),
    ("CAP", "PLAT", "INF"),
    (_field("headroom_pct", "current headroom below the autoscale ceiling, percent",
            INT(5, 90, 5)),
     _field("weekly_growth_pct", "weekly usage growth, percent", INT(0, 30)),
     _field("projected_weeks", "projected weeks until usage reaches the ceiling",
            INT(1, 16)),
     _field("peak_season_soon", "a peak season starts within four weeks", BOOL),
     _field("spare_budget", "unused capacity budget remains this quarter", BOOL)),
    {"boolean": "breach_within_month", "choice": "capacity_action",
     "score": "capacity_urgency"},
    {"boolean": ("Does the trend breach the ceiling within a month?",
                 "Is a capacity breach expected inside four weeks?",
                 "Will headroom run out within the next month on this trend?"),
     "choice": ("What is the proportionate capacity action?",
                "Which capacity decision fits this picture?",
                "Pick the right capacity response."),
     "score": ("Score the urgency of the capacity decision, 0 to 3.",
               "Rate how urgent capacity action is here, 0 to 3.",
               "How urgent is this capacity call, 0 to 3?")},
    ({}, {"false": "the ceiling is more than a month away on this trend",
          "true": "the trend reaches the ceiling within a month"}),
    {"provision_now": "Provision additional capacity immediately",
     "schedule_expansion": "Schedule an expansion inside this planning cycle",
     "reduce_growth": "Work demand reduction; budget does not cover the growth",
     "monitor": "No action needed beyond routine monitoring"},
    ("no urgency; trend is far from the ceiling",
     "breach is possible this quarter; keep watching",
     "breach expected within about a month",
     "breach imminent; capacity must be added now"),
    _capacity_oracle,
    "urgency is driven by weeks-to-ceiling and raw headroom; a near-term peak season "
    "or growth that outruns the budget escalates an otherwise watchable trend"),

# ---------------------------------------------------------------- security_posture

def _security_oracle(f):
    if not f["patch_available"]:
        choice = "monitor" if f["workaround_in_place"] else "deploy_workaround"
    elif f["internet_facing"] and f["exploit_public"]:
        choice = "emergency_patch"
    elif f["cvss"] >= 7:
        choice = "scheduled_patch"
    else:
        choice = "routine_patch"
    score = (int(f["internet_facing"]) + int(f["exploit_public"])
             + int(f["cvss"] >= 7) + int(not f["workaround_in_place"]))
    return {"boolean": bool(f["patch_available"] and f["internet_facing"]
                            and f["exploit_public"]),
            "choice": choice, "score": score}


_fam(
    "security_posture", "security-posture-evaluation",
    ("Vulnerability review for {service} ({ticket}):",
     "Security bulletin triage — {service} — {ticket}:",
     "Exposure assessment, {service} [{ticket}]:"),
    ("edge-proxy", "auth-gateway", "bastion-host", "batch-worker",
     "api-gateway", "internal-ci", "web-frontend", "queue-consumer"),
    ("SEC", "VULN", "SECOPS"),
    (_field("internet_facing", "affected hosts are reachable from the internet", BOOL),
     _field("exploit_public", "a public exploit is observed in the wild", BOOL),
     _field("patch_available", "a fixed build exists in the package mirror", BOOL),
     _field("cvss", "advisory CVSS base score", INT(1, 10)),
     _field("workaround_in_place", "a compensating control is deployed", BOOL)),
    {"boolean": "emergency_change_allowed", "choice": "remediation_path",
     "score": "exposure_level"},
    {"boolean": ("Does the emergency change path apply to these hosts?",
                 "Is emergency patching outside the normal window permitted here?",
                 "May this be patched under the emergency-change rule?"),
     "choice": ("Which remediation path is correct?",
                "What is the right remediation order for this exposure?",
                "Pick the remediation that matches the posture."),
     "score": ("Score the exposure level, 0 to 4.",
               "Rate how exposed this fleet is, 0 to 4.",
               "How severe is the exposure, 0 to 4?")},
    ({}, {"false": "emergency patching is not permitted for this posture",
          "true": "emergency patching is permitted"}),
    {"emergency_patch": "Emergency-patch the exposed hosts now",
     "scheduled_patch": "Patch inside the next scheduled window",
     "routine_patch": "Patch on the routine cadence",
     "deploy_workaround": "No fix exists; deploy a compensating control",
     "monitor": "No fix exists and a control is in place; monitor"},
    ("minimal exposure", "one exposure factor present",
     "two exposure factors present", "three exposure factors present",
     "maximum exposure: reachable, exploitable, severe and unmitigated"),
    _security_oracle,
    "without a fix the only options are a compensating control or monitoring; with a "
    "fix, internet-facing plus a public exploit forces the emergency path, high CVSS "
    "the scheduled window, and everything else the routine cadence"),

# ---------------------------------------------------------------- pipeline_health

def _pipeline_oracle(f):
    if f["downstream_blocked"]:
        choice = "blocked"
    elif f["schema_drift"]:
        choice = "degraded_schema"
    elif f["lag_minutes"] > f["freshness_slo_minutes"] or f["failed_tasks_pct"] > 10:
        choice = "degraded_freshness"
    else:
        choice = "healthy"
    if f["downstream_blocked"]:
        score = 3
    elif f["schema_drift"] or f["failed_tasks_pct"] > 25:
        score = 2
    elif (f["lag_minutes"] > f["freshness_slo_minutes"]
          or f["failed_tasks_pct"] > 5):
        score = 1
    else:
        score = 0
    meets = (f["lag_minutes"] <= f["freshness_slo_minutes"]
             and f["failed_tasks_pct"] <= 5 and not f["downstream_blocked"])
    return {"boolean": meets, "choice": choice, "score": score}


_fam(
    "pipeline_health", "data-pipeline-health",
    ("Pipeline status report for {service} ({ticket}):",
     "Data-pipeline health check — {service} — {ticket}:",
     "Freshness board, {service} [{ticket}]:"),
    ("etl-orders", "etl-clickstream", "warehouse-sync", "feature-pipeline",
     "metrics-rollup", "cdc-stream", "reporting-dag", "export-feed"),
    ("DATA", "PIPE", "ETL"),
    (_field("lag_minutes", "current end-to-end lag in minutes", INT(0, 600, 10)),
     _field("freshness_slo_minutes", "freshness objective in minutes", INT(30, 480, 30)),
     _field("failed_tasks_pct", "share of tasks failing, percent", INT(0, 100, 5)),
     _field("schema_drift", "upstream schema drift detected", BOOL),
     _field("downstream_blocked", "downstream consumers are blocked", BOOL)),
    {"boolean": "meets_freshness_objective", "choice": "pipeline_state",
     "score": "degradation_level"},
    {"boolean": ("Is the pipeline currently meeting its freshness objective?",
                 "Does this pipeline satisfy its stated freshness target?",
                 "Is the pipeline inside its freshness commitment right now?"),
     "choice": ("What is the pipeline's current state?",
                "How should this pipeline be classified?",
                "Pick the status that best describes the pipeline."),
     "score": ("Score the degradation level, 0 to 3.",
               "Rate how degraded this pipeline is, 0 to 3.",
               "How degraded is the pipeline, 0 to 3?")},
    ({}, {"false": "the freshness objective is not currently met",
          "true": "the freshness objective is met"}),
    {"healthy": "Pipeline is inside all objectives",
     "degraded_freshness": "Lag or failures exceed the freshness objective",
     "degraded_schema": "Upstream schema drift must be reconciled first",
     "blocked": "Downstream consumers are blocked; escalate"},
    ("within all objectives", "freshness objective missed but recoverable",
     "schema drift or heavy failures need intervention",
     "downstream blocked; immediate escalation"),
    _pipeline_oracle,
    "a blocked consumer outranks every other signal; absent that, schema drift beats "
    "freshness because drift can poison downstream tables silently"),

# ---------------------------------------------------------------- deploy_risk

def _deploy_oracle(f):
    over = f["canary_error_permille"] > f["budget_permille"]
    if over:
        choice = "rollback"
    elif not f["rollback_tested"]:
        choice = "hold_for_rollback_drill"
    elif f["peak_hours"] and f["blast_radius_pct"] > 25:
        choice = "schedule_offpeak"
    else:
        choice = "promote"
    score = (int(over) + int(not f["rollback_tested"]) + int(f["peak_hours"])
             + int(f["blast_radius_pct"] > 25))
    return {"boolean": not over, "choice": choice, "score": score}


_fam(
    "deploy_risk", "deploy-risk-assessment",
    ("Deploy gate review for {service} ({ticket}):",
     "Canary assessment — {service} — {ticket}:",
     "Release risk board, {service} [{ticket}]:"),
    ("payments-svc", "checkout-api", "search-api", "mobile-backend",
     "pricing-svc", "media-encode", "auth-gateway", "feed-ranker"),
    ("REL", "DEPLOY", "SHIP"),
    (_field("canary_error_permille", "observed canary error rate, per mille",
            INT(0, 50)),
     _field("budget_permille", "canary error budget, per mille", INT(1, 20)),
     _field("rollback_tested", "rollback path rehearsed this week", BOOL),
     _field("peak_hours", "the deploy lands inside peak traffic hours", BOOL),
     _field("blast_radius_pct", "share of traffic this deploy can reach, percent",
            INT(5, 100, 5))),
    {"boolean": "within_error_budget", "choice": "deploy_decision",
     "score": "deploy_risk_level"},
    {"boolean": ("Is the canary inside its error budget?",
                 "Does the observed error rate stay within the canary budget?",
                 "Is the canary metric within budget?"),
     "choice": ("What should the deploy pipeline do next?",
                "Pick the correct action for this deploy gate.",
                "Which deploy decision fits these signals?"),
     "score": ("Score the deploy risk, 0 to 4.",
               "Rate the risk of promoting this deploy, 0 to 4.",
               "How risky is this promotion, 0 to 4?")},
    ({}, {"false": "the canary is over its error budget",
          "true": "the canary is within its error budget"}),
    {"promote": "Promote the canary to full rollout",
     "hold_for_rollback_drill": "Hold until the rollback path is rehearsed",
     "schedule_offpeak": "Reschedule the promotion outside peak hours",
     "rollback": "Roll back the canary and investigate"},
    ("no risk factors present", "one risk factor present",
     "two risk factors present", "three risk factors present",
     "all four risk factors present"),
    _deploy_oracle,
    "a budget breach always rolls back; absent a breach, an unrehearsed rollback path "
    "or a wide blast radius at peak gates the promotion"),

# ---------------------------------------------------------------- dependency_upgrade

def _dependency_oracle(f):
    if f["freeze_active"] and not f["security_fix"]:
        choice = "wait_for_window"
    elif f["breaking_uses"] > 0 and f["coverage_pct"] < 80:
        choice = "staged_migration"
    elif f["breaking_uses"] > 0:
        choice = "upgrade_with_review"
    else:
        choice = "upgrade_direct"
    score = (int(f["major_bump"]) + int(f["breaking_uses"] > 0)
             + int(f["coverage_pct"] < 60)
             + int(f["freeze_active"] and not f["security_fix"]))
    return {"boolean": bool(f["freeze_active"] and not f["security_fix"]),
            "choice": choice, "score": score}


_fam(
    "dependency_upgrade", "dependency-upgrade-decision",
    ("Dependency upgrade review for {service} ({ticket}):",
     "Upgrade spike notes — {service} — {ticket}:",
     "Version bump assessment, {service} [{ticket}]:"),
    ("billing-svc", "reporting-api", "user-portal", "data-export",
     "notification-hub", "geo-indexer", "session-store", "audit-log"),
    ("PLAT", "DEP", "MAINT"),
    (_field("major_bump", "the upgrade crosses a major version boundary", BOOL),
     _field("breaking_uses", "call sites using APIs removed by the upgrade",
            INT(0, 20)),
     _field("coverage_pct", "test coverage over affected call sites, percent",
            INT(0, 100, 5)),
     _field("freeze_active", "a change freeze window is active", BOOL),
     _field("security_fix", "the upgrade carries a security fix", BOOL)),
    {"boolean": "blocked_by_freeze", "choice": "upgrade_path",
     "score": "upgrade_risk"},
    {"boolean": ("Is this upgrade blocked by the active freeze?",
                 "Does the freeze window currently block this upgrade?",
                 "Is the freeze blocking this bump right now?"),
     "choice": ("What is the correct path for this upgrade?",
                "Which upgrade strategy applies here?",
                "Pick the upgrade path that fits."),
     "score": ("Score the upgrade risk, 0 to 4.",
               "Rate the risk of landing this upgrade, 0 to 4.",
               "How risky is this upgrade, 0 to 4?")},
    ({}, {"false": "the freeze does not block this upgrade",
          "true": "the freeze blocks this upgrade"}),
    {"upgrade_direct": "Land the upgrade directly; no breaking call sites",
     "upgrade_with_review": "Land with focused review of the breaking call sites",
     "staged_migration": "Migrate call sites in stages before bumping",
     "wait_for_window": "Hold the upgrade until the freeze lifts"},
    ("trivial bump", "one risk factor", "two risk factors",
     "three risk factors", "four risk factors; treat as high risk"),
    _dependency_oracle,
    "a freeze blocks everything except security fixes; breaking call sites under "
    "thin coverage require a staged migration, under real coverage a focused review"),

# ---------------------------------------------------------------- oncall_triage

def _oncall_oracle(f):
    if not f["user_impact"] and f["scope_pct"] < 5:
        choice = "downgrade_to_ticket"
    elif f["runbook_match"]:
        choice = "follow_runbook"
    elif f["recurring"]:
        choice = "escalate_known_issue"
    else:
        choice = "page_secondary"
    score = 0
    if f["scope_pct"] >= 20 or f["recurring"]:
        score = 1
    if f["user_impact"] or f["scope_pct"] >= 50:
        score = 2
    if f["user_impact"] and f["scope_pct"] >= 50:
        score = 3
    return {"boolean": bool(f["user_impact"] or f["scope_pct"] >= 20),
            "choice": choice, "score": score}


_fam(
    "oncall_triage", "oncall-triage-priority",
    ("Pager triage for {service} ({ticket}):",
     "On-call handoff note — {service} — {ticket}:",
     "Alert triage board, {service} [{ticket}]:"),
    ("edge-cache", "queue-consumer", "sso-gateway", "reporting-api",
     "ml-serve", "cron-fleet", "web-frontend", "streaming-join"),
    ("PAGE", "ONCALL", "ALRT"),
    (_field("user_impact", "end users are affected", BOOL),
     _field("scope_pct", "share of users in scope, percent", INT(1, 100)),
     _field("runbook_match", "a runbook entry matches the symptoms", BOOL),
     _field("recurring", "the same alert paged twice this week", BOOL),
     _field("minutes_since_page", "minutes since the page fired", INT(1, 120))),
    {"boolean": "page_worthy", "choice": "triage_action",
     "score": "triage_severity"},
    {"boolean": ("Does this page justify waking the on-call?",
                 "Is this alert page-worthy rather than a ticket?",
                 "Should the on-call treat this as a real page?"),
     "choice": ("What is the correct triage action?",
                "Pick the triage step that fits this page.",
                "How should the on-call triage this alert?"),
     "score": ("Score the triage severity, 0 to 3.",
               "Rate how severe this page is, 0 to 3.",
               "How severe is this alert, 0 to 3?")},
    ({}, {"false": "this does not merit a page",
          "true": "this merits a page"}),
    {"downgrade_to_ticket": "Convert to a ticket; no user impact and tiny scope",
     "follow_runbook": "Execute the matching runbook entry",
     "escalate_known_issue": "Escalate as a recurring known issue",
     "page_secondary": "Page the secondary owner; no runbook covers this"},
    ("benign; no user impact and narrow scope",
     "some signal worth a look", "real user impact or wide scope",
     "broad user impact; treat as severe"),
    _oncall_oracle,
    "a page with no user impact and tiny scope is a ticket; a matching runbook is "
    "the cheapest safe action; recurrence without a runbook escalates"),

# ---------------------------------------------------------------- postmortem_review

def _postmortem_oracle(f):
    n = (int(f["root_cause_identified"]) + int(f["action_items_assigned"])
         + int(f["timeline_complete"]) + int(f["blameless_language"])
         + int(f["customer_impact_quantified"]))
    if n == 5:
        choice = "publish"
    elif f["root_cause_identified"] and f["action_items_assigned"]:
        choice = "needs_polish"
    elif f["root_cause_identified"]:
        choice = "incomplete_actions"
    else:
        choice = "major_gaps"
    return {"boolean": bool(f["root_cause_identified"]
                            and f["action_items_assigned"]),
            "choice": choice, "score": n}


_fam(
    "postmortem_review", "postmortem-completeness",
    ("Postmortem draft review for {service} ({ticket}):",
     "Incident writeup checklist — {service} — {ticket}:",
     "Postmortem readiness board, {service} [{ticket}]:"),
    ("checkout-api", "search-api", "orders-db", "edge-proxy",
     "auth-gateway", "billing-worker", "etl-orders", "feed-ranker"),
    ("PM", "INC", "RETR"),
    (_field("root_cause_identified", "root cause is identified and stated", BOOL),
     _field("action_items_assigned", "action items exist with named owners", BOOL),
     _field("timeline_complete", "the timeline section is complete", BOOL),
     _field("blameless_language", "the writeup uses blameless language", BOOL),
     _field("customer_impact_quantified", "customer impact is quantified", BOOL)),
    {"boolean": "actionable", "choice": "postmortem_state",
     "score": "completeness_level"},
    {"boolean": ("Does the draft have both a root cause and owned action items?",
                 "Are root cause and owned follow-ups both present?",
                 "Is the postmortem actionable as written?"),
     "choice": ("What is the state of this postmortem draft?",
                "How should this postmortem be classified?",
                "Pick the draft's current state."),
     "score": ("Score the postmortem's completeness, 0 to 5.",
               "Rate how complete this postmortem is, 0 to 5.",
               "How complete is the writeup, 0 to 5?")},
    ({}, {"false": "root cause or owned action items are missing",
          "true": "root cause and owned action items are both present"}),
    {"publish": "All sections complete; publish the postmortem",
     "needs_polish": "Substance is there; finish the remaining sections",
     "incomplete_actions": "Root cause known but follow-up work is thin",
     "major_gaps": "Root cause itself is missing; return to the authors"},
    ("nothing usable yet", "one required section done",
     "two required sections done", "three required sections done",
     "four required sections done", "all five required sections done"),
    _postmortem_oracle,
    "publishable requires all five sections; a root cause plus owned action items is "
    "the substantive core — without the root cause the draft has major gaps"),

# ---------------------------------------------------------------- cost_anomaly

def _cost_oracle(f):
    if not f["attributed"]:
        choice = "investigate_attribution"
    elif f["spike_pct"] > 100 and f["budget_exceeded"]:
        choice = "throttle_and_alert"
    elif f["spike_pct"] > 50:
        choice = "rightsizing_review"
    else:
        choice = "note_and_monitor"
    score = (int(f["spike_pct"] > 200) + int(f["budget_exceeded"])
             + int(not f["attributed"]) + int(f["spike_pct"] > 50))
    return {"boolean": bool(f["spike_pct"] > 50 or f["budget_exceeded"]),
            "choice": choice, "score": score}


_fam(
    "cost_anomaly", "cost-anomaly-response",
    ("Cost anomaly ticket for {service} ({ticket}):",
     "Spend review — {service} — {ticket}:",
     "Billing anomaly board, {service} [{ticket}]:"),
    ("gpu-training", "cdn-origin", "log-ingest", "warehouse-sync",
     "media-encode", "etl-clickstream", "sandbox-env", "backup-store"),
    ("COST", "FIN", "BILL"),
    (_field("spike_pct", "spend spike vs the 30-day baseline, percent",
            INT(0, 500, 10)),
     _field("attributed", "the spike is attributed to a known workload", BOOL),
     _field("budget_exceeded", "the monthly budget is already exceeded", BOOL),
     _field("new_service", "a new service launched inside the window", BOOL),
     _field("commitment_covered", "spend is covered by committed-use discounts",
            BOOL)),
    {"boolean": "needs_action", "choice": "cost_response",
     "score": "anomaly_severity"},
    {"boolean": ("Does this anomaly require an action beyond noting it?",
                 "Is action required on this cost spike?",
                 "Does this spike cross the act threshold?"),
     "choice": ("What is the correct response to this cost anomaly?",
                "Pick the proportionate cost response.",
                "Which response fits this anomaly?"),
     "score": ("Score the anomaly severity, 0 to 4.",
               "Rate the severity of this cost anomaly, 0 to 4.",
               "How severe is this cost anomaly, 0 to 4?")},
    ({}, {"false": "noting and monitoring is sufficient",
          "true": "an action is required"}),
    {"investigate_attribution": "Attribute the spike before taking action",
     "throttle_and_alert": "Throttle the workload and alert the budget owner",
     "rightsizing_review": "Schedule a rightsizing review this week",
     "note_and_monitor": "Record the anomaly and keep monitoring"},
    ("trivial fluctuation", "one severity factor", "two severity factors",
     "three severity factors", "four severity factors; severe anomaly"),
    _cost_oracle,
    "an unattributed spike is investigated before any throttle; attributed spikes "
    "scale with magnitude, and an exceeded budget turns a large spike into an "
    "immediate throttle"),

# ---------------------------------------------------------------- slo_breach

def _slo_oracle(f):
    if f["budget_remaining_pct"] <= 0:
        choice = "freeze_feature_work"
    elif f["burn_rate"] > 10:
        choice = "emergency_reliability"
    elif f["burn_rate"] > 4 or f["user_complaints"]:
        choice = "reliability_focus"
    else:
        choice = "normal_cadence"
    score = 0
    if f["burn_rate"] > 4 or f["user_complaints"]:
        score = 1
    if f["burn_rate"] > 10:
        score = 2
    if f["budget_remaining_pct"] <= 0:
        score = 3
    return {"boolean": f["budget_remaining_pct"] <= 0,
            "choice": choice, "score": score}


_fam(
    "slo_breach", "slo-breach-handling",
    ("Objective review for {service} ({ticket}):",
     "Error-budget board — {service} — {ticket}:",
     "Reliability review, {service} [{ticket}]:"),
    ("checkout-api", "search-api", "sso-gateway", "payments-svc",
     "mobile-backend", "feed-ranker", "notify-svc", "api-gateway"),
    ("REL", "OBJ", "SRE"),
    (_field("budget_remaining_pct", "error budget remaining, percent",
            INT(-50, 100, 5)),
     _field("burn_rate", "current burn-rate multiple", INT(1, 20)),
     _field("window_days_left", "days left in the objective window", INT(1, 30)),
     _field("user_complaints", "user complaints are arriving", BOOL)),
    {"boolean": "budget_exhausted", "choice": "reliability_policy",
     "score": "burn_urgency"},
    {"boolean": ("Is the error budget for this window exhausted?",
                 "Has the error budget run out?",
                 "Is the objective budget spent?"),
     "choice": ("Which reliability policy applies now?",
                "Pick the posture the team should take.",
                "What does the error-budget policy dictate here?"),
     "score": ("Score the burn urgency, 0 to 3.",
               "Rate how urgent the reliability response is, 0 to 3.",
               "How urgent is the burn situation, 0 to 3?")},
    ({}, {"false": "error budget remains in the window",
          "true": "the error budget is exhausted"}),
    {"freeze_feature_work": "Freeze feature work until the window resets",
     "emergency_reliability": "Drop everything for reliability work",
     "reliability_focus": "Prioritize reliability work over new features",
     "normal_cadence": "Budget is healthy; keep the normal cadence"},
    ("burn is healthy", "elevated burn or complaints; reliability gets priority",
     "burn is critical; emergency reliability work",
     "budget exhausted; feature work freezes"),
    _slo_oracle,
    "an exhausted budget freezes feature work outright; a critical burn rate is an "
    "emergency, and a merely elevated burn or arriving complaints shift priority to "
    "reliability"),

# ---------------------------------------------------------------- schema_migration

def _schema_migration_oracle(f):
    if not f["expand_deployed"]:
        choice = "deploy_expand_step"
    elif f["old_reads_present"] or f["wait_cycles_done"] < f["required_wait"]:
        choice = "wait"
    else:
        choice = "run_contract_step"
    score = (int(not f["expand_deployed"]) + int(f["old_reads_present"])
             + int(f["wait_cycles_done"] < f["required_wait"]))
    safe = (f["expand_deployed"] and not f["old_reads_present"]
            and f["wait_cycles_done"] >= f["required_wait"])
    return {"boolean": safe, "choice": choice, "score": score}


_fam(
    "schema_migration", "schema-migration-safety",
    ("Migration plan review for {service} ({ticket}):",
     "Expand-contract tracker — {service} — {ticket}:",
     "Schema change board, {service} [{ticket}]:"),
    ("orders-db", "users-db", "ledger-db", "catalog-db",
     "sessions-store", "events-store", "billing-db", "inventory-db"),
    ("MIG", "DB", "SCHEMA"),
    (_field("expand_deployed", "the expand step is fully deployed", BOOL),
     _field("old_reads_present", "production code still reads the old shape", BOOL),
     _field("wait_cycles_done", "release cycles elapsed since the expand step",
            INT(0, 10)),
     _field("required_wait", "release cycles the plan requires before contract",
            INT(1, 10))),
    {"boolean": "contract_step_safe", "choice": "migration_step",
     "score": "premature_contract_risk"},
    {"boolean": ("Is running the contract step safe right now?",
                 "May the contract step run today?",
                 "Is it safe to drop the old shape now?"),
     "choice": ("What is the next correct migration step?",
                "Pick the step that should run next.",
                "Which migration action applies?"),
     "score": ("Score the risk of contracting now, 0 to 3.",
               "Rate the risk of running the contract step today, 0 to 3.",
               "How risky is contracting at this point, 0 to 3?")},
    ({}, {"false": "the contract step is not safe yet",
          "true": "the contract step is safe"}),
    {"deploy_expand_step": "Deploy the expand step first",
     "wait": "Wait: readers remain or the rollback window has not elapsed",
     "run_contract_step": "Run the contract step; all preconditions cleared"},
    ("contract is safe to run now", "one blocker remains",
     "two blockers remain", "nothing about the contract step is ready"),
    _schema_migration_oracle,
    "expand-contract ordering is strict: the expand step must be deployed, no "
    "production reader may remain on the old shape, and the planned wait must have "
    "elapsed before the contract step is safe"),

# ---------------------------------------------------------------- alert_tuning

def _alert_oracle(f):
    if f["actioned_pct"] < 10:
        choice = "retune_or_remove"
    elif f["actioned_pct"] < 50 and f["sleep_pages"]:
        choice = "raise_threshold"
    elif not f["slo_aligned"]:
        choice = "realign_to_objective"
    else:
        choice = "keep"
    if f["actioned_pct"] >= 80:
        score = 4
    elif f["actioned_pct"] >= 50:
        score = 3
    elif f["actioned_pct"] >= 25:
        score = 2
    elif f["actioned_pct"] >= 10:
        score = 1
    else:
        score = 0
    noisy = (f["actioned_pct"] < 10
             or (f["alerts_per_week"] > 50 and f["actioned_pct"] < 30))
    return {"boolean": noisy, "choice": choice, "score": score}


_fam(
    "alert_tuning", "alert-quality-tuning",
    ("Alert quality review for {service} ({ticket}):",
     "Paging-signal audit — {service} — {ticket}:",
     "Alert tuning board, {service} [{ticket}]:"),
    ("checkout-api", "log-ingest", "edge-cache", "cron-fleet",
     "streaming-join", "metrics-rollup", "api-gateway", "batch-worker"),
    ("ALRT", "OBS", "MON"),
    (_field("alerts_per_week", "pages fired per week", INT(0, 200, 5)),
     _field("actioned_pct", "share of pages that led to an action, percent",
            INT(0, 100, 5)),
     _field("slo_aligned", "the page threshold is aligned to the objective", BOOL),
     _field("sleep_pages", "the alert pages during sleep hours", BOOL)),
    {"boolean": "alert_is_noisy", "choice": "tuning_decision",
     "score": "signal_quality"},
    {"boolean": ("Is this alert producing noise rather than signal?",
                 "Does this alert qualify as noisy?",
                 "Is the alert mostly noise on these numbers?"),
     "choice": ("What is the right tuning decision?",
                "Pick the action that improves this alert.",
                "Which tuning move applies here?"),
     "score": ("Score the alert's signal quality, 0 to 4.",
               "Rate this alert's signal quality, 0 to 4.",
               "How good is the alert's signal, 0 to 4?")},
    ({}, {"false": "the alert is acceptable as configured",
          "true": "the alert is noisy"}),
    {"keep": "Keep the alert as configured",
     "raise_threshold": "Raise the page threshold to cut sleep-hour noise",
     "realign_to_objective": "Realign the threshold to the service objective",
     "retune_or_remove": "Retune or remove: almost no page leads to action"},
    ("nearly pure noise", "weak signal", "mixed signal",
     "mostly signal", "excellent signal"),
    _alert_oracle,
    "an alert almost nobody acts on is noise to retune or remove; a middling alert "
    "that wakes people gets a higher threshold; a misaligned-but-useful alert is "
    "realigned to the objective"),

# ---------------------------------------------------------------- backup_assurance

def _backup_oracle(f):
    if f["last_backup_hours"] > 72:
        choice = "fix_backups_first"
    elif not f["checksum_ok"]:
        choice = "investigate_corruption"
    elif f["restore_tested_days"] > f["drill_required_days"]:
        choice = "run_restore_drill"
    elif not f["offsite_copy"]:
        choice = "add_offsite_copy"
    else:
        choice = "verified"
    score = (int(f["last_backup_hours"] > 72) + int(not f["checksum_ok"])
             + int(f["restore_tested_days"] > f["drill_required_days"])
             + int(not f["offsite_copy"]))
    verified = (f["checksum_ok"] and f["last_backup_hours"] <= 72
                and f["restore_tested_days"] <= f["drill_required_days"]
                and f["offsite_copy"])
    return {"boolean": verified, "choice": choice, "score": score}


_fam(
    "backup_assurance", "backup-restore-assurance",
    ("Backup assurance report for {service} ({ticket}):",
     "Restore-drill ledger — {service} — {ticket}:",
     "Backup verification board, {service} [{ticket}]:"),
    ("orders-db", "users-db", "ledger-db", "events-store",
     "config-store", "media-bucket", "warehouse-sync", "audit-log"),
    ("BKP", "DR", "DATA"),
    (_field("last_backup_hours", "hours since the last successful backup",
            INT(1, 720)),
     _field("checksum_ok", "the latest backup checksum verifies", BOOL),
     _field("restore_tested_days", "days since the last successful restore drill",
            INT(0, 400, 5)),
     _field("drill_required_days", "required restore-drill cadence in days",
            INT(30, 180, 30)),
     _field("offsite_copy", "an offsite copy of the backup set exists", BOOL)),
    {"boolean": "backup_set_verified", "choice": "assurance_state",
     "score": "assurance_gap"},
    {"boolean": ("Does this backup set count as verified under the runbook?",
                 "Is the backup set verified as recoverable?",
                 "May this backup set be called verified?"),
     "choice": ("What is the assurance state of this backup set?",
                "Pick the state that describes these backups.",
                "How should this backup posture be classified?"),
     "score": ("Score the assurance gap, 0 to 4.",
               "Rate the size of the assurance gap, 0 to 4.",
               "How large is the assurance gap, 0 to 4?")},
    ({}, {"false": "the set does not meet the verified bar",
          "true": "the set meets the verified bar"}),
    {"fix_backups_first": "Backups are stale; fix the backup job first",
     "investigate_corruption": "Latest checksum fails; investigate corruption",
     "run_restore_drill": "Run a restore drill; cadence has lapsed",
     "add_offsite_copy": "Add an offsite copy to close the gap",
     "verified": "Backups verified: recent, clean, drilled and offsite"},
    ("no assurance gap", "one assurance gap", "two assurance gaps",
     "three assurance gaps", "four assurance gaps; treat as unprotected"),
    _backup_oracle,
    "verification needs all of: a recent successful backup, a clean checksum, an "
    "in-cadence restore drill and an offsite copy; the first failed precondition "
    "in that order is the state"),

# ---------------------------------------------------------------- access_review

def _access_oracle(f):
    if not f["approval_recorded"]:
        choice = "deny"
    elif f["external_party"] and f["admin_scope"]:
        choice = "restrict_scope"
    elif not f["expiry_set"]:
        choice = "grant_with_expiry"
    elif not f["mfa_enrolled"]:
        choice = "grant_pending_mfa"
    else:
        choice = "grant"
    grantable = (f["approval_recorded"] and f["mfa_enrolled"] and f["expiry_set"]
                 and not (f["external_party"] and f["admin_scope"]))
    score = (int(f["admin_scope"]) + int(f["external_party"])
             + int(not f["mfa_enrolled"]) + int(not f["expiry_set"]))
    return {"boolean": grantable, "choice": choice, "score": score}


_fam(
    "access_review", "access-grant-review",
    ("Access grant review for {service} ({ticket}):",
     "Permission request — {service} — {ticket}:",
     "Access decision board, {service} [{ticket}]:"),
    ("admin-console", "prod-db", "deploy-pipeline", "secrets-vault",
     "billing-svc", "support-tools", "metrics-ui", "internal-ci"),
    ("ACC", "IAM", "SEC"),
    (_field("admin_scope", "the requested grant carries admin scope", BOOL),
     _field("mfa_enrolled", "the requester is enrolled in MFA", BOOL),
     _field("approval_recorded", "a manager approval is on record", BOOL),
     _field("expiry_set", "an expiry date is set on the grant", BOOL),
     _field("external_party", "the requester is outside the organization", BOOL)),
    {"boolean": "grantable_as_requested", "choice": "access_decision",
     "score": "grant_risk"},
    {"boolean": ("Is this grant approvable exactly as requested?",
                 "May the grant be issued as it stands?",
                 "Is the requested grant acceptable unchanged?"),
     "choice": ("What is the correct access decision?",
                "Pick the decision that fits this request.",
                "Which access outcome applies here?"),
     "score": ("Score the risk of this grant, 0 to 4.",
               "Rate the grant's risk, 0 to 4.",
               "How risky is this grant, 0 to 4?")},
    ({}, {"false": "the grant is not acceptable as requested",
          "true": "the grant is acceptable as requested"}),
    {"grant": "Grant the access as requested",
     "grant_with_expiry": "Grant only after an expiry date is set",
     "grant_pending_mfa": "Grant once the requester enrolls in MFA",
     "restrict_scope": "Grant only with reduced scope; external plus admin is barred",
     "deny": "Deny: no manager approval is on record"},
    ("no risk factors", "one risk factor", "two risk factors",
     "three risk factors", "four risk factors; treat as high risk"),
    _access_oracle,
    "no recorded approval means deny; an external requester with admin scope is "
    "restricted rather than granted; missing expiry or MFA become grant conditions"),

# ---------------------------------------------------------------- feature_flag_rollout

def _flag_oracle(f):
    if f["guardrail_breach"] or not f["error_rate_ok"]:
        choice = "rollback_flag"
    elif f["current_exposure_pct"] >= f["target_pct"]:
        choice = "complete"
    elif f["hours_at_stage"] < f["min_stage_hours"]:
        choice = "hold_stage"
    else:
        choice = "advance_stage"
    safe = (f["error_rate_ok"] and not f["guardrail_breach"]
            and f["hours_at_stage"] >= f["min_stage_hours"]
            and f["current_exposure_pct"] < f["target_pct"])
    score = (int(f["guardrail_breach"]) + int(not f["error_rate_ok"])
             + int(f["target_pct"] - f["current_exposure_pct"] > 50))
    return {"boolean": safe, "choice": choice, "score": score}


_fam(
    "feature_flag_rollout", "feature-flag-rollout",
    ("Rollout stage review for {service} ({ticket}):",
     "Flag exposure board — {service} — {ticket}:",
     "Progressive rollout status, {service} [{ticket}]:"),
    ("recommendations-v2", "new-checkout", "search-rerank", "pricing-v3",
     "onboarding-flow", "dark-mode", "upload-v2", "chat-assist"),
    ("FLAG", "FEAT", "EXP"),
    (_field("current_exposure_pct", "current traffic exposure, percent",
            INT(0, 100, 5)),
     _field("target_pct", "target exposure for the rollout, percent",
            INT(0, 100, 5)),
     _field("error_rate_ok", "error rate is inside the guardrail", BOOL),
     _field("guardrail_breach", "a guardrail metric has breached", BOOL),
     _field("hours_at_stage", "hours spent at the current stage", INT(0, 96)),
     _field("min_stage_hours", "minimum hours required per stage", INT(1, 48))),
    {"boolean": "safe_to_advance", "choice": "rollout_step",
     "score": "rollout_risk"},
    {"boolean": ("Is it safe to advance this rollout to the next stage?",
                 "May the rollout advance now?",
                 "Is advancing the flag safe at this point?"),
     "choice": ("What is the correct next step for this rollout?",
                "Pick the rollout action that applies.",
                "Which step should the rollout take?"),
     "score": ("Score the rollout risk, 0 to 3.",
               "Rate the risk of this rollout, 0 to 3.",
               "How risky is the rollout position, 0 to 3?")},
    ({}, {"false": "advancing now is not safe",
          "true": "advancing now is safe"}),
    {"advance_stage": "Advance to the next exposure stage",
     "hold_stage": "Hold the current stage until the minimum soak elapses",
     "complete": "Rollout is at target; mark complete",
     "rollback_flag": "Roll the flag back; a guardrail or error budget broke"},
    ("low risk; rollout on plan", "one risk factor",
     "two risk factors", "all risk factors present"),
    _flag_oracle,
    "a guardrail breach or a broken error budget always rolls back; at target the "
    "rollout is done; otherwise the stage advances only after its minimum soak"),

# ---------------------------------------------------------------- quota_enforcement

def _quota_oracle(f):
    if f["shared_ip"]:
        choice = "throttle_shared_pool"
    elif f["abuse_score"] > 80 or f["prior_warnings"] >= 3:
        choice = "suspend_key"
    elif f["burst_pct_of_quota"] > 150:
        choice = "throttle"
    else:
        choice = "allow"
    suspend = (not f["shared_ip"]
               and (f["abuse_score"] > 80 or f["prior_warnings"] >= 3))
    score = (int(f["abuse_score"] > 80) + int(f["prior_warnings"] >= 3)
             + int(f["burst_pct_of_quota"] > 200)
             + int(f["burst_pct_of_quota"] > 150))
    return {"boolean": suspend, "choice": choice, "score": score}


_fam(
    "quota_enforcement", "quota-enforcement-decision",
    ("Quota enforcement review for {service} ({ticket}):",
     "Abuse-score board — {service} — {ticket}:",
     "Client quota decision, {service} [{ticket}]:"),
    ("public-api", "graphql-edge", "webhook-relay", "search-api",
     "export-feed", "auth-gateway", "media-encode", "batch-api"),
    ("QUOTA", "API", "EDGE"),
    (_field("abuse_score", "client abuse score, 0 to 100", INT(0, 100, 5)),
     _field("paying_tier", "the client is on a paying tier", BOOL),
     _field("burst_pct_of_quota", "current burst as a share of quota, percent",
            INT(0, 300, 10)),
     _field("prior_warnings", "prior warnings issued to this client", INT(0, 5)),
     _field("shared_ip", "the client shares an egress IP with other tenants",
            BOOL)),
    {"boolean": "suspension_warranted", "choice": "enforcement_action",
     "score": "abuse_severity"},
    {"boolean": ("Is key suspension warranted for this client?",
                 "Does this client merit suspension rather than throttling?",
                 "Should the client's key be suspended?"),
     "choice": ("What enforcement action is correct?",
                "Pick the enforcement action that fits.",
                "Which quota action applies to this client?"),
     "score": ("Score the abuse severity, 0 to 4.",
               "Rate how abusive this client pattern is, 0 to 4.",
               "How severe is the abuse, 0 to 4?")},
    ({}, {"false": "suspension is not warranted",
          "true": "suspension is warranted"}),
    {"allow": "Allow the traffic; nothing warrants action",
     "throttle": "Throttle the client back inside quota",
     "throttle_shared_pool": "Throttle the shared pool; do not punish one tenant",
     "suspend_key": "Suspend the client key outright"},
    ("no abuse signal", "one severity factor", "two severity factors",
     "three severity factors", "four severity factors; severe abuse"),
    _quota_oracle,
    "a shared egress IP means throttle the pool, never the tenant; otherwise a high "
    "abuse score or repeated warnings suspend, and a large burst alone throttles"),

# ---------------------------------------------------------------- data_retention

def _retention_oracle(f):
    overdue = f["age_days"] > f["max_retention_days"]
    if f["legal_hold"]:
        choice = "preserve"
    elif f["contains_pii"] and overdue:
        choice = "expire_now"
    elif overdue:
        choice = "expire_scheduled"
    else:
        choice = "retain"
    score = (int(f["contains_pii"]) + int(overdue and not f["legal_hold"])
             + int(f["in_backups"]))
    return {"boolean": bool(overdue and not f["legal_hold"]),
            "choice": choice, "score": score}


_fam(
    "data_retention", "data-retention-posture",
    ("Retention review for {service} ({ticket}):",
     "Dataset lifecycle board — {service} — {ticket}:",
     "Retention posture, {service} [{ticket}]:"),
    ("support-chats", "clickstream-raw", "uploads-bucket", "analytics-events",
     "email-archive", "device-telemetry", "session-logs", "survey-exports"),
    ("RET", "PRIV", "DATA"),
    (_field("contains_pii", "records contain personal identifiers", BOOL),
     _field("age_days", "age of the oldest record in days", INT(0, 2000, 10)),
     _field("max_retention_days", "policy retention ceiling in days",
            INT(90, 1095, 15)),
     _field("legal_hold", "a legal hold is active on the dataset", BOOL),
     _field("in_backups", "the data is also inside backup archives", BOOL)),
    {"boolean": "retention_overdue", "choice": "retention_action",
     "score": "compliance_risk"},
    {"boolean": ("Is part of this dataset past its retention deadline?",
                 "Has this dataset overstayed its retention ceiling?",
                 "Is any of this data overdue for expiry?"),
     "choice": ("What is the correct retention action?",
                "Pick the retention decision that applies.",
                "Which retention action is required?"),
     "score": ("Score the compliance risk, 0 to 3.",
               "Rate the retention compliance risk, 0 to 3.",
               "How risky is this retention posture, 0 to 3?")},
    ({}, {"false": "nothing is overdue for expiry",
          "true": "overdue records exist"}),
    {"retain": "Retain the dataset; it is inside policy",
     "expire_scheduled": "Schedule expiry of the overdue records",
     "expire_now": "Expire the overdue personal records immediately",
     "preserve": "Preserve everything; a legal hold is active"},
    ("compliant posture", "one risk factor", "two risk factors",
     "three risk factors; act this sprint"),
    _retention_oracle,
    "a legal hold preserves everything; otherwise personal identifiers past the "
    "retention ceiling expire immediately, non-personal overdue data expires on "
    "schedule, and in-policy data is retained"),

# ---------------------------------------------------------------- release_readiness

def _release_oracle(f):
    if f["open_blockers"] > 0:
        choice = "block_release"
    elif f["freeze_window"]:
        choice = "wait_for_window"
    elif not f["tests_green"] or f["coverage_delta"] < 0:
        choice = "fix_first"
    elif f["signoffs"] < f["required_signoffs"]:
        choice = "gather_signoffs"
    else:
        choice = "ship"
    shippable = (f["open_blockers"] == 0 and not f["freeze_window"]
                 and f["tests_green"] and f["coverage_delta"] >= 0
                 and f["signoffs"] >= f["required_signoffs"])
    score = (int(f["tests_green"]) + int(f["coverage_delta"] >= 0)
             + int(f["signoffs"] >= f["required_signoffs"])
             + int(f["open_blockers"] == 0))
    return {"boolean": shippable, "choice": choice, "score": score}


_fam(
    "release_readiness", "release-gate-readiness",
    ("Release gate review for {service} ({ticket}):",
     "Ship-readiness board — {service} — {ticket}:",
     "Release checklist, {service} [{ticket}]:"),
    ("mobile-app", "web-frontend", "api-gateway", "desktop-client",
     "firmware-image", "plugin-sdk", "admin-console", "cli-tool"),
    ("REL", "SHIP", "GATE"),
    (_field("tests_green", "the full test suite is green", BOOL),
     _field("coverage_delta", "coverage change vs the main branch, points",
            INT(-20, 20)),
     _field("signoffs", "recorded release approvals", INT(0, 3)),
     _field("required_signoffs", "approvals the release policy requires",
            INT(1, 3)),
     _field("open_blockers", "open blocking issues", INT(0, 5)),
     _field("freeze_window", "a release freeze window is active", BOOL)),
    {"boolean": "shippable_now", "choice": "release_verdict",
     "score": "readiness_level"},
    {"boolean": ("May this release ship right now?",
                 "Is the release shippable as it stands?",
                 "Can the release go out at this moment?"),
     "choice": ("What is the release verdict?",
                "Pick the gate outcome for this release.",
                "Which verdict applies to this release?"),
     "score": ("Score release readiness, 0 to 4.",
               "Rate how ready this release is, 0 to 4.",
               "How ready is the release, 0 to 4?")},
    ({}, {"false": "the release may not ship now",
          "true": "the release may ship now"}),
    {"ship": "All gates green; ship the release",
     "gather_signoffs": "Collect the missing approvals, then ship",
     "fix_first": "Fix failing tests or the coverage regression first",
     "wait_for_window": "Hold until the freeze window lifts",
     "block_release": "Block: open blocking issues remain"},
    ("nothing ready", "one gate cleared", "two gates cleared",
     "three gates cleared", "all gates cleared; ready to ship"),
    _release_oracle,
    "open blockers stop the release outright; a freeze window holds it; failing "
    "tests or a coverage regression must be fixed first; missing approvals are the "
    "last gate before ship")

FAMILIES = tuple(spec["family"] for spec in FAMILY_DEFS)
FAMILY_MAP = {spec["family"]: spec for spec in FAMILY_DEFS}
QIDS = {qtype: {spec["family"]: spec["qids"][qtype] for spec in FAMILY_DEFS}
        for qtype in QUESTION_TYPES}

assert len(FAMILIES) == len(set(FAMILIES)) == 18, "v3 expects exactly 18 families"


# --------------------------------------------------------------------------------------
# deterministic fact sampling and oracle validation
# --------------------------------------------------------------------------------------

def _fact_key(facts):
    return json.dumps(facts, sort_keys=True, separators=(",", ":"))


def _oracle_checked(spec, facts):
    """Run the family oracle and enforce the answer contract on every fact set."""
    gold = spec["oracle"](facts)
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

    Domains at or below ENUMERATE_CAP are enumerated exhaustively and shuffled
    deterministically; larger domains are sampled with replacement up to
    SAMPLE_CAP unique fact sets.
    """
    keys = [f["key"] for f in spec["fields"]]
    values = [domain_values(f["domain"]) for f in spec["fields"]]
    total = 1
    for v in values:
        total *= len(v)
    rng = random.Random(f"nanojev-engineering-catalog-v3:{seed}:{spec['family']}")
    combos = []
    if total <= ENUMERATE_CAP:
        combos = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
        rng.shuffle(combos)
    else:
        seen = set()
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


def select_for_coverage(spec, candidates, target):
    """Order candidates so every label is covered early, then balance choice labels."""
    pool = list(candidates)
    selected, picked_ids = [], set()

    def take(pred, limit=None):
        for cand in pool:
            if id(cand) in picked_ids or not pred(cand["gold"]):
                continue
            picked_ids.add(id(cand))
            selected.append(cand)
            return True
        return False

    for key in spec["choice_criteria"]:
        take(lambda g, k=key: g["choice"] == k)
    for level in range(len(spec["score_criteria"])):
        take(lambda g, lv=level: g["score"] == lv)
    for value in (False, True):
        take(lambda g, v=value: g["boolean"] is v)
    # Balance the choice label histogram while filling to the soft cap.
    buckets = defaultdict(list)
    for cand in pool:
        if id(cand) not in picked_ids:
            buckets[cand["gold"]["choice"]].append(cand)
    soft_cap = min(len(pool), max(target * 3, target + len(spec["choice_criteria"])))
    while len(selected) < soft_cap:
        smallest = min(buckets, key=lambda k: sum(
            1 for c in selected if c["gold"]["choice"] == k))
        if not buckets[smallest]:
            break
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
# state rendering: five surface templates x per-pair entities/order/phrasing
# --------------------------------------------------------------------------------------

def _fmt(value, template):
    if isinstance(value, bool):
        if template in (1, 4):
            return "true" if value else "false"
        return "yes" if value else "no"
    return str(value)


def render_state(facts, spec, render):
    order = render["field_order"]
    labels = {f["key"]: f["label"] for f in spec["fields"]}
    template = render["template"]
    heading = (render["heading"].replace("{service}", render["service"])
               .replace("{ticket}", render["ticket"]))
    lines = []
    if template == 0:                                   # report bullets
        lines.append(heading)
        for key in order:
            lines.append(f"- {labels[key]}: {_fmt(facts[key], template)}")
    elif template == 1:                                 # ticket excerpt
        lines.append(f"{render['ticket']} {spec['family'].replace('_', '-')} "
                     f"({render['service']}):")
        for key in order:
            lines.append(f"  {key} = {_fmt(facts[key], template)}")
    elif template == 2:                                 # metric table
        lines.extend((heading, "| field | value |", "| --- | --- |"))
        for key in order:
            lines.append(f"| {labels[key]} | {_fmt(facts[key], template)} |")
    elif template == 3:                                 # checklist
        lines.append(heading)
        for key in order:
            value = facts[key]
            if isinstance(value, bool):
                lines.append(f"* [{'x' if value else ' '}] {labels[key]}")
            else:
                lines.append(f"* {labels[key]}: {_fmt(value, template)}")
    elif template == 4:                                 # config block
        lines.append(f"# {render['service']} {spec['family'].replace('_', '-')} "
                     f"review ({render['ticket']})")
        for key in order:
            lines.append(f"{key}={_fmt(facts[key], template)}")
    else:
        raise ValueError(f"unknown template {template!r}")
    return "\n".join(lines)


def make_render(spec, seed, ordinal, attempt=0):
    rng = random.Random(
        f"nanojev-engineering-v3-render:{seed}:{spec['family']}:{ordinal}:{attempt}")
    order = [f["key"] for f in spec["fields"]]
    rng.shuffle(order)
    return {"template": rng.randrange(5),
            "heading": spec["headings"][rng.randrange(len(spec["headings"]))],
            "service": spec["services"][rng.randrange(len(spec["services"]))],
            "ticket": f"{spec['tickets'][rng.randrange(len(spec['tickets']))]}"
                      f"-{rng.randrange(1000, 9999)}",
            "field_order": order,
            "text_variant": rng.randrange(64)}


def question_body(spec, qtype, variant):
    variants = spec["instructions"][qtype]
    instructions = variants[variant % len(variants)]
    if qtype == "boolean":
        criteria = spec["boolean_criteria"][variant % len(spec["boolean_criteria"])]
        body = {"type": "boolean", "instructions": instructions}
        if criteria:
            body["criteria"] = dict(criteria)
        return body
    if qtype == "choice":
        return {"type": "choice", "instructions": instructions,
                "criteria": dict(spec["choice_criteria"])}
    if qtype == "score":
        return {"type": "score", "instructions": instructions,
                "criteria": list(spec["score_criteria"])}
    raise ValueError(f"unknown question type {qtype!r}")


def build_request(spec, facts, render, state_id):
    return {"states": [{"id": state_id, "state": render_state(facts, spec, render),
                        "questions": {spec["qids"][qt]:
                                      question_body(spec, qt, render["text_variant"])
                                      for qt in QUESTION_TYPES}}]}


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
    return len(FAMILY_DEFS)


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
            for attempt in range(5):
                trial = make_render(spec, seed, ordinal, attempt)
                text = render_state(base_facts, spec, trial)
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
                variant_state = render_state(mutation["variant_facts"], spec, render)
                if variant_state not in used_states:
                    picked = (mutation, variant_state)
                    break
            if picked is None:
                continue
            mutation, variant_state = picked
            delta = v1.fact_delta(base_facts, mutation["variant_facts"])
            if len(delta) != 1 or delta[0][0] != mutation["key"]:
                raise ValueError(f"{family}#{ordinal}: mutation must change exactly "
                                 f"the named fact")
            if delta[0][1] == delta[0][2]:
                raise ValueError(f"{family}#{ordinal}: mutation does not change "
                                 "the value")
            used_states.add(base_state)
            used_states.add(variant_state)
            rule_id = f"{family}-r{made:03d}"
            pairs.append({
                "pair_id": f"ej3-{family}-p{made:03d}",
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
        missing_choice = sorted(set(spec["choice_criteria"])
                                - {g["choice"] for g in family_golds})
        missing_scores = sorted(set(range(len(spec["score_criteria"])))
                                - {g["score"] for g in family_golds})
        missing_bool = sorted({False, True} - {g["boolean"] for g in family_golds})
        if missing_choice or missing_scores or missing_bool:
            raise ValueError(f"{family}: label coverage gaps: choice "
                             f"{missing_choice}, score {missing_scores}, "
                             f"boolean {missing_bool}")
    return pairs


# --------------------------------------------------------------------------------------
# items, components, splits (V2 architecture, V3 content)
# --------------------------------------------------------------------------------------

def make_items(pair, member):
    family = pair["family"]
    spec = pair["rule"]
    facts = pair[f"{member}_facts"]
    request = build_request(spec, facts, pair["render"], pair["state_id"])
    items = []
    for qtype in QUESTION_TYPES:
        qid, gold = gold_for(family, facts, qtype)
        criteria_values = request["states"][0]["questions"][qid].get("criteria", {})
        vector = v1.probability_vector(qtype, criteria_values, gold)
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
            "expected": {
                "answer_type": qtype,
                "candidate_keys": vector["keys"],
                "gold": gold,
                "gold_index": vector["gold_index"],
                "distribution": vector["probabilities"],
                "distribution_kind": "hard_label",
                "calibrated": False,
            },
            "provenance": {
                "source_id": SOURCE_ID,
                "rule_id": pair["rule_id"],
                "catalog_version": CATALOG_VERSION,
                "rule": spec["why"],
                "why_correct": (f"under the frozen {family} rule the facts in this "
                                f"state fix {qid}={gold!r}"),
                "fact_basis": {key: facts[key] for key in sorted(facts)},
                "fact_keys": sorted(facts),
                "surface_template": pair["render"]["template"],
                "authoring": ("programmatic_base_state_from_family_domain"
                              if member == "base" else
                              "programmatic_one_fact_mutation_of_a_sampled_state"),
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

    for family, comps in per_family.items():
        for index, (_, comp_nodes) in enumerate(sorted(comps)):
            split = SPLIT_CYCLE[(index + abs(seed)) % len(SPLIT_CYCLE)]
            fps = sorted({_visible_fingerprint(item) for node in comp_nodes
                          for item in members[node]})
            group_id = "ejc-v3-" + digest_value(
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
        "version": "engineering-judgment-rule-v3",
        "statement": ("A pair is two served-contract request bodies whose states "
                      "differ in exactly one fact leaf of the family's declared "
                      "domain; the expected answers of both members are computed "
                      "from their own facts with the family's frozen oracle, and "
                      "the declared flip question types are required to differ. "
                      "A pair whose declared flip does not hold is never emitted."),
        "base_states": ("programmatically sampled/enumerated fact sets rendered "
                        "through five surface templates with per-pair entities, "
                        "headings, field order and phrasing variants"),
        "split_unit": ("canonical-input connected component: both members of a "
                       "contrastive pair plus every member sharing an identical "
                       "visible state+question always share one split"),
        "split_rule": ("deterministic per-family component position + seed offset "
                       "over a 20-slot cycle (~55/15/10/20 "
                       "train/dev/calibration/test); content never selects a split"),
        "question_types": list(QUESTION_TYPES),
        "families": list(FAMILIES),
        "label_coverage": ("every family must cover all choice candidates, all "
                           "score levels and both boolean values across its pair "
                           "members or the build fails"),
        "views": {
            "items/<split>.jsonl": ("audit view: one object per (member x question "
                                    "type) with the full request, declared gold, "
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
        "created_by": "scripts/build_engineering_corpus_v3.py",
        "catalog_version": CATALOG_VERSION,
        "scale_up": {
            "motivation": ("W42 T9d result: full-backbone training on corpus v2 "
                           "(228 items / 26 components) overfit instantly; data "
                           "scale and diversity are the binding constraint"),
            "canonical_input_components": len(grouped),
            "families": len(FAMILIES),
            "surface_templates": 5,
            "v1_corpus_preserved": "research/engineering_judgment_corpus_v1 unchanged",
            "v2_corpus_preserved": "research/engineering_judgment_corpus_v2 unchanged",
            "heldout_preserved": "research/engineering_heldout_v1 unchanged",
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
            "policy": ("V3 keeps the V1/V2 refusal wall: no source path naming an "
                       "evaluation corpus or the pre-registered abstention survey, "
                       "no reserved evaluation-corpus token in emitted content, "
                       "and no evaluation-derived declared source. Evaluation "
                       "cohorts are never a training, validation, calibration or "
                       "selection source."),
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

TRAINER_GOLD_PROBS_KIND = "deterministic_truth"
TRAINER_GOLD_LABEL_KIND = "deterministic_truth"


def trainer_row(item):
    state = item["request"]["states"][0]
    qid = item["qid"]
    return {
        "id": item["item_id"],
        "state_id": item["state_id"],
        "family_id": item["family"],
        "split": item["split"],
        "state": state["state"],
        "questions": {qid: state["questions"][qid]},
        "gold": {qid: item["expected"]["gold"]},
        "gold_probs": {qid: dict(item["expected"]["distribution"])},
        "gold_probs_kind": {qid: TRAINER_GOLD_PROBS_KIND},
        "gold_label_kind": {qid: TRAINER_GOLD_LABEL_KIND},
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
            "cohort": "engineering_judgment_corpus_v3",
            "catalog_version": CATALOG_VERSION,
            "derived_from_evaluation_corpus": False,
            "human_reviewed": False,
            "license": "CC0-1.0",
            "training_authorized": False,
        },
    }


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
    if set(expected["distribution"]) != set(expected["candidate_keys"]):
        errors.append(f"{item_id}: distribution keys must equal candidate keys")
    elif not 0 <= expected["gold_index"] < len(expected["candidate_keys"]):
        errors.append(f"{item_id}: gold_index is outside the candidate list")
    elif item["question_type"] == "boolean":
        if expected["candidate_keys"][expected["gold_index"]] != \
                ("true" if expected["gold"] is True else "false"):
            errors.append(f"{item_id}: gold_index does not address the gold answer")
    elif item["question_type"] == "choice":
        if expected["candidate_keys"][expected["gold_index"]] != expected["gold"]:
            errors.append(f"{item_id}: gold_index does not address the gold answer")
    elif expected["gold_index"] != expected["gold"]:
        errors.append(f"{item_id}: gold_index does not address the gold score level")
    if expected["distribution_kind"] != "hard_label" or \
            expected["calibrated"] is not False:
        errors.append(f"{item_id}: distribution must stay an uncalibrated hard label")
    if item["family"] in FAMILIES:
        recomputed = gold_for(item["family"], item["provenance"]["fact_basis"],
                              item["question_type"])
        if recomputed[0] != qid or recomputed[1] != expected["gold"]:
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


def validate_manifest(manifest):
    """Re-derive the corpus from the seed and fail on any hand edit."""
    errors = []
    if not isinstance(manifest, dict):
        return ["manifest must be a JSON object"]
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        errors.append(f"schema_version must be {MANIFEST_SCHEMA!r}")
    if manifest.get("created_by") != "scripts/build_engineering_corpus_v3.py":
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
            if base["expected"]["gold"] != variant["expected"]["gold"]:
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
        "builder": "scripts/build_engineering_corpus_v3.py",
        "components": (manifest["scale_up"]["canonical_input_components"]
                       if manifest else 0),
        "families": len(FAMILIES),
        "source_groups": manifest["source_group_count"] if manifest else 0,
        "pairs": manifest["pair_count"] if manifest else 0,
        "items": manifest["item_count"] if manifest else 0,
        "items_by_family": ({family: manifest["counts_by_family"][family]["items"]
                             for family in FAMILIES} if manifest else {}),
        "items_by_split": dict(manifest["counts_by_split"]) if manifest else {},
        "training_performed": False, "wrote_output": False, "errors": errors,
    }


# --------------------------------------------------------------------------------------
# read-only audit receipt (audit_engineering_corpus_v2 conventions, v3 schema)
# --------------------------------------------------------------------------------------

def _cohort_fingerprints(rows):
    from audit_engineering_corpus_v1 import references
    return references(rows)


def audit_report(corpus_dir, heldout_path=None, compare_dir=None,
                 compare_seed=None):
    """Read-only audit: builder integrity + split isolation + cross-corpus overlap."""
    from audit_engineering_corpus_v1 import audit_rows, file_hash
    from train_pipeline_decisions import read_training_records

    corpus_dir = Path(corpus_dir).resolve(strict=True)
    inputs = [corpus_dir / MANIFEST_NAME]
    inputs += [corpus_dir / TRAINER_DIR / f"{s}.jsonl" for s in SPLITS]
    inputs += [corpus_dir / ITEM_DIR / f"{s}.jsonl" for s in SPLITS]
    if heldout_path is not None:
        inputs.append(Path(heldout_path).resolve(strict=True))
    if compare_dir is not None:
        compare_dir = Path(compare_dir).resolve(strict=True)
        inputs += [compare_dir / MANIFEST_NAME]
        inputs += [compare_dir / TRAINER_DIR / f"{s}.jsonl" for s in SPLITS]
    before = {str(p): file_hash(p) for p in inputs}

    integrity_errors = check(corpus_dir)
    manifest = json.loads((corpus_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    rows, _ = read_training_records(corpus_dir / TRAINER_DIR)
    audit = audit_rows(rows)

    corpus_refs = _cohort_fingerprints(rows)
    isolation = {"method": ("exact canonical model-visible-input equality: sha256 "
                            "of {state, question.type, instructions, criteria}, "
                            "sorted-key JSON; the same fingerprint used by "
                            "scripts/heldout_v1_isolation_check.py")}

    heldout_report = None
    block_reasons = list(audit["block_reasons"])
    if integrity_errors:
        block_reasons.append("builder_integrity_check_failed")
    if heldout_path is not None:
        heldout_rows, _ = read_training_records(Path(heldout_path))
        heldout_refs = _cohort_fingerprints(heldout_rows)
        common = sorted(set(corpus_refs) & set(heldout_refs))
        per_split = {}
        for split in SPLITS:
            split_rows = [r for r in rows if r["split"] == split]
            shared = sorted(set(_cohort_fingerprints(split_rows))
                            & set(heldout_refs))
            per_split[split] = {"shared_canonical_inputs": len(shared),
                                "shared_input_sha256": shared}
        item_fps = {_visible_fingerprint(item) for item in manifest["items"]}
        heldout_item_overlap = sorted(item_fps & set(heldout_refs))
        identity = {}
        for key in ("id", "state_id", "family_id"):
            identity[key] = sorted({r[key] for r in rows}
                                   & {r[key] for r in heldout_rows})
        identity["source_group_id"] = sorted(
            {r.get("metadata", {}).get("source_group_id") for r in rows}
            & {r.get("metadata", {}).get("source_group_id")
               for r in heldout_rows} - {None})
        heldout_report = {
            "heldout": str(heldout_path), "heldout_items": len(heldout_rows),
            "common_canonical_inputs_total": len(common),
            "common_canonical_inputs_by_split": per_split,
            "item_view_overlap": len(heldout_item_overlap),
            "identity_overlap": identity,
            "overlap_details": [{"input_sha256": k,
                                 "references": corpus_refs[k] + heldout_refs[k]}
                                for k in common],
        }
        if common or heldout_item_overlap:
            block_reasons.append("heldout_shares_canonical_inputs_with_corpus_v3")
        if any(identity.values()):
            block_reasons.append("heldout_shares_declared_identities_with_corpus_v3")

    compare_report = None
    if compare_dir is not None:
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
        v3_item_fps = {_visible_fingerprint(item) for item in manifest["items"]}
        compare_report = {
            "compare_corpus": str(compare_dir),
            "policy": ("zero canonical-input overlap is required between corpus v3 "
                       "and ALL of corpus v2 (not only its test split): v2 remains "
                       "a separately versioned cohort and any shared visible input "
                       "would blur the isolation evidence"),
            "common_canonical_inputs_total": len(common_all),
            "common_canonical_inputs_by_v2_split": per_split,
            "item_view_overlap": len(compare_item_fps & v3_item_fps),
            "overlap_details": [
                {"input_sha256": k,
                 "references": corpus_refs[k] + compare_refs[k]}
                for k in common_all],
        }
        if common_all or (compare_item_fps & v3_item_fps):
            block_reasons.append("corpus_v3_shares_canonical_inputs_with_corpus_v2")

    # Component statistics straight from the manifest's declared groups.
    comp_sizes = Counter()
    comp_splits = {}
    for item in manifest["items"]:
        gid = item["source_group_id"]
        comp_sizes[gid] += 1
        comp_splits.setdefault(gid, set()).add(item["split"])
    cross_split_groups = sorted(g for g, s in comp_splits.items() if len(s) > 1)
    if cross_split_groups:
        block_reasons.append("source_group_crosses_splits")

    # Probe the V2 auditor in-process; it is schema-bound to V2 (its own docstring)
    # so the expected outcome on a V3 corpus is a clean refusal, recorded verbatim.
    v2_auditor_probe = {"ran": False}
    try:
        import audit_engineering_corpus_v2 as audit_v2
        v2_auditor_probe["ran"] = True
        try:
            report = audit_v2.audit_corpus(corpus_dir)
            v2_auditor_probe.update(
                status=report["status"],
                block_reasons=report["block_reasons"],
                holdout_block_reasons=report["holdout_block_reasons"])
        except Exception as error:  # noqa: BLE001 - record the refusal verbatim
            v2_auditor_probe.update(
                status="blocked_isolation_plan_only",
                error=f"{type(error).__name__}: {error}",
                note=("the v2 auditor hard-requires manifest schema v2 and the v1 "
                      "family table; a v3 corpus cannot pass it by design — the "
                      "substantive audit for v3 is this receipt"))
    except ImportError as error:
        v2_auditor_probe["error"] = f"ImportError: {error}"

    if manifest["item_count"] < 1000:
        block_reasons.append("below_scale_target_1000_items")
    if len(FAMILIES) < 15:
        block_reasons.append("below_family_target_15")

    after = {str(p): file_hash(p) for p in inputs}
    inputs_changed = before != after
    if inputs_changed:
        block_reasons.append("input_files_changed_during_audit")

    return {
        "schema_version": AUDIT_SCHEMA,
        "corpus": str(corpus_dir),
        "audit_subject": ("V3 scaled corpus; V1, V2 and engineering_heldout_v1 "
                          "preserved as immutable evidence"),
        "source_hashes": before,
        "source_hashes_after": after,
        "input_files_changed": inputs_changed,
        "builder_integrity_errors": integrity_errors,
        "builder": "scripts/build_engineering_corpus_v3.py",
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
        "heldout_isolation": heldout_report,
        "corpus_v2_overlap": compare_report,
        "v2_auditor_probe": v2_auditor_probe,
        "scale_targets": {"items_min": 1000, "families_min": 15,
                          "items_actual": manifest["item_count"],
                          "families_actual": len(FAMILIES),
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
            "leakage is addressed by disjoint families, domains, templates and "
            "declared provenance",
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
            "engineering_heldout_v1 remains the independent heldout; never train "
            "or select on it",
            "T9d-scale training on v3 requires its own protocol review; this "
            "corpus authorizes nothing",
            "V1 and V2 corpora remain immutable evidence; do not merge without "
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
    parser.add_argument("--heldout", type=Path,
                        help="heldout items.jsonl for the overlap check (audit)")
    parser.add_argument("--compare-corpus", type=Path,
                        help="second corpus dir for the overlap check (audit)")
    parser.add_argument("--output", type=Path, help="receipt path (audit, write-once)")
    parser.add_argument("--print-families", action="store_true")
    args = parser.parse_args(argv)

    if args.print_families:
        print(json.dumps({"families": list(FAMILIES), "count": len(FAMILIES),
                          "pairs_per_family_target": PAIRS_PER_FAMILY},
                         indent=2, sort_keys=True))
        return 0
    if args.self_test and (args.output_dir is not None or args.check is not None
                           or args.audit):
        parser.error("--self-test is mutually exclusive with --output-dir/--check/--audit")
    if args.audit:
        if args.corpus is None:
            parser.error("--audit requires --corpus")
        try:
            report = audit_report(args.corpus, args.heldout, args.compare_corpus)
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(json.dumps({"status": "error", "error": str(error),
                              "training_authorized": False}, indent=2,
                             sort_keys=True))
            return 2
        serialized = json.dumps(report, indent=2, ensure_ascii=False,
                                allow_nan=False, sort_keys=True) + "\n"
        if args.output:
            if args.output.resolve().is_relative_to(args.corpus.resolve()):
                parser.error("audit output cannot live inside the audited corpus")
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(serialized)
            print(json.dumps({"status": report["status"],
                              "output": str(args.output),
                              "block_reasons": report["block_reasons"]},
                             indent=2, sort_keys=True))
        else:
            print(serialized, end="")
        return 0 if not report["block_reasons"] else 2
    if not args.self_test and args.output_dir is None and args.check is None:
        parser.error("one of --self-test, --output-dir, --check or --audit is required")
    if args.self_test:
        report = self_test(args.seed)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] == "ok" else 1
    if args.check is not None:
        errors = check(args.check)
        print(json.dumps({"status": "ok" if not errors else "failed",
                          "checked": str(args.check), "errors": errors,
                          "training_performed": False}, indent=2, sort_keys=True))
        return 0 if not errors else 1
    try:
        manifest = build(args.output_dir, args.seed)
    except ValueError as error:
        print(json.dumps({"status": "failed", "error": str(error)},
                         indent=2, sort_keys=True))
        return 2
    print(json.dumps({
        "status": "ok", "output_dir": str(args.output_dir),
        "seed": manifest["seed"],
        "components": manifest["scale_up"]["canonical_input_components"],
        "families": len(FAMILIES),
        "source_groups": manifest["source_group_count"],
        "pairs": manifest["pair_count"], "items": manifest["item_count"],
        "items_by_split": dict(manifest["counts_by_split"]),
        "content_sha256": manifest["content_sha256"],
        "training_performed": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
