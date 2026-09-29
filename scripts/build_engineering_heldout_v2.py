#!/usr/bin/env python3
"""Engineering-judgment heldout V2: scaled independent evaluation cohort (X4).

``research/engineering_heldout_v1`` is 69 rows over 23 source groups — too small
for the statistical power X4 needs.  This builder emits
``research/engineering_heldout_v2/items.jsonl`` plus ``manifest.json``: a fresh
held-out cohort of 312 rows (104 states x 3 question types) over 104 source
groups, one group per state.

Independence properties, by construction:

* The 13 families are declared only in this file.  None of the corpus v1/v2/v3/v4
  family rules, fact domains, field sets, qids, instruction strings or candidate
  vocabularies is reused, and none of heldout_v1's scenarios is reused.
* States render through four heldout-specific surface formats (prose sentence
  chains, yaml-ish records, csv-style field dumps, bracketed log excerpts) that
  do not occur in the corpora; entity and reference vocabularies are disjoint.
* ``source_group_id`` values (``eh2-*``) live in a disjoint namespace from the
  corpora's ``ejc-*`` groups and heldout_v1's ``eh1-*``; row ``id``,
  ``state_id`` and ``family_id`` are disjoint by name and by content.
* Gold is deterministic per row: every fact set's answer is computed by the
  family's frozen oracle and the emitted ``gold`` is re-checked against that
  oracle at build time; ``gold_probs`` is the matching one-hot vector with kind
  ``deterministic_truth`` (train_pipeline_decisions contract).
* Each row carries a provenance block declaring authorship, the scenario's
  domain and an explicit independence statement.  All rows carry
  ``split="test"`` — evaluation only, never merged into train/dev/calibration.

State selection is deterministic: each family's fact domain is enumerated (or
sampled when very large) and eight states are picked round-robin across the
distinct choice labels, so every family covers several answers.  Option-count
strata: boolean (2), choice at 4/5/6/8/12 candidates, score at 3/4/5 levels.

This file authors data only.  It performs no training and no measurement, and
authorizes nothing.

Usage

    .venv/bin/python scripts/build_engineering_heldout_v2.py \
        --output-dir research/engineering_heldout_v2
    .venv/bin/python scripts/build_engineering_heldout_v2.py --self-test
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_toy_decisions import validate_request  # noqa: E402
from train_pipeline_decisions import validate_training_row  # noqa: E402
import build_engineering_corpus_v1 as v1  # noqa: E402

SCHEMA_VERSION = "nanojev-engineering-heldout-v2"
MANIFEST_SCHEMA = "nanojev-engineering-heldout-v2-manifest-v1"
ROW_SCHEMA = "nanojev-engineering-judgment-trainer-row-v1"
QUESTION_TYPES = ("boolean", "choice", "score")
DEFAULT_SEED = 20260923
STATES_PER_FAMILY = 8
SAMPLE_CAP = 4000
ENUMERATE_CAP = 20000

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


def _fam(family, domain_desc, entities, refs, fields, qids, instructions,
         boolean_criteria, choice_criteria, score_criteria, oracle, why):
    FAMILY_DEFS.append({
        "family": family, "domain_desc": domain_desc,
        "entities": tuple(entities), "refs": tuple(refs),
        "fields": tuple(fields), "qids": dict(qids),
        "instructions": {qt: tuple(v) for qt, v in instructions.items()},
        "boolean_criteria": tuple(boolean_criteria),
        "choice_criteria": dict(choice_criteria),
        "score_criteria": list(score_criteria),
        "oracle": oracle, "why": why,
    })


def _levels(*texts):
    return list(texts)


# ---------------------------------------------------------------- dns_failover_policy

def _dns_oracle(f):
    if not f["failover_armed"]:
        choice = "keep_primary"
    elif not f["primary_healthy"]:
        choice = "promote_secondary" if f["secondary_rtt"] <= 300 \
            else "hold_for_health"
    elif f["primary_rtt"] > 3 * f["secondary_rtt"]:
        choice = "shift_traffic"
    else:
        choice = "keep_primary"
    sev = (int(not f["failover_armed"]) + int(not f["primary_healthy"])
           + int(f["primary_rtt"] > 200) + int(f["dns_ttl"] > 300))
    return {"boolean": bool(f["failover_armed"] and f["primary_healthy"]),
            "choice": choice, "score": min(sev, 3)}


_fam(
    "dns_failover_policy", "dns traffic management",
    ("resolver-east", "resolver-west", "edge-dns", "authoritative-ns"),
    ("DNS", "EDGE"),
    (_field("failover_armed", "automatic failover is armed", BOOL),
     _field("primary_healthy", "the primary endpoint passes health checks", BOOL),
     _field("primary_rtt", "primary endpoint RTT in ms", INT(10, 400, 10)),
     _field("secondary_rtt", "secondary endpoint RTT in ms", INT(10, 400, 10)),
     _field("dns_ttl", "DNS TTL in seconds", INT(30, 600, 30))),
    {"boolean": "failover_ready", "choice": "dns_action",
     "score": "failover_risk"},
    {"boolean": ("Is failover armed with a healthy primary?",
                 "Is the DNS failover posture healthy and armed?"),
     "choice": ("What is the correct DNS failover action?",
                "Pick the action the failover policy dictates."),
     "score": ("Score the failover risk, 0 to 3.",
               "Rate the failover risk, 0 to 3.")},
    ({}, {"false": "failover is not armed or the primary is unhealthy",
          "true": "failover is armed and the primary is healthy"}),
    {"keep_primary": "Keep serving the primary endpoint",
     "shift_traffic": "Shift traffic toward the much faster secondary",
     "promote_secondary": "Primary unhealthy: promote the secondary",
     "hold_for_health": "Secondary too slow: hold and investigate health"},
    _levels("nominal", "elevated", "high", "critical"),
    _dns_oracle,
    "unarmed failover keeps the primary; an unhealthy primary promotes the "
    "secondary when it is fast enough else holds; a primary three times slower "
    "than the secondary shifts traffic; otherwise keep primary",)


# ---------------------------------------------------------------- blob_lifecycle

def _blob_oracle(f):
    if f["legal_hold"]:
        choice = "retain"
    elif f["tier"] == 2 and f["age_days"] > 180:
        choice = "purge_review"
    elif f["access_count"] == 0 and f["age_days"] > 90:
        choice = "demote"
    else:
        choice = "keep_tier"
    band = 0 if f["age_days"] <= 30 else (1 if f["age_days"] <= 120 else
          (2 if f["age_days"] <= 240 else 3))
    return {"boolean": bool(f["legal_hold"]), "choice": choice, "score": band}


_fam(
    "blob_lifecycle", "object storage lifecycle",
    ("media-vault", "backup-blobs", "telemetry-store", "artifact-store"),
    ("BLOB", "STOR"),
    (_field("legal_hold", "a legal hold is attached to the object", BOOL),
     _field("tier", "storage tier 0-2 (0 hot, 2 archive)", INT(0, 2)),
     _field("age_days", "object age in days", INT(1, 400)),
     _field("access_count", "reads in the last 90 days", INT(0, 50)),
     _field("size_gb", "object size in GB", INT(1, 200))),
    {"boolean": "protected_from_moves", "choice": "lifecycle_action",
     "score": "age_band"},
    {"boolean": ("Is this object protected from any lifecycle move?",
                 "Does a legal hold freeze this object in place?"),
     "choice": ("What lifecycle action applies to this object?",
                "Pick the correct lifecycle transition."),
     "score": ("Score the object's age band, 0 to 3.",
               "Rate the object age band, 0 to 3.")},
    ({}, {"false": "the object may transition under policy",
          "true": "a legal hold protects the object"}),
    {"keep_tier": "Keep the object at its current tier",
     "demote": "No reads in 90+ days: demote to a colder tier",
     "purge_review": "Archive-tier object older than 180 days: review for purge",
     "retain": "Legal hold: retain and skip all lifecycle moves"},
    _levels("fresh (<=30 days)", "settling (31-120)", "old (121-240)",
            "ancient (>240 days)"),
    _blob_oracle,
    "a legal hold retains the object; archive objects older than 180 days go to "
    "purge review; unread objects older than 90 days demote; otherwise keep "
    "the tier; the score bands object age",)


# ---------------------------------------------------------------- webhook_retry

def _webhook_oracle(f):
    if f["attempts"] >= f["max_attempts"]:
        choice = "dead_letter"
    elif f["status_class"] == 0:
        choice = "mark_delivered"
    elif f["status_class"] == 1:
        choice = "drop_event"
    elif f["status_class"] == 2:
        choice = "retry_long_backoff"
    else:
        choice = "retry_short_backoff"
    remaining = max(0, f["max_attempts"] - f["attempts"])
    band = 0 if remaining >= 3 else (1 if remaining == 2 else 2)
    return {"boolean": choice in ("retry_short_backoff", "retry_long_backoff"),
            "choice": choice, "score": band}


_fam(
    "webhook_retry", "webhook delivery retries",
    ("billing-hooks", "crm-hooks", "shipping-hooks", "identity-hooks"),
    ("HOOK", "RETRY"),
    (_field("attempts", "delivery attempts already made", INT(0, 8)),
     _field("max_attempts", "configured maximum attempts", INT(3, 8)),
     _field("status_class", "last response class: 0=2xx, 1=4xx-other, 2=429, "
            "3=5xx", INT(0, 3)),
     _field("jitter_on", "retry jitter is enabled", BOOL)),
    {"boolean": "will_retry", "choice": "retry_verdict",
     "score": "attempts_left"},
    {"boolean": ("Will this event be retried?",
                 "Does the retry policy schedule another attempt?"),
     "choice": ("What is the correct retry verdict?",
                "Pick the retry decision the policy dictates."),
     "score": ("Score remaining attempts, 0 to 2 (3+, 2, fewer).",
               "Rate how many retries remain, 0 to 2.")},
    ({}, {"false": "no further retry happens",
          "true": "another retry is scheduled"}),
    {"mark_delivered": "Last attempt succeeded: mark the event delivered",
     "drop_event": "Permanent client error: drop the event",
     "retry_long_backoff": "Rate-limited: retry on the long backoff",
     "retry_short_backoff": "Server error: retry on the short backoff",
     "dead_letter": "Attempts exhausted: move the event to the dead-letter queue"},
    _levels("three or more attempts left", "two attempts left",
            "one or zero attempts left"),
    _webhook_oracle,
    "exhausted attempts dead-letter; a 2xx marks delivered; a permanent 4xx "
    "drops; a 429 retries on the long backoff; a 5xx retries on the short "
    "backoff; the score bands remaining attempts",)


# ---------------------------------------------------------------- node_drain

def _drain_oracle(f):
    evictable = f["pods"] * f["evictable_pct"] // 100
    after = f["pods"] - evictable
    if not f["cordoned"]:
        choice = "cordon_first"
    elif f["pods"] == 0:
        choice = "remove_node"
    elif after < f["pdb_min"]:
        choice = "wait_for_pdb"
    elif f["spot_node"]:
        choice = "rush_drain"
    else:
        choice = "drain_gradual"
    urgency = (int(f["spot_node"]) + int(f["pods"] > 40)
               + int(after < f["pdb_min"]) + int(not f["cordoned"]))
    return {"boolean": bool(f["cordoned"] and after >= f["pdb_min"]),
            "choice": choice, "score": min(urgency, 3)}


_fam(
    "node_drain", "cluster node lifecycle",
    ("k8s-pool-a", "k8s-pool-b", "batch-nodes", "gpu-nodes"),
    ("NODE", "DRAIN"),
    (_field("cordoned", "the node is already cordoned", BOOL),
     _field("pods", "pods currently scheduled on the node", INT(0, 64)),
     _field("evictable_pct", "share of pods that can be evicted now, percent",
            INT(0, 100, 5)),
     _field("pdb_min", "minimum pods the disruption budget requires", INT(0, 8)),
     _field("spot_node", "the node is a reclaimable spot instance", BOOL)),
    {"boolean": "drainable_now", "choice": "drain_step",
     "score": "drain_urgency"},
    {"boolean": ("May the drain proceed immediately under the PDB?",
                 "Is the node drainable right now?"),
     "choice": ("What is the correct next drain step?",
                "Pick the step the drain policy dictates."),
     "score": ("Score the drain urgency, 0 to 3.",
               "Rate how urgent this drain is, 0 to 3.")},
    ({}, {"false": "cordon first or the PDB would be violated",
          "true": "the drain may proceed"}),
    {"cordon_first": "Cordon the node before anything else",
     "wait_for_pdb": "Wait: eviction would breach the disruption budget",
     "drain_gradual": "Drain gradually within the budget",
     "rush_drain": "Spot node: drain aggressively before reclaim",
     "remove_node": "Node is empty: remove it"},
    _levels("routine", "elevated", "high", "critical"),
    _drain_oracle,
    "an uncordoned node is cordoned first; an empty node is removed; eviction "
    "that would leave fewer pods than the PDB minimum waits; spot nodes rush; "
    "otherwise drain gradually",)


# ---------------------------------------------------------------- secret_scope_grant

def _scope_oracle(f):
    if f["scope_sensitivity"] == 2 and f["requester_role"] == 0:
        choice = "deny"
    elif not f["approved_ticket"] and f["scope_sensitivity"] > 0:
        choice = "deny"
    elif f["requester_role"] == 0 or f["expires_days"] == 0 \
            or f["expires_days"] > 60:
        choice = "grant_expiring"
    else:
        choice = "grant"
    risk = (int(f["scope_sensitivity"] == 2) + int(not f["approved_ticket"])
            + int(f["expires_days"] == 0 or f["expires_days"] > 60))
    return {"boolean": choice != "deny", "choice": choice, "score": min(risk, 2)}


_fam(
    "secret_scope_grant", "secrets access control",
    ("vault-prod", "kms-edge", "secrets-hub", "token-vault"),
    ("SEC", "VAULT"),
    (_field("requester_role", "requester role 0-2 (0 contractor, 2 owner)",
            INT(0, 2)),
     _field("scope_sensitivity", "scope sensitivity 0-2 (2 most sensitive)",
            INT(0, 2)),
     _field("approved_ticket", "an approved access ticket exists", BOOL),
     _field("expires_days", "requested grant lifetime in days (0 = indefinite)",
            INT(0, 90))),
    {"boolean": "grant_permitted", "choice": "grant_verdict",
     "score": "grant_risk"},
    {"boolean": ("May any grant be issued for this request?",
                 "Is some form of grant permitted here?"),
     "choice": ("What is the correct grant verdict?",
                "Pick the verdict the access policy dictates."),
     "score": ("Score the grant risk, 0 to 2.",
               "Rate the risk of this grant, 0 to 2.")},
    ({}, {"false": "no grant may be issued", "true": "a grant is permitted"}),
    {"grant": "Issue the grant as requested",
     "grant_expiring": "Issue only a short-lived expiring grant",
     "deny": "Deny the grant request",
     "grant_with_audit": "Grant with elevated audit logging"},
    _levels("low risk", "moderate risk", "high risk"),
    _scope_oracle,
    "contractors never receive the most sensitive scope; sensitive scopes "
    "without an approved ticket are denied; contractors and indefinite or "
    "over-60-day requests get expiring grants; everything else is granted",)


# ---------------------------------------------------------------- cache_warmup

def _warmup_oracle(f):
    if f["miss_spike"] and f["origin_capacity"] < 40:
        choice = "slow_warmup"
    elif f["warm_pct"] >= f["target_pct"]:
        choice = "open_traffic"
    elif f["warm_pct"] >= f["target_pct"] - 20:
        choice = "partial_traffic"
    else:
        choice = "hold_traffic"
    gap = max(0, f["target_pct"] - f["warm_pct"])
    band = 0 if gap == 0 else (1 if gap <= 15 else (2 if gap <= 35 else
          (3 if gap <= 60 else 4)))
    return {"boolean": f["warm_pct"] >= f["target_pct"],
            "choice": choice, "score": band}


_fam(
    "cache_warmup", "cache warmup before traffic",
    ("img-cdn", "api-front", "read-cache", "video-edge"),
    ("WARM", "EDGE"),
    (_field("warm_pct", "share of the hot set already warmed, percent",
            INT(0, 100, 5)),
     _field("target_pct", "warm share required before full traffic, percent",
            INT(50, 95, 5)),
     _field("miss_spike", "a miss spike is in progress", BOOL),
     _field("origin_capacity", "origin spare capacity, percent", INT(0, 100, 5))),
    {"boolean": "ready_for_traffic", "choice": "warmup_step",
     "score": "warmup_gap"},
    {"boolean": ("Is the cache warm enough for full traffic?",
                 "Has warmup reached the stated target?"),
     "choice": ("What is the correct warmup step?",
                "Pick the traffic decision for this cache."),
     "score": ("Score the warmup gap, 0 to 4.",
               "Rate how far below target the cache is, 0 to 4.")},
    ({}, {"false": "warmup is below the target", "true": "target reached"}),
    {"hold_traffic": "Hold traffic; the cache is far below target",
     "partial_traffic": "Admit partial traffic while warming finishes",
     "open_traffic": "Target reached: open full traffic",
     "slow_warmup": "Miss spike with a weak origin: slow the warmup"},
    _levels("at target", "small gap (<=15)", "moderate gap (16-35)",
            "large gap (36-60)", "extreme gap (>60)"),
    _warmup_oracle,
    "a miss spike with less than 40% origin spare slows the warmup; at or above "
    "target opens traffic; within 20 points admits partial traffic; otherwise "
    "hold",)


# ---------------------------------------------------------------- partition_rebalance

def _rebalance_oracle(f):
    if not f["quorum_ok"]:
        choice = "halt"
    elif f["move_running"]:
        choice = "wait_running"
    elif f["skew_pct"] > 60:
        choice = "urgent_rebalance"
    elif f["skew_pct"] > 30:
        choice = "plan_rebalance"
    elif f["hot_partitions"] > 8:
        choice = "split_hot"
    else:
        choice = "no_action"
    urgency = (int(f["skew_pct"] > 60) + int(f["skew_pct"] > 30)
               + int(f["hot_partitions"] > 8) + int(not f["quorum_ok"]))
    return {"boolean": bool(f["skew_pct"] > 30 or f["hot_partitions"] > 8),
            "choice": choice, "score": min(urgency, 3)}


_fam(
    "partition_rebalance", "stream partition assignment",
    ("events-core", "metrics-stream", "click-pipe", "audit-stream"),
    ("PART", "KAFKA"),
    (_field("quorum_ok", "the controller quorum is healthy", BOOL),
     _field("move_running", "a partition move is already running", BOOL),
     _field("skew_pct", "broker load skew, percent", INT(0, 90, 5)),
     _field("hot_partitions", "partitions above the hot threshold", INT(0, 12)),
     _field("total_partitions", "total partition count", INT(8, 64, 8))),
    {"boolean": "rebalance_needed", "choice": "rebalance_action",
     "score": "skew_urgency"},
    {"boolean": ("Does this cluster need a rebalance at all?",
                 "Is a rebalance warranted by skew or hot partitions?"),
     "choice": ("What is the correct rebalance action?",
                "Pick the action the partition policy dictates."),
     "score": ("Score the skew urgency, 0 to 3.",
               "Rate how urgent rebalancing is, 0 to 3.")},
    ({}, {"false": "no rebalance is needed", "true": "a rebalance is needed"}),
    {"no_action": "Cluster is balanced; no action",
     "split_hot": "Split the hottest partitions first",
     "plan_rebalance": "Plan a rebalance inside the next window",
     "urgent_rebalance": "Severe skew: rebalance now",
     "wait_running": "Wait for the in-flight move to finish",
     "halt": "Quorum unhealthy: halt all moves"},
    _levels("balanced", "watch", "urgent", "halted"),
    _rebalance_oracle,
    "an unhealthy quorum halts everything; a running move waits; skew above 60 "
    "rebalances urgently, above 30 plans one; many hot partitions split first; "
    "otherwise no action",)


# ---------------------------------------------------------------- log_pii_redaction

def _pii_oracle(f):
    if f["pii_detected"] and f["jurisdiction"] == 2:
        choice = "redact_now"
    elif f["pii_detected"] and not f["user_consent"]:
        choice = "redact_now"
    elif f["pii_detected"]:
        choice = "flag_for_review"
    elif f["detector_recall"] < 80:
        choice = "upgrade_detector"
    else:
        choice = "compliant"
    sev = (int(f["pii_detected"]) + int(f["jurisdiction"] == 2)
           + int(not f["user_consent"]) + int(f["detector_recall"] < 80))
    return {"boolean": choice == "compliant", "choice": choice,
            "score": min(sev, 3)}


_fam(
    "log_pii_redaction", "log privacy compliance",
    ("app-logs", "edge-logs", "audit-logs", "debug-logs"),
    ("PII", "LOGS"),
    (_field("pii_detected", "the detector found PII in the stream", BOOL),
     _field("jurisdiction", "strictest applicable jurisdiction 0-2 (2 strictest)",
            INT(0, 2)),
     _field("user_consent", "a documented user consent covers this data", BOOL),
     _field("detector_recall", "measured detector recall, percent",
            INT(50, 99))),
    {"boolean": "compliant_as_is", "choice": "pii_action",
     "score": "privacy_severity"},
    {"boolean": ("Is the log stream compliant with no further action?",
                 "May the pipeline continue unchanged?"),
     "choice": ("What is the correct PII handling action?",
                "Pick the action the privacy policy dictates."),
     "score": ("Score the privacy severity, 0 to 3.",
               "Rate how severe the privacy posture is, 0 to 3.")},
    ({}, {"false": "an action is required", "true": "compliant as-is"}),
    {"compliant": "No PII and a healthy detector: compliant",
     "upgrade_detector": "Detector recall too low: upgrade before trusting it",
     "flag_for_review": "PII with consent: flag for manual review",
     "redact_now": "Redact the stream immediately",
     "suspend_pipeline": "Suspend the pipeline until remediation lands"},
    _levels("clean", "watch", "serious", "critical"),
    _pii_oracle,
    "PII under the strictest jurisdiction or without consent redacts "
    "immediately; consented PII flags for review; a weak detector must be "
    "upgraded; otherwise the stream is compliant",)


# ---------------------------------------------------------------- gpu_batch_pack
# 8-candidate binding stratum.

def _gpu_pack_oracle(f):
    need = f["slots_needed"]
    free = [f[f"bin_{k}_free"] for k in range(8)]
    fitting = [k for k in range(8) if free[k] >= need]
    if fitting:
        pick = min(fitting, key=lambda k: (free[k] - need, k))
    else:
        pick = max(range(8), key=lambda k: (free[k], -k))
    total_free = sum(free)
    scarcity = 0 if total_free >= 24 else (1 if total_free >= 12 else
               (2 if total_free >= 4 else 3))
    return {"boolean": free[0] >= need, "choice": f"bin_{pick}",
            "score": scarcity}


_fam(
    "gpu_batch_pack", "gpu bin packing",
    ("train-cluster", "infer-pool", "render-gpus", "lab-rig"),
    ("GPU", "PACK"),
    tuple([_field(f"bin_{k}_free", f"free slots in gpu bin {k}", INT(0, 4))
           for k in range(8)]
          + [_field("slots_needed", "slots the job requires in one bin",
                    INT(1, 4)),
             _field("urgent", "the job is marked urgent", BOOL)]),
    {"boolean": "bin_zero_fits", "choice": "target_bin",
     "score": "pool_scarcity"},
    {"boolean": ("Does bin 0 have enough free slots for this job?",
                 "Can bin 0 take the job as-is?"),
     "choice": ("Which bin should take this job?",
                "Pick the bin the packing rule selects."),
     "score": ("Score pool scarcity, 0 to 3 (free slots 24+, 12-23, 4-11, <4).",
               "Rate how scarce free capacity is, 0 to 3.")},
    ({}, {"false": "bin 0 lacks the free slots", "true": "bin 0 fits the job"}),
    {f"bin_{k}": f"Pack the job into bin {k}" for k in range(8)},
    _levels("plenty of room", "tightening", "scarce", "almost full"),
    _gpu_pack_oracle,
    "pack into the tightest bin that still fits the job (ties to the lower bin); "
    "if no bin fits, take the bin with the most free slots as the documented "
    "fallback; the score bands total free slots",)


# ---------------------------------------------------------------- cron_storm

def _cron_oracle(f):
    if f["misfires"] > 10:
        choice = "pause_scheduler"
    elif f["window_overlap"] and f["top_offender_share"] > 50:
        choice = "stagger_top_offender"
    elif f["jobs_per_min"] > 80 and not f["jittered"]:
        choice = "add_jitter"
    elif f["jobs_per_min"] > 80:
        choice = "shard_schedule"
    elif f["window_overlap"]:
        choice = "shift_windows"
    else:
        choice = "healthy"
    band = 2 if (f["misfires"] > 10 or f["jobs_per_min"] > 80) else (
        1 if (f["window_overlap"] or f["jobs_per_min"] > 40) else 0)
    return {"boolean": f["jobs_per_min"] > 80 or f["misfires"] > 10,
            "choice": choice, "score": band}


_fam(
    "cron_storm", "scheduled job load",
    ("nightly-batch", "report-crons", "sync-crons", "cleanup-crons"),
    ("CRON", "SCHED"),
    (_field("jobs_per_min", "jobs firing per minute", INT(0, 120)),
     _field("misfires", "misfired jobs in the last hour", INT(0, 20)),
     _field("top_offender_share", "share of load from the busiest schedule, "
            "percent", INT(0, 100, 5)),
     _field("window_overlap", "heavy schedules overlap in one window", BOOL),
     _field("jittered", "schedules already carry start jitter", BOOL)),
    {"boolean": "storm_in_progress", "choice": "storm_action",
     "score": "storm_band"},
    {"boolean": ("Is a cron storm in progress?",
                 "Is the scheduler in a storm condition?"),
     "choice": ("What is the correct scheduler action?",
                "Pick the action the schedule policy dictates."),
     "score": ("Score the storm band, 0 to 2.",
               "Rate the schedule storm, 0 to 2.")},
    ({}, {"false": "no storm", "true": "a storm is in progress"}),
    {"healthy": "Schedule load is nominal",
     "shift_windows": "Separate the overlapping windows",
     "shard_schedule": "Shard the schedule across workers",
     "add_jitter": "Add start jitter to spread the load",
     "stagger_top_offender": "Stagger the dominant schedule first",
     "pause_scheduler": "Misfires are out of control: pause the scheduler"},
    _levels("calm", "busy", "storm"),
    _cron_oracle,
    "more than ten misfires pauses the scheduler; overlap dominated by one "
    "schedule staggers it; load above 80/min adds jitter or shards when jitter "
    "is already on; mere overlap shifts windows; otherwise healthy",)


# ---------------------------------------------------------------- artifact_promotion

def _artifact_oracle(f):
    if not f["scan_clean"] or not f["signed"]:
        choice = "hold_artifact"
    elif f["regressions"] > 0:
        choice = "investigate_regressions"
    elif f["soak_hours"] < f["required_soak"]:
        choice = "continue_soak"
    elif f["stage"] < 2:
        choice = "promote_next"
    else:
        choice = "publish"
    blockers = (int(not f["scan_clean"]) + int(not f["signed"])
                + int(f["regressions"] > 0)
                + int(f["soak_hours"] < f["required_soak"]))
    return {"boolean": bool(f["scan_clean"] and f["signed"]
                            and f["regressions"] == 0
                            and f["soak_hours"] >= f["required_soak"]),
            "choice": choice, "score": min(blockers, 4)}


_fam(
    "artifact_promotion", "release artifact pipeline",
    ("release-train", "nightly-builds", "sdk-artifacts", "driver-builds"),
    ("ART", "PROMO"),
    (_field("stage", "current stage 0-2 (0 dev, 1 staging, 2 prod-candidate)",
            INT(0, 2)),
     _field("scan_clean", "the vulnerability scan came back clean", BOOL),
     _field("signed", "the artifact is signed", BOOL),
     _field("soak_hours", "hours the artifact has soaked", INT(0, 72)),
     _field("required_soak", "required soak hours", INT(8, 48, 8)),
     _field("regressions", "open regression reports", INT(0, 10))),
    {"boolean": "promotable_now", "choice": "promotion_step",
     "score": "blocker_count"},
    {"boolean": ("Is the artifact promotable right now?",
                 "Has every promotion gate cleared?"),
     "choice": ("What is the correct promotion step?",
                "Pick the step the promotion policy dictates."),
     "score": ("Score the number of open blockers, capped at 4, 0 to 4.",
               "Rate how many promotion blockers remain, 0 to 4.")},
    ({}, {"false": "a promotion gate is still open",
          "true": "all gates cleared; promotable"}),
    {"hold_artifact": "Hold: unsigned or scan-dirty artifacts never promote",
     "investigate_regressions": "Investigate the open regressions first",
     "continue_soak": "Keep soaking until the required hours elapse",
     "promote_next": "Promote to the next stage",
     "publish": "All gates cleared at the final stage: publish"},
    _levels("no blockers", "one blocker", "two blockers", "three blockers",
            "four blockers"),
    _artifact_oracle,
    "unsigned or dirty artifacts are held; regressions are investigated; a "
    "short soak continues; mid-stage artifacts promote; a fully cleared "
    "prod-candidate publishes",)


# ---------------------------------------------------------------- edge_route_shift
# 12-candidate binding stratum.

def _edge_oracle(f):
    loads = [f[f"pop_{k}_load"] for k in range(12)]
    pick = min(range(12), key=lambda k: (loads[k], k))
    saturated = sum(1 for load in loads if load >= 50)
    band = 0 if saturated <= 3 else (1 if saturated <= 8 else 2)
    return {"boolean": pick == 0, "choice": f"pop_{pick:02d}",
            "score": band}


_fam(
    "edge_route_shift", "edge pop traffic shifting",
    ("edge-mesh-a", "edge-mesh-b", "cdn-front", "anycast-net"),
    ("POP", "EDGE"),
    tuple([_field(f"pop_{k}_load", f"point-of-presence {k} load, percent",
                  INT(0, 100, 5)) for k in range(12)]
          + [_field("shift_pct", "share of traffic to move, percent",
                    INT(5, 50, 5))]),
    {"boolean": "pop_zero_lightest", "choice": "shift_target",
     "score": "best_load_band"},
    {"boolean": ("Is pop 00 currently the lightest?",
                 "Does pop 00 carry the lowest load?"),
     "choice": ("Which pop should receive the shifted traffic?",
                "Pick the pop the routing rule selects."),
     "score": ("Score mesh saturation, 0 to 2 (<=3 pops at 50%+, 4-8, 9-12).",
               "Rate how saturated the edge mesh is, 0 to 2.")},
    ({}, {"false": "another pop is lighter", "true": "pop 00 is lightest"}),
    {f"pop_{k:02d}": f"Shift traffic to pop {k:02d}" for k in range(12)},
    _levels("at most 3 pops at 50% load or more", "4-8 pops saturated",
            "9-12 pops saturated"),
    _edge_oracle,
    "shift to the lowest-load pop (ties to the lower number); the boolean asks "
    "whether pop 00 is that minimum and the score counts pops at 50% load or "
    "more, banded 0-3 / 4-8 / 9-12",)


# ---------------------------------------------------------------- snapshot_prune

def _snapshot_oracle(f):
    if f["snap_count"] <= f["cap"]:
        choice = "no_prune"
    elif f["protected"] >= f["snap_count"] - f["cap"]:
        choice = "raise_cap"
    elif f["policy_strict"]:
        choice = "prune_oldest"
    else:
        choice = "compress_then_prune"
    over = max(0, f["snap_count"] - f["cap"])
    band = 0 if over == 0 else (1 if over <= 20 else (2 if over <= 60 else 3))
    return {"boolean": f["snap_count"] > f["cap"], "choice": choice,
            "score": band}


_fam(
    "snapshot_prune", "snapshot retention housekeeping",
    ("vm-fleet", "db-clones", "fs-snaps", "lab-images"),
    ("SNAP", "PRUNE"),
    (_field("snap_count", "snapshots currently stored", INT(0, 200)),
     _field("cap", "the configured snapshot cap", INT(20, 120, 10)),
     _field("protected", "snapshots pinned by policy", INT(0, 10)),
     _field("oldest_age", "age of the oldest snapshot in days", INT(1, 90)),
     _field("policy_strict", "the strict pruning policy applies", BOOL)),
    {"boolean": "over_cap", "choice": "prune_action",
     "score": "overflow_band"},
    {"boolean": ("Is the snapshot store over its cap?",
                 "Does the count exceed the configured cap?"),
     "choice": ("What is the correct pruning action?",
                "Pick the action the retention policy dictates."),
     "score": ("Score the overflow band, 0 to 3 (none, <=20, <=60, >60).",
               "Rate how far over cap the store is, 0 to 3.")},
    ({}, {"false": "within cap", "true": "over the cap"}),
    {"no_prune": "Within cap: no pruning",
     "compress_then_prune": "Compress old snapshots, then prune",
     "prune_oldest": "Strict policy: prune the oldest first",
     "raise_cap": "Too many pinned snapshots: raise the cap instead"},
    _levels("within cap", "mild overflow", "large overflow", "severe overflow"),
    _snapshot_oracle,
    "within cap does nothing; when pinning alone would cover the overflow the "
    "cap is raised; strict policy prunes the oldest; otherwise compress then "
    "prune",)


FAMILIES = tuple(spec["family"] for spec in FAMILY_DEFS)
FAMILY_MAP = {spec["family"]: spec for spec in FAMILY_DEFS}
assert len(FAMILIES) == len(set(FAMILIES)) == 13


# --------------------------------------------------------------------------------------
# heldout-specific surface rendering (deliberately unlike the corpus templates)
# --------------------------------------------------------------------------------------

def _fmt(value, template):
    if isinstance(value, bool):
        if template == 3:
            return "TRUE" if value else "FALSE"
        if template == 2:
            return "yes" if value else "no"
        return "true" if value else "false"
    return str(value)


def render_state(facts, spec, render):
    order = render["field_order"]
    labels = {f["key"]: f["label"] for f in spec["fields"]}
    template = render["template"]
    entity, ref = render["entity"], render["ref"]
    if template == 0:      # prose sentence chain
        sentences = [f"Situation report {ref} for {entity}."]
        for key in order:
            sentences.append(f"The record shows {labels[key]} = "
                             f"{_fmt(facts[key], template)}.")
        return " ".join(sentences)
    if template == 1:      # yaml-ish record
        lines = [f"record:", f"  ref: {ref}", f"  entity: {entity}",
                 f"  kind: {spec['family'].replace('_', '-')}"]
        for key in order:
            lines.append(f"  {key}: {_fmt(facts[key], template)}")
        return "\n".join(lines)
    if template == 2:      # csv-style field dump
        lines = [f"dump {ref} ({entity})", "field,value"]
        for key in order:
            lines.append(f"{key},{_fmt(facts[key], template)}")
        return "\n".join(lines)
    if template == 3:      # bracketed log excerpt
        lines = [f"[{ref}] {entity} {spec['family'].replace('_', '-')} snapshot"]
        for key in order:
            lines.append(f"[{ref}] {key}={_fmt(facts[key], template)}")
        return "\n".join(lines)
    raise ValueError(f"unknown template {template!r}")


def make_render(spec, seed, ordinal):
    rng = random.Random(
        f"nanojev-engineering-heldout-v2:{seed}:{spec['family']}:{ordinal}")
    order = [f["key"] for f in spec["fields"]]
    rng.shuffle(order)
    return {"template": rng.randrange(4),
            "entity": spec["entities"][rng.randrange(len(spec["entities"]))],
            "ref": f"{spec['refs'][rng.randrange(len(spec['refs']))]}"
                   f"-{rng.randrange(100, 99999)}",
            "field_order": order,
            "variant": rng.randrange(64)}


# --------------------------------------------------------------------------------------
# deterministic fact pools, state selection, rows
# --------------------------------------------------------------------------------------

def _oracle_checked(spec, facts):
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


def fact_pool(spec, seed):
    keys = [f["key"] for f in spec["fields"]]
    values = [domain_values(f["domain"]) for f in spec["fields"]]
    total = 1
    for v in values:
        total *= len(v)
    rng = random.Random(f"nanojev-heldout-v2-pool:{seed}:{spec['family']}")
    if total <= ENUMERATE_CAP:
        combos = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
        rng.shuffle(combos)
    else:
        combos, seen = [], set()
        attempts = 0
        while len(combos) < SAMPLE_CAP and attempts < SAMPLE_CAP * 4:
            attempts += 1
            facts = {key: rng.choice(domain) for key, domain in zip(keys, values)}
            tag = json.dumps(facts, sort_keys=True)
            if tag in seen:
                continue
            seen.add(tag)
            combos.append(facts)
    return [{"facts": facts, "gold": _oracle_checked(spec, facts)}
            for facts in combos]


def select_states(spec, seed, count=STATES_PER_FAMILY):
    """Seed guaranteed label coverage, then spread across uncovered choice labels."""
    pool = fact_pool(spec, seed)
    min_choice = min(3, len(spec["choice_criteria"]))
    min_score = min(3, len(spec["score_criteria"]))
    selected, picked = [], set()

    def take(pred):
        for cand in pool:
            if id(cand) in picked or not pred(cand["gold"]):
                continue
            picked.add(id(cand))
            selected.append(cand)
            return True
        return False

    # Phase A: coverage seeding — both boolean values, min_score levels,
    # min_choice choice labels (a single candidate may cover several).
    for value in (False, True):
        take(lambda g, v=value: g["boolean"] is v)
    seen_scores = {c["gold"]["score"] for c in selected}
    for level in range(len(spec["score_criteria"])):
        if len(seen_scores) >= min_score:
            break
        if level not in seen_scores and take(
                lambda g, lv=level: g["score"] == lv):
            seen_scores.add(level)
    seen_choices = {c["gold"]["choice"] for c in selected}
    for label in sorted(spec["choice_criteria"]):
        if len(seen_choices) >= min_choice:
            break
        if label not in seen_choices and take(
                lambda g, k=label: g["choice"] == k):
            seen_choices.add(label)

    # Phase B: fill remaining slots preferring still-uncovered choice labels.
    by_choice = defaultdict(list)
    for cand in pool:
        if id(cand) not in picked:
            by_choice[cand["gold"]["choice"]].append(cand)
    while len(selected) < count:
        progressed = False
        for label in sorted(by_choice):
            if label in seen_choices:
                continue
            cand = by_choice[label].pop(0)
            picked.add(id(cand))
            selected.append(cand)
            seen_choices.add(label)
            progressed = True
            if len(selected) >= count:
                break
        if not progressed:
            break
    for cand in pool:
        if len(selected) >= count:
            break
        if id(cand) not in picked:
            picked.add(id(cand))
            selected.append(cand)
    if len(selected) < count:
        raise ValueError(f"{spec['family']}: only {len(selected)} states available")
    golds = [c["gold"] for c in selected]
    distinct_choice = {g["choice"] for g in golds}
    distinct_score = {g["score"] for g in golds}
    bools = {g["boolean"] for g in golds}
    min_choice = min(3, len(spec["choice_criteria"]))
    min_score = min(3, len(spec["score_criteria"]))
    if len(distinct_choice) < min_choice:
        raise ValueError(f"{spec['family']}: choice coverage {distinct_choice}")
    if len(distinct_score) < min_score:
        raise ValueError(f"{spec['family']}: score coverage {distinct_score}")
    if bools != {False, True}:
        raise ValueError(f"{spec['family']}: boolean coverage {bools}")
    return selected


def candidate_ids(question):
    if question["type"] == "boolean":
        return ["false", "true"]
    if question["type"] == "choice":
        return list(question["criteria"])
    return [str(i) for i in range(len(question["criteria"]))]


def one_hot(ids, gold):
    if isinstance(gold, bool):
        index = ids.index("true" if gold else "false")
    elif isinstance(gold, int):
        index = gold
    else:
        index = ids.index(gold)
    return {key: (1.0 if i == index else 0.0) for i, key in enumerate(ids)}


def question_body(spec, qtype, variant):
    instructions = spec["instructions"][qtype][variant % len(spec["instructions"][qtype])]
    if qtype == "boolean":
        body = {"type": "boolean", "instructions": instructions}
        criteria = spec["boolean_criteria"][variant % len(spec["boolean_criteria"])]
        if criteria:
            body["criteria"] = dict(criteria)
        return body
    if qtype == "choice":
        return {"type": "choice", "instructions": instructions,
                "criteria": dict(spec["choice_criteria"])}
    return {"type": "score", "instructions": instructions,
            "criteria": list(spec["score_criteria"])}


def build_rows(seed=DEFAULT_SEED):
    used_states = set()
    rows = []
    for spec in FAMILY_DEFS:
        selected = select_states(spec, seed)
        for ordinal, cand in enumerate(selected):
            facts, gold = cand["facts"], cand["gold"]
            render, state_text = None, None
            for attempt in range(8):
                trial = make_render(spec, seed, ordinal * 8 + attempt)
                text = render_state(facts, spec, trial)
                if text not in used_states:
                    render, state_text = trial, text
                    break
            if render is None:
                raise ValueError(f"{spec['family']}: no unique render for state "
                                 f"{ordinal}")
            used_states.add(state_text)
            slug = f"{spec['family']}-s{ordinal:02d}"
            questions = {qt: question_body(spec, qt, render["variant"])
                         for qt in QUESTION_TYPES}
            qids = spec["qids"]
            for qtype in QUESTION_TYPES:
                qid = qids[qtype]
                question = questions[qtype]
                ids = candidate_ids(question)
                g = gold[qtype]
                probs = one_hot(ids, g)
                row = {
                    "id": f"eh2-{slug}-{qtype}",
                    "state_id": f"eh2-{slug}",
                    "family_id": f"eh2_{spec['family']}",
                    "split": "test",
                    "state": state_text,
                    "questions": {qid: question},
                    "gold": {qid: g},
                    "gold_probs": {qid: probs},
                    "gold_probs_kind": {qid: "deterministic_truth"},
                    "gold_label_kind": {qid: "deterministic_truth"},
                    "schema_version": ROW_SCHEMA,
                    "metadata": {
                        "source_group_id": f"eh2-{slug}",
                        "question_type": qtype,
                        "domain": spec["domain_desc"],
                        "fact_basis": {k: facts[k] for k in sorted(facts)},
                        "authoring": "programmatic_independent_heldout_v2",
                        "provenance_source_id":
                            "scripts/build_engineering_heldout_v2.py",
                        "derived_from_evaluation_corpus": False,
                        "training_authorized_by_this_corpus": False,
                    },
                    "provenance": {
                        "cohort": "engineering_heldout_v2",
                        "authoring": ("programmatic_frozen_oracle_states_for_"
                                      "independent_heldout"),
                        "domain": spec["domain_desc"],
                        "rule": spec["why"],
                        "fact_basis": {k: facts[k] for k in sorted(facts)},
                        "independence_statement": (
                            "Authored independently of every engineering "
                            "judgment corpus version and of "
                            "engineering_heldout_v1: no corpus or heldout rule, "
                            "family, field set, surface template, instruction "
                            "string or candidate vocabulary was reused; not a "
                            "reseed or paraphrase of any existing item."),
                        "derived_from_evaluation_corpus": False,
                        "human_reviewed": False,
                        "license": "CC0-1.0",
                        "training_authorized": False,
                    },
                }
                rows.append(row)
    return rows


def validate_rows(rows):
    """Every row must pass the real validators and oracle-checked gold."""
    errors = []
    if len(rows) < 300:
        errors.append(f"heldout v2 requires >=300 items, have {len(rows)}")
    groups = {r["metadata"]["source_group_id"] for r in rows}
    if len(groups) < 50:
        errors.append(f"heldout v2 requires >=50 source groups, have {len(groups)}")
    ids, states = set(), set()
    for row in rows:
        try:
            targets = validate_training_row(row)
            validate_request({"states": [{k: row[k]
                                          for k in ("id", "state", "questions")}]})
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{row.get('id')}: validation failed: {exc}")
            continue
        for qid, q in row["questions"].items():
            t = targets[qid]
            if t["gold_index"] is None or t["gold_distribution_probs"] is None:
                errors.append(f"{row['id']}:{qid}: missing gold")
            elif t["gold_distribution_probs"][t["gold_index"]] != 1.0:
                errors.append(f"{row['id']}:{qid}: gold_probs argmax != gold")
        family = row["family_id"].removeprefix("eh2_")
        if family not in FAMILY_MAP:
            errors.append(f"{row['id']}: unknown family {family!r}")
            continue
        spec = FAMILY_MAP[family]
        qtype = row["metadata"]["question_type"]
        recomputed = _oracle_checked(spec, row["metadata"]["fact_basis"])[qtype]
        if recomputed != row["gold"][qid]:
            errors.append(f"{row['id']}: gold does not match the family oracle")
        if row["id"] in ids:
            errors.append(f"duplicate row id {row['id']}")
        ids.add(row["id"])
        if row["split"] != "test":
            errors.append(f"{row['id']}: heldout rows must carry split=test")
        states.add(row["state_id"])
        hits = v1.reserved_hits(json.dumps(row, ensure_ascii=False, sort_keys=True))
        if hits:
            errors.append(f"{row['id']}: reserved evaluation tokens {hits}")
        if v1.forbidden_path_reason(row["metadata"]["provenance_source_id"]):
            errors.append(f"{row['id']}: refused evaluation source")
    if len(states) != len(groups):
        errors.append("each source group must contain exactly one state cluster")
    strata = sorted({len(q["criteria"])
                     for r in rows for q in r["questions"].values()
                     if q["type"] in ("choice", "score")})
    if len(strata) < 4:
        errors.append(f"option-count strata too thin: {strata}")
    return errors


def _jsonl(rows):
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                   for r in rows)


def manifest_for(rows, seed):
    per_family = {}
    for row in rows:
        fam = row["family_id"]
        per_family.setdefault(fam, {"rows": 0, "states": set(),
                                    "choice_options": None,
                                    "score_levels": None})
        per_family[fam]["rows"] += 1
        per_family[fam]["states"].add(row["state_id"])
        q = next(iter(row["questions"].values()))
        if q["type"] == "choice":
            per_family[fam]["choice_options"] = len(q["criteria"])
        if q["type"] == "score":
            per_family[fam]["score_levels"] = len(q["criteria"])
    for fam in per_family:
        per_family[fam]["states"] = len(per_family[fam]["states"])
    text = _jsonl(rows)
    return {
        "schema_version": MANIFEST_SCHEMA,
        "cohort": "engineering_heldout_v2",
        "created_by": "scripts/build_engineering_heldout_v2.py",
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": seed,
        "role": "independent_evaluation_heldout_only",
        "row_schema": ROW_SCHEMA,
        "row_count": len(rows),
        "state_count": len({r["state_id"] for r in rows}),
        "source_group_count": len({r["metadata"]["source_group_id"] for r in rows}),
        "families": per_family,
        "question_types": list(QUESTION_TYPES),
        "split": "test",
        "option_count_strata": sorted({len(q["criteria"])
                                       for r in rows for q in r["questions"].values()
                                       if q["type"] in ("choice", "score")}),
        "items_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "independence": {
            "corpora_excluded": ["engineering_judgment_corpus_v1",
                                 "engineering_judgment_corpus_v2",
                                 "engineering_judgment_corpus_v3",
                                 "engineering_judgment_corpus_v4",
                                 "engineering_heldout_v1"],
            "external_model_outputs": "none; golds are computed by frozen local oracles",
            "surface_templates": "4 heldout-specific formats unused by any corpus",
        },
        "provenance": dict(v1.PROVENANCE),
        "training_authorized": False,
        "training_performed": False,
        "network_model_calls": 0,
    }


def self_test(seed=DEFAULT_SEED):
    try:
        rows = build_rows(seed)
        errors = validate_rows(rows)
        manifest = manifest_for(rows, seed)
    except Exception as error:  # noqa: BLE001
        rows, manifest, errors = None, None, [f"{type(error).__name__}: {error}"]
    return {"schema_version": SCHEMA_VERSION, "mode": "self-test",
            "status": "ok" if not errors else "failed",
            "items": len(rows) if rows else 0,
            "questions": sum(len(r["questions"]) for r in rows) if rows else 0,
            "source_groups": (manifest["source_group_count"] if manifest else 0),
            "families": sorted(FAMILIES),
            "errors": errors, "training_performed": False, "wrote_output": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if args.output_dir is not None:
            parser.error("--self-test is mutually exclusive with --output-dir")
        report = self_test(args.seed)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] == "ok" else 1
    if args.output_dir is None:
        parser.error("one of --self-test or --output-dir is required")
    rows = build_rows(args.seed)
    errors = validate_rows(rows)
    if errors:
        print(json.dumps({"status": "failed", "errors": errors}, indent=2))
        return 2
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        print(json.dumps({"status": "refused",
                          "reason": "output directory must be empty"}, indent=2))
        return 2
    out.mkdir(parents=True, exist_ok=True)
    items_path = out / "items.jsonl"
    manifest = manifest_for(rows, args.seed)
    # Write-once: refuse to overwrite an existing heldout file.
    with items_path.open("x", encoding="utf-8") as stream:
        stream.write(_jsonl(rows))
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(items_path),
                      "items": len(rows),
                      "source_groups": manifest["source_group_count"],
                      "sha256": manifest["items_sha256"],
                      "training_performed": False}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
