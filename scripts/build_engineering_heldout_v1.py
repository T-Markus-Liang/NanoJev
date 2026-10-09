#!/usr/bin/env python3
"""Engineering-judgment heldout V1: independently authored evaluation cohort.

This builder emits ``research/engineering_heldout_v1/items.jsonl`` — a fresh
held-out cohort for the T9d domain-adaptation review (W35-C condition 2:
"heldout/OOD must be independently sourced; a reseeded corpus is not
independent").

Independence properties, by construction:

* Every scenario is hand-authored for this file.  None of the 9 frozen corpus-v2
  rules, state templates, family names, field sets, instruction strings or
  candidate vocabularies is reused.  State texts use deliberately different
  surface formats (incident timelines, ticket excerpts, metric tables, config
  diffs, chat-style notes) rather than the corpus's ``- label: value`` fact
  lists.
* ``source_group_id`` values (``eh1-*``) live in a disjoint namespace from the
  corpus's ``ejc-v2-*`` groups; ``state_id``/``family_id``/row ``id`` are
  likewise disjoint by name *and* by content.
* Gold is deterministic per item under the rubric stated in each question's
  instructions/criteria; ``gold_probs`` is the matching one-hot vector with
  kind ``deterministic_truth`` (train_pipeline_decisions contract).
* Each row carries a ``provenance`` block declaring authorship, the scenario's
  domain and an explicit independence statement.

Rows use the ``nanojev-engineering-judgment-trainer-row-v1`` schema (one
question per row, like the corpus ``trainer_view``) so the cohort loads through
``train_pipeline_decisions.read_training_records`` and predicts through
``predict_toy_decisions.validate_request`` unchanged.  All rows carry
``split="test"`` — this cohort is evaluation-only and must never be merged into
train/dev/calibration.

This file authors data only.  It performs no training and no measurement, and
authorizes nothing.

Usage

    .venv/bin/python scripts/build_engineering_heldout_v1.py \
        --output research/engineering_heldout_v1/items.jsonl
    .venv/bin/python scripts/build_engineering_heldout_v1.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_toy_decisions import validate_request  # noqa: E402
from train_pipeline_decisions import validate_training_row  # noqa: E402

SCHEMA_VERSION = "nanojev-engineering-heldout-v1"
ROW_SCHEMA = "nanojev-engineering-judgment-trainer-row-v1"
QUESTION_TYPES = ("boolean", "choice", "score")

# Shared 4-level urgency rubric used by the score questions.  Criteria lists are
# per-scenario (the model-visible text differs), but the ordinal semantics are
# consistent: 0 = no action, 1 = monitor/schedule, 2 = act soon, 3 = act now.
URGENCY_LEVELS = ("no action needed", "monitor or schedule routinely",
                  "act soon, within hours", "act immediately")


def scen(slug, family, domain, state, boolean, choice, score):
    """Compact scenario declaration; each question is (qid, instructions, criteria, gold)."""
    return {"slug": slug, "family": family, "domain": domain, "state": state,
            "boolean": boolean, "choice": choice, "score": score}


SCENARIOS = [
    scen(
        "pg-replica-promotion", "incident_triage", "database operations",
        "INC-4417 timeline (UTC):\n"
        "14:02 primary pg-01 reports disk latency 800ms p99\n"
        "14:05 orchestrator flags pg-01 unhealthy, proposes promoting pg-02\n"
        "14:05 replica pg-02 replay lag measured at 41 seconds\n"
        "14:06 writes still accepted on pg-01 at reduced rate\n"
        "Runbook DB-FAILOVER-3: promote only when replay lag is under 10 seconds "
        "or primary is unreachable. pg-01 remains reachable.",
        ("promote_replica_now",
         "Does the runbook permit promoting pg-02 to primary at this point in the timeline?",
         {"false": "promotion is not permitted under DB-FAILOVER-3 given the stated lag",
          "true": "promotion is permitted under DB-FAILOVER-3 right now"},
         False),
        ("best_next_action",
         "Which next action best follows DB-FAILOVER-3?",
         {"promote_now": "promote pg-02 immediately despite the lag",
          "wait_for_lag": "keep pg-01 writable, recheck lag, promote once under 10 s",
          "rollback_deploy": "roll back the most recent application deploy",
          "shutdown_writes": "stop all writes and freeze the cluster"},
         "wait_for_lag"),
        ("response_urgency",
         "Rate the urgency of acting on the failover decision.",
         ["no action needed", "monitor or schedule routinely",
          "act soon, within hours", "act immediately"],
         3),
    ),
    scen(
        "django-major-upgrade", "dependency_upgrade", "web framework maintenance",
        "Ticket PLAT-882: bump Django 4.2 LTS -> 5.1 in the billing service.\n"
        "Notes from the spike:\n"
        "* 3 direct uses of APIs removed in 5.0 found by codemod\n"
        "* postgres-backed sessions: no change required\n"
        "* staging suite green on 5.1rc; canary stage not yet run\n"
        "* billing freeze for quarter-end starts in 6 days\n"
        "Policy: no framework majors land inside a revenue freeze window.",
        ("land_inside_freeze",
         "May the Django 5.1 upgrade land inside the quarter-end freeze window?",
         None,
         False),
        ("upgrade_path",
         "What is the correct path for this upgrade?",
         {"land_now": "merge to main and deploy before the freeze",
          "land_after_freeze": "finish canary verification, then land once the freeze lifts",
          "pin_42_forever": "stay on 4.2 permanently",
          "rewrite_billing": "rewrite the billing service to avoid the upgrade"},
         "land_after_freeze"),
        ("upgrade_risk",
         "Rate the risk of shipping this upgrade inside the freeze window.",
         ["negligible risk", "low risk, acceptable anytime",
          "moderate risk, needs canary first", "high risk, blocked by policy"],
         3),
    ),
    scen(
        "orders-column-drop", "schema_migration", "database schema change",
        "Migration plan M-209 (orders service):\n"
        "  step 1: deploy code that stops reading orders.legacy_total\n"
        "  step 2: wait one full release cycle (7 days)\n"
        "  step 3: ALTER TABLE orders DROP COLUMN legacy_total\n"
        "Current state: step 1 deployed yesterday; step 2 has 6 days remaining.\n"
        "A reviewer proposes running step 3 today 'to save time'.",
        ("drop_column_today",
         "Is dropping orders.legacy_total today consistent with migration plan M-209?",
         {"false": "step 2's full release cycle has not elapsed",
          "true": "the column may be dropped today"},
         False),
        ("migration_safety",
         "Which property does the 7-day wait in step 2 primarily protect?",
         {"rollback_window": "a rollback to the previous code version still reads the column",
          "disk_space": "the table must shrink before the drop",
          "replication": "replicas need time to copy the new schema",
          "audit_compliance": "regulators require a 7-day notice period"},
         "rollback_window"),
        ("proposal_risk",
         "Rate the risk of running step 3 today as the reviewer proposes.",
         ["safe", "minor risk only", "risky but recoverable",
          "unsafe: defeats the rollback window"],
         3),
    ),
    scen(
        "v1-charges-endpoint", "api_deprecation", "public API lifecycle",
        "Deprecation record DEP-31 for GET /v1/charges:\n"
        "  announced: 2026-03-01, sunset date: 2026-09-01\n"
        "  today: 2026-08-20 (12 days to sunset)\n"
        "  last-7-day traffic: 41,200 calls from 17 distinct API keys\n"
        "  migration guide sent to all key owners on 2026-08-01\n"
        "  policy: endpoints may be hard-disabled only after sunset date AND "
        "after traffic drops below 1,000 calls/week.",
        ("disable_now",
         "May GET /v1/charges be hard-disabled today under the stated policy?",
         {"false": "sunset date has not passed and traffic is above the threshold",
          "true": "the endpoint may be hard-disabled today"},
         False),
        ("correct_step",
         "What is the correct next step for DEP-31?",
         {"disable_endpoint": "return 410 Gone for all callers now",
          "nudge_and_wait": "remind remaining key owners, re-evaluate after sunset and below threshold",
          "extend_sunset": "push the sunset date out by one quarter",
          "silence_alerts": "turn off the deprecation traffic monitor"},
         "nudge_and_wait"),
        ("remaining_work",
         "Rate how close this deprecation is to completion.",
         ["complete, no work left", "nearly done, routine follow-up",
          "significant work remains", "not started"],
         1),
    ),
    scen(
        "checkout-pool-saturation", "capacity_planning", "service capacity",
        "Capacity board, checkout-api (30-day lookback):\n"
        "| week | p95 CPU | p95 conns | errors |\n"
        "| W30  | 41%     | 38%       | 0      |\n"
        "| W31  | 55%     | 52%       | 0      |\n"
        "| W32  | 71%     | 69%       | 2      |\n"
        "| W33  | 84%     | 81%       | 19     |\n"
        "Forecast: same growth rate continues; autoscale ceiling is 90% CPU.",
        ("breach_next_week",
         "Does the trend imply checkout-api will hit its autoscale ceiling within the next week?",
         {"false": "the trend stays below the ceiling next week",
          "true": "the trend crosses the 90% ceiling next week"},
         True),
        ("capacity_response",
         "What is the proportionate capacity response?",
         {"ignore": "no action; weekly numbers are noisy",
          "raise_ceiling": "raise the autoscale ceiling and add headroom before next week",
          "halve_traffic": "reject half of checkout requests",
          "rewrite_service": "rewrite checkout-api for efficiency this week"},
         "raise_ceiling"),
        ("planning_urgency",
         "Rate the urgency of the capacity decision.",
         list(URGENCY_LEVELS),
         2),
    ),
    scen(
        "openssl-cve-3412", "security_patch", "vulnerability response",
        "Security bulletin SEC-2026-091:\n"
        "  CVE-2026-3412 (openssl): CVSS 9.8, remote code execution, public exploit observed\n"
        "  exposure scan of our fleet:\n"
        "    edge-proxy-01..04: openssl 3.0.13 (vulnerable), internet-facing\n"
        "    internal batch workers: openssl 3.0.13 (vulnerable), no inbound network\n"
        "  fixed version: 3.0.15, available in our mirror\n"
        "  change policy: emergency patches to internet-facing hosts bypass the weekly window.",
        ("emergency_patch_edges",
         "Does policy permit emergency-patching the edge proxies outside the weekly window?",
         {"false": "the weekly change window must still be honoured",
          "true": "emergency patching is permitted for these hosts"},
         True),
        ("patch_scope",
         "What is the correct patching scope order?",
         {"edges_first": "patch internet-facing edges first, then internal workers on the normal cadence",
          "internal_first": "patch batch workers first because there are more of them",
          "all_never": "do not patch; wait for the vendor's next release",
          "edges_only_ever": "patch the edges and leave internal workers vulnerable permanently"},
         "edges_first"),
        ("patch_urgency",
         "Rate the urgency of patching the edge proxies.",
         list(URGENCY_LEVELS),
         3),
    ),
    scen(
        "chat-logs-retention", "data_retention", "privacy and retention",
        "Retention ticket DATA-77:\n"
        "  dataset: support chat transcripts, includes customer email addresses\n"
        "  current retention: kept indefinitely in analytics bucket\n"
        "  privacy standard P-4: transcripts with personal identifiers expire after 400 days\n"
        "  oldest record age: 1,180 days\n"
        "  proposal: apply a 400-day TTL job to the bucket this sprint.",
        ("ttl_required",
         "Does standard P-4 require expiring part of this dataset?",
         {"false": "indefinite retention is allowed for transcripts",
          "true": "records older than 400 days must be expired"},
         True),
        ("retention_fix",
         "Which remediation matches P-4?",
         {"delete_all": "delete the entire analytics bucket",
          "ttl_job": "apply the 400-day TTL and purge the overdue records",
          "rename_bucket": "rename the bucket so the policy does not match",
          "export_to_csv": "export transcripts to CSV and keep the CSV instead"},
         "ttl_job"),
        ("compliance_gap",
         "Rate the severity of the current retention posture.",
         ["compliant", "minor documentation gap",
          "real violation needing scheduled work", "severe violation needing immediate work"],
         3),
    ),
    scen(
        "holiday-release-window", "release_freeze", "release management",
        "Release calendar note:\n"
        "  freeze window: Dec 20 00:00 - Jan 3 00:00 (all deploys except sev-1 hotfixes)\n"
        "  pending: feature-flag rollout 'new-recommendations', scheduled Dec 22\n"
        "  the flag targets 5% of users and is fully reversible\n"
        "  on-call asked whether the rollout may proceed inside the window.",
        ("rollout_allowed",
         "May the new-recommendations rollout proceed on Dec 22 under the freeze rule?",
         {"false": "the freeze permits only sev-1 hotfixes",
          "true": "any reversible change may ship during the freeze"},
         False),
        ("freeze_options",
         "What are the team's compliant options?",
         {"wait_or_exception": "wait until Jan 3, or request a formal freeze exception",
          "ship_anyway": "proceed because the flag is reversible",
          "rename_hotfix": "label the rollout a sev-1 hotfix",
          "cancel_rollout": "cancel the feature permanently"},
         "wait_or_exception"),
        ("exception_bar",
         "Rate how strong a justification a freeze exception would need.",
         ["no justification needed", "a brief note suffices",
          "a solid business case", "an emergency-level justification"],
         3),
    ),
    scen(
        "p95-latency-alert", "alert_threshold", "monitoring and alerting",
        "Alert review for search-api p95 latency:\n"
        "  current page threshold: 900 ms; alert fired 26 times last week\n"
        "  on-call acknowledged all 26; 0 led to any action (all benign spikes)\n"
        "  SLO for the service: p95 < 1200 ms; last-week actual p95 max: 1180 ms\n"
        "  proposal A: raise the page threshold to 1250 ms\n"
        "  proposal B: keep 900 ms and ask on-call to be more careful",
        ("alert_is_noisy",
         "Is the current 900 ms page threshold producing alert fatigue?",
         {"false": "26 pages that all resolve to no action are healthy",
          "true": "26 no-action pages in a week is a noisy alert"},
         True),
        ("threshold_choice",
         "Which proposal better fits alerting practice?",
         {"raise_threshold": "proposal A: page above the SLO line, watch trends on a dashboard",
          "keep_threshold": "proposal B: keep paging at 900 ms",
          "delete_alert": "remove the latency alert entirely",
          "page_everything": "page on every p95 measurement"},
         "raise_threshold"),
        ("alert_quality",
         "Rate the current alert's signal quality.",
         ["excellent signal", "mostly useful", "mostly noise",
          "pure noise, actively harmful"],
         2),
    ),
    scen(
        "payments-canary", "rollout_strategy", "deployment strategy",
        "Deploy plan for payments-service v2.31:\n"
        "  change: new retry layer around the card processor client\n"
        "  canary stage: 2% of traffic for 30 minutes, error budget 0.1%\n"
        "  observed at minute 12: error rate 0.6%, all retries to the same processor\n"
        "  the deploy pipeline offers: continue / pause / rollback",
        ("continue_rollout",
         "May the canary proceed to full rollout on these observations?",
         {"false": "the observed error rate exceeds the canary budget",
          "true": "0.6% errors are acceptable for a payments canary"},
         False),
        ("canary_action",
         "What should the pipeline operator select?",
         {"continue": "promote to 100% now",
          "pause": "hold at 2% and keep watching for another hour",
          "rollback": "roll back to v2.30 and investigate the retry layer",
          "silence_metrics": "mute the error metric and continue"},
         "rollback"),
        ("signal_strength",
         "Rate the strength of the signal that the canary is unhealthy.",
         ["no signal", "weak hint", "moderate evidence",
          "clear budget breach"],
         3),
    ),
    scen(
        "nightly-backup-verify", "backup_restore", "backup assurance",
        "Backup report, orders-db (nightly full dumps):\n"
        "  last 30 dumps: all completed, checksums clean\n"
        "  restore drills performed in the last 12 months: 0\n"
        "  last attempted restore (11 months ago): FAILED on missing WAL segment\n"
        "  runbook requirement: quarterly restore drills must pass before the\n"
        "  backup set is called 'verified'.",
        ("backups_verified",
         "Do the 30 clean dumps make this backup set 'verified' under the runbook?",
         {"false": "verification requires passing restore drills, not just clean dumps",
          "true": "clean checksums alone verify the set"},
         False),
        ("backup_next_step",
         "What is the correct next step?",
         {"declare_verified": "mark the backup set verified",
          "run_restore_drill": "run a restore drill and fix the WAL gap it reproduces",
          "increase_frequency": "take dumps twice nightly instead",
          "stop_dumps": "stop dumps until the drill passes"},
         "run_restore_drill"),
        ("drill_debt",
         "Rate the severity of the missing restore-drill coverage.",
         ["none", "cosmetic gap", "material assurance gap",
          "critical assurance gap"],
         3),
    ),
    scen(
        "crypto-pr-review", "code_review_gate", "review escalation",
        "PR #5571 summary:\n"
        "  author: external contributor (first PR)\n"
        "  change: new token-signing helper in security/crypto.py\n"
        "  CI: green; two automated reviewers approved (lint + coverage bots)\n"
        "  security-reviewer queue: untouched for 3 days\n"
        "  repo rule: changes under security/ require a human SECURITY-owner approval.",
        ("mergeable_now",
         "Is PR #5571 mergeable in its current state?",
         {"false": "a human SECURITY-owner approval is still required",
          "true": "green CI plus bot approvals satisfy the rule"},
         False),
        ("review_path",
         "What is the correct escalation path?",
         {"merge_now": "merge; the bots already approved",
          "request_owner": "wait for or explicitly request a SECURITY-owner review",
          "move_file": "move the helper out of security/ to dodge the rule",
          "close_pr": "close the PR because external PRs are not allowed"},
         "request_owner"),
        ("bypass_risk",
         "Rate the risk of merging without the owner review.",
         ["none", "process nit only", "meaningful risk",
          "serious risk on a security boundary"],
         3),
    ),
    scen(
        "runtime-flags-drift", "config_change", "runtime configuration",
        "Config drift report, fleet feature flags:\n"
        "  desired (flags.yaml @ main): checkout_v2=true, search_cache=true, ml_rank=false\n"
        "  observed live:             checkout_v2=true, search_cache=false, ml_rank=true\n"
        "  last config apply job: 19 days ago; it failed silently (exit 0, no diff emitted)\n"
        "  two incidents this month traced to operators 'fixing' flags by hand.",
        ("drift_present",
         "Does the report show live configuration drifted from the desired state?",
         {"false": "live flags match flags.yaml",
          "true": "two of three flags diverge from desired"},
         True),
        ("drift_response",
         "What is the correct response to this drift?",
         {"reconcile_and_fix_job": "re-apply flags.yaml and fix the silent-failure path in the apply job",
          "edit_yaml": "edit flags.yaml to match whatever is live",
          "ignore": "drift is expected and harmless",
          "disable_flags": "turn every flag off until next quarter"},
         "reconcile_and_fix_job"),
        ("drift_severity",
         "Rate the severity of a silently-failing config apply loop.",
         ["harmless", "minor hygiene issue", "real operational risk",
          "severe: desired and actual state can diverge unnoticed"],
         3),
    ),
    scen(
        "vendor-analytics", "vendor_evaluation", "third-party adoption",
        "Vendor assessment, 'Metricly' analytics SDK:\n"
        "  proposed use: client-side event counters in the mobile app\n"
        "  SDK behaviour (from vendor docs + our proxy capture): batches events\n"
        "    hourly; payload includes device_id and coarse location\n"
        "  our data policy: coarse location may leave the device only with\n"
        "    explicit user consent; the app has no consent flow today\n"
        "  legal review: scheduled, not yet completed",
        ("adopt_now",
         "May the Metricly SDK ship in the app today?",
         {"false": "consent flow and legal review are both unfinished",
          "true": "vendor documentation alone clears adoption"},
         False),
        ("adoption_path",
         "What sequence is required before shipping?",
         {"consent_then_legal": "add the consent flow and complete legal review first",
          "ship_then_fix": "ship now and add consent in a follow-up",
          "proxy_strip": "strip the location fields in our proxy and skip consent",
          "self_host": "self-host the vendor backend and skip review"},
         "consent_then_legal"),
        ("privacy_risk",
         "Rate the privacy risk of shipping without the consent flow.",
         ["none", "low", "moderate", "high: policy violation by default"],
         3),
    ),
    scen(
        "dark-launch-coverage", "observability_gap", "instrumentation",
        "Launch checklist, recommendations-service dark launch:\n"
        "  [x] service deployed to prod, receiving mirrored traffic\n"
        "  [x] request/error/latency dashboards live\n"
        "  [ ] business metric (click-through on recommendations) instrumented\n"
        "  [ ] per-tenant cost attribution wired\n"
        "  launch gate L-5: dark launches may not graduate without business metrics.",
        ("graduate_allowed",
         "May the service graduate from dark launch today under gate L-5?",
         {"false": "business metrics are not instrumented",
          "true": "technical dashboards are sufficient"},
         False),
        ("obs_next_step",
         "What is the correct next step?",
         {"graduate_anyway": "graduate now; dashboards look clean",
          "instrument_metric": "instrument the click-through metric, then re-review",
          "drop_metric": "remove the business-metric requirement from L-5",
          "turn_off_mirror": "stop mirrored traffic to save cost"},
         "instrument_metric"),
        ("blindness_risk",
         "Rate the risk of graduating without the business metric.",
         ["none", "minor", "moderate", "high: success is unmeasurable"],
         3),
    ),
    scen(
        "hotfix-vs-release", "hotfix_scope", "patch triage",
        "Bug triage note, INV-220:\n"
        "  defect: invoice PDF renders tax line with wrong currency symbol\n"
        "  impact: cosmetic only; amounts and totals are correct\n"
        "  affected: ~2% of invoices, EU tenants only\n"
        "  options: (a) hotfix to prod today, (b) fix in next weekly release in 4 days\n"
        "  hotfix policy: reserved for data corruption, security, or outage-class bugs.",
        ("hotfix_warranted",
         "Does INV-220 qualify for a hotfix under the stated policy?",
         {"false": "a cosmetic defect is not hotfix-class",
          "true": "any customer-visible defect may be hotfixed"},
         False),
        ("triage_choice",
         "Which resolution is correct?",
         {"hotfix_today": "hotfix to production today",
          "weekly_release": "fix in the next weekly release",
          "wontfix": "close as won't-fix; cosmetics do not matter",
          "revert_invoices": "stop sending invoices until fixed"},
         "weekly_release"),
        ("defect_severity",
         "Rate the defect's severity.",
         ["cosmetic/minor", "annoying but bounded", "major functional impact",
          "critical/outage-class"],
         0),
    ),
    scen(
        "edge-traffic-shed", "load_shedding", "overload protection",
        "Load event, edge tier (ongoing):\n"
        "  inbound RPS: 3.1x capacity; upstream provider degraded\n"
        "  request classes by share: search 55%, recommendations 30%, checkout 15%\n"
        "  shed policy P-9: shed lowest-revenue-criticality traffic first;\n"
        "    checkout traffic must never be shed while the tier is alive\n"
        "  proposal on the incident call: shed search first, then recommendations.",
        ("shed_checkout",
         "Does policy P-9 ever permit shedding checkout traffic in this event?",
         {"false": "checkout is never shed while the tier is alive",
          "true": "checkout may be shed once search is exhausted"},
         False),
        ("shed_order",
         "Which shedding order follows P-9?",
         {"search_then_recs": "search, then recommendations; checkout stays served",
          "checkout_first": "checkout first because it is the smallest share",
          "proportional": "shed all three classes proportionally",
          "none": "shed nothing; let the tier fall over"},
         "search_then_recs"),
        ("event_urgency",
         "Rate the urgency of the shedding decision.",
         list(URGENCY_LEVELS),
         3),
    ),
    scen(
        "orders-backfill", "data_backfill", "data pipeline repair",
        "Backfill plan BF-118, orders pipeline:\n"
        "  defect window: events between Aug 2-9 missing the region field\n"
        "  fix: re-emit ~4.1M events through the corrected transformer\n"
        "  constraint C-2: backfill traffic must stay under 15% of the topic's\n"
        "    consumer capacity so live traffic is never starved\n"
        "  computed safe rate: 480 events/s; plan proposes 2,000 events/s\n"
        "  at 480 events/s the backfill finishes in about 2.4 hours.",
        ("rate_compliant",
         "Is the proposed 2,000 events/s rate compliant with constraint C-2?",
         {"false": "2,000/s exceeds the computed 480/s safe rate",
          "true": "any rate is fine for a one-off backfill"},
         False),
        ("backfill_plan",
         "What is the correct backfill plan?",
         {"run_480": "run at 480 events/s for ~2.4 hours within C-2",
          "run_2000": "run at 2,000 events/s to finish faster",
          "run_batched_night": "drop live consumers at night and run unbounded",
          "skip_backfill": "leave the region field missing permanently"},
         "run_480"),
        ("starvation_risk",
         "Rate the risk of running the proposed 2,000 events/s.",
         ["none", "minor slowdown", "live-traffic starvation risk",
          "guaranteed outage"],
         2),
    ),
    scen(
        "billing-oncall-owner", "service_ownership", "ownership routing",
        "Ownership registry excerpt:\n"
        "  service: invoice-renderer   owner: team-billing   oncall: billing-primary\n"
        "  service: pdf-lib            owner: team-platform  oncall: platform-primary\n"
        "Alert at 03:10: invoice-renderer error rate 12%, pdf-lib healthy.\n"
        "Escalation rule E-1: page the owning team's oncall for the failing service;\n"
        "page the platform oncall only if a shared dependency is also failing.",
        ("page_platform",
         "Does rule E-1 direct a page to platform-primary for this alert?",
         {"false": "the failing service is owned by team-billing",
          "true": "the platform oncall owns all PDF rendering"},
         False),
        ("correct_page",
         "Who should be paged under E-1?",
         {"billing_oncall": "billing-primary, owner of the failing service",
          "platform_oncall": "platform-primary, owner of pdf-lib",
          "both": "page both oncalls simultaneously",
          "neither": "wait for the morning standup"},
         "billing_oncall"),
        ("misroute_cost",
         "Rate the cost of paging the wrong oncall here.",
         ["free", "minor annoyance", "delays the fix and burns another team",
          "catastrophic"],
         2),
    ),
    scen(
        "ab-checkout-button", "experiment_analysis", "A/B test readout",
        "Experiment readout EXP-90, checkout button copy test:\n"
        "  arms: control 48,102 sessions; variant 47,960 sessions\n"
        "  conversion: control 3.41%, variant 3.55%\n"
        "  pre-registered primary metric: conversion; min detectable effect 0.3pp\n"
        "  observed lift +0.14pp, 95% CI [-0.21pp, +0.49pp]; test ran full 14 days\n"
        "  a PM proposes shipping the variant 'because the point estimate is up'.",
        ("ship_on_readout",
         "Does the readout support shipping the variant on the primary metric?",
         {"false": "the CI includes zero and the lift is below the MDE",
          "true": "a positive point estimate is enough"},
         False),
        ("correct_read",
         "What is the correct interpretation?",
         {"inconclusive": "inconclusive on the primary metric; ship decision needs more evidence or a business call",
          "ship_now": "ship the variant",
          "kill_variant": "the variant is proven harmful",
          "extend_forever": "keep the test running until the CI excludes zero"},
         "inconclusive"),
        ("proposal_risk",
         "Rate the methodological risk of the PM's proposal.",
         ["sound practice", "minor sloppiness", "p-hacking-adjacent decision risk",
          "directly contradicts the pre-registered rule"],
         3),
    ),
    scen(
        "sessions-cache-stampede", "cache_invalidation", "caching strategy",
        "Cache design note, session store:\n"
        "  current: all session entries share TTL=3600s, written in one nightly batch\n"
        "  symptom: every night at batch+3600s, ~90% of keys expire simultaneously;\n"
        "    origin DB spikes to 4x normal read load for ~8 minutes\n"
        "  proposal: keep TTL=3600s but add uniform jitter +/-600s per key.",
        ("jitter_helps",
         "Does per-key TTL jitter address the nightly stampede?",
         {"false": "jitter cannot change expiry timing",
          "true": "jitter spreads expiries across ~20 minutes instead of one instant"},
         True),
        ("stampede_cause",
         "What is the root cause of the stampede?",
         {"shared_expiry": "one batch write time plus one shared TTL collapses all expiries together",
          "db_bug": "the database has a nightly maintenance bug",
          "ttl_too_long": "3600 seconds is simply too long a TTL",
          "cache_too_small": "the cache cluster is undersized"},
         "shared_expiry"),
        ("residual_risk",
         "Rate the residual risk after adding jitter.",
         ["none", "low: load spreads but peak still occurs nightly",
          "moderate", "high: jitter makes things worse"],
         1),
    ),
    scen(
        "celery-backlog-drain", "queue_backlog", "async job backlog",
        "Queue monitor, emails queue (celery):\n"
        "  backlog: 182,000 jobs, growing ~400/minute\n"
        "  workers: 12 online, consuming ~900/minute combined\n"
        "  job type: transactional receipts (idempotent, must eventually send)\n"
        "  scaling constraint: broker memory limit reached at ~250,000 queued jobs\n"
        "  options: add 12 workers now, or drop jobs older than 24h",
        ("drop_old_jobs",
         "May receipt jobs older than 24h be dropped to relieve the backlog?",
         {"false": "receipts are must-send; dropping loses customer email",
          "true": "old jobs may be discarded freely"},
         False),
        ("drain_plan",
         "What is the correct drain plan?",
         {"add_workers": "add the 12 workers; consumption doubles while arrivals stay 400/min",
          "drop_jobs": "drop jobs older than 24h",
          "restart_broker": "restart the broker to clear memory",
          "pause_producers": "halt email producers indefinitely"},
         "add_workers"),
        ("backlog_urgency",
         "Rate the urgency of acting on this backlog.",
         ["no rush", "routine", "urgent: broker limit approaches",
          "critical: memory limit imminent within hours"],
         3),
    ),
    scen(
        "kms-key-rotation", "key_management", "cryptographic operations",
        "KMS rotation record, key k-payments:\n"
        "  policy: rotate every 365 days; last rotation 402 days ago (overdue)\n"
        "  rotation type: symmetric data-encryption key; old material retained\n"
        "    for decrypt-only after rotation (envelope encryption)\n"
        "  blocker noted: rotation job requires a maintenance token that\n"
        "    expires during the weekend freeze; next token window is Monday",
        ("rotation_overdue",
         "Is k-payments overdue for rotation under the stated policy?",
         {"false": "402 days is within policy",
          "true": "402 days exceeds the 365-day interval"},
         True),
        ("rotation_plan",
         "What is the correct rotation plan?",
         {"rotate_monday": "rotate in the Monday token window; envelope crypto keeps old ciphertext readable",
          "rotate_now": "force rotation this weekend without a valid token",
          "never_rotate": "retire the rotation policy",
          "reencrypt_now": "re-encrypt all data this weekend"},
         "rotate_monday"),
        ("overdue_severity",
         "Rate the severity of a 37-day-overdue key rotation.",
         ["none", "policy hygiene issue", "material compliance gap",
          "immediate cryptographic compromise"],
         2),
    ),
]


def candidate_ids(question):
    if question["type"] == "boolean":
        return ["false", "true"]
    if question["type"] == "choice":
        return list(question["criteria"])
    return [str(i) for i in range(len(question["criteria"]))]


def one_hot(ids, gold):
    """One-hot distribution keyed by candidate id, gold expressed in row terms."""
    if isinstance(gold, bool):
        index = ids.index("true" if gold else "false")
    elif isinstance(gold, int):
        index = gold
    else:
        index = ids.index(gold)
    return {key: (1.0 if i == index else 0.0) for i, key in enumerate(ids)}


def build_rows():
    rows = []
    for s in SCENARIOS:
        questions = {
            "boolean": {"type": "boolean", "instructions": s["boolean"][1]},
            "choice": {"type": "choice", "instructions": s["choice"][1],
                       "criteria": dict(s["choice"][2])},
            "score": {"type": "score", "instructions": s["score"][1],
                      "criteria": list(s["score"][2])},
        }
        if s["boolean"][2] is not None:
            questions["boolean"]["criteria"] = dict(s["boolean"][2])
        golds = {"boolean": s["boolean"][3], "choice": s["choice"][3],
                 "score": s["score"][3]}
        qids = {"boolean": s["boolean"][0], "choice": s["choice"][0],
                "score": s["score"][0]}
        for qtype in QUESTION_TYPES:
            qid = qids[qtype]
            question = questions[qtype]
            ids = candidate_ids(question)
            gold = golds[qtype]
            probs = one_hot(ids, gold)
            row = {
                "id": f"eh1-{s['slug']}-{qtype}",
                "state_id": f"eh1-{s['slug']}",
                "family_id": f"eh1_{s['family']}",
                "split": "test",
                "state": s["state"],
                "questions": {qid: question},
                "gold": {qid: gold},
                "gold_probs": {qid: probs},
                "gold_probs_kind": {qid: "deterministic_truth"},
                "gold_label_kind": {qid: "deterministic_truth"},
                "schema_version": ROW_SCHEMA,
                "metadata": {
                    "source_group_id": f"eh1-{s['slug']}",
                    "question_type": qtype,
                    "domain": s["domain"],
                    "authoring": "hand_authored_independent_heldout_v1",
                    "derived_from_evaluation_corpus": False,
                    "provenance_source_id": "engineering_heldout_v1",
                    "training_authorized_by_this_corpus": False,
                },
                "provenance": {
                    "cohort": "engineering_heldout_v1",
                    "authoring": "hand_authored_for_independent_heldout",
                    "domain": s["domain"],
                    "independence_statement": (
                        "Authored independently of engineering_judgment_corpus_v2: no corpus "
                        "rule, state template, family, field set, instruction string or "
                        "candidate vocabulary was reused; not a reseed or paraphrase of any "
                        "corpus item or evaluation record."),
                    "derived_from_evaluation_corpus": False,
                    "human_reviewed": False,
                    "license": "CC0-1.0",
                    "training_authorized": False,
                },
            }
            rows.append(row)
    return rows


def validate_rows(rows):
    """Every row must pass the real trainer and served validators, and gold must
    be the argmax of gold_probs under the row's own candidate ordering."""
    errors = []
    if len(rows) < 40:
        errors.append(f"heldout requires >=40 items, have {len(rows)}")
    groups = {r["metadata"]["source_group_id"] for r in rows}
    if len(groups) < 20:
        errors.append(f"heldout requires >=20 source groups, have {len(groups)}")
    ids = set()
    for row in rows:
        try:
            targets = validate_training_row(row)
            validate_request({"states": [{k: row[k] for k in ("id", "state", "questions")}]})
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{row.get('id')}: validation failed: {exc}")
            continue
        for qid, q in row["questions"].items():
            t = targets[qid]
            if t["gold_index"] is None or t["gold_distribution_probs"] is None:
                errors.append(f"{row['id']}:{qid}: missing gold")
            elif t["gold_distribution_probs"][t["gold_index"]] != 1.0:
                errors.append(f"{row['id']}:{qid}: gold_probs argmax != gold")
        if row["id"] in ids:
            errors.append(f"duplicate row id {row['id']}")
        ids.add(row["id"])
        if row["split"] != "test":
            errors.append(f"{row['id']}: heldout rows must carry split=test")
    return errors


def self_test():
    rows = build_rows()
    errors = validate_rows(rows)
    return {"schema_version": SCHEMA_VERSION, "mode": "self-test",
            "status": "ok" if not errors else "failed",
            "items": len(rows), "questions": sum(len(r["questions"]) for r in rows),
            "source_groups": len({r["metadata"]["source_group_id"] for r in rows}),
            "families": sorted({r["family_id"] for r in rows}),
            "errors": errors, "training_performed": False, "wrote_output": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if args.output is not None:
            parser.error("--self-test is mutually exclusive with --output")
        report = self_test()
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] == "ok" else 1
    if args.output is None:
        parser.error("one of --self-test or --output is required")
    rows = build_rows()
    errors = validate_rows(rows)
    if errors:
        print(json.dumps({"status": "failed", "errors": errors}, indent=2))
        return 2
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
    # Write-once: refuse to overwrite an existing heldout file.
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)
    print(json.dumps({"status": "ok", "output": str(path), "items": len(rows),
                      "questions": sum(len(r["questions"]) for r in rows),
                      "source_groups": len({r["metadata"]["source_group_id"] for r in rows}),
                      "sha256": hashlib.sha256(text.encode()).hexdigest(),
                      "training_performed": False}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
