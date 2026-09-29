#!/usr/bin/env python3
"""Synthetic JevBench-schema fixtures for the NanoJev JevBench adapter (X1).

Emits ``research/jevbench_fixtures_v1/items.jsonl`` — a small hand-authored
cohort in the canonical JevBench task record shape (``jevbench/tasks.py`` at
the pinned revision ``83831807458d7df424a1e53e5724f3a3ffe2cf89``):

    {id, family, state, question:{type,instructions,criteria}, labels,
     expected, split, group, provenance}

Every item is authored for this file.  Nothing is copied, paraphrased or
derived from real JevBench public/private task rows — per the frozen adapter
contract (``research/jevbench_adapter_contract_v1.json``) benchmark rows may
not even be read before the contract hash is recorded, so these fixtures are
the ONLY items the adapter is validated against in this phase.

Quarantine marker: every item carries ``split="synthetic"`` (a value the
canonical schema does not use for real benchmark records) plus
``provenance.synthetic=true``.  The adapter's ``synthetic`` mode refuses any
item whose split is not ``synthetic``, which keeps this validation path
structurally unable to consume real benchmark rows.

Coverage: noul with and without criteria, choice at 2/4/12/255 options
(contract bounds), score at 2/5/10 levels (contract bounds), a structured
(JSON object) state, a paraphrase group pair, and one expected=null item
(unmeasured ground truth — kept for coverage reporting, excluded from
accuracy).

This file authors data only.  No training, no measurement, no benchmark fetch.

Usage

    .venv/bin/python scripts/build_jevbench_fixtures_v1.py \
        --output-dir research/jevbench_fixtures_v1
    .venv/bin/python scripts/build_jevbench_fixtures_v1.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

SCHEMA_VERSION = "nanojev-jevbench-fixtures-v1"
TASK_SCHEMA = "jevbench::v1.2"
FAMILY = "synthetic_adapter_validation"


def _task(item_id, state, qtype, instructions, criteria, labels, expected,
          group=None, notes=""):
    provenance = {
        "source": "hand-authored synthetic fixture",
        "source_id": item_id,
        "license": "same as repository",
        "imported_at": "2026-09-22",
        "synthetic": True,
        "exclude_reason": None,
        "notes": notes or "authored for adapter contract validation; not a benchmark row",
    }
    question = {"type": qtype, "instructions": instructions}
    if criteria is not None:
        question["criteria"] = criteria
    return {
        "id": item_id,
        "family": FAMILY,
        "state": state,
        "question": question,
        "labels": labels,
        "expected": expected,
        "split": "synthetic",
        "group": group,
        "provenance": provenance,
    }


def build_items():
    items = []

    # --- noul (JevBench boolean) -------------------------------------------------
    items.append(_task(
        "syn-noul-001",
        "Deploy log for release 2026.09.21:\n"
        "- 09:12 UTC: canary at 5% traffic, error rate 0.02%\n"
        "- 09:40 UTC: canary at 25%, error rate 0.03%\n"
        "- 10:05 UTC: rollout completed to 100%, error rate 0.02%\n"
        "- Post-rollout smoke suite: 48/48 checks passed\n"
        "Rollback policy: roll back if post-rollout error rate exceeds 0.5% or "
        "any smoke check fails.",
        "noul",
        "Does the deploy log satisfy the rollback policy's conditions for "
        "rolling back release 2026.09.21?",
        {"false": "The release meets all post-rollout criteria; no rollback "
                  "condition is triggered.",
         "true": "At least one rollback condition is triggered."},
        ["no", "yes"], "no",
        notes="noul with true/false criteria; answer is 'no'"))

    items.append(_task(
        "syn-noul-002",
        {"ticket": "SUP-88123",
         "customer_tier": "free",
         "request": "refund for a yearly plan purchased 40 days ago",
         "policy": {"refund_window_days": 30,
                    "paid_tiers_only": True}},
        "noul",
        "Is this refund request eligible under the stated policy?",
        None,
        ["no", "yes"], "no",
        notes="noul without criteria; structured dict state"))

    # --- choice ------------------------------------------------------------------
    items.append(_task(
        "syn-choice-2opt",
        "Monitoring alert STREAM-7 fired at 03:14 UTC: consumer lag on topic "
        "orders.raw grew from 200 to 48,000 messages over 20 minutes while "
        "producer rate stayed flat. The single consumer pod shows a crash-loop "
        "backoff event at 03:10.",
        "choice",
        "Which is the most probable primary cause of the growing lag?",
        {"deploy": "A new deploy slowed message processing.",
         "consumer_down": "The consumer pod is crash-looping and not "
                          "processing messages."},
        ["consumer_down", "deploy"], "consumer_down",
        notes="minimum-cardinality choice (2 options); labels ordered "
              "differently from criteria keys"))

    items.append(_task(
        "syn-choice-4opt",
        "Pull request #2211 changes `checkout.js`. CI results: lint passed, "
        "unit tests passed, the payments integration suite failed on "
        "'test_currency_rounding'. The PR diff touches only shipping-label "
        "rendering and a locale string table.",
        "choice",
        "What is the most appropriate reviewer action?",
        {"approve": "Approve; the failure is clearly unrelated.",
         "investigate": "Re-run or investigate the failing suite before "
                        "deciding.",
         "request_changes": "Request changes; the diff must have broken "
                            "rounding.",
         "merge_override": "Merge with an admin override to save time."},
        ["approve", "investigate", "merge_override", "request_changes"],
        "investigate",
        notes="4-option choice; plausible-but-wrong distractors"))

    items.append(_task(
        "syn-choice-12opt",
        "Incident review: the 18:00 UTC outage began right after feature flag "
        "`new_pricing_engine` was enabled for 100% of traffic. CPU, memory and "
        "database metrics were normal; p99 latency on /quote jumped from "
        "120ms to 9s. Reverting the flag at 18:22 restored normal latency.",
        "choice",
        "Which remediation should the team apply first?",
        {k: v for k, v in [
            ("revert_flag", "Keep the flag reverted and fix the engine "
                            "offline."),
            ("scale_pods", "Double the pod count to absorb latency."),
            ("add_index", "Add a database index on the quotes table."),
            ("restart_db", "Restart the primary database."),
            ("enable_cache", "Enable the CDN cache for /quote."),
            ("raise_timeout", "Raise the gateway timeout to 30s."),
            ("page_vendor", "Open a ticket with the cloud vendor."),
            ("disable_ssl", "Disable TLS on /quote to cut overhead."),
            ("shard_topic", "Shard the pricing Kafka topic."),
            ("rollback_deploy", "Roll back the morning's unrelated deploy."),
            ("increase_quota", "Raise the API rate-limit quota."),
            ("ignore", "No action; the incident self-resolved."),
        ]},
        ["add_index", "disable_ssl", "enable_cache", "ignore",
         "increase_quota", "page_vendor", "raise_timeout", "restart_db",
         "revert_flag", "rollback_deploy", "scale_pods", "shard_topic"],
        "revert_flag",
        notes="12-option choice with alphabetical label order independent of "
              "criteria order"))

    # 255-option edge: maximum contract cardinality.  Deterministically
    # generated option set; still hand-authored fixture content, not benchmark
    # data.
    many_criteria = {}
    many_labels = []
    for i in range(255):
        key = f"opt_{i:03d}"
        many_criteria[key] = (f"Route the request to handler {i}."
                              if i != 137 else
                              "Route the request to the billing-legacy handler, "
                              "which the state identifies as the owner of "
                              "plan_type='grandfathered-2019'.")
        many_labels.append(key)
    items.append(_task(
        "syn-choice-255opt",
        "Routing table excerpt: account A-9911 has plan_type "
        "'grandfathered-2019'. Policy ROUTE-MAP: every account is handled by "
        "the handler owning its plan_type; no other handler may accept it.",
        "choice",
        "Which handler must receive account A-9911's request?",
        many_criteria, many_labels, "opt_137",
        notes="255-option choice: maximum NanoJev contract cardinality"))

    # --- score -------------------------------------------------------------------
    items.append(_task(
        "syn-score-2lvl",
        "Battery sensor report: voltage 3.1V, within the rated 3.0–4.2V band; "
        "temperature normal; no error flags raised in the last 30 days.",
        "score",
        "Rate the urgency of servicing this sensor.",
        ["Service required immediately.", "No service required."],
        ["0", "1"], 1,
        notes="minimum-cardinality score (2 levels)"))

    items.append(_task(
        "syn-score-5lvl",
        "Customer message: 'Hi — the export button produces an empty CSV for "
        "our February data. All other months export fine. It is not blocking "
        "us, but finance will need it before month-end close next week.'",
        "score",
        "Rate the support priority of this ticket.",
        ["Critical: service down or data loss, all users.",
         "High: core feature broken, no workaround, many users.",
         "Medium: feature degraded, workaround or limited scope.",
         "Low: minor defect or cosmetic issue.",
         "Trivial: question or feedback only."],
        ["0", "1", "2", "3", "4"], 2,
        notes="5-level ordinal score"))

    items.append(_task(
        "syn-score-10lvl",
        "Code review note: function `compute_tax` returns the right value for "
        "all 214 conformance vectors, but uses a quadratic scan where a hash "
        "lookup is standard; runtime measured at 40ms p99 against a 100ms "
        "budget.",
        "score",
        "Rate the severity of this finding on the review rubric.",
        [f"Severity level {i}" for i in range(9)] +
        ["Severity level 9: correctness is fine, performance within budget, "
         "non-idiomatic implementation only."],
        [str(i) for i in range(10)], 9,
        notes="maximum-cardinality score (10 levels)"))

    # --- paraphrase group pair ----------------------------------------------------
    base_facts = ("Warehouse sensor W-12 reported 34.2C at 14:00 UTC and "
                  "34.6C at 14:30 UTC. The freezer's safe band is -25C to "
                  "-18C. Alarm ALM-TEMP triggers above -15C.")
    items.append(_task(
        "syn-noul-grp-a",
        base_facts,
        "noul",
        "Is sensor W-12's reading consistent with a normally operating "
        "freezer?",
        None, ["no", "yes"], "no", group="syn-grp-freezer",
        notes="paraphrase group member A"))
    items.append(_task(
        "syn-noul-grp-b",
        base_facts.replace("reported", "logged").replace("safe band",
                                                         "allowed range"),
        "noul",
        "Given the freezer's allowed temperature range, are W-12's readings "
        "within normal operating conditions?",
        None, ["no", "yes"], "no", group="syn-grp-freezer",
        notes="paraphrase group member B (same decision, rephrased)"))

    # --- unmeasured ground truth ---------------------------------------------------
    items.append(_task(
        "syn-choice-unmeasured",
        "A/B experiment EXP-44 ran 14 days on 2% of traffic: variant B shows "
        "+0.4% conversion (CI crosses zero) and +1.1% revenue per visitor.",
        "choice",
        "Which rollout decision does the experiment support?",
        {"ship": "Ship variant B to all traffic now.",
         "extend": "Extend the experiment for more power.",
         "abandon": "Abandon variant B."},
        ["abandon", "extend", "ship"], None,
        notes="expected=null: unmeasured ground truth; kept for coverage, "
              "excluded from accuracy"))

    return items


def validate_item(item):
    """Mirror of the canonical jevbench.tasks.Task.validate at the pinned
    revision, plus the adapter contract's label-consistency rules and the
    synthetic-split marker.  Fail-closed: any violation raises."""
    q = item["question"]
    if q.get("type") not in ("noul", "choice", "score"):
        raise ValueError(f"{item['id']}: bad question type {q.get('type')!r}")
    if item["split"] != "synthetic":
        raise ValueError(f"{item['id']}: fixture split must be 'synthetic'")
    if not item["labels"]:
        raise ValueError(f"{item['id']}: empty labels")
    labels = [str(x) for x in item["labels"]]
    criteria = q.get("criteria")
    if q["type"] == "noul":
        if labels != ["no", "yes"]:
            raise ValueError(f"{item['id']}: noul labels must be ['no','yes']")
        if criteria is not None and (
                not isinstance(criteria, dict)
                or not set(criteria) <= {"false", "true"}):
            raise ValueError(f"{item['id']}: noul criteria keys must be a "
                             "subset of false/true")
    elif q["type"] == "choice":
        if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 255:
            raise ValueError(f"{item['id']}: choice criteria must be a 2-255 "
                             "entry object")
        if set(labels) != set(criteria):
            raise ValueError(f"{item['id']}: choice labels != criteria keys")
    else:
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
            raise ValueError(f"{item['id']}: score criteria must be a 2-10 "
                             "entry list")
        if labels != [str(i) for i in range(len(criteria))]:
            raise ValueError(f"{item['id']}: score labels must be level "
                             "indices 0..k-1")
    exp = item.get("expected")
    if exp is not None:
        if q["type"] == "score":
            if not isinstance(exp, int) or str(exp) not in labels:
                raise ValueError(f"{item['id']}: bad score expected {exp!r}")
        elif exp not in item["labels"]:
            raise ValueError(f"{item['id']}: expected {exp!r} not in labels")
    if isinstance(item["state"], dict):
        for banned in ("expected", "label", "ground_truth", "answer_key"):
            if banned in item["state"]:
                raise ValueError(f"{item['id']}: state contains banned key "
                                 f"{banned!r}")
    if not item["provenance"].get("synthetic"):
        raise ValueError(f"{item['id']}: missing provenance.synthetic marker")


def manifest_for(items_path, items):
    blob = items_path.read_bytes()
    by_type = {}
    for it in items:
        by_type[it["question"]["type"]] = by_type.get(
            it["question"]["type"], 0) + 1
    cardinalities = [len(it["labels"]) for it in items]
    return {
        "schema_version": SCHEMA_VERSION,
        "task_schema": TASK_SCHEMA,
        "task_schema_source": {
            "repo": "https://github.com/fstandhartinger/jevbench",
            "revision": "83831807458d7df424a1e53e5724f3a3ffe2cf89",
            "file": "jevbench/tasks.py",
            "note": "record shape mirrored from the pinned harness; no task "
                    "rows were read or copied",
        },
        "path": str(items_path),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "items": len(items),
        "counts_by_type": by_type,
        "min_labels": min(cardinalities),
        "max_labels": max(cardinalities),
        "groups": sorted({it["group"] for it in items if it["group"]}),
        "unmeasured": sum(1 for it in items if it["expected"] is None),
        "quarantine": {
            "evaluation_only": True,
            "training_allowed": False,
            "calibration_fit_allowed": False,
            "per_item_failure_tuning_allowed": False,
            "contains_benchmark_rows": False,
        },
        "builder": "scripts/build_jevbench_fixtures_v1.py",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("research/jevbench_fixtures_v1"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    items = build_items()
    ids = [it["id"] for it in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate fixture ids")
    for it in items:
        validate_item(it)

    if args.self_test:
        print(json.dumps({"status": "self_test_passed", "items": len(items),
                          "types": sorted({it["question"]["type"]
                                           for it in items})}))
        return 0

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    items_path = out_dir / "items.jsonl"
    if items_path.exists():
        raise ValueError(f"refusing to overwrite {items_path}")
    with items_path.open("x", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = manifest_for(items_path, items)
    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False,
                  allow_nan=False)
        fh.write("\n")
    print(json.dumps({"status": "fixtures_written",
                      "items": manifest["items"],
                      "items_path": str(items_path),
                      "sha256": manifest["sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
