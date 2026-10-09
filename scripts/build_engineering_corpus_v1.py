#!/usr/bin/env python3
"""Deterministic engineering-judgment training-corpus builder (V1).

Why this file exists
--------------------

The local checkpoint ``checkpoints/local_atomic_seed17/variants/local_atomic_seed17``
abstains on 13/13 engineering-judgment questions at the 0.9 cutoff (max confidence
0.736) while its own maze family reaches 0.998 and is 97.8% correct at >= 0.9.  The
diagnosis (``docs/SKILL_ABSTENTION_DIAGNOSIS_V1.md``) localises the collapse to the
out-of-family question distribution: the checkpoint has **no engineering-judgment
training data at all**.  This builder removes that data blocker by emitting a
contrastive, provenance-carrying engineering-judgment corpus in the exact served
request contract of ``scripts/predict_toy_decisions.py``, together with a second
view in the row contract of ``scripts/train_pipeline_decisions.py``.

Boundaries enforced here
------------------------

* **No training.** Nothing is trained, no checkpoint is read or written, no
  optimizer exists in this file.  Every artefact records
  ``training_authorized_by_this_corpus: false``.
* **No network.** The imports are the standard library, the real request validator
  in ``scripts/predict_toy_decisions.py``, the real row validator in
  ``scripts/train_pipeline_decisions.py`` (imported lazily, for validation only) and
  the canonical-digest helper reused from ``scripts/build_gate_contrastive_v1.py``.
* **No evaluation-corpus reuse.** The builder refuses any source path naming
  ``local_maze_v1``/``ood.jsonl``, the context-relevance test/OOD corpus, the
  workflow V2 evaluation cohort or baseline, or the pre-registered abstention
  survey, and it refuses to emit any item whose content carries a reserved
  evaluation-corpus token or a refused ``source_id``.
* **Contrastive, with a computed flip proof.** A pair is two states whose fact sets
  differ in exactly one leaf.  The expected answers of both members are *computed*
  from the mutated fact sets with the family's frozen rule, and the declared flip
  question type is required to differ; otherwise the build fails loudly before
  anything is written.
* **Re-derivable.** Every item carries a content hash and ``validate_manifest``
  re-runs the whole generation from the seed, so a hand edit fails loudly.

Two emitted views (stated explicitly in ``docs/ENGINEERING_CORPUS_V1.md``)
------------------------------------------------------------------------

``items/<split>.jsonl``
    The **item view**: one JSON object per (pair member x question type), carrying
    the full request, the declared gold answer and distribution, the contrastive
    metadata and the provenance record.  This is the audit view.
``trainer_view/<split>.jsonl``
    The **trainer view**: one JSON object per (member x question) with exactly the
    keys ``scripts/train_pipeline_decisions.py`` requires (``id``, ``state_id``,
    ``family_id``, ``split``, ``state``, ``questions``) plus ``gold``,
    ``gold_probs``, ``gold_probs_kind``, ``gold_label_kind`` and a ``metadata`` block
    carrying ``source_group_id``, ``pair_id``, ``member``, ``question_type`` and the
    contrast record.  Both views are generated from the same manifest, item for item.

Usage
-----

    .venv/bin/python scripts/build_engineering_corpus_v1.py --self-test
    .venv/bin/python scripts/build_engineering_corpus_v1.py --output-dir research/engineering_judgment_corpus_v1
    .venv/bin/python scripts/build_engineering_corpus_v1.py --check research/engineering_judgment_corpus_v1
    .venv/bin/python -m unittest discover -s scripts -p 'test_engineering_corpus*.py'
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import re

# Reuse (do not duplicate) the frozen canonical-digest helper of the existing builder.
from build_gate_contrastive_v1 import digest_value

# The real served-request validator (no torch import at module level).
from predict_toy_decisions import validate_request


SCHEMA_VERSION = "nanojev-engineering-judgment-corpus-v1"
MANIFEST_SCHEMA = "nanojev-engineering-judgment-manifest-v1"
ITEM_SCHEMA = "nanojev-engineering-judgment-item-v1"
TRAINER_ROW_SCHEMA = "nanojev-engineering-judgment-trainer-row-v1"
SOURCE_GROUP_SCHEMA = "nanojev-engineering-source-group-v1"
MANIFEST_NAME = "manifest.json"
ITEM_DIR = "items"
TRAINER_DIR = "trainer_view"
SPLIT_FILES = {"train": "train.jsonl", "dev": "dev.jsonl",
               "calibration": "calibration.jsonl", "test": "test.jsonl"}
SPLITS = ("train", "dev", "calibration", "test")
DEFAULT_SEED = 20260919
CATALOG_VERSION = "engineering-judgment-catalog-v1"
TRAINER = "scripts/train_pipeline_decisions.py"

#: Split skeleton.  Catalog position inside a family, never content, assigns the
#: split, so editing a state's wording can never move an item across the boundary.
SPLIT_CYCLE = ("train", "train", "dev", "calibration", "train", "test")

FAMILIES = (
    "lifecycle_phase",
    "test_scope",
    "request_routing",
    "failure_classification",
    "risk_authorization",
    "context_retention",
    "checkpoint_readiness",
)
QUESTION_TYPES = ("boolean", "choice", "score")
ALLOWED_QUESTION_KEYS = {"type", "instructions", "criteria"}

# --------------------------------------------------------------------------------------
# refusal policy: evaluation corpora may never be a source, and never leak as content
# --------------------------------------------------------------------------------------

#: Path parts that name an evaluation corpus, an evaluation builder, or the
#: pre-registered abstention survey.  Any source path containing one is refused.
#: NOTE: ``build_gate_contrastive_v1.EXCLUDED_DIR_PARTS`` forbids the ``research``
#: directory outright, which would forbid this corpus's own output path
#: (``research/engineering_judgment_corpus_v1``); that check is therefore not reused
#: and the concrete evaluation corpora below are named exactly instead.
FORBIDDEN_PATH_MARKERS = (
    "local_maze_v1", "games_v4", "test.jsonl", "ood.jsonl", "scaled_maze",
    "context_relevance", "relevance_test", "relevance_ood", "workflow_v2",
    "workflow_challenge", "workflow_manifest", "tool_history_fixture",
    "skill_abstention_survey", "abstention_survey", "nanojev_v2_baseline",
    "build_local_maze_data", "build_context_relevance_v1", "build_workflow_challenge_v2",
    "build_tool_history_fixtures_v1", "build_workflow_decisions",
    "build_scaled_games", "build_game_decisions", "label_decision_dataset",
)

#: Content tokens that only exist as *data* in an evaluation cohort (family names, field
#: names, split kinds, rendered values).  These are the leakage tripwire: any one of them
#: in an emitted item or in the manifest fails the build.  Deliberately excluded from this
#: list are policy words that also occur in the refusal provenance flags themselves
#: (for example "superseded", which is a context-relevance split kind and would otherwise
#: match this builder's own "reuses_relevance_test_or_ood_corpus" bookkeeping); those are
#: covered by the path-marker refusals below, which name the corpora exactly.
RESERVED_TOKENS = (
    "required_field", "two_fields", "latest_correction", "limit_check", "wrong_entity",
    "wrong_field", "unit_price", "position_limit", "order_limit",
    "refund_days", "response_hours", "speed_limit", "distance_limit", "robotics",
    "multilingual",
    "catalog_lookup", "smart_home", "known_chance", "tool_history_fixture",
    "non_repeatable_result", "mutable_file_read", "secrets_credentials",
    "skill_abstention_survey", "abstention_survey", "max_confidence",
    "\u5e93\u5b58\u8bb0\u5f55", "\u968f\u673a\u62bd\u6837\u5b9e\u9a8c",
    "\u5bb6\u5ead\u63a7\u5236\u8bb0\u5f55", "\u65f6\u95f4", "\u5730\u70b9\u7f16\u53f7",
)
_RESERVED_PATTERNS = tuple(
    (token, re.compile(r"(?<![0-9A-Za-z])" + re.escape(token) + r"(?![0-9A-Za-z])"))
    for token in RESERVED_TOKENS)


def reserved_hits(text):
    return sorted({token for token, pattern in _RESERVED_PATTERNS if pattern.search(text)})


def forbidden_path_reason(path):
    """Return a fixed reason string when a path names an evaluation corpus."""
    lowered = Path(path).as_posix().lower()
    for marker in FORBIDDEN_PATH_MARKERS:
        if marker.lower() in lowered:
            return f"path contains the evaluation marker {marker!r}"
    return None


def guard_source(source_id, item, label):
    """Refuse an evaluation-corpus source signature or a reserved content token."""
    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError(f"{label}: source_id must be a non-empty string")
    reason = forbidden_path_reason(source_id)
    if reason:
        raise ValueError(f"{label}: source refused: {reason}")
    hits = reserved_hits(json.dumps(item, ensure_ascii=False, sort_keys=True))
    if hits:
        raise ValueError(f"{label}: contains reserved evaluation tokens {hits}")


# --------------------------------------------------------------------------------------
# deterministic rule tables (the frozen label oracle)
# --------------------------------------------------------------------------------------

def _lifecycle_phase(f):
    """The next lifecycle phase, in strict priority order.

    Each prerequisite is its own phase so that clearing any single prerequisite moves
    the answer to the next one: verification (the invariant check), measurement (the
    B0 measurement-integrity result), data (the missing training corpus), signoff
    (a frozen builder plus an independent signature) and finally train.
    """
    if not f["invariant_verified"]:
        return "verification"
    if not f["data_integrity_measured"]:
        return "measurement"
    if not f["dataset_present"]:
        return "data"
    if not (f["builder_frozen"] and f["reviewer_signoff"]):
        return "signoff"
    return "train"


def _lifecycle_ready(f):
    """0-4 readiness: independent cleared prerequisites, clamped to the 5-level rubric.

    The components are independent, so clearing one prerequisite raises the score by
    one.  The clamp only binds when every component is already cleared, which is why
    the two mutually exclusive ``signoff`` levers each still move the score.
    """
    components = (
        bool(f["invariant_verified"]),
        bool(f["data_integrity_measured"]),
        bool(f["dataset_present"]),
        bool(f["builder_frozen"]),
        bool(f["reviewer_signoff"]),
    )
    return min(sum(components) + int(f["training_authorized"]), 5)


def _test_scope(f):
    """A new public entry point forces the full suite; otherwise scope decides."""
    if f["new_public_api"]:
        return "full"
    if f["only_reached_modules"]:
        return "canary"
    if f["touches_shared_module"] or f["changed_lines"] >= 200:
        return "full"
    if not f["behavior_changed"]:
        return "skip"
    return "targeted"


def _test_depth(f):
    if not f["behavior_changed"] and not f["new_public_api"]:
        return 0
    if f["only_reached_modules"]:
        return 3
    if f["touches_shared_module"] or f["changed_lines"] >= 200 or f["new_public_api"]:
        return 2
    return 1


def _test_needs_new_test(f):
    if not f["behavior_changed"]:
        return False
    return bool(f["touches_shared_module"] or f["changed_lines"] >= 200 or f["new_public_api"])


def _route(f):
    if f["candidate_segments"] <= f["scoring_budget"]:
        return "gate_direct"
    if f["mergeable"] and f["candidate_segments"] <= 4 * f["scoring_budget"]:
        return "batch_merge"
    return "gate_bypass"


def _route_plan_ok(f):
    """0-4: does the plan respect the budget, the merge contract and the protected limit."""
    score = 0
    if f["candidate_segments"] <= 4 * f["scoring_budget"] and not f["protected_endpoint"]:
        score += 1
    if f["mergeable"] and not f["protected_endpoint"]:
        score += 1
    if f["candidate_segments"] <= f["scoring_budget"]:
        score += 1
    if not f["protected_endpoint"]:
        score += 1
    return score


def _failure_kind(f):
    if f["inputs_changed"]:
        return "data_bug"
    if f["expected_output_edited"]:
        return "test_bug"
    if not f["repeated_runs_equal"]:
        return "flaky"
    return "code_bug"


def _failure_regression(f):
    return f["product_path_affected"] and f["previously_passing"]


def _failure_confidence(f):
    if f["inputs_changed"] or f["expected_output_edited"]:
        return 2
    if not f["repeated_runs_equal"]:
        return 1
    return 3 if f["previously_passing"] else 2


def _risk_action(f):
    if f["blocked"]:
        return "pause"
    if f["destructive"] and not f["explicit_user_authorization"]:
        return "ask_human"
    if f["public_api_calls"] and not f["explicit_user_authorization"]:
        return "deny"
    return "proceed"


def _risk_level(f):
    penalty = int(f["destructive"]) + int(f["public_api_calls"]) + \
        int(f["blocked"]) + int(not f["explicit_user_authorization"])
    return min(penalty + 1, 3)


def _risk_human_needed(f):
    return (f["destructive"] or f["public_api_calls"]) and not f["explicit_user_authorization"]


def _retention(f):
    load_bearing = f["candidate_item"] == f["queried_item"] and \
        f["candidate_field"] == f["queried_field"] and not f["later_update_exists"]
    return "load_bearing" if load_bearing else "reducible"


def _retention_drop_safe(f):
    """0-3: how safe it is to drop the candidate segment.

    A candidate that is not even about the queried item or field is uninformative
    rather than redundant, so it scores below a redundant (genuinely removable) one.
    """
    if f["candidate_item"] != f["queried_item"] or f["candidate_field"] != f["queried_field"]:
        return 0
    return 3 if f["later_update_exists"] else 2


def _checkpoint_level(f):
    if not f["proposes_removals"]:
        return 0
    if f["protected_deletions"] > 0:
        return 0
    if not f["token_savings"] or f["invalid_outputs"]:
        return 1
    if f["ece"] <= 0.1 and f["maze_test_accuracy"] >= 0.75:
        return 3
    return 2


def _checkpoint_posture(f):
    """The removal posture: the one branch is whether protected segments are touched."""
    if not f["proposes_removals"]:
        return "abstains"
    if f["protected_deletions"] > 0:
        return "removals_proposed"
    return "removable_safe"


def _answer(family, fact):
    """The frozen choice/score rule of each family, keyed by ``<type>:<qid>``."""
    if family == "lifecycle_phase":
        return {"choice:next_phase": _lifecycle_phase(fact),
                "score:release_readiness": _lifecycle_ready(fact)}
    if family == "test_scope":
        return {"choice:which_test": _test_scope(fact),
                "score:verification_depth": _test_depth(fact)}
    if family == "request_routing":
        return {"choice:route": _route(fact),
                "score:plan_acceptability": _route_plan_ok(fact)}
    if family == "failure_classification":
        return {"choice:kind": _failure_kind(fact),
                "score:classification_confidence": _failure_confidence(fact)}
    if family == "risk_authorization":
        return {"choice:action": _risk_action(fact),
                "score:risk_level": _risk_level(fact)}
    if family == "context_retention":
        return {"choice:segment_role": _retention(fact),
                "score:drop_safety": _retention_drop_safe(fact)}
    if family == "checkpoint_readiness":
        return {"choice:removal_posture": _checkpoint_posture(fact),
                "score:readiness_level": _checkpoint_level(fact)}
    raise ValueError(f"unknown family {family!r}")


def _boolean_answer(family, fact):
    """A boolean proposition that the same facts fix (not a restatement of the choice)."""
    if family == "lifecycle_phase":
        return bool(fact["training_authorized"])
    if family == "test_scope":
        return _test_needs_new_test(fact)
    if family == "request_routing":
        return not fact["protected_endpoint"]
    if family == "failure_classification":
        return _failure_regression(fact)
    if family == "risk_authorization":
        return _risk_human_needed(fact)
    if family == "context_retention":
        return _retention(fact) == "load_bearing"
    if family == "checkpoint_readiness":
        return fact["token_savings"] > 0
    raise ValueError(f"unknown family {family!r}")


#: qid per (family, question type); every source group carries all three.
QIDS = {
    "boolean": {"lifecycle_phase": "training_authorized", "test_scope": "needs_new_test",
                "request_routing": "safe_to_drop", "failure_classification": "is_regression",
                "risk_authorization": "needs_human", "context_retention": "is_load_bearing",
                "checkpoint_readiness": "saves_tokens"},
    "choice": {"lifecycle_phase": "next_phase", "test_scope": "which_test",
               "request_routing": "route", "failure_classification": "kind",
               "risk_authorization": "action", "context_retention": "segment_role",
               "checkpoint_readiness": "removal_posture"},
    "score": {"lifecycle_phase": "release_readiness", "test_scope": "verification_depth",
              "request_routing": "plan_acceptability",
              "failure_classification": "classification_confidence",
              "risk_authorization": "risk_level", "context_retention": "drop_safety",
              "checkpoint_readiness": "readiness_level"},
}


def gold_for(family, fact, qtype):
    if qtype == "boolean":
        return QIDS["boolean"][family], _boolean_answer(family, fact)
    qid = QIDS[qtype][family]
    return qid, _answer(family, fact)[f"{qtype}:{qid}"]


def all_gold(family, fact):
    return {qtype: gold_for(family, fact, qtype)[1] for qtype in QUESTION_TYPES}


# --------------------------------------------------------------------------------------
# served-contract question definitions
# --------------------------------------------------------------------------------------

BOOLEAN_CRITERIA = {}

CHOICE_CRITERIA = {
    "lifecycle_phase": {
        "verification": "Re-run the invariant verification the claim rests on",
        "measurement": "Re-publish the measurement-integrity result the claim rests on",
        "data": "Author the missing training data for the blocked family",
        "signoff": "Freeze the builder and obtain the independent reviewer sign-off",
        "train": "Start the training run the frozen artifacts already permit",
    },
    "test_scope": {
        "targeted": "Run only the tests of the touched modules",
        "full": "Run the full repository test suite",
        "canary": "Run a canary subset on the reachable modules only",
        "skip": "Run no tests for this change",
    },
    "request_routing": {
        "gate_direct": "Score every candidate segment in one gate pass",
        "batch_merge": "Score bounded batches and merge the removal plans",
        "gate_bypass": "Let the gate bypass and keep the full context",
        "main_model": "Send the whole context to the main model",
    },
    "failure_classification": {
        "code_bug": "A defect in the implementation under test",
        "test_bug": "The test's own expectation is wrong",
        "data_bug": "The input fixture or label data is wrong",
        "flaky": "A non-deterministic timing or ordering effect",
    },
    "risk_authorization": {
        "proceed": "Run the action as scoped",
        "ask_human": "Ask the human for explicit authorization first",
        "deny": "Refuse the action in this environment",
        "pause": "Resolve the blocking prerequisite before acting",
    },
    "context_retention": {
        "load_bearing": "The segment decides the answer and must be retained",
        "reducible": "The segment can be dropped without changing the answer",
    },
    "checkpoint_readiness": {
        "removals_proposed": "The checkpoint proposes context removals",
        "removable_safe": "The checkpoint proposes removals without touching protected segments",
        "abstains": "The checkpoint abstains and proposes no removals",
    },
}

SCORE_CRITERIA = {
    "lifecycle_phase": ("no data foundation yet", "one prerequisite cleared",
                        "two prerequisites cleared", "three prerequisites cleared",
                        "four prerequisites cleared", "five prerequisites cleared"),
    "test_scope": ("no tests warranted", "targeted tests only", "full suite",
                   "full suite plus canary on reachable modules"),
    "request_routing": ("plan violates the budget", "single-pass scoring infeasible",
                        "batching and merging required", "protected segments present",
                        "plan is directly acceptable"),
    "failure_classification": ("unclassifiable with the facts given",
                               "one discriminator is present", "two discriminators agree",
                               "the classification is fixed by the facts"),
    "risk_authorization": ("no risk identified", "one caution applies",
                           "two cautions apply", "gated and blocked"),
    "context_retention": ("the segment is unrelated to both the item and the field",
                          "the segment is unrelated on one axis",
                          "the segment decides the answer with no later update",
                          "a later record supersedes the segment"),
    "checkpoint_readiness": ("not ready", "early prototype", "usable with supervision",
                             "production ready"),
}

FEATURE_MAP = {
    "lifecycle_phase": "lifecycle-phase-selection",
    "test_scope": "test-vs-skip-selection",
    "request_routing": "request-routing-under-scoring-budget",
    "failure_classification": "failure-classification",
    "risk_authorization": "risk-authorization-gating",
    "context_retention": "context-retention-decision",
    "checkpoint_readiness": "checkpoint-readiness-scoring",
}


# --------------------------------------------------------------------------------------
# frozen per-family field sets and question wording
# --------------------------------------------------------------------------------------

LIFECYCLE_FIELDS = (
    ("data_integrity_measured", "B0 measurement-integrity result published"),
    ("invariant_verified", "trained-invariance check passed"),
    ("dataset_present", "engineering-judgment dataset present"),
    ("builder_frozen", "corpus builder frozen"),
    ("reviewer_signoff", "independent reviewer sign-off recorded"),
    ("training_authorized", "training run authorized by the owner"),
)
TEST_FIELDS = (
    ("changed_lines", "changed lines in the diff"),
    ("touches_shared_module", "diff touches a module imported by other packages"),
    ("behavior_changed", "change alters observable behaviour"),
    ("only_reached_modules", "only modules reachable from the diff can be executed"),
    ("new_public_api", "change adds a public entry point"),
)
ROUTING_FIELDS = (
    ("candidate_segments", "scorable context segments in the request"),
    ("scoring_budget", "candidate paths the gate can score in one pass"),
    ("mergeable", "per-segment removal plans can be merged into one plan"),
    ("protected_endpoint", "request carries protected segments"),
)
FAILURE_FIELDS = (
    ("inputs_changed", "the fixture or input data changed in this diff"),
    ("expected_output_edited", "the expected value was edited in this diff"),
    ("repeated_runs_equal", "repeated runs give identical results"),
    ("previously_passing", "this test passed on the parent revision"),
    ("product_path_affected", "the failure is on a path users exercise"),
)
RISK_FIELDS = (
    ("destructive", "the operation cannot be undone"),
    ("public_api_calls", "the operation reaches a network or public API"),
    ("explicit_user_authorization", "the owner explicitly authorized this operation"),
    ("blocked", "an unresolved prerequisite gates the operation"),
)
RETENTION_FIELDS = (
    ("candidate_item", "item the candidate segment is about"),
    ("candidate_field", "field the candidate segment carries"),
    ("queried_item", "item the user question asks about"),
    ("queried_field", "field the user question asks for"),
    ("later_update_exists", "a later segment updates the same item and field"),
)
CHECKPOINT_FIELDS = (
    ("maze_test_accuracy", "local-maze test accuracy"),
    ("ece", "expected calibration error"),
    ("invalid_outputs", "invalid outputs in the last evaluation"),
    ("proposes_removals", "the checkpoint proposes context removals"),
    ("token_savings", "text tokens saved by proposed removals"),
    ("protected_deletions", "protected segments proposed for removal"),
)

FIELD_SETS = {
    "lifecycle_phase": LIFECYCLE_FIELDS,
    "test_scope": TEST_FIELDS,
    "request_routing": ROUTING_FIELDS,
    "failure_classification": FAILURE_FIELDS,
    "risk_authorization": RISK_FIELDS,
    "context_retention": RETENTION_FIELDS,
    "checkpoint_readiness": CHECKPOINT_FIELDS,
}

HEADINGS = {
    "lifecycle_phase": "NanoJev V2 working notes (repository facts, no recommendation):",
    "test_scope": "Change under review (repository facts, no recommendation):",
    "request_routing": "Scoring request under review (repository facts, no recommendation):",
    "failure_classification": "Failing test report (repository facts, no recommendation):",
    "risk_authorization": "Pending operation under review (repository facts, no recommendation):",
    "context_retention": "Context ledger under review (repository facts, no recommendation):",
    "checkpoint_readiness": "Local checkpoint measurement (repository facts, no recommendation):",
}

INSTRUCTIONS = {
    "lifecycle_phase": (
        "Is a training run authorized by this state?",
        "Which engineering lifecycle phase should run next?",
        "Score the release readiness of the item described here, 0 to 5."),
    "test_scope": (
        "Does this change require a new test?",
        "Which test action is most appropriate?",
        "Score the verification depth this change warrants, 0 to 3."),
    "request_routing": (
        "Is it safe to drop context from this request without a validated gate?",
        "Which component should handle this request?",
        "Score how acceptable this routing plan is, 0 to 4."),
    "failure_classification": (
        "Does this failure indicate a product regression?",
        "What kind of problem is this?",
        "Score how confident the classification is, 0 to 3."),
    "risk_authorization": (
        "Does this operation need a human decision before it runs?",
        "What should happen next?",
        "Score the risk this operation carries, 0 to 3."),
    "context_retention": (
        "Is the candidate segment load-bearing for the question?",
        "What is the role of the candidate segment?",
        "Score how safe it is to drop the candidate segment, 0 to 3."),
    "checkpoint_readiness": (
        "Does this checkpoint currently save tokens?",
        "What is the checkpoint's current removal posture?",
        "Score the readiness of this checkpoint, 0 to 3."),
}


# --------------------------------------------------------------------------------------
# frozen catalog: hand-authored base states and their declared one-fact mutations
# --------------------------------------------------------------------------------------

def _mutation(entry):
    """Normalize one declared mutation: ``(fact, patch)`` or ``(fact, patch, flips)``.

    The optional third element declares which question types *this specific* mutation
    must flip; the group-level ``flip`` is only the default.  A per-mutation override
    is needed where two mutations on the same state clear different prerequisites and
    therefore move different answer types (for example a fact that raises the
    readiness score without changing which phase comes next).
    """
    if len(entry) == 2:
        return {"fact": entry[0], "patch": entry[1], "flip": None}
    return {"fact": entry[0], "patch": entry[1], "flip": tuple(entry[2])}


def _rule(rule_id, family, facts, flip, mutations, source_id, why):
    """One hand-authored source group.

    ``flip`` names the question types whose gold answer the group's mutations must
    change by default.  It is a declaration, not an assumption: every declared flip
    is recomputed and asserted at build time.
    """
    return {"rule_id": rule_id, "family": family, "facts": facts, "flip": tuple(flip),
            "mutations": tuple(_mutation(entry) for entry in mutations),
            "source_id": source_id, "why": why}


CATALOG = (
    # ---------------------------------------------------------------- lifecycle_phase
    _rule("lifecycle-measurement-first", "lifecycle_phase",
          {"data_integrity_measured": False, "invariant_verified": False, "dataset_present": False,
           "builder_frozen": False, "reviewer_signoff": False, "training_authorized": False},
          ("choice", "score"),
          (("invariant_verified", {"invariant_verified": True}),
           ("data_integrity_measured", {"data_integrity_measured": True}, ("score",))),
          "scripts/build_engineering_corpus_v1.py",
          "the invariant check has not passed, so the next phase is verification whatever else "
          "is true; once it passes, the unpublished B0 measurement becomes the next blocker"),
    _rule("lifecycle-verify-before-training", "lifecycle_phase",
          {"data_integrity_measured": True, "invariant_verified": False, "dataset_present": True,
           "builder_frozen": True, "reviewer_signoff": False, "training_authorized": False},
          ("choice", "score"),
          (("invariant_verified", {"invariant_verified": True}),
           ("reviewer_signoff", {"reviewer_signoff": True}, ("score",))),
          "docs/GATE_CONTRASTIVE_PROTOCOL_V1.md",
          "the protocol freezes the construction rule and requires independent review of the "
          "builder, its tests and the no-leakage claim before any training run; with the "
          "invariant verified the missing signature is what still blocks training"),
    _rule("lifecycle-authorization-gap", "lifecycle_phase",
          {"data_integrity_measured": True, "invariant_verified": True, "dataset_present": True,
           "builder_frozen": True, "reviewer_signoff": False, "training_authorized": False},
          ("choice",),
          (("reviewer_signoff", {"reviewer_signoff": True}),),
          "docs/GATE_CONTRASTIVE_PROTOCOL_V1.md",
          "an owner authorization is not the protocol's independent review: the review gate is "
          "a separate prerequisite, so permission alone does not make the phase trainable"),
    _rule("lifecycle-data-missing", "lifecycle_phase",
          {"data_integrity_measured": True, "invariant_verified": True, "dataset_present": False,
           "builder_frozen": True, "reviewer_signoff": True, "training_authorized": True},
          ("choice",),
          (("dataset_present", {"dataset_present": True}),),
          "research/abstention-survey-v1 (fact source only; file refused by the builder)",
          "the abstention diagnosis records that the checkpoint has no engineering-judgment "
          "training data at all; here the review already signed off and the only gap is data"),
    _rule("lifecycle-unreviewed-authorization", "lifecycle_phase",
          {"data_integrity_measured": True, "invariant_verified": True, "dataset_present": False,
           "builder_frozen": False, "reviewer_signoff": False, "training_authorized": False},
          ("boolean",),
          (("training_authorized", {"training_authorized": True}),),
          "docs/GATE_CONTRASTIVE_PROTOCOL_V1.md",
          "with no dataset present the readiness score cannot rise on a permission alone; "
          "granting training authorization raises only the authorization component"),
    _rule("lifecycle-signoff-reviewer-only", "lifecycle_phase",
          {"data_integrity_measured": True, "invariant_verified": True, "dataset_present": True,
           "builder_frozen": True, "reviewer_signoff": False, "training_authorized": True},
          ("choice",),
          (("reviewer_signoff", {"reviewer_signoff": True}),),
          "docs/GATE_CONTRASTIVE_PROTOCOL_V1.md",
          "the protocol requires an independent sign-off as well as a frozen builder; with the "
          "builder frozen the missing signature is what keeps the phase at signoff"),
    _rule("lifecycle-signoff-builder-only", "lifecycle_phase",
          {"data_integrity_measured": True, "invariant_verified": True, "dataset_present": True,
           "builder_frozen": False, "reviewer_signoff": True, "training_authorized": True},
          ("choice",),
          (("builder_frozen", {"builder_frozen": True}),),
          "docs/GATE_CONTRASTIVE_PROTOCOL_V1.md",
          "a signature on an unfrozen builder is not a frozen artifact; freezing the builder is "
          "the one fact that completes the signoff precondition"),

    # ---------------------------------------------------------------- test_scope
    _rule("test-scope-shared-module", "test_scope",
          {"changed_lines": 12, "touches_shared_module": False, "behavior_changed": True,
           "only_reached_modules": False, "new_public_api": False},
          ("choice", "score"),
          (("touches_shared_module", {"touches_shared_module": True}),),
          "scripts/predict_toy_decisions.py",
          "a small behavior change confined to one module needs only that module's tests; "
          "touching a module other packages import makes the change shared"),
    _rule("test-scope-large-diff", "test_scope",
          {"changed_lines": 20, "touches_shared_module": False, "behavior_changed": True,
           "only_reached_modules": False, "new_public_api": False},
          ("choice", "score"),
          (("changed_lines", {"changed_lines": 420}),),
          "scripts/build_gate_contrastive_v1.py",
          "a large diff confined to one module is still targeted; the shared-module fact, not "
          "the size alone, is what forces the full suite"),
    _rule("test-scope-behavior-preserving", "test_scope",
          {"changed_lines": 30, "touches_shared_module": False, "behavior_changed": False,
           "only_reached_modules": False, "new_public_api": False},
          ("choice", "score"),
          (("behavior_changed", {"behavior_changed": True}),),
          "docs/GATE_CONTRASTIVE_PROTOCOL_V1.md",
          "the protocol's rule that a behaviour change requires a test: with behaviour "
          "unchanged no behavioural test run is warranted"),
    _rule("test-scope-canary", "test_scope",
          {"changed_lines": 40, "touches_shared_module": True, "behavior_changed": True,
           "only_reached_modules": True, "new_public_api": False},
          ("choice", "score"),
          (("only_reached_modules", {"only_reached_modules": False}),),
          "scripts/build_gate_contrastive_v1.py",
          "when only modules reachable from the diff can be executed, the honest plan is a "
          "canary subset rather than a claim that the full suite passed"),
    _rule("test-scope-new-api-flag-only", "test_scope",
          {"changed_lines": 25, "touches_shared_module": False, "behavior_changed": True,
           "only_reached_modules": False, "new_public_api": True},
          ("choice", "score"),
          (("new_public_api", {"new_public_api": False}),),
          "scripts/predict_toy_decisions.py",
          "a new public entry point demands a test even when the diff looks behaviour-"
          "preserving; the new-public-API fact is what removes the obligation to run tests"),

    # ---------------------------------------------------------------- request_routing
    _rule("routing-budget-exceeded", "request_routing",
          {"candidate_segments": 400, "scoring_budget": 32, "mergeable": True,
           "protected_endpoint": False},
          ("choice", "score"),
          (("candidate_segments", {"candidate_segments": 24}),),
          "research/abstention-survey-v1 (fact source only; file refused by the builder)",
          "the recorded routing geometry: about 400 scorable segments against a 32-candidate "
          "budget, so the gate must bypass with scoring_budget_exceeded instead of silently "
          "truncating the removal plan"),
    _rule("routing-mergeable-batching", "request_routing",
          {"candidate_segments": 96, "scoring_budget": 32, "mergeable": True,
           "protected_endpoint": False},
          ("choice", "score"),
          (("mergeable", {"mergeable": False}),),
          "docs/GATE_CONTRASTIVE_PROTOCOL_V1.md",
          "bounded batches are only mergeable into one removal plan when the per-segment plans "
          "are declared mergeable; otherwise batching cannot be recombined"),
    _rule("routing-unmergeable-bypass", "request_routing",
          {"candidate_segments": 80, "scoring_budget": 32, "mergeable": False,
           "protected_endpoint": False},
          ("choice", "score"),
          (("mergeable", {"mergeable": True}),),
          "docs/GATE_CONTRASTIVE_PROTOCOL_V1.md",
          "without mergeability the gate cannot reconstruct one decision for the request, so "
          "the safe route is bypass rather than a partial removal"),
    _rule("routing-protected-endpoint", "request_routing",
          {"candidate_segments": 120, "scoring_budget": 32, "mergeable": True,
           "protected_endpoint": True},
          ("boolean", "score"),
          (("protected_endpoint", {"protected_endpoint": False}),),
          "docs/GATE_CONTRASTIVE_PROTOCOL_V1.md",
          "the protocol's M2 hard gate: a protected segment may never be proposed for removal, "
          "so a request carrying protected segments is not safe to drop from"),
    _rule("routing-near-budget", "request_routing",
          {"candidate_segments": 40, "scoring_budget": 32, "mergeable": True,
           "protected_endpoint": False},
          ("choice", "score"),
          (("candidate_segments", {"candidate_segments": 20}),),
          "docs/GATE_CONTRASTIVE_PROTOCOL_V1.md",
          "just over the budget the gate still batches; within the budget a single pass is "
          "available and batching would be wasted work"),

    # ---------------------------------------------------------------- failure_classification
    _rule("failure-default-code-bug", "failure_classification",
          {"inputs_changed": False, "expected_output_edited": False, "repeated_runs_equal": True,
           "previously_passing": True, "product_path_affected": True},
          ("choice", "score"),
          (("repeated_runs_equal", {"repeated_runs_equal": False}),
           ("inputs_changed", {"inputs_changed": True}),
           ("expected_output_edited", {"expected_output_edited": True})),
          "scripts/predict_toy_decisions.py",
          "with unchanged inputs, unchanged expectations and a deterministic repeat, every "
          "discriminator points at the implementation"),
    _rule("failure-edited-expectation", "failure_classification",
          {"inputs_changed": False, "expected_output_edited": True, "repeated_runs_equal": True,
           "previously_passing": True, "product_path_affected": False},
          ("choice", "score"),
          (("expected_output_edited", {"expected_output_edited": False}),),
          "docs/SKILL_ABSTENTION_DIAGNOSIS_V1.md",
          "the diagnosis corrects a published claim that its own measurement contradicted; an "
          "expectation edited without a code change is a test defect, and for this shape of "
          "mismatch the implementation is not implicated"),
    _rule("failure-input-data-drift", "failure_classification",
          {"inputs_changed": True, "expected_output_edited": False, "repeated_runs_equal": True,
           "previously_passing": True, "product_path_affected": True},
          ("choice", "score"),
          (("inputs_changed", {"inputs_changed": False}),),
          "docs/GATE_CONTRASTIVE_PROTOCOL_V1.md",
          "a changed fixture invalidates the expectation written against the old fixture, so the "
          "defect is in the data rather than the implementation"),
    _rule("failure-flaky-repeat", "failure_classification",
          {"inputs_changed": False, "expected_output_edited": False, "repeated_runs_equal": False,
           "previously_passing": True, "product_path_affected": True},
          ("choice", "score"),
          (("repeated_runs_equal", {"repeated_runs_equal": True}),),
          "scripts/build_gate_contrastive_v1.py",
          "the contrastive builder pins determinism by construction (a repeated call returned "
          "bitwise-identical results in the diagnosis), so non-determinism here is a property "
          "of the run rather than of the expectation"),
    _rule("failure-nonregression-refactor", "failure_classification",
          {"inputs_changed": False, "expected_output_edited": False, "repeated_runs_equal": True,
           "previously_passing": False, "product_path_affected": False},
          ("score",),
          (("previously_passing", {"previously_passing": True}),),
          "docs/SKILL_ABSTENTION_DIAGNOSIS_V1.md",
          "a never-passing check on a path users do not exercise is a code defect but not a "
          "product regression; the confidence score rises once the expectation itself has a "
          "recorded passing history"),

    # ---------------------------------------------------------------- risk_authorization
    _rule("risk-authorized-write", "risk_authorization",
          {"destructive": True, "public_api_calls": False, "explicit_user_authorization": True,
           "blocked": False},
          ("choice", "boolean", "score"),
          (("explicit_user_authorization", {"explicit_user_authorization": False}),),
          "AGENTS.md",
          "the repository rule that a user directive may authorize a bounded local write; "
          "absent that authorization the same irreversible write needs the human first"),
    _rule("risk-destructive-without-consent", "risk_authorization",
          {"destructive": True, "public_api_calls": False, "explicit_user_authorization": False,
           "blocked": False},
          ("choice", "boolean", "score"),
          (("destructive", {"destructive": False}),),
          "AGENTS.md",
          "an irreversible action without recorded authorization escalates to the human; "
          "removing the irreversibility is the single fact that changes the answer"),
    _rule("risk-network-without-authorization", "risk_authorization",
          {"destructive": False, "public_api_calls": True, "explicit_user_authorization": False,
           "blocked": False},
          ("choice", "boolean", "score"),
          (("public_api_calls", {"public_api_calls": False}),),
          "docs/GATE_CONTRASTIVE_PROTOCOL_V1.md",
          "the protocol's no-network boundary: a provider or public API call sits outside the "
          "authorized envelope even when it is not destructive"),
    _rule("risk-blocked-prerequisite", "risk_authorization",
          {"destructive": False, "public_api_calls": False, "explicit_user_authorization": True,
           "blocked": True},
          ("choice", "score"),
          (("blocked", {"blocked": False}),),
          "docs/SKILL_ABSTENTION_DIAGNOSIS_V1.md",
          "an unresolved prerequisite gates the action regardless of authorization; the "
          "blockage, not the write, is the risk being scored"),
    _rule("risk-consent-for-destructive-push", "risk_authorization",
          {"destructive": True, "public_api_calls": True, "explicit_user_authorization": False,
           "blocked": False},
          ("choice", "boolean"),
          (("explicit_user_authorization", {"explicit_user_authorization": True}),),
          "AGENTS.md",
          "an irreversible network-reaching action without consent is gated; recorded consent "
          "is the one fact that lets it proceed in this local environment (the risk score is "
          "unchanged because two cautions still apply either way)"),

    # ---------------------------------------------------------------- context_retention
    _rule("retention-candidate-decides", "context_retention",
          {"candidate_item": "gate", "candidate_field": "threshold", "queried_item": "gate",
           "queried_field": "threshold", "later_update_exists": False},
          ("choice", "score"),
          (("later_update_exists", {"later_update_exists": True}),),
          "docs/GATE_CONTRASTIVE_PROTOCOL_V1.md",
          "the last-value-wins oracle frozen by the contrastive rule: the candidate is "
          "load-bearing exactly when dropping it changes the answer, and a later record for the "
          "same item and field removes that dependence"),
    _rule("retention-wrong-entity", "context_retention",
          {"candidate_item": "gate", "candidate_field": "threshold", "queried_item": "corpus",
           "queried_field": "threshold", "later_update_exists": False},
          ("choice", "score"),
          (("candidate_item", {"candidate_item": "corpus"}),),
          "docs/GATE_CONTRASTIVE_PROTOCOL_V1.md",
          "entity binding: a segment about a different item cannot decide the queried item's "
          "field, which is why the gate abstains instead of guessing the binding"),
    _rule("retention-wrong-field", "context_retention",
          {"candidate_item": "corpus", "candidate_field": "threshold", "queried_item": "corpus",
           "queried_field": "item_count", "later_update_exists": False},
          ("choice", "score"),
          (("candidate_field", {"candidate_field": "item_count"}),),
          "docs/GATE_CONTRASTIVE_PROTOCOL_V1.md",
          "field binding: a segment carrying a different field of the right item does not answer "
          "the queried field"),
    _rule("retention-later-update-evidence", "context_retention",
          {"candidate_item": "corpus", "candidate_field": "item_count", "queried_item": "corpus",
           "queried_field": "item_count", "later_update_exists": True},
          ("choice", "score"),
          (("later_update_exists", {"later_update_exists": False}),),
          "docs/GATE_CONTRASTIVE_PROTOCOL_V1.md",
          "supersession: a later record for the same item and field makes an on-topic earlier "
          "record redundant, so dropping it is safe only while that later record exists"),
    _rule("retention-same-field-different-item", "context_retention",
          {"candidate_item": "corpus", "candidate_field": "count", "queried_item": "gate",
           "queried_field": "count", "later_update_exists": True},
          ("score",),
          (("candidate_item", {"candidate_item": "gate"}),),
          "docs/GATE_CONTRASTIVE_PROTOCOL_V1.md",
          "a redundant segment for another item is uninformative, not removable; matching the "
          "queried item is what upgrades it from ignorable to removably redundant (the role "
          "answer itself stays reducible in both worlds)"),

    # ---------------------------------------------------------------- checkpoint_readiness
    _rule("checkpoint-abstains-everywhere", "checkpoint_readiness",
          {"maze_test_accuracy": 0.7784, "ece": 0.0851, "invalid_outputs": 0,
           "proposes_removals": False, "token_savings": 0, "protected_deletions": 0},
          ("score",),
          (("proposes_removals", {"proposes_removals": True}),),
          "docs/SKILL_ABSTENTION_DIAGNOSIS_V1.md",
          "the diagnosis records this checkpoint at 77.84% local-maze accuracy with zero "
          "proposed removals; a checkpoint that proposes nothing saves nothing, so readiness "
          "stays at the floor no matter how good its in-domain accuracy is"),
    _rule("checkpoint-protected-deletion", "checkpoint_readiness",
          {"maze_test_accuracy": 0.9100, "ece": 0.0400, "invalid_outputs": 0,
           "proposes_removals": True, "token_savings": 240, "protected_deletions": 1},
          ("score",),
          (("protected_deletions", {"protected_deletions": 0}),),
          "docs/GATE_CONTRASTIVE_PROTOCOL_V1.md",
          "M2 is a hard gate: a single protected-segment deletion fails the run immediately, "
          "regardless of every other metric"),
    _rule("checkpoint-calibration-gap", "checkpoint_readiness",
          {"maze_test_accuracy": 0.8100, "ece": 0.2200, "invalid_outputs": 0,
           "proposes_removals": True, "token_savings": 96, "protected_deletions": 0},
          ("score",),
          (("ece", {"ece": 0.0600}),),
          "docs/SKILL_ABSTENTION_DIAGNOSIS_V1.md",
          "the diagnosis reports ECE 0.0851 in-domain and treats calibration as part of the "
          "readiness story; a much worse ECE blocks the top level even when tokens are saved"),
    _rule("checkpoint-invalid-outputs", "checkpoint_readiness",
          {"maze_test_accuracy": 0.9000, "ece": 0.0500, "invalid_outputs": 3,
           "proposes_removals": True, "token_savings": 150, "protected_deletions": 0},
          ("score",),
          (("invalid_outputs", {"invalid_outputs": 0}),),
          "research/abstention-survey-v1 (fact source only; file refused by the builder)",
          "the survey records 0 invalid outputs for this checkpoint, and the diagnosis's "
          "proportional readout requires finite probabilities summing to one; invalid outputs "
          "cap readiness below the supervised-use level even though tokens are still saved"),
    _rule("checkpoint-low-accuracy-calibrated", "checkpoint_readiness",
          {"maze_test_accuracy": 0.6000, "ece": 0.0400, "invalid_outputs": 0,
           "proposes_removals": True, "token_savings": 120, "protected_deletions": 0},
          ("score",),
          (("maze_test_accuracy", {"maze_test_accuracy": 0.8600}),),
          "docs/SKILL_ABSTENTION_DIAGNOSIS_V1.md",
          "well-calibrated but weak in-domain accuracy stays at supervised use; the diagnosis's "
          "0.9-band evidence is what would support the top level"),
)


def _spec_for(rule_id):
    for rule in CATALOG:
        if rule["rule_id"] == rule_id:
            return {"rule_id": rule_id, "family": rule["family"],
                    "heading": HEADINGS[rule["family"]], "fields": FIELD_SETS[rule["family"]]}
    raise ValueError(f"no catalog rule {rule_id!r}")


def _validate_tables():
    """Fail loudly if the catalog and its field sets have drifted apart."""
    seen = set()
    count = 0
    for rule in CATALOG:
        rule_id, family = rule["rule_id"], rule["family"]
        if rule_id in seen:
            raise ValueError(f"duplicate catalog rule id {rule_id!r}")
        seen.add(rule_id)
        if family not in FAMILIES:
            raise ValueError(f"{rule_id}: unknown family {family!r}")
        if set(rule["facts"]) != {key for key, _ in FIELD_SETS[family]}:
            raise ValueError(f"{rule_id}: facts and the family field set disagree")
        if set(rule["flip"]) - set(QUESTION_TYPES):
            raise ValueError(f"{rule_id}: unknown flip question type")
        if not rule["mutations"]:
            raise ValueError(f"{rule_id}: a source group needs at least one mutation")
        for mutation in rule["mutations"]:
            if mutation["fact"] not in rule["facts"]:
                raise ValueError(f"{rule_id}: mutation names an unknown fact "
                                 f"{mutation['fact']!r}")
            if mutation["flip"] is not None and set(mutation["flip"]) - set(QUESTION_TYPES):
                raise ValueError(f"{rule_id}: mutation {mutation['fact']!r} declares an unknown "
                                 f"flip question type")
        count += 1
    return count


# --------------------------------------------------------------------------------------
# deterministic variation so catalog entries are not near-duplicates
# --------------------------------------------------------------------------------------

#: Only free fields are varied, and only within bounds that cannot change the base
#: state's rule outcome (re-validated on every build).
JITTER_BOUNDS = {
    "changed_lines": (0, 2),
    "candidate_segments": (0, 2),
    "scoring_budget": (0, 4),
    "token_savings": (0, 3),
}


def jitter(facts, key):
    """Deterministic variation of the free fields.

    A field that is exactly zero stays zero: several rules branch on a zero
    (``token_savings``, ``only_reached_modules``-style counts), so moving it could
    change the label.  Every other value moves by a small bounded step, and the
    result is checked against the un-jittered rule outcome on every build.
    """
    rng = random.Random(f"nanojev-engineering-catalog:{key}")
    varied = deepcopy(facts)
    for name, (low, high) in sorted(JITTER_BOUNDS.items()):
        if isinstance(varied.get(name), int) and varied[name] != 0:
            varied[name] = varied[name] + rng.randint(low, high)
    return varied


# --------------------------------------------------------------------------------------
# state rendering, item construction, contrastive pairs with a computed flip proof
# --------------------------------------------------------------------------------------

def render_state(facts, family):
    lines = [HEADINGS[family]]
    for key, label in FIELD_SETS[family]:
        value = facts[key]
        rendered = ("yes" if value else "no") if isinstance(value, bool) else str(value)
        lines.append(f"- {label}: {rendered}")
    return "\n".join(lines)


def question_body(qtype, family):
    if qtype == "boolean":
        return {"type": "boolean", "instructions": INSTRUCTIONS[family][0],
                "criteria": dict(BOOLEAN_CRITERIA)}
    if qtype == "choice":
        return {"type": "choice", "instructions": INSTRUCTIONS[family][1],
                "criteria": dict(CHOICE_CRITERIA[family])}
    if qtype == "score":
        return {"type": "score", "instructions": INSTRUCTIONS[family][2],
                "criteria": list(SCORE_CRITERIA[family])}
    raise ValueError(f"unknown question type {qtype!r}")


def build_request(facts, family, state_id):
    """One served-contract request carrying all three question types for the state."""
    return {"states": [{"id": state_id, "state": render_state(facts, family),
                        "questions": {QIDS[qtype][family]: question_body(qtype, family)
                                      for qtype in QUESTION_TYPES}}]}


def candidate_keys(qtype, criteria):
    """The candidate ids exactly as the served readout labels them."""
    if qtype == "boolean":
        return ["false", "true"]
    if qtype == "choice":
        return list(criteria)
    return [str(position) for position in range(len(criteria))]


def gold_index_of(qtype, gold):
    """The zero-based readout index of a gold answer, mirroring predict_toy_decisions."""
    if qtype == "boolean":
        return 1 if gold is True else 0
    if qtype == "choice":
        return None  # resolved against the candidate keys by the caller
    return int(gold)


def probability_vector(qtype, criteria, gold):
    keys = candidate_keys(qtype, criteria)
    if qtype == "choice":
        index = keys.index(gold)
    else:
        index = gold_index_of(qtype, gold)
    return {"keys": keys, "gold_index": index,
            "probabilities": {key: (1.0 if position == index else 0.0)
                              for position, key in enumerate(keys)}}


def fact_delta(base, variant):
    """Every (path, before, after) leaf difference between two fact dicts."""
    changes = []
    for key in sorted(set(base) | set(variant)):
        if key not in base or key not in variant:
            changes.append((key, base.get(key), variant.get(key)))
        elif base[key] != variant[key]:
            changes.append((key, base[key], variant[key]))
    return changes


def make_items(pair, member):
    """Materialize every question type of one member as its own item, with computed gold."""
    family = pair["family"]
    rule = pair["rule"]
    facts = pair[f"{member}_facts"]
    request = build_request(facts, family, pair["state_id"])
    items = []
    for qtype in QUESTION_TYPES:
        qid, gold = gold_for(family, facts, qtype)
        criteria_values = request["states"][0]["questions"][qid]["criteria"]
        vector = probability_vector(qtype, criteria_values, gold)
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
                "source_id": rule["source_id"],
                "rule_id": rule["rule_id"],
                "catalog_version": CATALOG_VERSION,
                "rule": rule["why"],
                "why_correct": (f"under the frozen {family} rule the facts in this state fix "
                                f"{qid}={gold!r}"),
                "fact_basis": {key: facts[key] for key in sorted(facts)},
                "fact_keys": sorted(facts),
                "authoring": ("hand_authored_state_and_rule_with_programmatic_rendering"
                              if member == "base" else
                              "programmatic_one_fact_mutation_of_a_hand_authored_state"),
                "human_reviewed": False,
                "derived_from_evaluation_corpus": False,
                "training_authorized": False,
            },
            "contrastive": {
                "pair_id": pair["pair_id"],
                "member": member,
                "variant_of": f"{pair['pair_id']}-{qtype}-base" if member == "variant" else None,
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


def pairs_for(seed):
    """Derive every base/variant pair and verify each declared flip by computation.

    The expected answers of both members are computed from their own fact sets.  A
    declared flip that does not actually change the computed answer raises before any
    item is emitted, so a broken contrast can never reach an output directory.
    """
    _validate_tables()
    pairs = []
    for rule in CATALOG:
        family = rule["family"]
        base_facts = jitter(rule["facts"], f"{seed}:{rule['rule_id']}")
        if all_gold(family, base_facts) != all_gold(family, rule["facts"]):
            raise ValueError(f"{rule['rule_id']}: jitter changed the base rule's answer")
        base_gold = all_gold(family, base_facts)
        for index, mutation in enumerate(rule["mutations"], start=1):
            mutated_fact, patch = mutation["fact"], mutation["patch"]
            variant_facts = deepcopy(base_facts)
            variant_facts.update(patch)
            delta = fact_delta(base_facts, variant_facts)
            if len(delta) != 1 or delta[0][0] != mutated_fact:
                raise ValueError(f"{rule['rule_id']}#{index}: mutation {mutated_fact!r} must "
                                 f"change exactly that fact, changed "
                                 f"{[name for name, _, _ in delta]}")
            if delta[0][1] == delta[0][2]:
                raise ValueError(f"{rule['rule_id']}#{index}: mutation {mutated_fact!r} does not "
                                 f"change the value")
            variant_gold = all_gold(family, variant_facts)
            flipped = tuple(qtype for qtype in QUESTION_TYPES
                            if base_gold[qtype] != variant_gold[qtype])
            declared = mutation["flip"] if mutation["flip"] is not None else tuple(rule["flip"])
            if set(declared) - set(flipped):
                missing = sorted(set(declared) - set(flipped))
                raise ValueError(
                    f"{rule['rule_id']}#{index}: the one-fact change to {mutated_fact!r} does "
                    f"not flip the declared question type(s) {missing}: "
                    f"{[(q, base_gold[q], variant_gold[q]) for q in missing]}")
            pair_id = f"ejc-{rule['rule_id']}" if index == 1 else \
                f"ejc-{rule['rule_id']}-p{index:02d}"
            state_id = rule["rule_id"] if index == 1 else f"{rule['rule_id']}-p{index:02d}"
            pairs.append({"pair_id": pair_id, "rule_id": rule["rule_id"],
                          "state_id": state_id, "family": family,
                          "feature": FEATURE_MAP[family], "mutated_fact": mutated_fact,
                          "value_before": delta[0][1], "value_after": delta[0][2],
                          "flip_question_types": flipped,
                          "gold_before": base_gold, "gold_after": variant_gold,
                          "base_facts": base_facts, "variant_facts": variant_facts,
                          "declared_flip": list(declared), "rule": rule})
    return pairs


def assign_splits(pairs, seed):
    """Assign splits by catalog position within each family; never by content."""
    per_family = {}
    for pair in pairs:
        per_family.setdefault(pair["family"], []).append(pair)
    for family, family_pairs in per_family.items():
        for index, pair in enumerate(family_pairs):
            pair["split"] = SPLIT_CYCLE[(index + abs(seed)) % len(SPLIT_CYCLE)]
            pair["source_group_id"] = digest_value({
                "schema": SOURCE_GROUP_SCHEMA, "seed": seed, "family": family,
                "pair_id": pair["pair_id"]})
    return pairs


# --------------------------------------------------------------------------------------
# manifest and the two emitted views
# --------------------------------------------------------------------------------------

PROVENANCE = {
    "source": "self_authored_engineering_judgment_states_and_frozen_rules",
    "license": "CC0-1.0",
    "contains_real_user_data": False,
    "contains_real_credentials": False,
    "contains_real_tool_output": False,
    "derived_from_any_existing_evaluation_record": False,
    "reuses_local_maze_evaluation_corpus": False,
    "reuses_relevance-test-or-ood-corpus": False,
    "reuses_workflow-v2-evaluation-cohort-or-baseline": False,
    "reuses_pre-registered-abstention-survey": False,
    "training_performed": False,
    "checkpoint_read": False,
    "checkpoint_modified": False,
    "network_access": False,
    "corpus_role": "training_data_prerequisite_only",
    "training_authorized_by_this_corpus": False,
    "labels_human_reviewed": False,
}

CONSTRUCTION_RULE = {
    "version": "engineering-judgment-rule-v1",
    "statement": ("A pair is two served-contract request bodies whose states differ in exactly "
                  "one fact leaf of the family's enumerated fact set; the expected answers of "
                  "both members are computed from their own facts with the family's frozen rule, "
                  "and the declared flip question type(s) are required to differ. A pair whose "
                  "declared flip does not hold raises before anything is written."),
    "base_states": "hand-authored repository-realistic fact tables in the served contract",
    "split_unit": "source_group_id (a pair's members always share one group and one split)",
    "split_rule": "catalog position within a family, never content and never item",
    "question_types": list(QUESTION_TYPES),
    "families": list(FAMILIES),
    "jitter": ("free fields (changed_lines, candidate_segments, scoring_budget, token_savings) "
               "are varied deterministically by small bounded steps per source group; jitter is "
               "rejected if it would change the base state's computed answers, and mutated fact "
               "leaves are never jittered, so the one-fact invariant holds leaf-for-leaf"),
    "views": {
        "items/<split>.jsonl": ("audit view: one object per (member x question type) with the "
                                "full request, declared gold, distribution, contrast and "
                                "provenance"),
        "trainer_view/<split>.jsonl": ("row view in the scripts/train_pipeline_decisions.py "
                                       "contract, generated from the same items"),
    },
    "provenance_field": ("every item records source_id, rule_id, the human-readable rule, "
                         "why_correct, the full fact basis and the authoring mode"),
}


def _family_counts(items, pairs):
    counts = {}
    for family in FAMILIES:
        family_items = [item for item in items if item["family"] == family]
        family_pairs = [pair for pair in pairs if pair["family"] == family]
        counts[family] = {
            "items": len(family_items),
            "pairs": len(family_pairs),
            "boolean": sum(1 for item in family_items if item["question_type"] == "boolean"),
            "choice": sum(1 for item in family_items if item["question_type"] == "choice"),
            "score": sum(1 for item in family_items if item["question_type"] == "score"),
            "splits": {split: sum(1 for item in family_items if item["split"] == split)
                       for split in SPLITS},
        }
    return counts


def derive_manifest(seed=DEFAULT_SEED):
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    _validate_tables()
    all_pairs = assign_splits(pairs_for(seed), seed)
    items = []
    for pair in all_pairs:
        items.extend(make_items(pair, "base"))
        items.extend(make_items(pair, "variant"))
    for item in items:
        guard_source(item["provenance"]["source_id"], item, f"item {item['item_id']}")
    pair_records = []
    for pair in all_pairs:
        pair_records.append({
            "pair_id": pair["pair_id"], "rule_id": pair["rule_id"],
            "state_id": pair["state_id"], "family": pair["family"],
            "feature": pair["feature"], "split": pair["split"],
            "source_group_id": pair["source_group_id"],
            "mutated_fact": pair["mutated_fact"], "value_before": pair["value_before"],
            "value_after": pair["value_after"],
            "flip_question_types": list(pair["flip_question_types"]),
            "gold_before": pair["gold_before"], "gold_after": pair["gold_after"],
            "item_ids": [item["item_id"] for item in items if item["pair_id"] == pair["pair_id"]],
            "declared_flip": list(pair["declared_flip"]),
        })
    split_geometry = {}
    for split in SPLITS:
        split_items = [item for item in items if item["split"] == split]
        split_geometry[split] = {
            "items": len(split_items),
            "pairs": sum(1 for pair in all_pairs if pair["split"] == split),
            "source_groups": sorted({item["source_group_id"] for item in split_items}),
            "families": sorted({item["family"] for item in split_items}),
            "question_types": sorted({item["question_type"] for item in split_items}),
        }
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "training_data_prerequisite_no_training_authorized",
        "corpus_role": "training_data_prerequisite_only",
        "created_by": "scripts/build_engineering_corpus_v1.py",
        "catalog_version": CATALOG_VERSION,
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": seed,
        "contract": {
            "validator": "scripts/predict_toy_decisions.py:validate_request",
            "shape": ('{"states":[{"id","state","questions":'
                      '{qid:{"type","instructions","criteria"}}}]}'),
            "trainer_row_contract": f"{TRAINER}:validate_training_row",
            "views": {"item": f"{ITEM_DIR}/<split>.jsonl",
                      "trainer": f"{TRAINER_DIR}/<split>.jsonl"},
            "validated": True,
        },
        "construction_rule": deepcopy(CONSTRUCTION_RULE),
        "provenance": dict(PROVENANCE),
        "exclusions": {
            "policy": ("The builder refuses any source path naming an evaluation corpus or the "
                       "pre-registered abstention survey, and refuses to emit any item whose "
                       "content carries a reserved evaluation-corpus token. Evaluation cohorts "
                       "are never a training, validation, calibration or selection source."),
            "refused_path_marker_count": len(FORBIDDEN_PATH_MARKERS),
            "refused_path_markers_sha256": digest_value(list(FORBIDDEN_PATH_MARKERS)),
            "reserved_token_count": len(RESERVED_TOKENS),
            "reserved_tokens_sha256": digest_value(list(RESERVED_TOKENS)),
            "refused_sources": [
                "dataset/games_v4/data/local_maze_v1/test.jsonl",
                "dataset/games_v4/data/local_maze_v1/ood.jsonl",
                "the context-relevance-v1 protocol corpus (test/OOD splits)",
                "the workflow-v2 evaluation cohort and its baseline manifest",
                "the pre-registered abstention survey and its gated run",
            ],
            "uses_evaluation_corpus_as_source": False,
        },
        "source_group_count": len({pair["source_group_id"] for pair in all_pairs}),
        "pair_count": len(all_pairs),
        "item_count": len(items),
        "counts_by_family": _family_counts(items, all_pairs),
        "counts_by_split": {split: split_geometry[split]["items"] for split in SPLITS},
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
    manifest["content_sha256"] = digest_value({key: value for key, value in manifest.items()
                                               if key not in ("content_sha256", "builder_sha256")})
    return manifest


# --------------------------------------------------------------------------------------
# trainer view
# --------------------------------------------------------------------------------------

TRAINER_GOLD_PROBS_KIND = "deterministic_truth"
TRAINER_GOLD_LABEL_KIND = "deterministic_truth"


def trainer_row(item):
    """One row in the ``scripts/train_pipeline_decisions.py`` record contract."""
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
    }


def trainer_rows(manifest):
    return [trainer_row(item) for item in manifest["items"]]


def load_trainer_validator():
    """Load the real trainer module for its validator (its torch import is lazy)."""
    import importlib.util
    import sys

    path = Path(__file__).with_name("train_pipeline_decisions.py")
    module_name = "engineering_corpus_trainer_validator"
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load the trainer validator")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return sys.modules[module_name].validate_training_row


def validate_trainer_row(row):
    return load_trainer_validator()(row)


# --------------------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------------------

def _first_difference(left, right, path=""):
    if type(left) is not type(right):
        return f"{path or '/'}: {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, dict):
        if set(left) != set(right):
            return f"{path or '/'}: keys differ ({sorted(set(left) ^ set(right))})"
        for key in sorted(left):
            difference = _first_difference(left[key], right[key], f"{path}/{key}")
            if difference:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path or '/'}: length {len(left)} != {len(right)}"
        for index, (first, second) in enumerate(zip(left, right)):
            difference = _first_difference(first, second, f"{path}/{index}")
            if difference:
                return difference
        return None
    if left != right:
        return f"{path or '/'}: {left!r} != {right!r}"
    return None


def validate_item(item):
    """Return contract/rule errors for one item; empty means sound."""
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
    if expected["distribution_kind"] != "hard_label" or expected["calibrated"] is not False:
        errors.append(f"{item_id}: distribution must stay an uncalibrated hard label")
    recomputed = gold_for(item["family"], item["provenance"]["fact_basis"], item["question_type"])
    if recomputed[0] != qid or recomputed[1] != expected["gold"]:
        errors.append(f"{item_id}: the declared gold is not what the family rule computes "
                      f"({recomputed[0]}={recomputed[1]!r})")
    hits = reserved_hits(json.dumps(item, ensure_ascii=False, sort_keys=True))
    if hits:
        errors.append(f"{item_id}: reserved evaluation tokens {hits}")
    if forbidden_path_reason(item["provenance"]["source_id"]):
        errors.append(f"{item_id}: refused evaluation source")
    if item["provenance"]["derived_from_evaluation_corpus"] is not False:
        errors.append(f"{item_id}: derived_from_evaluation_corpus must be false")
    if item["provenance"]["training_authorized"] is not False:
        errors.append(f"{item_id}: training_authorized must be false")
    if item["item_content_sha256"] != digest_value(
            {key: value for key, value in item.items() if key != "item_content_sha256"}):
        errors.append(f"{item_id}: item_content_sha256 does not match the item")
    return errors


def validate_manifest(manifest):
    """Re-derive the corpus from the seed and fail on any hand edit."""
    errors = []
    if not isinstance(manifest, dict):
        return ["manifest must be a JSON object"]
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        errors.append(f"schema_version must be {MANIFEST_SCHEMA!r}")
    if manifest.get("created_by") != "scripts/build_engineering_corpus_v1.py":
        errors.append("created_by must name this builder")
    if manifest.get("status") != "training_data_prerequisite_no_training_authorized":
        errors.append("status must record that no training is authorized")
    if manifest.get("corpus_role") != "training_data_prerequisite_only":
        errors.append("corpus_role must be training_data_prerequisite_only")
    if manifest.get("catalog_version") != CATALOG_VERSION:
        errors.append("catalog_version must match the builder")
    for flag, value in PROVENANCE.items():
        if isinstance(value, bool) and manifest.get("provenance", {}).get(flag) is not value:
            errors.append(f"provenance.{flag} must be {value}")
    if manifest.get("construction_rule") != CONSTRUCTION_RULE:
        errors.append("construction_rule must be the frozen rule")
    if manifest.get("contract", {}).get("validator") != \
            "scripts/predict_toy_decisions.py:validate_request":
        errors.append("contract.validator must name the real served validator")
    exclusions = manifest.get("exclusions", {})
    if exclusions.get("refused_path_marker_count") != len(FORBIDDEN_PATH_MARKERS):
        errors.append("exclusions.refused_path_marker_count must match the builder")
    if exclusions.get("refused_path_markers_sha256") != digest_value(list(FORBIDDEN_PATH_MARKERS)):
        errors.append("exclusions.refused_path_markers_sha256 must match the builder")
    if exclusions.get("reserved_token_count") != len(RESERVED_TOKENS):
        errors.append("exclusions.reserved_token_count must match the builder")
    if exclusions.get("reserved_tokens_sha256") != digest_value(list(RESERVED_TOKENS)):
        errors.append("exclusions.reserved_tokens_sha256 must match the builder")
    if exclusions.get("uses_evaluation_corpus_as_source") is not False:
        errors.append("exclusions.uses_evaluation_corpus_as_source must be false")
    if type(manifest.get("seed")) is not int:
        errors.append("seed must be an integer")
        return errors

    try:
        rederived = derive_manifest(manifest["seed"])
    except Exception as error:  # defensive: a corrupt manifest must not crash validation
        errors.append(f"re-derivation raised {type(error).__name__}: {error}")
        return errors

    difference = _first_difference(manifest.get("items"), rederived["items"], "/items")
    if difference:
        errors.append(f"declared items do not match the frozen rule: {difference}")
    difference = _first_difference(manifest.get("pairs"), rederived["pairs"], "/pairs")
    if difference:
        errors.append(f"declared pairs do not match the frozen rule: {difference}")
    for key in ("source_group_count", "pair_count", "item_count", "counts_by_family",
                "counts_by_split", "counts_by_question_type", "split_geometry", "item_digest",
                "pair_digest", "content_sha256"):
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
    seen_ids, groups, states = set(), {}, {}
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
        errors.extend(validate_item(item))
    pair_ids = {item["pair_id"] for item in items}
    if len(pair_ids) != manifest.get("pair_count"):
        errors.append("pair_count must equal the number of distinct pair ids")
    for pair_id in sorted(pair_ids):
        pair_items = [item for item in items if item["pair_id"] == pair_id]
        bases = [item for item in pair_items if item["member"] == "base"]
        variants = [item for item in pair_items if item["member"] == "variant"]
        if len(bases) != len(QUESTION_TYPES) or len(variants) != len(QUESTION_TYPES):
            errors.append(f"pair {pair_id!r} must carry all {len(QUESTION_TYPES)} question "
                          f"types for both members")
            continue
        if len({item["source_group_id"] for item in pair_items}) != 1:
            errors.append(f"pair {pair_id!r} must share exactly one source group")
        if len({item["split"] for item in pair_items}) != 1:
            errors.append(f"pair {pair_id!r} must not cross splits")
        delta = fact_delta(bases[0]["provenance"]["fact_basis"],
                           variants[0]["provenance"]["fact_basis"])
        if len(delta) != 1:
            errors.append(f"pair {pair_id!r} differs in {len(delta)} fact leaves, not 1")
        elif delta[0][0] != bases[0]["contrastive"]["mutated_fact"]:
            errors.append(f"pair {pair_id!r} mutated fact does not match the observed diff")
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
            errors.append(f"pair {pair_id!r} declares a flip that does not hold for "
                          f"{sorted(declared - flipped)}")
    hits = reserved_hits(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    if hits:
        errors.append(f"manifest contains reserved evaluation tokens {hits}")
    return errors


# --------------------------------------------------------------------------------------
# build / check / self-test / CLI
# --------------------------------------------------------------------------------------

def _jsonl(rows):
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
                   for row in rows).encode("utf-8")


def pretty(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            + "\n").encode("utf-8")


def output_files(manifest):
    """Both views plus the manifest, as relative path -> bytes."""
    files = {f"{ITEM_DIR}/{SPLIT_FILES[split]}":
             _jsonl([item for item in manifest["items"] if item["split"] == split])
             for split in SPLITS}
    rows = trainer_rows(manifest)
    files.update({f"{TRAINER_DIR}/{SPLIT_FILES[split]}":
                  _jsonl([row for row in rows if row["split"] == split])
                  for split in SPLITS})
    files[MANIFEST_NAME] = pretty(manifest)
    return files


def build(output_dir, seed=DEFAULT_SEED):
    """Derive, validate and materialize both views into an empty directory."""
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


def load_manifest(output_dir):
    return json.loads((Path(output_dir) / MANIFEST_NAME).read_text(encoding="utf-8"))


def check(output_dir):
    """Re-derive the corpus and compare it against every materialized file."""
    try:
        manifest = load_manifest(output_dir)
    except Exception as error:
        return [f"cannot read {MANIFEST_NAME}: {type(error).__name__}: {error}"]
    errors = validate_manifest(manifest)
    try:
        expected = output_files(manifest)
    except Exception as error:
        # A malformed manifest cannot address its own views; report that instead of raising.
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
    """Validate the frozen rule end to end without writing any output."""
    try:
        manifest = derive_manifest(seed)
        errors = validate_manifest(manifest)
        difference = _first_difference(manifest, derive_manifest(seed), "/")
        if difference:
            errors.append(f"a second in-process derivation differs: {difference}")
        if output_files(manifest)[MANIFEST_NAME] != pretty(manifest):
            errors.append("the manifest view is not stable")
    except Exception as error:
        manifest, errors = None, [f"{type(error).__name__}: {error}"]
    return {
        "schema_version": MANIFEST_SCHEMA,
        "mode": "self-test",
        "status": "failed" if errors else "ok",
        "seed": seed,
        "builder": "scripts/build_engineering_corpus_v1.py",
        "source_groups": manifest["source_group_count"] if manifest else 0,
        "pairs": manifest["pair_count"] if manifest else 0,
        "items": manifest["item_count"] if manifest else 0,
        "items_by_family": {family: manifest["counts_by_family"][family]["items"]
                            for family in FAMILIES} if manifest else {},
        "items_by_split": dict(manifest["counts_by_split"]) if manifest else {},
        "items_by_question_type": dict(manifest["counts_by_question_type"]) if manifest else {},
        "training_performed": False,
        "wrote_output": False,
        "errors": errors,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, help="empty directory for the corpus")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--self-test", action="store_true",
                        help="validate the frozen rule without writing output")
    parser.add_argument("--check", type=Path, help="re-derive and verify an existing corpus")
    parser.add_argument("--print-tables", action="store_true",
                        help="print the frozen catalog summary and exit")
    args = parser.parse_args(argv)
    if args.print_tables:
        print(json.dumps({
            "catalog_version": CATALOG_VERSION,
            "source_groups": len(CATALOG),
            "rules": [{"rule_id": rule["rule_id"], "family": rule["family"],
                       "flip": list(rule["flip"]),
                       "mutations": [name for name, _ in rule["mutations"]]}
                      for rule in CATALOG],
        }, indent=2, sort_keys=True))
        return 0
    if args.self_test and (args.output_dir is not None or args.check is not None):
        parser.error("--self-test is mutually exclusive with --output-dir/--check")
    if not args.self_test and args.output_dir is None and args.check is None:
        parser.error("one of --self-test, --output-dir or --check is required")
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
        print(json.dumps({"status": "failed", "error": str(error)}, indent=2, sort_keys=True))
        return 2
    print(json.dumps({
        "status": "ok",
        "output_dir": str(args.output_dir),
        "seed": manifest["seed"],
        "source_groups": manifest["source_group_count"],
        "pairs": manifest["pair_count"],
        "items": manifest["item_count"],
        "items_by_family": {family: manifest["counts_by_family"][family]["items"]
                            for family in FAMILIES},
        "items_by_split": dict(manifest["counts_by_split"]),
        "items_by_question_type": dict(manifest["counts_by_question_type"]),
        "content_sha256": manifest["content_sha256"],
        "training_performed": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
