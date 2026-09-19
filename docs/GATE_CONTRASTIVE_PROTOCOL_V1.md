# Gate contrastive protocol V1 (pre-registered draft)

Status: **pre-registered protocol draft. Nothing has been trained. This protocol requires
independent review BEFORE any training run.** It authorises no training, no checkpoint
change, no threshold change, no active context filtering, and no token-saving claim.

This document specifies work package **A2/A4 contrastive curation** for the context gate. It
is the protocol half of the pair whose data half is
`scripts/build_gate_contrastive_v1.py`. It follows the published recipe of
`bespokelabsai/nimble` as reviewed in
[JEV_COMMUNITY_REFERENCES.md](JEV_COMMUNITY_REFERENCES.md) ("Track A recipe and serving:
Bespoke Nimble"): *change one fact so the correct answer flips; the negative examples force
discrimination*. It maps that recipe onto the gate-training problem described in
[WORKFLOW_V2_BASELINE.md](WORKFLOW_V2_BASELINE.md) and
[NANOJEV_SKILL_READINESS_V1.md](NANOJEV_SKILL_READINESS_V1.md).

The gate's current behaviour is **correct abstention**: it proposes zero removals because it
cannot discriminate a distractor segment from a load-bearing one. Under irrelevant archived
context, Catalog Choice accuracy collapses from 100.00% to 54.17%. Contrastive one-fact pairs
are the pre-registered training signal for exactly that discrimination. This protocol defines
what would count as success *before* any such training exists.

## 1. Pre-registration statement

1. The construction rule, split geometry, endpoints, and failure criteria below are frozen at
   the moment this document is reviewed. Any change after review is a new protocol version and
   must be re-reviewed.
2. No training run may start until an independent reviewer (not the author, not the training
   executor) has reviewed this protocol, the builder, and the builder's test suite, and has
   recorded a PASS with any required amendments.
3. The held-out evaluation cohorts named here are evaluation-only. They may be *evaluated on*;
   they may never be used as training, validation, calibration, or early-stopping data. The
   builder enforces this by refusing excluded paths and by refusing reserved evaluation-corpus
   tokens in its inputs and outputs.
4. All comparisons are retained and reported, including flat and unfavourable ones. No
   best-of-three reporting and no post-hoc endpoint selection.

## 2. Hypothesis

* **H1 (discrimination).** Training on one-fact contrastive pairs raises the gate's ability to
  retain load-bearing context and drop genuinely reducible context, measured as pair
  discrimination on a held-out source-group-disjoint contrastive cohort.
* **H2 (downstream recovery).** The recovered discrimination restores downstream task success
  under irrelevant archived context, in particular Catalog Choice (`match`) accuracy from
  54.17% back toward the 100.00% no-archive baseline, without regressing the other frozen
  baseline rows.
* **H3 (calibration).** Gate confidence becomes positively associated with correctness on
  held-out engineering questions, so that abstention is a real uncertainty signal rather than
  a constant.
* **H0 (null).** Contrastive training does not move pair discrimination above the frozen
  baseline, or moves it while destroying protected retention/downstream success. H0 is the
  default; the experiment must beat it on evidence.

The measured reason for the current abstention, quoted from the baseline:
Catalog Choice test accuracy 100.00% (original) → 54.17% (archived distractor) → 58.33%
(structured envelope with archive). The gate cannot tell which archived segment matters, so
abstaining is the correct current behaviour.

## 3. Frozen contrastive construction rule

Rule version: `gate-contrastive-rule-v1`, implemented in
`scripts/build_gate_contrastive_v1.py` and covered by `scripts/test_gate_contrastive_v1.py`.

A **pair** is two request bodies that differ in **exactly one fact leaf of exactly one
record**. The pair's two members must receive opposite correct gate decisions (`keep` vs
`remove`). The rule is:

1. **World.** Four synthetic records in chronological order about two synthetic items, plus a
   user question asking for one field of one item. The oracle is last-value-wins: the answer is
   the chronologically last record matching the queried item and field.
2. **Candidate.** One record is the candidate segment (the segment the gate is asked about).
3. **Mutation.** Exactly one leaf of one record is changed, from the base member's value to the
   other member's value, and the two members are otherwise byte-identical outside that fact.
   Three mechanisms are frozen:
   * `supersession` — mutate the field label of a later record, so the candidate either is or
     is not the last value for the queried field;
   * `entity_binding` — mutate the candidate record's `entity`, so it either is or is not about
     the queried item;
   * `field_binding` — mutate the candidate record's `field`, so it either does or does not
     carry the queried field.
4. **Flip.** A pair is only emitted when the removal oracle computes different decisions for
   the two members. Both directions are emitted: `keep_to_remove` and `remove_to_keep`, so a
   model cannot learn "the mutated member is always the droppable one".
5. **Verification.** Every declared decision is re-derived from the materialized bytes and
   re-verified through the real, unchanged byte-preserving core `scripts/context_gate_v1.py`
   via `shadow_request` with a fixed synthetic oracle stub scorer (`threshold` frozen at 0.99).
   The stub is a deterministic stand-in for the gate checkpoint, never a model call.
6. **Record of change.** Every pair records `fact_path`, `fact_field`, `value_before`,
   `value_after`, `from_member`, `to_member` and `direction`; every item records a content
   hash of its materialized body, sidecar, and expected decision.
7. **Manifest.** The builder emits a machine-readable `contrastive_manifest.json` containing
   the frozen rule, the embedded base world set, split geometry, per-pair and per-item content
   hashes, and provenance flags. `validate_manifest` re-derives the whole cohort from the
   frozen rule and fails on any hand edit that contradicts it.
8. **Refusal.** The builder never reads, imports, or derives from the existing relevance
   test/OOD corpus or the workflow V2 evaluation cohort. It refuses a base fixture located in
   an excluded directory (`data/`, `results/`, `research/`), a path naming a relevance/OOD/
   challenge cohort, and any input or output containing a reserved evaluation-corpus token. It
   also refuses to write output into those locations.
9. **Determinism and isolation.** Given a seed the cohort is byte-identical on every rebuild.
   A different seed produces a fresh cohort whose source groups are disjoint from every other
   seed's groups; the frozen skeleton (pair ids, mechanisms, splits, directions) is unchanged.
   This is the mechanism for producing a held-out cohort that shares no source group with
   training data.

Default cohort: **12 pairs (24 items)** across 3 mechanisms and 3 wire formats
(`openai_chat`, `openai_responses`, `anthropic_messages`). Reproduce with:

```bash
.venv/bin/python scripts/build_gate_contrastive_v1.py --self-test
.venv/bin/python scripts/build_gate_contrastive_v1.py --output-dir /tmp/gate-contrastive-train --seed 20260919
.venv/bin/python scripts/build_gate_contrastive_v1.py --output-dir /tmp/gate-contrastive-heldout --seed 20260920
.venv/bin/python -m unittest discover -s scripts -p 'test_gate_contrastive*.py'
```

Measured at draft time (seed 20260919): `--self-test` reports `status: ok`, 12 pairs,
24 items, directions `keep_to_remove` and `remove_to_keep`; the test suite reports
**15 tests, OK**; the full repository suite reports **411 tests, OK, 2 skips**.

## 4. Split geometry

Source groups, not items, are the isolation unit. A pair's two members always share one source
group and one split; a pair is never split across train/dev/calibration/test.

| Split | Pairs (default cohort) | Purpose | Touched |
|---|---:|---|---|
| `train` | 6 | fitting | every run |
| `dev` | 2 | debugging / shape checks | per run |
| `calibration` | 2 | any threshold or abstention fitting | calibration only |
| `test` | 2 | the reported contrastive endpoint | once, after freezing |

* **Separation.** `source_group_id` is a hash over the semantic world (entities, fields,
  values, mechanism) and never over a split name, direction label, or file order. A group
  appears in exactly one split. The test split shares no source group with train/dev/calibration.
* **Mechanism coverage.** `train` contains all three mechanisms; `test` contains at least two.
  No split is a single-mechanism proxy.
* **Held-out cohort.** The primary pair endpoint is measured on a cohort built with a
  *different* seed from the training cohort, so no source group is shared (the builder's test
  suite asserts disjointness across seeds). The held-out cohort is generated once, frozen by
  hash, and evaluated once.
* **Scale-up for a real run.** If a training run needs more than 12 pairs, extend the frozen
  world table through `--base-fixture` **before** review, keep at least 20% of source groups in
  `test`, keep every pair intact inside one split, and re-freeze the manifest hashes.
* **Evaluation-only cohorts.** The relevance test/OOD corpus and the workflow V2 test/OOD
  cohort are used only as downstream/robustness *evaluation* sets (Section 5, M2/M3). They are
  never a training source, and no threshold may be selected on them.

## 5. Metrics that would count as success

The primary endpoint is **M1**. M2–M5 are secondary and cannot rescue a failed M1. Any
threshold fitted anywhere is fitted on `calibration` only.

**M1 — Dependency-pair retention (primary).** On the held-out contrastive cohort, for each
pair the `keep` member's load-bearing segment must be retained **and** the `remove` member's
segment must be dropped. The held-out cohort's pairs are *all* evaluation pairs; its own
train/dev/calibration labels are irrelevant and are never used for fitting. Success requires:
* pair-discrimination accuracy ≥ **0.95**, with the lower bound of a 95% source-group cluster
  bootstrap CI **> 0.50**;
* per-mechanism accuracy ≥ **0.85** for each of `supersession`, `entity_binding`,
  `field_binding` (no mechanism collapse).

For the default 12-pair cohort the bootstrap interval is descriptive; the 0.95 point estimate
is the binding bar. A scale-up cohort must pre-register a tighter CI bar before it is run.

**M2 — Protected-segment deletions must be 0 (hard gate).** Across every evaluation run
(held-out contrastive cohort, workflow V2 test/OOD, relevance test/OOD, and the tool-history
shadow fixtures), the number of protected segments proposed for removal and the number
actually removed must both be exactly **0**. A single protected deletion fails the run
immediately, regardless of every other metric.

**M3 — Downstream task success.** Paired evaluation on the frozen workflow V2 test cohort in
shadow mode, comparing the candidate gate against the current zero-drop checkpoint:
* Catalog Choice (`match`) accuracy under the **archived-distractor** variant ≥ **0.90**
  (baseline 0.5417; original 1.0000);
* no row of the four-row baseline table (original / archived distractor / structured envelope
  with archive / candidate order reversed) regresses by more than **3 percentage points**
  against the current checkpoint;
* Smart-home Boolean OOD accuracy under the archive does not fall more than **5 percentage
  points** below the current checkpoint;
* known-chance total-variation distance does not worsen by more than **0.05**.

**M4 — Accuracy-versus-confidence correlation.** On held-out engineering questions (workflow V2
test/OOD, using the frozen question set, never used for fitting), the Spearman correlation
between the gate's confidence and decision correctness must be **> 0**, with a 95%
source-group bootstrap CI excluding zero; and the count of confident-wrong decisions must not
exceed the current checkpoint's. Confidence must be defined from the gate's own readout before
the run and recorded in the receipt; choosing the confidence definition after seeing results
invalidates M4.

**M5 — Token savings are real, not assumed.** Mean proposed removable text tokens per eligible
request must be **> 0** on the evaluation cohorts, and the abstention/fail-open rate must not
exceed the current checkpoint's by more than **10 percentage points**. Savings only count
towards a claim when M1–M4 all hold. Proposed-token counts are `isolated_segment_text_only`
as the receipt declares; they are not provider billing evidence.

**Reporting standard.** Report all three matched training seeds of the same recipe (for example
17, 18, 19) and the **minimum** result across seeds, not the best; report per-seed M1–M5;
retain flat and unfavourable arms; report the exact builder, manifest, checkpoint, and
evaluator hashes. No multiplicity-corrected discovery claim is made from the descriptive
comparisons in this protocol.

## 6. Failure criteria

The experiment **fails** — stop, do not enable anything, and keep abstention as the default —
if any of the following hold:

1. **Any protected-segment deletion** is proposed or applied (M2 ≠ 0). Immediate stop.
2. Pair-discrimination accuracy < 0.90, or any single mechanism < 0.70 (M1).
3. Catalog Choice under the archived distractor stays < 0.70, or any baseline row regresses by
   more than 5 percentage points (M3).
4. Spearman accuracy–confidence correlation ≤ 0, or its 95% CI includes 0 (M4).
5. Confident-wrong decisions increase relative to the current checkpoint (M4).
6. The gate still proposes **zero** removals on held-out eligible segments (savings remain 0):
   this is a documented null result, not a success, and must be reported as such.
7. Any evidence of evaluation-cohort reuse: training, validating, calibrating, or selecting on
   the relevance test/OOD corpus, the workflow test/OOD cohort, or the held-out contrastive
   cohort.

A failed criterion is reported as a failure of the hypothesis, not reframed as a partial
success. A protocol that cannot be executed as written (for example the builder refusing an
input) is a blocker to escalate, not a licence to relax a criterion.

## 7. Pre-registered training configuration (not authorised by this document)

For completeness, the recipe this protocol pre-registers — modelled on the published Bespoke
Nimble recipe, adapted to the local checkpoint and the existing 255-candidate head contract —
is: LoRA rank 16, learning rate 5e-5, effective batch 8, seed 17 (plus matched 18 and 19),
one epoch, BF16, cross-entropy over the allowed answer tokens only, 2,048-token prompt cap.
The upstream 26-choice cap and normalised-probability readout are **not** adopted; the local
head contract is unchanged.

This configuration is written down so it cannot be tuned after seeing results. It is **not
authorised** by this document: no training run may start before independent review (Section 8),
and no production default, checkpoint, threshold, or kill-switch behaviour may change as part
of it.

## 8. Review gate (required before any training run)

Independent review must confirm, in writing and with a recorded PASS:

1. This protocol's endpoints, split geometry, and failure criteria are pre-registered and
   unambiguous, and the primary endpoint is M1.
2. The builder implements the construction rule as written: one-fact-only mutation verified
   structurally, decision flip verified through the real core, deterministic rebuild,
   per-item content hashes, manifest re-derivation on validation.
3. The no-leakage claims hold: no relevance test/OOD or workflow evaluation record can enter
   the training stream through the builder, its base fixtures, or its outputs.
4. The held-out cohort (different seed) really shares no source group with the training cohort.
5. The training configuration in Section 7, the calibration-only threshold policy, and the
   three-seed reporting plan are acceptable.
6. The evaluator and receipt plan can produce M2–M5 without touching the frozen cohorts for
   fitting.

Until that review exists, the correct state is: fixture builders and tests only, shadow mode,
zero removals, no token savings claimed.

## 9. What this protocol explicitly does NOT authorise

* No training, fine-tuning, LoRA, distillation, or checkpoint modification.
* No change to the production checkpoint, the frozen 0.99 shadow threshold, or any serving
  default; active context filtering stays disabled.
* No use of the relevance test/OOD corpus or the workflow V2 evaluation cohort as a training,
  validation, calibration, or early-stopping source.
* No network access, provider proxy change, or external model call as part of the builder.
* No token-saving, latency, quality, financial, or production-readiness claim. Zero savings
  remain the measured state until M1–M5 pass under review.
* No financial training, live trading, or third-party data-licence risk acceptance.

## 10. Limitations

1. The default cohort is a **skeleton** (12 pairs / 24 items), not a curriculum. It pins the
   construction rule and the protocol; a real run needs a larger reviewed world table.
2. The frozen oracle is a last-value-wins lookup. It defines load-bearing membership for this
   corpus only; contrastive training against it can overfit that definition and teach nothing
   about general semantic relevance. M3/M4 exist to detect exactly that.
3. The synthetic oracle stub scorer is **not** the gate model. It verifies the fixture contract,
   not model behaviour; no accuracy number in this document is a model result.
4. Labels are programmatic and **not human-reviewed** — the same disclosure the upstream recipe
   makes about its model-checked labels.
5. Triples of records per item and three mechanisms do not cover tool results, credentials,
   corrections, or long-document distractors; those live in other work packages and must not be
   inferred to work from this one.
6. The builder cannot add capability the head cannot express. If the gate's training objective
   cannot represent "this specific record is superseded", M1 will fail; that is a finding, and
   the fallback remains abstention.

## 11. Reproduction and evidence

```bash
# Rule and cohort integrity only; writes nothing.
.venv/bin/python scripts/build_gate_contrastive_v1.py --self-test

# Materialize a training cohort and a source-group-disjoint held-out cohort.
.venv/bin/python scripts/build_gate_contrastive_v1.py --output-dir /tmp/gate-contrastive-train --seed 20260919
.venv/bin/python scripts/build_gate_contrastive_v1.py --output-dir /tmp/gate-contrastive-heldout --seed 20260920

# The builder suite (determinism, one-fact-only diff, decision flip through the core,
# no leakage, manifest hash stability).
.venv/bin/python -m unittest discover -s scripts -p 'test_gate_contrastive*.py'

# Repository regression suite (expected at draft time: 411 tests, OK, 2 skips).
.venv/bin/python -m unittest discover -s scripts -p 'test_*.py'
```

Manifest layout (per output directory): `contrastive_manifest.json` plus
`pairs/<pair_id>/{keep,remove}/{request.json,sidecar.json,expected.json}`. The manifest records
the frozen rule, embedded base worlds, split geometry, per-pair mutation records, and per-item
content hashes; it re-derives and re-verifies its own contract on every validation.

See [WORKFLOW_V2_BASELINE.md](WORKFLOW_V2_BASELINE.md) for the measured baseline,
[TOOL_HISTORY_SHADOW_V1.md](TOOL_HISTORY_SHADOW_V1.md) and
[REVERSIBLE_FILTERING_V1.md](REVERSIBLE_FILTERING_V1.md) for the shadow and reversibility
contracts this data would feed, and
[JEV_COMMUNITY_REFERENCES.md](JEV_COMMUNITY_REFERENCES.md) for the upstream recipe and its
adoption boundary.
