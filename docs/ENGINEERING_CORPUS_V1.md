# Engineering-judgment corpus V1

**What this is.** A deterministic, contrastive, provenance-carrying engineering-judgment
training corpus in the exact served request contract of
`scripts/predict_toy_decisions.py`, emitted together with a second view in the row
contract of `scripts/train_pipeline_decisions.py`. It exists to remove the *data*
blocker identified by [SKILL_ABSTENTION_DIAGNOSIS_V1.md](SKILL_ABSTENTION_DIAGNOSIS_V1.md):
the checkpoint `local_atomic_seed17` abstains 13/13 on engineering-judgment questions at
the 0.9 cutoff because it has **no engineering-judgment training data at all**, while its
own maze family reaches 0.998 confidence and 97.8% accuracy at >= 0.9.

> **This corpus is a training-data prerequisite only. Producing it authorises no
> training.** Nothing here trains, fine-tunes, distils, evaluates, or modifies any
> checkpoint, threshold, serving default or production behaviour. Every item, every row
> and the manifest itself record `training_authorized_by_this_corpus: false`,
> `training_performed: false`, `checkpoint_read: false`, `network_access: false`. The
> independent review gate in
> [GATE_CONTRASTIVE_PROTOCOL_V1.md](GATE_CONTRASTIVE_PROTOCOL_V1.md) Section 8 remains
> untouched and unsatisfied by this work.

## Files

| Path | What it is |
|---|---|
| `research/engineering_judgment_corpus_v1/manifest.json` | The re-derivable manifest: frozen construction rule, exclusions, composition counts, all pair records, all items with per-item content hashes, `item_digest`, `pair_digest`, `content_sha256` |
| `research/engineering_judgment_corpus_v1/items/{train,dev,calibration,test}.jsonl` | The **item view** (audit view) |
| `research/engineering_judgment_corpus_v1/trainer_view/{train,dev,calibration,test}.jsonl` | The **trainer view** |
| `scripts/build_engineering_corpus_v1.py` | The builder/generator (`--self-test` writes nothing) |
| `scripts/test_engineering_corpus_v1.py` | The test suite that actually runs |

### The two views

Both views are generated from the same manifest, item for item; neither is a
hand-maintained copy.

* **Item view** — one JSON object per *(pair member x question type)*. It carries the
  full served `request`, the `expected` gold answer with an uncalibrated one-hot
  distribution, the `contrastive` record (pair id, member, mutated fact, before/after
  values, which question types flip), a full `provenance` record, and
  `item_content_sha256`. This is the audit and validation view.
* **Trainer view** — one JSON object per *(member x question)* with the keys
  `scripts/train_pipeline_decisions.py:validate_training_row` requires: `id`,
  `state_id`, `family_id`, `split`, `state`, `questions`, plus `gold`, `gold_probs`,
  `gold_probs_kind` and `gold_label_kind`. Gold is a hard one-hot label
  (`gold_probs_kind = gold_label_kind = "deterministic_truth"`). `source_group_id`,
  `pair_id`, `member`, `question_type`, `mutated_fact`, `is_flip_question` and the
  provenance summary ride along in `metadata`, which the trainer already reads for its
  source-group split guard. The trainer file is unmodified.

## Composition

Counts below are read from the emitted `manifest.json` (seed `20260919`).

| Family (feature) | Source groups | Contrastive pairs | boolean | choice | score | Items | train | dev | calibration | test |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lifecycle-phase selection | 9 | 9 | 18 | 18 | 18 | 54 | 30 | 6 | 6 | 12 |
| test-vs-skip selection | 5 | 5 | 10 | 10 | 10 | 30 | 12 | 6 | 6 | 6 |
| request routing under a scoring budget | 5 | 5 | 10 | 10 | 10 | 30 | 12 | 6 | 6 | 6 |
| failure classification | 7 | 7 | 14 | 14 | 14 | 42 | 18 | 6 | 6 | 12 |
| risk / authorization gating | 5 | 5 | 10 | 10 | 10 | 30 | 12 | 6 | 6 | 6 |
| context-retention decision | 5 | 5 | 10 | 10 | 10 | 30 | 12 | 6 | 6 | 6 |
| checkpoint-readiness scoring | 5 | 5 | 10 | 10 | 10 | 30 | 12 | 6 | 6 | 6 |
| **Total** | **41** | **41** | **82** | **82** | **82** | **246** | **108** | **42** | **42** | **54** |

By question type: **82 boolean / 82 choice / 82 score**. Every split carries all three
types, and the train:dev:calibration:test item ratio is 108:42:42:54.

**Contrastive-pair accounting (read this before quoting a pair count).** There are **41
contrastive pairs**. Each pair carries all three question types for both members, so
there are **123 pair-by-question-type contrastive instances** (82 of which are the
declared flip question; the rest are co-questions whose answers are computed from the
same facts but do not flip). "41 pairs" is the honest unit: 41 base states each paired
with a one-fact variant.

## How correctness provenance is recorded

There are no hand-written answer labels. Each source group is a hand-authored
**fact table** (a state in the served contract) plus a per-family **frozen rule**; the
expected answer is computed from the facts by that rule at build time. Every item
records:

* `provenance.source_id` — the concrete repository artefact or accepted rule that
  grounds the item (for example `docs/GATE_CONTRASTIVE_PROTOCOL_V1.md`, `AGENTS.md`,
  `scripts/predict_toy_decisions.py`, or a labelled fact-source reference);
* `provenance.rule_id` and `provenance.rule` — the frozen rule and a human-readable
  statement of it;
* `provenance.why_correct` — the completed sentence "under the frozen *family* rule the
  facts in this state fix *qid* = *gold*";
* `provenance.fact_basis` / `fact_keys` — the full enumerated fact assignment, so the
  answer is independently recomputable without trusting the stored gold;
* `provenance.authoring` — `hand_authored_state_and_rule_with_programmatic_rendering`
  for a base member, `programmatic_one_fact_mutation_of_a_hand_authored_state` for a
  variant member;
* `provenance.human_reviewed: false`, `derived_from_evaluation_corpus: false`,
  `training_authorized: false`.

`validate_item` recomputes the gold from `fact_basis` on every validation; a mismatch is
an error. The per-family rules are:

| Family | Rule (abbreviated) |
|---|---|
| lifecycle-phase | strict priority: unverified invariant -> `verification`; then unmeasured B0 -> `measurement`; then missing dataset -> `data`; then unfrozen builder or missing signature -> `signoff`; else `train`. Readiness 0-5 counts cleared prerequisites. |
| test-vs-skip | a new public entry point forces the full suite; otherwise a canary when only reachable modules can run; otherwise a shared module or a >=200-line diff forces the full suite; otherwise no behaviour change -> `skip`, else `targeted`. |
| request routing | within budget -> one gate pass; mergeable within 4x budget -> batched merge; otherwise bypass. Protected endpoints are never safe to drop from. |
| failure classification | changed fixture -> `data_bug`; else edited expectation -> `test_bug`; else a non-deterministic repeat -> `flaky`; else `code_bug`. Regression requires a user-facing path that previously passed. |
| risk / authorization | blocked prerequisite -> `pause`; else irreversible and unauthorised -> `ask_human`; else network-reaching and unauthorised -> `deny`; else `proceed`. |
| context retention | last-value-wins: load-bearing exactly when the candidate matches the queried item and field and no later record covers it; a matching but later-covered segment is the only fully safe drop. |
| checkpoint readiness | no proposed removals -> 0; any protected deletion -> 0; removals without savings or with invalid outputs -> 1; savings with ECE <= 0.1 and accuracy >= 0.75 -> 3; otherwise 2. |

## The contrastive rule and its proof

Frozen as `CONSTRUCTION_RULE` in the builder. A pair is two served request bodies whose
states differ in **exactly one fact leaf** of that family's enumerated fact set, and the
declared flip question type must actually change.

The proof is computed, not assumed. For every group, `pairs_for` derives the base facts
and each mutated fact set, then:

1. asserts the leaf diff is exactly the one declared fact (and that the value really
   changes);
2. **computes** `all_gold` for both members from their own facts with the family rule;
3. raises before anything is written if a declared flip does not actually flip, naming
   the question type, the before/after answers, and the mutated fact.

`validate_manifest` re-runs the same derivation from the emitted seed on every
validation and re-checks the flip per pair from the materialized items.

### Split geometry

The isolation unit is the **source group**, never the item. A pair's members always share
one `source_group_id`, one `state_id` and one split. Splits are assigned by catalog
position within a family (`SPLIT_CYCLE`), never by content, so editing a state's wording
cannot move an item across the train/test boundary. `validate_manifest` fails if any
`source_group_id` or `state_id` crosses splits, and the test suite re-checks that the
test split shares no source group with train/dev/calibration.

`source_group_id` is a hash over `{seed, family, pair_id}`; a different seed yields
disjoint source groups with an unchanged skeleton (asserted by the test suite).

## Refusals: no evaluation-corpus reuse

The builder refuses, by path marker, any source naming `local_maze_v1`/`games_v4`/
`test.jsonl`/`ood.jsonl`/`scaled_maze`, the context-relevance test/OOD corpus, the
workflow V2 evaluation cohort or baseline manifest, the tool-history shadow fixtures, the
NanoJev V2 baseline results, or the pre-registered abstention survey and its gated run
(`FORBIDDEN_PATH_MARKERS`, 26 markers). It additionally refuses to emit any item whose
content contains a reserved evaluation-corpus data token (`RESERVED_TOKENS`, 32 tokens:
family names, field names, split kinds, rendered values).

Three source groups take their *facts* from the publicly recorded numbers in
`research/skill_abstention_survey_v1.json` (the ~400-segment/32-candidate routing
geometry, the zero-removal checkpoint, the zero-invalid-output checkpoint). Those facts
are re-authored here in this corpus's own wording; the survey file is listed as a refused
source and no item is derived from it, which is why its `source_id` is recorded as
"abstention-survey-v1 (fact source only; file refused by the builder)" rather than as a
path. Policy words that also occur in the builder's own refusal bookkeeping (for example
`superseded`, a context-relevance split kind) are deliberately **not** in
`RESERVED_TOKENS`; those corpora remain covered by the path-marker refusals.

## Reproduce

```bash
# Validate the frozen rule end to end; writes nothing.
.venv/bin/python scripts/build_engineering_corpus_v1.py --self-test

# Materialize both views.
.venv/bin/python scripts/build_engineering_corpus_v1.py --output-dir research/engineering_judgment_corpus_v1

# Re-derive and compare every emitted file against the manifest.
.venv/bin/python scripts/build_engineering_corpus_v1.py --check research/engineering_judgment_corpus_v1

# Print the frozen catalog.
.venv/bin/python scripts/build_engineering_corpus_v1.py --print-tables

# The builder suite (25 tests).
.venv/bin/python -m unittest discover -s scripts -p 'test_engineering_corpus*.py'
```

Measured at emission (seed 20260919): `--self-test` reports `status: ok`, 41 source
groups, 41 pairs, 246 items; two builds are byte-identical; the committed corpus equals a
fresh build; `--check` reports `status: ok` with an empty error list;
`content_sha256 = d473135243d51707e296d75aa93ed1edd434e5f4be11989d5b89e0ecf12900be`.
The builder suite reports **25 tests, OK**.

## Authoring disclosure: hand-authored versus programmatic

Stated plainly, because it bounds what this corpus can be claimed to be:

* **Hand-authored**: 41 base fact tables (one per source group) and their 41 declared
  one-fact mutations, across 7 decision families — the catalog in
  `scripts/build_engineering_corpus_v1.py` (`CATALOG`, 41 `_rule` entries). Each carries a
  written justification (`why`) naming the fact and the project rule that makes the
  answer correct.
* **Programmatic**: the contract rendering (state prose, question bodies and criteria),
  the expected answers and one-hot distributions, the variant members produced by
  applying the declared mutation, the pair/flip verification, the split assignment, the
  deterministic jitter of free fields, both output views, and every hash. 205 of the 246
  items are variant-derived; all 246 have machine-computed labels.
* Numeric free fields (`changed_lines`, `candidate_segments`, `scoring_budget`,
  `token_savings`) are jittered deterministically per source group so no two groups are
  near-duplicates. A field that is exactly zero stays zero; the build rejects any jitter
  that would change a rule's answer.

## Limits (explicit)

1. **Scale is below the requested target.** The request asked for at least 400 items and
   120 contrastive pairs. This corpus delivers **246 items and 41 contrastive pairs**.
   The gap is deliberate and is not padded. Padding was available in three cheap forms —
   extra paraphrases of the same flip, extra mutations that do not flip the designated
   question, or randomly enumerated fact vectors — and all three would either inflate the
   pair count with near-duplicates or weaken the exact-one-fact-flip property that makes
   this corpus useful. Every one of the 41 pairs changes exactly one fact leaf and its
   declared answer really flips; that property was protected over the count. Reaching 120
   honest pairs needs roughly 120 hand-authored base states plus their marginal-fact
   mutations (see limit 2), which is a further authoring pass, not a bigger loop over the
   same table.
2. **Single-fact flips are scarce by construction.** A one-fact change only flips an
   answer when that fact is the *marginal* blocker in a priority rule. Most plausible
   mutations of a state are below the priority cut and change nothing; those were
   detected during development and are either declared as score-only/no-op sub-mutations
   with a written reason or dropped. This is why the hand-authored catalog is the
   bottleneck rather than the generator.
3. **Labels are not human-reviewed.** They are deterministic and re-derivable, but they
   encode *this project's* frozen rules, not an independent engineering ground truth. The
   upstream contrastive recipe makes the same disclosure. The rules themselves are
   contestable; where they are, the corpus teaches the contestable rule.
4. **Priority orderings are design decisions.** The lifecycle priority (invariant ->
   measurement -> data -> signoff -> train), the test-scope priority (new public API
   first), and the risk priority (blocked -> irreversible -> network) are chosen to match
   `GATE_CONTRASTIVE_PROTOCOL_V1.md` and `AGENTS.md` as written at emission. A rule
   change invalidates the affected items and requires re-emission; the manifest records
   `catalog_version` so this is detectable.
5. **Coverage is a bounded subset of engineering judgment.** One contract style, seven
   families, five to nine source groups each, short synthetic state renderings. Not
   covered: multi-turn or tool-history states, long-document distractors, credentials and
   corrections as distinct retention mechanisms, financial decisions, and any judgment
   requiring reading a real diff. Nothing may be inferred from this corpus about
   behaviour on those distributions.
6. **Two of the seven families expose fewer than three flip facets.** The
   request-routing and checkpoint-readiness groups have five groups each, so several
   families are thinner than `lifecycle_phase` (nine) or `failure_classification`
   (seven). Composition per split is balanced by position, not stratified per family, so
   split-by-family cells range from 6 to 30 items.
7. **No model behaviour is measured here.** This corpus makes no claim about the
   abstention rate, confidence, calibration, token savings, accuracy or readiness of any
   checkpoint. In particular, producing this corpus does **not** change the diagnosis's
   finding that the current checkpoint abstains 13/13 on the engineering survey; that
   remains the measured state until a reviewed training run exists.
8. **The reserved-token guard is content, not semantic.** It detects evaluation-corpus
   data tokens and refused sources; it cannot detect a paraphrase of an evaluation item.
   The three fact-source groups in limit-adjacent note above are the known, disclosed
   place where recorded public numbers are re-authored by hand.
9. **One seed, one catalog, one machine.** Determinism and disjointness across seeds are
   asserted by the test suite; nothing here is evidence about a different seed, a
   different catalog, or a different platform.
