# Valen Nano v3 — context_relevance_v3 expansion

**Status:** built and validated. `data/valen_nano_v3/` = all `valen_nano_v2`
sources + the newly generated `data/context_relevance_v3/`. Data dirs are
gitignored; this document, the generator, the builder and the tests are the
tracked artifacts. No training or deployment decisions are attached.

## Why v3

`valen_nano_v2` fixed the correction/overlap_distractor label contradiction
(see `VALEN_NANO_V2_CONTRACT_FIX_V1.md`) but kept the same narrow template
space: each family×kind is one template, so only ~48 distinct normalized
conversation shapes exist and eval saturates. v3 adds a new self-authored
programmatic source that (a) covers all 14 candidate kinds already present
across the two v2 sources, (b) adds two hard-negative families aimed at the
known correction/superseded error cluster, and (c) marks confusable records
with `meta.hard_negative` so future comparisons do not saturate.

## Artifacts

| artifact | role |
|---|---|
| `scripts/gen_context_relevance_v3_v1.py` | new generator; `--seed` (default 20261005), `--groups-per-family` (default 122), `--out` (default `data/context_relevance_v3`) |
| `data/context_relevance_v3/` | source dir: `train/dev/calibration.jsonl` + `manifest.json` (schema `nanojev-context-relevance-v3`) |
| `external/valen/scripts/build_nanojev_valen_v3.py` | merge builder → `data/valen_nano_v3/` (schema `nanojev-valen-nano-v3`) |
| `scripts/test_gen_context_relevance_v3_v1.py` | 16 unit tests |
| `scripts/validate_valen_label_consistency_v1.py` | existing validator; run against `data/valen_nano_v3` |

## Candidate-kind roster (16)

All 14 kinds reused with their existing semantics, plus 2 new families:

| kind | label | hard | source of semantics |
|---|---|---|---|
| required_evidence | keep | | context_relevance_v1 |
| user_constraint | keep | | context_relevance_v1 |
| tool_dependency | keep | | context_relevance_v1 |
| unrelated | drop | | both |
| stale_fact | drop | | context_relevance_v1 |
| wrong_entity | drop | yes | both |
| wrong_field | drop | yes | oracle_v1 |
| required_field | keep | | oracle_v1 |
| two_fields | keep | | oracle_v1 |
| limit_check | keep | | oracle_v1 |
| latest_correction | keep | yes | oracle_v1 |
| superseded | drop | yes | oracle_v1 |
| correction | keep **or** drop | yes | context_relevance_v1, split into two variants |
| overlap_distractor | keep **or** drop | yes | context_relevance_v1, split into two variants |
| **near_duplicate_evidence** | drop | yes | **new** |
| **correction_confirmed** | keep | yes | **new** |

### Variant rule for correction / overlap_distractor

Each `correction`/`overlap_distractor` record draws one variant per record
(`metadata.variant` records which):

- `current_in_state` → **drop**: the request states the current value, so the
  stale candidate is redundant (e.g. "…use port 8<i>N</i>…" vs "A stale note
  says the service once used port <old>").
- `sole_evidence` → **keep**: the request references the corrected value
  without stating it, so the stale candidate is the sole evidence
  ("answer false when uncertainty remains").

This is exactly the v2 contract applied at generation time, per record rather
than per family — the v1 defect (label attached to the declared arm instead of
the state) cannot recur because the label is emitted from the same branch that
chooses the request variant.

### New hard-negative families

- `near_duplicate_evidence` (drop): the candidate restates information already
  present in the state with trivial surface differences ("Restated request
  parameters: service code-0088, port 8176."). Under the contract the current
  value is in the state by construction (the `withval` request), so the
  restatement is certainly irrelevant.
- `correction_confirmed` (keep): the candidate documents a correction that was
  later re-confirmed ("Correction history for order-0042: quantity 5 -> 3; the
  customer later re-confirmed 3."). It resembles a stale-superseded record but
  carries the operative value — targets the correction/superseded error
  cluster directly.

## Domains and languages

Six domains, matching v1/v2: `code`, `order`, `risk`, `support`, `robotics`,
`multilingual` — each contributes 122 groups × 16 kinds. The multilingual
family mixes zh/en: ~70% of requests are Chinese, ~30% English, and ~35% of
zh-request records use an English candidate twin (plus `user_constraint`
suffixes in the other language and English tool messages); 777 records carry
`metadata.mixed_language: true`.

## Counts

`data/context_relevance_v3` (seed 20261005):

| split | records | hard_negative | mixed_language | groups |
|---|---:|---:|---:|---:|
| train | 8384 | 4192 | 605 | 524 |
| dev | 1664 | 832 | 86 | 104 |
| calibration | 1664 | 832 | 86 | 104 |
| **total** | **11712** | **5856** | **777** | **732** |

`data/valen_nano_v3` (merged, split rule `sha256(group_id) mod 100 < 20 -> eval`):

| | records | groups | labels (false/true) | hard_negative |
|---|---:|---:|---|---:|
| train | 12122 | 1732 | 6727 / 5395 | 4720 |
| eval | 2950 | 410 | 1631 / 1319 | 1136 (**38.5%**) |
| total | 15072 | — | — | — |

v3 contributes 9440 train / 2272 eval. Eval contains exactly 142 records of
each of the 16 kinds (groups are homogeneous and land whole in one split).

## Contract adherence

- Labels are a pure function of state content: every drop-arm that cites an
  outdated value is emitted only alongside a request that states the current
  value (`metadata.variant = "current_in_state"`); sole-evidence stale
  candidates keep `irrelevant=false`.
- `validate_valen_label_consistency_v1.py data/valen_nano_v3` →
  `{records: 15072, normalized_states: 4081, violations: 0}`.
- The unit test adds a stricter global check (one label per normalized state
  across *all* splits; the official validator is per-split): 0 conflicts.
- v3 normalized-state diversity: ~1550 distinct normalized states for 11712
  records (vs ~48 for the whole v1 source), via per-record parameter draws,
  variant branches, filler/request variants and multilingual language flips.

## Group isolation

Each v3 group = one parametrized scenario emitting all 16 kinds under a single
`source_group_id = sha256("context_relevance_v3:<seed>:<family>:<gidx>")`; the
builder prefixes `context_relevance_v3:` as lineage. Verified: train/eval group
intersection is empty on the built output.

## Determinism

`--seed` parametrizes a single `random.Random` instance; all draws happen in
fixed iteration order; records serialize with `ensure_ascii=False,
separators=(",", ":")`. Regenerating with the same seed is byte-identical
(covered by test and re-verified on the built dir).

## Known limitations

- v3 emits only `train/dev/calibration` — no frozen `test/ood` splits exist for
  the new source. If a future gate needs them, regenerate with a disjoint seed
  and mark them frozen before any selection use.
- Templates are still finite and self-authored synthetic; normalized
  near-duplicates spanning train/eval remain by design (same label — redundancy,
  not contradiction), so eval is partly a generalization-over-paraphrase test.
- `hard_negative` is a design-time flag (the kind is confusable), not a measured
  model-error label; hard fraction is 50% of v3 records, 38.5% of merged eval.
- `correction_confirmed`/`near_duplicate_evidence` rely on the v2 contract
  reading: a re-confirmed correction is evidence (keep); a pure restatement of
  already-present content is certainly irrelevant (drop). Both interpretations
  are asserted by tests but remain judgment calls of the synthetic protocol.
- v2/oracle records carry no `hard_negative` key (flag exists only on v3
  records) — evaluators should treat missing as not-hard.

## Reproduce

```bash
python3 scripts/gen_context_relevance_v3_v1.py            # -> data/context_relevance_v3
python3 external/valen/scripts/build_nanojev_valen_v3.py . # -> data/valen_nano_v3
python3 scripts/validate_valen_label_consistency_v1.py data/valen_nano_v3
python3 -m unittest scripts.test_gen_context_relevance_v3_v1
```
