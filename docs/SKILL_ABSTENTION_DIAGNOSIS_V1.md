# Skill abstention diagnosis V1 — where the confidence collapse happens

**Question.** The local NanoJev checkpoint `local_atomic_seed17` abstained 13/13 on the
engineering-judgement survey at the default 0.9 threshold. Is that because (a) the model is
genuinely uncertain everywhere, (b) it is uncertain only out-of-domain, or (c) confidence exists
but is not calibrated to correctness?

**Answer: (b).** The collapse is **out-of-domain**. Replayed through the *same* local service, the
same checkpoint produces a well-separated, positively-calibrated high-confidence mode on its own
maze task family — 37.5% of in-domain questions at confidence ≥ 0.9, and those are **97.8%**
correct — while all 13 natural-language engineering questions land in the compressed band
[0.255, 0.736] with median 0.505. The engineering survey sits entirely below the in-domain median,
and it is there — and only there — that the checkpoint loses its confidence.

Companion receipt: `results/skill_abstention_diagnosis_v1.json` (per-question confidences, exact
commands, event ids, threshold tables, provenance, capture hashes).

## Method

| Item | Value |
|---|---|
| Service | `python3 /Users/markus/.codex/skills/nanojev-local-decider/scripts/nanojev_skill.py health --start` |
| Service URL | `http://127.0.0.1:8876` (port 8765 was occupied; the helper's remembered fallback was used) |
| Health | `ready: true`, `model_loaded_once: true`, **`provider_calls: 0`** |
| Checkpoint | `checkpoints/local_atomic_seed17/variants/local_atomic_seed17` (`Qwen/Qwen3-0.6B`, config sha256 `3a6b014a…`, 596 250 498 params) |
| Serving | device `mps`, precision `fp32`, parameter storage `float32`, 0 autoregressive decode steps, `network_model_calls: 0` on every request |
| Limits respected | ≤ 32 states / ≤ 96 questions / ≤ 256 candidate paths per request |

**Survey.** The `request` object of `research/skill_abstention_survey_v1.json` was lifted verbatim
to `/tmp/njdiag/survey_request.json` (6 states / 13 questions / 28 candidate paths). No
`abstain_below` was supplied, so every recorded `confidence` is the **raw max probability**;
abstention rates below are recomputed from those raws at each threshold.

**In-domain.** `dataset/games_v4/data/local_maze_v1/test.jsonl` (44 states × 4 boolean questions =
176 questions) and `ood.jsonl` (16 states × 4 = 64 questions). The 44-state test split exceeds the
96-question cap, so it was split into two bounded batches (24 states/96 questions, then 20
states/80 questions); the OOD split fits in one batch (16 states/64 questions). `state_id` →
`state.id`, `state` → `state.state`, and the question objects passed through unchanged; gold was
never sent. The batches were merged before aggregation.

**Fidelity check.** The replayed in-domain metrics reproduce `results/nanojev_v2_baseline_seed17.json`
to floating-point precision (test: accuracy `0.7784090909090909`, NLL `0.44068301`, ECE
`0.085127211429856`; OOD: accuracy `0.765625`, NLL `0.44663747`, ECE `0.07338943984359503`;
agreement ≤ 1e-8, the residual being float32-vs-float64 summation). Batching and the HTTP
service path therefore did not perturb inference. A repeated call on the same 96-question batch
returned **bitwise-identical** confidences (max abs diff `0.0`).

## Result 1 — the survey reproduces, including the 13/13 abstention

The survey is fully reproduced: min `0.255`, median `0.505`, max `0.736`, mean `0.494` — matching
`measured_2026_09_19` in the survey file. Re-running the same request with `abstain_below: 0.9` on
every question gives `decision_summary.abstained = 13`, all answers flagged `abstained`, all
selected values nulled. The 13/13 claim is a measurement, not an inference from the raws.

| state:question | type | model answer | fixed answer | ✓/✗ | confidence |
|---|---|---|---:|---:|---:|
| phase:next_phase | choice | testing | – | – | 0.2676 |
| phase:b0_first | boolean | false | true | ✗ | 0.6153 |
| phase:blocked | boolean | false | true | ✗ | 0.6574 |
| tests:which_test | choice | targeted | – | – | 0.4003 |
| tests:needs_new_test | boolean | true | true | ✓ | 0.6677 |
| routing:route | choice | main_model | – | – | 0.4901 |
| routing:safe_to_drop | boolean | true | false | ✗ | **0.7361** |
| failure:kind | choice | code_bug | – | – | 0.2550 |
| failure:is_regression | boolean | false | – | – | 0.5048 |
| risk:action | choice | ask | – | – | 0.3998 |
| risk:check_secrets | boolean | true | true | ✓ | 0.6220 |
| quality:level | score | level 2 | – | – | 0.2699 |
| quality:token_savings | boolean | false | false | ✓ | 0.5323 |

Fixed-answer subset: **3/6 = 50.0%**; the two safety-relevant misses are `safe_to_drop` → `true`
at 0.7361 (the highest confidence in the survey) and `blocked` → `false` at 0.6574.

## Result 2 — in-domain the model is confident, and its confidence carries signal

240 in-domain questions, all answered, none near-uniform.

| Statistic | Survey (OOD, n=13) | Maze test (n=176) | Maze OOD (n=64) | In-domain pooled (n=240) |
|---|---:|---:|---:|---:|
| min | 0.2550 | 0.5172 | 0.5072 | 0.5072 |
| p25 | 0.3998 | 0.6724 | 0.5524 | 0.6292 |
| **median** | **0.5048** | **0.7890** | **0.6744** | **0.7792** |
| p75 | 0.6220 | 0.9429 | 0.9756 | 0.9462 |
| max | 0.7361 | 0.9978 | 0.9962 | 0.9978 |
| mean | 0.4937 | 0.7972 | 0.7363 | 0.7810 |
| fraction ≥ 0.9 | 0/13 = 0% | 68/176 = 38.6% | 22/64 = 34.4% | 90/240 = 37.5% |

**The entire survey distribution lies below the in-domain median** and below every in-domain
question in the ≥0.9 band. The highest confidence the model ever emits on an engineering question
(0.7361) is lower than **140/240 = 58.3%** of its in-domain confidences.

## Result 3 — abstention at each threshold

| Threshold | Survey abstain | Maze test abstain | Maze OOD abstain | In-domain pooled abstain |
|---:|---:|---:|---:|---:|
| 0.50 | 6/13 = 46.2% | 0/176 = 0.0% | 0/64 = 0.0% | 0/240 = 0.0% |
| 0.60 | 8/13 = 61.5% | 25/176 = 14.2% | 26/64 = 40.6% | 51/240 = 21.3% |
| 0.70 | 12/13 = 92.3% | 55/176 = 31.3% | 35/64 = 54.7% | 90/240 = 37.5% |
| 0.80 | 13/13 = 100% | 91/176 = 51.7% | 39/64 = 60.9% | 130/240 = 54.2% |
| **0.90** | **13/13 = 100%** | 108/176 = 61.4% | 42/64 = 65.6% | **150/240 = 62.5%** |

The two regimes never meet: at 0.5 the survey still abstains on 46.2% of its questions while the
in-domain cohorts abstain on none; at 0.8 the survey is at 100% while in-domain is at 54.2%.
**But note the second-order fact**: the 0.9 gate abstains on 62.5% of *in-domain* questions too.
The out-of-domain failure is the disappearance of the high-confidence mode, not that in-domain
questions pass the gate comfortably.

## Result 4 — accuracy versus confidence (the (b)-vs-(c) discriminator)

| Threshold t | Survey: answered / accuracy | Maze test: answered / accuracy | Maze OOD: answered / accuracy | Pooled in-domain: answered / accuracy |
|---:|---:|---:|---:|---:|
| 0.50 | 6 / 50.0% | 176 / 77.8% | 64 / 76.6% | 240 / 77.5% |
| 0.60 | 5 / 40.0% | 151 / 82.1% | 38 / 84.2% | 189 / 82.5% |
| 0.70 | 1 / 0.0% | 121 / 90.1% | 29 / 93.1% | 150 / 90.7% |
| 0.80 | 0 / – | 85 / 90.6% | 25 / 96.0% | 110 / 91.8% |
| **0.90** | 0 / – | 68 / **97.1%** | 22 / **100%** | 90 / **97.8%** |

Confidence bins (pooled in-domain, 240 questions):

| Confidence bin | n | mean confidence | accuracy |
|---|---:|---:|---:|
| 0.50–0.60 | 51 | 0.556 | 58.8% |
| 0.60–0.70 | 39 | 0.655 | 51.3% |
| 0.70–0.80 | 40 | 0.755 | 87.5% |
| 0.80–0.90 | 20 | 0.842 | 65.0% |
| **0.90–1.00** | **90** | **0.961** | **97.8%** |

In-domain, confidence is **positively** associated with correctness: point-biserial
r = **+0.4157** (test), **+0.3763** (OOD), **+0.3970** (pooled); accuracy ≥0.9 is 97.8% versus
65.3% below 0.9; ECE is 0.0851 (test) / 0.0734 (OOD), reproducing the baseline receipt.

On the 6 fixed-answer survey questions the direction reverses weakly — mean confidence 0.670 on
the 3 wrong answers versus 0.607 on the 3 correct ones — but every association statistic at this
sample size is uninformative and they do not agree in magnitude: point-biserial r = −0.505,
Spearman ρ = −0.293, Kendall τ-a = −0.200, and an exact permutation test on the mean difference
over all C(6,3) = 20 label splits gives one-sided p = 0.80. This is a 3/3 split at n=6; the rank
pattern claimed by the original readiness doc does not hold (see below).

## Verdict

**The failure is (b): the model is uncertain only out-of-domain.** Concretely:

- **(a) rejected.** Confidence is not low everywhere. In-domain the checkpoint emits confidences up
  to 0.998, 37.5% of questions ≥ 0.9, and the ≥0.9 answers are 97.8% correct.
- **(c) rejected for the in-domain regime.** In-domain, confidence is monotonically informative at
  the top (97.8% accuracy ≥0.9 vs 65.3% below), positively correlated with correctness
  (r = +0.397), and roughly calibrated (ECE 0.072–0.085). Confidence exists *and* is calibrated to
  correctness inside the trained family. What is broken is the transfer of that confidence
  machinery to the natural-language engineering domain, where the whole distribution collapses
  into [0.255, 0.736].
- **(c) has only weak, non-established residual support *inside the survey*.** With n=6 the
  anti-correlation on the fixed-answer subset is a direction, not a measurement, and the prior
  doc's "three highest confidences are the three wrong answers" claim is factually wrong (below).

**Practical reading.** At the 0.9 gate the engineering survey abstains 13/13 because the model has
no in-domain analogue for those questions — abstention there is correct domain-rejection
behaviour, and the prior doc's "do not lower the threshold" conclusion survives. The new fact is
that this is **domain transfer**, not a globally mute model: any future use of this checkpoint must
be restricted to question distributions that resemble the maze family, and the confidence head
should not be assumed dead. Conversely, the 0.9 gate is still strict in-domain (62.5% abstention),
so "zero removals proposed" is not evidence that confidence is saturated everywhere either.

## Corrections to `docs/NANOJEV_SKILL_READINESS_V1.md`

The readiness doc was amended in the working tree while this diagnosis was running; its original
Result 2 claims are reproduced below with the independent confirmation and the statistics the
amendment still lacks.

| Prior claim | What was measured |
|---|---|
| "The three wrong answers carry the three highest confidences in the entire survey (0.736, 0.657, 0.615)" | Ranks by confidence: 0.736 `safe_to_drop` (wrong), **0.668 `needs_new_test` (correct)**, 0.657 `blocked` (wrong), 0.622 `check_secrets` (correct), 0.615 `b0_first` (wrong), 0.532 `token_savings` (correct). Wrong answers hold ranks 1/3/5, so a correct answer sits between two wrong ones. Only the weaker mean statement survives: 0.670 mean confidence on wrong vs 0.607 on correct. |
| "confidence is anti-correlated with correctness on this sample" | Direction reproduces on all three association measures (r = −0.505, ρ = −0.293, τ-a = −0.200) but on n=6 (3/3) with permutation p = 0.80; no magnitude, rate or interval is claimable. |
| "the model cannot discriminate" (global) | True out-of-domain, false in-domain: 97.8% accuracy at ≥0.9 in-domain is strong discrimination. |

## What is NOT established

1. **The survey's anti-correlation is not a measurement.** n=6 fixed-answer questions, 3 correct /
   3 wrong. The direction is consistent across point-biserial (−0.505), Spearman (−0.293) and
   Kendall τ-a (−0.200), and the exact permutation test on the mean difference gives p = 0.80, so
   no rate, interval, magnitude or ranking claim is supported.
2. **The cause of the out-of-domain collapse is not identified.** The survey differs from the maze
   cohorts in language distribution, question type (choice/score vs boolean), option wording and
   state length all at once. This study localises the collapse; it does not attribute it.
3. **No context-filter / retention questions were measured.** The gate's actual removal decisions
   are a different question type from the maze boolean questions. The 62.5% in-domain abstention at
   0.9 is *not* a prediction of the gate's removal rate and must not be quoted as one.
4. **One checkpoint, one machine, one frozen cohort.** Only `local_atomic_seed17`, only `mps`/`fp32`,
   only these splits. Nothing was trained, tuned or modified; this is inference-only evidence.
5. **No threshold was tuned on test data.** All thresholds in the tables are post-hoc re-readings of
   the same raw confidences; the 0.5–0.9 grid is reported in full so no single threshold is
   presented as selected.
6. **Small bins in the middle of the in-domain curve.** The 0.50–0.60 and 0.80–0.90 bins hold 51 and
   20 pooled questions and the 0.80–0.90 bin is locally non-monotone (65.0% accuracy). The monotone
   story rests on the ≥0.9 bucket (90 questions), not on the middle bins.
7. **"OOD" here means size extrapolation, not format shift.** `ood.jsonl` is the same `scaled_maze`
   family at maze size 50 (test uses 8/16/32) and it stays confident. The engineering survey is the
   only true out-of-family probe in this study.

## Reproduction

```bash
# 1. service (fell back to the remembered port 8876 because 8765 was occupied)
python3 /Users/markus/.codex/skills/nanojev-local-decider/scripts/nanojev_skill.py health --start

# 2. survey, raw probabilities (no abstain_below -> confidence == max probability)
#    /tmp/njdiag/survey_request.json is the "request" object of
#    research/skill_abstention_survey_v1.json, copied verbatim.
python3 /Users/markus/.codex/skills/nanojev-local-decider/scripts/nanojev_skill.py decide \
  --input /tmp/njdiag/survey_request.json --source codex \
  --task-tag skill_abstention_diagnosis_survey_raw

# 3. in-domain, bounded batches (24+20 states for test, 16 for ood)
for f in test_batch1 test_batch2 ood_batch1; do
  python3 /Users/markus/.codex/skills/nanojev-local-decider/scripts/nanojev_skill.py decide \
    --input /tmp/njdiag/$f.json --source codex \
    --task-tag skill_abstention_diagnosis_indomain_$f
done

# 4. aggregation (quantiles, threshold tables, bins, Brier/NLL/ECE)
/Users/markus/Documents/NanoJev/.venv/bin/python /tmp/njdiag/analyze.py
```

Event ids: survey `849d2def-6ea4-42e8-862b-1a75d8e2853e`; survey at `abstain_below=0.9`
`51c85ae3-a47c-4490-82fd-c63e8b37803d`; test batch 1 `7b142512-660b-4875-9f2d-220c7b65e6a0`;
test batch 2 `3898f8e2-df9d-470e-a276-0275bdec26cb`; ood `ef7700f3-be02-4b53-93a9-20096a49c22a`;
determinism repeat `2aa881b2-8cb2-4182-8220-e1b04984a8de`. Log:
`~/.codex/nanojev/usage.jsonl`. All six events plus capture-file sha256s are in the receipt.
