# Skill calibration V1 — can a fitted temperature produce a usable operating point?

**Question (2026-09-19).** The local decision model serves probability distributions whose
confidence is documented as uncalibrated, and on 13 engineering-judgment questions it abstained
13/13 at the default 0.9 threshold while its 3 wrong answers carried the highest confidences
(`docs/NANOJEV_SKILL_READINESS_V1.md`). Can a temperature fitted on a **calibration split only**,
never on test data, produce a usable operating point — and at what accuracy cost?

**Answer. No — not for the abstention problem, and not measurably for held-out calibration either.**

1. The fitted value is **T = 0.854226** (NLL-minimising, fitted on `calibration.jsonl` only,
   192 Boolean questions / 48 states).
2. On the untouched test and OOD splits the fitted temperature changes **nothing that survives
   a state-clustered bootstrap**: NLL, Brier and ECE deltas all have 95% intervals containing
   zero, and **accuracy is identical to the fourth decimal** because temperature scaling cannot
   change an argmax.
3. On the engineering questions it is far too weak: the survey's maximum confidence moves
   **0.736 → 0.769**, still 0.13 below the 0.9 gate, so abstention stays **13/13**. Answering even
   one engineering question at 0.9 would need **T ≤ 0.467**, and that first question is the
   wrong, safety-critical one.
4. The reason is structural, not numerical: temperature scaling is strictly monotone, so it
   **cannot reorder confidences**. The engineering errors are the *most* confident answers, so
   sharpening admits them first. No value of a single scalar temperature fixes that.

A separate calibration split **does** exist and is legitimate; that is the one positive enabling
fact this study establishes and documents below.

---

## 1. How temperature is applied, and how to apply a candidate without editing the service

The serving path already carries a temperature concept:

- `scripts/predict_toy_decisions.py:294` — `probabilities = (scores / temperature).softmax(-1)`.
- `DecisionPredictor.predict(payload, batch_questions=0, temperature=1.0)` accepts the scalar and
  validates it as a finite positive number.
- The response echoes `temperature: {value, fitted_by_this_command: false, note: "…默认1不表示模型已校准"}`
  — i.e. the service is explicitly telling the caller that **it applied a scalar it did not fit,
  and that 1.0 does not mean calibrated.**
- `scripts/serve_decisions.py` never sets the argument, so the HTTP service always serves
  temperature 1.0. The HTTP layer would need no change to serve another value: only the caller's
  argument would change.

**Applying a candidate temperature with zero edits to any service file.** Two routes were used and
cross-checked:

| Route | Mechanism | Role |
|---|---|---|
| Inference (authoritative) | `DecisionPredictor.predict(payload, temperature=T)` on the unmodified module | every number reported here |
| Arithmetic (search) | `softmax(log(p) / T)`, valid because `softmax(z/T)` is shift-invariant in `z` | the grid search, which is otherwise 4000 model runs |

The two routes agree to **≤ 8.7e-08** maximum absolute probability difference on all four splits
(`service_path_verification` in the results JSON), well inside the service's own 1e-5 probability-sum
tolerance. So the arithmetic route is a faithful stand-in for the serving path, and the fitted
temperature is confirmed to be applicable by the real path.

No model weights were trained and no gradient step was taken. Temperature is a single post-hoc
scalar.

## 2. Split provenance — a calibration split exists, and it is not test data

**A separate calibration split does exist.** It is a frozen artifact of the cohort builder, not
something this study carved out:

```
dataset/games_v4/data/local_maze_v1/
  manifest.json  train.jsonl  dev.jsonl  calibration.jsonl  test.jsonl  ood.jsonl
```

| split | rows | Boolean questions | source groups | sha256 (first 16) |
|---|---:|---:|---:|---|
| train | 144 | 576 | 34 | `502f2840f3e15ffa` |
| dev | 48 | 192 | 11 | `bd7f9d4402112770` |
| **calibration** | **48** | **192** | **12** | **`94a233e71eaf3460`** |
| test | 44 | 176 | 11 | `be7783c395715f9f` |
| ood | 16 | 64 | 4 | `e6772998a189880c` |

Why fitting on `calibration.jsonl` is legitimate and is **not** tuning on test:

- **Hash-pinned.** Every split file's sha256 equals the value recorded in `manifest.json`
  (`outputs.<split>.sha256`). `calibration.jsonl` = `94a233e7…`, `test.jsonl` = `be7783c3…`,
  `ood.jsonl` = `e6772998…`. A copy of `test.jsonl` dropped into the calibration slot fails this
  check.
- **Disjoint by state.** `state_id` overlap between calibration and test is **0**, and with OOD
  is **0** — computed from the data, not read from prose. The same holds for every split pair.
- **Disjoint by maze group.** `source_group_id` overlap between calibration and test is **0**, and
  with OOD is **0**. No maze leaks across the boundary, so the split is not merely distinct rows of
  the same mazes.
- **Never trained on.** The checkpoint config records `train_questions: 576`, which is exactly the
  whole of `train.jsonl` (144 × 4). `calibration.jsonl` contributes 0 of those.
- **Never used for model selection.** The same config records
  `selection: "minimum dev target CE; held-out test first evaluated after training and checkpoint
  selection"`. Selection used dev; test and OOD were held back.
- The cohort manifest's own claims agree: `cross_split_source_groups: 0`,
  `cross_split_state_ids: 0`, `resampling: false`, `api_calls: 0`. These were independently
  re-derived rather than trusted.

The config references the cohort as `data/local_maze_v1` (a path from the training host); the frozen
local copy is `dataset/games_v4/data/local_maze_v1/`, and its file hashes match the config's
`data_sha256` block exactly.

**Enforced guard, not a comment.** `scripts/fit_decision_temperature_v1.py` refuses a forbidden fit
in five independent layers:

| Layer | What it stops |
|---|---|
| **G1** name | `--fit-split test` / `--fit-split ood` (and `--also-fit`) abort before any file is read |
| **G2** hash | any split file whose sha256 differs from the frozen manifest |
| **G3** row | a fitting row whose own `split` field is not the declared fit split |
| **G4** data | fitting `state_id`/`source_group_id` overlapping test or OOD, **always** computed even when test/OOD are not evaluated |
| **G5** objective | the loss loop re-asserts each scored row's split |

`--guard-self-test` proves G1 at runtime (`{"ood": "rejected", "test": "rejected"}`,
`{"calibration": "admitted", …}`, `passed: true`), `--verify-only` runs G2–G4 without loading the
model, and the shipped JSON records `fitted_on_test_or_ood: false`.

## 3. Method

- **Fit objective:** NLL (Brier also fitted as a cross-check). Log-spaced grid 0.05 → 20.0
  (4000 points, 1.0 forced onto the grid) then a deterministic 200-step golden-section refinement
  in log space.
- **Fit data:** `calibration.jsonl` only — 48 states, 192 Boolean questions.
- **Metrics** (per split, at each temperature): accuracy; NLL with a 1e-12 floor; Brier reported
  both as the scalar Bernoulli form and as the repo's two-class sum; ECE with 10 **fixed-width**
  bins on `p(true)` and, as a bin-sensitivity check, 10 **adaptive** (equal-count) bins; MCE.
- **Operating points:** a question is "answered" at threshold *t* when
  `max(p_true, 1 − p_true) ≥ t` — the exact semantics of the skill's own `confidence()` and
  `abstain_below` (`nanojev_skill.py:241-272`).
- **Uncertainty:** 2000-resample bootstrap clustered on `state_id` (the four questions of a maze
  share a state), seed 17.
- **Post-hoc engineering probe:** the frozen survey in `research/skill_abstention_survey_v1.json`
  (sha256 `25ccd45c…`) is run at 1.0 and at the fitted temperature. It is **applied, never fitted**;
  it contributes to no fitting or selection decision.
- **No network:** `network_model_calls: 0` on every call; the script aborts otherwise.

## 4. Results

### 4.1 The fitted value, and its instability

| fit split | objective | optimal T | loss at T | loss at 1.0 |
|---|---|---:|---:|---:|
| **calibration** | **NLL** | **0.854226** | **0.377361** | 0.380179 |
| calibration | Brier | 0.999438 | 0.252659 | 0.252659 |
| dev | NLL | 1.109016 | 0.481778 | 0.482880 |
| dev | Brier | 1.190477 | 0.327103 | 0.328454 |

Two things already undermine the value: on the **same** calibration data, Brier says "do nothing"
(0.999) while NLL says "sharpen 15%"; and on an independent held-out split, **dev**, both
objectives point the **other way** (T > 1, soften). A quantity whose sign flips between two
non-test splits is sample noise, not a stable property of the model. The NLL gain on calibration
is 0.0028 nats — 0.7% relative.

### 4.2 Untouched test and OOD, temperature 1.0 versus 0.854226

Accuracy is reported for completeness; it is **exactly invariant** by construction.

| split | T | accuracy | NLL | Brier (scalar) | ECE fixed-10 | ECE adaptive-10 | conf max |
|---|---:|---:|---:|---:|---:|---:|---:|
| calibration | 1.0 | 0.8229 | 0.3802 | 0.1263 | 0.1025 | 0.0795 | 0.9954 |
| calibration | 0.8542 | 0.8229 | **0.3774** | 0.1270 | 0.1050 | 0.0824 | 0.9981 |
| **test** | 1.0 | 0.7784 | 0.4407 | 0.1436 | 0.0960 | 0.0435 | 0.9978 |
| **test** | 0.8542 | 0.7784 | 0.4459 | 0.1446 | 0.0653 | 0.0538 | 0.9992 |
| **ood** | 1.0 | 0.7656 | 0.4466 | 0.1560 | 0.0814 | 0.1093 | 0.9962 |
| **ood** | 0.8542 | 0.7656 | 0.4466 | 0.1565 | 0.0997 | 0.1066 | 0.9985 |

Answer rate (`max(p, 1−p) ≥ t`) and accuracy among answered:

| split | T | ans@0.5 | ans@0.7 | ans@0.9 | acc\|ans@0.7 | acc\|ans@0.9 |
|---|---:|---:|---:|---:|---:|---:|
| calibration | 1.0 | 1.000 | 0.708 | 0.411 | 0.8971 | 1.0000 |
| calibration | 0.8542 | 1.000 | 0.755 | 0.438 | 0.8828 | 1.0000 |
| **test** | 1.0 | 1.000 | 0.688 | 0.386 | 0.9008 | 0.9706 |
| **test** | 0.8542 | 1.000 | 0.750 | 0.409 | 0.8636 | 0.9583 |
| **ood** | 1.0 | 1.000 | 0.453 | 0.344 | 0.9310 | 1.0000 |
| **ood** | 0.8542 | 1.000 | 0.500 | 0.344 | 0.8750 | 1.0000 |

### 4.3 What survives uncertainty — nothing, except the invariance

State-clustered bootstrap (2000 resamples) of `fitted − 1.0`, test and OOD:

| metric | test Δ (95% CI) | OOD Δ (95% CI) |
|---|---|---|
| accuracy | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] |
| NLL | +0.00545 [−0.00963, +0.02625] | −0.00013 [−0.00930, +0.00976] |
| Brier (scalar) | +0.00098 [−0.00203, +0.00432] | +0.00048 [−0.00298, +0.00416] |
| ECE fixed-10 | −0.02009 [−0.05959, +0.01834] | +0.00100 [−0.03803, +0.03376] |
| ECE adaptive-10 | +0.00002 [−0.01304, +0.01272] | −0.00202 [−0.01053, +0.00532] |
| answer rate @0.7 | **+0.06242 [+0.02273, +0.10227]** | +0.04752 [+0.00000, +0.12500] |
| accuracy\|ans @0.7 | **−0.03668 [−0.06899, −0.00955]** | −0.05530 [−0.12514, +0.00000] |
| answer rate @0.9 | +0.02337 [+0.00000, +0.05682] | +0.00000 [+0.00000, +0.00000] |

Reading:

- **Accuracy: identically zero change.** Not "small" — exactly zero, CI degenerate. This is the
  defining property of a monotone rescaling and it is confirmed empirically.
- **NLL, Brier, ECE: every 95% interval contains zero** on both held-out splits. The apparent
  fixed-bin ECE gain on test (−0.031 full-sample) does **not** survive clustering, and the
  adaptive-bin ECE point estimate is +0.00002 — i.e. the fixed-bin "improvement" is a binning
  artefact. This is the honest reason to distrust the single most flattering number in the table.
- **The only significant moves are a trade, not a gain:** on test at 0.7 the fitted temperature
  answers ~6.2 points more questions and those extra answers are ~3.7 points less accurate. More
  coverage, lower precision.

### 4.4 How far the usable-answer rate moves

At the documented default gate of **0.9**, which is the actual question asked:

| split | answer rate @0.9, T=1.0 | @0.9, T=0.8542 | movement | accuracy among answered |
|---|---:|---:|---:|---|
| test | 0.386 | 0.409 | **+2.3 pp** | 0.9706 → 0.9583 |
| ood | 0.344 | 0.344 | **0.0 pp** | 1.0000 → 1.0000 |

So in-domain the usable-answer rate moves by about **two percentage points on test and not at all
on OOD**. Note what this also reveals: in-domain the model is **not** abstention-crippled — at 0.9
it already answers 38.6% of test with 97.1% precision. The 13/13 abstention in
`NANOJEV_SKILL_READINESS_V1.md` is a property of the *engineering questions*, not of the gate.

### 4.5 The engineering questions, with the fitted temperature applied

Running the frozen 13-question survey through the same checkpoint (this exactly reproduces the
documented figures: max confidence 0.7361, abstained 13/13 at 0.9, accuracy 3/6):

| | T = 1.0 | T = 0.854226 |
|---|---:|---:|
| answered at 0.9 | **0/13** | **0/13** |
| answered at 0.7 | 1/13 | 1/13 |
| answered at 0.5 | 7/13 | 8/13 |
| max confidence | 0.7361 | 0.7686 |
| accuracy among answered at 0.7 (6 fixed-answer questions) | 0.0 (n=1) | 0.0 (n=1) |
| accuracy on the 6 fixed-answer questions | 3/6 = 0.500 | 3/6 = 0.500 |

The fitted temperature lifts the survey's best confidence from 0.736 to **0.769 — still 0.131 below
the gate.** Abstention at the documented threshold is unchanged at 13/13.

Because the transform is monotone the shortfall can be inverted exactly. The largest temperature at
which each question would still clear 0.9:

| question | confidence @1.0 | T needed for 0.9 | correct? |
|---|---:|---:|---|
| `safe_to_drop` | 0.736 | **0.467** | **wrong** (safety-critical direction) |
| `needs_new_test` | 0.668 | 0.318 | ok |
| `blocked` | 0.657 | 0.297 | **wrong** |
| `check_secrets` | 0.622 | 0.227 | ok |
| `b0_first` | 0.615 | 0.214 | **wrong** |
| `token_savings` | 0.532 | 0.059 | ok |
| remaining 7 (no fixed answer) | 0.255–0.505 | 0.006–0.168 | — |

- To answer **one** question at 0.9 you must go to **T ≤ 0.467** — and that first question is the
  wrong one that directly negates the project's central safety rule.
- **Correction (2026-09-19, against two independent audits).** The wrong fixed-answer questions
  are **not** the three highest-confidence ones: the ranks are 1 (`safe_to_drop` 0.736, wrong),
  2 (`needs_new_test` 0.668, **correct**), 3 (`blocked` 0.657, wrong), 4 (`check_secrets` 0.622,
  correct), 5 (`b0_first` 0.615, wrong), 6 (`token_savings` 0.532, correct) — the wrongs hold
  ranks 1/3/5. What survives, and is what matters here, is that the **single highest-confidence
  answer is the safety-critical wrong one**. Since `T > 0` is strictly monotone, no value of `T`
  can reorder confidences, so any threshold that admits an answer admits that one first: this is
  an **ordering failure**, and a monotone rescaling cannot repair an ordering.
- To answer **all 13** at 0.9 would need **T ≤ 0.0061** — a ~140× sharper transform than the fitted
  0.854 — at which point the distribution is saturated and the fixed-answer accuracy is still
  3/6 = 50%.

## 5. Verdict

**Temperature alone cannot fix the engineering-question abstention problem. It only acts
in-domain, and there it does not measurably help either.**

- **Structurally impossible for the engineering set.** Temperature scaling is strictly monotone in
  the log-odds. It can move *how many* questions clear a threshold; it can never change *which*
  questions are more confident than which. On this model the engineering errors are the
  highest-confidence answers, so no single scalar temperature can separate them from the correct
  ones. This is an ordering failure, and a monotone rescaling cannot repair an ordering.
- **Empirically absent in-domain.** The fitted value buys nothing that survives a state-clustered
  bootstrap on test or OOD: accuracy delta exactly 0, and NLL, Brier and ECE intervals all contain
  zero. The usable-answer rate at 0.9 moves +2.3 pp on test and 0.0 pp on OOD, while precision among
  answered falls slightly. The one nominally significant effect measured is a coverage/precision
  trade at 0.7, not a calibration gain.
- **The fitted value is not even stable.** Brier on the same calibration data says T ≈ 1.0; dev's
  NLL and Brier both say T > 1 (soften). The sign flips between non-test splits, so the point
  estimate is sample noise rather than a property of the checkpoint.
- **The in-domain problem is accuracy, not confidence.** The model is already sharp in-domain
  (max confidence ≈ 0.996) and already answers ~39% of test at 0.9 with ~97% precision; its ceiling
  is 0.78 accuracy on test and 0.77 on OOD. Temperature rescaling has no leverage on that ceiling.
- **What the study does establish as positive:** a legitimate calibration split exists, is
  hash-pinned and provably disjoint from test/OOD by both state and maze group, and the served
  `temperature.value` is fully controllable by the caller without editing any service file — with
  the arithmetic route agreeing with the real inference path to 8.7e-8.

**Recommendation.** Leave the served temperature at 1.0. Do not lower the 0.9 gate to obtain
engineering answers: the first answer to appear is the wrong, highest-confidence safety-critical
one. The fix remains model and data quality (T9 contrastive curation), not a scalar.

## 6. What is NOT established

1. **Nothing about other checkpoints.** All of this is one checkpoint,
   `checkpoints/local_atomic_seed17/variants/local_atomic_seed17`, and one cohort.
2. **Nothing about engineering questions as a calibration target.** The shortfall is quantified,
   but the survey is 13 questions of which only 6 have a defensible fixed answer. No confidence
   interval is claimed on the 3/6 figure; it is far too small.
3. **No claim that T = 0.8542 is the best in-domain operating point.** It is the NLL optimum of one
   192-question calibration sample whose Brier optimum is ≈ 1.0 and whose dev optimum is > 1. The
   honest reading is "indistinguishable from 1.0", and the bootstrap supports exactly that.
4. **Boolean-only.** The entire local-maze cohort is Boolean (176/176 test questions). Choice and
   score calibration — the types used by 5 of the 13 engineering questions and by 7 of the 13 — were
   **not** measured. A per-type temperature is untested and is not what was fitted here.
5. **ECE is bin-sensitive and finite-sample.** Two binnings are reported precisely because they
   disagree on test (fixed −0.031 vs adaptive +0.00002). Neither is a gold standard at n = 176 / 64.
6. **The bootstrap mean is not the full-sample point estimate.** With 44 (test) / 16 (OOD) state
   clusters the resampled mean sits slightly off the full-sample delta (e.g. test ECE fixed:
   full-sample −0.031, bootstrap mean −0.020). Point estimates in §4.2 are full-sample; §4.3 is for
   inference only.
7. **No distribution-shift claim.** OOD here means held-out maze groups from the same generator,
   not a different task. The engineering result is a post-hoc transfer demonstration on a frozen
   artifact, not a validated out-of-domain calibration.
8. **Not verified through the HTTP service.** Temperature was applied via the unmodified
   `DecisionPredictor`/`predict` module, which is the code `serve_decisions.py` calls. The HTTP
   endpoint itself was not exercised with a non-default temperature, because `serve_decisions.py`
   exposes no way to pass one — and editing it was out of scope.

## 7. Reproduction

```bash
# Prove the guard rejects test/OOD and admits calibration (no model load):
.venv/bin/python scripts/fit_decision_temperature_v1.py --guard-self-test

# Prove the split hashes and disjointness without loading the model:
.venv/bin/python scripts/fit_decision_temperature_v1.py --verify-only

# Full study (offline, MPS/CPU, ~2 minutes):
.venv/bin/python scripts/fit_decision_temperature_v1.py --output results/temperature_fit_v1.json

# Refuses (exit 2) before reading a file:
.venv/bin/python scripts/fit_decision_temperature_v1.py --fit-split test
```

Artifacts: `results/temperature_fit_v1.json` (fitted value, split definition with hashes, all
metrics at 1.0 vs fitted on calibration/dev/test/OOD, per-split service-path verification, the
post-hoc survey probe, and the bootstrap intervals) and this document.

Observed facts of record: service up on `http://127.0.0.1:8876`, `ready: true`,
`provider_calls: 0`; model device `mps`, precision `fp32`, 0 autoregressive decode steps,
`network_model_calls: 0` on every call. The mandatory advisory lifecycle call for this phase
abstained (confidence 0.436 < 0.9) and proposed fitting on `dev` rather than `calibration`;
that advisory was recorded as `abstained` and handled independently as the task specifies.
