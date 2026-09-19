# NanoJev skill readiness V1

**Question asked (2026-09-19): can the local NanoJev be packaged as a skill for seamless
integration, or will it frequently abstain?**

**Answer: the packaging works; the model does not.** At the documented default threshold of 0.9
the skill abstained on **13 of 13** realistic engineering questions. When the threshold is
lowered far enough to get answers, the answers are at **chance (3/6)**, and the single
highest-confidence answer in the whole survey was **wrong in the safety-critical direction**.
Abstention is not a tuning defect here — it is the correct behaviour for a model that cannot
discriminate.

## Method

Service: the installed skill's own helper,
`/Users/markus/.codex/skills/nanojev-local-decider/scripts/nanojev_skill.py`, started with
`health --start`. It came up on `http://127.0.0.1:8876` (the default 8765 is occupied by another
process, so the helper used its remembered fallback port) reporting
`ready: true`, `model_loaded_once: true`, **`provider_calls: 0`** — confirming no remote call.

Checkpoint: `checkpoints/local_atomic_seed17/variants/local_atomic_seed17` (Qwen3-0.6B base,
attention set head), device `mps`, precision `fp32`, 0 autoregressive decode steps.

Survey: 6 states / 13 questions (5 choice, 7 boolean, 1 score) written to describe **this
project's actual engineering decisions** — lifecycle phase selection, test selection, request
routing under the `MAX_SCORED=32` limit, failure classification, pre-commit risk gating, and
checkpoint readiness. All questions were asked in one batched request (28 candidate paths),
inside the service's documented limits.

## Result 1 — abstention at the default threshold is total

`abstain_below` was left at the documented default of 0.9. Raw confidences were recorded so the
rate can be recomputed at any threshold.

| Threshold | Abstained | Answered |
|---:|---:|---:|
| **0.90 (documented default)** | **13/13 = 100.0%** | 0/13 |
| 0.80 | 13/13 = 100.0% | 0/13 |
| 0.70 | 12/13 = 92.3% | 1/13 |
| 0.60 | 8/13 = 61.5% | 5/13 |
| 0.50 | 6/13 = 46.2% | 7/13 |
| 0.40 | 4/13 = 30.8% | 9/13 |

Confidence distribution: min 0.255, p25 0.400, **median 0.505**, max 0.736.

**The maximum confidence observed anywhere in the survey (0.736) is 0.164 below the default
threshold.** There is no setting of the existing threshold at which this checkpoint answers
these questions and is also being used as designed.

## Result 2 — below the threshold, accuracy is chance and the errors are confidence-ranked

Six questions have answers fixed by this project's own recorded facts, so they can be scored
without judgement:

| Question | Expected | Model | Confidence | |
|---|---|---|---|---|
| B0 before financial baselines? | true | **false** | 0.615 | ✗ |
| Project blocked from financial claims? | true | **false** | 0.657 | ✗ |
| Safe to drop context without a validated gate? | false | **true** | 0.736 | ✗ |
| Behavior change needs a new test? | true | true | 0.668 | ✓ |
| Scan staged files for secrets? | true | true | 0.622 | ✓ |
| Does this checkpoint save tokens? | false | false | 0.532 | ✓ |

**3/6 = 50.0%.** The three wrong answers carry the **three highest confidences in the entire
survey** (0.736, 0.657, 0.615) while the three correct answers are lower (0.668, 0.622, 0.532).
Confidence is therefore **anti-correlated with correctness** on this sample, which is exactly
what the "confidence is uncalibrated" caveat predicts — and why lowering the threshold is not a
repair.

The most serious single result: asked whether it is safe to remove context without a validated
gate, the model answered **`true` at 0.736** — the highest confidence in the survey, and the
direct negation of the project's central safety rule. A workflow that lowered the threshold to
obtain answers would have received this as its most confident recommendation.

## Result 3 — the project's own history agrees

`summary` over the skill's usage log, before this survey was added:

| Metric | Value |
|---|---|
| Decision events | 8 |
| Questions asked | 23 |
| **Abstained questions** | **10 (43.5%)** |
| Mean confidence | 0.508 |
| Latency p50 / p95 | 103.9 ms / 348.1 ms |
| Feedback events | 6 |
| **Feedback labels** | **6 × `fallback`, 0 × anything else** |

Every recorded outcome in the project's history is `fallback`: a stronger model or a
deterministic rule made the actual decision. **The skill has never once been the load-bearing
decision in this project.**

## Result 4 — what does work

These are real, verified properties, and they are why the packaging question has a positive half:

- **On-demand startup works.** `health --start` brought the service up and `model_loaded_once`
  was true; the helper handles the occupied-port case by itself.
- **Deterministic.** Two identical runs produced identical confidences and identical answers
  across all 13 questions. Byte-comparable, not merely "similar".
- **Offline.** `provider_calls: 0`; the helper rejects remote URLs and redirects.
- **Fast.** 0.27–0.33 s of server evaluation for 13 questions / 28 candidate paths; end-to-end
  helper latency 274–327 ms, consistent with the historical p50 of 104 ms for smaller calls.
- **Receipts.** Every call produced an event id, a privacy-preserving log line, and accepted
  feedback labels — the plumbing required by the AGENTS.md participation rule functions.

## Verdict

**Technically skill-ready; not decision-ready.**

- The skill can be packaged, installed, started on demand, and called reliably, offline and
  deterministically. Nothing blocks the *mechanism*.
- The *model* abstains on essentially every realistic engineering question at the documented
  threshold, and is at chance with confidence-anti-correlated errors below it. This is consistent
  with the separately measured reason the gate proposes zero removals: under irrelevant archived
  context, Catalog Choice accuracy collapses 100% → 54.17%. The checkpoint currently cannot tell
  a load-bearing segment from a distractor, so abstaining is the **correct** behaviour.
- Therefore the practical rule is: **invoke the skill for the receipt, never treat its output as
  the decision.** That is already what the project does — 6 of 6 recorded outcomes are
  `fallback`.

**Do not lower the threshold to obtain answers.** The 0.9 gate is doing its job; the fix is model
and data quality (the contrastive-curation path in T9), not the threshold.

## What would change this verdict

1. A checkpoint that discriminates distractor from load-bearing context (T9, contrastive
   curation), re-measured on this same survey plus fresh engineering questions.
2. A confidence distribution that is **positively** correlated with correctness on a held-out
   engineering set — the anti-correlation above must reverse, not merely improve.
3. Confirmation that abstention falls below roughly 20% at the default threshold **while**
   accuracy on answered questions stays above chance.

## Reproduction

```bash
python3 /Users/markus/.codex/skills/nanojev-local-decider/scripts/nanojev_skill.py health --start
python3 /Users/markus/.codex/skills/nanojev-local-decider/scripts/nanojev_skill.py decide \
  --input /tmp/njsurvey/survey.json --source codex --task-tag abstention_survey
python3 /Users/markus/.codex/skills/nanojev-local-decider/scripts/nanojev_skill.py summary
```

Event ids recorded for this measurement: `0e2fba4f-d012-4066-b856-5b74c13e8da7` (lifecycle call,
abstained, feedback `abstained`) and `3a2a011c-683b-449f-b806-b05af4a309cd` (survey, feedback
`fallback`).

**Limits.** One 13-question survey on one checkpoint on one machine, plus the project's 23
historical questions; the ground-truth subset is 6 questions, which is too small for a
confidence interval. The direction is unambiguous, but the precise rates are not precise
estimates.
