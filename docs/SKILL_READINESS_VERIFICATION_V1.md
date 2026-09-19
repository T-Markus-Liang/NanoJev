# Skill readiness V1 — adversarial verification

**Scope.** Independent falsification attempt against `docs/NANOJEV_SKILL_READINESS_V1.md`
("the readiness doc", 2026-09-19) and its survey `research/skill_abstention_survey_v1.json`
("the survey file"). Verification pass run 2026-09-19T14:2x UTC at repo HEAD `94d9ac3`, working
tree dirty (pre-existing). Read-only: nothing in this pass modified any repository file except
this document. No network was used; all model evidence came from the loopback service already
on `http://127.0.0.1:8876`.

**Evidence used (all local, all already on disk before this pass):**

| Source | What it establishes |
|---|---|
| `research/skill_abstention_survey_v1.json` | question set, the 6 declared labels, the header claims (`13/13`, `accuracy 3/6`, `max_confidence 0.736`) |
| `~/.codex/nanojev/usage.jsonl` (skill usage log) | the per-event record of what was actually asked, what `abstain_below` was applied, and the per-event `abstained_count` / confidence extremes |
| `/tmp/njsurvey/survey.json`, `run1.json`, `run2.json` (raw artifacts of the original run) | the full per-question answer, probability distribution and confidence for all 13 questions, and the one-byte-level determinism check |
| `/Users/markus/.codex/skills/nanojev-local-decider/scripts/nanojev_skill.py` | threshold semantics of `decide` vs `lifecycle` |
| `docs/NANOJEV_V2_ROADMAP.md`, `docs/CURRENT_PROGRESS_AND_HANDOFF.md`, `docs/PAPER_TRADE_REAL_DATA_V1.md`, `docs/WORKFLOW_V2_BASELINE.md`, `docs/EXECUTION_REVIEW_LOG.md`, `results/paper_trade_b0_baseline_*.json`, `results/b0_periods/*.json` | the repository's own recorded facts for the label audit |

**Concurrent-correction note.** While this verification was in progress, a parallel workstream
edited `docs/NANOJEV_SKILL_READINESS_V1.md` (mtime 22:25:41) and added
`docs/SKILL_ABSTENTION_DIAGNOSIS_V1.md`. The edit withdraws the "three highest confidences /
anti-correlated" claim — the same falsification reported here as C5/C6 — and re-titles Result 2
to "out-of-domain accuracy is chance". I read the doc before that edit and re-read the diff after
it. Findings C1, C2, C7, C8, C9, C10, C11 and C12 apply to the edited version unchanged; C5 and
C6 are now **already corrected in the source document**, and my §3/§5 entries for them serve as
the independent confirmation plus the statistics the correction still lacks. My wording fixes in
§5 items 1–4, 6–11 have not been applied and remain live.

**Overall verdict: the measurement is weaker than the doc claims, but its practical conclusion
survives.** One headline sentence of the doc is not merely unsupported but factually wrong on the
doc's own numbers (§3). The abstention result is a post-hoc recomputation presented as an
observed run (§4). Four of the six ground-truth labels are solid; two are arguable (§1).
The n=6 statistics cannot support the words used ("at chance", "anti-correlated"), though they
also do not show the model is better than chance (§3). The "not decision-ready" verdict and the
"do not lower the threshold" rule survive because the maximum confidence in the survey (0.736)
is far below the 0.9 gate and the checkpoint proposes zero removals — those facts are
independently confirmed.

---

## 0. Claim-by-claim verdict index

| # | Claim (as published) | Verdict |
|---|---|---|
| C1 | Abstained **13/13** at the default 0.9 threshold | **Overstated / mislabeled.** True only as a recomputation: the recorded run had **no** threshold and abstained **0/13**. The 0.9 default exists only on the `lifecycle` path, not the `decide` path the survey used. |
| C2 | The maximum survey confidence 0.736 is 0.164 below the threshold | **Upheld** (arithmetic and record). |
| C3 | "There is no setting of the existing threshold at which this checkpoint answers these questions and is also being used as designed" | **Upheld, with wording caveat** — see C1; correct if stated as "no threshold below 0.9 is a used-as-designed setting", not "the default abstains". |
| C4 | Below the threshold, accuracy is **3/6 = 50.0%**, reported as "chance" | **Numerically upheld, statistically overstated.** 3/6 is also the *most likely* outcome under chance (p=0.3125), so it is consistent with — not evidence for — chance. 95% CI [11.8%, 88.2%] (Clopper–Pearson), [18.8%, 81.2%] (Wilson). |
| C5 | "The three wrong answers carry the **three highest confidences in the entire survey**" | **Falsified by the doc's own table.** The correct answer `needs_new_test` sits at 0.668, between the wrong 0.736 and the wrong 0.657. Two of the top three are wrong; the top-three set is {wrong, **correct**, wrong}. |
| C6 | "Confidence is therefore **anti-correlated with correctness**" | **Unsupported.** Mean confidence wrong 0.6696 vs correct 0.6073 (Δ=0.0623); permutation p=0.25 (one-sided, exact over all C(6,3) splits); rank correlation is **+0.29, i.e. positive, not anti-**. |
| C7 | "The single highest-confidence answer in the whole survey was **wrong in the safety-critical direction**" | **Upheld.** 0.736 on `safe_to_drop` = `true` against the project's central no-drop-without-a-gate rule. |
| C8 | History: 8 decisions / 23 questions / **10 abstained (43.5%)** / mean 0.508 / p50 104 ms / p95 348 ms / 6 feedback labels all `fallback` | **Partly wrong.** From the same log, before the survey: **7/23 abstained (30.4%)**, mean 0.504, p50 103.9 ms, p95 348.1 ms, **6 × fallback confirmed**. The 10/23 figure matches the 15-decision, 303-question log as it stands now (11/303 = 3.6%), not the pre-survey window as stated. |
| C9 | "**The skill has never once been the load-bearing decision in this project.**" | **Overstated.** Feedback implies reliance; it does not prove absence of reliance. The same log shows later `skill_abstention_diagnosis_*` calls answered 96/80/64 questions with max confidence 0.993–0.998 and mean 0.74–0.80 — i.e. the checkpoint *does* answer confidently in-domain, which is the opposite flavour of evidence. Narrow the claim to what the feedback records. |
| C10 | Determinism: "two identical runs produced identical confidences and identical answers… Byte-comparable, not merely 'similar'" | **Upheld for the model output; overstated for the artifact.** `run1.json` vs `run2.json` are **not** byte-identical (different `server_evaluation_seconds`, `inference_call_index`, `latency_ms`, `event_id`); after removing latency/index/event they are identical. Confidences and answers are bit-identical, which is the substantive claim. |
| C11 | Ground-truth labels are "fixed by this project's own recorded facts" | **4 of 6 upheld, 2 arguable** (§1): `check_secrets` is a generic practice, not a recorded project fact; `token_savings` is stated in the handoff prose but not backed by a stored per-run measurement. |
| C12 | Verdict "technically skill-ready; not decision-ready" and "do not lower the threshold" | **Upheld** on independent grounds (max confidence 0.736 ≪ 0.9; zero proposed removals; confidence distribution below threshold). |

---

## 1. Audit of the six ground-truth labels

| Label | Declared | Repository record | Verdict |
|---|---|---|---|
| `b0_first` = true | B0 before financial baselines | Roadmap: "B0. Measurement integrity — now the first gate. Status: **NOT met; blocking every other financial claim**… This gate precedes B1–B8"; task board gives T11 (B3/R3 financial baselines) `Depends on: T5, T6`, i.e. baseline work is downstream of B0; governing principle "do not build a model on top of a measurement that cannot yet be trusted". Handoff §4.0 likewise sequences B0 first. | **Recorded, defensible.** |
| `blocked` = true | Project blocked from financial claims | Roadmap: "blocking every other financial claim". Handoff: "B0 测量完整性… 未达成，阻塞所有金融结论" and the risk light "执行经济性（Track B）🔴 Red". Caveat for honesty: `results/paper_trade_b0_baseline_*.json` now show the three venues at −34,080 / −37,597 / −38,045 (spread 4.5% of capital, vs the pre-B0 271,945 = 2.7× capital), and `on_divergence: report`, `capacity: fixed` are in place. So the *block* is real and documented, but the underlying measurements are much closer to interpretable than "2.7× capital" suggests. | **Recorded, defensible** — but the label is only as strong as T5 (sensitivity report, still ⬜) being incomplete. |
| `safe_to_drop` = false | Safe to drop context without a validated gate | Handoff 方向 1: "接入 ≠ 被授权主动裁剪：网关默认 shadow、active 关闭。启用生产主动裁剪仍需 Track A 全部门禁 + ≥3 主模型族配对下游质量与净 token 成本证据"; roadmap Track A acceptance list requires "100% fail-open behavior for… decisions below the confidence threshold" and three model families. | **Recorded, defensible — the strongest label in the set.** |
| `needs_new_test` = true | A behavior change requires a new test | P2 acceptance: "确定性重放…"; roadmap: no acceptance without a proving command; handoff §5 forbids "因测试失败希望删测试、放宽门槛"; every task board row has a `Verify` command and the project's own convention is test + receipt. | **Defensible, but near-tautological.** Anyone would answer this `true`; it measures comprehension of a platitude, not engineering judgment. |
| `check_secrets` = true | Scan staged files for secrets before committing | **No repository rule says this.** The recorded rules are: never commit `.env`/provider credentials (`JEV_COMPARISON_PROTOCOL.md`), "Do not retain broker or venue credentials anywhere in this repository" (`FINANCIAL_DATA_PLAN_V1.md` §412), gateway receipts must not persist credentials (`REVERSIBLE_FILTERING_V1.md`), and a `secrets_credentials` *fixture class* in the A4 corpus (`TOOL_HISTORY_SHADOW_V1.md`). None of these is a pre-commit secret-scan rule. | **Arguable / not a project fact.** It is a good universal practice, so `true` is the answer a competent engineer gives; but scoring it as "fixed by this project's own recorded facts" is wrong. |
| `token_savings` = false | Checkpoint currently saves tokens | Handoff: "生产参考仍为 `checkpoints/local_atomic_seed17/variants/local_atomic_seed17`，**实际 token 节省为零**"; roadmap Track A acceptance: "complete… receipts…"; the same checkpoint "proposes zero context removals". | **Recorded in prose, defensible** — but note the support is a prose statement about the pipeline, not a stored zero-savings receipt for this checkpoint. |

**Answers I judged arguable and therefore unscoreable (which the survey correctly excluded):**
`next_phase` (no fixed answer; the model's `testing` is a *reasonable* reading of "run proportionate
tests" plus "T1/B0 尚未关闭"), `which_test` (`targeted` is a defensible reading of the project's
"run proportionate tests" convention, `full` of "full suite OK" acceptance), `route` (the model's
`main_model` matches the roadmap's "Requests with more than 32 scorable candidates bypass
entirely" and "the most likely to hit this", yet the roadmap's own T8 task is to build the
`batch_merge` path, so both are arguable), `kind`, `is_regression`, `action`, `level`.

**Net effect on the headline:** the exclusion rule is defensible in direction. But two of the six
scored labels (`check_secrets`, `needs_new_test`) are generic rather than project-specific, and
one of them (`check_secrets`) is not recorded anywhere in the repository. Removing
`check_secrets` would leave 2/5; discarding both generic items leaves 1/4 — the accuracy figure
is that sensitive to label choice. That sensitivity, not the raw 3/6, is the real finding.

---

## 2. Survey fairness audit

1. **The question set is loaded toward the conclusion.** Of 13 questions, 6 are scored and 3 of
   those 6 (`b0_first`, `blocked`, `safe_to_drop`) are answered by a single repository sentence
   each, in the direction the repository is emphatic about; 2 more are engineer platitudes. The
   scored subset therefore contains no genuinely open engineering trade-off. The seven excluded
   questions (including `route`, where the model's answer agrees with the roadmap's recorded
   behaviour) are exactly the ones where the model looks better or where the answer is
   genuinely debatable. A neutral design would pre-register which questions are scorable and
   which are not, before seeing answers.
2. **A scoring rule that is not stated in the doc.** `check_secrets` and `needs_new_test` are
   policy/engineering defaults. Scoring them as "fixed by project facts" is the weakest
   methodological step in the survey. The survey file's note is right to say the other seven have
   no single defensible ground truth; it is too generous to itself about these two.
3. **"At chance" and "the top three confidences are the errors" cannot both be the intended
   refutation, because they are different tests.** Chance-level accuracy *without* confidence
   ranking is the signature of no discrimination; chance-level accuracy *with* confidence
   anti-ranking is the stronger and more alarming claim. The doc asserts both, but the sample
   supports the first and falsifies the second (C5, C6).
4. **One question is answerable only by guessing, and the doc treats the guess as informative.**
   `kind` (code bug vs test bug vs data bug vs flaky) is genuinely underdetermined by the state
   string; the model's four probabilities are 0.2550/0.2513/0.2486/0.2450 — a uniform
   distribution within 1 point. This is consistent with "the model knows it cannot know", i.e.
   partial competent abstention-by-flatness, and the doc's own framing of total indiscrimination
   is not the only reading. (It is unscored, so it changes no number — but it weakens the
   "cannot discriminate at all" narrative.)
5. **State strings are thin.** Each state is one sentence and the candidate descriptions carry
   most of the semantics ("Let the gate bypass with scoring_budget_exceeded"). That is a
   legitimate prompt format, but it means the survey measures judgment given a heavily
   compressed state, and it is unfair to compare that to "engineering-judgment questions" in
   general. Two of the six labels would likely be answered correctly by a bag-of-words baseline.
6. **A reasonable engineer would dispute:** `check_secrets` (a policy choice, not a fact),
   `which_test` (targeted vs full both defensible), `is_regression` (regime priority ordering is
   a specification in `PAPER_TRADE_REAL_DATA_V1.md`, which makes "regression" arguable),
   `action` (the state says the user authorized it, so `commit_push` vs `ask` depends on an
   unstated secret/capability policy). The doc should say which of its scored items it considers
   indisputable and why, per item.
7. **What is fair and holds up:** the safety-critical item (`safe_to_drop`) is genuinely fair,
   genuinely scored, and genuinely answered wrongly at the survey's maximum confidence. That
   single item, not the 3/6, is the survey's real content.

---

## 3. Statistics (n=6 cannot carry the words used)

Computed from the six recorded confidences/correctness pairs (system Python, exact arithmetic):

| Quantity | Value |
|---|---|
| Accuracy | 3/6 = 0.500 |
| P(exactly 3/6 \| p=0.5) | 0.3125 — the **modal** outcome under chance |
| P(≥3/6 \| p=0.5) | 0.65625 |
| Exact two-sided binomial test vs p=0.5 | p = 1.0 (0.65625 one-sided) — no evidence of deviation from chance in either direction |
| 95% CI, Clopper–Pearson | **[0.118, 0.882]** |
| 95% CI, Wilson | **[0.188, 0.812]** |
| Confidence ranking | 1 `safe_to_drop` 0.7361 **wrong** · 2 `needs_new_test` 0.6677 *correct* · 3 `blocked` 0.6574 **wrong** · 4 `check_secrets` 0.6220 *correct* · 5 `b0_first` 0.6153 **wrong** · 6 `token_savings` 0.5323 *correct* |
| Mean confidence wrong vs correct | 0.6696 vs 0.6073 → Δ = +0.0623 |
| Exact permutation p (all C(6,3)=20 splits, one-sided Δ≥observed) | **0.25** (two-sided 0.50) |
| Rank correlation (confidence rank, correctness) | **+0.293** (positive: higher confidence weakly *more* likely correct) |
| P(the 3 wrong are exactly the top 3), under random correctness with 3 wrong fixed | 1/C(6,3) = **0.05** — the only number in the doc that is conventionally significant, and it is a *different* claim from the one made |

**What n=6 can and cannot say.**

- It **can** say: 3 of 6 is compatible with chance; it is also compatible with 88% accuracy. Nothing
  about a rate.
- It **cannot** say "at chance" as a finding, because chance is the null, not a result — under the
  null, 3/6 is the single most likely score. The honest statement is "the observed score is
  indistinguishable from chance", with the interval.
- It **cannot** support any claim about confidence ordering: the exact permutation p for the
  mean-confidence gap is 0.25, and the sign of the rank correlation is *opposite* to the claim.
- It **cannot** support "anti-correlated": with 6 points and a 0.06 mean gap between groups whose
  ranges overlap (correct answers include 0.668, wrong answers include 0.615), the anti-correlation
  is one of the least likely readings of the data. The doc's own table already violates it.
- **What n would be needed.** For a one-sided exact binomial test at α=0.05 with 80% power to
  distinguish 0.8 accuracy from 0.5, **n=18** is the minimum (≥13 correct); to distinguish 0.7
  from 0.5, **n=37**; to distinguish 0.65 from 0.5, **n=69**. For a 95% Wilson interval narrow
  enough to be useful (±0.1), n≈100. For the confidence-vs-correctness question specifically, a
  paired design with ≥40–60 labelled items, pre-registered direction, and a corrected test is the
  minimum before "anti-correlated" is a finding rather than a coincidence. At n=6 the *only*
  defensible confidence statement is descriptive: "the highest-confidence answer was also the
  most dangerous one."

**Multiplicity.** The doc effectively tests several claims on the same 6 points (accuracy ≠ 0.5;
errors are top-ranked; sign of the correlation). With three tests, even the p=0.05 top-3 event
does not survive a Bonferroni threshold of 0.017. The readiness doc's "Limits" paragraph is
therefore too generous to itself: it concedes "too small for a confidence interval" but still
publishes "at chance" and "anti-correlated" as result-2 conclusions and repeats them in the
abstract.

---

## 4. Abstention audit: the recorded run did not abstain, and 0.9 is not the `decide` default

**Threshold semantics (from `nanojev_skill.py`, read directly):**

- `decide` (lines 473–480) has **no** `--abstain-below` argument and no default. A question is
  gated only if the input JSON itself carries `abstain_below` (line 229, applied line 272). With
  no such field, `is_abstained` is `False` for every question.
- `lifecycle` (line 485) has `--abstain-below` with `default=0.9`, and it injects that value into
  the single `next_check` question it builds (line 445). It then calls `command_decide`.

So "the documented default threshold of 0.9" is the **`lifecycle` path's default only**. The
`decide` path — the one the readiness doc's own Reproduction section uses for the survey
(`decide --input /tmp/njsurvey/survey.json`) — has no default threshold at all. The doc's
`Method` paragraph ("`abstain_below` was left at the documented default of 0.9") is not
implementable on the command it printed.

**What the record shows.** The survey event in `~/.codex/nanojev/usage.jsonl`:

```
event_id    e2b6b800-a603-4bf3-94ff-ffb47c6ddcfc
task_tag    abstention_survey       timestamp 2026-09-19T14:13:29Z
input_sha256 c25c42fba029b5ae752c154954659712ca7d8e3addfb03768841640ec4b7cb7d
question_count 13   state_count 6   candidate_paths 28
abstained_count 0   confidence_min 0.25499996542930603
confidence_max 0.7360702157020569   confidence_mean 0.4937087068190941
```

I recomputed `sha256(canonical_json({"states": <survey file's request.states>}))` and it equals
`c25c42fb…` exactly, while the same payload with `abstain_below: 0.9` inserted into every question
hashes to `54b8fe79…`. **The recorded survey input had no gate.** The event log says
`abstained_count: 0`, and so does the second recorded run tagged `determinism_check`, and so does
a third tagged `skill_abstention_diagnosis_survey_raw`.

Consequently: the survey's answers were *not* produced under a 0.9 gate; the 13/13 abstention in
the doc is a **post-hoc recomputation** from the raw confidences (which are all < 0.9, so the
recomputation is arithmetically correct and I reproduced it exactly, including the per-threshold
row 0.8→13, 0.7→12, 0.6→8, 0.5→6, 0.4→4). Presenting it under `## Result 1 — abstention at the
default threshold is total`, with the `decide` command in Reproduction, overstates what was
observed. The doc's event-id footnote compounds this: `3a2a011c-…` is tagged
`determinism_check`, not the survey, and its feedback is attached to a survey whose recorded
`abstained_count` is 0 — so neither recorded event is a "13/13 abstained" event. The only
genuinely gated, genuinely abstaining event is `0e2fba4f-…` (one `lifecycle` question,
three candidates, confidence 0.351), which is what the `lifecycle` default actually does. My own
verification `lifecycle` call this pass reproduced that behaviour (one question, confidence
0.27295, `abstained: true` at the 0.9 default).

**What is unaffected.** The max confidence 0.736 and the min/p25/median (0.255 / 0.400 / 0.505)
are exactly reproduced. `max < 0.9` is therefore a true statement about the model on this survey.
Result 4's offline/determinism/fast/receipt properties are also reproducible (with the byte-level
caveat in C10). And `results/workflow_v2_baseline_seed17.json` plus the readiness doc's own
citation of the 100%→54.17% distractor collapse remain independent support for "zero proposed
removals".

**Reasons a reader could have been misled.** (a) The doc reads as if a single command produced
both the abstention rate and the raw confidences; no single invocation can do that. (b) The
distinction between the `lifecycle` default (0.9) and the `decide` default (none) is exactly the
distinction the readiness doc blurs, and it matters because `AGENTS.md` mandates the `lifecycle`
path while the measurement used the `decide` path. (c) The survey file committed to `research/`
stores no `abstain_below` and no raw per-question confidences, so an independent reader cannot
recompute the headline from the repository alone — the recomputation is only possible from the
uncommitted `/tmp/njsurvey/run1.json` and the usage-log extremes. The claim "Raw confidences were
recorded so the rate can be recomputed at any threshold" is true of a temporary file, not of the
committed survey.

---

## 5. Corrections — exact replacement wording

1. **Title/abstract sentence.** Replace
   > At the documented default threshold of 0.9 the skill abstained on **13 of 13** realistic
   > engineering questions. When the threshold is lowered far enough to get answers, the answers
   > are at **chance (3/6)**, and the single highest-confidence answer in the whole survey was
   > **wrong in the safety-critical direction**.

   with
   > Under the `lifecycle` command's default `abstain_below=0.9` this checkpoint would abstain on
   > **13 of 13** questions, because every recorded confidence (max 0.736) is below the gate. The
   > survey itself was run through `decide`, which applies no threshold unless the input requests
   > one, and the recorded event shows **0 abstentions**. On the six questions with recorded
   > answers, the model scored **3/6** — indistinguishable from chance at this sample size
   > (95% CI 0.12–0.88) — and the single highest-confidence answer was **wrong in the
   > safety-critical direction**.

2. **Result 1 heading and table.** Replace `## Result 1 — abstention at the default threshold is
   total` with `## Result 1 — every recorded confidence is below the 0.9 lifecycle gate`, and add
   before the table:
   > This table is a recomputation from the recorded confidences, not an observed abstention
   > count. The recorded survey event (`e2b6b800-…`) reports `abstained_count: 0`; the gate
   > applies only if the input carries `abstain_below`, which the committed survey file does not.
   > The only gated event in the log (`0e2fba4f-…`, one `lifecycle` question) abstained, as the
   > `lifecycle` default requires.

3. **Method paragraph.** Replace "`abstain_below` was left at the documented default of 0.9" with
   > The survey was sent through `decide`, which has no default threshold; the 0.9 gate exists
   > only on `lifecycle` (`--abstain-below`, default 0.9), which injects it into its own
   > `next_check` question. The 0.9 figure below is therefore the lifecycle default applied
   > post hoc to recorded confidences, not a gate that was active during the survey run.

4. **Result 2 heading.** Replace `## Result 2 — below the threshold, accuracy is chance and the
   errors are confidence-ranked` with `## Result 2 — 3/6 correctness below the gate, with the
   highest-confidence answer wrong`.

5. **The "three highest confidences" sentence.** *(STATUS: already replaced in the readiness doc
   by the concurrent correction described at the top; kept here as the independent confirmation
   and because downstream docs still carry the old sentence — see item 11.)* Replace
   > The three wrong answers carry the **three highest confidences in the entire survey**
   > (0.736, 0.657, 0.615) while the three correct answers are lower (0.668, 0.622, 0.532).
   > Confidence is therefore **anti-correlated with correctness** on this sample…

   with
   > Two of the three wrong answers are the top two confidences, but the claim that the three
   > wrong answers are the three highest is **not true even in this table**: the *correct*
   > `needs_new_test` at 0.668 outranks the wrong `blocked` at 0.657 and the wrong `b0_first` at
   > 0.615. Mean confidence is 0.670 for the wrong answers and 0.607 for the correct ones; with
   > n=6 the exact permutation p for that gap is 0.25 and the rank correlation is **+0.29**, so
   > these data neither establish nor suggest anti-correlation.

6. **"At chance".** Anywhere "at chance (3/6)" / "3/6 = 50.0%" appears as a finding, write
   > 3 of 6 correct (50%), which is the most likely score under chance (P=0.3125) and therefore
   > cannot distinguish the model from chance; the 95% interval is 0.12–0.88
   > (Clopper–Pearson), so neither "at chance" nor "above chance" is supported by n=6.

7. **Determinism bullet.** Replace "Byte-comparable, not merely 'similar'" with
   > Identical answers and bit-identical confidences across two runs; the receipt JSON as a whole
   > is not byte-identical (latency and evaluation-time fields differ), so the determinism claim
   > applies to the model output, not to the artifact.

8. **Result 3 numbers.** Replace `| **Abstained questions** | **10 (43.5%)** |` with
   `| **Abstained questions** | **7 (30.4%)** — all 7 in gated `lifecycle` calls; the 13-question
   survey abstained 0 |`, and state the window ("the 8 decisions recorded before the survey").
   Keep `6 × fallback` (verified) and the latency figures (verified: p50 103.9 ms, p95 348.1 ms
   over those 8 events; the log now holds 15 events / 303 questions, over which the rate is
   11/303 = 3.6%).

9. **"Never load-bearing" sentence.** Replace
   > **The skill has never once been the load-bearing decision in this project.**

   with
   > Every one of the six feedback labels recorded before this survey is `fallback`, i.e. the
   > recorded outcomes all attribute the decision to a stronger model or a deterministic rule.
   > The log records only the feedback that was filed, so this is evidence of consistent
   > fallback labelling, not proof that no decision ever relied on the model.

10. **Add a limitation line to `Limits`.** Replace "the ground-truth subset is 6 questions, which
    is too small for a confidence interval" with
    > the ground-truth subset is 6 questions: its 95% interval is 0.12–0.88, it cannot separate
    > chance from 65–80% accuracy, and it cannot support any confidence-ordering claim (exact
    > permutation p = 0.25; observed rank correlation is +0.29, i.e. the opposite sign from the
    > published claim).

11. **Two downstream documents repeat the falsified sentence and should be corrected separately
    (not by this pass, which is read-only outside its own file):**
    `docs/CURRENT_PROGRESS_AND_HANDOFF.md:67` and `docs/NANOJEV_V2_ROADMAP.md:139` both say the
    errors are "恰是置信度最高三个" / "errors at the top 3 confidences". Both statements are
    contradicted by the readiness doc's own table (0.668 correct > 0.657 wrong) and by the raw
    run.

---

## 6. What a stronger follow-up measurement needs

**Sample size and design**

- **≥40 labelled engineering decisions** for any "accuracy vs chance" claim (this gives a 95%
  Wilson half-width near ±0.15 and 80% power against 0.7); **≥70** to resolve 0.65-vs-0.5; **≥18**
  is the bare minimum that could ever reject chance at 0.8 accuracy. Report the interval, never
  the point estimate alone.
- A **paired confidence–correctness design of ≥40–60 items** with the direction pre-registered,
  if the confidence-calibration claim is to be made at all. Report Spearman/permutation with the
  multiplicity correction stated up front.
- **Pre-register** the scored subset, the labels, the threshold, and the tests *before* looking at
  model output. The current survey chose its scored subset after seeing answers.

**Question provenance and label audit**

- Every scored label must cite a **specific recorded fact** (file + line, receipt + hash), the way
  §1 of this document does. Generic best-practice items (`check_secrets`, `needs_new_test`) must
  either be dropped or reclassified as a separate "common-sense" subscale, and reported
  separately. If a label's supporting fact is prose rather than a receipt (`token_savings`),
  store the receipt.
- Include at least one item per real engineering decision class from the roadmap's T1–T16 board,
  drawn *by rule* (e.g. every third board row) rather than by hand, so the failure is not
  concentrated by construction.
- Keep genuinely open trade-offs (targeted vs full tests; bypass vs batch-merge) **and** report
  them, marked as unscoreable, rather than removing them from the narrative.

**Threshold and path honesty**

- State the command and the exact `abstain_below` that produced every number, and record for each
  question whether the gate was active. Never present a recomputation as an observed run.
- Commit the exact run input (including `abstain_below` if used) and the full per-question
  confidence table to `research/` so the headline is recomputable from the repository.

**Blinding and independence**

- Two annotators label correctness independently from the recorded facts and reconcile blind;
  disagreements are reported, not resolved silently by the measurement author.
- Freeze the question set and its labels (hash in the manifest) before running the model; if the
  model output is already known, the labels must be shown to a reviewer who has not seen it.
- Re-run on ≥2 checkpoints and ≥2 seeds/configs, and cross-check against the existing in-domain
  batches (96/80/64 questions) so the report distinguishes "cannot answer engineering-judgment
  questions" from "has not been shown to".

**Reporting**

- Report per-item results (13 items, not 1 rate), the six-item and open-item subsets separately,
  the full confidence vector, and the exact tests with their p-values and intervals. A claim like
  "anti-correlated" must survive a pre-registered test at the stated n or be dropped to a
  descriptive sentence.

---

## 7. Scope and residual uncertainty

- **Not re-run:** I did not re-run the survey end to end. The original `/tmp/njsurvey/run1.json`
  matches the usage-log hashes and confidence extremes exactly, so it is sound evidence; a fresh
  gated re-run would additionally prove the 13/13 recomputation as an observed fact and is the
  single cheapest repair to C1.
- **One mandated local-model call was made** (AGENTS.md participation rule): a `lifecycle`
  `development` call recording this pass, event `546f72a3-6518-4367-a9a7-a821a39a3229`,
  confidence 0.27295, abstained. It answered nothing relevant to the verdict — which is itself
  consistent with the readiness doc's practical rule.
- **Log drift:** the usage log now contains 15 decisions / 303 questions / 11 abstentions, plus
  the four `skill_abstention_diagnosis_*` batches added after the readiness run. Any figure quoted
  from "the project's history" must name its window; the readiness doc's 8-decision window is
  reproducible but its abstention count (10) is not.
- **Files I did not touch:** the readiness doc, the roadmap, the handoff, the review log, the
  survey JSON, and every script. Corrections in §5 are wording for their owner to apply.

**Strongest surviving conclusion.** The checkpoint's maximum confidence on thirteen
engineering-judgment questions is 0.736 — 0.164 below the 0.9 gate — and the one question it
answered most confidently (0.736) was "it is safe to drop context without a validated gate",
which is the direct negation of the project's central safety rule, while the checkpoint proposes
zero removals and collapses 100%→54.17% under distractor context. Therefore the 0.9 gate is not
the defect and lowering it is not a repair; the skill is technically packageable but not
decision-ready. The survey's 3/6 accuracy and its confidence-ranking analysis do not add to that
conclusion — they are too small to carry it, and one of them is wrong as stated.
