# NanoJev local decision benchmark

**Frozen offline bundle:** 1,720 rows / 893 labeled rows across official JevBench public, SemIf authored/perturbation/shape fixtures, WANLI, and Every retrieval.
**Protocol:** evaluation-only, pinned source/model revisions, no benchmark-derived training or calibration, and no generated answer tokens unless an upstream engine requires them. See [`docs/BENCHMARK_PROTOCOL_V1.md`](docs/BENCHMARK_PROTOCOL_V1.md).
**Publication boundary:** this scorecard publishes aggregate metrics only; raw provider responses and per-item receipts remain local.

Ranked by labeled accuracy. Local wall time is measured end-to-end on Apple Silicon; Official Jev direct uses remote response latency and is not strictly comparable for speed.

| Rank | Candidate | Acc | Δ vs Jev | Speed× | BalAcc | NLL ↓ | Brier ↓ | Cov@0.9 | CW@0.9 | Perturb flip | Every R@1 | Every MRR | Wall s | p50 s | p95 s | Err | Tok in | Tok out |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Official Jev direct | 0.8858 | +0.0000 | 1.00× | 0.8443 | 0.9648 | 0.1775 | 0.7180 | 29 | 0.0000 | 1.0000 | 1.0000 | 642.7 | 0.292 | 0.729 | 0 | 2028925 | 63261 |
| 2 | Winnow-12B Q8 | 0.8824 | -0.0034 | 1.30× | 0.8374 | 0.9295 | 0.2069 | 0.8744 | 59 | 0.0278 | 1.0000 | 1.0000 | 494.6 | 0.152 | 0.831 | 0 | 1745656 | 0 |
| 3 | Kev-9B | 0.8186 | -0.0672 | 0.92× | 0.7548 | 0.9985 | 0.2599 | 0.5058 | 35 | 0.0093 | 1.0000 | 1.0000 | 700.4 | 0.174 | 1.444 | 0 | 1556839 | 88723 |
| 4 | Kev-4B | 0.8018 | -0.0840 | 1.07× | 0.7048 | 1.1326 | 0.3055 | 0.4942 | 47 | 0.0185 | 1.0000 | 1.0000 | 601.5 | 0.128 | 1.399 | 0 | 1556839 | 88597 |
| 5 | OpenAlternative Qwen3.5-4B | 0.7917 | -0.0941 | 0.24× | 0.7436 | 1.0762 | 0.3094 | 0.4459 | 31 | 0.1574 | 1.0000 | 1.0000 | 2682.9 | 0.537 | 3.012 | 0 | 0 | 0 |
| 6 | Decider-2B | 0.7917 | -0.0941 | 0.61× | 0.6875 | 1.1044 | 0.3142 | 0.5558 | 43 | 0.0370 | 1.0000 | 1.0000 | 1051.7 | 0.282 | 1.163 | 0 | 0 | 0 |
| 7 | Reflex stable | 0.7895 | -0.0963 | 2.61× | 0.7803 | 1.0509 | 0.2913 | 0.2343 | 10 | 0.0741 | 1.0000 | 1.0000 | 246.4 | 0.089 | 0.404 | 0 | 0 | 0 |
| 8 | this-that-model-1.0 | 0.7850 | -0.1008 | 0.62× | 0.6366 | 1.4306 | 0.3658 | 0.7436 | 101 | 0.0278 | 0.9474 | 0.9737 | 1037.9 | 0.229 | 1.195 | 0 | 1400024 | 0 |
| 9 | SemIf Qwen3.5-4B MLX4 | 0.7671 | -0.1187 | 1.57× | 0.7423 | 1.1401 | 0.3365 | 0.5384 | 39 | 0.1389 | 1.0000 | 1.0000 | 409.6 | 0.099 | 0.450 | 0 | 1699225 | 0 |
| 10 | Open-Jev-2B | 0.7413 | -0.1445 | 0.18× | 0.6173 | 1.1894 | 0.3662 | 0.2605 | 23 | 0.0463 | 0.8421 | 0.8991 | 3532.6 | 0.673 | 4.599 | 0 | 3416399 | 0 |
| 11 | Decider-0.8B | 0.7256 | -0.1601 | 0.65× | 0.6453 | 1.2150 | 0.3871 | 0.3203 | 33 | 0.0926 | 0.9474 | 0.9737 | 989.8 | 0.264 | 1.093 | 0 | 0 | 0 |
| 12 | SmallJev-v9 | 0.6540 | -0.2318 | 1.08× | 0.5003 | 1.3454 | 0.4559 | 0.1070 | 17 | 0.2037 | 0.7895 | 0.8860 | 596.4 | 0.153 | 0.647 | 0 | 0 | 0 |
| 13 | NanoJev MiniCPM seed20 | 0.6529 | -0.2329 | 0.26× | 0.6694 | 0.7705 | 0.4603 | 0.2721 | 21 | 0.0833 | 0.9474 | 0.9737 | 2519.0 | 0.355 | 3.031 | — | — | — |
| 14 | Kev-0.6B | 0.6394 | -0.2464 | 9.97× | 0.5839 | 2.5804 | 0.5362 | 0.2721 | 95 | 0.1667 | 0.8947 | 0.9263 | 64.4 | 0.023 | 0.141 | 0 | 1468620 | 88039 |
| 15 | Gavel-base | 0.6338 | -0.2520 | 9.47× | 0.5097 | 1.3056 | 0.4575 | 0.0378 | 3 | 0.1111 | 0.6316 | 0.7544 | 67.9 | 0.027 | 0.067 | 0 | 0 | 0 |
| 16 | Laya base | 0.5834 | -0.3024 | 12.12× | 0.5620 | 1.4658 | 0.5596 | 0.1227 | 33 | 0.1759 | 0.6842 | 0.7803 | 53.0 | 0.032 | 0.046 | 0 | 0 | 0 |
| 17 | OpenDecision | 0.5588 | -0.3270 | 5.85× | 0.5393 | 1.4892 | 0.5722 | 0.0797 | 18 | 0.2037 | 0.4211 | 0.6469 | 109.8 | 0.063 | 0.114 | 0 | 1714130 | 0 |
| 18 | Laya typed-decisions | 0.5543 | -0.3315 | 7.25× | 0.4942 | 1.4329 | 0.5441 | 0.0081 | 1 | 0.1667 | 0.6316 | 0.7662 | 88.7 | 0.032 | 0.090 | 0 | 0 | 0 |
| 19 | AgentJev-0.6B | 0.5398 | -0.3460 | 1.25× | 0.4912 | 1.4790 | 0.5663 | 0.0052 | 2 | 0.0741 | 0.7895 | 0.8596 | 513.0 | 0.055 | 0.655 | 0 | 2831732 | 0 |
| 20 | OpenJev DeBERTa-v3-large | 0.5330 | -0.3527 | 9.39× | 0.6097 | 1.4749 | 0.5750 | 0.0395 | 15 | 0.2315 | 0.7895 | 0.8605 | 68.4 | 0.049 | 0.061 | 0 | 0 | 0 |
| 21 | J-FAST seed30 | 0.3953 | -0.4905 | 2.66× | 0.2042 | 1.0867 | 0.6539 | 0.0021 | 2 | 0.3889 | 0.2632 | 0.4674 | 241.5 | 0.038 | 0.292 | — | — | — |
| 22 | Certo | 0.3572 | -0.5286 | 11.73× | 0.2942 | 1.5909 | 0.6318 | 0.0000 | 0 | 0.1296 | 0.4211 | 0.5593 | 54.8 | 0.031 | 0.037 | 0 | 0 | 0 |
| 23 | OpenJev Verdict 151M | 0.3471 | -0.5386 | 23.71× | 0.4946 | 4.1188 | 0.7152 | 0.0081 | 0 | 0.1296 | 0.3158 | 0.4979 | 27.1 | 0.018 | 0.020 | 0 | 0 | 0 |

Metric notes:
- Acc / BalAcc: top-label accuracy over labeled rows; BalAcc averages class recall.
- NLL / Brier: probability quality; lower is better.
- Cov@0.9: fraction with max probability ≥0.9; CW@0.9 is the count of confident errors.
- Perturb flip: argmax changes across the 108 SemIf perturbation pairs.
- Every R@1/MRR: retrieval metrics on the 19-query Every subset.
- Δ vs Jev: candidate accuracy minus Official Jev direct accuracy.
- Speed×: Official Jev direct wall / candidate wall; >1 means faster on this bundle.
- Err: receipt rows with transport/runtime/schema errors.

Bundle: [`data/jevbench_offline_bundle_v1`](data/jevbench_offline_bundle_v1) (manifest SHA-256 `1fc3234acf806016adfd9906fade547ca597f87acf749aecac1ba706f6dd8ee5`).

## Domain-specific decision-head ladder (Valen training stack)

**Domain-specific evals — not the general JevBench task.** Second-generation experiment: lightweight decision heads trained on the open-source [Valen](https://github.com/Valen-Team) SFT+RLCD stack (Qwen3.5 backbones, frozen for head-only stages). All numbers below were measured on a single A100-80G / Apple Silicon MPS under identical evaluation code; official Jev was called through the direct TypeSafe API (`jev-latest`, reported version `jev-1.13.0`). Aggregates only — raw per-item outputs remain local. Same ladder as [README.md — Domain-trained decision heads](README.md#domain-trained-decision-heads-valen-training-stack); error analysis and the planned rebuild are tracked in the W132/W133 cross-roadmap note in [`docs/NANOJEV_V2_ROADMAP.md`](docs/NANOJEV_V2_ROADMAP.md).

### Context-filter task — `valen_nano_v1` held-out eval (678 noul questions)

Domain-specific. Own CC0-licensed data (`context_relevance_v1` + `oracle`; world variants split across train/eval). The target use case: candidate-context relevance filtering.

| Scorer | Acc | Brier ↓ | NLL ↓ | Notes |
|---|---:|---:|---:|---|
| **nano_rlcd_v2** (0.8B, SFT→RLCD) | **0.9012** | **0.1349** | **0.2058** | current `:8093` sidecar backend |
| nano_rlcd_2b (2B, SFT→RLCD) | 0.8953 | 0.1521 | 0.2529 | RLCD +3.4pp over its SFT init |
| nano_sft_v2 (0.8B, 12 ep) | 0.8894 | 0.1570 | 0.2380 | |
| nano_rlcd_v1 (0.8B, RLCD pilot) | 0.8746 | — | — | ECE 0.0377 |
| nano_sft_2b (2B, 12 ep) | 0.8614 | 0.2032 | 0.3250 | 2B underperforms 0.8B here |
| Winnow-12B Q8 (production) | 0.8599 | — | — | ECE 0.1191 |
| Official Jev (`jev-1.13.0`) | 0.7341 | — | — | ECE 0.1038; 662/678 answered, 16 API timeouts (~5 s) |
| nano_sft_v1 (0.8B, 4 ep) | 0.6445 | — | — | undertrained reference |
| Valen-Preview-0923 head | 0.5100 | — | — | general-domain head, no transfer |
| untrained head | 0.5220 | — | — | |

A 0.8B head trained on ~10k in-domain records beats official Jev by +16.7pp on its own task — and beats it on latency (~0.9 s local vs ~1.5 s remote, free and offline). RLCD adds measurable gains on top of converged SFT (+1.2pp acc, better Brier/NLL); on the undertrained 2B init the lift is +3.4pp. Official-Jev accuracy is over answered items only; scoring the 16 timeouts as failures gives 0.7168 (486/678).

**Label-contract caveat (error analysis):** the `valen_nano_v1` generator has a label-contract contradiction — v1 `correction` (keep) and v1 `overlap_distractor` (drop) produce normalized-identical states with opposite labels, covering 12.7% of records (427/3,360). The eval ceiling is therefore **0.9572** under pointer-aware normalization (the earlier ~0.9322 estimate over-merged pointer indices); the current 0.9012 leaves ~5.6pp of real headroom, and more training on contradictory labels only teaches noise. A corrected **`valen_nano_v2` rebuild is delivered** (0 violations, eval ceiling 1.0; roadmap tasks T163→T164/T165→T167; W132/W133/W134 note in [`docs/NANOJEV_V2_ROADMAP.md`](docs/NANOJEV_V2_ROADMAP.md)).

**⚠ v1 label caveat (W134 clean-label re-eval):** the v1 numbers in the table above carry up to ~1.3pp of label-noise inflation. On the fixed `valen_nano_v2` eval (same-device MPS bf16, 678 questions) the ordering is **nano_sft_v2 0.8938 > nano_rlcd_2b 0.8923 > nano_rlcd_v2 0.8879** — rlcd_v2's v1 lead was partly memorization of the contradictory labels (31 eval records relabeled; rlcd_v2 went 16→15 correct on them = net loss). The expanded `valen_nano_v3` rebuild (12,122 train / 2,950 eval, 38.5% hard negatives) is delivered — results in the archived v3 table below. Detail: [`docs/VALEN_NANO_V2_REEVAL_V1.md`](docs/VALEN_NANO_V2_REEVAL_V1.md).

**Device-drift caution:** bf16 forward results drift across hardware — identical weights and identical requests produced 50/678 different predicted labels between A100 and MPS. Cross-device accuracy deltas of ~±0.01 (~±1pp) are device noise; cross-device comparisons require a same-device re-measurement. (The archived v3 fp32 parity check below shows this drift disappears under fp32.)

### Context filtering — `valen_nano_v4` (4,251 held-out questions)

The task: given a user request, conversation history, and one candidate
context snippet, judge whether the snippet is irrelevant (safe to drop) or
needed (must keep). Two error directions: false-drop loses evidence (high
risk); missed-drop wastes context (low risk).

This eval supersedes v3 — **do not compare its numbers with the
678-question v1 table above** or with the archived v3 numbers below.
`valen_nano_v4` keeps v3's 2,950 clean rows and adds 1,301 new
hard-family rows (tool-result dependencies, cross-pointer evidence,
long-context dilution, topic shifts, adversarial restatements) for 4,251
eval questions; data sha256 prefix in `data/valen_nano_v4/manifest.json`.
NanoJev heads are an fp32 forward on A100; Winnow/Kev run on a local Mac;
JEMM-27B is bf16 sharded across 2×L40; official Jev via its hosted API
(`jev-latest` = 1.13.0). Aggregates only — no per-item outputs.

| Scorer | What it is | Accuracy | False-drops | Missed-drops | p50 latency |
|---|---|---:|---:|---:|---:|
| nano_sft_text_v4 | NanoJev 0.8B + LoRA-tuned backbone | 1.0000 | 0 | 0 | 70ms |
| nano_rlcd_v4 | NanoJev 0.8B, preference-optimized | 0.8946 | 245 | 203 | 47ms |
| Winnow-12B Q8 | local 12B general model | 0.8292 | 15 | 711 | 542ms |
| JEMM-27B | external head-free 27B baseline | 0.8165 | 148 | 632 | 192ms |
| Kev | local encoder backend | 0.8022 | 375 | 466 | 1116ms |
| nano_sft_v4 | NanoJev 0.8B head-only | 0.7944 | 765 | 109 | 45ms |
| Official Jev 1.13.0 | TypeSafe cloud API | 0.6765 | 0 | 1375 | 687ms |

**Notes:**

1. Accuracy is the fraction answered correctly. False-drop (FP on the drop
   decision) = a needed snippet wrongly marked irrelevant — it loses
   evidence. Missed-drop (FN) = an irrelevant snippet wrongly kept — it
   wastes context. Latency is the median per request; per-row hardware is
   listed in the paragraph above.
2. Official Jev's failure is systematic, not noise: 0 false-drops against
   1,375 missed-drops — it almost never answers "irrelevant". JEMM shows
   the same keep-bias direction (632 FN vs 148 FP) but weaker.
3. The legacy v2 heads and the head-free readout controls were measured on
   the v3 subset only — see the archived v3 table below; they are not
   carried forward.
4. A perfect 1.0000 means this synthetic suite is saturated again —
   including the new hard families — not that production performance is
   proven; a real-transcript eval is still pending.
5. JEMM-27B reproduced across hardware: 0.8165 on A100 vs 0.8167 on L40×2
   (25/4,251 decision flips) — same-weights cross-device consistency
   confirmed.

**Saturation caveat:** `nano_sft_text_v4`'s 1.0000 means this *synthetic* benchmark is saturated again — now including the new hard families. It does **not** prove production performance; the production context-filter backend is unchanged, and specialist heads still do not transfer out-of-domain (see the JevBench check below).

#### Archived: `valen_nano_v3` eval (2,950 held-out questions)

Superseded by the v4 table above; kept for the legacy v2-head and
readout-control rows not carried forward, and for the fp32 cross-device
parity check.

The task: given a user request, conversation history, and one candidate
context snippet, judge whether the snippet is irrelevant (safe to drop) or
needed (must keep). Two error directions: false-drop loses evidence (high
risk); missed-drop wastes context (low risk).

This eval set is newer and harder — **do not compare its numbers with the
678-question v1 table above.** The `valen_nano_v3` rebuild removes the v1
label-contract contradiction (0 violations, eval ceiling 1.0) and expands to
12,122 train / 2,950 eval questions with 38.5% hard negatives
(`data_sha256 c95286ff00512b28…`). All local models run on Apple Silicon
(fp32); official Jev via its hosted API (`jev-latest` = 1.13.0, 0 timeouts).
Aggregates only — no per-item outputs.

| Scorer | What it is | Accuracy | False-drops | Missed-drops | p50 latency |
|---|---|---:|---:|---:|---:|
| nano_sft_text_v3 | in-house 0.8B model, LoRA-tuned backbone | 1.0000 | 0 | 0 | 228ms |
| nano_rlcd_v3 | in-house 0.8B model, preference fine-tuned | 0.9847 | 34 | 11 | 213ms |
| nano_sft_v3 | in-house 0.8B model, decision head only | 0.9173 | 5 | 239 | 219ms |
| Kev | local general-purpose encoder backend | 0.8600 | 177 | 236 | 553ms |
| Winnow-12B Q8 | local 12B general LLM | 0.8166 | 7 | 534 | 425ms |
| nano_sft_v2 | previous-gen head, trained on v1 data | 0.7268 | 235 | 571 | 300ms |
| nano_rlcd_v2 | previous-gen head, current production | 0.7210 | 307 | 516 | 266ms |
| Official Jev 1.13.0 | TypeSafe hosted API | 0.6871 | 0 | 923 | 625ms |
| LM-head readout, LoRA backbone | label-token readout control arm | 0.5620 | — | — | — |
| LM-head readout, raw backbone | untrained blank control | 0.5530 | — | — | — |

**Notes:**

1. Accuracy is the fraction answered correctly. False-drop (FP on the drop
   decision) = a needed snippet wrongly marked irrelevant — it loses
   evidence. Missed-drop (FN) = an irrelevant snippet wrongly kept — it
   wastes context. Latency is the median per request; local rows measured
   on Apple Silicon fp32, official Jev via its hosted API.
2. Official Jev's failure is systematic, not noise: 0 false-drops against
   923 missed-drops — it almost never answers "irrelevant". It over-keeps
   on hard negatives.
3. The bottom two rows are control arms: the same backbone without the
   trained head collapses to always answering "keep" — at 0.8B the trained
   head carries the skill.
4. A perfect 1.0000 means the synthetic benchmark is fully learned
   (saturated), not that production performance is proven; a real-traffic
   eval is still pending.

**Saturation caveat:** `nano_sft_text_v3`'s 1.0000 means this *synthetic* benchmark is saturated — the template family is fully learned. It does **not** prove production performance; the production context-filter backend is unchanged, and specialist heads still do not transfer out-of-domain (see the JevBench check below).

**Cross-device parity (fp32):** A100 fp32 vs Apple Silicon MPS fp32 — `nano_rlcd_v3` 0.9858 → 0.9847 (3/2,950 label flips, max Δp 0.051); `nano_sft_text_v3` 1.0000 → 1.0000 (0 flips, max Δp 4e-05). The earlier bf16 comparison produced ~50/678 flips; fp32 computation (LayerCast-style) eliminates decision drift.

### General-domain check — JevBench public 231 questions

The same specialist head on the official JevBench public split (139 choice / 74 noul / 18 score). Specialist heads do **not** transfer out-of-domain — the symmetric result to the v1 table above:

| Scorer | Acc |
|---|---:|
| Official Jev direct* | ~0.886 |
| Winnow-12B Q8* | 0.8824 |
| OmniJev-4B | 0.688 |
| OmniJev-2B / NanoJev MiniCPM seed20 | 0.654 |
| OmniJev-0.8B | 0.524 |
| **nano_rlcd_v2** | **0.307** |

`*` measured on the pinned offline bundle containing these rows (the 1,720-row scorecard above). Specialist-head scores by type: noul 0.473, choice 0.216, score 0.333.

### Financial regime task — `valen_fin_v1` eval (3,000 records / 9,000 questions)

Domain-specific. Crypto-perp daily decisions: `regime_gate` (read `btc_ret20` from state), `fwd5_bucket` (5-day forward-return bucket, chance 0.25), `xs_outperform_5d` (cross-sectional median beat, chance 0.50).

| Scorer | Overall | fwd5 | regime | xs5d |
|---|---:|---:|---:|---:|
| fin_sft_v1 (0.8B) | **0.6202** | 0.329 | 0.9997 | 0.532 |
| fin_rlcd_v1 (0.8B) | 0.6086 | 0.300 | 1.0000 | 0.526 |
| OmniJev-4B | 0.435 | 0.174 | 0.648 | 0.484 |

`regime_gate` is largely state-reading, not market prediction; the real outcome-prediction questions (`fwd5`, `xs5d`) are weak — consistent with the project's paper-ledger findings that regime/ranking structure carries the signal.

Domain notes:

- **Specialized accelerator, not a general Jev replacement.** The architecture generalizes; the priors are domain-specific. A small in-domain head wins on its own task (0.9012 vs 0.7341) and loses everywhere else (0.307 on JevBench public). Winnow-12B Q8 and official Jev remain the general-path scorers; the production service keeps Winnow as the general default and routes the context-filter question to the specialist head.
- Acc / Brier / NLL as in the scorecard above; ECE is expected calibration error where recorded. `—` marks metrics not captured for that run.
- Official Jev rows are remote API calls (`jev-latest`, reported `jev-1.13.0`); all other rows are local heads.
