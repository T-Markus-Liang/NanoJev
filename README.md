# NanoJev — A nano replica of [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)

**English** | [简体中文](README.zh-CN.md)

**A 0.6B parallel decision model. States and questions in, complete probability distributions out—with zero output-token decoding.**

[Model](https://huggingface.co/C-Tianyu/NanoJev) · [Dataset](https://huggingface.co/datasets/C-Tianyu/NanoJev-Data)

## Recorded NanoJev measurements

The repository publishes NanoJev measurements and aggregate comparison metrics under
pinned, disclosed protocols. Raw provider outputs and private side-by-side receipts are
retained locally and are not part of the public source tree.

### Find the exit: 50×50 maze

The model judges four local directions. Code remembers collisions, explores untried
edges, and repositions through verified open paths. In the recorded NanoJev run it
reached the goal in **244 attempts with 36 collisions**.

### Keep growing: 12×12 Snake

The common planner filters immediate collisions and finds static paths toward the visible
food. The model breaks ties between the remaining actions; a single remaining action is
a code-forced move. In the recorded NanoJev run (**seed 61005**, greedy controller), it
collected **27 food over 256 steps** and remained alive at the evaluation horizon.

## Local decision benchmark

The repository includes a frozen **1,720-row / 893-labeled** offline bundle for typed decisions: official JevBench public rows, SemIf authored/perturbation/shape fixtures, WANLI, and Every retrieval. All runs are evaluation-only; aggregate metrics are published without raw provider responses or per-item receipts.

| Rank | System | Acc | Δ vs Jev | Speed× | BalAcc | NLL ↓ | Brier ↓ | Cov@0.9 | CW@0.9 | Perturb flip | Every R@1 | Wall s | p50/p95 s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | Official Jev direct | **0.8858** | — | 1.00× | **0.8443** | **0.9648** | **0.1775** | 0.7180 | **29** | **0.0000** | **1.0000** | 642.7* | 0.292 / 0.729 |
| 2 | **Winnow-12B Q8** | **0.8824** | **-0.0034** | **1.30×** | **0.8374** | **0.9295** | 0.2069 | **0.8744** | 59 | 0.0278 | **1.0000** | **494.6** | **0.152 / 0.831** |
| 3 | Kev-9B | 0.8186 | -0.0672 | 0.92× | 0.7548 | 0.9985 | 0.2599 | 0.5058 | 35 | 0.0093 | 1.0000 | 700.4 | 0.174 / 1.444 |
| 4 | Kev-4B | 0.8018 | -0.0840 | 1.07× | 0.7048 | 1.1326 | 0.3055 | 0.4942 | 47 | 0.0185 | 1.0000 | 601.5 | 0.128 / 1.399 |
| 5 | Decider-2B | 0.7917 | -0.0941 | 0.61× | 0.6875 | 1.1044 | 0.3142 | 0.5558 | 43 | 0.0370 | 1.0000 | 1051.7 | 0.282 / 1.163 |
| 6 | Reflex stable | 0.7895 | -0.0963 | **2.61×** | 0.7803 | 1.0509 | 0.2913 | 0.2343 | **10** | 0.0741 | 1.0000 | **246.4** | **0.089 / 0.404** |
| 7 | this-that-model-1.0 | 0.7850 | -0.1008 | 0.62× | 0.6366 | 1.4306 | 0.3658 | 0.7436 | 101 | 0.0278 | 0.9474 | 1037.9 | 0.229 / 1.195 |
| 8 | SemIf Qwen3.5-4B MLX4 | 0.7671 | -0.1187 | 1.57× | 0.7423 | 1.1401 | 0.3365 | 0.5384 | 39 | 0.1389 | 1.0000 | 409.6 | 0.099 / 0.450 |
| 13 | NanoJev MiniCPM seed20 | 0.6529 | -0.2329 | 0.26× | 0.6694 | 0.7705 | 0.4603 | 0.2721 | 21 | 0.0833 | 0.9474 | 2519.0 | 0.355 / 3.031 |

`*` Official Jev direct is remote response latency; local wall times are not strictly speed-comparable. `Cov@0.9` is coverage at confidence threshold 0.9; `CW@0.9` is the number of confident errors. Full scorecard: [BENCHMARKS.md](BENCHMARKS.md). Bundle: [`data/jevbench_offline_bundle_v1`](data/jevbench_offline_bundle_v1) (manifest SHA-256 `1fc3234acf806016adfd9906fade547ca597f87acf749aecac1ba706f6dd8ee5`).

## Domain-trained decision heads (Valen training stack)

Second-generation experiment: lightweight decision heads trained on the open-source
[Valen](https://github.com/Valen-Team) SFT+RLCD stack (Qwen3.5 backbones, frozen for
head-only stages). All numbers below were measured on a single A100-80G / Apple
Silicon MPS under identical evaluation code; official Jev was called through the
direct TypeSafe API (`jev-latest`, reported version `jev-1.13.0`). Aggregates only —
raw per-item outputs remain local.

### Context-filter task — `valen_nano_v1` held-out eval (678 noul questions)

Own CC0-licensed data (`context_relevance_v1` + `oracle`; world variants split
across train/eval). The target use case: candidate-context relevance filtering.

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

**A 0.8B head trained on ~10k in-domain records beats official Jev by +16.7pp on
its own task** — and beats it on latency (~0.9 s local vs ~1.5 s remote, free and
offline). RLCD adds measurable gains on top of converged SFT (+1.2pp acc, better
Brier/NLL); on the undertrained 2B init the lift is +3.4pp.

*Label caveat: v1 numbers carry up to ~1.3pp label-noise inflation — a clean-label `valen_nano_v2` re-eval and `valen_nano_v3` rebuild are tracked in `BENCHMARKS.md` / `docs/VALEN_NANO_V2_REEVAL_V1.md`.*

### Context filtering — `valen_nano_v4` (4,251 held-out questions)

The task: given a user request, conversation history, and one candidate
context snippet, judge whether the snippet is irrelevant (safe to drop) or
needed (must keep). Two error directions: false-drop loses evidence (high
risk); missed-drop wastes context (low risk).

This eval supersedes v3 — the 4,251 questions are v3's 2,950 clean rows
plus 1,301 new hard-family rows. **Do not compare its numbers with the
678-question v1 table above** (data sha256 prefix in
`data/valen_nano_v4/manifest.json`).

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

1. Accuracy is the fraction answered correctly. A false-drop is a needed
   snippet wrongly marked irrelevant — it loses evidence. A missed-drop is
   an irrelevant snippet wrongly kept — it wastes context. The 4,251
   questions are v3's 2,950 rows plus 1,301 new hard-family rows
   (tool-result dependencies, cross-pointer evidence, long-context
   dilution, topic shifts, adversarial restatements). Latency is the
   median per request: NanoJev heads are an fp32 forward on A100; Winnow
   and Kev run on a local Mac; JEMM is bf16 sharded across 2×L40; official
   Jev is the API round-trip. Data sha256 prefix in
   `data/valen_nano_v4/manifest.json`.
2. The 1.0000 means this synthetic suite is saturated again — including
   the new hard families. A real-transcript eval is still pending; treat
   it as a learnable-benchmark ceiling, not production proof.
3. Official Jev's failure stays structural: 0 false-drops against 1,375
   missed-drops — it almost never judges content droppable. JEMM shows the
   same keep-bias direction (632 missed vs 148 false) but weaker.
4. Legacy v2 heads and the head-free readout controls were measured on the
   v3 subset only — see the archived v3 table below; not carried forward.
5. JEMM-27B reproduced across hardware: 0.8165 on A100 vs 0.8167 on L40×2
   (25/4,251 decision flips) — same-weights cross-device consistency
   confirmed.

#### Archived: `valen_nano_v3` eval (2,950 held-out questions)

Superseded by the v4 table above; kept for the legacy v2-head and
readout-control rows that were not carried forward.

The task: given a user request, conversation history, and one candidate
context snippet, judge whether the snippet is irrelevant (safe to drop) or
needed (must keep). Two error directions: false-drop loses evidence (high
risk); missed-drop wastes context (low risk).

This eval set is newer and harder — **do not compare its numbers with the
678-question v1 table above** (`data_sha256 c95286ff00512b28…`).

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

1. Accuracy is the fraction answered correctly. A false-drop is a needed
   snippet wrongly marked irrelevant — it loses evidence. A missed-drop is
   an irrelevant snippet wrongly kept — it wastes context. Latency is the
   median per request; local rows were measured on Apple Silicon fp32,
   official Jev via its hosted API.
2. Official Jev's failure is systematic, not noise: 0 false-drops against
   923 missed-drops — it almost never answers "irrelevant". It over-keeps.
3. The bottom two rows are control arms: the same backbone without the
   trained head collapses to always answering "keep" — at 0.8B the trained
   head carries the skill.
4. A perfect 1.0000 means the synthetic benchmark is fully learned
   (saturated), not that production performance is proven; a real-traffic
   eval is still pending.

### General-domain check — JevBench public 231 questions

The same head on the official JevBench public split (139 choice / 74 noul / 18
score). Specialist heads do **not** transfer out-of-domain — the symmetric result
to the v1 table above:

| Scorer | Acc |
|---|---:|
| Official Jev direct* | ~0.886 |
| Winnow-12B Q8* | 0.8824 |
| OmniJev-4B | 0.688 |
| OmniJev-2B / NanoJev MiniCPM seed20 | 0.654 |
| OmniJev-0.8B | 0.524 |
| **nano_rlcd_v2** | **0.307** |

`*` measured on the pinned offline bundle containing these rows. Specialist-head
scores by type: noul 0.473, choice 0.216, score 0.333.

### Financial regime task — `valen_fin_v1` eval (3,000 records / 9,000 questions)

Crypto-perp daily decisions: `regime_gate` (read `btc_ret20` from state),
`fwd5_bucket` (5-day forward-return bucket, chance 0.25), `xs_outperform_5d`
(cross-sectional median beat, chance 0.50).

| Scorer | Overall | fwd5 | regime | xs5d |
|---|---:|---:|---:|---:|
| fin_sft_v1 (0.8B) | **0.6202** | 0.329 | 0.9997 | 0.532 |
| fin_rlcd_v1 (0.8B) | 0.6086 | 0.300 | 1.0000 | 0.526 |
| OmniJev-4B | 0.435 | 0.174 | 0.648 | 0.484 |

`regime_gate` is largely state-reading, not market prediction; the real
outcome-prediction questions (`fwd5`, `xs5d`) are weak — consistent with the
project's paper-ledger findings that regime/ranking structure carries the signal.

**Takeaway:** the architecture generalizes; the priors are domain-specific. A small
in-domain head wins on its own task (0.90 vs 0.73) and loses everywhere else —
which is exactly why the production service keeps Winnow as the general default
and routes the context-filter question to the specialist head.

## Features

- **0.6B LLM backbone.** Qwen3-0.6B with decision heads for structured outputs.
- **Multiple states and questions in one forward.** Batch independent decisions together.
- **Dynamic Choice.** Supply **2–255 candidates** and receive a probability for every candidate.
- **Boolean decisions.** Receive the probability that a complete proposition is true.
- **Ordered Score.** Supply **2–10 levels** and receive the level distribution and expected score.
- **Complete distributions.** Use the same output for ranking, greedy selection, or probability sampling.
- **Zero output decoding.** Read decisions directly from a forward pass.
- **Persistent serving.** Load a checkpoint once and reuse it across requests.

Measured in the running service: **6 states · 18 questions · 44 candidate paths · 1 backbone forward**.

## Larger games and calibrated decisions

- **Full-size environments:** 8×8, 16×16, 32×32, and 50×50 mazes, four topologies, multiple positions per map, and configurable larger sizes.
- **Local judgments + code planning:** matched 5×5 observations, four parallel safety judgments, movement memory, and model-guided exploration.
- **Snake dynamics:** reproducible food generation, body growth, collision rules, tail movement, dynamic action candidates, and safety questions.
- **Probability learning:** observed-event datasets, CE/Brier training, paired proper-reward learning, exact gradient checks, and completed Qwen3-0.6B runs.
- **Verified evaluation:** map-separated data, frozen game cohorts, real model execution, and independent trajectory replay.

The local safety model reaches **77.84% accuracy on test questions** and **76.56% on 50×50 OOD questions**. The probability-learning pilot's paired proper-reward arm reaches **0.11844 test / 0.06202 OOD distribution error**, measured as the sum of squared differences from the simulator's event probabilities.

[RLCD implementation and results](docs/RLCD_EXPERIMENT.md) · [Input contract](docs/TYPESAFE_CONTRACT.md) · [V2 roadmap](docs/NANOJEV_V2_ROADMAP.md)

## Earlier 40-map NanoJev navigation benchmark

With T=1 probability sampling, NanoJev completed **19/20 (95%)** of the 4×4 test maps
and **18/20 (90%)** of the 6×6 OOD maps. These are NanoJev-only measurements on the
frozen 20-test/20-OOD cohort.

## How it works

Each decision is defined by a **state**, a **question**, and its **candidate set**. Every candidate path carries the relevant input into the backbone. Shared decision heads return a distribution over the candidates supplied for that question.

Choice uses a shared scalar head and set attention. Boolean uses a single-path sigmoid. Score evaluates its ordered level descriptions and returns their probability-weighted expectation.

1. **Build queries.** Generate states, questions, candidate descriptions, and target distributions.
2. **Organize data.** Keep related maps, rules, and their variations in the same split.
3. **Train.** Initialize Qwen3-0.6B, warm up the decision heads, and train with complete-question distribution losses.
4. **Evaluate.** Measure probability quality and execute game controllers with recorded actions.
5. **Serve and visualize.** Reuse a persistent model endpoint and replay complete trajectories in the browser.

[Complete pipeline commands](research/pipeline_runbook.md)

## Download the showcase models

| Use | Checkpoint in [C-Tianyu/NanoJev](https://huggingface.co/C-Tianyu/NanoJev/tree/main/variants) |
|---|---|
| **50×50 maze demo** | `variants/local_atomic_seed17` |
| **Snake demo** | `variants/games_gold_seed17` |
| Full-map evaluation | `variants/games_api_seed17` |
| Calibrated-decision experiments | `variants/events_ce_seed17`, `variants/events_brier_seed17`, `variants/events_paired_seed17` |

```python
from pathlib import Path
from huggingface_hub import snapshot_download

variant = "local_atomic_seed17"  # Select "games_gold_seed17" for Snake.
snapshot = snapshot_download(
    repo_id="C-Tianyu/NanoJev",
    allow_patterns=[f"variants/{variant}/*"],
)
checkpoint_dir = Path(snapshot) / "variants" / variant
```

The [game data package](https://huggingface.co/datasets/C-Tianyu/NanoJev-Data/tree/main/games_v4) contains the matching training splits and frozen evaluation inputs.

## Download and run the model

The [model](https://huggingface.co/C-Tianyu/NanoJev) and [dataset](https://huggingface.co/datasets/C-Tianyu/NanoJev-Data) are public. Prepare an NVIDIA CUDA or Apple Silicon environment with the recorded [Python dependencies](requirements-toy.txt):

```bash
python -m pip install -r requirements-toy.txt
```

Download the base release checkpoint and dataset. The root checkpoint is the initialization model and the earlier navigation baseline:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="C-Tianyu/NanoJev", local_dir="checkpoints/NanoJev",
    allow_patterns=["best.safetensors", "config.json", "tokenizer/*", "backbone_config/*"],
)
snapshot_download(
    repo_id="C-Tianyu/NanoJev-Data", repo_type="dataset", local_dir="data/NanoJev",
)
```

Start the persistent service:

```bash
python scripts/serve_decisions.py \
  --checkpoint-dir checkpoints/NanoJev \
  --web-root web --port 8765
```

Open **http://127.0.0.1:8765**. The service loads the model once and accepts repeated batches through **`POST /api/evaluate`**.

Apple Silicon checkpoint inference uses MPS with FP32. See [Apple Silicon inference](docs/APPLE_SILICON.md).

The [pipeline runbook](research/pipeline_runbook.md) covers data generation, training, evaluation, checkpoint creation, and continuing from the downloaded model and data.

## Roadmap

The active plan is maintained in [NANOJEV_V2_ROADMAP.md](docs/NANOJEV_V2_ROADMAP.md). Post-W45, the critical path is architecture convergence and independent external evaluation rather than another bespoke multi-model leaderboard.

- [x] **Establish the local baseline** — Independent heldout, three-seed head/full/LoRA runs, and grouped-calibration diagnosis.
- [ ] **Converge the readout** — Compare direct logits, the current LoRA head, and LoRA+pointer; require candidate-order and shared-prefix parity.
- [ ] **Expand independent evidence** — Build corpus v4 and a larger engineering heldout without benchmark/provider outputs.
- [ ] **External validation** — Add a pinned [JevBench](https://github.com/fstandhartinger/jevbench) adapter, run a frozen public milestone, then request maintainer-held-out evaluation.

Architecture reference: [SemIf](https://github.com/TheoLeeCJ/SemIf). Local application/runtime reference: [laya-mlx](https://github.com/mizorewww/laya-mlx). External benchmark and leaderboard: [JevBench on Benchmark Heaven](https://benchmarkheaven.com/jev-models).
