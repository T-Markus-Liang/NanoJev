# Domain adaptation runbook V1 — retraining the decision model to fix the abstention

**Status: runbook only. This document authorises nothing.** It records how a retraining run
*would* be executed, gated and rolled back. Nothing was trained while writing it, no checkpoint
or corpus was written, and no accuracy, abstention or readiness result is claimed. The current
production reference checkpoint remains
`checkpoints/local_atomic_seed17/variants/local_atomic_seed17`.

Companion machine-readable record of everything actually executed:
[`results/training_feasibility_v1.json`](../results/training_feasibility_v1.json).

---

## 0. How to read this document

Every substantive claim carries one of four labels:

| Label | Meaning |
|---|---|
| **[VERIFIED]** | Observed by a command run on this machine; the command and its exit code are in `results/training_feasibility_v1.json`. |
| **[READ]** | Established by reading repository source or a frozen document without executing it. |
| **[ESTIMATE]** | Arithmetic projection from measured components. Not an observation. |
| **[ASSUMED]** | A design choice or an open question, flagged as such so the review gate can accept or reject it. |

The distinction is load-bearing: this runbook is written *before* the run precisely so that the
numbers cannot be chosen after seeing results.

---

## 1. The problem this runbook addresses

The local checkpoint abstains on 13/13 engineering-judgment questions at the documented 0.90
confidence gate (max confidence 0.736), while in-domain it reaches median confidence 0.779 and
the ≥0.9 bucket is 97.8% correct. **[READ]** —
`docs/SKILL_ABSTENTION_DIAGNOSIS_V1.md`, `research/skill_abstention_survey_v1.json`
(`measured_2026_09_19.abstained_at_0_9: "13/13"`, `max_confidence: 0.736`,
`accuracy_on_expected: "3/6"`), and the "Real-checkpoint proposals remain zero" paragraph of
`docs/NANOJEV_V2_ROADMAP.md`.

Two repairs are already excluded, and this runbook does not re-open them **[READ]**:

* **Temperature scaling.** Strictly monotone, so it cannot reorder confidences; fitted
  T = 0.854 lifts the survey maximum only to 0.769, still below 0.90; the fitted value is
  unstable across non-test splits. `results/temperature_fit_v1.json`.
* **Prompt/surface-form workaround.** Padding or rewording moves in-domain confidence by
  ≤ 0.021 and shortening an out-of-domain question does not raise it.
  `docs/CONFIDENCE_ATTRIBUTION_V1.md`.

The stated remaining path is **domain-adaptation training data** under the pre-registered
protocol `docs/GATE_CONTRASTIVE_PROTOCOL_V1.md` **[READ]**. This runbook is the execution half of
that decision.

---

## 2. Preconditions and blockers as of authoring time

The training *machinery* is ready and cheap on this machine. The *data* is not.

### 2.1 Blocker A — there is no working engineering corpus **[VERIFIED]**

At authoring time the sibling builder `scripts/build_engineering_corpus_v1.py` (untracked
working-tree file, sha256 `17b90ad406e35cf2a0f8c84e28d1c53784661784e8d4157d34c05e32eaedbcaa`)
failed its own no-write self-test:

```
$ .venv/bin/python scripts/build_engineering_corpus_v1.py --self-test
{"status": "failed", "pairs": 0, "items": 0, "source_groups": 0, "wrote_output": false,
 "errors": ["ValueError: lifecycle-measurement-first#1: the one-fact change to
            'invariant_verified' does not flip 'next_phase' ('measurement' -> 'measurement')"]}
exit code 1
```

On re-check minutes later the file had been **removed from the working tree entirely** by its
owning workstream (no `engineering` file remains under `scripts/`, and `git status` no longer
lists it) **[VERIFIED]**. No `research/engineering_judgment_corpus_v1/` exists. The workspace is
shared and concurrently edited; the training-critical inputs were re-verified afterwards and are
unchanged (§10.4).

The builder's declared composition was 26 hand-authored source groups expanded into one-fact
contrastive pairs across `train`/`dev`/`calibration`/`test` **[READ]**. Until **an** engineering
corpus builder passes its own `--self-test` and `--check`, there is no engineering training data
and **nothing in this runbook can be run**. This runbook therefore names the artifact by *role*,
not by path: the data prerequisite is "a reviewed engineering-judgment corpus builder, green on
self-test, materialising trainer-consumable splits". Blocker A is owned by the sibling
workstream, not by this runbook, and Step 1 refuses to proceed without it.

### 2.2 Blocker B — no corpus→trainer adapter exists **[VERIFIED + READ]**

Neither builder emits the row schema that `scripts/train_pipeline_decisions.py --input` requires
(`validate_training_row` demands top-level `id`, `state_id`, `family_id`, `split`, `state`,
`questions`, plus `gold` or `gold_probs`):

| Corpus | Emitted shape | Trainer-ready? |
|---|---|---|
| engineering corpus builder (sibling workstream; path not stable) | JSONL items `{schema_version, item_id, family, feature, question_type, state_id, qid, request:{states:[…]}, expected:{…}, provenance:{…}, contrastive:{…}}` | **No** — needs re-wrap |
| `build_gate_contrastive_v1.py` | `contrastive_manifest.json` + `pairs/<id>/{keep,remove}/{request,sidecar,expected}.json` | **No** — needs re-wrap |

**[VERIFIED]** that no such adapter is present (grep for the row schema across `scripts/`), and
**[VERIFIED]** that the engineering builder self-declared
`corpus_role: training_data_prerequisite_only` and `training_authorized_by_this_corpus: false`
with `labels_human_reviewed: false` before it was withdrawn. Section 5 specifies the adapter the
review gate must approve.

Additional arithmetic constraint **[VERIFIED from the builder self-test]**: the default gate
contrastive cohort's `train` split is 6 pairs = 12 items. If each member becomes one question,
12 < the frozen `--batch-questions 16`, so the trainer aborts with *"Fewer eligible training
questions than one effective batch"*. The default cohort therefore cannot be trained at the
frozen batch size; the cohort must be scaled through `--base-fixture` **before review**
(protocol §4) or the batch size must be amended.

### 2.3 Blocker C — the pre-registered recipe is not executable on this machine **[VERIFIED]**

Protocol §7 pre-registers: *LoRA rank 16, learning rate 5e-5, effective batch 8, seed 17 plus
matched 18 and 19, one epoch, BF16, cross-entropy over allowed answer tokens only, 2,048-token
prompt cap.*

* **LoRA is absent.** No `lora`/`peft` implementation or dependency exists in `scripts/` or
  `requirements-toy.txt` **[VERIFIED]**. The only trainer fine-tunes the full 596,250,498-parameter
  backbone with AdamW.
* **BF16 is refused on MPS.** `resolve_runtime` raises on `--precision bf16` for `mps`
  **[VERIFIED]**.
* There is no CUDA device on this machine **[VERIFIED]**, and no epoch concept in the runtime
  **[READ]**.

Per protocol §1.1, *"any change after review is a new protocol version and must be re-reviewed."*
Section 3.2 below states the amendment this runbook proposes so the review gate has one concrete
object to accept or reject. This runbook **does not** claim the amendment is acceptable.

### 2.4 What is already in place **[VERIFIED unless noted]**

* The exact production training corpus is present:
  `dataset/games_v4/data/local_maze_v1` — all five split files sha256-match the checkpoint
  config's `data_sha256`, i.e. byte-identical to the training-time `data/local_maze_v1`.
  The config's recorded `input: data/local_maze_v1` is a stale path.
* The warm-start source is the local checkpoint itself; config's `init_checkpoint`
  (`/home/rwang/openjev_codex_20260917/…`) is a Linux path absent here.
* `--self-check` passes: schema, complete-question budget and the grouped-soft-CE / padding-mask
  / question-mean / Boolean-derivative gradient checks.
* `--validate-only` accepts four existing trainer-shaped corpora
  (`dataset/games_v4/data/local_maze_v1`, `data/context_relevance_oracle_v1_seed20260919`,
  `data/workflows_v2`, `data/workflow_challenge_v2`).
* The gate contrastive builder's `--self-test` is green: 12 pairs, 24 items, both directions,
  three mechanisms, no output written.
* The production checkpoint cannot be clobbered: the trainer refuses an existing output
  directory (`ValueError: Use a new output directory; existing checkpoints are not overwritten`).

---

## 3. The frozen protocol (fixed BEFORE training)

### 3.1 Inheritance

The pre-registered protocol `docs/GATE_CONTRASTIVE_PROTOCOL_V1.md` **[READ]** is adopted
**verbatim for everything it already fixes**:

* construction rule `gate-contrastive-rule-v1` (§3) — one-fact mutation, verified flip through
  the real core, both directions;
* split geometry (§4) — source groups, not items, are the isolation unit; a pair is never split;
  the held-out cohort is built with a *different* seed and shares no source group;
* calibration-only policy — any threshold or abstention fit happens on `calibration` only;
* the primary endpoint is **M1**; M2–M5 are secondary and cannot rescue a failed M1;
* the held-out contrastive cohort, the relevance test/OOD corpus and the workflow V2 test/OOD
  cohort are evaluation-only and may never be trained, validated, calibrated or early-stopped on;
* the reporting standard — three matched seeds (17, 18, 19), report the **minimum**, retain flat
  and unfavourable arms, report exact hashes, no best-of-three, no post-hoc endpoint selection.

This runbook does not invent a replacement protocol and does not restate those clauses as its own.

### 3.2 The amendment that must be reviewed (the only proposed delta) **[ASSUMED until approved]**

Because of Blocker C, this runbook proposes to execute the domain adaptation with the **existing
runtime**, not with §7's LoRA recipe:

| §7 clause | Proposed amendment | Rationale |
|---|---|---|
| LoRA rank 16 | full-backbone fine-tune, fresh AdamW, `--backbone-lr 2e-5`, `--head-lr 2e-4` | No LoRA exists; the full-backbone path is the one that produced the current checkpoint and the three context-relevance runs |
| BF16 | FP32 pointwise storage, `--precision fp32` | MPS refuses BF16; the runtime's historical A100 recipe is also `parameter_storage: float32` |
| effective batch 8 | `--batch-questions 16`, `--microbatch-questions 4`, `--max-microbatch-tokens 16384` | The frozen maze recipe's batch; only the *effective* number differs from §7 |
| one epoch | fixed `--steps 300` with `--eval-every 50` | The runtime has no epoch concept |
| CE over allowed answer tokens | unchanged (`--objective gold_distribution --loss ce`) | Already the documented objective |
| 2,048-token cap | unchanged (`--max-length 2048`) | Identical to §7 |

**This substitution is a protocol change and therefore requires the §8 review before any run.**
The reviewer may instead require a CUDA host or a new LoRA implementation; in that case this
runbook's Steps 3–5 are unchanged in structure and only the command lines change.

### 3.3 Frozen seeds, splits, objective and selection

**Seeds [frozen].** `--seed 17`, `18`, `19`. Each seed also controls `random.sample` over the
training questions and — on CUDA only — the CUDA RNG; on MPS the head initialisation differs per
seed through `torch.manual_seed` **[READ]**. Every seed is reported; the acceptance verdict uses
the minimum across seeds.

**Splits [frozen at the review gate].** One merged corpus directory per arm, five JSONL files,
source-group-disjoint:

| Split | Composition | Used for |
|---|---|---|
| `train` | maze train (576 q) ∪ engineering train | fitting |
| `dev` | maze dev (192 q) ∪ engineering dev, composition frozen before the run | `dev target CE` selection only |
| `calibration` | engineering calibration | any threshold/abstention fit (none planned: T stays 1.0) |
| `test` | held-out engineering test cohort, source-group-disjoint from train/dev/calibration | the primary abstention endpoint, evaluated **once** after selection |
| `ood` | the different-seed held-out cohort (protocol §4) | robustness reporting only |

Two facts make this work mechanically **[READ]**:

* `read_training_records` refuses a `state_id` or `metadata.source_group_id` that appears in two
  splits, so the adapter must carry `metadata.source_group_id` and the builder's source-group
  isolation is enforced by the trainer.
* `main()` requires nonempty `train`, `dev` and `test`; `ood` and `calibration` are optional but
  must be present for the protocol's reporting.

**Objective [frozen].** `--objective gold_distribution --loss ce`. The maze rows already carry
`gold_probs` of kind `deterministic_truth`; the adapter maps each engineering item's
`expected.distribution` to `gold_probs` with the same kind. `--loss paired_brier_pg` is **not**
usable: it requires `observed_outcome`, the maze corpus has zero `observed_outcome` targets
**[VERIFIED]**, and MPS refuses the loss **[VERIFIED]**.

**Selection [frozen].** The runtime's rule is `minimum dev target CE`, with the initialisation
(step 0) as the incumbent — *"a trained candidate must beat the unchanged initialization on dev,
not win by default"* **[READ]**. Because `dev` is the merged set, selection is on the pooled
in-domain + engineering dev CE. The engineering `test` cohort is never part of selection, and
the abstention endpoint is never computed during the run. **No temperature is fitted** (the
summary records `temperature: 1.0`, `temperature_fitted: false`) **[READ]**. The 0.90 gate is not
changed and no threshold is tuned on test.

### 3.4 Stop conditions [frozen]

Stop immediately and report — do not continue, do not re-select — if any of:

1. any protected-segment deletion is proposed or applied (protocol §6.1, hard stop);
2. a nonfinite loss or nonfinite evaluation logits is raised (the runtime already raises);
3. `dev target CE` never beats step 0 within the declared budget → stop and report
   *"no learning signal"*, not a partial success;
4. any in-domain maze accuracy floor (§7 G4) is breached at the first two dev evaluations;
5. the declared wall-clock budget is exhausted (§4.3) — 45 minutes per seed **and** 3 hours for
   the whole three-seed campaign, both including evaluation;
6. any evidence of evaluation-cohort reuse appears (protocol §6.7).

A criterion that cannot be executed as written is an escalation to the review gate, never a
licence to relax it (protocol §6).

---

## 4. Resource reality on THIS machine

Machine: Apple M5-class (T6050), macOS 26.5.1 arm64, 128 GiB unified memory, 18 CPUs,
611 GiB free on the workspace volume, Python 3.14.6 in `.venv`, torch 2.14.0,
transformers 5.17.0, safetensors 0.8.0. **[VERIFIED]**

### 4.1 Device and objective support **[VERIFIED]**

`training_runtime()` probe, called directly from `scripts/train_pipeline_decisions.py`:

| `--device` | `--precision` | `--loss` | Accepted | Resolved |
|---|---|---|---|---|
| `mps` | `fp32` | `ce` | **yes** | mps / fp32 |
| `mps` | `fp32` | `brier` | yes | mps / fp32 |
| `mps` | `fp32` | `paired_brier_pg` | **no** | `ValueError: MPS sampled policy-gradient training is not validated; use CPU or CUDA` |
| `mps` | `bf16` | `ce` | **no** | `ValueError: MPS path currently fixed to FP32` |
| `cpu` | `fp32` / `auto` | `ce` | yes | cpu / fp32 |
| `auto` | `auto` | `ce` | yes | mps / fp32 |
| `cuda` | `bf16` | `ce` | **no** | no CUDA in this environment |
| `mps` | `fp32` | `ce` + `--disable-native-triton` | **no** | flag applies only to CUDA |

**Consequences.** (i) The only accelerated path is `mps` + `fp32` + `ce`/`brier`. (ii) The A100
recipe's `--disable-native-triton` **must be omitted** here or the run aborts. (iii)
`paired_brier_pg` is unusable on this machine *and* on this corpus — the roadmap's T14 note
("plan CPU or CUDA") is confirmed, and CPU-only sampling would be a separate, far slower arm
that this runbook does not schedule.

### 4.2 Memory **[VERIFIED measured lower bound + ESTIMATE]**

| Quantity | Value | Label |
|---|---|---|
| Host peak RSS during a real forward+backward on the checkpoint | 5.26 GB | **[VERIFIED]** |
| MPS current allocated after a step | 2.40 GB | **[VERIFIED]** |
| MPS driver allocated after a step | 2.83 GB | **[VERIFIED]** |
| Parameters (fp32) | 2.385 GB | **[VERIFIED]** (2.385 GB `best.safetensors`, 596,250,498 params) |
| Gradients + AdamW `exp_avg`/`exp_avg_sq` | ≈ 7.15 GB | **[ESTIMATE]** analytic |
| **Estimated real peak** | **≈ 8.9 GiB** | **[ESTIMATE]** |

The measured RSS is a lower bound because the probe deliberately never called
`optimizer.step()`, so AdamW moment buffers were never allocated. Even the estimate leaves ~14×
headroom under 128 GiB unified memory: **memory is not the constraint.** Activations are also
small because maze questions tokenise to at most 184 tokens and each microbatch packs only 4
complete questions **[VERIFIED]**.

### 4.3 Wall clock **[VERIFIED components + ESTIMATE totals]**

Measured on this machine, fp32 MPS, warm process:

| Component | Measured |
|---|---|
| Load checkpoint (tokenizer + `from_config` + safetensors + `.to("mps")`) | 6.33 s |
| Tokenise maze train, 300 records / 1200 questions, `max_length 2048` | 0.23 s |
| **One optimizer step**, 16 questions, 4 microbatches, `--gradient-checkpointing` | **2.13 s**, then **1.70 s** |
| **One optimizer step**, same, no gradient checkpointing | **1.41 s**, then **1.41 s** |
| Dev evaluation, 192 questions | 4.28 s |
| Calibration evaluation, 192 questions | 4.26 s |
| Test evaluation, 176 questions | 3.84 s |
| OOD evaluation, 64 questions | 1.58 s |

Cross-check **[VERIFIED]**: the three existing MPS runs
`runs/context_relevance_oracle_v1_seed{17,18,19}` (68 optimizer steps, batch 16, `max_length 512`,
no gradient checkpointing, 3 × 240-question dev evaluations, warm-started from the production
checkpoint) took **298.7 s / 271.3 s / 290.1 s**.

**Projected cost of the frozen 300-step maze recipe [ESTIMATE]:**

| Component | No grad-ckpt | Grad-ckpt |
|---|---|---|
| 300 optimizer steps | 423 s | 509–639 s |
| initial dev evaluation | 4.3 s | 4.3 s |
| 6 periodic dev evaluations (`--eval-every 50`) | 25.7 s | 25.7 s |
| final dev+calibration+test+ood evaluations | 14.0 s | 14.0 s |
| load + tokenise | 6.6 s | 6.6 s |
| `best.safetensors` writes (up to 7 × 2.385 GB ≈ 16.7 GB) | seconds–tens of s | same |
| **Total** | **≈ 8 minutes** | **≈ 9.5–11.5 minutes** |
| Conservative ceiling under contention | 25 min/seed | 25 min/seed |

**Sensitivity to the real (larger, longer) engineering corpus [ESTIMATE].** Per-step cost scales
with padded candidate tokens per step. The maze step processes ≈ 4 questions × 1 path × 184
tokens per microbatch; if engineering questions average 4 candidate paths × 512 tokens, a step
processes ≈ 2.8× the tokens → ≈ 4 s/step → ≈ 20 minutes for 300 steps. **This must be measured on
the assembled corpus before committing to the full campaign** (Step 2 below); the runbook's
budget in §3.4 uses the conservative 45 min/seed.

Because step cost does not depend on corpus size (each step samples `--batch-questions`), the
campaign cost is governed by step count, not by how much data the sibling workstream produces.

### 4.4 Disk **[VERIFIED]**

611.25 GiB free. Each candidate checkpoint directory is ≈ 2.4 GB (`best.safetensors` alone is
2,385,039,280 bytes). A three-seed candidate arm plus three control arms plus merged corpora is
well under 30 GB.

---

## 5. Required new artifact: the corpus adapter **[ASSUMED until reviewed]**

This is the only new code the runbook requires, and it must be reviewed and hash-frozen before
Step 1 can run. It is specified here so the review is against a concrete contract.

**Inputs.** (a) the gate contrastive cohort directory (materialized by
`build_gate_contrastive_v1.py`), (b) the engineering corpus directory (materialized by
the engineering corpus builder after Blocker A is fixed — see §12 for the withdrawal note),
(c) `dataset/games_v4/data/local_maze_v1`.

**Output.** `research/domain_adaptation_v1/merged/` containing `train.jsonl`, `dev.jsonl`,
`calibration.jsonl`, `test.jsonl`, `ood.jsonl` plus `manifest.json`.

**Per-row mapping** (must satisfy `validate_training_row` exactly):

| Trainer field | Source |
|---|---|
| `id` | `item_id` (globally unique; duplicates are rejected) |
| `state_id` | `item.state_id` |
| `family_id` | `item.family` (nonempty string) |
| `split` | `item.split` ∈ {train, dev, calibration, test, ood} |
| `state` | `item.request.states[0].state` |
| `questions` | `item.request.states[0].questions` |
| `gold_probs` | `{key: p}` from `item.expected.distribution` over `expected.candidate_keys`; keys must exactly match the candidate IDs; must sum to 1 |
| `gold_probs_kind` | `deterministic_truth` (the label is a frozen programmatic rule), so the one-hot check must hold |
| `gold_label_kind` | `deterministic_truth` |
| `metadata.source_group_id` | `item.contrastive.pair_id` for paired items, otherwise `state_id` |

**Hard requirements.**

1. **Source-group isolation.** The two members of a pair share `metadata.source_group_id` and
   live in one split. The trainer rejects cross-split state/source groups **[READ]**, so a
   violated merge fails loudly rather than silently leaking.
2. **No evaluation reuse.** No maze `test`/`ood` row, no `data/workflows_v2` test/ood row, no
   `data/workflow_challenge_v2` row, no context-relevance test/OOD row, and no
   `research/skill_abstention_survey_v1.json` content may enter any merged split. The adapter
   must carry the builders' refusal guards forward and record the guarantees in its manifest.
3. **Determinism and provenance.** Byte-identical rebuild for a fixed seed; `manifest.json`
   records per-file sha256, per-split question and source-group counts, and the input corpus
   hashes. The trainer independently records every input file's sha256 into its own `config.json`
   **[READ]**, so corpus provenance travels with the checkpoint.
4. **Dev composition is frozen.** The ratio of maze to engineering questions in `dev.jsonl` is
   declared in the manifest *before* the run, because `dev` drives selection.
5. **The survey is secondary.** `research/skill_abstention_survey_v1.json` is a 13-question
   diagnostic; it may be *replayed* as a fixed secondary endpoint but never used as training
   data, validation data or a selection signal.

---

## 6. The runbook, step by step

All commands run from the repository root with `.venv/bin/python`. `[DRY RUN]` marks a step that
writes no model state.

### Step 0 — Dependency and schema self-checks `[DRY RUN]` **[VERIFIED: all exit 0]**

```bash
.venv/bin/python scripts/train_pipeline_decisions.py --self-check
.venv/bin/python scripts/train_pipeline_decisions.py --help
```

Expected `--self-check` output:
`{"schema_checks": "passed", "complete_question_budget": "passed", "numerical_checks": "passed: grouped soft CE gradient, padding mask, question-mean accumulation and Boolean derivative"}`

### Step 1 — Build and validate the corpora `[DRY RUN for the trainer]` **[VERIFIED for the builders]**

```bash
# Gate contrastive cohort (green today)
.venv/bin/python scripts/build_gate_contrastive_v1.py --self-test
.venv/bin/python scripts/build_gate_contrastive_v1.py \
  --output-dir /tmp/domain-adaptation-v1/contrastive-train --seed 20260919
.venv/bin/python scripts/build_gate_contrastive_v1.py \
  --output-dir /tmp/domain-adaptation-v1/contrastive-heldout --seed 20260920

# Engineering corpus (Blocker A: no green builder exists as of authoring time; path to be
# fixed at the review gate by whoever owns the corpus. Run only once --self-test is green.)
.venv/bin/python scripts/<engineering-corpus-builder>.py --self-test
.venv/bin/python scripts/<engineering-corpus-builder>.py \
  --output-dir /tmp/domain-adaptation-v1/engineering
.venv/bin/python scripts/<engineering-corpus-builder>.py \
  --check /tmp/domain-adaptation-v1/engineering

# Adapter (new, reviewed) then the trainer's own stdlib dry run
.venv/bin/python scripts/<reviewed-adapter>.py \
  --output-dir research/domain_adaptation_v1/merged
.venv/bin/python scripts/train_pipeline_decisions.py \
  --input research/domain_adaptation_v1/merged --validate-only
```

`--validate-only` is the trainer's only dry-run mode **[READ]**: it parses the JSONL, enforces
the schema, the split enum, the unit-sum and one-hot target rules, the duplicate-ID rule and the
state/source-group cross-split rule, and prints the record/question counts per split and the
count of questions eligible for `gold_distribution`. **Do not proceed unless it exits 0 and the
printed `train/gold_distribution` count is at least `--batch-questions`.**

There is no `--dry-run` flag that executes the model path without writing. A bounded preflight
that *does* exercise the model path is Step 2.

### Step 2 — Bounded preflight on the real corpus `[DRY RUN]`

Two verified mechanisms are available today, plus a recommended one-step cost probe.

**(a) Objective-support preflight** — proves the warm start and tokenizer load on the real
corpus and that the objective has eligible targets, aborting before any write. It exits 1 by
design; the output directory must be empty afterwards.

```bash
.venv/bin/python scripts/train_pipeline_decisions.py \
  --input research/domain_adaptation_v1/merged \
  --init-checkpoint checkpoints/local_atomic_seed17/variants/local_atomic_seed17 \
  --output-dir /tmp/nj-preflight --objective observed_outcome --loss ce \
  --device mps --precision fp32 --steps 300 --head-steps 0 \
  --batch-questions 16 --microbatch-questions 4 --max-microbatch-tokens 16384 \
  --max-length 2048 --seed 17 --eval-every 50
ls -A /tmp/nj-preflight   # must print nothing
```

**[VERIFIED]** on the maze corpus this aborts with *"Fewer eligible training questions than one
effective batch"* after loading the checkpoint and tokenising, leaving the directory empty. On
the merged corpus it must abort for the same reason — if it does not, the adapter accidentally
supplied `observed_outcome` targets and the corpus must be re-reviewed.

**(b) Packing-budget preflight** — proves every complete question fits the declared token budget.

```bash
.venv/bin/python scripts/train_pipeline_decisions.py \
  --input research/domain_adaptation_v1/merged \
  --init-checkpoint checkpoints/local_atomic_seed17/variants/local_atomic_seed17 \
  --output-dir /tmp/nj-preflight2 --objective gold_distribution --loss ce \
  --device mps --precision fp32 --steps 300 --head-steps 0 \
  --batch-questions 16 --microbatch-questions 4 --max-microbatch-tokens 1 \
  --max-length 2048 --seed 17
```

**[VERIFIED]** it exits 1 with *"Complete question … requires N padded tokens, over budget 1"*.
If the real `--max-microbatch-tokens 16384` is not comfortably above the reported demand,
raise it in the amendment — never split a question's softmax (the error message says so
explicitly).

**(c) One-step cost probe (recommended, not yet a repository script).** Re-measure the per-step
cost on the assembled corpus before committing to 300 steps, without writing anything. This is a
self-contained, read-only adaptation of the probe used for §4.3:

```python
# /tmp/measure_step.py — forward+backward only; no optimizer.step(); writes nothing.
import os, sys, time
from pathlib import Path
REPO = Path("/Users/markus/Documents/NanoJev"); sys.path.insert(0, str(REPO / "scripts"))
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer
from predict_toy_decisions import local_checkpoint_files, read_json
from train_pipeline_decisions import load_training_examples, pack_complete_questions
from train_toy_decisions import DecisionModel

CKPT = REPO / "checkpoints/local_atomic_seed17/variants/local_atomic_seed17"
CORPUS = Path(sys.argv[1])                      # e.g. research/domain_adaptation_v1/merged
root, paths = local_checkpoint_files(str(CKPT)); cfg = read_json(paths["run_config"])
tok = AutoTokenizer.from_pretrained(str(paths["tokenizer"]), local_files_only=True, trust_remote_code=False)
if tok.pad_token_id is None: tok.pad_token = tok.eos_token
bc = AutoConfig.from_pretrained(str(paths["body_config"]), local_files_only=True, trust_remote_code=False)
model = DecisionModel(AutoModel.from_config(bc, attn_implementation="sdpa", trust_remote_code=False).float(),
                      cfg["set_head"])
w = load_file(str(paths["weights"]), device="cpu"); model.load_state_dict(w, strict=True); del w
model.backbone.config.use_cache = False; model = model.to("mps")
ex, _ = load_training_examples(CORPUS, tok, 2048)
train = [e for e in ex if e["split"] == "train"]
groups = pack_complete_questions(train[:16], 4, 16384)
model.train(); torch.mps.synchronize(); t0 = time.perf_counter()
for g in groups:
    logits, _ = model(g, tok.pad_token_id); z = logits.float().log_softmax(-1); loss = 0.0
    for i, e in enumerate(g):
        t = torch.tensor(e["gold_distribution_probs"], dtype=z.dtype, device=z.device)
        loss = loss - (t * z[i, :len(e["candidate_ids"])]).sum()
    (loss / 16).backward()
torch.mps.synchronize()
print("seconds_per_step", time.perf_counter() - t0,
      "peak_group_paths", max(sum(len(e["leaf_tokens"]) for e in g) for g in groups))
```

**[VERIFIED]** the equivalent probe on the maze corpus reports 1.41 s/step (no gradient
checkpointing) / 2.13 s/step (with it), 184 max path tokens, 4 paths per microbatch.

### Step 3 — Control arm: re-fit the frozen maze recipe, seeds 17/18/19 **[WRITES CHECKPOINTS]**

The control isolates the effect of the new corpus from the effect of simply continuing to train.
`results/nanojev_v2_baseline_seed17.json` already exists, but matched baseline reports for seeds
18 and 19 do not **[VERIFIED]**, so the control must be run at all three seeds.

```bash
for SEED in 17 18 19; do
  .venv/bin/python scripts/train_pipeline_decisions.py \
    --input dataset/games_v4/data/local_maze_v1 \
    --init-checkpoint checkpoints/local_atomic_seed17/variants/local_atomic_seed17 \
    --output-dir "runs/domain_adaptation_v1_control_seed${SEED}" \
    --objective gold_distribution --loss ce \
    --device mps --precision fp32 \
    --steps 300 --head-steps 0 --batch-questions 16 \
    --microbatch-questions 4 --max-microbatch-tokens 16384 \
    --eval-every 50 --max-length 2048 --seed "${SEED}" \
    --backbone-lr 2e-5 --head-lr 2e-4 --head-warmup-lr 1e-3 \
    --gradient-checkpointing
done
```

Notes **[READ/VERIFIED]**:

* `--disable-native-triton` is **deliberately absent**: it is refused off CUDA **[VERIFIED]**.
* `--head-steps 0` because this is a checkpoint warm start (the runtime would default to 0 here
  anyway); the warm start also forces `set_head` from the checkpoint config.
* `--gradient-checkpointing` matches the production config; §4.3 shows it costs ~30–50% per step
  on this machine and saves nothing that matters (5.26 GB measured RSS vs 128 GiB). Keeping it
  preserves the frozen recipe exactly; dropping it is a declaration the amendment must state.
* The trainer refuses to reuse an existing output directory, which is the rollback safety net.
* **Do not re-use `runs/local_atomic_seed17`** (it may be an archived artifact) and **never**
  point `--output-dir` at anything under `checkpoints/local_atomic_seed17`.

### Step 4 — Candidate arm: domain-adapted, seeds 17/18/19 **[WRITES CHECKPOINTS]**

```bash
for SEED in 17 18 19; do
  .venv/bin/python scripts/train_pipeline_decisions.py \
    --input research/domain_adaptation_v1/merged \
    --init-checkpoint checkpoints/local_atomic_seed17/variants/local_atomic_seed17 \
    --output-dir "checkpoints/domain_adaptation_v1_seed${SEED}" \
    --objective gold_distribution --loss ce \
    --device mps --precision fp32 \
    --steps 300 --head-steps 0 --batch-questions 16 \
    --microbatch-questions 4 --max-microbatch-tokens 16384 \
    --eval-every 50 --max-length 2048 --seed "${SEED}" \
    --backbone-lr 2e-5 --head-lr 2e-4 --head-warmup-lr 1e-3 \
    --gradient-checkpointing
done
```

Each run writes `config.json` (with `data_sha256` for every merged input file and the dependency
versions), `target_audit.json`, `initial_dev_metrics.json`, `train_log.json`, `summary.json`,
`best.safetensors`, `predictions_*.jsonl` and `predictions.jsonl`. `summary.json` records
`best_step`, `best_dev_target_ce`, per-split metrics, `temperature: 1.0`, `temperature_fitted:
false` and `training_seconds` **[READ]** — these are the artifacts the reviewer inspects.

### Step 5 — Evaluation **[READ-ONLY over the new checkpoints]**

**5a. In-domain maze regression (matched driver, so the comparison is valid).**

```bash
for SEED in 17 18 19; do
  .venv/bin/python scripts/benchmark_nanojev_v2.py \
    --checkpoint "checkpoints/domain_adaptation_v1_seed${SEED}" \
    --input dataset/games_v4/data/local_maze_v1 \
    --splits test,ood --device mps --precision fp32 --seed-label "seed-${SEED}" \
    --bootstrap-samples 1000 \
    --output "results/domain_adaptation_v1_seed${SEED}_maze.json"
done

# Paired comparison, one invocation per seed against the frozen baseline report.
.venv/bin/python scripts/compare_nanojev_v2.py \
  --baseline-report results/nanojev_v2_baseline_seed17.json \
  --candidate-report results/domain_adaptation_v1_seed17_maze.json \
  --output results/domain_adaptation_v1_seed17_compare.json
```

`compare_nanojev_v2.py` requires the baseline and candidate to have the *same* `seed_label`, the
same sample-ID set and identical cohort identity and candidate ordering **[READ]**, so the
candidate report must be produced on the same input with the same splits. For seeds 18 and 19,
first produce matched baseline reports from the Step 3 control checkpoints.

Frozen baseline to compare against **[VERIFIED]**, `results/nanojev_v2_baseline_seed17.json`
(device mps, precision fp32):

| Split | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| test (176 q) | 0.7784090909090909 | 0.4406830089906102 | 0.2872815926118046 | 0.085127211429856 |
| ood (64 q) | 0.765625 | 0.44663747693227196 | 0.31201684639476185 | 0.07338943984359503 |

**5b. Engineering abstention endpoint.** `benchmark_nanojev_v2.py` already emits, per question,
the full `probabilities` dict plus `gold_index` in its `samples` array **[READ]**, so the
abstention rate and the confidence–correctness correlation are derivable offline from an
evaluation run over the held-out engineering cohort:

```bash
.venv/bin/python scripts/benchmark_nanojev_v2.py \
  --checkpoint "checkpoints/domain_adaptation_v1_seed${SEED}" \
  --input research/domain_adaptation_v1/heldout \
  --splits test --device mps --precision fp32 --seed-label "seed-${SEED}" \
  --bootstrap-samples 2000 \
  --output "results/domain_adaptation_v1_seed${SEED}_engineering.json"
```

with, per question, `confidence = max(probabilities.values())`, `answered = confidence >= 0.90`,
`correct = argmax(probabilities) == gold_index`. The derivation script must be written, reviewed
and hash-frozen **before** Step 4, together with the source-group cluster bootstrap.

> **Caveat [ASSUMED].** No repository script currently computes the abstention rate or a
> Spearman accuracy–confidence correlation end to end; the diagnosis used the skill path
> (`integrations/codex-skill/nanojev-local-decider/scripts/nanojev_skill.py`). The gate-evaluation
> driver is therefore a second required artifact, and choosing or changing the confidence
> definition after seeing results invalidates the endpoint (protocol §M4).

**5c. M2 protected-segment deletions.** Run the protocol's M2 evaluation set: the held-out
contrastive cohort, the workflow V2 test/OOD cohort, the relevance test/OOD corpus and the
tool-history shadow fixtures, counting protected segments proposed and actually removed. Both
must be **exactly 0**.

---

## 7. Acceptance gates

**Primary engineering abstention endpoint (`G1`) — the fix for the abstention.** Measured once on
the frozen held-out engineering cohort (source-group-disjoint from every training split;
minimum size **N ≥ 60 questions / ≥ 20 source groups** so the interval means something).
Baseline: **0 of 13 answered at 0.90** (100% abstention).

| Gate | Threshold | Baseline |
|---|---|---|
| **G1 (primary)** | held-out answered-at-0.90 rate **A ≥ 0.50** *and* the 95% source-group cluster-bootstrap **lower bound of A ≥ 0.30** | A = 0.00 |
| G1b (secondary, necessary) | the original 13-question survey replayed at 0.90 is no longer 13/13 abstained: **A_survey ≥ 1/13** | 0/13 |
| **G2** | among answered questions: fixed-answer-subset accuracy **≥ 0.80** over all six fixed-answer items; **zero confident-wrong safety-critical items**, where the safety-critical pair is `safe_to_drop` (must answer `false`) and `blocked` (must answer `true`) — the two misses the diagnosis names; and **zero confident-wrong decisions in total**, because protocol M4 requires the count not to exceed the baseline's | baseline at 0.90: 0 answered, hence 0 confident-wrong; raw confidences already include 2 safety-relevant misses (`safe_to_drop`→true at 0.736, `blocked`→false at 0.657) |
| **G3 (= protocol M4)** | Spearman(confidence, correctness) **> 0** with 95% source-group cluster-bootstrap CI **excluding 0**; confident-wrong count not increased | not established (n=6) |
| **G0 (= protocol M2, hard)** | protected-segment deletions proposed **and** applied = **0** across every evaluation cohort | 0 |
| **G4** | in-domain maze test accuracy **≥ 0.773409** (−0.5 pp vs 0.778409) and OOD accuracy **≥ 0.760625** (−0.5 pp vs 0.765625); **and** no NLL/Brier/ECE point estimate worse than baseline whose paired source-group bootstrap CI excludes zero | baseline table in §5a |
| **G5** | all of G0, G1, G1b, G2, G3 and G4 hold at the **minimum across seeds 17/18/19**; every seed reported, no best-of-three | — |

**Measure the baseline first.** Before Step 3, run the same Step 5b evaluation against the
*unmodified* production checkpoint on the frozen held-out engineering cohort and record its
answered-at-0.90 rate, fixed-answer accuracy and confident-wrong count in the receipt. The
13-question survey baseline (A = 0.00, 3/6, and two safety-relevant raw misses) is known, but the
held-out cohort's baseline is not, and every gate above is a *delta* against it. If that baseline
is not 0, the reviewer must restate G1's bar as an absolute number before the candidate arm runs;
choosing it afterwards is exactly the post-hoc endpoint selection protocol §1.4 forbids.

**Why G4 is genuinely binding.** The three-seed relevance curriculum
(`docs/CONTEXT_RELEVANCE_V1.md`) reached 93.6–96.4% relevance accuracy yet regressed maze
accuracy by up to 4.69 pp at seed 17 and worsened NLL and Brier in **all three** seeds, and was
therefore not promoted **[READ]**. A candidate that buys engineering confidence by degrading the
in-domain probability quality has not fixed anything.

**Fixed rules.** No temperature is fitted (T stays 1.0) and no threshold is tuned; any threshold
fit is on `calibration` only; the 0.90 gate itself is not lowered (temperature scaling and
gate-lowering are both already excluded by prior measurement **[READ]**).

**G2 is deliberately strict, and the strictness is the point.** Because the baseline answers
nothing at 0.90, it has zero confident-wrong decisions; protocol M4 therefore requires the
candidate to also have zero. Combined with `G1` (A ≥ 0.50), that means at least half the
engineering questions must be answered *and correct* — a fix that raises confidence without
raising correctness fails, which is exactly the failure mode (the most confident answer being
the safety-critical wrong one) that made abstention the correct current behaviour.

---

## 8. Failure criteria, including the explicit null result

| # | Condition | Verdict and required action |
|---|---|---|
| F0 | Any protected-segment deletion (G0 ≠ 0) | **Immediate stop**, regardless of every other metric. Report the offending items; keep abstention as default. |
| F1 | **G1 fails** | **The explicit null: "abstention is unchanged."** Report it in exactly those terms: at the documented 0.90 gate the model still has no confident engineering mode. It means the confidence *ordering* on engineering content did not move under this corpus/objective/capacity combination — the same class of failure temperature scaling could not repair, but now tested with in-domain training signal rather than rescaling. Consequences: the phrase-based scope guard stays; the 0.90 gate stays; **do not** lower the threshold, **do not** re-fit temperature, and **do not** reframe a small movement as partial success. The next attempt must change corpus scale/diversity, model capacity, or the readout — not the gate. |
| F2 | G1 passes but G2 or G3 fails | **Fail, and it is worse than abstention:** the model has become confidently wrong on engineering judgment, including safety-critical items. Revert; the scope guard is mandatory. |
| F3 | G4 fails | **Fail.** Domain adaptation traded away in-domain probability quality (the exact failure mode that sank the relevance curriculum). Revert. |
| F4 | Any evaluation-cohort reuse (training/validating/calibrating/selecting on the held-out engineering cohort, maze test/OOD, workflow challenge, relevance test/OOD, tool-history shadow fixtures) | **Fail** (protocol §6.7). Report which cohort and how. |
| F5 | `dev target CE` never beats step 0 within budget | Stop early. **"No learning signal."** Not a partial success. |
| F6 | The protocol/builder/adapter cannot be executed as written | Escalate to the review gate as a **blocker**; never relax a criterion (protocol §6 closing clause). |

Every failure is reported as a failure of the hypothesis, with all arms and seeds retained,
including flat and unfavourable ones.

---

## 9. Review gate, promotion, rollback

### 9.1 Who must approve, and what they approve

Per protocol §8, an **independent reviewer** — not the author of this runbook, not the training
executor — must record a written PASS covering:

1. that §3's frozen seeds, splits, objective, selection rule and stop conditions, and the §7
   acceptance gates, are pre-registered and unambiguous before any run;
2. the §3.2 amendment (recipe substitution) or a decision to require a CUDA host / new LoRA code
   instead;
3. the corpus adapter (§5): source-group isolation, no evaluation reuse, determinism, manifest
   hashes;
4. that the engineering corpus builder is green (`--self-test` and `--check`) and that the
   trainer's `--validate-only` passes on the merged corpus with adequate eligible counts;
5. the corpus lineage: that the gate contrastive builder's and the engineering builder's
   no-leakage guarantees hold for the *merged* corpus, not just for each half;
6. the gate-evaluation driver and confidence definition (§5b) are hash-frozen before Step 4.
7. the exact command lines, seed list, budget and rollback path.

Additionally, **promotion** — pointing serving at a new checkpoint — is a separate decision by
the repository owner. Per `AGENTS.md`, production active-context removal is **not authorised** by
this runbook, and no training may start before the §8 review PASS is recorded.

### 9.2 Rollback path

* **The production checkpoint is structurally protected [VERIFIED].** The trainer raises
  `ValueError: Use a new output directory; existing checkpoints are not overwritten` when
  `--output-dir` already contains `config.json` or `best.safetensors`; its timestamps were
  unchanged by this work. Every run therefore lands in a **new** directory and cannot mutate
  `checkpoints/local_atomic_seed17/variants/local_atomic_seed17`.
* **Serving default is untouched until promotion.** Until a reviewed promotion, the local
  service keeps `checkpoints/local_atomic_seed17/variants/local_atomic_seed17`, temperature 1.0,
  the 0.90 gate and zero removals.
* **Rollback = repoint, do not delete.** Revert the serving pointer/configuration to the frozen
  checkpoint path. No data is mutated, because the new checkpoint lives elsewhere.
* **Confirm the rollback.** Re-run
  `.venv/bin/python scripts/benchmark_nanojev_v2.py --checkpoint checkpoints/local_atomic_seed17/variants/local_atomic_seed17 --input dataset/games_v4/data/local_maze_v1 --splits test,ood --device mps --precision fp32 --seed-label seed-17`
  and confirm the §5a baseline numbers reproduce to floating-point precision.
* **Retain everything.** Keep every candidate and control run directory with its `config.json`
  (which carries the input `data_sha256`) and `summary.json`, including failed arms. A failed run
  is evidence, not garbage.

---

## 10. What was verified by execution vs read from code

### 10.1 Verified by execution **[VERIFIED]**

| Check | Result |
|---|---|
| Device/runtime probe (`torch` 2.14.0, mps available, cuda absent) | mps yes, cuda no |
| `training_runtime()` 9-case matrix | mps+fp32 `ce`/`brier` accepted; mps+`paired_brier_pg` refused; mps+bf16 refused; `--disable-native-triton` refused off CUDA; cpu+fp32 accepted; `auto` → mps/fp32 |
| `--help` | exit 0; no `--dry-run` flag exists |
| `--self-check` | exit 0; schema, budget and numerical gradient checks passed |
| `--validate-only` × 4 corpora | exit 0; counts recorded in the feasibility JSON |
| Warm-start preflight on MPS (`observed_outcome` abort) | exit 1 as designed; output dir empty; checkpoint loaded and corpus tokenised |
| Packing-budget preflight | exit 1 as designed; *"requires 177 padded tokens, over budget 1"*; output dir empty |
| Overwrite guard against the production checkpoint | exit 1; `best.safetensors` and `config.json` untouched |
| Maze corpus identity | all five sha256 match `config.json data_sha256` |
| Maze label census | 576/576 boolean, `deterministic_truth`, zero `observed_outcome` targets |
| One-step cost (16 q, fp32 MPS) | 1.41 s (no grad-ckpt), 2.13 s / 1.70 s (with grad-ckpt) |
| Evaluation cost per split | dev 4.28 s, calibration 4.26 s, test 3.84 s, ood 1.58 s |
| Memory | peak host RSS 5.26 GB; MPS allocated 2.40 GB / 2.83 GB driver |
| Checkpoint load time | 6.33 s |
| Tokenisation of 1200 maze questions at `max_length 2048` | 0.23 s |
| Token demand | max 184 path tokens/question; ≤ 4 paths per microbatch; `max_microbatch_tokens 16384` never binding |
| Gate contrastive builder `--self-test` | exit 0; 12 pairs / 24 items; no output written; re-verified green after the workspace churn |
| Engineering corpus builder `--self-test` | **exit 1; 0 pairs / 0 items; one-fact flip violated** — and the file was then removed from the working tree by its owning workstream |
| Existing MPS runs | `runs/context_relevance_oracle_v1_seed{17,18,19}` trained on mps/fp32 in 271–299 s |
| Disk | 611.25 GiB free; checkpoint dir ≈ 2.4 GB |

### 10.2 Read from code or frozen documents, not executed **[READ]**

* The trainer's control flow: `validate_training_row`, split/duplicate/cross-split enforcement,
  complete-question packing, `evaluate_pipeline`, selection on minimum dev target CE with step 0
  as the incumbent, test evaluated only after selection, temperature recorded as 1.0 and not
  fitted, `data_sha256` captured in `config.json`.
* `compare_nanojev_v2.py`'s matched-sample, seed-label and paired source-group bootstrap
  requirements; `benchmark_nanojev_v2.py`'s metric definitions, ECE binning and per-question
  `samples` schema (which is what makes the abstention rate and Spearman derivable offline).
* Protocol `GATE_CONTRASTIVE_PROTOCOL_V1.md`: §1.1 (post-review changes need a new version),
  §3–§6 (rule, split geometry, M1–M5, failure criteria), §7 (the LoRA/BF16 recipe), §8 (review
  gate), §9 (what is not authorised), §10 (limitations).
* The engineering builder's item schema and its `training_authorized_by_this_corpus: false` /
  `labels_human_reviewed: false` / `corpus_role: training_data_prerequisite_only` declarations.
* `docs/CONTEXT_RELEVANCE_V1.md` (the prior three-seed curriculum and its maze regressions),
  `docs/SKILL_ABSTENTION_DIAGNOSIS_V1.md`, `docs/CONFIDENCE_ATTRIBUTION_V1.md`,
  `docs/APPLE_SILICON.md`, `docs/NANOJEV_V2_ROADMAP.md` (T8/T9/T14),
  `results/temperature_fit_v1.json`, `results/nanojev_v2_baseline_seed17.json`.
* `AGENTS.md` authority boundaries (no production active-context removal, no financial training
  before review gates).

### 10.3 Estimated or assumed, not observed **[ESTIMATE] / [ASSUMED]**

* Full-run wall clock (§4.3): arithmetic projection from measured per-step and per-evaluation
  costs. **No 300-step run was performed** — that would have written a checkpoint.
* Peak memory with AdamW moments (§4.2): analytic, because no optimizer step was taken.
* Step cost on a larger engineering corpus (§4.3): scaling estimate tied to padded-token ratio;
  must be measured on the assembled corpus.
* Whether the 596M-parameter backbone can express the required engineering discrimination at all:
  **unknown**. Protocol §10.6 names exactly this as a live way for M1 to fail.
* That the engineering corpus's programmatic labels correspond to human engineering judgement:
  **not established**; the builder says the labels are not human-reviewed.
* The exact dev composition (§3.3) and adapter details (§5): proposed here, subject to the review
  gate.

### 10.4 Workspace churn observed during authoring **[VERIFIED]**

The workspace is shared with concurrently running sibling workstreams and changed while this
runbook was being written. `scripts/build_engineering_corpus_v1.py` was present and failing its
self-test, and was then **removed entirely** by its owning workstream (re-checked at
`2026-09-19T14:44:57Z`: no `engineering` file under `scripts/`, not listed by `git status`).
New untracked files also appeared (`scripts/safe_dedup_v1.py`, `scripts/test_safe_dedup_v1.py`,
`results/safe_dedup_v1.json`) from another workstream.

After the churn, every training-critical input was re-verified unchanged: the five maze split
files still sha256-match the checkpoint config, the checkpoint `config.json` still hashes to
`3a6b014a…def176`, `best.safetensors` is still 2,385,039,280 bytes, and the gate contrastive
builder's `--self-test` is still green. Nothing in this runbook depends on the withdrawn file
remaining at a particular path — the corpus is named by role — but the dated failure is retained
as evidence that the data prerequisite was unmet at authoring time.

---

## 11. Limitations of this runbook

1. **Nothing was trained.** No number here is a model result. Every acceptance threshold is a
   pre-registered bar, not a measurement.
2. **The data does not exist yet.** At authoring time the engineering corpus builder failed its
   own self-test and was then withdrawn from the working tree by its owning workstream (§10.4);
   the runbook's Steps 3–5 are blocked on that sibling workstream, not on this document.
3. **Two new artifacts are required** (the corpus adapter and the gate-evaluation driver) and
   neither exists; the runbook specifies them so the review gate has a concrete object.
4. **The proposed recipe is an amendment**, not the pre-registered §7 recipe. Until the review
   gate accepts it (or demands a CUDA host), no run is authorised.
5. **Timings are one machine, one torch build, one quiescent moment**; they are not a
   cross-machine benchmark, and the three existing runs' 271–299 s figures come from a similar
   M-series host.
6. **The engineering endpoint rests on synthetic labels** from a frozen programmatic rule; a
   model that passes G1–G3 has learned that rule, which is necessary but not demonstrated to be
   sufficient for real engineering judgement.
7. **Small-cohort risk.** The protocol's default cohort is 12 pairs (a skeleton, protocol §10.1);
   the primary engineering endpoint's N ≥ 60 / ≥ 20 source groups is a floor proposed here, and
   the review gate may raise it. Below that floor the bootstrap interval is descriptive only.
8. **The G1 threshold (A ≥ 0.50, CI lower bound ≥ 0.30) is a judgement call** made before any
   data exists. It is deliberately conservative relative to "the abstention is fixed" (which
   would be A near 1.0) and deliberately above the 0.90-gate's in-domain strictness (62.5%
   abstention in-domain, so even the in-domain model would only barely clear A = 0.50 on a
   comparable cohort).
9. **Not authorised here:** production context removal, serving-default changes, threshold or
   temperature changes, and any token-saving or readiness claim.

---

## 12. Reproduction index

```bash
# Everything this runbook verified, in order, with no writes:
.venv/bin/python scripts/train_pipeline_decisions.py --help
.venv/bin/python scripts/train_pipeline_decisions.py --self-check
.venv/bin/python scripts/train_pipeline_decisions.py --input dataset/games_v4/data/local_maze_v1 --validate-only
.venv/bin/python scripts/train_pipeline_decisions.py --input data/context_relevance_oracle_v1_seed20260919 --validate-only
.venv/bin/python scripts/train_pipeline_decisions.py --input data/workflows_v2 --validate-only
.venv/bin/python scripts/train_pipeline_decisions.py --input data/workflow_challenge_v2 --validate-only
.venv/bin/python scripts/build_gate_contrastive_v1.py --self-test
.venv/bin/python scripts/<engineering-corpus-builder>.py --self-test     # no green builder existed at authoring time
.venv/bin/python /tmp/measure_step.py research/domain_adaptation_v1/merged
```

Artifacts produced by this work:

* `docs/DOMAIN_ADAPTATION_RUNBOOK_V1.md` — this document.
* `results/training_feasibility_v1.json` — commands, exit codes, probe matrices, measurements,
  estimates and non-claims.

No other file in the repository was modified; no checkpoint, corpus or run directory was written.
