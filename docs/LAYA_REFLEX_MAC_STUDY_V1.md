# laya and reflex on a Mac — study and what to adopt

Studied 2026-09-19 because both are Mac-deployable decision models, and this project's blocker is
out-of-domain decision capability. Every claim is marked **[verified here]**, **[repo claim]**, or
**[our measurement]**.

Sources: `github.com/NandhaKishorM/laya` (690 stars, Apache-2.0) and
`github.com/kshetrajna12/reflex` (73 stars, MIT), both read 2026-09-19, plus local execution on
this machine.

## 1. What they are

| | **laya** | **reflex** |
|---|---|---|
| Base | ModernBERT-large encoder, **421M**, non-autoregressive | Qwen3.5-4B decoder (default); 0.8B in the browser demo |
| Readout | typed heads over `choice` / `score` / `noul` | **direct logits** — reads the next-token scores of the answer labels (A/B/C, Yes/No), **no trained heads** |
| Context | 512 (English) / 1024 (multilingual, typed-decisions) | not stated as a limit; fits the GPU |
| Training | **RL against strictly proper scoring rules (RLCD)** | supervised; a **single temperature** fitted on labelled data |
| Choice cap | high-cardinality supported but budget-limited (see §5) | **26 options** hard cap |
| Licence | Apache-2.0 | MIT |

## 2. Mac deployment — the two paths are different, and only one is server-side

**laya — [verified here].** `pip install laya`; runs on Apple Silicon MPS with no GPU server.
Measured on this machine: model loads **offline in 27 s** and reports `device: mps:0`; the earlier
E1 pass measured **in-process warm p50 40.8 ms** over 4 questions. It is a plain Python dependency.
It also ships a Colab notebook, a Hugging Face Space, and a `Router` that loads checkpoints lazily.

**reflex — [repo claim].** The Python package explicitly requires **"a Linux machine with an NVIDIA
GPU"**. The **Mac path is the browser demo**: `kshetrajna12.github.io/reflex` runs a 650 MB
Qwen3.5-0.8B ONNX q4f16 model through **WebGPU in Chrome/Edge/Safari 18+**, entirely client-side.
So reflex on a Mac means *browser*, not a server component, and its own README warns the demo is
"smaller and less calibrated" than the Python version and re-reads state per question.

**Consequence for us:** if the goal is a Mac-local decision service, laya is the directly usable
shape. reflex's Mac story is a client-side demo, which is a different product surface.

## 3. laya's architecture has three ideas we do not have

1. **A Router that dispatches per request.** Three checkpoints (`english` 421M, `multilingual`
   322M, `typed-decisions` 421M) with lazy loading, chosen per request. This is a principled
   version of the crude phrase-based scope guard this project currently uses for out-of-domain
   input.
2. **Workflow presets**, including an **"Intelligent Model Router (routes to small vs frontier
   models)"**, prompt-guardrail, moderation, and support-triage question schemas. These are
   pre-tuned question sets — i.e. the "specialise per task" answer in shipped form.
3. **Automated confidence gating built on proper-scoring training**: `if confidence >= 0.85: act
   automatically else escalate`. Its own README claims the probabilities are "statistically
   meaningful" *because* they were trained against strictly proper scoring rules (RLCD).

## 4. Both projects are explicit about calibration, and both say "fit it yourself"

- **laya [repo claim]:** "Both checkpoints are **over-confident as shipped**. Refitting **one
  temperature per (question type, option count)** on held-out data moves mean ECE **0.466 → 0.081**
  (`laya`) and **0.314 → 0.106** (`laya-multilingual`). `laya-multilingual` ships with **no fitted
  temperatures at all**."
- **reflex [repo claim]:** "A single 'temperature' number, fitted on labelled data, makes the
  percentages honest."

**This is a methodological correction to our own temperature work.** Our calibration study fitted
**one global scalar** and found no usable gain. laya fits **one temperature per (question type,
option count)** — a grouped calibration. Those are not the same experiment, and grouped
temperature can repair over-confidence that a single global scalar cannot. Our conclusion
("recalibration cannot fix the abstention") remains valid for the *ordering* reason, but the
weaker claim that a single scalar does not help should not be generalised to grouped calibration.

## 5. laya's honest limits are the most useful thing in either repository

Verbatim [repo claim]:

> "**The base checkpoints are near chance on typed-decisions zero-shot** — 0.362 and 0.352 against a
> 0.318 random baseline and a 0.461 majority-class baseline. The 0.766 figure comes from the
> checkpoint fine-tuned on that benchmark's own training split. **Laya is a fast base to specialise,
> not a zero-shot decision engine.**"

Also documented: high-cardinality `choice` accuracy **falls off sharply** when the per-option token
budget shrinks (77 options at default settings give ~3–4 tokens per label; 0.425 vs 0.870), and
ordinal `score` is its weakest primitive.

**This reframes our problem.** Out-of-domain collapse is **not a NanoJev defect** — it is the
documented behaviour of the whole model class. The industry answer is *specialise per task and
route*, not "make one small model good at everything".

## 6. The experiment: laya on our own 13 engineering questions

**[our measurement]** The same 13 questions NanoJev abstains on, run through laya on `mps:0`
(offline, no network):

| | NanoJev 0.6B | **laya 421M** |
|---|---:|---:|
| Median confidence | 0.505 | **0.656** |
| Max confidence | 0.736 | **1.000** |
| Answered at 0.90 | **0 / 13** | **3 / 13** |
| Answered at 0.50 | 7 / 13 | 10 / 13 |

laya is clearly **more willing to answer**. But the safety-relevant rows are the point:

| Question | Expected | NanoJev | laya |
|---|---|---|---|
| `safe_to_drop` — is it safe to remove context without a validated gate? | **false** | true @ 0.736 | **true @ 0.878** |
| `blocked` — is the project blocked from financial claims? | **true** | false @ 0.657 | **false @ confidence 1.000** |

**laya makes the same safety-critical error, more confidently.** Its single most confident answer
in the batch (1.000) is wrong.

### What this means

1. **Swapping in a stronger open decision model would not fix our problem; it would convert
   abstention into confident wrongness.** For a gate that can remove context, that is strictly
   worse than abstaining.
2. It is also a direct, independent confirmation of laya's own README: an un-specialised decision
   model is near chance, and confidence is not a safety signal out of domain.
3. **Therefore the fix remains task specialisation.** Our scope guard is the correct interim
   behaviour; a router (laya's pattern) is the right generalisation once specialised checkpoints
   exist.

## 7. What to adopt, concretely

**Adopt:**
- **The Router/dispatch pattern** — multiple specialised checkpoints plus an explicit
  route-or-refuse decision, replacing the phrase-based guard when specialised checkpoints exist.
- **Grouped calibration** — fit temperature per (question type, option count), not one global
  scalar. Our single-scalar negative result must not be read as ruling this out.
- **Workflow presets as shipped artifacts** — a pre-tuned question set per task family is what
  makes a small decision model usable; our corpus work should emit these.
- **The encoder-as-scorer hypothesis gets stronger** — 421M encoder, single forward pass, 40.8 ms
  on this Mac, and a training recipe (RLCD with proper scoring rules) that matches our Track B4
  direction. Worth an A/B against our 0.6B decoder heads **on a specialised task**, which is the
  only comparison that would mean anything.

**Do not adopt:**
- reflex's Python path for a Mac server (Linux + NVIDIA required); only its WebGPU demo is Mac-local.
- Any assumption that higher confidence means better decisions out of domain — measured false here.
- laya's checkpoints as a drop-in replacement: un-specialised, they are near chance by their own
  documentation and confidently wrong on our safety-critical item.

## 8. What this changes in the roadmap

- **T9 (gate model) gains a concrete architectural option**: encoder scorer with per-task
  specialisation and a router, rather than only "train our decoder heads on more data".
- **A new comparison belongs in the plan**: our decoder heads vs a laya-style encoder scorer,
  both specialised on the same engineering corpus, on the same frozen holdout. Without
  specialisation the comparison is meaningless — laya's own numbers say so.
- **The calibration task (S2) should be reopened narrowly**: grouped per-(type, cardinality)
  temperature is untested here.
- **Nothing here authorises active context removal, training, or a model swap.** The scope guard
  stays until acceptance gates pass.

## Limits of this study

- One checkpoint per project, one machine, one 13-question probe. laya's 3/13 and its two
  safety-critical rows are **13-question observations with no confidence interval**.
- laya was run through its published package on cached weights; **no laya training, fine-tuning, or
  hyperparameter work was done**, and none of its benchmark numbers were reproduced here.
- reflex was **not executed**. Its Mac deployment claim is taken from its README and its browser
  demo was only confirmed reachable earlier (E1), not re-verified now.
- Comparisons against NanoJev are on identical questions and identical machine, but the two models
  expose different primitives (`noul` vs `boolean`, different `score` encodings), so confidence is
  compared as the max probability, not as a like-for-like calibrated quantity.
- No provider was contacted and no data left the machine: laya ran with `HF_HUB_OFFLINE=1`.

## Reproduction

```bash
PYTHONPATH=/tmp/laya_pkg HF_HUB_OFFLINE=1 .venv/bin/python /tmp/laya_probe.py
# probe source kept at scripts/probe_laya_engineering_v1.py
```

---

# Addendum (2026-09-19): the reference set was incomplete, and one readout idea was tested

## A. Three HuggingFace projects we had missed

The first pass studied only GitHub. Searching HuggingFace revealed decision-model projects absent
from this project's reference list — two of them larger or more directly relevant than reflex.

| Project | Where | Why it matters |
|---|---|---|
| **[TheoLeeCJ/SemIf](https://github.com/TheoLeeCJ/SemIf)** — **1,770 stars, MIT**, created 2026-09-16 | GitHub + HF `Meanblock/JEV-CPU` | "Semantic ifs from open models, on a 3090 at home." **Larger than laya (690) and reflex (73) combined**, and it is the author of the compression objection already recorded in this roadmap |
| **[leesk212/JEV-CPU](https://github.com/leesk212/JEV-CPU)** — MIT, 2026-09-19 | HF `Meanblock/JEV-CPU` | A CPU port of SemIf: **"on a laptop CPU, no GPU"**, `Qwen/Qwen3-0.6B` in float32 (~2.4 GB), ~1 s per decision, demonstrated across **eight domains** including support, **code-review triage** and **incident severity** — i.e. engineering judgment |
| **[kotoba-lang/typed-decisions](https://github.com/kotoba-lang/typed-decisions)** / HF `com-kotobalabs/open-jev-deberta-v3-large` | Apache-2.0, 20 likes | Jev-shaped typed decisions on DeBERTa-v3-large with **choice over up to 255 options** — matching our contract, not reflex's 26 cap — plus a corpus builder, ablations and ADRs |
| HF `mobarmg/jev-schema-scorer-deberta-v3-large` | MIT, 158 downloads | States the principle outright: "the question text, criteria and option ids are read **at inference time, never baked into the weights**, so the same checkpoint answers new questions over new label sets **without retraining**" |

**Correction to the reflex section above.** reflex's README also contains material the first pass
missed, and it is the most directly useful part:

- **A calibration workflow**: `reflex-eval-mmlu --n 1200 --fit-temperature ...`, with measured
  results — Qwen3.5-4B 72% accuracy, ECE **0.090 → 0.039** after fitting; Qwen3-8B 71%, ECE
  0.264 → 0.061. (Jev reportedly reports 0.031.)
- **A data builder**, `reflex-data`, that constructs labelled files "from **eight public datasets**,
  one recipe each", covering routing intents, exam questions, toxicity with **soft labels**,
  hallucination checks and **passage relevance**.
- **The training recipe, stated plainly**: "Temperature fixes over-confidence but cannot make the
  model *better* at a task. For that you train it... penalise it with a proper scoring rule (log
  loss or Brier), which is minimised only by the true probabilities. **That is the supervised form
  of the 'RLCD' training Jev uses.**"
- **reflex has no HuggingFace weights of its own** — its HF references are only the base Qwen
  checkpoints. It is a toolkit and recipe over someone else's base model.

## B. Two architectures, and ours is the harder one

| | **Task-fitted heads** (our NanoJev, laya's typed-decisions checkpoint) | **Schema-conditioned logit readout** (SemIf/JEV-CPU, reflex, jev-schema-scorer) |
|---|---|---|
| Decision | trained head per question type | `P(option letter \| evidence, criterion, options)` from one forward pass |
| New task | needs retraining | claimed **without retraining** |
| Documented weakness | collapses out of domain (ours: 13/13 abstain) | needs a model whose letter distribution reflects the prompt |

This is the sharpest framing the study produced. Our abstention is a symptom of the first column;
the second column is explicitly designed to avoid it.

## C. Tested: the SemIf readout, on our checkpoint — a position artifact

**[our measurement]** We applied SemIf's readout to **this project's fine-tuned checkpoint**,
holding the weights fixed and changing only the readout. Naively the result looked spectacular:

| Readout, same weights | Median confidence | Max | Answered at 0.9 |
|---|---:|---:|---:|
| Our trained heads | 0.505 | 0.736 | **0 / 13** |
| SemIf letter-choice logits | **0.899** | **0.948** | **6 / 13** |

**It is an artifact.** Every one of the 13 answers was option **A**. An option-permutation control
settles it:

| Control | Result |
|---|---|
| Chosen **letter** identical under permutation | **13 / 13** (always A) |
| Chosen **description** identical under permutation | **0 / 13** |

The model answers "A" regardless of what A denotes, so the probabilities carry **no decision
content**. Our own trained head passes this same control (zero top-choice flips across 272 Choice
questions), which is why the project tests it.

**Why this happens, and the correct comparison.** SemIf/JEV-CPU apply this readout to a **base
instruct model**, whose next-token distribution over letters reflects the prompt. Our checkpoint's
backbone was fine-tuned with task-fitted heads, and its letter slots no longer carry the criteria.
The honest comparison is therefore **base Qwen3-0.6B + SemIf readout** versus **our checkpoint +
trained heads** — not the hybrid above. That comparison is the open follow-up; the base weights are
downloading as this note is written.

## D. What to take from the addendum

1. **Our reference list had a 1,770-star gap.** SemIf/JEV-CPU are the closest published analogues
   to our problem and must be pinned before any further architecture work.
2. **The architectural question is now explicit**: task-fitted heads versus schema-conditioned
   readout, and the second is claimed to generalise without retraining. Our corpus work assumes the
   first; that assumption should be re-examined before a training run, not after.
3. **A readout that raises confidence is not a fix until the permutation control passes.** The
   artifact above would have looked like a 0.505 → 0.899 triumph if the control had not been run.
4. **Both laya and reflex say the same thing about calibration and about training**: fit a
   temperature, then train with a proper scoring rule, and expect nothing more from the scalar.
   reflex supplies a soft-label data builder and calls its recipe the supervised form of RLCD.

## Reproduction

```bash
.venv/bin/python scripts/probe_semif_readout_v1.py          # naive readout
.venv/bin/python scripts/probe_semif_permutation_control_v1.py   # the control that refutes it
```

Receipts: `results/semif_readout_probe_v1.json` (naive result plus the permutation control).
