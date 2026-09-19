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
