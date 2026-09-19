# Confidence attribution V1 — the collapse is the content domain, not the surface form

**Question.** The local NanoJev checkpoint abstains 13/13 on engineering-judgment questions while
answering in-domain maze questions confidently. An earlier diagnosis localised the collapse to
out-of-domain input but could not **attribute** it: the engineering survey differed from the maze
cohort in language, question type, option wording and state length **all at once**. If the cause
were surface form — length or phrasing — a prompt-format fix might repair it without retraining.
This probe separates the two.

## Design

Two directions, one batch, 18 states / 66 questions / 71 candidate paths (within the service's
32 / 96 / 256 limits). Read-only inference against the running local service; no training, no
checkpoint writes, `network_model_calls: 0`.

**Direction 1 — in-domain content, four surface forms.** Four maze states (`test.jsonl`) replayed as:

| Variant | State | Questions |
|---|---|---|
| `v0_plain` | original | original |
| `v1_padded` | original + an irrelevant engineering paragraph | original |
| `v2_reworded` | original | rewritten into engineering prose |
| `v3_both` | padded | rewritten |

**Direction 2 — out-of-domain content, shortened and simplified.** Two engineering questions
(lifecycle phase, routing) reduced to a one-clause state and plainer wording, i.e. moved *toward*
the maze surface form.

## Result

### Direction 1 — surface form barely moves in-domain confidence

| Content | plain | padded | reworded | both |
|---|---:|---:|---:|---:|
| maze0 | 0.604 | 0.589 | 0.605 | 0.627 |
| maze1 | 0.881 | 0.868 | 0.877 | 0.866 |
| maze2 | 0.856 | 0.862 | 0.854 | 0.859 |
| maze3 | 0.774 | 0.763 | 0.753 | 0.744 |

Median confidence per variant (4 questions each). The largest movement from **padding** is 0.015,
and from **rewording** 0.021 — both far inside the noise of a 4-question probe. Adding an
irrelevant engineering paragraph to a maze state does **not** suppress confidence.

### Direction 2 — shortening does not recover out-of-domain confidence

| Engineering question | Original survey | Shortened + simplified |
|---|---:|---:|
| lifecycle phase choice | 0.268 | **0.275** |
| routing under a scoring budget | 0.490 | **0.444** |

Moving the out-of-domain question *toward* the in-domain surface form leaves confidence where it
was, and in one case slightly lower. Length and wording are not the barrier.

## Verdict

**The collapse is attributable to the content domain.** Surface form is excluded as the cause:
it neither suppresses in-domain confidence when degraded, nor restores out-of-domain confidence
when improved.

**Consequence for the fix — this matters more than the measurement.** There is no prompt-format or
wording workaround. Repairing the abstention requires the model to see engineering-judgment
content during training, which means:

- the data blocker is real and primary (a sibling workstream is building that corpus);
- temperature scaling is already excluded (it is monotone and cannot reorder confidences);
- a scope guard is the correct behaviour **until** a retrained checkpoint passes the acceptance
  gates, not a substitute for fixing it.

This closes the last cheap hypothesis. The remaining path to fixing the abstention is domain
adaptation under the pre-registered protocol in
[GATE_CONTRASTIVE_PROTOCOL_V1](GATE_CONTRASTIVE_PROTOCOL_V1.md), which requires independent review
before any training run.

## What this does NOT establish

- **Four maze states (16 questions) per variant and two shortened engineering questions.** This is
  a probe, not a study; it can exclude a large effect, not measure a small one.
- The padding is irrelevant *engineering prose*, not real repository noise, and confidence was the
  only metric — accuracy on the padded/reworded maze variants was not scored, so "confidence
  unchanged" is not the same as "answers unchanged".
- One checkpoint, one device, one batch. The survey questions used here are two of thirteen.
- This says nothing about how much engineering data would be enough, or whether the 0.6B head can
  express the discrimination at all.

## Reproduction

```bash
python3 scripts/probe_confidence_attribution_v1.py
```

Receipt: `results/confidence_attribution_v1.json` (per-state medians, raw per-question
confidences, event id `7b868f9e-6f29-44a8-aaaf-1e77fe9c4a47`).
