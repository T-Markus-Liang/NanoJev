# The drop-label dilemma — full analysis (W148, 2026-10-07)

Evidence from v5 → v6 ablation → v7 → clef adjudication → v8 epoch-1.
This document consolidates every experiment into one causal picture and
defines the remaining viable paths.

## 1. The question

The frozen main tier (525) plus drop supplement (65) plus v5ext (588) is
the only distribution that matters: real transcript windows with
adjudicated labels. Production needs BOTH:

- FP≈0 at the real keep boundary (safety: never drop needed evidence)
- dRec>~0.5 at the real drop boundary (utility: actually filter)

v4 achieves (FP 10, dRec .65 merged) but fails hard-keeps (v5ext FP 304).
Every attempt to add drop-labeled training data since v4 has failed to
improve on this trade-off. Why?

## 2. Evidence matrix

| source of drop labels | rows | FP effect | dRec effect | verdict |
|---|---|---|---|---|
| v4_base synthetic (v1-v4 era) | ~8,000 | baseline | .65 | works — clean generation |
| F2/F4 mined drops (v5) | ~341 | **poison** (FP→178) | no help | 67% debatable; clef rejects 364/364 |
| rule-audit verified (v7) | 151 | FP fixed | .04 — too few | direction right, volume wrong |
| foreign injection (v8) | 928 | FP fine | .04 — doesn't transfer | pattern too easy |
| clef adjudicated (v8) | 0/364 passed | — | — | teacher too conservative |
| mined keeps (v5-v8) | ~928 | **protective** (removing → FP 272) | — | keep them |

Contrastive arms prove causality:
- conservative (no mined drops): FP=0, dRec=0
- minus_f2 (no mining at all): FP=272 — mined keeps carry the keep boundary
- minus_f1f3 (no synthetic families): real .912 but synth .566 — synthetic
  families teach the synthetic distribution and nothing else
- v7b ≈ v7, v8b > v8 on real tiers: synthetic families are net noise

## 3. The failure taxonomy

Every drop-label source fails one of two ways:

**Poisoned** — label says drop but truth is keep-or-debatable (mined
drops). Trains confident over-dropping.

**Untransferable** — label is mechanically correct but the pattern is too
easy (foreign injection: "obviously alien content = drop"). The real
boundary is "related-looking but actually superseded/task-shifted" —
subtle, not alien.

The gap in the middle — *subtle but certainly irrelevant* — is exactly
what neither mining nor mechanical generation produces correctly.

## 4. Why v4 still wins

v4_base's drops were **designed synthetic drops** (generator families
v1-v4): clean semantics, unambiguous, diverse. They transfer to real
drops at .65 recall because the generator encoded the *semantic* notion
(superseded, task-shifted, boilerplate) not a surface pattern.

Its weakness (v5ext FP 304) is over-dropping on hard keeps — covered in
production by winnow∧consensus (FP=0).

## 5. Viable paths forward

### Path A — agent-adjudicated mining at scale (the honest one)
Mine liberally (looser proposals → 2-5k candidates), adjudicate each with
the *main model* (not clef — it proved useless as a drop judge, dRec .26).
Cost: thousands of adjudications (~2-5h agent time), but produces
correct subtle-drop labels at the scale v7 lacked (151 → target 2k+).
Risk: none — this is what worked for the 590-row eval set.

### Path B — harder mechanical drops (cheap, incremental)
Upgrade injection: instead of foreign-content, inject **superseded-value**
segments — take a real transcript, find a fact that was later corrected,
re-attach the OLD value as the candidate under the post-correction
anchor. Label drop is mechanically correct (superseded = certainly
irrelevant by construction) and the pattern is subtle (same domain,
plausible content). Complements Path A, doesn't replace it.

### Path C — production status quo (zero work)
winnow + lora_v4 + consensus already achieves FP=0 with usable recall.
Marginal value of a better single head is modest — the remaining gap is
drop coverage on the hardest 35% of supp, which even clef-9B misses.

### Path D — scale the backbone (structural)
If the 0.8B head can't express the subtle boundary even with perfect
labels, the ceiling is representational. clef-flash's failure on supp
(.26) suggests even 9B struggles — but it's out-of-domain. A
LoRA-on-9B trained on adjudicated real data is the obvious next step up;
L40×2 handles 9B LoRA fine. Depends on A (needs the same labels anyway).

## 6. Recommendation

**Sequence: B first (hours, mechanical, no risk), then A (the real fix),
then re-evaluate whether D is needed.**

- B produces a v9 corpus overnight: conservative + 151 + subtle-injection
  (~1-2k pairs). If v9 lifts dRec to ~.3+ while holding FP≈0, subtlety
  was the missing ingredient.
- A is the real answer regardless: ~2-5k adjudicated labels settles the
  question definitively — either the boundary is learnable (v10 wins) or
  it's not (v4+consensus is the ceiling, stop burning GPU).
- D only after A proves labels aren't the bottleneck.

## 7. What we stop doing

- No more mined drops as labels (proven poison 3 ways)
- No more synthetic families F1/F3 (net noise on real tiers)
- No more unaugmented foreign injection alone (doesn't transfer)
- No retraining without adjudicated labels — every GPU run without them
  has produced the same conclusion
