# V7_RECIPE_DESIGN_V1 — v7 context-filter recipe design (post-v6 ablation)

**Status:** design spec + acceptance protocol. v7/v7b training already running
on L40×2 (W145); this doc pins the causal argument, predictions, gates, and
contingencies **before** scoring lands, so the verdict is pre-registered, not
post-hoc.
**Date:** 2026-10-02
**Inputs:** `docs/V5_ACCEPTANCE_VERDICT_V1.md` (v5 FAIL + v6 ablation verdict),
`docs/V5_DATA_DESIGN_V1.md` (families F1–F4, gate scheme G1–G6),
`data/v5_mining/label_audit.json` (rule audit of mined drop labels),
`docs/CURRENT_PROGRESS_AND_HANDOFF.md` (W142–W145).
**Privacy:** compressed descriptions only — no raw transcript text; real-corpus
artifacts stay under the gitignored `data/` root.

## 1. v6 ablation evidence — what poisoned v5

Four ablation arms of the v5 recipe, LoRA warmstart from `nano_sft_v4`, scored
on the same tiers (main 525 / supp 65 = frozen 590; v5ext 588 mined hard-keeps;
merged = weighted real acc; confFP = FP at noul≥0.9; dRec = drop recall):

| arm | recipe | main | supp | v5ext | merged | confFP | dRec |
|---|---|---|---|---|---|---|---|
| v4 incumbent | — | .988/FP4 | .615/FP6 | .481/FP304 | .947 | 7 | .65 |
| v5 | v4 + F1+F2+F3+F4 | .670/FP172 | .492/FP6 | .929/FP40 | .651 | 118 | .50 |
| minus_f2 | v5 − all mining (drops+keeps) | .478/FP272 | .815/FP6 | — | .515 | 253 | .89 |
| minus_f3 | v5 − F3 boilerplate pairs | .726/FP142 | .477/FP2 | .991/FP3 | .698 | 110 | .41 |
| minus_f1f3 | v5 − F1 − F3 synth | .912/FP44 | .354/FP1 | .990/FP4 | .851 | 22 | .24 |
| **conservative** | v5 − ALL mined drops | **.996/FP0** | .169/FP0 | **.997/FP0** | .905 | **0** | .00 |

Causal chain (three eliminations, one positive identification):

1. **f1/f3 synthetic drop labels are innocent.** conservative keeps them
   verbatim and reaches FP=0 on every real tier, confFP=0.
2. **Mined keeps are protective, not neutral.** minus_f2 removed drops *and*
   keeps → catastrophic FP 272. Removing the keep mass unbalanced training
   toward the drop prior; the mined keeps were actively holding the boundary.
3. **f1f3 removal trades real for synth.** minus_f1f3 improves real FP (44)
   but synth eval collapses to 0.566 — synthetic families carry the synthetic
   distribution; they are not what damaged the real boundary.
4. **The poison = ~341 mined drop labels.** Rule audit
   (`data/v5_mining/label_audit.json`): F2 yes-proposals 288 → 193 debatable /
   91 likely_correct / 4 likely_wrong (`cross_task_reanchor` = 261 of them,
   ~67% debatable); F4 76 → 60 likely_correct / 16 debatable. Borderline-grade
   proposals entered train as hard `true` labels — "foreign-looking under the
   anchor ⇒ drop" — exactly the confident-over-drop signature v5 showed.

## 2. v7 / v7b recipes

| arm | composition | rows | role |
|---|---|---|---|
| **v7** | conservative (v4_base + f1 + f3 + mined keeps, zero debatable drops) **+ 151 rule-audit-verified drops** (91 F2 `likely_correct` + 60 F4 `likely_correct`) | 20,782 | candidate recipe |
| **v7b** | v4_base + mined keeps + same 151 drops — **no synthetic families at all** | 18,240 | control arm |

Both LoRA warmstart from `nano_sft_v4`, L40 GPU0/GPU1 (~3h each).

**Rationale.** conservative proved the safe recipe (FP=0) but paid for it with
dRec=0.00 — with no mined drop positives at all, the head reverts to "keep
everything" on real-distribution candidates (supp acc .169 ≈ drops nothing).
v7 adds back only the audit-verified slice: real-distribution drop positives
whose labels survive the rule audit, i.e. the same mechanism minus the
debatable mass.

**Predictions (pre-registered):**

- **v7**: FP ≈ 0–2 on frozen 590, confFP = 0, **nonzero dRec** (151 verified
  drops restore real drop-positive signal; magnitude unknown — anywhere from
  v4's .65 down is informative). v5ext stays ≥ conservative's .997 (keeps
  unchanged, drops are verified non-overlapping). Synth eval ≈ 1.0 (f1/f3
  retained).
- **v7b**: answers "are synthetic families needed at all?" If v7b ≈ v7 on all
  real tiers → the entire v1–v4 synthetic emitter stack is dispensable for the
  real boundary, and the future recipe collapses to v4_base + adjudicated real
  labels. Expected: real tiers near v7, synth eval collapses (minus_f1f3
  precedent, 0.566) — that collapse is diagnostic, not a defect: it would mean
  synthetic families mainly teach the synthetic distribution.

## 3. Acceptance criteria — v7/v7b vs v4 incumbent

Same-device fp32 MPS scoring; identical tier set for every arm; τ=0.5 primary
with the fixed diagnostic grid reported (eval never selects production τ).

| gate | tier | criterion | verdict rule |
|---|---|---|---|
| V7-G1 FP superiority | frozen 590 | v7 FP < v4's 10 **and** confFP(noul≥0.9) = 0 | hard gate |
| V7-G2 recall non-regression | frozen 590 | v7 dRec > 0 **and** ≥ v4's 62.5% for full PASS; 0 < dRec < .625 = partial (see §4b) | hard gate at >0 |
| V7-G3 merged | frozen 590 | merged acc ≥ v4's .947 | hard gate |
| V7-G4 v5ext | 588 mined hard-keeps | acc ≥ .99 / FP ≈ 0 (conservative = .997/FP0) | hard gate |
| V7-G5 synth | v4 eval + v5 synth eval | ≥ 0.99 each (conservative scored 1.000) | v7 only; v7b exempt, collapse is diagnostic |
| V7-G6 consensus | frozen 590 | `v7 ∧ winnow` AND-drop FP = 0 | hard gate |
| V7-G7 asymmetry | all | error mass FN-sided; any FP on F1/F4-type required-evidence rows = fail | hard gate |

**PASS** = all hard gates for v7. **v7b interpretation** (reported alongside,
does not block v7): v7b ≈ v7 on real tiers ⇒ synthetic families unnecessary
for the real boundary (recipe simplification opportunity); v7b ≪ v7 ⇒ f1/f3
carry real-boundary value (keep them); v7b ≫ v7 ⇒ synthetic families actively
harmed the real boundary even in v5-minus-poison form.

## 4. Failure contingencies

| observation | interpretation | action |
|---|---|---|
| **4a. v7 still over-drops** (FP > 0 on frozen or any confFP) | rule-audit `likely_correct` is still not adjudication-grade; no rule-verified drop label is trustworthy | drop the verified 151 entirely; ship **pure conservative** (FP=0 measured); future drops only via owner-delegated adjudication per `REAL_CONTEXT_LABELING_GUIDE_V1` — never rule-only |
| **4b. v7 under-drops** (FP=0 but dRec ≈ 0 or ≪ .625) | 151 verified rows insufficient mass, or verified slice is too easy a distribution | grow the verified set via **stricter mining**: keep only high-precision strategies (`stale_superseded` ~all likely_correct; `cross_task_reanchor` needs explicit task-shift marker + zero downstream reference), then route through the audit → adjudication before train |
| **4c. v7 FP=0 but supp/v5ext regress** | verified drops re-taught a form prior | bisect the 151 by strategy (F2-anchor vs F4-outcome) in a follow-up arm |
| **4d. v7b ≫ v7 on real** | synthetic families net-negative on real boundary | v8 = v4_base + adjudicated real labels; retire f1/f3 emitters for real-target training |

## 5. The deeper lesson — proposals are weak labels, not ground truth

v5's failure was a **label-authority violation**, not a mining failure. The
design doc already said it (`V5_DATA_DESIGN_V1` §6: "model/scorer proposals are
never labels"), yet in execution 288 F2 yes-proposals entered train as hard
`true` labels gated only by a regex veto. Post-hoc rule audit showed 67% of
them debatable — under the asymmetric certain-only-drop contract, a debatable
drop label is a **wrong** label, and ~190 wrong hard labels were enough to
destroy the natural-distribution boundary (FP 172→0 when removed).

Corollaries now measured, not hypothesized:

- Mined keeps are cheap and protective; mined drops are expensive and
  dangerous — the asymmetry is intrinsic to the contract, so mining budgets
  should skew keep-heavy by default.
- Rule-audit verdicts (`debatable`/`likely_correct`/`likely_wrong`) are
  **triage grades for an adjudicator**, not a substitute for one. v7 tests
  whether `likely_correct` alone is trainable-grade; 4a is the contingency
  if not.
- The durable fix is an adjudication pass with `evidence_basis` per row
  (owner-delegated agent adjudication, the same pass that built the frozen
  590 + v5ext labels) — mining generates candidates, adjudication mints
  labels. Any future recipe that skips that pass repeats v5.
