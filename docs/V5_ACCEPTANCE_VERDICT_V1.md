# V5 Acceptance Verdict — FAIL (2026-10-01)

**Decision: v5 recipe rejected. Production unchanged: winnow default +
lora_v4 shadow + v4∧winnow consensus (FP=0).**

## Gate results

| Gate | requirement | v5 result | verdict |
|---|---|---|---|
| G1 | acc>0.9453, FP<11, FP=0 @noul≥0.9 | acc 0.647, FP 179, confident-FP 118 | FAIL |
| G2 | consensus FP=0 | v5∧winnow FP=4 | FAIL |
| G3 | transfer gap ≤10pp | not measured (slots empty) | NOT_READY |
| G4 | synth non-regression, no saturation | v4eval 1.000, v5eval 0.991 | PASS |
| G5 | FP≤FN asymmetric | FP 179 vs FN 32 | FAIL |
| G6 | calibration tail not worse | ECE 0.309 vs incumbent 0.041 | FAIL |

## Failure mode

- 179 FPs on frozen 590 at τ=0.5; **79 still FP at τ=0.99** — confident wrong drops,
  not a calibration issue. Threshold cannot rescue it.
- FP breakdown: assistant_text 75 / tool_result 67 / user_turn 36 — all kinds.
- Confident-drop content: task-plan restatements, "继续" micro-instructions,
  task constraints — surfaces that resemble harness boilerplate.

## Root-cause hypothesis

F3 boilerplate contrastive pairs (750 pairs, envelope→drop) + F1 surface-drop
side + F2 stale-segment mined labels taught a **form prior** ("text that looks
like boilerplate = drop") instead of the intended **anchor-relevance** rule.
On real data, planning text/constraints/micro-turns all look boilerplate-like.
Symmetric mirror: v4 drops 304/586 of the mined hard-keep set (v5-ext) while
v5 drops only 40 — v5 did learn the adversarial-keep side, but at the cost of
the natural-distribution boundary.

## Next options (requires L40, ~3h per arm)

- Ablation: retrain minus F3 / minus F2 to localize the poison family
- Or rebalance: F3 drop side is likely over-weighted vs real keep mass
