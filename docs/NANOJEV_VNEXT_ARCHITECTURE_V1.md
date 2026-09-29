# NanoJev vNext architecture decision V1

Status: **frozen implementation target for the next research build; no deployment or
active filtering decision**.

## Question

After the open-source reproduction sweep and the V2 context-filter fixture run,
should NanoJev vNext be:

1. a direct wrapper around Winnow-12B;
2. a controller plus small fast-path scorer plus Winnow fallback;
3. a continued standalone small-model training effort?

## Evidence reviewed

- Frozen offline bundle: `1,720` rows / `893` labeled.
- Public aggregate scorecard: `BENCHMARKS.md`.
- Winnow audit: `docs/WINNOW_REVIEW_V1.md`.
- Context-gate shadow run: `docs/CONTEXT_GATE_WINNOW_SHADOW_V1.md`.
- Context-filter value fixtures: `docs/CONTEXT_FILTER_VALUE_V2.md`.
- Threshold policy: `docs/CONTEXT_FILTER_THRESHOLD_POLICY_V1.md`.
- Fast-path comparison: `docs/FAST_PATH_CASCADE_COMPARISON_V1.md`.
- Local paired evaluator: `docs/LOCAL_MAIN_MODEL_PAIR_V1.md`.
- Offline cascade replay: `results/cascade_simulation_winnow_fallback_v1.json`.

## Key measurements

### Frozen bundle quality

| Path | Accuracy | Estimated wall | Confident wrong@0.9 | Notes |
|---|---:|---:|---:|---|
| Winnow-12B Q8 alone | 0.8824 | 494.6s | 59 | strongest local quality baseline |
| Reflex stable alone | 0.7895 | 246.4s | 10 | fast and conservative, but lower accuracy |
| SemIf Qwen3.5-4B MLX4 alone | 0.7671 | 409.6s | 39 | lower accuracy |
| Reflex ≥0.9 → Winnow fallback | 0.8802 | 385.1s | 62 | near-Winnow accuracy, 59.6% fallback |
| SemIf ≥0.9 → Winnow fallback | 0.8757 | 261.3s | 76 | best speed/accuracy replay point measured |
| SemIf ≥0.95 → Winnow fallback | 0.8802 | 284.2s | 69 | closer to Winnow accuracy |

### V2 context-filter value fixture

| Path | Removed bytes | Removed local-BPE tokens | Unsafe removals | Paired answers | Fast / strong paths |
|---|---:|---:|---:|---:|---:|
| Winnow direct @0.90 | 5,560 | 1,431 | 0 | 25/25 | 0 / 19 |
| Reflex→Winnow @fast0.95 | 5,560 | 1,431 | 0 | 25/25 | 0 / 19 |
| SemIf→Winnow @fast0.80 | 5,278 | 1,348 | 0 | 25/25 | 1 / 18 |
| Decider-2B→Winnow @fast0.90 | 5,560 | 1,431 | 0 | 25/25 | 1 / 18 |
| Kev-4B direct @0.90 | 5,560 | 1,431 | 0 | 25/25 | n/a |
| Kev-9B direct @0.90 | 4,799 | 1,227 | 0 | 25/25 | n/a |
| Kev-4B→Winnow @fast0.95 | 5,560 | 1,431 | 0 | 25/25 | 13 / 6 |

The V2 result is still synthetic evidence preservation, not real downstream model
quality or provider billing.

Subsequent pinned local-generation checks now cover `Qwen3-0.6B`,
`Qwen2.5-3B-Instruct`, and `Qwen3.5-4B`. After chat-template support and normalized
answer matching, all preserve their original answer counts under Winnow `0.90`
(`8/8`, `17/17`, and `18/18` originally successful cases respectively). These are
still small local models, so they strengthen—but do not complete—downstream quality
evidence.

## Frozen decision

Adopt this architecture for the next research build:

```text
NanoJev controller
    │  owns segmentation, protected/dependency policy, scorer routing,
    │  receipts, restore manifests, and fail-open behavior
    ▼
optional fast-path scorer
    │  current reference: Kev-4B /v1/systemone
    │  alternatives measured: Reflex, SemIf-4B, Decider-0.8B/2B
    ▼
Winnow-12B Q8 strong path
    │  current local quality baseline and fallback scorer
    ▼
shadow removal plan / restore manifest
```

The controller contract is the owned component. The scorer role is replaceable:

- `strong`: Winnow-12B Q8 over loopback `/v1/systemone`.
- `fast`: Kev-4B is the first credible measured fast path.
- `future`: a NanoJev-owned scorer may replace Kev only after independently
  trained/calibrated evidence beats the current cascade on a development cohort.

## Why not direct Winnow-only as the architecture

- Winnow is the best local quality baseline, but it is a 12B external artifact.
- Its private training mixture prevents a contamination audit.
- It has `59` confident-wrong rows at `0.9` on the frozen bundle.
- The architecture goal is a local decision layer that can swap scorers; hard-wiring
  NanoJev to one 12B scorer would remove that flexibility.

## Why not small-model-only

The best measured small/frozen candidates remain materially below Winnow on the
bundle, and only Kev-4B matched the full V2 removal set in this fixture. A single
small scorer is not yet a safe replacement.

## Why Kev-4B is the current fast-path reference

- Direct @0.90: same `5,560` byte / `1,431` local-token reduction as Winnow on V2.
- Cascade @fast0.95: `13/19` candidate states resolved without a strong call.
- Zero unsafe removals and `25/25` deterministic paired answers preserved.
- Kev-9B was more conservative and slower (`4,799` bytes removed; p50 scorer
  latency about `1.01s`).

Limitation: Kev-4B's scorer latency (`~706ms` p50 direct) is currently worse than
Winnow's llama.cpp path (`~252ms` p50) on this harness. The cascade reduces strong
calls but is not yet a measured end-to-end latency win.

## Implementation gates

Before treating the cascade as a product path:

1. Route only clearly eligible assistant/history text to the small scorer.
2. Keep protected segments, dependencies, credentials, tool linkage, and user
   intent outside the removable set before scoring.
3. Keep thresholds diagnostic until a separate operational policy selects an
   operating point on a declared development cohort.
4. Add provider-reported prompt/completion token accounting.
5. Add a stronger representative downstream cohort or a real-request holdout; the
   small pinned local-generation backends now pass the V2 suite but remain a proxy.
6. Add scorer timeout/error/cache-eviction/kill-switch stress tests.
7. Measure memory pressure and end-to-end latency, not just scorer-call latency.

## Explicit non-decisions

- No active context deletion is authorized.
- No production threshold or release candidate is selected.
- No external benchmark submission is triggered.
- No financial-track model or trading action is implicated.
- Kev-4B is a reference fast-path scorer, not a NanoJev-owned model.
