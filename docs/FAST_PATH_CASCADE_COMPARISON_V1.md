# Fast-path scorer and cascade comparison V1

Status: **local synthetic shadow/offline measurement; no active filtering decision**.

This report compares local fast-path scorers on the expanded 25-case V2 fixture.
The strong path is Winnow `/v1/systemone`. The final removal threshold is the
diagnostic `0.90`; the cascade fast threshold controls whether a fast answer avoids
the strong scorer.

## Measured arms

| Fast path / scorer | Cascade fast threshold | Fast paths | Strong paths | Removed bytes | Removed local-BPE tokens | Unsafe removals | Paired answers | Scorer p50 ms | Scorer p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Winnow only | n/a | 0 | 19 effective | 5,560 | 1,431 | 0 | 25/25 | 251.9 | 411.5 |
| Reflex | 0.95 | 0 | 19 | 5,560 | 1,431 | 0 | 25/25 | 446.1 | 729.5 |
| SemIf-4B MLX4 | 0.80 | 1 | 18 | 5,278 | 1,348 | 0 | 25/25 | 353.7 | 437.1 |
| Decider-2B | 0.80 | 4 | 15 | 4,567 | 1,170 | 0 | 25/25 | not rerun after latency instrumentation | — |
| Decider-2B | 0.90 | 1 | 18 | 5,560 | 1,431 | 0 | 25/25 | 421.1 | 583.3 |
| Kev-4B direct | n/a | n/a | n/a | 5,560 | 1,431 | 0 | 25/25 | 706.0 | 1,046.7 |
| Kev-9B direct | n/a | n/a | n/a | 4,799 | 1,227 | 0 | 25/25 | 1,010.9 | 1,488.1 |
| Kev-4B → Winnow | 0.95 | 13 | 6 | 5,560 | 1,431 | 0 | 25/25 | 911.1 | 1,286.6 |

Direct scorer observations on the same fixture:

- Reflex maximum observed `p_irrelevant`: about `0.773`; no direct removals at
  `0.99`.
- SemIf maximum observed `p_irrelevant`: about `0.852`; no direct removals at
  `0.99`.
- Decider-0.8B maximum observed `p_irrelevant`: about `0.575`; no direct removals
  at `0.99`.
- Decider-2B maximum observed `p_irrelevant`: about `0.913`; still no removals at
  frozen `0.99`, but one clearly irrelevant candidate can be handled by the fast
  path at cascade fast threshold `0.90`.
- Kev-4B's `p_irrelevant` values are mostly `0.90–0.98`; at diagnostic `0.90` it
  removes the same 5,560 bytes as Winnow with zero unsafe removals.
- Kev-9B is more conservative on this fixture: `4,799` removed bytes at `0.90`,
  zero unsafe removals, and roughly `1.01s` p50 scorer latency.

## Interpretation

1. **Reflex is not an effective fast path on this fixture.** At `0.95`, it never
   produces a confident answer, so all 19 eligible candidate states go to Winnow.
2. **SemIf offloads one candidate at fast `0.80`, but loses one expected removal.**
   It produces `5,278` removed bytes instead of Winnow's `5,560`; the reduction is
   still safe by the paired evidence contract, but coverage is lower.
3. **Decider-2B has a narrow fast-path signal.** At cascade fast threshold `0.90`,
   it handles `mutable_file_read_state` locally, preserves the same reduction as
   Winnow, and leaves 18 candidates to the strong path. At `0.80` it offloads four
   candidates but loses three expected removals.
4. **Kev-4B is the best measured fast path.** As a direct scorer at `0.90`, it
   matches Winnow's reduction with zero unsafe removals. In cascade mode with fast
   threshold `0.95`, it resolves `13/19` candidate states and still preserves the
   full Winnow reduction. Its local server latency is currently worse than
   Winnow's llama.cpp server on this fixture, so the route share is useful but not
   yet a net-latency win.
5. **Deterministic and local-generation checks both pass on this suite.** All
   measured arms retain protected evidence and deterministic paired answers. The
   pinned local-generation backends (`Qwen3-0.6B`, `Qwen2.5-3B-Instruct`,
   `Qwen3.5-4B`) preserve their original answer counts under Winnow @0.90 after
   answer normalization was fixed.
6. **This is not a production latency measurement.** `scorer_latency_ms` measures
   this Python harness's scorer call on the synthetic fixture; it includes the
   selected cascade path but excludes provider work, request serving overhead, and
   memory pressure.

## Files

- Reflex cascade: `results/filter_value_v2_cascade_reflex_t090_f095_v1.json`
- SemIf cascade: `results/filter_value_v2_cascade_semif_t090_f080_v1.json`
- Decider-2B cascade fast `0.80`: `results/filter_value_v2_cascade_decider2b_t090_f080_v1.json`
- Decider-2B cascade fast `0.90`: `results/filter_value_v2_cascade_decider2b_t090_f090_v1.json`
- SemIf direct: `results/filter_value_v2_semif_v1.json`
- Decider-0.8B direct: `results/filter_value_v2_decider08_v1.json`
- Decider-2B direct: `results/filter_value_v2_decider2b_v1.json`
- Kev-4B direct: `results/filter_value_v2_kev4b_t090_v1.json`
- Kev-9B direct: `results/filter_value_v2_kev9b_t090_v1.json`
- Kev-4B cascade: `results/filter_value_v2_cascade_kev4b_t090_f095_v1.json`
- Local-generation Qwen3-0.6B control/dedup: `results/filter_value_v2_control_dedup_localgen_qwen3_0p6b_v1.json`
- Local-generation Qwen3-0.6B Winnow: `results/filter_value_v2_winnow_t090_localgen_qwen3_0p6b_v1.json`
- Local-generation Qwen2.5-3B control/dedup: `results/filter_value_v2_control_dedup_localgen_qwen25_3b_v1.json`
- Local-generation Qwen2.5-3B Winnow: `results/filter_value_v2_winnow_t090_localgen_qwen25_3b_v1.json`
- Local-generation Qwen3.5-4B control/dedup: `results/filter_value_v2_control_dedup_localgen_qwen35_4b_v1.json`
- Local-generation Qwen3.5-4B Winnow: `results/filter_value_v2_winnow_t090_localgen_qwen35_4b_v1.json`

## Conclusion

The best current architecture proposal remains:

```text
controller + small local fast path + Winnow strong-path fallback
```

The measured fast-path ranking is now clear:

1. Kev-4B: `13/19` cascade fast paths and full Winnow reduction, but slower than
   Winnow in this Python/MLX-PyTorch harness.
2. Decider-2B: `1/19` fast paths without losing reduction at fast `0.90`.
3. SemIf-4B: `1/19` fast paths and lower reduction at fast `0.80`.
4. Reflex: `0/19` fast paths at fast `0.95`.

T20 therefore supports keeping the cascade design and marks Kev-4B as the first
credible local fast path. It does not yet justify replacing Winnow or enabling an
active path: the evidence is synthetic, latency is not a net win, and local
generation remains a small-model diagnostic rather than a representative production
main-model check.
