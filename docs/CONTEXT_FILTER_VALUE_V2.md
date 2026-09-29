# Context filter value fixture V2

Status: **expanded synthetic shadow/offline measurement; no provider call; no active filtering decision**.

V2 expands the eight-case value fixture to **25 self-authored cases** across **25
named families**. It keeps the V1 cases and adds multilingual, policy/credential,
correction, non-repeatable/mutable tool evidence, retrieval distractor, protected
instruction, and unsupported/bypass coverage.

Manifest:

- `research/context_filter_value_fixture_manifest_v2.json`
- Builder: `scripts/build_filter_value_fixtures_v2.py`
- Runner: `scripts/benchmark_filter_value_v1.py`
- Paired evaluator: `scripts/local_main_model_evaluator_v1.py`

The manifest is deterministic and can be checked with:

```bash
python3 scripts/build_filter_value_fixtures_v2.py --check
```

## Added coverage

- Multilingual clear-drop and ambiguous-retain cases.
- Policy plus placeholder-credential context; placeholders remain protected.
- Correction chains where a stale value is required by dependency closure.
- Non-repeatable export IDs and mutable file-read timestamps.
- Long tool traces and long unrelated tails.
- Cited retrieval evidence versus uncited distractors.
- Current user instructions and code-review configuration evidence.
- Pending tool call, orphan tool result, unsupported image-like content,
  unresolved server context, and unknown envelope fields, all requiring bypass.

## Results

### Control and safe dedup

| Arm | Removed bytes | Byte reduction | Removed local-BPE tokens | Token reduction | Paired answers | Unsafe removals |
|---|---:|---:|---:|---:|---:|---:|
| control | 0 | 0.0% | 0 | 0.0% | 25/25 | 0 |
| safe_dedup | 246 | 1.7% | 74 | 2.0% | 25/25 | 0 |
| deterministic stub | 6,452 | 44.1% | unavailable | unavailable | 24/25 | **1 detected** |

The stub still intentionally drops required evidence in
`required_evidence_marked_eligible`; the paired evaluator records a regression.

### Model arms at frozen threshold `0.99`

| Arm | Removed bytes | Token reduction | Paired answers | Unsafe removals |
|---|---:|---:|---:|---:|
| Winnow `/v1/systemone` | 0 | 0.0% | 25/25 | 0 |
| Reflex stable | 0 | 0.0% | 25/25 | 0 |

Winnow's eligible-segment irrelevant probabilities ranged from about `0.027` for
ambiguous/retain cases to `0.982` for clear distractors; all clear-drop candidates
were below the frozen `0.99` gate. Reflex's observed maximum irrelevant probability
was about `0.773`.

### Diagnostic threshold `0.90`

| Arm | Removed bytes | Byte reduction | Removed local-BPE tokens | Token reduction | Paired answers | Unsafe removals |
|---|---:|---:|---:|---:|---:|---:|
| Winnow `/v1/systemone` | 5,560 | 38.0% | 1,431 | 38.4% | 25/25 | 0 |
| Kev-4B `/v1/systemone` | 5,560 | 38.0% | 1,431 | 38.4% | 25/25 | 0 |
| Kev-9B `/v1/systemone` | 4,799 | 32.8% | 1,227 | 32.9% | 25/25 | 0 |
| Reflex→Winnow cascade, fast threshold 0.95 | 5,560 | 38.0% | 1,431 | 38.4% | 25/25 | 0 |
| SemIf→Winnow cascade, fast threshold 0.80 | 5,278 | 36.1% | 1,348 | 36.2% | 25/25 | 0 |
| Decider-2B→Winnow cascade, fast threshold 0.90 | 5,560 | 38.0% | 1,431 | 38.4% | 25/25 | 0 |
| Kev-4B→Winnow cascade, fast threshold 0.95 | 5,560 | 38.0% | 1,431 | 38.4% | 25/25 | 0 |

Winnow removed all declared clear-drop segments on 14 scored cases while retaining
required evidence, ambiguous history, dependency-linked stale values, tool linkage,
policy/credential placeholder context, and unsupported/bypass requests.

Route accounting differs sharply by fast path:

```text
Reflex→Winnow fast0.95:      0 fast / 19 strong
SemIf→Winnow fast0.80:       1 fast / 18 strong
Decider-2B→Winnow fast0.90:  1 fast / 18 strong
Kev-4B→Winnow fast0.95:     13 fast / 6 strong
```

Kev-4B is the first measured fast path that preserves the full Winnow reduction.
It is still slower than Winnow-only in this harness, so it is a credible route
candidate rather than a demonstrated latency win.

### Pinned local-generation downstream check

A second paired evaluator generates deterministic answers with pinned local models.
After chat-template support and normalized answer comparison, three local backends
preserve all originally successful answers under Winnow `0.90`:

| Backend | Control answers | Dedup answers | Winnow reduced answers | Paired regressions |
|---|---:|---:|---:|---:|
| `Qwen3-0.6B` | 8/25 | 8/25 | 8/25 | 0 |
| `Qwen2.5-3B-Instruct` | 17/25 | 17/25 | 17/25 | 0 |
| `Qwen3.5-4B` | 18/25 | 18/25 | 18/25 | 0 |

The models' differing baseline success rates are useful stress evidence, but these
are still small local models and not representative of a production main model.

## Interpretation

1. **V2 broadens the value evidence:** Winnow's diagnostic `0.90` result is no longer
   based only on the original eight cases.
2. **Structural and local-generation checks now agree on this suite:** protected
   drops, required-string failures, and paired local-generation regressions are all
   zero for Winnow at `0.90` across Qwen3-0.6B, Qwen2.5-3B-Instruct, and
   Qwen3.5-4B after answer normalization was fixed.
3. **Bypass coverage is working:** pending/orphan tool linkage, unsupported content,
   unresolved server context, and unknown envelope fields never reach scorer removal.
4. **The frozen `0.99` policy remains non-productive:** it retains every segment.
5. **The result is still not production evidence:** cases are synthetic, downstream
   answers are deterministic, and token counts are local estimates.

## Files

- Manifest: `research/context_filter_value_fixture_manifest_v2.json`
- Builder: `scripts/build_filter_value_fixtures_v2.py`
- Tests: `scripts/test_filter_value_v1.py`
- V2 self-test: `results/filter_value_v2_selftest_v1.json`
- Winnow `0.99`: `results/filter_value_v2_winnow_t099_v1.json`
- Winnow `0.90`: `results/filter_value_v2_winnow_t090_v1.json`
- Reflex `0.99`: `results/filter_value_v2_reflex_v1.json`
- SemIf direct `0.99`: `results/filter_value_v2_semif_v1.json`
- Decider-0.8B direct `0.99`: `results/filter_value_v2_decider08_v1.json`
- Decider-2B direct `0.99`: `results/filter_value_v2_decider2b_v1.json`
- Kev-4B `0.90`: `results/filter_value_v2_kev4b_t090_v1.json`
- Kev-9B `0.90`: `results/filter_value_v2_kev9b_t090_v1.json`
- Reflex cascade `0.90`: `results/filter_value_v2_cascade_reflex_t090_f095_v1.json`
- SemIf cascade `0.90`: `results/filter_value_v2_cascade_semif_t090_f080_v1.json`
- Decider cascade `0.90`: `results/filter_value_v2_cascade_decider2b_t090_f090_v1.json`
- Kev cascade `0.90`: `results/filter_value_v2_cascade_kev4b_t090_f095_v1.json`
- Local-generation control/dedup: `results/filter_value_v2_control_dedup_localgen_qwen3_0p6b_v1.json`
- Local-generation Winnow `0.90`: `results/filter_value_v2_winnow_t090_localgen_qwen3_0p6b_v1.json`
- Local-generation Qwen2.5-3B control/dedup: `results/filter_value_v2_control_dedup_localgen_qwen25_3b_v1.json`
- Local-generation Qwen2.5-3B Winnow: `results/filter_value_v2_winnow_t090_localgen_qwen25_3b_v1.json`
- Local-generation Qwen3.5-4B control/dedup: `results/filter_value_v2_control_dedup_localgen_qwen35_4b_v1.json`
- Local-generation Qwen3.5-4B Winnow: `results/filter_value_v2_winnow_t090_localgen_qwen35_4b_v1.json`
- Held-out V3 report: `docs/CONTEXT_FILTER_VALUE_V3.md`

## Non-decisions

- No production threshold is selected.
- No active context deletion is authorized.
- No provider token billing is claimed.
- No real LLM downstream quality is claimed.
- No external milestone or release candidate is promoted.
