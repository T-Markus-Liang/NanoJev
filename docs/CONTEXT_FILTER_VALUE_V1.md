# Context filter value fixture V1

Status: **synthetic shadow/offline measurement; no provider call; no active filtering decision**. The expanded 25-case suite is in [Context filter value fixture V2](CONTEXT_FILTER_VALUE_V2.md).

## Why this exists

The A4 fixture suite proved the context gate fails open and never removes protected
segments. It did not prove filtering value because the strong scorer proposed zero drops
under the frozen `0.99` threshold.

`research/context_filter_value_fixture_manifest_v1.json` adds eight self-authored
synthetic cases designed to distinguish three behaviors:

- clearly irrelevant eligible history should be removable;
- required evidence marked eligible by a bad sidecar must still be retained;
- ambiguous history and dependency-linked stale values must still be retained.

The suite includes clear irrelevant text, a long noise segment, a text-part removal,
a tool call/result pair, an exact duplicate, a dependency-protected correction,
required evidence, and ambiguous history.

## Runner

`scripts/benchmark_filter_value_v1.py` evaluates:

- `control`: retain everything;
- `safe_dedup`: deterministic exact-duplicate arm;
- `systemone`: local Winnow `/v1/systemone`;
- `reflex`: local Reflex/Qwen3.5-4B in-process scorer;
- `cascade`: Reflex first, Winnow fallback for uncertain answers;
- `stub`: deterministic high-irrelevance test scorer used only to prove unsafe drops
  are detected.

For model arms, the normal `shadow_request` receipt is converted into a removal plan
and applied only in-process through the existing reversible reduction applicator.
No reduced bytes are sent to any provider.

`scripts/local_main_model_evaluator_v1.py` then evaluates the original and reduced
request as a pair through the `deterministic-evidence-v1` backend. It answers the
declared expected answer only when every declared required string remains; otherwise
it returns `insufficient_evidence`. Reports include only status, answer booleans,
response hashes, required-string hashes, and local BPE token counts. This is a
deterministic downstream check, not a real model-quality benchmark.

## Results

### Control and deterministic arms

| Arm | Removed bytes | Byte reduction | Removed local-BPE tokens | Token reduction | Unsafe removals |
|---|---:|---:|---:|---:|---:|
| control | 0 | 0.0% | 0 | 0.0% | 0 |
| safe_dedup | 246 | 4.8% | 74 | 5.7% | 0 |
| deterministic stub | 2,768 | 53.9% | unavailable | unavailable | **1 detected** |

The stub intentionally drops required evidence in `required_evidence_marked_eligible`;
the runner flags that case as unsafe. This is a negative control, not a filtering result.

### Model arms at frozen threshold `0.99`

| Arm | Removed bytes | Unsafe removals | Paired answers | Observation |
|---|---:|---:|---:|---|
| Winnow `/v1/systemone` | 0 | 0 | 8/8 | irrelevant p-values were 0.948–0.976, below 0.99 |
| Reflex stable | 0 | 0 | 8/8 | observed p-values were 0.207–0.750 |
| Reflex→Winnow cascade | 0 | 0 | 8/8 | all 9 eligible candidates routed to Winnow |

### Diagnostic threshold `0.90`

| Arm | Removed bytes | Byte reduction | Removed local-BPE tokens | Token reduction | Paired answers | Unsafe removals |
|---|---:|---:|---:|---:|---:|---:|
| Winnow `/v1/systemone` | 2,307 | 44.9% | 569 | 44.1% | 8/8 | 0 |
| Reflex→Winnow cascade, fast threshold 0.95 | 2,307 | 44.9% | 569 | 44.1% | 8/8 | 0 |

Winnow dropped the clear irrelevant history, the unrelated tool-history note, the
irrelevant text part, both duplicate weather notes, and the long-tail noise segment.
It retained required evidence and ambiguous history. The dependency case bypassed
before scoring because dependency closure retained the only eligible candidate.

All five non-empty reduction plans were byte-reversible: `5/5` restore round-trips
passed. No required evidence string was lost.

## Interpretation

1. **The safety mechanics work:** protected/tool/dependency structures stayed intact,
   and the runner detects unsafe evidence loss.
2. **`0.99` is too conservative for value measurement:** Winnow's strongest irrelevant
   scores were around `0.95–0.98`, so the frozen threshold produced zero savings.
3. **Winnow can produce large hypothetical savings on intentionally noisy context:**
   `44.9%` bytes and `44.1%` local-BPE tokens in this suite, with `8/8` paired
   deterministic answers preserved and zero required-string loss.
4. **Reflex did not provide a useful fast path here:** its highest irrelevant score
   was `0.750`, so a `0.95` cascade acceptance threshold sent every candidate to
   Winnow.
5. **The cascade adapter is functional but unproven as a speed optimization:** the
   measured cascade produced identical filtering to Winnow but always used the strong
   path.

## Files

- Fixture manifest: `research/context_filter_value_fixture_manifest_v1.json`
- Runner: `scripts/benchmark_filter_value_v1.py`
- Paired evaluator: `scripts/local_main_model_evaluator_v1.py`
- Evaluator contract: `docs/LOCAL_MAIN_MODEL_PAIR_V1.md`
- Self-test: `results/filter_value_selftest_v1.json`
- Winnow/control/dedup at `0.90`: `results/filter_value_winnow_t090_v1.json`
- Reflex at `0.99`: `results/filter_value_reflex_v1.json`
- Cascade at `0.90`: `results/filter_value_cascade_t090_v1.json`

## Non-decisions

- No production threshold is selected.
- No active context deletion is authorized.
- No provider token savings are claimed; local BPE counts are estimates over canonical
  JSON bodies, not provider billing.
- No downstream model task-success measurement has been run yet.
