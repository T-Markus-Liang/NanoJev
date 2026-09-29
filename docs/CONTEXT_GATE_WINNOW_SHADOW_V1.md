# Winnow / Reflex context-gate shadow V1

Status: **shadow-only diagnostic complete; active filtering remains disabled**.
This run connects local typed-decision scorers to the frozen A4 tool-history fixture
suite through `context_gate_v1.shadow_request`. It does not send provider calls,
does not modify request bytes, and does not tune thresholds on held-out outcomes.

## Adapter work delivered

`scorer_adapters_v1.py` now includes `SystemOneHTTPScorer`, a loopback-only adapter
that translates the gate's `{"states": [...]}` batch into individual
`POST /v1/systemone` calls and maps `noul` back to the gate's boolean
`{false,true}` contract.

`main_model_gateway_v1.py` accepts:

```bash
--scorer systemone \
--scorer-url http://127.0.0.1:8091 \
--scorer-endpoint /v1/systemone \
--scorer-model Winnow-12B
```

The adapter refuses non-loopback origins and carries no API key. A fake-service
gateway test verifies that shadow mode forwards the original bytes unchanged while
recording a proposed plan.

## Frozen A4 fixture result

Workload: `research/tool_history_fixture_manifest_v1.json`, seed `20260919`,
11 synthetic tool-history cases. Three cases bypass at the structural layer
(`unresolved_tool_link`); eight cases present one eligible candidate each.

| Scorer | Threshold | Scored cases | Proposed drops | Protected drops | Scorer calls | p50 / p95 latency |
|---|---:|---:|---:|---:|---:|---|
| Winnow-12B Q8 `/v1/systemone` | 0.99 | 1 / 8 | 0 | 0 | 8 | 389.1 / 561.0 ms |
| Winnow-12B Q8 `/v1/systemone` | 0.90 diagnostic | 2 / 8 | 0 | 0 | 8 | 386.9 / 462.5 ms |
| Reflex Qwen3.5-4B stable | 0.99 | 0 / 8 | 0 | 0 | 8 | 332.6 / 1745.4 ms |
| Reflex Qwen3.5-4B stable | 0.90 diagnostic | 0 / 8 | 0 | 0 | 8 | 273.8 / 323.3 ms |

All runs forwarded the original bytes unchanged and produced content-free receipts.
No protected segment was proposed for removal. No actual token savings occurred.

## Interpretation

The result is **safe but not useful yet**:

- Winnow's strongest irrelevant probability on an eligible candidate was `0.8892`;
  the others were below the frozen `0.99` gate, so almost every scored case failed
  open as `uncertain_score`.
- Reflex's maximum irrelevant probability was `0.5774`, so every scored case
  bypassed at both thresholds.
- The fixture suite proves fail-open behavior and adapter integration; it does not
  establish token savings or downstream quality.
- The observed latency is compatible with local shadow analysis, but neither scorer
  currently creates a removal plan on this fixture set.

## Receipts

Local receipts:

```text
results/context_gate_winnow_shadow_v1.json
results/context_gate_winnow_shadow_t090_v1.json
results/context_gate_reflex_shadow_v1.json
results/context_gate_reflex_shadow_t090_v1.json
```

These receipts contain pointers, suggestions, probabilities, hashes, and aggregate
metadata; they do not contain raw fixture text. They are diagnostics only.

## Next required evidence

Before any active filtering claim:

1. Add more realistic eligible-history cases where the semantic answer is clearly
   irrelevant, not only A4's conservative synthetic tool transcripts.
2. Measure paired downstream task success and provider-reported token counts.
3. Add a cascade policy such as deterministic retain for protected/dependency
   segments, small-model screening, and Winnow fallback for uncertain candidates.
4. Stress concurrent requests, scorer failures, context eviction, and cache
   isolation through the actual gateway.
5. Keep active mode disabled until the Track A acceptance gates are met.
