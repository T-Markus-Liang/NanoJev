# A5 filter-versus-rebuild protocol V1

Status: **protocol, preflight, and synthetic controls only — READY_FOR_REVIEW.** No arm has
run, no model or provider was called, no context was filtered, and nothing here is measurement
evidence or authorization. The protocol is frozen before measurement.

## Purpose

The community compression debate (filtering versus reconstruction) is unresolved. A5 makes it a
frozen, paired experiment instead of an argument. The frozen protocol lives at
[`research/nanojev_v2_t10_a5_filter_vs_rebuild_protocol_v1.json`](../research/nanojev_v2_t10_a5_filter_vs_rebuild_protocol_v1.json),
the preflight/control package at
[`scripts/filter_vs_rebuild_v1.py`](../scripts/filter_vs_rebuild_v1.py), and its tests at
[`scripts/test_filter_vs_rebuild_v1.py`](../scripts/test_filter_vs_rebuild_v1.py).

## Frozen arms

| # | Arm | Transformation | Lossy |
|--:|---|---|---|
| 0 | `unfiltered_control` | identity baseline | no |
| 1 | `safe_dedup` | deterministic exact-duplicate removal | no |
| 2 | `relevance_filter` | A2/A4 verbatim-retain relevance removal | no |
| 3 | `abstractive_summary` | abstractive reconstruction | yes |
| 4 | `retrieval_rebuild` | retrieval reconstruction over stored evidence | yes |

All arms fail open: every arm's failure action is *forward the original bytes*. A summary
fallback is a separate policy and is never substituted silently.

## Source binding and fixture families

The protocol binds the frozen A4 manifest read-only by exact SHA-256
(`research/tool_history_fixture_manifest_v1.json`, digest
`0c8253c9424245315d3aaf1501a6c683b8cda39bc494f8c32005ab219adf89e0`). Fixture content is
never copied into the protocol or receipts; only the digest and structural facts are bound.

The frozen fixture-kind-to-task-family map is:

| Task family | A4 fixture kinds | Protected |
|---|---|:--:|
| `tool_lookup` | `success` | no |
| `parallel_tool_use` | `parallel_calls` | no |
| `error_evidence` | `error` | yes |
| `ambiguous_linkage` | `duplicate_ids`, `orphan_result`, `pending_call` | yes |
| `side_effect_safety` | `non_repeatable_result` | yes |
| `snapshot_consistency` | `mutable_file_read` | yes |
| `correction_dependency` | `stale_output`, `user_correction` | yes |
| `credential_handling` | `secrets_credentials` | yes |

A family is protected exactly when one of its A4 kinds carries a hard blocking flag
(`non_repeatable`, `credential_bearing`, `contains_error`, `stale_evidence`,
`correction_dependent`, `volatile_source`) or has unresolved pairing. The validator re-derives
this set from the A4 manifest and rejects any protocol that disagrees.

## Metrics and cost contract

| Metric | Unit |
|---|---|
| `dependency_pair_retention` | ratio |
| `evidence_fidelity` | ratio |
| `protected_segment_deletion` | count (must be zero) |
| `downstream_task_success` | ratio |
| `restore_cost` | provider tokens |
| `end_to_end_cost` | provider tokens |

Cost is provider-reported or tokenizer-based only. Character estimates and manual character
counts are forbidden, and tokenizer counts require a tokenizer id. `end_to_end_cost` includes
scorer, rebuild, retry, and tool re-execution cost. Acceptance targets are the Track A gates:
no more than a 0.5 percentage-point task-success regression, at least 30% median token
reduction, and zero protected-segment deletion, reported per task family with paired
confidence intervals.

## Fail-open and fail-closed rules

* Shadow-only, synthetic-only, `applied` is false: no active pruning exists in this package.
* Any protected-family loss — a protected-segment deletion or a success regression beyond the
  threshold — blocks an aggregate win outright, even if the aggregate shows a large reduction.
* The preflight fails closed on schema, SHA-256, enum (arm/metric/family), authorization,
  network, raw-content, and path mismatches. Missing measurement evidence also blocks.

## Synthetic controls and CLI

The package emits deterministic placeholder receipts only: all provider, quality, physical, and
cost fields are `null`, `applied` is `false`, `network_model_calls` is `0`, `model_loaded` and
`measurement_evidence` are `false`, and every authorization flag is `false`. Each receipt binds the
protocol byte hash and carries its own content-free receipt hash. Receipts contain no raw fixture
content.

This is a repository-level control package, not a standalone wheel: it reuses the existing A4 builder
and shadow core (`scripts/build_tool_history_fixtures_v1.py`, `scripts/context_gate_v1.py`,
`scripts/context_restore_v1.py`, and their `predict_toy_decisions.py` dependency) read-only. A future
offline release must include these dependencies in its manifest/SBOM before claiming installability.

```
python scripts/filter_vs_rebuild_v1.py --self-test          # validate binding, write nothing
python scripts/filter_vs_rebuild_v1.py --stdout             # print the synthetic bundle
python scripts/filter_vs_rebuild_v1.py --output bundle.json # deterministic exclusive create
```

`--output` writes canonical JSON and refuses to overwrite an existing file (exit code 2).
`--self-test` validates the protocol and A4 binding without writing.

## Boundaries

No provider or model call, no network access, no training, quantization, deployment, or active
pruning is performed. Nothing is adopted from this package; adoption still requires the Track A
acceptance gates and paired downstream-quality evidence. The preflight status is
`protocol_valid_not_authorized`.
