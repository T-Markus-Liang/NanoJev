# Context filter threshold policy V1

Status: **diagnostic policy only; no production threshold selected**.

Policy file:

- `research/context_filter_threshold_policy_v1.json`

The policy separates evidence collection from operating-point selection. The scorer
`p_irrelevant` value is not calibrated accuracy, so a threshold is an operating point
and must be declared before review.

## Declared values

| Threshold | Role | Production candidate |
|---:|---|---:|
| `0.99` | Frozen shadow control | No |
| `0.95` | Diagnostic sweep point | No |
| `0.90` | Diagnostic sweep point | No |
| `0.85` | Diagnostic sweep point | No |

Current evidence:

- `0.99` is the frozen conservative control and currently produces zero removals on
  the value fixtures.
- `0.95` removes 5,212 bytes / 1,337 local-BPE tokens (35.7% / 35.9%) on V2.
- `0.90` removes 5,560 bytes / 1,431 local-BPE tokens (38.0% / 38.4%) on V2.
- `0.85` produces the same V2 reduction as `0.90`; no additional unsafe removals are
  observed on this synthetic suite.
- All three lower sweep points remain diagnostic only.
- Cascade fast-path diagnostic values are declared separately as `0.95`, `0.90`, and
  `0.80`; they control whether the fast scorer's answer avoids the strong scorer, not
  the final removal threshold.

## Review mode

`benchmark_filter_value_v1.py` accepts:

```bash
--threshold-policy research/context_filter_threshold_policy_v1.json --review-mode
```

In review mode the runner:

- requires a threshold-policy file;
- records the policy path and SHA-256 in the result;
- fails closed on undeclared scorer thresholds;
- fails closed on undeclared cascade fast-path thresholds.

Example:

```bash
python3 scripts/benchmark_filter_value_v1.py \
  --manifest research/context_filter_value_fixture_manifest_v2.json \
  --threshold-policy research/context_filter_threshold_policy_v1.json \
  --review-mode \
  --threshold 0.90 \
  --arms control safe_dedup systemone \
  --output results/example.json
```

## Veto conditions

A diagnostic sweep can be recorded even when it produces unsafe removals, but no
operating point may be proposed unless all of these are true:

- protected drops are zero;
- required-evidence failures are zero;
- paired answer regressions are zero;
- every attempted reduction restores byte-identically;
- unsupported or unresolved requests bypass before scoring;
- the fixture manifest and threshold policy are hash-pinned;
- provider calls remain absent from the local paired harness.

## Production selection rule

No production trial threshold is currently selected. If one is proposed later, it
must be selected on a declared development cohort before held-out review. Evaluation
fixtures remain evaluation-only; their outputs must not be used for training or
calibration fitting.

## Non-decisions

- `0.90` is not a production threshold.
- This policy does not authorize active filtering.
- This policy does not claim provider billing savings or real model-quality
  preservation.
