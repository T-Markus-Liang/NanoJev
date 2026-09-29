# Phase1 shadow-measurement readiness gate V1

Status: **ready for owner Phase1 authorization; Phase1 itself is not authorized**.

This is the current major milestone. It means the local evidence and control
plumbing are ready to request a scoped owner authorization for Phase1 shadow
measurement. It does not mean provider calls or active filtering are enabled.

## Gate receipt

```text
results/phase1_readiness_gate_v1.json
status=ready_for_owner_phase1_authorization
phase1_authorized=false
provider_calls_allowed=false
active_filtering_allowed=false
SHA-256=3b92bd884d1ad9c5790e229df625c36ece64da0e3c1267f5299407b0212e4ad8
```

## Metric requirements

| Metric | Required | Observed |
|---|---:|---:|
| Track A review packet | owner-authorized internal pass | pass |
| Phase0 cases | 17/17 pass | 17/17 |
| Phase1 shadow dry-run cases | 8/8 pass | 8/8 |
| Provider calls during preparation | 0 | 0 |
| Unsafe actions | 0 | 0 |
| Paired regressions in selected evidence | 0 | 0 |
| Kill switch verified | yes | yes |
| Restore round-trip verified | yes | yes |
| Unsupported-path bypass verified | yes | yes |
| Scorer fail-open verified | yes | yes |
| Content-free receipts | yes | yes |
| Phase1 authorization | false | false |
| Active filtering authorization | false | false |

## Evidence artifacts

- Track A decision: `results/track_a_review_decision_v1.json`
- Canary preflight: `results/active_mode_canary_preflight_v1.json`
- Phase0 loopback receipt: `results/active_mode_phase0_loopback_v1.json`
- Phase1 authorization template check:
  `results/phase1_shadow_authorization_preflight_v1.json`
- Phase1 dry-run receipt: `results/phase1_shadow_dry_run_v1.json`
- Real-context holdout preflight:
  `results/real_context_holdout_preflight_v1.json`
- Collection authorization check:
  `results/real_context_collection_authorization_preflight_v1.json`

## What this node means

Ready to ask owner for a bounded Phase1 scope containing:

```text
upstream
request/source scope
request budget
expiry
```

## What it does not mean

- No provider calls are authorized.
- No active filtering is authorized.
- No production threshold is selected.
- No provider billing savings are claimed.
- No release candidate exists.
- No real-context collection has started.
