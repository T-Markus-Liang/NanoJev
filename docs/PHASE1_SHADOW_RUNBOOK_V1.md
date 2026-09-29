# Phase1 shadow measurement runbook V1

Status: **authorization template ready; Phase1 not authorized**.

Phase1 is a measurement-only shadow phase. It may score requests and record
provider/usage metadata, but it must forward the original request bytes. It must
never send reduced bytes.

## Artifact

- Authorization template: `research/phase1_shadow_authorization_v1.json`
- Validator: `scripts/validate_phase1_shadow_authorization_v1.py`
- Current preflight: `results/phase1_shadow_authorization_preflight_v1.json`

Current status:

```text
status=template_ready_not_authorized
phase1_authorized=false
provider_calls_allowed=false
active_filtering_allowed=false
SHA-256=5cdb5f15871bf4a4390d1e5ce7b19f69575eb3dcbab65a6db53c2707a532c2e3
```

## Milestone definition

This node is **Phase1 shadow measurement readiness**, not Phase1 execution.

The readiness packet must satisfy:

| Metric | Required value |
|---|---:|
| Phase0 loopback cases | 17/17 pass |
| Phase0 status | `phase0_pass` |
| Phase1 shadow dry-run cases | 8/8 pass |
| Provider calls during preparation | 0 |
| Unsafe actions | 0 |
| Paired downstream regressions in selected evidence | 0 |
| Content-free receipts | required |
| Kill switch | required and verified |
| Restore round-trip | required and verified |
| Unsupported-path bypass | required and verified |
| Scorer failures | fail open |
| Active filtering | disabled |
| Reduced bytes | never sent |

## Required owner fields before any Phase1 run

```text
authorization.granted=true
authorization.granted_by=<owner identity>
authorization.granted_at=<timestamp>
authorization.scope=<allowed request/source scope>
authorization.expires_at=<expiry timestamp>
authorization.upstream=<approved upstream>
```

## Operator checklist

| Step | Check |
|---|---|
| 1 | Name upstream, request source scope, request budget, expiry |
| 2 | Confirm measurement-only shadow mode |
| 3 | Confirm actual savings requires paired provider-reported usage |
| 4 | Confirm no raw prompt/response/credential/tool-output logging |
| 5 | Confirm stop conditions and rollback path |

## Stop immediately if

- any request would send reduced bytes;
- a protected or required-evidence pointer is proposed for removal;
- restore round-trip fails;
- actual savings is claimed without paired provider usage;
- a receipt contains raw text;
- owner revokes scope or requests stop.

## Current decision

```text
PHASE1 SHADOW: NOT AUTHORIZED
PROVIDER CALLS: DISABLED
ACTIVE FILTERING: DISABLED
```
