# Active-mode phase0 loopback dry-run V1

Status: **phase0 passed; provider calls remain disabled**.

This run exercises the gateway's active-mode path only against a loopback fake
upstream. No provider is contacted and no real request is filtered.

## Runner

- `scripts/run_active_mode_phase0_v1.py`
- `scripts/test_active_mode_phase0_v1.py`
- Receipt: `results/active_mode_phase0_loopback_v1.json`

Current receipt:

```text
status=phase0_pass
cases=17
provider_calls=0
SHA-256=da797daf1671748d498a93202ab27c508ae17b2684f9b95c462147825cde3b38
```

## Cases

| Case | Expected result | Observed |
|---|---|---|
| `active_eligible_drop` | eligible assistant text removed, evidence/user retained, restore manifest emitted | pass |
| `provider_accounting_baseline` | reduced request forwarded; paired baseline header produces `savings.claim=actual` | pass |
| `anthropic_eligible_drop` | Anthropic text-part reduction and restore manifest | pass |
| `responses_eligible_drop` | OpenAI Responses input reduction and restore manifest | pass |
| `unsupported_embeddings` | unsupported wire path forwarded unchanged | pass |
| `active_no_reduction` | active mode forwards original when no plan exists | pass |
| `shadow_control` | original bytes forwarded, no restore manifest | pass |
| `kill_switch` | original bytes forwarded, no scoring/reduction | pass |
| `scorer_error` | original bytes forwarded, `gate_reason=scorer_error` | pass |
| `malformed_sidecar` | original bytes forwarded, `gate_reason=caller_bypass` | pass |
| `dependency_closure` | retained dependent protects its required dependency | pass |
| `uncertain_score` | uncertain scorer output fails open | pass |
| `invalid_score_response` | malformed scorer output fails open | pass |
| `scorer_timeout` | scorer deadline fails open with `scorer_failure_kind=timeout` | pass |
| `restore_manifest_too_large` | oversized restore header fails open | pass |
| `oversized_request_body` | request rejected before upstream with 413 | pass |
| `internal_header_stripping` | internal `X-NanoJev-*` headers do not reach upstream | pass |

## Verified properties

1. Active mode applied only the declared eligible pointer in each supported wire
   format.
2. The fake upstream received reduced JSON containing protected evidence and user
   intent.
3. `X-NanoJev-Restore-Manifest` was emitted only for applied reductions.
4. The manifest reconstructed the original request hash and matched the reduced
   request hash.
5. The restore header contained no raw system/evidence/noise/user text.
6. Shadow mode forwarded the original bytes and emitted no restore header.
7. Kill switch forwarded the original bytes without scoring.
8. Scorer exception, timeout, uncertain score, malformed score response,
   malformed sidecar, oversized restore header, and oversized request all failed
   open or were rejected before upstream.
9. Dependency closure prevented a proposed dependency drop when its dependent
   segment was retained.
10. Internal `X-NanoJev-*` headers were stripped before upstream forwarding.

## Boundary

Phase0 proves transport/applicator/reversibility plumbing only. It does not prove
real model quality, provider billing savings, or production safety. Phase1 and
Phase2 still require explicit owner operational authorization.
