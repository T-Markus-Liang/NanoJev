# Active-mode canary protocol V1

Status: **protocol valid, not authorized**.

This is the next gate after T22. It defines how a future isolated active-mode
canary could be run, but it does **not** enable active filtering and does **not**
authorize provider calls.

## Inputs

- Protocol: `research/active_mode_canary_protocol_v1.json`
- Validator: `scripts/validate_active_mode_canary_protocol_v1.py`
- Preflight receipt: `results/active_mode_canary_preflight_v1.json`
- Review decision: `results/track_a_review_decision_v1.json`
- Review replay: `results/track_a_review_replay_v1.json`

Current preflight:

```text
status=protocol_valid_not_authorized
active_filtering_allowed=false
provider_calls_allowed=false
SHA-256=84e421c3f741e3e7b9f9e73687598c46ab604ef19bfbe93061fc2aa6eddd81d4
```

## Phase ladder

| Phase | Mode | Upstream | Provider calls | Owner authorization | Purpose |
|---|---|---|---:|---:|---|
| phase0_loopback_fake_upstream | active | loopback fake only | no | no | test applicator, restore header, receipts, supported wire formats, baseline accounting, unsupported path, kill switch, fail-open |
| phase1_shadow_provider_measure | shadow | provider/local main model | no | yes | measure original-request gate plan and provider usage |
| phase2_paired_canary | active | provider/local main model | no | yes | paired original/reduced accounting and quality check |

Only phase0 may run without a new operational authorization because it uses a
loopback fake upstream and sends no provider traffic. Its required case list is
now the 17-case multi-wire/stress coverage set: OpenAI chat, paired baseline
accounting, Anthropic messages, OpenAI Responses, unsupported path, active
no-reduction, shadow control, kill switch, scorer error, malformed sidecar,
dependency closure, uncertain score, invalid score response, scorer timeout,
oversized restore manifest, oversized request body, and internal header
stripping.

## Accounting rule

Actual savings can be claimed only when both sides are provider-reported:

```text
baseline provider prompt tokens - reduced provider prompt tokens
```

The trusted baseline header is:

```text
X-NanoJev-Baseline-Provider-Prompt-Tokens
```

Local tokenizer estimates remain estimates and are never billing claims.

## Safety requirements

- global deterministic kill switch: `NANOJEV_GATEWAY_KILL_SWITCH`;
- shadow fallback;
- restore-manifest header budget: `4096` bytes;
- protected segment veto;
- dependency closure;
- round-trip verification;
- content-free receipts;
- no raw prompt/response/credential/tool-output logging;
- max `12` requests per phase in V1;
- max request body `65,536` bytes.

## Stop conditions

The canary must stop and fail open on:

- any protected segment proposed for removal;
- any required evidence missing from reduced request;
- any paired downstream answer regression;
- any restore-manifest reconstruction failure;
- any scorer timeout/malformed output/exception in phase0;
- any body over budget;
- missing provider prompt usage when actual savings is claimed;
- any raw text/credential/tool-output field in a receipt;
- owner revocation or stop request.

## Remaining prerequisites before provider traffic

1. Explicit owner operational authorization naming the exact phase and scope.
2. A real-context holdout under `REAL_CONTEXT_HOLDOUT_PROTOCOL_V1.md`, or an
   explicitly approved substitute.
3. Provider usage fields verified for the chosen upstream wire format.
4. A bounded request list with provenance and expected outcomes.
5. A rehearsed rollback/kill-switch procedure.

## Phase0 result

`results/active_mode_phase0_loopback_v1.json` reports `phase0_pass` with five
loopback cases and `provider_calls=0`. See `docs/ACTIVE_MODE_PHASE0_V1.md`.

## Current decision

```text
PROTOCOL VALIDATED
PHASE0 LOOPBACK: PASSED
ACTIVE FILTERING: DISABLED OUTSIDE LOOPBACK TESTS
PROVIDER CANARY: NOT AUTHORIZED
```
