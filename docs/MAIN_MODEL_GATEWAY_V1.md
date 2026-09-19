# Main-model gateway V1

Status: **forwarding gateway implemented and unit-tested against a loopback fake upstream.
Shadow mode is the default; active filtering is explicit opt-in and remains disabled by
default. No real provider was called. No production token-saving claim is made.**

This document specifies the missing integration piece for Track A: a local HTTP gateway
that runs the existing byte-preserving context gate *in front of* a main model, so the
same gate can sit before GPT, Claude, Kimi, OpenCode, or a local model server.

The gateway does **not** replace or fork the shadow core. It calls
[`shadow_request(...)`](../scripts/context_gate_v1.py) and reuses its parsing,
protection, dependency-closure, score-validation, and fail-open behavior verbatim. The
gateway only adds transport, a removal-plan applicator for the opt-in active path,
content-free receipts, and token accounting.

## 1. Files

| File | Role |
|---|---|
| `scripts/main_model_gateway_v1.py` | Loopback HTTP gateway, shadow/active modes, kill switch, reduction applicator, JSONL receipts |
| `scripts/scorer_adapters_v1.py` | Pluggable scorers: existing NanoJev `/api/evaluate` service, in-process callable, bounded-deadline and failure-recording wrappers, documented laya-style encoder shape |
| `scripts/test_main_model_gateway_v1.py` | unittest coverage with a loopback fake upstream started inside the test |
| `docs/MAIN_MODEL_GATEWAY_V1.md` | This specification |

Nothing else in the repository is modified by this milestone.

## 2. Architecture

```text
caller (GPT/Claude/Kimi/local client)
        |
        |  POST /v1/chat/completions   or   POST /v1/messages   (or /v1/responses)
        v
+---------------------------------------------------------------+
|  main_model_gateway_v1  (listens on a literal loopback addr)  |
|                                                               |
|  1. wire-format lookup from path                              |
|  2. kill switch?           ---> forward ORIGINAL bytes         |
|  3. unsupported?           ---> forward ORIGINAL bytes         |
|  4. sidecar from config + trusted X-NanoJev-Sidecar header     |
|  5. shadow_request(bytes, format, sidecar, scorer)  <---------+-- reuse core
|        |                                                       |
|        |  gate receipt (content-free) + unchanged bytes        |
|        v                                                       |
|  6. shadow: forward ORIGINAL bytes                             |
|     active: verify removal plan, build reduced bytes,          |
|             re-validate with parse_segments, else ORIGINAL     |
|                                                               |
|  7. forward to upstream origin (stdlib http.client,            |
|     no env proxy, no redirect following)                       |
|  8. write content-free JSONL receipt                           |
+---------------------------------------------------------------+
        |
        v
configured upstream base URL (real provider or local model)
```

The upstream is a bare origin (`https://api.openai.com`, `http://127.0.0.1:8080`). The
gateway listens only on a literal loopback address; `GatewayConfig` rejects non-loopback
listen hosts. Caller headers (including provider credentials) are forwarded opaquely and
are never read, stored, or logged; the gateway never requires or inspects an API key.

## 3. Endpoints and wire formats

| Path | Wire format | Notes |
|---|---|---|
| `POST /v1/chat/completions` | `openai_chat` | OpenAI-compatible |
| `POST /v1/messages` | `anthropic_messages` | Anthropic-compatible |
| `POST /v1/responses` | `openai_responses` | Already supported by the shadow core; included for completeness |
| anything else / non-POST | bypass | Forwarded unchanged, receipt reason `unsupported_wire_format` or `unsupported_method` |

Only methods and formats the core can parse safely are gated. Everything else is
forwarded byte-identically.

## 4. Modes and exact safety invariants

**Shadow mode (default).** The original request bytes are forwarded unchanged and the
upstream response is returned unmodified. The gate records a hypothetical removal plan
but applies nothing.

**Active mode (explicit opt-in).** Enabled only by `--mode active` or
`GatewayConfig(mode="active")`. It is never enabled by a request field, prompt text, or
sidecar. Active mode always fails open.

Invariants, enforced in order:

1. The gateway never parses or protects segments itself. All of that comes from
   `shadow_request`, including transitive dependency closure (before and after scoring)
   and cycle handling.
2. System/developer instructions, current user intent, tool schemas, cited evidence
   (`citations`/`annotations`), tool-call linkage, caller-pinned (`pinned`, `cited`,
   `safety`, `credential`, `dependency`, `exact_text`) segments, and unknown-schema
   requests are retained by the core. A request body can never grant itself eligibility.
3. The 0.99 provisional threshold is passed through unchanged. It is not tuned here and
   is still not a calibration claim.
4. Before applying any reduction, the gateway re-checks the gate's plan: **every** drop
   pointer must have role `assistant` and reason `high_irrelevance_score`. Any other
   drop reason (for example `protected_structure`, `caller_protected`,
   `required_dependency`, `whole_request_fallback`) means the removal set contains a
   protected segment, so the whole request is forwarded unchanged with reason
   `protected_segment_in_removal_set`.
5. The reduced body is built only for pointers matching
   `/{messages|input}/{i}/content` or `/{messages|input}/{i}/content/{j}/text`, must still
   contain the user message, and is re-validated with the core's `parse_segments` before
   it is forwarded. Any failure forwards the ORIGINAL bytes.
6. Shadow mode never builds or sends reduced bytes under any circumstance.
7. The kill switch is checked before scoring and forwards everything unchanged.

### Fail-open reason codes

Any of these forwards the ORIGINAL bytes untouched; `forwarded_unchanged` is `true`:

| `forward_reason` | Cause |
|---|---|
| `gate_error` | `shadow_request` itself raised |
| `scorer_error` | Scorer raised; `scorer_failure_kind` is `exception` or `timeout` |
| `invalid_score_response` | Malformed, partial, duplicate, non-unit, or nonfinite scores (inside `gate_receipt.reason`) |
| `uncertain_score` | Any candidate inside the uncertain interval |
| `protected_segment_in_removal_set` | Plan contained a protected/unknown drop |
| `reduction_error` | Reduced bytes could not be built or failed core re-validation |
| `unsupported_wire_format` / `unsupported_method` | Not a gated endpoint |
| `kill_switch` | Global kill switch engaged; no scoring performed |
| `active_no_reduction` | Gate scored but proposed nothing to drop |
| `shadow_mode` | Default mode |

Scorer failures never copy exception text into the receipt; only the fixed
`scorer_failure_kind` code is recorded, because exception messages can contain prompt text.

## 5. Deterministic global kill switch

* CLI: `--kill-switch`
* Environment: `NANOJEV_GATEWAY_KILL_SWITCH=1` (also `true`/`yes`/`on`)

When engaged, the gateway does not score at all and forwards every request unchanged, in
both modes. It is deterministic: no request, header, or sidecar can disable it.

## 6. Scorer adapters

`build_scorer(kind, ...)` selects a scorer:

* `none` — no scorer. The core reports `scorer_unavailable` and retains everything.
* `http` — `NanoJevHTTPScorer`, reusing `context_gate_local.LoopbackPredictor`
  (`POST /api/evaluate`, literal loopback only, no proxy, no redirects, bounded timeout).
* `inprocess` — `InProcessScorer` around a caller-supplied callable.
* `laya` — `LayaEncoderScorerAdapter`, a **documented, configurable shape** for a
  laya-style local encoder decision service. `laya` is never installed, downloaded, or
  imported; the adapter only fixes the loopback HTTP contract and fails with
  `ScorerError` until an operator explicitly configures a loopback origin.

`DeadlineScorer` bounds any scorer with a wall-clock deadline and raises `ScorerTimeout`.
Python threads cannot be force-killed, so the deadline stops the *gateway* from waiting;
it does not cancel the underlying call. `FailureRecordingScorer` records whether the last
failure was a timeout or an exception without retaining its text.

Trusted per-request metadata is supplied by integration, never parsed from prompt text:

* `X-NanoJev-Sidecar: base64(JSON)` — caller sidecar (`segments`, `bypass`). Malformed
  input becomes `{"bypass": true}` and can never grant eligibility. The header is stripped
  before forwarding.
* `X-NanoJev-Baseline-Provider-Prompt-Tokens: <int>` — caller-retained provider-reported
  prompt-token count for the same request in its unfiltered form. Used only for paired
  token accounting and stripped before forwarding.

All `X-NanoJev-*` headers and hop-by-hop headers are stripped before the upstream request.

## 7. Receipts

One JSONL line per request at `--receipt-log`. Receipts contain: schema version, event
id, mode, kill-switch flag, wire format, method, path, request/forwarded SHA-256, the
nested content-free gate receipt (fixed JSON pointers, segment hashes, retain/drop
suggestions, fixed reason codes, probabilities), proposed and applied pointers, upstream
status, gate latency, total latency, and token accounting.

Receipts never contain raw prompt text, tool output, response bodies, headers, or
credentials. Paths are recorded without query strings.

## 8. Measured-versus-estimated token accounting

The rule is deliberately conservative:

* **Shadow mode reports only an ESTIMATE.** Nothing was removed, so the receipt reports
  `savings.claim = "estimate"`, `savings.basis = "shadow_estimate_only"`, and the isolated
  removed-segment text token count from the core, labelled with the tokenizer identity
  (`estimate.tokenizer_id` plus its SHA-256). The default CLI tokenizer is a whitespace
  word split explicitly named `whitespace-word-split-v1-not-a-provider-tokenizer`; it is
  not a BPE tokenizer and not provider billing.
* **Active mode reports ACTUAL savings only from provider-reported usage.** When a
  genuinely reduced request was sent, the gateway reads the upstream response's
  `usage.prompt_tokens` (Anthropic/Responses: `usage.input_tokens`). That number is always
  recorded under `provider_reported`.
  * With a caller-supplied provider baseline, `savings.claim = "actual"` and
    `savings.tokens = baseline - provider_reported.prompt_tokens`; both ends are
    provider-reported. Basis `provider_paired_baseline`.
  * Without a paired baseline, the gateway refuses to call it actual:
    `savings.claim = "estimate"`, basis
    `provider_usage_observed_no_paired_baseline` (or `provider_usage_unavailable`).
* **Active mode with no reduction** reports `savings.claim = "none"`, basis
  `no_reduction_sent`.

Character counts, whitespace-word estimates, and local tokenizer counts are never
reported as provider tokens or money savings.

## 9. Running against a real provider (documented, NOT executed here)

Start the existing local NanoJev service, then the gateway. Shadow mode first:

```bash
.venv/bin/python scripts/main_model_gateway_v1.py \
  --upstream https://api.openai.com \
  --listen-host 127.0.0.1 --listen-port 8790 \
  --mode shadow \
  --scorer http --scorer-url http://127.0.0.1:8765 \
  --receipt-log /tmp/nanojev_gateway_receipts.jsonl
```

Point the client at the gateway instead of the provider (the client keeps sending its own
credential headers, which the gateway forwards without reading):

```bash
curl -s http://127.0.0.1:8790/v1/chat/completions \
  -H "Authorization: Bearer $PROVIDER_KEY" \
  -H "Content-Type: application/json" \
  -d @request.json
```

Active mode is the same command with `--mode active`, and only after the Track A
acceptance gates pass. The kill switch is `--kill-switch` or
`NANOJEV_GATEWAY_KILL_SWITCH=1`. This section is documentation only; it was not executed,
no provider was contacted, and no API key was used.

## 10. Reproduction

```bash
.venv/bin/python -m unittest discover -s scripts -p 'test_main_model_gateway*.py'
```

The suite starts a loopback fake upstream and a loopback fake NanoJev `/api/evaluate`
service inside the tests. It contacts no real provider, uses no API key, and requires no
external network. It covers: shadow byte-identity, active reduction with provider-reported
savings, protected-segment fail-open, scorer exception/timeout/absent fail-open,
malformed/partial/nonfinite/duplicate/non-unit/uncertain score fail-open, defensive
refusal of a protected drop, kill switch, unsupported format/method bypass, receipt
privacy, upstream error pass-through, upstream-unreachable 502, config validation, and the
scorer adapter shapes (including that `laya` is never imported).

## 11. What is NOT established

* **No real provider was called.** All evidence is loopback fake-upstream unit tests. No
  GPT, Claude, Kimi, or local main-model endpoint was contacted.
* **No paired downstream-quality comparison exists.** Task success, safety, citation
  fidelity, tool selection, code correctness, and instruction following were not measured
  with and without filtering. Track A's A3 cohorts and the ≤0.5 percentage-point
  regression gate are not satisfied.
* **No production token-saving claim.** There is no measured ≥30% median main-model
  input-token reduction on eligible frozen workloads. The only token numbers here are
  labelled local estimates and, in active mode, provider-reported counts for a reduced
  request; a savings delta additionally requires a caller-supplied provider baseline.
* **Active filtering remains disabled by default** and should stay that way until the
  Track A acceptance gates pass.
* **The 0.99 threshold is unchanged and uncalibrated.** It was not tuned, and earlier
  evidence shows required and irrelevant cases score similarly under it, with zero actual
  token savings in the real-checkpoint shadow run.
* **Scoring coverage is narrow.** Only assistant text explicitly marked eligible by a
  trusted sidecar can be dropped. User messages, tool results, files, retrieval, images,
  and server-side conversation references are not candidate removal sources in V1.
* **No service identity/attestation.** The gateway trusts whatever responds at the
  configured loopback scorer URL; there is no authenticated model-identity check.
* **Deadline is not cancellation.** An in-process scorer that exceeds `DeadlineScorer`
  keeps running in its thread after the gateway fails open.
* **No streaming, caching, retry, or cost accounting.** Streamed responses are forwarded
  as opaque bytes; per-request cost, cached input, tool re-execution, and retry costs are
  not measured.
* **No independent review.** These files have not been reviewed by the project's
  independent reviewer, and no `results/` receipt was produced.
