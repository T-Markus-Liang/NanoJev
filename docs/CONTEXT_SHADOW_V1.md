# Context gate: shadow integration V1

Status: **byte-preserving integration implemented; learned token reduction not achieved**.

The [shadow core](../scripts/context_gate_v1.py) accepts an original request body as bytes, a wire-format name, trusted integration metadata, and a scorer. It returns the identical original bytes and a content-free analysis receipt. There is no active deletion code path, provider proxy, application configuration change, or automatic capture of real conversations.

## Supported boundaries

| Wire format | Analyzed content | Protected / bypassed content |
|---|---|---|
| `openai_chat` | Text messages and text blocks | System/developer/user/tool content, schemas, completed tool-call structures protected |
| `openai_responses` | String input or message items with text blocks | Instructions protected; function-call/opaque items and server-side conversation references bypassed |
| `anthropic_messages` | Text messages and text blocks | System/user content and completed tool-use/result structures protected |

Images, files, encrypted reasoning, unresolved tool linkage, unsupported fields, invalid JSON, unknown dependency pointers, and oversized requests bypass analysis. Bypassing returns the exact original bytes; it does not truncate or rewrite them. The supported subset is deliberately explicit, not a claim of compatibility with every provider extension.

Only assistant text explicitly marked eligible by the integration can be scored in V1. All user messages are protected, not just the latest message. Broader file/retrieval/tool-result selection still needs provenance-aware adapters and validation. The user request body cannot grant itself eligibility.

## Trusted sidecar contract

Sidecar metadata is supplied separately by the integration, never parsed from prompt text. Example for an eligible historical assistant message:

```json
{
  "segments": {
    "/messages/1/content": {"eligible": true},
    "/messages/2/content": {"depends_on": ["/messages/1/content"]}
  }
}
```

In this example, if message 2 is a protected user request, message 1 is retained as a dependency even if eligible. `pinned`, `cited`, `safety`, `credential`, `dependency`, and `exact_text` are trusted protection flags and override eligibility. A top-level `bypass: true` preserves exact-text/legal or otherwise unsupported workflows without scoring. Integrations are responsible for supplying these flags and dependencies; the model is not an authority for discovering which constraints may be discarded.

The scorer receives all parsed context, not a silently truncated approximation. One Boolean question per candidate asks whether it is certainly irrelevant. A fixed provisional threshold of 0.99 is **not** a calibration claim. Any intermediate-confidence candidate makes the entire hypothetical removal plan retain all segments. Required dependencies are closed transitively before and after scoring, including cycles. Missing/malformed/nonfinite/partial results and scorer exceptions also retain everything.

## Local invocation and logs

[context_gate_local.py](../scripts/context_gate_local.py) provides a logged scorer and explicit CLI. It does not start a service or change a provider. The selected NanoJev service must already be running at the caller's chosen address. Only literal loopback HTTP origins are accepted; environment proxies and redirects are not used. A socket timeout is bounded to 30 seconds; timeout does not cancel work already running on the server, and an arbitrary in-process scorer has no automatic deadline.

```bash
.venv/bin/python scripts/context_gate_local.py --input request.json --format openai_chat --sidecar eligibility.json --url http://127.0.0.1:8765 --log data/context_usage.jsonl --receipt-log data/context_receipts.jsonl
```

The CLI analyzes a caller-supplied local file and writes receipts only. It is not an HTTP forwarding proxy. Embedders call `shadow_request(...)` and use its unchanged byte result for their normal provider request. No SDK or live provider integration has been installed globally by this milestone.

Successful inference uses the existing `nanojev-usage-v1` helper for counts, probabilities, timing, and checkpoint configuration identity. The shadow receipt links to that decision event for later feedback. Raw-debug payload logging is forcibly omitted for this integration even if the debug environment setting is enabled. Scorer failures remain visible in the shadow receipt without copying exception messages containing possible private text.

Receipts contain fixed JSON pointers, segment/request hashes, fixed reason codes, probability values, and policy identity, not raw content. Hashes enable verification against an original retained by the caller; they cannot reconstruct it alone and are not encryption or a guarantee against dictionary attacks. Use local access controls for logs. Configured checkpoint identity is not server weight attestation. Exact token billing is unavailable: optional token counts describe isolated segment text using a declared tokenizer, not cached/prompt/API billing or savings.

## Measured evidence

The [benchmark receipt](../results/context_shadow_v1_seed17.json) contains code/checkpoint/log hashes and runtime identity.

- 3,000 deterministic stress cases across three wire formats passed unchanged-byte, protected-segment, privacy, and fallback assertions. These are stubbed guardrail checks, not model-quality scores.
- 24 real-model requests covering eight underlying relevant/irrelevant scenarios were sent through a temporary loopback HTTP server using the actual service handler and MPS checkpoint. All 24 returned HTTP 200 and linked usage events.
- All 24 scores fell into the uncertain interval and produced retain-all plans. Proposed deletions: zero. Actual token savings: **zero**.
- Gate round-trip p50 was 32.46 ms and p95 37.67 ms in this local sample, excluding 7.40 seconds of model loading. This is not isolated model compute or financial-decision latency; 24 samples are insufficient for a tail-latency release claim.
- Required and irrelevant cases received similarly high raw irrelevance probabilities, approximately 0.706 to 0.823. Lowering the threshold to force savings is not an acceptable fix; task-specific learning and calibration are required.

## Reproduction and next gates

Use fresh log paths because the benchmark refuses to append a second run into an existing audit file:

```bash
.venv/bin/python scripts/benchmark_context_shadow_v1.py --checkpoint checkpoints/local_atomic_seed17/variants/local_atomic_seed17 --output results/context_shadow_v1_seed17.json --usage-log data/context_shadow_v1_http_usage.jsonl --receipt-log data/context_shadow_v1_http_receipts.jsonl
.venv/bin/python -m unittest discover -s scripts -p 'test_context_gate*.py'
```

The benchmark shuts down its temporary HTTP server. No persistent service or proxy changes remain.

Follow-up experiment: the first provenance-aware relevance curriculum has now been trained under a frozen three-seed protocol and audited in [Context relevance V1](CONTEXT_RELEVANCE_V1.md). The synthetic latest-record dependency task is learnable (initialization ≈46% → 92–97% accuracy), but the outcome is mixed and **no candidate is promoted**: seed 17 regresses local-maze accuracy beyond budget, seed 18 false-drops required segments at the frozen 0.99 threshold, and seed 19's zero false drops on 30/45 source groups is not a safety certificate. The 0.99 threshold remains a provisional diagnostic, not a calibrated claim. Active deletion and main-model token savings remain zero.

Remaining requirements include independently reviewed challenge families, calibration-only threshold fitting on calibration data, candidate-selection coverage beyond assistant text, authenticated/model-identity-checked operational service integration, paired main-model answers across three model families, exact provider-token accounting, and all roadmap quality/efficiency release gates. None are replaced by the safe retain-all result or by the relevance training outcome.

Repository validation at this milestone reports 152 unittest cases with two optional dataset skips, plus three existing local-skill tests. As of the Context relevance V1 documentation pass (2026-09-19), the suite has grown to 165 cases (163 executed, 2 optional skips) plus 3 local-skill tests; the newer tests cover the relevance data builder, report integrity, and training-runtime additions. The external read-only worker timed out without edits; it is not counted as successful independent review.
