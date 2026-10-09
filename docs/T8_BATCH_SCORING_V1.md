# T8 — bounded batch scoring and original-order merge

2026-09-20. **Implementation and verification complete; READY_FOR_REVIEW.**
No production activation, model promotion, training, threshold change or provider change.
The full roadmap goal remains active. T8g's data review blockers remain independent.

## Outcome

`MAX_SCORED=32` now limits **one scorer call**, not the whole request. A request with
65 eligible candidates produces batches of 32/32/1 rather than immediately returning
`scoring_budget_exceeded`. Every candidate still sees the **entire original conversation**.
The gate returns the original byte object; only a hypothetical plan is merged.

**The model did not become a better relevance judge.** Five real-local cases, 26 HTTP
inferences and 392 candidate scores all produced retain-all plans at the unchanged 0.99
threshold. Proposed real-model drops and actual saved tokens are both **zero**.

## Contract and safety

- Preserve original segment order and map response IDs only inside their batch. The
  request parser, trusted eligibility, protected roles, threshold and scorer renderer
  are unchanged. No evidence truncation, labels or previous batch answers enter a prompt.
- A thrown, malformed, missing, duplicate, nonfinite or otherwise invalid batch result
  retains **all candidates in that batch**. Never accept its valid-looking subset.
  Other batches still run. `partial_batch_fallback` means some batches were usable;
  `all_batches_failed` means none were.
- Apply dependency closure **globally after merging**. Retained/failed candidates can
  protect evidence in earlier or later batches, transitively and through cycles.
- Any uncertain valid score still retains the **whole request**. Conflicting checkpoint
  metadata fingerprints across valid batches also retain all (`scorer_identity_mismatch`).
  Metadata consistency is not cryptographic attestation of serving weights.
- Receipt schema remains `nanojev-context-shadow-v1`; additive `batching_policy` is
  `nanojev-context-batch-v1`. `batches` and `scorer_event_ids` preserve per-call audit
  identity. Legacy single-batch `scorer_event_id` remains. No raw errors/text are logged.
- The gateway's failure recorder survives later successful batches and is recreated per
  request, so an early timeout is neither hidden nor carried into another request.

### Restore-header guard

The old 32-candidate cap indirectly bounded the restore manifest, as the earlier G2 review
noted. T8 removes that implicit protection. Before sending **any** reduced bytes, the
gateway now checks the complete `x-nanojev-restore-manifest: ...\r\n` line is at most
**4096 ASCII bytes**, after the existing real restore round-trip. Over-budget plans use
`restore_manifest_header_too_large`: send original bytes and omit the manifest header.
No truncated manifest, alternate content channel, or unrecoverable reduction is emitted.

This conservative per-header bound does not guarantee compatibility with smaller/aggregate
intermediary limits. Large active plans therefore currently fail open; redesigning the
restore transport is a separate task, not a reason to weaken this guard. Active paths were
exercised only with synthetic scorers and loopback fake upstreams, not production traffic.

## Verified evidence

### Deterministic and integration checks

- **15 new core tests:** 1/32/33/64/65/127 candidates; three wire formats; complete judge
  context; original order and shuffled result IDs; synthetic numerical/plan equivalence;
  failed first/middle/final batches; all failures; malformed ID types; uncertainty;
  cross-batch dependency cycles; identity mismatch; usage links and privacy; unchanged
  128-segment bound. Existing core/local tests also pass (32 total in that test family).
- **8 new gateway tests:** real loopback forwarding and three-format restore round-trips,
  failed/timeout batch retention, dependency preservation, failure telemetry, large-header
  original-byte fallback, and exact header-boundary/one-byte-over checks.
- **6 probe tests:** fixed fixture contract, missing/numerically divergent scores cannot
  pass merely because plans agree, near-threshold plan disagreement, exclusive outputs.
- **60 old/new parity cases** across three formats × counts 1/2/16/32 × drop/retain/
  uncertainty/exception/malformed outcomes: **identical scorer payloads and every legacy
  semantic receipt field**. Only IDs/timing and the additive batch fields were excluded.
  Old core is the unmodified `47da116` version; hashes and result are in
  `results/context_batch_legacy_parity_v1.json`.
- Final whole-repository suite: **576 tests, OK (2 skipped)** in 60.308 s.
  Existing unclosed-file `ResourceWarning` remains; no tests failed.
  `git diff --check` passed. There was no installed-skill or production configuration change.

### Real local model partition comparison

Probe: `scripts/probe_context_batches_v1.py`; immutable output directory:
`results/context_batch_probe_v1/` (protocol, per-arm shadow receipts, usage events, report).
The protocol was written before the first probe request; SHA-256:
`837d7c53e8cbbb553de1c013fff5ed20987b069df982c70dbcdfafca0577d673`.

| Format / candidates | 32-state partition | 16-state reference | Max probability difference | Plans |
|---|---|---|---|---|
| OpenAI Chat / 32 | 32 | 16/16 | 0 | identical, retain all |
| OpenAI Chat / 33 | 32/1 | 16/16/1 | 0 | identical, retain all |
| OpenAI Chat / 65 | 32/32/1 | 16/16/16/16/1 | 0 | identical, retain all |
| OpenAI Responses / 33 | 32/1 | 16/16/1 | 0 | identical, retain all |
| Anthropic Messages / 33 | 32/1 | 16/16/1 | 0 | identical, retain all |

All **392 scores** are present; no failed batch is being counted as model abstention.
All **26 usage events** link to the calls, one metadata fingerprint across every arm,
MPS/FP32, temperature 1.0, reported remote model calls **0**. Source/checkpoint files hash
identically before and after. Server compute summed to 61.50 s over these observations;
this is **not a speedup, per-decision latency or tail-latency benchmark**.

**Timeout mismatch is a measured operational limitation:** the probe uses a **30-second
HTTP timeout**, not the gateway's default 5-second deadline. **Two calls exceeded 5 s**;
maximum server compute was **8.244 s**. Those calls would miss the default deadline
(and be retained); this probe is **not proof that default gateway settings can complete
every tested batch**. No timeout was raised in production. Smaller/adaptive batches and
whole-request latency/cancellation need separate measurements before a readiness claim.

The 16-state reference changes only a process-local partition constant in the probe. It
does not change the persistent service, weights, maximum path length or configuration.
Both partitions respect the HTTP service's existing 32-state cap. This provides local
numerical-equivalence evidence, not downstream task correctness or deletion safety.

## Reproduce

```bash
.venv/bin/python -m unittest discover -s scripts -p 'test_context_gate*.py' -v
.venv/bin/python -m unittest discover -s scripts -p 'test_main_model_gateway_batch_v1.py' -v
.venv/bin/python -m unittest discover -s scripts -p 'test_probe_context_batches_v1.py' -v
.venv/bin/python -m unittest discover -s scripts -p 'test_*.py'

# Existing local service must be healthy. Use a NEW output directory on each run.
.venv/bin/python scripts/probe_context_batches_v1.py \
  --url http://127.0.0.1:8876 --output-dir results/context_batch_probe_v1_repeat
```

Main implementation hashes:

| File | SHA-256 |
|---|---|
| `scripts/context_gate_v1.py` | `dfacd69f375b4b64e04e644ca1ee31d694635156271dfaa533d1f284effb8da7` |
| `scripts/main_model_gateway_v1.py` | `089fc3713c5cd856e8b7ed7f6f176b5fe4588f7f02a4be21fc64d85bb9c7363d` |
| `scripts/scorer_adapters_v1.py` | `d497aef0dd016a3f72989f2ec3f522014356f623968b1d7e15ce41322722d883` |
| `scripts/probe_context_batches_v1.py` | `9ab02766e9660131606d159bdd0dd032ef4e3569526f59a486413b4391fe2dc6` |
| real probe `report.json` | `91539e2c5d030f12b99bbd1037b38dcf22e0f39925c053b1ec1b5b2ea2a39dba` |

## Execution provenance and limits

Official DeepSeek V4.1 Flash job `1789835760-4f1f602bffd8` implemented the staged core
and 13 tests in an isolated temporary directory. Official route:
`https://api.deepseek.com/v1`, `deepseek-flash`, version mapping checked; no fallback.
Its own 27-test run passed. The main agent reviewed the actual diff and unchanged input
copies, imported with `apply_patch`, hardened malformed-type/usage handling, added the
versioned additive receipt policy, gateway protections, two extra core tests, eight
integration tests and six probe tests, and executed the local-model comparison.
This is implementation assistance, **not independent production approval**.

Local NanoJev lifecycle events (all real local inference and non-actionable abstentions):

- development: `7948e36b-6376-428b-8890-f05c0ca449c0`, confidence 0.379;
- testing: `7e70edb1-4f49-4961-9454-9ffda5d4573a`, confidence 0.373;
- optimization/parity measurement: `6e9a3787-5baf-4d75-bab0-627dfe791b67`, confidence 0.364;
- optimization/new timeout evidence: `4cdb9e3d-d7af-4a14-b635-c55354737a86`, confidence 0.348.

No deployment phase ran. The installed skill still lacks the separately maintained scope
guard; it has not been silently synchronized. Feedback is based on deterministic outcomes,
not on treating the model's suggested choice as a successful decision.
All four lifecycle events have `fallback` feedback. The 26 probe decisions have separate
linked feedback in `results/context_batch_probe_v1/feedback.jsonl`; the original usage
log and its recorded hash were preserved.

Remaining limits: 128 total segments, 128,000 request bytes, 2 MB scorer transport budget,
2048 model path tokens (overlong inputs refused, not truncated), at most four sequential
scorer calls, per-call rather than whole-request timeout (default at most about 20 seconds
of scoring wait plus overhead), no forced cancellation or prefix sharing. No general
long-context, quality, financial, safety or net-token-savings claim follows from T8.
T8g's frozen routing examples still describe the old 32-candidate bypass; review their
rule freshness in a **new** corpus version, without editing V1 in place.
