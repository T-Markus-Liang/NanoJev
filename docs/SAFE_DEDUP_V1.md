# Safe deduplication V1

Status: **deterministic, model-free deduplication arm implemented, self-tested, and
measured over every request-shaped fixture in this repository. It removes zero segments on
every real fixture, because those fixtures contain no exact duplicate of an arm-eligible
segment; and even where it does find exact duplicates, the unmodified gateway refuses the
plan and fails open. Net token saving sent to a provider: 0. No provider was contacted. No
model was called. No savings are claimed.**

This document specifies a deterministic arm that exists to answer one question honestly:
**can a model-free, no-score, no-threshold rule deliver real token savings today?** The
local NanoJev gate cannot: its Catalog Choice accuracy collapses 100% -> 54.17% under
irrelevant archived context, so it currently, correctly, proposes zero removals and the
whole savings claim is hostage to a model that is not ready. This arm removes that model
dependency from the decision and measures what remains.

It is the companion to [Main-model gateway V1](MAIN_MODEL_GATEWAY_V1.md) (transport,
modes, fail-open reason codes, `estimate`-vs-`actual` accounting),
[Reversible filtering V1](REVERSIBLE_FILTERING_V1.md) (the restore-manifest contract), and
[Tool-history shadow V1](TOOL_HISTORY_SHADOW_V1.md) (the A4 fixture contract). It changes
none of them.

## 1. Files

| File | Role |
|---|---|
| `scripts/safe_dedup_v1.py` | The deterministic rule, `--help`, `--self-test`, `--plan`, measurement CLI |
| `scripts/test_safe_dedup_v1.py` | 40 unittest cases: determinism, exact-duplicate-only removal, protected retention, byte-identical restore, tamper rejection, gateway fail-open, contract validity |
| `results/safe_dedup_v1.json` | The measurement receipt with provenance and the accounting status |
| `docs/SAFE_DEDUP_V1.md` | This specification |

Nothing else in the repository is modified. `scripts/context_gate_v1.py`,
`scripts/context_restore_v1.py`, and `scripts/main_model_gateway_v1.py` are imported
read-only, never edited, wrapped, forked, or monkey-patched.

## 2. The rule, precisely

Given request bytes that the existing shadow core can parse, a wire format, and the same
trusted caller sidecar the core already accepts, **SafeDedup V1** decides as follows, with
no randomness, no score, no probability, no threshold, no model, and no I/O:

1. **Enumerate** segments with `context_gate_v1.parse_segments`. If it raises `Bypass`
   (`unresolved_tool_link`, `unsupported_content`, `request_budget_exceeded`, ...) the plan
   is empty and reports the bypass reason. Nothing is removed from a request the core
   cannot fully enumerate.
2. **Protect** with `context_gate_v1.sidecar_policy` and
   `context_gate_v1.protect_dependencies`, unchanged. A segment is **arm-eligible** only
   when the core assigns it *no retention reason at all*: it must be text-bearing,
   structurally unprotected (role not in `system/developer/user/tool/control`; not
   tool-linked; not `pinned`/cited/annotated), carry none of the six caller protection
   flags (`pinned`, `cited`, `safety`, `credential`, `dependency`, `exact_text`),
   be explicitly marked `eligible: true` in the trusted sidecar, and be in the restore
   module's pointer grammar. Every core retention reason is respected verbatim --
   including `not_explicitly_eligible`, which `docs/TOOL_HISTORY_SHADOW_V1.md` declares
   protected. With no sidecar grant, the arm removes nothing.
3. **Normalize by identity.** The declared normalization is
   `identity-v1-exact-codepoint-equality`: `normalized_text(x) = x`. Two segments are
   duplicates only when their texts are equal code point for code point. There is no case
   folding, no NFC/NFKC, no whitespace collapsing, no truncation, no tokenization. Identity
   is the only normalization that guarantees the bytes removed are exactly the bytes that
   remain in the retained copy.
4. **Group** arm-eligible segments by `(kind, role, normalized_text)`, where `kind` is
   `message` (a whole `/messages|input/{i}/content` string) or `text_part`
   (`/messages|input/{i}/content/{j}/text`). A whole-message copy and a text-part copy of
   the same string are not the same unit and never deduplicate against each other.
5. **Keep the first occurrence; drop only later exact duplicates.** In request order the
   first occurrence of every group is always retained, so a surviving identical copy exists
   for every removal. A segment with no earlier exact duplicate is never touched. Novel
   text is never removed.
6. **Re-close dependencies** over the effective retained set (all core-retained segments
   plus every retained first occurrence), reusing the core's own `protect_dependencies`. A
   duplicate that a retained segment declares `depends_on` is retained.
7. **Prove reversibility before removing.** Each candidate is added only if the resulting
   plan survives the same sequence the gateway's active path uses:
   `build_reduced_request` (the gateway's own applicator) -> `parse_segments` re-validation
   -> `build_restore_manifest` (content-free) -> `removed_segments_from_raw` ->
   `restore_request` must reproduce the canonical original **byte-identically**. Any
   failure -- an unbuildable plan, a pointer outside the restore grammar, a reconstruction
   that does not hash to the recorded original, a tamper -- means the candidate is
   *retained* with reason `retain_not_reversible`.

Anything ambiguous, novel, protected, dependency-bound, non-canonical, or not provably
reversible is retained. The arm fails **closed** on the plan (retain) and the existing
gateway fails **open** on the request (forward the original bytes).

### Known irreversible shapes the proof retains

The restore module reconstructs a dropped text part as `{"type": "text", "text": ...}`.
Therefore a text part that is not exactly that shape -- an `openai_responses`
`input_text`/`output_text` part, or a `text` part carrying `cache_control` -- cannot be
reconstructed byte-identically. The proof detects this and retains the candidate. This is
behaviour, not a bug: the arm does not send a reduction the caller could not undo.

## 3. The measurement

Fixtures measured (all on disk, no network):

* **A4 tool-history fixtures** -- `research/tool_history_fixture_manifest_v1.json`
  (sha256 `0c8253c9...`), materialized in memory with the builder's own tag substitution
  (`scripts/build_tool_history_fixtures_v1.py`, seed 20260919). 11 cases.
* **Request-shaped corpus** -- every body in
  `data/context_relevance_oracle_v1_seed20260919/*.jsonl` (2400 bodies across 5 splits).
  This is the largest request-shaped corpus in the repository. It is **not** long-context:
  the largest body measured 760 bytes.
* **Synthetic mechanism check** -- 8 bodies authored inside `safe_dedup_v1.py` purely to
  exercise the rule's positive and negative paths. Reported separately; its reductions are
  **not** savings.

Tokenizer: a **real, local, offline BPE tokenizer**
(`checkpoints/local_atomic_seed17/variants/local_atomic_seed17/tokenizer/tokenizer.json`,
vocab 151669, model type `BPE`, sha256 `be756060...`). `tiktoken` is not installed; no
provider tokenizer is used or available. Tokens are counted over the canonical JSON body
text, which is a local estimate and **not** a provider chat-template render and **not**
provider billing.

### Result table (real fixtures)

| Source | Cases | Segments examined | Segments removed | Canonical chars (orig -> reduced) | Tokenizer tokens (orig -> reduced) | Removed-segment text | Protected deletions | Removal round-trips |
|---|---|---|---|---|---|---|---|---|
| A4 tool-history (8 parsed, 3 bypass) | 11 | 55 | **0** | 6959 -> 6959 (0) | 2230 -> 2230 (0) | 0 chars / 0 tokens | **0** | none attempted (0 removals) |
| Request-shaped corpus | 2400 | 11400 | **0** | 1345414 -> 1345414 (0) | 361029 -> 361029 (0) | 0 chars / 0 tokens | **0** | none attempted (0 removals) |
| **Total, real fixtures** | **2411** | **11455** | **0** | **1352373 -> 1352373 (0.00%)** | **363259 -> 363259 (0.00%)** | **0 / 0** | **0** | **0 attempted** |

Zero-removal requests still exercised the manifest contract with an empty plan:
**2408 / 2408** no-op manifest checks passed (A4 8, corpus 2400). Those are contract checks,
not restoration round-trips.

### Result table (synthetic mechanism check, not a saving)

| Source | Cases | Segments examined | Segments removed | Chars | Tokens | Removed-segment text | Protected deletions | Round-trips |
|---|---|---|---|---|---|---|---|---|
| Synthetic mechanism check | 8 | 43 | 5 | 2615 -> 2352 (-10.06%) | 618 -> 563 (-8.90%) | 93 chars / 20 tokens | **0** | **2 / 2 = 100%** |

The synthetic check also proves the negative paths: a duplicated tool result, a duplicated
user message, a duplicated system message, a caller-`safety`-flagged duplicate, an
undeclared (not explicitly eligible) duplicate, and a dependency-bound duplicate are all
retained; and 2 candidates were retained specifically because their restore would not be
byte-identical (an `output_text` duplicate and a `cache_control` duplicate).

### Restoration round-trip

* Every removal the arm performs is restored byte-identically. Combined over all sources:
  **2 / 2 reduced requests restored = 100%**; across the real fixtures there were **0**
  removals, so the rate over those sources is **undefined**, not 100% -- the receipt says
  so explicitly (`round_trip_success_rate: null` with a note).
* Tamper rejection is exercised in the module self-test and the test suite: tampered
  manifest hash, tampered manifest count, tampered reduced bytes, tampered removed
  segment, missing segment, and extra segment all raise `RestoreError`. The arm never
  returns a wrong reconstruction.
* The existing gateway's own fail-open path was exercised end-to-end with only its
  transport stubbed: a well-formed reduction whose restore fails is not sent
  (`forward_reason: reduction_error`, original bytes forwarded, no
  `x-nanojev-restore-manifest` header, `savings.claim: none`).

### `estimate`-vs-`actual` accounting under the gateway's existing rule

`docs/MAIN_MODEL_GATEWAY_V1.md` section 8 allows an **actual** savings claim only when a
genuinely reduced request was sent and the caller supplies a provider-reported baseline,
so both ends are provider-reported. Here:

* **No provider was contacted** and **no reduced bytes were sent** (`bytes_sent_to_provider: 0`).
* The unmodified gateway refuses this arm's plan: `removal_plan` accepts only
  `role: assistant` drops whose receipt reason is the model reason
  `high_irrelevance_score`. This arm is model-free and does not -- must not -- emit that
  reason, so the gateway resolves the plan to
  `protected_segment_in_removal_set` and forwards the ORIGINAL bytes.
* Therefore the gateway's rule reports **`claim: "none"`, `basis: "no_reduction_sent"`,
  `tokens: null`**. The only number the rule permits is a shadow **`estimate`**
  (`basis: "shadow_estimate_only"`), which is **0 tokens** on every real fixture here.
* **`actual_provider_savings: null`. `net_sent_token_saving: 0`.**

### Gateway compatibility blocker

Today the arm cannot be carried by the existing gateway without a reviewed gateway change
(out of scope, and the rules forbid editing the gateway). This is the second, independent
reason the net saving is zero: even if the fixtures had contained exact duplicates, the
unchanged gateway would fail open on the arm's plan reason and send the original bytes.

## 4. What the numbers do and do not establish

**They do establish:**

* A deterministic, model-free dedup rule exists, is implemented, is deterministic, and is
  covered by 40 passing tests plus a `--self-test`.
* On the exact shape it targets (identical repeated assistant text), it removes duplicates,
  keeps the first copy, and restores every reduction byte-identically (2/2 synthetic
  round-trips, 100%).
* It never removes a protected segment: 0 protected deletions and 0 tool-linked deletions
  over 11 455 real segments, including duplicated tool results, user messages, system
  messages, caller-flagged, undeclared, and dependency-bound segments.
* On **every real fixture available in this repository it removes nothing**: the A4 bodies
  and the relevance corpus contain no exact duplicate of an arm-eligible segment. That is
  the honest answer to "can this arm deliver savings today on the available evidence?" --
  **no measured saving**.
* The character and token reductions are exactly 0 on real fixtures, so no percentage can
  be quoted from them.

**They do NOT establish:**

* **No real token saving.** Nothing was sent to a provider. A reduction that is not sent is
  not a saving, and a character reduction is not a token reduction. The net sent token
  saving is **0**.
* **No provider token count.** The token numbers come from a real local BPE tokenizer over
  canonical JSON body text. They are a local estimate. Provider tokenizers, chat templates,
  cached-input pricing, and output tokens differ and were not measured.
* **No long-context evidence.** There is no long-context fixture in this repository. The
  corpora measured are small (max body 760 bytes); long tool histories with genuine
  repeated blocks were not available, so the arm's real-world duplicate rate is unknown.
* **No value for the *omitted* context.** Deduplication assumes the retained identical copy
  is an acceptable substitute for the removed position. That is a strong assumption; the
  paired downstream-quality comparison required by Track A was not performed.
* **No end-to-end integration.** The arm is a library + CLI, not wired into the gateway.
  The gateway refuses its plan reason; enabling it needs a reviewed change plus the full
  Track A acceptance gates.
* **No interaction with tool re-execution, caching, retries, or cost accounting.** None of
  those were modelled.
* **No independent review.** These files have not been reviewed by the project's
  independent reviewer.

## 5. Safety statement

**No reduction may be sent to a real provider on the basis of this document alone.**
The arm is offline evidence that a model-free rule is deterministic, protectively
conservative, and reversible. It is not authorization to enable active filtering: active
filtering remains disabled by default, the gateway's plan vocabulary does not accept this
arm's reason, and the Track A acceptance gates, paired downstream-quality evidence across
at least three main-model families, and a protected-segment stress set with zero deletions
are still required before any such change.

## 6. Reproduction

```bash
# Rule help and contract surface.
.venv/bin/python scripts/safe_dedup_v1.py --help

# Deterministic in-process checks (17 checks; writes nothing).
.venv/bin/python scripts/safe_dedup_v1.py --self-test

# The full measurement; rewrites the receipt deterministically.
.venv/bin/python scripts/safe_dedup_v1.py --out results/safe_dedup_v1.json

# Plan a single request without measuring.
.venv/bin/python scripts/safe_dedup_v1.py --plan request.json \
  --wire-format openai_chat --sidecar sidecar.json

# The test suite (40 tests).
.venv/bin/python -m unittest discover -s scripts -p 'test_safe_dedup_v1.py'
```

The measurement is deterministic: `repository_head`, file SHA-256 digests, and the
tokenizer identity are recorded, and re-running produces byte-identical JSON. The arm makes
no network call, opens no socket, calls no model, and writes nothing except the `--out`
receipt.
