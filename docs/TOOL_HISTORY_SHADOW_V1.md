# Tool-history compaction: shadow fixtures and contract V1

Status: **fixtures and shadow-only tooling implemented; no active filtering, no receipt run, no token savings.**

This document specifies work package **A4** ("tool-history compaction shadow experiment") from
[CURRENT_PROGRESS_AND_HANDOFF.md](CURRENT_PROGRESS_AND_HANDOFF.md). It builds the fresh, source-group-isolated
fixture manifest that [JEV_COMMUNITY_REFERENCES.md](JEV_COMMUNITY_REFERENCES.md) (Track A:
`fast-jev-compaction`) calls for, and it exercises the existing byte-preserving shadow core without forking it.
It does not change the [shadow integration V1](CONTEXT_SHADOW_V1.md) contract, any production request path, or
any frozen artifact.

## 1. What was implemented

| File | Role |
|---|---|
| `research/tool_history_fixture_manifest_v1.json` | Frozen fixture contract: 11 cases, each declaring the expected protected/eligible status of every segment and the atomic call/result pairing. |
| `scripts/build_tool_history_fixtures_v1.py` | Deterministic (seeded) materializer. Refuses a non-empty output directory. `--self-test` validates manifest integrity without writing output. |
| `scripts/test_tool_history_fixtures_v1.py` | unittest suite for manifest integrity, deterministic rebuild, source-group isolation, orphan results, atomic pairing, and protected-segment assertions. |
| `docs/TOOL_HISTORY_SHADOW_V1.md` | This spec. |

The manifest was frozen before materialization: `status` is `frozen_before_shadow_materialization`, and the
builder treats it as read-only input (it never writes back to `research/`). The builder's `--self-test`
re-derives every declared status from the real core on every run, so a manifest edit that contradicts the core
fails loudly instead of silently redefining the contract.

## 2. Reuse of the existing shadow core (no fork)

All analysis goes through `scripts/context_gate_v1.py` exactly as it exists:

* `parse_segments(raw, wire_format)` enumerates the segments and the structural tool linkage;
* `sidecar_policy(...)` and `shadow_request(raw, wire_format, sidecar, scorer, threshold=0.99)` produce the
  shadow plan and the text-free receipt.

The builder imports `Bypass`, `FORMATS`, `parse_segments`, `shadow_request`, and `serialized` from that module
and does not modify, wrap, or re-implement its logic. `shadow_request` returns the **identical `bytes` object**
it was given (`context_gate_v1.py` line 345: `return raw, receipt`), and the builder asserts `output is raw`,
`request_sha256 == forwarded_sha256`, `actual_removed_segments == 0`, and `actual_removed_tokens == 0` for every
fixture. The production request therefore remains byte-identical by construction, and the fixture suite is a
regression test for that guarantee.

The frozen `0.99` threshold is untouched: the manifest declares `threshold_frozen: true`,
`probabilities_calibrated: false`, and the receipts must report `policy_threshold == 0.99`. No transaction
filtering code, provider proxy, network call, or service configuration is added.

## 3. The three allowed outcomes

Every declared call/result pair lists `allowed_outcomes`, a subset of exactly three operations. These are the
**only** operations any future active design may choose from; this experiment evaluates none of them.

| Outcome | Meaning | When it is permitted |
|---|---|---|
| `keep_both` | Retain the call segment and every result segment verbatim. | Always. This is the fail-open default and the only outcome for unresolved pairs. |
| `keep_call_bounded_result` | Keep the call and replace the result with an **explicitly marked** bounded view (for example a declared truncation marker plus an offset into the retained original). Never silent truncation. | Only for pairs that are repeatable and carry no error, credential, non-repeatable side effect, stale evidence, or correction dependency. `mutable_file_read` permits it because the recorded value can be kept inside the bound. |
| `remove_pair_atomically` | Remove the call segment **and all** of its result segments together. One side is never removed alone. | Only for pairs marked repeatable, non-volatile, and free of every blocking flag. In this manifest that is `success` and `parallel_calls`; both are synthetic, reproducible reads. |

Blocking flags and their effect (enforced by `validate_manifest`):

| Flag | `keep_both` | `keep_call_bounded_result` | `remove_pair_atomically` |
|---|---|---|---|
| `contains_error` | yes | no | no |
| `credential_bearing` | yes | no | no |
| `non_repeatable` | yes | no | no |
| `stale_evidence` | yes | no | no |
| `correction_dependent` | yes | no | no |
| `volatile_source` | yes | yes | no |
| `pairing_resolved == false` | yes | no | no |
| plain repeatable read | yes | yes | yes |

A delegated report (`scripts/build_tool_history_fixtures_v1.py --self-test`) prints the number of scored and
bypassed cases, so the outcome mix is auditable without opening the manifest.

## 4. Why retained text must stay verbatim

The core, the manifest, and the tests share one rule: **retained means byte-for-byte retained.**

* Tool results are evidence, not summaries. An omitted result body can hold the only error code, the only
  version string, the only transaction id, or the value a later user correction refers back to (cases `error`,
  `non_repeatable_result`, `stale_output`, `user_correction`).
* Re-execution is not a recovery mechanism. Re-running a tool may be costly, state-changing, unavailable, or
  may simply return different data (`mutable_file_read`, `non_repeatable_result`); the receipt therefore
  records `applied: false` and the plan never triggers a re-run.
* The judge sees full context. `scoring_payload` is rendered from the parsed conversation without truncating
  the state, and a retained segment's hash is computed over the untouched value. Any partial rewrite would make
  `fingerprint(segment.value)` incomparable with the caller's original.
* Receipts carry SHA-256 hashes and fixed reason codes, not content. Hashes cannot reconstruct the original and
  are not encryption, so the caller — not the receipt — remains the only holder of the retained text.
* Dependency closure is transitive and cyclic-safe. If an eligible segment is required to interpret a protected
  segment, it is retained as `required_dependency` regardless of score (`user_correction`).

## 5. Fixture manifest structure

```
research/tool_history_fixture_manifest_v1.json
├── gate_contract        core, entrypoint, frozen 0.99 threshold, three outcomes, pairing rule
├── tag_contract         @SYNTH_TAG@ placeholder and its seed-derived substitution rule
├── provenance           self-authored synthetic only; no real user data/credentials/tool output
├── source_groups[11]    one isolated synthetic world per fixture
└── cases[11]            body, sidecar, declared segments, pairing, expected gate, leak tokens
```

Each case declares:

* `body` — the exact JSON request that is materialized (with the tag placeholder);
* `sidecar` — the trusted metadata pointer map (`eligible`, `depends_on`, protection flags);
* `segments[]` — every segment pointer, its role, `expected_status` (`protected` or `eligible`),
  `expected_suggestion` (`retain` or `drop`), and `expected_reason_code`;
* `pairing[]` — atomic pairs: `call_segment`, `result_segments`, and the dependency flags with
  `allowed_outcomes`; `pairing_resolved` says whether the core can attribute the result unambiguously;
* `expected_gate` — `status`, `reason`, whether segments are enumerated, and whether the whole request is
  retained;
* `leak_tokens` — literals that must never appear in a receipt.

### Source-group isolation

Each fixture owns exactly one source group (`isolation_unit: single_fixture`) whose `group_id` is
`sha256` over the case's declared synthetic world, never over a file name, split, or ordering. The suite proves
isolation three ways:

1. group ids are unique and each matches the world hash;
2. entity tokens are unique and no token is a substring of another, so byte-level absence is unambiguous;
3. each materialized `request.json` contains its own tag and entity tokens and **none** of the other fixtures'.

The seed-derived tag (`tg-` + 12 hex characters from `sha256("nanojev-tool-history:{seed}:{case_id}")`) makes
the isolation check also a determinism check: the same seed reproduces the same bytes, a different seed produces
different synthetic identifiers while leaving every pointer and declared status unchanged. No fixture reads
from `data/`, `results/`, or the frozen relevance corpus; the manifest declares
`reuses_frozen_relevance_corpus: false`.

## 6. Case coverage

11 cases, 8 scored (enumerated) and 3 whole-request bypasses.

| # | Case | Wire | Pairing | Allowed outcomes | What it pins down |
|---|---|---|---|---|---|
| 1 | `success` | openai_chat | resolved | all three | Baseline atomic pair; the only pair where bounded/removal is declared safe. |
| 2 | `error` | openai_chat | resolved | `keep_both` | The result is the only failure evidence; also exercises `not_explicitly_eligible`. |
| 3 | `parallel_calls` | anthropic_messages | resolved | all three | Two `tool_use` blocks with two `tool_result` blocks keep true 1:1 pairs. |
| 4 | `duplicate_ids` | openai_chat | unresolved | `keep_both` | Two calls share one id; the core bypasses instead of guessing attribution. |
| 5 | `orphan_result` | openai_chat | unresolved | `keep_both` | A result id that no call issued must never be adopted by a nearby call. |
| 6 | `pending_call` | openai_chat | unresolved (no result) | `keep_both` | An unfinished call stays; no result segment exists to pair with. |
| 7 | `non_repeatable_result` | openai_chat | resolved | `keep_both` | Committed side effect plus explicit `safety` sidecar flags; never re-run, never trim. |
| 8 | `mutable_file_read` | openai_chat | resolved | `keep_both`, bounded | A snapshot of a volatile source: keep the recorded value, never refresh. |
| 9 | `stale_output` | openai_chat | resolved | `keep_both` | A stale value is needed to interpret the correction that supersedes it. |
| 10 | `user_correction` | openai_chat | resolved | `keep_both` | User corrections are protected; dependency closure retains the pre-correction restatement. |
| 11 | `secrets_credentials` | openai_chat | resolved | `keep_both` | Placeholder credential only; receipt must contain hashes, never the literal. |

## 7. Dependency-label semantics

`depends_on` is the only mechanism that makes an otherwise eligible segment non-removable, and it is supplied
by the trusted integration, never parsed from prompt text. The direction is: *"this segment needs that segment
to be interpretable"*. When a segment is retained for any reason, `protect_dependencies` closes transitively
over its `depends_on` list, so the dependency is retained too.

`expected_status` in the manifest means: **`protected`** = must never be proposed for removal (structural
protection, a caller protection flag, dependency closure, or not being explicitly eligible); **`eligible`** =
a legal scoring candidate that the model may propose for removal. A protected segment is always
`expected_suggestion: retain`.

## 8. Ambiguities and how they were resolved

1. **Parallel calls in `openai_chat` share one call segment.** A single assistant message carries
   `tool_calls: [A, B]` and `parse_segments` emits one `/messages/N/tool_calls` segment holding both ids. A
   1:1 `call_segment → result_segment` mapping is therefore impossible in that wire format, so the parallel
   case is expressed in `anthropic_messages`, where each `tool_use`/`tool_result` part is its own segment and
   true 1:1 pairs hold. The manifest still permits a call segment to own several result segments (a result id
   plus its content in `openai_chat`), so pair validation requires *disjointness and full coverage*, not
   cardinality 1:1.
2. **A result whose id matches no call.** Two readings exist: attach the result to the nearest call, or refuse
   to attribute it. The reference warns against guessing, so the contract refuses: `orphan_result` declares the
   *intended* pair but `pairing_resolved: false`, and the core returns `unresolved_tool_link` with no shadow
   plan. The same rule covers `duplicate_ids` and `pending_call`.
3. **A call without a result.** No result segment exists, so the declared pair has an empty
   `result_segments` list and is unresolved. This is the one place where a pair legitimately has no results;
   `validate_manifest` requires `pairing_resolved: false` in that situation, so an empty result set can never
   masquerade as a complete pair.
4. **Which side of `depends_on` protects a segment.** In `user_correction`, the assistant restatement is
   eligible on its own, so it is protected only because the *protected* final user request declares
   `depends_on` it. The manifest documents this direction explicitly rather than marking the assistant text
   protected directly, so the test exercises dependency closure instead of restating structural protection.
5. **`not_explicitly_eligible` is a retention reason, not an eligibility class.** A plain assistant segment
   with no sidecar note is retained (`error`, `duplicate_ids`, `orphan_result`, `pending_call`). It is declared
   `protected` in the manifest, but its reason code is `not_explicitly_eligible`, distinct from
   `protected_structure`; the tests assert the reason code separately so the two cannot be conflated.
6. **Stale evidence versus a redundant restatement.** In `stale_output` the raw tool pair is
   `keep_both` (it is needed to interpret the correction) while a later assistant restatement of the same stale
   value remains an eligible candidate. The two are separate segments precisely so the contract can be
   conservative about evidence without freezing redundant prose.
7. **Where a bounded result is acceptable.** `keep_call_bounded_result` is meaningful only when the omitted
   span is not the evidence itself. `mutable_file_read` keeps the recorded value and allows a bound around the
   rest; `error`, `secrets_credentials`, `non_repeatable_result`, `stale_output`, and `user_correction` forbid
   it because the omitted span *is* the evidence.
8. **How unresolved cases are still structurally verified.** The core refuses to enumerate a bypassed request,
   so its declared per-segment statuses cannot be compared directly. The manifest therefore declares a minimal
   `ambiguity_probe` (repair the call-id typo, or append the missing result) used **only** to enumerate the
   structural contract; the probe is never materialized into a fixture, and the unprobed request must still
   bypass with the declared reason.

## 9. What is explicitly NOT done

* No active context filtering, deletion, rewriting, truncation, or summarization. No code path removes a
  segment; `applied` is `false` everywhere.
* No provider proxy, HTTP forwarding, network access, or service configuration change. The production request
  remains byte-identical and untouched.
* No modification or fork of `scripts/context_gate_v1.py`; it is imported read-only.
* No shadow receipts generated over the three eligible operations, no real transcript run, and no end-to-end
  cost/token measurement. A4 deliverable 3 (receipts) remains open.
* No downstream model comparison, no paired task correctness, no token/cache/retry accounting, and no
  three-model-family evaluation. A4 deliverables 4–5 remain open.
* No threshold tuning of any kind. The `0.99` threshold is frozen and remains a provisional diagnostic, not a
  calibration claim; upstream's `0.5` is not copied.
* No real user transcript, real credential, or real tool output. Fixtures are self-authored synthetic data;
  the credential value is the literal `not-a-real-secret-0000`.
* No reuse of the frozen relevance corpus, its splits, or its OOD families as fixture data.
* No `git commit` or `git push`, and no edits to any existing repository file.

## 10. Reproduction

```bash
# Manifest integrity only; writes nothing.
.venv/bin/python scripts/build_tool_history_fixtures_v1.py --self-test

# Help / contract surface.
.venv/bin/python scripts/build_tool_history_fixtures_v1.py --help

# Materialize fixtures into an empty directory (temporary for experiments).
.venv/bin/python scripts/build_tool_history_fixtures_v1.py --output-dir /tmp/toolhist-fixtures

# The full A4 fixture suite (scoped pattern; no global discovery).
.venv/bin/python -m unittest discover -s scripts -p 'test_tool_history*.py'
```

Materialized layout, per case and deterministic across runs with the same seed:

```
<output-dir>/fixture_index.json          # seed, per-case tag, SHA-256 of every file
<output-dir>/<case>/request.json         # exact bytes fed to shadow_request
<output-dir>/<case>/sidecar.json         # trusted eligibility/dependency metadata
<output-dir>/<case>/expected.json        # declared segments, pairing, expected gate, leak tokens
```

The expected test result is **13 tests, OK**. Materialization refuses any non-empty output directory, so
running it twice against the same directory fails with `output directory must be empty` rather than
overwriting fixtures.

## 11. Privacy and safety notes

* Receipts and `expected.json` contain pointers, hashes, reason codes, and flags only. The suite asserts that
  the synthetic tag, entity tokens, and the credential placeholder never appear in a receipt.
* `--seed` changes only synthetic identifiers; it cannot introduce real data.
* The builder never calls a tool, never re-runs one, and never opens a socket.
* The manifest is the single frozen source of truth. Editing `context_gate_v1.py` behaviour would break
  `--self-test`, which is the intended tripwire.

## 12. Review gate (RA) and next steps

Per the handoff plan's review point **RA**, new eligibility/truncation rules, external model requests, and any
active truncation require review before execution. This package stays inside the approved shadow-only scope:
fixtures, deterministic tooling, tests, and a frozen contract.

Open items for a future, separately reviewed package:

1. A4 deliverable 3 — shadow receipts for three eligible operations (with the original request preserved),
   including timeout, malformed score, oversized context, unsupported block, and uncertain-relevance arms.
2. A4 deliverable 4 — the frozen paired comparison (unfiltered, deterministic safe deduplication,
   upstream-style policy, local gate) with result-omitted versus evidence-bearing judge states reported
   separately.
3. A4 deliverable 5 — paired downstream task correctness and measured input tokens across at least three
   main-model families, before any default active use.
4. A bounded-result encoding for `keep_call_bounded_result`: the core has no field for an explicitly marked
   bound today, so that outcome is declared and tested as a contract but deliberately not implemented.
5. Independent review of the manifest's outcome assignments, especially that `success` and `parallel_calls`
   are the only pairs where atomic removal is declared permissible.

See [CURRENT_PROGRESS_AND_HANDOFF.md](CURRENT_PROGRESS_AND_HANDOFF.md) and
[JEV_COMMUNITY_REFERENCES.md](JEV_COMMUNITY_REFERENCES.md) for the surrounding boundaries.
