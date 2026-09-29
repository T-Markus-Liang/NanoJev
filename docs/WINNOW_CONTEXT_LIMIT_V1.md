# WINNOW_CONTEXT_LIMIT_V1 — winnow 400 rejections in real-context eval

Date: 2026-09-28. Scope: diagnosis of `results/winnow_real_candidates_v1.jsonl`
(220/547 rows, `status 400`) plus local scorer fleet health. Read-only
diagnosis; no rebuild performed.

## Fleet health (verified live, 2026-09-28)

| Port | Service | Probe | Result |
|------|---------|-------|--------|
| 8091 | winnow (Winnow-12B Q8_0, gemma4 arch) | `GET /health` | `{"status":"ok"}` 200 |
| 8091 | winnow | `POST /v1/systemone` (noul, tiny state) | 200, ~430 ms, `noul` returned |
| 8091 | winnow | `POST /v1/winnow/inspect` (547 candidates) | 200, all tokenized, ~6 s total |
| 8092 | kev (`jaredpalmer/kev-4b`, Qwen3.5-4B base) | `GET /health` | 404 — no `/health` route (expected; FastAPI app) |
| 8092 | kev | `GET /v1/models` | 200, `kev-latest` (alias `jev-latest`) |
| 8092 | kev | `POST /v1/systemone` | 200, ~400 ms, `noul` returned — responsive post-respawn |
| 8093 | valen (`nano_rlcd_v2/valen-head@Qwen3.5-0.8B`) | `GET /health` | 200 |
| 8093 | valen | `POST /v1/systemone` | 200, ~2.3 s (cold head) |
| 8094 | lora-shadow (`nano_sft_text_v4_fp32/valen-head@Qwen3.5-0.8B`) | `GET /health` | 200 |
| 8094 | lora-shadow | `POST /v1/systemone` | 200, ~1.0 s |
| 8876 | main (`scripts/serve_decisions.py`) | `GET /health` | 404 — use `GET /api/health` |
| 8876 | main | `GET /api/health` | `{"ready":true,...,"backends":{"winnow":true,"kev":true,"valen":true,"valen_lora":true}}` |
| 8876 | main | `POST /v1/systemone` and `?backend=kev` | 200 both — routing works |

Note: kev (:8092) requires `questions.<name>.instructions` (noul variant); a
criteria-only question returns 422. valen/lora-shadow return 500
`"instructions must be nonempty text"` for the same shape. The candidates'
real question payload (`{"type":"noul","instructions":...}`) is accepted by
all four backends.

## Root cause of the 220 rejections

Winnow is served by `external/winnow-inference/scripts/serve.py` (launchd:
`deploy/launchd/ai.nanojev.winnow.plist`) with `--context 8192`, which sets
both the llama server `--ctx-size 8192` and `WINNOW_CONTEXT=8192` for the
decision engine. Server log confirms `n_ctx_slot = 8192`; `/v1/winnow/inspect`
runtime reports `context: 8192`.

Two distinct 400s exist in the decision path:

1. `native/engine.h:439-440` — `prefix_size >= llama_n_ctx(ctx)` throws
   `"State must leave space for question suffixes inside the context"`.
   205 of the 220 rejects hit this (winnow prefix tokens 8,192–10,672).
2. `native/planner.h:127-140` (`wave_end`) — when `prefix + question suffix`
   exceeds capacity, `"State plus question exceeds context capacity"`.
   15 of the 220 rejects hit this (winnow prefix tokens 8,115–8,188; the
   rendered `irrelevant` question suffix is 96 tokens).

Effective ceiling, measured: winnow **prefix ≤ 8,096 tokens** (8,192 − 96
suffix). Verified live — candidate i=329 at prefix 8,096 returns 200 with
`usage.input_tokens = 8192` exactly; candidate i=0 at prefix 8,115 returns
400. Maximum accepted prefix across all 327 scored rows is exactly 8,096.

## Why candidates at ≤7,000 "packed tokens" still overflowed

`scripts/build_real_context_eval_v1.py` windows segments to
`TOKEN_BUDGET = 7_000` using the **Qwen tokenizer** (`PackedTokenCounter`,
valen compiler convention, line ~94-204) — sized for OUR scorer's 8,192
limit, leaving 1,192 tokens of headroom *in Qwen tokens*.

Winnow-12B is a Gemma-4 12B fine-tune (`external/models/Winnow-12B/README.md`:
`base_model: google/gemma-4-12B-it`) with a different tokenizer. Winnow
prefix tokens vs Qwen packed tokens on this corpus:

- ratio winnow/qwen: min 1.057, median 1.193, max 1.569
- rejected rows span 5,950–7,000 Qwen packed tokens; accepted rows reach
  6,997 — the ranges overlap because the ratio is content-dependent
- worst ratio on multilingual/Chinese-heavy transcripts: rejected set is
  151 `multi` / 69 `en` vs accepted 303 `multi` / 24 `en`
- additional overhead: winnow wraps `state` in a fixed system/user preamble
  (`native/protocol.h:56-66`) and JSON-re-dumps it via `safe_data` (`<`
  escaped as `\u003c`)

Distribution of the 220 rejects (winnow prefix tokens): min 8,115,
median 8,649, max 10,672. Accepted: min 1,174, median 7,421, max 8,096.
There are zero accepted rows in 8,097–8,114 — the ceiling is sharp.

## Recommendation

**(b) accept winnow partial coverage for real eval v1** — zero work and
already safe by construction: `scripts/build_review_queue_v1.py` maps
missing `noul` → `"uncertain"`, which routes every rejected record to human
review (line ~66-72). Winnow simply abstains on 220/547 (40.2%) candidates.
Coverage is biased toward shorter/more-English transcripts, which the review
queue's disagreement/uncertainty composition already absorbs.

Option (a) — rebuild a winnow-compatible subset — is feasible and cheap if
full coverage is wanted later:

- Re-window only the 220 rejected records. Do **not** lower `--token-budget`
  on the Qwen counter: the winnow/qwen ratio varies 1.06–1.57 per document,
  so no fixed Qwen budget cleanly separates pass/fail (a conservative
  ~5,100-token Qwen budget would work but discards ~1,900 tokens of usable
  context on most rows).
- Correct approach: reuse `_window_indices`/`_render_window` from
  `scripts/build_real_context_eval_v1.py` with a winnow-token counter. The
  exact count is already available via `POST /v1/winnow/inspect`
  (tokenize-only, ~10 ms/candidate; all 547 measured in ~6 s for this
  report). Loop: drop segments until `prefix_tokens ≤ 8,096 − margin`
  (suggest margin ≥ 32, i.e. target ≤ 8,064, to cover suffix variation if
  question text ever changes — the current suffix is a fixed 96 tokens).
- Then re-run `eval_systemone_backend_v1.py --url
  http://127.0.0.1:8091/v1/systemone` on the rebuilt subset (~220 × ~7 s
  ≈ 26 min wall clock) and merge rows into the results file by `i`.
- Estimated effort: ~2–3 hours (small driver script ~100 lines, rebuild,
  rescore, merge, verify `error` count → 0 for covered rows). Any record
  whose minimal single-segment window still exceeds 8,064 stays a winnow
  abstain by design.

Do **not** raise `--context` in the plist instead: the Winnow-12B README
documents an 8,192-token LoRA training cap (`external/models/Winnow-12B/
README.md`); the deployment's `--context 8192` matches it. Serving beyond
8K is out-of-envelope for the fine-tune even though `serve.py` defaults to
65,536 and the base Gemma-4 supports it.

## Reproduction

```
# per-candidate winnow prefix token count (tokenization only, no scoring)
curl -s http://127.0.0.1:8091/v1/winnow/inspect \
  -H 'Content-Type: application/json' -d @candidate_request.json
# → {"prefix_tokens": N, "suffix_tokens": [96], ...}

# boundary check: prefix 8096 → 200; prefix 8115 →
#   400 "State plus question exceeds context capacity"; prefix ≥ 8192 →
#   400 "State must leave space for question suffixes inside the context"
```

Measured token table: `/tmp/winnow_inspect_tokens.json` (scratch; recompute
via the inspect endpoint — do not commit).

## T178 执行结果

Optional recovery path implemented (default policy unchanged — partial
coverage remains the posture; this builds the off-switch, not the switch):

- `scripts/rewindow_for_winnow_v1.py` — measures every candidate's winnow
  prefix via `POST /v1/winnow/inspect` (localhost-only, sequential), then
  re-windows records >8,000 prefix tokens by dropping the OLDEST
  non-protected entries first (protected = candidate pointer, final user
  turn, control/system entries incl. `/elided/N` sidecar pointers;
  `user_messages_in_order` preserved verbatim), re-measuring after each
  drop until ≤8,000 or the protected floor is reached. Fail-open:
  `meta.winnow_overflow: true` on unfittable records.
- `scripts/test_rewindow_for_winnow_v1.py` — 10 offline unit tests
  (synthetic states, injected measurer), all passing.

Run result on all 547 candidates (2026-09-28, sequential inspect):

- 242 records measured >8,000 prefix tokens: the 220 rejects (8,115–10,672)
  plus 22 previously-accepted records in the 8,003–8,096 margin band.
- After re-windowing: **219 of the 220 rejects fit ≤8,000** (1–17 segments
  dropped per record, median 3). Fitted set coverage: **546/547 (99.8%)**.
- 1 unfittable: `realctx:c8e82d6fe7a5:seg0017` — protected-floor window is
  still 8,161 tokens (9,725 measured); stays a winnow abstain by design.
- Output: `data/real_context_eval_v1/candidates_winnow_fit.jsonl` — same
  record_ids/order; `meta.winnow_rewindow` carries measured/final prefix
  tokens and the dropped-pointer list; candidate pointers verified to
  resolve in every emitted conversation.

Rescore readiness: 327 rows already scored. Rescoring the 219 fitted
rejects on winnow (~500 ms each ≈ 2 min; inspect pass ~6 s) is cheap and
worthwhile if near-full coverage is wanted — command:
`eval_systemone_backend_v1.py --url http://127.0.0.1:8091/v1/systemone
--data data/real_context_eval_v1/candidates_winnow_fit.jsonl` and merge by
`i`. NOT run here (recovery path remains opt-in).
