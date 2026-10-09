# Local tooling acceptance — 2026-09-19

User priority: fix and verify official DeepSeek V4.1 Worker routing and require local
NanoJev participation throughout development, testing, optimization, and deployment,
before resuming roadmap work. These prerequisites are delivered; B0 remains open.

## Official Worker

- Runtime: `/Users/markus/code/clis/deepseek-worker`; installed command
  `/Users/markus/.local/bin/deepseek-worker`.
- Direct API: `https://api.deepseek.com/v1`, SDK `@ai-sdk/deepseek`, model ID
  `deepseek-flash`. Official [model table](https://api-docs.deepseek.com/quick_start/pricing/)
  and [2026-09-10 release](https://api-docs.deepseek.com/news/news260910) identify this as
  **DeepSeek-V4.1-Flash**. The expired `...expires-on-0910` name is not the production route.
- Live official `/v1/models` returned `deepseek-flash` and `deepseek-v4-pro`.
  A minimal official completion returned model `deepseek-flash` and content `OK`.
  The guessed name `deepseek-v4.1-flash` returned HTTP 400; it is not an API ID.
- Every Worker execution verifies the official model-version mapping; failure stops it.
  Automatic routing contains only `deepseek-official`. Explicit CommandCode execution is
  rejected before any API call. The Codex main provider was not changed.
- Worker-owned process configuration overrides the official endpoint/model/SDK/credential;
  no credentials were printed or copied into this repository. Legacy bridge/native profiles
  remain diagnostic artifacts, not approved task routes.
- Fixed result observation: an already-completed result is accepted even if polled after
  its deadline, rather than misclassified as a timeout.

Live receipts:

| Job | Evidence | Outcome |
|---|---|---|
| `1789826655-28b36fb346e0` | Read a fresh file token with a real tool, 3.18 s | success, no writes, no fallback |
| `1789826954-4170b41d8452` | Independently ran 8 skill tests and wrote `probe_result.json`, 9.17 s | success; only declared file changed; no fallback |

Inspect with `deepseek-worker --json inspect --job-id JOB_ID`. These are bounded
functional checks, not a general reliability or model-quality benchmark. Earlier failed
attempts were retained, not relabelled as successes.

## NanoJev skill

- Source: `integrations/codex-skill/nanojev-local-decider`.
- Installed: `/Users/markus/.codex/skills/nanojev-local-decider`.
- All five maintained skill files matched byte-for-byte at acceptance. Automatic discovery
  is enabled. Project `AGENTS.md` requires lifecycle calls and observed-outcome feedback.
- Persistent service: `http://127.0.0.1:8876` (8765 was occupied); MPS FP32;
  `checkpoints/local_atomic_seed17/variants/local_atomic_seed17`.
- On-disk weight SHA-256:
  `2b06e4423861f47da7ca53ad9c23a26aebdbfe79de3a238b36c2954ae9c6abf0`.
  This is the file identity, not a claim that a path alone authenticates a running process.
- `lifecycle --stage development|testing|optimization|deployment` performs actual inference,
  records an event ID and explicitly returns `authorizes_execution: false`.
- Remote URLs, redirects, invalid probability distributions, missing answers, a mismatched
  checkpoint and reported remote model calls are rejected. Model startup is offline.
  Abstention clears actionable choice/value; scores remain available for analysis.
- Boolean, Noul, Choice and Score were exercised in one real request: four questions,
  seven candidate paths, one forward pass, zero autoregressive decoding and zero remote
  model calls. Event: `7857fc13-5305-4ba6-b2f3-1daccc783753`.

Lifecycle evidence (helper round-trip latency; different prompts, not a speed comparison):

| Stage | Event ID | Latency | Result |
|---|---|---:|---|
| development | `0c688ce0-1f25-4f38-9b56-e7b6d70763f7` | 93.300 ms | abstain |
| testing | `49ee99fa-0de0-49f1-8507-2f2da5237977` | 95.446 ms | abstain |
| optimization | `cb99ee03-27f5-43ab-80a6-45fa723f1798` | 60.783 ms | abstain |
| deployment | `93722ade-1e98-45a5-bf00-b9b3cd670bc7` | 48.225 ms | abstain |

All four calls were MPS-local with `network_model_calls: 0`; feedback recorded fallback
to independent evidence. The game-trained checkpoint has **not** demonstrated engineering
decision quality. Do not lower thresholds to conceal abstentions, report token/cost savings
from these calls, or let a score bypass a review, permission, test, or deployment gate.
Raw input logging stays off by default; the existing explicit debug opt-in remains sensitive.

## Verification and continuation

- Worker official-only tests: **9 passed** (`tests/test_official_only.py`). The Worker
  repository's complete legacy suite was not rerun; it includes historical third-party routing tests.
- Skill tests: **8 passed**, both source and installed copy; also independently run by Worker.
- Skill validators: installed NanoJev and DeepSeek Worker skills valid.
- NanoJev main suite: **380 tests, OK, 2 skipped** (378 executed), 25.906 s.
  One existing unclosed-file ResourceWarning appeared; not a failed assertion.
- Both repository diffs passed whitespace checks. No commit/push, live trade, financial
  training, active context filtering or global main-provider change was performed.

Next: resume T1–T5/B0 with lifecycle calls. `scripts/paper_trade_perp_v1.py` contains
unverified in-progress roadmap changes from the interrupted turn; its protocol helper,
frozen pins and new driver acceptance tests are not delivered yet. Existing suite success
does not certify those new CLI paths. Do not mark T1 complete until they are verified.
