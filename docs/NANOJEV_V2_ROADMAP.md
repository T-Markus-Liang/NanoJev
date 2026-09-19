# NanoJev V2 roadmap

NanoJev should evolve as a fast, calibrated decision layer rather than a smaller chat model. V2 has two primary product goals:

1. **Universal model integration:** place NanoJev in front of GPT, Claude, Kimi, OpenCode, local models, and other providers to identify irrelevant input segments before the main-model call, reducing input-token cost without losing instructions, evidence, or task success.
2. **Financial secondary-market decisions:** explore an independently reproducible training approach inspired by Reinforcement Learning for Calibrated Decisions (RLCD) for bounded trading decisions, with calibrated probabilities, explicit abstention, realistic costs, and millisecond-level latency as an acceptance target.

The financial track is the higher strategic priority. The universal integration track is the first reusable deployment surface and provides the filtering, telemetry, calibration, fallback, and low-latency infrastructure needed by both tracks.

Neither goal is a current capability claim. Token reduction, financial utility, calibration, and latency must be demonstrated on frozen, reproducible evaluations before release claims or live use.

The official Jev API is proprietary and its public latency and cost claims are workload-specific. NanoJev therefore uses explicit contracts, public cohorts, benchmark receipts, and acceptance gates instead of claiming architectural equivalence or reproducing marketing numbers.

## Current execution handoff

The [current progress and AI handoff plan](CURRENT_PROGRESS_AND_HANDOFF.md) is the execution entry point: it records worktree status, bounded work packages, acceptance evidence, and mandatory review checkpoints. The [review log](EXECUTION_REVIEW_LOG.md) holds the per-work-package verification record and the reviewer's findings. The [three-seed relevance result](CONTEXT_RELEVANCE_V1.md) is documented: training and receipts exist, independent implementation/evidence review remains pending, and **no candidate is promoted**. This does not close Phase 0 or authorize active filtering.

**Delivered since the last roadmap revision** (all independently re-verified by the reviewer):

| Package | Delivered | Evidence |
|---|---|---|
| P0 | Evidence closure and review log | frozen-artifact hashes MATCH; independent report rebuild is semantically identical |
| P1 / P1b | Financial data, rights, PIT, and experiment contract draft, **stopped at R1** | 14 sources surveyed; perpetual-only scope; validator projection key-set verified |
| P2 / P2b | Perpetual execution simulator plus paper-trading backtest | **80 tests**; conservation, determinism, funding monotonicity, liquidation and margin-sufficiency checks pass |
| A4 | Tool-history shadow fixtures | 11 fixtures, 13 tests, **11/11 semantics match through the real gateway** |
| G1 | Main-model workflow gateway | 42 tests; shadow, active, real-scorer, and `actual` token-accounting paths verified end to end |
| G2 | Reversible filtering | 65 restore tests; byte-identical round-trip through the real gateway; one reviewer-found contract violation fixed |
| E1 | Mac-local read-only measurement of the community references | laya runs on `mps:0`; readouts are diagnostics only, not a controlled comparison |
| W1 | **Real-data paper trading across four venues** | 6,554-record real PIT cohort accepted by the unmodified validator at exit 0; three venue receipts |

Full suite: **380 tests OK (2 skipped)**. Nothing was purchased; no broker, account, key, or order
was ever touched. `HEAD` remained at `be0303c` throughout the work; this revision is the first
commit of that work.

**User directive (2026-09-19): the local decision model must be integrated into the main-model workflow to increase speed and reduce token consumption.** The concrete missing deliverable is the G1 provider gateway (work package G1 in the handoff plan): a loopback gateway exposing OpenAI- and Anthropic-compatible pass-through endpoints, defaulting to byte-preserving shadow mode, failing open on every error path, with a documented measured-versus-estimated token accounting rule. Building and testing that gateway is authorized now. **Enabling active pruning in production is not**: it still requires every Track A acceptance gate below plus paired downstream-quality and net-token-cost evidence across at least three main-model families and a zero-deletion protected-segment stress suite. *(Status: G1 is delivered and verified — see the delivered table above; this paragraph records the directive, not current state.)*

**User scope decision (2026-09-19) for Track B: perpetual-contract crypto trading only, never spot**, across **Binance, Bybit, Aster, and Hyperliquid**, with RLCD training and a paper-trading backtest as explicit deliverables. Live capital remains out of scope. Consequences that must be visible in every plan: the spot-era draft (long-only, gross price-move label) must become two-sided with leverage, margin, funding, and liquidation semantics; the existing simulator refuses short sales and models no margin or funding, which is a hard blocker for perp backtesting *(resolved in P2b — the simulator now models shorting, margin, funding, and liquidation; 80 tests)*; and the previously recommended Binance Vision archive is CC BY-NC-SA 4.0 (non-commercial) with a §4.2 clause prohibiting live proprietary trading execution, so it can never support an execution phase and must be re-assessed per venue.

**User directive (2026-09-19): advance real-data simulated (paper) trading first — it is the cheaper path.** This reordered the financial track: the data/simulator/backtest plumbing was delivered end to end on **real** venue data before any RLCD training. See [Real-data paper trading V1](PAPER_TRADE_REAL_DATA_V1.md) and [Venue data licensing audit V1](VENUE_DATA_LICENSING_V1.md).

**User directive (2026-09-19): authorise Bybit/Aster/Hyperliquid as data sources and connect them.** All four venues are now connected. Two of them were blocked purely by **egress**, not by the venues: `api.bybit.com` is DNS-poisoned in this environment and `fapi.asterdex.com` refuses direct connections; routing through the local egress proxy reaches both. **This was a reviewer oversight** — the first pass tested direct connections only and wrongly concluded Aster was unreachable.

**User directive (2026-09-19): update progress and forward plan in the docs, and commit/sync important material to git.** This revision records the delivered work packages, the negative results, and the re-ordered plan below.

### The single most important V2 result so far is a negative one

Over real data the same reference strategy produced **four mutually contradictory** results:

| Run | Data | Net PnL |
|---|---|---:|
| A | Binance, before a 3-day index fix | −49,920.90 |
| B | Binance, after the fix | +21,173.77 |
| C | Bybit, same period and instruments | −77,919.27 |
| D | Aster, same period and instruments | +194,025.96 |

The cross-venue spread is **271,945 on 100,000 of initial capital — 2.7× the capital** — while the
venues' mark prices agree to within **1–2 basis points** (Aster vs Bybit mean absolute deviation
0.0199%, Aster vs Binance 0.0122%). Runs A and B differ by three days and one rejected order;
that single order flips the sign.

**Price-level differences are too small to explain the spread** — but that alone does not exclude
data issues: time alignment, units, funding handling, volume accounting, and missing measurements
have not all been audited to exclusion. What is established is that the located **candidate
mechanisms** are in the execution path, and three amplifiers are identified and must be fixed
before any financial number can carry meaning:

1. a **rejected or partially filled** order silently changes the trade size and timing, and the
   strategy then compounds from a different path (position *level* self-heals because
   `long`/`short` are target-position actions sized from the ledger's actual holdings — what does
   not heal is the size and timing of each trade, hence fees, funding, and PnL);
2. the **capacity policy interacts with venue-reported volume** (`fill ≤ 10% × bar volume`;
   Aster's median daily BTCUSDT bar volume is 9,959 vs Binance's 128,882), so the same strategy
   receives different position sizes per venue — this is the confirmed source of Aster's 27
   partial fills;
3. position sizing is **already path-independent** (quantity from initial cash and decision-day
   close only, verified in code) — an earlier draft of this document incorrectly claimed
   equity-compounding sizing; the record is corrected here and the property is locked by a test
   under B0-B.

This is why the financial track still has **no** utility, edge, or profitability claim, and why
Track B's first acceptance gate is now measurement integrity rather than model quality.

## Community references and adoption plan

Reviewed 2026-09-19. See [the pinned reference review](JEV_COMMUNITY_REFERENCES.md) for evidence, limits, and executable follow-up gates.

| Reference | Role in this roadmap | First deliverable | Adoption boundary |
|---|---|---|---|
| [tamaratran/fast-jev-compaction](https://github.com/tamaratran/fast-jev-compaction) | Track A: verbatim retention, paired tool-call/result pruning, bounded scoring requests | A4 tool-history shadow corpus and paired compression benchmark | Do not inherit its 0.5 keep threshold, character-based savings claim, or result-omission policy without validation |
| [hr98w/jev-visual](https://github.com/hr98w/jev-visual) | Track B and shared runtime: shared context, direct candidate scoring, independent-forward parity | B8 structured-versus-visual state ablation plus runtime parity specification | This is an educational visual inference project, not a financial strategy, RLCD trainer, or proof of millisecond trading |
| [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) | Shared runtime and Track B recipe: 421M ModernBERT encoder decision engine, proper-scoring GRPO-style training, script-detection routing | Read-only local install on this Mac laptop (MPS/CPU latency and memory receipt) plus zero-shot probe on frozen workflow samples | T4 latency and Jev comparison are non-local/third-party numbers; base checkpoints are near-chance zero-shot by its own disclosure; 512/1024-token context does not cover Track A long-context loads; its GRPO-style recipe is an RLCD candidate, not the definition |
| [kshetrajna12/reflex](https://github.com/kshetrajna12/reflex) | Shared runtime and model quality: Qwen3.5 direct-logits readout without trained heads, packed/batched branch isolation with numerical-equivalence tests, calibration protocol, TypeSafe-compatible API shape | WebGPU demo verification on this Mac (Safari 18+), then a reviewed direct-logits-versus-trained-heads ablation spec on frozen cohorts | Direct-logits caps Choice at 26 candidates versus our 255 contract; MMLU/Jev ECE figures are cross-source comparisons; supervised proper-scoring LoRA is not RLCD; GB10/WebGPU timings do not transfer to our MPS service |

Financial calibrated decisions remain the highest strategic priority. The already-running Track A relevance experiment is closed and reported (no promotion); advance financial data/simulator foundations next. Visual demos, plugin packaging, and the laya/reflex Mac-local exploratory checks must not displace that work. No reference changes the frozen training protocol or authorizes provider reconfiguration, external transcript uploads, active pruning, or trading.

## Ecosystem update (2026-09-19, second sweep)

A broader ecosystem sweep was provided by the project owner. Every item is classified by evidence level; **nothing here changes acceptance gates**. Star counts are GitHub API snapshots taken 2026-09-19.

**Verified to exist** (GitHub API, 2026-09-19 snapshot):

| Project | Stars | Relevance to this roadmap |
|---|---:|---|
| [tamaratran/fast-jev-compaction](https://github.com/tamaratran/fast-jev-compaction) | 3,729 | Largest community Jev project; Claude Code plugin replacing summary-based compaction with per-item Jev decisions and verbatim retention — the same semantics as our A2/A4. Its README states token sizes are **character-count estimates without a tokenizer**, with a 25k estimated-token state cap and a 30k request cap under Jev's 32k request ceiling. The reported "156,000 → 62,000 tokens, 78% → 31%" figure is **not in the README** and remains unverified. |
| [vercel-labs/json-render](https://github.com/vercel-labs/json-render) | 16,681 | Generative UI framework; its compose path uses Jev to pick components and actions |
| [vercel-labs/fx](https://github.com/vercel-labs/fx) | 3,064 | Unix-like coding agent in Zig; ships a `typesafe_permission_reviewer` |
| [vercel-labs/ai-python](https://github.com/vercel-labs/ai-python) | 184 | Official Python SDK whose evaluation op supports Jev |
| [cline/plugins](https://github.com/cline/plugins) | 23 | Official curated plugins; includes `jev-browser` |
| [jaredpalmer/kev](https://github.com/jaredpalmer/kev) | 287 | Minimal Jev-like model on Qwen2.5-0.5B, trainable and runnable on a MacBook — relevant to the Mac-training ladder below |
| [bespokelabsai/nimble](https://github.com/bespokelabsai/nimble) | 147 | Open recipe + open weights (Apache-2.0 adapter): Qwen3.5-9B LoRA with **contrastive data curation** (change one fact so the answer flips). Directly relevant to our Track A zero-drops blocker; full review in the [pinned references](JEV_COMMUNITY_REFERENCES.md) |
| [achimala/jevinci](https://github.com/achimala/jevinci) | 19 | Creative: parallel per-pixel color decisions; confidence maps to brush width |
| [luiginotmario/postgres-Jev](https://github.com/luiginotmario/postgres-Jev) | 0 | Natural-language PostgreSQL predicates (`WHERE jev(...)`-style) |
| [sosopop/jev_stock](https://github.com/sosopop/jev_stock) | 6 | Experimental short-term stock-direction forecasting with a first-trading-day backtest script |
| [rorshopping/jev-on-a-laptop](https://github.com/rorshopping/jev-on-a-laptop) | 14 | Unofficial Jev-style parallel typed-decisions study |
| [TianyuCodings/NanoJev](https://github.com/TianyuCodings/NanoJev) (upstream) | 626 | The upstream project this fork tracks; repo created 2026-09-17 — the sweep's "72 hours" framing is roughly consistent with that timeline |

**Provided but not verified** (recorded as unverified, not as false): Atomic (structured-output provider claim; no matching repo found), Jevinik (finance; search returns unrelated repos), the Monad on-chain trading bot (claimed real orders every 300 ms block; not located), the DuckDB extension ("1,000 rows ≈ 10 s"; not located), the cost case studies ($0.09 per 724 ad breakdowns; $2.17 for 3M replay events → 132 rage-click clusters and 213 fix PRs; $0.19 for 384 news items versus a same-day frontier-model comparison; no primary sources checked), the 2,276-like objection thread, Decider-2B and System-One 4B (no matching repos found), the Jev-compatible public API (Qwen3.6-35B-A3B), and the "Jev-ify any HF model" library.

### What the sweep changes — absorbed as tasks, never as capabilities

1. **The compression controversy becomes a formal experiment arm (new A5).** The largest community project is filtering-based (score each tool call/result, delete the unneeded, keep survivors verbatim) — the same semantics as our A2/A4. The loudest public objection argues compaction is reconstruction, not filtering. Both positions are unproven. A5 runs a frozen, paired comparison across verbatim-retain, deterministic safe dedup, relevance filtering, abstractive summary, and retrieval-rebuild arms, measuring dependency retention, evidence fidelity, downstream success, restore cost, and end-to-end cost with paired confidence intervals per task family. Fail-open stays "forward the original bytes"; a summary fallback is a different policy and must never be substituted silently. Retained text being verbatim does not imply deletion is lossless, and byte-reversible restore does not imply reduced-context inference is equivalent.
2. **Cost becomes a first-class test standard.** The ecosystem's most persuasive artifacts are cost receipts (cents per task). Phase 0 gains a rule: every evaluation reports cost per decision, per classified row, or per defect found, with full denominators (eligible versus all requests), including scorer, cache-discount, rebuild, retry, and tool-re-execution costs. Our paired estimate-versus-actual token accounting is the foundation; character-count estimates (as used by fast-jev-compaction) do not qualify.
3. **"Judgment as a code primitive" gets a capability matrix — C-track candidates, not commitments.** Three surfaces appeared: structured-output providers sharing a resolver, database row predicates (Postgres/DuckDB), and SDK evaluation operators. Gates: protocol, schema, and semantic compatibility are verified separately per provider/model/version; a database predicate must expose unknown/abstain, budgets, snapshot identity, batch and failure semantics, and must never act as a permission, row-level security, or deterministic constraint; batch throughput is not per-decision latency.
4. **Finance signals are safety-requirement sources only.** Community projects claim real-money execution (a Monad bot placing real orders every 300 ms block). That changes nothing here: real market data plus simulated fills remains the ceiling, and any live execution needs a new, separate authorization. `jev_stock`'s first-trading-day backtest pattern matches our W1 shape and reinforces the per-venue/per-regime breakdown standard.
5. **Mac-trainable small models get a five-level acceptance ladder.** kev (0.5B, trains and runs on a MacBook) and our own training pipeline both target this. Levels: loads → infers → backpropagates → full training run reproducible (checkpoint save/reload identity, peak unified memory, wall time, no hidden cloud/CUDA dependency) → meets frozen quality/cost targets. "Loads" is not "trains". A small-model field (Laya 421M, kev 0.5B, NanoJev 0.6B, plus unverified Decider-2B / Reflex / System-One 4B) is forming; our differentiator — complete probability distributions with zero output-token decoding under frozen evaluation — must be demonstrated, not asserted.

## Baseline: V1.0

Current reference checkpoint: `variants/local_atomic_seed17`.

| Gate | Current result | Source |
|---|---:|---|
| local-maze test accuracy | 77.8409% | 176 public questions |
| local-maze OOD accuracy | 76.5625% | 64 public questions |
| test ECE | 0.0851 | 10-bin top-label ECE |
| OOD ECE | 0.0734 | 10-bin top-label ECE |
| warm MPS latency | about 65 ms | 3 questions, 8 candidate paths |
| 255-candidate Choice | about 1.28 s | M5 Max, FP32 MPS |
| invalid probability outputs | 0 observed | full local smoke and public evaluation |

The baseline report must be regenerated before comparing a new checkpoint. See [`benchmark_nanojev_v2.py`](../scripts/benchmark_nanojev_v2.py).
The current seed-17 receipt, including all paired sample keys and provenance hashes, is [`nanojev_v2_baseline_seed17.json`](../results/nanojev_v2_baseline_seed17.json). Candidate reports are compared with [`compare_nanojev_v2.py`](../scripts/compare_nanojev_v2.py); a single-seed comparison deliberately reports no training-seed confidence interval.

The [workflow baseline and challenge report](WORKFLOW_V2_BASELINE.md) adds 816 original questions and a 3,264-question paired robustness suite. All variants share 272 source groups. Catalog Choice test accuracy drops from 100% to 54.17% under explicitly irrelevant archived context; smart-home Boolean test accuracy is 43.75%. These are measured blockers for active context removal, not evidence of deployment readiness. The checkpoint was not retrained or selected using these results.

## Phase 0: measurement and contract gates

Status: **in progress**

- [x] Freeze public test and OOD cohorts.
- [x] Record cold-load and warm-request latency separately.
- [x] Test candidate counts through 255.
- [x] Test candidate permutation invariance.
- [x] Add one machine-readable benchmark command.
- [x] Add deterministic source-group confidence intervals, multi-seed aggregation, and paired checkpoint comparison tooling.
- [x] Add a benchmark manifest containing dataset, checkpoint, dependency, and hardware hashes.
- [x] Add the broad workflow baseline covering Boolean, Choice, Score, rule-based tasks, and known probability distributions, with committed metric receipts and reproduction commands.
- [x] Run a four-variant, 3,264-question robustness challenge with paired source-group intervals and complete failure reporting.
- [x] Add a reversible-filtering round-trip property with a content-free restore manifest, verified byte-identical through the real gateway.
- [x] Verify the gateway's token accounting distinguishes `estimate` from provider-paired `actual` savings.
- [ ] Add human-reviewed challenges and genuinely held-out task families beyond synthetic rendering variants.
- [ ] Evaluate the first V2.1 candidate and its baseline with at least three matched training seeds.
- [ ] **[NEW] Add a mandatory sensitivity/spread report** to every financial result so no point estimate can be published alone (see B0).
- [ ] **[NEW] Cost-native reporting**: every evaluation reports cost per decision, per classified row, or per defect found, with full denominators (eligible versus all requests) and including scorer, cache-discount, rebuild, retry, and tool-re-execution costs. Token counts must be provider-reported or tokenizer-based; character-count estimates do not qualify.
- [ ] **[NEW] External-claim evidence ledger**: every ecosystem number used in a document carries an evidence level (author claim / source audit / local reproduction / paired comparison), source URL, snapshot timestamp, and comparison scope.

Exit gate: every model change reports accuracy, NLL, Brier, ECE, invalid-output count, abstention, cold start, warm p50/p95/p99, throughput, and peak memory on the same frozen cohort. Product-specific evaluations add token savings or financial utility without replacing these shared metrics.

## Track A: universal context gate

Priority: **primary deployment goal**

NanoJev will run before a main model as a provider-neutral context gate. It will score structured input segments, retain the information required to complete the task, and remove only segments that are both eligible and confidently irrelevant. This is not lossy prompt summarization and it must not rewrite the user's meaning.

### A1. Context contract and safety invariants

- Segment inputs by provenance and role: system/developer instructions, user intent, conversation history, tool schemas, tool results, cited evidence, files, retrieved memories, and optional context.
- Always preserve system and developer instructions, safety boundaries, current user intent, tool schemas required by the task, cited evidence, dependency context, and segments explicitly pinned by the caller.
- Treat credentials, authorization rules, data-loss constraints, output contracts, and unresolved user corrections as protected context.
- Never silently drop an uncertain segment. Low confidence, unsupported input, parser failure, distribution shift, or conflicting evidence must fail open to the unfiltered request.
- Produce a reversible decision receipt containing segment IDs or hashes, retained/dropped status, reason codes, confidence, policy version, model version, latency, and token counts. Raw sensitive text must remain excluded from normal telemetry.
- Keep filtering independent from provider authentication and routing so the same gate can serve GPT, Claude, Kimi, OpenCode, local models, and future providers.

### A2. Shadow mode and adapters

Status: **shadow library, loopback adapter, main-model forwarding gateway, and reversible filtering are implemented and independently verified**; not globally installed, and active filtering stays **disabled by default**. See [Context shadow V1](CONTEXT_SHADOW_V1.md), [Main-model gateway V1](MAIN_MODEL_GATEWAY_V1.md), and [Reversible filtering V1](REVERSIBLE_FILTERING_V1.md).

Delivered and verified at five levels — library tests (42), CLI entry point, real-scorer end to end, process-level `active` reduction, and the `actual` token-accounting path:

- **Shadow** forwards the original bytes, passes opaque credentials through untouched, leaks no `x-nanojev-*` header upstream, and writes a content-free receipt.
- **Active** reduction works end to end and preserves system instructions, user intent, and protected segments.
- **Token accounting is honest by construction.** Savings are reported as `estimate` unless the caller supplies a provider-reported paired baseline; with one, the claim becomes `actual` and equals `baseline − provider_prompt_tokens` (verified: 412 − 300 = 112). A reduction that was genuinely sent still refuses to claim `actual` without that baseline.
- **Reversible filtering** returns a content-free restore manifest in `x-nanojev-restore-manifest`. Independent verification confirms byte-identical round-trip restoration through the real gateway and rejects tampering (`removed_segment_hash_mismatch`).
- **A reviewer-found contract violation was fixed**: the gateway could forward a reduction the caller could not restore, because its guard checked only that the manifest *built*, not that restoration *succeeded*. The gateway now performs a real round-trip against the caller's own bytes before sending, and fails open on any mismatch.

Two measured limits now bound what this can claim:

- **`MAX_SCORED = 32`.** Requests with more than 32 scorable candidates bypass entirely with `scoring_budget_exceeded` and zero reduction. Long coding-agent histories — the case the gate exists for — are the most likely to hit this. Serviceability of long contexts is therefore an **open design question**, not a solved one.
- **Real-checkpoint proposals remain zero**, so actual token savings are still zero. Every Track A gate below is unmet.

Also open: main-model quality comparisons, broader provenance adapters, calibrated threshold fitting on calibration data, and independent challenge families.

- Build a canonical request envelope and adapters for the major chat, coding-agent, and tool-use message formats.
- Start in shadow mode: score and log proposed removals while sending the original unfiltered request to the main model.
- Replay identical tasks with filtered and unfiltered context against multiple model families, using fixed seeds or deterministic settings where available.
- Add explicit bypasses for unsupported modalities, opaque encrypted payloads, exact-transcription tasks, legal or policy text requiring full retention, and requests whose dependency graph cannot be established safely.
- Route low-confidence cases to the full request and expose the fallback reason to the caller.

### A3. Token-efficiency evaluation

Measure token reduction and downstream quality together. A reduction is invalid if it hides a regression in task success, safety, citation fidelity, tool selection, code correctness, or user instruction following.

Evaluation cohorts must include:

- long coding-agent histories with tool traces and repeated file content
- multi-document retrieval with distractors and cited evidence
- customer-support and operational workflows with policy constraints
- multilingual requests and provider-specific message formats
- adversarial dependency cases where an early detail changes a late answer
- unsupported tasks that must bypass filtering

### A4. Tool-history compaction reference experiment

Status: **implemented and independently verified, including through the gateway.** 11 deterministic fixtures (8 scored + 3 bypass), 13 unit tests, and a build script; see [Tool history shadow V1](TOOL_HISTORY_SHADOW_V1.md).

The A4 fixtures were built against the shadow core while the gateway was built separately, so a reviewer closed the gap by replaying **all 11 fixtures through the real gateway process** in `active` mode with a deliberately maximal always-drop scorer:

| Check | Result |
|---|---|
| gate `status` matches each fixture's declaration | **11/11** |
| gate `reason` matches | **11/11** |
| retain semantics match | **11/11** |
| receipts containing leak tokens | **0** |

The three tool-linkage ambiguity cases (`duplicate_ids`, `orphan_result`, `pending_call`) **bypass wholesale** at the gateway layer — no attribution guessing — which is the core safety property A4 was designed for. Auditing what was actually dropped showed only assistant drafts or restatements of a tool value were removed, with the authoritative tool result, system instructions, and user intent always retained; the leak tokens were present in the forwarded bytes and absent from the receipts.

- Keep first/current intent, recent context, pinned evidence, and pending calls protected.
- Retain retained text verbatim. Never imply that deletion is lossless merely because the surviving text was not rewritten.
- Unsupported or ambiguous histories must pass through unchanged.
- Measure provider-tokenizer or API-reported input counts, gate overhead, tool re-execution, downstream failures, and total cost. Character reduction alone is not token or money savings.
- Ship only shadow receipts until Track A acceptance gates pass. Do not tune the 0.99 gate using held-out outcomes or copy upstream's 0.5 threshold.

### A5. Filter-versus-rebuild comparison arm

Status: **planned; added 2026-09-19 in response to the ecosystem sweep.** The compression debate (filtering versus reconstruction) is the community's most contested question and both positions are unproven. This arm makes it a frozen, paired experiment rather than a mailing-list argument.

- Arms: unfiltered control, deterministic safe deduplication, relevance filtering (our A2/A4 semantics), abstractive summary, and retrieval-rebuild. All arms see the same frozen tool-history and long-context fixtures.
- Measure: dependency-pair retention, evidence fidelity, protected-segment deletion (must be zero), downstream task success, restore cost, and **end-to-end cost per request** including scorer, rebuild, retry, and tool re-execution — provider-reported or tokenizer-based tokens only, never character estimates.
- Report paired confidence intervals per task family; a win on the aggregate that loses on a protected family is a fail.
- Fail-open is "forward the original bytes" in every arm. A summary fallback is a separate policy and is never substituted silently.
- No arm is adopted from this experiment alone; adoption still requires the Track A acceptance gates below.

### Track A acceptance gates

Targets are release gates, not current results:

- at least 30% median main-model input-token reduction on eligible frozen workloads
- no more than 0.5 percentage-point task-success regression overall or within any protected task family
- zero critical-instruction, safety-boundary, required-tool-schema, or cited-evidence deletions in the stress suite
- 100% fail-open behavior for parser errors, unsupported tasks, and decisions below the confidence threshold
- complete, replayable decision receipts for every active filtering request
- filtering p95 low enough that end-to-end latency and total cost still improve; report gate latency separately from provider latency
- successful evaluation against at least three materially different main-model families before enabling active filtering by default

## Track B: financial secondary-market RLCD research

Priority: **highest strategic priority**

The initial target is a bounded decision engine, not free-form market commentary and not an autonomous live-trading system. The model should emit complete probability distributions and an abstain/no-trade decision for clearly defined horizons and market states. Per the user scope decision recorded above, the instrument class is **perpetual contracts only** (Binance, Bybit, Aster, Hyperliquid), so the action space is two-sided and the state contract must carry margin, leverage, funding, and liquidation state; spot is out of scope.

### B0. Measurement integrity — now the first gate

Status: **NOT met; blocking every other financial claim.**

The real-data result recorded at the top of this document shows a 2.7×-of-capital spread across
three venues whose prices agree to 1–2 bps. Until that is fixed, no financial number — including
any future RLCD result — can be distinguished from path noise. This gate precedes B1–B8.

Required before any further financial measurement:

- **A fill divergence must be an error, not a silent state change.** When an order is rejected,
  partially filled, or expires, the harness must either raise or explicitly reconcile the
  intended position with the actual one. Continuing from an assumed position is forbidden.
- **Position sizing must be declared and path-independent, and locked by test.** The driver
  already sizes from initial cash and decision-day close only (verified; an earlier draft here
  claimed equity-compounding — that claim was wrong and is corrected). B0-B adds the locking
  test and a documented `fixed_notional` mode; an equity-compounding variant would require
  interleaved decide/replay and is deferred to B5.
- **Capacity and cost parameters must not be coupled to venue-reported volume** unless that
  coupling is the object of study, because it makes cross-venue comparison meaningless.
- **A sensitivity report is mandatory** for every financial result: the same run under
  perturbed seeds, day-sets, and venues, reported together, with the spread stated. A single
  number with no spread must not be presented as a result.
- **Determinism is not robustness.** Byte-identical re-runs were verified and are necessary but
  insufficient; the receipt must state both.

Exit gate: the same strategy on the same period across at least three venues produces results
whose spread is small enough to be interpretable, **or** the harness explicitly reports that it
cannot and declines to emit a headline number.

### B1. Decision and state contract

Status: **PIT validator, perpetual execution simulator, real multi-venue data, and an end-to-end paper backtest are implemented; the state/action contract is still incomplete.**

- [Financial PIT V1](FINANCIAL_PIT_V1.md): 12,000 synthetic records, three walk-forward folds, 1,000 rejected timing mutations, 1,000 label-isolation checks.
- [Financial data plan V1](FINANCIAL_DATA_PLAN_V1.md) + [venue licensing audit V1](VENUE_DATA_LICENSING_V1.md): 14 sources surveyed, perpetual-only scope, four venues, per-venue licence verdicts.
- [Financial simulator V1](FINANCIAL_SIMULATOR_V1.md): perpetual layer with margin sufficiency, funding, and liquidation; 80 tests; **71 policy parameters explicitly provisional pending R1**.
- [Real-data paper trading V1](PAPER_TRADE_REAL_DATA_V1.md): **real** venue data end to end.

**Real data is now in place.** 925 public archive/API files were fetched with per-file provenance and SHA-256 in three manifests, and **nothing was purchased**; all sources are public read-only endpoints, no key, no account, no order.

| Venue | Files | Bytes |
|---|---:|---:|
| Binance (bulk archive) | 880 | 1,185,364 |
| Bybit (v5 API via proxy) | 30 | 3,734,654 |
| Hyperliquid (`/info`) | 15 | 14,131,517 |
| Aster (v3 API via proxy) | 35 | 4,032,704 |

Build results: a **6,554-record real PIT cohort** (45.03% positive base rate) that the **unmodified** `financial_pit_v1.py` accepts at exit 0 against the frozen protocol core, with all four phases nonempty in all three folds (test 905 / 905 / 465, dev and calibration 120 each).

**What is still missing from B1:**

- The cohort is a **PILOT**, not the R1 cohort. The R1 feature allowlist is a **closed 12-feature set**, and three features cannot be built: `open_interest_level` and `open_interest_log_change_1d` need Binance's daily metrics archive (obtainable), while **`liquidation_intensity_1d` is published historically by no venue in the set** (not obtainable). The pilot therefore declares a documented 9-feature subset and must not be presented as the R1 cohort.
- **No order-book or microstructure features**, no position/inventory state, no borrow or capacity state. The state contract below is still aspirational.
- **No real-data receipt for the risk gate**, and the execution-cost parameters remain declared guesses (the archive has no order book, so bid/ask equal the mark close and all spread cost is one parameter).

Start with finite actions whose outcomes can be labeled and simulated:

- buy/hold/sell or long/flat/short
- place order/pass
- position-size bucket
- execution venue and order type
- risk gate: allow, reduce, block, or exit

The state contract should support point-in-time versions of:

- order-book and market-microstructure features
- price, volume, volatility, liquidity, and cross-sectional features
- event and catalyst state available at the decision timestamp
- current position, inventory, exposure, limits, and outstanding orders
- fees, spread, estimated slippage, latency, borrow cost, and capacity constraints

Every feature needs a source timestamp, availability timestamp, transformation version, and lineage receipt. Features unavailable at decision time are forbidden even if they later appear in historical data.

### B2. Calibrated targets

Train and evaluate explicit event probabilities separately from action policy:

- forward-return threshold events over fixed horizons
- fill and partial-fill probabilities
- adverse-selection probability after execution
- drawdown, stop, liquidation, and other risk-event probabilities
- market-regime probabilities

An action distribution is not automatically a calibrated success probability. Event prediction, action selection, execution quality, and portfolio utility must have separate metrics and receipts.

### B3. Baselines before RLCD

- Establish deterministic and supervised probabilistic baselines first.
- Compare exact cross-entropy and exact Brier objectives under matched data, seeds, initialization, and training budgets.
- Establish simple non-neural references such as logistic regression, gradient boosting, and rule-based risk gates where applicable.
- Freeze training, development, calibration, test, instrument-holdout, and regime-holdout protocols before RL experiments.
- Calibrate only on the calibration split; never tune labels, thresholds, execution rules, or position sizing on test periods.

### B4. RLCD-like experiments

- Treat the existing paired categorical Brier policy-gradient estimator as one research candidate, not as the definition of RLCD.
- Compare estimator variance, bias checks, sample efficiency, calibration, selective risk, expected utility, and stability against exact proper-loss baselines.
- Separate rewards for prediction quality, execution quality, risk, and net utility so one aggregate reward cannot hide a failure mode.
- Use counterfactual or simulator-derived rewards only when their assumptions and coverage are explicitly recorded.
- Do not describe an experiment as RLCD merely because reinforcement learning or a policy gradient is present. The approach must optimize and verify calibrated decisions under a documented objective.

### B5. Financial validation protocol

- Use point-in-time datasets with survivorship-bias controls and delisted/inactive instruments where relevant.
- Use purged walk-forward evaluation with an embargo around overlapping labels and dependent samples.
- Hold out complete regimes, time periods, and instruments; report each separately rather than only an aggregate score.
- Include realistic fees, bid/ask spread, slippage, market impact, latency, rejected orders, partial fills, borrow constraints, and capacity.
- Prohibit test-period threshold tuning, feature selection, early stopping, and strategy selection.
- Report gross and net results, turnover, drawdown, tail loss, exposure, calibration, abstention coverage, and uncertainty intervals.
- Compare against passive, random, simple-rule, and deterministic NanoJev baselines under the same execution simulator.

### B6. Risk and deployment controls

- Make abstain/no-trade a first-class action and measure selective risk as coverage changes.
- Enforce position, exposure, inventory, loss, turnover, and order-rate limits outside the model.
- Provide a deterministic kill switch and stale-data, clock-drift, and feed-integrity gates.
- Require offline replay, then shadow decisions, then paper trading. Live capital remains out of scope until all paper/shadow gates pass and a separate human approval process is defined.
- Preserve a full decision receipt for post-trade reconstruction without storing secrets in ordinary telemetry.

### B7. Latency tiers

"Millisecond-level" is an acceptance target, not a present claim. Measure three scopes separately:

1. model compute on a prepared bounded tensor
2. feature preparation plus model compute
3. end-to-end paper decision from timestamped market input to validated action

Optimize only after correctness and leakage controls are in place. Candidate techniques include compact feature encoders, fixed-shape batches, preallocated buffers, persistent services, quantization, compiled kernels, prefix sharing, and asynchronous feature updates.

### B8. Shared scoring and optional visual financial states

Status: **planned; inference reference reviewed, financial transfer unverified**.

- Use `jev-visual` as a systems reference for shared state prefill, batched bounded candidates, selective output projection, and comparison with independent full forwards. Its normalized candidate scores are not calibrated financial event probabilities.
- Start with timestamped structured market features. Test chart images only as an optional ablation against the same source observations, instruments, horizons, and chronological splits; do not make a vision model a prerequisite for financial research.
- Render each chart strictly from point-in-time data with fixed trailing windows, axis rules, and transformations. No future candles, future-dependent scaling, revised indicators, or post-event annotations. Rendering and image encoding count toward end-to-end latency.
- Compare structured-only, image-only, and combined states using identical labels and budgets. Require incremental held-out calibration/utility or a justified operational benefit before retaining visual complexity.
- Preserve an independent scoring path; test cache isolation, candidate permutations, padding, mixed lengths, overlapping token sequences, multilingual candidates, and recurrent state where applicable. Architecture-specific cache code is not portable by assumption.
- Do not transplant the reported M4 visual timings to NanoJev or financial workloads. Record model compute, feature/image preparation, queueing, and complete validated paper-decision latency separately.
- Keep CE/exact-Brier baselines ahead of RLCD-like experiments. This reference contributes inference engineering, not evidence of an RLCD training reproduction.

### Track B initial acceptance gates

Targets are provisional and must be revisited after the first frozen dataset and simulator:

- **[NEW, first] measurement integrity (B0)**: cross-venue and cross-perturbation spread of the same strategy is reported and small enough to interpret, or the harness declines to emit a headline number
- warm local model compute p50 below 5 ms and p99 below 20 ms for bounded state/candidate workloads on declared hardware
- end-to-end paper-decision p99 below 50 ms, with feature freshness and validation included
- better calibration and selective-risk curves than the deterministic baseline on frozen holdouts
- positive net performance after realistic costs across multiple purged walk-forward periods, with uncertainty intervals and no single-period dependency
- stable behavior across instrument and regime holdouts, with explicit abstention under distribution shift
- zero live trading until offline, shadow, and paper-trading gates pass

### Data-rights gates

- **[NEW] Every venue used must have a read verdict recorded** before its data drives a result. Read so far: **Binance** (CC BY-NC-SA 4.0, research permitted §4.1, live execution prohibited §4.2) and **Aster** (Terms of 2026-02-25; §6.1(b) requires prior written consent to download their material and §6.2(e) requires express permission for automated access, with **no research carve-out** — the project's own authorisation is not the venue's permission). **Not read: Bybit** (JS SPA; the headless browser cannot use the egress proxy) and **Hyperliquid** (no terms page exists in its public docs; the main site returns 403).
- Aster is the only **read, concrete, unresolved** prohibition. Closing it requires written permission from Aster, an authenticated API key under whatever terms accompany it, or an explicit recorded decision to accept the risk.
- Unread venues stay labelled **unverified** in every artifact; they must never be described as cleared.
- Live execution is a separate question from data rights and is blocked for Binance by §4.2 independently.

## Shared model-quality program

The existing V2.1 decision-success work supports both primary tracks. It is no longer an independent product priority.

- Build a balanced task matrix across Boolean/Noul, Choice, and Score instead of optimizing only local-maze Boolean questions.
- Add hard negatives: near-identical states, contradictory evidence, distractor candidates, overlapping vocabulary, long irrelevant context, and critical dependency needles.
- Separate semantic paraphrases from source-group duplicates so OOD measures transfer rather than memorization.
- Preserve exact probability targets where available; keep deterministic labels, observed outcomes, and teacher distributions as separate target kinds.
- Add human-reviewed challenge sets that cannot be generated from training templates.
- Compare shared scalar heads, set-attention heads, and candidate-conditioned cross-attention under matched seeds.
- Train with proper losses appropriate to each target and fit temperature only on calibration data.
- Expose confidence and explicit abstention so uncertain cases can fall back to full context, a stronger model, a deterministic risk gate, or human review.
- Measure per-task-family and per-candidate-count performance; reject aggregate gains that hide protected-family regressions.

Initial shared targets remain:

- local-maze test accuracy at least 82%, with no more than 0.5 percentage-point regression on any existing split
- local-maze OOD accuracy at least 80%
- calibration ECE below 0.08 on the fixed calibration protocol
- zero schema or probability-sum failures across the stress suite
- challenge-set accuracy reported separately, with no hidden tuning

## Shared runtime and efficiency program

- Implement prefix/tree sharing for states and question prefixes; verify numerical equivalence against repeated-path inference.
- Add tokenizer and encoded-prefix caches for persistent services, with bounded memory and request-isolation tests.
- Benchmark PyTorch SDPA, MPS kernels, CUDA BF16, compiled execution, and target-specific runtimes as separate profiles.
- Evaluate quantization and lower-memory checkpoint formats only after comparing accuracy and probability calibration.
- Add controlled microbatching and concurrency tests; distinguish queueing, feature preparation, model compute, and network latency.
- Record cold start, warm p50/p95/p99, candidate paths per second, questions per second, memory, and energy where available.
- Keep numerical equivalence or task-quality checks for every optimized execution path.

The previous M5 Max targets remain useful for the general runtime, but they do not substitute for Track A end-to-end token economics or Track B market-decision latency.

## Contract, SDK, and ecosystem

- Support structured instructions and candidate descriptions with a versioned canonical serializer.
- Add explicit `noul` compatibility with Jev's native naming while preserving the local `boolean` alias.
- Return confidence, score legend, calibration status, model version, abstention reason, and decision receipt metadata.
- Add richer candidate types and nullable/structured fields without ambiguous string coercion.
- Add dependent multi-stage workflows as separate requests with explicit state transitions.
- Publish a versioned Python/HTTP SDK and provider-neutral adapters with stable schemas.
- Provide OpenAI-style structured decision endpoints without pretending to be a text-generation API.
- Publish model registry metadata, checkpoint hashes, benchmark receipts, and reproducible release bundles.
- Compare NanoJev and official Jev on identical frozen requests when official API access is available.
- Publish failure analyses, not only aggregate scores.

## Research and safety rules

- Never mix training, development, calibration, test, OOD, regime-holdout, or instrument-holdout data.
- Never use future or revised information in a point-in-time financial feature.
- Never compare a local warm request with an end-to-end remote request without reporting both scopes.
- Never call a recorded Jev trajectory an independent fresh API measurement.
- Never claim token savings without paired downstream-quality results.
- Never claim profitable trading from gross returns, one period, one instrument set, or a tuned test period.
- Never present a financial point estimate without its cross-venue and cross-perturbation spread; a bare number from one data set is not a result.
- Never treat determinism as robustness. A byte-identical re-run of a wrong-sign number is still a wrong-sign number.
- Never let a rejected, partial, or expired fill pass silently as an assumed position.
- Never describe a data source as licensed or cleared when its terms were not read; "unread" and "unverified" must appear in every artifact that uses it.
- Never treat the project owner's authorisation as the venue's permission where a clause requires the venue's consent.
- Never connect a data source whose terms prohibit the access method without recording the conflict explicitly.
- Never enable active context removal without fail-open fallback and protected-segment tests.
- Never treat an ecosystem direction as a delivered capability, a star count as product evidence, or a passing test as real-world validity.
- Never treat character-count reduction as token savings, or token savings as net bill savings.
- Never treat a normalized probability as calibrated, calibration as trading profitability, or real-data paper trading as live trading.
- Never treat schema validity as semantic correctness, API-shape compatibility as architectural equivalence, or batch throughput divided by item count as single-decision latency.
- Never treat "loads and runs on this Mac" as "trains on this Mac"; the five-level ladder (load, infer, backpropagate, reproducible full run, quality/cost targets) applies.
- Never call a third-party recipe "RLCD" equivalence because its author named it that; it is a candidate to be compared against our frozen baselines.
- Never quote an external performance, cost, accuracy, or adoption number without its evidence level, source URL, snapshot time, and comparison scope.
- Never enable live trading as part of model research or benchmark automation.
- Every speed optimization needs a numerical-equivalence or task-quality report.
- Every capability extension needs a versioned input/output contract and a migration test.

## Current development slice

Re-ordered 2026-09-19 after the real-data results, and rewritten as an execution task board.
The governing principle: **do not build a model on top of a measurement that cannot yet be
trusted.** Effort scale: S < half a day, M = days, L = a week or more. Every task lists the
exact files it touches and the command that proves it done.

### Task board

| # | Task | Track | Effort | Depends on | Status |
|---|---|---|---|---|---|
| T1 | B0-0 frozen replay baseline | B | S | — | 🟡 protocol file created; driver wiring pending |
| T2 | B0-A order/fill divergence state machine | B | M | T1 | ⬜ |
| T3 | B0-B path-independent sizing | B | S | T2 | ⬜ |
| T4 | B0-C capacity decoupling | B | S | T2 | ⬜ |
| T5 | B0-D PnL attribution + mandatory sensitivity report | B | M | T2–T4 | ⬜ |
| T6 | R1 decision package (feature set, 71 params, numeraire) | B | M | T5 evidence | ⬜ |
| T7 | Aster licence resolution | B | S | owner action | ⬜ |
| T8 | A2 batch-and-merge scoring past `MAX_SCORED=32` | A | M | — | ⬜ |
| T9 | A2 gate model that proposes drops (contrastive curation protocol) | A | L | review gate | ⬜ |
| T10 | A5 filter-versus-rebuild paired comparison | A | M | A4 fixtures (done) | ⬜ |
| T11 | B3/R3 financial baselines (rules, logistic, GBM, CE, exact-Brier) | B | M | T5, T6 | ⬜ |
| T12 | B5 real-data validation protocol in `financial_backtest_v1.py` | B | M | T5 | ⬜ |
| T13 | B6 risk gate on real data | B | S | T12 | ⬜ |
| T14 | B4 RLCD-like estimator comparisons | B | L | T11 | ⬜ |
| T15 | E2 ecosystem verification ledger (ongoing) | shared | S | — | 🟡 running |
| T16 | P0 three-seed relevance independent review | A | external | owner | ⬜ |

### T1 — B0-0 frozen replay baseline

**Goal.** Make every paper-trading number reproducible from a single frozen config, so "same
input, same result" is checkable and "different result" always has an explainable cause.
Effort S. Artifact: `research/paper_trade_b0_protocol.json` + protocol hashes in receipts.

**Status.** The protocol file [`research/paper_trade_b0_protocol.json`](../research/paper_trade_b0_protocol.json)
**was created 2026-09-19** (venues, symbols, window, strategy and policy parameters, seed,
contracts, input-manifest paths). Remaining: the driver-side wiring below.

**Changes.**
- `scripts/paper_trade_perp_v1.py`: add `--protocol PATH` — when given, the protocol overrides
  every run parameter, the referenced input manifest must exist, and the receipt records
  `protocol_sha256` and `input_manifest_sha256`. Unknown schema versions abort.

**Verify.** Two runs of the same command produce byte-identical receipts; flipping one parameter
in the protocol changes `protocol_sha256` and aborts the run.
`grep -q '"protocol_sha256"' results/paper_trade_perp_binance_v1.json`

**Gate.** No receipt without a protocol hash; no headline number without `replay_sha256`.

### T2 — B0-A order/fill divergence state machine

**Goal.** A rejected, partially filled, or expired order must become an explicit event, never a
silent change to the trade path. Depends on T1. Effort M. Artifact: receipts carry a
`divergences` block; new tests in `scripts/test_perp_pipeline_v1.py`.

**Code facts established while planning (2026-09-19).** `long`/`short` are target-position
actions: the simulator sizes each order as the delta from the **ledger's actual** position, so
position *level* self-heals after a divergence. What does not heal is the size and timing of
each trade (capacity clamps at `10% × bar volume`, rejections, TTL expiry), which changes fees,
funding, and the PnL path. The simulator's `result["orders"]` already carries
`decision_id`, `requested_quantity`, `filled_quantity`, and `status_history` — sufficient for a
post-replay audit with no simulator change.

**Changes.**
- `scripts/paper_trade_perp_v1.py`: add `find_divergences(result)` (terminal status != `filled`
  or `filled_quantity != requested_quantity` → divergence record) and
  `--on-divergence {error,report}` (default `error`). `error` writes the receipt, marks
  `divergence_error: true`, and exits non-zero naming the first divergences. `report` records
  the full divergence list and continues.

**Verify.** Tests: (a) synthetic rejected order → `error` mode exits non-zero; (b) `report`
mode → receipt lists the divergence with reason codes; (c) clean run → `divergence_count: 0`.
Command: `.venv/bin/python -m unittest scripts.test_perp_pipeline_v1 -v`.

**Gate.** Zero silent divergences in any mode; divergence count in every receipt.

### T3 — B0-B path-independent sizing (verify + lock)

**Goal.** Lock the property that sizing cannot compound with the PnL path. Depends on T2.
Effort S. Artifact: sizing test + receipt field.

**Code fact.** The driver already computes quantity from **initial** cash and the decision-day
close (`0.25 × initial_cash × leverage / close`) — path-independent notional, verified
2026-09-19. The earlier "equity compounding" claim in this document was wrong.

**Changes.** `scripts/paper_trade_perp_v1.py`: add `--sizing fixed_notional` (the only driver
mode; recorded in the receipt). An equity-compounding variant requires interleaved
decide/replay and is deferred to T12 (B5).

**Verify.** Test: identical price series with shuffled replay outcomes produces identical
decision quantities (sizing depends on price and initial cash only).

**Gate.** Receipts record `sizing: fixed_notional`; the test locks the property.

### T4 — B0-C capacity decoupling

**Goal.** Cross-venue comparisons must not be driven by venue-reported volume differences.

**Changes.** `scripts/paper_trade_perp_v1.py`: add `--capacity {fixed_bps,venue_volume,off}`,
default `fixed_bps` with a declared participation rate (e.g. 1% of a **fixed reference notional
volume** identical across venues). `venue_volume` keeps the current behavior for study runs.

**Verify.** Test: with `fixed_bps`, fill counts for the same strategy are identical across
venues; with `venue_volume`, Aster's lower volume produces fewer fills (documenting the
coupling we are removing).

**Gate.** Cross-venue tables in docs must come from `fixed_bps` runs.

### T5 — B0-D PnL attribution + mandatory sensitivity report

**Goal.** Every financial number ships with its decomposition and its spread.

**Changes.**
- New `scripts/paper_trade_report_v1.py`: reads one or more receipts and emits
  `results/paper_trade_b0_report_v1.json` with (a) PnL attribution — price/signal, fees,
  funding, spread, unfilled-capacity remainder — reconciling to net PnL within 1e-6;
  (b) a sensitivity matrix over seeds × venues × day-subsets with min/median/max and spread;
  (c) a `headline_allowed` boolean that is false when the spread exceeds a declared threshold.
- `docs/PAPER_TRADE_REAL_DATA_V1.md`: replace the four-number table with the attributed,
  sensitivity-qualified version once T2–T4 land.

**Verify.** Attribution sums to net PnL (test); sensitivity report present and
`headline_allowed` computed (test); doc refuses a bare number (review checklist).

**Gate (B0 exit).** Same strategy across ≥3 venues produces an interpretable spread, or the
report sets `headline_allowed: false` and says why.

### T6 — R1 decision package

**Goal.** Convert the three open R1 questions into decision-ready material for the owner.

**Changes.** New `docs/FINANCIAL_R1_DECISIONS_V1.md`: for each question (PILOT 9-feature vs
closed 12-feature allowlist; the 71 provisional parameters + 19 construction values;
single-numeraire scope), list options, consequences, evidence from T5, and a recommendation.
The **decision itself is the owner's**, recorded in this file once made.

**Gate.** No Track B training work (T11/T14) starts before these are decided and frozen.

### T7 — Aster licence resolution

**Goal.** Close the only read, concrete, unresolved venue prohibition (Aster §6.1(b)/§6.2(e)).

**Changes.** We prepare `docs/ASTER_PERMISSION_REQUEST_V1.md` (draft request text and the exact
usage description). **Owner actions**: send the request, obtain an API key under its terms, stop
using Aster data, or record an accepted-risk decision in the licensing audit.

**Gate.** Until resolved, Aster results stay in a separate, clearly-labelled appendix and never
in headline tables.

### T8 — A2 batch-and-merge scoring past `MAX_SCORED = 32`

**Goal.** Requests with more than 32 scorable candidates get scored instead of bypassing.

**Changes.** `scripts/context_gate_v1.py`: split the candidate list into ≤32 batches, score each
batch, merge removal plans in original order. Fail-open semantics unchanged; batches that error
cause that batch's candidates to be retained.

**Verify.** New tests: numerical equivalence with single-batch scoring on ≤32-candidate
requests; a >32-candidate request now produces a removal plan instead of `scoring_budget_exceeded`;
fail-open preserved on injected batch errors.

**Gate.** Merged plans are byte-identical to single-batch plans where both apply.

### T9 — A2 gate model that proposes drops (contrastive curation)

**Goal.** A scorer that discriminates distractor from load-bearing context, fixing the measured
Catalog Choice 100% → 54.17% collapse.

**Changes.** New `docs/GATE_CONTRASTIVE_PROTOCOL_V1.md` (pre-registered protocol, fresh splits,
following the nimble contrastive-curation recipe: change one fact so the correct gate decision
flips); then data builder `scripts/build_gate_contrastive_v1.py`; training via the existing
runtime. **Review gate before any training**; the existing relevance test/OOD stays untouched.

**Gate.** Promotion requires the full Track A gates; a contrastive checkpoint that still
proposes zero drops is a valid, reportable negative result.

### T10 — A5 filter-versus-rebuild paired comparison

**Goal.** Settle filter-versus-rebuild on our fixtures with evidence.

**Changes.** New `scripts/filter_vs_rebuild_v1.py` + tests: five arms (unfiltered, safe dedup,
relevance filter, abstractive summary, retrieval-rebuild) over the A4 fixtures; metrics per the
A5 section; cost accounting per the Phase 0 cost-native rule.

**Gate.** Paired intervals per task family; aggregate win with a protected-family loss is a fail.

### T11 — B3/R3 financial baselines

**Goal.** Baselines before any RLCD-like estimator.

**Changes.** New `scripts/financial_baselines_v1.py`: deterministic rules, logistic regression,
gradient boosting, cross-entropy and exact-Brier objectives on the frozen cohort with matched
seeds/initialization/budget; paired source-group intervals.

**Gate.** Beats determinism with intervals, or we report that it does not.

### T12 — B5 real-data validation protocol

**Goal.** The real data flows through `financial_backtest_v1.py` proper, not a bespoke driver.

**Changes.** Extend `scripts/financial_backtest_v1.py` with a real-data path (external bars +
funding instead of synthetic path generation); purged walk-forward with embargo; regime and
instrument holdouts reported separately.

**Gate.** The harness emits equity curve, drawdown, and per-regime analytics for the real cohort.

### T13 — B6 risk gate on real data

**Changes.** Run the existing out-of-model risk guards on the real replay; produce the first
real-data risk receipt (`results/risk_real_v1.json`).

### T14 — B4 RLCD-like estimators

Only after T11 passes. Report calibration, selective risk, costs, regime stability. Remember:
`paired_brier_pg` is refused on this Mac's MPS — plan CPU or CUDA.

### T15 — E2 ecosystem verification ledger (ongoing)

**Goal.** No ecosystem number enters a project document without evidence level, source URL,
snapshot time, and comparison scope. Effort S, recurring. Artifact: updates to the ecosystem
section and one subsection per verified item in
[the pinned reference review](JEV_COMMUNITY_REFERENCES.md).

**Recurring checklist.**
- Re-snapshot star counts (GitHub API) before quoting any of them; record the UTC date.
- Locate primary sources for the still-unverified sweep items: Atomic, Jevinik, the Monad
  live-trading bot, the DuckDB extension, the three cost case studies, Decider-2B, System-One
  4B, the Jev-compatible public API, and the HF-model adapter library.
- Re-check the fast-jev-compaction 156k→62k claim if the project publishes benchmark receipts
  (its README currently documents character-count token estimates only).
- Never install plugins, upload private data, use accounts, or place orders to verify a claim.

### T16 — P0 independent review (external)

**Goal.** The three-seed relevance result (no candidate promoted) gets an independent
implementation/evidence review; we do not self-certify. Effort external. Artifact: a review
verdict filed in [the execution review log](EXECUTION_REVIEW_LOG.md).

**Blockers on our side.** None — evidence package is complete (frozen protocol, three run
receipts, hashes all MATCH, independent report rebuild semantically identical). Waiting on
reviewer availability.

## Codex local skill and telemetry

Status: **implemented for structured local decisions; the new shadow library reuses its telemetry, but active universal filtering remains unimplemented**

NanoJev is packaged as the `nanojev-local-decider` Codex skill under [`integrations/codex-skill/nanojev-local-decider`](../integrations/codex-skill/nanojev-local-decider). The installed copy lives in the local Codex skills directory and uses a persistent loopback HTTP service, preferring `127.0.0.1:8765` and remembering an automatic fallback port if that port is already occupied by another local service.

The current skill is intended for routing, candidate selection, confidence gates, verification, and computer-use decisions. It is not yet a universal context filter, chat replacement, financial trading system, or live execution engine. It records one privacy-preserving JSONL event per decision by default, including task tags, schema types, candidate cardinalities, latency, confidence, abstention, runtime, checkpoint identity, and a request fingerprint. Raw states and criteria remain excluded unless a user deliberately enables local debug payload capture.

Downstream outcomes are recorded separately through `record-feedback`, using `correct`, `incorrect`, `abstained`, `fallback`, or `human_override`. The summary command exposes latency, confidence, abstention, task type, and feedback coverage for later calibration, dataset construction, and optimization. This telemetry is an input to both primary tracks; it is not ground truth until feedback or an independently verified outcome exists.
