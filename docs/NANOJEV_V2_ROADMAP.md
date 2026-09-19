# NanoJev V2 roadmap

NanoJev should evolve as a fast, calibrated decision layer rather than a smaller chat model. V2 has two primary product goals:

1. **Universal model integration:** place NanoJev in front of GPT, Claude, Kimi, OpenCode, local models, and other providers to identify irrelevant input segments before the main-model call, reducing input-token cost without losing instructions, evidence, or task success.
2. **Financial secondary-market decisions:** explore an independently reproducible training approach inspired by Reinforcement Learning for Calibrated Decisions (RLCD) for bounded trading decisions, with calibrated probabilities, explicit abstention, realistic costs, and millisecond-level latency as an acceptance target.

The financial track is the higher strategic priority. The universal integration track is the first reusable deployment surface and provides the filtering, telemetry, calibration, fallback, and low-latency infrastructure needed by both tracks.

Neither goal is a current capability claim. Token reduction, financial utility, calibration, and latency must be demonstrated on frozen, reproducible evaluations before release claims or live use.

The official Jev API is proprietary and its public latency and cost claims are workload-specific. NanoJev therefore uses explicit contracts, public cohorts, benchmark receipts, and acceptance gates instead of claiming architectural equivalence or reproducing marketing numbers.

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

## Phase 0: measurement and contract gates

Status: **in progress**

- [x] Freeze public test and OOD cohorts.
- [x] Record cold-load and warm-request latency separately.
- [x] Test candidate counts through 255.
- [x] Test candidate permutation invariance.
- [x] Add one machine-readable benchmark command.
- [x] Add deterministic source-group confidence intervals, multi-seed aggregation, and paired checkpoint comparison tooling.
- [x] Add a benchmark manifest containing dataset, checkpoint, dependency, and hardware hashes.
- [ ] Commit and publish the broad workflow baseline covering Boolean, Choice, Score, rule-based tasks, and known probability distributions.
- [ ] Evaluate the first V2.1 candidate and its baseline with at least three matched training seeds.

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

The initial target is a bounded decision engine, not free-form market commentary and not an autonomous live-trading system. The model should emit complete probability distributions and an abstain/no-trade decision for clearly defined horizons and market states.

### B1. Decision and state contract

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

### Track B initial acceptance gates

Targets are provisional and must be revisited after the first frozen dataset and simulator:

- warm local model compute p50 below 5 ms and p99 below 20 ms for bounded state/candidate workloads on declared hardware
- end-to-end paper-decision p99 below 50 ms, with feature freshness and validation included
- better calibration and selective-risk curves than the deterministic baseline on frozen holdouts
- positive net performance after realistic costs across multiple purged walk-forward periods, with uncertainty intervals and no single-period dependency
- stable behavior across instrument and regime holdouts, with explicit abstention under distribution shift
- zero live trading until offline, shadow, and paper-trading gates pass

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
- Never enable active context removal without fail-open fallback and protected-segment tests.
- Never enable live trading as part of model research or benchmark automation.
- Every speed optimization needs a numerical-equivalence or task-quality report.
- Every capability extension needs a versioned input/output contract and a migration test.

## Current development slice

Work proceeds in this order:

1. Complete and publish the broad baseline and challenge evaluation, including the current rule-based Boolean and probability-calibration weaknesses.
2. Build the provider-neutral universal context-gating adapter in shadow mode with protected segments, receipts, and fail-open fallback.
3. Establish a paired token-savings and downstream-quality benchmark across multiple main-model families.
4. Define and freeze the financial point-in-time dataset contract, leakage checks, purged walk-forward splits, execution simulator, and risk controls.
5. Establish deterministic, cross-entropy, and exact-Brier financial baselines before any policy-gradient experiment.
6. Begin RLCD-like estimator comparisons only after the supervised baselines and simulator pass their integrity checks.
7. Optimize model, feature, and end-to-end latency only on frozen tasks, without weakening calibration, risk, or leakage gates.

## Codex local skill and telemetry

Status: **implemented for structured local decisions; universal filtering is not yet implemented**

NanoJev is packaged as the `nanojev-local-decider` Codex skill under [`integrations/codex-skill/nanojev-local-decider`](../integrations/codex-skill/nanojev-local-decider). The installed copy lives in the local Codex skills directory and uses a persistent loopback HTTP service, preferring `127.0.0.1:8765` and remembering an automatic fallback port if that port is already occupied by another local service.

The current skill is intended for routing, candidate selection, confidence gates, verification, and computer-use decisions. It is not yet a universal context filter, chat replacement, financial trading system, or live execution engine. It records one privacy-preserving JSONL event per decision by default, including task tags, schema types, candidate cardinalities, latency, confidence, abstention, runtime, checkpoint identity, and a request fingerprint. Raw states and criteria remain excluded unless a user deliberately enables local debug payload capture.

Downstream outcomes are recorded separately through `record-feedback`, using `correct`, `incorrect`, `abstained`, `fallback`, or `human_override`. The summary command exposes latency, confidence, abstention, task type, and feedback coverage for later calibration, dataset construction, and optimization. This telemetry is an input to both primary tracks; it is not ground truth until feedback or an independently verified outcome exists.
