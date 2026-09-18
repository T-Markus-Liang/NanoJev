# NanoJev V2 roadmap

NanoJev should evolve as a decision system, not as a smaller chat model. The upgrade order is deliberate:

1. Improve decision success on the tasks the system already claims to support.
2. Reduce warm and cold inference cost without changing the decision contract.
3. Expand the contract and task families only after the first two are measured and stable.

The official Jev API is proprietary and its public latency/cost claims are workload-specific. NanoJev therefore uses fixed public cohorts and explicit acceptance gates instead of trying to reproduce marketing numbers.

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
- [ ] Evaluate the first V2.1 candidate and its baseline with at least three matched training seeds.

Exit gate: every model change reports accuracy, NLL, Brier, ECE, invalid-output count, cold start, warm p50/p95, throughput, and peak memory on the same cohort.

## Phase 1: V2.1 decision success

Priority: **highest**

The first model upgrade should improve the probability that a decision is correct within the intended task distribution, including difficult and ambiguous cases.

### Data and target quality

- Build a balanced task matrix across Boolean/Noul, Choice, and Score instead of optimizing only local maze Boolean questions.
- Add hard negatives: near-identical states, contradictory evidence, distractor candidates, candidate descriptions with overlapping vocabulary, and long irrelevant context.
- Separate semantic paraphrases from source-group duplicates so OOD measures transfer rather than memorization.
- Preserve exact probability targets where available; keep deterministic labels, observed outcomes, and teacher distributions as separate target kinds.
- Add a small human-reviewed challenge set that cannot be generated from the training templates.

### Training and decision quality

- Compare shared scalar heads, set-attention heads, and candidate-conditioned cross-attention under matched seeds.
- Train with a mixture of proper losses: cross-entropy for deterministic labels, Brier for calibrated probabilities, and distribution loss for soft targets.
- Fit temperature only on calibration data; never select a checkpoint or threshold on test/OOD labels.
- Add selective prediction: expose confidence and an explicit abstain state for low-confidence decisions so downstream systems can route uncertain cases to a stronger model or human review.
- Measure per-task-family and per-candidate-count performance; do not accept an aggregate gain that hides a regression in one primitive.

### V2.1 acceptance targets

Targets are gates for the next release, not claims about the current checkpoint:

- local-maze test accuracy: at least 82%, with no more than 0.5 percentage-point regression on any existing split
- local-maze OOD accuracy: at least 80%
- calibration ECE: below 0.08 on the fixed calibration protocol
- zero schema or probability-sum failures across the stress suite
- challenge-set accuracy reported separately, with no hidden tuning

## Phase 2: V2.2 inference speed and efficiency

Priority: **second**

Speed work must be measured against the same outputs and must not silently reduce task quality.

- Implement prefix/tree sharing for states and question prefixes; verify numerical equivalence against repeated-path inference.
- Add tokenizer and encoded-prefix caches for persistent service workloads, with bounded memory and request-isolation tests.
- Benchmark PyTorch SDPA, MPS kernels, CUDA BF16, and optional `torch.compile` as separate runtime profiles.
- Evaluate weight-only quantization and lower-memory checkpoint formats only after comparing accuracy and probability calibration.
- Add controlled microbatching and concurrency tests; distinguish queueing latency from model compute.
- Record cold-start time, warm p50/p95/p99, candidate paths per second, questions per second, memory, and energy where available.

### V2.2 provisional targets

- M5 Max warm p50 at least 1.5x faster for the current 3-question smoke request
- 255-candidate Choice below 800 ms on the same machine, without a measurable fixed-cohort accuracy regression
- persistent-service memory below 2.5 GiB for the current FP32 checkpoint
- exact or tolerance-bounded equivalence checks for every optimized execution path

## Phase 3: V2.3 capability coverage

Priority: **third**

Expand the decision contract only after V2.1 and V2.2 gates are green.

- Support structured instructions and candidate descriptions with a versioned canonical serializer.
- Add explicit `noul` compatibility with Jev's native naming while preserving the local `boolean` alias.
- Return response metadata parity: confidence, score legend, calibration status, model version, and abstention reason.
- Add richer candidate types and nullable/structured fields without ambiguous string coercion.
- Add dependent multi-stage workflows as separate requests with explicit state transitions; do not leak prior answers into independent questions.
- Evaluate candidate sets above 255 only as an extension experiment, not as an undocumented contract change.
- Add adapters for routing, extraction, verification, tool selection, and safety gates using task-specific datasets.

## Phase 4: V2.4 system and ecosystem

- Publish a versioned Python/HTTP SDK with stable schemas.
- Provide OpenAI-style structured decision endpoints without pretending to be a text-generation API.
- Add model registry metadata, checkpoint hashes, benchmark receipts, and reproducible release bundles.
- Compare NanoJev and official Jev on identical frozen requests when official API access is available.
- Publish failure analyses, not only aggregate scores.

## Research rules

- Never mix training, calibration, and test/OOD data.
- Never compare a local warm request with an end-to-end remote request without reporting both scopes.
- Never call a recorded Jev trajectory an independent fresh API measurement.
- Every speed optimization needs a numerical equivalence or task-quality report.
- Every capability extension needs a versioned input/output contract and a migration test.

## Current development slice

This branch starts Phase 0 and prepares Phase 1:

1. Add the reproducible V2 benchmark command.
2. Establish the M5 Max baseline from the public local-maze data.
3. Use that baseline to evaluate hard-negative, calibration, and multi-seed experiments.
4. Select the first V2 checkpoint only after the acceptance gates above are measured.

## Codex local skill and telemetry

Status: **implemented**

NanoJev is also packaged as the `nanojev-local-decider` Codex skill under [`integrations/codex-skill/nanojev-local-decider`](../integrations/codex-skill/nanojev-local-decider). The installed copy lives in the local Codex skills directory and uses a persistent loopback HTTP service, preferring `127.0.0.1:8765` and remembering an automatic fallback port if that port is already occupied by another local service.

The skill is intended for routing, candidate selection, confidence gates, verification, and computer-use decisions. It is not a chat or code-generation replacement. It records one privacy-preserving JSONL event per decision by default, including task tags, schema types, candidate cardinalities, latency, confidence, abstention, runtime, checkpoint identity, and a request fingerprint. Raw states and criteria remain excluded unless a user deliberately enables local debug payload capture.

Downstream outcomes are recorded separately through `record-feedback`, using `correct`, `incorrect`, `abstained`, `fallback`, or `human_override`. The summary command exposes latency, confidence, abstention, task-type, and feedback coverage for later calibration, dataset construction, and optimization. This telemetry is an input to V2.1 success-rate work and V2.2 runtime work; it is not treated as ground truth until feedback or an independently verified outcome exists.
