# Jev community references: adoption review

Reviewed: 2026-09-19. This is a source/document review, not a local reproduction of any external project's benchmarks. No third-party plugin, weights, hook, or global provider setting was installed or changed for this review.

Target hardware context: the adoption evaluations below are framed for this project's working machine — an Apple Silicon Mac laptop running NanoJev via PyTorch MPS in FP32. Every upstream latency figure was measured on other hardware (T4, GB10, M4) and must be re-measured locally before any comparison or adoption decision.

## Scope and immutable sources

| Project | Reviewed revision | License reported by GitHub | Relevant sources |
|---|---|---|---|
| `tamaratran/fast-jev-compaction` | `e3f262a7f4d42bd8dd32ced30d26176f7cb545b0` | MIT | [README](https://github.com/tamaratran/fast-jev-compaction/blob/e3f262a7f4d42bd8dd32ced30d26176f7cb545b0/README.md), [compaction implementation](https://github.com/tamaratran/fast-jev-compaction/blob/e3f262a7f4d42bd8dd32ced30d26176f7cb545b0/src/compact.ts) |
| `hr98w/jev-visual` | `19af545f096e8db4c4dd5d47aed42d92ec252111` | MIT | [README](https://github.com/hr98w/jev-visual/blob/19af545f096e8db4c4dd5d47aed42d92ec252111/README.md), [inference contract](https://github.com/hr98w/jev-visual/blob/19af545f096e8db4c4dd5d47aed42d92ec252111/docs/inference.md), [scoring implementation](https://github.com/hr98w/jev-visual/blob/19af545f096e8db4c4dd5d47aed42d92ec252111/jev_visual/scoring.py), [published benchmark](https://github.com/hr98w/jev-visual/blob/19af545f096e8db4c4dd5d47aed42d92ec252111/benchmarks/RESULTS.md) |
| `NandhaKishorM/laya` | `4937897c680dcaa041d05a1040c507f70cba2aeb` (v0.3.1) | Apache-2.0 | [README](https://github.com/NandhaKishorM/laya/blob/4937897c680dcaa041d05a1040c507f70cba2aeb/README.md), [benchmark report](https://github.com/NandhaKishorM/laya/blob/4937897c680dcaa041d05a1040c507f70cba2aeb/BENCHMARKS.md), [fine-tuning notebook](https://github.com/NandhaKishorM/laya/blob/4937897c680dcaa041d05a1040c507f70cba2aeb/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb), [HF model](https://huggingface.co/convaiinnovations/laya) |
| `kshetrajna12/reflex` | `3124257395eccd5716fba61713d59f2e0acb1a8d` | MIT | [README](https://github.com/kshetrajna12/reflex/blob/3124257395eccd5716fba61713d59f2e0acb1a8d/README.md), [architecture](https://github.com/kshetrajna12/reflex/blob/3124257395eccd5716fba61713d59f2e0acb1a8d/docs/ARCHITECTURE.md), [LoRA results](https://github.com/kshetrajna12/reflex/blob/3124257395eccd5716fba61713d59f2e0acb1a8d/docs/results/lora-mix-qwen3.5-4b.md), [browser demo](https://kshetrajna12.github.io/reflex/) |

Before copying source, review the actual LICENSE, third-party notices, dependency licenses, and model-weight terms at the pinned revision. Preserve attribution. No upstream source is vendored by this document change.

## Track A: fast-jev-compaction

### What the project actually does

- Claude Code plugin and TypeScript library using a pluggable `JevAsker` transport; the default client calls the TypeSafe Jev API.
- Scores tool calls and results separately, preserving first/recent messages and leaving retained conversational text verbatim and ordered.
- Chooses keep, retain-call/truncate-result, or remove-call-and-result together. This maintains call/result pairing rather than deleting unrelated fragments independently.
- Fits a judge state to a bounded budget and batches questions; state is repeated for each batch. Tool-result bodies are omitted from that judge state, and other state may be shortened to fit.
- Defaults include `keepThreshold=0.5`, six recent messages, approximately 25k state tokens, approximately 30k request tokens, and a 300-character result head. Token sizing is heuristic; `reductionRatio` uses character counts.
- Library failures throw; the documented Claude hook falls back to built-in summarization on failure or insufficient reduction. This is not NanoJev's unchanged-request fail-open guarantee.

### Adopt, test, or reject

Adopt the design ideas of atomic tool dependencies, separate call/result decisions, pinned context, verbatim survivors, injectable transport, bounded requests, and inspectable decisions.

Do not inherit a numerical threshold without calibration. A score above/below 0.5 is not proof of safe deletion; conversely NanoJev's current 0.99 gate is provisional, not established calibration. Omitted result contents can contain the only error, constraint, evidence, or correction needed later. Tool re-execution may be costly, state-changing, unavailable, or return different data.

Our fallback remains the original request. Built-in summarization may be compared as a separately named experimental arm, never silently substituted for fail-open. Input token counts, cached-token pricing, repeated judge context, provider charges, and retries/tool re-runs must be included in net economics.

### A4 deliverables and gates

1. A fresh tool-transcript fixture manifest with source-group isolation and explicit dependency labels, separate from the existing frozen relevance corpus.
2. Protocol-specific pairing tests for success, error, parallel, duplicate, orphan, pending, and non-repeatable tool cases. Assert no orphan result, reordered survivor, changed protected text, or destructive re-execution.
3. Shadow receipts for three eligible operations, with original request preserved. Test timeout, malformed score, oversized context, unsupported blocks, and uncertain relevance.
4. Frozen paired comparison: unfiltered, deterministic safe deduplication, upstream-style policy, and local gate. Compare result-omitted versus evidence-bearing judge states separately.
5. Downstream task correctness and actual input tokens on at least three main-model families before default active use, with the roadmap's protected-context and quality gates intact.

Current status (2026-09-19): reference review complete; **the tool-history experiment is now implemented** as work package A4 — 11 deterministic fixtures, 13 unit tests, and all 11 replayed through the real gateway with 11/11 status/reason matches and zero receipt leaks ([Tool history shadow V1](TOOL_HISTORY_SHADOW_V1.md)). The paired main-model quality/cost comparison remains open.

## Track B: jev-visual

### What the project actually does

An educational local visual inference project using Qwen3.5-0.8B with MLX on Apple Silicon. It explicitly disclaims reproducing Jev's proprietary architecture, RLCD, calibration, and serving system. It is not a financial dataset, backtester, trading strategy, or RLCD trainer.

Useful implementation details:

- One shared image/context prefill, followed by bounded question/candidate batches; an independent full-forward mode is the numerical reference.
- Stable-label or native single-token scoring, plus teacher-forced multi-token sequence scores. Sequence mode normalizes over the full vocabulary at every step and includes EOS; it is not autoregressive sampled JSON generation.
- Candidate token round-trip, special-token, collision, and boundary-merge checks; shared mode enforces a request length bound.
- Architecture-specific branching includes recurrent/convolution state as well as attention KV. The documented implementation copies caches; it is not zero-copy or cross-request caching.
- Normalized candidate probabilities and entropy concentration are not correctness or event-calibration certificates.

### What its public numbers do and do not establish

The published M4/16GB, 4-bit-weight/FP32-compute benchmark uses one image, three measured repetitions, and repeated criteria for scaling. Shared mode reports median totals of 630.0 ms for one decision, 678.6 ms for four, 1051.5 ms for sixteen, and 2399.1 ms for sixty-four. These are upstream measurements, not measurements made on this Mac or NanoJev.

This supports investigating reuse for multi-question workloads. It does not establish sub-20-ms financial inference, p99 latency, general accuracy, or calibration. Dividing a batch's duration by its decision count is throughput accounting, not the latency of an individual live trading decision.

### B8 deliverables and gates

1. Structured market-state baseline first, with availability timestamps, leakage checks, costs, abstention, and deterministic risk limits from Track B.
2. Independent-versus-shared scoring specification on bounded states/candidates: probability parity, candidate-order checks, padding, token boundaries, recurrent-state isolation where applicable, and explicit tolerances frozen before timing.
3. Optional paired structured/image/combined ablation from identical point-in-time observations. Exclude future-dependent chart scaling, future candles, revised indicators, and post-event overlays. Hold out periods, instruments, and regimes.
4. Report feature preparation, chart rendering, vision encoding, model compute, queueing, validation, and full paper-decision p50/p95/p99 separately. Compare calibrated event probabilities and net financial utility, not only formatted actions.
5. Keep visual processing only if it adds reproducible value under the existing accuracy, calibration, risk, and latency gates. Do not assume a chart encoder belongs on the critical execution path.

Current status: source review complete; no local reproduction, financial transfer, or new RLCD training result is claimed.

## Shared runtime and Track B recipe: laya (ModernBERT 421M, encoder decision engine)

### What the project actually does

- A non-autoregressive decision engine on **encoder** backbones, not decoder LLMs: `laya` is ModernBERT-large at 421M parameters with 512-token context; `laya-multilingual` is mmBERT-base at 322M with 1024-token context; `laya-typed-decisions` is ModernBERT-large at 421M with 1024-token context. All questions of a request are answered in a single forward pass.
- Ships the same three primitives (choice, score, noul) with probabilities and confidence, plus a `Router` that picks a checkpoint per request. Script detection is the primary routing signal, motivated by their own finding that the English checkpoint collapses outside English while remaining confident (Khmer accuracy 0.000 at 95.2% reported confidence).
- Training recipe: "RLCD" as proper-scoring-rule rewards with a GRPO-style policy gradient, plus per-(question type, option count) post-hoc temperature fitting. The public notebook runs on two free Kaggle T4s (about 4–5 hours, 4 epochs, about 30k questions). Weights are on Hugging Face (`convaiinnovations/laya*`), install via `pip install laya`.
- Reports measured speed on a T4: 33–39.5 ms for one question, 7.2 ms/question batched at 10 questions. Reports a typed-decisions benchmark accuracy of 0.766 against TypeSafe Jev 1.13.0's **published** 0.727.

### What its public numbers do and do not establish

- All latency figures are Tesla T4 measurements, not Apple Silicon, not MPS, not this Mac. They justify investigating an encoder gate for latency; they establish nothing about local performance until re-measured.
- The Jev comparison columns use third-party published Jev numbers; Laya's authors state they had no TypeSafe API access. Sample sizes and prompts differ. This is not a paired measurement and must not be quoted as one.
- By the authors' own "honest limits": the base checkpoints are near chance zero-shot on typed decisions (0.362 and 0.342 against a 0.318 random baseline); the 0.766 figure comes from the checkpoint fine-tuned on that benchmark's own training split. Capability comes from fine-tuning — the same lesson as our own relevance experiment.
- Both checkpoints are over-confident as shipped (mean ECE 0.466/0.314), improving to 0.081/0.106 only after refitting temperatures on held-out data; `laya-multilingual` ships with no fitted temperatures at all. Their headline ECE is a post-refit number.
- Encoder context is 512/1024 tokens. That is far below Track A's long coding-history and multi-document workloads; an encoder gate would need a chunking or selection design, not a drop-in replacement.
- Choice questions degrade above roughly 20 options (fixed label token budget). Our contract supports 2–255 candidates.

### Mac laptop fit

This is the most Mac-friendly runtime reference of the four: a 421M encoder is substantially lighter than our 0.6B decoder, is `pip`-installable, and should run under PyTorch MPS or even CPU. If a context-gate scorer can be served by a small encoder, gate latency and memory on this laptop improve by design rather than by optimization. This is a hypothesis to measure, not a result.

### Adopt, test, or reject

- Adopt as reference: encoder-plus-heads as a candidate gate architecture; the script-detection router as prior art for multilingual Track A cohorts; the discipline of publishing "honest limits" and per-language collapse evidence; temperature refits scoped per (type, cardinality).
- Test before any use: local MPS/CPU latency and memory on this Mac; zero-shot behavior on our frozen workflow cohorts (expect near-chance per their own disclosure — that is a data point, not a failure); license/weight terms at the pinned revision before any vendoring.
- Reject: transplanting T4 latency or the Jev comparison table into our materials; treating its GRPO-style recipe as the definition of RLCD (roadmap B4 requires documented objectives and verification); any suggestion that a 512-token encoder solves long-context gating unaided.

### Deliverables and gates (laya)

1. Read-only local evaluation on this Mac: install at the pinned revision in an isolated environment, run its own quickstart, record MPS/CPU latency and memory next to our V1.0 baseline table. No integration into `scripts/` or the service.
2. Zero-shot relevance probe on a small frozen sample of the workflow baseline, clearly labeled as upstream-weights zero-shot, no threshold tuning.
3. A short written comparison of its GRPO-style proper-scoring recipe against our paired Brier estimator and exact proper-loss baselines, filed under Track B4 as a research candidate.
4. Encoder-gate architecture question (encoder scorer versus our decoder heads for Track A) is a design decision requiring review before any prototyping beyond step 1–2.

Current status (2026-09-19): **step 1 done** — laya installed in an isolated environment and ran on this Mac's MPS: in-process warm p50 **40.8 ms** over 4 questions (load 241.8 s including one-time weight download); see the review log's E1 entry. Steps 2–4 remain open; no integration, training, or quality claim is made.

## Shared runtime and model quality: reflex (Qwen3.5 + direct-logits readout)

### What the project actually does

- A Jev re-creation on a **stock pretrained decoder** (Qwen3.5-4B by default; Qwen3 and Qwen3-VL supported) with **no trained decision heads**: at each question branch's last token it takes next-token logits, restricts them to the label tokens (`A/B/C…`, `Yes/No`), temperature-scales, and softmaxes. That restricted softmax over the model's native logits is the entire readout.
- Shared state encoding with an LRU KV cache keyed by content hash ("prefix sharing"); all question branches run in one forward pass, isolated from each other. Two strategies: `packed` (custom 4D attention mask, per-branch position ids, for attention-only backbones) and `batched` (batch-expanded state-cache copy for hybrid linear-attention backbones like Qwen3.5, where a mask cannot isolate a recurrent scan). GPU tests assert numerical identity with per-question independent forwards.
- Calibration protocol: fit one temperature on half of 1,200 MMLU items, report the held-out half; then a LoRA trainer on the label-restricted logits with cross-entropy/Brier against hard or soft labels ("RLCD-lite": when the output is the distribution, expected proper reward is differentiable in closed form, so RL collapses to supervised minimization).
- Serving: FastAPI `POST /v1/systemone` deliberately compatible with TypeSafe's hosted API shape. A WebGPU browser demo runs Qwen3.5-0.8B (ONNX q4f16, about 650 MB) entirely client-side in Chrome/Edge/Safari 18+.

### What its public numbers do and do not establish

- MMLU (1,200 items): Qwen3.5-4B raw readout accuracy 72.0%, ECE 0.090, improving to 0.039 with one held-out-fitted temperature (Jev publishes 0.031 on a possibly different sample — cross-source comparison only). One epoch of LoRA on an eight-dataset mix moved held-out accuracy 62.7% → 76.8% and ECE 0.120 → 0.051 → 0.024 with temperature; the authors flag that the last temperature was fitted on the same held-out set, so 0.024 is slightly optimistic by their own note.
- Server latency is measured on a GB10 (DGX Spark): about 100 ms for four text questions with the state cached. The browser demo's laptop WebGPU numbers (about 381 ms cold / 300 ms warm for four questions) are the closest upstream evidence to this Mac, but still not measured here.
- Direct-logits readout caps Choice at 26 single-token letter options; Jev and NanoJev support 255. Letter-position bias exists and is mitigated with permutation averaging at N× branch cost. The ONNX/WebGPU path cannot do batch-expanded cache continuations (an operator limitation), forcing per-question continuations there — a concrete portability lesson for any MPS/ONNX runtime work.

### Mac laptop fit

Two cheap, zero-risk evaluation paths exist on this machine today: the WebGPU demo runs in Safari 18+ with no install, and the Python engine supports 0.8B-class checkpoints that fit laptop memory (the Python README targets CUDA; MPS compatibility is unverified and is itself a test item). Reflex is the best available reference for the "no trained heads" alternative and for browser-side demos.

### Adopt, test, or reject

- Adopt as reference: direct-logits readout as a documented architectural alternative to our trained decision heads; the packed-mask/batched branch-isolation design with **numerical-equivalence tests against independent forwards** — exactly the discipline our shared runtime program requires for prefix sharing; state-cache keyed by content hash; calibration fitted on a split and reported on another; TypeSafe-compatible request shape as a contract cross-check.
- Test before any use: WebGPU demo on this Mac (Safari 18+), comparing its browser and Python answers on identical requests as upstream claims; direct-logits versus trained-head quality on our frozen cohorts would be a NanoJev-side ablation, not an adoption of their weights.
- Reject: replacing our 255-candidate Choice contract with a 26-letter readout outside an explicitly labeled ablation arm; quoting its MMLU ECE against Jev's published number as a paired result; calling its supervised proper-scoring LoRA "RLCD" (roadmap B4); assuming the GB10 or WebGPU timings transfer to our MPS service.

### Deliverables and gates (reflex)

1. Browser-demo verification on this Mac: run the WebGPU page, record cold/warm timings and memory, compare outputs against the naive per-question path it exposes (`batch: false`), and file a short receipt. Read-only; nothing installed into the repo.
2. A direct-logits ablation specification for NanoJev: same frozen local-maze/workflow cohorts, same backbone, trained heads versus label-token readout, with permutation-bias measurement and the 26-candidate cap stated as a contract limitation. Requires review before execution; it is a model-quality experiment, not a runtime change.
3. Feed the packed-mask isolation and content-hash cache ideas into the shared runtime program's prefix-sharing design review, with numerical-equivalence tests as the acceptance bar.

Current status (2026-09-19): the WebGPU demo page is confirmed reachable and usable in this Mac's browser (zero install). The timing verification (step 1), the direct-logits ablation spec (step 2), and the prefix-sharing design input (step 3) all remain open.

## Execution order

The already-frozen three-seed relevance experiment is now trained and reported in [Context relevance V1](CONTEXT_RELEVANCE_V1.md) without changing its splits, losses, or threshold; the mixed outcome (one seed regressing local-maze, one seed false-dropping at 0.99) means no candidate is promoted and no further winner-search on that test/OOD is permitted. Next, prioritize financial data and simulation foundations (handoff packages P1–P3) while extending A4 tool-history shadow tests in parallel.

The laya and reflex local evaluations (laya deliverable 1–2, reflex deliverable 1) are cheap, read-only, Mac-local checks that may run in parallel as exploratory work, but they must not displace the financial work package, must not modify the frozen cohorts or production checkpoint, and their receipts are upstream-weights measurements, not NanoJev capability claims. The encoder-gate architecture question (laya deliverable 4) and the direct-logits ablation (reflex deliverable 2) both require review before execution.

The four references inform those deliverables; none changes the highest strategic priority or authorizes production deployment. See the [current progress and handoff plan](CURRENT_PROGRESS_AND_HANDOFF.md) for work-package boundaries and review gates.

Track all adoption work in [the V2 roadmap](NANOJEV_V2_ROADMAP.md). Human approval for live trading and explicit consent for transmitting private transcripts remain separate from research authorization.
