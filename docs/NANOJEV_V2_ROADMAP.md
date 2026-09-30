# NanoJev V2 roadmap

NanoJev should evolve as a fast, calibrated decision layer rather than a smaller chat model. V2 has two primary product goals:

1. **Universal model integration:** place NanoJev in front of GPT, Claude, Kimi, OpenCode, local models, and other providers to identify irrelevant input segments before the main-model call, reducing input-token cost without losing instructions, evidence, or task success.
2. **Financial secondary-market decisions:** explore an independently reproducible training approach inspired by Reinforcement Learning for Calibrated Decisions (RLCD) for bounded trading decisions, with calibrated probabilities, explicit abstention, realistic costs, and millisecond-level latency as an acceptance target.

Post-W51 owner direction is **open-source reproduction before custom optimization**: inventory runnable Jev-class systems from JevBench/Hugging Face/GitHub, prefer macOS laptop-compatible artifacts, reproduce unmodified candidates locally, run the same offline benchmark, and keep only strong native baselines before further NanoJev training or architecture work. The financial track remains separate and safety-gated, but T11–T14 found no estimator improvement over base rate; it returns to the critical path only after a new point-in-time signal/data hypothesis passes protocol review.

Neither goal is a current capability claim. Token reduction, financial utility, calibration, and latency must be demonstrated on frozen, reproducible evaluations before release claims or live use.

The official Jev API is proprietary and its public latency and cost claims are workload-specific. NanoJev therefore uses explicit contracts, public cohorts, benchmark receipts, and acceptance gates instead of claiming architectural equivalence or reproducing marketing numbers.

## Current execution handoff

Internal handoff, execution-review, and completion-audit documents remain local-only and are excluded from Git. This public roadmap records release-safe status and NanoJev-only evidence; the [V3 local-decision-substrate roadmap](NANOJEV_V3_ROADMAP.md) and the [V4 next-paradigm roadmap](NANOJEV_V4_NEXT_PARADIGM_ROADMAP_V1.md) are conditional designs, not promotion decisions. The [three-seed relevance result](CONTEXT_RELEVANCE_V1.md) is documented: training and receipts exist, independent implementation/evidence review remains pending, and **no candidate is promoted**. This does not close Phase 0 or authorize active filtering.

**Delivered since the last roadmap revision** (状态以每行证据为准；标为 `READY_FOR_REVIEW` 的项目仍待独立 reviewer，不能视作已接受):

| Package | Delivered | Evidence |
|---|---|---|
| **Goal 1 / Track A core (T57–T61)** | **Universal model/context-filter integration — owner-scored complete (100/100, 2026-09-24)** | Unified service `127.0.0.1:8876` (launchd-persistent, deterministic); official-Jev billed eval: 10.55% (fixtures) / 20.81% / 26.52% (stitched real corpus, 70 cases) real input-token savings; official scorer agreement 4/5, 7/7, 10/12; real-data joint-scoring flaw found and fixed; 200-case open-corpus regression clean. See task table T57–T61 |
| P0 | Evidence closure and review log | frozen-artifact hashes MATCH; independent report rebuild is semantically identical |
| P1 / P1b | Financial data, rights, PIT, and experiment contract draft, **stopped at R1** | 14 sources surveyed; perpetual-only scope; validator projection key-set verified |
| P2 / P2b | Perpetual execution simulator plus paper-trading backtest | **80 tests**; conservation, determinism, funding monotonicity, liquidation and margin-sufficiency checks pass |
| A4 | Tool-history shadow fixtures | 11 fixtures, 13 tests, **11/11 semantics match through the real gateway** |
| G1 | Main-model workflow gateway | 42 tests; shadow, active, real-scorer, and `actual` token-accounting paths verified end to end |
| G2 | Reversible filtering | 65 restore tests; byte-identical round-trip through the real gateway; one reviewer-found contract violation fixed |
| E1 | Mac-local read-only measurement of the community references | laya runs on `mps:0`; readouts are diagnostics only, not a controlled comparison |
| W1 | **Real-data paper trading across four venues** | 6,554-record real PIT cohort accepted by the unmodified validator at exit 0; three venue receipts |
| B0 (T1–T4) | Frozen protocol with verified digest, order-outcome divergence audit, locked path-independent sizing, decoupled capacity | **Found and fixed a window defect that invalidated every earlier number**; the all-three-venue diagnostic spread is 271,945 → **3,964.56** (Aster appendix included), while T5's primary Binance/Bybit full-window spread is **3,517.11** and its 2026 YTD spread is **6,454.37 (>5%)**; `headline_allowed=false`; the "chaotic" conclusion retracted |
| Skill measurement + hardening | Abstention measured on 13 engineering questions; scope guard; out-of-domain diagnosis; attribution probe; temperature study | Current installed skill regression **17 tests OK**; guard `nanojev-scope-guard-v1`; **3 of the author's own claims withdrawn by an adversarial pass**. Historical aggregate counts remain in the execution log and are not capability evidence. |
| Engineering corpus + runbook | 41 contrastive pairs / 246 items over 7 families; domain-adaptation runbook | 25 corpus tests; every row validated through the real trainer validator; **no training performed** |
| Safe dedup (model-free savings arm) | Deterministic exact-duplicate removal | **Negative**: 0 removals on every repo fixture; the unchanged gateway also cannot carry such a plan |
| T10 / A5 filter-versus-rebuild protocol (2026-09-20) | Frozen five-arm paired contract, A4 manifest hash/family binding, fail-closed preflight, and synthetic placeholder receipts | **Protocol/preflight complete, READY_FOR_REVIEW**: 27 tests; `protocol_valid_not_authorized`; 5 arms / 8 families. Provider comparison outputs remain private; the public roadmap reports NanoJev-only evidence. |
| T9a three-arm readout diagnostic (2026-09-20) | Pinned base + letter / fine-tuned + letter / current head; all option permutations, full probabilities and input hashes | **Negative drop-in result**: base answers 12/13 but only 10/13 are semantic-invariant; 4/6 accuracy equals constant-true baseline, including high-confidence unsafe wrong answer. [Evidence](T9A_READOUT_COMPARISON_V1.md); 47 probe tests, full suite 523 OK (2 skipped) |
| V3 N1 contract-first benchmark (2026-09-20) | Versioned local benchmark contract, lazy harness, deterministic/content-free receipts, fail-closed zero-network check | **Implementation/verification complete, READY_FOR_REVIEW**: 25 N1 tests; offline MPS/FP32 smoke is deterministic with `network_model_calls=0`; one measured workload receipt, no performance/quality promotion |
| V3 N2 domain-pack preflight (2026-09-20) | Versioned read-only manifest/split contract, portable report identity/source paths, provenance/source-group/lineage isolation, canonical-input and protected-case checks | **Implementation/verification complete, READY_FOR_REVIEW; clean receipt absent**: 40 synthetic tests; clean synthetic CLI is not a real N2 domain-pack receipt or training authorization; blocked V1 remains unchanged and N3-S stays blocked |
| V3 N3 architecture/readout protocol (2026-09-20) | Versioned paired-arm readout ladder with frozen split roles, permutation/OOD/protected/baseline controls and fail-closed preflight | **Protocol/verification complete, READY_FOR_REVIEW**: 22 synthetic tests; real preflight is `protocol_valid_not_authorized`; no arm run, training, deployment or production pruning |
| V3 N3 synthetic control runner (2026-09-20) | Synthetic-only paired receipts for the N3 controls; no checkpoint, dataset, or network access | **Implementation/verification complete, READY_FOR_REVIEW**: 12 tests; 3 arms/8 cases/123 receipts; `synthetic_controls_passed_not_model_evidence`; zero network/model load/authorization; not model-quality evidence |
| V3 N3-S readiness preflight (2026-09-20) | Read-only pinned-evidence/review-declaration gate before real N3 paired measurement; W20 compatibility plus W21 portable report/source-path binding | **Implementation/verification complete, BLOCKED as expected**: 53 tests, including real-producer synthetic integration, strict JSON types and exact source-path binding; run3 receipt remains `n3_s_readiness_blocked` because N2 receipt is missing and four independent reviews are pending; declarations are not identity/authentication or authorization |
| V3 N4 footprint contract/preflight (2026-09-20) | Versioned FP32/FP16/INT8/INT4/distilled candidate ladder with paired resource/quality receipt fields | **Contract/preflight complete, READY_FOR_REVIEW**: 24 tests; five candidates; `footprint_contract_valid_not_authorized`; no artifact/model load, quantization, training or deployment |
| V3 N4 synthetic footprint controls (2026-09-20) | Network-free placeholder receipts and fail-closed control-path regression for the frozen candidate ladder | **Implementation/verification complete, READY_FOR_REVIEW**: 14 tests; five candidates/five receipts; `synthetic_footprint_controls_passed_not_measurement_evidence`; all physical/quality fields `null`, no authorization |
| W27 V1 corpus isolation plan (2026-09-20) | Read-only connected-component plan over pair lineage and canonical model-visible inputs; explicit cross-split/provenance blockers | **Plan-only and blocked**: run3 has 246 records/28 components/6 split-conflicted components/27 canonical cross-split groups/18 evaluation-derived records; 10 direct boundary tests; no corpus rewrite or training |
| W28 V4-S0.4 scope-binding coverage (2026-09-20) | Read-only negative tests for M1 cold and M2 warm/cold target scope binding | **32 V4-S0.4 tests; clean contract remains `metrics_contract_valid_not_authorized`; no measurement/quantization/training/deployment** |
| W29 plan receipt boundary flags (2026-09-20) | Explicit false machine-readable measurement/deployment authorization and performed fields; run3 preserved, run4 exclusive-created | **Run4 preserves all isolation counts/block reasons/components; no corpus mutation or execution** |
| W31 T15 ecosystem ledger refresh (2026-09-20) | Read-only GitHub metadata refresh with a common UTC snapshot and canonical redirect note | **17 rows reconciled; stars remain metadata only; no gate or capability claim changed** |
| W52 offline benchmark bundle + official Jev reference run (2026-09-21) | Provenance-separated local bundle (`data/jevbench_offline_bundle_v1/`, gitignored): official JevBench v1.2.4 public 231 rows + SemIf pinned-rev authored144/perturbations108/shape777 + WANLI 256 and Every 204 rebuilt by upstream pinned build scripts; manifest + `--verify` + 6 tests | **Bundle byte-identical to upstream pins (git blob SHA verified)**; official Jev scored on all 1,493 labeled rows via direct `api.typesafe.ai` (`jev-1.13.0`, the only supported channel): JevBench 0.8571, WANLI 0.7695, authored144 0.9653, perturbations 1.0, Every-154 0.9675; p50 292ms / p95 729ms; ~2.03M input tokens ≈$0.09. TypeSafe-102 track absent (no redistributable snapshot); shape777 has no gold. Aggregates publishable per the 2026-09-21 publication update; raw per-item outputs remain local (`research/official_jev_direct_run_v1.jsonl`). A one-time Vercel gateway run exists in local receipts for reference only. Diagnostic only — not an X5 milestone, not NanoJev evidence |
| W53 open-source Jev reproduction wave 1 (2026-09-22) | Local unmodified-artifact reproduction using the frozen offline bundle and common TypeSafe/native adapters: `kev-0.6b`, `openjev-verdict`/`rlcd-modernbert-151m`, `open-jev-deberta-v3-large`, `gavel-base`, `opendecision`, `laya`, `laya-typed-decisions`, `certo`, `smalljev-semantic-v9`, `semif` Qwen3.5-4B MLX4, `reflex` Qwen3.5-4B stable, `open-alternative` Qwen3.5-4B separate, `kev-4b`, `decider-2b`, `decider-0.8b`, `kev-9b` | Local labeled-row accuracy on the common 893-label subset: official Jev 0.8858, Kev-9B 0.8186, Kev-4B 0.8018, open-alternative-4B 0.7917, Decider-2B 0.7917, Reflex-stable 0.7895, SemIf-4B-MLX4 0.7671, Decider-0.8B 0.7256, SmallJev-v9 0.6540, MiniCPM seed20 0.6529, Kev-0.6B 0.6394, Gavel-base 0.6338, Laya base 0.5834, OpenDecision 0.5588, Laya typed-decisions 0.5543, OpenJev DeBERTa 0.5330, Certo 0.3572, Verdict-151M 0.3471, J-FAST seed30 0.3953. Kev-9B is the best reproduced open artifact; Kev-4B/open-alternative/Decider-2B/Reflex form the 4B tier; Decider-0.8B is the strongest sub-1B artifact. Scorecard `results/offline_bundle_comparison_v1.json`. Diagnostic only — no production or promotion claim |
| W54 local-decision vs sequential-planning boundary (2026-09-22) | Absorb the @karminski3 Jev-vs-xoshiro256++ maze experiment as a capability-boundary control: compare Jev greedy/sample, uniform random, deterministic planner/oracle, and planner-wrapped Jev on held-out maps; vary history visibility, cycle traps, detours, finite horizon, and multi-seed random baselines | **Design decision**: Jev/NanoJev is a single-step typed decision primitive, not a planner. Random can beat a locally biased controller through exploration, but finite-horizon results require multi-seed confidence intervals; Pólya recurrence does not imply good bounded-time performance. No planner or active control promotion until cycle avoidance, irreversible-action gates, and System-2 escalation are measured |
| W55–W60 open-source reproduction and context-gate measurement (2026-09-23) | Reproduced runnable public Jev-class artifacts on the frozen offline bundle; audited Winnow; added `/v1/systemone` and cascade scorer adapters; measured A4 shadow fixtures, value-oriented fixtures, deterministic safe dedup, and paired fake-provider checks | **Complete reproduction phase, not promotion**: Winnow-12B Q8 reaches 0.8824 vs Official Jev 0.8858 on the same 893-label bundle and is retained as local baseline/shadow scorer; A4 remains zero-drop at 0.99; new value fixture shows 44.9% byte / 44.1% local-BPE-token hypothetical reduction at diagnostic 0.90 with 8/8 paired answers and zero unsafe removals. No active filtering, production threshold, provider billing, or downstream model-quality claim |
| W134 valen_nano clean-label fix + re-eval (2026-09-27) | Label-contract fix applied to `correction`/`overlap_distractor`; `data/valen_nano_v2/` rebuilt (0 contradictions, eval ceiling 1.0; v1 untouched for audit) and expanded to `data/valen_nano_v3/` (12,122 train / 2,950 eval, 38.5% hard negatives); all heads re-evaluated on v2, same-device MPS bf16 | **Delivered; T167 retrain blocked on GPU (owner directive: no local-Mac training)**: clean-label ordering **sft_v2 0.8938 > rlcd_2b 0.8923 > rlcd_v2 0.8879** — the v1 0.9012 figure was partly label-noise memorization (31 eval relabels; rlcd_v2 16→15 = net loss), so v1 numbers carry up to ~1.3pp inflation. bf16 cross-device drift measured: identical model+requests gave 50/678 label flips A100↔MPS ⇒ ~±0.01 deltas are device noise, cross-device comparison needs same-device re-measurement. Winnow Platt calibration: eval ECE 0.119→0.092 (isotonic recommended next). Docs: `VALEN_NANO_V2_CONTRACT_FIX_V1.md`, `VALEN_NANO_V2_REEVAL_V1.md`, `WINNOW_CALIBRATION_V1.md` |
|| W135–W139 valen v3/v4 training + JEMM baseline + L40×2 migration (2026-09-28) | v3 batch on clean data: sft_v3 0.9156 / rlcd_v3 0.9858 / sft_text_v3 (LoRA) 1.0000 on the 2,950-question v3 eval; v4: `valen_nano_v4` (17,124 train / 4,251 eval incl. five new hard families), sft_v4 0.7944 / rlcd_v4 0.8946 / sft_text_v4 1.0000; JEMM-27B external baseline measured 0.8165 (A100) → 0.8167 (L40×2 sharded bf16) on the same eval; A100 released, workspace rebuilt on dual L40-48GB; fp32 parity re-verified (LoRA 0 flips vs MPS); public benchmark tables switched to v4 primary | **Delivered, not promoted**: synthetic saturation ≠ production performance — v4-new families still expose backend gaps (kev 0.6710, official Jev 0.6526 on v4-new only). Production head unchanged (`nano_rlcd_v2` via `?backend=valen`; `valen_lora` shadow opt-in); real-transcript labeling (547 candidates) is the remaining gate before any production switch. Aggregates publishable per the 2026-09-21 publication decision |


At the T9a checkpoint (2026-09-20): **523 tests OK (2 skipped)**. The post-T8b/T15
worktree audit reran **651 tests OK (2 skipped)**; after adding the V3 N1 harness, the N1
increment was **676 tests OK (2 skipped)**; after the V3 N2 validator, the latest
worktree regression was **715 tests OK (2 skipped)**, followed by **737 tests OK (2 skipped)**
after the V3 N3 protocol validator, **749 tests OK (2 skipped)** after the synthetic runner,
**773 tests OK (2 skipped)** after the N4 footprint validator, then **787 tests OK (2 skipped)**
after the N4 synthetic footprint-control runner, and **806 tests OK (2 skipped)** after the N3-S
readiness validator; **837 tests OK (2 skipped)** after the W20 real-producer compatibility
and fail-closed regressions; **841 tests OK (2 skipped)** after the W21 portable-report and
source-path binding regressions; **893 tests OK (2 skipped)** after the W24 T10/A5 protocol,
preflight, and synthetic-control package; **922 tests OK (2 skipped)** after the W26 V4-S0.4
contract/preflight regression; **932 tests OK (2 skipped)** after the W27 isolation-boundary
tests; **935 tests OK (2 skipped)** after the W28 scope-binding negative tests. The installed local skill
remains **17 tests OK**.
The earlier delivery baseline was 380 tests at
`be0303c`; that historical count does not describe the current tree.
T9a started at `47da116`. No purchase, broker access or live order was part of this work.

**User directive (2026-09-19): the local decision model must be integrated into the main-model workflow to increase speed and reduce token consumption.** The concrete missing deliverable is the G1 provider gateway (work package G1 in the handoff plan): a loopback gateway exposing OpenAI- and Anthropic-compatible pass-through endpoints, defaulting to byte-preserving shadow mode, failing open on every error path, with a documented measured-versus-estimated token accounting rule. Building and testing that gateway is authorized now. **Enabling active pruning in production is not**: it still requires every Track A acceptance gate below plus paired downstream-quality and net-token-cost evidence across at least three main-model families and a zero-deletion protected-segment stress suite. *(Status: G1 is delivered and verified — see the delivered table above; this paragraph records the directive, not current state.)*

**User scope decision (2026-09-19) for Track B: perpetual-contract crypto trading only, never spot**, across **Binance, Bybit, Aster, and Hyperliquid**, with RLCD training and a paper-trading backtest as explicit deliverables. Live capital remains out of scope. Consequences that must be visible in every plan: the spot-era draft (long-only, gross price-move label) must become two-sided with leverage, margin, funding, and liquidation semantics; the existing simulator refuses short sales and models no margin or funding, which is a hard blocker for perp backtesting *(resolved in P2b — the simulator now models shorting, margin, funding, and liquidation; 80 tests)*; and the previously recommended Binance Vision archive is CC BY-NC-SA 4.0 (non-commercial) with a §4.2 clause prohibiting live proprietary trading execution, so it can never support an execution phase and must be re-assessed per venue.

**User directive (2026-09-19): advance real-data simulated (paper) trading first — it is the cheaper path.** This reordered the financial track: the data/simulator/backtest plumbing was delivered end to end on **real** venue data before any RLCD training. See [Real-data paper trading V1](PAPER_TRADE_REAL_DATA_V1.md) and [Venue data licensing audit V1](VENUE_DATA_LICENSING_V1.md).

**User directive (2026-09-19): authorise Bybit/Aster/Hyperliquid as data sources and connect them.** All four venues are now connected. Two of them were blocked purely by **egress**, not by the venues: `api.bybit.com` is DNS-poisoned in this environment and `fapi.asterdex.com` refuses direct connections; routing through the local egress proxy reaches both. **This was a reviewer oversight** — the first pass tested direct connections only and wrongly concluded Aster was unreachable.

**User directive (2026-09-19): update progress and forward plan in the docs, and commit/sync important material to git.** This revision records the delivered work packages, the negative results, and the re-ordered plan below.

### The most important V2 result so far: the measurement was broken, and fixing it changed the conclusion

Over real data the same reference strategy first appeared to produce **four mutually
contradictory** results spanning **271,945 on 100,000 of capital — 2.7× the capital**:

| Run | Data | Net PnL |
|---|---|---:|
| A | Binance, before a 3-day index fix | −49,920.90 |
| B | Binance, after the fix | +21,173.77 |
| C | Bybit, same period and instruments | −77,919.27 |
| D | Aster, same period and instruments | +194,025.96 |

**A first draft of this document concluded that the strategy was "chaotic with respect to its
input". That conclusion is retracted.** Two tooling defects, not strategy behaviour, produced most
of the spread:

1. **The receipts misstated their own sample.** `--first-day`/`--last-day` were recorded in the
   receipt but never applied, so every run silently used the whole archive
   (2023-01-01 … 2026-08-31) while claiming 2024-01-01 … 2026-08-31 — the venues were never
   actually compared over the same period. It was caught by a reproduction check that could not
   pass: a "half" of the sample returned results identical to the whole.
2. **The participation cap was coupled to venue-reported volume** (`fill ≤ 10% × bar volume`;
   Aster's median daily BTCUSDT bar volume is 9,959 vs Binance's 128,882), producing Aster's
   partial fills.

With the window enforced and the cap decoupled, the same strategy over the same period gives:

| Venue | Decisions | Net PnL | Divergences |
|---|---:|---:|---:|
| Binance | 61 | −34,080.29 | 0 |
| Bybit | 61 | −37,597.40 | 0 |
| Aster | 61 | −38,044.85 | 0 |

**The all-three-venue diagnostic spread collapses from 271,945 to 3,964.56 (3.9646% of capital),
with all three venues agreeing in sign and magnitude; Aster is appendix-only in T5.** The primary
Binance/Bybit T5 table separately reports 3,517.11 for the full window and 6,454.37 for 2026 YTD,
so `headline_allowed=false`. The capacity-coupling effect alone now measures **820**, not 279,000.
See [B0 window defect V1](B0_WINDOW_DEFECT_V1.md) and [T5 report](PAPER_TRADE_REAL_DATA_V1.md).

What survives, and is honest to say:

- **The reference strategy loses money on all three venues** over this window under declared
  provisional costs. It is a mechanical moving-average crossover, not a model, and it authorises
  nothing.
- A **≈4%-of-capital** spread is still too large to ignore, so no single-venue number is a result
  yet — but it is bounded, interpretable dispersion rather than chaos. Characterising it is T5.
- **Measurement integrity was not a formality before the science here: it was the finding.** Both
  defects produced plausible numbers that had already been quoted in documents.

Also verified: position sizing is **path-independent** (quantity from initial cash and
decision-day close only), so an earlier draft's equity-compounding claim is withdrawn; and the
replay is **insensitive to the seed** while `fill_probability = 1.0` and `reject_probability =
0.0`, so a multi-seed sensitivity axis would be theatre — the meaningful axes are venue and
period.

The financial track still has **no** utility, edge, or profitability claim, and Track B's first
acceptance gate remains measurement integrity rather than model quality.

## Community references and adoption plan

Reviewed 2026-09-20. See the [Ecosystem verification ledger V1](ECOSYSTEM_VERIFICATION_LEDGER_V1.md) for evidence levels, primary URLs, UTC metadata snapshots, and comparison scope. The detailed working reference review remains local-only. The 2026-09-19 table below is retained as historical context; new numeric claims require a fresh ledger snapshot.

| Reference | Role in this roadmap | First deliverable | Adoption boundary |
|---|---|---|---|
| [tamaratran/fast-jev-compaction](https://github.com/tamaratran/fast-jev-compaction) | Track A: verbatim retention, paired tool-call/result pruning, bounded scoring requests | A4 tool-history shadow corpus and paired compression benchmark | Do not inherit its 0.5 keep threshold, character-based savings claim, or result-omission policy without validation |
| [hr98w/jev-visual](https://github.com/hr98w/jev-visual) | Track B and shared runtime: shared context, direct candidate scoring, independent-forward parity | B8 structured-versus-visual state ablation plus runtime parity specification | This is an educational visual inference project, not a financial strategy, RLCD trainer, or proof of millisecond trading |
| [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) | Shared runtime and Track B recipe: 421M ModernBERT encoder decision engine, proper-scoring GRPO-style training, script-detection routing | Read-only local install on this Mac laptop (MPS/CPU latency and memory receipt) plus zero-shot probe on frozen workflow samples | T4 latency and Jev comparison are non-local/third-party numbers; base checkpoints are near-chance zero-shot by its own disclosure; 512/1024-token context does not cover Track A long-context loads; its GRPO-style recipe is an RLCD candidate, not the definition |
| [kshetrajna12/reflex](https://github.com/kshetrajna12/reflex) | Shared runtime and model quality: Qwen3.5 direct-logits readout without trained heads, packed/batched branch isolation with numerical-equivalence tests, calibration protocol, TypeSafe-compatible API shape | WebGPU demo verification on this Mac (Safari 18+), then a reviewed direct-logits-versus-trained-heads ablation spec on frozen cohorts | Direct-logits caps Choice at 26 candidates versus our 255 contract; MMLU/Jev ECE figures are cross-source comparisons; supervised proper-scoring LoRA is not RLCD; GB10/WebGPU timings do not transfer to our MPS service |
| [TheoLeeCJ/SemIf](https://github.com/TheoLeeCJ/SemIf) | MIT reference for direct native-logit decisions, Apple MLX, serial prefix reuse, parallel suffix scoring, evidence manifests, and perturbation checks | Same-backbone direct-logit/LoRA-head/pointer-head ablation plus independent-forward parity tests on NanoJev-owned cohorts | Reuse architecture and test discipline, not TypeSafe records or upstream benchmark conclusions; preserve the 255-candidate contract |
| [JevBench](https://github.com/fstandhartinger/jevbench) / [Benchmark Heaven](https://benchmarkheaven.com/jev-models) | Independent typed-decision benchmark with public harness/tasks, four raw axes, adapters, and maintainer-held-out evaluation | Pinned NanoJev adapter → one public milestone run → external heldout submission | Evaluation-only: never train/calibrate on benchmark tasks; rank is secondary to raw axes; do not copy other systems' outputs into this repository |
| [mizorewww/laya-mlx](https://github.com/mizorewww/laya-mlx) | Apache-2.0 application/runtime reference: native MLX typed decisions, packaged Python API, batched questions, Snake demo with an explicit deterministic safety layer, opt-in compile/cache, parity/stability checks, and checksummed releases | Use its evidence format and application composition as a checklist for NanoJev's local demo/runtime acceptance; consider a separately reviewed MLX backend only after the current MPS candidate is frozen | Independent Laya port, not official upstream; author performance/gameplay claims are not NanoJev evidence; preserve `NOTICE` if code is reused; upstream weights have separate terms; no outputs or benchmark rows enter training/calibration |

After the W45 negative financial estimator comparison and the positive LoRA result, the critical path shifts to NanoJev model capability: SemIf-style readout/runtime ablations, independently authored corpus growth, and JevBench external validation. Financial work remains protocol-preserving and headline-free until a new signal/data hypothesis is reviewed. No reference authorizes provider reconfiguration, external transcript uploads, active pruning, deployment, or trading.

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
| [bespokelabsai/nimble](https://github.com/bespokelabsai/nimble) | 147 | Open recipe + open weights (Apache-2.0 adapter): Qwen3.5-9B LoRA with **contrastive data curation** (change one fact so the answer flips). Directly relevant to our Track A zero-drops blocker; the detailed review remains local-only |
| [achimala/jevinci](https://github.com/achimala/jevinci) | 19 | Creative: parallel per-pixel color decisions; confidence maps to brush width |
| [luiginotmario/postgres-Jev](https://github.com/luiginotmario/postgres-Jev) | 0 | Natural-language PostgreSQL predicates (`WHERE jev(...)`-style) |
| [sosopop/jev_stock](https://github.com/sosopop/jev_stock) | 6 | Experimental short-term stock-direction forecasting with a first-trading-day backtest script |
| [rorshopping/jev-on-a-laptop](https://github.com/rorshopping/jev-on-a-laptop) | 14 | Unofficial Jev-style parallel typed-decisions study |
| [TheoLeeCJ/SemIf](https://github.com/TheoLeeCJ/SemIf) | **1,770** | **The closest published analogue to our problem**, and its author is the source of the compression objection already recorded here. "Semantic ifs from open models": reads typed option probabilities from an open model in one forward pass. MIT, created 2026-09-16. |
| [leesk212/JEV-CPU](https://github.com/leesk212/JEV-CPU) / HF `Meanblock/JEV-CPU` | 1 (HF: 1 like) | A CPU port of SemIf: **"on a laptop CPU, no GPU"**, `Qwen/Qwen3-0.6B` in float32 — **our exact base model** — ~1 s per decision, demonstrated across **eight domains** including code-review triage and incident severity. |
| HF `com-kotobalabs/open-jev-deberta-v3-large` | 20 likes | Jev-shaped typed decisions on DeBERTa-v3-large with **choice over up to 255 options** (matching our contract, not reflex's 26 cap); ships a corpus builder, ablations and ADRs. Apache-2.0. |
| HF `mobarmg/jev-schema-scorer-deberta-v3-large` | 158 downloads | States the key principle outright: "the question text, criteria and option ids are read **at inference time, never baked into the weights**, so the same checkpoint answers new questions over new label sets **without retraining**". MIT. |
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
| local skill: 13 engineering questions | **13/13 abstained** (default 0.9); max confidence 0.736 | [Skill readiness V1](NANOJEV_SKILL_READINESS_V1.md) |
| local skill: accuracy below threshold | **3/6 = 50%**; the three errors occupy confidence ranks **1, 3, and 5** (not a claim of a top-three block or correlation) | 6 questions with fixed answers |
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

Status: **shadow library, loopback adapter, main-model forwarding gateway, reversible filtering, local `/v1/systemone` scoring, and a small→strong cascade scorer are implemented**; not globally installed, and active filtering stays **disabled by default**. See [Context shadow V1](CONTEXT_SHADOW_V1.md), [Main-model gateway V1](MAIN_MODEL_GATEWAY_V1.md), [Reversible filtering V1](REVERSIBLE_FILTERING_V1.md), [Winnow context shadow V1](CONTEXT_GATE_WINNOW_SHADOW_V1.md), and [Context-filter value V1](CONTEXT_FILTER_VALUE_V1.md).

Delivered and verified at five levels — library tests (42), CLI entry point, real-scorer end to end, process-level `active` reduction, and the `actual` token-accounting path:

- **Shadow** forwards the original bytes, passes opaque credentials through untouched, leaks no `x-nanojev-*` header upstream, and writes a content-free receipt.
- **Active** reduction works end to end and preserves system instructions, user intent, and protected segments.
- **Token accounting is honest by construction.** Savings are reported as `estimate` unless the caller supplies a provider-reported paired baseline; with one, the claim becomes `actual` and equals `baseline − provider_prompt_tokens` (verified: 412 − 300 = 112). A reduction that was genuinely sent still refuses to claim `actual` without that baseline.
- **Reversible filtering** returns a content-free restore manifest in `x-nanojev-restore-manifest`. Independent verification confirms byte-identical round-trip restoration through the real gateway and rejects tampering (`removed_segment_hash_mismatch`).
- **A reviewer-found contract violation was fixed**: the gateway could forward a reduction the caller could not restore, because its guard checked only that the manifest *built*, not that restoration *succeeded*. The gateway now performs a real round-trip against the caller's own bytes before sending, and fails open on any mismatch.

Two measured limits now bound what this can claim:

- **T8 batch scoring delivered (2026-09-20).** `MAX_SCORED=32` is now a per-batch cap, with original-order merge and per-batch error retention plus global dependency closure. The 128-segment / 128,000-byte / model-path limits remain. Real local 32/33/65-candidate probes score all candidates but still propose zero drops; 32-vs-16 partition probabilities match exactly. Large restore headers now trigger original-byte fallback before any active test reduction. This resolves the old 33-candidate bypass, **not general long-context serviceability or model quality**; see [T8](T8_BATCH_SCORING_V1.md).
- **A4/frozen-threshold proposals remain zero**, so actual token savings are still zero. The newer value fixture found Winnow can propose a 44.9% byte reduction at a diagnostic 0.90 threshold with no detected unsafe removals, but that is not a production threshold or actual provider savings. Every Track A gate below is unmet. *(Superseded 2026-09-24 by T57–T61: the same reviewer-text remained accurate when written, but the official-Jev billed upstream eval now shows 10.55–26.52% REAL billed input-token savings on dropped cases with 83–100% official scorer agreement; the acceptance gates below still govern production active-mode defaults.)* The reason the old NanoJev checkpoint still fails is measured: on 13 engineering-judgment questions the checkpoint **abstains 13/13 at a 0.9 cutoff** (an observed gated run, `research/skill_abstention_survey_gated_run_v1.json`; max confidence 0.736), and its single most confident answer is the safety-critical wrong one (the recorded historical abstention rate is **7/23 = 30.4%**; an earlier figure of 43.5% wrongly included this survey's own calls) — "safe to remove context without a validated gate" = `true` at 0.736. An independent diagnosis shows the collapse is **out-of-domain**: in-domain the median confidence is 0.779, 37.5% of questions are ≥ 0.9 and those are 97.8% correct, and confidence correlates positively with accuracy (point-biserial +0.397). The checkpoint is **technically skill-ready** (starts on demand, deterministic model output, offline, 0.27–0.33 s for 13 questions) but **not decision-ready out of domain**, and a phrase-based scope guard now marks engineering-judgment questions out of scope. **Recalibration was tried and does not repair it** — with the caveat that only a single global scalar was tested, while laya fits one temperature per (question type, option count), which is a different experiment (T9c): temperature fitted on the frozen calibration split only (T = 0.854) lifts the survey maximum from 0.736 to 0.769 — still 0.131 below the gate, so abstention stays 13/13. Answering even one question at 0.9 would need T ≤ 0.467 and the first question admitted is the safety-critical wrong one; because temperature scaling is strictly monotone it cannot reorder confidences. In-domain the fitted value produces no measurable held-out gain either (state-clustered bootstrap: accuracy delta exactly [0, 0] and all proper-loss/ECE intervals contain zero), and it is unstable across non-test splits. The 0.9 gate is not the defect and lowering it is not a repair. **The reference set was incomplete, and the architectural choice is now explicit.** A HuggingFace
search found **TheoLeeCJ/SemIf (1,770 stars)** and its CPU port **Meanblock/JEV-CPU** — the closest
published analogue to this problem, running on **Qwen3-0.6B, the same base model, on a laptop CPU
across eight domains including code review and incident severity** — plus `open-jev-deberta-v3-large`
(255-option choice) and `jev-schema-scorer` ("criteria read at inference time, never baked into the
weights"). These split decision models into two architectures: **task-fitted heads** (ours; strong
in-domain, collapses out-of-domain) versus **schema-conditioned logit readout** (theirs; claimed to
generalise to new tasks **without retraining**). T9 must be re-scoped to answer which architecture is
right **before** a training run, since the corpus work assumes the first. A cautionary measurement:
applying SemIf's readout to our checkpoint raised median confidence 0.505 → 0.899 and answered-at-0.9
from 0/13 to 6/13, but an option-permutation control showed it was a pure **letter-A position artifact**
(letter identical 13/13, description identical 0/13). The base-model comparison is the open follow-up.
The detailed Mac study remains local-only.

**Out-of-domain collapse is class-wide, not a NanoJev defect.** The reference `laya` (421M
RLCD-trained encoder) documents that its own base checkpoints are "near chance on typed-decisions
zero-shot" and calls itself "a fast base to specialise, not a zero-shot decision engine". Measured
here on the same 13 questions, laya answers 3/13 at 0.9 (median confidence 0.656, max 1.000) where
this checkpoint answers 0/13 — but it makes the **same safety-critical error more confidently**
(`safe_to_drop` = true at 0.878, and `blocked` = false at confidence 1.000). **Swapping in a
stronger open decision model converts abstention into confident wrongness**, which for a gate that
can remove context is strictly worse. The industry answer is specialise-per-task plus a router.
The detailed Mac study remains local-only.

A probe that separates content domain from surface form settles the last cheap hypothesis: padding an in-domain state with irrelevant text or rewording it moves confidence by <=0.021, while shortening an out-of-domain question to one clause does not raise it — **the barrier is the content domain, not length or wording**, so the fix is domain-adaptation training data and there is no prompt-format workaround. See [skill readiness V1](NANOJEV_SKILL_READINESS_V1.md), [abstention diagnosis V1](SKILL_ABSTENTION_DIAGNOSIS_V1.md), and [confidence attribution V1](CONFIDENCE_ATTRIBUTION_V1.md).

Also open: main-model quality comparisons, broader provenance adapters, calibrated threshold fitting on calibration data, and independent challenge families.

- Build a canonical request envelope and adapters for the major chat, coding-agent, and tool-use message formats.
- Start in shadow mode: score and log proposed removals while sending the original unfiltered request to the main model.
- Replay identical tasks with filtered and unfiltered context against multiple model families, using fixed seeds or deterministic settings where available.
- Add explicit bypasses for unsupported modalities, opaque encrypted payloads, exact-transcription tasks, legal or policy text requiring full retention, and requests whose dependency graph cannot be established safely.
- Route low-confidence cases to the full request and expose the fallback reason to the caller.

### A3. Token-efficiency evaluation

Status: **first synthetic paired-evidence slice measured (2026-09-23); broader cohorts and real main-model responses remain open.**

The new value fixture measured removal value rather than only safety. At the diagnostic `0.90` threshold, Winnow removed 44.9% of bytes / 44.1% of local-BPE tokens while preserving 8/8 deterministic paired answers and producing zero unsafe removals. This is not provider billing evidence or a real model-quality claim.

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

Status: **protocol/preflight complete, READY_FOR_REVIEW (2026-09-20); real paired comparison not started.** The compression debate (filtering versus reconstruction) is the community's most contested question and both positions are unproven. This arm makes it a frozen, paired experiment rather than a mailing-list argument.

- Arms: unfiltered control, deterministic safe deduplication, relevance filtering (our A2/A4 semantics), abstractive summary, and retrieval-rebuild. All arms see the same frozen tool-history and long-context fixtures.
- Measure: dependency-pair retention, evidence fidelity, protected-segment deletion (must be zero), downstream task success, restore cost, and **end-to-end cost per request** including scorer, rebuild, retry, and tool re-execution — provider-reported or tokenizer-based tokens only, never character estimates.
- Report paired confidence intervals per task family; a win on the aggregate that loses on a protected family is a fail.
- Fail-open is "forward the original bytes" in every arm. A summary fallback is a separate policy and is never substituted silently.
- No arm is adopted from this experiment alone; adoption still requires the Track A acceptance gates below.

### A6. Context-filter value and paired downstream evidence

Status: **first implementation and synthetic paired slice delivered (2026-09-23); expanded cohorts and real local main-model response checks are the next critical path.**

A6 extends A4/A5 from “did the gate fail open?” to “did the gate preserve the evidence needed to answer while removing measurable bytes/tokens?” The first slice uses eight self-authored synthetic cases, a deterministic fake provider, reversible in-process reductions, and content-free receipts.

Delivered artifacts:

- `research/context_filter_value_fixture_manifest_v1.json`
- `research/context_filter_value_fixture_manifest_v2.json`
- `scripts/build_filter_value_fixtures_v2.py`
- `scripts/benchmark_filter_value_v1.py`
- `scripts/local_main_model_evaluator_v1.py`
- `scripts/test_filter_value_v1.py`
- `scripts/test_local_main_model_evaluator_v1.py`
- `docs/CONTEXT_FILTER_VALUE_V1.md`
- `docs/CONTEXT_FILTER_VALUE_V2.md`
- `docs/LOCAL_MAIN_MODEL_PAIR_V1.md`
- `results/filter_value_*_v1.json`
- `results/filter_value_v2_*_v1.json`

Measured first slice:

- `0.99` frozen threshold: Winnow, Reflex, and cascade all retain every segment.
- `0.90` diagnostic threshold: Winnow removes 2,307 bytes / 569 local-BPE tokens (44.9% / 44.1%) with 8/8 paired answers preserved, zero unsafe removals, and 5/5 restore round-trips.
- Safe deterministic dedup removes 246 bytes / 74 tokens (4.8% / 5.7%).
- Reflex is safe but too uncertain; all nine eligible candidates fall back to Winnow in the measured cascade.
- V2 expanded suite: Winnow `0.90` removes 5,560 bytes / 1,431 local-BPE tokens (38.0% / 38.4%) across 25 cases with 25/25 paired answers, zero unsafe removals, and correct bypass behavior for five unsupported/unresolved families; Winnow `0.99` and Reflex remain zero-removal. The Reflex→Winnow cascade still routes all 19 eligible candidates to Winnow.

Next required work:

- an optional pinned local model backend that compares generated original/reduced responses, not just the deterministic evidence contract;
- an explicit fast-path scorer comparison so the cascade is not always delegated to Winnow;
- an architecture freeze and independent review before any release candidate or active filtering claim.

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

Priority: **next focus after Goal 1 closure (2026-09-24)** — T11–T14 negative estimator evidence stands; the gate to resume training-side work is a reviewed new PIT signal/data hypothesis. Concrete next steps are T62–T63 below: survey candidate signals/data hypotheses from the open literature, then route the best one through the existing review gate before any training.

The initial target is a bounded decision engine, not free-form market commentary and not an autonomous live-trading system. The model should emit complete probability distributions and an abstain/no-trade decision for clearly defined horizons and market states. Per the user scope decision recorded above, the instrument class is **perpetual contracts only** (Binance, Bybit, Aster, Hyperliquid), so the action space is two-sided and the state contract must carry margin, leverage, funding, and liquidation state; spot is out of scope.

### B0. Measurement integrity — now the first gate

Status: **T1–T5 implementation/verification complete; T5 is READY_FOR_REVIEW and the financial headline remains explicitly refused.**

Delivered and verified: a frozen protocol pins every run parameter with a verified digest (T1); a
post-replay order-outcome audit reports any divergence and fails the run under `--on-divergence
error` (T2, catches 11 on Aster in coupled mode, 0 in decoupled mode); sizing is proven and locked
path-independent (T3); the participation cap is decoupled from venue-reported volume (T4); and a
**defect that invalidated every earlier number** was found and fixed — `--first-day`/`--last-day`
were recorded in receipts but never applied, so runs silently used the whole archive while
claiming a narrower window.

With the window enforced and the cap decoupled, the same strategy gives **Binance −34,080.29 /
Bybit −37,597.40 / Aster −38,044.85**: the all-three-venue diagnostic spread is **3,964.56
(3.9646% of capital)**, Aster included, against **271,945 (2.7× capital)** before. The T5 primary
Binance/Bybit report separately records a **3,517.11** full-window spread and **6,454.37** for
2026 YTD; the latter exceeds the predeclared 5% threshold, so `headline_allowed=false`. The earlier
"the strategy is chaotic with respect to its input" conclusion is **retracted**. See [B0 window defect V1](B0_WINDOW_DEFECT_V1.md)
and [T5 report](PAPER_TRADE_REAL_DATA_V1.md).

Also measured: the replay is **seed-insensitive** while `fill_probability = 1.0` and
`reject_probability = 0.0`, so a multi-seed axis is degenerate by construction and the meaningful
sensitivity axes are **venue and period**. T5 attribution closes to a maximum residual of
`3.6307028494775295e-9`, but source-calendar and YTD dispersion still force an explicit headline
refusal and owner/reviewer decision before any financial conclusion.

The real-data result recorded above is now an auditable accounting/plumbing result, not a trading
edge: source-calendar mismatch, provisional economics, and the 2026 YTD spread keep
`headline_allowed=false`. No financial number — including any future RLCD result — is promoted
until T6/R1 and the required independent review are resolved. This gate precedes B1–B8.

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
| T1 | B0-0 frozen replay baseline | B | S | — | ✅ done (protocol + hashes; verified) |
| T2 | B0-A order/fill divergence state machine | B | M | T1 | ✅ done (audit catches 27 on Aster venue_volume, exit 2) |
| T3 | B0-B path-independent sizing | B | S | T2 | ✅ done (locked by test) |
| T4 | B0-C capacity decoupling | B | S | T2 | ✅ done (removes the +194k Aster artefact) |
| T5 | B0-D PnL attribution + mandatory sensitivity report | B | M | T2–T4 | ✅ **ACCEPTED 2026-09-20** by independent reviewer (W33-A): report/study hashes match, 36-cell rebuild field-identical, max residual 3.63e−9 recomputed, matrix/refusal/coverage/seed-degeneracy claims all verified; accounting evidence only, headline still refused |
| T6 | R1 decision package (feature set, 71 params, numeraire) | B | M | T5 evidence | ✅ **gate complete** (W35-A + receipt): protocol v2 frozen `ab1eec40…`, def hash `cffd4921…`, 90 params frozen, real-data PIT receipt `results/financial_t6_pit_receipt_v1.json` — 3,266 records / 11 features / unmodified validator: all 4 phases nonempty × 3 folds. T11/T14 unblocked at this gate. Limitation: Binance-only V1, venue holdout unsatisfiable |
| T7 | Aster licence resolution | B | S | owner action | ✅ **OWNER_DECIDED 2026-09-20: Aster dropped** (option 3). Request draft kept as history, not sent; no risk acceptance recorded; Aster receipts stay historical appendix only. Bybit/Hyperliquid terms still unread (W32-B) |
| T8 | A2 batch-and-merge scoring past `MAX_SCORED=32` | A | M | — | ✅ **ACCEPTED 2026-09-20** by independent reviewer (W33-B): file hashes match, batching/fail-open/global-closure code verified, 60 parity checks + 5 probe rows all identical plans, 26 calls / 392 scores confirmed, zero drops / zero savings confirmed; timeout gap (probe 30s vs default 5s) remains documented. No deployment authorization |
| T8b | Skill scope guard + measured envelope (added 2026-09-19) | A | S | — | ✅ source and installed skill copies synchronized; `nanojev-scope-guard-v1`, 17 tests, real local lifecycle checks; no production active-filter authorization |
| T8c | Out-of-domain confidence diagnosis (added 2026-09-19) | A | S | — | ✅ done (verdict: collapse is out-of-domain) |
| T8d | Temperature-scaling repair attempt (added 2026-09-19) | A | S | — | ✅ done (**negative**: monotone rescaling cannot repair an ordering failure; fitted T also unstable and all held-out deltas bootstrap-insignificant) |
| T9 | A2 gate model that proposes drops (contrastive curation protocol) | A | L | review gate | 🟢 **contract gap closed (W41-B)**: adapter contract frozen `f6607806…`; independent heldout 69 items / 23 new families, zero canonical overlap (`isolation_passed_evaluation_only`); G4 eng-heldout baseline measured (acc 0.464, answered@0.9=0). Training still requires T9d authorization action |
| T9a | Test base Qwen3-0.6B schema readout against current head and fine-tuned readout | A | S | — | ✅ **ACCEPTED 2026-09-20** by independent reviewer (W33-C): protocol/survey hashes match, all arm metrics recomputed from raw rows, both confident-wrong safety answers (0.999989/0.999974) reproduced, 4/6 = constant-true baseline confirmed. Negative drop-in result stands. No architecture promotion |
| T9b | Encoder/router architecture option (from the laya study, added 2026-09-19) | A | M | corpus exists | ✅ **done (W44-A)**: internal architecture review recommends kev-style block-causal packing + LoRA r16 + pointer head; LoRA lr 5e-5 (matches W42/J-D5 backbone-damage evidence); DeBERTa encoder gated on cheap checks |
| T9c | Grouped calibration: temperature per (question type, option count) | A | S | — | ✅ protocol/preflight complete, **READY_FOR_REVIEW**: 25 tests, clean `protocol_valid_not_authorized` receipt; no grouped fit, model load, service-temperature change or promotion |
| T8e | Model-free safe dedup arm (added 2026-09-19) | A | S | — | ✅ done (**negative**: 0 removals on every repo fixture; the unchanged gateway also cannot carry such a plan) |
| T8f | Domain-adaptation runbook (added 2026-09-19) | A | S | — | ✅ done (runbook + feasibility receipt) |
| T8g | Corpus→trainer adapter (blocker B) | A | S | review gate | 🟢 **adapter contract frozen (W41-B)**: `engineering_adapter_contract_v1.json` + checker pass; heldout/OOD satisfied by `engineering_heldout_v1` (69 items, 0 overlap); remaining: merged-adapter review + training authorization |
| T9d | Protocol §7 amendment for FP32 full-backbone on MPS (blocker C) | A | S | **owner approval** | 🟡 owner-approved; protocol v2 frozen `7d687325…02b1a` (W34-A), trainer/predictor hash pins live; formal review **ACCEPT_WITH_CONDITIONS** (W35-C): corpus v2 satisfies the clean-corpus condition (W35-B); **W42 NEGATIVE at v2 scale; W44-C v3-scale two-arm run: head-only arm WINS (heldout 0.522 mean, +5.8pp vs baseline, 3/3 seeds ≥ baseline); full arm 0.473 ≈ baseline. Next: LoRA/pointer-head per T9b Option B. Prior:**: 3-seed MPS/FP32 run per frozen protocol; train loss→1e-4 while dev CE worsened (1.13→2.70); 2/3 seeds dev-selected step 0; heldout acc 0.362 < atomic baseline 0.464. Full-backbone training at 102-question scale does not generalize. Next: J-T3 corpus scale-up or head/LoRA protocol amendment |
| T10 | A5 filter-versus-rebuild paired comparison | A | M | A4 fixtures (done) | ✅ **A5 five-arm run complete (W41-D)**: 3 local arms measured (zero removals/regressions/savings on A4 fixtures), 2 rebuild arms fail-closed because components are absent; no winner is claimable. Provider-oracle measurements remain private and are excluded from public evidence. |
| T11 | B3/R3 financial baselines (rules, logistic, GBM, CE, exact-Brier) | B | M | T5, T6 | ✅ **real fit complete (W41-C)**: 3-fold fit on frozen R1 cohort; no baseline beats base_rate significantly (fold2 dev pick significantly worse on test — reported as-is); `fit_completed_not_promoted`: feature-only runner boundary, numeric CE/Brier/boosted-stump cores, fixed instrument holdout, pinned protocol, 30 focused tests; real fit not authorized. See `FINANCIAL_BASELINES_V1.md` |
| T12 | B5 real-data validation protocol in `financial_backtest_v1.py` | B | M | T5 | ✅ **done (W43-A)**: frozen protocol `507ee3e2…`, real-data replay receipt (45 cells, byte-identical double replay, headline-free), 25+139 tests pass; implemented via new adapter module (R1 hash pin preserved). T13 unblocked |
| T13 | B6 risk gate on real data | B | S | T12 ✅ | ✅ **done (W44-B)**: frozen R1 limits verbatim on real path; 1247 decisions → 14 allow/13 reduce/1220 block (position_limit dominant; min_position_qty=0 blocks all net-shorts — declared frozen property); receipt results/risk_real_v1.json; 20 new + 151 financial tests pass |
| T14 | B4 RLCD-like estimator comparisons | B | L | T11 ✅ | ✅ **done (W45-C), honest negative**: 13 arms × 3 folds; no arm beats base_rate on test CI; correctness_pg is the expected biased negative control. Receipt results/financial_rlcd_fit_20260921_v1.json |
| T15 | E2 ecosystem verification ledger (ongoing) | shared | S | — | 🟡 running; W46 adds pinned SemIf + JevBench/Benchmark Heaven audits; later targeted audit adds pinned laya-mlx as an application/runtime evidence reference with Apache-2.0/NOTICE and upstream-weight boundaries |
| T16 | P0 three-seed relevance independent review | A | external | owner | ✅ **ACCEPTED 2026-09-20** by an independent reviewer session (W32-A): all hashes MATCH, oracle/labels 2,400/2,400 re-derived, split isolation 0 overlap, dev-only selection confirmed, metrics recomputed, report rebuilt with zero field diff. Conclusion "no promotion" stands; remark: dev evaluated at 4 points only, best always the last point |
| J-D | 根源锁定诊断 ×5（位置偏置/boolean 分解/校准分解/弃权门映射/backbone 归因） | A | S | NanoJev baselines | ✅ **all 5 complete (W41-A)**: position bias refuted; dominant boolean failure = operating-point bias; residual engineering gap = domain data/head; full-backbone capacity is not the first lever |
| J-T | 能力提升构建 | A | M | J-D done | 🟢 **major progress (W44/W45)**: head-only heldout mean 0.522; LoRA r16 mean 0.536, best 0.580; grouped calibration helps atomic only. Revised next path: SemIf-style direct-logit/pointer ablation + prefix parity + independently authored corpus v4 |
| J-M | NanoJev milestone measurement | shared | S | frozen candidate | 🔁 internal heldout remains the selection gate; bespoke community-wide comparison is retired in favor of pinned JevBench public milestone + maintainer-held-out submission |
| X0 | External benchmark contract and data quarantine | shared | S | T15 | ✅ **done (W46-A)**: contract `a3cb3579…` freezes JevBench v1.2.4 pin, type mapping, timing/cost scope, quarantine flags; 7-source data quarantine manifest; zero real task rows read |
| X1 | NanoJev JevBench adapter | A | M | X0 | ✅ **done (W46-A)**: `jevbench_adapter_v1.py` self-hash-bound to contract; synthetic-only mode enforced; 12 hand fixtures (incl. 255-option); 34 tests pass; milestone/fetch paths hard-gated for X5 |
| X2 | SemIf architecture transfer ablation | A | M | T9d-v4 | ✅ **done (W47)**: base native readout 0.536–0.623 competitive but 21/23 order flips (not deployable); fine-tuning damages native readout (J-D5 replicated); LoRA+pointer honest negative (mean 0.415 < baseline). Readout is NOT the binding constraint → X4 corpus growth is higher-value axis |
| X3 | Shared-prefix runtime parity | shared | M | X2 | ✅ **done (W48)**: packed tree-mask path opt-in; parity gate PASSED (max Δp 7.75e-7 ≤1e-5, zero flips, 77 questions incl. 255-opt); warm speedup 1.4×–5.1× (k=255: 5.79s→1.14s); 19 new tests green |
| X4 | Corpus v4 + larger heldout | A | L | X2 diagnosis | ✅ **done (W48/X4 audit)**: corpus v4 4,320 rows/36 families; heldout_v2 312 rows/104 groups; zero canonical overlap; CC0/local oracles; no external outputs |
| X5 | JevBench public milestone | shared | M | X1 + frozen candidate | 🟡 **public diagnostic allowed/done for seed20 (W51-Y)**: 231/534 public rows only, accuracy 0.6537; official full milestone remains one-shot after internal freeze; report raw axes and limitations, not just composite rank; always paired with the custom deep benchmark scorecard |
| X6 | External heldout submission | shared | external | X5 + internal gates | ⬜ maintainer-held-out evaluation for release candidate; publish NanoJev result by linking the external source |
| X7 | Dual-track benchmark scorecard | shared | M | X5 protocol + J-D7 | 🟡 **seed20 scorecard complete (W51-Y)**: public JevBench diagnostic 231-row accuracy 0.6537 plus custom heldout_v1 0.7536 / heldout_v2 0.5224; public diagnostics are allowed on unfrozen candidates but are not promotion evidence; no composite merge, no public-row tuning, no heldout leakage. W52 adds the external reference point on the same 231 public rows: official Jev 0.8571 (direct `jev-1.13.0`, sole channel) |
| T9d-v5 | LoRA retrain on corpus v4 | A | M | X4 ✅ | ✅ **done (W50-B, negative)**: 3 seeds + gradient checkpointing; heldout_v1 mean 0.454 (v3 LoRA mean 0.536), heldout_v2 mean 0.331; cov@0.9≈0, heldout_v2 protected errors 0/3/2; no J-E1/X5 promotion |
| J-C | Calibration + selective-risk serving layer | A | M | T9d-v5 | ✅ **done (W50-A), honest negative**: layer built (23 tests); LoRA s18 no threshold reaches ≥0.9 selective acc above ~5% coverage; fit thresholds fail OOD transfer (0/8 heldout_v2); 20 protected errors @0.9 on heldout_v2; choice all unreachable → OOD scope signal is the prerequisite, not thresholds |
| J-DP | Domain packs | shared | L | J-E1 frozen candidate | ⬜ per-domain train/dev/calibration/heldout/OOD isolation; no shared eval sets; seed-reshuffle is not independence |
| J-P | Offline typed SDK + packaging | shared | L | J-C | 🟡 **spec done (W50-C)**: `docs/OFFLINE_SDK_SPEC_V1.md` — pack=checkpoint+manifest+policy+receipts, hash-verify at load, shadow-only until real OOD scope signal exists, N-1 rollback + dual kill-switch, 10 acceptance gates; implementation pending frozen candidate |
| J-D6 | v5 failure decomposition | A | S | T9d-v5 ✅ | ✅ **done (W51-A)**: aggregate-only receipt `ad8beb62…`; heldout_v2 choice accuracy falls with k (4=.367, 5=.198, 6=.188, 8=.083, 12=.042); v4 train has no choice k=6/12; score-5=.062 and weak seed agreement show semantic shift plus candidate-set mismatch, not data volume alone |
| J-D7 | MiniCPM v6 post-capacity shift diagnosis | A | S | J-A4b ✅ | ✅ **done (W51-U)**: generic shift diagnostic over 3 seeds; heldout_v2 choice still falls by k (4=.617, 5=.469, 6/8/12≈.333/.167/.333) and score stays ≈.44; seed agreement is only .490/.471 for choice/score. Capacity helps but candidate-set/domain semantics remain the binding failure |
| J-D8 | Sequential-planning boundary and controller safety | shared | M | J-D7 + W54 | 🟡 **principles absorbed, measurement pending**: local typed decisions must not be marketed as planning; add finite-horizon multi-seed random baseline, cycle/revisit diagnostics, detour traps, irreversible-action gate, and explicit System-2/planner escalation. A Jev answer can rank the next action, but the controller owns memory, search, loop breaking, rollback, and stopping |
| J-CV5 | independently authored corpus v5 repair arm | A | L | J-D7 | 🟡 **seed20 complete, not promoted (W51-W)**: v4 base + 8 repair families, 5,280 items / 880 pairs; audit passed. Remote seed20 hash-verified and CUDA→MPS parity passed; heldout_v1 0.754, heldout_v2 0.522 with 15 confident errors. Need separate seed21/22 stability review before promotion |
| J-PERF | optimized inference/runtime path | A | M | J-CV5 candidate freeze | 🟡 **bulk+shared-prefix diagnostic passed (W51-AE)**: public subset wall 284.46s → 65.64s bulk / 69.72s serial-shared (4.33x/4.08x); serial-shared p50 0.075s, p95 1.283s, argmax flips 0, max Δp 7.09e-6; official speed gate and KV/runtime design remain open |
| J-FAST | small fast-backbone arm | A | L | J-PERF baseline | 🟡 **unblocked after OSS reproduction (W60)**: marker_before made option embeddings near-duplicates (0.937 vs 0.416 cosine); bundle official Jev direct 0.886 vs MiniCPM seed20 0.653 vs JFast seed30 0.395, JFast 10.4x faster than MiniCPM; marker_after amendment drafted. Next evidence is a fast-path comparison against SemIf/Kev/Decider on value fixtures before any custom retrain |
| J-OSS | open-source Jev-class reproduction on macOS | A | L | J-PERF/J-FAST evidence | ✅ **done (W60)**: runnable candidates reproduced on the frozen 1,720-row / 893-label bundle; Winnow-12B Q8 selected as local strong baseline (0.8824 vs official 0.8858), slow Open-Jev/Nimble/jev-local paths timeout-eliminated or deferred, public aggregate scorecard delivered. No artifact promotion or training claim |
| J-A4a | stronger-base compatibility ladder | A | M | J-D6 | 🟡 **MiniCPM gate passed (W51-C)**: pinned Apache-2.0 rev `12a3808…`; real MPS FP32 forward, 9.10GB peak, 294 LoRA modules, zero-adapter Δlogit=0. Qwen3.5-4B rev `851bf6e…` config resolves but needs reviewed language-backbone adapter (`hidden_size` is under `text_config`) before real-weight probe |
| J-A4b | stronger-base controlled arm | A | L | J-A4a partial pass | 🟠 **3-seed stability complete (W51-T)**: MiniCPM5-2B seeds 17/18/19 completed on A100; all checkpoint hashes, merge parity, and CUDA→MPS owned parity passed. Heldout_v1/v2 mean accuracy 0.691/0.544 vs v5 means 0.454/0.331, but heldout_v2 confident errors mean 27.3; root-cause diagnostics next, no promotion |
| T17 | Controlled local main-model paired evaluator | A | M | A6 first slice | ✅ **done (W61/W63/W65)**: `local_main_model_evaluator_v1.py` supports deterministic evidence and pinned local-generation backends; value runner treats original-pass/reduced-fail as unsafe. After chat-template and answer-normalization fixes, Qwen3-0.6B / Qwen2.5-3B / Qwen3.5-4B preserve 8/8, 17/17, and 18/18 originally successful answers under Winnow @0.90 |
| T18 | Expanded value/stress fixture suite | A | M | A6 first slice | ✅ **done (W61/W66)**: V2 has 25 self-authored development cases; V3 adds a 12-case held-out synthetic cohort. Winnow diagnostic 0.90 removes 38.0% on V2 and 30.7% on held-out V3 with zero paired regressions/unsafe removals in deterministic and Qwen3.5-4B local-generation checks |
| T19 | Diagnostic threshold policy | A | S | T17 + T18 | ✅ **done (W61)**: pinned policy JSON/docs declare 0.99/0.95/0.90/0.85 as diagnostic-only; runner records policy hash and review-mode fails closed on undeclared scorer/cascade thresholds. V2 Winnow sweep: 0.95 removes 35.7% bytes, 0.90/0.85 remove 38.0%, all with 25/25 paired answers and zero unsafe removals; no production threshold selected |
| T20 | Fast-path scorer/cascade comparison | A | M | T18 | ✅ **done (W61)**: compared safe dedup, Winnow, Reflex, SemIf-4B, Decider-0.8B/2B, Kev-4B/9B on V2. Kev-4B direct matches Winnow's 38.0% byte / 38.4% local-token reduction with 0 unsafe removals and 25/25 paired answers; Kev-4B→Winnow cascade offloads 13/19 candidate states while preserving the full reduction. Latency still favors Winnow-only in this harness; no production fast path selected |
| T21 | NanoJev vNext architecture freeze | shared | M | T17–T20 | ✅ **done (W62)**: architecture target frozen as NanoJev controller + optional Kev-4B fast path + Winnow-12B strong fallback; Kev-4B is reference fast path (13/19 routes, full V2 reduction), Winnow remains quality baseline, custom NanoJev scorer deferred. Decision records still forbid active filtering/production threshold/release candidate |
| T22 | Track A review/release gate | shared | owner/internal | T21 | ✅ **done (W67, owner-authorized internal review)**: checklist + deterministic replay + scorer stress receipt + V2/V3 held-out results reviewed; `docs/TRACK_A_REVIEW_DECISION_V1.md` records a conditional pass for the Track A evidence packet. This is not external independent-review equivalence; active filtering, production threshold, and release candidate remain unauthorized |
| T23 | Active-mode canary protocol/preflight | A | S | T22 | ✅ **done (W68)**: `research/active_mode_canary_protocol_v1.json` + validator/preflight define a three-phase loopback→shadow→paired canary ladder; status `protocol_valid_not_authorized`, active filtering and provider calls remain disabled |
| T24 | Phase0 loopback active-mode dry-run | A | S | T23 | ✅ **done (W69)**: `run_active_mode_phase0_v1.py` passed 5 loopback cases — eligible reduction + restore manifest, shadow control, kill switch, scorer error, malformed sidecar; provider_calls=0 |
| T25 | Real-context holdout intake protocol | A | S | T23/T24 | ✅ **done (W70)**: `real_context_holdout_protocol_v1.json` + validator define local-only raw storage, manual labels, V2/V3 isolation, privacy stop conditions, and shadow→localgen→optional provider evaluation order; collection remains `not_collecting` |
| T26 | Real-context holdout manifest builder/validator | A | S | T25 | ✅ **done (W71)**: `real_context_holdout_manifest_v1.py` builds content-free manifests from local case metadata, enforces hashes/pointers/labels/coverage/credential scan, and refuses overwrite; no raw cases collected |
| T27 | Real-context deterministic shadow evaluator | A | S | T26 | ✅ **done (W72)**: `run_real_context_holdout_shadow_v1.py` consumes manifests + local raw files, runs deterministic shadow scoring, checks protected/required drops and restore round-trips, emits content-free receipts; provider_calls=0 |
| T28 | Real-context paired local-generation evaluator | A | S | T27 | ✅ **done (W73)**: `run_real_context_holdout_localgen_v1.py` reads manifest + local contract files, builds deterministic-shadow reduced requests in-process, compares original/reduced via deterministic or pinned local-generation backend, and flags paired regressions/unsafe evidence loss |
| T29 | Real-context end-to-end preflight | A | S | T28 | ✅ **done (W74)**: `run_real_context_holdout_preflight_v1.py` chains protocol → manifest → shadow → localgen into one receipt; supports deterministic or pinned local-generation backend and `--require-ready`; provider_calls=0 |
| T30 | Real-context collection authorization runbook | A | S | T29 | ✅ **done (W75)**: `real_context_collection_authorization_v1.json` + validator/runbook define owner-scope fields, operator checklist, forbidden actions, stop conditions, and rollback; status `template_ready_not_authorized` |
| T31 | Multi-wire Phase0 active-mode dry-run | A | S | T24/T30 | ✅ **done (W76)**: Phase0 expanded to 10 loopback cases covering OpenAI chat, Anthropic messages, OpenAI Responses, baseline accounting, unsupported path, no-reduction, shadow, kill switch, scorer error, malformed sidecar; provider_calls=0 |
| T32 | Phase0 stress/denial coverage | A | S | T31 | ✅ **done (W77)**: Phase0 expanded to 17 cases adding dependency closure, uncertain/invalid scorer output, scorer timeout, oversized restore manifest, oversized request body, and internal header stripping; provider_calls=0 |
| T33 | Phase1 shadow authorization template/runbook | A | S | T32 | ✅ **done (W78)**: `phase1_shadow_authorization_v1.json` + validator/runbook define measurement-only shadow scope, owner fields, metrics, forbidden actions, and stop conditions; status `template_ready_not_authorized` |
| T34 | Phase1 synthetic shadow dry-run | A | S | T33 | ✅ **done (W78)**: `run_phase1_shadow_dry_run_v1.py` passes 8 loopback shadow cases across supported wire formats, headers, unsupported path, kill switch, and scorer error; provider_calls=0 |
| T35 | Phase1 readiness gate | A | S | T34 | ✅ **done (W78)**: `evaluate_phase1_readiness_v1.py` aggregates Track A, Phase0, Phase1 auth/dry-run, and holdout/collection gates; status `ready_for_owner_phase1_authorization` |
| T36 | Minimum-reduction gate (borrowed) | A | S | T35 | ✅ **done (W80)**: `GatewayConfig.min_reduction_bytes` / `--min-reduction-bytes`; verified reduction below the floor forwards original bytes with `forward_reason=below_min_reduction`, no restore manifest; 2 new gateway tests, 49/49 pass |
| T37 | Staged scorer-state fitting (borrowed) | A | S | T36 | ✅ **done (W81)**: `fit_scoring_payload` degrades judge view full → contents<=2000/500/120 → collapsed_context; `max_scorer_payload_bytes` on gate + `--max-scorer-payload-bytes` on gateway; `state_stage` in batch receipts; impossible budget fails open `scorer_state_budget_exceeded` |
| T38 | Local scorer service stability pass | A | S | — | ✅ **done (W82)**: `check_local_services_v1.py` probes NanoJev lifecycle service (8765 + 8876–8890 fallback) + optional scorer endpoints; `--smoke` posts one tiny evaluate; `LOCAL_SERVICES_RUNBOOK_V1.md` covers restart via `nanojev_skill.py health --start` |
| T39 | Winnow-12B Q8 as NanoJev default scorer | A | S | T38 | ✅ **done (W83)**: winnow-server running on 8091 (`/v1/systemone` verified, ~0.28s/state); wired into health check, runbook start command, and gateway `--scorer systemone` usage; still shadow/loopback only |
| T40 | Gateway ↔ live Winnow end-to-end shadow | A | S | T39 | ✅ **done (W84)**: `run_gateway_winnow_shadow_v1.py` passes 7 cases with the real HTTP scorer — eligible proposals on all 3 wire formats, no-sidecar/unsupported/kill-switch/internal-headers verified; scorer_calls=4, mean ~0.31s |
| T41 | Winnow concurrency/cache-isolation stress | A | S | T39 | ✅ **done (W84)**: `run_winnow_stress_v1.py` — 20 calls × 4 workers: deterministic, per-state consistent, clean isolation (0.998/0.991 vs 0.001/0.0002), p50 ~0.46s under load |
| T42 | Live Kev-4B → Winnow cascade | A | S | T39 | ✅ **done (W85)**: `run_cascade_live_v1.py` — Kev-4B (8092) confident fast path (0.95/0.05), simulated outage routes all states to Winnow strong (0.951/0.004); `provider_calls=0` |
| T43 | Local stack keepalive | A | S | T38 | ✅ **done (W86)**: `start_local_stack.sh` idempotently starts lifecycle service + winnow + kev-4b then runs the health check; logs to `/tmp` |
| T44 | Cascade scorer in gateway CLI | A | S | T42 | ✅ **done (W87)**: `--scorer cascade --scorer-url <fast> --scorer-strong-url <strong>` wired through `build_scorer`; live gateway shadow over Kev→Winnow passes 7/7 (`gateway_cascade_shadow_v1.json`) |
| T45 | Active-mode loopback with live cascade | A | S | T44 | ✅ **done (W88)**: `run_gateway_live_active_v1.py` — live cascade proposes + applies drops on all 3 wire formats; restore manifest + round-trip verified; no raw text in header; scorer ~0.64s/call |
| T46 | Live cascade fixture eval vs Winnow-only | A | S | T44 | ✅ **done (W89)**: V3 held-out manifest via `benchmark_filter_value_v1.py --arms cascade systemone control` — cascade matches Winnow decisions exactly (1830B, 0 unsafe, 0 regressions) but is slower on MPS (p50 796ms vs 290ms; Kev-4B Qwen3.5 lacks fast DeltaNet kernels); fast-path hit 4/8 |
| T47 | Stack operator sheet | A | S | T43 | ✅ **done (W90)**: `docs/LOCAL_STACK_OPERATOR_V1.md` — one-page daily ops: services, start/check/stop, scorer choice, gateway flags, regression receipts, boundaries |
| T48 | Phase1 shadow measurement (synthetic scope) | A | S | T35 | ✅ **done (W91)**: owner-granted `phase1_shadow_scope_synthetic_v1.json` (≤25 requests, shadow only, no provider/real data); `run_phase1_shadow_measurement_v1.py` measured 12 held-out cases through live Winnow — 0 unsafe, 0 changed bytes, 5 proposed / 7 retained |
| T49 | Unified nanojev service | A | S | T43 | ✅ **done (W92)**: `serve_decisions.py` is now the single entrypoint — `/api/evaluate` + `/v1/systemone` (`?backend=winnow|kev|cascade`) + `/v1/context-gate` shadow eval + aggregated `/api/health`; one port 8876, backends internal |
| T50 | Agent-facing API contract | A | S | T49 | ✅ **done (W93)**: `docs/NANOJEV_SERVICE_API_V1.md` — endpoints, request/response shapes, advisory + shadow + content-free contract rules for local agent CLIs |
| T51 | launchd keepalive | A | S | T49 | ✅ **done (W93)**: `deploy/launchd/` three user agents (service + winnow + kev backends), RunAtLoad+KeepAlive; `install.sh` / `--uninstall`; live verified all up under launchd |
| T52 | nanojev-local-service skill | A | S | T50 | ✅ **done (W94)**: `integrations/codex-skill/nanojev-local-service/` — SKILL.md + `nanojev_eval.py` (jev-eval-compatible CLI, `--backend winnow/kev/cascade`, `--context-gate`, auto-start); installed to `~/.codex/skills/` + `~/.local/bin/nanojev-eval`; live parallel call with official jev verified |
| T53 | Service stability hardening | A | S | T51 | ✅ **done (W95)**: determinism probe added (`--determinism`: identical `/v1/systemone` called twice must match); earlier "nondeterminism" re-tested — actually different payloads; same input is bit-identical across 6 calls |
| T54 | Agent adoption + feedback channel | A | S | T52 | ✅ **done (W95)**: skill synced to ~/.codex ~/.claude ~/.agents ~/.config/devin ~/.copilot skills dirs; `nanojev-eval --feedback` + content-free call log at `~/.local/state/nanojev-eval/log.jsonl`; global CLAUDE.md updated |
| T55 | Traffic-split router (jev-route) | A | S | T54 | ✅ **done (W96)**: `jev_route.py` — deterministic sha256-bucket split (default 30% local), `--local-ratio`/`JEV_ROUTE_LOCAL_RATIO` configurable, local-down → official fallback flagged in output, routing logged content-free |
| T56 | Quality-fallback routing mode | A | S | T55 | ✅ **done (W97)**: `jev-route --mode quality` — local answers first, escalate to official when top-prob < `--escalate-below` (default 0.95); `route.escalated` + `local_confidence` recorded; live verified (confident stays local, ambiguous choice escalated to jev-1.13.0) |
| T57 | E2E real local upstream | A | M | T45 | ✅ **done (W98)**: `run_e2e_local_upstream_v1.py` — gateway ACTIVE mode in front of real Winnow-12B chat upstream (loopback-authorized); 37 fixture cases, real `prompt_tokens` from upstream accounting: **24.47% savings, 0 answer regressions** (temp=0 pinned after one sampling flake); `results/e2e_local_upstream_v1.json` |
| T58 | Real open-corpus traffic eval | A | M | T57 | ✅ **done (W99)**: `fetch_open_corpus_v1.py` sampled 100 real ultrachat multi-turn convs (HF datasets-server, public, provenance recorded) → auto-sidecar → e2e. **Finding: 0.05% savings on coherent chat (scorer retains relevant history — correct), vs 24.47% on noise-heavy fixtures.** 4/100 judge disagreements all had 0 bytes dropped = upstream/judge noise, not filtering. Safety on real traffic: verified. `results/e2e_open_corpus_v1.json` |
| T59 | Official-Jev billed upstream eval | A | M | T58 | ✅ **done (W100)**: `run_e2e_official_upstream_v1.py` — owner-authorized real provider accounting. 12 fixtures → official jev-1.13.0, 22 calls / 8,513 billed input tokens; dropped cases **10.55% real billed savings**; official agreed with 4/5 local drops (the 1 miss: borderline 0.89 vs 0.90 threshold). Fixed part-pointer bug found by this eval. `results/e2e_official_fixtures_v1.json` |
| T60 | Joint candidate scoring + real-noise corpus | A | L | T59 | ✅ **done (W101)**: real-data eval exposed a judge-view flaw — per-candidate scoring let stale content look "relevant" because its paired earlier user turn stayed in view (0.02-0.08 false-relevant on real stitched noise). **Fix: one batch state × N per-candidate questions** (joint scoring) + earlier user turns moved out of `conversation` into `earlier_user_messages` + per-segment `uncertain_score` retain (was whole-request bypass). Result on stitched real corpus: **20.81% real billed savings, official agreed 7/7 drops**. Also ~N× cheaper (prefix-shared call). 71/71 tests green. `results/e2e_official_stitched_v1.json` |
| T61 | Scale-verified official eval + corpus regression | A | M | T60 | ✅ **done (W102)**: stitched official eval scaled to 70 cases total — **26.52% billed savings** on second batch, 10/12 official agreement (disagreements at 0.79-0.88 borderline, honest judgment calls). Full-corpus regression with joint scoring: 200 cases, 12 drops, 0.79% savings, disagreement rate ~3.5% ≈ upstream noise floor. Evidence chain complete. `results/e2e_official_stitched_v2.json`, `results/e2e_open_corpus_v2.json` |
| T62 | Track B signal/data hypothesis survey | B | M | — | ✅ **done (W103)**: `docs/TRACK_B_SIGNAL_HYPOTHESES_V1.md` — ranked 6-hypothesis shortlist from literature+OSS survey; **H1 (extreme negative funding → +0.5% 24h return, asymmetric tail effect) recommended first** — explains T11/T14 linear-baseline nulls; H2 (funding-z fade) + H3 (basis convergence) as controls; carry explicitly falsified by external pre-registered study; liquidations blocked (no historical source, same gap in OSS) |
| T63 | Route best hypothesis through review gate | B | S | T62 | ✅ **done (W104)**: protocol + owner auth (`financial_signal_hypotheses_protocol_v1.json` sha 04c08caf, auth receipt) → `financial_signal_hypotheses_v1.py` replay on real 6,554-record cohort. **Results: H1 falsified on our label (45.42% vs 45.5% base, p=0.98 — external +0.5%/24h effect doesn't transfer to 25bps/1d binary); H2 direction-right but insignificant (48.2%, p=0.44, n=164); H3 basis convergence 82.2% p≈0 but partially tautological (extreme values regress mechanically — needs sharper formulation).** `results/financial_signal_hypotheses_v1.json` |
| T98 | Daily cohort → 15 assets | B | M | — | ⏸ blocked: local archive only has 5 symbols; expansion needs ~1,760-file fetch — needs owner approval for bounded fetch |
| T99 | Hyperliquid replication | B | M | T81 | ✅ done (W117): `financial_signal_hyperliquid_v1.py` on local HL data — **STRONGEST validation yet: spec-gated signal (funding≥80pct + BTC-20d-trend) replicates OFF-Binance at ρ=+0.061 (vs Binance +0.042), FDR-pass; ungated = powered null → the trend gate is real and portable, not a Binance artifact**. Top-decile +209bps gated. HL-Binance funding spread null (consistent w/ T80) |
| T100 | Literature round 3 | B | S | T64 | ✅ done (W117): top new arm = **distance-from-high momentum** (t=+4.93 on large caps, predicts positive sign on our cohort type); liquidity-conditioned reversal/momentum interaction (Zaremba); XS funding-rank carry triple-killed (don't test); **our settlement sawtooth is NOT published — plausibly small novel finding**; hourly reversal documented in lit |
| T101 | Distance-from-high + Amihud arms | B | M | T100 | ✅ done (W118): `financial_signal_dfh_v1.py` — **SECOND real signal: dfh20 (dist from 20d high) ρ=+0.031 FDR-pass, quintile spread +157bps t=5.03, AND adds ρ=+0.119 inside f2b2 cell** — direction prior confirmed (near-high → continue); Amihud liquidity flip REJECTED (opposite direction). Caveat: window-sensitive (dfh10/1d-horizon null), part market-beta |
| T102 | XS pilot 10-asset cohort | B | M | — | ✅ done (W118): 5 new symbols (ADA/DOGE/LINK/LTC/DOT) × 3mo fetched (60/60), 890-rec pilot cohort built, XS-rank arms null-but-underpowered (10 assets thin — viability check only); fetch pipeline proven extensible |
| T103 | Composite dfh×f2b2 spec v2 test | B | M | T101 | ✅ done (W119): `financial_signal_composite_v1.py` — **verdict: KEEP spec v1 frozen**. dfh gate fails upgrade bar: within-base contrast p=0.036 misses FDR; dfh is NOT orthogonal (r=+0.30 w/ funding_pct, redundant w/ BTC-trend leg); +24% fewer trades for ~zero per-trade gain; fold-inconsistent. dfh = descriptive overlay, not spec component |
| T104 | Full-history XS cohort (10 assets) | B | M | T102 | ✅ done (W119): 440/440 files fetched (klines+funding only), `perp_pit_xs_v1` = **13,380 recs / 10 symbols / 2023-01→2026-08**. XS funding-rank carry NULL (prior confirmed, 1/4 folds). **XS dfh-rank POSITIVE: +54bps/day-spread, t=3.34, 4/4 folds same sign — purely cross-sectional (within-asset ρ≈0)** — new candidate signal |
| T105 | Neg-regime reversal OOS confirmation | B | M | T96 | ✅ done (W119): `financial_signal_reversal_oos_v1.py` — **CONFIRMED**: frozen test replicates on confirm half (+125bps, FDR-pass, n thin=50); regime exclusivity holds on confirm half only (dev positive-control +36bps) — honest stability caveat |
| T106 | XS dfh robustness battery | B | M | T104 | ✅ done (W120): **CONDITIONAL SPEC-CANDIDATE** — survives net-costs (net +13.5bps/day, Sharpe 0.99, turnover 27%), all horizons/depths/lookbacks, placebo-collapsed; **but regime-gated: uptrend +111bps vs downtrend −14bps** — fails regime-free, merits spec status only WITH BTC-trend condition |
| T107 | Bybit dfh external replication | B | M | T101/T104 | ✅ done (W120): partial replication, tail/gate-concentrated — quintile +154.7bps FDR-pass; **XS top2/bot2 +54.5bps t=3.60 ≈ Binance +54.06 (near-identical)**; inside-gate dfh ρ=+0.138 vs Binance +0.119; spec gate +219bps on-vs-off FDR-pass. Pooled Spearman weaker at powered n |
| T108 | Benchmark v3 (XS+dfh blocks) | B | M | T104 | ✅ done (W120): 21 cells, 14 PASS; daily+intraday byte-identical to v2; **dfh20 PASS daily + XS PASS (+54bps, 4/4 folds) → cross_scale `replicated_both_scales`**; XS funding carry FAIL as predicted; deterministic ×2 |
| T109 | Spec draft `xs_dfh_carry_v1` | B | S | T106 | ✅ done (W121): `research/financial_signal_spec_xs_dfh_v1.json` — status `spec_candidate_pending_review`; froze the MEASURED gate (ret20>0, not SMA20 — caught spec-drift before it entered a frozen doc); all evidence sha256-pinned |
| T110 | Forward ledger v2 (XS sleeve) | B | M | T109 | ✅ done (W121): `financial_forward_ledger_v2.py` + XS refresh + health probe; backfilled 1362 dates (net +10.8kbps); **launchd chain updated + installed + kickstart exit 0** — both sleeves now tracked daily. Current book: long BTC+LTC / short DOGE+DOT, gate ON |
| T111 | Master-gate joint system sim | B | M | T106/T110 | ✅ done (W121): `financial_signal_master_gate_v1.py` — **combined gated system +160% net / Sharpe 1.43 / mdd 17.9% / active 53%** vs always-long +137%/0.69/69.2%; gate adds value (+17.75 vs −1.28bps/day conditional); sleeves 50% overlap (not independent alpha); 2026 weakest (S0.72 — fading late-window, consistent w/ instability caveat) |
| T112 | Mega cohort (301 symbols) + XS decile arms | B | L | T104 | ✅ done (W122): imported sibling project's `data/futures` (301 syms × 2021-01→2025-12, funding+OI for 52) → `perp_pit_mega_v1` = **287,514 recs / 277 syms / 5yr incl. 11 delisted**. **XS dfh replicates at 25× cross-section: +68.9bps t=3.31 FDR-pass, 4/5 folds, gate-on +192.6 vs gate-off −63.7**; XS mom20 same pattern; XS funding carry +76bps but 3/5 folds → NOT YET (positive at scale, contradicts null prior, unstable); survivorship control passes |
| T113 | Sibling project cross-validation | B | S | T112 | ✅ done (W122): their regime gate = **BTC_atrp (vol) not trend** — alternative master-gate parameterization worth testing; their crowded-long entry counter-evidence (f3 ρ=−0.233) is different object (leveraged entry timing, no funding PnL) — tension noted not resolved; their pullback>breakout confirms anti-chase direction of dfh short leg; methods to borrow: symbol-block bootstrap, plateau IS-argmax test, regime occupancy reporting |
| T114 | Master-gate shootout (ret20 vs atrp) | B | M | T113 | ✅ done (W123): `financial_signal_gate_shootout_v1.py` — **ret20 wins decisively (mean Welch t=+4.32 vs atrp_d −0.83); atrp is refinement not replacement**; joint trend+vol +308bps median-split (4/5yr); their frozen 0.025 threshold never fires post-2022 here (occupancy 13.4%); dfh_pct ρ FLIPS sign under ret20 (+0.080/−0.127); exception: funding carry is vol-gated |
| T115 | XS funding carry deep-dive | B | M | T112 | ✅ done (W123): **verdict (c) noise/concentration-luck — +76bps is a 2021 event (ex-2021: −0.24bps t=−0.02), top-3 symbols = 65.6% of PnL**; NOT spec candidate; sub-finding: F1×D3 'stealth rally' cell +160bps (near-high without funding euphoria); churn 0.66/day falsifies 'slow=cheap' hypothesis |
| T116 | OI arm at 52 symbols | B | M | T112 | ✅ done (W123): **OI null REPLICATED at 10× symbols** — metrics.csv is twice-monthly snapshots only (1st/15th); all actionable arms |t|<0.7; 3 FDR 'survivors' proven day-composition pseudo-replication (+456bps pooled → −3.9bps per-day); arm conclusively closed at this granularity |
| T120 | Campaign ledger v3 sleeve | B | M | T119 | ✅ done (W124): live on 10-asset universe — honestly reports LOSS (0.432x; composite entry doesn't replicate on majors-only); launchd chain updated (v1+v2+v3), not yet installed |
| T121 | 4h roll-engine port | B | L | T119 | ✅ done (W124): **roll mechanics NEGATIVE at 4h** (rolled < flat < BTC hold both arms); ratchet-stop turns continuations into scratch exits; our entry > their E1 on identical engine |
| T122 | Campaign param grid (36 cells) | B | M | T119 | ✅ done (W124): **parameter_lucky — median cell loses to BTC hold, default rank 2/36, ex-2021=0.219x**; robust residue: pullback-band + real trail stop structure; T119's 34.4x DOWNGRADED to not-credible |
| T123 | Principled campaign rebuild | B | M | T122 | ✅ done (W125): **0/9 vol-scaled cells beat BTC hold; ex-2021 0.557x** — campaign wrapper confirmed dead; entry filter still beats no-filter but doesn't clear hold |
| T124 | Mega 4h cohort + hourly arms | B | L | — | ✅ done (W125): 1.28M recs / 283 syms. mom_4h reversal replicates at 70× scale (universal, NOT neg-regime — T96 exclusivity was small-sample) but fades to 0 in 2025; settlement sawtooth replicates −5bps; 4h funding carry flips sign (no generalization) |
| T125 | ML probe walkforward ridge | B | M | T112 | ✅ done (W125): **ML_VIABLE — ridge_all 4/4 folds OOS positive ρ+spread (pooled +378bps top-decile); interaction dfh20×btc_ret20 carries effect** — model learned the gate itself; gated-training arm fails. First viable model formulation |
| T126 | 2025 decay autopsy | B | M | T124 | ✅ done (W126): **premise partially wrong** — XS dfh NOT faded in 2025 (2nd-best year, best normalized since 2021); mom_4h reversal = REAL decay (dead on incumbents too); funding carry never stable (2025 rates pinned ~0 → starved); dispersion drives XS spread size (corr 0.87) |
| T127 | Ridge v2 + net tradability | B | M | T125 | ✅ done (W126): **ridge_min3 wins — LS book net +7.28bps/day, Sharpe 0.666, 4/4 years positive**; 14-feature expansion BREAKS consistency (2/4); only dfh20×btc + xs_rank_mom coefficients stable; long-only marginal (+1.75bps/day) |
| T128 | Ledger v4 ridge sleeve | B | M | T125 | ✅ done (W126): live daily scoring on 10-asset universe; backfill net +11.6kbps; **OOS flag: 2026 ranking spearman −0.096 on thin universe (model trained on 277-asset XS)** — gated book still net-positive; chain updated, not yet installed |
| T129 | Live universe 10→30 assets | B | L | T128 | ✅ done (W127): top-30-by-qvol universe (20 new incl. 1000*-alias fix for PEPE/SHIB/BONK), `perp_pit_xs_v2` 46k recs, xs2 refresh + `--universe xs_v2` on ledgers; **v4 book spearman OOS: −0.027→+0.144, net equity 1.126→1.494**; found+healed refresh tail-rewrite fragility (lost bars on transient failures) |
| T130 | Domain-matched ridge retrain | B | M | T128 | ✅ done (W127): **top30-qvol domain-matched training fixes the inversion** (pooled ρ −0.057→+0.025, daily +0.056 on live window); xs10-only halves inversion; recent-window-only BACKFIRES (−0.107 — pre-2023 history helps); recommendation: refit v4 scorer with ridge_min3@top30 full-history — **APPLIED: `--model ridge_min3_top30` added to ledger v4 + wired into launchd chain on xs_v2 universe** (verified: OOS book +83bps realized vs +5bps for ridge_all@xs_v2) |
| T134 | Mirror regime-switching book | B | M | T132 | ✅ done (W128): mirror confirmed at book level (mom +16.3/−11.1 × rev +9.8/−7.5); M-only gated book net +141.8% Sharpe 0.744 — reversal side real but not monetizable after costs |
| T135 | Age-conditioned model/book | B | M | T131 | ✅ done (W128): **age → book overlay YES (+8.2bps/d, Sharpe 0.778 vs 0.245), model features NO** (ρ up but OOS net down); gt365 top-dfh short isolated +7.31bps/d all-years-positive |
| T136 | Refit policy evaluation | B | M | T128 | ✅ done (W128): expanding_monthly wins (+24.2bps/d, 4/4yr); recency windows hurt; v4's full-history refit already correct — monthly cadence preferable |
| T137 | Valen (开源 Jev 复现) 本地移植 | A | L | — | ✅ done (W129): **full SFT+RLCD pipeline reproduced on MPS** — only ONE code edit (float64→CPU in distributed.py); 107 tests pass; SFT smoke 15s / RLCD smoke 38s end-to-end; Preview-2B ckpt inference verified (sha-matched). Feasibility: warmup/head-only overnight ✅, LoRA SFT needs fp32 (bf16 backward unsupported on MPS) → ~1-3k samples feasible, 10k+ not; real-scale RLCD impractical on laptop. `external/valen/` + `MPS_PATCH_NOTES.md` |
| T138 | NanoJev 决策数据→Valen 格式转换 | A | M | T137 | ✅ done (W129): jevbench bundle 是 evaluation_only 禁用→改用 `context_relevance_v1`+`oracle` 自有 CC0 数据（3360 rec，hash 分裂 2682/678）——`data/valen_nano_v1/` |
| T139 | Valen baseline 评测（未训头/Preview） | A | S | T137 | ✅ done (W129): 未训头 0.522 / Preview-2B 0.510 on our eval；OmniJev-0.8B 本地 MPS 跑通（1 行 patch），jevbench 231 题 acc **0.524** vs 官方 Jev 0.857 / NanoJev 0.654 |
| T140 | SFT warmup 头训练（本地数据） | A | M | T138 | ✅ done (W129): **首个自训练决策头**——0.8B backbone 冻结、head 524K 参数、bf16 MPS、4 epochs 40 分钟、CE 0.73→0.45（未收敛） |
| T141 | 训练后头 vs 基线评测（acc/ECE） | A | S | T140 | ✅ done (W129): **trained 0.645 vs untrained 0.522 vs Preview 0.510**（+12pt）；保守 dropper（precision 1.0 / recall 0.27——过滤场景的安全方向）；过自信，需更多 epochs / LoRA / 温度缩放 |
| T142 | RLCD pilot（本地数据） | A | M | T141 | ✅ done (W132): rlcd_v1 pilot 完成（0.8746）后被 T144-eval 反超判定；**A100 上 nano_rlcd_v2（RLCD@sft_v2 收敛头，300 步）acc 0.9012/brier 0.1349/nll 0.2058——推翻"收敛后 RLCD 无效"的先前结论，pilot 是欠训练**；ckpt 已回传并上线 sidecar `?backend=valen`（生产最强头） |
| T144 | 收敛版 SFT（多 epochs + 校准） | A | M | T140 | ✅ done (W131/W132): sft_v2 12ep 收敛 acc 0.8894/nll 0.238/brier 0.157；A100 上 nano_rlcd_v2 再 +1.2pp→0.9012 且校准更优（rlcd_v2 conf≥0.99 的 459 题仅 1 错） |
| T145 | 金融 regime 头 SFT | B | L | T143 | ✅ done (W132): fin_sft_v1 45,192×3ep 收敛，9k 题 eval overall **0.6202**（regime_gate 0.9997=读题/fwd5_bucket 0.329/xs_outperform 0.532≈chance）；fin_rlcd_v1 0.6086 无增益——**域内头胜 OmniJev-4B（0.435）但真预测只略胜随机，收益饱和点在数据不在训练** |
| T147 | 修复 lifecycle 非 JSON bug | A | S | — | ✅ done (W130): **根因=端口漂移**——skill 默认 8765 被无关进程占用返回 HTML；服务一直在 8876 正确运行。修复+6 份 skill 副本同步+lifecycle 端到端验证通过（含对抗路径） |
| T148 | ledger v5 年龄叠加 sleeve | B | M | T135 | ✅ done (W129): 回填 +15.2kbps 净（Sharpe 0.653）；**诚实发现：固定宇宙的年轻池已空（新币都老过365d）——袖子短期只能做空腿，直到宇宙扩新币** |
| T149 | 全 ledger 健康探针 | B | S | T148 | ✅ done (W130): `check_forward_ledgers_all_v1.py` 覆盖 9 账本；7 个链上全新鲜；发现尾行 cumulative 漂移伪影（append-only+迟到 bar 重定价）；v4_xs2 孤儿账本标注 |
| T150 | OmniJev-2B/4B 本地对标 | A | M | T139 | ✅ done (W130): **完整阶梯 0.8B 0.524 / 2B 0.654(=NanoJev) / 4B 0.688**——模型规模单调有效但都被官方 Jev ~0.87 压住；4B noul 被缺失的 bias 项低估（clone 早于 v1.1） |
| T143 | 金融 regime 决策数据集（Valen 格式） | B | M | T133 | ✅ done (W129): `data/valen_fin_v1/` 20k train/3k eval——fwd5 桶 choice + XS跑赢 noul + regime choice（±1% deadband）；schema 验证通过；标签尾部偏斜已记录 |
| T151 | 宇宙扩 30→50（复活 v5 年轻池） | B | L | T148 | ✅ done (W130): 20 个 2025-26 新上市币加入（POWER/BEAT/MMT/RIVER…），年轻池 2025-08 起无缝覆盖；**但诚实发现：50 资产 v5 回填反而更差（1.84x vs 2.70x）——2026 新币多头腿 −50.8bps/leg 大亏，年龄叠加是 2023-24 上市潮现象非稳健效应** |
| T152 | OmniJev-4B bias 修复复测 | A | S | T150 | ✅ done (W130): v1.1 bias 正确 backport 但 acc 不变（0.688）——bias 只修 ECE 不修 accuracy，4B noul 0.514 是真 OOD 行为；阶梯定论 0.524/0.654/0.688 |
| T153 | 研究状态文档整合 | B | M | — | ✅ done (W130): TRACK_B 重写为结构化状态文档（确认机制/可货币化袖子/坟场/基建/开放问题）+ handoff 补齐 W119-W130 |
| T154 | 探针尾行漂移范围收窄 + xs2/xs3 refresh 补缺 | B | S | T149 | ✅ done (W131): cumulative 重算检查限定非尾行 → 9 账本全绿（迟到 bar 重定价降为 informational note） |
| T144-eval | sft_v2(12ep) vs v1 vs RLCD 收敛对比 | A | S | T144 | ✅ done (W131): **结论反转——收敛 SFT 反超 RLCD**：sft_v2 acc 0.8894/nll 0.238/brier 0.157 vs rlcd_v1 0.8746/0.251/0.167 vs sft_v1(4ep) 0.6445——RLCD 先前优势实为"救了欠拟合"；sidecar 已切到 sft_v2 checkpoint 上线 |
| T155 | spec-v2 晋级门预声明 | B | S | T109 | ✅ done (W131): `research/spec_v2_promotion_gate_v1.json` + `docs/SPEC_V2_PROMOTION_GATE_V1.md`——min 90 OOS 日/Sharpe≥0.5/mdd≤1200bps/集中度上限/否决分支全预声明；**发现阻塞项：live v2 账本门是 SMA20 而 spec 合约是 ret20（15.6% 天数分歧）→ T159 修复** |
| T159 | v2 账本门漂移修复（SMA20→ret20 合约一致臂） | B | S | T155 | ✅ done (W131): `financial_forward_ledger_v2.py` 加 `--gate`；新合约账本 `forward_ledger_v2_ret20.jsonl` 全史回填 **net +12,187bps vs sma20 +11,018——合约门更优**；老账本保留为对照；已进每日链。修程中两次静默丢 edit（edit tool 对热文件不可靠）→ heredoc 直写验证 |
| T156 | RLCD 头接服务 scorer | A | M | T142 | ✅ done (W131): **已完成端到端集成**——`valen_head_scorer_v1.py`（同 eval acc 0.8746 vs winnow 0.8599，ECE 0.038 vs 0.119）→ sidecar `valen_head_server_v1.py` :8093（单线程服务不阻塞）→ `SCORER_BACKENDS["valen"]` 一行接线 → `ai.nanojev.valen.plist` launchd 常驻；`/v1/systemone?backend=valen` 实测通过（稳态 ~0.8s/req） |
| T157 | OmniJev-4B 金融 regime OOD 检验 | A/B | M | T143 | ✅ done (W131): **不迁移**——500 recs/1500 题：overall 0.435 vs chance 0.361；真实结果题 fwd5_bucket 0.174（低于 25% chance）、xs_outperform 0.484≈chance；regime_gate 0.648 是文本复述非金融判断（标签自监督于 state 里印着的 btc_ret20）；**对照：域内 0.8B SFT 头仅训 200 步已 0.611**——架构通用，决策先验必须域内训练 |
| T158 | xs3 refresh + v5@xs_v3 进每日链 | B | S | T151 | ✅ done (W131): plist 补 `refresh_binance_xs3_v1.py` + `financial_forward_ledger_v5.py --cohort perp_pit_xs_v3 --ledger forward_ledger_v5_xs3`（4 个 supplement 全带）→ 已装 launchd；snapshot 验证年轻池 19 标的存活；**xs_v3 保持对照臂定位（T151 回填更差），不晋升主 v5** |
| T160 | A100 周末训练冲刺计划 | A | — | T144 | ✅ done (W131): `docs/A100_WEEKEND_PLAN_V1.md`——9 项任务按依赖排序（P0 环境冒烟→A1 fin 续训→A2 RLCD@sft_v2→A3 Valen 100k 全量四阶段[旗舰]→A4-A8）；本机 fin_sft_v1 已安全暂停（10,130/45,192 ckpt 存），MPS patch 对 CUDA 无影响 |
| T95 | BTC→alt hourly lead-lag | B | S | T89 | ✅ done (W116): `financial_signal_leadlag_v1.py` — **no directional lead-lag**: contemporaneous ρ 0.75-0.86, lagged residual ≈0 mildly negative (structural beta decay + shared mean-reversion). Real finding: big-BTC bars carry ~50% elevated alt vol (FDR-pass) — diagnostic, untradable |
| T96 | Mom reversal + settlement contamination | B | S | T89,T93 | ✅ done (W116): `financial_signal_reversal_v1.py` — reversal sub-fee (gross +3.8bps < fees); **regime-dependent: negative-funding regime tilt +14.9bps FDR-pass**; settlement-window exclusion flips funding_pct PASS→FAIL (marginal) — contamination note verified |
| T97 | Ledger launchd live | B | S | T90 | ✅ done (W116): job installed + firing daily 07:10. **Hit + fixed real TCC issue**: `/usr/bin/python3` (CLT) has no Documents folder grant under launchd — switched interpreter to `.venv/bin/python` (service-proven context); last exit code 0. Manual chain also ran: refresh to 09-23, 1 bar+1 exit appended (BTC/ETH time-stop closed), all assets out |
| T92 | Benchmark v2 on 5mo data | B | S | T89,T91 | ✅ done (W115): rerun on 17,960 recs — funding_pct PASS at ρ=+0.026 (halved vs 2mo but placebo-clear, cross_scale=replicated_both_scales); mom_1h/4h now significant **negative** ρ (intraday reversal); taker_pct PASS; basis/rv FAIL. Daily block still byte-identical |
| T93 | Post-settlement tilt | B | S | T89 | ✅ done (W115): `financial_signal_settlement_v1.py` — **real pattern, correct characterization = settlement-centered sawtooth**: weak ±2h around settlement, positive mid-cycle; dips under BOTH funding signs (not payment-mechanical); strongest at 2h (−6.27bps p≈0), high-rv only; untradable after fees → **diagnostic: post-settlement bars contaminate other signals** |
| T94 | Funding-sign regime gates | B | S | T89 | ✅ done (W115): `financial_signal_funding_regime_v1.py` — **scale asymmetry resolved: hourly funding signal lives ONLY in positive-funding regime (ρ +0.048 pos / −0.007 neg; f2b2 arm sign-flips in neg regime −2.6bps); daily signal is regime-ROBUST (ρ +0.041 pos / +0.074 neg!) — BTC-trend remains the operative gate at daily scale**. Daily pooled ρ=+0.0422 exactly reproduces frozen spec — construction validated |
| T89 | Intraday 5-month expansion | B | S | T87 | ✅ done (W114): 61 files → **17,960 hourly records** (5 sym × Apr-Aug 2026; one real venue gap at 06-29 correctly skipped by contiguity guards). **Key honest result: funding replication WEAKENED with 3× data** — ρ +0.033→+0.015 (2mo window was ~87% positive-funding; Apr-Jun had 44-50% negative months → level effect is positive-regime-specific). New emergent: post-settlement 0-2h −4.8bps FDR-pass (single effective test); taker_pct → PROVISIONAL_PASS. **Funding-follow is regime-conditional at hourly scale too** |
| T90 | Ledger launchd automation | B | S | T83 | ✅ done (W114): `ai.nanojev.forward-ledger.plist` (daily 07:10: refresh→ledger→snapshot→log), install.sh wired, `check_forward_ledger_v1.py` health probe (exit 0 ok / 1 stale/missing), lint clean. **Not loaded — install left as owner action** |
| T91 | Benchmark v2 cross-scale | B | M | T82,T88 | ✅ done (W114): `financial_signal_benchmark_v2.py` — daily block byte-identical to v1 + intraday layer (7 cells) + cross_scale block. NOTE: v2 ran on the 2-month intraday set before T89 expansion; re-run it for the 5-month numbers |
| T86 | Funding-timing hourly arms | B | S | T85 | ✅ done (W113): `financial_signal_funding_timing_v1.py` — pilot nulls were underpowered; on expanded 7,300-rec set: **funding sign × distance all 4 buckets FDR-pass (+16bps diff), A4 top-vs-bottom funding quintile +10.7 vs +0.8bps t=3.26 FDR-pass; distance-only/post-settlement timing = null**. Funding signal is a level effect, not settlement microstructure |
| T87 | Intraday expansion 5×2 | B | S | T85 | ✅ done (W113): 40/40 files fetched (5 symbols × Jul+Aug 2026), **7,300 hourly PIT records** at `data/perp_pit_intraday_v1/` (label base rate 0.342); honest-abort + manifest-merge + contiguity guards added |
| T88 | Hourly baseline arms | B | S | T87 | ✅ done (W113): `financial_signal_intraday_v1.py` — **funding_pct replicates at 4h scale: rho +0.033, p=0.0049, FDR+placebo → PROVISIONAL_PASS**; taker_pct FDR-pass but placebo-fail → NO_SIGNAL; mom/rv/basis all null. Same carry mechanism at both scales |
| T83 | Forward paper-trade ledger | B | M | T81 | ✅ done (W112, subagent): `financial_forward_ledger_v1.py` — append-only content-light ledger (`results/forward_ledger_v1.jsonl`), idempotent replay, full spec state machine (entry f2b2+BTC-trend, exit 5-bar/regime-off, funding-accrual PnL). Dry-run: 6,668 bars, 228 entries, 226 closed, **2 currently open (BTC+ETH since 09-18, unrealized +655/+542bps gross)**. Plus `refresh_binance_cohort_v1.py` — pulled 114 supplement records (→2026-09-22) from Binance Vision; ledger is now LIVE-forward-capable |
| T84 | Agent dogfooding wiring | A | S | T53-T56 | ✅ done (W112, subagent): `nanojev_usage_report_v1.py` (tool/route/escalation/feedback stats), runbook dogfooding section, dogfood blocks appended to all 5 skill dirs. Current usage: 26 self-test calls only — real usage now discoverable via report |
| T85 | Intraday data scaffold | B | M | — | ✅ done (W112, subagent): `fetch_binance_intraday_v1.py` + `build_perp_pit_intraday_v1.py` — real BTCUSDT 1h data fetched (Aug 2026: 744 bars + 93 settlements), **716 hourly PIT records** at `data/perp_pit_intraday_v1/` with distance-to-settlement feature + 4h forward label. Lead-lag/funding-timing now feasible (pilot scale) |
| T82 | Unified signal benchmark | B | M | T81 | ✅ done (W111): `financial_signal_benchmark_v1.py` + `docs/FINANCIAL_SIGNAL_BENCHMARK_V1.md` — one contract (pinned cohort/5d label/frozen folds/180-PIT/5bps+funding cost), 10 cells, **placebo label-shuffle controls (3 seeds)**, determinism (byte-identical x2), always-long baseline, FDR family, PASS/ECON_ONLY/FAIL verdicts. Scorecard: **6 PASS** (funding, basis, fxb, abnvol weak, f2b2, f2b2_btc, tri), 2 FAIL (rv, taker), 1 ECON_ONLY (fmom). New arms plug into the same family |
| T81 | Seal signal as review candidate | B | S | T80 | ✅ done (W111): `research/financial_signal_spec_v1.json` — `crowded_long_carry_follow_v1` frozen: full entry/exit definition, evidence refs w/ receipt hashes, residual risks, forbidden interpretations, next-gate = RLCD review (conditioned state feature / sparse policy; new protocol + owner auth required before fitting) |
| T80 | Binance↔Bybit funding spread | B | S | T79 | ✅ done (W110): `financial_signal_crossvenue_v1.py` on local pinned Bybit funding (4,072 8h settlements ×5 symbols) — **all null**: spread ρ=0.0096, deciles/f2b2-contrast/test-fold all n.s. Venue settlement rates near-synchronous → no signal. Cross-venue arm closed |
| T79 | Regime-gate arms on f2b2 cell | B | M | T78 | ✅ done (W110, 3 subagents + replay): `financial_signal_regime_v1.py` — **trend gates sharpen the signal strongly: BTC-uptrend gate +228.6 vs −52.3bps (t=4.36, p=1.3e-5 FDR), own-asset 20d-trend +225 vs +3.3 (t=3.5 FDR), high-vol×up-trend +252 vs +43 (FDR)**; funding-streak and extreme-funding splits null. Fold-level: gates fix f0/f2 but **f1 test window is negative under all gates — period effect not fully removable**. Conditional-strategy subagent: tri-gate +294.8bps/trade but only 110 trades (too sparse); double-gate +172bps n=302 defensible, tail-driven. Venue audit: **Binance↔Bybit funding spread IS locally feasible** (~4,070 aligned 8h settlements ×5 symbols); Aster owner-blocked, Hyperliquid lacks mark/OI, lead-lag needs intraday (absent). `results/financial_signal_{regime,conditional}_v1.json` |
| T78 | Supervised ridge on continuous label | B | M | T72 | ✅ done (W109): `financial_signal_fit_v1.py` — 6 pct-features + funding×basis cross, frozen-fold ridge. **Honest negative: MSE never beats base (1.009-1.019); rho unstable across folds (+0.009 / −0.102 / +0.026 — fold1 actively reverses); top-decile tail works in 2/3 folds (+137/+296bps) but collapses in fold1 (−304bps)**. Signal is pooled-real but **temporally unstable** — era-dependent, consistent with the falsification-lab prior. Not model-ready as a linear global fit |
| T68 | funding×vol conditional grid | B | S | T67 | ✅ done (W108): `financial_signal_grid_v1.py` — edge concentrates in high-vol regime: vol_t2 top-vs-bottom funding +151.5 vs −37.3bps, t=4.35, **FDR-pass**; low vol → no contrast. Per-asset: SOL/XRP strong, BTC/BNB null |
| T69 | horizon/threshold robustness | B | S | T68 | ✅ done (W108): `financial_signal_robustness_v1.py` — **12/12 cells survive FDR**; rho grows 3d→10d (0.035→0.049); 10d top-90th +344.6bps vs +45.3 rest. Not a parameter accident |
| T70 | funding momentum + funding×basis | B | S | T69 | ✅ done (W108): momentum ρ=0.0004 (noise — level not change); **funding×basis joint: f2b2 +164.4bps vs f2b0 −8.9bps, p=0.004 FDR-pass — the signal is really "crowded longs at a premium" (both high)** |
| T71 | real harness + frozen risk gate | B | M | T70 | ✅ done (W108): `financial_signal_gated_run_v1.py` — conditional long/flat through unmodified backtest+RiskEngine on bound cohort: **12/15 cells beat always-long, total +4,760 vs −26,366** (win = avoiding bad windows); 4 gate-blocked decisions, 0 liquidations, byte-identical replays |
| T72 | OI arms on v2 cohort | B | S | T71 | ✅ done (W108): `financial_signal_oi_v1.py` — OI adds nothing (all null); **funding effect replicates on independent v2 cohort** (f2 col 202-217bps) |
| T73 | Track A post-fix regression | A | S | — | ✅ done (W108): full 37-case local e2e post joint-scoring — **24.47% savings, 0 regressions, identical to pre-fix** (`e2e_local_upstream_full_v2.json`) |
| T74 | signal runner unit tests | B | S | T72 | ✅ done (W108): `test_financial_signal_runners_v1.py` — 9 tests (stat helpers + 5 runner schema/ planted-signal checks), all pass |
| T75 | Track B research-state doc | B | S | T72 | ✅ done (W108): `docs/TRACK_B_RESEARCH_STATE_V1.md` — full verdict table, current signal understanding, vetoed/blocked list, boundaries |
| T76 | ops sweep | A | S | — | ✅ done (W108): services up + deterministic + smoke pass; eval log 26 entries (own test calls only — no external agent usage yet) |
| T67 | Net-of-cost backtest of the signal | B | M | T66 | ✅ **done (W108)**: `financial_signal_net_backtest_v1.py` — 5bps taker + funding accrual + non-overlapping 5d holds on real cohort. **Long arm survives costs: +151.8bps net/trade (gross 178), vs always-long +60.8 — ~2.5× unconditional edge; but median −27bps (right-tail skew), win 48%, short arm negative (−58bps), per-asset split (SOL/XRP +116, BTC/ETH negative — edge concentrates in less-efficient names)**. Signal is real but thin alone → valid as feature/conditioner, not standalone strategy. `results/financial_signal_net_backtest_v1.json` |
| T66 | Label-redesign experiment (T66) | B | M | T65 | ✅ **done (W107) — FIRST REAL SIGNAL**: `financial_signal_labels_v4.py` — the 25bps/1d binary label was indeed destroying signal. On continuous targets: **funding pct → 5-day forward return Spearman rho=+0.042, t=3.17, p=0.0015, SURVIVES BH-FDR; top-decile funding → +190.8bps 5d vs +22.5bps rest (t=4.78, p=2e-06); holds in test folds (rho 0.051)**. Direction = funding-FOLLOWING, matching carry literature (opposite of the falsified fade). Same funding arm on 1d binary label: null — label was the problem. `results/financial_signal_labels_v4.json` |
| T65 | Round-3 replay: H6-H9 corrected-direction arms | B | M | T64 | ✅ **done (W106)**: `financial_signal_hypotheses_v3.py` — all 6 arms NULL (p 0.40-0.61, none survive FDR). Even literature's strongest daily-horizon results fail to transfer: abnormal-volume (peer-reviewed −0.5%/day) → ours +0.014pp wrong-sign; basis momentum (t=7.10) → null. **Cumulative: 15+ falsified arms across 3 rounds → conclusion: the 25bps/1d binary label itself likely destroys signal (base rate 45% ≈ coin flip); next rational step is LABEL redesign (continuous/vol-normalized forward return, longer horizon) before more feature arms.** `results/financial_signal_hypotheses_v3.json` |
| T64 | Parallel push: v2 arms + lit round-2 + V2 re-eval | B/A | L | T63 | ✅ **done (W105, 3 subagents)**: (a) `financial_signal_hypotheses_v2.py` — H3-sharp **falsified** (50.9% sign agreement ≈ coin flip → confirms H3's 82% was tautology), H4 vol-expansion 1.8× but direction-neutral n=57, H5 noise; nothing survives FDR. (b) Literature round-2: **top new hypotheses = funding-FOLLOWING (external lab found fade literally backwards), abnormal-volume direction (peer-reviewed −0.5%/day on Binance data), basis momentum (Chi et al t=7.10), vol-regime conditioning**; crypto-edge-search falsification lab killed ~167 hypotheses → strong null prior. (c) V2 fixtures re-eval post joint-scoring: **identical metrics** (no regression, fix verified live). `results/financial_signal_hypotheses_v2.json`, `results/e2e_official_fixtures_v2_new.json` |
| T161 | A100 周末冲刺执行 | A | L | T160 | ✅ done (W132): 全部训练完成——fin_sft_v1(3ep) 0.6202、nano_rlcd_v2 **0.9012**（生产头）、nano_sft_2b 0.8614→+RLCD 0.8953（2B 不及 0.8B）、a3 5k 多模态头 0.515（通用头不迁移）；fin_rlcd_v1 0.6086。**同题对标：官方 Jev jev-1.13.0 在 valen_nano 678 题上 0.7341（662 答/16 超时）vs rlcd_v2 0.9012，+16.7pp**；反向镜像：rlcd_v2 在 jevbench 231 题仅 0.307。fla+causal-conv1d 内核修复后训练提速 ~25-37×、678 题 eval 45s。README 三版已发布聚合阶梯 |
| T162 | valen_nano_v1 错误分析 | A | S | T161 | ✅ done (W133): **发现数据生成器标签矛盾**——v1 `correction`(keep) 与 v1 `overlap_distractor`(drop) 归一化后是同构 state 却标签相反：427/3360=12.7% 记录受污染 → **eval 天花板 0.9572（pointer-aware；早期 0.9322 估算过度合并了 pointer 索引），当前 0.9012 剩余约 5.6pp 真实空间**。错误结构：96% 集中在 4 个 correction 系题型；与官方 Jev 交叉：共同错 27（真难/噪声）、仅我方错 36（correction 类 34 条可修复）、仅官方错 149。校准干净（conf≥0.99 错率 0.2%）。**优化前必须先修标签契约** |
| T163 | 修复 context_relevance_v1 生成器标签契约 | A | S | T162 | ✅ done (W134, subagent): **根因=模板碰撞**——v1 生成器的 `correction`/`overlap_distractor` 两臂在全部 6 family 下产出逐字节同构模板，标签挂在声明臂而非 state（该缺陷早已记录在 `context_relevance_v1_protocol.json` 的 invalid_predecessors）。修复=按陈旧事实契约重打标：**仅当 state 中已有取代它的当前值时才 drop；孤证陈旧事实 keep**。316 条重打标（226 条进 valen 消费 split），污染 427→0，**eval 天花板 0.9572→1.0**。产出 `data/context_relevance_v2_seed20260919/` + `data/valen_nano_v2/`（2682/678，同 hash 分裂契约）+ `validate_valen_label_consistency_v1.py`（0 violations）+ 6 tests + `docs/VALEN_NANO_V2_CONTRACT_FIX_V1.md` |
| T164 | valen_nano 数据集重建 + 难负例扩产 | A | M | T163 ✅ | ✅ done (W134, subagent e90676d9): `gen_context_relevance_v3_v1.py`（16 kinds=14 沿用+`near_duplicate_evidence`/`correction_confirmed` 两难负例族；correction/overlap_distractor 改为 current_in_state→drop / sole_evidence→keep 双变体，v1 bug 结构性不可能复现）→ `data/valen_nano_v3/` **train 12,122 / eval 2,950**（v3 贡献 9,440/2,272）；归一化状态多样性 ~1,550（v1 仅 ~48）；0 violations；16 tests OK |
| T165 | eval 扩容与硬化 | A | M | T163 ✅ | ✅ done（并入 T164）：eval 678→**2,950** 题、hard_negative=38.5%、group 隔离验证 train∩eval=∅；天花板修正至 1.0 后扩容目的转为区分度 |
| T166 | LoRA backbone text 阶段（干净数据） | A | M | T164+GPU | ✅ done (W135, A100 2026-09-28)：`nano_sft_text_v3`（init sft_v2 收敛头，LoRA 21.6M + head 524K，3ep=36,366 states，bf16 CUDA+fla 内核）→ **v3 eval fp32 acc 1.0000**（2950/2950，brier ~9e-11，零 borderline 预测）。决定性结论：head-only 的"天花板"是冻结特征不足，解冻 backbone 把生成器规则族学透——**v3 合成 benchmark 已饱和**，区分度耗尽，后续评测需真实/OOD 数据 |
| T167 | 修正数据上的 SFT→RLCD 重训 | A | M | T164 (T166 可选) + **GPU** | ✅ done (W135, A100)：`nano_sft_v3` head-only 6ep=72,732 states（20.3 sps，CUDA 内核生效）→ fp32 acc **0.9156**；`nano_rlcd_v3`（init sft_v3/latest，300 步，group16）→ fp32 acc **0.9858**（RLCD **+7.0pp**，远超 v1 的 +1.2pp——证明此前增益小是标签噪声压制）。fp32 跨设备 parity 实测 **50→1/678 flips**（LayerCast 路线有效，部署用 fp32 MPS）。产物：`external/valen/output_v3_20260928.tgz`（526MB 本地备份，6 ckpt+3 eval metrics+日志） |
| T168 | Winnow-12B 校准修正 | A | S | — | ✅ done (W134, subagent 76013eca): 全量 3,360 条自训数据拟合——**温度缩放失败**（T=1.914 反而 ECE 恶化 0.121→0.135，Winnow 失真是非对称的：keep 极端过自信+drop 中区间欠自信，单标量 logit 修不了）；**Platt（a=0.778,b=+1.496）eval ECE 0.119→0.092、Brier 0.128→0.099、acc 还 +0.3pp**；isotonic 诊断 0.026 提示残差是函数形状问题 → 推荐 per-domain isotonic/查表校准 bundle（J-P 方向）。receipt `results/winnow_calibration_v1.json`，doc `WINNOW_CALIBRATION_V1.md`；未改生产配置 |
| T169 | A100 收尾/成本决策 | ops | S | T161 | ✅ done (W139)：v3+v4 全部训练/评测收尾后 **A100-80G 已释放**（先退卡→整机停机，产物此前已全部回传本地）；算力切换为 **L40-48GB ×2**（ssh `-p <port> root@<gpu-host>`，key `<ssh-key>`），环境按 `docs/L40_MIGRATION_V1.md` 重建完成 |
| T170 | valen_nano_v4 数据集 + v4 难族生成器 | A | M | T164 | ✅ done (W136/W137)：`gen_context_relevance_v4_v1.py`（新五族：工具配对/跨指针/长稀释/话题切换/对抗重述）→ `data/context_relevance_v4_seed20261115/`（eval 1,301 + dev 199，0 violations，training_allowed=false）+ `data/valen_nano_v4/`（train 17,124 = v3 + 5,002 / eval 4,251）；实测击穿 v3 头——lora_v3 在 v4 eval 仅 0.4458（v3 满分=模板边界记忆非语义能力） |
| T171 | v4 全量训练 + eval（A100） | A | M | T170+GPU | ✅ done (W137)：sft_v4 0.7944 / rlcd_v4 0.8946 / **sft_text_v4 (LoRA) 1.0000**（v4 难族亦被学透）；产物 `external/valen/output_v4_20260928.tgz`（322MB 已回传校验）；结论：合成生成器族永远可被 LoRA 学透——**真实 eval 是唯一剩余区分度** |
| T172 | JEMM-27B 外部对标 | A | M | GPU | ✅ done (W137/W139)：A100 bf16 v4 eval **0.8165**（FP148 FN632，brier 0.237；v3 部分 0.8125 / v4 新族 0.8255）→ L40×2 `device_map=auto` 分片复现 0.8167（25 flips）；常驻 `:8790` ~222ms/q，site-packages patch `jemm/model.py:57`（重装即回退，见服务器 INSTALL_NOTES）；低于专项头 ~18pp，但无头+真实语料路线值得长期参照 |
| T173 | A100→L40×2 迁移 | ops | S | T169 | ✅ done (W139)：`docs/L40_MIGRATION_V1.md` runbook 执行完毕（tar pipes 替代 rsync、aliyun 镜像、causal-conv1d sm89 补丁、NCCL cu13 修复等偏差记录在执行回执节）；v4 头 fp32 parity 复测：lora 0 / rlcd 109 / sft 165 flips vs MPS fp32 |
| T174 | 真实 transcript 打标 + eval | A | M | T171 | 🟡 in progress (W139/W140)：547 候选就绪；**三后端提案已入库**（labeler_type=model 永不记分）+ pairwise adjudication + `review_queue.jsonl`（446 行，按严重度排序）已生成；owner 工作流就绪——`owner_review.md` 顶部 ~40 条实质分歧 + `owner_labels_template.csv` + `csv_to_labels_json.py` 转换器；lora_v3 头在真实候选上 537/547 判 keep |
| T175 | 生产头切换评审（v4 候选 vs nano_rlcd_v2） | A | S | T174 | ✅ **DECIDED (W141, owner 委托)——NO switch**：真实 eval（590 标签）显示 lora 虽最优但双向缺陷（FP=11 含 noul=0.92 高置信误删必需证据、漏删 21）；生产路由不变（winnow 默认 + `?backend=valen` rlcd_v2 + v4 保持 shadow :8094）。**新增 `?backend=consensus`（lora∧winnow AND-drop，真实集 FP=0/drop 精度 1.000/召回 17.9%）已在 :8876 上线为 shadow/advisory 模式**，作为 canary 阶梯的可部署安全形态；ConsensusScorer + parse_state=False 修复 + 8 项测试。备忘录 `docs/PRODUCTION_SWITCH_REVIEW_V1.md`（local-only）。复审触发：§4 四条件全过或分布显著漂移 |
| T176 | joint-batch 契约回归迁移收尾 | A | S | — | ✅ done (W140)：`context_gate_v1.py` joint-batch 契约（`irrelevant_N`/`candidate_pointers`）重写遗留的 **13 项测试回归全部修复**（7 个文件 stub scorer 迁移 + 2 处 uncertain_score 语义更新：整单 bypass → 按段保留）+ `report_context_relevance` 契约改名兼容 + `predictor repin` 链 v2；`test_jevbench_adapter_v1` 为 torch 2.6.0 MPS 内核 SIGABRT（环境项，非回归） |
| T177 | commit 审计执行 | ops | S | — | 🟡 待 owner 确认 commit：执行完毕——4 文档基础设施信息脱敏（IP/key → `<gpu-host>:<port>` 占位符）、12 处断链修复、26GB `remote_archive` 已确认忽略、handoff 脱轨；最终 add 白名单在 `docs/COMMIT_AUDIT_20260929.md` 执行结果节；**待决**：corpus_v3/v4/v5 (~119MB) 是否入库、deploy plists、审计文件本身 |
| T178 | winnow 真实候选覆盖 | A | S | T174 | ✅ done (W140)：`scripts/rewindow_for_winnow_v1.py`（10 tests 全过）实测——winnow 上限 prefix ≤8,096（自身 tokenizer）；220 拒绝 + 22 边界带共 242 条重窗口后 **219 条装入 ≤8,000**（1 条保护集下限仍超，按设计弃权）→ 覆盖率 546/547 (99.8%)；`candidates_winnow_fit.jsonl` 已产出（opt-in，未被消费）；rescore 待 owner 决定（~2min） |
| T179 | owner 复核打标（真实 eval 硬门禁） | A | S | T174 | ✅ done (W140, owner 委托 agent 裁决)：547 标签全入库（525 scored：523 keep + 2 drop；22 uncertain 剔除）；provenance=agent-adjudicated-under-owner-delegation 已披露；owner 抽查修改则需重算 |
| T180 | 真实 eval finalize + 分族报告 | A | S | T179 | ✅ done (W140)：`docs/REAL_CONTEXT_EVAL_RESULTS_V1.md`——590 条真实标签（525 主 + 65 drop 补集挖掘自同批 transcript 跨任务重锚定）：**合并 lora 0.9453/drop召回62.5%/FP11，kev 0.8927，winnow 0.9151 但召回仅23%**；lora 合并口径仍领先但暴露双向缺陷（漏删21 + 误删11 含 noul=0.92 高置信误删必需证据）；合成 vs 真实排名重排证实合成不可外推；winnow 覆盖经重窗口补至 546/547 |
| T181 | 开源生态调研（复刻候选清单） | A | M | — | ✅ done (W140)：`docs/ECOSYSTEM_RESEARCH_V1.md`——18 候选全表 + Top-3 复刻项（ModernBERT×Provence 配方 / EXIT 句级决策 / LLMLingua-2 标注管线）+ 可借用 eval/真实分布语料（SWE-chat、OpenHands 轨迹、HotpotQA supporting-facts）+ v5 设计七条建议；关键外部证据：kev-0.8B train→new-source gap 15–18pp 实证我们的记忆化问题 |
| T182 | 候选方案本机复刻+对标（Provence/EXIT/LLMLingua-2 管线） | A | L | T181 | 🟡 预研完成 (W140)：**Provence×ModernBERT** GO——ModernBERT-base MPS 冒烟通过（50.6ms/fwd @1159tok，22.9k tok/s），MVP 1.5–3 人日全量 5–9 人日（`docs/PROVENCE_REPRO_PREP_V1.md`）；**EXIT 配方** GO——配方级移植到现有 valen 契约 = S 工作量（纯数据活：`state`+`irrelevant` 问题已是 EXIT 条件化形状），不引入 Gemma 底座；**推荐合成**＝EXIT 数据配方（真值推导+2:1:1 难/随机负例）× valen 契约 + Provence 低阈值非对称操作点；LLMLingua-2 权重 CC-BY-NC-SA 只能借管线 |

### T17 — Controlled local main-model paired evaluator

**Status.** ✅ **deterministic + pinned local-generation slice done (W61/W63/W65)**: `scripts/local_main_model_evaluator_v1.py` provides `deterministic-evidence-v1` plus explicit local model directories / pinned HF-cache snapshots. After chat-template and normalization fixes, `Qwen3-0.6B`, `Qwen2.5-3B-Instruct`, and `Qwen3.5-4B` preserve `8/8`, `17/17`, and `18/18` originally successful answers under Winnow `0.90`, with zero unsafe removals. See [Local main-model pair V1](LOCAL_MAIN_MODEL_PAIR_V1.md).

**Goal.** Move from “required strings survived” to “a controlled downstream responder still produces the expected answer from the reduced request.” Depends on the A6 first slice. Effort M.

**Changes.**
- Add `scripts/local_main_model_evaluator_v1.py` and `scripts/test_local_main_model_evaluator_v1.py`.
- Add `docs/LOCAL_MAIN_MODEL_PAIR_V1.md` describing the paired contract, available local backends, output-normalization rules, privacy boundary, and receipt schema.
- Extend `scripts/benchmark_filter_value_v1.py` or add a thin caller that evaluates original/reduced request pairs through the evaluator.
- Support a deterministic evidence-conditioned backend first; optionally support an offline local model backend only after a pinned revision/hash is declared. No external provider calls.

**Verify.** `PYTHONPATH=scripts python3 -m unittest scripts.test_local_main_model_evaluator_v1 -v`; synthetic smoke must show original and reduced requests produce the declared answer while a deliberately unsafe drop fails.

**Gate.** No downstream-quality claim from string presence alone; every paired result reports answer status, usage estimate, latency, backend identity, and content-free response hashes.

### T18 — Expanded value/stress fixture suite

**Status.** ✅ **done (W61/W66)**: `research/context_filter_value_fixture_manifest_v2.json` has 25 development cases; `research/context_filter_value_fixture_manifest_v3.json` adds 12 held-out cases. V2 Winnow `0.90` removes 5,560 bytes / 38.0% with 25/25 deterministic paired answers; V3 removes 1,830 bytes / 30.7% with 12/12 deterministic paired answers and `9/9` Qwen3.5-4B originally successful answers preserved. Both report zero unsafe removals. See [Context filter value fixture V2](CONTEXT_FILTER_VALUE_V2.md) and [Context filter value fixture V3](CONTEXT_FILTER_VALUE_V3.md).

**Goal.** Make the value benchmark broad enough that one lucky synthetic set cannot select a threshold. Depends on T17 only for evaluator semantics. Effort M.

**Changes.**
- Add `research/context_filter_value_fixture_manifest_v2.json`, builder, validator, and tests.
- Cover at least: multilingual requests; policy/credential/correction cases; non-repeatable and mutable tool outputs; long tool traces; retrieval distractors with cited evidence; pending/duplicate/orphan tool links; unsupported tasks that must bypass; and protected families where zero deletion is mandatory.
- Keep all cases self-authored, deterministic, and free of real credentials, user data, benchmark rows, or provider responses.

**Verify.** Manifest integrity, deterministic rebuild, schema checks, and negative-control tests pass; unsupported/protected cases bypass or retain 100% of protected bytes.

**Gate.** No threshold or architecture selection from the current eight-case fixture alone.

### T19 — Diagnostic threshold policy

**Status.** ✅ **done (W61)**: `research/context_filter_threshold_policy_v1.json` and [threshold policy doc](CONTEXT_FILTER_THRESHOLD_POLICY_V1.md) pin `0.99/0.95/0.90/0.85` as diagnostic-only. `benchmark_filter_value_v1.py --review-mode` records policy hash and fails closed on undeclared scorer/cascade thresholds. V2 Winnow sweep: `0.95` removes 35.7% bytes / 35.9% local tokens; `0.90` and `0.85` remove 38.0% / 38.4%; all keep 25/25 paired answers and zero unsafe removals.

**Goal.** Separate exploratory threshold sweeps from a production operating point. Depends on T17/T18. Effort S.

**Changes.**
- Add `research/context_filter_threshold_policy_v1.json` and `docs/CONTEXT_FILTER_THRESHOLD_POLICY_V1.md`.
- Define development-only sweep thresholds, held-out fixture roles, confidence intervals, protected-family veto, and explicit “diagnostic-only” labels for values such as `0.90`.
- If a future production trial is proposed, select the operating point on a declared development cohort before touching held-out cases; do not tune on downstream outcomes from evaluation-only rows.

**Verify.** Policy file hash-pinned; runner reports the policy version and rejects undeclared thresholds in review mode; focused tests pass.

**Gate.** `0.90` remains diagnostic evidence only. No active filtering trial until the predeclared policy is reviewed.

### T20 — Fast-path scorer/cascade comparison

**Status.** ✅ **done (W61)**: [fast-path comparison](FAST_PATH_CASCADE_COMPARISON_V1.md) compares safe dedup, Winnow, Reflex, SemIf-4B MLX4, Decider-0.8B/2B, and Kev-4B/9B on the V2 suite. Kev-4B direct matches Winnow's 38.0% byte / 38.4% local-token reduction with zero unsafe removals and 25/25 paired answers. Kev-4B→Winnow at fast `0.95` offloads `13/19` candidate states and preserves the full reduction. Kev-9B is more conservative and slower. Latency still favors Winnow-only in this harness; this is evidence for a fast path, not a production selection.

**Goal.** Determine whether any smaller local scorer can reduce strong-path calls without adding unsafe reductions. Depends on T18. Effort M.

**Changes.**
- Generalize the value runner to select fast-path implementations rather than hard-coding Reflex.
- Compare safe dedup, Reflex, SemIf-4B, Kev-4B/9B where feasible, Decider-2B/0.8B, and any NanoJev-owned scorer under the same fixture suite.
- Report fast/strong route share, scorer latency and memory, paired downstream result, unsafe removals, and estimated net cost/latency.

**Verify.** `scripts/test_filter_value_v1.py` plus new scorer-selection tests; all route accounting is content-free and deterministic.

**Gate.** A fast path is useful only if it offloads a meaningful share while preserving zero protected drops and zero paired-answer regressions on the fixture suite. Reflex’s first-slice result is `0/9` fast paths, so it is not yet a useful fast path.

### T21 — NanoJev vNext architecture freeze

**Status.** ✅ **done (W62)**: [architecture decision](NANOJEV_VNEXT_ARCHITECTURE_V1.md) and `research/nanojev_vnext_architecture_decision_v1.json` now freeze the research target as **NanoJev controller + optional Kev-4B fast path + Winnow-12B strong fallback**. Kev-4B is the reference fast path because it resolved `13/19` V2 candidate states and preserved the full `5,560`-byte reduction; Winnow remains the local quality baseline. A custom NanoJev scorer is deferred until independent evidence beats this cascade. The freeze is diagnostic only: no active filtering, production threshold, release candidate, or external milestone.

**Goal.** Turn the measured evidence into a frozen implementation decision rather than continuing open-ended model search. Depends on T17–T20. Effort M.

**Changes.**
- Update `docs/NANOJEV_VNEXT_ARCHITECTURE_V1.md` and `research/nanojev_vnext_architecture_decision_v1.json`.
- Freeze the controller contract, scorer roles, cascade order, failure modes, receipts, model/precision/revision pins, and fallback policy.
- Decide whether the next build is: (a) controller + Winnow strong path, (b) controller + small fast path + Winnow fallback, or (c) a deferred custom NanoJev model.

**Verify.** Decision JSON validates against the policy fields; every referenced artifact has revision/hash; architecture doc explicitly separates third-party scorer evidence from NanoJev-owned model evidence.

**Gate.** Architecture freeze does not authorize active filtering or external submission; it only selects the next implementation target.

### T22 — Track A review/release gate

**Status.** ✅ **done (W67, owner-authorized internal review)**: [review checklist](TRACK_A_REVIEW_CHECKLIST_V1.md), deterministic replay, scorer stress receipt, V2 development results, V3 held-out results, and pinned local-generation results were reviewed after the owner waived the third-party reviewer requirement. `docs/TRACK_A_REVIEW_DECISION_V1.md` records a conditional pass for the Track A evidence packet; `results/track_a_review_decision_v1.json` is the machine-readable receipt. This is not external independent-review equivalence and does not authorize active filtering, a production threshold, or a release candidate.

**Goal.** Require a reviewer to replay the evidence before any active trial, release candidate, or external milestone. For this repository state, the owner explicitly waived third-party review and authorized direct internal review. Depends on T21. Effort owner/internal.

**Changes.**
- Review checklist covers fixture provenance, scorer provenance, threshold policy, paired downstream evidence, route accounting, restore manifests, privacy, and publication boundaries.
- Review replay reruns manifest validators, selected value-result invariants, negative controls, scorer stress checks, and focused tests.
- Review decision records the owner waiver, evidence hashes, accepted findings, and remaining limitations.

**Verify.** `results/track_a_review_replay_v1.json` reports `ok=true` / `failure_count=0`; `results/track_a_review_decision_v1.json` records `conditional_pass_owner_authorized_internal_review`.

**Gate.** T22 review is now closed for research handoff. Active filtering remains disabled until a separate operational authorization and canary plan exist. Production defaults remain fail-open/shadow.

### T23 — Active-mode canary protocol/preflight

**Status.** ✅ **done (W68)**: `research/active_mode_canary_protocol_v1.json` defines the isolated canary ladder; `scripts/validate_active_mode_canary_protocol_v1.py` validates review-gate status, cohort hashes, diagnostic threshold, provider accounting, safety controls, stop conditions, rollback, and scorer identity. `results/active_mode_canary_preflight_v1.json` reports `protocol_valid_not_authorized` with `active_filtering_allowed=false` and `provider_calls_allowed=false`. See [Active-mode canary protocol V1](ACTIVE_MODE_CANARY_PROTOCOL_V1.md).

**Goal.** Convert the reviewed evidence packet into a bounded, rollback-ready experiment plan without enabling active filtering. Depends on T22. Effort S.

**Changes.**
- Phase0: active-mode loopback fake upstream only.
- Phase1: shadow provider/main-model measurement after explicit owner authorization.
- Phase2: paired active canary after explicit owner authorization.
- Require paired provider-reported usage before any actual savings claim.
- Define kill switch, restore-manifest budget, stop conditions, content-free receipts, and bounded request limits.

**Verify.** Protocol validator emits `protocol_valid_not_authorized`; unit tests cover phase authorization, diagnostic threshold, and provider-paired accounting.

**Gate.** This protocol authorizes no active filtering and no provider calls. Phase1/Phase2 require a separate explicit owner operational authorization and a real-context holdout or approved substitute.

### T24 — Phase0 loopback active-mode dry-run

**Status.** ✅ **done (W69)**: `scripts/run_active_mode_phase0_v1.py` executes the gateway in active mode only against a loopback fake upstream. `results/active_mode_phase0_loopback_v1.json` reports `phase0_pass`, 5/5 cases, `provider_calls=0`. See [Active-mode phase0 dry-run V1](ACTIVE_MODE_PHASE0_V1.md).

**Goal.** Prove the active applicator, restore manifest, content-free receipts, kill switch, and fail-open paths without sending any provider traffic. Depends on T23. Effort S.

**Changes.**
- Loopback fake upstream records only hashes/statuses in the phase receipt.
- Eligible assistant segment is removed while evidence/user intent stays.
- Restore manifest reconstructs the original request and contains no raw text.
- Shadow control, kill switch, scorer error, and malformed sidecar all fail/pass as expected.

**Verify.** `scripts/test_active_mode_phase0_v1.py` passes; phase receipt contains no provider calls and no raw prompt/response fields.

**Gate.** Phase0 is transport/applicator evidence only. Phase1/Phase2 remain unauthorized.

### T25 — Real-context holdout intake protocol

**Status.** ✅ **done (W70)**: `research/real_context_holdout_protocol_v1.json` defines raw-local-only intake under `data/real_context_holdout_v1/`, owner-selected sources, forbidden benchmark/provider-label sources, manual labels, V2/V3 isolation, privacy stop conditions, and the required evaluation order. `scripts/validate_real_context_holdout_protocol_v1.py` emits `results/real_context_holdout_preflight_v1.json` with `protocol_valid_not_collecting`. See [Real-context holdout protocol V1](REAL_CONTEXT_HOLDOUT_PROTOCOL_V1.md).

**Goal.** Remove ambiguity before collecting any real context so raw data cannot leak into public receipts and so evaluation starts with deterministic shadow/local-generation checks rather than provider traffic. Depends on T23/T24. Effort S.

**Changes.**
- Raw root is under ignored `data/` storage.
- Manifests may store only hashes/counts/status-like fields.
- Labels are manual; scorer outputs are not labels.
- V2/V3 request-hash overlap is prohibited.
- Provider and active canary phases require new owner scope.

**Verify.** Protocol validator reports `protocol_valid_not_collecting`; tests cover collection gate, manual labels, isolation, and evaluation order.

**Gate.** Collection has not started. Provider calls and active filtering remain disabled.

### T26 — Real-context holdout manifest builder/validator

**Status.** ✅ **done (W71)**: `scripts/real_context_holdout_manifest_v1.py` can build and validate a content-free holdout manifest from local-only case metadata. It enforces case ID/uniqueness, protocol source/wire/status contracts, SHA-256 fields, protected/eligible separation, downstream contracts, request-file confinement, credential-like pattern scans, duplicate request hashes, coverage reporting, and overwrite refusal. See [Real-context holdout manifest V1](REAL_CONTEXT_HOLDOUT_MANIFEST_V1.md).

**Goal.** Give future owner-selected raw contexts a deterministic intake boundary before evaluation. Depends on T25. Effort S.

**Changes.**
- Manifest contains hashes/counts/pointers/labels, not raw request text.
- Case metadata may reference `request_file`, but the file stays under the case directory and local `data/` root.
- Scored cases must declare downstream evidence or expected-answer hash.
- Coverage report determines `ready_for_evaluation`.

**Verify.** `scripts/test_real_context_holdout_manifest_v1.py` covers valid manifests, duplicate hashes, raw metadata keys, credential-like requests, and protected/eligible overlap.

**Gate.** No real cases were collected; provider calls and active filtering remain disabled.

### T27 — Real-context deterministic shadow evaluator

**Status.** ✅ **done (W72)**: `scripts/run_real_context_holdout_shadow_v1.py` validates a holdout manifest, reads its local raw request files, derives trusted sidecar metadata, runs deterministic shadow scoring, checks expected gate status, unsafe protected/required drops, optional expected suggestions, and hypothetical reduced-request restore round-trips. See [Real-context holdout shadow V1](REAL_CONTEXT_HOLDOUT_SHADOW_V1.md).

**Goal.** Make future real-context cases immediately measurable without sending any bytes to a provider. Depends on T26. Effort S.

**Changes.**
- Bypass, scored, and protected-only cases follow the manifest contract.
- Deterministic scorer marks eligible candidates irrelevant at 0.999.
- Protected, dependency, and required-evidence drops are unsafe.
- Reduced bytes are built only in-process and verified for restore before receipt.

**Verify.** `scripts/test_real_context_holdout_shadow_v1.py` covers scored restore, bypass, protected-only, unsafe evidence drop, and invalid-manifest failure.

**Gate.** No real cases collected; provider calls and active filtering remain disabled.

### T28 — Real-context paired local-generation evaluator

**Status.** ✅ **done (W73)**: `scripts/run_real_context_holdout_localgen_v1.py` validates a holdout manifest, reads local raw requests plus `<case_id>.contract.json`, verifies downstream hash bindings, constructs deterministic-shadow reduced bytes in-process, and compares original/reduced outcomes with either `DeterministicEvidenceResponder` or pinned `LocalGenerationResponder`. See [Real-context paired local-generation V1](REAL_CONTEXT_HOLDOUT_LOCALGEN_V1.md).

**Goal.** Give future real-context cases a downstream quality check beyond structural evidence preservation. Depends on T27. Effort S.

**Changes.**
- Local contract files keep raw expected answers/required strings out of manifests and receipts.
- Contract hashes must match the manifest.
- Non-scored cases are skipped.
- Paired regressions and missing reduced evidence are unsafe.

**Verify.** `scripts/test_real_context_holdout_localgen_v1.py` covers deterministic pass, evidence-drop regression, expected-answer hash mismatch, and non-scored skip.

**Gate.** No real cases collected; provider calls and active filtering remain disabled.

### T29 — Real-context end-to-end preflight

**Status.** ✅ **done (W74)**: `scripts/run_real_context_holdout_preflight_v1.py` chains the holdout protocol validator, manifest validation, deterministic shadow evaluation, and paired local-generation evaluation into one content-free receipt. It supports protocol-only mode, deterministic or pinned local-generation backends, and `--require-ready`. See [Real-context E2E preflight V1](REAL_CONTEXT_HOLDOUT_E2E_PREFLIGHT_V1.md).

**Goal.** Give future real-context cohorts a single deterministic gate before any provider shadow/canary discussion. Depends on T28. Effort S.

**Changes.**
- `protocol_ready_no_manifest` for protocol-only checks.
- `e2e_preflight_pass` only when protocol + manifest + shadow + local-generation all pass.
- `--require-ready` enforces coverage minima.
- Receipt remains content-free and records `provider_calls=0`.

**Verify.** `scripts/test_real_context_holdout_preflight_v1.py` covers protocol-only, full-chain pass, coverage readiness failure, and unsafe-case propagation.

**Gate.** No real cases collected; provider calls and active filtering remain disabled.

### T30 — Real-context collection authorization runbook

**Status.** ✅ **done (W75)**: `research/real_context_collection_authorization_v1.json` is a validated not-authorized template; `scripts/validate_real_context_collection_authorization_v1.py` verifies owner-scope fields, required checklist items, forbidden actions, stop conditions, and artifact availability. `docs/REAL_CONTEXT_COLLECTION_RUNBOOK_V1.md` documents the operator procedure.

**Goal.** Make future collection explicitly scoped and reversible before raw real-context files are introduced. Depends on T29. Effort S.

**Changes.**
- Authorization template remains `template_ready_not_authorized`.
- Required owner fields: authorized flag, authorized_by, authorized_at, scope, expires_at.
- Checklist covers scope, sources, privacy, labels, manifest, E2E preflight, rollback.
- Provider calls, remote scorers, active filtering, scorer-derived labels, and training/calibration use are forbidden during collection.

**Verify.** `results/real_context_collection_authorization_preflight_v1.json` reports `template_ready_not_authorized`; tests cover authorization fields, checklist, forbidden actions, and stop conditions.

**Gate.** Collection is not authorized; provider calls and active filtering remain disabled.

### T31 — Multi-wire Phase0 active-mode dry-run

**Status.** ✅ **done (W76)**: `scripts/run_active_mode_phase0_v1.py` now covers 10 loopback cases: OpenAI chat, paired baseline accounting, Anthropic messages, OpenAI Responses, unsupported embeddings path, active no-reduction, shadow control, kill switch, scorer error, and malformed sidecar. `results/active_mode_phase0_loopback_v1.json` reports `phase0_pass` with `provider_calls=0`.

**Goal.** Prove the active-path safety contract is not OpenAI-chat-specific before any future provider-authorized phase. Depends on T24; effort S.

**Changes.**
- Parameterized phase0 request construction by wire format.
- Verifies pointer-specific reduction, evidence retention, restore manifest, and round-trip hashes for supported formats.
- Verifies baseline header produces `savings.claim=actual` only in the paired-baseline case.
- Verifies unsupported paths and no-reduction cases remain byte-identical.

**Verify.** `scripts/test_active_mode_phase0_v1.py` checks the expanded receipt contract and all 10 case IDs.

**Gate.** Loopback-only evidence; provider calls and active filtering outside tests remain disabled.

### T32 — Phase0 stress/denial coverage

**Status.** ✅ **done (W77)**: Phase0 now covers 17 loopback cases, adding dependency closure, uncertain scorer output, invalid scorer response, scorer timeout, oversized restore manifest, oversized request body, and internal header stripping. The canary protocol validator enforces the expanded required case list.

**Goal.** Prove denial/failure paths before any future operational canary. Depends on T31. Effort S.

**Changes.**
- `run_case` supports bounded request size, restore-header limit injection, upstream header inspection, and no-upstream outcomes.
- Deadline scorer validates timeout fail-open classification.
- Dependency case proves a retained dependent protects its required dependency.
- Internal `X-NanoJev-*` headers are verified absent from fake-upstream requests.

**Verify.** `results/active_mode_phase0_loopback_v1.json` reports `phase0_pass` across 17 cases; `scripts/test_active_mode_phase0_v1.py` checks the expanded case set.

**Gate.** Loopback-only stress evidence; provider calls and active filtering outside tests remain disabled.

### T33 — Phase1 shadow authorization template/runbook

**Status.** ✅ **done (W78)**: `research/phase1_shadow_authorization_v1.json`, `scripts/validate_phase1_shadow_authorization_v1.py`, and `docs/PHASE1_SHADOW_RUNBOOK_V1.md` define the measurement-only Phase1 scope. The template remains `template_ready_not_authorized`; owner must supply granted_by/granted_at/scope/expires_at/upstream before any future Phase1 run.

**Goal.** Separate local readiness evidence from a future scoped shadow measurement. Depends on T32. Effort S.

**Verify.** `results/phase1_shadow_authorization_preflight_v1.json` reports `template_ready_not_authorized`; tests cover shadow-only contract, metrics, checklist, forbidden actions, and stop conditions.

**Gate.** Phase1 not authorized; provider calls and active filtering remain disabled.

### T34 — Phase1 synthetic shadow dry-run

**Status.** ✅ **done (W78)**: `scripts/run_phase1_shadow_dry_run_v1.py` runs 8 loopback shadow cases across OpenAI chat, Anthropic messages, OpenAI Responses, no-sidecar, internal headers, unsupported path, kill switch, and scorer error. `results/phase1_shadow_dry_run_v1.json` reports `phase1_shadow_dry_run_pass`, `provider_calls=0`, `active_filtering_applied=false`.

**Goal.** Prove Phase1 measurement plumbing preserves original bytes while recording shadow proposals. Depends on T33. Effort S.

**Verify.** `scripts/test_phase1_shadow_dry_run_v1.py` checks all cases forward unchanged and emit no restore header.

**Gate.** Loopback-only; no provider calls or active filtering.

### T35 — Phase1 readiness gate

**Status.** ✅ **done (W78)**: `scripts/evaluate_phase1_readiness_v1.py` aggregates the Track A decision, canary preflight, Phase0 17-case receipt, Phase1 authorization template, Phase1 dry-run, real-context protocol, and collection authorization into `results/phase1_readiness_gate_v1.json`. Status: `ready_for_owner_phase1_authorization`.

**Goal.** Declare the next major node: ready to request owner authorization for bounded Phase1 shadow measurement, while keeping Phase1/provider/active explicitly unauthorized. Depends on T34. Effort S.

**Metric requirements.** Track A pass; Phase0 17/17; Phase1 dry-run 8/8; provider calls 0; unsafe actions 0; selected paired regressions 0; kill switch/restore/unsupported bypass/scorer fail-open verified; content-free receipts; Phase1 and active filtering still false.

**Verify.** `scripts/test_phase1_readiness_v1.py` checks gate status and all milestone metrics.

**Gate.** This is readiness, not authorization. Provider calls and active filtering remain disabled.

### T36 — Minimum-reduction gate (borrowed)

**Status.** ✅ **done (W80)**. Borrowed from `tamaratran/fast-jev-compaction` (`minReductionRatio`).

**Goal.** If a proposed reduction saves less than a configured floor, forward the original request unchanged — small savings are not worth applicator risk or scorer overhead.

**Changes.** `GatewayConfig.min_reduction_bytes` (default 0, non-negative int) and CLI `--min-reduction-bytes`. The check runs after the candidate reduction is fully verified and before it is applied: savings below the floor forward original bytes with `forward_reason=below_min_reduction` and emit no restore manifest.

**Verify.** `test_active_skips_verified_reduction_below_min_floor` and `test_active_reduction_at_or_above_min_floor_still_applies` in `scripts/test_main_model_gateway_v1.py`; 49/49 gateway tests pass; Phase0/canary tests unaffected.

**Gate.** Byte-floor only (no token floor); default 0 preserves existing behavior; fail-open semantics unchanged.

### T37 — Staged scorer-state fitting (borrowed)

**Status.** ✅ **done (W81)**. Borrowed from `fast-jev-compaction` (`fitState` stages).

**Goal.** When the scorer request exceeds its budget, degrade the judge's view deterministically instead of failing, so the local scorer keeps working on long contexts.

**Changes.**
- `context_gate_v1.py`: `scoring_payload` gained `content_limit`/`collapse` render options via `_scoring_state`; new `fit_scoring_payload(segments, candidates, max_bytes)` walks stages `full → contents<=2000 → contents<=500 → contents<=120 → collapsed_context`; `shadow_request(..., max_scorer_payload_bytes=None)` fits per batch and records `batches[].state_stage`; impossible budgets raise `scorer_state_budget_exceeded` → batch fails open.
- `main_model_gateway_v1.py`: `GatewayConfig.max_scorer_payload_bytes` (default None) + CLI `--max-scorer-payload-bytes`, passed through to the gate.

**Safety invariants.** The candidate being judged and user messages are never removed from the view (truncated only); no protected-pointer/dependency rule changed; an unfittable state fails the batch open rather than silently scoring a gutted state; stage is receipted for audit.

**Verify.** `test_staged_scorer_state_fitting` in `scripts/test_context_gate_v1.py`; 15/15 gate tests, 67/67 batch+local+gateway tests pass.

**Gate.** Default unset preserves existing full-state behavior; no scorer/judgment-quality claim — fitting degrades evidence the judge sees, which is why the stage is receipted and the hard failure stays fail-open.

### T38 — Local scorer service stability pass

**Status.** ✅ **done (W82)**. Standing goal: keep the local model stack reliably runnable on this Mac.

**Goal.** A small daily-driver pass so day-to-day progress does not silently depend on a dead service.

**Changes.**
- `scripts/check_local_services_v1.py`: probes NanoJev lifecycle service across 8765 + 8876–8890 fallback ports (`/api/health` ready check) plus informational legacy scorer endpoints 8091/8092; `--smoke` posts one tiny `/api/evaluate` to prove the loaded model answers; JSON receipt, exit 0/1.
- `scripts/test_local_services_check_v1.py`: fake-server unit tests for probe/smoke/required-down logic.
- `docs/LOCAL_SERVICES_RUNBOOK_V1.md`: service map, daily check commands, restart via `nanojev_skill.py health --start` or direct `serve_decisions.py`.

**Verify.** Live check: service on 8876 `{"ready": true}`, smoke evaluate ~83ms, receipt `results/local_services_health_v1.json`; 3/3 unit tests pass.

**Gate.** Read-only probes only — the checker never starts, stops, or reconfigures services; no provider calls.

### T39 — Winnow-12B Q8 as NanoJev default scorer

**Status.** ✅ **done (W83)**. Owner directed: NanoJev uses Winnow-12B Q8 directly as the scorer.

**Changes.**
- Started `external/winnow-inference/scripts/serve.py` with the measured Apple profile: `external/models/Winnow-12B/Winnow-12B-Q8_0.gguf`, ctx 8192, f16 KV, decision_parallel 1, `127.0.0.1:8091`.
- Live-verified `POST /v1/systemone` (~0.28 s/state) and the full gate path: `parse_segments → scoring_payload → SystemOneHTTPScorer` returns `p_irrelevant=0.911` on a stale assistant segment.
- `check_local_services_v1.py` now reports `winnow_scorer_8091`; `docs/LOCAL_SERVICES_RUNBOOK_V1.md` documents the start command and gateway wiring (`--scorer systemone --scorer-url http://127.0.0.1:8091`).

**Gate.** Winnow is the default *scorer*, not an authorization mechanism — its 59 confident-wrong@0.9 failures remain documented; gateway stays shadow by default, active filtering still disabled, loopback-only scorer transport unchanged.

### T40 — Gateway ↔ live Winnow end-to-end shadow

**Status.** ✅ **done (W84)**.

**Goal.** Prove the real integration path: gateway shadow mode → `SystemOneHTTPScorer` → live `winnow-server` on 8091 — not in-process stubs.

**Changes.** `scripts/run_gateway_winnow_shadow_v1.py` runs the Phase0 loopback topology with a counting live scorer across OpenAI chat, Anthropic messages, OpenAI Responses, no-sidecar, unsupported path, kill switch, and internal-header stripping.

**Verify.** `results/gateway_winnow_shadow_v1.json` (`winnow_shadow_pass`, `scorer_calls=4`, mean ~0.31s/state, `provider_calls=0`); `scripts/test_gateway_winnow_shadow_v1.py` checks shadow invariants and that Winnow proposed the declared eligible pointer on every supported wire.

**Gate.** Shadow only — every case forwards original bytes; active filtering still disabled.

### T41 — Winnow concurrency/cache-isolation stress

**Status.** ✅ **done (W84)**. Covers the "concurrency and cache-isolation checks" gap listed in `WINNOW_REVIEW_V1.md`.

**Changes.** `scripts/run_winnow_stress_v1.py` fires 20 `/v1/systemone` calls (4 distinct ticket states × 5 repeats, 4 workers) and checks determinism (same request → identical noul), per-state consistency under concurrency, and answer isolation (refund/replacement states ≥0.99, others ≤0.001 — no cross-contamination).

**Verify.** `results/winnow_stress_v1.json` (`winnow_stress_pass`, deterministic, consistent, isolated; p50 ~0.46s under 4-way parallel with `decision_parallel=1`); `scripts/test_winnow_stress_v1.py`.

**Gate.** Read-only load on the server; loopback only; no provider calls.

### T42 — Live Kev-4B → Winnow cascade

**Status.** ✅ **done (W85)**. Turns the cascade from offline replay into a live two-service path.

**Changes.**
- Started `kev.serve` for `jaredpalmer/kev-4b` on `127.0.0.1:8092` (`kev-latest`, TypeSafe `/v1/systemone`).
- `scripts/run_cascade_live_v1.py`: synthetic payload through `CascadeScorer` — confident fast answers stay on Kev (stale segment 0.95, evidence segment 0.05); a simulated fast outage routes all states to Winnow (0.951/0.004).
- Health check now reports `kev_scorer_8092`; runbook documents the start command.

**Verify.** `results/cascade_live_v1.json` (`cascade_live_pass`, `provider_calls=0`); `scripts/test_cascade_live_v1.py`.

**Gate.** Loopback only; cascade is a scorer composition — gate invariants and fail-open semantics unchanged.

### T43 — Local stack keepalive

**Status.** ✅ **done (W86)**.

**Changes.** `scripts/start_local_stack.sh` idempotently brings the stack up: probes each endpoint and only starts what is missing — NanoJev lifecycle service (`nanojev_skill.py health --start`), `winnow-server` on 8091, `kev.serve` on 8092 — then prints the health-check receipt. Logs go to `/tmp` (`NANOJEV_STACK_LOG_DIR` override).

**Verify.** Ran with all services up → no-op, health receipt `ok`; `LOCAL_SERVICES_RUNBOOK_V1.md` documents daily usage.

**Gate.** Starts local loopback services only; never touches provider or config.

### T44 — Cascade scorer in gateway CLI

**Status.** ✅ **done (W87)**.

**Changes.**
- `scorer_adapters_v1.py`: `SCORER_KINDS` + `build_scorer` gained `cascade` — `url` is the fast `/v1/systemone` service, `strong_url` the strong one; missing either URL raises.
- `main_model_gateway_v1.py`: `--scorer cascade` plus `--scorer-strong-url` / `--scorer-strong-model`; cascade defaults to `/v1/systemone` endpoints.
- `run_gateway_winnow_shadow_v1.py` gained `--scorer-kind cascade --strong-url` for the live shadow regression.

**Verify.** `test_cascade_scorer_confident_fast_and_strong_fallback` (confident fast stays, uncertain routes to strong) + `build_scorer` shape checks; `results/gateway_cascade_shadow_v1.json` — 7/7 shadow cases pass through live Kev-4B → Winnow cascade (`scorer_calls=4`, `provider_calls=0`); 50/50 gateway tests.

**Gate.** Same fail-open rules; strong-path failures still propagate to fail open. Cascade scorer never lowers protected-pointer rules.

### T45 — Active-mode loopback with live cascade

**Status.** ✅ **done (W88)**. Last unverified integration point: the apply path (reduction + restore manifest) with real model outputs instead of deterministic stubs.

**Changes.** `scripts/run_gateway_live_active_v1.py` runs active-mode loopback cases through the live Kev-4B → Winnow cascade. For each wire format the receipt validates whichever outcome the scorer produces: a proposed drop must satisfy all apply-path invariants (reduced bytes correct, restore manifest present, round-trip hashes match, no raw text in header); a retained request must forward original bytes.

**Verify.** `results/gateway_live_active_v1.json` (`live_active_pass`): all 3 wires dropped with live models, restore round-trip verified, scorer mean ~0.64s/call, `provider_calls=0`; `scripts/test_gateway_live_active_v1.py`.

**Gate.** Loopback upstream only; active mode remains test-only — no change to the production authorization boundary.

### T46 — Live cascade fixture eval vs Winnow-only

**Status.** ✅ **done (W89)**.

**Goal.** Measure the live cascade's actual value: fast-path hit rate, decision parity vs Winnow-only, latency delta.

**Changes.** Ran `benchmark_filter_value_v1.py` on the V3 held-out manifest (`--review-mode --threshold 0.90`) with `cascade` (Kev-4B @8092 → Winnow-12B @8091), `systemone` (Winnow-only), and `control` arms.

**Findings** (`results/filter_value_v3_live_cascade_v1.json`).
- Cascade decisions match Winnow-only exactly: same 1830 removed bytes, 0 unsafe removals, 0 paired answer regressions.
- Fast-path hit rate 4/8 states; uncertain states fell through to Winnow correctly.
- Cascade is *slower* on this Mac: p50 ~796ms vs ~290ms Winnow-only — Kev-4B is a Qwen3.5 hybrid and MPS has no fast DeltaNet kernels (upstream README: ~0.78s vs 0.17s for the Qwen3 variant).
- Operational conclusion: keep Winnow-12B as the default scorer; revisit cascade only if a faster fast-path appears (Qwen3 Kev build, MLX backend, or a smaller decision model).

**Verify.** `scripts/test_filter_value_v3_live_v1.py` — parity and safety invariants on the live receipt.

**Gate.** Fixture eval only; no provider calls; no active filtering.

### T47 — Stack operator sheet

**Status.** ✅ **done (W90)**.

**Changes.** `docs/LOCAL_STACK_OPERATOR_V1.md`: one-page daily operations — service table, `start_local_stack.sh` usage, measured scorer choice (Winnow-only default; cascade slower on MPS), gateway flags, regression receipt commands, stop commands, boundaries.

**Gate.** Documentation only.

### T48 — Phase1 shadow measurement (synthetic scope)

**Status.** ✅ **done (W91)**. Owner granted continuing; executed Phase1 shadow at the minimal self-contained scope — no provider, no real user data.

**Changes.**
- `research/phase1_shadow_scope_synthetic_v1.json`: owner scope grant (the authorization template itself stays `granted=false` by design) — synthetic fixture traffic, loopback upstream, live Winnow scorer, shadow only, ≤25 requests, expires 2026-09-24.
- `scripts/run_phase1_shadow_measurement_v1.py`: validates the grant deterministically before any request (mode=shadow, provider/active/real-data all forbidden, budget bound), then measures each held-out fixture case through the loopback gateway with the live Winnow scorer.

**Findings** (`results/phase1_shadow_measurement_v1.json`).
- 12/12 requests: original bytes upstream, no restore manifests, `provider_calls=0`.
- Winnow proposed drops on 5 cases, retained 7; **0 unsafe proposals**.
- Every declared retain-expectation honoured; the only deviation is `heldout_exact_duplicate` (expected a drop, scorer retained — safe/conservative direction).
- Scorer latency mean ~0.30s, p-max ~0.40s.

**Gate.** This is scoped synthetic measurement only — it does NOT authorize provider traffic, real-user requests, or active filtering. Those still need their own scope grants.

### T49 — Unified nanojev service

**Status.** ✅ **done (W92)**. Owner asked to consolidate the stack into one callable "nanojev service".

**Changes.** `scripts/serve_decisions.py` is now the single entrypoint on 8876 (auto-fallback preserved):
- `GET /api/health` → `ready` + per-backend status (winnow/kev probed, not started).
- `POST /api/evaluate` → unchanged NanoJev typed decisions.
- `POST /v1/systemone` → proxies to the scorer backend; `?backend=kev` or `?backend=cascade` (kev fast → winnow strong when the fast answer isn't confident ≥0.95).
- `POST /v1/context-gate` → `{"request", "wire_format"?, "sidecar"?, "backend"?, "threshold"?, "max_scorer_payload_bytes"?}` → shadow-mode gate receipt (content-free).

Health checker renamed the required probe to `nanojev` and the scorer probes to `*_backend_*`. `nanojev_skill.py health --start` still starts the service; `start_local_stack.sh` still starts all three processes (service + 2 backends).

**Verify.** `scripts/test_serve_nanojev_v1.py` (4 tests: confidence gate, routing, cascade fallback, injected-scorer context-gate); live verified — winnow/kev/cascade all answer through 8876, context-gate proposes a drop at threshold 0.9 and retains at 0.99 (uncertain_score fail-open).

**Gate.** Loopback only, shadow semantics unchanged; backends are internal dependencies, not new services.

### T50 — Agent-facing API contract

**Status.** ✅ **done (W93)**.

**Changes.** `docs/NANOJEV_SERVICE_API_V1.md` — the contract other local agent CLIs code against: base URL + health, four contract rules (advisory only, shadow semantics, content-free receipts, fail-open), `/api/evaluate` input/output, `/v1/systemone` backend routing table, `/v1/context-gate` payload + receipt field semantics, availability commands.

**Gate.** Documentation only; explicitly documents that no response authorizes an action.

### T51 — launchd keepalive

**Status.** ✅ **done (W93)**.

**Changes.** `deploy/launchd/` with three user agents — `ai.nanojev.service` (serve_decisions on 8876), `ai.nanojev.winnow` (8091 backend), `ai.nanojev.kev` (8092 backend) — all `RunAtLoad` + `KeepAlive`, logs to `/tmp/`. `install.sh` bootstraps/bootouts; `--uninstall` removes cleanly.

**Verify.** Installed, `launchctl list` shows all three loaded, `/api/health` reports `backends: {winnow: true, kev: true}` — the whole stack now survives logout/reboot and crashes without manual restart.

**Gate.** User-scope launch agents on loopback services only; reversible.

### T52 — nanojev-local-service skill

**Status.** ✅ **done (W94)**. Packages the unified service as a general skill callable in parallel with the official direct Jev API.

**Changes.**
- `integrations/codex-skill/nanojev-local-service/SKILL.md`: usage, parallel-call pattern with `jev-eval`, endpoint table, advisory/shadow/fail-open semantics, measured quality ordering (official 0.886 > Winnow 0.882 > Kev-9B 0.819 > Kev-4B 0.802).
- `scripts/nanojev_eval.py`: jev-eval-compatible CLI — same `{state, questions}` input shape, same `{answers, usage, model}` output plus `backend`/`source`/`elapsed_ms`; `--backend winnow|kev|cascade`, `--context-gate`, `--url`; auto-starts the service via `nanojev_skill.py health --start` when down.
- Installed: `~/.codex/skills/nanojev-local-service/` + `~/.local/bin/nanojev-eval` symlink.

**Verify.** Live: `nanojev-eval` (winnow noul 0.9993), `--backend cascade` (kev unconfident → winnow 0.9995), `--context-gate` (receipt, fail-open on uncertain score); parallel `jev-eval` + `nanojev-eval` on the same payload produced shape-identical outputs. `test_nanojev_eval.py` 3/3.

**Gate.** Loopback only, no key material, advisory semantics preserved.

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

**Status (2026-09-20).** Complete and ready for review; [attributed report](PAPER_TRADE_REAL_DATA_V1.md),
`results/paper_trade_b0_report_v1.json`, final raw evidence `results/paper_trade_t5_v1_run2/`.
75 new tests; full suite 651 OK (2 skipped). Price/signal independently reconstructed from
signed mid-price fill cash flows and terminal inventory; spread/slippage/tick rounding split;
unfilled opportunity PnL stays unknown, not fabricated. 36 cells and one repeated receipt,
all final input/source hashes unchanged. Old corrected full-window ledgers/replay hashes match.
Seed axis is degenerate, windows restart independently, Binance lacks 2026-06-29 mark/index.
Primary Binance/Bybit YTD spread is 6.4544% of initial cash, above declared exploratory 5%.
`headline_allowed=false`: B0 explicit-refusal branch, not robustness or training approval.
Aster licence/source metadata conflict remains in a separate appendix. Next: T6 owner decision package.

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

**Status (2026-09-20).** The owner has recorded D1-a (11-feature realizable schema with OI and
no liquidation), D2-a (all 71+19 values frozen as explicit code-default assumptions), D3-a
(USDT-margined linear only), and C1–C9 in [FINANCIAL_R1_DECISIONS_V1.md](FINANCIAL_R1_DECISIONS_V1.md).
The protocol rebase and real-data PIT gate are now also complete: the authoritative v5 receipt binds
3,266 Binance-only records with 11 features and reports `preflight_passed_not_training_authorized`.
T11/T14 are no longer blocked by T6, but no financial training, measurement, order submission, live
trading, or venue permission is implied; the V1 venue-holdout limitation remains.

**Goal.** Convert the three open R1 questions into decision-ready material for the owner.

**Changes.** New `docs/FINANCIAL_R1_DECISIONS_V1.md`: for each question (PILOT 9-feature vs
closed 12-feature allowlist; the 71 provisional parameters + 19 construction values;
single-numeraire scope), list options, consequences, evidence from T5, and a recommendation.
The owner decision is recorded in the decision package; an independent reviewer may still reject
the protocol or request a new version before formal financial work begins.

**Gate.** No Track B training work (T11/T14) starts before these are decided and frozen.

### T7 — Aster licence resolution

**Status (2026-09-20).** Draft prepared at
[ASTER_PERMISSION_REQUEST_V1.md](ASTER_PERMISSION_REQUEST_V1.md). It requests written consent
under §6.1(b) and express automated-access permission under §6.2(e), documents the exact
read-only endpoints and 35 retained files, and offers retention/deletion/rate-limit/attribution
commitments. It also discloses the existing manifest `base_url` conflict without rewriting it.
The draft was not sent or submitted; no venue response exists, so Aster stays appendix-only and
the permission decision remains with the owner. A venue response or owner decision is required
before the gate can close.

**Goal.** Close the only read, concrete, unresolved venue prohibition (Aster §6.1(b)/§6.2(e)).

**Changes.** We prepare `docs/ASTER_PERMISSION_REQUEST_V1.md` (draft request text and the exact
usage description). **Owner actions**: send the request, obtain an API key under its terms, stop
using Aster data, or record an accepted-risk decision in the licensing audit.

**Gate.** Until resolved, Aster results stay in a separate, clearly-labelled appendix and never
in headline tables.

### T8 — A2 batch-and-merge scoring past `MAX_SCORED = 32`

**Status (2026-09-20).** Implemented and verified; [T8 report](T8_BATCH_SCORING_V1.md).
Tests cover original-order merge, batch-local malformed/exception/timeout fallback,
global uncertainty and cross-batch dependency closure, metadata identity, and restore
header overflow. 60 old/new single-batch cases preserve all legacy semantic fields and
payloads. Production remains shadow; no model or threshold changed.

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

**Status (2026-09-20).** The protocol/preflight subtask and the first five-arm run are complete.
Three local arms produced zero removals, regressions, or savings on the A4 fixtures; two rebuild arms
failed closed because their components are absent. Provider comparison outputs are internal-only and
are not evidence published by this roadmap.

**Goal.** Settle filter-versus-rebuild on our fixtures with evidence.

**Changes.** Added [`docs/A5_FILTER_VS_REBUILD_PROTOCOL_V1.md`](A5_FILTER_VS_REBUILD_PROTOCOL_V1.md),
the frozen protocol [`research/nanojev_v2_t10_a5_filter_vs_rebuild_protocol_v1.json`](../research/nanojev_v2_t10_a5_filter_vs_rebuild_protocol_v1.json),
`scripts/filter_vs_rebuild_v1.py`, and its 27-test suite. The package binds the A4 manifest by exact
SHA-256, freezes the five arms and eight task families, rejects character-count cost claims, enforces
paired-CI/protected-family gates, and emits only content-free synthetic receipts. The smoke receipt is
[`results/nanojev_v2_t10_a5_filter_vs_rebuild_preflight_20260920.json`](../results/nanojev_v2_t10_a5_filter_vs_rebuild_preflight_20260920.json).

**Gate.** Paired intervals per task family; aggregate win with a protected-family loss is a fail.
The protocol/preflight is not a paired result: no provider/model was called, no request was rewritten,
and no active pruning, training, or deployment is authorized.

### T11 — B3/R3 financial baselines

**Status (W39).** Numerical estimator cores, a feature-only projection boundary and a pinned T11
protocol/preflight are implemented. The original R1 PIT receipt still verifies. The earlier stress-
purged diagnostic remains as a historical failure receipt; T11 v1 treats those windows as descriptive
test strata and defers point-in-time regime masks to T12. The current protocol is valid but real fit
still requires independent review and explicit fit authorization; do not use test for selection.
See [baseline report](FINANCIAL_BASELINES_V1.md).

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
section and one release-safe ledger entry per verified item. Detailed working notes remain local-only.

**Recurring checklist.**
- Re-snapshot star counts (GitHub API) before quoting any of them; record the UTC date.
- Locate primary sources for the still-unverified sweep items: Atomic, Jevinik, the Monad
  live-trading bot, the DuckDB extension, the three cost case studies, Decider-2B, System-One
  4B, the Jev-compatible public API, and the HF-model adapter library.
- Re-check the fast-jev-compaction 156k→62k claim if the project publishes benchmark receipts
  (its README currently documents character-count token estimates only).
- Never install plugins, upload private data, use accounts, or place orders to verify a claim.

**2026-09-20 cycle.** A public GitHub metadata refresh is recorded in
[Ecosystem verification ledger V1](ECOSYSTEM_VERIFICATION_LEDGER_V1.md). It updates 17
repository star counts with a common UTC snapshot and explicitly leaves the previously
unverified claims unpromoted. This is bookkeeping evidence only; it does not change any
acceptance gate or adoption decision.

### T16 — P0 independent review (external)

**Goal.** The three-seed relevance result (no candidate promoted) gets an independent
implementation/evidence review; we do not self-certify. Effort external. Artifact: a review
verdict filed in the local-only execution review log.

**Blockers on our side.** None — evidence package is complete (frozen protocol, three run
receipts, hashes all MATCH, independent report rebuild semantically identical). Waiting on
reviewer availability.

## W26 cross-roadmap note — V4-S0.4 metrics contract/preflight（2026-09-20）

V4-S0.4 的 5 候选、3 latency scopes、M1/M2 target profiles、paired quality/cost/reliability 字段、
canonical protocol hash、只读 validator、29 项专项测试和 run2 preflight 已交付。官方 DeepSeek V4.1
analysis Worker `1789857734-79419c9726da` 对修正后的隔离 staging 返回 `success`、`files_changed=[]`，
确认 cold p99、scope binding、protocol hash、exclusive-create、freeze/approval 与 `pre_approval_only`
六项缺口均已闭合。Worker 只能做只读静态复核，主模型独立重算 hash；官方身份固定为
`https://api.deepseek.com/v1` + `deepseek-flash` → `DeepSeek-V4.1-Flash`、`official_only=true`、无第三方回退。

这只是下一阶段 V4 的 contract/preflight 进展，**不改变本 V2 T1–T16 的外部门禁**：T10 的 provider 对比只保留为本地私有证据，公开路线图仅记录 NanoJev 自有结果；
T5/T8/T9a 已由独立 reviewer ACCEPTED（W33-A/B/C），T6 protocol/PIT gate 已完成但不含训练或测量授权，T7 owner 已决定停用 Aster（W32-B：不申请许可、不记接受风险，Bybit/Hyperliquid 条款仍未读），T9d 独立 protocol review ACCEPT_WITH_CONDITIONS（W33-D，待新协议版本+干净语料），T16 独立复核已 ACCEPTED（W32-A），T8g/T9 阻塞，T11–T14 尚未开始；真实 measurement、训练、
量化、serving、active pruning 和部署均未授权。run2 仍为 `metrics_contract_valid_not_authorized`，不能写成
模型质量、体积、速度、成本或发布证据。W26 回归为主仓库 **922 tests OK（2 skipped）**、安装版 skill
**17/17**，并已通过 `compileall`、`py_compile` 与 `git diff --check`；这些仍只证明代码/契约路径。完整 W26 记录保留在 local-only execution review 与 completion audit 中。

## W27 cross-roadmap note — V1 corpus isolation plan（2026-09-20）

为回应 T8g/T9 的语料隔离阻塞，本轮新增只读计划器及 10 项专项测试。计划同时按 `pair_id` 血缘和
canonical model-visible input digest 建立 deterministic connected components，并显式 fail-closed 报告
`pair_id_crosses_splits`、`source_group_id_crosses_splits`、跨 split components 与
evaluation-derived provenance；不写新语料、不改冻结 split/label、不合并、不训练。

权威收据 `results/engineering_corpus_v2_isolation_plan_20260920_run3.json` 的 SHA-256 为
`904c48ab99ab333fcaa3e8dc6490929b51cd7909aed5cee5e5546258e148f884`，状态为
`blocked_isolation_plan_only`，关键计数为 246 records、28 components、6 split-conflicted components、
27 canonical-input cross-split groups、18 evaluation-derived records、cross-seed common inputs 129、
affected records 174/174；`training_authorized=false`、`merged_rows_written=0`。W27 专项 **10/10**，
主仓库最新 **932/2**，安装版 skill **17/17**。这只是隔离/证据进展，不是训练准备度或模型能力提升。

官方 DeepSeek V4.1 Flash analysis Worker `1789859941-6b1300836f21` 只读复核成功、无文件修改；
doctor/任务收据固定为 `https://api.deepseek.com/v1`、`deepseek-flash` → `DeepSeek-V4.1-Flash`、
`official_only=true`、`fallback_attempted=false`。Worker 未在隔离 staging 执行测试，主仓库回归是实际验证来源。
其残余风险（source-group 目前为 plan-level 阻断、run3 不含这两类 violation、determinism 未有专门
byte-identical 单测）已记录，不能解除 T8g/T9 或 N2/N3-S 门禁。V2 仍为
`AUDIT_COMPLETE_WITH_EXTERNAL_GATES_OPEN`；真实 measurement、训练、量化、serving、active pruning
和部署继续禁止。随后官方文档复核 job `1789860583-1bcca79905ea` 确认 W27/V2/V4 文档口径一致；
W27 receipt 未新增 machine-readable `measurement_authorized`/`deployment_authorized` 字段，保持历史
receipt 不变并由 V4-S0.4 契约继续 fail-closed。

## W28 cross-roadmap note — V4-S0.4 scope-binding negative coverage（2026-09-20）

针对 W26/W27 复核留下的 scope-binding 负向覆盖缺口，新增三条只读测试：M1 cold、M2 warm、M2 cold
target scope 偏离 `local_serving` 时必须报告 `scope_binding` violation。V4-S0.4 专项现为 **32/32**，
主仓库为 **935/2**，安装版 skill **17/17**；contract 仍为 `metrics_contract_valid_not_authorized`，
所有 measurement/training/deployment/network flags 仍为 false。没有加载模型、运行真实 latency、量化、
训练、serving 或部署。

官方 DeepSeek V4.1 Flash analysis Worker `1789861172-bc6de5143aaa` 只读复核成功、无文件修改，确认
新增三条测试与 validator 的双 profile scope 检查一致，cold p99 与授权扫描未被削弱；staging 未执行测试，
主仓库回归是实际验证来源。官方身份固定为 `https://api.deepseek.com/v1`、`deepseek-flash` →
`DeepSeek-V4.1-Flash`、`official_only=true`、`fallback_attempted=false`。

W28 只关闭 V4-S0.4 的一项测试覆盖缺口，不改变 V2 的外部门禁、W27 语料隔离阻塞或 V4-S1/S2/S3 的阶段顺序。

## W29 cross-roadmap note — plan receipt machine-readable boundary flags（2026-09-20）

W27 plan-only receipt 的 schema 已补齐四个显式 false 字段：`measurement_authorized`、
`measurement_performed`、`deployment_authorized`、`deployment_performed`。旧 run3 保留，新 run4
`results/engineering_corpus_v2_isolation_plan_20260920_run4.json` 以 exclusive-create 写出，SHA-256
为 `8800a8192a51bb9accd7a825f4ae0a5040e2fefd64fb7efeba7df9ebf401ee6c`；run3/run4 的 isolation counts、
block reasons、source hashes 与 components 一致，训练/测量/部署均未执行。W29 focused 10/10，主仓库
935/2，skill 17/17。

官方 DeepSeek V4.1 Flash retry review `1789861830-a96a57205f73` 成功、无文件修改；首轮同范围 job
`1789861770-86bd4828e5c2` 为 invalid structured result，亦无文件修改、无第三方回退。W29 不改变 V2
外部门禁或 V4 阶段顺序。

## W30 cross-roadmap note — 文档一致性收口（2026-09-20）

官方 DeepSeek V4.1 Flash 文档复核 job `1789862099-a3fdc102aa0e` 返回 `success`、`files_changed=[]`，
指出并核对三处陈旧口径：W27 的 W29 前临时 run4 注记、审计文档的 latest 回归计数、以及“now 787”措辞。
本轮已将临时输出明确标为历史、审计回归更新至 W27/W28/W29 的 **935/2**，并把 787 改为历史节点；没有
改写 run3/run4 收据或改变 V2 外部门禁。官方身份仍为 `https://api.deepseek.com/v1`、`deepseek-flash` →
`DeepSeek-V4.1-Flash`、`official_only=true`、无第三方回退。

修正后的最终只读复核 job `1789863133-df273189f1e7` 返回 `success`、`files_changed=[]`，独立重算 run3/run4
hash 并确认 W27/W28/W29 计数和四个 false boundary fields 一致；无文件修改或第三方回退。

本轮本地 NanoJev development event `da37c160-883f-4619-a079-26d4a67d2967` 在本地 MPS/FP32、零网络下
abstain；最近决策 event `cee7f210-776b-40bf-a8a3-5e59d92d983f` 的 `fallback` feedback 已记录（receipt
`4c336585-b617-49c5-9929-a89b6bf35420`）；收口 testing event `2e6f2c4d-eedb-4487-84d2-4a84395ce092`
的 feedback receipt 为 `37dd6893-bd8e-4c10-8536-7a4e532ed0b5`=`fallback`，development feedback receipt
为 `85aa4feb-cce9-4437-970a-9ae17255d5df`=`fallback`；optimization event
`748a343b-54f2-4407-b256-8ae394fab089` 的 feedback receipt 为
`5e57511f-c5ca-4d2f-87d0-62305de98481`=`fallback`；deployment readiness event
`888bb9a6-0c92-43bd-9c8a-50845a50d641` 的 feedback receipt 为
`cc589821-ff0d-49ba-bdae-589c50fd69eb`=`fallback`。这些 advisory 记录不构成授权或能力证据。

## W31 cross-roadmap note — T15 ecosystem ledger refresh（2026-09-20）

T15 只读刷新统一使用 UTC snapshot `2026-09-20T00:18:05Z`，账本包含 17 个公开 GitHub 仓库记录。
`achimala/jevinci` 的 API 请求重定向到 canonical `achimala/jev-paint`，已在
`docs/ECOSYSTEM_VERIFICATION_LEDGER_V1.md` 中记录。W31 只更新公开 metadata bookkeeping，不改变
T6/T7/T8g/T9/T9d/T16、N2/N3-S、真实 measurement、训练、量化、serving 或部署门禁；star 数和 README
声明不被当作能力、质量、许可或成本证据。

## W34 cross-roadmap note — T9d protocol v2 与 T8g/T9 isolation run5（2026-09-20）

按 W33-D 条件新增版本化 T9d 协议 `research/nanojev_v2_t9d_fp32_mps_protocol_v2.json`，SHA-256
为 `7d687325040159241d7c92df1103cbdf04f4e85f0e3654e8576c41f8aed02b1a`，并配套说明
`docs/NANOJEV_V2_T9D_FP32_MPS_PROTOCOL_V2.md` 与 fail-closed validator。协议明确
`batch_questions=16`、`max_length=2048`、`device=mps`、`precision=fp32` 是 amendment，
不再把它们表述为运行时默认；同时保留 `brier` 可用、`paired_brier_pg` 在 MPS 被拒的事实。
4 项专项单测和 validator 均通过，但 protocol review、干净语料 review 和训练授权仍未完成。

冻结 V1 语料的只读 isolation run5 收据为
`results/engineering_corpus_v2_isolation_plan_20260920_run5.json`，SHA-256
`8800a8192a51bb9accd7a825f4ae0a5040e2fefd64fb7efeba7df9ebf401ee6c`；结果仍为
`blocked_isolation_plan_only`（246 records、28 components、6 跨 split components、27 canonical
cross-split groups、18 provenance conflicts、129 cross-seed common inputs，174/174 affected）。
未写入 merged rows，`training_authorized=false`，没有改动原始语料。W34 本地 NanoJev development/testing
事件分别为 `92c789ed-8379-4205-85e2-d2c1308b6a13` / `56b92c77-e5e3-4297-97eb-4f776e4262f5`，均
MPS/FP32、零网络、低于 0.9 而 abstain；feedback 已记录。W34 不改变 T8g/T9、T10、T11–T14、
N2/N3-S 或真实训练/量化/serving/deployment 门禁。

## W132/W133 cross-roadmap note — A100 结果与 valen_nano 数据契约修复计划（2026-09-27）

**W132（A100 周末冲刺，已完成）改变了三个先前结论：**

1. **"收敛后 RLCD 无效"被推翻**（T144-eval 的先前判定需修正）：nano_rlcd_v2 在收敛的
   sft_v2 头上 +300 步 RLCD → acc 0.8894→0.9012、brier 0.157→0.135、nll 0.238→0.206。
   rlcd_v1 pilot 的劣势是欠训练假象。RLCD 在 0.8B（+1.2pp）和 2B（+3.4pp）上都有效。
2. **规模不是瓶颈**：2B backbone 反而输 0.8B（SFT 0.861 vs 0.889；+RLCD 0.895 vs 0.901）。
   0.8B 是本任务甜点。
3. **域特异先验双向成立（同题实测）**：valen_nano 678 题——rlcd_v2 0.9012 vs 官方
   jev-1.13.0 0.7341（+16.7pp，官方另有 16 题超时未答）；jevbench 231 题——rlcd_v2 仅
   0.307 vs 官方 ~0.886。**专项头统治自己的域，不是通用 Jev 替代**；生产定位维持
   "本地专项头主力 + `jev-route --mode quality` 官方兜底"。

**W133（错误分析）给出当前最重要的工程事实：**

`valen_nano_v1` 的 v1 生成器存在**标签契约矛盾**——`correction`（keep）与
`overlap_distractor`（drop）在归一化后是同构 state 却标签相反，污染 12.7% 记录，
**eval 理论天花板 0.9572**（pointer-aware 口径；当前 0.9012，剩余真实空间约 5.6pp）。因此优化顺序是
**先修数据契约（T163），再扩数据/扩 eval（T164/T165），最后才谈训练**（T166/T167）——
在矛盾标签上加训练量只会教模型记忆噪声。官方 Jev 在 correction 类题上做对（36 条
仅我方错误中 34 条属此类）证明区分信息客观存在，是标签而非容量问题。

**推进序列（当前阶段细化，2026-09-29）：**

```text
Phase 0 — ✅ 完成 (2026-09-29)
  T179 547 标签入库（owner 委托 agent 裁决，provenance 已披露）
  T180 真实分族表已出（REAL_CONTEXT_EVAL_RESULTS_V1）——
  关键发现：lora FP=5（短/空/负结果+纠错段过度判删，含 0.92 高置信误删）；
  真实 vs 合成排名完全重排

Phase 1 — 并行可做的本机工程（T178/T177 已完成）
  T177 commit 已落地（5cdc203 + 129580f；held-back 文档与大语料保留未提交）
  T178 winnow 重窗口完成，refill 219 条后台补打
  T182a/b 预研完成（Provence GO / EXIT GO）

Phase 1 — 并行可做的本机工程（不等 T179）
  T177 commit 审计执行：脱敏新文档基础设施信息 → owner 确认 → git add 白名单
  T178 winnow 覆盖决策：默认接受部分覆盖；可选 2h 重窗口（winnow tokenizer 重切）
  T182a Provence×ModernBERT 复刻预研：装 transformers + ModernBERT-large，
        读 bergen/scripts/provence 配方，本机推理冒烟（无训练）
  T182b EXIT 配方复刻预研：读 recipe，评估 Qwen3-0.8B LoRA 移植成本
  shadow 8094 继续跑 v4 头攒真实流量观测

Phase 2 — T180 出结果后的分叉
  A) 真实表现达标 → 走 T175 切换评审（shadow 数据 + 独立复核 + 可回退方案）
  B) 真实表现不达标 → v5 数据设计（按 ECOSYSTEM_RESEARCH_V1 §4 七条：
     按源切分 eval / SWE 轨迹结果推导标签 / 真实分布负例 / 非对称规范 /
     多粒度反捷径 / MinHash 去重 / 教师 ensemble）
     → 数据就绪后开 L40×2 重训（runbook 15min 重建）
  C) 中间态（部分族达标）→ 针对性补该族训练数据，不必全量重训

Phase 3 — 长期选项（Phase 2 后）
  T182 完整复刻对标：Provence/EXIT/LLMLingua-2 逐个上 v4+真实 eval
  kev 脚手架参考：--init_from 迁移 / frozen manifest 做法借入我们的 CI
  JevBench 公开子集（231 题）第二意见 eval
```

**关键路径**：T179（owner 打标）是唯一的串行瓶颈；其余全部可并行。

**GPU 触发条件**：仅 Phase 2B/2C 的"需重训"分支。届时 L40×2 按
`docs/L40_MIGRATION_V1.md` 重建（~15min）+ fla sm89 验证 + fp32 parity 重测。

**诚实现状**：合成 v4 已证明"永远可被 LoRA 学透"（lora_v4 eval 1.0000 且
lora_v3 在真实候选 537/547 判 keep 呈过度保留先兆）——合成 acc 不再构成
生产证据，T179/T180 是生产切换前无法绕过的最后一道门。



**定位。** 长期目标保持不变，但公开路线图仅使用 NanoJev 自有数据、固定公开评测输入和
本地可复现收据。第三方模型输出、对比数字和 side-by-side 收据均为本地私有证据，不进入
公开源码树，也绝不进入训练数据。本节不授权任何部署或生产变更；既有 review 门禁继续生效。

### J-0 NanoJev 自有基线与进展

| 测量面 | NanoJev 自有指标 | 当前解释 |
|---|---:|---|
| L1 迷宫 test 原子判断 | accuracy 0.750；Brier 0.166 | 域内能力存在，仍需改善校准 |
| L1 迷宫 OOD 任务完成率 | 0.50 | OOD 迁移仍是缺口 |
| L2 workflow boolean | 0.42 | 系统性 say-true 操作点偏差 |
| L2 workflow choice / score | 0.81 / 0.88 | 域内结构化选择明显强于 boolean |
| L2 `known_chance` boolean | 0.21 | 当前最严重的校准/操作点失败 |
| L3 工程判断 choice / score | 0.32 / 0.25 | 工程 OOD 泛化不足 |
| 独立 heldout 原子基线 | 0.464 | 69 题、23 个独立 source group |
| v3 head-only 三 seed 均值 | 0.522 | 相对自有基线 +5.8pp |
| v4 LoRA r16 三 seed 均值 | **0.536**；最佳 seed 0.580 | 当前最佳训练臂；差值仍处于小样本噪声带 |

### J-D 阶段：根源锁定诊断

| ID | 诊断目标 | NanoJev-only 方法 | 结论/判据 |
|---|---|---|---|
| J-D1 | choice 失败是否源于位置偏置 | 对相同题目做 identity/reversed/rotated 候选排列 | 已推翻位置偏置：语义 acc 不随排列变化 |
| J-D2 | boolean 是域失配还是操作点失败 | 按 family 分解混淆矩阵与 oracle threshold | say-true 偏置为主；信号存在但 operating point 错 |
| J-D3 | `known_chance` 是模式错还是置信错 | 报告 argmax 一致率、KL 与 TV | choice/score 过锐；boolean 模式错且平 |
| J-D4 | 0.9 弃权门是否可服务 | 绘制 NanoJev confidence→accuracy/coverage 曲线 | 默认门覆盖率 0%，不可通过降门限绕过 |
| J-D5 | backbone 与 head 归因 | base letter-readout、fine-tuned readout、NanoJev head 同题比较 | head/训练是主问题；小语料全骨干训练会损伤迁移 |

### J-R 阶段：基于现有证据重新选路（W46 修订）

已经得到的证据改变了执行顺序：

1. v2 小语料全骨干训练过拟合；扩大到 v3 后全骨干不再崩溃，但独立 heldout 没有净增益。
2. head-only 三 seed 有正增益；LoRA r16 进一步取得当前最佳均值，说明首要问题是
   **可训练参数规模 × 数据规模 × readout 形式**，不是简单堆全骨干参数。
3. grouped calibration 只帮助未校准 atomic 基线，对较好的已训模型没有迁移收益；因此不再把
   post-hoc calibration 当作主能力路线。
4. SemIf 已公开 direct-logit、MLX、shared-prefix/parallel-suffix 和证据校验实现；JevBench 已公开
   通用 typed-decision harness、原始能力轴和外部 heldout 路径。继续自建社区排行榜是重复劳动。

因此，下一阶段从“扩自建比较 → 继续调参”改为：
**内部数据与安全门禁 → SemIf-style readout/runtime ablation → 冻结候选 → JevBench 外部验收**。

### J-A 阶段：架构收敛

| ID | 内容 | 固定比较 | 通过条件 |
|---|---|---|---|
| J-A1 | **同 backbone readout 矩阵** | Qwen3-0.6B 上比较 current LoRA head、native direct logits、LoRA+pointer head | 相同序列化、相同训练数据、相同 dev-only 选点；三 seed |
| J-A2 | **候选顺序与基数压力** | identity/reverse/rotation；2/4/8/32/255 candidates | ✅ **done (W46-B)**: LoRA seed18 + atomic 均 6 种排列零 argmax 翻转（max \|Δp\| ≤7.5e-8）——位置不变性成立；基数升高稳定性下降但 LoRA 全面优于 atomic；255-option contract 验证通过 |
| J-A3 | **shared-prefix / parallel-suffix** | fresh independent forward 为数值参考 | 概率误差在预注册容差内；protected cases 零 answer flip，之后才测速度 |
| J-A4 | **分阶段更强基座容量臂** | 0.6B 在 corpus v4 + heldout_v2 上未泛化，容量门已满足，但先做 J-D6 诊断 | 先 MiniCPM5-2B 兼容/内存/owned-fixture smoke；再 Qwen3.5-4B（SemIf/JevBench 公开证据最强）；每个模型固定 revision/license，新协议、新预算、新基线；不得使用 SemIf/JevBench/官方模型输出训练或选参 |

SemIf 是实现与测试方法参考，不是可直接宣称的 NanoJev 结果。若复制源代码，保留 MIT
attribution；若重写实现，也必须保留独立-forward parity 和 evidence-manifest 纪律。

#### W51 公开参考复核后的容量路线

- **SemIf（pinned `ca3ba65f…`）**在其 owned workload 报告 Qwen3-0.6B 0.440、
  MiniCPM5-2B 0.686、Qwen3.5-4B 0.813 balanced accuracy；并显示 direct typed logits、
  shared-state reuse 和 parallel suffix 是有价值的系统方法。这是外部参考证据，不是 NanoJev
  结果，也不能把其 rows/outputs 用于训练或选参。
- **JevBench v1.2.4（pinned `83831807…`）**的公开排行榜同样显示 Qwen3.5-4B
  typed-decision systems 接近领先组，但也记录某实现仅交换 yes/no 选项顺序即从 72% 降到
  21%。因此更强基座不能跳过 permutation/candidate-set gate，且 public per-task failures
  继续禁止进入调参循环。
- **laya-mlx（pinned `fc1df628…`）**证明 421M 双向 encoder 可以在 Apple Silicon
  达到低延迟并通过 port parity；这支持未来 domain-pack/专用 encoder 路线，但其 512/1024
  context、上游权重与训练方法不同，不能直接解释 NanoJev v5 退化。
- **TypeSafe live docs（W51 复核）**明确 confidence 只是分布形状统计，阈值必须按具体
  workload 和风险验证；这与 J-C 的 OOD 迁移失败完全一致，禁止把更大模型的高 confidence
  当作安全证据。

由此采用**诊断优先、容量阶梯**：J-D6 先用已有冻结输出拆解 v5 失败；J-A4a 先验证
MiniCPM5-2B，再验证公开 typed-decision 证据更强但资源更重的 Qwen3.5-4B。只有兼容、
内存、parity、排列和 owned-fixture 门都通过才进入 J-A4b 训练；不直接开启三 seed 4B。

### J-Data 阶段：继续扩数据，但隔离外部 benchmark

| 数据资产 | 下一版目标 | 隔离规则 |
|---|---|---|
| corpus v4 train/dev/calibration/test | ≥4,000 items、≥30 source families；扩大 choice/score、否定、规则应用和跨域状态绑定 | 独立编写或许可证明确的 gold labels；跨 split component 零重叠 |
| engineering heldout v2 | ≥300 items、≥50 source groups；三题型与 option-count 分层 | 评测专用；与 v1/v2/v3/v4 canonical input 零重叠 |
| JevBench public tasks | 固定为 external milestone | `evaluation_only=true`、`training_allowed=false`、`calibration_fit_allowed=false` |
| SemIf/TypeSafe/其他模型记录 | 架构研究或私有比较 | 输出、概率、标签、分布均不得进入训练或 calibration fit |

任何基于 JevBench 单题错误的定向训练都会污染外部测试。公开 milestone 只允许消费聚合的
family/topic 结果来规划**独立新题族**，不得复制题目、改写同题或把输出当 label。

### J-E 阶段：双层验收

#### J-E1 内部选择门（决定是否冻结候选）

- 只用 NanoJev dev 选择 checkpoint/hyperparameters；test 与 heldout 只读一次。
- 至少三 seed；报告均值、标准差、state/source-group clustered interval 和最差 seed。
- expanded heldout 相对当前 LoRA 均值必须有预注册的实际增益，并且 boolean/choice/score
  任一类型不得显著退化。
- NLL、Brier、ECE、coverage-risk 同报；confidence≥0.9 错误不得增加。
- 通过 candidate-order、255-option、save/reload identity、prefix parity 和 MPS/FP32 重现门。

#### J-E2 JevBench 外部里程碑（不参与选模循环）

1. X0 冻结 v1.2.4 source hash、adapter mapping、primitive semantics、raw latency/cost scope 和数据禁用标志。
2. X1 用 synthetic fixtures 验证 adapter；review 通过前不读取 public benchmark rows。
3. 候选在 J-E1 冻结后只运行一次 JevBench public milestone；同时报告 Intelligence、Calibration、
   Speed、Cost 原始轴、raw p50/p95 和 benchmark-adjusted 值，综合排名只作摘要。
4. 不根据 public per-item failures 回训；下一次公开运行必须对应新的预注册 major candidate。
5. 达到内部发布门后，提交 maintainer-held-out evaluation。公开结果通过外部来源链接，不复制
   其他系统的 per-task 输出或官方 Jev 概率到仓库。

### 外部战略评审吸收（对齐 "本地高速决策操作系统" 范式）

外部评审提出五个方向。与 NanoJev 已有证据的对照和吸收方式：

1. **实时决策运行时（p95/p99 而非参数量）——确认并已基本落地。**
   X3 已交付 shared-prefix/packed candidates（parity gate 通过、k=255 提速 5.09×）。
   剩余差距：early-exit、CPU/MPS 自适应调度、并发/取消/超时语义、内存预算基准——
   归入 J-P 产品化阶段，不阻塞当前训练关键路径。
2. **校准优先——采纳但修正表述。** T9c 证明 post-hoc 校准对弱基线有效、对已训模型
   无效；"校准优先"被重述为"**selective-risk 服务层优先**"（J-C）：coverage-risk 曲线、
   per-type 操作点、OOD abstention routing 是独立于最终模型的 machinery，可在
   v4 训练并行期构建，fit 只用 calibration split。
3. **Domain packs——采纳为 J-DP。** 与现有 family/group 隔离纪律同构；硬性规则：
   每个 pack 独立 train/dev/calibration/heldout/OOD，换 seed 不等于独立数据。
   金融 pack 保持 research-only。
4. **主模型前置层——已是既定定位。** 与本地 decider skill 的 advisory 角色一致；
   "减少 token" 必须用真实 provider usage + 下游成功率证明（沿用 A5 已确立标准）。
5. **离线产品化——采纳为 J-P。** 断网可用、无凭据、hash 钉权重/协议/tokenizer、
   升级/回滚/kill-switch、默认 shadow、receipt 不落原始私有文本。

**修正后的阶段顺序**（外部建议的 X4–X9 编号与既有任务冲突，已重映射）：

```text
X4 corpus v4（done）→ T9d-v5 v4 LoRA 重训 → J-C selective-risk 层（可并行构建 machinery）
→ J-E1 内部冻结 → X5/X6 JevBench → J-DP domain packs → J-P 离线 SDK → shadow gateway
```

外部建议与证据一致的不建议方向同步确认：不做通用聊天模型、不单纯扩参（J-A4 保持
条件触发）、不只追平均延迟、不用高置信掩盖错误、不把官方输出当训练数据、
OOD 未稳定前不启用主动上下文删除、金融模拟≠交易能力。

### 停止项与降级项

- **停止**扩展自建 community-wide leaderboard；JevBench 已覆盖该职责。
- **停止**把频繁官方 provider 调用作为训练迭代信号；它只能是可选、私有、owner-reviewed sanity check。
- **降级**单独 temperature sweep：只有新模型在独立 calibration split 上出现明确可校准误差时才重开。
- **保留**NanoJev 自有 L1/L2/L3/engineering heldout，因为外部排行榜不能替代项目特定安全、
  abstention、255-option 和 context-removal 门禁。

### NanoJev 公开成功条件

- **Level 1 — OOD 改善**：迷宫 OOD 完成率相对自有 0.50 基线显著提升，paired CI 排除零。
- **Level 2 — 校准改善**：冻结 cohort 上 NLL、Brier、ECE 与 coverage-risk 均改善，且不新增
  confidence≥0.9 的错误。
- **Level 3 — 工程泛化**：expanded engineering heldout 的三 seed 均值达到 0.70，且任一题型不低于 0.60。
- **Level 4 — 外部可比性**：内部 Level 1–3 通过后完成 JevBench public + maintainer-held-out；
  报告 NanoJev 自己的四轴结果、协议版本和限制。
- **Level 5 — 长期目标**：在不牺牲本地性、typed contract、安全门和证据完整性的前提下，
  在独立外部同工作负载测量中达到领先；不由单一 composite score 或一次运行宣布。

### 与其他线的关系

- J-A/J-Data/J-E 不解除 T8g adapter contract、N2/N3-S 或部署门禁；金融线独立推进。
- TypeSafe MCA §2.3(b) 的蒸馏、模仿训练和竞品开发限制继续作为 fail-closed 边界。
- 公开仓只保存 NanoJev 代码、协议、许可证允许的公开评测输入和 NanoJev 自有指标；
  provider outputs 与私有 side-by-side receipts 保留在 ignored local paths。



Status: **implemented for structured local decisions and local context-gate shadow/value measurement; active universal filtering remains disabled and unreviewed**

NanoJev is packaged as the `nanojev-local-decider` Codex skill under [`integrations/codex-skill/nanojev-local-decider`](../integrations/codex-skill/nanojev-local-decider). The installed copy lives in the local Codex skills directory and uses a persistent loopback HTTP service, preferring `127.0.0.1:8765` and remembering an automatic fallback port if that port is already occupied by another local service.

The current skill is intended for routing, candidate selection, confidence gates, verification, and computer-use decisions. It is not yet a universal context filter, chat replacement, financial trading system, or live execution engine. It records one privacy-preserving JSONL event per decision by default, including task tags, schema types, candidate cardinalities, latency, confidence, abstention, runtime, checkpoint identity, and a request fingerprint. Raw states and criteria remain excluded unless a user deliberately enables local debug payload capture.

Downstream outcomes are recorded separately through `record-feedback`, using `correct`, `incorrect`, `abstained`, `fallback`, or `human_override`. The summary command exposes latency, confidence, abstention, task type, and feedback coverage for later calibration, dataset construction, and optimization. This telemetry is an input to both primary tracks; it is not ground truth until feedback or an independently verified outcome exists.
