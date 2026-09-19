# Workflow V2 baseline and robustness challenge

Status: measured local baseline, **not a release-quality context gate or financial decision engine**.

## Scope and receipts

Checkpoint: `checkpoints/local_atomic_seed17/variants/local_atomic_seed17`, unchanged throughout both evaluations. Device: local macOS MPS, FP32. No external model calls, training, threshold selection, or provider changes were performed.

- [Original evaluation](../results/workflow_v2_baseline_seed17.json): 816 questions from 272 source groups (528 test questions, 288 OOD questions).
- [Challenge evaluation](../results/workflow_challenge_v2_seed17.json): 3,264 questions, four variants of those **same 272 groups**, not 1,088 independent market/user situations.
- [Paired challenge analysis](../results/workflow_challenge_v2_paired_seed17.json): 54 split/family/type/variant comparisons, 1,000 source-group bootstrap resamples with seed 17.

The underlying workflow dataset contains 1,008 states / 3,024 questions across all five splits. Training, development, and calibration records were validated for source overlap but were not sent to the model in these evaluations. Families are smart-home rules, catalog lookup, and known-chance distributions. Each has Boolean, Choice, and Score questions.

The evaluator records exact input files, evaluated-record and prediction hashes, checkpoint hashes, evaluator/dependency code hashes, runtime dependencies, hardware, batching settings, and effective input length. Metadata and gold labels do not enter the model payload. The normal loader audits state/source-group isolation before selecting evaluation splits.

## Results

Deterministic accuracy excludes analytic-probability questions: each variant has 336 deterministic test questions and 192 deterministic OOD questions. Score accuracy here is argmax-level accuracy, not the expected-score readout used by some applications.

| Input variant | Test deterministic accuracy | OOD deterministic accuracy |
|---|---:|---:|
| Original | 85.42% | 95.83% |
| Unrelated archived distractor | 75.30% | 74.48% |
| Structured envelope with archived distractor | 76.49% | 76.56% |
| Choice candidate order reversed | 85.42% | 95.83% |

The archived material includes irrelevant authorization, warehouse, and probability facts. Questions explicitly instruct the model to use only the current record. Structured envelopes also contain the archive, so that arm does **not** isolate serialization effects from distraction. The existing runtime renders object states using Python string representation, not a new canonical JSON serializer.

| Test task | Original | Archived distractor | Structured envelope |
|---|---:|---:|---:|
| Smart-home Boolean authorization | 43.75% | 43.75% | 43.75% |
| Catalog Boolean packed/not-shipped | 75.00% | 56.25% | 58.33% |
| Catalog Choice color/size matching | 100.00% | 54.17% | 58.33% |
| Smart-home Choice device selection | 100.00% | 95.31% | 96.88% |

Known-chance test Boolean total variation from the true distribution is 0.34284 originally, 0.35704 with the archive, and 0.37543 with the structured envelope (lower is better). These are analytically known distributions; no realized random outcomes were sampled. They are not trading calibration results.

Candidate-order reversal produced zero top-choice flips across 272 Choice questions. The maximum absolute probability drift was about `1.19e-7`. Boolean/Score questions in that arm are unchanged controls and must not be counted as candidate-permutation tests.

## Paired uncertainty and failure interpretation

Examples from the full paired report, with accuracy deltas expressed in percentage points:

- Catalog Boolean test, archive minus original: -18.75 pp; conditional 95% cluster-bootstrap interval [-31.25, -8.33] pp.
- Smart-home Boolean OOD, archive minus original: -43.75 pp; interval [-62.50, -28.125] pp.
- Smart-home Score test retains the same top decisions with the archive, but expected Brier worsens by 0.10583; unchanged accuracy does not imply unchanged probability quality.

The intervals resample complete source groups, keeping related questions paired. They describe one fixed checkpoint on this synthetic cohort, not uncertainty across training seeds. The 54 intervals are descriptive and are not multiplicity-corrected discovery tests. All comparisons are retained, including flat or unfavorable results. Confidence >=0.9 selected/wrong counts are also reported; this fixed diagnostic is not a calibrated production threshold.

These failures block any claim that the checkpoint is ready for autonomous context removal. They motivate training-only provenance/distractor curricula, calibrated abstention, and fail-open shadow adapters. They do not justify fine-tuning on these evaluation records or selecting a deployment threshold from them.

## Runtime

| Measurement | Original | Four-variant challenge |
|---|---:|---:|
| Questions | 816 | 3,264 |
| Requests / forward passes | 34 / 34 | 136 / 136 |
| Questions/second | 26.72 | 10.44 |
| Warm batched request p50 | 873.43 ms | 2,108.90 ms |
| Warm batched request p95 | 1,275.70 ms | 3,927.53 ms |
| Warm batched request p99 | 1,318.91 ms | 4,527.10 ms |
| Model load | 7.60 s | 7.34 s |
| Peak process RSS | 5,279,416,320 bytes | 5,282,807,808 bytes |

Each request contains eight states / 24 questions with varying candidate paths. Request latency includes tokenization, inference, and host output conversion. Throughput includes first-request time and result processing, but excludes loading and provenance hashing. These are not single-decision compute times, an isolated system benchmark, or evidence of millisecond financial latency. RSS is not a complete measurement of Metal/unified-memory footprint. Input lengths and candidate composition differ between runs; the throughput ratio is not an optimization comparison.

## Reproduction

Run from the repository with its Python environment and existing local checkpoint. Use new output directories for fresh reproductions; the challenge builder refuses to overwrite a nonempty directory. Large source/prediction JSONL files stay in the local ignored `data/` directory; committed reports retain their hashes. GPU arithmetic may differ slightly across runtime/hardware, so byte-identical probabilities are not promised across devices.

```bash
.venv/bin/python scripts/build_workflow_decisions.py --seed 20260917 --output-dir data/workflows_v2
.venv/bin/python scripts/evaluate_raw_dataset_v2.py --checkpoint checkpoints/local_atomic_seed17/variants/local_atomic_seed17 --input data/workflows_v2 --splits test,ood --batch-states 8 --device mps --precision fp32 --predictions data/workflow_v2_baseline_predictions.jsonl --output results/workflow_v2_baseline_seed17.json
.venv/bin/python scripts/build_workflow_challenge_v2.py --input data/workflows_v2 --output-dir data/workflow_challenge_v2
.venv/bin/python scripts/evaluate_raw_dataset_v2.py --checkpoint checkpoints/local_atomic_seed17/variants/local_atomic_seed17 --input data/workflow_challenge_v2 --splits test,ood --batch-states 8 --device mps --precision fp32 --predictions data/workflow_challenge_v2_predictions.jsonl --output results/workflow_challenge_v2_seed17.json
.venv/bin/python scripts/compare_workflow_variants_v2.py --predictions data/workflow_challenge_v2_predictions.jsonl --output results/workflow_challenge_v2_paired_seed17.json --bootstrap-samples 1000 --seed 17
.venv/bin/python -m unittest discover -s scripts -p 'test_*.py'
.venv/bin/python -m unittest discover -s integrations/codex-skill/nanojev-local-decider/scripts -p 'test_*.py'
.venv/bin/python scripts/test_calibrated_objectives.py --output data/calibrated_objectives_recheck.json --mc-repeats 2000 --threads 1
```

Validation at this milestone: unittest reports 126 tests with two skips (optional frozen three-map pilot and optional `scaled_games_v4` artifact absent); three local-skill tests pass. The separate calibrated-objective checks also pass exact enumeration, 2,000-repeat Monte Carlo, dynamic/Boolean checks, invalid-input checks, and grouped-mean/padding checks. The external read-only review worker timed out without edits; its attempt is not counted as successful independent review.

## Remaining evidence

- Human-reviewed and genuinely new task-family challenges are still missing.
- Three matched training seeds, calibration-only threshold fitting, and production-style task success are still required.
- Provider-neutral shadow gating and paired main-model token/quality measurements remain unimplemented.
- No financial point-in-time cohort, simulator, live execution, or profitable strategy is established by this work.
- Next implementation should preserve the original request in shadow mode and keep all protected instructions outside any learned deletion policy.
