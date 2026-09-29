# Offline benchmark protocol V1

This document defines the public offline comparison published in
[`BENCHMARKS.md`](../BENCHMARKS.md) and summarized in the root READMEs.

## Scope

The frozen bundle is [`data/jevbench_offline_bundle_v1`](../data/jevbench_offline_bundle_v1):

- `1,720` total rows;
- `893` labeled rows;
- official JevBench public rows;
- SemIf authored, perturbation, and unlabeled shape fixtures;
- WANLI-derived labeled rows;
- Every-derived labeled/retrieval rows.

Manifest SHA-256:

```text
1fc3234acf806016adfd9906fade547ca597f87acf749aecac1ba706f6dd8ee5
```

The bundle is **evaluation-only**. Do not use benchmark rows, labels, provider
outputs, model probabilities, or benchmark-derived thresholds for training or
calibration fitting.

## Reproduction commands

Verify the public bundle bytes and manifest:

```bash
python3 scripts/build_offline_jevbench_bundle_v1.py \
  --root data/jevbench_offline_bundle_v1 \
  --verify
```

Normalize a completed receipt:

```bash
python3 scripts/score_official_jev_bundle_v1.py \
  --bundle-root data/jevbench_offline_bundle_v1 \
  --receipt <receipt.jsonl> \
  --output-prefix results/offline_bundle_<candidate>_v1
```

Rebuild the aggregate scorecard:

```bash
python3 scripts/update_offline_bundle_comparison.py \
  --candidate winnow_12b_q8=results/offline_bundle_winnow12b_q8_v1.summary.json \
  --markdown BENCHMARKS.md
```

The updater intentionally requires at least one `--candidate` argument but merges
all normalized `results/offline_bundle_*.summary.json` files that are already
present. Raw receipts remain local and are not required to render the public table.

## Metric definitions

| Metric | Definition |
|---|---|
| Acc | Top-label accuracy over labeled rows. |
| Δ vs Jev | Candidate accuracy minus Official Jev direct accuracy on the same labeled rows. |
| Speed× | Official Jev direct wall time divided by candidate wall time. This is a coarse comparison only. |
| BalAcc | Mean per-class recall over labeled rows. |
| NLL | Negative log likelihood of the gold label; lower is better. |
| Brier | Multiclass Brier score; lower is better. |
| Cov@0.9 | Fraction of labeled rows whose selected-label probability/confidence is at least 0.9. |
| CW@0.9 | Count of wrong answers at confidence at least 0.9; lower is better. |
| Perturb flip | Fraction of the 108 SemIf perturbation pairs whose argmax changes. |
| Every R@1 / MRR | Retrieval ranking metrics on the 19-query Every subset. |
| Wall s | End-to-end local wall time for all submitted bundle rows. |
| p50 / p95 s | Per-row latency percentiles. |
| Err | Rows with transport/runtime/schema errors. |
| Tok in / Tok out | Adapter-reported token totals. `0` can mean “not instrumented” for some adapters. |

Official Jev direct uses remote response latency; local wall times are not strictly
speed-comparable with it. The table reports the observed numbers without treating
remote service latency as local model compute.

## Timeout and elimination policy

`local_eval_timeout_policy_v1` is recorded in
`research/oss_jev_reproduction_matrix_v1.json`:

- after 200 rows, stop if per-row p95 exceeds 5 seconds;
- stop if projected full-bundle wall time exceeds 3,600 seconds;
- stop if no progress is made for 300 seconds;
- retain the partial receipt and mark the candidate `timeout_eliminated`;
- an explicit owner exception is required to continue.

## Publication boundary

Public artifacts may contain:

- aggregate scores and run metadata;
- bundle inputs, labels, licenses, manifests, and provenance;
- normalized summary JSON;
- the generated `BENCHMARKS.md` table.

Private/local-only artifacts must not be published:

- raw Official Jev or other provider per-item responses;
- raw per-item probabilities or prediction receipts;
- side-by-side provider receipts;
- API keys, credentials, local paths containing secrets, or transient logs.

## Candidate evidence ledger

Candidate source URLs, revisions, hashes, statuses, and timeout notes live in
`research/oss_jev_reproduction_matrix_v1.json`. Public tables should cite that
matrix or the linked upstream sources rather than copying mutable claims without a
snapshot.

## Current interpretation

Winnow-12B Q8 is the strongest reproduced open candidate on this bundle
(`0.8824` accuracy, `494.6s` wall), but it has `59` confident-wrong answers at the
0.9 threshold and a private training mixture. Reflex is the strongest low-latency
operating point (`0.7895`, `246.4s`, `10` confident-wrong). Neither result alone
authorizes active filtering or deployment.
