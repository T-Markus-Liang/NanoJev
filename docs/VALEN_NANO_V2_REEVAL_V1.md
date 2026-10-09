# Valen Nano — re-eval on `valen_nano_v2` (clean-label eval set)

**Status:** local re-evaluation complete. No training, deployment, or config changes are
attached to this document. Follow-up to `VALEN_NANO_V2_CONTRACT_FIX_V1.md`.

## Setup

- Eval data: `data/valen_nano_v2/eval.jsonl` (678 records / 678 `noul` questions,
  sha256-prefix `data_sha256` recorded in each `metrics.json`).
- Entrypoint: `python -m valen.evaluate --checkpoint CKPT --data DATA --output OUT --device mps`
  from `external/valen`, venv `external/valen/.venv` (torch 2.6.0, transformers 5.4.0),
  dtype bf16 from checkpoint config (forward-only; fine on MPS per `MPS_PATCH_NOTES.md`).
- Runtime: ~3 min per checkpoint on MPS (single process, ~0.27 s/record).
- Outputs: `external/valen/output/<name>_eval_v2/{metrics.json,predictions.jsonl}`.
- Because the v1 reference numbers were produced on other hardware (A100 / CPU) and bf16
  forward results drift across devices (see caveat), each checkpoint was also re-run on
  `data/valen_nano_v1/eval.jsonl` on MPS into `output/<name>_eval_v1_mps/` for an
  apples-to-apples same-device comparison.

## Label delta v1 → v2 (join by `meta.record_id`)

- 678/678 record ids match 1:1; zero request-payload changes.
- **31 records changed labels**, all on qid `irrelevant`:
  `true→false`: 23, `false→true`: 8.
- By domain: risk 8, robotics 7, order 5, multilingual 4, support 4, code 3.

## Accuracy: v1 vs v2

| checkpoint | v1 acc (orig device) | v1 acc (MPS) | v2 acc (MPS) | Δ v1→v2 (MPS) |
|---|---|---|---|---|
| `nano_rlcd_v2` (production head) | 0.9012 (A100) | 0.8894 | 0.8879 | −0.0015 |
| `nano_sft_v2` (SFT baseline) | 0.8894 (CPU) | 0.8894 | 0.8938 | +0.0044 |
| `nano_rlcd_2b` | 0.8953 (A100) | 0.8879 | 0.8923 | +0.0044 |

v2 per-domain accuracy (MPS):

| domain | rlcd_v2 | sft_v2 | rlcd_2b |
|---|---|---|---|
| code | 0.8769 | 0.8923 | 0.8846 |
| multilingual | 0.8197 | 0.9016 | 0.8689 |
| order | 0.9103 | 0.8690 | 0.9172 |
| risk | 0.8957 | 0.9202 | 0.8773 |
| robotics | 0.8810 | 0.9524 | 0.9048 |
| support | 0.8978 | 0.8686 | 0.8978 |

v2 overall brier / nll (MPS): rlcd_v2 0.1616 / 0.2498; sft_v2 0.1483 / 0.2274;
rlcd_2b 0.1666 / 0.2760.

## v1-error breakdown (same-device MPS predictions)

Predictions are bit-stable across identical-request MPS runs (0 label flips between the
v1-data and v2-data runs), so correctness on relabeled records simply inverts with the label.

| model | v1 errors | (a) on relabeled records, now correct | (b) still wrong on v2 | new v2 errors (all relabel-caused) | v2 errors |
|---|---|---|---|---|---|
| nano_rlcd_v2 | 75 | 15 | 60 | 16 | 76 |
| nano_sft_v2 | 75 | 17 | 58 | 14 | 72 |
| nano_rlcd_2b | 76 | 17 | 59 | 14 | 73 |

On the 31 relabeled records, correctness moved 16→15 (rlcd_v2), 14→17 (sft_v2),
14→17 (rlcd_2b). The relabel net-hurts the production head by 1 record and net-helps
the other two by 3 — i.e., rlcd_v2's v1 lead over sft_v2 was partly alignment with the
contradictory labels; on the clean contract the two 0.8B heads are within noise
(0.8879 vs 0.8938) and sft_v2 nominally leads.

## Caveats

- The original v1 metrics ran on A100 (rlcd_v2, rlcd_2b) and CPU (sft_v2). bf16 forward
  logits drift substantially across devices: 50/678 predicted labels differ between the
  A100 v1 predictions (`output/nano_rlcd_v2_eval/predictions.jsonl`, mirrored at
  `/tmp/rlcd_v2_preds.jsonl`) and the same-requests MPS run, with 102 unchanged-label
  records moving p(true) by >0.1. Cross-device accuracy deltas of ~±0.01 are within this
  device noise; the same-device columns above are the reliable comparison.
- For reference, the A100-prediction breakdown (v1 errors = 67): 15 relabel-fixed,
  52 still wrong under v2 labels, 16 v1-correct made wrong by relabeling — directionally
  identical to the same-device analysis.
- `nano_rlcd_2b` required `models/Qwen3.5-2B` (present locally, 4.3 GB).
