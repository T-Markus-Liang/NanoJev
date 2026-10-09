# V5_ACCEPTANCE_RUNBOOK_V1 — v5 head acceptance loop

Companion to `docs/V5_DATA_DESIGN_V1.md` (§7.2 eval tiers, §9 gates G1–G6).
This runbook covers: assemble the eval manifest → train on L40×2 → score each
tier locally → run the gate checker. It authorizes nothing beyond what the
design doc already does; consensus stays shadow/advisory.

## 0. Components

| piece | path | role |
|---|---|---|
| eval assembler | `scripts/assemble_v5_eval_v1.py` | emits `data/v5_eval/eval_manifest.json` — merged pointers for tiers T1–T4 (read-only; no payload copies) |
| gate checker | `scripts/check_v5_gates_v1.py` | G1–G6 pass/fail/not_ready over a results dir |
| scorer harness | `scripts/eval_systemone_backend_v1.py` | posts each eval record's `request` to a `/v1/systemone` backend, writes `{i, ms, noul, model, target}` rows |
| tests | `scripts/test_v5_gates_v1.py` | synthetic fixtures; `python3 scripts/test_v5_gates_v1.py` must exit 0 |

## 1. Assemble the eval manifest

```bash
python3 scripts/assemble_v5_eval_v1.py
# -> data/v5_eval/eval_manifest.json
```

Tier status at first run:

| tier | content | status |
|---|---|---|
| T1 real-frozen | `data/real_context_eval_v1/eval.jsonl` (525) + `drop_supp/eval.jsonl` (65) = **590 scored** | ready |
| T2 real-v5-ext | reserved pool = `data/v5_mining/pool_split.json:eval_pool_hashes` (**18 transcripts**); eval lands at `data/real_context_eval_v5_ext/eval.jsonl` | pending_mining_and_labeling |
| T3 synthetic | `data/valen_nano_v4/eval.jsonl` (4251) ready; v5 synth eval expected at `data/valen_nano_v5/eval.jsonl` (or `data/context_relevance_v5_seed20261201/eval.jsonl`) | partial |
| T4 transfer holdout | `data/v5_transfer_holdout/eval.jsonl`; withheld source family fixed once the v5 train manifest exists | pending_definition |

Everything is `training_allowed=false`, `evaluation_only=true`. T2/T4 pending
is normal — the checker reports those gates as `NOT_READY` rather than
failing.

## 2. Train on L40×2 (GPU)

Follow `docs/L40_MIGRATION_V1.md` verbatim, with v5 substitutions:

```bash
# sync data + code to the L40 box (mirror the Mac path via symlinks)
rsync -a data/valen_nano_v5/ root@<HOST>:/root/gpufree-data/nanojev/data/valen_nano_v5/

# dual-GPU, output dirs bumped to nano_*_v5 — never overwrite v4 artifacts
CUDA_VISIBLE_DEVICES=0 python -m valen.train --config configs/cuda/sft_nano_v5_cuda.json &
CUDA_VISIBLE_DEVICES=1 python -m valen.train --config configs/cuda/sft_text_nano_v5_lora.json \
    --initialize output/nano_sft_v4/latest
```

Mandatory after hardware/family change (L40 runbook §6): fp32 parity retest
before accepting numbers. Pull the `nano_*_v5` checkpoint back; do not modify
`nano_*_v4` dirs.

## 3. Serve + score locally (MPS fp32, same-device rule)

Deploy the v5 head on a fresh port — keep `:8094` (lora_v4) running as the
incumbent. Suggested: v5 sidecar on `:8095`.

Score every tier that exists into one results dir, using the slot filenames
the checker expects:

```bash
RD=results/v5_acceptance_<tag>; mkdir -p $RD
B=http://127.0.0.1:8095/v1/systemone

# T1 frozen 590 (G1, G2, G5, G6)
python3 scripts/eval_systemone_backend_v1.py --url $B \
  --data data/real_context_eval_v1/eval.jsonl          --output $RD/lora_v5_frozen_main.jsonl
python3 scripts/eval_systemone_backend_v1.py --url $B \
  --data data/real_context_eval_v1/drop_supp/eval.jsonl --output $RD/lora_v5_frozen_supp.jsonl

# T3 synthetic (G4 leg A). Leg B only once the v5 synth eval exists.
python3 scripts/eval_systemone_backend_v1.py --url $B \
  --data data/valen_nano_v4/eval.jsonl                 --output $RD/lora_v5_synth_v4.jsonl
python3 scripts/eval_systemone_backend_v1.py --url $B \
  --data data/valen_nano_v5/eval.jsonl                 --output $RD/lora_v5_synth_v5.jsonl   # when built

# T4 / G3 (when the transfer holdout is defined and built)
python3 scripts/eval_systemone_backend_v1.py --url $B \
  --data data/v5_seen_source/eval.jsonl                --output $RD/lora_v5_seen.jsonl
python3 scripts/eval_systemone_backend_v1.py --url $B \
  --data data/v5_transfer_holdout/eval.jsonl           --output $RD/lora_v5_transfer.jsonl

# T2 mined-eval extension (informational; feeds G5 protected-row coverage)
python3 scripts/eval_systemone_backend_v1.py --url $B \
  --data data/real_context_eval_v5_ext/eval.jsonl      --output $RD/lora_v5_mined_ext.jsonl  # when built
```

G2 needs no rescoring of winnow: the checker defaults to the incumbent files
`results/winnow_real_candidates_full_v1.jsonl` + `results/winnow_drop_supp_v1.jsonl`.
Override with `--slot winnow_main=... winnow_supp=...` only if winnow is
re-scored (e.g. refill).

Non-default eval paths (e.g. v5 synth eval emitted as
`data/context_relevance_v5_seed20261201/eval.jsonl`) → pass
`--synth-v5-eval`, `--transfer-eval`, etc.

## 4. Run the gates

```bash
python3 scripts/check_v5_gates_v1.py --results-dir $RD \
  --out $RD/gate_report.json
# exit 0 = all gates pass; 1 = a gate failed/tripwire; 2 = some gate not_ready
```

| gate | reads | criterion (§9) |
|---|---|---|
| G1 | T1 + incumbent `results/lora_*` | acc > 0.9453 **and** drop-recall > 62.5% **and** FP < 11 **and** zero FP at noul ≥ 0.9 |
| G2 | T1 v5 × winnow | AND-drop (τ=.5, double-covered intersect): FP=0, precision 1.000, recall ≥ 17.9% |
| G3 | seen + transfer | seen-acc − transfer-acc ≤ 10pp |
| G4 | synth v4 + v5 | v4 acc ≥ 0.99; v5 acc ≥ 0.99 **and** ≥1 borderline prediction — saturated ~1.0 with zero borderline = `TRIPWIRE` (extend the set, don't claim victory) |
| G5 | all scored sets | FP ≤ FN overall **and** zero FP on protected-family keep rows (`meta.family` ∈ F1/F4 set, or `meta.protected`) |
| G6 | T1 | 0.55–0.92 FP tail band ≤ incumbent's, ECE(15-bin) ≤ budget (default = incumbent ECE × 1.05 + 0.005; override `--ece-max`) + recall-at-p0.1 reported |

Alignment detail: results row `i` indexes the file it was scored on. For T1
main the incumbent lora file was scored on `candidates.jsonl` (superset of
`eval.jsonl`); the checker joins `i → meta.record_id → eval label`, so
unlabeled/uncertain candidates are skipped automatically. Rows carrying a
`record_id` field join directly.

Useful flags: `--only G1,G2` (subset), `--tau` (default 0.5),
`--tail-band 0.55:0.92`, `--borderline 0.05:0.95`, `--slot NAME=PATH`,
`--incumbent-dir` (default `results/`).

## 5. Reading the report

- `pass` / `fail` / `tripwire` / `not_ready` per gate; `overall` aggregates
  (any fail/tripwire/error → fail; else any not_ready → not_ready).
- `measured` carries n/acc/TP/FP/FN/recall/precision, ECE, tail counts, and —
  for G1 — a paired-bootstrap 95% CI of the acc delta vs the measured
  incumbent (seed 20261201, 2000 resamples).
- A gate whose inputs don't exist yet reports `not_ready` with a `missing`
  list — that is the expected state until the v5 corpus / transfer holdout
  are built. `not_ready` is **not** a pass: exit code 2 keeps CI honest.

## 6. Failure playbook

| signal | meaning | next step |
|---|---|---|
| G1 fail, `zero_confident_fp` | the 0.919-class bug persists | more F1 mined/synth; inspect the offending keep rows |
| G1 fail, recall | still under-drops | F2 hard negatives too weak or too few |
| G2 fail | consensus safety property regressed | do not proceed; check winnow alignment + coverage |
| G3 fail | >10pp source gap | T4 family needs mined representation in train, or report honestly |
| G4 `tripwire` | v5 memorized its own synthetic eval | extend `context_relevance_v5` eval pool; not a victory |
| G5 fail, `protected_fp` | FP on F1/F4-type keep row | protected-family regression (V4-S0); block |
| G6 fail | FP tail band grew or ECE over budget | calibration regression; inspect keep-side noul tail |
| any `not_ready` | missing artifact | build the tier (T2 mining+labels, v5 synth eval, T4 definition) |

## 7. Guardrails (unchanged)

- All eval files are `training_allowed=false`; the 590 frozen labels and the
  T2/T4 pools are forbidden inputs for train (§7.3, manifest `excluded`).
- Real transcript content stays local; scorer receipts stay in `results/`
  (gitignored). Publication = aggregates only (AGENTS.md).
- Gates select nothing in production: thresholds are the fixed diagnostic
  grid; no production τ is chosen here. T175's no-switch stands.
- Verify before trusting: `python3 scripts/test_v5_gates_v1.py` → exit 0.
