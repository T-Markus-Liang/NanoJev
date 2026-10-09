# Test Triage — 2026-09-29

Sweep of 26 failing `scripts/test_*.py` files. The original sweep ran them with
system `python3` (3.14.6, no torch/numpy), so every file importing torch or numpy
failed with `ModuleNotFoundError`. Each file was retried with the correct
interpreter `external/valen/.venv/bin/python` (Python 3.11.15, torch 2.6.0,
numpy 2.4.6), 240s timeout each, cwd `scripts/`.

## Verdict counts

- **env** (pass under correct interpreter): **11**
- **env** (still fails under venv — hardware/kernel issue): **1** (`test_jevbench_adapter_v1`)
- **regression** (fails under venv; caused by this week's uncommitted changes): **13**
- **stale**: **1** (`test_validate_nanojev_v2_t9d_fp32_mps_protocol_v2`)

## REAL REGRESSIONS — read first

A single root cause accounts for 12 of the 13 regressions: the **uncommitted
rewrite of `scripts/context_gate_v1.py`** (worktree-modified, mtime 2026-09-24;
+225-line diff). The scoring contract changed from per-segment states
(`id: segment_N`, question `irrelevant`) to one joint `batch` state with
`irrelevant_N` questions (`scoring_payload`, `validated_scores`). Every consumer
whose scorer still emits the old contract now raises
`Bypass("invalid_score_response")`, so the gate returns status `bypass` /
`invalid_score_response`, applies no reductions, and downstream receipts lose
`applied_pointers` / `restore_round_trip`. Companion worktree edits to
`main_model_gateway_v1.py` (+81), `scorer_adapters_v1.py` (+190),
`serve_decisions.py` (+158, backend-table work), and
`build_context_relevance_v1.py` (gold key renamed `irrelevant` → `irrelevant_0`)
complete the blast radius.

Failing files attributable to the gate-contract rewrite:
`test_context_gate_batch_v1`, `test_tool_history_fixtures_v1`,
`test_filter_vs_rebuild_v1`, `test_filter_value_v1`, `test_gate_contrastive_v1`,
`test_main_model_gateway_batch_v1`, `test_safe_dedup_v1`, `test_serve_nanojev_v1`,
`test_real_context_holdout_shadow_v1`, `test_real_context_holdout_localgen_v1`,
`test_real_context_holdout_preflight_v1`, `test_active_mode_phase0_v1`.

Plus one contract rename: `test_report_context_relevance_v1`
(`report_context_relevance_v1.join_predictions` still reads `gold["irrelevant"]`;
`build_context_relevance_v1.make_record` now emits `gold["irrelevant_0"]`).

Un-migrated stub scorers found in: `build_gate_contrastive_v1.oracle_scorer`
(reads `state["state"]["candidate_pointer"]`, answers `{"irrelevant": ...}`),
`build_tool_history_fixtures_v1` deterministic scorer, and the test-file stubs
in `test_context_gate_batch_v1` / `test_main_model_gateway_batch_v1` /
`test_serve_nanojev_v1` / holdout runners.

## Table

| file | venv result | last-touch | verdict | root cause |
|---|---|---|---|---|
| test_active_mode_phase0_v1.py | FAIL (KeyError 'restore_round_trip', run_active_mode_phase0_v1:303) | untracked, mtime 09-23 | regression | gate bypasses under new batch contract → active mode applies no reduction → receipt lacks `restore_round_trip` |
| test_calibrated_objectives.py | PASS | commit 2026-09-17 | env | needs torch (imports torch at line 17); system py3 lacks it |
| test_context_gate_batch_v1.py | FAIL (24 subtest failures; `scorer_event_ids` []) | untracked, mtime 09-20 | regression | `context_gate_v1.validated_scores` now requires joint `batch` state + `irrelevant_N` answers; test stubs emit old `segment_N`/`irrelevant` contract |
| test_filter_value_v1.py | FAIL (2: `unsafe_removals` 0 != 1) | untracked, mtime 09-23 | regression | gate bypass (`invalid_score_response`) → deterministic-scorer unsafe removals never counted |
| test_filter_vs_rebuild_v1.py | FAIL (1: manifest expects `scored`/`shadow_only`, got `bypass`/`invalid_score_response`) | untracked, mtime 09-20 | regression | same gate-contract break inside `build_tool_history_fixtures_v1` manifest validation |
| test_financial_rlcd_v1.py | PASS | untracked, mtime 09-20 | env | needs numpy |
| test_gate_contrastive_v1.py | FAIL (setUpClass: `core did not score (scorer_error)`) | commit 2026-09-19 | regression | `build_gate_contrastive_v1.oracle_scorer` (line 489-509) still emits old per-segment contract → scorer_error via new `validated_scores` |
| test_jevbench_adapter_v1.py | CRASH (SIGABRT, MPSGraph matmul "incompatible dimensions", LLVM ERROR) | untracked, mtime 09-21 | env | test hard-requires `device="mps"` (line 292) and asserts `execution.device=="mps"`; torch 2.6.0 MPS kernel shape-inference bug — not a code regression |
| test_jfast_modernbert.py | PASS | untracked, mtime 09-21 | env | needs torch |
| test_lora_merge_v4.py | PASS | untracked, mtime 09-20 | env | needs torch |
| test_main_model_gateway_batch_v1.py | FAIL (4: `applied_pointers` [] vs expected drops, all 3 wire formats) | untracked, mtime 09-20 | regression | gateway/gate bypass under new contract → removal plan empty; `main_model_gateway_v1.py` also worktree-modified 09-24 |
| test_predict_runtime.py | PASS | commit 2026-09-19 (+worktree edit) | env | needs torch |
| test_real_context_holdout_localgen_v1.py | FAIL (2: `gate did not score case: invalid_score_response`, KeyError `answer_regression`) | untracked, mtime 09-23 | regression | gate bypass under new contract in `run_real_context_holdout_localgen_v1` |
| test_real_context_holdout_preflight_v1.py | FAIL (2: valid chain fails `local_generation`; bad case not caught at `shadow`) | untracked, mtime 09-23 | regression | cascades from shadow/localgen gate-contract failures |
| test_real_context_holdout_shadow_v1.py | FAIL (2: valid case flagged `unsafe`; required-evidence drop not flagged) | untracked, mtime 09-23 | regression | gate bypass inverts holdout verdicts under new contract |
| test_report_context_relevance_v1.py | FAIL (4 errors: KeyError `irrelevant` at report_context_relevance_v1.py:51) | commit 2026-09-19 | regression | `build_context_relevance_v1.make_record` renamed gold key to `irrelevant_0` (uncommitted, 09-24); `report_context_relevance_v1` not updated |
| test_run_financial_baselines_v1.py | PASS | untracked, mtime 09-20 | env | needs numpy |
| test_safe_dedup_v1.py | FAIL (2: `forward_reason` `active_no_reduction` vs `active_reduced`/`reduction_error`) | commit 2026-09-19 | regression | gateway active path never reduces under new gate contract |
| test_serve_nanojev_v1.py | FAIL (1: `status` `bypass` != `scored`) | untracked, mtime 09-24 | regression | `serve_decisions` context-gate eval path hits `invalid_score_response` bypass (serve_decisions worktree-modified 09-28, backend-table work) |
| test_tool_history_fixtures_v1.py | FAIL (setUpClass: manifest validation — all cases `bypass`/`invalid_score_response`) | commit 2026-09-19 | regression | `build_tool_history_fixtures_v1` deterministic scorer emits old contract → every case bypasses |
| test_training_runtime.py | PASS | commit 2026-09-19 | env | needs torch |
| test_validate_nanojev_v2_t9d_fp32_mps_protocol_v2.py | FAIL (2: `repin nanojev_predictor_repin_v1.json does not match current scripts/predict_toy_decisions.py hash`) | untracked, mtime 09-20 | stale | predictor repin fixture not refreshed after `predict_toy_decisions.py` edits (mtime 09-21); drift detection working as designed — needs repin |
| test_winnow_isotonic_v1.py | PASS | untracked, mtime 09-28 | env | needs numpy (winnow isotonic work itself is fine) |
| test_x2_pointer_head.py | PASS | untracked, mtime 09-20 | env | needs torch |
| test_x2_semif_readout.py | PASS | untracked, mtime 09-20 | env | needs torch (via `predict_toy_decisions`) |
| test_x3_shared_prefix.py | PASS | untracked, mtime 09-20 | env | needs torch (via `predict_toy_decisions`) |

## Notes for follow-up

- Re-run command per file: `cd scripts && ../external/valen/.venv/bin/python <test>.py`
- The 12 gate-contract regressions share one fix path: either migrate all stub
  scorers/consumers to the joint `batch` + `irrelevant_N` contract, or restore
  per-segment compatibility in `context_gate_v1.validated_scores`. Given the
  rewrite is uncommitted in-flight work (part of this week's context-relevance /
  serve-decisions batch), this is expected breakage, not an old bug — but it
  currently leaves the suite red.
- `test_jevbench_adapter_v1` needs a CPU-capable path or a torch/MPS pin before
  it can run on this Mac; the crash is inside MetalPerformanceShadersGraph, not
  project code.
- `test_validate_nanojev_v2_t9d_fp32_mps_protocol_v2` will go green once
  `nanojev_predictor_repin_v1.json` is re-pinned to the current
  `predict_toy_decisions.py` hash (or the predictor edit is reverted/committed
  with a repin).
