# NanoJev V3 N3 Readout Protocol V1

**Status: contract (N3 deliverable), 2026-09-20.**
**Scope:** read-only, fail-closed research protocol for comparing paired readout arms on
frozen split roles: fields, controls, receipt requirements and explicit boundaries.
**Implementation:** `scripts/validate_nanojev_v3_n3_readout_protocol_v1.py`; tests:
`scripts/test_validate_nanojev_v3_n3_readout_protocol_v1.py`.
**Protocol:** `research/nanojev_v3_n3_readout_protocol_v1.json`.
**Synthetic control smoke:** `scripts/run_nanojev_v3_n3_synthetic.py`;
`results/nanojev_v3_n3_synthetic_smoke_20260920.json`。该 runner 只使用内存 fixture，不加载
模型/数据集、不联网，输出 `synthetic_controls_passed_not_model_evidence`；它不是架构质量或部署证据。

This contract corresponds to the N3 work package of the
[V3 roadmap](NANOJEV_V3_ROADMAP.md): compare a task head, a schema-conditioned readout and a
lightweight encoder/router, each as an independent architecture hypothesis, and report a
per-pack Pareto front instead of a global winner. It does **not** train, quantize, serve,
promote or prune anything.

## 0. Non-negotiable statements

1. **A clean protocol is not an authorization.** The validator always reports
   `training_authorized=false` and `deployment_authorized=false`. A `status` of
   `protocol_valid_not_authorized` only says the protocol is structurally complete and
   internally consistent.
2. **Read-only.** The validator reads exactly one JSON protocol file. It never loads a model,
   checkpoint, tokenizer or dataset, never imports a third-party package and never accesses the
   network. The only optional write is an exclusive report (`--output`); an existing report is
   never overwritten.
3. **Fail-closed.** Any missing or malformed required field, duplicate or missing control,
   or truthy authorization-like flag blocks the protocol with exit code `2` and a
   machine-readable report.
4. **No training, no deployment, no production pruning.** These are declared boundaries in the
   protocol and enforced by the validator, not merely prose.

## 1. CLI contract

```
python3 scripts/validate_nanojev_v3_n3_readout_protocol_v1.py \
    --protocol research/nanojev_v3_n3_readout_protocol_v1.json \
    [--output report.json]
```

- `--protocol` defaults to `research/nanojev_v3_n3_readout_protocol_v1.json`.
- `--output` is optional and exclusive: if the path already exists the run is blocked with exit
  code `2`.
- The report is always printed to stdout as JSON; exit code `0` means valid, `2` means blocked.

## 2. Paired arms

A paired protocol declares exactly three arms, each compared on the same frozen splits, controls
and receipt fields:

| Arm | Role | Readout |
|---|---|---|
| `trained_head` | incumbent baseline | task-head distribution over the declared option set |
| `schema_readout` | candidate | slot-restricted softmax over the declared options |
| `encoder_router` | candidate | route distribution plus abstain mass |

Arm ids must be unique and exactly cover the three required values. Each arm declares
`paired=true`, and the top-level `pairing` block requires `paired`, `same_frozen_splits`,
`same_controls` and `same_receipt_fields` to be true. A shared prefill is allowed only when each
request stays isolated.

## 3. Frozen split roles

Five roles are mandatory: `train`, `dev`, `calibration`, `test`, `ood`. Every role must declare
`frozen=true`. Each role declares its `usable_for` purposes, and the `test` and `ood` roles must
not be usable for any fitting, selection, calibration, tuning or promotion step. This is the
isolation invariant: a split assignment is only frozen if no downstream selection touches it.

## 4. Controls

Required control ids (each unique, each `required=true`, each declaring the arms it `applies_to`):

`option_permutation`, `label_permutation`, `boolean_coverage`, `score_coverage`, `ood_cases`,
`protected_cases`, `constant_baselines`.

- **Option permutation** (`option_permutation`): evaluate every option order with semantic option
  ids fixed, identity order first. It applies to Boolean, Score and Choice question types. A
  semantic argmax flip rejects the arm as a permutation-stable readout; a slot-only flip is a
  high-confidence position artifact and also rejects the arm.
- **Label permutation** (`label_permutation`): permute gold labels while preserving the option set
  and the label multiset. No calibrated coverage or accuracy gain may survive this negative
  control.
- **Boolean and Score coverage** (`coverage`): Boolean questions must expose two options, Score
  questions must expose ordered options of size two to four.
- **OOD and protected cases**: OOD cases come from the `ood` split and must abstain; protected
  cases must belong to `test` or `ood` and are reported per arm. A confident protected error
  rejects the arm for that pack.
- **Constant baselines** (`constant_baselines`): `constant_true` and `constant_false` are
  mandatory; the protocol also declares `constant_majority` and `abstain_all`.

## 5. Paired receipt fields

Every arm, every permutation and every baseline emits the same fields so comparisons stay paired.
Required fields include `arm_id`, `split`, `case_id`, `question_type`, `option_order`,
`label_assignment`, `predicted_label`, `probabilities`, `confidence`, `coverage`, `correct`,
`constant_true_correct`, `constant_false_correct`, `control_id`, `receipt_sha256`,
`network_model_calls`, `training_authorized` and `deployment_authorized`. The protocol also declares
`arm_role`, `option_permutation_id`, `label_permutation_id`, `protected_case`, `latency_ms` and
`memory_bytes`. Receipt fields must be unique and complete.

## 6. Boundaries

The `boundaries` block must set every authorization-like flag to `false`:
`training_authorized`, `deployment_authorized`, `production_pruning_authorized`,
`active_pruning_authorized`, `promotion_authorized`, `authorizes_execution`, `authorizes_training`,
`authorizes_deployment`, `model_weights_modified`, `checkpoint_written` and `network_allowed`.
`shadow_only` remains `true`. The validator additionally scans the whole protocol tree and rejects
any authorization-like key (`*_authorized`, `*_authorization`, `authorizes_*`) carrying a truthy
value.

## 7. Fail-closed checklist

The protocol is blocked (exit `2`) when any of the following holds:

1. the protocol root is not a JSON object, or the file is missing / unreadable / not valid JSON;
2. `schema_version` is missing or is not exactly `nanojev-v3-n3-readout-protocol-v1`;
3. a required string field is absent or empty (`protocol_id`, `purpose`, `status`, `mode`,
   `decision_rule`), or `limitations` is not a non-empty list of strings;
4. `frozen_before_inference` or `read_only` is not `true`, or `network_model_calls` is not `0`;
5. a paired arm is missing, duplicated, unknown, unpaired, or lacks a required field; the
   `pairing` block is missing or declares a false pairing flag;
6. a required split role is missing or not frozen; `test`/`ood` is usable for fit/select/calibrate/
   tune/train/promote; an unknown split role is declared;
7. a control is missing, duplicated, not `required`, or references an unknown arm;
8. `option_permutation` is disabled or not identity-first; it does not cover Boolean, Score and
   Choice; `label_permutation` is disabled or does not preserve the option set and label multiset;
9. `coverage.boolean` or `coverage.score` is missing or not required, or option counts are invalid;
10. `ood` is missing, not required, not sourced from the `ood` split, does not require abstain, or
    permits threshold tuning on OOD;
11. `protected_cases` is empty, has a duplicate `case_id`, lacks a `reason`, or declares a split
    other than `test`/`ood`;
12. a required constant baseline is missing or the list has duplicates;
13. a required receipt field is missing or the field list has duplicates;
14. a `boundaries` authorization flag is missing or not exactly `false`;
15. any authorization-like key anywhere in the protocol has a truthy value.

## 8. Report shape

The report records `report_version`, `schema_version`, `source`, `valid`, `status`,
`violations` (each with `code`, `path`, `message`), `block_reasons`, `counts`,
`arms_present`, `split_roles_present`, `controls_present`, and always:

```
training_authorized=false
deployment_authorized=false
production_pruning_authorized=false
training_performed=false
deployment_performed=false
model_loaded=false
network_model_calls=0
```

## 9. Tests

`scripts/test_validate_nanojev_v3_n3_readout_protocol_v1.py` builds synthetic protocols in
temporary directories only. It covers the clean pass, schema and field failures, arm and pairing
failures, frozen-split and OOD isolation failures, duplicate/missing controls, disabled
permutation controls, coverage and protected-case failures, receipt-field failures, truthy
authorization-like flags, blocked reports still declaring no authorization, file parsing
failures, CLI exit codes and exclusive output. No model, checkpoint, dataset or network is used.

```
python3 -m unittest discover -s scripts -p test_validate_nanojev_v3_n3_readout_protocol_v1.py
```

Synthetic-only control smoke（独立于真实 arm）:

```text
.venv/bin/python -m unittest discover -s scripts -p 'test_run_nanojev_v3_n3_synthetic.py' -> 12 tests OK
.venv/bin/python scripts/run_nanojev_v3_n3_synthetic.py \
  --protocol research/nanojev_v3_n3_readout_protocol_v1.json \
  --output results/nanojev_v3_n3_synthetic_smoke_20260920.json -> exit 0
```

该 smoke 收据为 3 arms/8 cases/123 receipts，零网络调用、未加载模型、无训练/部署/生产裁剪
授权；通过只说明控制和收据路径有效，不代表准确率、校准、速度、体积、成本或部署 ready。

## 10. Non-goals

- Not training, distillation, quantization, serving or production pruning.
- Not a global architecture winner; only a per-pack Pareto front when all gates pass.
- Not a deployment or promotion authorization, and not a substitute for owner or independent
  review of the N2 corpus, protected cases and OOD composition.
