# Local main-model paired evaluator V1

Status: **deterministic and pinned local-generation backends implemented; no external
provider calls; no production-quality claim**.

## Purpose

`scripts/local_main_model_evaluator_v1.py` compares an original request and a reduced
request as a pair. The goal is to make downstream evidence preservation explicit:

- if required evidence survives, the evaluator returns the declared expected answer;
- if any required evidence is missing, it returns `insufficient_evidence`;
- a request that answers before filtering but fails after filtering is a paired
  regression and therefore unsafe.

This is the first controlled downstream layer for Track A. It is stronger than a raw
byte/token count because it directly tests whether the request still contains the
declared evidence contract. It is still deterministic and does **not** prove real LLM
behavior.

## Backend

### `deterministic-evidence-v1`

Inputs:

- original request bytes;
- reduced request bytes;
- downstream contract with:
  - `expected_answer`;
  - `required_strings`;
- optional tokenizer callback for local usage estimates.

The responder checks whether every `required_strings` entry appears in the request
bytes. It does not inspect model semantics and does not call a model. The returned
receipt contains:

- backend ID;
- answer status and boolean;
- prompt-token estimate;
- SHA-256 hashes of missing required strings;
- SHA-256 hash of the private response object;
- pair status and answer-regression flag;
- prompt-token difference.

Raw request text, expected answers, required strings, and response bodies are not
written to the receipt.

### `local-generation:*`

The optional local-generation backend loads a pinned local model snapshot and
generates a short deterministic answer (`do_sample=false`) for each
original/reduced request. It supports direct local directories or pinned Hugging
Face cache snapshots; no network request is required. If the tokenizer provides a
chat template, it uses that template; otherwise it uses the plain evaluator prompt.
Generated answers are normalized by removing `<think>` blocks and comparing
normalized alphanumeric tokens, while common conjunctions/punctuation do not create
false mismatches.

Measured pinned backends:

| Backend | Control answers | Dedup answers | Winnow @0.90 reduced answers | Paired regressions |
|---|---:|---:|---:|---:|
| `Qwen/Qwen3-0.6B@c1899de2` | 8/25 | 8/25 | 8/25 | 0 |
| `Qwen/Qwen2.5-3B-Instruct@aa8e7253` | 17/25 | 17/25 | 17/25 | 0 |
| `Qwen/Qwen3.5-4B@851bf6e8` | 18/25 | 18/25 | 18/25 | 0 |

The backend stores only status, match boolean, missing-required-string hashes,
prompt/completion estimates, and a private response-object SHA-256. Raw prompts and
raw generated text are not written to the receipt.

Result files:

- `results/filter_value_v2_control_dedup_localgen_qwen3_0p6b_v1.json`
- `results/filter_value_v2_winnow_t090_localgen_qwen3_0p6b_v1.json`
- `results/filter_value_v2_control_dedup_localgen_qwen25_3b_v1.json`
- `results/filter_value_v2_winnow_t090_localgen_qwen25_3b_v1.json`
- `results/filter_value_v2_control_dedup_localgen_qwen35_4b_v1.json`
- `results/filter_value_v2_winnow_t090_localgen_qwen35_4b_v1.json`
- `results/filter_value_v3_control_dedup_localgen_qwen35_4b_v1.json`
- `results/filter_value_v3_winnow_t090_localgen_qwen35_4b_v1.json`

Interpretation: after chat-template and answer-normalization correction, all three
local-generation backends preserve their original answer counts under Winnow `0.90`.
This is stronger evidence than required-string preservation alone, but these models
still answer only `8–18` of 25 synthetic prompts; it is not a frontier-model
quality claim.

## Value-runner integration

`benchmark_filter_value_v1.py` now calls `evaluate_pair` for every case/arm. The
existing safety check treats a paired answer regression as unsafe even if every other
structural check passes.

CLI smoke example:

```bash
python3 scripts/local_main_model_evaluator_v1.py \
  --request request.json \
  --reduced-request reduced.json \
  --contract contract.json \
  --output pair_result.json
```

`contract.json` is a local-only JSON object, for example:

```json
{
  "expected_answer": "42",
  "required_strings": ["limit=42", "What is the limit?"]
}
```

## Interpretation boundaries

- `passed` means the reduced request still satisfies the declared evidence contract.
- It does not mean a real LLM will answer correctly.
- `prompt_tokens` is a local estimator when a tokenizer is available; it is not
  provider billing.
- The local-generation backends are pinned and deterministic, but the largest local
  check is still a small Qwen3.5-4B model; it is not a frontier downstream model.
- External providers remain excluded from this evaluator.

## Files

- Evaluator: `scripts/local_main_model_evaluator_v1.py`
- Tests: `scripts/test_local_main_model_evaluator_v1.py`
- Integrated runner: `scripts/benchmark_filter_value_v1.py`
- Fixture contract: `research/context_filter_value_fixture_manifest_v1.json`
