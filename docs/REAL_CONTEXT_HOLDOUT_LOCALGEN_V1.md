# Real-context holdout paired local-generation evaluator V1

Status: **evaluator ready; no real cases collected**.

This runner consumes the same content-free manifest and local raw request files
as the deterministic shadow evaluator. For each `scored` case it also reads a
local-only downstream contract, builds the deterministic-shadow reduced request,
and compares original vs reduced outcomes through a deterministic or pinned
local-generation responder.

## Commands

Deterministic contract backend:

```bash
python3 scripts/run_real_context_holdout_localgen_v1.py \
  --manifest data/real_context_holdout_v1/manifest.json \
  --case-dir data/real_context_holdout_v1/cases \
  --contract-dir data/real_context_holdout_v1/cases/contracts \
  --backend deterministic \
  --output results/real_context_holdout_localgen_v1.json
```

Pinned local-generation backend:

```bash
python3 scripts/run_real_context_holdout_localgen_v1.py \
  --manifest data/real_context_holdout_v1/manifest.json \
  --case-dir data/real_context_holdout_v1/cases \
  --contract-dir data/real_context_holdout_v1/cases/contracts \
  --backend local-generation \
  --model-path external/models/Qwen3.5-4B \
  --device mps \
  --output results/real_context_holdout_localgen_qwen35_4b_v1.json
```

## Local contract file

Each scored case needs a local-only contract file:

```text
<case_id>.contract.json
```

Shape:

```json
{
  "expected_answer": "...",
  "required_strings": ["..."]
}
```

The manifest stores only the corresponding SHA-256 values. The runner verifies:

- `expected_answer_sha256`;
- `required_evidence_sha256`, when declared;
- request-file hash;
- manifest validity.

## Checks

For each scored case:

1. deterministic shadow scorer proposes eligible drops;
2. reduced bytes are built in-process only;
3. original and reduced requests are evaluated by the selected backend;
4. original-pass / reduced-fail is an `answer_regression`;
5. reduced missing required evidence is unsafe;
6. non-scored cases are skipped with reason.

## Output boundary

The receipt records hashes, statuses, pointer lists, counts, and regressions. It
never writes raw prompts, generated text, expected answers, required strings, or
tool outputs.

## Current tests

`scripts/test_real_context_holdout_localgen_v1.py` covers:

- deterministic pair pass;
- required-evidence drop regression;
- expected-answer hash mismatch;
- non-scored skip.

## Boundary

This is paired evaluation plumbing. It does not call providers, enable active
filtering, select a production threshold, or prove real downstream quality until
real authorized holdout cases and a representative pinned backend are used.
