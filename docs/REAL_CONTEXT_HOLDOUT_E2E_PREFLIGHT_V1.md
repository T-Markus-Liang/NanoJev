# Real-context holdout end-to-end preflight V1

Status: **runner ready; no real cases collected**.

This runner chains the complete local-only holdout pipeline into one receipt:

```text
protocol validation
→ manifest validation
→ deterministic shadow evaluation
→ paired local-generation evaluation
```

No provider calls and no active filtering are performed.

## Commands

Protocol-only check:

```bash
python3 scripts/run_real_context_holdout_preflight_v1.py \
  --output results/real_context_holdout_e2e_preflight_v1.json
```

Full manifest chain:

```bash
python3 scripts/run_real_context_holdout_preflight_v1.py \
  --manifest data/real_context_holdout_v1/manifest.json \
  --case-dir data/real_context_holdout_v1/cases \
  --contract-dir data/real_context_holdout_v1/cases/contracts \
  --backend deterministic \
  --output results/real_context_holdout_e2e_preflight_v1.json
```

Pinned local-generation backend:

```bash
python3 scripts/run_real_context_holdout_preflight_v1.py \
  --manifest data/real_context_holdout_v1/manifest.json \
  --case-dir data/real_context_holdout_v1/cases \
  --contract-dir data/real_context_holdout_v1/cases/contracts \
  --backend local-generation \
  --model-path external/models/Qwen3.5-4B \
  --device mps \
  --output results/real_context_holdout_e2e_qwen35_4b_v1.json
```

## Status values

| Status | Meaning |
|---|---|
| `protocol_ready_no_manifest` | protocol is valid and no manifest was supplied |
| `e2e_preflight_pass` | protocol, manifest, shadow, and local-generation checks all passed |
| `e2e_preflight_fail` | at least one stage failed |
| `preflight_fail` | protocol validation failed before manifest evaluation |

`--require-ready` additionally fails when manifest coverage does not satisfy the
protocol minima.

## Output boundary

The unified receipt contains nested protocol, shadow, and local-generation
receipts plus status/count fields. It contains no raw prompts, generated text,
expected answers, required strings, credentials, or provider data.

## Current tests

`scripts/test_real_context_holdout_preflight_v1.py` covers:

- protocol-only pass;
- full-chain pass;
- `--require-ready` failure on partial coverage;
- unsafe case failure across shadow/local-generation stages.

## Boundary

This is a deterministic gate runner. It does not authorize collection, provider
calls, active filtering, a production threshold, or release.
