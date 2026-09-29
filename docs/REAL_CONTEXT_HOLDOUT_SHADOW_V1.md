# Real-context holdout deterministic shadow evaluator V1

Status: **evaluator ready; no real cases collected**.

This runner consumes the content-free manifest produced by
`real_context_holdout_manifest_v1.py`, reads the local raw request files it
references, and evaluates the existing shadow gate deterministically. It never
calls a provider and never applies active filtering.

## Command

```bash
python3 scripts/run_real_context_holdout_shadow_v1.py \
  --manifest data/real_context_holdout_v1/manifest.json \
  --case-dir data/real_context_holdout_v1/cases \
  --threshold 0.90 \
  --output results/real_context_holdout_shadow_v1.json
```

## What it checks per case

- manifest is valid and content-free;
- `request_file` stays inside the case directory;
- raw request hash matches `request_sha256`;
- sidecar is derived only from declared pointers:
  - bypass cases get `{"bypass": true}`;
  - protected/dependency pointers are pinned;
  - eligible pointers are scored;
- deterministic scorer marks eligible candidates irrelevant at `0.999`;
- expected gate status matches (`scored`, `bypass`, or `protected_only`);
- suggested drops are a subset of eligible pointers;
- protected/dependency/required-evidence pointers are never dropped;
- optional `expected_suggestions` match the proposed pointers;
- hypothetical reduced bytes are reversible through the restore manifest.

## Output boundary

The receipt contains only:

- case IDs;
- request/reduced hashes;
- pointer lists;
- counts;
- gate status/reason;
- unsafe flags;
- restore status.

It does not contain raw prompt, response, credential, or tool-output text.

## Current test coverage

`scripts/test_real_context_holdout_shadow_v1.py` covers:

- eligible scored reduction + restore;
- bypass status;
- protected-only status;
- unsafe required-evidence drop detection;
- invalid manifest failure.

## Boundary

This is deterministic plumbing/safety evidence. It does not select a production
threshold, call a provider, or prove downstream model quality.
