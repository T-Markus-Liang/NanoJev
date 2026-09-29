# Real-context holdout manifest V1

Status: **intake validator ready; no real cases collected**.

This is the deterministic manifest contract for the future local-only
real-context holdout. It converts owner-supplied local case metadata into a
content-free manifest containing hashes, counts, pointers, and labels only.

## Tool

```bash
python3 scripts/real_context_holdout_manifest_v1.py build \
  --case-dir data/real_context_holdout_v1/cases \
  --data-root data/real_context_holdout_v1 \
  --output data/real_context_holdout_v1/manifest.json

python3 scripts/real_context_holdout_manifest_v1.py validate \
  --manifest data/real_context_holdout_v1/manifest.json \
  --output results/real_context_holdout_manifest_check_v1.json
```

The builder refuses to overwrite a non-empty manifest.

## Case metadata contract

Each case metadata file is a JSON object in the case directory. It references a
relative `request_file`; the manifest never copies the raw request.

Required fields:

- `case_id`
- `family`
- `language`
- `wire_format`
- `source_type`
- `source_hash`
- `request_file`
- `captured_or_redacted_at`
- `expected_gate_status`
- `protected_pointers`
- `downstream`
- `labels`

Optional fields:

- `eligible_candidate_pointers`
- `dependency_pointers`
- `expected_suggestions`
- `tool_linked`
- `long_tail`
- `mixed_part`
- `tool_result_mutability`

## Enforced checks

1. `case_id` is a bounded slug and unique.
2. `wire_format`, `source_type`, and `expected_gate_status` match the intake
   protocol.
3. `source_hash` and hash-valued downstream fields are SHA-256.
4. Protected and eligible pointers cannot overlap.
5. Scored cases must declare a downstream contract.
6. Low-confidence labels cannot be used for scored cases.
7. `request_file` must stay inside the case directory, exist, and parse as UTF-8
   JSON.
8. Raw request bytes are scanned for obvious credential-like patterns.
9. Duplicate `request_sha256` fails.
10. Raw-text-like metadata keys (`request`, `messages`, `content`, `prompt`,
    `response`, `required_strings`, `expected_answer`, `text`, `raw`) are
    forbidden anywhere in the case metadata.

## Coverage report

The manifest computes:

- case count;
- distinct families;
- bypass cases;
- tool-linked cases;
- multilingual cases;
- protected-only cases;
- long-tail cases;
- mixed-part cases;
- mutable/nonrepeatable tool-result cases;
- `ready_for_evaluation` against the protocol minima.

## Boundary

This tool only builds and checks manifests. It does not collect data, call a
provider, enable active filtering, or make a production-quality claim.
