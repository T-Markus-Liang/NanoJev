# NanoJev V3 N2 Domain-Pack Contract V1

**Status: contract (N2 deliverable), 2026-09-20.**
**Scope:** read-only preflight contract for a versioned domain pack: manifest fields,
split inputs, isolation invariants and fail-closed behavior.
**Implementation:** `scripts/validate_nanojev_v3_domain_pack_v1.py`; tests:
`scripts/test_validate_nanojev_v3_domain_pack_v1.py`.

This contract defines the gate that runs **before** any corpus is merged or any optimizer is
started. It is derived from the T8g audit (`docs/T8G_ADAPTER_REVIEW_V1.md`) and the N1 benchmark
contract (`docs/NANOJEV_V3_BENCHMARK_CONTRACT_V1.md`): ID-only disjointness does not establish an
independent holdout, so split assignment must be verified on canonical model-visible content.

## 0. Non-negotiable statements

1. **A clean preflight is not training authorization.** A clean run writes
   `training_authorized=false`, `training_performed=false`, `merged_rows_written=0`. It only says
   the staged pack satisfies the schema and isolation checks in this contract.
2. **Read-only.** The validator reads the manifest and split files, records their sha256 before
   and after, and fails closed if any input byte changed. It never rewrites, relabels,
   repartitions, deduplicates or merges rows, and never loads a model or checkpoint.
3. **The only write is an optional exclusive report.** An existing report path is never
   overwritten, and no report may be written under the pack root.

## 1. Pack layout

```
<pack>/
  manifest.json
  splits/train.jsonl
  splits/dev.jsonl
  splits/calibration.jsonl
  splits/test.jsonl
  splits/ood.jsonl
```

All five split names are mandatory. Split paths are declared in the manifest, must stay inside
the pack root, and must exist and be non-empty.

## 2. Manifest schema

`manifest.json` is a JSON object with:

| Field | Rule |
|---|---|
| `schema_version` | exactly `nanojev-v3-domain-pack-v1` |
| `pack_id` | non-empty string |
| `pack_version` | non-empty string |
| `heldout` | object with non-empty `identity` and `description` (explicit heldout identity) |
| `protected_cases` | non-empty list of `{"id", "split", "reason"}` |
| `splits` | object mapping each required split name to a relative JSONL path |

`protected_cases[*].split` must be `test` or `ood`. Every declared protected case must resolve
to a record whose actual split equals the declared split; a protected case that is absent from
test/OOD is a violation.

## 3. Record schema

Each JSONL line is a JSON object. Required fields:

| Field | Rule |
|---|---|
| `id` | non-empty string, unique across the whole pack |
| `split` | optional, but if present must equal the containing split file |
| `source_group_id` | non-empty string |
| `lineage_id` | non-empty string |
| `provenance` | object with `source_alias` and `derived_from_evaluation_corpus` |
| `state` | model-visible evidence text |
| `question` | object with `type`, `instructions`, `criteria` |
| `target` | gold distribution (excluded from isolation hashing) |

## 4. Canonical model-visible input

Isolation is measured on the digest of exactly the model-visible input:

```
{"state", "type", "instructions", "criteria"}
```

- Record id, question id, split, source/lineage id, provenance and target/gold are excluded
  because they never enter the model input.
- Dictionary key order is normalized (`criteria` choice maps are order-insensitive).
- Ordinal **Score** criteria lists preserve their order.
- The same canonical input appearing in more than one split is a violation. This is an exact
  lower bound on leakage, not a semantic paraphrase guarantee.

## 5. Fail-closed checklist

Any of the following blocks the pack (exit 2):

1. pack root, `manifest.json`, or a declared split file missing or unreadable;
2. malformed JSON, non-object manifest/record, empty split, missing record `question` fields or
   `target`, malformed provenance/protected-case fields, or `schema_version` mismatch;
3. missing/invalid manifest fields, including explicit `heldout` identity;
4. missing, empty, or wrongly-split protected-case declarations; protected case absent from
   test/OOD;
5. missing or duplicate record `id`;
6. missing `source_group_id` or `lineage_id`;
7. `source_group_id` or `lineage_id` crossing splits;
8. canonical model-visible input crossing splits;
9. evaluation-derived provenance: explicit `derived_from_evaluation_corpus=true`, or a
   `source_alias` normalizing (hyphen/underscore/space) to a known evaluation-derived source;
10. input hashes changing before/after validation;
11. manifest/split paths resolving outside the pack root;
12. report output already existing, or output path inside the pack root.

## 6. Exit codes and report

- `0`: clean preflight; report `status=preflight_passed_not_training_authorized`,
  `block_reasons=[]`, `training_authorized=false`.
- `2`: blocked report (violations) or hard read/write error. A blocked report is still emitted
  when `--output` is given; the output file is created exclusively.

```
python3 scripts/validate_nanojev_v3_domain_pack_v1.py --pack DIR [--output report.json]
```

The report records a portable `pack` identity (`pack_id@pack_version`, or `unresolved` when the
manifest cannot be read), `source_paths` (the manifest and the five manifest-declared relative
split paths), `source_hashes` (before) and `source_hashes_after`, `input_files_changed`,
`counts_by_split`, `record_count`, `violations`, `block_reasons`, `heldout`, `protected_cases`,
`training_authorized=false`, `training_performed=false` and `merged_rows_written=0`. The report
does not embed the absolute checkout path, so identical pack bytes produce identical report
identity fields across temporary directories; the saved report's own path is still external metadata.

## 7. Tests

`scripts/test_validate_nanojev_v3_domain_pack_v1.py` fabricates packs in temporary directories
only, covering clean preflight, blocked schema, content overlap (including choice-dict
normalization and ordinal score lists), source/lineage crossing, provenance aliases, protected
coverage, missing/duplicate IDs, heldout identity, hash immutability, and exclusive output.
No real corpus is audited and no network/model/agent is used.

```
python3 -m unittest discover -s scripts -p test_validate_nanojev_v3_domain_pack_v1.py
```

## 8. Non-goals

- Not training, quantization, serving or production pruning.
- Not a semantic deduplication guarantee; canonical equality is a lower bound.
- Not approval of labels, corpus scale, split proportions or the T9d training protocol.
- Not a substitute for human review of the T8g provenance and heldout-composition decisions.
