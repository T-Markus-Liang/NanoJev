# Offline JevBench benchmark bundle

This directory is an offline evaluation bundle. It keeps the official JevBench
public track separate from SemIf supplemental fixtures; SemIf rows are not
renamed or merged into the official benchmark.

## Tracks

- `official_jevbench_v1.2.4_public/`: the pinned JevBench public release at
  revision `83831807458d7df424a1e53e5724f3a3ffe2cf89` (231 public rows across
  `original`, `easy`, and the public half of `hard`). `public_231.jsonl` is a
  deterministic concatenation of those three official files. Use this track
  with the existing NanoJev adapter's `public_diagnostic` mode.
- `semif_owned/`: SemIf's authored 144-row workload, 108 perturbations, and
  37x21 `shape777` systems fixture.
- `semif_manifests/`: SemIf's evaluation matrix, metric contract, and source
  selection records.
- `semif_external_sources/`: hash-verified WANLI test source and Every source
  snapshots used to rebuild SemIf's external rows.
- `semif_rebuilt_external/`: the fixed SemIf selections rebuilt offline from
  those snapshots: WANLI 256 rows and Every inference 204 rows / labeled 154
  rows plus firewall actions.

The selected 102-row TypeSafe subset is intentionally marked unavailable in
`offline_bundle_manifest.json`: SemIf requires source snapshots that are not
shipped with its repository. Provider outputs are not reconstructed or used as
labels here.

All tracks are evaluation-only. Do not use any row, expected label, provider
output, or benchmark-derived probability for training or calibration fitting.
This bundle is a disclosed public evaluation fixture; raw provider responses,
per-item probabilities, and local receipts remain outside the public tree.

Verify the bundle against the manifest (sha256/bytes/rows; non-zero on mismatch):

```bash
python3 scripts/build_offline_jevbench_bundle_v1.py --verify
```

Run official Jev over every row via the direct TypeSafe API at
`api.typesafe.ai` (the only channel; the Vercel AI Gateway path was removed).
Receipts go to the gitignored `research/official_jev_direct_run_v1.jsonl` and a
scored summary JSON:

```bash
node scripts/eval_official_jev_bundle_v1.mjs --live
```

Regenerate the hash/count manifest from the downloaded files with:

```bash
python3 scripts/build_offline_jevbench_bundle_v1.py
```

The current working-tree adapter self-hash matches
`research/jevbench_adapter_contract_v1.json`. A dry-run over the bundled
`public_231.jsonl` validated all 231 official public records. This is schema
validation only; it is not a model score, official milestone, or promotion
authorization.
