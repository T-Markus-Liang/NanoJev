# Context filter held-out value fixture V3

Status: **held-out synthetic shadow/offline replay complete; active filtering remains
disabled**.

V3 is a 12-case held-out synthetic cohort. Its cases are new self-authored
fixtures and do not reuse the 25-case V2 development set. The goal is to reduce
threshold/adapter overfitting risk before review handoff.

## Fixture contract

Manifest:

- `research/context_filter_value_fixture_manifest_v3.json`
- SHA-256: `805bd209a3c14bca6225d6284729da5f52603ce2c39df6266a391f6f35f6403e`
- cases: `12`
- split: `heldout_replay`
- base development cohort: `research/context_filter_value_fixture_manifest_v2.json`

Coverage:

- clear irrelevant archive notes;
- ambiguous status history;
- dependency-linked correction;
- protected tool call/result plus unrelated noise;
- mixed text parts;
- multilingual request;
- exact duplicate;
- required value marked eligible but expected to retain;
- protected-only zero-drop request;
- long-tail noise;
- pending tool linkage bypass;
- unsupported envelope bypass.

## Results

### Deterministic paired evidence backend

Result:

- `results/filter_value_v3_winnow_t090_v1.json`
- SHA-256: `1571c47beee5921047a49d3e130e997f398bb9e2add4c50bdb7147dabc016af7`

| Metric | Winnow @0.90 |
|---|---:|
| Request bytes | 5,960 |
| Removed bytes | 1,830 |
| Byte reduction | 30.7% |
| Paired answers | 12/12 |
| Paired regressions | 0 |
| Unsafe removals | 0 |
| Required-string failures | 0 |
| Restore round-trips | 5/5 OK |

Removals were observed on the clear irrelevant, tool-noise, mixed-part,
Spanish-language, and long-tail cases. Ambiguous/correction/protected-only cases
produced no unsafe reduction.

### Pinned local-generation backend

Result:

- `results/filter_value_v3_control_dedup_localgen_qwen35_4b_v1.json`
  - SHA-256: `bbd3928e21943fd81bb97ad889ca97aa0846bfcb5e7f98bf99a14fd155a653d5`
- `results/filter_value_v3_winnow_t090_localgen_qwen35_4b_v1.json`
  - SHA-256: `e63a031bf5f59a09fb4cffc80bf0d08959fc36477d46b4eedb92215c731ae736`

Backend:

```text
Qwen/Qwen3.5-4B @ 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
```

| Arm | Original answers | Reduced answers | Paired regressions | Unsafe removals |
|---|---:|---:|---:|---:|
| control | 9/12 | 9/12 | 0 | 0 |
| safe_dedup | 9/12 | 9/12 | 0 | 0 |
| Winnow @0.90 | 9/12 | 9/12 | 0 | 0 |

The local-generation run preserves the 9 originally successful answers. The lower
absolute answer count reflects the small local model and held-out case mix; it is
not a production-main-model quality estimate.

## Interpretation

1. V3 provides held-out evidence that the V2 diagnostic protocol is not entirely
   fixture-specific.
2. Winnow @0.90 still produces measurable hypothetical reduction on unseen
   synthetic cases: `30.7%` bytes.
3. Deterministic evidence preservation and Qwen3.5-4B generation both report zero
   paired regressions.
4. This remains synthetic and small. It does not authorize active filtering or a
   production threshold.
5. Provider billing savings remain unmeasured; local token estimates are not
   provider accounting.
