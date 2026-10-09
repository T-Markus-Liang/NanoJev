# Valen Nano v2 — context_relevance_v1 label-contract fix

**Status:** fix applied and rebuilt as `data/valen_nano_v2/`; `data/valen_nano_v1/` left untouched
for audit. No training or deployment decisions are attached to this document.

## Diagnosis

`data/context_relevance_v1` was produced by a programmatic generator whose script is not in this
repository (the dataset is `self_authored_programmatic`; the same defect is already listed under
`invalid_predecessors` in `research/context_relevance_v1_protocol.json` as "contradictory overlap
labels").

Inspection shows each (family × candidate_kind) arm emits exactly one template. The
`correction` and `overlap_distractor` arms emit **the same template** in all six families —
states differ only in scenario ids and numbers:

| family | shared template shape (user → candidate) |
|---|---|
| code | "Update the `<id>` service to use port `<N>`…" → "A stale note says the `<id>` service once used port `<N>`." |
| order | "confirm whether `<N>` units at `<N>` are within the requested budget." → "An old quote for `<id>` used quantity=<N> and unit_price=<N>." |
| risk | "open a position of `<N>` units under its current limit?" → "A previous risk snapshot for `<id>` had a limit of `<N>` units." |
| support | "preserve the customer's requested refund deadline…" → "previously requested a refund by Monday before the customer corrected it to Friday." |
| robotics | "select the safe speed after the operator's latest correction." → "An earlier plan proposed `<N>` m/s before the operator correction." |
| multilingual | 保留用户最后确认的时间和地点 → 旧草稿…但用户后来已经更正 |

Labels were attached to the *declared arm* rather than the state content: all 206 `correction`
records carry `irrelevant=false` (keep) and all 221 `overlap_distractor` records carry
`irrelevant=true` (drop). After normalizing ids/numbers in conversation content, every record of
one kind collides with every same-family record of the other: **427/3360 records in
`data/valen_nano_v1` share normalized-identical states with opposite labels.**

This is a pure template collision — there is no hidden distinguishing field; the two arms are
byte-identical modulo ids. The oracle source
(`context_relevance_oracle_v1_seed20260919`) is unaffected: its `superseded`/`latest_correction`
states differ structurally (candidate pointer position) and its labels are consistent.

## Chosen contract

> A stale/superseded candidate fact is `irrelevant=true` (drop) **only if the current value it
> was superseded by is present elsewhere in the state**. When no current value exists, the stale
> fact is the sole evidence and stays `irrelevant=false` (keep) — matching the question's own
> instruction ("answer false … when uncertainty remains").

Applied per family to the shared template:

| family | superseding current value elsewhere? | contract label |
|---|---|---|
| code | yes — new port is in the user request | drop (true) |
| order | yes — current quantity/unit_price are in the user request | drop (true) |
| risk | no — "current limit" is never stated | keep (false) |
| support | no — corrected deadline exists only in the candidate | keep (false) |
| robotics | no — corrected speed is absent | keep (false) |
| multilingual | no — corrected time/place are absent | keep (false) |

### Why relabel (option a), not a marker (b) or dropping templates (c)

- (b) adding a distinguishing field would fabricate state content the generator never produced,
  and cannot repair `code`/`order` correction records, whose states already contain the current
  value in the user request — under the contract they are honestly *irrelevant*, so the label,
  not the state, was wrong.
- (c) dropping both templates deletes 427 records and removes clean coverage of two valuable
  cases: "stale fact is sole evidence → keep" (risk/support/robotics/multilingual) and "stale
  fact superseded by the request itself → drop" (code/order).
- (a) makes the label a pure function of the state, which is exactly the property the
  contradiction violated.

## Implementation

- `scripts/fix_context_relevance_v1_contract_v1.py` — relabels `correction`/`overlap_distractor`
  records per the contract and writes `data/context_relevance_v2_seed20260919/` (all five splits;
  test/ood relabelled by the same rule and remain frozen). Changed rows get
  `metadata.contract_fix` with the previous label and reason.
- `external/valen/scripts/build_nanojev_valen_v2.py` — builds `data/valen_nano_v2/` with
  `schema_version` `nanojev-valen-nano-v2`, same split rule (`sha256(group_id) mod 100 < 20 ->
  eval`), same group isolation and same eval-only exclusions. `group_id` keeps the
  `context_relevance_v1` lineage prefix for the fixed source so the split assignment — and hence
  the eval set — is identical to v1.
- `scripts/validate_valen_label_consistency_v1.py` — the regression check that caught this bug:
  normalizes digit runs in conversation content and `scenario_id` (pointers are kept verbatim;
  message indices are positional semantics), then asserts no normalized-identical state group
  carries more than one argmax label. Exits 1 on violations.
- `scripts/test_fix_context_relevance_v1_contract_v1.py` — unit tests for the relabel rule, the
  builder manifest, and the validator.

## Contamination and counts

| metric | valen_nano_v1 | valen_nano_v2 |
|---|---:|---:|
| contradictory records (normalized-state, opposite labels) | 427 / 3360 | **0 / 3360** |
| normalized-state label ceiling, eval | 0.9572 | **1.0000** |
| normalized-state label ceiling, train | 0.9433 | **1.0000** |

(The earlier estimate of ~0.932 used coarser normalization that also collapsed pointer indices;
with pointer-aware normalization the v1 eval ceiling is 0.9572 — 29 eval records sit in
contradictory groups with at most `max(count_true, count_false)` recoverable per group.)

Relabel counts in the valen-consumed splits (train/dev/calibration): **226 records flipped** —
73 `correction` on code/order (false→true) and 153 `overlap_distractor` on
risk/support/robotics/multilingual (true→false). All 5 splits of the v2 source: 316 flips
(train 164, dev 28, calibration 34, test 46, ood 44).

| valen_nano_v2 | records | groups | label_counts |
|---|---:|---:|---|
| train | 2682 | 1142 | false 1400 / true 1282 |
| eval | 678 | 268 | false 361 / true 317 |
| total | 3360 | 1410 | — |

Record set, group ids and split assignment are identical to v1; only `targets` (and
`meta.source_dataset`) changed on the 226 flipped records. Known remaining property: each
family×kind is one template, so normalized near-duplicates span the train/eval split by design —
after the fix they carry consistent labels, so this is redundancy, not contradiction.
