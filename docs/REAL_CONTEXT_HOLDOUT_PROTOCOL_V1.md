# Real-context holdout protocol V1

Status: **protocol valid, not collecting**.

This protocol defines how a future real-context holdout may be collected and
evaluated. It does not start collection, does not call a provider, and does not
enable active filtering.

## Artifacts

- Protocol: `research/real_context_holdout_protocol_v1.json`
- Validator: `scripts/validate_real_context_holdout_protocol_v1.py`
- Preflight: `results/real_context_holdout_preflight_v1.json`

Current preflight:

```text
status=protocol_valid_not_collecting
collection_started=false
provider_calls_allowed=false
active_filtering_allowed=false
SHA-256=e460e8f11597053faefeb741f8639d3488fb215a13f6cee0a11aa1040ca654b2
```

## Data boundary

Raw holdout data, when authorized and collected, must live only under:

```text
data/real_context_holdout_v1/
```

`data/*` is already gitignored except for the disclosed public JevBench bundle.
No raw request, response, credential, or tool-output text may enter public
manifests or receipts.

## Allowed sources

- owner-selected local requests;
- owner-selected sanitized tool traces;
- owner-selected application logs after credential removal.

## Forbidden sources

- JevBench or other benchmark rows;
- provider outputs used as labels;
- real credentials/tokens;
- third-party private data without permission;
- unredactable sensitive requests.

## Labeling contract

Labels are manual and conservative:

- protected pointers are manually required;
- required downstream evidence must be declared;
- scorer outputs cannot become labels;
- uncertain context is retained;
- scored cases require a downstream contract.

## Coverage targets

When collection is authorized, the first cohort should contain at least:

| Requirement | Minimum |
|---|---:|
| total cases | 30 |
| distinct families | 8 |
| bypass cases | 3 |
| tool-linked cases | 3 |
| multilingual cases | 2 |
| protected-only cases | 2 |
| long-tail cases | 2 |
| mixed-part cases | 1 |
| mutable tool-result cases | 1 |
| nonrepeatable tool-result cases | 1 |

## Isolation

- No overlap with V2/V3 by `request_sha256`.
- Near-duplicates require manual review.
- Evaluation-only; no training or calibration fitting.

## Evaluation order

1. manifest schema/hash validation;
2. privacy and isolation checks;
3. deterministic shadow evaluation;
4. pinned local-generation paired evaluation;
5. optional phase1 shadow provider measurement after explicit authorization;
6. optional phase2 paired active canary after explicit authorization.

## Stop conditions

Stop immediately on:

- raw context outside the ignored data root;
- failed credential scan;
- V2/V3 or intra-holdout `request_sha256` overlap;
- unclear protected pointers or required evidence;
- unredactable sensitive content;
- attempted provider call or active filtering before explicit authorization.
