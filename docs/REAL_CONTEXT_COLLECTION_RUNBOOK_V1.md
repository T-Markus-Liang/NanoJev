# Real-context collection runbook V1

Status: **authorization template ready; collection not authorized**.

This runbook is the operator checklist for any future owner-authorized collection
of real-context holdout cases. It does not start collection.

## Artifacts

- Authorization template: `research/real_context_collection_authorization_v1.json`
- Validator: `scripts/validate_real_context_collection_authorization_v1.py`
- Preflight receipt: `results/real_context_collection_authorization_preflight_v1.json`
- Intake protocol: `docs/REAL_CONTEXT_HOLDOUT_PROTOCOL_V1.md`
- E2E runner: `docs/REAL_CONTEXT_HOLDOUT_E2E_PREFLIGHT_V1.md`

Current authorization preflight:

```text
status=template_ready_not_authorized
collection_authorized=false
provider_calls_allowed=false
active_filtering_allowed=false
SHA-256=49db63deedc00016fe481f46de9ef798fc736aa284aa97cdb07db46a7ef14d51
```

## Before collecting anything

The owner must explicitly set these fields in a new scoped artifact, not by
silently editing the template:

```text
collection.authorized=true
collection.authorized_by=<owner identity>
collection.authorized_at=<timestamp>
collection.scope=<allowed source scope>
collection.expires_at=<expiry timestamp>
```

The default template keeps all of those unset.

## Operator checklist

| Step | Required check |
|---|---|
| 1 | Name the source scope, max case budget, and expiry |
| 2 | Confirm every source is owner-selected and allowed by the intake protocol |
| 3 | Remove credentials, secrets, personal identifiers, and unredactable content |
| 4 | Manually label protected pointers, required evidence, downstream contract, and gate status |
| 5 | Build the content-free manifest |
| 6 | Run manifest validation and E2E preflight |
| 7 | Confirm rollback: stop collection and remove local-only files |

## Collection-time rules

- Raw files stay under `data/real_context_holdout_v1/`.
- `data/*` remains gitignored.
- No benchmark rows.
- No provider outputs as labels.
- No scorer outputs as labels.
- No raw context or contract files in public docs/receipts.
- No provider calls.
- No remote scorer calls.
- No active filtering.
- No training or calibration use.

## Commands after collection

```bash
python3 scripts/real_context_holdout_manifest_v1.py build \
  --case-dir data/real_context_holdout_v1/cases \
  --data-root data/real_context_holdout_v1 \
  --output data/real_context_holdout_v1/manifest.json

python3 scripts/run_real_context_holdout_preflight_v1.py \
  --manifest data/real_context_holdout_v1/manifest.json \
  --case-dir data/real_context_holdout_v1/cases \
  --contract-dir data/real_context_holdout_v1/cases/contracts \
  --backend deterministic \
  --require-ready \
  --output results/real_context_holdout_e2e_preflight_v1.json
```

## Stop immediately if

- a credential-like pattern is detected;
- raw context appears outside the ignored data root;
- provenance is unclear;
- protected pointers cannot be labeled confidently;
- manifest or E2E preflight fails;
- owner revokes scope or requests stop.

## Current decision

```text
COLLECTION: NOT AUTHORIZED
PROVIDER: DISABLED
ACTIVE FILTERING: DISABLED
```
