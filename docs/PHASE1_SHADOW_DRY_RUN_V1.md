# Phase1 shadow dry-run V1

Status: **loopback dry-run passed; Phase1 not authorized**.

This runner validates the Phase1 measurement-only path against a fake loopback
upstream. It exercises shadow scoring and receipts while proving that original
bytes are forwarded unchanged and no restore/reduction is sent.

## Command

```bash
python3 scripts/run_phase1_shadow_dry_run_v1.py \
  --output results/phase1_shadow_dry_run_v1.json
```

## Latest receipt

```text
status=phase1_shadow_dry_run_pass
cases=8
provider_calls=0
active_filtering_applied=false
SHA-256=152ac85eb905440e9619b1d1cb8a7e016395af4311ddc87f9975f20279e306d6
```

## Coverage

| Case | Verified behavior |
|---|---|
| `shadow_openai_eligible_plan` | OpenAI chat shadow proposal recorded; original bytes sent |
| `shadow_anthropic_eligible_plan` | Anthropic shadow proposal recorded; original bytes sent |
| `shadow_responses_eligible_plan` | OpenAI Responses shadow proposal recorded; original bytes sent |
| `shadow_no_sidecar` | no eligibility → no proposal; original bytes sent |
| `shadow_internal_headers` | internal headers stripped before upstream |
| `shadow_unsupported_embeddings` | unsupported path bypasses gate unchanged |
| `shadow_kill_switch` | kill switch forwards unchanged |
| `shadow_scorer_error` | scorer exception cannot modify forwarded bytes |

## Boundary

This is Phase1 plumbing evidence only. It does not authorize a provider call,
active filtering, or billing-savings claims.
