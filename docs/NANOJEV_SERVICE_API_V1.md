# NanoJev service API contract V1

For local agent CLIs calling the unified nanojev service on this machine.

## Base

```text
http://127.0.0.1:8876   (auto-fallback may pick 8765 or 8877-8890 if 8876 is busy)
GET /api/health → {"ready": true, "backends": {"winnow": true, "kev": true}}
```

No authentication — loopback-only, protected by binding. Do not expose.

## Contract rules (read first)

1. **Advisory only.** Every endpoint returns typed suggestions/probabilities.
   Nothing returned authorizes an action. Callers must run their own
   deterministic checks before side effects.
2. **Shadow semantics.** `/v1/context-gate` evaluates what *would* be removed;
   it never modifies or forwards anything. There is no active filtering.
3. **Content-free receipts.** Responses contain hashes, JSON pointers,
   reason codes, and probabilities — never raw request text.
4. **Fail-open.** Scorer/backend errors return HTTP 502/500 or a bypass
   receipt — treat failure as "keep everything".

## POST /api/evaluate — NanoJev typed decisions

The pinned local 0.6B model. Input:

```json
{
  "states": [{"id": "s1", "state": {"any": "json"}}],
  "questions": {"q": {"type": "boolean"|"choice"|"noul"|"score", ...}}
}
```

Returns per-state typed answers with probabilities. The model abstains
(`choice: null`, `abstained: true`) when below threshold — abstention is an
answer, not an error, and never an authorization.

## POST /v1/systemone — scorer routing

Body is a TypeSafe `/v1/systemone` request (`{"model"?, "state", "questions"}`).

| Query | Backend | Notes |
|---|---|---|
| *(none)* or `?backend=winnow` | Winnow-12B Q8 @8091 | default, ~0.3s/state |
| `?backend=kev` | Kev-4B @8092 | faster small model |
| `?backend=cascade` | kev → winnow fallback | confident answers stay on kev |

Response is the backend's native `{model, answers, usage}` map.

## POST /v1/context-gate — request filtering evaluation

```json
{
  "request": {"model": "x", "messages": [...]},
  "wire_format": "openai_chat",            // or anthropic_messages / openai_responses
  "sidecar": {"segments": {"/messages/2/content": {"eligible": true}}},
  "backend": "winnow",                     // optional
  "threshold": 0.9,                        // optional, default 0.99
  "max_scorer_payload_bytes": 8000         // optional, staged fitting
}
```

Returns a `nanojev-context-shadow-v1` receipt:

- `status`: `scored` | `bypass` — bypass reasons: `unsupported_wire_format`,
  `unsupported_envelope_fields`, `unresolved_tool_link`, `no_eligible_segments`,
  `uncertain_score`, `scorer_failure`, `scorer_state_budget_exceeded`
- `segments[]`: per-pointer `{role, suggestion, reason, p_irrelevant, applied}`
  — `applied` is always `false` in shadow mode
- `forwarded_unchanged`: always `true`
- Protected pointers (system/user roles, credentials, tool links, non-eligible
  segments) always show `suggestion: retain`.

A `drop` suggestion means "this segment scored as safely removable" — it is
advice about a hypothetical reduction, not an instruction.

## Availability

```bash
bash scripts/start_local_stack.sh    # bring up service + backends
python3 scripts/check_local_services_v1.py --smoke
```

If launchd keepalive is installed (`docs/LOCAL_SERVICES_RUNBOOK_V1.md`),
the service comes up on login automatically.
