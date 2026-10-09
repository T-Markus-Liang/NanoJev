---
name: nanojev-local-service
description: Call the local unified NanoJev service (typed decisions, context filtering evaluation, scorer routing) on 127.0.0.1:8876 — a free, private, keyless local twin of the official TypeSafe Jev `/v1/systemone` API. Use for parallel local+official calls, offline/private judgments, context-filter shadow evaluation, or any bounded routing/Boolean/Noul/Score question where a local advisory answer suffices.
---

# NanoJev Local Service

A single local service at `http://127.0.0.1:8876` fronts the whole NanoJev
stack. It speaks the same `/v1/systemone` request/response shape as the
official TypeSafe Jev API, so the same payload can be sent to both for
parallel comparison, or to the local service alone when offline/privacy
matters. **No API key, no network egress, no cost.**

## Quick call (jev-eval-compatible CLI)

```bash
# Same input shape as jev-eval: {state, questions}
nanojev-eval input.json            # or stdin

# Backend selection: winnow (default, 12B, ~0.3s), kev (4B fast), cascade
nanojev-eval --backend cascade < input.json

# Context-gate: shadow filtering evaluation on a wire-format request
nanojev-eval --context-gate <<'EOF'
{"request": {"model": "x", "messages": [...]},
 "wire_format": "openai_chat",
 "sidecar": {"segments": {"/messages/2/content": {"eligible": true}}},
 "threshold": 0.9}
EOF
```

`nanojev-eval` auto-starts the service if it is down. Output mirrors
jev-eval: `{"answers": ..., "usage": ..., "model": ..., "elapsed_ms": ...,
"backend": ..., "source": "nanojev-local"}`.

## Parallel local + official Jev

```bash
# Official direct TypeSafe Jev (keyed, billed, strongest)
jev-eval <<'EOF' > /tmp/jev.json
{"state": {...}, "questions": {"q": {"type": "noul", "instructions": "..."}}}
EOF

# Local NanoJev service (free, private, advisory)
nanojev-eval <<'EOF' > /tmp/nanojev.json
{"state": {...}, "questions": {"q": {"type": "noul", "instructions": "..."}}}
EOF

# Compare probabilities; both are advisory — deterministic code decides.
```

## Traffic splitting — `jev-route`

Deterministically split calls between official and local by ratio
(default **30% local**). The same payload always routes the same way
(sha256 bucket), so splits are reproducible and auditable:

```bash
jev-route < input.json                     # ~30% local / ~70% official
jev-route --local-ratio 0.5 < input.json   # any ratio in [0,1]
JEV_ROUTE_LOCAL_RATIO=0.5 jev-route < input.json
```

Output adds `route: {planned, actual, local_ratio, fallback}`. If the local
path is down it falls back to official with `fallback: true`. Every routing
decision is appended to `~/.local/state/nanojev-eval/log.jsonl`
(content-free).

### Quality mode — official fallback for weak local answers

```bash
jev-route --mode quality < input.json              # local first
jev-route --mode quality --escalate-below 0.9 < input.json
```

Local answers first; when every answer's top probability reaches
`--escalate-below` (default 0.95) the local answer is returned. Otherwise
the request escalates to official Jev (`route.escalated: true`,
`route.local_confidence` recorded). Quality floor without a fixed ratio —
confident traffic stays free and local; uncertain cases get the strong
model. Local-down also falls back to official.

Use the official API when accuracy is critical; use the local service for
high-volume, private, or offline work. Never substitute local answers for a
required official check without measuring the gap.

## Raw HTTP endpoints

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | `{"ready", "backends": {"winnow", "kev"}}` |
| `POST /api/evaluate` | pinned NanoJev 0.6B typed decisions |
| `POST /v1/systemone` | scorer routing, `?backend=winnow\|kev\|cascade` |
| `POST /v1/context-gate` | shadow-mode context filtering receipt |

## Semantics — read before relying on output

- **Advisory only.** Answers are probabilities/suggestions; nothing
  authorizes an action. Deterministic caller code decides.
- **Shadow only.** `/v1/context-gate` reports what *would* be removed;
  it never modifies requests and `applied` is always `false`.
- **Fail-open.** Backend errors → HTTP 5xx or a bypass receipt; treat as
  "keep everything".
- **Quality ordering.** Measured on this project's frozen benchmark:
  official Jev 0.886 > Winnow-12B 0.882 > Kev-9B 0.819 > Kev-4B 0.802.
- **Keep local.** The service binds 127.0.0.1 only; never expose it.

## Usage log & feedback

Every `nanojev-eval` call appends a **content-free** line to
`~/.local/state/nanojev-eval/log.jsonl` (endpoint, backend, latency, status —
never payload text). If the service behaves oddly (wrong-looking answer,
error, slowness), leave a note for the owner:

```bash
nanojev-eval --feedback "short description of what went wrong"
```

Keep notes free of secrets and raw prompts. The owner reviews this log when
tuning the service — your reports directly improve it.

## Availability

The stack is launchd-managed (`deploy/launchd/`): it starts at login and
restarts on crash. Manual fallback: `bash scripts/start_local_stack.sh`.
Health + determinism: `python3 scripts/check_local_services_v1.py --smoke --determinism`.
