# Local stack operator sheet V1

One-page daily operations for the NanoJev local stack.

## The one service: `nanojev` on 127.0.0.1:8876

Callers only ever talk to this port. Scorer backends are internal.

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | ready + backend status |
| `POST /api/evaluate` | NanoJev typed decisions |
| `POST /v1/systemone` | SystemOne scoring, `?backend=winnow` (default) / `kev` / `cascade` |
| `POST /v1/context-gate` | shadow filter eval: `{"request": <wire body>, "wire_format"?, "sidecar"?, "backend"?, "threshold"?}` |
| `GET /` | web demo (static) |

Backends (not caller-facing): Winnow-12B Q8 @8091, Kev-4B @8092.

## Daily start

```bash
bash scripts/start_local_stack.sh           # start whatever is down + health check
bash scripts/start_local_stack.sh --smoke   # same + one tiny evaluate call
```

`nanojev_skill.py health --start` alone starts the unified service;
the stack script also starts both scorer backends. For login auto-start +
crash keepalive: `bash deploy/launchd/install.sh` (reversible with
`--uninstall`).

For agent CLIs calling this service, see `docs/NANOJEV_SERVICE_API_V1.md`.

## Scorer choice (measured)

| Scorer | Latency/state | Notes |
|---|---:|---|
| Winnow-12B Q8 only | ~0.29 s | **default** — 0.8824 on the frozen bundle |
| Cascade Kev-4B → Winnow | ~0.80 s | same decisions on V3 fixtures; slower on MPS |

Keep Winnow-only unless a faster fast-path shows up.

## Gateway (shadow default)

```bash
python3 scripts/main_model_gateway_v1.py --upstream <origin> \
  --receipt-log receipts.jsonl \
  --scorer systemone --scorer-url http://127.0.0.1:8091 \
  --scorer-model Winnow-12B --mode shadow
```

Useful flags: `--min-reduction-bytes N` (skip tiny savings),
`--max-scorer-payload-bytes N` (staged scorer-state fitting),
`--kill-switch` (forward everything, no scoring).

## Regression receipts (all loopback, provider_calls=0)

```bash
python3 scripts/run_gateway_winnow_shadow_v1.py      # shadow × live Winnow
python3 scripts/run_gateway_live_active_v1.py      # apply path × live cascade
python3 scripts/run_cascade_live_v1.py             # fast/strong routing
python3 scripts/run_winnow_stress_v1.py            # concurrency/isolation
python3 scripts/run_active_mode_phase0_v1.py       # 17-case deterministic suite
```

## Stop

```bash
pkill -f serve_decisions.py
pkill -f "kev.serve"
kill $(pgrep -f winnow-server)
```

## Boundaries

Shadow by default; active filtering stays test-only; no provider calls;
receipts are content-free.
