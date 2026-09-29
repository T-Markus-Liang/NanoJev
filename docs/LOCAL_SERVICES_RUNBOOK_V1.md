# Local services runbook V1

Daily-driver checks so progress does not silently depend on a dead service.

## Service map

| Service | Port(s) | Health endpoint | Required for daily work |
|---|---|---|---|
| NanoJev unified service | 8765 default, auto-fallback 8876–8890 | `GET /api/health` → `{"ready": true, "backends": {...}}` | yes — the only caller-facing port |
| Winnow-12B Q8 backend | 8091 | `GET /health` → `{"status": "ok"}` | scorer backend of the service |
| Kev-4B backend | 8092 | `GET /v1/models` → `kev-latest` | scorer backend of the service |

The NanoJev service is the single entrypoint: it serves the pinned local
checkpoint (`checkpoints/local_atomic_seed17/variants/local_atomic_seed17`),
the web demo, `/api/evaluate` typed decisions, `/v1/systemone` scorer routing
(`?backend=winnow|kev|cascade`, default winnow), and `/v1/context-gate`
shadow-mode request filtering evaluation. Backends are probed, never
started, by this service. Weights load once at startup; `provider_calls`
stays `0`.

## Keepalive (launchd)

`deploy/launchd/` ships three user agents so the stack survives reboots and
crashes (RunAtLoad + KeepAlive):

```bash
bash deploy/launchd/install.sh              # install + start at login
bash deploy/launchd/install.sh --uninstall  # remove agents
launchctl list | grep nanojev               # check status
```

Logs: `/tmp/nanojev-service.log`, `/tmp/winnow-server.log`,
`/tmp/kev-serve.log`. Without launchd, `bash scripts/start_local_stack.sh`
does the same job manually.

## Daily check

```bash
python3 scripts/check_local_services_v1.py            # probe only
python3 scripts/check_local_services_v1.py --smoke    # + one tiny evaluate call
python3 scripts/check_local_services_v1.py --smoke --determinism  # + same probe twice, must match
python3 scripts/check_local_services_v1.py --smoke --output results/local_services_health_v1.json
```

Or bring the whole stack up and check it in one step (idempotent — only
starts what is down):

```bash
bash scripts/start_local_stack.sh           # start missing services + health check
bash scripts/start_local_stack.sh --smoke   # same, with the evaluate smoke
```

Service logs land in `/tmp/winnow-server.log` and `/tmp/kev-serve.log`
(override with `NANOJEV_STACK_LOG_DIR`).

Exit code `0` when the required service is up (and smoke passed, if requested);
`1` otherwise. Latest observed: service on port 8876, smoke evaluate ~83ms.

## Restart

If the NanoJev service is down or wedged, the skill entrypoint can start it
(it auto-picks a free port in 8876–8890):

```bash
/Users/markus/.codex/skills/nanojev-local-decider/scripts/nanojev_skill.py health --start
```

Or start it directly:

```bash
python3 scripts/serve_decisions.py \
  --checkpoint-dir checkpoints/local_atomic_seed17/variants/local_atomic_seed17 \
  --web-root web --host 127.0.0.1 --port 8876 \
  --device auto --precision auto
```

Then re-run the health check. The service logs HTTP method/path/status only —
request states are never logged.

## Winnow-12B Q8 scorer (default strong scorer)

The gateway's strong-path scorer is the native `winnow-server` exposing
`POST /v1/systemone` on `127.0.0.1:8091`. Start it with the measured Apple
profile (12 GB model, loads once, Metal):

```bash
python3 external/winnow-inference/scripts/serve.py \
  --model external/models/Winnow-12B/Winnow-12B-Q8_0.gguf \
  --context 8192 --decision-parallel 1 \
  --chat-parallel 1 --memory exclusive
```

Verify:

```bash
curl http://127.0.0.1:8091/health          # {"status":"ok"}
curl http://127.0.0.1:8091/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"Winnow-12B","state":{"task":"smoke"},"questions":{"ok":{"type":"noul","instructions":"Is this a smoke check?"}}}'
```

Use it in the gateway:

```bash
python3 scripts/main_model_gateway_v1.py --upstream <origin> \
  --receipt-log receipts.jsonl \
  --scorer systemone --scorer-url http://127.0.0.1:8091 \
  --scorer-model Winnow-12B --mode shadow
```

Live check on this machine: `/v1/systemone` answers (~0.28 s per state),
`checkpoint.models=["Winnow-12B"]`.

## Kev-4B fast-path scorer

The cascade fast path is Kev-4B served by `kev.serve` on `127.0.0.1:8092`
(TypeSafe-compatible `/v1/systemone`):

```bash
cd external/kev
uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8092
```

Verify: `curl http://127.0.0.1:8092/v1/models` → `kev-latest`.

Live cascade (Kev fast → Winnow strong):

```bash
python3 scripts/run_cascade_live_v1.py   # receipt: results/cascade_live_v1.json
```

Or run the whole gateway on the cascade (fast `--scorer-url` = Kev 8092,
`--scorer-strong-url` = Winnow 8091):

```bash
python3 scripts/main_model_gateway_v1.py --upstream <origin> \
  --receipt-log receipts.jsonl \
  --scorer cascade --scorer-url http://127.0.0.1:8092 \
  --scorer-model kev-latest \
  --scorer-strong-url http://127.0.0.1:8091 \
  --scorer-strong-model Winnow-12B --mode shadow
```

Shadow regression through the live cascade:

```bash
python3 scripts/run_gateway_winnow_shadow_v1.py \
  --scorer-kind cascade --scorer-url http://127.0.0.1:8092 \
  --strong-url http://127.0.0.1:8091 \
  --output results/gateway_cascade_shadow_v1.json
```

## Notes

- Scorer endpoints on 8091/8092 are external model servers; the checker only
  reports them. They are expected to be down unless a scorer service was
  deliberately started.
- A `down` required service means lifecycle/`decide` calls will hang or fall
  back to a cold start; run the restart command before development work.

## Dogfooding — how agents should call it

This subsection is the usage contract for agents working on this machine; it
adds to the sections above and does not change them.

- **Prefer local for bounded questions.** Routing, classification, context
  filtering, candidate ranking, and obvious-risk flags should go to the local
  service first — it is free, keyless, private, and offline-capable. Open
  conversation, code generation, and high-stakes authorization stay out of
  Jev entirely (see global rules).
- **Default entrypoint: `jev-route --mode quality`.** Local answers first;
  anything below the confidence floor (default 0.95, tune with
  `--escalate-below`) escalates to official Jev automatically. Confident
  traffic stays free; uncertain traffic gets the strong model. Use
  `nanojev-eval` directly only when local-only is required (offline,
  privacy, volume) and `jev-eval` only when official quality is mandatory.
- **Leave `--feedback` notes.** When a local answer looks wrong, errors, or
  is slow: `nanojev-eval --feedback "one-line description"`. Notes are
  content-free (no prompts, no secrets) and land in
  `~/.local/state/nanojev-eval/log.jsonl`; the owner tunes the service from
  them.
- **Observability.** `python3 scripts/nanojev_usage_report_v1.py` summarizes
  the same log: calls by tool, local-vs-official split, quality-mode
  escalation rate, feedback entries, and the implied usage span.

## Lifecycle "Expecting value: line 1 column 1" — root cause and fix (T147, 2026-09-26)

**Symptom.** Subagents running
`nanojev_skill.py lifecycle --stage ...` intermittently failed with
`Expecting value: line 1 column 1` while `curl http://127.0.0.1:8876` looked
fine and one diagnostic saw `/health` → 404.

**Root cause — port drift, not a schema mismatch.** The service exposes
`/api/health` (not `/health`) and `/api/evaluate`; those were always correct.
The skill's built-in default URL was `http://127.0.0.1:8765`, but the unified
service (T49/W93) is pinned to `127.0.0.1:8876` by the `ai.nanojev.service`
launchd keepalive, and 8765 is now occupied by an unrelated HTTP server
(`ui.app_server`) that answers `/` with HTML 200 and unknown paths with HTML
errors. The skill only discovered 8876 via `~/.codex/nanojev/service.url`.
Whenever that file was absent, stale, or unreadable in the caller's context,
the health probe hit the foreign server (non-JSON/HTML body → `json.loads`
raised `Expecting value: line 1 column 1`, or HTTP 404), and the
spawn-fallback then tried to start a second full model server on 8877–8890
(slow cold start, sometimes inside the per-call timeout) — surfacing as
"lifecycle endpoint unavailable" in parallel sessions.

**Fix (skill side only; service behavior unchanged for `nanojev-eval` /
`/v1/systemone` / ledger callers).** In
`integrations/codex-skill/nanojev-local-decider/scripts/nanojev_skill.py`
(synced to all installed agent skill dirs):

- `DEFAULT_URL` is now `http://127.0.0.1:8876` (the launchd-pinned port);
  `NANOJEV_URL` still overrides.
- `ensure_service` / `health` now run `discover_service`: after the default
  URL and `service.url` fail, they probe the known ports (8765 + 8876–8890,
  same list as `check_local_services_v1.py`) and use the first healthy
  NanoJev (`ready:true`, `provider_calls:0`) instead of spawning a duplicate.
  A discovered URL is written back to `service.url` (self-healing).
- A non-JSON `/api/health` body now reports "foreign service may occupy the
  port" instead of a bare JSON error; malformed `--candidates` JSON reports
  the expected shape.

**Operational notes.** `service.url` is no longer load-bearing. If 8876 is
squatted by a foreign server, the service falls back to 8877–8890 and
discovery still finds it. The service is single-threaded: a health probe can
block behind an in-flight `/api/evaluate` (~seconds–~30s), so concurrent
lifecycle calls may still see slow first probes — discovery and the launchd
keepalive now cover that path.

## Forward paper-trade ledger (T90)

The daily `ai.nanojev.forward-ledger` launchd job (installed by the same
`deploy/launchd/install.sh`; `StartCalendarInterval` 07:10 local) runs the
Binance supplement refresh, then the append-only ledger, then `--snapshot`.
Output appends to `logs/forward-ledger.log`; the refresh step is
offline-safe (noops cleanly when data.binance.vision is unreachable).

Status 2026-09-24: INSTALLED — bootstrapped into `gui/501` and verified by
`launchctl kickstart`: refresh `refreshed` (111 records, newest-day archive
404s benign), ledger `+3 bars / +1 exit`, snapshot written; job exit 0 and
health probe `ok`. TCC note: the plist must invoke a Documents-granted
interpreter (`.venv/bin/python`); `/usr/bin/python3` fails in the launchd
context with `Operation not permitted`, and the `job-working-directory:
getcwd` line in the log is benign noise from the `/bin/bash` wrapper.

Health probe:

```bash
python3 scripts/check_forward_ledger_v1.py
```

Exit `0` when the newest ledger bar is within 48h of now; `1` with
`status: stale`/`missing`/`empty` otherwise, including the last bar date
per asset and whether the refresh supplement ran ahead of the ledger.
Use `--max-age-hours` to tune the staleness window.
