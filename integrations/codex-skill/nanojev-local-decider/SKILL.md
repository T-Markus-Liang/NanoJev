---
name: nanojev-local-decider
description: Use the local NanoJev model for bounded structured decisions and advisory checks throughout NanoJev development, testing, optimization, and deployment; also supports routing, choice, Boolean/Noul and Score queries. Not a code generator or an authorization system.
---

# NanoJev Local Decider

Use this skill when a task needs a bounded decision rather than generated prose: choose one candidate, estimate whether a proposition is true, rank ordered levels, route a request, select a tool, or gate a risky action. Keep the primary work in the stronger general model and use NanoJev as a local decision component.

## NanoJev engineering lifecycle (user-required)

For work on NanoJev, call this skill at the start of each development, testing, optimization,
and deployment phase with current observed facts and a bounded candidate set. Call again
when material new evidence changes the decision. This is a real local inference requirement,
not permission to substitute a canned response, skip a phase, or claim the model was used
without an event ID. One batched request can cover independent checks within a phase.

Use `lifecycle --stage development|testing|optimization|deployment`. Example (absolute helper path):

```bash
python3 /Users/markus/.codex/skills/nanojev-local-decider/scripts/nanojev_skill.py lifecycle \
  --stage testing \
  --state 'The local-only URL guard changed; its unit tests have not run.' \
  --candidates '{"unit":"Run URL and redirect unit tests","benchmark":"Benchmark inference latency"}'
```

- Development: prioritize a bounded implementation/check from observed requirements.
- Testing: choose relevant tests or classify observed failure evidence; actually run tests separately.
- Optimization: compare measured candidates; do not invent timing, quality, or token savings.
- Deployment: advise which readiness check to run. Deterministic gates and user authority decide deployment.

The current `local_atomic_seed17` checkpoint is game-trained. Engineering-task quality is
not established. Confidence is uncalibrated; even a high-confidence proposal needs independent
verification. Default threshold is 0.9. An abstention clears the actionable choice/value and
must be handled by the main model or a deterministic rule. Never lower the threshold merely
to get an answer. Record observed outcomes as feedback, not the model's own predictions.

If service start/inference fails, report the failed local invocation and repair or escalate;
never silently route to a cloud model or fabricate a decision. Do not perform deployment
or bypass a mandatory roadmap review gate because of any NanoJev score.

## Operating rules

- Use only the local service (default `127.0.0.1:8765`; a conflicting port is automatically avoided and remembered, currently `8876`). The helper rejects remote URLs and redirects, disables proxy inheritance, and starts the model in offline mode.
- Do not send credentials, secrets, private user text, or raw application state to a remote provider through this skill.
- Do not use NanoJev for conversation, code generation, explanation, planning prose, or decisions that require broad world knowledge.
- Make independent questions explicit in one `states` request. Do not put prior model answers into a later question unless the workflow intentionally makes that dependency part of the state.
- Use `choice` for selecting among named candidates, `boolean` for a proposition, `noul` when Jev-compatible naming is required, and `score` for an ordered scale. The helper maps `noul` to the current local `boolean` core and maps the response back.
- For consequential actions, treat low confidence as a reason to abstain or escalate. Set `abstain_below` on a question when a threshold is appropriate; the helper applies this gate after inference and never pretends an abstention is a confident answer.
- Always record the decision through the helper. Logs contain privacy-preserving metadata by default: counts, types, candidate cardinalities, latency, confidence, device, checkpoint identity, and a SHA-256 request fingerprint. Raw state and criteria are not logged unless the user explicitly enables `NANOJEV_LOG_PAYLOADS=1`.
- When the downstream result becomes known, record feedback with `record-feedback`. Use `correct`, `incorrect`, `abstained`, `fallback`, or `human_override`.
- All lifecycle output is advisory and carries `authorizes_execution: false`. No command, file mutation, trade, active context deletion, or provider change is automatically executed by this helper.

## Helper commands

The deterministic helper is `scripts/nanojev_skill.py` in this skill directory. It can start the persistent service on demand.

```bash
python3 scripts/nanojev_skill.py health --start
python3 scripts/nanojev_skill.py decide --input request.json --source codex --task-tag routing
python3 scripts/nanojev_skill.py record-feedback --event-id EVENT_ID --label correct
python3 scripts/nanojev_skill.py summary
```

Use `--input -` to read a JSON request from stdin. The request follows the local NanoJev `{"states": [...]}` contract. Environment overrides are available for `NANOJEV_URL`, `NANOJEV_PROJECT_ROOT`, `NANOJEV_CHECKPOINT`, `NANOJEV_PYTHON`, `NANOJEV_LOG`, `NANOJEV_RUNTIME_DIR`, `NANOJEV_DEVICE`, and `NANOJEV_PRECISION`.

Read [references/usage-schema.md](references/usage-schema.md) when adding an integration, interpreting telemetry, or changing the event format. Keep changes backward-compatible with `schema_version: nanojev-usage-v1`.
