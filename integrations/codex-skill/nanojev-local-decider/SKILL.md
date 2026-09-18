---
name: nanojev-local-decider
description: Use the locally deployed NanoJev checkpoint for structured routing, candidate selection, confidence gates, verification, and computer-use decisions; do not use it for chat, code generation, or long-form reasoning.
---

# NanoJev Local Decider

Use this skill when a task needs a bounded decision rather than generated prose: choose one candidate, estimate whether a proposition is true, rank ordered levels, route a request, select a tool, or gate a risky action. Keep the primary work in the stronger general model and use NanoJev as a local decision component.

## Operating rules

- Prefer the local-only service at `127.0.0.1:8765`. The helper disables proxy inheritance for loopback requests and the NanoJev checkpoint is loaded from disk with offline Hugging Face and Transformers settings.
- Do not send credentials, secrets, private user text, or raw application state to a remote provider through this skill.
- Do not use NanoJev for conversation, code generation, explanation, planning prose, or decisions that require broad world knowledge.
- Make independent questions explicit in one `states` request. Do not put prior model answers into a later question unless the workflow intentionally makes that dependency part of the state.
- Use `choice` for selecting among named candidates, `boolean` for a proposition, `noul` when Jev-compatible naming is required, and `score` for an ordered scale. The helper maps `noul` to the current local `boolean` core and maps the response back.
- For consequential actions, treat low confidence as a reason to abstain or escalate. Set `abstain_below` on a question when a threshold is appropriate; the helper applies this gate after inference and never pretends an abstention is a confident answer.
- Always record the decision through the helper. Logs contain privacy-preserving metadata by default: counts, types, candidate cardinalities, latency, confidence, device, checkpoint identity, and a SHA-256 request fingerprint. Raw state and criteria are not logged unless the user explicitly enables `NANOJEV_LOG_PAYLOADS=1`.
- When the downstream result becomes known, record feedback with `record-feedback`. Use `correct`, `incorrect`, `abstained`, `fallback`, or `human_override`.

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
