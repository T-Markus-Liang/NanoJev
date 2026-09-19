---
name: nanojev-local-decider
description: Use the local NanoJev model for bounded structured decisions and advisory checks throughout NanoJev development, testing, optimization, and deployment; also supports routing, choice, Boolean/Noul and Score queries. Deterministically refuses engineering-judgment and authorization questions (status out_of_scope). Not a code generator or an authorization system.
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

## Scope guard: engineering judgment and authorization are out of scope

The helper applies a deterministic, documented text guard (`nanojev-scope-guard-v1`) before any
result is presented. It has no model component, no network, and no configuration flag to disable
it. When a question asks for engineering judgment or authorization, the helper marks that answer
`status: "out_of_scope"`, sets `out_of_scope: true`, records a machine-readable
`out_of_scope_reason` (`engineering_judgment_or_authorization_out_of_scope`) plus the matched
pattern IDs and phrases, and clears the presented selection (`value`/`choice`), even when the raw
confidence is high. The underlying `probabilities`, `p_true`, and `confidence` are preserved
unchanged: the guard changes the presented status, never the measurement.

It fires on these pattern families (exact lowercase substring matching after whitespace
normalization, over the state text, question instructions, and choice-candidate descriptions):

- `safety_judgment`: "is it safe to …", "is this safe", "safe to remove/delete/drop/deploy/…".
- `authorization_decision`: "should we approve/deploy/commit/merge/ship/release/proceed/…",
  "is it ok/okay/acceptable to …", "sign off", "green light", "go/no-go", "permission to proceed".
- `test_or_gate_requirement`: "does this change need a test", "need(s) a test", "is a test
  required", "skip/remove/replace/delete/waive/bypass the test(s)".
- `gate_or_approval_bypass`: "skip/bypass the gate or review", "without a validated gate",
  "replace/skip/remove/waive the approval".
- `context_or_artifact_removal`: "remove/delete/drop context", "remove active/production context",
  "should we remove/delete", "can this be removed".

The guard is redundant with the measurement on purpose and is expected to be the load-bearing
safety property for this checkpoint. It is deliberately fail-closed: if request wording resembles
an authorization or removal decision, the answer is suppressed rather than presented as advice.
An `out_of_scope` answer is not an abstention and not a negative answer — it is a refusal of the
question, and it must be routed to the main model plus deterministic gates and human authority.

## Operating envelope (measured)

Every `decide`/`lifecycle` result carries an `operating_envelope` block:

- `checkpoint_default_abstain_threshold: 0.9` — the documented default; do not lower it.
- At that threshold the shipped checkpoint abstained on **13/13 (100%)** realistic
  engineering-judgment questions; the highest confidence observed anywhere in that survey
  (0.736) was below the gate.
- Below the threshold it scored **3/6 (chance)**; the single highest-confidence answer in the
  survey — "safe to remove context without a validated gate" = `true` at 0.736 — was wrong in the
  safety-critical direction. An earlier "confidence is anti-correlated with correctness" claim was
  withdrawn as unsupported at n=6 (see the correction in the evidence doc); the conclusion is
  unchanged: this checkpoint cannot be trusted to rank its own answers.
- Engineering-task quality is **not established** for this game-trained checkpoint.
- Full evidence: [`docs/NANOJEV_SKILL_READINESS_V1.md`](../../../docs/NANOJEV_SKILL_READINESS_V1.md).

Practical rule: **invoke the skill for the receipt, never treat its output as the decision.**
The guard exists so that even a caller that ignores this advice cannot obtain a confident-looking
local answer to an authorization question.

## Operating rules

- Use only the local service (default `127.0.0.1:8765`; a conflicting port is automatically avoided and remembered, currently `8876`). The helper rejects remote URLs and redirects, disables proxy inheritance, and starts the model in offline mode.
- Do not send credentials, secrets, private user text, or raw application state to a remote provider through this skill.
- Do not use NanoJev for conversation, code generation, explanation, planning prose, or decisions that require broad world knowledge.
- Make independent questions explicit in one `states` request. Do not put prior model answers into a later question unless the workflow intentionally makes that dependency part of the state.
- Use `choice` for selecting among named candidates, `boolean` for a proposition, `noul` when Jev-compatible naming is required, and `score` for an ordered scale. The helper maps `noul` to the current local `boolean` core and maps the response back.
- For consequential actions, treat low confidence as a reason to abstain or escalate. Set `abstain_below` on a question when a threshold is appropriate; the helper applies this gate after inference and never pretends an abstention is a confident answer.
- Per-answer `status` is one of `in_scope_advisory` (bounded, answered, still not a decision), `abstained` (confidence below `abstain_below`), or `out_of_scope` (the deterministic scope guard refused an engineering-judgment/authorization question). Only `in_scope_advisory` presents a selection, and even that is advisory only.
- Always record the decision through the helper. Logs contain privacy-preserving metadata by default: counts, types, candidate cardinalities, latency, confidence, device, checkpoint identity, scope/out-of-scope counts, and a SHA-256 request fingerprint. Raw state and criteria are not logged unless the user explicitly enables `NANOJEV_LOG_PAYLOADS=1`.
- When the downstream result becomes known, record feedback with `record-feedback`. Use `correct`, `incorrect`, `abstained`, `fallback`, or `human_override`.
- All helper output is advisory, carries `authorizes_execution: false` (top level, per answer, in `scope_assessment`, in `operating_envelope`, and in the workflow/log receipt), and no command, file mutation, trade, active context deletion, or provider change is automatically executed by this helper.

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
