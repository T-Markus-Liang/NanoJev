# NanoJev usage schema

The helper writes one JSON object per line to `~/.codex/nanojev/usage.jsonl` by default. The schema is intentionally small and privacy-preserving.

## Decision event

`event_type` is `decision`, `schema_version` is `nanojev-usage-v1`, and `event_id` is the identifier used when recording downstream feedback.

- `source` and `task_tag`: integration-owned labels, not raw user text.
- `checkpoint`: local checkpoint identity and configuration hash.
- `authorizes_execution`: always `false`.
- `operating_envelope`: default abstain threshold (`0.9`), the evidence pointer
  (`docs/NANOJEV_SKILL_READINESS_V1.md`), and `authorizes_execution: false`.
- `runtime`: device and precision reported by the service.
- `request`: counts, question types, candidate cardinalities, and a SHA-256 fingerprint of the normalized request.
- `result`: end-to-end latency, confidence summary, abstention count, out-of-scope count, per-status
  counts, scope policy id, candidate paths, forward passes, network model calls, and local inference index.
- `workflow` (lifecycle calls): stage (`development`, `testing`, `optimization`, `deployment`),
  `advisory_only: true`, `requires_independent_verification: true`, and `authorizes_execution: false`.

Missing/invalid response probabilities, incomplete answers, a mismatched checkpoint, or a
reported remote model call are errors, not successful decision events. Abstention leaves
scores available but clears `choice`/`value`; it never authorizes an operation.

## Scope guard and answer status

Every answer carries a `status`:

- `in_scope_advisory`: bounded question, answered, advisory only. `value` or `choice` is present.
- `abstained`: confidence was below the question's `abstain_below` gate. `abstain_reason` is
  `confidence_below_threshold`; `value`/`choice` are cleared and the raw proposal is kept in
  `suggested_value`.
- `out_of_scope`: the deterministic guard (`nanojev-scope-guard-v1`) recognised an engineering
  judgment or authorization question (safety, approval/deploy/commit, test/gate requirement or
  bypass, context/artifact removal). The answer is refused regardless of confidence:
  `out_of_scope: true`, `out_of_scope_reason`
  (`engineering_judgment_or_authorization_out_of_scope`), `out_of_scope_patterns` and
  `out_of_scope_matched_phrases` name the trigger, and `value`/`choice` are cleared while
  `suggested_value` preserves the raw proposal.

The guard changes presented status only. `probabilities`, `p_true`, and `confidence` are recorded
unchanged in every case, and `abstained` remains a pure confidence-gate measurement (it can be
`false` while `status` is `out_of_scope`). The response-level `scope_assessment` block reports
`out_of_scope`, `out_of_scope_count`, `out_of_scope_questions`, the policy id, the required
action, and `authorizes_execution: false`. `decision_summary` adds `out_of_scope`,
`answered_advisory`, and `status_counts`. `operating_envelope` on the response repeats the
measured threshold behaviour and the evidence pointer. Authoritative behaviour is documented in
`SKILL.md`; evidence is in `docs/NANOJEV_SKILL_READINESS_V1.md`.

Raw states and criteria are omitted. Set `NANOJEV_LOG_PAYLOADS=1` only for a deliberate local debugging session, and turn it off afterward. Do not enable that mode for credentials, tokens, or sensitive user data.

## Feedback event

Feedback references a decision through `decision_event_id` and uses one of:

- `correct`: downstream outcome supports the decision.
- `incorrect`: downstream outcome disproves the decision.
- `abstained`: escalation was the right outcome or the model abstained.
- `fallback`: a stronger model or another rule made the final decision.
- `human_override`: a human changed the selected decision.

Feedback is deliberately separate from inference. This prevents the service from silently treating its own choice as ground truth and makes later calibration and task-family analysis possible.

## Compatibility

Readers should ignore unknown fields and accept future `schema_version` values only after an explicit migration. The helper's `summary` command reports aggregate latency, confidence, abstention, out-of-scope, question-type, and feedback statistics, plus the operating envelope, without printing raw payloads.
