# NanoJev usage schema

The helper writes one JSON object per line to `~/.codex/nanojev/usage.jsonl` by default. The schema is intentionally small and privacy-preserving.

## Decision event

`event_type` is `decision`, `schema_version` is `nanojev-usage-v1`, and `event_id` is the identifier used when recording downstream feedback.

- `source` and `task_tag`: integration-owned labels, not raw user text.
- `checkpoint`: local checkpoint identity and configuration hash.
- `runtime`: device and precision reported by the service.
- `request`: counts, question types, candidate cardinalities, and a SHA-256 fingerprint of the normalized request.
- `result`: end-to-end latency, confidence summary, abstention count, candidate paths, forward passes, network model calls, and local inference index.

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

Readers should ignore unknown fields and accept future `schema_version` values only after an explicit migration. The helper's `summary` command reports aggregate latency, confidence, abstention, question-type, and feedback statistics without printing raw payloads.