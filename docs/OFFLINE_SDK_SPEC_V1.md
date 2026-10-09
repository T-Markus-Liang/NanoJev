# Offline Typed SDK + Packaging — Design Specification V1 (J-P)

**Status: design specification only, 2026-09-21. Nothing in this document is implemented unless a
"Existing component" reference says otherwise.**
**Scope:** roadmap task J-P (`docs/NANOJEV_V2_ROADMAP.md`, row J-P): the offline-capable typed
SDK and packaging layer for the NanoJev decision model — "local high-speed decision OS":
typed structured judgment + strict calibration/abstention + front-layer routing to bigger
models (W49 positioning). This document specifies pack layout, API surface, offline
guarantees, lifecycle, serving policy, runtime budgets, receipts, acceptance gates and
non-goals. It authorizes nothing: `deployment_authorized=false` until §8 gates are green.
**Upstream evidence:** `scripts/predict_toy_decisions.py` (DecisionPredictor + X3
shared-prefix path), `scripts/serve_decisions.py` (loopback HTTP), `scripts/selective_risk_v1.py`
+ `research/selective_risk_layer_v1.json` (serving policy layer), `integrations/codex-skill/
nanojev-local-decider` (skill, scope guard, usage schema), `docs/ECOSYSTEM_VERIFICATION_LEDGER_V1.md`
(laya-mlx reference checklist: packaged Python API, hash-pinned weights, opt-in compile/cache,
parity/stability checks, privacy-preserving receipts; Apache-2.0, weights have separate provenance).

---

## 0. Non-negotiable statements

1. **This spec authorizes nothing.** A conformant implementation still requires every
   acceptance gate in §8 before any deployment claim. `authorizes_execution` is `false` in
   every artifact, response and receipt this spec defines.
2. **Fail-closed everywhere.** Missing bytes, bad hashes, unknown `schema_version`, unknown
   detector id, expired deadline, absent scope signal — all are load/serving refusals, never
   degraded silent operation. There is no "warn and continue" path.
3. **Shadow is the default and the only unauthenticated mode.** A fresh install evaluates and
   records receipts but never surfaces an executable answer. Advisory answering requires an
   implemented, measured scope gate (§5.3); no mode executes anything.
4. **The J-C finding is a load-bearing constraint.** `research/selective_risk_layer_v1.json`
   records that calibration-fitted confidence thresholds fail OOD transfer (0/8 targets met on
   heldout_v2) and that protected errors at confidence ≥0.9 exist on OOD cohorts. Therefore a
   confidence threshold alone is **never** sufficient to serve; scope-gating is a first-class
   interface in the pack, the API, and every receipt.
5. **Honesty over capability.** Thresholds that are `unreachable_on_fit_split`, detectors that
   are `placeholder_interface_not_implemented`, and budgets that were never measured are
   reported as such. A spec-compliant artifact may not round a negative into a positive.

---

## 1. Package layout — the NanoJev decision pack

A **decision pack** is a versioned, self-describing, hash-pinned directory. It is deliberately
a **superset of the existing checkpoint directory contract**: a conformant pack root is itself
a valid `DecisionPredictor` checkpoint dir (`local_checkpoint_files`,
`scripts/predict_toy_decisions.py`), so today's runtime reads a pack with zero adaptation.

```
<pack_id>@<pack_version>/
  manifest.json                  # REQUIRED. nanojev-decision-pack-v1 (§1.1)
  best.safetensors               # REQUIRED. full merged weights (strict=True load)
  config.json                    # REQUIRED. run config, openjev-decision-pipeline-v1
  backbone_config/               # REQUIRED. HF config directory (config.json, …)
  tokenizer/                     # REQUIRED. HF tokenizer directory (all files)
  protocol/                      # REQUIRED. frozen protocol + amendments used to build/evaluate
    <protocol>.json              #   e.g. nanojev_v2_t9d_v5_lora_amendment_v2.json
  policy/
    selective_risk.json          # REQUIRED. nanojev-selective-risk-layer-v1 (§5)
    scope_gate.json              # REQUIRED. scope-gate policy declaration (§5.3)
  receipts/                      # REQUIRED. pack-build evidence (§8 inputs)
    parity_probe.json            #   parity-gate receipt for every non-default compute path
    scope_eval.json              #   measured scope-detector behaviour on OOD cohort
    budget_probe.json            #   cold/warm latency + peak RSS measurement receipt
    selective_risk_report.json   #   coverage-risk curves + transfer table + protected errors
  LICENSE.NOTICES                # REQUIRED when any artifact carries third-party terms
```

Every file under the pack root — including every file inside `backbone_config/` and
`tokenizer/` — appears exactly once in `manifest.artifacts`. A file on disk that is not in the
manifest, or a manifest entry whose file is absent, is a verification failure.

### 1.1 Manifest schema (`nanojev-decision-pack-v1`)

`manifest.json` is a JSON object. Unknown fields are rejected (closed schema), matching the
strict `validate_request` convention in `predict_toy_decisions.py`.

| Field | Rule |
|---|---|
| `schema_version` | exactly `"nanojev-decision-pack-v1"` |
| `pack_id` | non-empty string, `[a-z0-9][a-z0-9-]*` (e.g. `eng-judgment`) |
| `pack_version` | non-empty string; ordering key used by upgrade/rollback (§4) |
| `created_at_utc` | ISO-8601 UTC timestamp |
| `status` | one of `candidate`, `frozen`, `retired`; only `frozen` packs may leave shadow evaluation for advisory claims (§8 G1) |
| `checkpoint` | `{base_model, base_revision, resolved_model_revision, set_head}` — mirrors `config.json`; `set_head ∈ {none, attention, pointer}` |
| `contract` | request/response contract (below) |
| `artifacts` | non-empty list; each `{path, sha256, bytes, role}` (below) |
| `calibration` | `{spec: "policy/selective_risk.json", fitted_on_split, fitted_on_sha256}` — names the calibration cohort and its hash, never fitted on eval splits (isolation per `selective_risk_v1.py`) |
| `scope_gate` | `{policy: "policy/scope_gate.json", detector, status}` — §5.3 |
| `runtime_budget` | `{peak_rss_bytes, device_policy, request_limits}` — §6; values may be `null` only with `status:"unmeasured"` on the sibling field |
| `provenance` | `{training_run, corpus_sha256, protocol_sha256[], source_checkpoint}` — audit trail, not authorization |
| `manifest_sha256` | self-hash: sha256 of the canonical JSON (UTF-8, `sort_keys`, `separators=(",",":")` — the `canonical_json` convention in `nanojev_skill.py`) of the manifest **without** this field |

`contract` object:

| Field | Rule |
|---|---|
| `request_schema` | `"openjev-toy-inference-v1"` (the `{"states":[…]}` payload validated by `validate_request`) |
| `response_schema` | `"nanojev-decision-response-v1"` (§2.2) |
| `question_types` | subset of `["boolean","choice","score"]` the pack was evaluated for |
| `max_candidates` | `255` for choice (2–255 criteria), `10` for score (2–10 levels), `2` boolean — must match `validate_request` limits |
| `max_length` | token budget per candidate path; inputs exceeding it are errors, never truncated |
| `abstain_gate` | default `0.9`; per-question `abstain_below` may override within `[0,1]` |
| `protected_error_gate` | `0.9` — the zero-tolerance confident-error gate |

`artifacts[*]` object:

| Field | Rule |
|---|---|
| `path` | relative POSIX path; must resolve inside the pack root; no symlinks |
| `sha256` | 64-hex lowercase digest of file bytes |
| `bytes` | exact file size |
| `role` | one of `weights`, `run_config`, `backbone_config`, `tokenizer`, `protocol`, `policy`, `receipt`, `license`, `other` |

Hash verification order at load: (1) manifest parses and `schema_version` matches;
(2) `manifest_sha256` self-hash verifies; (3) every artifact's `bytes` then `sha256` verifies;
(4) `config.json` decodes, `set_head` is legal, `max_length ≤ max_position_embeddings`;
(5) only then is any weight byte touched by the model loader. Any failure raises
`PackVerificationError` and **no partial model state survives**.

---

## 2. API surface

### 2.1 Python API (`nanojev.sdk` — proposed module name)

Minimal surface, consistent with `DecisionPredictor` semantics:

```python
from nanojev.sdk import load_pack, Decider, PackVerificationError

pack = load_pack("~/.nanojev/packs/eng-judgment/0.3.1")   # §1 verify; raises on any mismatch
pack.identity()   # {pack_id, pack_version, manifest_sha256, checkpoint:{…}, contract:{…},
                  #  scope_gate:{detector,status}, status}
pack.verify()     # re-run full artifact verification (used by rollback drill + audits)

with Decider(pack, mode="shadow", device="auto", precision="fp32") as d:
    result = d.predict(
        {"states": [ … ]},           # openjev-toy-inference-v1 request
        task_tag="routing",          # receipt label; never raw user text
        deadline_ms=5000,            # queue+compute budget; timeout is an error, not a partial
        request_id=None,             # for cancellation (§6.3)
    )
```

`Decider` construction performs one persistent model load (the
`persistent_model_load_count: 1` convention) and honours §4 kill-switch and §6 budgets.
`mode ∈ {"shadow","advisory"}`; requesting `advisory` on a pack whose scope gate is a
placeholder raises at construction time, not at first request.

### 2.2 Response schema (`nanojev-decision-response-v1`)

The response **embeds the existing predictor payload unchanged** and adds a policy overlay —
no probability is recomputed, rescaled or hidden:

```json
{
  "schema_version": "nanojev-decision-response-v1",
  "pack": {"pack_id": "…", "pack_version": "…", "manifest_sha256": "…",
           "checkpoint": {"directory": "…", "base_model": "…", "set_head": "…"}},
  "mode": "shadow",
  "policy": {"abstain_gate": 0.9, "scope_detector": "nanojev-scope-guard-v1",
             "selective_risk_spec": "policy/selective_risk.json",
             "protected_error_gate": 0.9},
  "latency_ms": {"queue": 1.2, "compute": 1180.4, "total": 1181.6},
  "predictor": { "schema_version": "openjev-toy-inference-v1", "…": "unchanged predictor payload" },
  "states": [ {"id": "…", "answers": {"q": {
      "type": "choice", "probabilities": {"a": 0.6, "b": 0.4},
      "confidence": 0.6, "status": "abstained",
      "abstain_reason": "confidence_below_threshold",
      "out_of_scope": false,
      "value": null, "choice": null, "suggested_value": "a",
      "presented_value": null,
      "authorizes_execution": false }}} ],
  "decision_summary": {"questions": 1, "abstained": 1, "out_of_scope": 0,
                       "answered_advisory": 0, "status_counts": {"abstained": 1},
                       "confidence_min": 0.6, "confidence_max": 0.6, "confidence_mean": 0.6},
  "scope_assessment": {"policy": "nanojev-scope-guard-v1", "out_of_scope": false,
                       "out_of_scope_count": 0, "authorizes_execution": false},
  "network_model_calls": 0,
  "authorizes_execution": false
}
```

Answer `status` reuses the installed skill vocabulary (`usage-schema.md`):
`in_scope_advisory` | `abstained` | `out_of_scope`, plus `shadow` when `mode="shadow"`
(always: `presented_value=null`, raw `probabilities`/`suggested_value` preserved). Abstention
and out-of-scope both **clear `value`/`choice` and preserve raw scores** — the recorded
distribution is never altered by the gate (`raw_probabilities_preserved`).

### 2.3 Loopback HTTP service

The packaged service extends `scripts/serve_decisions.py`, keeping its exact constraints:
`http.server` on `127.0.0.1`/`::1`/`localhost` only (port range 8876–8890 convention from the
skill's `fallback_url`), POST body ≤ 2,000,000 bytes, cross-origin requests refused, no
redirects, no proxy honouring, method/path/status-only access logging.

| Endpoint | Behaviour |
|---|---|
| `GET /api/health` | `{ready, model_loaded_once, provider_calls:0}` **plus** `{pack identity, mode, scope_gate:{detector,status}, kill_switch:false}` |
| `GET /api/pack` | full pack identity + `manifest_sha256` + budget declarations (read-only) |
| `POST /api/evaluate` | `openjev-toy-inference-v1` request → `nanojev-decision-response-v1`; kill-switch active → `503 {"error":"nanojev_kill_switch_active"}`; queue full → `429`; deadline exceeded → `504` with no partial answers |

Client-side URL rules are inherited unchanged from `validate_local_url`: HTTP scheme, loopback
host only, no credentials, no query, no fragment. The service never binds a non-loopback
interface; a `--host` value outside the loopback set is a startup refusal.

---

## 3. Offline guarantees

"Offline" is a verifiable property, not a setting. At inference time the runtime MUST NOT:

- open any non-loopback socket, for any reason;
- contact Hugging Face Hub or any model/weights/tokenizer host (`HF_HUB_OFFLINE=1`,
  `TRANSFORMERS_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1` are set by the loader today and MUST
  remain set for the process lifetime);
- read credential material: no `.env`, no `HF_TOKEN`, no API keys, no auth headers, no
  credential helpers — a pack contains no secrets and needs none;
- honour HTTP(S) proxy environment variables for any call (the skill already builds an
  opener with `ProxyHandler({})`); pack code makes no HTTP calls at all;
- follow redirects (none are issued; any is an error);
- perform DNS resolution, telemetry, crash-reporting, update-checking, licence-phoning or
  NTP/other "harmless" beacons;
- download or fetch pack content at runtime — packs arrive on the filesystem before load
  (transport/distribution is out of runtime scope, §9).

**Enforcement.** Every response and receipt carries `network_model_calls: 0` (existing
execution field). The acceptance suite includes an offline drill (§8 G7): the full
load→predict→receipt path executed under network isolation must complete and report zero
calls. Hash verification at load (§1.1) makes corrupt/missing artifacts a `PackVerificationError`
before any inference — there is no "best effort" load.

---

## 4. Lifecycle management

### 4.1 Registry

A local registry directory (default `~/.nanojev/`, override `NANOJEV_HOME`) holds:

```
~/.nanojev/
  packs/<pack_id>/<pack_version>/     # unpacked packs, verified on install
  active.json                        # nanojev-pack-registry-v1, atomically replaced
  activations.jsonl                  # nanojev-pack-activation-v1 audit log (append-only)
  config.json                        # runtime config: {serving:{enabled,mode}, device_policy}
  KILL                               # sentinel file: presence = kill-switch (§4.4)
```

`active.json` (`nanojev-pack-registry-v1`): `{schema_version, pack_id, active_version,
previous_version, activated_at_utc, activation_id}`.

### 4.2 Install / upgrade

- `install`: verify the pack fully (§1.1) into `packs/<id>/<version>/`; a pack that fails
  verification is never registered. Reinstalling an identical version is a no-op only if
  hashes match; otherwise refused.
- `activate <id>@<version>`: re-verify the target pack, then write `active.json` to a temp
  file and `os.replace` it (atomic switch). The outgoing version is recorded as
  `previous_version` and **retained (N-1)**. Activation appends an audit event.
- Upgrade = install + activate. Downgrade = activate of an older registered version —
  same code path, same audit event.

### 4.3 Rollback

`rollback` re-verifies `previous_version`'s pack and atomically switches `active.json` back.
Rollback refuses if the previous pack no longer verifies (corrupt disk is not a reason to
serve). The drill in §8 G5 requires: activate v → activate v+1 → rollback → re-run a frozen
probe set → predictions must be **identical** to the pre-upgrade v predictions (pack identity
restored, not merely similar outputs).

### 4.4 Kill-switch — two independent mechanisms, either suffices

1. **Config flag:** `config.json` → `serving.enabled=false` (checked per request).
2. **File sentinel:** existence of `~/.nanojev/KILL` (checked per request, before config —
   works even if config is unreadable or maliciously edited).

Either condition → `Decider.predict` raises `KillSwitchActive`; HTTP returns `503
{"error":"nanojev_kill_switch_active"}`. Clearing requires removing the file **and** setting
`enabled=true` — restoring service is deliberate, never implicit (e.g. never auto-cleared by
an activation). Every kill/unkill transition appends an audit event with actor and reason.

### 4.5 Audit log (`nanojev-pack-activation-v1`)

Append-only JSONL; one object per line:

```json
{"event_type": "activate|rollback|install|kill|unkill|verify_fail|mode_change",
 "event_id": "uuid", "timestamp": "utc", "schema_version": "nanojev-pack-activation-v1",
 "pack_id": "…", "pack_version": "…", "manifest_sha256": "…",
 "actor": "cli|sdk|service", "reason": "…", "previous_version": "…|null",
 "authorizes_execution": false}
```

`verify_fail` events record which artifact failed (path + expected/actual digest) — tamper
evidence without raw file content.

---

## 5. Serving policy layer

Implements `research/selective_risk_layer_v1.json` (`nanojev-selective-risk-layer-v1`) as the
shipped policy contract. The layer remains **advisory measurement**: it routes, abstains and
reports; it does not re-calibrate probabilities and does not authorize anything.

### 5.1 Modes

| Mode | Semantics |
|---|---|
| `shadow` (**default**, the only mode a fresh install may serve) | Full prediction + gates + receipts run; every answer gets `status:"shadow"`, `presented_value=null`. Caller sees distributions only. Explicitly: **evaluate but never act, and never even present an acted value.** |
| `advisory` | Answers surface with `in_scope_advisory`/`abstained`/`out_of_scope` statuses. Requires a non-placeholder scope detector **and** a `receipts/scope_eval.json` in the pack (§5.3). `value`/`choice` are still advisory; `authorizes_execution` stays `false`. |

There is intentionally **no `act`/`execute` mode**. Routing to a stronger model or a human is
always the caller's job (§5.2).

### 5.2 Abstention → caller routing

Per `abstention_semantics`: `confidence < abstain_below` → `status="abstained"`,
`value=null`, `choice=null`, `suggested_value` preserves the argmax. The routing target is
`caller_or_main_model`: the SDK returns the refusal, the caller decides the escalation
(main model, rules, human). The SDK itself **never silently answers** below threshold and
never substitutes a fallback model — there is no fallback model (`no teacher fallback`,
existing `serve_decisions.py` error contract).

### 5.3 Scope gate — first-class interface (J-C prerequisite)

`policy/scope_gate.json` declares the pack's OOD/scope detector:

```json
{"schema_version": "nanojev-scope-gate-policy-v1",
 "detector": "nanojev-scope-guard-v1 | <learned-detector-id> | placeholder_none",
 "status": "implemented | placeholder_interface_not_implemented",
 "interface": {"input": "question record (state text, question type, candidate set)",
               "output": {"out_of_scope": "bool", "score": "float|null", "reason": "string|null"}},
 "routing": "out_of_scope=true routes to caller_or_main_model identically to confidence abstention",
 "eval_receipt": "receipts/scope_eval.json | null"}
```

Rules:

- The interface is exactly the `ood_detector` block of `selective_risk_layer_v1.json`. A
  learned detector ships inside the pack: its artifacts are listed in `manifest.artifacts`
  and hash-pinned like weights.
- `detector="placeholder_none"` or `status="placeholder_interface_not_implemented"` → the pack
  may serve **shadow only**. `Decider(mode="advisory")` raises at construction.
- The existing lexical guard (`nanojev-scope-guard-v1`, `assess_question_scope` in
  `nanojev_skill.py`) is a legitimate declared detector — it is deterministic, auditable and
  has measured on-domain behaviour — but the manifest must still carry a `scope_eval.json`
  receipt describing its measured envelope, and the manifest must state honestly that it is a
  lexical heuristic, not a learned OOD detector.
- Scope refusal is **presentation-only**: probabilities, confidence and `abstained` are
  recorded unchanged; only `status`/`presented_value` change (existing skill semantics).
- Per-question `abstain_below` in the request remains supported (validated `[0,1]`), as in the
  installed skill.

### 5.4 Protected-error reporting

`protected_error_policy` from the layer spec is carried into serving: every eval receipt and
the §7 aggregate report count wrong-argmax answers at `confidence ≥ 0.9`
(`protected_error_gate`), per question type, with `limit: 0` — a non-zero count on-domain is a
serving blocker (§8 G4), and on OOD cohorts it is reported, never hidden.

---

## 6. Runtime budget spec

### 6.1 Memory budgets

- `manifest.runtime_budget.peak_rss_bytes` is declared **at pack build time** from a measured
  probe (e.g. N1 smoke measured ~5.27 GB peak RSS for the current workload class on MPS). A
  pack without measurement declares `{"peak_rss_bytes": null, "status":"unmeasured"}` and may
  only serve shadow.
- `request_limits` (defaults inherited from `serve_decisions.py`): `max_states=32`,
  `max_questions=96`, `max_candidate_paths=256` per request; `max_length` per candidate path.
  Exceeding any is a `400`, never truncation.
- Runtime measures actual RSS after load and per request; sustained exceedance of the
  declared budget is a health-reportable condition, not a silent one.

### 6.2 Concurrency model

- **v1: single-worker serialized inference.** At most one forward batch in flight per
  `Decider`/service process (the model is not re-entrant; concurrent MPS forwards are
  explicitly out of scope).
- Requests beyond the in-flight slot enter a bounded FIFO queue (default depth 8; deeper →
  `429`/`QueueFull`, never unbounded growth).
- Parallelism across questions inside one request is the existing `batch_questions` /
  shared-prefix machinery — internal batching, not concurrent requests.

### 6.3 Cancellation and timeout

- Every request carries `deadline_ms` covering **queue wait + compute** (the skill's per-call
  `timeout` convention). Deadline exceeded → error, and **no partial answers** — matching the
  predictor's existing rule that non-finite or incomplete results are never returned
  ("未返回部分预测").
- `Decider.cancel(request_id)` is cooperative: checked between `complete_question_batches`
  batches / packed-row chunks; an in-flight forward is not preempted mid-kernel. Cancelled
  work releases its queue slot; the cancellation is recorded in the event receipt.

### 6.4 Early-exit / accelerated-path policy

Any computation shortcut that changes the executed graph — shared-prefix packing (X3),
candidate pruning, layer early-exit, KV tricks — is governed by one rule, already proven by
X3: **opt-in, off by default, gated by a recorded parity receipt.**

- `shared_prefix` remains the reference implementation of this policy: `parity_probe.json`
  must show max `|Δp| ≤ 1e-5` and zero argmax flips on a frozen probe set that includes a
  255-candidate question (the X3 gate: max Δp 7.75e-7, 77 questions).
- A future layer early-exit mechanism ships disabled; enabling requires the same parity
  receipt *for that pack*, and the receipt is listed in `manifest.artifacts`.
- Layer-type guard preserved: packed/exotic paths fall back to independent forward when the
  backbone is not `full_attention` throughout (existing `_shared_prefix_logits` check).

### 6.5 CPU/MPS dispatch

- `device="auto"`: `cuda` (bf16 if supported) → `mps` → `cpu`, identical to
  `resolve_runtime`. **Reference target is MPS/FP32**; MPS and CPU paths are FP32-fixed.
- **Adaptive dispatch rule:** when MPS is unavailable or its residency would push the process
  over the declared memory budget (e.g. a concurrent training process occupies the GPU), the
  runtime dispatches CPU **deterministically** and records `device_fallback` + the reason in
  the event receipt and `execution.device`. Device choice is never silent.
- Device is process/pack level, not per-request; a request may not force a device that
  violates the pack's declared `device_policy`.

### 6.6 Warm/cold/queue measurement

Every receipt separates:

- `cold`: first-call latency including model load and first forward (persistent-load model —
  load happens once at construction, so cold = load + first predict, reported separately);
- `warm`: steady-state predict latency;
- `queue_ms` vs `compute_ms` per request (§2.2 `latency_ms`).

Aggregate reports give **p50/p95/p99** for warm and cold separately (the
`nanojev-usage-summary-v1` percentile convention, extended to p99). Warmup calls are counted
but flagged. Receipts must state device, precision, `prefix_sharing`, `max_length`, and
request-size profile alongside numbers — a latency without a workload description is not a
measurement.

---

## 7. Receipts / telemetry

Event schema `nanojev-decision-event-v1` — a superset of the installed `nanojev-usage-v1`
(`references/usage-schema.md`), append-only JSONL (default `~/.nanojev/usage.jsonl`):

```json
{"event_type": "decision", "event_id": "uuid", "timestamp": "utc",
 "schema_version": "nanojev-decision-event-v1",
 "source": "codex|sdk|http", "task_tag": "…",
 "pack": {"pack_id": "…", "pack_version": "…", "manifest_sha256": "…",
          "checkpoint": {"directory": "…", "config_sha256": "…"}},
 "mode": "shadow",
 "authorizes_execution": false,
 "request": {"state_count": 1, "question_count": 1,
             "question_types": ["choice"],
             "candidate_count_min": 4, "candidate_count_max": 4,
             "input_sha256": "…"},
 "result": {"latency_ms": {"queue": 1.2, "compute": 1180.4, "total": 1181.6},
            "confidence_min": 0.6, "confidence_max": 0.6, "confidence_mean": 0.6,
            "abstained_count": 1, "out_of_scope_count": 0,
            "status_counts": {"abstained": 1},
            "scope_policy": "nanojev-scope-guard-v1",
            "device": "mps", "device_fallback": null, "precision": "fp32",
            "forward_passes": 1, "candidate_paths": 4,
            "network_model_calls": 0, "queue_depth": 0,
            "inference_call_index": 7}}
```

Privacy rules (unchanged from the installed schema, restated):

- **No raw `state` text, `instructions`, or `criteria` in any event.** The request identity is
  `input_sha256` over the canonical payload — a fingerprint for dedup/replay, not content.
- `task_tag`/`source` are integration labels, never user text.
- The **only** escape hatch is the existing debug flag `NANOJEV_LOG_PAYLOADS=1`, which adds
  `raw_payload` for a deliberate local debugging session; it is documented as dangerous and
  must never carry credentials or sensitive data.
- Feedback events (`correct|incorrect|abstained|fallback|human_override`, keyed by
  `decision_event_id`) are unchanged; aggregate summaries (`nanojev-usage-summary-v1`
  extended with p99 + queue split) derive from events only.
- Pack lifecycle events live in `activations.jsonl` (§4.5), separate from decision events.

---

## 8. Acceptance gates

All gates must hold **before any deployment claim**; each produces a receipt artifact. Until
then the pack `status` stays `candidate` and mode stays `shadow`.

| Gate | Requirement | Evidence |
|---|---|---|
| G1 frozen candidate | Pack built from a checkpoint whose evaluation receipt is frozen (J-E1/frozen-candidate dependency; e.g. `checkpoints/domain_adaptation_v4_lora_seed18` class artifact) | `provenance` + frozen eval receipt |
| G2 scope signal | A non-placeholder scope detector implemented **and** measured on an OOD cohort (J-C prerequisite; lexical guard acceptable only with its measured envelope stated honestly) | `receipts/scope_eval.json` |
| G3 parity | Every enabled accelerated path passes parity vs independent forward: max `|Δp| ≤ 1e-5`, zero argmax flips, frozen probe set incl. a 255-candidate case | `receipts/parity_probe.json` |
| G4 selective-risk | Operating points validated **on-domain**: protected errors @0.9 = 0 on the held-out domain cohort; unreachable targets reported as unreachable; transfer table included verbatim | `receipts/selective_risk_report.json` |
| G5 rollback drill | activate v → v+1 → rollback → probe predictions identical to pre-upgrade v | `activations.jsonl` + drill receipt |
| G6 kill-switch drill | config-flag kill and `KILL`-file kill each refuse service (503/raise) and restore only on deliberate clear | drill receipt |
| G7 offline drill | full load→predict→receipt under network isolation; `network_model_calls=0` | drill receipt |
| G8 tamper drill | single-byte corruption of any artifact → `PackVerificationError`, no partial load | drill receipt |
| G9 shadow-default | fresh install serves shadow; any mode change writes an audit event | `activations.jsonl` |
| G10 budgets | cold/warm p50/p95/p99 + queue/compute split + peak RSS measured and declared on the target device class | `receipts/budget_probe.json` |

A deployment claim made without all ten receipts is non-conformant by definition — this
mirrors the repo rule that a clean preflight is not authorization.

---

## 9. Explicit non-goals

1. **No active context removal / production pruning.** Not authorized by AGENTS.md; out of
   scope for this layer entirely.
2. **No provider calls, no remote fallback.** There is no network path to any model or
   service; abstention routes to the *caller*, not to a provider inside the SDK.
3. **No chat or generation.** The model scores bounded candidate sets; `autoregressive_decode_steps`
   is 0 by contract. No free-text output exists to leak.
4. **No autonomous authorization.** `authorizes_execution=false` in every schema; the SDK is
   a measurement/advisory component, never a permission system, approver or gate-bypasser.
5. **No re-calibration at serving time.** Probabilities pass through at T=1.0; fitted
   temperatures and thresholds live in the calibration spec and are reported as measured
   quantities, never silently applied (J-C/T8d evidence: rescaling cannot reorder confidence).
6. **No runtime pack distribution.** Packs arrive on the filesystem before load; fetching,
   verifying-remote, or auto-update channels are out of scope (open question, §10).
7. **No concurrency beyond the serialized model.** v1 does not promise multi-worker or
   multi-device inference; queuing is a fairness/budget mechanism, not throughput scaling.
8. **No truncation, no partial results.** Over-budget inputs and timed-out requests are
   errors with receipts, never silently clipped answers.

---

## 10. Open design questions (for owner)

1. **Pack authenticity vs integrity.** sha256 pinning proves bytes match a known-good
   reference, not that the reference is trustworthy. Is a signing/trust-root mechanism
   (e.g. owner-signed `manifest_sha256` allowlist) wanted in v1, or is filesystem provenance
   sufficient for the single-operator local model?
2. **Advisory-mode bar.** §5.3 lets the existing lexical guard qualify for `advisory` only
   with a measured `scope_eval.json`. Should advisory instead be hard-blocked until a learned
   OOD detector exists (stricter reading of the J-C verdict)?
3. **MPS memory accounting.** Unified memory makes `peak_rss_bytes` an imperfect MPS budget
   signal; is a Metal-heap-based measure (or a conservative RSS cap) the declared metric for
   G10?
4. **Version ordering.** `pack_version` ordering semantics (semver vs date-based `vN`
   matching the repo's `_vN` convention) need a single rule before rollback can order N-1.
5. **Early-exit scope.** §6.4 treats "early-exit" as any graph-changing shortcut under the
   X3 parity rule. If J-P intends a specific mechanism (e.g. per-layer confidence exits), its
   parity protocol needs a dedicated section.
6. **HTTP service hardening.** The `http.server`-based service is single-process; whether the
   packaged service keeps it (simplest, matches current) or adopts a queued worker front is
   an implementation decision the spec leaves open as long as §6 semantics hold.
7. **Registry location.** `~/.nanojev` vs the skill's `~/.codex/nanojev` — unify or keep
   separate with explicit precedence?

---

*Prepared as the J-P design deliverable. Implementation, drills and gate receipts are
separate roadmap work; this document itself authorizes none of them.*
