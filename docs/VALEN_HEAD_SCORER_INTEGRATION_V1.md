# Valen head scorer backend — integration design (T156, prototype only)

Status: prototype. `scripts/valen_head_scorer_v1.py` loads
`external/valen/output/nano_rlcd_v1/latest` (RLCD decision head on a frozen
Qwen3.5-0.8B backbone, head-only checkpoint: `head.decision`/`head.candidate`
256×1024) and exposes `score(record)` with the same input contract that
`serve_decisions.py`/`nanojev-eval` forward to winnow: a `{"state",
"questions"}` request body (or a full eval-format record carrying `request`).
Output mirrors `/v1/systemone`: an `answers` map (`{"type": "noul", "noul":
p}`) plus a `decision` summary (`keep = 1 - noul`, `drop = noul` for the
`irrelevant` question). Load-once singleton via `get_scorer()`.

## Why a sidecar, not in-process

`serve_decisions.py` is a single-threaded `HTTPServer`: one blocking call
stalls every endpoint (`/v1/systemone`, `/v1/context-gate`, `/api/evaluate`,
static). A valen-head forward on MPS measures ~0.3–0.8 s per candidate
question (see `results/valen_head_scorer_v1_eval.json`); inside the service
that would serialize with — and delay — all other traffic, including winnow
proxying. Two further blockers:

- **Interpreter/venv split.** The service runs the main `.venv`; the head
  needs the valen venv (transformers 5.4 with `Qwen3_5ForConditionalGeneration`,
  peft). Merging environments risks breaking the production service.
- **Memory.** ~1.6 GB bf16 backbone resident next to the existing checkpoint
  engine; a crash or OOM in the head would take down the whole service.

Recommendation: **sidecar process** — a tiny loopback HTTP server (e.g.
`127.0.0.1:8093`) running this module under `external/valen/.venv`, exposing
`POST /v1/systemone` that accepts `{state, questions}` and returns
`{model, answers, usage}`. Then integration into `serve_decisions.py` is a
one-line change: add `"valen": ("127.0.0.1", 8093, MODEL_ID)` to
`SCORER_BACKENDS`. `systemone_route`, `context_gate_eval`'s `SystemOneHTTPScorer`
path, `/api/health` probing, and `nanojev-eval --backend valen` all work
unchanged — the service already treats scorers as probed, never-started
loopback backends. `start_local_stack.sh` gains a fourth stanza that launches
the sidecar (model load ~20–40 s on MPS) when 8093 is not answering.

Alternative considered — lazy in-worker load: feasible only after the service
moves to a threaded server *and* the venv split is resolved; still leaves the
head's forward serialized behind the request loop. Sidecar wins on isolation,
venv separation, and zero changes to the request path beyond the backend table.

## Contract notes

- Question semantics preserved verbatim: `irrelevant`/`noul`,
  `true == certainly irrelevant == drop`.
- `state` may be a JSON string (eval records serialize the conversation state)
  or a messages object — the valen compiler handles both, matching training.
- No truncation: requests exceeding `max_length` (8192 tokens) raise — the
  sidecar should map that to a 4xx so the gate fails open.
- The head is advisory/shadow like every local scorer; thresholding stays in
  `context_gate_v1`, not in the scorer.
