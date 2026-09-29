#!/usr/bin/env bash
# Idempotent starter for the local NanoJev stack.
# Safe to run any time: it only starts what is not already answering.
# After it finishes it prints the health-check receipt.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${NANOJEV_STACK_LOG_DIR:-/tmp}"

# 1. NanoJev lifecycle service (auto-selects a free port in 8876-8890).
if ! curl -fsS --max-time 2 http://127.0.0.1:8876/api/health >/dev/null 2>&1; then
  echo "[stack] starting NanoJev lifecycle service"
  /Users/markus/.codex/skills/nanojev-local-decider/scripts/nanojev_skill.py health --start >/dev/null
fi

# 2. Winnow-12B Q8 scorer on 8091.
if ! curl -fsS --max-time 2 http://127.0.0.1:8091/health >/dev/null 2>&1; then
  echo "[stack] starting winnow-server on 8091 (model load takes ~5-30s)"
  nohup python3 "$ROOT/external/winnow-inference/scripts/serve.py" \
    --model "$ROOT/external/models/Winnow-12B/Winnow-12B-Q8_0.gguf" \
    --context 8192 --decision-parallel 1 --chat-parallel 1 \
    --memory exclusive > "$LOG_DIR/winnow-server.log" 2>&1 &
fi

# 3. Kev-4B fast-path scorer on 8092.
if ! curl -fsS --max-time 2 http://127.0.0.1:8092/v1/models >/dev/null 2>&1; then
  echo "[stack] starting kev-4b on 8092 (model load takes ~10-60s)"
  (cd "$ROOT/external/kev" && nohup uv run --extra serve \
    python -m kev.serve --run jaredpalmer/kev-4b --port 8092 \
    > "$LOG_DIR/kev-serve.log" 2>&1 &)
fi

PYTHONPATH="$ROOT/scripts" python3 "$ROOT/scripts/check_local_services_v1.py" "$@"
