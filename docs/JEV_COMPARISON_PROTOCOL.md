# NanoJev and official Jev comparison protocol

Use identical frozen requests for both systems. Do not compare NanoJev's local latency against selected Jev marketing examples or compare different task cohorts.

## Required cohorts

1. Public `local_maze_v1` test and OOD questions for atomic Boolean decisions.
2. The same fixed composed-maze episodes for controller-level outcomes.
3. A general structured-decision cohort covering Boolean/Noul, Choice, Score, candidate counts, long inputs, and paraphrases.

## Metrics

- task quality: accuracy, NLL, Brier score, ECE, and OOD degradation
- contract quality: schema validity, missing candidates, non-finite values, and probability-sum failures
- runtime: end-to-end latency p50/p95/p99, throughput, cold start, and concurrency
- economics: measured request cost and input size, reported separately from local hardware cost
- robustness: candidate permutation, irrelevant-context injection, paraphrases, and candidate counts up to 255
- workflow outcomes: completion rate, collisions, steps, and path efficiency

Report local model-loading time separately from warm inference. Keep network latency in Jev measurements and exclude it only in a separately labeled provider-compute experiment if the provider exposes such timing.

## Existing entry points

The repository already supports both engines for composed evaluations:

```bash
PYTHONPATH=scripts .venv/bin/python scripts/evaluate_composed_maze.py \
  --episodes /path/to/frozen_episodes.jsonl \
  --engine checkpoint \
  --checkpoint /path/to/local_atomic_seed17 \
  --precision auto \
  --output results/nanojev_maze.json

PYTHONPATH=scripts .venv/bin/python scripts/evaluate_composed_maze.py \
  --episodes /path/to/frozen_episodes.jsonl \
  --engine jev \
  --env-file .env \
  --journal-dir results/jev_journal \
  --budget-usd 1 \
  --output results/jev_maze.json
```

The Jev path requires an `AI_GATEWAY_API_KEY` with access to `typesafe-ai/jev`. Never commit `.env`, request journals containing sensitive inputs, or provider credentials.

## Interpretation

Type-safe output means malformed schema output can be prevented by construction; it does not guarantee semantic correctness or calibrated probabilities. Jev is proprietary, so this protocol compares observable API behavior and outcomes, not architecture or training equivalence.
