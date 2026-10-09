#!/usr/bin/env bash
# T9d-v3 two-arm sequential runner — amendment research/nanojev_v2_t9d_v3_amendment_v1.json
# Arm A (headonly): scripts/train_pipeline_decisions_v3.py --freeze-backbone --backbone-lr 0
# Arm B (full):     scripts/train_pipeline_decisions.py --backbone-lr 2e-5 (frozen v2 recipe)
# Serialized on MPS: each heldout predict runs only after its training run exits.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
INPUT=research/engineering_judgment_corpus_v3/trainer_view
INIT=checkpoints/local_atomic_seed17/variants/local_atomic_seed17
HELDOUT=research/engineering_heldout_v1/predict_input.json

run_seed() { # arm_label trainer_script extra_args... -- seed
  local arm="$1"; local trainer="$2"; local seed="$3"; shift 3
  local out="checkpoints/domain_adaptation_v3_${arm}_seed${seed}"
  echo "=== arm ${arm} seed ${seed} train start $(date +%H:%M:%S) ==="
  "$PY" "scripts/${trainer}" \
    --input "$INPUT" --output-dir "$out" \
    --init-checkpoint "$INIT" \
    --objective gold_distribution --loss ce \
    --steps 300 --head-steps 0 --eval-every 50 \
    --batch-questions 16 --microbatch-questions 4 --max-microbatch-tokens 16384 \
    --max-length 2048 --head-lr 2e-4 \
    --device mps --precision fp32 --seed "$seed" "$@"
  local rc=$?
  echo "=== arm ${arm} seed ${seed} train exit ${rc} $(date +%H:%M:%S) ==="
  if [ $rc -eq 0 ] && [ -f "$out/best.safetensors" ]; then
    echo "=== arm ${arm} seed ${seed} heldout predict start $(date +%H:%M:%S) ==="
    "$PY" scripts/predict_toy_decisions.py \
      --checkpoint-dir "$out" --input "$HELDOUT" \
      --output "results/heldout_v1_post_v3_${arm}_seed${seed}.json" \
      --device mps --precision fp32 --batch-questions 8 --max-length 2048
    echo "=== arm ${arm} seed ${seed} heldout predict exit $? $(date +%H:%M:%S) ==="
  else
    echo "=== arm ${arm} seed ${seed} SKIP heldout (train rc=${rc} or no best.safetensors) ==="
  fi
}

for seed in 17 18 19; do
  run_seed headonly train_pipeline_decisions_v3.py "$seed" --freeze-backbone --backbone-lr 0
done
for seed in 17 18 19; do
  run_seed full train_pipeline_decisions.py "$seed" --backbone-lr 2e-5
done
echo "=== ALL RUNS COMPLETE $(date +%H:%M:%S) ==="
