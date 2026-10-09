#!/usr/bin/env bash
# v5_posttrain_acceptance_v1.sh — pull v5 checkpoints, serve on :8095,
# score all eval tiers, run G1-G6 gates. Run after remote training finishes.
# Usage: bash scripts/v5_posttrain_acceptance_v1.sh <PORT>
set -euo pipefail
PORT="${1:-30335}"
HOST=120.209.70.195
KEY=~/.ssh/nanojev_gpu_ed25519
SSH="ssh -i $KEY -o IdentitiesOnly=yes -p $PORT root@$HOST"
REMOTE=/root/gpufree-data/nanojev
OUTDIR=external/valen/output

echo "== 0. verify remote training done"
$SSH "tail -3 $REMOTE/valen/logs/lora_v5.log | head -3; ls $REMOTE/valen/output/nano_sft_text_v5/latest/"

echo "== 1. pull checkpoints (v5 only)"
mkdir -p $OUTDIR
$SSH "cd $REMOTE/valen && tar -czf - output/nano_sft_text_v5 output/nano_sft_v5 output/nano_rlcd_v5 2>/dev/null" \
  | tar -xzf - -C $OUTDIR --strip-components=0
ls $OUTDIR/nano_sft_text_v5/latest/

echo "== 2. hash manifest for pulled artifacts"
(cd $OUTDIR && find nano_sft_text_v5 nano_sft_v5 nano_rlcd_v5 -type f 2>/dev/null \
  | sort | xargs shasum -a 256) > research/v5_checkpoint_manifest.sha256 || true
wc -l research/v5_checkpoint_manifest.sha256

echo "== 3. deploy v5 LoRA head on :8095 (launchd/manual)"
echo "   run: scripts/serve_nanojev_v1-style command for checkpoint nano_sft_text_v5"
echo "   (keeps :8094 lora_v4 incumbent untouched)"

echo "== 4. score tiers into results/  (fill in after :8095 is up)"
cat <<'EOF'
# T1 frozen 590:
python3 scripts/eval_systemone_backend_v1.py --url http://127.0.0.1:8095/v1/systemone \
  --data data/real_context_eval_v1/eval.jsonl --output results/lora_v5_frozen_main.jsonl --timeout 60
python3 scripts/eval_systemone_backend_v1.py --url http://127.0.0.1:8095/v1/systemone \
  --data data/real_context_eval_v1/drop_supp/eval.jsonl --output results/lora_v5_frozen_supp.jsonl --timeout 60
# T2 v5-ext:
python3 scripts/eval_systemone_backend_v1.py --url http://127.0.0.1:8095/v1/systemone \
  --data data/real_context_eval_v5_ext/eval.jsonl --output results/lora_v5_v5ext.jsonl --timeout 60
# T3 synthetic (v4 synth + v5 synth eval):
python3 scripts/eval_systemone_backend_v1.py --url http://127.0.0.1:8095/v1/systemone \
  --data data/valen_nano_v4/eval.jsonl --output results/lora_v5_synth_v4.jsonl --timeout 60
python3 scripts/eval_systemone_backend_v1.py --url http://127.0.0.1:8095/v1/systemone \
  --data data/valen_nano_v5/eval.jsonl --output results/lora_v5_synth_v5.jsonl --timeout 60
EOF

echo "== 5. gates"
echo "   python3 scripts/check_v5_gates_v1.py --results-dir results"
