#!/usr/bin/env bash
# l40_v5_bringup_v1.sh — v5 training session on L40x2.
# Run AFTER ssh key auth works (see docs/L40_MIGRATION_V1.md §1).
# Usage: bash scripts/l40_v5_bringup_v1.sh <PORT>
# Assumes: host 120.209.70.195, key ~/.ssh/nanojev_gpu_ed25519,
# remote workspace /root/gpufree-data/nanojev persists from W139.
set -euo pipefail
PORT="${1:-30335}"
HOST=120.209.70.195
KEY=~/.ssh/nanojev_gpu_ed25519
SSH="ssh -i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -p $PORT root@$HOST"
RSYNC="rsync -az -e ssh -i $KEY -o IdentitiesOnly=yes -p $PORT"
REMOTE=/root/gpufree-data/nanojev

echo "== 0. connectivity + persistence check"
$SSH "nvidia-smi --query-gpu=name --format=csv,noheader | head -2 && \
      ls $REMOTE/valen-venv/bin/python $REMOTE/valen/models/Qwen3.5-0.8B 2>&1 | head -3 && \
      df -h /root/gpufree-data | tail -1"

echo "== 1. v5 data"
$SSH "mkdir -p $REMOTE/data/valen_nano_v5"
rsync -az -e "ssh -i $KEY -o IdentitiesOnly=yes -p $PORT" \
  data/valen_nano_v5/{train,eval,dev}.jsonl data/valen_nano_v5/manifest.json \
  root@$HOST:$REMOTE/data/valen_nano_v5/

echo "== 2. v5 configs + any code drift"
rsync -az -e "ssh -i $KEY -o IdentitiesOnly=yes -p $PORT" \
  external/valen/configs/ root@$HOST:$REMOTE/valen/configs/
rsync -az -e "ssh -i $KEY -o IdentitiesOnly=yes -p $PORT" \
  --exclude output --exclude .venv --exclude models --exclude __pycache__ \
  external/valen/valen/ root@$HOST:$REMOTE/valen/valen/

echo "== 3. remote env sanity (fla fast path + parity preconditions)"
$SSH "cd $REMOTE/valen && source $REMOTE/valen-venv/bin/activate && \
      python -c 'import torch;print(torch.cuda.device_count(),torch.cuda.get_device_name(0))' && \
      python -c 'import fla' 2>&1 | tail -1"

echo "== 4. launch (tmux, dual-GPU)"
$SSH "cd $REMOTE/valen && source $REMOTE/valen-venv/bin/activate && \
      mkdir -p logs && \
      tmux new-session -d -s v5sft  'CUDA_VISIBLE_DEVICES=0 python -m valen.train --config configs/cuda/sft_nano_v5_cuda.json 2>&1 | tee logs/sft_v5.log' && \
      tmux new-session -d -s v5lora 'CUDA_VISIBLE_DEVICES=1 python -m valen.train --config configs/gpu/sft_text_nano_v5_lora.json --initialize output/nano_sft_v4/latest 2>&1 | tee logs/lora_v5.log' && \
      tmux ls"
echo "== launched. monitor: $SSH 'tail -20 $REMOTE/valen/logs/*.log'"
