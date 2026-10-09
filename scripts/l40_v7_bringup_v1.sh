#!/usr/bin/env bash
# l40_v7_bringup_v1.sh — reconnect, sync v7 data, launch v7 training, then
# pull + fp32 + score once done. Run: bash scripts/l40_v7_bringup_v1.sh
set -euo pipefail
SSH="ssh -i $HOME/.ssh/nanojev_gpu_ed25519 -o IdentitiesOnly=yes -o ConnectTimeout=15 -p 30335 root@120.209.70.195"
R=/root/gpufree-data/nanojev

echo "=== connectivity ==="
$SSH "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader"

echo "=== sync v7 data ==="
tar -C data/valen_nano_v7 -cf - . | $SSH "mkdir -p $R/data/valen_nano_v7 && tar -xf - -C $R/data/valen_nano_v7 && wc -l $R/data/valen_nano_v7/train.jsonl"

echo "=== launch v7 (GPU0) ==="
$SSH "cd $R/valen && source ../valen-venv/bin/activate && tmux kill-session -t v7 2>/dev/null; tmux new-session -d -s v7 'CUDA_VISIBLE_DEVICES=0 python -m valen.train --config configs/gpu/sft_text_v7.json --initialize output/nano_sft_v4/latest 2>&1 | tee logs/v7.log' && sleep 30 && tail -2 logs/v7.log | tail -c 200"

echo "v7 training launched. Monitor: ssh ... tail -1 logs/v7.log"
