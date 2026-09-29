# A100 Next Session Runbook V1 (v4 data)

> **SUPERSEDED (2026-09-28)** — this plan was fully executed on the A100
> (W137), then the instance was released. The live GPU workspace is now
> dual L40-48GB; authoritative runbook + execution receipt:
> `docs/L40_MIGRATION_V1.md`. Kept for history only.

Prereq done: `data/valen_nano_v4` (17,124 train = v3 12,122 + v4 五族 5,002 / 4,251 eval 含新难族冻结行, label-contract fixed). Configs already in repo. Estimated total GPU time ~2–3 h on A100-80G.

## Sync

```bash
# from repo root on Mac
scp -P <port> -i ~/.ssh/<ssh-key> -o IdentitiesOnly=yes \
  data/valen_nano_v4/train.jsonl data/valen_nano_v4/eval.jsonl \
  data/valen_nano_v4/manifest.json \
  root@<gpu-host>:/root/gpufree-data/nanojev/data/valen_nano_v4/   # mkdir -p first
# configs are already in external/valen/configs/{cuda,gpu}/ — rsync/tar the repo dir if the checkout predates them
```

Data paths in configs use `/Users/markus/Documents/NanoJev/...` — the server has a symlink mirroring that prefix onto `/root/gpufree-data/nanojev` (same convention as the last session; keep it).

## Run order (env: `/root/gpufree-data/nanojev/valen-venv/bin/python`, `cd valen`)

```bash
# 1) head-only SFT on v3 — ~1 h
python -m valen.train --config configs/cuda/sft_nano_v3_cuda.json
# 2) LoRA text stage initialized from sft_v3 head — ~1-1.5 h (only stage never run on our data)
python -m valen.train --config configs/gpu/sft_text_nano_v3_lora.json \
  --initialize output/nano_sft_v4/latest
# 3) RLCD on sft_v3 — ~0.5 h
python -m valen.train --config configs/cuda/rlcd_nano_v3_cuda.json \
  --initialize output/nano_sft_v4/latest
# 4) eval each on the 2,950-question v3 eval (~1 min each)
for ck in nano_sft_v4 nano_sft_text_v4 nano_rlcd_v4; do
  python -m valen.evaluate --checkpoint output/$ck/latest \
    --data /Users/markus/Documents/NanoJev/data/valen_nano_v4/eval.jsonl \
    --output output/${ck}_eval --device cuda
done
```

## Rules for this session

- **Same-device comparisons only** — bf16 drift across A100↔MPS flipped 50/678 labels on identical weights; never mix devices in one table.
- v1 numbers (`0.9012` etc.) carry a label-noise caveat (≤~1.3pp); v3 eval is the clean benchmark going forward.
- No training on the local Mac (owner directive).
- Pull back `latest/` checkpoints + `metrics.json` for all runs; tarball the whole `output/` dir before shutdown (`tar czf output_backup.tgz output`).

## 硬件选型备忘（2026-09-28 owner 决策）

下次起机改用 **L40×2**（¥4.56/h，48GB/卡）替代 A100-80G（¥5.99/h）：
- 训练任务峰值仅 ~14GB 显存，A100 带宽/显存从未用上；双卡独占并行优于单卡共享
- JEMM 27B：单卡 fp8（~27G）或双卡 TP bf16；数据盘需扩到 ~150GB
- 换卡后必做：`fla` 内核在 sm89 上重验编译；fp32 parity 重测（L40↔MPS）
