# L40×2 Migration Runbook (2026-09-28)

A100 session archived. Target: dual L40-48G (¥4.56/h) replacing A100-80G (¥5.99/h).
Everything below is sufficient to rebuild the workspace from scratch — the old
instance can be released after this file exists. Local copies already hold all
irreplaceable artifacts (v3/v4 checkpoints, eval predictions, datasets, code).

## 1. SSH bootstrap

```bash
# one-time root password from provider console, then inject our key:
sshpass -p '<password>' ssh -o StrictHostKeyChecking=accept-new -p <PORT> root@<HOST> \
  "mkdir -p ~/.ssh && echo 'ssh-ed25519 <public-key>' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
# afterwards: ssh -i ~/.ssh/<ssh-key> -o IdentitiesOnly=yes -p <PORT> root@<HOST>
```

## 2. Workspace layout (keep the symlink convention)

```bash
mkdir -p /root/gpufree-data/nanojev
# configs reference the Mac path; mirror it with symlinks so configs need no edits:
mkdir -p /Users/markus/Documents/NanoJev
ln -s /root/gpufree-data/nanojev /Users/markus/Documents/NanoJev/external/valen   # adjust to actual layout used last session
ln -s /root/gpufree-data/nanojev/data /Users/markus/Documents/NanoJev/data
# last session: /Users/markus/Documents/NanoJev/data/<ds> each symlinked into gpufree-data/data/<ds>
```

## 3. Python env

- Python 3.12 venv at `/root/gpufree-data/nanojev/valen-venv`
- Reference env: `research/remote_archive/env_manifest_20260928.txt`
  (kernel 5.15, driver 580.126.09, torch 2.10.0+cu128, py3.12.14)
- Pinned package list: `research/remote_archive/pip_freeze_20260928.txt`
- **Critical**: `flash-linear-attention` + `causal-conv1d` — required for the
  Qwen3.5 fast path (~30× vs torch fallback). Re-verify compile on Ada sm89
  (L40); if kernels fail on sm89, fall back to torch path (works, ~30× slower)
  and flag before accepting training numbers.
- pypi.org reachable directly; **huggingface.co is NOT reachable** —
  use `HF_ENDPOINT=https://hf-mirror.com` for all hub operations.

## 4. Data + repo sync

```bash
mkdir -p /root/gpufree-data/nanojev/data/valen_nano_v4
scp -P <PORT> -i ~/.ssh/<ssh-key> -o IdentitiesOnly=yes \
  data/valen_nano_v4/{train,eval,dev}.jsonl data/valen_nano_v4/manifest.json \
  root@<HOST>:/root/gpufree-data/nanojev/data/valen_nano_v4/
# push the valen code dir (includes configs/cuda/*.json):
rsync -az -e "ssh -p <PORT> -i ~/.ssh/<ssh-key> -o IdentitiesOnly=yes" \
  --exclude output --exclude .venv --exclude models \
  external/valen/ root@<HOST>:/root/gpufree-data/nanojev/valen/
# base model weights (2GB):
rsync -az -e "ssh ..." external/valen/models/ root@<HOST>:/root/gpufree-data/nanojev/valen/models/
```

## 5. Run (dual-GPU — true parallelism, not shared)

```bash
cd /root/gpufree-data/nanojev/valen
source ../valen-venv/bin/activate
# GPU0: head-only SFT          GPU1: LoRA (init from prior converged head)
CUDA_VISIBLE_DEVICES=0 python -m valen.train --config configs/cuda/sft_nano_v3_cuda.json &
CUDA_VISIBLE_DEVICES=1 python -m valen.train --config configs/cuda/sft_text_nano_v3_lora.json --initialize output/nano_sft_v4/latest &
# when SFT finishes: RLCD on whichever GPU is free
CUDA_VISIBLE_DEVICES=0 python -m valen.train --config configs/cuda/rlcd_nano_v3_cuda.json --initialize output/nano_sft_v4/latest
# evals: parallel per-GPU
CUDA_VISIBLE_DEVICES=0 python -m valen.evaluate --checkpoint output/nano_sft_v4/latest --data .../eval.jsonl --output output/nano_sft_v4_v4eval_fp32 --device cuda
```

Configs already point at `valen_nano_v4` and write to `nano_*_v4` outputs.
**Note**: output dir names on the new session should be bumped to v5 if v4
checkpoints are retrained — do not overwrite pulled artifacts.

## 6. Mandatory after hardware change

- **fp32 parity retest**: L40 fp32 vs MPS fp32 flips/max-Δp — the A100↔MPS
  numbers (LoRA 0/2,950 flips) do NOT transfer to a new GPU model.
- Same-device-only rule for all benchmark tables.
- Verify `fla` fast path actually active (log line "fast path not available"
  = fallback; check throughput ≈19 sps head-only / ~5 sps LoRA as baseline).

## 7. JEMM redeploy (optional, when disk allows)

```bash
# needs ~55GB free on the data disk (expand to ~150GB when creating instance)
HF_ENDPOINT=https://hf-mirror.com python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("MaestroYan/JEMM"); snapshot_download("Qwen/Qwen3.8-27B")
EOF
pip install git+https://github.com/ypcypc/JEMM   # github reachable, pypi reachable
# bf16 needs ~54GB VRAM: single L40 (48G) insufficient → fp8 variant on one card,
# or tensor-parallel across both L40s if jemm.serve supports it (check --help)
python -m jemm.serve --adapter MaestroYan/JEMM --port 8790
# eval: scripts/eval_systemone_backend_v1.py --url http://127.0.0.1:8790/v1/systemone
# (already validated end-to-end on A100: 0.8165 on valen_nano_v4 eval, ~200ms/q)
```

## What is NOT backed up (by design — all recreatable)

- `hf-jemm/` 53GB weights → re-download via hf-mirror (~14 min)
- `valen-venv/` 8.3GB → rebuild from pip_freeze
- `models/Qwen3.5-0.8B` → local copy is canonical
- `data/` → pushed from local; local is canonical
- remote `output/` → v3/v4 already pulled (`output_v3_20260928.tgz`,
  `output_v4_20260928.tgz` extracted under `external/valen/output_v3|v4/`);
  older v1/v2 heads already live in local `external/valen/output/`

## 执行回执 (2026-09-28)

Migration executed same-day on `ssh -p <port> root@<gpu-host>` (key
`<ssh-key>` verified); A100-80G released after pull-verify
(cardless → full shutdown). Server-side details live in
`/root/gpufree-data/nanojev/INSTALL_NOTES.md` — that file is authoritative
for rebuilds (pip reinstall reverts the patches below).

**Applied verbatim**: SSH key bootstrap; symlink convention under
`/Users/markus/Documents/NanoJev`; venv `valen-venv` rebuilt from
`pip_freeze_20260928.txt`; data/code/base-model sync layout;
dual-GPU `CUDA_VISIBLE_DEVICES` split; `hf-mirror.com` for all HF ops
(huggingface.co unreachable confirmed again).

**Deviations encountered**:
- `rsync` unavailable on the fresh instance → all sync done via `tar`
  pipes over ssh (`tar czf - … | ssh … 'tar xzf -'`), same result.
- `pip freeze` restore stalled mid-install → completed with aliyun mirror
  (`-i https://mirrors.aliyun.com/pypi/simple/`).
- `causal-conv1d` had no sm89 (Ada) support → required a source patch
  before compile (procedure in INSTALL_NOTES); `fla` fast path verified
  active afterwards.
- NCCL cu12/cu13 version conflict surfaced on first multi-GPU use →
  fix recorded in INSTALL_NOTES.

**JEMM deploy result**: bf16 (~54GB) does not fit one L40 — deployed via
`device_map=auto` sharding across both cards, needing a 1-line patch at
`valen-venv/.../jemm/model.py:57` so `--device auto` is honored
(site-packages patch; reinstalling jemm reverts it). `jemm.serve` resident
on `:8790`, ~222 ms/question; v4 eval 4,251 questions reproduced the A100
result within noise (**0.8167 vs 0.8165**, 25 label flips). fp32 parity
retest vs Mac MPS done per §6 — flips: lora 0 / rlcd 109 / sft 165
(handoff W138/W139).
