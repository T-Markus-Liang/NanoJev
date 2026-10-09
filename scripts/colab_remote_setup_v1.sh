#!/usr/bin/env bash
# colab_remote_setup_v1.sh — runs ON the Colab VM (uploaded by colab_bringup_v1.sh).
# Rebuilds the valen training environment per INSTALL_NOTES.md (L40 recipe, sm89).
# Usage: bash colab_remote_setup_v1.sh  (detached: writes setup.log + .done/.fail)
set -euo pipefail
R=/content/nanojev
LOG=$R/setup.log
exec > >(tee -a "$LOG") 2>&1
trap 'touch $R/.setup_fail' ERR
cd "$R"

echo "=== [1/6] system ==="
python3 --version
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader
nvcc --version | tail -1 || echo "no system nvcc (will use pip toolchain)"

echo "=== [2/6] pip deps ==="
PIP="python3 -m pip install"
command -v uv >/dev/null && PIP="uv pip install --system" || true
$PIP -r colab_requirements_v1.txt \
  --index-url https://pypi.org/simple \
  --extra-index-url https://mirrors.aliyun.com/pypi/simple/ || \
$PIP -r colab_requirements_v1.txt -i https://mirrors.aliyun.com/pypi/simple/

echo "=== [2.5/6] valen package (editable, no-deps — deps already pinned) ==="
# repo root = $R (pyproject.toml + valen/ package were extracted flat into $R)
python3 -m pip install -e "$R" --no-deps || uv pip install --system -e "$R" --no-deps || true

echo "=== [3/6] causal-conv1d sm89 build ==="
tar xzf conv1d_sm89_src.tgz
cd causal-conv1d-1.7.0-sm89
export FORCE_CUDA=1
if command -v nvcc >/dev/null; then
  export CUDA_HOME=/usr/local/cuda
else
  # pip toolchain fallback per INSTALL_NOTES: nvcc/crt/nvvm must match (13.0.x)
  python3 -m pip install --quiet nvidia-cuda-nvcc==13.0.88 nvidia-cuda-crt==13.0.88 \
    nvidia-nvvm==13.0.88 nvidia-cuda-cccl==13.0.85 nvidia-cuda-runtime==13.0.96
  SP=$(python3 -c "import site;print(site.getsitepackages()[0])")
  export CUDA_HOME=$SP/nvidia/cu13
  [ -f "$CUDA_HOME/lib/libcudart.so" ] || ln -sf libcudart.so.13 "$CUDA_HOME/lib/libcudart.so" || true
fi
TORCH_CUDA_ARCH_LIST="8.9" MAX_JOBS=8 python3 -m pip install --no-build-isolation --no-deps -v . \
  > "$R/cc1d_build.log" 2>&1 || { tail -40 "$R/cc1d_build.log"; exit 1; }
cd "$R"

echo "=== [4/6] verify fast path ==="
python3 - <<'PY'
import json, torch
out = {
  "torch": torch.__version__,
  "cuda_avail": torch.cuda.is_available(),
  "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
  "cap": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
}
try:
    from causal_conv1d import causal_conv1d_fn
    x = torch.randn(2, 64, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(64, 4, device="cuda", dtype=torch.bfloat16)
    causal_conv1d_fn(x=x, weight=w)
    out["causal_conv1d"] = "sm89_native_ok"
except Exception as e:
    out["causal_conv1d"] = f"FAIL: {e}"
try:
    import fla
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    out["fla"] = "ok"
except Exception as e:
    out["fla"] = f"FAIL: {e}"
print(json.dumps(out, indent=2))
assert out["cuda_avail"] and "ok" in out.get("causal_conv1d", ""), "fast path required"
PY

echo "=== [4.5/6] Drive staging (optional — only if rclone.conf was uploaded) ==="
if [ -f "$R/rclone.conf" ]; then
  if ! command -v rclone >/dev/null; then
    curl -fsSL https://rclone.org/install.sh | bash >/dev/null 2>&1 || true
  fi
  mkdir -p /root/.config/rclone && cp "$R/rclone.conf" /root/.config/rclone/rclone.conf
  if [ -f "$R/stage_data.txt" ]; then
    while read -r ds; do
      [ -n "$ds" ] && rclone copy "gdrive:nanojev-staging/data/$ds" "$R/data/$ds" --checksum || true
    done < "$R/stage_data.txt"
    echo "staged datasets pulled: $(cat "$R/stage_data.txt" | tr '\n' ' ')"
  fi
fi

echo "=== [5/6] workspace layout (Mac-path symlinks, zero config edits) ==="
mkdir -p "$R/data" /Users/markus/Documents
ln -sfn "$R" /Users/markus/Documents/NanoJev_workdir 2>/dev/null || true
# configs reference /Users/markus/Documents/NanoJev/data/... :
mkdir -p /Users/markus/Documents/NanoJev
ln -sfn "$R/data" /Users/markus/Documents/NanoJev/data
ln -sfn "$R/valen" /Users/markus/Documents/NanoJev/valen_link

echo "=== [6/6] base models (HF reachable natively; pinned revisions) ==="
python3 - <<'PY'
from huggingface_hub import snapshot_download
for repo, rev in [("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"),
                  ("Qwen/Qwen3.5-2B",  "15852e8c16360a2fea060d615a32b45270f8a8fc")]:
    name = repo.split("/")[-1]
    p = snapshot_download(repo, revision=rev)
    import os, shutil
    dst = f"/content/nanojev/models/{name}"  # configs use relative model_path from $R
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        shutil.copytree(p, dst)
    # re-assert manifest
    print("model ready:", dst)
PY

python3 - <<'PY'
import json, platform, subprocess
r = {
  "python": platform.python_version(),
  "pip_freeze": subprocess.run(["python3","-m","pip","freeze"],capture_output=True,text=True).stdout.splitlines(),
}
open("/content/nanojev/env_receipt.json","w").write(json.dumps(r,indent=2))
print("env receipt written")
PY
touch "$R/.setup_done"
echo "SETUP_COMPLETE"
