#!/usr/bin/env bash
# colab_bringup_v1.sh — Colab L4 bringup + train + retrieve (replaces L40 workflow)
# Usage: bash scripts/colab_bringup_v1.sh <DATASET_DIR> <REMOTE_CONFIG> [RUN_NAME]
#   e.g. bash scripts/colab_bringup_v1.sh data/valen_nano_v9 configs/cuda/sft_nano_v9_cuda.json v9sft
# Prereqs: colab CLI authenticated; research/remote_archive/l40_notes/causal-conv1d-1.7.0-sm89/
# Env is rebuilt per session (VM is ephemeral — see docs/COLAB_WORKFLOW_V1.md).
set -euo pipefail
# colab.dev 长连接/大传输走 星岛梦(7894) 会 SSL EOF 断流；
# 龙猫云(7892, SG) 通畅但额度会耗尽 — 默认 7892，可用 COLAB_PROXY 覆盖。
COLAB_PROXY="${COLAB_PROXY:-http://127.0.0.1:7892}"
export https_proxy="$COLAB_PROXY" HTTPS_PROXY="$COLAB_PROXY" \
       http_proxy="$COLAB_PROXY" HTTP_PROXY="$COLAB_PROXY" \
       all_proxy="$COLAB_PROXY" ALL_PROXY="$COLAB_PROXY"
DATASET="${1:?dataset dir, e.g. data/valen_nano_v9}"
CONFIG="${2:?remote config path relative to valen/, e.g. configs/cuda/sft_nano_v9_cuda.json}"
RUN="${3:-colab_run}"
SESSION="nanojev_${RUN}"
R=/content/nanojev
PULL_EVERY="${PULL_EVERY:-300}"   # seconds between checkpoint polls
KEEP="${KEEP:-0}"                 # KEEP=1 leaves session running at end

phase() { echo; echo "########## $* ##########"; }

phase "0. preflight"
command -v colab >/dev/null || { echo "colab CLI missing"; exit 1; }
colab usage | tail -3
test -f scripts/colab_requirements_v1.txt
test -f scripts/colab_remote_setup_v1.sh
test -d research/remote_archive/l40_notes/causal-conv1d-1.7.0-sm89 || \
  { echo "patched conv1d source missing (expected research/remote_archive/l40_notes/)"; exit 1; }

phase "1. build upload bundles"
B=$(mktemp -d)
# env bundle: reqs + setup + slim conv1d source (no build/ artifacts)
cp scripts/colab_requirements_v1.txt scripts/colab_remote_setup_v1.sh "$B"/
tar czf "$B/conv1d_sm89_src.tgz" -C research/remote_archive/l40_notes \
  --exclude build --exclude '*.o' --exclude '*.so' causal-conv1d-1.7.0-sm89
# code bundle: explicit include list (exclude-tar walks 7G models dir on bsdtar)
tar czf "$B/valen_code.tgz" -C external/valen \
  --exclude '._*' --exclude '__pycache__' --exclude '.pytest_cache' \
  valen configs evaluation scripts tests docs assets data \
  pyproject.toml README.md README_zh.md LICENSE .gitignore \
  MPS_PATCH_NOTES.md NANOJEV_REPRO_NOTES.md env.sh 2>/dev/null || \
tar czf "$B/valen_code.tgz" -C external/valen \
  --exclude '._*' --exclude '__pycache__' --exclude '.pytest_cache' \
  valen configs evaluation scripts tests docs assets data pyproject.toml
# data bundle — STAGE_DRIVE=1 routes large data via Drive instead of proxy upload
if [ "${STAGE_DRIVE:-0}" = "1" ]; then
  bash scripts/drive_sync_v1.sh stage-data "$DATASET"
  echo "$(basename "$DATASET")" > "$B/stage_data.txt"
  cp ~/.config/rclone/rclone.conf "$B/rclone.conf"   # credential: upload-only, never committed
else
  tar czf "$B/data.tgz" -C data "$(basename "$DATASET")"
fi
ls -lh "$B"

phase "2. colab new --gpu L4"
colab new -s "$SESSION" --gpu L4 || { echo "L4 unavailable — try --gpu T4 or retry later"; exit 1; }
trap 'echo "!! aborted; session $SESSION left running (colab stop -s $SESSION to clean)"' ERR

phase "3. upload + remote env setup (detached)"
echo "import pathlib; [pathlib.Path(p).mkdir(parents=True,exist_ok=True) for p in ['$R','$R/data','$R/valen']]; print('dirs ok')" | colab exec -s "$SESSION"
for f in colab_requirements_v1.txt colab_remote_setup_v1.sh conv1d_sm89_src.tgz valen_code.tgz data.tgz rclone.conf stage_data.txt; do
  [ -f "$B/$f" ] && colab upload -s "$SESSION" "$B/$f" "$R/$f"
done
echo "extract code+data on VM"
echo "import subprocess; subprocess.run(['bash','-c','''
mkdir -p $R/data $R/valen
cd $R && tar xzf valen_code.tgz -C $R
[ -f data.tgz ] && tar xzf data.tgz -C $R/data
rm -f $R/.setup_done $R/.setup_fail
nohup bash $R/colab_remote_setup_v1.sh > /dev/null 2>&1 &
print(\"setup launched\")'''],check=True)" | colab exec -s "$SESSION"

phase "4. poll env setup (~10-20min)"
for i in $(seq 1 120); do
  S=$(echo "import pathlib
p=pathlib.Path('$R')
if (p/'.setup_done').exists(): print('DONE')
elif (p/'.setup_fail').exists(): print('FAIL')
else:
  t=(p/'setup.log'); print('RUNNING', t.read_text().count(chr(10)) if t.exists() else 0)" \
    | colab exec -s "$SESSION" | tail -1)
  echo "  setup: $S"
  case "$S" in
    DONE*) break ;;
    FAIL*) echo "== setup FAILED; last log:"; echo "print(open('$R/setup.log').read()[-3000:])" | colab exec -s "$SESSION" | tail -40; exit 1 ;;
  esac
  sleep 15
done

phase "5. launch training (detached)"
# cwd=$R (repo root on VM): configs resolve relative data/ + models/ + output/ from there;
# PYTHONPATH covers the case where the editable install hasn't run yet.
echo "import subprocess
subprocess.run(['bash','-c','cd $R && mkdir -p logs && PYTHONPATH=$R nohup python -m valen.train --config $CONFIG > logs/$RUN.log 2>&1 & echo \$! > $R/train.pid'],check=True)
print('training launched')" | colab exec -s "$SESSION"

phase "6. monitor + stream checkpoints"
LOCAL_OUT="external/valen/output_colab/$RUN"
mkdir -p "$LOCAL_OUT"
LAST_SHA=""
while true; do
  ST=$(echo "import pathlib,subprocess,hashlib
p=pathlib.Path('$R')
log=(p/'logs/$RUN.log')
pid=(p/'train.pid')
import glob
cks=glob.glob(str(p/'output/*/latest/checkpoint.pt'))
sha=hashlib.sha256(open(cks[0],'rb').read()).hexdigest()[:16] if cks else 'none'
alive=subprocess.run(['bash','-c','kill -0 '+open(pid).read().strip()+' 2>/dev/null && echo ALIVE || echo DEAD'],capture_output=True,text=True).stdout.strip() if pid.exists() else 'NOPID'
tail=log.read_text().splitlines()[-1][:160] if log.exists() else 'no-log'
print(f'{alive}|{sha}|{tail}')" | colab exec -s "$SESSION" | tail -1)
  echo "  $ST"
  ALIVE=${ST%%|*}; REST=${ST#*|}; SHA=${REST%%|*}
  # pull checkpoint when it changed
  if [ "$SHA" != "$LAST_SHA" ] && [ "$SHA" != "none" ]; then
    LAST_SHA="$SHA"
    echo "import subprocess; subprocess.run(['bash','-c','cd $R && tar czf $R/ckpt_$RUN.tgz --exclude=\'._*\' output 2>/dev/null'],check=True)" | colab exec -s "$SESSION" >/dev/null 2>&1 || true
    colab download -s "$SESSION" "$R/ckpt_$RUN.tgz" "$LOCAL_OUT/ckpt_$RUN.tgz" 2>/dev/null && \
      tar xzf "$LOCAL_OUT/ckpt_$RUN.tgz" -C "$LOCAL_OUT" && echo "  checkpoint pulled ($SHA)"
  fi
  [ "$ALIVE" = "DEAD" ] && break
  [ "$ALIVE" = "NOPID" ] && { echo "pid missing — training ended or failed"; break; }
  sleep "$PULL_EVERY"
done

phase "7. final pull"
echo "import subprocess; subprocess.run(['bash','-c','cd $R && tar czf $R/final_$RUN.tgz output logs env_receipt.json setup.log 2>/dev/null'],check=True)" | colab exec -s "$SESSION"
colab download -s "$SESSION" "$R/final_$RUN.tgz" "$LOCAL_OUT/final_$RUN.tgz"
tar xzf "$LOCAL_OUT/final_$RUN.tgz" -C "$LOCAL_OUT"
echo "artifacts in $LOCAL_OUT"

phase "8. teardown"
if [ "$KEEP" = "1" ]; then
  echo "KEEP=1 — session $SESSION left running. Stop later: colab stop -s $SESSION"
else
  colab stop -s "$SESSION"
fi
trap - ERR
echo "DONE — post-steps: fp32 parity vs MPS + 4-layer eval + G1-G6 gates (per V5_ACCEPTANCE_RUNBOOK_V1)"
