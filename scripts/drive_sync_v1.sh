#!/usr/bin/env bash
# drive_sync_v1.sh — NanoJev ↔ Google Drive archive/staging layer
# Usage:
#   bash scripts/drive_sync_v1.sh push     # upload cold archives to Drive
#   bash scripts/drive_sync_v1.sh check    # verify remote vs local (rclone check)
#   bash scripts/drive_sync_v1.sh status   # show remote dir sizes
#   bash scripts/drive_sync_v1.sh pull <remote_subdir> <local_dir>
# Remote layout:  gdrive:nanojev-archive/{checkpoints,remote_archive,valen_tgz}
#                 gdrive:nanojev-staging/{data,ckpt}   (Colab hot-path staging)
# Proxy: Drive API 走龙猫云 7892（星岛梦长传会 SSL EOF，与 colab 同理）。
set -euo pipefail
# 代理出口：7892(龙猫云) 可能耗尽/断流；默认 7894(星岛梦)，可用 DRIVE_PROXY 覆盖
export https_proxy="${DRIVE_PROXY:-http://127.0.0.1:7894}" HTTPS_PROXY="${DRIVE_PROXY:-http://127.0.0.1:7894}" \
       http_proxy="${DRIVE_PROXY:-http://127.0.0.1:7894}" HTTP_PROXY="${DRIVE_PROXY:-http://127.0.0.1:7894}"
cd "$(dirname "$0")/.."
REMOTE_ARCHIVE=gdrive:nanojev-archive
REMOTE_STAGE=gdrive:nanojev-staging
FLAGS="--checksum --transfers=8 --checkers=8 --stats=30s --stats-one-line -q"

cmd="${1:-status}"
case "$cmd" in
  push)
    echo "== checkpoints (95G, local-only crown jewels) =="
    rclone copy checkpoints "$REMOTE_ARCHIVE/checkpoints" $FLAGS
    echo "== remote_archive (A100/L40 verified copies) =="
    rclone copy research/remote_archive "$REMOTE_ARCHIVE/remote_archive" $FLAGS
    echo "== valen tgz archives =="
    rclone copy external/valen "$REMOTE_ARCHIVE/valen_tgz" \
      --include '*.tgz' --exclude '*' $FLAGS
    echo "PUSH COMPLETE — run: bash scripts/drive_sync_v1.sh check"
    ;;
  check)
    rclone check checkpoints "$REMOTE_ARCHIVE/checkpoints" --checksum || true
    rclone check research/remote_archive "$REMOTE_ARCHIVE/remote_archive" --checksum || true
    ;;
  status)
    rclone about gdrive:
    echo "== archive =="; rclone size "$REMOTE_ARCHIVE" 2>/dev/null || echo "(empty)"
    echo "== staging =="; rclone size "$REMOTE_STAGE" 2>/dev/null || echo "(empty)"
    ;;
  stage-data)  # push a dataset dir for Colab pull: stage-data data/valen_nano_v9
    d="${2:?dataset dir}"
    rclone copy "$d" "$REMOTE_STAGE/data/$(basename "$d")" $FLAGS
    echo "staged at $REMOTE_STAGE/data/$(basename "$d")"
    ;;
  pull)
    rclone copy "${2:?remote path}" "${3:?local dir}" $FLAGS
    ;;
  *) echo "usage: push|check|status|stage-data <dir>|pull <remote> <local>"; exit 1 ;;
esac
