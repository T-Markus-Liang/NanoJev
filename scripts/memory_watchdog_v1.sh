#!/usr/bin/env bash
# memory_watchdog_v1.sh — external RSS watchdog for the NanoJev model
# services whose binaries cannot self-monitor (kev-4b, winnow-12B).
# A service over its RSS ceiling is restarted via launchctl kickstart -k;
# KeepAlive relaunches it with a clean heap/compressor footprint.
#
# thresholds (GB RSS): kev 40 / winnow 24 / valen-lora 32 (safety net on top
# of its in-process watchdog) / serve_decisions 8.
set -u
UID_=$(id -u)
check() { # <launchd label> <port> <rss_limit_gb>
  local label=$1 port=$2 limit=$3
  local pid rss_kb rss_gb
  pid=$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null | head -1) || true
  [ -z "${pid:-}" ] && return 0   # down — launchd KeepAlive owns restart
  rss_kb=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ') || return 0
  [ -z "${rss_kb:-}" ] && return 0
  rss_gb=$((rss_kb / 1048576))
  if [ "$rss_gb" -gt "$limit" ]; then
    echo "$(date -u +%FT%TZ) $label pid=$pid rss=${rss_gb}GB > ${limit}GB — kickstart"
    launchctl kickstart -k "gui/$UID_/$label" || \
      echo "$(date -u +%FT%TZ) $label kickstart failed"
  fi
}
check ai.nanojev.kev        8092 40
check ai.nanojev.winnow     8091 24
check ai.nanojev.valen-lora 8094 32
check ai.nanojev.service    8876 8
