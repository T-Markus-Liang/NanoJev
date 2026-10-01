#!/usr/bin/env bash
# Install/remove the nanojev stack launchd agents (login-time auto start +
# crash keepalive). Reversible: run with --uninstall to remove.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
AGENTS_DIR="$HOME/Library/LaunchAgents"
# Note: the forward-ledger agent's daily chain (legacy refresh -> ledger v1 ->
# XS refresh -> ledger v2 -> snapshots) is defined in that plist's
# ProgramArguments; edit the plist, not this script, to change the chain.
LABELS="ai.nanojev.service ai.nanojev.winnow ai.nanojev.kev ai.nanojev.valen ai.nanojev.valen-lora ai.nanojev.memwatch ai.nanojev.forward-ledger"

if [ "${1:-}" = "--uninstall" ]; then
  for label in $LABELS; do
    launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
    rm -f "$AGENTS_DIR/$label.plist"
    echo "removed $label"
  done
  exit 0
fi

mkdir -p "$AGENTS_DIR"
for label in $LABELS; do
  launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
  cp "$HERE/$label.plist" "$AGENTS_DIR/$label.plist"
  launchctl bootstrap "gui/$(id -u)" "$AGENTS_DIR/$label.plist"
  echo "installed $label"
done
echo "logs: /tmp/nanojev-service.log /tmp/winnow-server.log /tmp/kev-serve.log /tmp/valen-head-server.log /tmp/valen-lora-server.log /tmp/nanojev-memwatch.log $HERE/../../logs/forward-ledger.log"
