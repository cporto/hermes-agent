#!/bin/sh
# launchd entrypoint for ai.hermes.gateway: wait for volume, preflight, exec gateway.
# Install root-owned (agents must not be able to edit it); see docs in the commit message.
REPO=/Volumes/X10Pro/hermes/hermes-agent
VENV=/Users/hermes/.hermes/venv-gateway-internal
LOG=/Users/hermes/.hermes/logs-launchd/preflight.log
GUARD="$(dirname "$0")/preflight.py"

/bin/wait4path "$REPO" || exit 1
"$VENV/bin/python" "$GUARD" --repo "$REPO" --python "$VENV/bin/python" --log "$LOG"
rc=$?
# 1 = broken and not recoverable: don't start (launchd retries in 30s). 2 = guard bug: fail open.
[ "$rc" -eq 1 ] && exit 1
exec "$VENV/bin/hermes" gateway run --external-supervisor
