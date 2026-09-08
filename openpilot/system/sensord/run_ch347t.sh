#!/usr/bin/env bash
set -euo pipefail

# Run the CH347 USB-I2C IMU daemon. Builds the binary on first invocation,
# then execs it. If no CH347 device is present the daemon exits cleanly.
#
# Launched by the manager as a NativeProcess with cwd = <repo>/openpilot/system/sensord

SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SELF_DIR/../../.." && pwd)"

BIN="$SELF_DIR/ch347t"

if [ ! -x "$BIN" ]; then
  echo "[ch347t] building..."
  bash "$SELF_DIR/build_ch347t.sh" "$REPO_ROOT"
fi

exec "$BIN"
