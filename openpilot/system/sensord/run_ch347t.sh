#!/usr/bin/env bash
set -euo pipefail

# Run the CH347 USB-I2C IMU daemon. Builds the binary on first invocation,
# then execs it. If no CH347 device is present the daemon exits cleanly.
#
# Launched by the manager as a NativeProcess with cwd = <repo>/openpilot/system/sensord

# Resolve symlinks so REPO_ROOT lands in the real repo, not a symlink parent.
# Physical path from sensord/ to repo root is 2 levels up (system → ajouatom),
# but the logical path openpilot/system/sensord/ is 3 levels. Using pwd -P
# resolves symlinks, so we need ../.. not ../../..
SELF_DIR="$(cd "$(dirname "$0")" && pwd -P)"
REPO_ROOT="$(cd "$SELF_DIR/../.." && pwd -P)"

BIN="$SELF_DIR/ch347t"

if [ ! -x "$BIN" ]; then
  echo "[ch347t] building..."
  bash "$SELF_DIR/build_ch347t.sh" "$REPO_ROOT"
fi

exec "$BIN"
