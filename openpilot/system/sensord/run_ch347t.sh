#!/usr/bin/env bash
set -euo pipefail

# Run the CH347 USB-I2C IMU daemon. Builds the binary on first invocation,
# then execs it. If no CH347 device is present the daemon exits cleanly.
#
# Launched by the manager as a NativeProcess with cwd = <repo>/openpilot/system/sensord

# Resolve our own real directory. REPO_ROOT is *not* derived by counting
# "../.." levels any more — the layout differs (old: openpilot/ is a symlink,
# new: openpilot/ is a real subdir). build_ch347t.sh finds the repo root itself
# by walking up to the first dir containing msgq_repo/.
SELF_DIR="$(cd "$(dirname "$0")" && pwd -P)"

BIN="$SELF_DIR/ch347t"

# Rebuild also when the source is newer than the binary: the cereal schema
# changes across forks and a stale binary fails to publish / crashes manager.
if [ ! -x "$BIN" ] || [ "$SELF_DIR/ch347t.cc" -nt "$BIN" ]; then
  echo "[ch347t] building..."
  bash "$SELF_DIR/build_ch347t.sh"
fi

exec "$BIN"
