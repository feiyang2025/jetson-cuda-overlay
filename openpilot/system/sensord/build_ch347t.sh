#!/usr/bin/env bash
set -euo pipefail

# Build the CH347 (USB-I2C) IMU daemon for AGX Orin / Linux.
# Produces  openpilot/system/sensord/ch347t
#
# Usage: bash build_ch347t.sh [repo-root]   (default: current dir)
#
# Links only against libc/libstdc++/libdl + the CH347 vendor .so at runtime
# (dlopen), so it does NOT need json11 or a full openpilot build.

REPO_ROOT="${1:-$(pwd)}"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
SRC="$REPO_ROOT/openpilot/system/sensord/ch347t.cc"
OUT="$REPO_ROOT/openpilot/system/sensord/ch347t"
MSGQ="$REPO_ROOT/openpilot/cereal/messaging"

echo "[build_ch347t] src=$SRC"
echo "[build_ch347t] out=$OUT"

if [ ! -f "$SRC" ]; then
  echo "ERROR: $SRC not found (is this a sunnypilot tree?)" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"

# Compile msgq + cereal support sources (path-independent of old/new layout).
# Old layout uses cereal/messaging at root; new layout under openpilot/.
if [ -f "$REPO_ROOT/cereal/messaging/socketmaster.cc" ]; then
  CEREAL_ROOT="$REPO_ROOT"
elif [ -f "$REPO_ROOT/openpilot/cereal/messaging/socketmaster.cc" ]; then
  CEREAL_ROOT="$REPO_ROOT/openpilot"
else
  echo "ERROR: cannot find cereal/messaging" >&2
  exit 1
fi

MSGQ_SRC=""
for f in impl_msgq impl_zmq impl_fake ipc event; do
  for base in "$CEREAL_ROOT/../msgq_repo/msgq" "$REPO_ROOT/msgq_repo/msgq"; do
    if [ -f "$base/$f.cc" ]; then MSGQ_SRC="$MSGQ_SRC $base/$f.cc"; break; fi
  done
done

COMMON_SRC=""
for f in util.cc swaglog.cc ratekeeper.cc; do
  for base in "$CEREAL_ROOT/common"; do
    if [ -f "$base/$f" ]; then COMMON_SRC="$COMMON_SRC $base/$f"; break; fi
  done
done

g++ -O2 -std=c++17 -o "$OUT" "$SRC" \
  "$CEREAL_ROOT/cereal/messaging/socketmaster.cc" \
  $MSGQ_SRC \
  $COMMON_SRC \
  -I"$CEREAL_ROOT" -I"$REPO_ROOT/msgq_repo" \
  -pthread -ldl -lzmq -lcapnp -lkj

echo "[build_ch347t] done: $OUT"
