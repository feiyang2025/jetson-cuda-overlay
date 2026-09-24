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
REPO_ROOT="$(cd "$REPO_ROOT" && pwd -P)"
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

# Link against pre-built libraries instead of compiling from source.
# This avoids complex dependency chains (json11, msgq internal deps).
# In the new layout these live under $CEREAL_ROOT (= <root>/openpilot),
# in the old layout directly under $REPO_ROOT — resolve both.
find_lib() { # relpath
  if [ -f "$REPO_ROOT/$1" ]; then
    echo "$REPO_ROOT/$1"
  elif [ -f "$CEREAL_ROOT/$1" ]; then
    echo "$CEREAL_ROOT/$1"
  else
    return 1
  fi
}

MSGQ_LIB="$(find_lib msgq_repo/libmsgq.a)" || { echo "ERROR: required library not found: msgq_repo/libmsgq.a (also tried \$CEREAL_ROOT)" >&2; exit 1; }
COMMON_LIB="$(find_lib common/libcommon.a)" || { echo "ERROR: required library not found: common/libcommon.a (also tried \$CEREAL_ROOT)" >&2; exit 1; }
JSON11_O="$(find_lib third_party/json11/json11.o)" || { echo "ERROR: required library not found: third_party/json11/json11.o (also tried \$CEREAL_ROOT)" >&2; exit 1; }

g++ -O2 -std=c++17 -o "$OUT" "$SRC" \
  "$CEREAL_ROOT/cereal/messaging/socketmaster.cc" \
  "$MSGQ_LIB" "$COMMON_LIB" "$JSON11_O" \
  -I"$CEREAL_ROOT" -I"$CEREAL_ROOT/cereal/gen/cpp" -I"$REPO_ROOT/msgq_repo" -I"$REPO_ROOT" \
  -pthread -ldl -lzmq -lcapnp -lkj

echo "[build_ch347t] done: $OUT"
