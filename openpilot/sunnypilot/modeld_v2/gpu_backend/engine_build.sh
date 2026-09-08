#!/usr/bin/env bash
set -euo pipefail

# ONNX -> TensorRT .plan builder for AGX Orin (sm_87).
# Usage:
#   ./engine_build.sh <model.onnx> <output.plan> [--fp16] [workspace_mb]

ONNX="${1:?usage: engine_build.sh <onnx> <plan> [--fp16] [workspace_mb]}"
OUT="${2:?usage: engine_build.sh <onnx> <plan> [--fp16] [workspace_mb]}"
PRECISION="${3:---fp16}"
WORKSPACE_MB="${4:-4096}"

if ! command -v trtexec >/dev/null 2>&1; then
  echo "ERROR: trtexec not found (install TensorRT or add to PATH)" >&2
  exit 1
fi

echo "[engine_build] $ONNX -> $OUT ($PRECISION, workspace=${WORKSPACE_MB}MB)"
mkdir -p "$(dirname "$OUT")"

trtexec \
  --onnx="$ONNX" \
  --saveEngine="$OUT" \
  "$PRECISION" \
  --memPoolSize="workspace:${WORKSPACE_MB}" \
  --noDataTransfers

echo "[engine_build] done: $OUT"
