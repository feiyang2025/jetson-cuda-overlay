#!/usr/bin/env bash
set -euo pipefail

# AGX Orin CUDA backend deploy/verify script for master-c3.
# Run ON the Orin after pulling the repo.
# Usage: ./deploy.sh [--fp16]
#
# Steps:
#   1. build libcuda_transform.so from cuda_transform.cu
#   2. generate metadata.pkl for each model dir (get_model_metadata.py)
#   3. (optional) build TRT .plan from onnx via trtexec
#   Prints a verification checklist.

REPO_ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"   # master-c3 (root)
OPENPILOT="$REPO_ROOT/openpilot"
MODELD="$OPENPILOT/selfdrive/modeld"
TRANSFORMS="$MODELD/transforms"
MODELS="$MODELD/models"
PRECISION="${1:---fp16}"
PY="${2:-${VENV_PYTHON:-python3}}"

echo "[deploy] repo=$REPO_ROOT precision=$PRECISION"

# --- 1. CUDA transform ---
echo "[deploy] building libcuda_transform.so"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc not found; install CUDA toolkit first" >&2
  exit 1
fi
(cd "$TRANSFORMS" && nvcc -arch=sm_87 -shared -O2 -o libcuda_transform.so cuda_transform.cu -I.)
echo "[deploy] libcuda_transform.so ok"

# --- 2. metadata ---
for model_dir in "$MODELS"/*/; do
  name="$(basename "$model_dir")"
  for onnx in "$model_dir"/*.onnx; do
    [ -e "$onnx" ] || continue
    pyfile="$MODELD/get_model_metadata.py"
    echo "[deploy] metadata for $onnx"
    "$PY" "$pyfile" "$onnx" || echo "[deploy] WARN metadata failed for $onnx"
  done
done

echo "[deploy] done. Verify manually:"
echo "  1. $TRANSFORMS/libcuda_transform.so exists"
echo "  2. model metadata pkl exist under $MODELS"
echo "  3. env CUDA_BACKEND=1 (default on) + DISABLE_CUDA_BACKEND=0"