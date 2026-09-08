#!/usr/bin/env bash
set -euo pipefail

# Apply the AGX Orin CUDA/TensorRT + V4L2/VIC camera overlay onto a
# sunnypilot/openpilot tree (old OR new layout).
#
# Usage:
#   bash jetson-cuda-overlay/apply_cuda.sh <target-project-root>
#   (no arg -> current dir)
#
# This is version-independent and idempotent:
#   1. copies gpu_backend + gpu_model_state + cuda_transform + tensorrt_runner
#   2. copies V4L2/VIC camera adapter under system/camerad/webcam/ and patches
#      the camerad import to prefer it on Linux
#   3. runs patch_modeld.py to inject `_make_model()` (CUDA-first, tinygrad fallback)
#
# Re-run after `git pull upstream` to re-apply everything.

OVERLAY_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${1:-$(pwd)}"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
MODELD_PY="$REPO_ROOT/openpilot/sunnypilot/modeld_v2/modeld.py"
CAMERAD_PY="$REPO_ROOT/openpilot/system/camerad/webcam/camerad.py"

install() { # relpath
  local src="$OVERLAY_DIR/$1" dst="$REPO_ROOT/$1"
  mkdir -p "$(dirname "$dst")"
  if [ -d "$src" ]; then
    mkdir -p "$dst" && cp -rf "$src/." "$dst/"
  else
    cp -f "$src" "$dst"
  fi
  echo "  + $1"
}

echo "[apply_cuda] overlay=$OVERLAY_DIR"
echo "[apply_cuda] target=$REPO_ROOT"

if [ ! -f "$MODELD_PY" ]; then
  echo "ERROR: modeld_v2 not found at $MODELD_PY (not a sunnypilot tree?)" >&2
  exit 1
fi

echo "Installing CUDA model backend..."
install "openpilot/sunnypilot/modeld_v2/gpu_backend"
install "openpilot/sunnypilot/modeld_v2/gpu_model_state.py"
install "openpilot/selfdrive/modeld/transforms/cuda_transform.cu"
install "openpilot/selfdrive/modeld/transforms/cuda_transform.h"
install "openpilot/selfdrive/modeld/runners/tensorrt_runner.py"

echo "Installing V4L2/VIC camera adapter..."
install "openpilot/system/camerad/webcam/v4l2_dmabuf_camera.py"
install "openpilot/system/camerad/webcam/v4l2_camera.py"
install "openpilot/system/camerad/webcam/camera_cuda.py"
install "openpilot/system/camerad/webcam/cuda_ipc_bridge.py"
install "openpilot/system/camerad/webcam/cuda_jpeg_decoder.py"
if [ -f "$CAMERAD_PY" ]; then
  python3 "$OVERLAY_DIR/patch_camerad.py" "$CAMERAD_PY" || \
    echo "[apply_cuda] WARN camerad patch skipped (camera import not found)"
fi

echo "Patching modeld.py..."
python3 "$OVERLAY_DIR/patch_modeld.py" "$MODELD_PY"

echo "Installing CH347 (USB-I2C IMU) daemon..."
install "openpilot/system/sensord/ch347t.cc"
install "openpilot/system/sensord/ch347t.py"
install "openpilot/system/sensord/build_ch347t.sh"
install "openpilot/system/sensord/run_ch347t.sh"
install "openpilot/third_party/ch347"
chmod +x "$REPO_ROOT/openpilot/system/sensord/build_ch347t.sh" "$REPO_ROOT/openpilot/system/sensord/run_ch347t.sh" 2>/dev/null || true

# Patch process_config to register sensord_ch347 as an optional daemon
# (auto-exits if no CH347 device is present).
PCONFIG="$REPO_ROOT/openpilot/system/manager/process_config.py"
if [ -f "$PCONFIG" ] && ! grep -q "sensord_ch347" "$PCONFIG"; then
  python3 "$OVERLAY_DIR/patch_ch347_manager.py" "$PCONFIG" || \
    echo "[apply_cuda] WARN ch347 manager patch skipped"
fi

echo "[apply_cuda] done."
echo
echo "Next on-device steps:"
echo "  1. (Orin) cd $REPO_ROOT/openpilot/selfdrive/modeld/transforms"
echo "            nvcc -arch=sm_87 -shared -O2 -o libcuda_transform.so cuda_transform.cu -I."
echo "  2. generate metadata pkl: python3 openpilot/selfdrive/modeld/get_model_metadata.py <onnx>"
echo "  3. ensure .plan engines exist under openpilot/selfdrive/modeld/models/<Name>/"
echo "  4. env: USE_V4L2_CAMERA=1 (default), DISABLE_CUDA_BACKEND unset (default CUDA on)"