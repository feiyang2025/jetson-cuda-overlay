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
REPO_ROOT="$(cd "$REPO_ROOT" && pwd -P)"

# Detect directory layout (new sunnypilot vs old openpilot)
if [ -f "$REPO_ROOT/openpilot/sunnypilot/modeld_v2/modeld.py" ]; then
  LAYOUT="new"
  MODELD_PY="$REPO_ROOT/openpilot/sunnypilot/modeld_v2/modeld.py"
  CAMERAD_PY="$REPO_ROOT/openpilot/system/camerad/webcam/camerad.py"
  TRANSFORMS_DIR="$REPO_ROOT/openpilot/selfdrive/modeld/transforms"
  RUNNERS_DIR="$REPO_ROOT/openpilot/selfdrive/modeld/runners"
  SENSORD_DIR="$REPO_ROOT/openpilot/system/sensord"
  MANAGER_DIR="$REPO_ROOT/openpilot/system/manager"
elif [ -f "$REPO_ROOT/selfdrive/modeld/modeld.py" ]; then
  LAYOUT="old"
  MODELD_PY="$REPO_ROOT/selfdrive/modeld/modeld.py"
  CAMERAD_PY="$REPO_ROOT/tools/webcam/camerad.py"
  TRANSFORMS_DIR="$REPO_ROOT/selfdrive/modeld/transforms"
  RUNNERS_DIR="$REPO_ROOT/selfdrive/modeld/runners"
  SENSORD_DIR="$REPO_ROOT/system/sensord"
  MANAGER_DIR="$REPO_ROOT/system/manager"
else
  echo "ERROR: Cannot detect project layout (no modeld.py found)" >&2
  exit 1
fi

echo "[apply_cuda] Detected layout: $LAYOUT"

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

install_file() { # src dst
  local src="$1" dst="$2"
  mkdir -p "$(dirname "$dst")"
  cp -f "$src" "$dst"
  echo "  + $(basename "$dst")"
}

echo "[apply_cuda] overlay=$OVERLAY_DIR"
echo "[apply_cuda] target=$REPO_ROOT"

echo "Installing CUDA model backend..."
if [ "$LAYOUT" = "new" ]; then
  install "openpilot/sunnypilot/modeld_v2/gpu_backend"
  install "openpilot/sunnypilot/modeld_v2/gpu_model_state.py"
else
  # Old layout: install compat modules
  install "openpilot/sunnypilot/modeld_v2/compat"
  install_file "$OVERLAY_DIR/openpilot/sunnypilot/modeld_v2/gpu_model_state.py" "$REPO_ROOT/openpilot/sunnypilot/modeld_v2/gpu_model_state.py"
fi
install_file "$OVERLAY_DIR/openpilot/selfdrive/modeld/transforms/cuda_transform.cu" "$TRANSFORMS_DIR/cuda_transform.cu"
install_file "$OVERLAY_DIR/openpilot/selfdrive/modeld/transforms/cuda_transform.h" "$TRANSFORMS_DIR/cuda_transform.h"
install_file "$OVERLAY_DIR/openpilot/selfdrive/modeld/runners/tensorrt_runner.py" "$RUNNERS_DIR/tensorrt_runner.py"
if [ -f "$OVERLAY_DIR/openpilot/selfdrive/modeld/runners/trt_c_api.so" ]; then
  install_file "$OVERLAY_DIR/openpilot/selfdrive/modeld/runners/trt_c_api.so" "$RUNNERS_DIR/trt_c_api.so"
else
  echo "  - trt_c_api.so not bundled in overlay (device-side artifact); skip copy (ensure the target tree already provides it)"
fi

echo "Installing V4L2/VIC camera adapter..."
if [ "$LAYOUT" = "new" ]; then
  install "openpilot/system/camerad/webcam/v4l2_dmabuf_camera.py"
  install "openpilot/system/camerad/webcam/v4l2_camera.py"
  install "openpilot/system/camerad/webcam/camera_cuda.py"
  install "openpilot/system/camerad/webcam/cuda_ipc_bridge.py"
  install "openpilot/system/camerad/webcam/cuda_jpeg_decoder.py"
else
  # Old layout: tools/webcam/
  install_file "$OVERLAY_DIR/openpilot/system/camerad/webcam/v4l2_dmabuf_camera.py" "$REPO_ROOT/tools/webcam/v4l2_dmabuf_camera.py"
  install_file "$OVERLAY_DIR/openpilot/system/camerad/webcam/v4l2_camera.py" "$REPO_ROOT/tools/webcam/v4l2_camera.py"
  install_file "$OVERLAY_DIR/openpilot/system/camerad/webcam/camera_cuda.py" "$REPO_ROOT/tools/webcam/camera_cuda.py"
  install_file "$OVERLAY_DIR/openpilot/system/camerad/webcam/cuda_ipc_bridge.py" "$REPO_ROOT/tools/webcam/cuda_ipc_bridge.py"
  install_file "$OVERLAY_DIR/openpilot/system/camerad/webcam/cuda_jpeg_decoder.py" "$REPO_ROOT/tools/webcam/cuda_jpeg_decoder.py"
fi
if [ -f "$CAMERAD_PY" ]; then
  # 水土不服根治: patch_camerad.py 只认官方结构的 import 锚点, 对胡萝卜系
  # (openpilot. 前缀) fork 会静默跳过 → 旧 camerad.py 保留 → VIC/FrameSync
  # 链路接不上。这里检测旧版(无 "V4L2 DMABUF" 字样)就直接用 overlay 完整
  # 适配版覆盖(备份原文件), 保证任何 fork 拿到 sp 同款链路。
  if grep -q "V4L2 DMABUF\|v4l2_dmabuf_camera" "$CAMERAD_PY" 2>/dev/null; then
    echo "  - camerad.py 已是 V4L2/VIC 适配版, 不动"
  else
    cp -f "$CAMERAD_PY" "$CAMERAD_PY.orig_openpilot" 2>/dev/null || true
    install_file "$OVERLAY_DIR/openpilot/system/camerad/webcam/camerad.py" "$CAMERAD_PY"
    echo "  - camerad.py 已替换为 overlay V4L2/VIC 完整适配版 (原文件备份为 .orig_openpilot)"
  fi
  python3 "$OVERLAY_DIR/patch_camerad.py" "$CAMERAD_PY" || \
    echo "[apply_cuda] WARN camerad patch skipped (camera import not found)"
fi

echo "Patching modeld.py..."
python3 "$OVERLAY_DIR/patch_modeld.py" "$MODELD_PY"

echo "Installing CH347 (USB-I2C IMU) daemon..."
install_file "$OVERLAY_DIR/openpilot/system/sensord/ch347t.cc" "$SENSORD_DIR/ch347t.cc"
install_file "$OVERLAY_DIR/openpilot/system/sensord/ch347t.py" "$SENSORD_DIR/ch347t.py"
install_file "$OVERLAY_DIR/openpilot/system/sensord/build_ch347t.sh" "$SENSORD_DIR/build_ch347t.sh"
install_file "$OVERLAY_DIR/openpilot/system/sensord/run_ch347t.sh" "$SENSORD_DIR/run_ch347t.sh"
if [ "$LAYOUT" = "new" ]; then
  install "openpilot/third_party/ch347"
else
  mkdir -p "$REPO_ROOT/third_party/ch347"
  cp -rf "$OVERLAY_DIR/openpilot/third_party/ch347/." "$REPO_ROOT/third_party/ch347/"
fi
chmod +x "$SENSORD_DIR/build_ch347t.sh" "$SENSORD_DIR/run_ch347t.sh" 2>/dev/null || true

# Patch process_config to register sensord_ch347 as an optional daemon
# (auto-exits if no CH347 device is present).
PCONFIG="$MANAGER_DIR/process_config.py"
if [ -f "$PCONFIG" ] && ! grep -q "sensord_ch347" "$PCONFIG"; then
  python3 "$OVERLAY_DIR/patch_ch347_manager.py" "$PCONFIG" || \
    echo "[apply_cuda] WARN ch347 manager patch skipped"
fi

echo "Installing camera-calibration toolkit..."
# tools/calib/*: FCAM/ECAM intrinsic + ECAM extrinsic calibration (log-based,
# no hardcoded paths, layout-agnostic). Run with the *system* python3 for the
# scripts that need cv2 (wide_calibrator).
mkdir -p "$REPO_ROOT/tools/calib"
cp -rf "$OVERLAY_DIR/openpilot/tools/calib/." "$REPO_ROOT/tools/calib/"
echo "  + tools/calib/ ($(ls -1 "$OVERLAY_DIR/openpilot/tools/calib" | wc -l) files)"

# Patch calendar intrinsics plumbing (camera.py) + the Param keys it needs.
# patch_calib.py returns 0=no-op, 1=error, 2=applied. Guard against set -e
# so the success code (2) does not abort the whole script.
set +e
python3 "$OVERLAY_DIR/patch_calib.py" "$REPO_ROOT"
CALIB_RC=$?
set -e
if [ "$CALIB_RC" = "1" ]; then
  echo "[apply_cuda] WARN calibration patch reported anchors that need manual merge (see above)"
elif [ "$CALIB_RC" = "2" ]; then
  echo "[apply_cuda] NOTE camera.py / params_keys.h changed -> rebuild the params module:"
  echo "[apply_cuda]        cd $REPO_ROOT && source .venv/bin/activate && scons -j8 common/"
fi

echo "[apply_cuda] done."
echo
echo "Next on-device steps:"
echo "  1. (Orin) cd $TRANSFORMS_DIR"
echo "            nvcc -arch=sm_87 -shared -O2 -o libcuda_transform.so cuda_transform.cu -I."
echo "  2. generate metadata pkl: python3 $REPO_ROOT/selfdrive/modeld/get_model_metadata.py <onnx>"
echo "  3. ensure .plan engines exist under $REPO_ROOT/selfdrive/modeld/models/<Name>/"
echo "  4. env: USE_V4L2_CAMERA=1 (default), DISABLE_CUDA_BACKEND unset (default CUDA on)"