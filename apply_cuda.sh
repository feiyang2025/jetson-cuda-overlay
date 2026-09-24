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

echo "Installing modeld kit (kits/modeld, TRT 兜底闭环)..."
MODELD_KIT="$OVERLAY_DIR/kits/modeld"
MODELD_KIT_DST="$REPO_ROOT/selfdrive/modeld"
mkdir -p "$MODELD_KIT_DST/runners" "$MODELD_KIT_DST/transforms"
for f in modeld.py modeld_bigcombo.py; do
  if [ -f "$MODELD_KIT_DST/$f" ] && ! grep -q "TRT_LOAD_ATTEMPTS\|BigComboTrtUnavailable" "$MODELD_KIT_DST/$f" 2>/dev/null; then
    cp -f "$MODELD_KIT_DST/$f" "$MODELD_KIT_DST/$f.orig_openpilot" 2>/dev/null || true
    install_file "$MODELD_KIT/$f" "$MODELD_KIT_DST/$f"
    echo "  - $f 已替换为 TRT 兜底闭环版 (原文件备份 .orig_openpilot)"
  else
    echo "  - $f 已是兜底闭环版, 不动"
  fi
done
install_file "$MODELD_KIT/tensorrt_runner.py" "$MODELD_KIT_DST/runners/tensorrt_runner.py"
install_file "$MODELD_KIT/cuda_transform.cu" "$MODELD_KIT_DST/transforms/cuda_transform.cu"
install_file "$MODELD_KIT/cuda_transform.h" "$MODELD_KIT_DST/transforms/cuda_transform.h"

echo "Installing V4L2/VIC camera adapter..."
CAMERA_KIT="$OVERLAY_DIR/kits/camerad"
# 布局检测以目标 fork 实际目录为准 (不猜 new/old):
# sp 系 = tools/webcam; 官方新系 = system/camerad/webcam (或 openpilot/ 前缀)
if [ -d "$REPO_ROOT/openpilot/system/camerad/webcam" ]; then
  CAM_DST_DIR="$REPO_ROOT/openpilot/system/camerad/webcam"
elif [ -d "$REPO_ROOT/system/camerad/webcam" ]; then
  CAM_DST_DIR="$REPO_ROOT/system/camerad/webcam"
elif [ -d "$REPO_ROOT/tools/webcam" ]; then
  CAM_DST_DIR="$REPO_ROOT/tools/webcam"
else
  CAM_DST_DIR="$REPO_ROOT/tools/webcam"
fi
mkdir -p "$CAM_DST_DIR"
for f in camerad.py v4l2_dmabuf_camera.py v4l2_camera.py camera_cuda.py packed_to_nv12.cu nvbuf_import.cu; do
  if [ -f "$CAM_DST_DIR/$f" ] && [ "$f" = "camerad.py" ]; then
    cp -f "$CAM_DST_DIR/$f" "$CAM_DST_DIR/$f.orig_openpilot" 2>/dev/null || true
  fi
  install_file "$CAMERA_KIT/$f" "$CAM_DST_DIR/$f"
done
echo "  -> camerad kit 已安装到 $CAM_DST_DIR"
# 水土不服防护: kit 版自带自适应 import, 这里只验证不修复
if grep -q "V4L2 DMABUF\|v4l2_dmabuf_camera" "$CAM_DST_DIR/camerad.py" 2>/dev/null; then
  echo "  - camerad.py kit 版确认 (自适配入口)"
else
  echo "  [WARN] camerad.py 非 kit 版, 检查 $CAM_DST_DIR/camerad.py"
fi

# camerad 的 CUDA 辅助 so: 缺才编译 (libpacked_to_nv12.so 必编; libnvbuf_import.so 仅 SP_NVBUF_ZEROCOPY=1 需要, 仍一并编译兜底)
cd "$CAM_DST_DIR"
NVCC_BIN="$(command -v nvcc || true)"
[ -z "$NVCC_BIN" ] && [ -x /usr/local/cuda/bin/nvcc ] && NVCC_BIN=/usr/local/cuda/bin/nvcc
if [ -z "$NVCC_BIN" ]; then
  echo "  [WARN] 无 nvcc, 跳过 libpacked_to_nv12.so / libnvbuf_import.so 编译 (camerad 将走 CPU 回退)"
elif [ ! -f libpacked_to_nv12.so ]; then
  echo "  + 编译 libpacked_to_nv12.so ..."
  nvcc -arch=sm_87 -shared -O2 -o libpacked_to_nv12.so packed_to_nv12.cu && echo "    OK" || echo "    [WARN] 编译失败, camerad 走 CPU 回退"
fi
if [ -n "$NVCC_BIN" ] && [ ! -f libnvbuf_import.so ]; then
  echo "  + 编译 libnvbuf_import.so ..."
  nvcc -arch=sm_87 -shared -O2 -o libnvbuf_import.so nvbuf_import.cu -lcuda && echo "    OK" || echo "    [WARN] 编译失败, SP_NVBUF_ZEROCOPY 将无法启用(其余链路不受影响)"
fi
cd "$OVERLAY_DIR"

echo "Applying msgq zerocopy patch (write_and_send + refcount)..."
MSGQ_PATCH="$OVERLAY_DIR/kits/camerad/msgq/0001-visionipc-zerocopy.patch"
for MSGQ_DIR in "$REPO_ROOT/msgq" "$REPO_ROOT/msgq_repo" "$REPO_ROOT/openpilot/msgq"; do
  if [ -d "$MSGQ_DIR" ]; then
    if grep -rq "write_and_send" "$MSGQ_DIR/visionipc" 2>/dev/null; then
      echo "  - $MSGQ_DIR 已有 write_and_send, 跳过"
    elif git -C "$MSGQ_DIR" apply --check "$MSGQ_PATCH" 2>/dev/null; then
      git -C "$MSGQ_DIR" apply "$MSGQ_PATCH" && echo "  + $MSGQ_DIR zerocopy patch 已应用"
    else
      echo "  [WARN] $MSGQ_DIR patch 应用失败(子模块版本不同?), 手工:"
      echo "         git -C $MSGQ_DIR apply $MSGQ_PATCH"
    fi
    break
  fi
done

echo "Patching SConstruct (aarch64 -D__JETSON__)..."
SCONSTRUCT="$REPO_ROOT/SConstruct"
if [ -f "$SCONSTRUCT" ] && ! grep -q "__JETSON__" "$SCONSTRUCT"; then
  python3 - "$SCONSTRUCT" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
anchor = 'elif arch == "aarch64":'
if anchor in s and '-D__JETSON__' not in s:
  ins = anchor + '\n  # Jetson (Orin): visionbuf_jetson.cc CUDA 零拷贝映射 (overlay kits/modeld)\n  cflags += ["-D__JETSON__"]\n  cxxflags += ["-D__JETSON__"]\n  cpppath += ["/usr/local/cuda/include"]'
  s = s.replace(anchor, ins, 1)
  open(p, 'w').write(s)
  print("[apply_cuda] SConstruct aarch64: + -D__JETSON__ + CUDA include")
else:
  print("[apply_cuda] SConstruct 无 aarch64 锚点或已有 __JETSON__")
PY
else
  echo "  - SConstruct 已有 __JETSON__"
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