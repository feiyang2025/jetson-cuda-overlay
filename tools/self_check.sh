#!/usr/bin/env bash
set -uo pipefail
# self_check.sh — camerad/modeld kit 完整性自检 (相机链路 + 推理兜底 + 配套改动)
# 用法: bash tools/self_check.sh [目标树根]   (默认当前目录)
# 风格同 panda_boot_check.sh: 只检查不改动, 输出 PASS/FAIL 清单。

REPO_ROOT="$(cd "${1:-$(pwd)}" && pwd -P)"
FAIL=0
fail() { echo "  [FAIL] $1"; FAIL=1; }
pass() { echo "  [PASS] $1"; }
warn() { echo "  [WARN] $1"; }

echo "== self_check: $REPO_ROOT =="

# --- 布局探测 ---
if [ -d "$REPO_ROOT/tools/webcam" ]; then
  CAM_DST="$REPO_ROOT/tools/webcam"
elif [ -d "$REPO_ROOT/system/camerad/webcam" ]; then
  CAM_DST="$REPO_ROOT/system/camerad/webcam"
elif [ -d "$REPO_ROOT/openpilot/system/camerad/webcam" ]; then
  CAM_DST="$REPO_ROOT/openpilot/system/camerad/webcam"
else
  CAM_DST=""
fi

echo "[相机链路 camerad kit]"
if [ -n "$CAM_DST" ] && [ -f "$CAM_DST/camerad.py" ]; then
  if grep -q "V4L2 DMABUF\|v4l2_dmabuf_camera" "$CAM_DST/camerad.py"; then
    pass "camerad.py kit 版 (V4L2 DMABUF 自适应入口)"
  else
    fail "camerad.py 非 kit 版 ($CAM_DST/camerad.py)"
  fi
  grep -q "GMSL_CHROMA_LAYOUT" "$CAM_DST/camerad.py" && pass "twgmsl 色度规则在" || fail "GMSL_CHROMA_LAYOUT 缺失"
  grep -q "write_and_send" "$CAM_DST/camerad.py" && pass "零拷贝检测逻辑在" || warn "零拷贝检测逻辑缺失"
  for f in v4l2_dmabuf_camera.py packed_to_nv12.cu; do
    [ -f "$CAM_DST/$f" ] && pass "$f 在" || fail "$f 缺失"
  done
  [ -f "$CAM_DST/libpacked_to_nv12.so" ] && pass "libpacked_to_nv12.so (设备侧产物)" || warn "libpacked_to_nv12.so 未编译 (CPU 回退, 可接受)"
else
  fail "camerdad kit 未安装 (无 $CAM_DST/camerad.py)"
fi

echo "[msgq 零拷贝配套]"
MSGQ_DIR=""
for d in "$REPO_ROOT/msgq" "$REPO_ROOT/msgq_repo"; do
  [ -d "$d" ] && MSGQ_DIR="$d" && break
done
if [ -n "$MSGQ_DIR" ]; then
  grep -rq "write_and_send" "$MSGQ_DIR/visionipc" 2>/dev/null && pass "msgq write_and_send 在 ($MSGQ_DIR)" || warn "msgq 无 write_and_send (零拷贝自动回退主机路径)"
  grep -rq "acquire" "$MSGQ_DIR/visionipc/visionbuf.h" 2>/dev/null && pass "visionbuf refcount 在" || warn "visionbuf refcount 缺失"
else
  warn "未找到 msgq 子模块目录"
fi

echo "[模型推理 modeld kit]"
MD="$REPO_ROOT/selfdrive/modeld"
if [ -f "$MD/modeld.py" ]; then
  grep -q "TRT_LOAD_ATTEMPTS" "$MD/modeld.py" && pass "modeld.py TRT 兜底闭环在" || fail "modeld.py 非兜底闭环版 (缺 TRT_LOAD_ATTEMPTS)"
  grep -q "BigComboTrtUnavailable" "$MD/modeld_bigcombo.py" 2>/dev/null && pass "modeld_bigcombo.py BigCombo 降级闭环在" || fail "modeld_bigcombo.py 缺 BigComboTrtUnavailable"
else
  fail "selfdrive/modeld/modeld.py 不存在"
fi
[ -f "$MD/runners/tensorrt_runner.py" ] && pass "tensorrt_runner.py 在" || fail "tensorrt_runner.py 缺失"
[ -f "$MD/transforms/cuda_transform.cu" ] && pass "cuda_transform.cu 在" || fail "cuda_transform.cu 缺失"
[ -f "$MD/transforms/libcuda_transform.so" ] && pass "libcuda_transform.so (设备侧产物)" || warn "libcuda_transform.so 未编译 (需 nvcc 编)"
[ -d "$MD/models" ] && ls "$MD/models"/*/ 2>/dev/null | grep -q "\.plan" && pass "有 .plan 引擎" || warn "models/ 下无 .plan (需 trtexec 编)"

echo "[SConstruct JETSON]"
grep -q "__JETSON__" "$REPO_ROOT/SConstruct" 2>/dev/null && pass "-D__JETSON__ 在 SConstruct" || warn "SConstruct 缺 -D__JETSON__ (零拷贝 CUDA 映射不生效)"

echo "[本地散改补丁]"
LP="$REPO_ROOT/sunnypilot/selfdrive/controls/lib/longitudinal_planner.py"
if [ -f "$LP" ]; then
  grep -q "smartCruiseControl" "$LP" && pass "SCC-V/SCC-M 补丁在 (longitudinal_planner.py)" || warn "SCC 补丁丢失! 重打: git apply patches/sp_longitudinal_planner_scc.patch"
else
  warn "无 sunnypilot 树 (非 sp 系, 跳过 SCC 检查)"
fi

echo
if [ "$FAIL" = "0" ]; then
  echo "self_check: 全部通过"
else
  echo "self_check: 有 FAIL 项, 见上"
  exit 1
fi