#!/usr/bin/env python3
"""Bind the camera driver's lifecycle to the openpilot launcher (AGX Orin / twgmsl).

Idempotent. Ports the relevant parts of sunnypilot-cuda/aunch_pc.sh into this tree's
launch_chffrplus.sh.

Why: the camera contract requires the serdes to be initialised *before* camerad runs
(`/home/nvidia/公共/jetson-cuda-overlay/kits/camerad/CAMERAD_CONTRACT.md`). When the
driver is absent or stale, camerad's layout probe returns -1, falls back to guessing
G_FMT geometry + byte order and just runs dead: the UI shows no picture and nothing
in the log is an error. Leftover processes from a crashed/killed previous run make it
worse (they hold /dev/videoN, the VisionIPC socket and the ZMQ ports).

What it does (all inside launch(), before start_fake_panda):
  * kill_other_instances : force-close any other local openpilot process tree
                           (last one to start wins, like aunch_pc.sh's mutex).
  * ensure_camera_driver : start `sudo tw_camera_cfg` (service mode, NO `bring` --
                           bring is the GStreamer preview mode which spawns preview
                           pipelines that hold /dev/videoN), then wait up to 8s for
                           /dev/i2c-9 to be opened by it.
  * cleanup_camera_driver: on exit/INT/TERM, stop tw_camera_cfg and the fake panda.

Usage: python3 patch_launcher_camera.py <repo-root>
"""
import os
import sys

SENTINEL = "# ==== 相机驱动生命周期 + 单实例互斥"

BLOCK = r'''  # ==== 相机驱动生命周期 + 单实例互斥 (2026-10-11, 参考 sunnypilot-cuda/aunch_pc.sh) ====
  # 相机链路的先决条件: serdes 必须先于 camerad 初始化
  # (overlay/kits/camerad/CAMERAD_CONTRACT.md)。驱动缺失或状态陈旧时, camerad 的
  # 布局探测返回 -1 -> 退回"猜 G_FMT 几何 + 猜字节序"照跑 -> UI 黑屏且日志无 error。
  # 上一次崩溃/强杀残留的进程树还会占住 /dev/videoN、VisionIPC socket、ZMQ 端口。
  # 注意: pkill/pgrep -f 会匹配到自己的命令行, 所有模式一律写成 [x] 形式规避。
  kill_other_instances() {
    local pats=(
      "manager[.]py"
      "openpilot[.]system[.]manager"
      "openpilot[.]system[.]camerad"
      "webcam[.]camerad"
      "openpilot[.]selfdrive[.]ui"
      "selfdrive/ui/ui"
      "openpilot[.]system[.]loggerd"
      "openpilot[.]system[.]sensord"
      "[f]ake_panda_state"
      "[t]w_camera_cfg"
    )
    local hits="" p
    for p in "${pats[@]}"; do
      pgrep -f "$p" > /dev/null 2>&1 && hits="$hits ${p//[\[\]]/}"
    done
    [ -n "$hits" ] || { echo "[mutex] 无其它 openpilot 实例"; return 0; }
    echo "[mutex] 检测到其它 openpilot 实例:$hits"
    echo "[mutex] 强制关闭中 (后启动者只留自己)..."
    for p in "${pats[@]}"; do pkill -f "$p" 2>/dev/null || true; done
    sleep 1
    for p in "${pats[@]}"; do pkill -9 -f "$p" 2>/dev/null || true; done
    sleep 1
    echo "[mutex] 已清理"
  }

  # 森云相机驱动 (IMX390 + MAX9295A/MAX96712): 必须先于 camerad 起来,
  # 否则 VI 无帧输出 -> camerad 探测失败 -> UI 黑屏。
  # 服务模式 = `sudo tw_camera_cfg` 不带参数; 不要用 bring(那是 GStreamer 预览模式,
  # 会 spawn 预览管道占住 /dev/video0/1)。
  ensure_camera_driver() {
    if pgrep -x tw_camera_cfg > /dev/null 2>&1; then
      echo "[camera] tw_camera_cfg 已在运行, 跳过启动"
      return 0
    fi
    if ! sudo -n true 2>/dev/null; then
      echo "[camera] 警告: 当前无免密 sudo, 无法自动拉起相机驱动; 请手动执行 sudo tw_camera_cfg"
      return 0
    fi
    echo "[camera] 启动森云相机驱动 (tw_camera_cfg, serdes 初始化)..."
    sudo tw_camera_cfg > /tmp/tw_camera_cfg.log 2>&1 &
    local i pid
    for i in $(seq 1 16); do
      pid=$(pgrep -xn tw_camera_cfg 2>/dev/null || true)
      if [ -n "$pid" ] && sudo lsof -p "$pid" 2>/dev/null | grep -q "/dev/i2c-9"; then
        echo "[camera] 相机驱动就绪 (i2c 已初始化)"
        return 0
      fi
      sleep 0.5
    done
    echo "[camera] 警告: 8s 内未确认 i2c 就绪, 继续启动 (黑屏时先看 /tmp/tw_camera_cfg.log)"
  }

  # 退出(Ctrl-C/TERM/正常结束)时收掉相机驱动与假 panda, 不留占用设备的孤儿进程
  cleanup_camera_driver() {
    if pgrep -x tw_camera_cfg > /dev/null 2>&1; then
      echo ""
      echo "[camera] 停止森云相机驱动 (tw_camera_cfg)..."
      pkill -f "[t]w_camera_cfg" 2>/dev/null || true
    fi
    pkill -f "[f]ake_panda_state" 2>/dev/null || true
  }
  trap 'cleanup_camera_driver; exit 0' INT TERM
  trap 'cleanup_camera_driver' EXIT

  kill_other_instances
  ensure_camera_driver
  # ================================================================================

'''

ANCHOR = "  start_fake_panda() {\n"


def main():
  root = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
  path = os.path.join(root, "launch_chffrplus.sh")
  if not os.path.isfile(path):
    print(f"!! launch_chffrplus.sh not found under {root}", file=sys.stderr)
    return 1
  s = open(path, encoding="utf-8").read()
  if SENTINEL in s:
    print("  - launch_chffrplus.sh: already patched (no change)")
    return 0
  if ANCHOR not in s:
    print("!! anchor `start_fake_panda() {` not found -> SKIP (patch by hand)")
    return 1
  s = s.replace(ANCHOR, BLOCK + ANCHOR, 1)
  open(path, "w", encoding="utf-8").write(s)
  print("  + launch_chffrplus.sh: camera-driver lifecycle + single-instance mutex added")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
