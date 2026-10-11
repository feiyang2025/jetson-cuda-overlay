#!/usr/bin/env python3
"""calibd —— 常驻自动相机内参标定 (manager 监督).

为什么是常驻进程而不是 UI 按钮: 按钮 spawn 的子进程在重启/整机断电后就没了, 采集等于
随机发生; 而"设备自己把这事做完"才是目标 —— 用户不需要知道有这个流程。

行为:
  * 只读订阅 (carState/radarState/modelV2/deviceState), 不发布任何驾驶相关消息;
  * onroad 且没有手动标定在跑时开一趟采样 (CalibSession, 与手动 --live 同一份口径);
  * offroad (或手动标定接管/收到 SIGTERM) 时收尾: 融合计算 + 通过门限才写结果参数;
  * AutoCalib 打开时再走决策层 (连续 N 趟一致才自动写内参, 写前备份、可一键回滚)。

摄像头选择: 跟随 modeld —— 它把 modelV2 算出来的那台相机就是我们需要标定的那台
(与 modeld.py:456 同一判据: 路流不在可用流里 → 主相机是 wide)。手动标定 (FCAM Live /
Wide Calibration) 在跑时本进程让位, 免得两边抢同一份数据、给出两个结论。

开关 (Params):
  AutoCalib          BOOL  自动写入内参开关 (关 → 只收数据、只在面板给建议)
  AutoCalibHistory   JSON  每趟结果与动作 (最近 20 条, 供面板/排查看趋势)
  AutoCalibBackup    JSON  自动写入前的旧内参备份
  AutoCalibRollback  BOOL  置 1 → 回滚到备份并清除该标志
"""

from __future__ import annotations

import os
import signal
import sys
import time

from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot.calib.controller import AutoCalibController

# tools/calib 在树根 (不是 openpilot/ 里), 而算法只此一份 → 显式把该目录加进 sys.path,
# 免得把 SelfCalibrator 复制到 openpilot/ 下再分家一次。
_CALIB_DIR = os.path.join(BASEDIR, "tools", "calib")
if _CALIB_DIR not in sys.path:
  sys.path.insert(0, _CALIB_DIR)

LOG_PERIOD_S = 30.0          # 采样期间的状态打印周期
IDLE_PERIOD_S = 2.0          # offroad/让位时的轮询周期
SAMPLE_TIMEOUT_MS = 100      # SubMaster 阻塞上限


def _log(msg: str) -> None:
  print(f"[calibd] {msg}", flush=True)


def detect_main_camera() -> str:
  """modeld 用哪台相机算 modelV2, 我们就标哪台 (判据与 modeld.py:456 一致)。

  本代 msgq 不导出 VisionStreamType (modeld 也是同样兜底成数值 0/1/2), 所以这里照抄它。
  """
  vst_road = 0
  try:
    from msgq.visionipc import VisionStreamType          # 老 msgq 有类型
    vst_road = int(VisionStreamType.VISION_STREAM_ROAD)
  except ImportError:
    vst_road = 0                                          # 新代 msgq: 数值与 sp 枚举一致
  try:
    from msgq.visionipc import VisionIpcClient
    streams = VisionIpcClient.available_streams("camerad", block=False)
    if streams:
      return "ecam" if vst_road not in [int(s) for s in streams] else "fcam"
  except Exception as e:
    _log(f"检测主相机失败 ({e}), 按 fcam 处理")
  return "fcam"


def main() -> int:
  from openpilot.cereal import messaging
  from self_calibrator import CalibSession        # tools/calib/self_calibrator.py

  exit_now = False

  def _handler(sig, fr):
    nonlocal exit_now
    exit_now = True

  signal.signal(signal.SIGTERM, _handler)
  signal.signal(signal.SIGINT, _handler)

  params = Params()
  cam = detect_main_camera()
  ecam = cam == "ecam"
  ctl = AutoCalibController(params, cam=cam, log=_log)
  sm = messaging.SubMaster(["deviceState", "carState", "radarState", "modelV2"])
  _log(f"启动: 主相机={cam} ({ctl.key}), 自动写入={'开' if ctl.enabled() else '关'} (AutoCalib); "
       f"采样口径与手动标定一致")

  session = None
  last_log = 0.0
  enabled_prev = ctl.enabled()

  while not exit_now:
    # 回滚请求 (面板/命令行置位)
    try:
      if params.get_bool("AutoCalibRollback"):
        ctl.rollback()
    except Exception:
      pass

    # 开关状态变化要可见 (排查"我打开了怎么没写"时一眼看到)
    enabled_now = ctl.enabled()
    if enabled_now != enabled_prev:
      enabled_prev = enabled_now
      _log(f"AutoCalib {'已打开 (连续 3 趟一致即自动写入)' if enabled_now else '已关闭 (只给建议, 不动内参)'}")

    sm.update(SAMPLE_TIMEOUT_MS)
    onroad = bool(sm["deviceState"].started) if sm.seen["deviceState"] else False
    manual = params.get_bool("FcamLiveActive") or params.get_bool("WideCalibActive")

    if onroad and not manual:
      if session is None:
        session = CalibSession(ecam=ecam)
        last_log = 0.0
        _log("开始采样 (onroad)")
      session.feed(sm)
      now = time.monotonic()
      if now - last_log >= LOG_PERIOD_S:
        last_log = now
        rv, lw, cv = session.counts
        _log(f"采样中 rv={rv} lw={lw} cv={cv} 共={session.total}")
    else:
      if session is not None:
        total = session.total
        _log(f"收尾: 共 {total} 个样本" + (" (手动标定接管/offroad)" if not manual else ""))
        try:
          result = session.finish(ecam=ecam, params=params)
          ctl.consider(result)
        except Exception as e:
          cloudlog.exception(f"calibd: 收尾失败 {e}")
          _log(f"收尾失败: {e}")
        session = None
      else:
        time.sleep(IDLE_PERIOD_S)

  # SIGTERM: 有数据就先把这趟算完 (整车断电/manager 关停都走这条路)
  if session is not None and session.total > 0:
    _log(f"收到退出信号, 先收尾这趟 (共 {session.total} 个样本)")
    try:
      ctl.consider(session.finish(ecam=ecam, params=params))
    except Exception as e:
      _log(f"退出前收尾失败: {e}")
  _log("退出")
  return 0


if __name__ == "__main__":
  sys.exit(main())
