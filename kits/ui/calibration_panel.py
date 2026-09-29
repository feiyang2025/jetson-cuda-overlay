"""摄像头标定面板（开发者页）—— 把 sp 的 Qt developer_panel 4 个标定按钮移植到新 UI。

四个入口（与 sp 的 ButtonControl 一一对应）:
  FCAM Calibration        [RUN]        扫行车日志算 FCAM 内参 → 确认后写 FcamIntrinsics
  FCAM Live Calibration   [START/STOP] 行车中实时收样本, STOP 时算并应用
  Wide Calibration        [START/STOP] 把 ECAM 顶到 road 流, 实时收 ECAM 样本, STOP 时算并应用
  Camera Calibration      [VIEW]       看当前内参/外参

设计约束（都是踩过的坑）:
- 子进程 stdout 由后台线程读, 线程里只写 self._pending_line[name],
  真正 set_description 放到 UI 线程的 update() 里做 —— 别从工作线程碰渲染对象。
- road/wide 换流与内参互换由 camerad 自己轮询 WideCalibMode 完成
  (见 kits/camerad/camerad.py), 这里只负责翻 param 与应用结果, 不重启 camerad。
- 写内参必须同时写 calibrations["<W>x<H>"]: 全栈按分辨率键取值, 只写顶层 fl 在
  换分辨率时会退化成默认值。
- 清 CalibrationParams 必须在 offroad 窗口内做 (calibrationd 是 only_onroad,
  跑着的时候会把旧值写回来)。sp 用 OffroadMode 造这个窗口, 本 UI 的 hardwared
  同样认 OffroadMode (system/hardware/hardwared.py)。
"""
import datetime
import json
import os
import subprocess
import sys
import threading
import time

from openpilot.common.params import Params
from openpilot.common.hardware.hw import Paths
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.widgets import DialogResult
from openpilot.system.ui.widgets.confirm_dialog import ConfirmDialog
from openpilot.system.ui.widgets.list_view import button_item
from openpilot.system.ui.sunnypilot.widgets.html_render import HtmlModalSP

# 与 sp developer_panel.cc 对齐的默认焦距（用于没标过时的提示值）
DEFAULT_FL_FCAM = 2520
DEFAULT_FL_ECAM = 670

# 换流后短暂等待: camerad 轮询 WideCalibMode 的周期是 ~2s
SWAP_SETTLE_S = 3.0


def _find_op_dir() -> str:
  """找到带 tools/calib 的项目根（从 cwd 向上, 再看 env）。"""
  cands = []
  env_root = os.environ.get("OPENPILOT_ROOT") or os.environ.get("PYTHONPATH", "").split(":")[0]
  if env_root:
    cands.append(env_root)
  d = os.getcwd()
  while True:
    cands.append(d)
    parent = os.path.dirname(d)
    if parent == d:
      break
    d = parent
  here = os.path.dirname(os.path.abspath(__file__))
  for _ in range(8):  # 模块自身位置再往上兜底
    cands.append(here)
    here = os.path.dirname(here)
  for c in cands:
    if c and os.path.isdir(os.path.join(c, "tools", "calib")):
      return c
  return os.getcwd()


def _find_python(op_dir: str) -> str:
  venv = os.path.join(op_dir, ".venv", "bin", "python3")
  return venv if os.path.exists(venv) else sys.executable


def _cam_dims(key: str) -> tuple[int, int]:
  """取该相机流的输出尺寸（与 launch_env.sh 的 CAM_WIDTH/HEIGHT 一致）。"""
  if key == "FcamIntrinsics":
    w = os.environ.get("ROAD_CAM_WIDTH") or os.environ.get("CAM_WIDTH")
    h = os.environ.get("ROAD_CAM_HEIGHT") or os.environ.get("CAM_HEIGHT")
  else:
    w = os.environ.get("WIDE_CAM_WIDTH") or os.environ.get("CAM_WIDTH")
    h = os.environ.get("WIDE_CAM_HEIGHT") or os.environ.get("CAM_HEIGHT")
  try:
    return int(w), int(h)
  except (TypeError, ValueError):
    return 1920, 1080


def _pstr(p: Params, key: str) -> str:
  v = p.get(key)
  if v is None:
    return ""
  return v.decode() if isinstance(v, bytes) else str(v)


def _current_fl(p: Params, key: str, default_fl: int) -> int:
  try:
    data = json.loads(_pstr(p, key) or "{}")
    return int(data.get("fl", default_fl))
  except Exception:
    return default_fl


def _extract_fl(obj: dict) -> int:
  """结果 JSON 里 fl 建议值；scan 版在 fused 下, wide_calibrator 版在顶层。"""
  if isinstance(obj.get("fused"), dict):
    return int(obj["fused"].get("fl_suggested") or 0)
  try:
    return int(obj.get("fl_suggested") or 0)
  except (TypeError, ValueError):
    return 0


def write_intrinsics(p: Params, key: str, fl: int, clear_calibration: bool = True) -> None:
  """写内参: 顶层 fl/cx/cy + calibrations[<W>x<H>], 可选清 CalibrationParams。"""
  w, h = _cam_dims(key)
  intr = {}
  existing = _pstr(p, key)
  if existing:
    try:
      doc = json.loads(existing)
      if isinstance(doc, dict):
        intr = doc
    except Exception:
      intr = {}
  intr["fl"] = fl
  intr["cx"] = w / 2.0
  intr["cy"] = h / 2.0
  calibs = intr.get("calibrations") or {}
  calibs[f"{w}x{h}"] = {"fl": fl, "cx": w / 2.0, "cy": h / 2.0}
  intr["calibrations"] = calibs
  p.put(key, json.dumps(intr))

  if clear_calibration:
    # 换了内参, 旧的俯仰/偏航外参不再成立, 清掉让 calibrationd 重新标
    p.remove("CalibrationParams")


class CalibrationPanel:
  """4 个标定项 + 子进程管理。UI 线程只调 build_items() / update()。"""

  def __init__(self, params: Params | None = None):
    self.params = params or Params()
    self.op_dir = _find_op_dir()
    self.python = _find_python(self.op_dir)
    self._procs: dict[str, subprocess.Popen] = {}
    self._threads: dict[str, threading.Thread] = {}
    self._lock = threading.Lock()
    self._pending_line: dict[str, str] = {}
    self._pending_done: dict[str, int] = {}
    self._flash: dict[str, tuple[str, float]] = {}
    self._msg = ""

    self.items = {
      "fcam": button_item(tr("FCAM Calibration"), tr("RUN"),
                          tr("Analyze radar-vision lead distance, lane width, and IMU-model curvature "
                             "to verify camera intrinsic calibration. Requires recent drive logs."),
                          callback=self.on_fcam_scan),
      "fcam_live": button_item(tr("FCAM Live Calibration"), tr("START"),
                               tr("Toggle ON before driving: collects radar-vision lead distance, lane width "
                                  "and IMU-model curvature during the drive. Computes FCAM calibration on "
                                  "shutdown and saves the result."),
                               callback=self.on_fcam_live),
      "wide": button_item(tr("Wide Calibration"), tr("START"),
                          tr("Switch the wide-angle camera onto the road stream for ECAM calibration. "
                             "START resets extrinsic calibration and starts live sample collection "
                             "(needs rv>=30 + lw>=10 + cv>=10). STOP computes ECAM intrinsics; "
                             "if data is insufficient nothing is saved."),
                          callback=self.on_wide_calib),
      "view": button_item(tr("Camera Calibration"), tr("VIEW"),
                          tr("View current FCAM and ECAM intrinsic focal lengths and the "
                             "extrinsic calibration (roll/pitch/yaw)."),
                          callback=self.on_view_calib),
    }

  # ------------------------------------------------------------------ 子进程
  def _running(self, name: str) -> bool:
    p = self._procs.get(name)
    return p is not None and p.poll() is None

  def _spawn(self, name: str, args: list[str]) -> None:
    if self._running(name):
      return
    argv = [self.python, "-u"] + args
    print(f"[calib] spawn: {' '.join(argv)}", flush=True)
    proc = subprocess.Popen(argv, cwd=self.op_dir, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    self._procs[name] = proc

    def _reader():
      try:
        for line in proc.stdout:
          line = line.strip()
          if line:
            with self._lock:
              self._pending_line[name] = line[:100]
      except Exception:
        pass
      rc = proc.wait()
      with self._lock:
        self._pending_done[name] = rc

    t = threading.Thread(target=_reader, name=f"calib-{name}", daemon=True)
    self._threads[name] = t
    t.start()

  def _terminate(self, name: str, timeout: float = 30.0) -> int | None:
    """SIGTERM 让脚本走完'计算并写结果'的收尾路径 (run_live 靠 SIGTERM 触发)。"""
    proc = self._procs.get(name)
    if proc is None:
      return None
    if proc.poll() is None:
      proc.terminate()
      try:
        proc.wait(timeout=timeout)
      except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=3)
    return proc.returncode

  def _set_flash(self, name: str, text: str, seconds: float = 5.0) -> None:
    self._flash[name] = (text, time.monotonic() + seconds)

  def _flash_now(self, name: str) -> str | None:
    f = self._flash.get(name)
    if f is None:
      return None
    if time.monotonic() > f[1]:
      self._flash.pop(name, None)
      return None
    return f[0]

  # ------------------------------------------------------------------ 应用结果
  def _offer_intrinsics(self, key: str, camera_name: str, fl: int, default_fl: int, on_done=None) -> None:
    cur = _current_fl(self.params, key, default_fl)
    if fl <= 0:
      self._set_flash("fcam" if key == "FcamIntrinsics" else "wide", tr("NO DATA"))
      return

    def _cb(result: DialogResult):
      if result == DialogResult.CONFIRM:
        write_intrinsics(self.params, key, fl)
        print(f"[calib] {key} <- fl={fl} (was {cur}), CalibrationParams cleared", flush=True)
      if on_done:
        on_done()

    msg = (f"{camera_name} {tr('calibration result')}: fl={fl} ({tr('current')}: {cur})\n"
           f"{tr('Update intrinsics? Extrinsic calibration will be reset.')}")
    gui_app.push_widget(ConfirmDialog(msg, tr("Update"), rich=False, callback=_cb))

  # ------------------------------------------------------------------ 回调
  def on_fcam_scan(self) -> None:
    if self._running("fcam"):
      return
    script = os.path.join(self.op_dir, "tools", "calib", "self_calibrator.py")
    if not os.path.exists(script):
      self._set_flash("fcam", tr("script not found"))
      return
    out = os.path.join(self.op_dir, "calib_result.json")
    self._spawn("fcam", [script, "--scan", "--base-dir", Paths.log_root(), "--output", out])
    self._msg = ""

  def on_fcam_live(self) -> None:
    active = self.params.get_bool("FcamLiveActive")
    if active:
      # OFF: 先翻 param 让脚本退出采集循环, 再 SIGTERM 触发计算收尾
      self.params.put_bool("FcamLiveActive", False)
      self.items["fcam_live"].action_item.set_text(tr("COMPUTING..."))
      self._terminate("fcam_live")
      obj = self._read_result_param("FcamCalibResult")
      fl = _extract_fl(obj) if obj else 0
      self._set_flash("fcam_live", f"fl={fl}" if fl > 0 else tr("NO DATA"))
      if fl > 0:
        self._offer_intrinsics("FcamIntrinsics", "FCAM", fl, DEFAULT_FL_FCAM,
                               on_done=lambda: self.items["fcam_live"].action_item.set_text(tr("START")))
      else:
        self.items["fcam_live"].action_item.set_text(tr("START"))
    else:
      script = os.path.join(self.op_dir, "tools", "calib", "self_calibrator.py")
      if not os.path.exists(script):
        self._set_flash("fcam_live", tr("script not found"))
        return
      out = os.path.join(self.op_dir, "fcam_live_result.json")
      self.params.put_bool("FcamLiveActive", True, block=True)
      self._spawn("fcam_live", [script, "--live", "--output", out])
      self.items["fcam_live"].action_item.set_text(tr("STOP"))

  def on_wide_calib(self) -> None:
    active = self.params.get_bool("WideCalibActive")
    if active:
      self.params.put_bool("WideCalibActive", False)
      self.items["wide"].action_item.set_text(tr("COMPUTING..."))
      self._terminate("wide")
      obj = self._read_result_param("WideCalibResult")
      fl = _extract_fl(obj) if obj else 0
      self._set_flash("wide", f"fl={fl}" if fl > 0 else tr("NO DATA"))
      # 先写新 ECAM, 再关 WideCalibMode: camerad 的还原路径会认出 Ecam 已被改过而保留新值
      if fl > 0:
        self._offer_intrinsics("EcamIntrinsics", "ECAM", fl, DEFAULT_FL_ECAM,
                               on_done=lambda: self._finish_wide())
      else:
        self._finish_wide()
    else:
      script = os.path.join(self.op_dir, "tools", "calib", "self_calibrator.py")
      if not os.path.exists(script):
        self._set_flash("wide", tr("script not found"))
        return
      # 进入: 造 offroad 窗口清外参 -> 翻 WideCalibMode (camerad ~2s 内换流) -> 起 ECAM 采集
      self.params.put_bool("WideCalibMode", True, block=True)
      self.params.put_bool("WideCalibActive", True, block=True)
      self.params.put_bool("OffroadMode", True, block=True)
      self.items["wide"].action_item.set_text(tr("SWITCHING..."))
      self.params.remove("WideCalibIntrinsicsFcamBackup")
      self.params.remove("WideCalibIntrinsicsEcamBackup")
      self.params.remove("WideCalibResult")

      def _after_offroad():
        time.sleep(SWAP_SETTLE_S)
        self.params.remove("CalibrationParams")
        self.params.put_bool("OffroadMode", False, block=True)
        print("[calib] wide: CalibrationParams cleared, back onroad; camerad polls WideCalibMode", flush=True)
        out = os.path.join(self.op_dir, "ecam_calib_result.json")
        self._spawn("wide", [script, "--live", "--ecam", "--output", out])
        self.items["wide"].action_item.set_text(tr("STOP"))

      threading.Thread(target=_after_offroad, name="calib-wide-switch", daemon=True).start()

  def _finish_wide(self) -> None:
    """收尾: 关 WideCalibMode, camerad 自己还原 Fcam/Ecam 内参。"""
    self.params.put_bool("WideCalibMode", False, block=True)
    self.items["wide"].action_item.set_text(tr("START"))

  def on_view_calib(self) -> None:
    lines = []
    for key, name in (("FcamIntrinsics", "FCAM"), ("EcamIntrinsics", "ECAM")):
      raw = _pstr(self.params, key)
      if not raw:
        lines.append(f"<b>{name}</b>: {tr('not calibrated')}")
      else:
        try:
          d = json.loads(raw)
        except Exception:
          d = {}
        fl = d.get("fl", "-")
        lines.append(f"<b>{name}</b>: fl={fl} cx={d.get('cx', '-')} cy={d.get('cy', '-')}")
        calibs = d.get("calibrations") or {}
        for res, entry in calibs.items():
          lines.append(f"&nbsp;&nbsp;{res}: fl={entry.get('fl', '-')}")
    extra = self._extrinsic_summary()
    if extra:
      lines.append(extra)
    text = "<br>".join(lines) if lines else tr("No calibration data")
    gui_app.push_widget(HtmlModalSP(text=text))

  def _extrinsic_summary(self) -> str:
    raw = self.params.get("CalibrationParams")
    if not raw:
      return f"<b>{tr('Extrinsics')}</b>: {tr('not calibrated')}"
    try:
      from openpilot.cereal import log as logmod
      evt = logmod.Event.from_bytes(raw)
      cp = evt.liveCalibrationData
      rpy = list(cp.rpyCalib)
      return (f"<b>{tr('Extrinsics')}</b>: roll={rpy[0]:.3f} pitch={rpy[1]:.3f} yaw={rpy[2]:.3f} "
              f"(valid={cp.validBlocks})")
    except Exception as e:
      return f"<b>{tr('Extrinsics')}</b>: {tr('unreadable')} ({e})"

  # ------------------------------------------------------------------ 每帧刷新
  def _read_result_param(self, key: str) -> dict | None:
    raw = _pstr(self.params, key)
    if not raw:
      return None
    try:
      return json.loads(raw)
    except Exception:
      return None

  def _consume_finished(self) -> None:
    with self._lock:
      done = dict(self._pending_done)
      self._pending_done.clear()
      line = dict(self._pending_line)
    for name, rc in done.items():
      self._procs.pop(name, None)
      if name == "fcam":
        self.items["fcam"].action_item.set_text(tr("RUN"))
        if rc == 0:
          try:
            with open(os.path.join(self.op_dir, "calib_result.json")) as f:
              obj = json.load(f)
          except Exception:
            obj = None
          fl = _extract_fl(obj) if obj else 0
          self._set_flash("fcam", f"fl={fl}" if fl > 0 else tr("NO DATA"))
          if fl > 0:
            self._offer_intrinsics("FcamIntrinsics", "FCAM", fl, DEFAULT_FL_FCAM)
        else:
          self._set_flash("fcam", tr("FAILED"))
      elif name in ("fcam_live", "wide"):
        pass  # STOP 路径自己收尾
    for name, text in line.items():
      item = self.items.get(name)
      if item is not None:
        item.set_description(text)

  def update(self) -> None:
    """UI 线程每帧调用。"""
    self._consume_finished()
    if self._running("fcam") and not self._flash_now("fcam"):
      self.items["fcam"].action_item.set_text(tr("RUNNING..."))

    fcam_live = self.params.get_bool("FcamLiveActive")
    if self._running("fcam_live") and fcam_live:
      self.items["fcam_live"].action_item.set_text(tr("STOP"))
    elif not self._running("fcam_live"):
      f = self._flash_now("fcam_live")
      self.items["fcam_live"].action_item.set_text(f or tr("START"))

    wide_active = self.params.get_bool("WideCalibActive")
    if self._running("wide") and wide_active:
      self.items["wide"].action_item.set_text(tr("STOP"))
    elif not self._running("wide"):
      f = self._flash_now("wide")
      self.items["wide"].action_item.set_text(f or tr("START"))

    for name in ("fcam", "fcam_live", "wide"):
      f = self._flash_now(name)
      if f:
        self.items[name].action_item.set_text(f)
