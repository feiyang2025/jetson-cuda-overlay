"""自动标定的决策层 (纯逻辑, params 注入 → 可离线单测)。

分工:
  * 采样与融合在 tools/calib/self_calibrator.py (CalibSession), 手动/自动共用;
  * 本模块只回答"这一趟的结果要不要采纳、要不要自动写进内参、要不要回滚";
  * calibd.py 只是把两者串起来的常驻外壳。

安全立场 (为什么不是"算完就写"):
  fl 直接决定视觉测距 → 跟车距离、AEB 触发点。所以自动写入必须同时满足:
    1. 融合已采纳 (≥2 个方法一致, 见 CalibSession);
    2. 修正幅度在 ±MAX_DELTA_PCT 以内 (超出通常说明别的东西坏了: 相机没装好/张冠李戴);
    3. **连续 MIN_RUNS 趟**结果互相一致 (±CONSISTENCY_PCT) —— 单趟可能是偶发;
    4. 开关 AutoCalib 打开; 写前自动备份旧内参, 支持一键回滚。
  不满足就给"只写结果 + 建议", 不动内参 —— 人看一眼就能点确认, 谁都不必记流程。
"""

from __future__ import annotations

import os
import time

from openpilot.sunnypilot.calib.intrinsics import (
  INTRINSICS_ROLLBACK_FLAG,
  effective_fl,
  read_intrinsics,
  restore_intrinsics,
  write_intrinsics,
)

# ---- 门限 ----
# 触发以**里程**为主: 同一台相机累计跑够 MIN_DISTANCE_KM 才可能写, 期间把所有"已采纳"的
# 趟数汇成一个池子取加权中位数 —— 这样"跑多少 km 自动写"就是可预期、可解释的, 不需要人
# 盯着开关。里程按 vEgo 积分, 存在 AutoCalibHistory 里, 重启不清零。
MIN_DISTANCE_KM = 30.0      # 累计多少 km 才有资格写 (可被 env CALIB_MIN_KM 覆盖)
MIN_RUNS = 2                # 池子里至少几趟有效数据 (一趟 10-20km, 30km 通常 2-3 趟)
CONSISTENCY_PCT = 1.5       # 池子里各趟建议值彼此允许的离散 (%)
MAX_DELTA_PCT = 15.0        # 修正幅度上限 (%)
MIN_APPLY_DELTA_PCT = 2.0   # 偏差小于这个就不动 (属于测量噪声)
HISTORY_LEN = 20            # 历史保留条数

PARAM_ENABLE = "AutoCalib"
PARAM_HISTORY = "AutoCalibHistory"
PARAM_BACKUP = "AutoCalibBackup"


def _now() -> str:
  return time.strftime("%Y-%m-%d %H:%M:%S")


def _weighted_median(vals: list, weights: list) -> float:
  """按样本数加权的中位数 —— "最合适"的那个值 (比取最后一趟稳)。"""
  pairs = sorted(zip(vals, [max(int(w), 1) for w in weights]))
  total = sum(w for _, w in pairs)
  acc = 0
  for v, w in pairs:
    acc += w
    if acc >= total / 2:
      return float(v)
  return float(pairs[-1][0])


class AutoCalibController:
  def __init__(self, params, cam: str = "fcam", log=print):
    assert cam in ("fcam", "ecam"), cam
    self.params = params
    self.cam = cam
    self.key = "EcamIntrinsics" if cam == "ecam" else "FcamIntrinsics"
    self.log = log

  # ------------------------------------------------------------------ params
  def enabled(self) -> bool:
    """自动写入是否开启。

    默认**开** (键不存在 = 自动): 装好就该自己收敛, 不需要人知道有这条流程。
    要关就显式写 False (kill switch), 那时只收数据/只给建议。
    """
    try:
      v = self.params.get(PARAM_ENABLE)
    except Exception:
      return True
    if v is None or v == "":
      return True
    return bool(self.params.get_bool(PARAM_ENABLE))

  def min_distance_km(self) -> float:
    try:
      return float(os.environ.get("CALIB_MIN_KM") or MIN_DISTANCE_KM)
    except (TypeError, ValueError):
      return MIN_DISTANCE_KM

  def history(self) -> list:
    try:
      h = self.params.get(PARAM_HISTORY)
      if isinstance(h, list):
        return h
      if isinstance(h, dict):
        return list(h.get("runs") or [])
    except Exception:
      pass
    return []

  def _save_history(self, hist: list, state: dict | None = None) -> None:
    doc = {"runs": hist[-HISTORY_LEN:]}
    doc.update(state if state is not None else self.state())
    try:
      self.params.put(PARAM_HISTORY, doc, block=True)
    except Exception as e:
      self.log(f"[calibd] 写历史失败: {e}")

  def state(self) -> dict:
    """与里程相关的持久状态 {auto: {distance_km, since, written}}。"""
    try:
      d = self.params.get(PARAM_HISTORY)
      if isinstance(d, dict) and isinstance(d.get("auto"), dict):
        return {"auto": d["auto"]}
    except Exception:
      pass
    return {}

  def distance_km(self) -> float:
    return float((self.state().get("auto") or {}).get("distance_km") or 0.0)

  def add_distance(self, km: float) -> float:
    """累加本窗口的行驶里程 (calibd 在每个 onroad tick 调)。"""
    if km <= 0:
      return self.distance_km()
    st = self.state()
    auto = dict(st.get("auto") or {})
    auto["distance_km"] = round(float(auto.get("distance_km") or 0.0) + km, 3)
    auto.setdefault("since", _now())
    st["auto"] = auto
    self._save_history(self.history(), st)
    return auto["distance_km"]

  def _reset_window(self, written: dict) -> None:
    """写成功之后: 里程清零重新累计, 保留历史与已写入记录。"""
    st = {"auto": {"distance_km": 0.0, "since": _now(), "written": written}}
    self._save_history(self.history(), st)

  # ------------------------------------------------------------------ 决策
  def consider(self, result: dict) -> dict:
    """把一趟的结果交给决策层: 记历史 + 决定动作。返回这条历史条目。"""
    fused = (result or {}).get("fused") or {}
    fl_cur = float(fused.get("fl_current") or effective_fl(self.params, self.key) or 0.0)
    fl_sug = fused.get("fl_suggested")
    entry = {
      "ts": _now(),
      "cam": self.cam,
      "accepted": bool(fused.get("accepted")),
      "confidence": fused.get("confidence"),
      "n_methods": fused.get("n_methods"),
      "spread_pct": fused.get("spread_pct"),
      "samples": fused.get("total_observations"),
      "fl_current": round(fl_cur, 2) if fl_cur else None,
      "fl_suggested": round(float(fl_sug), 2) if fl_sug else None,
      "delta_pct": fused.get("fl_delta_pct"),
      "reason": fused.get("reason") or "",
      "action": "none",
    }

    if not entry["accepted"]:
      entry["action"] = "reject"
      # 没通过融合门限: 结果 param 已由 CalibSession 清空, 只记历史
    elif not fl_cur or not fl_sug:
      entry["action"] = "reject"
      entry["reason"] = entry["reason"] or "缺 fl_current/fl_suggested"
    elif abs(entry["delta_pct"] or 0.0) < MIN_APPLY_DELTA_PCT:
      entry["action"] = "no_change"
    elif not self.enabled():
      entry["action"] = "suggest_only"     # 面板可见 (结果 param 已写), 但不动内参
    else:
      entry["action"] = self._maybe_apply(fl_cur, float(fl_sug), entry) and "applied" or "pending"

    hist = self.history()
    hist.append(entry)
    self._save_history(hist)
    self.log(f"[calibd] 结果 {self.cam}: fl={entry['fl_current']}→{entry['fl_suggested']} "
             f"({entry['delta_pct']}%) 置信={entry['confidence']} "
             f"里程={self.distance_km():.1f}/{self.min_distance_km():.0f}km 动作={entry['action']}"
             + (f" — {entry['reason']}" if entry["reason"] else ""))
    return entry

  def _maybe_apply(self, fl_cur: float, fl_sug: float, entry: dict) -> bool:
    """里程触发: 累计够 km + 池内各趟一致 → 写"最合适"的值 (加权中位数)。"""
    delta = abs((fl_sug / fl_cur - 1.0) * 100.0)
    if delta > MAX_DELTA_PCT:
      entry["reason"] = f"修正幅度 {delta:.1f}% 超上限 {MAX_DELTA_PCT}% → 不自动写 (先查相机安装/型号)"
      return False

    need_km = self.min_distance_km()
    done_km = self.distance_km()
    pool = [h for h in self.history() if h.get("accepted") and h.get("cam") == self.cam
            and h.get("fl_suggested")]
    vals = [float(h["fl_suggested"]) for h in pool] + [fl_sug]
    weights = [int(h.get("samples") or 0) for h in pool] + [int(entry.get("samples") or 0)]

    if done_km < need_km:
      entry["reason"] = f"里程 {done_km:.1f}/{need_km:.0f} km (差 {need_km - done_km:.1f} km), 继续跑"
      return False
    if len(vals) < MIN_RUNS:
      entry["reason"] = f"里程够了但有效趟数不足 ({len(vals)}/{MIN_RUNS}) → 再跑一趟"
      return False
    mean = sum(vals) / len(vals)
    spread = (max(vals) - min(vals)) / mean * 100.0
    if spread > CONSISTENCY_PCT:
      entry["reason"] = f"池内 {len(vals)} 趟离散 {spread:.2f}% > {CONSISTENCY_PCT}% → 数据还不稳, 继续收"
      return False

    apply_fl = _weighted_median(vals, weights)
    self.apply(apply_fl, reason=f"累计 {done_km:.1f} km / {len(vals)} 趟一致 (离散 {spread:.2f}%)")
    entry["applied_fl"] = round(apply_fl, 2)
    entry["applied_km"] = round(done_km, 1)
    entry["applied_spread_pct"] = round(spread, 2)
    self._reset_window({"ts": _now(), "fl": round(apply_fl, 2), "km": round(done_km, 1),
                        "runs": len(vals)})
    return True

  # ------------------------------------------------------------------ 执行
  def apply(self, fl: float, reason: str = "") -> dict:
    """写内参 (先备份旧值 -> 清 CalibrationParams)。"""
    old = read_intrinsics(self.params, self.key)
    try:
      self.params.put(PARAM_BACKUP, {"ts": _now(), "key": self.key, "reason": reason,
                                     "doc": old, "fl": old.get("fl")}, block=True)
    except Exception as e:
      self.log(f"[calibd] 备份失败, 放弃自动写: {e}")
      return {"ok": False, "error": str(e)}
    write_intrinsics(self.params, self.key, fl, clear_calibration=True)
    self.log(f"[calibd] 已自动写入 {self.key}: fl={old.get('fl')} → {fl} ({reason}); "
             f"旧值已备份, 需要回滚就把 {INTRINSICS_ROLLBACK_FLAG} 置 1")
    return {"ok": True, "key": self.key, "fl": fl, "previous_fl": old.get("fl")}

  def rollback(self) -> dict:
    """把 AutoCalibBackup 里的内参写回去, 并清掉请求标志。"""
    try:
      bak = self.params.get(PARAM_BACKUP)
      if isinstance(bak, str):
        import json
        bak = json.loads(bak)
    except Exception:
      bak = None
    if not isinstance(bak, dict) or not bak.get("doc"):
      self.params.put_bool(INTRINSICS_ROLLBACK_FLAG, False)
      return {"ok": False, "error": "没有可用的备份"}
    restore_intrinsics(self.params, bak.get("key") or self.key, bak["doc"], clear_calibration=True)
    self.params.put_bool(INTRINSICS_ROLLBACK_FLAG, False)
    self.log(f"[calibd] 已回滚 {bak.get('key')}: fl ← {bak.get('fl')} (备份于 {bak.get('ts')})")
    return {"ok": True, "key": bak.get("key"), "fl": bak.get("fl")}
