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

import time

from openpilot.sunnypilot.calib.intrinsics import (
  INTRINSICS_ROLLBACK_FLAG,
  effective_fl,
  read_intrinsics,
  restore_intrinsics,
  write_intrinsics,
)

# ---- 门限 (与 tools/calib 的融合门限叠加使用) ----
MIN_RUNS = 3                # 连续多少趟一致才自动写
CONSISTENCY_PCT = 1.5       # 这几趟的建议值彼此允许的离散 (%)
MAX_DELTA_PCT = 15.0        # 单趟修正幅度上限 (%)
MIN_APPLY_DELTA_PCT = 2.0   # 偏差小于这个就不动 (属于测量噪声)
HISTORY_LEN = 20            # 历史保留条数

PARAM_ENABLE = "AutoCalib"
PARAM_HISTORY = "AutoCalibHistory"
PARAM_BACKUP = "AutoCalibBackup"


def _now() -> str:
  return time.strftime("%Y-%m-%d %H:%M:%S")


class AutoCalibController:
  def __init__(self, params, cam: str = "fcam", log=print):
    assert cam in ("fcam", "ecam"), cam
    self.params = params
    self.cam = cam
    self.key = "EcamIntrinsics" if cam == "ecam" else "FcamIntrinsics"
    self.log = log

  # ------------------------------------------------------------------ params
  def enabled(self) -> bool:
    try:
      return bool(self.params.get_bool(PARAM_ENABLE))
    except Exception:
      return False

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

  def _save_history(self, hist: list) -> None:
    try:
      self.params.put(PARAM_HISTORY, {"runs": hist[-HISTORY_LEN:]}, block=True)
    except Exception as e:
      self.log(f"[calibd] 写历史失败: {e}")

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
             f"({entry['delta_pct']}%) 置信={entry['confidence']} 动作={entry['action']}"
             + (f" — {entry['reason']}" if entry["reason"] else ""))
    return entry

  def _maybe_apply(self, fl_cur: float, fl_sug: float, entry: dict) -> bool:
    """滞回检查: 连续 MIN_RUNS 趟一致才写。返回是否真的写了。"""
    delta = abs((fl_sug / fl_cur - 1.0) * 100.0)
    if delta > MAX_DELTA_PCT:
      entry["reason"] = f"修正幅度 {delta:.1f}% 超上限 {MAX_DELTA_PCT}% → 不自动写 (先查相机安装/型号)"
      return False

    recent = [h for h in self.history() if h.get("accepted") and h.get("cam") == self.cam
              and h.get("fl_suggested")]
    recent = recent[-(MIN_RUNS - 1):]          # 含本趟共 MIN_RUNS 趟
    if len(recent) < MIN_RUNS - 1:
      entry["reason"] = f"还需要 {MIN_RUNS - len(recent) - 1} 趟一致的结果才自动写 (已有 {len(recent)} 趟)"
      return False
    vals = [float(h["fl_suggested"]) for h in recent] + [fl_sug]
    spread = (max(vals) - min(vals)) / (sum(vals) / len(vals)) * 100.0
    if spread > CONSISTENCY_PCT:
      entry["reason"] = f"最近 {len(vals)} 趟离散 {spread:.2f}% > {CONSISTENCY_PCT}% → 再等等"
      return False

    apply_fl = sorted(vals)[len(vals) // 2]    # 取中位数, 别用最后一趟
    self.apply(apply_fl, reason=f"连续 {len(vals)} 趟一致 (离散 {spread:.2f}%)")
    entry["applied_fl"] = round(apply_fl, 2)
    entry["applied_spread_pct"] = round(spread, 2)
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
