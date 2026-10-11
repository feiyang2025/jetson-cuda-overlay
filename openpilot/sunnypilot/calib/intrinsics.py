"""内参 (fl/cx/cy) 的读写 —— 标定 daemon 与 UI 面板共用的唯一实现。

为什么单独一个模块: 这两个消费方原来各写一份 (面板在 UI 里, 手动标定脚本在 tools/calib),
口径一旦分家就会出现"面板写进去的值和 daemon 写的语义不同"。这里只依赖 Params, 不碰 raylib,
所以 daemon 也能安全导入。

本 fork 这些键是 JSON 型: 必须 put(dict) 且 block=True
(python2cpp 只认 dict, 传 json 字符串直接 TypeError; 不 block 会立刻读不到)。
"""

from __future__ import annotations

import os

INTRINSICS_KEYS = {"fcam": "FcamIntrinsics", "ecam": "EcamIntrinsics"}
INTRINSICS_ROLLBACK_FLAG = "AutoCalibRollback"


def pjson(params, key: str) -> dict | None:
  """读 JSON 型 param (dict / JSON 字符串 / bytes 三种都兜; 空键或非 dict 返回 None)。

  本房的 C++ Params 对 JSON 型键: get() 直接返回 dict; 老实现/别的 fork 可能返回字符串。
  """
  v = params.get(key)
  if v is None:
    return None
  if isinstance(v, (bytes, bytearray)):
    v = v.decode(errors="replace")
  if isinstance(v, str):
    if not v.strip():
      return None
    try:
      import json
      v = json.loads(v)
    except Exception:
      return None
  return v if isinstance(v, dict) else None


_pjson = pjson      # 模块内部沿用旧名


def cam_dims(key: str) -> tuple[int, int]:
  """该相机流的输出尺寸 (与 launch_env.sh 的 CAM_WIDTH/HEIGHT 一致)。"""
  if key == "FcamIntrinsics":
    w = os.environ.get("ROAD_CAM_WIDTH") or os.environ.get("CAM_WIDTH")
    h = os.environ.get("ROAD_CAM_HEIGHT") or os.environ.get("CAM_HEIGHT")
  else:
    w = os.environ.get("WIDE_CAM_WIDTH") or os.environ.get("CAM_WIDTH")
    h = os.environ.get("WIDE_CAM_HEIGHT") or os.environ.get("CAM_HEIGHT")
  try:
    return int(w or 1344), int(h or 760)
  except (TypeError, ValueError):
    return 1344, 760


def read_intrinsics(params, key: str) -> dict:
  return dict(_pjson(params, key) or {})


def effective_fl(params, key: str) -> float:
  """顶层 fl (即该相机流当前生效的焦距)。没有就返回 0。"""
  try:
    return float(read_intrinsics(params, key).get("fl") or 0.0)
  except (TypeError, ValueError):
    return 0.0


def write_intrinsics(params, key: str, fl: float, clear_calibration: bool = True) -> dict:
  """写内参: 顶层 fl/cx/cy + calibrations["<W>x<H>"], 可选清 CalibrationParams。

  返回写入后的文档 (便于调用方/备份使用)。
  """
  w, h = cam_dims(key)
  intr = read_intrinsics(params, key)
  intr["fl"] = float(fl)
  intr["cx"] = w / 2.0
  intr["cy"] = h / 2.0
  calibs = dict(intr.get("calibrations") or {})
  calibs[f"{w}x{h}"] = {"fl": float(fl), "cx": w / 2.0, "cy": h / 2.0}
  intr["calibrations"] = calibs
  params.put(key, intr, block=True)

  if clear_calibration:
    # 换了内参, 旧的俯仰/偏航外参不再成立, 清掉让 calibrationd 重新标
    params.remove("CalibrationParams")
  return intr


def restore_intrinsics(params, key: str, doc: dict, clear_calibration: bool = True) -> None:
  """回滚: 把整份文档写回去。"""
  params.put(key, dict(doc), block=True)
  if clear_calibration:
    params.remove("CalibrationParams")
