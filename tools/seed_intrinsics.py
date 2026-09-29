#!/usr/bin/env python3
"""按本树 `common/transformations/camera.py` 的厂商参考值播 Fcam/EcamIntrinsics。

来源与规则都取自目标树自己（不是我编的）:
  common/transformations/camera.py:121-131
    fcam = road  : vendor pinhole fl ~= 961.76 @1920x1080
    ecam = wide  : vendor fl 1740.0 @1920x1080
    fl 换分辨率 = base_fl * (width / 1920)   ← 同 `_camera_config_from_params`
  写入三个分辨率键, 让 1344x760 / 1928x1208 / 1920x1080 都走精确匹配
  (精确匹配优先于"按第一个键缩放", 见 _read_calib_from_params)。

已有非空值不会被覆盖, 除非显式 --force —— 免得把真标定结果冲掉。
写 JSON 型 param 必须 put(dict) 且 block=True(传字符串 TypeError + 不 block 读不到)。

用法: python3 tools/seed_intrinsics.py <树根> [--force]
"""
import os
import sys

TREE = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
FORCE = "--force" in sys.argv
sys.path.insert(0, TREE)
sys.path.insert(0, os.path.join(TREE, "openpilot"))

from openpilot.common.params import Params  # noqa: E402

# 厂商参考焦距 @1920x1080 (camera.py 注释里的原值)
BASE_FL = {"FcamIntrinsics": 961.76, "EcamIntrinsics": 1740.0}
BASE_W = 1920
RESOLUTIONS = [(1344, 760), (1928, 1208), (1920, 1080)]


def build(base_fl: float, w: int, h: int) -> dict:
  fl = round(base_fl * (w / BASE_W), 2)
  return {"fl": fl, "cx": w / 2.0, "cy": h / 2.0}


params = Params()
for key, base_fl in BASE_FL.items():
  cur = params.get(key)
  if cur not in (None, "", {}) and not FORCE:
    print(f"[seed] {key} 已有值, 跳过 (要覆盖加 --force): {cur}")
    continue
  calibs = {f"{w}x{h}": build(base_fl, w, h) for w, h in RESOLUTIONS}
  primary = calibs["1344x760"]
  doc = {"fl": primary["fl"], "cx": primary["cx"], "cy": primary["cy"], "calibrations": calibs}
  params.put(key, doc, block=True)
  back = params.get(key)
  print(f"[seed] {key} 写入 (base {base_fl} @1920x1080):")
  for r, e in calibs.items():
    print(f"        {r}: fl={e['fl']} cx={e['cx']} cy={e['cy']}")
  print(f"       读回校验: {'OK' if back == doc else '不一致! ' + repr(back)}")

print()
print("[seed] 完成。这些值只是能跑的起点(厂商参考), 真正准的要靠 tools/calib/ 标定后写回。")
