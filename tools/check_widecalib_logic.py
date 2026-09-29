#!/usr/bin/env python3
"""WideCalibMode 纯逻辑自检 (不碰相机/msgq, 可以和别的树同时在跑)。

覆盖两件事:
  1. _wc_output_of: road <-> wide 的输出映射 (换流只改输出身份)
  2. _swap_calib_intrinsics: 进/出时的内参互换与还原语义
     (含 sp 那条"Ecam 被标定改过就保留新值"的分支)

用法: python3 tools/check_widecalib_logic.py <树根>
"""
import os
import sys
import types

TREE = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
sys.path.insert(0, TREE)
sys.path.insert(0, os.path.join(TREE, "openpilot"))
CAM_DIR = os.path.join(TREE, "openpilot", "system", "camerad", "webcam")
if not os.path.isdir(CAM_DIR):
  CAM_DIR = os.path.join(TREE, "tools", "webcam")
sys.path.insert(0, CAM_DIR)

import camerad as C  # noqa: E402

FAIL = 0


def check(cond, msg):
  global FAIL
  print(("  [PASS] " if cond else "  [FAIL] ") + msg)
  if not cond:
    FAIL = 1


class FakeParams:
  def __init__(self, d=None):
    self.d = dict(d or {})

  def get(self, k):
    return self.d.get(k)

  def get_bool(self, k):
    return bool(self.d.get(k, False))

  def put(self, k, v):
    self.d[k] = v

  def remove(self, k):
    self.d.pop(k, None)


def fake_self(params):
  o = types.SimpleNamespace()
  o.params = params
  o._wc_swapped = False
  o._wc_intrinsics_done = False
  o._swap_calib_intrinsics = types.MethodType(C.Camerad._swap_calib_intrinsics, o)
  return o


FCAM = '{"fl": 2520, "cx": 960, "cy": 540}'
ECAM = '{"fl": 670, "cx": 960, "cy": 540}'

print("[1] _wc_output_of 输出映射")
if C._MSG_WIDE:
  check(C._wc_output_of(C._MSG_ROAD, C._VST_ROAD) == (C._MSG_WIDE, C._VST_WIDE_ROAD),
        f"{C._MSG_ROAD} -> {C._MSG_WIDE}")
  check(C._wc_output_of(C._MSG_WIDE, C._VST_WIDE_ROAD) == (C._MSG_ROAD, C._VST_ROAD),
        f"{C._MSG_WIDE} -> {C._MSG_ROAD}")
else:
  check(C._wc_output_of(C._MSG_ROAD, C._VST_ROAD) == (C._MSG_ROAD, C._VST_ROAD),
        "无 wide 相机时映射应为恒等 (原样返回)")

print("[2] _swap_calib_intrinsics 进入 (road 改用 ECAM)")
p = FakeParams({"FcamIntrinsics": FCAM, "EcamIntrinsics": ECAM})
s = fake_self(p)
s._swap_calib_intrinsics(True)
check(p.d.get("FcamIntrinsics") == ECAM, "FcamIntrinsics 换成 ECAM")
check(p.d.get("EcamIntrinsics") == FCAM, "EcamIntrinsics 换成 FCAM")
check(p.d.get("WideCalibIntrinsicsFcamBackup") == FCAM, "备份 Fcam 原值")
check(p.d.get("WideCalibIntrinsicsEcamBackup") == ECAM, "备份 Ecam 原值")
s._swap_calib_intrinsics(True)
check(p.d.get("FcamIntrinsics") == ECAM, "重复进入不二次互换 (幂等)")

print("[3] 退出: Ecam 未被改过 -> 两个都还原")
s._swap_calib_intrinsics(False)
check(p.d.get("FcamIntrinsics") == FCAM, "FcamIntrinsics 还原")
check(p.d.get("EcamIntrinsics") == ECAM, "EcamIntrinsics 还原")
check("WideCalibIntrinsicsFcamBackup" not in p.d and "WideCalibIntrinsicsEcamBackup" not in p.d, "备份已清理")

print("[4] 退出: Ecam 被标定改写 -> 保留新值")
p = FakeParams({"FcamIntrinsics": FCAM, "EcamIntrinsics": ECAM})
s = fake_self(p)
s._swap_calib_intrinsics(True)
NEW_ECAM = '{"fl": 640, "cx": 960, "cy": 540}'  # 标定写回来的新 ECAM 内参
p.put("EcamIntrinsics", NEW_ECAM)
s._swap_calib_intrinsics(False)
check(p.d.get("FcamIntrinsics") == FCAM, "FcamIntrinsics 还原")
check(p.d.get("EcamIntrinsics") == NEW_ECAM, "EcamIntrinsics 保留标定后的新值")

print("[5] 无备份时退出不应崩")
p = FakeParams({"FcamIntrinsics": FCAM})
s = fake_self(p)
s._swap_calib_intrinsics(False)
check(p.d.get("FcamIntrinsics") == FCAM, "无备份 -> 原样保留, 未抛异常")

print()
print("WideCalib 逻辑自检: " + ("全部通过" if FAIL == 0 else "有 FAIL 项"))
sys.exit(FAIL)
