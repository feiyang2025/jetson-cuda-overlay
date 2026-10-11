#!/usr/bin/env python3
"""calibd 决策层自测 (纯逻辑, 不需要相机/车辆/编译产物)。

验的是"要不要自动写内参"这套门限, 这是整个自动标定里最需要被证明的部分 —— fl 直接决定
视觉测距 (跟车/AEB), 写错比不写危险得多。

覆盖:
  1. 融合未采纳        -> 只记历史, 不动内参
  2. AutoCalib 关      -> suggest_only, 不动内参
  3. 打开后第 1/2 趟   -> pending (还差几趟)
  4. 连续 3 趟一致     -> applied: 写中位数, 先备份, 清 CalibrationParams
  5. 幅度 > 15%        -> 拒绝自动写 (通常是相机装错/型号不对, 得先查)
  6. 三趟互相不一致    -> pending (偶发抖动不写)
  7. 偏差 < 2%         -> no_change (测量噪声范围)
  8. 回滚              -> 内参回到备份, 请求标志清掉
  9. ECAM 走 EcamIntrinsics 键

用法: .venv/bin/python3 tools/selftest_calibd.py <树根>
"""
import os
import sys

TREE = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
sys.path.insert(0, TREE)
os.environ.setdefault("CAM_WIDTH", "1344")
os.environ.setdefault("CAM_HEIGHT", "760")

from openpilot.sunnypilot.calib import controller as C  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'OK ' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


class FakeParams:
    def __init__(self):
        self.store = {}

    def get(self, k):
        return self.store.get(k)

    def get_bool(self, k):
        return bool(self.store.get(k, False))

    def put(self, k, v, block=False):
        self.store[k] = v

    def put_bool(self, k, v):
        self.store[k] = bool(v)

    def remove(self, k):
        self.store.pop(k, None)


def result(fl_current, fl_suggested, accepted=True, delta_pct=None, confidence="high",
           n_methods=3, spread_pct=1.0, samples=300):
    if delta_pct is None:
        delta_pct = (fl_suggested / fl_current - 1) * 100
    return {"fused": {"accepted": accepted, "fl_current": fl_current, "fl_suggested": fl_suggested,
                      "fl_delta_pct": round(delta_pct, 2), "confidence": confidence,
                      "n_methods": n_methods, "spread_pct": spread_pct,
                      "total_observations": samples, "reason": "" if accepted else "不足2个方法"}}


def fresh(key="FcamIntrinsics", fl=961.76):
    p = FakeParams()
    p.store[key] = {"fl": fl, "cx": 672.0, "cy": 380.0,
                    "calibrations": {"1344x760": {"fl": fl, "cx": 672.0, "cy": 380.0}}}
    p.store["CalibrationParams"] = {"rpyCalib": [0, 0, 0]}
    return p


print("1) 融合未采纳 -> 只记历史")
p = fresh()
p.store["AutoCalib"] = True
ctl = C.AutoCalibController(p, "fcam", log=lambda *a: None)
e = ctl.consider(result(961.76, 700.0, accepted=False, delta_pct=-27.0))
check("action=reject", e["action"] == "reject", e["action"])
check("内参未动", p.store["FcamIntrinsics"]["fl"] == 961.76, str(p.store["FcamIntrinsics"]["fl"]))
check("历史已记录", len(p.store["AutoCalibHistory"]["runs"]) == 1)

print("2) 通过门限但开关关着 -> suggest_only")
p = fresh()
e = ctl.__class__(p, "fcam", log=lambda *a: None).consider(result(961.76, 930.0))
check("action=suggest_only", e["action"] == "suggest_only", e["action"])
check("内参未动", p.store["FcamIntrinsics"]["fl"] == 961.76)

print("3) 开关打开: 第 1/2 趟只累计, 第 3 趟一致才写")
p = fresh()
p.store["AutoCalib"] = True
ctl = C.AutoCalibController(p, "fcam", log=lambda *a: None)
e1 = ctl.consider(result(961.76, 930.0))
check("第1趟 pending", e1["action"] == "pending", f"{e1['action']} — {e1.get('reason')}")
e2 = ctl.consider(result(961.76, 931.0))
check("第2趟 pending", e2["action"] == "pending", f"{e2['action']} — {e2.get('reason')}")
check("两趟都没写内参", p.store["FcamIntrinsics"]["fl"] == 961.76)
e3 = ctl.consider(result(961.76, 930.5))
check("第3趟 applied", e3["action"] == "applied", f"{e3['action']} — {e3.get('reason')}")
check("内参写成中位数 930.5", abs(p.store["FcamIntrinsics"]["fl"] - 930.5) < 1e-6,
      str(p.store["FcamIntrinsics"]["fl"]))
check("calibrations 同步更新", abs(p.store["FcamIntrinsics"]["calibrations"]["1344x760"]["fl"] - 930.5) < 1e-6)
check("旧值已备份", p.store["AutoCalibBackup"]["fl"] == 961.76, str(p.store["AutoCalibBackup"]["fl"]))
check("CalibrationParams 已清 (外参要重标)", "CalibrationParams" not in p.store)

print("4) 幅度超 15% -> 拒绝自动写")
p = fresh()
p.store["AutoCalib"] = True
ctl = C.AutoCalibController(p, "fcam", log=lambda *a: None)
for _ in range(3):
    e = ctl.consider(result(961.76, 1300.0))     # +35%
check("拒绝 (上限说明)", e["action"] == "pending" and "上限" in e["reason"], f"{e['action']} — {e['reason']}")
check("内参未动", p.store["FcamIntrinsics"]["fl"] == 961.76)

print("5) 三趟互相不一致 -> 继续等")
p = fresh()
p.store["AutoCalib"] = True
ctl = C.AutoCalibController(p, "fcam", log=lambda *a: None)
# 三趟都超过 2% (会进入滞回判断), 但彼此差得远 (约 9%) → 必须继续等
ctl.consider(result(961.76, 900.0))              # -6.4%
ctl.consider(result(961.76, 940.0))              # -2.3%
e = ctl.consider(result(961.76, 985.0))          # +2.4%
check("离散超限 -> 不写", e["action"] == "pending" and "离散" in e["reason"], f"{e['action']} — {e['reason']}")
check("内参未动", p.store["FcamIntrinsics"]["fl"] == 961.76)

print("6) 偏差 < 2% -> 视为噪声, 不动")
p = fresh()
p.store["AutoCalib"] = True
e = C.AutoCalibController(p, "fcam", log=lambda *a: None).consider(result(961.76, 965.0))
check("action=no_change", e["action"] == "no_change", f"{e['action']} — {e.get('reason')}")
check("内参未动", p.store["FcamIntrinsics"]["fl"] == 961.76)

print("7) 回滚")
p = fresh()
p.store["AutoCalib"] = True
ctl = C.AutoCalibController(p, "fcam", log=lambda *a: None)
for fl in (930.0, 930.5, 930.2):
    ctl.consider(result(961.76, fl))
check("已写入", abs(p.store["FcamIntrinsics"]["fl"] - 930.2) < 1.0, str(p.store["FcamIntrinsics"]["fl"]))
p.store["AutoCalibRollback"] = True
r = ctl.rollback()
check("回滚成功", r.get("ok") and abs(p.store["FcamIntrinsics"]["fl"] - 961.76) < 1e-6, str(r))
check("回滚后请求标志已清", p.store["AutoCalibRollback"] is False, str(p.store["AutoCalibRollback"]))

print("8) ECAM 走 EcamIntrinsics")
p = fresh(key="EcamIntrinsics", fl=1740.0)
p.store["AutoCalib"] = True
ctl = C.AutoCalibController(p, "ecam", log=lambda *a: None)
for fl in (1680.0, 1682.0, 1681.0):
    ctl.consider(result(1740.0, fl))
check("写到 EcamIntrinsics", abs(p.store["EcamIntrinsics"]["fl"] - 1681.0) < 1e-6, str(p.store["EcamIntrinsics"]["fl"]))
check("FcamIntrinsics 没被碰", "FcamIntrinsics" not in p.store)

print()
if FAILS:
    print(f"FAILED {len(FAILS)}: {FAILS}")
    sys.exit(1)
print("全部通过")
