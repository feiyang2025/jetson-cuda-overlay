#!/usr/bin/env python3
"""calibd 决策层自测 (纯逻辑, 不需要相机/车辆/编译产物)。

触发是**里程**驱动的: 累计跑够 MIN_DISTANCE_KM (默认 30, env CALIB_MIN_KM 可覆盖) 且期间
各趟结论一致 (离散 ≤1.5%)、幅度在 ±15% 内、偏差 >2%, 才把"最合适"的值 (按样本数加权中位数)
写进内参; 写前备份、写后里程清零重新累计、随时可一键回滚。

覆盖:
  1. 默认自动开 / 显式 False 才是关
  2. 融合未采纳            -> 只记历史, 不动内参
  3. 显式关闭              -> suggest_only, 不动内参
  4. 里程不够              -> pending, 报"还差多少 km"
  5. 里程够但趟数不足      -> pending
  6. 里程够 + 2 趟一致     -> applied: 加权中位数写入 + 备份 + 清外参 + 里程清零
  7. 写入后重新累计        -> 再次写入要重新跑够里程
  8. 幅度 > 15%            -> 拒绝自动写
  9. 池内离散 > 1.5%       -> 继续收
 10. 偏差 < 2%             -> 不动 (测量噪声)
 11. 回滚 / ECAM 键

用法: .venv/bin/python3 tools/selftest_calibd.py <树根>
"""
import os
import sys

TREE = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
sys.path.insert(0, TREE)
os.environ.setdefault("CAM_WIDTH", "1344")
os.environ.setdefault("CAM_HEIGHT", "760")
os.environ["CALIB_MIN_KM"] = "30"        # 固定门槛, 免得测试受环境影响

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


def fresh(key="FcamIntrinsics", fl=961.76, disable=False):
    p = FakeParams()
    p.store[key] = {"fl": fl, "cx": 672.0, "cy": 380.0,
                    "calibrations": {"1344x760": {"fl": fl, "cx": 672.0, "cy": 380.0}}}
    p.store["CalibrationParams"] = {"rpyCalib": [0, 0, 0]}
    if disable:
        p.store["AutoCalib"] = False
    return p


def ctl_of(p, cam="fcam"):
    return C.AutoCalibController(p, cam, log=lambda *a: None)


print("1) 默认自动开 / 显式 False 才是关")
check("键不存在 -> 自动开", ctl_of(fresh()).enabled() is True)
check("显式 False -> 关", ctl_of(fresh(disable=True)).enabled() is False)

print("2) 融合未采纳 -> 只记历史, 不动内参")
p = fresh()
e = ctl_of(p).consider(result(961.76, 700.0, accepted=False, delta_pct=-27.0))
check("action=reject", e["action"] == "reject", e["action"])
check("内参未动", p.store["FcamIntrinsics"]["fl"] == 961.76)
check("历史已记录", len(p.store["AutoCalibHistory"]["runs"]) == 1)

print("3) 显式关闭 -> suggest_only")
p = fresh(disable=True)
e = ctl_of(p).consider(result(961.76, 930.0))
check("action=suggest_only", e["action"] == "suggest_only", e["action"])
check("内参未动", p.store["FcamIntrinsics"]["fl"] == 961.76)

print("4) 里程不够 -> pending 且报还差多少 km")
p = fresh()
ctl = ctl_of(p)
ctl.add_distance(12.0)
e = ctl.consider(result(961.76, 930.0))
check("pending", e["action"] == "pending", e["action"])
check("原因含里程", "里程" in (e.get("reason") or ""), e.get("reason"))
check("内参未动", p.store["FcamIntrinsics"]["fl"] == 961.76)

print("5) 里程够但有效趟数不足 -> pending")
p = fresh()
ctl = ctl_of(p)
ctl.add_distance(30.0)
e1 = ctl.consider(result(961.76, 930.0))
check("第1趟 pending(趟数不足)", e1["action"] == "pending" and "趟数" in (e1.get("reason") or ""),
      f"{e1['action']} — {e1.get('reason')}")
check("仍没写", p.store["FcamIntrinsics"]["fl"] == 961.76)

print("6) 里程够 + 第2趟一致 -> applied (加权中位数) + 备份 + 清外参 + 里程清零")
e2 = ctl.consider(result(961.76, 931.0, samples=400))
check("applied", e2["action"] == "applied", f"{e2['action']} — {e2.get('reason')}")
# 权重: 第一趟 300 样本、第二趟 400 样本 -> 加权中位数取 931.0
check("写入加权中位数 931.0", abs(p.store["FcamIntrinsics"]["fl"] - 931.0) < 1e-6,
      str(p.store["FcamIntrinsics"]["fl"]))
check("calibrations 同步", abs(p.store["FcamIntrinsics"]["calibrations"]["1344x760"]["fl"] - 931.0) < 1e-6)
check("旧值已备份", p.store["AutoCalibBackup"]["fl"] == 961.76)
check("CalibrationParams 已清", "CalibrationParams" not in p.store)
check("里程已清零", abs(ctl.distance_km()) < 1e-9, str(ctl.distance_km()))

print("7) 写入后要重新累计里程才可能再写")
ctl.add_distance(5.0)
e3 = ctl.consider(result(931.0, 900.0))
check("又变 pending", e3["action"] == "pending" and "里程" in (e3.get("reason") or ""),
      f"{e3['action']} — {e3.get('reason')}")

print("8) 幅度 > 15% -> 拒绝")
p = fresh()
ctl = ctl_of(p)
ctl.add_distance(40.0)
for _ in range(2):
    e = ctl.consider(result(961.76, 1300.0))
check("拒绝且提示上限", "上限" in (e.get("reason") or ""), e.get("reason"))
check("内参未动", p.store["FcamIntrinsics"]["fl"] == 961.76)

print("9) 池内离散 > 1.5% -> 继续收")
p = fresh()
ctl = ctl_of(p)
ctl.add_distance(40.0)
ctl.consider(result(961.76, 900.0))      # -6.4%
e = ctl.consider(result(961.76, 985.0))  # +2.4% -> 池内离散约 9%
check("不写", e["action"] == "pending" and "离散" in (e.get("reason") or ""),
      f"{e['action']} — {e.get('reason')}")
check("内参未动", p.store["FcamIntrinsics"]["fl"] == 961.76)

print("10) 偏差 < 2% -> 视为噪声")
p = fresh()
e = ctl_of(p).consider(result(961.76, 965.0))
check("no_change", e["action"] == "no_change", e["action"])

print("11) 回滚")
p = fresh()
ctl = ctl_of(p)
ctl.add_distance(30.0)
ctl.consider(result(961.76, 900.0, samples=300))
ctl.consider(result(961.76, 901.0, samples=400))   # 权重更大 -> 加权中位数取它
check("已写入", abs(p.store["FcamIntrinsics"]["fl"] - 901.0) < 1e-6, str(p.store["FcamIntrinsics"]["fl"]))
p.store["AutoCalibRollback"] = True
r = ctl.rollback()
check("回滚成功", r.get("ok") and abs(p.store["FcamIntrinsics"]["fl"] - 961.76) < 1e-6, str(r))
check("请求标志已清", p.store["AutoCalibRollback"] is False)

print("12) ECAM 走 EcamIntrinsics")
p = fresh(key="EcamIntrinsics", fl=1740.0)
ctl = ctl_of(p, "ecam")
ctl.add_distance(30.0)
ctl.consider(result(1740.0, 1680.0, samples=200))
ctl.consider(result(1740.0, 1682.0, samples=300))   # 权重更大 -> 取 1682

check("写到 EcamIntrinsics", abs(p.store["EcamIntrinsics"]["fl"] - 1682.0) < 1e-6,
      str(p.store["EcamIntrinsics"]["fl"]))
check("FcamIntrinsics 没被碰", "FcamIntrinsics" not in p.store)

print()
if FAILS:
    print(f"FAILED {len(FAILS)}: {FAILS}")
    sys.exit(1)
print("全部通过")
