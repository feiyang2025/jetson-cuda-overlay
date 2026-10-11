#!/usr/bin/env python3
"""self_calibrator 融合逻辑的自测 (合成数据, 不需要真车/日志)。

验的是"三方法加权融合 + 一致性门限"这件事本身:
  1. 三方法互相接近 -> 采纳, 且融合值比任一单方法更接近真值 (偏差摊薄);
  2. 一个方法偏 10%   -> MAD 剔除它, 其余照常采纳;
  3. 两方法分歧 6%    -> 不采纳 (分歧超限), 不写 Params;
  4. 只有一个方法     -> 不采纳 (要 ≥2 个独立真值交叉验证);
  5. 样本不足         -> 不采纳;
  6. ECAM 门限更严 + 置信度不判 high;
  7. 曲率票的尺度数学: κ_model/κ_imu = fl_true/fl_assumed。

用法: .venv/bin/python3 tools/calib/selftest_fusion.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import self_calibrator as sc  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'OK ' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def fuse(cands, ref_fl=2520, ecam=False):
    cal = sc.SelfCalibrator(ecam=ecam, fcam_fl=ref_fl)
    return cal._fuse_methods(list(cands))


def c(name, fl, samples):
    return {"name": name, "fl": fl, "samples": samples}


print("1) 三方法接近 + 独立偏差 -> 融合值比任一单方法更接近真值")
truth = 2520.0
res = fuse([c("radar_vision_ratio", truth * 1.012, 300),
            c("lane_width", truth * 0.988, 200),
            c("curvature", truth * 1.006, 100)])
err_single = min(abs(truth * m - truth) for m in (1.012, 0.988, 1.006)) / truth
err_fused = abs(res["fl_suggested"] - truth) / truth
check("采纳", res["accepted"] is True, f"confidence={res['confidence']} spread={res['spread_pct']}%")
check("融合误差小于最好的单方法", err_fused < err_single,
      f"融合 {err_fused * 100:.2f}% vs 最好单方法 {err_single * 100:.2f}%")
check("三个方法都在贡献里", len(res["contributors"]) == 3, str([x["name"] for x in res["contributors"]]))

print("2) 一个方法偏 10% -> 被 MAD 剔除")
res = fuse([c("radar_vision_ratio", truth * 1.01, 300),
            c("lane_width", truth * 0.99, 200),
            c("curvature", truth * 1.10, 100)])
names = [x["name"] for x in res["contributors"]]
rejected = [x["name"] for x in res["rejected"]]
check("离群方法被剔除", rejected == ["curvature"], f"rejected={rejected}")
check("其余两方法仍采纳", res["accepted"] is True and len(names) == 2, f"contributors={names}")

print("3) 两方法分歧 6% -> 不采纳")
res = fuse([c("radar_vision_ratio", truth * 1.03, 300),
            c("lane_width", truth * 0.97, 200)])
check("不采纳", res["accepted"] is False, f"reason={res['reason']}")
check("fl_suggested 为空", res["fl_suggested"] is None, f"raw={res['fl_suggested_raw']}")

print("4) 只有单一方法 -> 不采纳")
res = fuse([c("radar_vision_ratio", truth * 1.02, 300)])
check("不采纳", res["accepted"] is False, f"reason={res['reason']}")

print("5) 样本不足 -> 不采纳")
res = fuse([c("radar_vision_ratio", truth * 1.00, 20), c("lane_width", truth * 1.01, 20)])
check("不采纳", res["accepted"] is False, f"reason={res['reason']}")

print("6) ECAM: 门限更严 (2%) 且置信度不判 high")
res = fuse([c("radar_vision_ratio", truth * 1.02, 300), c("lane_width", truth * 0.995, 300)],
           ecam=True, ref_fl=605)
check("2.5% 分歧在 ECAM 下不采纳", res["accepted"] is False, f"spread={res['spread_pct']}% reason={res['reason']}")
res = fuse([c("radar_vision_ratio", truth * 1.005, 300), c("lane_width", truth * 0.999, 300)],
           ecam=True, ref_fl=605)
check("ECAM 一致时采纳但最高 medium", res["accepted"] is True and res["confidence"] == "medium",
      f"confidence={res['confidence']}")

print("7) 曲率票: κ_model/κ_imu = fl_true/fl_assumed")
ref = 605.0
for ratio in (0.94, 1.0, 1.06):
    cvcal = sc.CurvatureCalibrator(fl_current=ref)
    k_imu = 0.01                      # rad/m, 半径 100m 的弯
    for _ in range(40):
        cvcal.add(yaw_rate=k_imu * 15.0, v_ego=15.0, model_curv=k_imu * ratio)
    out = cvcal.compute()
    ok = out.get("valid") and abs(out["fl_suggested"] - ref * ratio) < 1.0
    check(f"ratio={ratio} -> fl≈{ref * ratio:.0f}", ok,
          f"got={out.get('fl_suggested')} ratio_median={out.get('ratio_median')}")

print("8) 曲率票样本不足时不参与")
cvcal = sc.CurvatureCalibrator(fl_current=ref)
for _ in range(5):
    cvcal.add(yaw_rate=0.01 * 15.0, v_ego=15.0, model_curv=0.0106)
check("弯道样本不足 -> valid False", cvcal.compute().get("valid") is False,
      str(cvcal.compute().get("error")))

print()
if FAILS:
    print(f"FAILED {len(FAILS)}: {FAILS}")
    sys.exit(1)
print("全部通过")
