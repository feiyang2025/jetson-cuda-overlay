#!/usr/bin/env python3
"""
Self-consistency calibration for camera intrinsics.
Three methods fused:
  1. Radar-Vision lead distance comparison
  2. Lane width measurement (speed-classified road type)
  3. IMU-Model curvature consistency

Usage:
  # Analyze one route
  python self_calibrator.py --route 0000003b--b85bf4ef47--2

  # Analyze multiple routes
  python self_calibrator.py --routes 0000003b--b85bf4ef47--2,0000003c--b85bf4ef47--0

  # Scan all routes in realdata/
  python self_calibrator.py --scan

  # Live collection mode (toggle ON during driving, SIGTERM to compute)
  python self_calibrator.py --live --output result.json
"""

import numpy as np
import os
import sys
import json
import signal
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from openpilot.tools.lib.logreader import LogReader


CAMERA_RADAR_OFFSET = 1.5  # m; camera (windshield) behind radar (bumper)
FL_CURRENT = 2520  # default FCAM fl; overridden in main() for --ecam
HIGHWAY_SPEED_MS = 90 / 3.6   # 25 m/s
EXPRESS_SPEED_MS = 60 / 3.6   # 16.7 m/s
STRAIGHT_THRESHOLD = 0.008    # rad/m max |curv| for "straight road" filter
LANE_MARKING_WIDTH = 0.15     # m; standard Chinese lane marking width (inner-edge to center)

# --- 曲率法 (第三票) 参数 ---
# 直线样本 (|κ|<STRAIGHT_THRESHOLD) 只用来诊断偏置: 那里 κ≈0, 对 fl 的尺度完全没信息量。
# 要给 fl 提供尺度, 必须用**弯道**样本: 模型曲率与 IMU 曲率的比值就是 fl 的真实倍率。
CURVED_THRESHOLD = 0.002      # rad/m; |κ_imu| 超过它才算"弯道样本" (半径 ≲500m)
CURVED_MIN_SPEED_MS = 8.0     # m/s; 低速时 yawRate/vEgo 噪声大, 且轮胎侧偏不可忽略
CURVED_MIN_SAMPLES = 30
CURVED_RATIO_CLIP = (0.5, 2.0)  # 只接受 0.5x-2x 的 fl 修正, 之外当样本被污染

# --- 融合 (三方法加权 + 一致性门限) ---
# 每个方法背后的"真值"不同 (雷达测距 / 3.5-3.75m 车道标准 / IMU 偏航率), 系统偏差互相独立,
# 加权融合能把各自的系统偏差摊薄; 单用任何一个 = 把它的偏差 100% 吃进来。
METHOD_RELIABILITY = {
    "radar_vision_ratio": 1.0,   # 雷达测距最可信, 但受安装偏移与雷达噪声影响
    "radar_vision": 1.0,
    "lane_width": 0.8,           # 依赖"标准车道宽"假设, 假设错 1:1 传到 fl
    "curvature": 0.6,            # 依赖 IMU 偏置与车辆实际循迹 (侧偏)
}
FUSION_MIN_SAMPLES = 50          # 总样本下限
FUSION_SPREAD_LIMIT = 0.03       # 方法间相对离散上限 (FCAM)
FUSION_SPREAD_LIMIT_ECAM = 0.02  # ECAM 更严: 广角畸变使边缘测量有系统偏差
FUSION_MAD_K = 3.0               # MAD 离群剔除倍数



class RadarVisionCalibrator:
    def __init__(self, fl_current: float = 2520):
        self.pairs = []
        self._fl = fl_current

    def add(self, d_radar: float, d_vision: float, v_ego: float):
        if d_radar > 3 and d_vision > 3:
            self.pairs.append((float(d_radar), float(d_vision), float(v_ego)))

    @property
    def count(self) -> int:
        return len(self.pairs)

    def compute(self) -> Dict:
        if len(self.pairs) < 10:
            return {"valid": False, "samples": len(self.pairs), "error": "不足10个样本"}

        d_r = np.array([p[0] for p in self.pairs])
        d_v = np.array([p[1] for p in self.pairs])

        # Iterative outlier removal by delta
        delta = d_v - d_r  # expected ≈ -1.5m (camera behind radar)
        for _ in range(3):
            q1, q3 = np.percentile(delta, [25, 75])
            iqr = q3 - q1
            mask = (delta >= q1 - 2.5 * iqr) & (delta <= q3 + 2.5 * iqr)
            if sum(mask) < 10:
                break
            delta = delta[mask]
            d_r = d_r[mask]
            d_v = d_v[mask]

        if len(d_r) < 10:
            return {"valid": False, "samples": len(self.pairs), "error": "过多样本被过滤"}

        # Linear regression: d_vision = k * d_radar + b
        # On correct fl: k ≈ 1, b ≈ -CAMERA_RADAR_OFFSET
        A = np.vstack([d_r, np.ones(len(d_r))]).T
        k, b = np.linalg.lstsq(A, d_v, rcond=None)[0]

        # If k > 1: vision grows faster → fl may be too low
        # fl correction: d_vision_corrected = d_vision / k
        # But we want d_vision_corrected ≈ d_radar - offset
        # so k_adjustment = 1/k
        fl_factor = 1.0 / k

        # Stratified
        strata = {}
        for label, cond in [("近距离 <30m", d_r < 30), ("中距 30-80m", (d_r >= 30) & (d_r < 80)), ("远距 ≥80m", d_r >= 80)]:
            n = int(sum(cond))
            if n >= 3:
                dv_m = np.mean(d_v[cond]); dr_m = np.mean(d_r[cond])
                strata[label] = {"n": n, "mean_delta": float(dv_m - dr_m)}
            else:
                strata[label] = {"n": n, "mean_delta": None}

        return {
            "valid": True,
            "samples": len(d_r), "samples_total": len(self.pairs),
            "slope_k": float(k), "intercept_b": float(b),
            "mean_delta": float(np.mean(delta)),
            "fl_factor": float(fl_factor),
            "fl_suggested": self._fl * fl_factor,
            "stratified": strata,
        }


class LaneWidthCalibrator:
    def __init__(self, fl_current: float = 2520, lane_marking_width: float = LANE_MARKING_WIDTH):
        self.obs = []
        self._marking_w = lane_marking_width  # single-side marking → inner-edge offset
        self._fl = fl_current

    def add(self, width: float, v_ego: float):
        if v_ego < 15 or not (2.0 < width < 6.0):
            return
        if v_ego >= HIGHWAY_SPEED_MS:
            road = "highway"
            expected = 3.75
        elif v_ego >= EXPRESS_SPEED_MS:
            road = "expressway"
            expected = 3.50
        else:
            return
        self.obs.append((width, expected, road, v_ego))

    @property
    def count(self) -> int:
        return len(self.obs)

    def compute(self) -> Dict:
        if len(self.obs) < 10:
            return {"valid": False, "samples": len(self.obs), "error": "不足10个样本"}

        by_type = defaultdict(list)
        for w, exp, rt, _ in self.obs:
            by_type[rt].append((w, exp))

        results = {}
        all_w, all_exp = [], []
        for rt in ["highway", "expressway"]:
            if rt not in by_type:
                continue
            ws = np.array([x[0] for x in by_type[rt]])
            exp = by_type[rt][0][1]
            # Lane model detects inner edges of lane markings.
            # True lane center-to-center = measured inner-edge gap + marking width.
            all_w.extend(ws + self._marking_w)
            all_exp.extend([exp] * len(ws))
            mw_raw = float(np.mean(ws))
            mw_corrected = mw_raw + self._marking_w
            results[rt] = {
                "n": len(ws), "mean_width_raw": mw_raw, "mean_width": mw_corrected,
                "marking_width": self._marking_w, "expected": exp,
                "ratio": mw_corrected / exp, "std": float(np.std(ws)),
            }

        if not results:
            return {"valid": False, "samples": len(self.obs), "error": "无有效道路类型数据"}

        combined_ratio = np.average(
            [r["ratio"] for r in results.values()],
            weights=[r["n"] for r in results.values()]
        )
        return {
            "valid": True,
            "samples": len(self.obs),
            "by_type": results,
            "combined_ratio": float(combined_ratio),
            "fl_factor": float(1.0 / combined_ratio),
            "fl_suggested": self._fl / combined_ratio,
        }


class CurvatureCalibrator:
    """IMU-模型曲率一致性。

    两种样本:
      * 直线 (|κ_imu| < STRAIGHT_THRESHOLD): 只做偏置诊断 (κ≈0, 对 fl 尺度无信息量);
      * 弯道 (|κ_imu| ≥ CURVED_THRESHOLD): 才是 fl 的第三票 ——
        模型物理量按 1/fl 缩放, 所以 κ_model/κ_imu = fl_true/fl_assumed,
        于是 fl_true = fl_assumed × median(κ_model/κ_imu)。
    """

    def __init__(self, fl_current: float = 2520):
        self.pairs = []          # 直线样本 (诊断)
        self.curved = []         # 弯道样本 (尺度)
        self._fl = fl_current

    def add(self, yaw_rate: float, v_ego: float, model_curv: float):
        if v_ego < 5:
            return
        curv_imu = yaw_rate / v_ego
        if abs(curv_imu) < STRAIGHT_THRESHOLD and abs(model_curv) < 0.02:
            self.pairs.append((curv_imu, model_curv, v_ego))
            return
        if v_ego < CURVED_MIN_SPEED_MS or abs(curv_imu) < CURVED_THRESHOLD:
            return
        # 只在同号时取比值: 反向说明这一帧的模型曲率被别的东西污染了
        if model_curv * curv_imu <= 0:
            return
        ratio = model_curv / curv_imu
        if CURVED_RATIO_CLIP[0] <= ratio <= CURVED_RATIO_CLIP[1]:
            self.curved.append((curv_imu, model_curv, v_ego, ratio))

    @property
    def count(self) -> int:
        return len(self.pairs) + len(self.curved)

    def compute(self) -> Dict:
        out: Dict = {"samples": self.count, "samples_straight": len(self.pairs),
                     "samples_curved": len(self.curved)}
        if len(self.pairs) >= 10:
            c_imu = np.array([p[0] for p in self.pairs])
            c_model = np.array([p[1] for p in self.pairs])
            bias = c_model - c_imu
            out.update({
                "bias_mean": float(np.mean(bias)),
                "bias_std": float(np.std(bias)),
                "bias_max_abs": float(np.max(np.abs(bias))),
            })

        if len(self.curved) < CURVED_MIN_SAMPLES:
            out["valid"] = False
            out["error"] = f"弯道样本不足{CURVED_MIN_SAMPLES} ({len(self.curved)})"
            return out

        ratios = np.array([p[3] for p in self.curved])
        med = float(np.median(ratios))
        mad = float(np.median(np.abs(ratios - med)))
        out.update({
            "valid": True,
            "ratio_median": med,
            "ratio_spread": mad,                      # MAD, 用于一致性判定
            "fl_factor": med,
            "fl_suggested": self._fl * med,
        })
        return out


class SelfCalibrator:
    def __init__(self, ecam: bool = False, fcam_fl: int = 2520):
        # ECAM mode uses a different reference fl; both fallback algorithms must
        # base their fl_suggested on it (module-global FL_CURRENT is unreliable).
        ref_fl = 605 if ecam else fcam_fl
        # ECAM: 调用方 (main --ecam) 会把 EcamIntrinsics 的当前 fl 当 fcam_fl 传进来,
        # 用它做参考; 只有没传时才退回 605 默认。
        if ecam and fcam_fl:
            ref_fl = fcam_fl
        self.rv = RadarVisionCalibrator(fl_current=ref_fl)
        self.lw = LaneWidthCalibrator(fl_current=ref_fl)
        self.cv = CurvatureCalibrator(fl_current=ref_fl)
        self._ecam = ecam
        self._fcam_fl = fcam_fl
        self._ref_fl = ref_fl      # 本次要标定的那台相机的当前 fl
        self._last_v = 0
        self._last_yaw = 0
        self._last_radar_d = 0
        self._last_vision_x = 0

    def process_log(self, path: str) -> Tuple[int, int]:
        total_msgs = 0
        frames_model = 0
        for m in LogReader(path):
            total_msgs += 1
            w = m.which()
            if w == 'carState':
                self._last_v = m.carState.vEgo
                self._last_yaw = getattr(m.carState, 'yawRate', 0)
            elif w == 'radarState':
                rs = m.radarState
                if _lead_present(rs.leadOne):
                    self._last_radar_d = rs.leadOne.dRel
            elif w == 'modelV2':
                frames_model += 1
                md = m.modelV2
                leads = md.leadsV3
                if leads[0].prob > 0.5:
                    self._last_vision_x = leads[0].x[0]
                    self.rv.add(self._last_radar_d, self._last_vision_x, self._last_v)
                probs = list(md.laneLineProbs)
                if len(probs) >= 3 and probs[1] > 0.5 and probs[2] > 0.5:
                    try:
                        l1 = md.laneLines[1]; l2 = md.laneLines[2]
                        x1, y1 = np.array(list(l1.x)), np.array(list(l1.y))
                        x2, y2 = np.array(list(l2.x)), np.array(list(l2.y))
                        y1_0 = np.interp(0, x1, y1); y2_0 = np.interp(0, x2, y2)
                        self.lw.add(y2_0 - y1_0, self._last_v)
                    except:
                        pass
                try:
                    mu = md.action
                    self.cv.add(self._last_yaw, self._last_v, mu.desiredCurvature)
                except:
                    pass
        return total_msgs, frames_model

    def _compute_ratio(self) -> Dict:
        """Ratio-based radar-vision calibration, auto-detects FCAM vs ECAM.

        Two-pass:
          1. Compute overall ratio histogram to detect mode.
          2. If ECAM mode (median > 1.5), filter to ECAM-only pairs and recompute.
        """
        pairs = [(p[0], p[1]) for p in self.rv.pairs if 5 < p[0] < 80 and p[1] > 3]
        if len(pairs) < 30:
            return {"valid": False, "error": f"不足30样本 ({len(pairs)})", "samples": len(pairs)}

        d_r = np.array([p[0] for p in pairs])
        d_v = np.array([p[1] for p in pairs])
        ratios = d_v / d_r

        for _ in range(3):
            q1, q3 = np.percentile(ratios, [25, 75])
            iqr = q3 - q1
            mask = (ratios > q1 - 2.5 * iqr) & (ratios < q3 + 2.5 * iqr)
            if sum(mask) < 30:
                break
            ratios = ratios[mask]
            d_r = d_r[mask]
            d_v = d_v[mask]

        med_all = float(np.median(ratios))
        is_ecam = med_all > 1.5

        # Two-pass for ECAM: filter out FCAM frames (ratio near 1)
        if is_ecam:
            # Find the ECAM peak: use ratio > 1.5 as threshold
            ecam_mask = ratios > 1.5
            if int(ecam_mask.sum()) >= 30:
                ratios = ratios[ecam_mask]
                d_r = d_r[ecam_mask]
                d_v = d_v[ecam_mask]
                # Refilter outliers within ECAM population
                for _ in range(2):
                    q1, q3 = np.percentile(ratios, [25, 75])
                    iqr = q3 - q1
                    m = (ratios > q1 - 2.5 * iqr) & (ratios < q3 + 2.5 * iqr)
                    if sum(m) < 30: break
                    ratios = ratios[m]; d_r = d_r[m]; d_v = d_v[m]

        med_ratio = float(np.median(ratios))
        mean_ratio = float(np.mean(ratios))
        std_ratio = float(np.std(ratios))
        fl_actual = self._fcam_fl / med_ratio if med_ratio > 0 else self._fcam_fl

        strata = {}
        for d_min, d_max, label in [(5, 20, "近距5-20m"), (20, 50, "中距20-50m"), (50, 80, "中远距50-80m")]:
            m = (d_r >= d_min) & (d_r < d_max)
            if int(m.sum()) >= 5:
                r = float(np.mean(ratios[m]))
                strata[label] = {"n": int(m.sum()), "mean_ratio": r, "fl_suggested": int(self._fcam_fl / r)}

        return {
            "valid": True,
            "samples": len(ratios),
            "samples_total": len(self.rv.pairs),
            "ratio_mean": mean_ratio,
            "ratio_median": med_ratio,
            "ratio_std": std_ratio,
            "fcam_fl": self._fcam_fl,
            "fl_actual": int(fl_actual),
            "fl_factor": float(self._fcam_fl / fl_actual) if fl_actual > 0 else 1.0,
            "fl_suggested": int(fl_actual),
            "is_ecam": is_ecam,
            "stratified": strata,
        }

    # ---------------------------------------------------------------- 融合
    def _fuse_methods(self, cands: List[Dict]) -> Dict:
        """三方法加权融合 + 一致性门限。

        cands: [{name, fl, samples}] —— 每项都是"这台相机"的绝对 fl 建议值。
        返回 fused 字典, 含 accepted 标志 (只有 accepted 才允许写 Params)。
        """
        valid = [c for c in cands if c.get("fl") and c["fl"] > 0 and c.get("samples", 0) > 0]
        total_samples = sum(int(c.get("samples", 0)) for c in valid)
        info = {"contributors": [], "rejected": [], "spread_pct": None, "accepted": False}

        if not valid:
            info.update({"reason": "没有任何方法产出 fl 建议值", "confidence": "low",
                         "fl_suggested": None, "total_observations": total_samples})
            return info

        # 1) 加权中位数 (权重 = 样本数 × 方法可靠度)
        for c in valid:
            c["weight"] = int(c.get("samples", 0)) * METHOD_RELIABILITY.get(c["name"], 0.5)
        ordered = sorted(valid, key=lambda c: c["fl"])
        wsum = sum(c["weight"] for c in ordered) or 1.0
        acc = 0.0
        fused_fl = ordered[-1]["fl"]
        for c in ordered:
            acc += c["weight"]
            if acc >= wsum * 0.5:
                fused_fl = c["fl"]
                break

        # 2) MAD 离群剔除: 偏离中位数超过 max(3*MAD, 3%) 的方法不参与定值
        fls = np.array([c["fl"] for c in valid], dtype=float)
        med = float(np.median(fls))
        mad = float(np.median(np.abs(fls - med)))
        tolerance = max(FUSION_MAD_K * mad, FUSION_SPREAD_LIMIT * med)
        kept, dropped = [], []
        for c in valid:
            (kept if abs(c["fl"] - med) <= tolerance else dropped).append(c)
        if not kept:                      # 全被剔了说明离散极大, 只保留最接近中位数的那个
            kept = [min(valid, key=lambda c: abs(c["fl"] - med))]
            dropped = [c for c in valid if c is not kept[0]]
        for c in dropped:
            info["rejected"].append({"name": c["name"], "fl": round(c["fl"], 1),
                                     "samples": c["samples"],
                                     "delta_pct": round((c["fl"] / med - 1) * 100, 2),
                                     "reason": "MAD 离群"})

        # 3) 保留集内取加权平均 —— 这一步才是"结合"的意义所在:
        #    各方法的系统偏差互相独立, 加权平均把偏差摊薄 (中位数只会挑中某一个方法,
        #    等于还是单一真值)。离群已由上面的 MAD 剔除承担。
        kept_sorted = sorted(kept, key=lambda c: c["fl"])
        kept_fls = [c["fl"] for c in kept_sorted]
        wsum = sum(c["weight"] for c in kept_sorted) or 1.0
        fused_fl = sum(c["fl"] * c["weight"] for c in kept_sorted) / wsum
        spread = (max(kept_fls) - min(kept_fls)) / (fused_fl or 1.0)
        limit = FUSION_SPREAD_LIMIT_ECAM if self._ecam else FUSION_SPREAD_LIMIT
        for c in kept_sorted:
            info["contributors"].append({"name": c["name"], "fl": round(c["fl"], 1),
                                         "samples": c["samples"], "weight": round(c["weight"], 1),
                                         "delta_pct": round((c["fl"] / fused_fl - 1) * 100, 2)})

        accepted = len(kept) >= 2 and spread <= limit and total_samples >= FUSION_MIN_SAMPLES
        if accepted:
            confidence = "high" if (spread <= limit * 0.67 and total_samples >= 200) else "medium"
        else:
            confidence = "low"
        if self._ecam and confidence == "high":
            confidence = "medium"      # 广角无畸变校正, 不判 high

        reason = ""
        if not accepted:
            if len(kept) < 2:
                reason = f"有效方法不足2个 (仅 {len(kept)}): 需≥2种独立真值交叉验证"
            elif spread > limit:
                reason = f"方法间分歧 {spread * 100:.1f}% 超过 {limit * 100:.0f}%"
            elif total_samples < FUSION_MIN_SAMPLES:
                reason = f"总样本 {total_samples} < {FUSION_MIN_SAMPLES}"

        info.update({
            "accepted": accepted,
            "fl_current": self._ref_fl,
            "fl_suggested": round(fused_fl) if accepted else None,
            "fl_suggested_raw": round(fused_fl),
            "fl_factor": round(fused_fl / self._ref_fl, 4) if self._ref_fl else 1.0,
            "fl_delta_pct": round((fused_fl / self._ref_fl - 1) * 100, 1) if self._ref_fl else 0.0,
            "confidence": confidence,
            "total_observations": total_samples,
            "spread_pct": round(spread * 100, 2),
            "spread_limit_pct": round(limit * 100, 1),
            "n_methods": len(kept),
            "method": "fusion",
            "reason": reason,
        })
        return info

    def compute(self) -> Dict:
        """三方法 (=三套独立真值) 加权融合。

        修正史: 旧实现只取"雷达-视觉比率"一个方法 (主路径), 车道宽仅在该方法无效时
        参与, 曲率法任何路径都不使用 —— 等于把单一方法的系统偏差 100% 吃进来。
        """
        rv_r = self._compute_ratio()
        lw_r = self.lw.compute()
        cv_r = self.cv.compute()

        result = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "camera": "ECAM" if self._ecam else "FCAM",
            "methods": {"radar_vision_ratio": rv_r, "lane_width": lw_r, "curvature": cv_r},
            "fused": {},
        }

        cands = []
        if rv_r.get("valid") and rv_r.get("fl_suggested"):
            cands.append({"name": "radar_vision_ratio", "fl": float(rv_r["fl_suggested"]),
                          "samples": int(rv_r.get("samples", 0))})
        if lw_r.get("valid") and lw_r.get("fl_suggested"):
            cands.append({"name": "lane_width", "fl": float(lw_r["fl_suggested"]),
                          "samples": int(lw_r.get("samples", 0))})
        if cv_r.get("valid") and cv_r.get("fl_suggested"):
            cands.append({"name": "curvature", "fl": float(cv_r["fl_suggested"]),
                          "samples": int(cv_r.get("samples_curved", cv_r.get("samples", 0)))})

        result["fused"] = self._fuse_methods(cands)
        return result

    def report(self, result: Dict = None) -> str:
        if result is None:
            result = self.compute()
        f = result["fused"]

        lines = ["=" * 58,
                 "  自标定报告 | Self-Calibration Report",
                 "=" * 58, "",
                 f"时间: {result['timestamp']}",
                 f"置信度: {f['confidence']}",
                 f"总样本: {f['total_observations']}", ""]

        # Method 1: ratio-based or linear
        is_ratio = "radar_vision_ratio" in result["methods"]
        rv_key = "radar_vision_ratio" if is_ratio else "radar_vision"
        rv = result["methods"][rv_key]
        lw = result["methods"]["lane_width"]
        cv = result["methods"]["curvature"]

        lines.append("─" * 58)
        if is_ratio:
            lines.append("  方法1: 雷达-视觉比率 (d_v/d_r) — 抗外参污染")
            lines.append("─" * 58)
            if rv.get("valid"):
                dtype = "ECAM" if rv["ratio_median"] > 1.5 else "FCAM"
                lines.append(f"  样本: {rv['samples']}/{rv['samples_total']}")
                lines.append(f"  比率中位数: {rv['ratio_median']:.3f} (均值: {rv['ratio_mean']:.3f} ± {rv['ratio_std']:.3f})")
                lines.append(f"  检测到: {dtype} (ratio > 1.5 为 ECAM)")
                lines.append(f"  FCAM参考fl: {rv['fcam_fl']}")
                for label, d in rv.get("stratified", {}).items():
                    lines.append(f"    {label}: ratio={d['mean_ratio']:.3f} → fl={d['fl_suggested']} (n={d['n']})")
                lines.append(f"  → 推算fl: {rv['fl_suggested']}")
            else:
                lines.append(f"  {rv.get('error', '数据不足')} (n={rv['samples']})")
        else:
            lines.append("  方法1: 雷达-视觉前车距离 (线性回归)")
            lines.append("─" * 58)
            if rv.get("valid"):
                lines.append(f"  样本: {rv['samples']}/{rv['samples_total']}")
                lines.append(f"  回归: d_vision = {rv['slope_k']:.4f} * d_radar + {rv['intercept_b']:.2f}")
                lines.append(f"  期望斜率为1, 截距≈-{CAMERA_RADAR_OFFSET}m (安装偏移)")
                for label, d in rv.get("stratified", {}).items():
                    if d["mean_delta"] is not None:
                        lines.append(f"    {label}: Δ={d['mean_delta']:.2f}m (n={d['n']})")
                lines.append(f"  → fl校正因子: {rv['fl_factor']:.4f}")
            else:
                lines.append(f"  {rv.get('error', '数据不足')} (n={rv['samples']})")

        lines.append("")
        lines.append("─" * 58)
        lines.append("  方法2: 车道宽度 (速度分类)")
        lines.append("─" * 58)
        if lw.get("valid"):
            lines.append(f"  总样本: {lw['samples']}")
            marking_w = lw.get("by_type", {}).get(list(lw["by_type"])[0], {}).get("marking_width", 0)
            if marking_w:
                lines.append(f"  (标线宽 {marking_w:.0f}cm: 内缘→中线校正)")
            for rt, d in lw.get("by_type", {}).items():
                label = {"highway": "高速 ≥90km/h→3.75m", "expressway": "快速路60-90→3.50m"}.get(rt, rt)
                lines.append(f"  {label}: {d['mean_width']:.2f}m (内缘 {d['mean_width_raw']:.2f}m) ± {d['std']:.2f} (n={d['n']}, 期望{d['expected']}m)")
            lines.append(f"  → fl校正因子: {lw['fl_factor']:.4f}")
        else:
            lines.append(f"  {lw.get('error', '数据不足')} (n={lw['samples']})")

        lines.append("")
        lines.append("─" * 58)
        lines.append("  方法3: IMU-模型曲率一致性")
        lines.append("─" * 58)
        lines.append(f"  直线样本(偏置诊断): {cv.get('samples_straight', 0)}  弯道样本(尺度): {cv.get('samples_curved', 0)}")
        if "bias_mean" in cv:
            lines.append(f"  偏置 (模型-IMU): {cv['bias_mean']:.6f} ± {cv['bias_std']:.6f} rad/m")
            lines.append(f"  最大偏置: {cv['bias_max_abs']:.6f} rad/m")
        if cv.get("valid"):
            lines.append(f"  弯道 κ_model/κ_imu 中位数: {cv['ratio_median']:.4f} (MAD {cv['ratio_spread']:.4f})")
            lines.append(f"  → 推算fl: {cv['fl_suggested']:.0f}")
        else:
            lines.append(f"  {cv.get('error', '数据不足')} — 该票不参与融合")

        lines.append("")
        lines.append("─" * 58)
        lines.append("  融合 (三真值加权平均 + MAD 剔除 + 一致性门限)")
        lines.append("─" * 58)
        if f.get("contributors"):
            for c in f["contributors"]:
                lines.append(f"  {c['name']}: fl={c['fl']} n={c['samples']} 权重={c['weight']} "
                             f"(相对融合值 {c['delta_pct']:+.2f}%)")
        for c in f.get("rejected", []):
            lines.append(f"  [剔除] {c['name']}: fl={c['fl']} ({c['delta_pct']:+.2f}%) — {c['reason']}")
        if f.get("spread_pct") is not None:
            lines.append(f"  方法间离散: {f['spread_pct']:.2f}% (上限 {f.get('spread_limit_pct')}%) "
                         f"有效方法数: {f.get('n_methods')}")

        lines.append("")
        lines.append("═" * 58)
        if not f.get("accepted"):
            lines.append("  结果: 未通过门限 -> 不写入 Params (继续采集)")
            lines.append(f"  原因: {f.get('reason') or '未知'}")
            if f.get("fl_suggested_raw"):
                lines.append(f"  参考值 (仅供判断, 未采纳): fl {f.get('fl_current')} → "
                             f"{f['fl_suggested_raw']} ({f.get('fl_delta_pct'):+.1f}%)")
        elif abs(f.get("fl_delta_pct", 0.0)) > 2:
            lines.append(f"  建议: fl {f['fl_current']} → {f['fl_suggested']} "
                         f"({f['fl_delta_pct']:+.1f}%) 置信={f['confidence']}")
            lines.append("  通过门限, 可由面板确认写入内参; 建议多趟数据交叉验证")
        else:
            lines.append(f"  结论: 内参在合理范围内 (偏差 {f.get('fl_delta_pct'):+.1f}%)")
            lines.append(f"  fl = {f.get('fl_current')} 无需调整")
        lines.append("=" * 58)
        return "\n".join(lines)


def _lead_present(lead) -> bool:
    """leadOne 是否有前车。

    字段名在 cereal 里改过: 老 schema 是 `status`, 新 schema (openpilot master /
    master-c3) 是 `present`。capnp 读不存在的字段会抛 AttributeError,
    所以两个都试, 都不行就当没有 —— 标定脚本不该因为 schema 改名整段跑不起来。
    """
    for attr in ("present", "status"):
        try:
            return bool(getattr(lead, attr))
        except Exception:
            continue
    return False


def find_routes(base_dir: str = None) -> List[str]:
    if base_dir is None:
        base_dir = _default_realdata()
    routes = []
    if not os.path.isdir(base_dir):
        print(f"错误: 日志目录不存在: {base_dir}")
        return routes
    for d in os.listdir(base_dir):
        full = os.path.join(base_dir, d)
        if os.path.isdir(full):
            for f in os.listdir(full):
                if f.endswith(".zst") and "rlog" in f:
                    routes.append(os.path.join(full, f))
    return sorted(routes)


def _default_realdata():
    """Path::log_root() —— 布局自适应, 且绝不抛异常。

    两点必须这样写:
      1. hw 模块在 fork 间搬过家 (新布局 openpilot/common/hardware/hw,
         老布局 openpilot/system/hardware/hw), 所以两个路径都试。
      2. 它被当作 argparse 的 default 求值, 在解析 --base-dir 之前就执行;
         一旦抛异常, 连显式传 --base-dir 都进不去 (master-c3 实测)。
    """
    for _mod in ("openpilot.common.hardware.hw", "openpilot.system.hardware.hw"):
        try:
            return __import__(_mod, fromlist=["Paths"]).Paths.log_root()
        except Exception:
            continue
    return os.path.expanduser("~/.comma/media/0/realdata")


def run_live(output_path: str, ecam: bool = False, fcam_fl: int = 2520):
    """Live collection mode: subscribe to messages, collect data, compute on exit."""
    # 导入要带 openpilot. 前缀: 本 fork 的源码在 <树根>/openpilot/ 子目录,
    # 裸的 `from common.params import ...` 只在老布局 (树根即 openpilot) 才能解析。
    try:
        from openpilot.common.params import Params
        from openpilot.cereal import messaging
    except ImportError:
        from common.params import Params
        from cereal import messaging

    exit_now = False
    def _handler(sig, fr):
        nonlocal exit_now
        exit_now = True
    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)

    params = Params()
    cal = SelfCalibrator(ecam=ecam, fcam_fl=fcam_fl)
    sm = messaging.SubMaster(['carState', 'radarState', 'modelV2'])

    print("实时采集已启动，等待数据...", flush=True)
    tick = 0
    start_time = time.time()
    active_key = "WideCalibActive" if ecam else "FcamLiveActive"
    while not exit_now and params.get_bool(active_key):
        sm.update()
        if sm.updated['carState']:
            cal._last_v = sm['carState'].vEgo
            cal._last_yaw = getattr(sm['carState'], 'yawRate', 0)
        if sm.updated['radarState']:
            rs = sm['radarState']
            if _lead_present(rs.leadOne):
                cal._last_radar_d = rs.leadOne.dRel
        if sm.updated['modelV2']:
            md = sm['modelV2']
            leads = md.leadsV3
            if leads[0].prob > 0.5:
                cal._last_vision_x = leads[0].x[0]
                cal.rv.add(cal._last_radar_d, cal._last_vision_x, cal._last_v)
            probs = list(md.laneLineProbs)
            if len(probs) >= 3 and probs[1] > 0.5 and probs[2] > 0.5:
                try:
                    l1 = md.laneLines[1]; l2 = md.laneLines[2]
                    x1, y1 = np.array(list(l1.x)), np.array(list(l1.y))
                    x2, y2 = np.array(list(l2.x)), np.array(list(l2.y))
                    y1_0 = np.interp(0, x1, y1); y2_0 = np.interp(0, x2, y2)
                    cal.lw.add(y2_0 - y1_0, cal._last_v)
                except:
                    pass
            try:
                mu = md.action
                cal.cv.add(cal._last_yaw, cal._last_v, mu.desiredCurvature)
            except:
                pass
        tick += 1
        if tick % 50 == 0:
            elapsed = time.time() - start_time
            print(f"收集 rv={cal.rv.count} lw={cal.lw.count} cv={cal.cv.count} 时间={elapsed:.0f}s", flush=True)

    print(f"计算 进度=0%", flush=True)
    print(f"\n采集结束，共 {cal.rv.count + cal.lw.count + cal.cv.count} 个样本，开始计算...")
    result = cal.compute()
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print(f"结果已保存: {output_path}")
    print()
    print(cal.report(result))
    f = result["fused"]
    label = "ECAM" if ecam else "FCAM"
    if f.get("accepted"):
        fl_c = f["fl_current"]
        fl_s = f["fl_suggested"]
        pct = (fl_s / fl_c - 1) * 100
        print(f"结果 {label} fl={fl_c}→{fl_s} ({pct:+.1f}%) 置信={f['confidence']} "
              f"方法数={f.get('n_methods')} 离散={f.get('spread_pct')}% 样本={f['total_observations']}", flush=True)
    else:
        print(f"结果 {label}: 未通过一致性门限, 不写入 Params — {f.get('reason')}", flush=True)
        if f.get("fl_suggested_raw"):
            print(f"  (参考值 未采纳: fl {f.get('fl_current')} → {f['fl_suggested_raw']})", flush=True)

    # Save to Params for UI to read
    # 两道关:
    #   1. 必须走 params_json: 本 fork 的 FcamCalibResult/WideCalibResult 是 JSON 型键,
    #      put(key, json.dumps(x)) 会 TypeError 被裸 except 吞掉 -> UI 永远读不到结果。
    #   2. 只有通过融合门限 (≥2 个方法一致 + 样本够) 才允许写入 —— fl 直接影响视觉测距,
    #      不合格宁可让面板显示"无数据", 也不能把一个可疑值送进内参。
    try:
        from params_json import params_json_put
        result_key = "WideCalibResult" if ecam else "FcamCalibResult"
        if f.get("accepted"):
            params_json_put(result_key, result, params)
            print(f"已写入 Params: {result_key}", flush=True)
        else:
            params.remove(result_key)      # 清掉上一轮的旧结果, 免得面板拿着老值让你确认
            print(f"未采纳: 已清空 {result_key}", flush=True)
    except Exception as e:
        print(f"写入 Params 失败: {e}", flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Self-calibration for camera intrinsics")
    parser.add_argument("--route", type=str, help="Route name, e.g. 0000003b--b85bf4ef47--2")
    parser.add_argument("--routes", type=str, help="Comma-separated route names")
    parser.add_argument("--scan", action="store_true", help="Scan all routes in realdata/")
    parser.add_argument("--live", action="store_true", help="Live collection mode (SIGTERM to stop)")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--output", type=str, help="Save JSON result to file")
    parser.add_argument("--base-dir", type=str, default=_default_realdata())
    parser.add_argument("--ecam", action="store_true", help="ECAM (wide camera) calibration mode")
    args = parser.parse_args()

    # Read true FCAM focal length from Params (fallback 2520)
    fcam_fl = 2520
    try:
        # JSON 型键在本 fork 的 get() 已返回 dict, json.loads 会 TypeError 被 except 吞掉,
        # 结果标定好的 fl 读不到、永远用 2520 默认值。
        from params_json import params_json_get
        fcam_fl = int((params_json_get("FcamIntrinsics") or {}).get("fl", 2520))
    except: pass
    FL_CURRENT = fcam_fl

    # For ECAM mode, we override FL_CURRENT but keep fcam_fl for ratio calc
    if args.ecam:
        FL_CURRENT = 605
        try:
            from params_json import params_json_get
            _fl = (params_json_get("EcamIntrinsics") or {}).get("fl")
            if _fl: FL_CURRENT = int(_fl)
        except: pass

    if args.live:
        # ECAM 模式下要标定的是 ECAM, 参考 fl 必须传 ECAM 的当前值; 否则融合会把
        # 建议值算到 FCAM 的 2520 上 (面板随后按 EcamIntrinsics 比对, 直接错位)。
        run_live(args.output, ecam=args.ecam, fcam_fl=(FL_CURRENT if args.ecam else fcam_fl))
        return

    if args.json and args.route and args.route.endswith(".json"):
        with open(args.route) as f:
            result = json.load(f)
            print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return

    paths = []
    if args.route:
        for candidate in [args.route, os.path.join(args.base_dir, args.route)]:
            if os.path.isdir(candidate):
                for f in sorted(os.listdir(candidate)):
                    if "rlog" in f and f.endswith(".zst"):
                        paths.append(os.path.join(candidate, f))
                break
            elif os.path.isfile(candidate) and candidate.endswith(".zst"):
                paths.append(candidate)
                break
        else:
            prefix = args.route + "--"
            for entry in sorted(os.listdir(args.base_dir)):
                if entry.startswith(prefix):
                    p = os.path.join(args.base_dir, entry, "rlog.zst")
                    if os.path.isfile(p):
                        paths.append(p)
    elif args.routes:
        for r in args.routes.split(","):
            prefix = r + "--"
            for entry in sorted(os.listdir(args.base_dir)):
                if entry.startswith(prefix):
                    p = os.path.join(args.base_dir, entry, "rlog.zst")
                    if os.path.isfile(p):
                        paths.append(p)
    elif args.scan:
        paths = find_routes(args.base_dir)

    if not paths:
        print("未找到日志文件")
        return

    cal = SelfCalibrator(fcam_fl=fcam_fl)
    total_msgs = 0
    total_frames = 0
    print(f"分析 {len(paths)} 个日志段...")
    for i, p in enumerate(paths):
        short = os.path.basename(os.path.dirname(p))
        print(f"  [{i+1}/{len(paths)}] {short}...", end=" ", flush=True)
        msgs, frames = cal.process_log(p)
        total_msgs += msgs
        total_frames += frames
        print(f"{msgs}条消息, {frames}帧模型")

    result = cal.compute()
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print(f"结果已保存: {args.output}")
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    else:
        print()
        print(cal.report(result))


if __name__ == "__main__":
    main()
