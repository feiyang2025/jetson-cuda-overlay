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
    def __init__(self):
        self.pairs = []

    def add(self, yaw_rate: float, v_ego: float, model_curv: float):
        if v_ego < 5:
            return
        curv_imu = yaw_rate / v_ego
        if abs(curv_imu) < STRAIGHT_THRESHOLD and abs(model_curv) < 0.02:
            self.pairs.append((curv_imu, model_curv, v_ego))

    @property
    def count(self) -> int:
        return len(self.pairs)

    def compute(self) -> Dict:
        if len(self.pairs) < 30:
            return {"valid": False, "samples": len(self.pairs), "error": "不足30个样本"}

        c_imu = np.array([p[0] for p in self.pairs])
        c_model = np.array([p[1] for p in self.pairs])
        bias = c_model - c_imu

        return {
            "valid": True,
            "samples": len(self.pairs),
            "bias_mean": float(np.mean(bias)),
            "bias_std": float(np.std(bias)),
            "bias_max_abs": float(np.max(np.abs(bias))),
        }


class SelfCalibrator:
    def __init__(self, ecam: bool = False, fcam_fl: int = 2520):
        # ECAM mode uses a different reference fl; both fallback algorithms must
        # base their fl_suggested on it (module-global FL_CURRENT is unreliable).
        ref_fl = 605 if ecam else fcam_fl
        self.rv = RadarVisionCalibrator(fl_current=ref_fl)
        self.lw = LaneWidthCalibrator(fl_current=ref_fl)
        self.cv = CurvatureCalibrator()
        self._ecam = ecam
        self._fcam_fl = fcam_fl
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
                if rs.leadOne.status:
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

    def compute(self) -> Dict:
        # Unified ratio-based method — robust to extrinsics for both FCAM and ECAM
        rv_r = self._compute_ratio()
        lw_r = self.lw.compute()
        cv_r = self.cv.compute()

        result = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "methods": {"radar_vision_ratio": rv_r, "lane_width": lw_r, "curvature": cv_r},
            "fused": {},
        }

        if rv_r.get("valid"):
            n = rv_r["samples"]
            conf = "high" if n > 200 else "medium" if n > 50 else "low"
            ratio = rv_r["ratio_median"]
            label = "ECAM" if ratio > 1.5 else "FCAM"
            suggested = rv_r["fl_suggested"]
            result["fused"] = {
                "fl_current": self._fcam_fl,
                "fl_suggested": suggested,
                "fl_factor": round(self._fcam_fl / suggested if suggested > 0 else 1.0, 4),
                "fl_delta_pct": round((suggested / self._fcam_fl - 1) * 100, 1),
                "confidence": conf,
                "total_observations": len(self.rv.pairs),
                "method": f"ratio_{label}",
                "ratio_median": round(ratio, 2),
            }
        else:
            # Fallback: linear regression
            rv_lin = self.rv.compute()
            lw_r = self.lw.compute()
            fl_factors, weights = [], []
            if rv_lin.get("valid"):
                fl_factors.append(rv_lin["fl_factor"]); weights.append(rv_lin["samples"] * 2)
            if lw_r.get("valid"):
                fl_factors.append(lw_r["fl_factor"]); weights.append(lw_r["samples"])
            combined = float(np.average(fl_factors, weights=weights)) if fl_factors else 1.0
            total = self.rv.count + self.lw.count + self.cv.count
            conf = "high" if (rv_lin.get("valid") and total > 200) else "medium" if total > 50 else "low"
            result["fused"] = {
                "fl_current": self._fcam_fl,
                "fl_suggested": round(self._fcam_fl * combined),
                "fl_factor": round(combined, 4),
                "fl_delta_pct": round((combined - 1) * 100, 1),
                "confidence": conf,
                "total_observations": total,
            }

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
        if cv.get("valid"):
            lines.append(f"  样本: {cv['samples']}")
            lines.append(f"  偏置 (模型-IMU): {cv['bias_mean']:.6f} ± {cv['bias_std']:.6f} rad/m")
            lines.append(f"  最大偏置: {cv['bias_max_abs']:.6f} rad/m")
        else:
            lines.append(f"  {cv.get('error', '数据不足')} (n={cv['samples']})")

        lines.append("")
        lines.append("═" * 58)
        method = f.get("method", "linear")
        if "ratio" in method:
            is_ecam = "ECAM" in method
            rv = result["methods"].get("radar_vision_ratio", {})
            fcam_fl = rv.get("fcam_fl", f["fl_current"])
            fl_suggested = f["fl_suggested"]
            if is_ecam:
                # Compare against current ECAM setting
                ecam_current = 605
                try:
                    from openpilot.common.params import Params as P
                    _e = json.loads(P().get("EcamIntrinsics") or "{}").get("fl")
                    if _e: ecam_current = int(_e)
                except: pass
                delta = (fl_suggested / ecam_current - 1) * 100
                lines.append(f"  检测到: ECAM 模式")
                lines.append(f"  FCAM参考fl: {fcam_fl}")
                if abs(delta) > 2:
                    lines.append(f"  建议: ECAM fl {ecam_current} → {fl_suggested} ({delta:+.1f}%)")
                else:
                    lines.append(f"  结论: ECAM fl = {ecam_current} 合理")
            else:
                lines.append(f"  检测到: FCAM 模式")
                if abs(f["fl_delta_pct"]) > 2:
                    lines.append(f"  建议: fl {f['fl_current']} → {fl_suggested} ({f['fl_delta_pct']:+.1f}%)")
                    lines.append(f"  警告: 建议确认后再修改 camera.py")
                else:
                    lines.append(f"  结论: 内参在合理范围内 (偏差 {f['fl_delta_pct']:+.1f}%)")
        else:
            if abs(f["fl_delta_pct"]) > 2 and f["confidence"] != "low":
                lines.append(f"  建议: fl {f['fl_current']} → {f['fl_suggested']} ({f['fl_delta_pct']:+.1f}%)")
                lines.append(f"  警告: 建议确认后再修改 camera.py")
            elif f["confidence"] == "low":
                lines.append(f"  数据不足, 继续收集...")
                lines.append(f"  fl当前: {f['fl_current']}, 建议: {f['fl_suggested']}")
            else:
                lines.append(f"  结论: 内参在合理范围内 (偏差 {f['fl_delta_pct']:+.1f}%)")
                lines.append(f"  fl = {f['fl_current']} 无需调整")
        lines.append("=" * 58)
        return "\n".join(lines)


def find_routes(base_dir: str = None) -> List[str]:
    if base_dir is None:
        from openpilot.system.hardware.hw import Paths
        base_dir = Paths.log_root()
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
    from openpilot.system.hardware.hw import Paths
    return Paths.log_root()


def run_live(output_path: str, ecam: bool = False, fcam_fl: int = 2520):
    """Live collection mode: subscribe to messages, collect data, compute on exit."""
    from common.params import Params
    from cereal import messaging

    exit_now = False
    def _handler(sig, fr):
        nonlocal exit_now
        exit_now = True
    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)

    params = Params()
    cal = SelfCalibrator(fcam_fl=fcam_fl)
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
            if rs.leadOne.status:
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
    if f.get("fl_suggested"):
        fl_c = f["fl_current"]
        fl_s = f["fl_suggested"]
        pct = (fl_s / fl_c - 1) * 100
        print(f"结果 {label} fl={fl_c}→{fl_s} ({pct:+.1f}%) 置信={f['confidence']} 样本={f['total_observations']}", flush=True)

    # Save to Params for UI to read
    try:
        result_key = "WideCalibResult" if ecam else "FcamCalibResult"
        params.put(result_key, json.dumps(result))
    except:
        pass


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
        from openpilot.common.params import Params
        import json
        fcam_fl = json.loads(Params().get("FcamIntrinsics") or "{}").get("fl", 2520)
    except: pass
    FL_CURRENT = fcam_fl

    # For ECAM mode, we override FL_CURRENT but keep fcam_fl for ratio calc
    if args.ecam:
        FL_CURRENT = 605
        try:
            from openpilot.common.params import Params as P2
            _fl = json.loads(P2().get("EcamIntrinsics") or "{}").get("fl")
            if _fl: FL_CURRENT = int(_fl)
        except: pass

    if args.live:
        run_live(args.output, ecam=args.ecam, fcam_fl=fcam_fl)
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
