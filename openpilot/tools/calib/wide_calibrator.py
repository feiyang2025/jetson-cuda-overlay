#!/usr/bin/env python3
"""
Wide-angle camera (ECAM) calibration via stereo with FCAM.

Simple workflow:
  UI toggle ON → starts this process
  During driving → collects FCAM/ECAM stereo matches via visionipc
  Openpilot shutdown (SIGTERM) or toggle OFF (SIGINT):
    → auto-computes ECAM fl calibration
    → saves result to /tmp/wide_calib_result.json + Params
    → exits

  Next startup: checks for unprocessed data from last run.

Usage:
  python wide_calibrator.py                    # collect + auto-compute on exit
  python wide_calibrator.py --compute          # compute from saved matches
  python wide_calibrator.py --status           # show last result
"""

import sys, os

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_log = None
def log(msg):
    global _log
    if _log is None:
        _log = open("/tmp/wide_calib_debug.log", "w", buffering=1)
    _log.write(f"{msg}\n")
    sys.stdout.write(f"{msg}\n")
    sys.stdout.flush()

log("=== wide_calibrator start ===")
log(f"argv={sys.argv}")
log(f"cwd={os.getcwd()}")

import numpy as np
import json
import signal
import time
from typing import Dict, List, Tuple

ECAM_FL_CURRENT = 605
FCAM_FL = 2690
R_F2E = np.eye(3, dtype=np.float64)
t_F2E = np.zeros(3, dtype=np.float64)

# (Re)built from the current fl globals; module-level F/K1/K2 are refreshed by
# _load_intrinsics() so a Params-provided calibration is actually honored.
K1 = None
K2 = None
F = None


def _rebuild_intrinsics():
    global K1, K2, F
    K1 = np.array([[FCAM_FL, 0, 960], [0, FCAM_FL, 540], [0, 0, 1]], dtype=np.float64)
    K2 = np.array([[ECAM_FL_CURRENT, 0, 960], [0, ECAM_FL_CURRENT, 540], [0, 0, 1]], dtype=np.float64)
    K1_inv = np.linalg.inv(K1)
    K2_inv_T = np.linalg.inv(K2).T
    t_skew = np.array([
        [0, -t_F2E[2], t_F2E[1]],
        [t_F2E[2], 0, -t_F2E[0]],
        [-t_F2E[1], t_F2E[0], 0]
    ], dtype=np.float64)
    F = K2_inv_T @ t_skew @ R_F2E @ K1_inv


def _load_intrinsics():
    global ECAM_FL_CURRENT, FCAM_FL
    try:
        from common.params import Params
        _f = json.loads(Params().get("FcamIntrinsics") or "{}").get("fl")
        if _f: FCAM_FL = int(_f)
        _e = json.loads(Params().get("EcamIntrinsics") or "{}").get("fl")
        if _e: ECAM_FL_CURRENT = int(_e)
    except Exception:
        pass
    _rebuild_intrinsics()


_rebuild_intrinsics()

OUTPUT_DIR = "."  # set at runtime; use ./ for current working dir
MATCHES_PATH = "wide_calib_matches.npz"
RESULT_PATH = "wide_calib_result.json"
FLAG_PENDING = "wide_calib_pending.flag"


class StereoMatcher:
    def __init__(self):
        import cv2
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def match(self, img1: np.ndarray, img2: np.ndarray) -> List[Tuple[float, float, float, float]]:
        import cv2
        kp1, des1 = self.orb.detectAndCompute(img1, None)
        kp2, des2 = self.orb.detectAndCompute(img2, None)
        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            log(f"match: too few features fcam={len(kp1) if kp1 else 0} ecam={len(kp2) if kp2 else 0}")
            return []
        matches = self.bf.match(des1, des2)
        log(f"match: bf found {len(matches)} matches")
        if len(matches) < 10:
            return []
        pts1 = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
        pts2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)
        h1 = np.hstack([pts1, np.ones((len(pts1), 1))])
        h2 = np.hstack([pts2, np.ones((len(pts2), 1))])
        Fx1 = (F @ h1.T).T
        Ftx2 = (h2 @ F).T
        d = np.sum(h2 * Fx1.T, axis=1)
        sampson = d ** 2 / np.maximum(Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[0, :] ** 2 + Ftx2[1, :] ** 2, 1e-10)
        mask = sampson < 3.0
        if sum(mask) < 10:
            mask = sampson < 8.0
        filtered = [(p1[0], p1[1], p2[0], p2[1]) for p1, p2 in zip(pts1[mask], pts2[mask])]
        log(f"match: after F filter {len(filtered)}")
        return filtered


class WideCalibComputer:
    def __init__(self):
        self.matches = []

    def load(self, path: str = MATCHES_PATH) -> bool:
        if not os.path.exists(path):
            return False
        d = np.load(path)
        m = d.get("matches")
        if m is not None and len(m) > 0:
            self.matches = [(r[0], r[1], r[2], r[3]) for r in m if len(r) >= 4]
        return len(self.matches) >= 20

    def load_from_list(self, matches: list):
        self.matches = matches

    def compute(self) -> Dict:
        if len(self.matches) < 20:
            return {"valid": False, "samples": len(self.matches), "fl_current": ECAM_FL_CURRENT}

        pts = np.array(self.matches)
        u1, v1, u2, v2 = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
        x1n = (u1 - K1[0, 2]) / K1[0, 0]
        y1n = (v1 - K1[1, 2]) / K1[1, 1]
        x2n = (u2 - K2[0, 2]) / K2[0, 0]
        y2n = (v2 - K2[1, 2]) / K2[1, 1]
        P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
        n_samp = min(len(self.matches), 500)

        def err_for(fl):
            K2t = K2.copy()
            K2t[0, 0] = K2t[1, 1] = fl
            P2 = K2t @ np.hstack([R_F2E, t_F2E.reshape(3, 1)])
            errs = []
            for i in range(n_samp):
                try:
                    A = np.zeros((4, 4))
                    A[0] = x1n[i] * P1[2] - P1[0]; A[1] = y1n[i] * P1[2] - P1[1]
                    A[2] = x2n[i] * P2[2] - P2[0]; A[3] = y2n[i] * P2[2] - P2[1]
                    _, _, Vt = np.linalg.svd(A); X = Vt[-1]
                    if abs(X[3]) < 1e-6: continue
                    X3 = X[:3] / X[3]
                    if X3[2] < 0: continue
                    x = P2 @ np.append(X3, 1)
                    if abs(x[2]) < 1e-6: continue
                    e = (u2[i] - x[0] / x[2]) ** 2 + (v2[i] - x[1] / x[2]) ** 2
                    if e < 10000: errs.append(e)
                except: continue
            return float('inf') if len(errs) < 5 else float(np.sqrt(np.mean(errs)))

        best_fl, best_err = ECAM_FL_CURRENT, err_for(ECAM_FL_CURRENT)
        results = []
        candidates = np.linspace(400, 900, 51)
        for i, fl in enumerate(candidates):
            e = err_for(fl)
            results.append((fl, e))
            if e < best_err: best_fl, best_err = fl, e
            if (i + 1) % 10 == 0 or i == len(candidates) - 1:
                log(f"计算 进度={i+1}/{len(candidates)} fl={fl:.0f} err={e:.2f}px")

        if len(results) >= 3:
            fls = np.array([r[0] for r in results])
            errs = np.array([r[1] for r in results])
            mi = int(np.argmin(errs))
            if 0 < mi < len(results) - 1:
                x0, x1, x2 = fls[mi-1:mi+2]; y0, y1, y2 = errs[mi-1:mi+2]
                d = (x0-x1)*(x0-x2)*(x1-x2)
                if abs(d) > 1e-10:
                    a = (x2*(y1-y0)+x1*(y0-y2)+x0*(y2-y1))/d
                    if abs(a) > 1e-10:
                        refined = -((x2**2*(y0-y1)+x1**2*(y2-y0)+x0**2*(y1-y2))/d)/(2*a)
                        if 400 < refined < 900: best_fl = refined

        return {
            "valid": True, "samples": len(self.matches),
            "fl_current": ECAM_FL_CURRENT, "fl_suggested": round(best_fl),
            "fl_factor": round(best_fl / ECAM_FL_CURRENT, 4),
            "error_pixels": round(best_err, 3),
            "fl_range": (round(results[0][0]), round(results[-1][0])),
        }


def collect_matches() -> List[Tuple]:
    """Grab one FCAM/ECAM frame pair and return matches."""
    try:
        from msgq.visionipc import VisionIpcClient, VisionStreamType
    except ImportError:
        return None  # None = fatal, can't init

    attr = collect_matches.__dict__
    if "_state" not in attr:
        attr["_state"] = "init"
        attr["_matcher"] = StereoMatcher()
        attr["_fcam"] = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_ROAD, False)
        attr["_ecam"] = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD, False)
        if not attr["_fcam"].connect(True) or not attr["_ecam"].connect(True):
            attr["_state"] = "failed"
            return None
        attr["_state"] = "ready"

    if attr["_state"] != "ready":
        return None

    import cv2
    try:
        fcam_buf = attr["_fcam"].recv()
        ecam_buf = attr["_ecam"].recv()
    except Exception as e:
        log(f"collect recv error: {e}")
        return None
    if fcam_buf is None or ecam_buf is None:
        log(f"collect recv None: fcam={'OK' if fcam_buf else 'NONE'} ecam={'OK' if ecam_buf else 'NONE'}")
        return []
    ts_f = attr["_fcam"].timestamp_sof
    ts_e = attr["_ecam"].timestamp_sof
    if abs(ts_f - ts_e) > 2e7:
        log(f"collect ts diff too large: fcam={ts_f} ecam={ts_e} diff={abs(ts_f-ts_e)}")
        return []

    try:
        fcam_img = cv2.cvtColor(fcam_buf.data.reshape(fcam_buf.height * 3 // 2, fcam_buf.width), cv2.COLOR_YUV2GRAY_I420)
        ecam_img = cv2.cvtColor(ecam_buf.data.reshape(ecam_buf.height * 3 // 2, ecam_buf.width), cv2.COLOR_YUV2GRAY_I420)
    except Exception as e:
        log(f"collect img convert error: {e}")
        return []

    result = attr["_matcher"].match(fcam_img, ecam_img)
    log(f"collect match result: {len(result)} matches, fcam_size={fcam_buf.width}x{fcam_buf.height}")
    return result


def _paths(out_dir: str = None) -> dict:
    d = out_dir or OUTPUT_DIR
    return {
        "matches": os.path.join(d, "wide_calib_matches.npz"),
        "result": os.path.join(d, "wide_calib_result.json"),
        "pending": os.path.join(d, "wide_calib_pending.flag"),
    }


def save_matches(matches: list, out_dir: str = None):
    p = _paths(out_dir)
    np.savez_compressed(p["matches"], matches=np.array(matches))
    open(p["pending"], "w").close()


def compute_and_save(matches: list, out_dir: str = None):
    if len(matches) < 20:
        return {"valid": False, "samples": len(matches), "fl_current": ECAM_FL_CURRENT}

    computer = WideCalibComputer()
    computer.load_from_list(matches)
    result = computer.compute()

    p = _paths(out_dir)
    with open(p["result"], "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    try:
        from common.params import Params
        Params().put("WideCalibResult", json.dumps(result))
    except:
        pass

    for k in ["matches", "pending"]:
        try: os.remove(p[k])
        except: pass

    return result


def main():
    try:
        _main()
    except Exception as e:
        import traceback
        log(f"FATAL: {e}")
        log(traceback.format_exc())

def _main():
    import argparse
    parser = argparse.ArgumentParser(description="Wide-angle camera calibration")
    parser.add_argument("--compute", action="store_true", help="Compute from saved matches")
    parser.add_argument("--status", action="store_true", help="Show last result")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: project root)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    log(f"_main: args={args}")
    _load_intrinsics()

    try:
        from common.params import Params
    except Exception as e:
        log(f"Params import failed: {e}")
        return

    if args.output_dir is None:
        try:
            from openpilot.common.basedir import BASEDIR
            out_dir = BASEDIR
        except:
            out_dir = "."
    else:
        out_dir = args.output_dir
    p = _paths(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # ── Status mode ──
    if args.status:
        if os.path.exists(p["result"]):
            with open(p["result"]) as f:
                log(json.dumps(json.load(f), indent=2, ensure_ascii=False))
        else:
            log("No result yet")
        return

    # ── Compute mode ──
    if args.compute:
        if not os.path.exists(p["matches"]):
            log("No match data found")
            return
        d = np.load(p["matches"])
        m = d.get("matches")
        matches = [(r[0], r[1], r[2], r[3]) for r in m if len(r) >= 4] if m is not None else []
        result = compute_and_save(matches, out_dir)
        if args.json:
            log(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            r = result
            log(f"ECAM fl: {r['fl_current']} → {r.get('fl_suggested', r['fl_current'])} ({r.get('fl_factor', 1.0):+.4f})")
        return

    # ── Collection mode (default) ──
    if os.path.exists(p["pending"]) and os.path.exists(p["matches"]):
        d = np.load(p["matches"])
        m = d.get("matches")
        pending_matches = [(r[0], r[1], r[2], r[3]) for r in m if len(r) >= 4] if m is not None else []
        if pending_matches:
            compute_and_save(pending_matches, out_dir)

    exit_now = False
    def _handler(sig, fr):
        nonlocal exit_now
        exit_now = True
    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)

    params = Params()
    all_matches = []
    last_save = 0
    frame_count = 0
    start_time = time.time()
    last_status = 0

    log("收集 初始化中...")
    while not exit_now and params.get_bool("WideCalibActive"):
        new_m = collect_matches()
        elapsed = time.time() - start_time

        if new_m is None:
            if elapsed > 5 and time.time() - last_status > 5:
                log(f"等待 无法连接摄像头（camerad 未运行?） 时间={elapsed:.0f}s")
                last_status = time.time()
            time.sleep(1)
            continue

        if new_m:
            all_matches.extend(new_m)
            frame_count += 1

            if frame_count % 100 == 0 and (time.time() - last_save) > 30:
                save_matches(all_matches, out_dir)
                log(f"保存 帧={frame_count} 匹配={len(all_matches)} 时间={elapsed:.0f}s")
                last_save = time.time()
            if frame_count % 10 == 0:
                rate = len(all_matches) / max(elapsed, 1)
                log(f"收集 帧={frame_count} 匹配={len(all_matches)} 时间={elapsed:.0f}s 速率={rate:.1f}/s")
                last_status = time.time()
        else:
            if time.time() - last_status > 5:
                if frame_count > 0:
                    log(f"等待 已收集{frame_count}帧 {len(all_matches)}匹配 等待更多数据...")
                else:
                    log(f"等待 等待摄像头数据... 时间={elapsed:.0f}s")
                last_status = time.time()
            time.sleep(0.05)

    if all_matches:
        log("计算 进度=0/51")
        result = compute_and_save(all_matches, out_dir)
        if result.get("valid"):
            fl_c = result["fl_current"]
            fl_s = result["fl_suggested"]
            pct = (fl_s / fl_c - 1) * 100
            err = result["error_pixels"]
            log(f"结果 ECAM fl={fl_c}→{fl_s} ({pct:+.1f}%) err={err}px 样本={result['samples']}")
        else:
            log(f"结果 无效 样本={result['samples']}")
            save_matches(all_matches, out_dir)
    else:
        log("结果 无数据")
        rp = _paths(out_dir)
        if os.path.exists(rp["pending"]) and os.path.exists(rp["matches"]):
            d = np.load(rp["matches"])
            m = d.get("matches")
            pending = [(r[0], r[1], r[2], r[3]) for r in m if len(r) >= 4] if m is not None else []
            if pending:
                log(f"结果 处理历史数据 样本={len(pending)}")
                compute_and_save(pending, out_dir)


if __name__ == "__main__":
    main()
