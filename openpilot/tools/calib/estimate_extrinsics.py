#!/usr/bin/env python3
"""
Estimate ECAM extrinsics from radar-vision mismatch.

Model: pinhole camera with pitch θ, height h.
  true distance d_r → image row y → model estimates d_v using wrong (h₀, θ₀)

  Image projection (true):  y = cy - fl * (-h*cosθ + d_r*sinθ) / (h*sinθ + d_r*cosθ)
  Model inversion (wrong):  d_v = h₀ * (y - cy + fl*sinθ₀) / (fl*cosθ₀ - (y-cy)*sinθ₀)
  
  Substituting y into d_v gives d_v(d_r, h, θ, h₀, θ₀, fl).
"""

import numpy as np
import os
import json
import argparse
from openpilot.tools.lib.logreader import LogReader


def find_rlogs(base_dir: str):
    rlogs = []
    for entry in sorted(os.listdir(base_dir)):
        full = os.path.join(base_dir, entry)
        rlog = os.path.join(full, "rlog.zst")
        if os.path.isfile(rlog):
            rlogs.append(rlog)
    return rlogs


def collect_pairs(rlog_paths: list):
    pairs = []
    for p in rlog_paths:
        radar_d = None
        for m in LogReader(p):
            w = m.which()
            if w == 'radarState' and m.radarState.leadOne.status:
                radar_d = m.radarState.leadOne.dRel
            elif w == 'modelV2' and m.modelV2.leadsV3[0].prob > 0.5 and radar_d is not None:
                d_v = float(m.modelV2.leadsV3[0].x[0]) * np.sign(m.modelV2.leadsV3[0].x[0])
                if d_v > 2:
                    pairs.append((float(radar_d), d_v))
    return pairs


def d_vision_pinhole(d, h, pitch, h0, pitch0, fl):
    """Model computes distance using wrong extrinsics (h0, pitch0).
       Returns: d_v given true distance d.
    """
    sp, cp = np.sin(pitch), np.cos(pitch)
    sp0, cp0 = np.sin(pitch0), np.cos(pitch0)

    # True projection: (y - cy) / fl = Y' / Z'
    # Y' = -h*cosθ + d*sinθ,   Z' = h*sinθ + d*cosθ
    num = -h * cp + d * sp
    den = h * sp + d * cp
    if abs(den) < 1e-10:
        return d
    y_prime = num / den  # (y - cy) / fl

    # Model inversion using wrong (h0, pitch0):
    #   y_prime = (-h0*cosθ0 + d_v*sinθ0) / (h0*sinθ0 + d_v*cosθ0)
    # → d_v = h0 * (y_prime*sinθ0 + cosθ0) / (sinθ0 - y_prime*cosθ0)
    numerator = h0 * (y_prime * sp0 + cp0)
    denominator = sp0 - y_prime * cp0
    if abs(denominator) < 1e-10:
        return d
    return numerator / denominator


def cost_params(params, pairs, h0_fixed=None, pitch0_fixed=None):
    if h0_fixed is not None and pitch0_fixed is not None:
        fl, h, pitch = params
        h0, pitch0 = h0_fixed, pitch0_fixed
    elif len(params) == 5:
        fl, h, pitch, h0, pitch0 = params
    else:
        fl, h, pitch = params
        h0, pitch0 = 1.3, 0.05

    errs = []
    for d_r, d_v in pairs:
        pred = d_vision_pinhole(d_r, h, pitch, h0, pitch0, fl)
        errs.append(d_v - pred)
    return np.array(errs)


def grid_search(pairs, d_r, d_v, fl_init=605):
    print("Grid search over (h, pitch, h0, pitch0, fl)...")
    best = None
    best_rmse = float('inf')
    results = []

    for fl in [500, 605, 700, 800, 900, 1000]:
        for h in np.arange(0.8, 2.1, 0.2):
            for pitch in np.arange(0.01, 0.15, 0.02):
                for dh in [-0.5, 0, 0.5]:
                    h0 = h + dh
                    for dp in [-0.02, 0, 0.02]:
                        pitch0 = pitch + dp
                        if h0 < 0.3 or pitch0 < 0:
                            continue
                        preds = np.array([d_vision_pinhole(x, h, pitch, h0, pitch0, fl) for x in d_r])
                        rmse = float(np.sqrt(np.mean((preds - d_v)**2)))
                        results.append((rmse, fl, h, pitch, h0, pitch0))
                        if rmse < best_rmse:
                            best_rmse = rmse
                            best = (fl, h, pitch, h0, pitch0, rmse)

    results.sort(key=lambda x: x[0])
    print(f"\nGrid search complete. Best RMSE={best_rmse:.2f}m")
    print(f"  fl={best[0]}  h={best[1]:.3f}m  pitch={np.degrees(best[2]):.1f}°")
    print(f"  (assumed) h0={best[3]:.3f}m  pitch0={np.degrees(best[4]):.1f}°")
    print(f"\nTop 5:")
    for r in results[:5]:
        print(f"  RMSE={r[0]:.2f}m  fl={r[1]}  h={r[2]:.2f}  pitch={np.degrees(r[3]):.1f}°  h0={r[4]:.2f}  pitch0={np.degrees(r[5]):.1f}°")

    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", help="Directory with segment subdirs")
    parser.add_argument("--route", type=str)
    parser.add_argument("--base-dir", default="/home/dengjian/realdata")
    parser.add_argument("--output", default="extrinsics_result.json")
    parser.add_argument("--fl-init", type=int, default=605, help="Initial focal length")
    args = parser.parse_args()

    paths = []
    if args.route:
        if os.path.isfile(args.route) and args.route.endswith(".zst"):
            paths.append(args.route)
        elif args.dir:
            paths = find_rlogs(args.dir)
        else:
            prefix = args.route + "--"
            for entry in sorted(os.listdir(args.base_dir)):
                if entry.startswith(prefix):
                    p = os.path.join(args.base_dir, entry, "rlog.zst")
                    if os.path.isfile(p):
                        paths.append(p)
    elif args.dir:
        paths = find_rlogs(args.dir)

    if not paths:
        print("No rlog files found")
        return

    print(f"Reading {len(paths)} segments...")
    pairs = collect_pairs(paths)
    print(f"Collected {len(pairs)} radar-vision pairs")

    if len(pairs) < 50:
        print("Too few samples")
        return

    d_r = np.array([p[0] for p in pairs])
    d_v = np.array([p[1] for p in pairs])

    # Baseline: linear fit
    A = np.vstack([d_r, np.ones(len(d_r))]).T
    k, b = np.linalg.lstsq(A, d_v, rcond=None)[0]
    preds_lin = k * d_r + b
    rmse_lin = float(np.sqrt(np.mean((preds_lin - d_v)**2)))
    print(f"Linear baseline:  d_v = {k:.4f}*d_r + {b:.2f}  RMSE={rmse_lin:.2f}m")

    best = grid_search(pairs, d_r, d_v, args.fl_init)
    fl_opt, h_opt, pitch_opt, h0_opt, pitch0_opt, rmse_opt = best

    result = {
        "fl": int(fl_opt),
        "height_true": round(h_opt, 3),
        "pitch_true_rad": round(pitch_opt, 4),
        "pitch_true_deg": round(np.degrees(pitch_opt), 1),
        "height_assumed": round(h0_opt, 3),
        "pitch_assumed_rad": round(pitch0_opt, 4),
        "pitch_assumed_deg": round(np.degrees(pitch0_opt), 1),
        "rmse_linear": round(rmse_lin, 2),
        "rmse_pinhole": round(rmse_opt, 2),
    }

    print(f"\n{'='*50}")
    print(f"  ECAM Extrinsics Estimate")
    print(f"{'='*50}")
    print(f"  Assumed (current) params:")
    print(f"    height:   {result['height_assumed']:.2f} m")
    print(f"    pitch:    {result['pitch_assumed_deg']:.1f}° down")
    print(f"  Estimated (true) params:")
    print(f"    height:   {result['height_true']:.2f} m")
    print(f"    pitch:    {result['pitch_true_deg']:.1f}° down")
    print(f"  Focal length: {result['fl']}")
    print(f"  RMSE improvement: {rmse_lin:.1f}m → {rmse_opt:.1f}m")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
