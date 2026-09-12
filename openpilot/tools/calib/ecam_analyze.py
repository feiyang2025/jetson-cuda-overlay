#!/usr/bin/env python3
"""
Analyze per-frame radar-vision ratio to separate FCAM/ECAM frames
and estimate ECAM focal length.

For each frame with a valid lead:
  ratio = d_vision / d_radar
  
  With correct extrinsics:  ratio ≈ fl_fcam / fl_cam
    FCAM: ratio ≈ 1
    ECAM: ratio ≈ fl_fcam / fl_ecam ≈ 2656/605 ≈ 4.4

  With wrong extrinsics: ratio deviates due to pitch/height errors.
  We cluster frames by ratio and pick the ECAM cluster.
"""

import numpy as np
import os
import json
import argparse
import sys
from collections import Counter
from openpilot.tools.lib.logreader import LogReader


def find_rlogs(base_dir: str, route: str = None):
    rlogs = []
    if route:
        prefix = route + "--"
        for entry in sorted(os.listdir(base_dir)):
            if entry.startswith(prefix):
                p = os.path.join(base_dir, entry, "rlog.zst")
                if os.path.isfile(p):
                    rlogs.append(p)
    else:
        for entry in sorted(os.listdir(base_dir)):
            full = os.path.join(base_dir, entry)
            rlog = os.path.join(full, "rlog.zst")
            if os.path.isfile(rlog):
                rlogs.append(rlog)
    return rlogs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", help="Directory with segment subdirs")
    parser.add_argument("--route", type=str)
    parser.add_argument("--base-dir", default="/home/dengjian/realdata")
    parser.add_argument("--output", default="ecam_analyze.json")
    parser.add_argument("--ratio-min", type=float, default=1.5,
                        help="Minimum d_vision/d_radar ratio to consider ECAM (default: 1.5)")
    parser.add_argument("--fcam-fl", type=float, default=2656,
                        help="FCAM focal length (default: 2656 from FcamIntrinsics)")
    args = parser.parse_args()

    paths = []
    if args.dir:
        paths = find_rlogs(args.dir)
    elif args.route:
        paths = find_rlogs(args.base_dir, args.route)
    else:
        print("Specify --dir or --route")
        return

    if not paths:
        print("No rlog files found")
        return

    # Read FCAM fl from Params if available
    fcam_fl = args.fcam_fl
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
        import json, importlib
        Params = importlib.import_module("openpilot.common.params").Params
        _fl = json.loads(Params().get("FcamIntrinsics") or "{}").get("fl")
        if _fl: fcam_fl = int(_fl)
    except: pass
    print(f"FCAM fl: {fcam_fl}")

    # Collect per-frame ratios
    ratios = []
    pairs = []
    for p in paths:
        radar_d = None
        seg_name = os.path.basename(os.path.dirname(p))
        for m in LogReader(p):
            w = m.which()
            if w == 'radarState' and m.radarState.leadOne.status:
                radar_d = m.radarState.leadOne.dRel
            elif w == 'modelV2' and m.modelV2.leadsV3[0].prob > 0.5 and radar_d is not None:
                d_v = float(m.modelV2.leadsV3[0].x[0])
                d_r = float(radar_d)
                if d_r > 5 and d_v > 5:
                    ratio = d_v / d_r
                    ratios.append(ratio)
                    pairs.append((d_r, d_v, ratio, seg_name))

    ratios = np.array(ratios)
    print(f"\nTotal pairs: {len(ratios)}")
    print(f"Ratio stats:")
    print(f"  min:  {ratios.min():.3f}")
    print(f"  max:  {ratios.max():.3f}")
    print(f"  mean: {ratios.mean():.3f}")
    print(f"  median: {np.median(ratios):.3f}")
    print(f"  std:  {ratios.std():.3f}")

    # Histogram
    hist, bins = np.histogram(ratios, bins=80, range=(0, 10))
    print(f"\nRatio distribution (top 10 peaks):")
    peak_indices = np.argsort(hist)[-10:][::-1]
    for idx in peak_indices:
        if hist[idx] > 0:
            print(f"  ratio≈{bins[idx]+(bins[idx+1]-bins[idx])/2:.2f}: {hist[idx]} frames")

    # Separate ECAM frames: ratio > threshold
    ecam_mask = ratios > args.ratio_min
    fcam_mask = ~ecam_mask
    n_ecam = int(ecam_mask.sum())
    n_fcam = int(fcam_mask.sum())
    print(f"\nFCAM frames (ratio ≤ {args.ratio_min}): {n_fcam}")
    print(f"ECAM frames (ratio > {args.ratio_min}): {n_ecam}")

    # Per-segment breakdown
    print(f"\nPer-segment breakdown:")
    seg_data = {}
    for d_r, d_v, ratio, seg in pairs:
        if seg not in seg_data:
            seg_data[seg] = {"ecam_pairs": [], "fcam_pairs": []}
        bucket = "ecam_pairs" if ratio > args.ratio_min else "fcam_pairs"
        seg_data[seg][bucket].append((d_r, d_v, ratio))

    for seg in sorted(seg_data.keys()):
        d = seg_data[seg]
        n_ec = len(d["ecam_pairs"])
        n_fc = len(d["fcam_pairs"])
        ratios_ec = np.array([x[2] for x in d["ecam_pairs"]])
        ratios_fc = np.array([x[2] for x in d["fcam_pairs"]])
        med_ec = np.median(ratios_ec) if n_ec > 0 else 0
        med_fc = np.median(ratios_fc) if n_fc > 0 else 0
        ec_fl = fcam_fl / med_ec if med_ec > 0 else 0
        print(f"  {seg}:  ECAM={n_ec} (median ratio={med_ec:.2f}, fl_ecam≈{ec_fl:.0f})  FCAM={n_fc} (ratio={med_fc:.2f})")

    # For ECAM frames, filter by distance and compute fl
    ecam_pairs = [p for i, p in enumerate(pairs) if ecam_mask[i]]
    ecam_ratios = ratios[ecam_mask]
    ecam_d_r = np.array([p[0] for p in ecam_pairs])
    ecam_d_v = np.array([p[1] for p in ecam_pairs])

    # Remove ratio outliers (3-sigma)
    r_mean, r_std = np.mean(ecam_ratios), np.std(ecam_ratios)
    r_mask = np.abs(ecam_ratios - r_mean) < 3 * r_std
    ecam_ratios_clean = ecam_ratios[r_mask]
    ecam_d_r_clean = ecam_d_r[r_mask]
    ecam_d_v_clean = ecam_d_v[r_mask]

    print(f"\nECAM analysis ({len(ecam_ratios)} frames):")
    print(f"  Mean ratio: {np.mean(ecam_ratios):.3f} ± {np.std(ecam_ratios):.3f}")
    print(f"  Median ratio: {np.median(ecam_ratios):.3f}")
    print(f"  ECAM fl = fcam_fl / ratio:")

    med_ratio = np.median(ecam_ratios_clean)
    mean_ratio = np.mean(ecam_ratios_clean)
    print(f"    from median: fl_ecam = {fcam_fl} / {med_ratio:.3f} = {fcam_fl/med_ratio:.0f}")
    print(f"    from mean:   fl_ecam = {fcam_fl} / {mean_ratio:.3f} = {fcam_fl/mean_ratio:.0f}")

    # By distance range
    print(f"\n  By distance range:")
    for d_min, d_max in [(5, 20), (20, 50), (50, 100)]:
        m = (ecam_d_r_clean >= d_min) & (ecam_d_r_clean < d_max)
        if m.sum() > 5:
            r_sub = ecam_ratios_clean[m]
            print(f"    {d_min}-{d_max}m:  ratio={r_sub.mean():.3f}±{r_sub.std():.3f}  fl={fcam_fl/r_sub.mean():.0f}  (n={int(m.sum())})")

    # For near-range frames only (less affected by pitch error)
    near_mask = ecam_d_r_clean < 30
    r_near = ecam_ratios_clean[near_mask]
    if len(r_near) > 10:
        print(f"\n  Near range (<30m, best for fl estimation):")
        print(f"    ratio={np.mean(r_near):.3f}±{np.std(r_near):.3f}  fl_ecam={fcam_fl/np.mean(r_near):.0f}  (n={len(r_near)})")

    # Linear fit per distance bin
    print(f"\n  Linear fit on ECAM pairs (near, b forced close to 0):")
    A = np.vstack([ecam_d_r_clean, np.ones(len(ecam_d_r_clean))]).T
    k, b = np.linalg.lstsq(A, ecam_d_v_clean, rcond=None)[0]
    print(f"    d_vision = {k:.4f} * d_radar + {b:.2f}")
    print(f"    fl_ecam = fcam_fl / k = {fcam_fl/k:.0f}")

    # Only near range linear fit
    near_idx = ecam_d_r_clean < 30
    if near_idx.sum() > 20:
        A_n = np.vstack([ecam_d_r_clean[near_idx], np.ones(near_idx.sum())]).T
        k_n, b_n = np.linalg.lstsq(A_n, ecam_d_v_clean[near_idx], rcond=None)[0]
        print(f"    near only (<30m): d_v = {k_n:.4f} * d_r + {b_n:.2f}  fl_ecam={fcam_fl/k_n:.0f}")

    # Save
    result = {
        "fcam_fl": int(fcam_fl),
        "ecam_ratio_threshold_used": args.ratio_min,
        "n_fcam": n_fcam,
        "n_ecam": int(n_ecam),
        "ecam_ratio_mean": round(float(np.mean(ecam_ratios)), 3),
        "ecam_ratio_median": round(float(np.median(ecam_ratios)), 3),
        "ecam_ratio_std": round(float(np.std(ecam_ratios)), 3),
        "ecam_fl_from_median": int(fcam_fl / float(np.median(ecam_ratios_clean))),
        "ecam_fl_from_mean": int(fcam_fl / float(np.mean(ecam_ratios_clean))),
        "ecam_fl_from_near": int(fcam_fl / float(np.mean(ecam_ratios_clean[ecam_d_r_clean < 30]))) if (ecam_d_r_clean < 30).sum() > 10 else 0,
    }
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
