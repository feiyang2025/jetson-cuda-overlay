#!/usr/bin/env python3
"""
Plot radar-vision calibration — three methods in one figure.

Usage:
  python plot_calib.py --dir /home/dengjian/.commaspold/media/0/realdata
  python plot_calib.py --dir /home/dengjian/.commaspold/media/0/realdata --output calib.png
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from openpilot.tools.lib.logreader import LogReader

try:
  from openpilot.common.transformations.camera import _read_calib_from_params
  import os
  _w = int(os.environ.get("ROAD_CAM_WIDTH", 1920))
  _h = int(os.environ.get("ROAD_CAM_HEIGHT", 1080))
  _base = 2580.0 * (_w / 1920.0)
  FL_CURRENT = int(_read_calib_from_params("FcamIntrinsics", _w, _h, _base)[0])
except Exception:
  FL_CURRENT = 2520

FCAM_FL = FL_CURRENT
STD_CLIP = 2.5
D_MIN, D_MAX = 5, 80
LANE_MARKING_WIDTH = 0.15
HIGHWAY_V = 90 / 3.6
EXPRESS_V = 60 / 3.6
STRAIGHT_THRESH = 0.008


def find_rlogs(base_dir):
    rlogs = []
    for entry in sorted(os.listdir(base_dir)):
        full = os.path.join(base_dir, entry)
        rlog = os.path.join(full, "rlog.zst")
        if os.path.isfile(rlog):
            rlogs.append(rlog)
    return rlogs


def collect_all(paths):
    rv_pairs = []
    lw_obs = []
    cv_pairs = []
    seg_counts = []

    for i, path in enumerate(paths):
        seg_name = os.path.basename(os.path.dirname(path))
        try:
            lr = LogReader(path)
            n_rv, n_lw, n_cv = 0, 0, 0
            radar_d, v_ego, yaw_rate = None, 0, 0
            for m in lr:
                w = m.which()
                if w == 'carState':
                    v_ego = m.carState.vEgo
                    yaw_rate = getattr(m.carState, 'yawRate', 0)
                elif w == 'radarState':
                    rs = m.radarState
                    radar_d = rs.leadOne.dRel if rs.leadOne.status else None
                elif w == 'modelV2':
                    md = m.modelV2
                    leads = md.leadsV3

                    # 1. Radar-Vision
                    if radar_d and leads[0].prob > 0.5:
                        vd = leads[0].x[0]
                        if D_MIN < radar_d < D_MAX and vd > 3:
                            rv_pairs.append((radar_d, vd))
                            n_rv += 1

                    # 2. Lane width
                    probs = list(md.laneLineProbs)
                    if len(probs) >= 3 and probs[1] > 0.5 and probs[2] > 0.5:
                        try:
                            l1, l2 = md.laneLines[1], md.laneLines[2]
                            x1, y1 = np.array(list(l1.x)), np.array(list(l1.y))
                            x2, y2 = np.array(list(l2.x)), np.array(list(l2.y))
                            w_left = np.interp(0, x1, y1)
                            w_right = np.interp(0, x2, y2)
                            lw_obs.append((w_right - w_left, v_ego))
                            n_lw += 1
                        except Exception:
                            pass

                    # 3. Curvature
                    try:
                        mu = md.action
                        cv_pairs.append((yaw_rate, v_ego, mu.desiredCurvature))
                        n_cv += 1
                    except Exception:
                        pass

            any_data = n_rv + n_lw + n_cv
            if any_data > 0:
                seg_counts.append(any_data)
                print(f"  [{i+1}/{len(paths)}] {seg_name}: rv={n_rv} lw={n_lw} cv={n_cv}")
        except Exception as e:
            print(f"  [{i+1}/{len(paths)}] {seg_name}: SKIP ({e})")

    return rv_pairs, lw_obs, cv_pairs, seg_counts


# ── Radar-Vision ──

def compute_rv(rv_pairs):
    if len(rv_pairs) < 30:
        return None
    d_r = np.array([p[0] for p in rv_pairs])
    d_v = np.array([p[1] for p in rv_pairs])
    ratios = d_v / d_r
    for _ in range(3):
        q1, q3 = np.percentile(ratios, [25, 75])
        iqr = q3 - q1
        mask = (ratios > q1 - STD_CLIP * iqr) & (ratios < q3 + STD_CLIP * iqr)
        if sum(mask) < 30:
            break
        ratios = ratios[mask]
        d_r = d_r[mask]
        d_v = d_v[mask]
    med = float(np.median(ratios))
    fl_sug = int(FCAM_FL / med) if med > 0 else FCAM_FL
    strata = {}
    for lo, hi, lbl in [(5, 20, "5-20m"), (20, 50, "20-50m"), (50, 80, "50-80m")]:
        m = (d_r >= lo) & (d_r < hi)
        if int(m.sum()) >= 5:
            strata[lbl] = {"n": int(m.sum()), "ratio": float(np.mean(d_v[m] / d_r[m]))}
    return {
        "d_r": d_r, "d_v": d_v, "ratios": ratios,
        "median": med, "fl_suggested": fl_sug,
        "samples_total": len(rv_pairs), "samples": len(d_r),
        "strata": strata,
    }


# ── Lane Width ──

def compute_lw(lw_obs):
    if len(lw_obs) < 10:
        return None
    by_type = defaultdict(list)
    for w, v in lw_obs:
        if v < 15 or not (2 < w < 6):
            continue
        if v >= HIGHWAY_V:
            by_type["highway"].append((w, 3.75))
        elif v >= EXPRESS_V:
            by_type["expressway"].append((w, 3.50))
    if not by_type:
        return None
    results = {}
    for rt, exp_w in [("highway", 3.75), ("expressway", 3.50)]:
        if rt not in by_type:
            continue
        ws = np.array([x[0] for x in by_type[rt]])
        w_corr = ws + LANE_MARKING_WIDTH
        r = float(np.mean(w_corr)) / exp_w
        results[rt] = {
            "n": len(ws), "mean_raw": float(np.mean(ws)),
            "mean_corr": float(np.mean(w_corr)), "expected": exp_w,
            "ratio": r, "std": float(np.std(ws)),
        }
    combined_r = float(
        np.average([v["ratio"] for v in results.values()],
                   weights=[v["n"] for v in results.values()])
    ) if results else 1.0
    return {
        "results": results, "combined_ratio": combined_r,
        "fl_factor": 1.0 / combined_r,
        "fl_suggested": int(FCAM_FL / combined_r),
        "samples": len(lw_obs),
    }


# ── Curvature ──

def compute_cv(cv_pairs):
    if len(cv_pairs) < 30:
        return None
    c_imu = np.array([p[0] / p[1] for p in cv_pairs if p[1] > 5])
    c_mod = np.array([-p[2] for p in cv_pairs if p[1] > 5])  # negate: align sign with IMU
    if len(c_imu) < 30:
        return None
    bias = c_mod - c_imu
    return {
        "bias": bias, "c_imu": c_imu, "c_mod": c_mod,
        "samples": len(bias),
        "bias_mean": float(np.mean(bias)),
        "bias_std": float(np.std(bias)),
    }


# ── Plot ──

def _plot(rv, lw, cv, seg_info, output):
    fig = plt.figure(figsize=(18, 12), facecolor='#1a1a1a')
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3,
                           height_ratios=[1.1, 1])

    # ──── Row 1: Radar-Vision ────
    ax_rv = fig.add_subplot(gs[0, 0])
    ax_rh = fig.add_subplot(gs[0, 1])
    ax_lw = fig.add_subplot(gs[0, 2])

    if rv:
        d_r, d_v, ratios = rv["d_r"], rv["d_v"], rv["ratios"]
        med = rv["median"]
        fl_sug = rv["fl_suggested"]
        fl_delta = (fl_sug / FCAM_FL - 1) * 100

        # 2D histogram
        ax_rv.set_facecolor('#2a2a2a')
        h = ax_rv.hist2d(d_r, d_v, bins=200, cmap='plasma',
                          range=[[0, 100], [0, 100]], cmin=1)
        fig.colorbar(h[3], ax=ax_rv, label='Obs', pad=0.01)
        xl = np.linspace(5, 100, 200)
        ax_rv.plot(xl, med * xl, '-', color='cyan', lw=2.5,
                   label=f'ratio = {med:.4f}')
        ax_rv.plot(xl, xl, ':', '#555', lw=1, label='ideal = 1.0')
        ax_rv.set_xlim(0, 100); ax_rv.set_ylim(0, 100)
        ax_rv.set_xlabel('Radar distance (m)', color='white')
        ax_rv.set_ylabel('Vision distance (m)', color='white')
        ax_rv.legend(loc='upper left', facecolor='#333', edgecolor='#555',
                     labelcolor='white', fontsize=8)
        ax_rv.grid(True, alpha=0.15)
        ax_rv.tick_params(colors='white')
        ax_rv.set_title('Radar-Vision: d_v vs d_r', color='white', fontsize=11)
        stats_rv = (f"n={rv['samples']:,}/{rv['samples_total']:,}\n"
                    f"median={med:.4f}\n"
                    f"fl: {FCAM_FL}→{fl_sug} ({fl_delta:+.1f}%)")
        ax_rv.text(0.97, 0.05, stats_rv, transform=ax_rv.transAxes,
                   fontsize=9, va='bottom', ha='right', color='white',
                   bbox=dict(boxstyle='round', facecolor='#333', edgecolor='#555', alpha=0.85))

        # Ratio histogram
        ax_rh.set_facecolor('#2a2a2a')
        ax_rh.hist(ratios, bins=120, color=plt.cm.plasma(0.5), alpha=0.8, edgecolor='none')
        ax_rh.axvline(med, color='cyan', lw=2, label=f'median={med:.4f}')
        y0 = ax_rh.get_ylim()[1] * 0.85
        for lbl, d in rv["strata"].items():
            ax_rh.axvline(d["ratio"], color='orange', lw=0.8, ls='--', alpha=0.6)
            ax_rh.text(d["ratio"], y0, f' {lbl} n={d["n"]}',
                       color='orange', fontsize=7, ha='center',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='#333', alpha=0.7))
            y0 -= ax_rh.get_ylim()[1] * 0.09
        ax_rh.set_xlabel('d_v / d_r', color='white')
        ax_rh.set_ylabel('Count', color='white')
        ax_rh.set_title('Ratio distribution', color='white', fontsize=11)
        ax_rh.legend(loc='upper right', facecolor='#333', edgecolor='#555',
                     labelcolor='white', fontsize=8)
        ax_rh.tick_params(colors='white')
        ax_rh.grid(True, alpha=0.15)
    else:
        for ax in [ax_rv, ax_rh]:
            ax.set_facecolor('#2a2a2a')
            ax.text(0.5, 0.5, 'Insufficient data', color='#888', fontsize=14,
                    ha='center', va='center', transform=ax.transAxes)
            ax.tick_params(colors='white')

    # ──── Row 1, Col 3: Lane Width ────
    ax_lw.set_facecolor('#2a2a2a')
    if lw:
        res = lw["results"]
        types = list(res.keys())
        x = np.arange(len(types))
        w = 0.28
        for i, rt in enumerate(types):
            d = res[rt]
            ax_lw.bar(i - w / 2, d["mean_raw"], w, label=f'{rt} raw' if i == 0 else '',
                      color='#e67e22', alpha=0.7)
            ax_lw.bar(i + w / 2, d["mean_corr"], w, label=f'{rt} corrected' if i == 0 else '',
                      color='#f1c40f', alpha=0.7)
            ax_lw.axhline(d["expected"], color='cyan', lw=1.5, ls=':',
                          xmin=(i + 0.2) / len(types),
                          xmax=(i + 0.8) / len(types))
        ax_lw.set_xticks(x)
        labels = []
        for rt in types:
            d = res[rt]
            labels.append(f'{rt}\n(d={d["std"]:.2f}, n={d["n"]})')
        ax_lw.set_xticklabels(labels, color='white', fontsize=8)
        ax_lw.set_ylabel('Width (m)', color='white')
        ax_lw.set_title(f'Lane Width  (factor={lw["fl_factor"]:.4f})',
                        color='white', fontsize=11)
        ax_lw.legend(loc='upper right', facecolor='#333', edgecolor='#555',
                     labelcolor='white', fontsize=7)
        ax_lw.tick_params(colors='white')
        ax_lw.grid(True, alpha=0.15, axis='y')
        stats_lw = (f'samples={lw["samples"]:,}\n'
                    f'combined ratio={lw["combined_ratio"]:.4f}\n'
                    f'fl→{lw["fl_suggested"]}')
        ax_lw.text(0.97, 0.05, stats_lw, transform=ax_lw.transAxes,
                   fontsize=8, va='bottom', ha='right', color='white',
                   bbox=dict(boxstyle='round', facecolor='#333', edgecolor='#555', alpha=0.85))
    else:
        ax_lw.text(0.5, 0.5, 'Insufficient data', color='#888', fontsize=14,
                   ha='center', va='center', transform=ax_lw.transAxes)
        ax_lw.tick_params(colors='white')

    # ──── Row 2: Curvature ────
    ax_cv_s = fig.add_subplot(gs[1, 0])
    ax_cv_h = fig.add_subplot(gs[1, 1])
    ax_sum = fig.add_subplot(gs[1, 2])

    if cv:
        bias = cv["bias"]
        c_imu, c_mod = cv["c_imu"], cv["c_mod"]

        # Scatter
        ax_cv_s.set_facecolor('#2a2a2a')
        ax_cv_s.scatter(c_imu, c_mod, s=1, color=plt.cm.plasma(0.3), alpha=0.3)
        lim = max(abs(c_imu).max(), abs(c_mod).max()) * 1.1
        ax_cv_s.plot([-lim, lim], [-lim, lim], ':', '#555', lw=1)
        ax_cv_s.set_xlim(-lim, lim); ax_cv_s.set_ylim(-lim, lim)
        ax_cv_s.set_xlabel('IMU curvature (rad/m)', color='white')
        ax_cv_s.set_ylabel('Model curvature (rad/m)', color='white')
        ax_cv_s.set_title('Curvature: IMU vs Model', color='white', fontsize=11)
        ax_cv_s.tick_params(colors='white')
        ax_cv_s.grid(True, alpha=0.15)
        ax_cv_s.set_aspect('equal')

        # Bias histogram
        ax_cv_h.set_facecolor('#2a2a2a')
        ax_cv_h.hist(bias, bins=80, color=plt.cm.plasma(0.5), alpha=0.8, edgecolor='none')
        ax_cv_h.axvline(0, color='#555', lw=1, ls=':', label='0')
        ax_cv_h.axvline(np.mean(bias), color='cyan', lw=2,
                        label=f'μ={np.mean(bias):.5f}')
        ax_cv_h.set_xlabel('Bias (model - IMU) rad/m', color='white')
        ax_cv_h.set_ylabel('Count', color='white')
        ax_cv_h.set_title(f'Curvature bias  (σ={np.std(bias):.5f})',
                          color='white', fontsize=11)
        ax_cv_h.legend(loc='upper right', facecolor='#333', edgecolor='#555',
                       labelcolor='white', fontsize=8)
        ax_cv_h.tick_params(colors='white')
        ax_cv_h.grid(True, alpha=0.15)
        stats_cv = f'n={cv["samples"]:,}'
        ax_cv_h.text(0.97, 0.05, stats_cv, transform=ax_cv_h.transAxes,
                     fontsize=9, va='bottom', ha='right', color='white',
                     bbox=dict(boxstyle='round', facecolor='#333', alpha=0.85))
    else:
        for ax in [ax_cv_s, ax_cv_h]:
            ax.set_facecolor('#2a2a2a')
            ax.text(0.5, 0.5, 'Insufficient data', color='#888', fontsize=14,
                    ha='center', va='center', transform=ax.transAxes)
            ax.tick_params(colors='white')

    # ──── Row 2, Col 3: Summary ────
    ax_sum.set_facecolor('#2a2a2a')
    fl_results = []
    if rv:
        fl_results.append(("Radar-Vision", rv["fl_suggested"],
                           (rv["fl_suggested"] / FCAM_FL - 1) * 100, rv["samples"]))
    if lw:
        fl_results.append(("Lane Width", lw["fl_suggested"],
                           (lw["fl_suggested"] / FCAM_FL - 1) * 100, lw["samples"]))
    ax_sum.axis('off')
    lines = ["Three-Method Summary", "", "Fl suggestion:"]
    for name, fl_sug, delta, n in fl_results:
        lines.append(f"  {name}: fl {FCAM_FL}→{fl_sug} ({delta:+.1f}%) n={n:,}")
    if lw and rv:
        mean_fl = int(np.mean([x[1] for x in fl_results]))
        lines.append("")
        lines.append(f"  Mean: {mean_fl}")
    ax_sum.text(0.05, 0.95, "\n".join(lines), transform=ax_sum.transAxes,
                fontsize=11, color='white', va='top', ha='left',
                fontfamily='monospace')

    # Title
    fig.suptitle(f'Three-Method Calibration  {seg_info}',
                 color='white', fontsize=14, y=0.98)
    fig.savefig(output, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output}")

    # Console summary
    print(f"\n{'='*50}")
    print("  Radar-Vision:")
    if rv:
        delta = (rv["fl_suggested"] / FCAM_FL - 1) * 100
        print(f"    median ratio={rv['median']:.4f}  n={rv['samples']:,}")
        print(f"    fl: {FCAM_FL} → {rv['fl_suggested']} ({delta:+.1f}%)")
        for lbl, d in rv["strata"].items():
            print(f"      {lbl}: ratio={d['ratio']:.4f} n={d['n']}")
    else:
        print("    (insufficient data)")
    print("  Lane Width:")
    if lw:
        print(f"    combined ratio={lw['combined_ratio']:.4f}  n={lw['samples']:,}")
        for rt, d in lw["results"].items():
            print(f"      {rt}: {d['mean_corr']:.2f}m vs {d['expected']}m (raw={d['mean_raw']:.2f}m) n={d['n']}")
        print(f"    fl: {FCAM_FL} → {lw['fl_suggested']} ({(lw['fl_suggested']/FCAM_FL-1)*100:+.1f}%)")
    else:
        print("    (insufficient data)")
    print("  Curvature:")
    if cv:
        print(f"    bias μ={cv['bias_mean']:.5f} σ={cv['bias_std']:.5f} n={cv['samples']:,}")
    else:
        print("    (insufficient data)")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", help="Directory with segment subdirs (rlog.zst)")
    parser.add_argument("--route", help="Single rlog.zst path")
    parser.add_argument("--output", default="calib_three_methods.png")
    parser.add_argument("--max", type=int, help="Max segments to process")
    args = parser.parse_args()

    paths = []
    if args.route:
        paths = [args.route]
    elif args.dir:
        paths = find_rlogs(args.dir)
    if not paths:
        parser.print_help()
        return
    if args.max:
        paths = paths[:args.max]
    print(f"Found {len(paths)} rlog files, processing...")

    rv_pairs, lw_obs, cv_pairs, seg_counts = collect_all(paths)
    total_samples = len(rv_pairs) + len(lw_obs) + len(cv_pairs)
    if total_samples < 10:
        print(f"Too few samples: {total_samples}")
        return

    rv = compute_rv(rv_pairs)
    lw = compute_lw(lw_obs)
    cv = compute_cv(cv_pairs)

    seg_info = f"segs={len(seg_counts)}" if seg_counts else ""
    _plot(rv, lw, cv, seg_info, args.output)


if __name__ == "__main__":
    main()
