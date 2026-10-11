#!/usr/bin/env python3
"""calibd 结果报告 —— 把 AutoCalibHistory + 当前结果参数打成人能读的表。

用法:
  .venv/bin/python3 tools/calibd_report.py            # 看历史与结论
  .venv/bin/python3 tools/calibd_report.py --verbose  # 连每个方法的分项也打出来

看什么:
  * 每趟: 有哪些方法参与 (n_methods)、彼此离散 (spread_pct)、样本数、结论 (accepted);
  * 三方法 fl 是否互相接近 (离散 ≤3% 才是可写的信号);
  * 还差什么: 常见是曲率票要弯道样本、雷达-视觉要前车。
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "openpilot")))

from openpilot.common.params import Params  # noqa: E402


def _hist(p: Params) -> list:
    h = p.get("AutoCalibHistory")
    if isinstance(h, dict):
        return list(h.get("runs") or [])
    return h if isinstance(h, list) else []


def _result(p: Params, key: str) -> dict:
    d = p.get(key)
    return d if isinstance(d, dict) else {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()
    p = Params()

    raw = p.get("AutoCalib")
    auto_on = raw is None or raw == "" or p.get_bool("AutoCalib")
    print(f"自动写入内参: {'开' if auto_on else '关(显式关闭)'}"
          + ("    ← 累计里程够 + 各趟一致(≤1.5%) + 幅度≤15% 就自动写"
             if auto_on else "    ← 只给建议, 不动内参 (要恢复: put_bool('AutoCalib', True))"))
    # 里程进度: 触发是按里程算的
    h = p.get("AutoCalibHistory")
    auto = (h.get("auto") if isinstance(h, dict) else None) or {}
    km = float(auto.get("distance_km") or 0.0)
    need = float(os.environ.get("CALIB_MIN_KM") or 30.0)
    w = auto.get("written")
    print(f"里程窗口: 已累计 {km:.1f} / {need:.0f} km"
          + (f"  (上次自动写入: fl={w.get('fl')} 于 {w.get('ts')}, {w.get('km')} km/{w.get('runs')} 趟)"
             if isinstance(w, dict) else ""))
    bak = p.get("AutoCalibBackup")
    if isinstance(bak, dict):
        print(f"备份(可回滚): {bak.get('key')} fl={bak.get('fl')} 于 {bak.get('ts')}  ({bak.get('reason')})")

    for label, key in (("FCAM", "FcamCalibResult"), ("ECAM", "WideCalibResult")):
        r = _result(p, key)
        if not r:
            continue
        f = r.get("fused") or {}
        print(f"\n当前结果参数 {key}: accepted={f.get('accepted')} "
              f"fl={f.get('fl_current')}→{f.get('fl_suggested')} 置信={f.get('confidence')} "
              f"方法数={f.get('n_methods')} 离散={f.get('spread_pct')}% 样本={f.get('total_observations')}")
        if f.get("reason"):
            print(f"  未通过原因: {f['reason']}")
        if args.verbose:
            for name, m in (r.get("methods") or {}).items():
                if isinstance(m, dict):
                    print(f"  [{name}] valid={m.get('valid')} samples={m.get('samples')} "
                          f"fl_suggested={m.get('fl_suggested')} {m.get('error', '')}")

    hist = _hist(p)
    print(f"\n历史 (共 {len(hist)} 趟):")
    if not hist:
        print("  (还没有记录 —— calibd 只在 onroad 采样, 停车/台架不会有)")
    for h in hist:
        print(f"  {h.get('ts')}  {h.get('cam')}  "
              f"{'采纳' if h.get('accepted') else '未采纳'}  "
              f"fl {h.get('fl_current')}→{h.get('fl_suggested')} ({h.get('delta_pct')}%)  "
              f"置信={h.get('confidence')} 方法数={h.get('n_methods')} 离散={h.get('spread_pct')}% "
              f"样本={h.get('samples')}  动作={h.get('action')}"
              + (f"  — {h.get('reason')}" if h.get("reason") else ""))

    ok = [h for h in hist if h.get("accepted") and h.get("fl_suggested")]
    if ok:
        vals = [float(h["fl_suggested"]) for h in ok[-5:]]
        spread = (max(vals) - min(vals)) / (sum(vals) / len(vals)) * 100
        print(f"\n最近 {len(vals)} 趟采纳值: {[round(v, 1) for v in vals]}")
        print(f"  离散 {spread:.2f}% (>1.5% 就还不会被自动写), 中位数 {sorted(vals)[len(vals)//2]:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
