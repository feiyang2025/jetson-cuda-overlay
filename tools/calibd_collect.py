#!/usr/bin/env python3
"""标定体检包 —— 一条命令把"这次标定到底发生了什么"打包成一个 txt, 直接发出去就能判读。

用法:
  .venv/bin/python3 tools/calibd_collect.py                 # 默认写到 /tmp/calib_check_<时间>.txt
  .venv/bin/python3 tools/calibd_collect.py --deep          # 额外读最近两段日志统计可用场景(慢)

包含: 环境与进程状态、相机驱动/端口占用、标定相关参数全量、calibd 报告、最近日志段概况、
      /tmp 下的启动日志尾部 (若有 [calibd]/[V4L2Camera]/lane_lines 等关键行会一并抓出来)。

配套文档: /home/nvidia/文档/相机内参自动标定_calibd_交接与排障_20261012.md
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

TREE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PY = os.path.join(TREE, ".venv", "bin", "python3")
KEY_PARAMS = ["AutoCalib", "AutoCalibHistory", "AutoCalibBackup", "AutoCalibRollback",
              "FcamIntrinsics", "EcamIntrinsics", "FcamCalibResult", "WideCalibResult",
              "CalibrationParams", "Model", "UseWideCamera", "WideCalibActive", "FcamLiveActive"]
ENV_KEYS = ["CAM_WIDTH", "CAM_HEIGHT", "ROAD_CAM_WIDTH", "ROAD_CAM_HEIGHT", "WIDE_CAM_WIDTH",
            "WIDE_CAM_HEIGHT", "CALIB_MIN_KM", "GMSL_CHROMA_LAYOUT", "SP_CHROMA_SWAP",
            "CAM_PACKED_LAYOUT", "CAM_YUV_LAYOUT", "USE_WEBCAM", "ROAD_CAM", "WIDE_CAM",
            "ENABLE_FAKE_PANDA", "SP_TIMING", "SP_NVBUF_ZEROCOPY"]
PROC_PATTERNS = ["openpilot.system.manager", "camerad", "modeld", "calibd", "tw_camera_cfg",
                 "carrot_server", "fake_panda", "launch_chffrplus"]
LOG_GREP = ["[calibd]", "[V4L2Camera]", "[CudaUyvyConverter]", "[TRT_CTYPES]", "lane_lines",
            "Missing output", "EADDRINUSE", "[mutex]", "[camera]", "fingerprinted"]


def sh(cmd: str, timeout: int = 20) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or "") + (r.stderr or "")
    except Exception as e:
        return f"(命令失败: {e})"


def out(lines: list, title: str) -> None:
    lines.append("")
    lines.append("=" * 70)
    lines.append(f"## {title}")
    lines.append("=" * 70)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"/tmp/calib_check_{time.strftime('%m%d_%H%M')}.txt")
    ap.add_argument("--deep", action="store_true", help="额外读最近两段日志统计可用场景 (较慢)")
    args = ap.parse_args()

    L = [f"标定体检包  生成于 {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
         f"树根: {TREE}",
         "配套文档: /home/nvidia/文档/相机内参自动标定_calibd_交接与排障_20261012.md"]

    out(L, "1. 环境与版本")
    L.append(sh(f"cd {TREE} && git rev-parse --short HEAD && git rev-parse --abbrev-ref HEAD"))
    L.append("-- 关键文件是否被本地改过 (head = 干净):")
    L.append(sh(f"cd {TREE} && git status --short openpilot/selfdrive/modeld/parse_model_outputs.py "
                f"openpilot/sunnypilot/calib openpilot/common/transformations/camera.py "
                f"tools/calib/self_calibrator.py launch_chffrplus.sh openpilot/common/params_keys.h"))
    L.append("-- overlay:")
    L.append(sh(f"git -C /home/nvidia/公共/jetson-cuda-overlay log --oneline -3 2>/dev/null || echo '(无 overlay)'"))
    L.append("-- python / 库:")
    L.append(f"{PY}  exists={os.path.exists(PY)}")
    L.append(sh(f"{PY} -c \"import numpy,sys;print('numpy',numpy.__version__, sys.version.split()[0])\" 2>&1"))
    L.append("-- 相关 env (本 shell; 多数只在启动脚本里 export, 所以下面再从运行进程读一遍):")
    for k in ENV_KEYS:
        L.append(f"  {k}={os.environ.get(k)!r}")
    L.append("-- 运行中进程的真实 env (从 /proc/<pid>/environ 提取, 这才是栈里生效的值):")
    for pat in ("camerad", "modeld", "calibd", "openpilot.system.manager"):
        pid = sh(f"pgrep -f '{pat}' | head -1").strip()
        if not pid.isdigit():
            L.append(f"  [{pat}] 未运行")
            continue
        L.append(f"  [{pat}] pid={pid}")
        try:
            raw = open(f"/proc/{pid}/environ", "rb").read().decode(errors="replace").split("\0")
            vals = dict(kv.split("=", 1) for kv in raw if "=" in kv)
            for k in ENV_KEYS:
                if k in vals:
                    L.append(f"    {k}={vals[k]}")
        except Exception as e:
            L.append(f"    (读不到 /proc/{pid}/environ: {e})")

    out(L, "2. 进程与端口 (两棵树是否同时在跑 / 谁占着相机与端口)")
    for p in PROC_PATTERNS:
        L.append(f"-- {p}:")
        L.append(sh(f"ps -eo pid,etime,cmd | grep -F '{p}' | grep -v grep | grep -v calibd_collect | head -5"))
    L.append("-- 端口 7000 / 7714:")
    L.append(sh("ss -ltnp 2>/dev/null | grep -E ':(7000|7714)' || echo '(未监听)'"))
    L.append("-- 相机节点:")
    L.append(sh("ls -l /dev/video0 /dev/video1 2>&1"))

    out(L, "2b. 相机模型当前生效的内参 (与 camera.py 同源, 标定基准)")
    camw = ""
    pid = sh("pgrep -f camerad | head -1").strip()
    if pid.isdigit():
        try:
            raw = open(f"/proc/{pid}/environ", "rb").read().decode(errors="replace").split("\0")
            vals = dict(kv.split("=", 1) for kv in raw if "=" in kv)
            camw = " ".join(f"{k}={vals[k]}" for k in ("CAM_WIDTH", "CAM_HEIGHT", "ROAD_CAM_WIDTH",
                                                       "WIDE_CAM_WIDTH") if k in vals)
        except Exception:
            pass
    L.append(f"  (取自 camerad 进程的 env: {camw or '无'})")
    L.append(sh(f"cd {TREE} && PYTHONPATH=\"$PWD\" {camw} {PY} tools/calib/camera_std.py 2>&1"))
    L.append("  ↑ 这里显示的 fl 就是标定的基准; 若与 FcamIntrinsics 里的 1344x760 条目不一致, 就是空间错位")

    out(L, "3. 标定相关参数 (全量)")
    L.append(sh(f"cd {TREE} && PYTHONPATH=\"$PWD\" {PY} tools/calibd_params_dump.py 2>&1"))

    out(L, "4. calibd 报告 (自动模式 / 里程窗口 / 每趟结论与还差什么)")
    L.append(sh(f"cd {TREE} && PYTHONPATH=\"$PWD\" {PY} tools/calibd_report.py -v 2>&1", timeout=60))

    out(L, "5. 启动日志里的关键行 (/tmp/*.log; 没有就下次用 | tee 存一份)")
    logs = sorted(glob.glob("/tmp/op*.log") + glob.glob("/tmp/*.log"), key=os.path.getmtime)[-4:]
    L.append(f"候选日志: {logs or '(无)'}")
    for f in logs:
        L.append(f"-- {f}  (mtime={time.strftime('%m-%d %H:%M', time.localtime(os.path.getmtime(f)))})")
        pat = "|".join(g.replace("[", r"\[") for g in LOG_GREP)
        L.append(sh(f"grep -aE '{pat}' {f} 2>/dev/null | tail -60"))

    out(L, "6. 最近日志段 (有没有真跑出数据)")
    realdata = os.path.expanduser("~/.comma/media/0/realdata")
    segs = sorted([d for d in glob.glob(os.path.join(realdata, "*")) if os.path.isdir(d)],
                  key=os.path.getmtime)[-6:]
    for d in segs:
        sz = sh(f"du -sh {d} 2>/dev/null | cut -f1").strip()
        L.append(f"  {os.path.basename(d):32s} {sz:>8s}  "
                 f"{time.strftime('%m-%d %H:%M', time.localtime(os.path.getmtime(d)))}")
    if args.deep and segs:
        L.append("-- 场景统计 (只有前车/车道线/运动都有帧, 标定才可能有样本):")
        L.append(sh(f"cd {TREE} && PYTHONPATH=\"$PWD\" {PY} tools/calibd_segments_scan.py "
                    f"{' '.join(segs[-3:])} 2>&1", timeout=600))

    out(L, "7. 提示")
    L.extend([
        "  * 只有第 4/6 节显示有可用场景 (前车/双车道线/运动) 时, 标定才可能有样本;",
        "  * 台架停车段没有样本是正常的 (曲率票还要弯道);",
        "  * 若第 5 节为空: 下次启动用 `./launch_openpilot.sh 2>&1 | tee /tmp/op_run.log` 再跑;",
        "  * 本文件不含任何凭据 (只读参数与进程状态)。",
    ])

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"已生成: {args.out}  ({os.path.getsize(args.out)} 字节)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
