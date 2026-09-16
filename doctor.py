#!/usr/bin/env python3
"""一键体检（傻瓜式，无参数）：把整台 AGX Orin 的 openpilot 环境按 overlay 的功能面过一遍。

检查项（每项都给出"是什么/怎么修"）：
  1. 环境      平台/Ubuntu/CUDA/TensorRT/clang++/venv
  2. 分支巡查  发现的分支、布局(新/旧)、panda 子模块状态、.venv 有无
  3. 相机链路  /dev/video*、twgmsl 驱动、V4L2 适配文件、tw_camera_cfg
  4. CUDA 后端 分支里的 gpu_backend/gpu_model_state/cuda_transform/tensorrt_runner 是否就位
               libcuda_transform.so 是否存在、libcuda 解析是否 nvgpu、.plan 引擎清单
  5. CH347 IMU 设备节点、sensord_ch347 是否注册、ch347t 二进制、imu_calibration.json
  6. 相机标定   tools/calib 是否就位、FcamIntrinsics/EcamIntrinsics 参数、params 键是否编译进去
  7. Panda     固件签名/协议一致性(委托 verify_protocol.py)+ 是否有 panda 在 USB 上

用法: python3 doctor.py            # 全量
      python3 doctor.py --short    # 只打印有问题的项
"""
from __future__ import annotations

import ctypes
import glob
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOTS = ["/data/openpilot", "/opt/openpilot"]
SHORT = "--short" in sys.argv
problems: list[str] = []


def head(title: str) -> None:
    if not SHORT:
        print(f"\n=== {title} ===")


def line(ok: bool | None, text: str) -> None:
    tag = "OK " if ok else ("!! " if ok is False else "-- ")
    if SHORT and ok is not False:
        return
    print(f"  [{tag}] {text}")
    if ok is False:
        problems.append(text)


def run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


# ---------------------------------------------------------------- 1. 环境
head("1. 环境")
line(os.path.exists("/etc/nv_tegra_release"), "Jetson 平台(L4T)")
ub = run(["lsb_release", "-ds"])
line("24.04" in ub or ub.startswith("24"), f"Ubuntu {ub or '?'}")
nvcc = shutil.which("nvcc") or (glob.glob("/usr/local/cuda-*/bin/nvcc") or [""])[0]
line(bool(nvcc), f"nvcc: {nvcc or '缺失(编 CUDA 变换库需要)'}")
trt = glob.glob("/usr/lib/aarch64-linux-gnu/libnvinfer.so*")
line(bool(trt), f"TensorRT: {len(trt)} 个库文件")
clang = shutil.which("clang++")
line(bool(clang), f"clang++: {clang or '缺失'}")

# ---------------------------------------------------------------- 2. 分支
head("2. 分支巡查")
forks = []
for root in ROOTS:
    for e in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        f = os.path.join(root, e)
        if os.path.isdir(os.path.join(f, "selfdrive")) or os.path.isdir(os.path.join(f, "openpilot")):
            forks.append(f)
for f in forks:
    name = os.path.basename(f)
    new = os.path.isdir(os.path.join(f, "openpilot", "sunnypilot", "modeld_v2"))
    has_panda = os.path.isdir(os.path.join(f, "panda", "python"))
    venv = os.path.isfile(os.path.join(f, ".venv", "bin", "python3"))
    gpu = os.path.isdir(os.path.join(f, "openpilot", "sunnypilot", "modeld_v2", "gpu_backend")) or \
        os.path.isdir(os.path.join(f, "sunnypilot", "modeld_v2", "gpu_backend"))
    calib = os.path.isdir(os.path.join(f, "tools", "calib"))
    line(True, f"{name:<24} 布局={'新' if new else '旧'} panda={'有' if has_panda else '无'} "
               f"venv={'有' if venv else '无'} CUDA后端={'有' if gpu else '无'} 标定工具={'有' if calib else '无'}")

# ---------------------------------------------------------------- 3. 相机
head("3. 相机链路")
vids = sorted(glob.glob("/dev/video*"))
line(bool(vids), f"/dev/video*: {', '.join(vids) if vids else '没有设备节点'}")
lsmod = run(["lsmod"])
line("twgmsl" in lsmod, "twgmsl 驱动已加载" if "twgmsl" in lsmod else "twgmsl 驱动未加载(modprobe twgmsl)")
line(os.path.exists("/etc/tw_camera_cfg.ini"), "厂商相机配置 /etc/tw_camera_cfg.ini")
line(bool(shutil.which("tw_camera_cfg")), "tw_camera_cfg 可用(必须先于 camerad 启动)")
line(True, f"tw_camera_cfg 正在跑: {'是' if run(['pgrep','-x','tw_camera_cfg']) else '否'}")

# ---------------------------------------------------------------- 4. CUDA 后端
head("4. CUDA 后端与模型")
libcuda = run(["bash", "-lc", "ldconfig -p | grep -E 'libcuda\\.so' | head -3"])
libcuda_ok = False
try:
    libcuda_ok = ctypes.CDLL("libcuda.so.1") is not None
except Exception:
    libcuda_ok = False
line(libcuda_ok, f"libcuda.so 解析: {libcuda.replace(chr(10),' | ') or '未解析(需要 nvgpu 驱动的 libcuda)'}")
for f in forks:
    name = os.path.basename(f)
    so = glob.glob(os.path.join(f, "**", "libcuda_transform.so"), recursive=True)
    plans = glob.glob(os.path.join(f, "**", "*.plan"), recursive=True)
    trt_c = glob.glob(os.path.join(f, "**", "trt_c_api.so"), recursive=True)
    line(True, f"{name:<24} libcuda_transform.so={'有' if so else '无(需 nvcc 编)'} "
               f"trt_c_api.so={'有' if trt_c else '无'} .plan引擎={len(plans)} 个")
    for p in plans[:4]:
        line(True, f"      {os.path.relpath(p, f)} ({os.path.getsize(p)//1048576}MB)")

# ---------------------------------------------------------------- 5. CH347
head("5. CH347 IMU")
devs = glob.glob("/dev/ch34x_pis*") + glob.glob("/dev/ttyACM*")
line(None if not devs else True, f"设备节点: {', '.join(devs) if devs else '未插 CH347(可选功能, 不插就跳过)'}")
line(True, f"imu_calibration.json: {'有' if glob.glob('/data/openpilot/*/imu_calibration.json') else '无'}")
for f in forks:
    pc = os.path.join(f, "system", "manager", "process_config.py")
    if os.path.isfile(pc):
        line("sensord_ch347" in open(pc, errors="replace").read(),
             f"{os.path.basename(f)}: sensord_ch347 已注册")

# ---------------------------------------------------------------- 6. 标定
head("6. 相机标定")
params_d = os.path.expanduser("~/.commaspold/params/d")
for k in ("FcamIntrinsics", "EcamIntrinsics"):
    p = os.path.join(params_d, k)
    if os.path.isfile(p):
        line(True, f"{k} = {open(p, errors='replace').read()[:110]}")
    else:
        line(None, f"{k} 未生成(首次加载 camera.py 时自动创建)")
for f in forks:
    hh = os.path.join(f, "common", "params_keys.h")
    if os.path.isfile(hh):
        n = len(set(re.findall(r'"(FcamIntrinsics|EcamIntrinsics|FcamCalibResult|WideCalib\w*|FcamLiveActive|PendingCalibReset)"', open(hh, errors="replace").read())))
        line(n >= 8, f"{os.path.basename(f)}: 标定 params 键 {n}/10")
        line(os.path.isfile(os.path.join(f, "common", "params_pyx.so")), f"{os.path.basename(f)}: params_pyx.so 已编译")

# ---------------------------------------------------------------- 7. Panda
head("7. Panda")
usb = run(["bash", "-lc", "lsusb | grep -E '3801|bbaa' || true"])
line(True if usb else None, f"USB 上的 panda: {usb if usb else '未插(不插也可, 只是不能实车验证 CAN)'}")
vp = os.path.join("/data/openpilot/panda_版本核对", "verify_protocol.py")
if os.path.isfile(vp):
    out = run(["python3", vp])
    ok = "RESULT: 全部分支 panda 协议一致" in out
    line(ok, "panda 固件/协议一致性")
    if not ok and SHORT:
        print(out)
else:
    line(False, "panda_版本核对 未安装(跑 jetson-cuda-overlay/panda/install.sh)")

# ---------------------------------------------------------------- 总结
print()
if problems:
    print(f"体检结果: {len(problems)} 项需要处理")
    for p in problems:
        print("  - " + p)
    sys.exit(1)
print("体检结果: 全部通过 ✅")
