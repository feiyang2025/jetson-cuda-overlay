#!/usr/bin/env python3
"""road/wide 双目同步诊断 (T906G / twgmsl / MAX9295A + MAX9296A)

回答一个问题: **这两路相机到底有没有硬件曝光同步?**

结论不能只看单次测量的 MAD。两路挂同一颗串行器/解串器、共享同源时钟时,
帧率会完全一致、单次运行内相位极稳, 但**绝对相位会随每次初始化随机跳变** ——
那不是同步, 是 free-run。所以本工具的核心是 `verdict`: 对比多次重启后的偏置,
只有偏置可复现才判定为硬件同步。

用法
  # 1. 寄存器判读 (只读, 不需要停栈)
  python3 stereo_sync_probe.py regs

  # 2. 单次相位测量 (需相机空闲: 先停 manager/camerad/预览)
  python3 stereo_sync_probe.py skew --json /tmp/sync_run1.json

  # 3. 重启相机驱动后再测一次, 然后对比
  #    sudo pkill -x tw_camera_cfg; sudo tw_camera_cfg &   (或 bring)
  python3 stereo_sync_probe.py skew --json /tmp/sync_run2.json
  python3 stereo_sync_probe.py verdict /tmp/sync_run1.json /tmp/sync_run2.json

  # 一次跑完 regs + skew
  python3 stereo_sync_probe.py all --json /tmp/sync_run1.json

判据 (来自 T906G 实测 2026-11-02)
  单次内 MAD < 50us      两路节拍一致 (共用时钟的必然结果, 不是同步证据)
  多次重启偏置可复现     偏置稳定在 <100us 内 -> 才有硬件同步嫌疑
  多次重启偏置跳变       数毫秒级跳变 -> free-run, 软件配对是唯一可行路径

只读工具: 不写任何 serdes 寄存器, 不启停服务。
"""
from __future__ import annotations

import argparse
import ctypes
import fcntl
import json
import mmap
import os
import shutil
import statistics as st
import subprocess
import sys
import threading
import time

# ---------------------------------------------------------------- V4L2 常量
# 注意: 本驱动 (twgmsl vi-output) 只接受 88 字节 (0xC058...) 的 buf 结构,
# 旧的 68 字节 (0xC044...) 恒返回 ENOTTY —— 实测踩过。
VIDIOC_REQBUFS = 0xC0145608
VIDIOC_QUERYBUF = 0xC0585609
VIDIOC_QBUF = 0xC058560F
VIDIOC_DQBUF = 0xC0585611
VIDIOC_STREAMON = 0x40045612
VIDIOC_STREAMOFF = 0x40045613
V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_MEMORY_MMAP = 1
NBUF = 8

# tegra 的 v4l2_buffer: 88 字节有效, ctypes 结构按 8 对齐后 sizeof=96,
# ioctl 的 size 字段取 88 (见上), 所以这里结构体定义必须与内核字段顺序一致。
I2C_BUS_DEFAULT = "9"     # 组 0 (video0/video1 同颗 MAX9296A)
DES_ADDR_DEFAULT = "0x4a"
SER_ADDRS_DEFAULT = ("0x41", "0x42")

DES_REGS = (
    (0x000D, "DEV_ID (0x94=MAX9296A)"),
    (0x0013, "CTRL3 (bit3=LOCKED)"),
    (0x0308, "PLL (0x61=locked)"),
    (0x0010, "MODE (0x03=splitter)"),
    (0x02B0, "GPIO_A (MFP passthrough)"),
    (0x02B1, "GPIO_B"),
    (0x02C2, "MFP/GPIO_C (0x83=input+passthrough)"),
    (0x02C3, "GPIO_D"),
)

SER_REGS = (
    (0x000D, "DEV_ID"),
    (0x0010, "MODE"),
    (0x02BE, "GPIO"),
    (0x02BF, "GPIO"),
    (0x02C1, "GPIO"),
    (0x02C3, "GPIO"),
    (0x02D3, "FSYNC enable"),
    (0x02D5, "FSYNC period"),
    (0x02D6, "FSYNC delay"),
    (0x02D8, "FSYNC vertical offset"),
)


# ---------------------------------------------------------------- V4L2 结构
class v4l2_requestbuffers(ctypes.Structure):
    _fields_ = [("count", ctypes.c_uint32), ("type", ctypes.c_uint32),
                ("memory", ctypes.c_uint32), ("capabilities", ctypes.c_uint32),
                ("reserved", ctypes.c_uint32)]


class v4l2_timeval(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_usec", ctypes.c_long)]


class v4l2_timecode(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint32), ("flags", ctypes.c_uint32),
                ("frames", ctypes.c_uint8), ("seconds", ctypes.c_uint8),
                ("minutes", ctypes.c_uint8), ("hours", ctypes.c_uint8),
                ("userbits", ctypes.c_uint8 * 4)]


class v4l2_buffer(ctypes.Structure):
    _fields_ = [("index", ctypes.c_uint32), ("type", ctypes.c_uint32),
                ("bytesused", ctypes.c_uint32), ("flags", ctypes.c_uint32),
                ("field", ctypes.c_uint32), ("timestamp", v4l2_timeval),
                ("timecode", v4l2_timecode), ("sequence", ctypes.c_uint32),
                ("memory", ctypes.c_uint32), ("m", ctypes.c_uint64),
                ("length", ctypes.c_uint32), ("input", ctypes.c_uint32),
                ("reserved", ctypes.c_uint32 * 4)]


# ---------------------------------------------------------------- I2C 只读
def _i2c_read(bus: str, addr: str, reg: int) -> str:
    """读 16 位寄存器, 返回 '0xNN' 或 'NACK'。"""
    if not shutil.which("i2ctransfer"):
        return "NO_I2CTRANSFER"
    hi, lo = (reg >> 8) & 0xFF, reg & 0xFF
    try:
        r = subprocess.run(
            ["i2ctransfer", "-f", "-y", bus, f"w2@{addr}", f"0x{hi:02x}", f"0x{lo:02x}", "r1"],
            capture_output=True, text=True, timeout=3)
    except (subprocess.TimeoutExpired, OSError):
        return "NACK"
    out = r.stdout.strip()
    return out if out else "NACK"


def cmd_regs(args) -> int:
    print(f"[regs] bus={args.bus}  des={args.des}  ser={','.join(args.ser)}")
    des_vals = {}
    print(f"\n  MAX9296A @ {args.des}")
    for reg, desc in DES_REGS:
        v = _i2c_read(args.bus, args.des, reg)
        des_vals[reg] = v
        print(f"    0x{reg:04x} = {v:>8}   {desc}")

    for sa in args.ser:
        print(f"\n  MAX9295A @ {sa}")
        for reg, desc in SER_REGS:
            v = _i2c_read(args.bus, sa, reg)
            print(f"    0x{reg:04x} = {v:>8}   {desc}")

    # 判读
    print("\n  ---- 判读 ----")
    try:
        lock = int(des_vals.get(0x0013, "0x0"), 16)
        print(f"    des lock : {'LOCKED' if (lock >> 3) & 1 else 'NOT LOCKED'}  (0x0013 & 0x08)")
    except ValueError:
        print("    des lock : 无应答, 链路可能未上电/未初始化")
    try:
        pll = int(des_vals.get(0x0308, "0x0"), 16)
        print(f"    des PLL  : {'locked' if pll == 0x61 else f'0x{pll:02x} (非 0x61)'}")
    except ValueError:
        pass
    try:
        na, nb = int(des_vals.get(0x02C2, "0x0"), 16), int(des_vals.get(0x02B0, "0x0"), 16)
        if na == 0x83 or nb == 0x83:
            print("    MFP 透传 : 已配置 (0x83 = 输入 + passthrough)")
        else:
            print("    MFP 透传 : 未配置")
    except ValueError:
        pass
    print("    提示     : 寄存器配置到位 ≠ 曝光同步; 必须跑 skew + 多次重启 verdict")
    return 0


# ---------------------------------------------------------------- 相位测量
class _Cam(threading.Thread):
    """单路独立 DQBUF 线程。

    两路必须并行读: 顺序 DQBUF (先 road 后 wide) 会把读取延迟差混进测量,
    实测能造成几百 us 到毫秒级的假偏置。
    """

    def __init__(self, dev: str, name: str, frames: int, duration: float):
        super().__init__(daemon=True)
        self.dev = dev
        self.name = name
        self.frames = frames
        self.duration = duration
        self.ts: list[int] = []
        self.err: str | None = None
        self._halt = False

    def run(self):
        try:
            fd = os.open(self.dev, os.O_RDWR | os.O_NONBLOCK)
        except OSError as e:
            self.err = f"打开失败: {e}"
            return
        maps = []
        try:
            rb = v4l2_requestbuffers()
            rb.count, rb.type, rb.memory = NBUF, V4L2_BUF_TYPE_VIDEO_CAPTURE, V4L2_MEMORY_MMAP
            fcntl.ioctl(fd, VIDIOC_REQBUFS, rb)
            if rb.count == 0:
                self.err = "REQBUFS 返回 0 个 buffer (驱动未就绪 / 设备被占用?)"
                return
            for i in range(rb.count):
                qb = v4l2_buffer()
                qb.index, qb.type, qb.memory = i, V4L2_BUF_TYPE_VIDEO_CAPTURE, V4L2_MEMORY_MMAP
                fcntl.ioctl(fd, VIDIOC_QUERYBUF, qb)
                maps.append(mmap.mmap(fd, qb.length, mmap.MAP_SHARED,
                                      mmap.PROT_READ, offset=qb.m))
                b = v4l2_buffer()
                b.index, b.type, b.memory = i, V4L2_BUF_TYPE_VIDEO_CAPTURE, V4L2_MEMORY_MMAP
                fcntl.ioctl(fd, VIDIOC_QBUF, b)
            fcntl.ioctl(fd, VIDIOC_STREAMON, ctypes.c_uint32(V4L2_BUF_TYPE_VIDEO_CAPTURE))

            deadline = time.monotonic() + self.duration
            while not self._halt and len(self.ts) < self.frames and time.monotonic() < deadline:
                b = v4l2_buffer()
                b.type, b.memory = V4L2_BUF_TYPE_VIDEO_CAPTURE, V4L2_MEMORY_MMAP
                try:
                    fcntl.ioctl(fd, VIDIOC_DQBUF, b)
                except BlockingIOError:
                    time.sleep(0.001)
                    continue
                self.ts.append(b.timestamp.tv_sec * 1_000_000_000
                               + b.timestamp.tv_usec * 1_000)
                fcntl.ioctl(fd, VIDIOC_QBUF, b)
            try:
                fcntl.ioctl(fd, VIDIOC_STREAMOFF, ctypes.c_uint32(V4L2_BUF_TYPE_VIDEO_CAPTURE))
            except OSError:
                pass
        except Exception as e:  # noqa: BLE001
            self.err = repr(e)
        finally:
            for m in maps:
                try:
                    m.close()
                except Exception:
                    pass
            os.close(fd)


def _periods(ts: list[int], lo_ns: int = 20_000_000, hi_ns: int = 60_000_000) -> list[int]:
    d = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
    return [x for x in d if lo_ns < x < hi_ns]


def measure(road: str, wide: str, frames: int, duration: float) -> dict:
    a, b = _Cam(road, "road", frames, duration), _Cam(wide, "wide", frames, duration)
    a.start()
    b.start()
    # 有界 join: DQBUF 在设备没出帧/被占用时可能一直阻塞, 不能让整个工具挂死。
    # 超时后置 halt 并放弃该线程 (daemon=True, 进程退出即回收)。
    limit = duration + 10.0
    for t in (a, b):
        t.join(timeout=limit)
    for t in (a, b):
        if t.is_alive():
            t._halt = True
            if t.err is None:
                t.err = f"采集超时 (> {limit:.0f}s 未返回): 该路没出帧或设备被占用"

    if a.err or b.err:
        return {"ok": False, "road_err": a.err, "wide_err": b.err,
                "road_frames": len(a.ts), "wide_frames": len(b.ts)}

    n = min(len(a.ts), len(b.ts))
    if n < 20:
        return {"ok": False, "road_err": None, "wide_err": None,
                "road_frames": len(a.ts), "wide_frames": len(b.ts),
                "reason": "可用帧不足 (<20), 至少一路没稳定出帧"}

    ta, tb = a.ts[:n], b.ts[:n]
    da, db = _periods(ta), _periods(tb)
    diff = [tb[i] - ta[i] for i in range(n)]
    # 丢弃超过半帧的离群 (启动期/丢帧造成的错配), 只保留同相位的一组
    diff_f = [d for d in diff if abs(d) < 50_000_000]
    if not diff_f:
        return {"ok": False, "reason": "所有配对都超过半帧, 两路很可能不在同一节拍"}
    med = st.median(diff_f)
    mad = st.median([abs(d - med) for d in diff_f])

    return {
        "ok": True,
        "frames": n,
        "road_dev": road,
        "wide_dev": wide,
        "road_period_us": st.median(da) / 1e3 if da else None,
        "wide_period_us": st.median(db) / 1e3 if db else None,
        "road_sigma_us": st.stdev(da) / 1e3 if len(da) > 1 else None,
        "wide_sigma_us": st.stdev(db) / 1e3 if len(db) > 1 else None,
        "offset_us": med / 1e3,
        "mad_us": mad / 1e3,
        "offset_min_us": min(diff_f) / 1e3,
        "offset_max_us": max(diff_f) / 1e3,
        "pairs_valid": len(diff_f),
        "ts_wall": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _print_measure(r: dict):
    if not r.get("ok"):
        rf, wf = r.get("road_frames", 0), r.get("wide_frames", 0)
        print(f"  测量失败: road 拿到 {rf} 帧, wide 拿到 {wf} 帧")
        for tag, key in (("road", "road_err"), ("wide", "wide_err")):
            if r.get(key):
                print(f"    {tag}: {r[key]}")
        if r.get("reason"):
            print(f"    {r['reason']}")
        return
    print(f"  帧数        : {r['frames']} (有效配对 {r['pairs_valid']})")
    print(f"  road 周期   : {r['road_period_us']:.1f} us  sigma {r['road_sigma_us']:.2f} us"
          f"  -> {1e6 / r['road_period_us']:.3f} fps")
    print(f"  wide 周期   : {r['wide_period_us']:.1f} us  sigma {r['wide_sigma_us']:.2f} us"
          f"  -> {1e6 / r['wide_period_us']:.3f} fps")
    print(f"  wide-road   : {r['offset_us']:+.1f} us  (MAD {r['mad_us']:.1f} us)")
    print(f"  范围        : {r['offset_min_us']:+.1f} .. {r['offset_max_us']:+.1f} us")
    if r["mad_us"] < 50:
        print("  → 单次内相位极稳。**这不是同步证据** —— 共用时钟的 free-run 也长这样。")


def cmd_skew(args) -> int:
    r = measure(args.road, args.wide, args.frames, args.duration)
    _print_measure(r)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(r, f, indent=2)
        print(f"  已保存: {args.json}")
    if not r.get("ok"):
        print("  排障: 确认相机空闲 (停 manager/camerad, 关掉 gst 预览), 再重试。")
        return 1
    print("\n  下一步: 重启相机驱动 (tw_camera_cfg) 再测一次, 然后")
    print("          python3 stereo_sync_probe.py verdict run1.json run2.json")
    return 0


def cmd_all(args) -> int:
    rc = cmd_regs(args)
    print()
    return max(rc, cmd_skew(args))


def cmd_verdict(args) -> int:
    runs = []
    for p in args.runs:
        try:
            with open(p) as f:
                runs.append((p, json.load(f)))
        except (OSError, json.JSONDecodeError) as e:
            print(f"  跳过 {p}: {e}")
    runs = [(p, r) for p, r in runs if r.get("ok")]
    if len(runs) < 2:
        print("  至少需要 2 次有效测量 (每次在不同初始化之后) 才能判同步。")
        return 1
    if len(runs) < 3:
        print("  ⚠ 只有 2 次测量, 样本不足以区分'恰好接近'与'真复现'。")
        print("    建议再做第 3 次 (重启后测) 再下结论。")
        print("    当前仅报告区间, 不当作最终同步判决。\n")

    print("  各次测量 (应按顺序在不同 tw_camera_cfg 初始化之后采集):")
    offs = []
    for p, r in runs:
        offs.append(r["offset_us"])
        print(f"    {os.path.basename(p):24} offset {r['offset_us']:+10.1f} us"
              f"   MAD {r['mad_us']:6.1f} us   {r['ts_wall']}")

    spread = max(offs) - min(offs)
    print(f"\n  偏置跨重启极差 : {spread:.1f} us")
    print(f"  单次内 MAD 上限: {max(r['mad_us'] for _, r in runs):.1f} us")

    if len(runs) < 3:
        print(f"\n  → 结论: 证据不足 (仅 {len(runs)} 次), 不能判定。")
        print("     区间跨度已看到, 但样本太少, 未下同步/free-run 定论。")
        return 4

    if spread < 100:
        print("\n  → 结论: 偏置跨重启可复现 (<100us), **疑似真硬件同步**。")
        print("     仍需光学 spinner 或示波器确认 (V4L2 timestamp 不是真实 SOF)。")
        return 0
    if spread < 1000:
        print("\n  → 结论: 偏置部分复现但有百 us 级漂移, 同步不完整 (可能只有帧率同源)。")
        print("     建议: 确认 FSYNC 脉冲存在且被两路共同使用。")
        return 2

    print(f"\n  → 结论: 偏置跨重启跳变 {spread / 1000:.2f} ms, **两路 free-run, 没有共同曝光触发**。")
    print("     软件配对 (FrameSync + camera_sync) 是当前唯一可行路径;")
    print("     不要为此收紧 modeld 容差, 也不要默认开 trigmode=trigger。")
    return 3


def cmd_auto(args) -> int:
    """按 sysfs 名称自动选组 0 的两路 (同颗 MAX9296A)。"""
    found = {}
    for i in range(16):
        p = f"/sys/class/video4linux/video{i}/name"
        try:
            with open(p) as f:
                found[f"/dev/video{i}"] = f.read().strip()
        except OSError:
            continue
    for d, n in found.items():
        print(f"  {d:14} {n}")
    same_group = [d for d, n in found.items() if "twgmsl 9-" in n]
    print(f"\n  组 0 (同颗 MAX9296A) 节点: {sorted(same_group)}")
    print("  说明: road/wide 必须落在同一组, MFP 透传才可能成立 (跨组不行)。")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="road/wide 双目同步诊断 (只读)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--bus", default=I2C_BUS_DEFAULT)
        p.add_argument("--des", default=DES_ADDR_DEFAULT)
        p.add_argument("--ser", nargs="+", default=list(SER_ADDRS_DEFAULT))
        p.add_argument("--road", default="/dev/video1")
        p.add_argument("--wide", default="/dev/video0")
        p.add_argument("--frames", type=int, default=150)
        p.add_argument("--duration", type=float, default=6.0)
        p.add_argument("--json", default=None)

    p_regs = sub.add_parser("regs", help="只读寄存器并判读链路/FSYNC 配置")
    common(p_regs)
    p_regs.set_defaults(func=cmd_regs)

    p_skew = sub.add_parser("skew", help="并行测两路时间戳相位 (单次)")
    common(p_skew)
    p_skew.set_defaults(func=cmd_skew)

    p_all = sub.add_parser("all", help="regs + skew")
    common(p_all)
    p_all.set_defaults(func=cmd_all)

    p_ver = sub.add_parser("verdict", help="对比多次重启后的偏置, 判定是否真同步")
    p_ver.add_argument("runs", nargs="+", help="skew --json 产出的文件")
    p_ver.set_defaults(func=cmd_verdict)

    p_auto = sub.add_parser("auto", help="列出 video 节点并按组归类")
    p_auto.set_defaults(func=cmd_auto)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
