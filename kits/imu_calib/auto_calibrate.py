#!/usr/bin/env python3
"""
CH347T + LSM6DS3 一键自动校准（auto-calibrate）
=================================================
用途：为 sunnypilot-cuda 的 ch347t（system/sensord/ch347t.cc）生成
imu_calibration.json 校准文件。

流程：检测 CH347T hidraw → 初始化 LSM6DS3 → 引导摆 N 个静止姿态
（自动判静止，不静止不采）→ 静止段椭球拟合（复用 imu_calibration.py）
→ 写 JSON（默认项目根，即 ch347t.cc 的读取路径）→ 静止自检，
打印校准后 gyro≈0 rad/s、accel 模长≈9.81 m/s²。

单位约定（与 ch347t.cc 一致）：
  read_raw  : gyro °/s（245dps），accel g（±2g）
  JSON      : imuBiasGyro[3] °/s, imuBiasAccel[3] g, imuCalibMatrix[9], gravity
  ch347t 应用: gyro_rad = (raw_dps - imuBiasGyro) * pi/180
              accel_ms2 = imuCalibMatrix @ (raw_mps2 - 9.81*imuBiasAccel)

用法：
  python3 tools/imu_calib/auto_calibrate.py            # 6 姿态 × 6s，写项目根
  python3 tools/imu_calib/auto_calibrate.py --poses 8 --secs 5 --out my.json
  python3 tools/imu_calib/auto_calibrate.py --lib /path/libch347.so
"""
import argparse
import builtins
import ctypes
import glob
import json
import os
import re
import sys
import time

import numpy as np

from imu_calibration import accel_ellipsoid, gyro_bias, stationary_mask

# libch347.so 每个 I2C 事务都往 stdout printf 调试行（mLength: ..），
# 会把交互提示淹没 —— 把 fd 1 重定向到 /dev/null 压掉库输出，
# 脚本自身 print 全部改走 stderr（input() 无 prompt，不受影响）。
def _print(*a, **k):
    k.setdefault("file", sys.stderr)
    builtins.print(*a, **k)


_orig_print = builtins.print
print = _print  # noqa: A001

# ---------------------------------------------------------------------------
# 硬件访问（与 system/sensord/ch347t.py 相同的调用约定，不依赖 cereal）
# ---------------------------------------------------------------------------
VID, PID = 0x1A86, 0x55DC
LSM6_ADDRS = (0x6B, 0x6A)
WHO_IDS = (0x69, 0x6A)
ACC_SENS = 2.0 / 32768.0      # ±2g
GYRO_SENS = 245.0 / 32768.0   # 245dps


def _s16(lo, hi):
    v = lo | (hi << 8)
    return v - 0x10000 if v & 0x8000 else v


def find_lib_path(project_root):
    for arch in ("aarch64", "x64"):
        p = os.path.join(project_root, "third_party", "ch347", "lib", arch, "dynamic", "libch347.so")
        if os.path.exists(p):
            return p
    return "libch347.so"


def find_hidraw():
    # 官方 imu_lsm6ds3.py 同款判定：sysfs uevent 含 VID/PID，且 realpath 指向
    # usb 接口 :1.1/:1.2（SPI+I2C 接口）。曾用 "input1" 字符串判断在多接口枚举下
    # 选错过（选到 UART 接口导致写失败），realpath 判断经真机验证可靠。
    vid, pid = f"{VID:04X}", f"{PID:04X}"
    fallback = None
    for hidraw in sorted(glob.glob("/sys/class/hidraw/hidraw*")):
        try:
            with open(os.path.join(hidraw, "device", "uevent")) as f:
                uevent = f.read()
        except OSError:
            continue
        if vid in uevent and pid in uevent:
            dev = "/dev/" + os.path.basename(hidraw)
            try:
                real = os.path.realpath(os.path.join(hidraw, "device"))
                if ":1.1" in real or ":1.2" in real:
                    return dev
            except OSError:
                pass
            if fallback is None:
                fallback = dev
    return fallback


class CH347LSM6:
    """libch347.so 封装（当前库为索引式约定；argtypes 全部显式声明——
    否则 ctypes 无 argtypes 时指针/None 参数转换不可靠，I2C 写会失败）"""

    def __init__(self, dev_path, lib_path):
        self.lib = ctypes.CDLL(lib_path)
        self.lib.CH347OpenDevice.restype = ctypes.c_int
        self.lib.CH347OpenDevice.argtypes = [ctypes.c_ulong]      # 索引式（open() 里会切换）
        self.lib.CH347CloseDevice.argtypes = [ctypes.c_int]
        self.lib.CH347CloseDevice.restype = ctypes.c_bool
        self.lib.CH347I2C_Set.argtypes = [ctypes.c_int, ctypes.c_ulong]
        self.lib.CH347I2C_Set.restype = ctypes.c_bool
        self.lib.CH347StreamI2C.argtypes = [ctypes.c_int, ctypes.c_ulong,
                                            ctypes.c_void_p, ctypes.c_ulong, ctypes.c_void_p]
        self.lib.CH347StreamI2C.restype = ctypes.c_bool
        self.dev_path = dev_path
        self.fd = -1
        self.addr = None

    def _stream_read(self, wb, rlen, retries=3):
        wbuf = (ctypes.c_ubyte * len(wb))(*wb)
        rbuf = (ctypes.c_ubyte * rlen)()
        # USB-HID 桥偶发单次事务失败（实测出现过），短重试后再抛
        for attempt in range(retries):
            # 统一用两库都有的 CH347StreamI2C（RetAck 变体仅部分库导出，不强求）
            if self.lib.CH347StreamI2C(self.fd, len(wb), wbuf, rlen, rbuf):
                return bytes(rbuf)
            time.sleep(0.02)
        raise OSError("I2C read failed")

    def _stream_write(self, wb, retries=3):
        wbuf = (ctypes.c_ubyte * len(wb))(*wb)
        for attempt in range(retries):
            if self.lib.CH347StreamI2C(self.fd, len(wb), wbuf, 0, None):
                return
            time.sleep(0.02)
        raise OSError("I2C write failed")

    def read_block(self, reg, length):
        return self._stream_read([self.addr << 1, reg], length)

    def write_u8(self, reg, val):
        self._stream_write([self.addr << 1, reg, val])

    def open(self):
        dev = self.dev_path or find_hidraw()
        if not dev:
            raise RuntimeError(
                "CH347T hidraw 未找到（VID:PID 1a86:55dc）。\n"
                "  - 检查 USB 连接（需经 USB hub 到 CH347T 开发板）\n"
                "  - 检查拨码开关 Mode2（S1=OFF, S2=ON）\n"
                "  - 重新插拔后运行: lsusb | grep 1a86")
        self.dev_path = dev
        m = re.match(r"/dev/hidraw(\d+)", dev)
        hidraw_n = int(m.group(1)) if m else -1
        # 双约定自适应（ch347t.cc 同款）：
        #   新库 CH347OpenDevice(ULONG idx)，内部拼 /dev/hidraw<idx+2>
        #   旧库 CH347OpenDevice(const char *path)
        # ctypes 未声明 argtypes 时指针会按 c_int 截断，故每次调用前显式设 argtypes
        attempts = []
        if hidraw_n >= 0:
            attempts.append(("idx", ctypes.c_ulong(hidraw_n - 2),
                             [ctypes.c_ulong]))
        attempts.append(("path", dev.encode(), [ctypes.c_char_p]))
        last_err = "no attempt"
        for kind, arg, argtypes in attempts:
            self.lib.CH347OpenDevice.argtypes = argtypes
            self.fd = self.lib.CH347OpenDevice(arg)
            if self.fd < 0:
                last_err = f"{kind} open failed"
                continue
            try:
                # 超时/辅助符号两库不同（有则用，无则跳过）—— ch347t.cc 同款策略
                for sym in ("CH34xSetTimeout", "CH347SetTimeout"):
                    fn = getattr(self.lib, sym, None)
                    if fn is not None:
                        try:
                            fn(self.fd, 2000, 2000)
                            break
                        except Exception:
                            pass
                if not self.lib.CH347I2C_Set(self.fd, 0x01):  # 100kHz
                    raise OSError("CH347I2C_Set failed")
                for sym, arg in (("CH347I2C_SetIgnoreNack", 1),
                                 ("CH347I2C_SetStretch", True)):
                    fn = getattr(self.lib, sym, None)
                    if fn is not None:
                        try:
                            fn(self.fd, arg)
                        except Exception:
                            pass
                time.sleep(0.02)
            except OSError as e:
                self.lib.CH347CloseDevice(self.fd)
                self.fd = -1
                last_err = f"{kind}: {e}"
                continue
            for _ in range(8):
                for addr in LSM6_ADDRS:
                    try:
                        who = self._stream_read([addr << 1, 0x0F], 1)[0]
                    except OSError:
                        continue
                    if who in WHO_IDS:
                        self.addr = addr
                        print(f"[ok] LSM6DS3 @ 0x{addr:02X} (WHO_AM_I=0x{who:02X}) via {dev} [{kind}]")
                        return
                time.sleep(0.02)
            self.lib.CH347CloseDevice(self.fd)
            self.fd = -1
            last_err = f"{kind}: WHO_AM_I not detected"
        raise RuntimeError(f"CH347 open failed ({last_err})。检查接线和拨码开关")

    def init_sensor(self):
        self.write_u8(0x12, 0x01)   # soft reset
        time.sleep(0.1)
        self.write_u8(0x12, 0x44)   # CTRL3_C: BDU(bit6, 防止高低字节跨采样) + IF_INC
        self.write_u8(0x10, 0x40)   # CTRL1_XL: 208Hz ±2g
        self.write_u8(0x11, 0x40)   # CTRL2_G : 208Hz 245dps
        time.sleep(0.2)

    def read_raw(self):
        # 0x22 起 12 字节连读: gyro XYZ(6) + accel XYZ(6)，数据寄存器始终支持自动递增
        d = self.read_block(0x22, 12)
        if len(d) != 12:
            raise OSError(f"短读 {len(d)} != 12")
        g = [_s16(d[0], d[1]), _s16(d[2], d[3]), _s16(d[4], d[5])]
        a = [_s16(d[6], d[7]), _s16(d[8], d[9]), _s16(d[10], d[11])]
        # 轴序与 ch347t.cc 一致: 输出帧 x=传感器 Y, y=-X, z=Z
        gx, gy, gz = g
        g = [gy * GYRO_SENS, -gx * GYRO_SENS, gz * GYRO_SENS]
        a = [x * ACC_SENS for x in a]
        return g, a

    def close(self):
        if self.fd >= 0:
            try:
                self.write_u8(0x10, 0x00)
                self.write_u8(0x11, 0x00)
            except Exception:
                pass
            self.lib.CH347CloseDevice(self.fd)
            self.fd = -1


# ---------------------------------------------------------------------------
# 采集辅助
# ---------------------------------------------------------------------------
def collect_pose(imu, secs, label):
    """引导采集一个静止姿态：等待静止（gyro 滚动 std）后采 secs 秒。返回样本 np.ndarray。"""
    for try_no in range(1, 4):   # I2C 偶发失败时整姿态重试，不崩流程
        try:
            return _collect_pose_once(imu, secs, label, try_no)
        except OSError as e:
            print(f"    [重试 {try_no}/3] 读取失败({e})，请检查连接后重新摆好该姿态")
    raise RuntimeError(f"姿态 {label} 连续 3 次采集失败，中止")


def _collect_pose_once(imu, secs, label, try_no):
    if try_no > 1:
        print(f"\n>>> 姿态 {label}（第 {try_no} 次尝试）")
    else:
        print(f"\n>>> 姿态 {label}")
    print("    请把 IMU 板（或装好的设备）静止放在该姿态，然后按回车开始…")
    input()
    # 预热：跳过传感器稳定期 + 采集滚动窗口判静止
    warm = 0.6
    wins = []
    t0 = time.time()
    while time.time() - t0 < warm:
        g, _ = imu.read_raw()
        wins.append(g)
        time.sleep(0.01)
    while True:
        g, _ = imu.read_raw()
        wins.append(g)
        if len(wins) > 50:
            wins.pop(0)
        if len(wins) >= 30:
            arr = np.asarray(wins[-30:])
            std = np.std(arr, axis=0)          # 每轴 std, °/s
            # 实测本板噪声底 ~0.1 dps（USB-HID 桥 + 板子电源噪声），
            # 阈值取 0.3 dps：静止可过，运动/振动（一般 >1 dps）被拒
            if np.all(std < 0.3):
                if np.linalg.norm(g) > 30:     # 明显的运动（>30°/s）先提示
                    print("    检测到运动，请保持静止…")
                break
        time.sleep(0.01)
    # 正式采集
    samples = []
    t_end = time.time() + secs
    while time.time() < t_end:
        g, a = imu.read_raw()
        samples.append([g[0], g[1], g[2], a[0], a[1], a[2]])
        time.sleep(0.01)
    arr = np.asarray(samples, dtype=np.float64)
    n = len(arr)
    a_norm = np.linalg.norm(arr[:, 3:6], axis=1).mean()
    print(f"    采集 {n} 样本 | accel 模长均值 {a_norm:.3f} g | gyro 均值 "
          f"[{arr[:,0].mean():+.3f} {arr[:,1].mean():+.3f} {arr[:,2].mean():+.3f}] °/s")
    return arr


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="CH347T+LSM6DS3 一键自动校准")
    ap.add_argument("--poses", type=int, default=9, help="静止姿态数（默认 9；建议 ≥9 且含 45° 斜放，矩阵更准）")
    ap.add_argument("--secs", type=float, default=6.0, help="每姿态采集秒数（默认 6）")
    ap.add_argument("--out", default=None, help="输出 JSON 路径（默认项目根 imu_calibration.json）")
    ap.add_argument("--lib", default=None, help="libch347.so 路径（默认 third_party 自动查找）")
    ap.add_argument("--skip-verify", action="store_true", help="跳过采集后自检")
    args = ap.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_path = args.out or os.path.join(project_root, "imu_calibration.json")
    lib = args.lib or find_lib_path(project_root)

    # 压掉 libch347.so 的 stdout 调试刷屏（脚本自身输出走 stderr）
    try:
        os.dup2(os.open(os.devnull, os.O_WRONLY), 1)
    except OSError:
        pass

    print("========== CH347T + LSM6DS3 自动校准 ==========")
    print(f"libch347.so : {lib}")
    if not os.path.exists(lib):
        print(f"[错] 库不存在: {lib}")
        sys.exit(1)

    imu = CH347LSM6(None, lib)
    try:
        imu.open()
        imu.init_sensor()
    except RuntimeError as e:
        print(f"[错] {e}")
        sys.exit(2)

    print(f"开始采集 {args.poses} 个静止姿态 × {args.secs}s…")
    print("提示：覆盖 6 个基本朝向（平放正/反、竖立各方向），再加 2-3 个 45° 斜放姿态；")
    print("      每姿态完全静止即可（自动判静止后开始采），板子可手持，不动就行。")

    all_samples = []
    try:
        for i in range(args.poses):
            arr = collect_pose(imu, args.secs, f"{i + 1}/{args.poses}")
            all_samples.append(arr)
    except KeyboardInterrupt:
        print("\n中断，已采集样本放弃本次输出。")
        imu.close()
        sys.exit(130)

    imu.close()
    data = np.vstack(all_samples)
    gyro, accel = data[:, :3], data[:, 3:6]

    # ---- 拟合：先按姿态剔除运动段，再全体拟合 ----
    # 注意：不能用 norm(gyro)<阈值 判静止 —— 零偏本身 ~5 dps 会撑爆 norm，
    # 导致 mask 全 False 走 fallback、把运动段(如姿态4那次)混进拟合。
    # 正确做法：每姿态 gyro std 大 = 该姿态采集时没静止 → 整段剔除。
    keep, dropped = [], []
    for idx, arr in enumerate(all_samples):
        std = np.std(arr[:, :3], axis=0)
        if np.all(std < 0.5):
            keep.append(arr)
        else:
            dropped.append((idx + 1, std.round(3)))
    data = np.vstack(keep)
    if dropped:
        print(f"[info] 剔除运动姿态段: {dropped}（其余样本均为静止，直接全体拟合）")
    if len(keep) < 4:
        print("[警告] 有效姿态 < 4，椭球拟合不可靠，建议重采")
    gyro, accel = data[:, :3], data[:, 3:6]
    g_bias = gyro.mean(axis=0)
    a_bias, M = accel_ellipsoid(accel, 9.81)

    result = {
        "imuBiasGyro": [round(float(x), 6) for x in g_bias],
        "imuBiasAccel": [round(float(x), 6) for x in a_bias],
        "imuCalibMatrix": [round(float(x), 6) for x in M.flatten()],
        "gravity": 9.81,
        "note": "backend: accel_ms2 = (imuCalibMatrix @ (raw_accel_g - imuBiasAccel)) / gravity; "
                "gyro_rad = (raw_dps - imuBiasGyro) * pi/180",
        "samples": int(len(data)),
        "poses": args.poses,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[ok] 校准 JSON 已写入: {out_path}")
    print("  imuBiasGyro (dps) :", g_bias.round(4))
    print("  imuBiasAccel (g)  :", a_bias.round(4))
    det = float(np.linalg.det(M))
    print(f"  imuCalibMatrix 行列式: {det:.4f}（这是标度/耦合矩阵，det 不必=1，以自检为准）")
    if not (0.9 < det < 1.1):
        print("  [info] det 偏离 1 属正常（含标度）；若自检模长≈9.81 即为合格")

    # ---- 自检：静止几秒，按 ch347t.cc 公式验证 ----
    if not args.skip_verify:
        print("\n===== 自检（请保持设备静止 5 秒）=====")
        imu2 = CH347LSM6(None, lib)
        try:
            imu2.open()
            imu2.init_sensor()
            gs, as_ = [], []
            t_end = time.time() + 5.0
            while time.time() < t_end:
                g, a = imu2.read_raw()
                gs.append(g)
                as_.append(a)
                time.sleep(0.01)
            gs, as_ = np.asarray(gs), np.asarray(as_)
            gc = (gs - np.asarray(g_bias)) * np.pi / 180.0
            a0 = as_ - np.asarray(a_bias)
            ac = (M @ a0.T).T
            ac_ms2 = ac * 9.81
            print(f"  校准后 gyro 均值: [{gc[:,0].mean():+.4f} {gc[:,1].mean():+.4f} {gc[:,2].mean():+.4f}] rad/s （期望≈0）")
            print(f"  校准后 accel 模长: {np.linalg.norm(ac_ms2, axis=1).mean():.3f} m/s² （期望≈9.81）")
            if abs(np.linalg.norm(ac_ms2, axis=1).mean() - 9.81) < 0.2 and np.max(np.abs(gc.mean(axis=0))) < 0.02:
                print("  [ok] 自检通过 ✔")
            else:
                print("  [警告] 自检偏差偏大：重新采集或检查姿态覆盖")
            imu2.close()
        except RuntimeError as e:
            print(f"  自检失败（不影响已写入 JSON）: {e}")

    print("\n完成。ch347t 下次启动会自动加载该校准；日志会打印")
    print("  'IMU calibration loaded: gyro_bias_dps=... accel_bias_g=...'")


if __name__ == "__main__":
    main()