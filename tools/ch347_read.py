#!/usr/bin/env python3
"""Standalone CH347 (USB-I2C) LSM6DS3 reader for AGX Orin.

为什么不用 sensord_ch347 守护进程：
  1. 本板 CH347<->LSM6DS3 的 I2C 只有 20kHz(0x00) 稳定；100kHz/400kHz 会挂总线。
  2. LSM6DS3 控制寄存器必须"写一个立即回读校验一个"，连续批量写会丢。
  3. 当前 ch347t 的 init_sensor() 批量写 + 新版候选速率代码在 -O2 下会段错误。

用法:
  python3 ch347_read.py            # 连续打印 accel/gyro
  python3 ch347_read.py --rate 50  # 采样率 Hz (默认 50)
"""
import argparse
import ctypes
import glob
import os
import sys
import time
from pathlib import Path

DEFAULT_LIB = ("/home/nvidia/公共/openpilot_cp_egpucp/openpilot/openpilot/"
               "third_party/ch347/lib/aarch64/dynamic/libch347.so")
ADDR = 0x6B
WHO_REG = 0x0F
WHO_ID = 0x69


def hys_read(dev, key):
  try:
    for line in Path(f"/sys/class/hidraw/{os.path.basename(dev)}/device/uevent").read_text().splitlines():
      if line.startswith(key):
        return line[len(key):]
  except OSError:
    pass
  return None


def pick_device():
  best = None
  for dev in sorted(glob.glob("/dev/hidraw*")):
    hid = hys_read(dev, "HID_ID=")
    if not hid:
      continue
    try:
      _bus, vid, pid = (int(x, 16) for x in hid.split(":"))
    except ValueError:
      continue
    if vid != 0x1A86 or pid not in (0x55DB, 0x55DC):
      continue
    phys = hys_read(dev, "HID_PHYS=") or ""
    idx = int(phys.rsplit("input", 1)[1]) if "input" in phys else -1
    if best is None or idx > best[0]:
      best = (idx, dev)
  return best[1] if best else None


def s16(lo, hi):
  v = lo | (hi << 8)
  return v - 0x10000 if v & 0x8000 else v


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--lib", default=os.environ.get("CH347_LIB", DEFAULT_LIB))
  ap.add_argument("--rate", type=float, default=50.0)
  args = ap.parse_args()

  dev = pick_device()
  if not dev:
    sys.exit("未找到 CH347 (1a86:55db/55dc) hidraw 节点")

  lib = ctypes.CDLL(args.lib)
  lib.CH347OpenDevice.restype = ctypes.c_int
  lib.CH347I2C_Set.argtypes = [ctypes.c_int, ctypes.c_int]
  lib.CH347StreamI2C.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
  lib.CH347StreamI2C.restype = ctypes.c_bool

  n = int(dev.rsplit("hidraw", 1)[1])
  fd = lib.CH347OpenDevice(ctypes.c_ulong(n - 2).value)  # 部署库: 索引 = hidrawN - 2
  if fd < 0:
    sys.exit("CH347OpenDevice 失败")

  lib.CH347I2C_Set(fd, 0x00)  # 20kHz
  if hasattr(lib, "CH347I2C_SetStretch"):
    lib.CH347I2C_SetStretch(fd, 1)
  if hasattr(lib, "CH347I2C_SetDelaymS"):
    lib.CH347I2C_SetDelaymS(fd, 5)
  time.sleep(0.05)

  def rd(reg, length=1):
    w = (ctypes.c_ubyte * 2)(ADDR << 1, reg)
    r = (ctypes.c_ubyte * length)()
    return bytes(r) if lib.CH347StreamI2C(fd, 2, w, length, r) else None

  def wr(reg, val):
    w = (ctypes.c_ubyte * 3)(ADDR << 1, reg, val)
    return lib.CH347StreamI2C(fd, 3, w, 0, None)

  def preheat():
    for _ in range(600):
      if rd(WHO_REG) == bytes([WHO_ID]) and all(rd(WHO_REG) == bytes([WHO_ID]) for _ in range(5)):
        return True
      time.sleep(0.003)
    return False

  def wr_verify(reg, val, tries=20):
    for _ in range(tries):
      wr(reg, val)
      if rd(reg) == bytes([val]):
        return True
      time.sleep(0.002)
    return False

  if not preheat():
    sys.exit("IMU 无应答 (WHO_AM_I != 0x69)")

  wr(0x12, 0x01)  # soft reset
  time.sleep(0.05)
  wr_verify(0x10, 0x60)  # CTRL1_XL: 416Hz, +-2g
  wr_verify(0x11, 0x60)  # CTRL2_G : 416Hz, 245dps
  wr_verify(0x12, 0x44)  # BDU + IF_INC

  print(f"IMU OK: {dev}  WHO_AM_I=0x{WHO_ID:02X}")
  print("%8s %9s %9s %9s | %9s %9s %9s" % ("n", "aX(g)", "aY(g)", "aZ(g)", "gX(dps)", "gY(dps)", "gZ(dps)"))
  A = 2.0 / 32768
  G = 245.0 / 32768
  i = 0
  period = 1.0 / args.rate if args.rate > 0 else 0.0
  try:
    while True:
      t0 = time.time()
      b = rd(0x22, 12)  # gyro 0x22..0x27 + accel 0x28..0x2D
      if b is None:
        preheat()
        wr_verify(0x10, 0x60)
        wr_verify(0x11, 0x60)
        continue
      if not any(b):
        wr_verify(0x10, 0x60)
        wr_verify(0x11, 0x60)
        continue
      v = [s16(b[j], b[j + 1]) for j in range(0, 12, 2)]
      ax, ay, az = (v[j] * A for j in (3, 4, 5))
      gx, gy, gz = (v[j] * G for j in (0, 1, 2))
      i += 1
      print("%8d %9.3f %9.3f %9.3f | %9.1f %9.1f %9.1f" % (i, ax, ay, az, gx, gy, gz))
      dt = period - (time.time() - t0)
      if dt > 0:
        time.sleep(dt)
  except KeyboardInterrupt:
    pass
  finally:
    lib.CH347CloseDevice(fd)


if __name__ == "__main__":
  main()
