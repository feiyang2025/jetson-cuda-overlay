#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CH347 (Mode 2 / HID) + LSM6DS3 - I2C scan & IMU data reader
For NVIDIA AGX Orin (Linux aarch64)

Usage:
    sudo python3 imu_lsm6ds3.py          # stream IMU data
    sudo python3 imu_lsm6ds3.py --scan   # scan I2C bus only

Requires: lib/libch347.so (aarch64) placed next to this script or in ./lib/
Device:   CH347T dip switch S1=OFF, S2=ON (Mode 2, PID 55DC)
"""

import ctypes
import sys
import os
import time
import math
import glob
import argparse

# ---------------------------------------------------------------------------
# load libch347.so
# ---------------------------------------------------------------------------
def load_lib():
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "libch347.so"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib", "libch347.so"),
        "/usr/lib/libch347.so",
        "/usr/local/lib/libch347.so",
    ]
    for p in candidates:
        if os.path.exists(p):
            return ctypes.CDLL(p)
    sys.exit("ERROR: libch347.so not found (looked in: %s)" % ", ".join(candidates))


lib = load_lib()

# API: int CH347OpenDevice(char *devpath)   -- HID mode: pass /dev/hidrawX path
#      void CH347CloseDevice(int fd)
#      BOOL CH347I2C_Set(int fd, ULONG mode)
#      BOOL CH347StreamI2C(int fd, ULONG wlen, void *wbuf, ULONG rlen, void *rbuf)
lib.CH347OpenDevice.restype = ctypes.c_int
lib.CH347OpenDevice.argtypes = [ctypes.c_char_p]
lib.CH347CloseDevice.restype = None
lib.CH347CloseDevice.argtypes = [ctypes.c_int]
lib.CH347I2C_Set.restype = ctypes.c_int
lib.CH347I2C_Set.argtypes = [ctypes.c_int, ctypes.c_ulong]
lib.CH347StreamI2C.restype = ctypes.c_int
lib.CH347StreamI2C.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p,
                                ctypes.c_ulong, ctypes.c_void_p]

# ---------------------------------------------------------------------------
# find CH347 hidraw device (VID 0x1A86, PID 0x55DC)
# ---------------------------------------------------------------------------
def find_ch347_hidraw():
    """return (device_path, uevent) - prefer the SPI+I2C interface (MI_01)"""
    fallback = None
    for hidraw in sorted(glob.glob("/sys/class/hidraw/hidraw*")):
        try:
            with open(os.path.join(hidraw, "device/uevent")) as f:
                uevent = f.read()
            if "VID_1A86" in uevent and "PID_55DC" in uevent:
                dev = "/dev/" + os.path.basename(hidraw)
                if "MI_01" in uevent:      # SPI+I2C interface
                    return dev, uevent
                if fallback is None:
                    fallback = (dev, uevent)
        except IOError:
            continue
    return fallback


# ---------------------------------------------------------------------------
# I2C helpers
# ---------------------------------------------------------------------------
fd = -1

def read_reg(addr, reg, length):
    wbuf = bytes([addr << 1, reg])
    rbuf = (ctypes.c_ubyte * length)()
    if not lib.CH347StreamI2C(fd, len(wbuf), wbuf, length, rbuf):
        return None
    return bytes(rbuf)


def write_reg(addr, reg, value):
    wbuf = bytes([addr << 1, reg, value])
    return lib.CH347StreamI2C(fd, len(wbuf), wbuf, 0, None)


def probe(addr):
    """probe: write 1 byte, check return (HID lib returns 0 on NACK)"""
    wbuf = bytes([addr << 1])
    return lib.CH347StreamI2C(fd, 1, wbuf, 0, None)


def s16(lo, hi):
    v = lo | (hi << 8)
    return v - 0x10000 if v & 0x8000 else v


# ---------------------------------------------------------------------------
# LSM6DS3
# ---------------------------------------------------------------------------
IMU_ADDR = 0x6B          # SDO=high; try 0x6A if not found
ACC_SENS = 2.0 / 32768   # +-2g
GYRO_SENS = 245.0 / 32768  # 245dps


def imu_init():
    who = read_reg(IMU_ADDR, 0x0F, 1)
    if not who or who[0] != 0x69:
        print("WHO_AM_I read failed or wrong chip: %s" %
              ("0x%02X" % who[0] if who else "None"))
        sys.exit(1)
    print("LSM6DS3 found (WHO_AM_I=0x69)")

    write_reg(IMU_ADDR, 0x12, 0x01)  # soft reset
    time.sleep(0.1)
    write_reg(IMU_ADDR, 0x10, 0x60)  # accel: 416Hz, +-2g
    write_reg(IMU_ADDR, 0x11, 0x60)  # gyro: 416Hz, 245dps
    write_reg(IMU_ADDR, 0x12, 0x44)  # BDU + auto-increment
    time.sleep(0.2)


def imu_read():
    d = read_reg(IMU_ADDR, 0x22, 12)  # gyro(0x22..27) + accel(0x28..2D)
    gx, gy, gz = (s16(d[i], d[i + 1]) * GYRO_SENS for i in (0, 2, 4))
    ax, ay, az = (s16(d[i], d[i + 1]) * ACC_SENS for i in (6, 8, 10))
    return ax, ay, az, gx, gy, gz


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    global fd
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", action="store_true", help="I2C scan only")
    parser.add_argument("--addr", type=lambda x: int(x, 0), default=IMU_ADDR)
    parser.add_argument("--rate", type=float, default=50.0, help="sample rate Hz")
    args = parser.parse_args()

    dev, uevent = find_ch347_hidraw()
    if not dev:
        sys.exit("CH347 (VID:PID 1A86:55DC) hidraw device not found.\n"
                 "Check: dip switch S1=OFF S2=ON (Mode 2), ls /dev/hidraw*")
    print("CH347 device: %s" % dev)

    fd = lib.CH347OpenDevice(dev.encode())
    if fd < 0:
        sys.exit("CH347OpenDevice failed (try sudo, or check udev rules)")
    print("opened, fd=%d" % fd)

    if not lib.CH347I2C_Set(fd, 1):  # 100kHz
        sys.exit("CH347I2C_Set failed")
    print("I2C initialized (100kHz)")
    time.sleep(0.2)

    try:
        if args.scan:
            print("\nI2C scan 0x03-0x77:")
            found = []
            for a in range(0x03, 0x78):
                if read_reg(a, 0x0F, 1) is not None and probe(a):
                    # ACK check: verify a real register read works
                    t = read_reg(a, 0x0F, 1)
                    if t:
                        found.append(a)
                        print("  0x%02X  WHO_AM_I=0x%02X" % (a, t[0]))
            print("found:", [hex(a) for a in found])
        else:
            imu_init()
            print("\nStreaming (Ctrl+C to stop)")
            print("%6s | %7s %7s %7s | %8s %8s %8s" %
                  ("n", "aX g", "aY g", "aZ g", "gX dps", "gY dps", "gZ dps"))
            n = 0
            t0 = time.time()
            while True:
                ax, ay, az, gx, gy, gz = imu_read()
                n += 1
                print("%6d | %7.3f %7.3f %7.3f | %8.1f %8.1f %8.1f" %
                      (n, ax, ay, az, gx, gy, gz))
                if n % 50 == 0:
                    print("      --- %.0f Hz ---" % (n / (time.time() - t0)))
                time.sleep(1.0 / args.rate)
    except KeyboardInterrupt:
        print("\nstopped, %.0f samples" % n)
    finally:
        lib.CH347CloseDevice(fd)


if __name__ == "__main__":
    main()
