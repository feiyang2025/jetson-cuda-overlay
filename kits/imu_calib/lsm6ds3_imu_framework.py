"""
347T IMU 接入框架 (LSM6DS3 over CH347T)
=====================================
和具体 CH347T 驱动解耦：只把"传输层"抽象成 IMUTransport，
你重编过的驱动只要实现 open/close/read_reg/write_reg 四个方法即可。
LSM6DS3 寄存器逻辑与"基准参数校准"的应用逻辑与驱动无关，基本不用改。

参考：
  - reference/mouxan_helpers.py    : mouxan 的 imuCalibMatrix(3x3 基准参数) 思路
  - reference/agx347_imu_lsm6ds3.py : CH347T Mode2 HID 的参考实现（可整段替换传输层）
"""

import os
import ctypes
import math
import json
import glob
import time

from cereal import log


# ===========================================================================
# 1) 传输层（你按自己重编的 CH347T 驱动来填这一段）
# ===========================================================================
class IMUTransport:
    def open(self):
        raise NotImplementedError   # TODO: 打开你重编的 CH347T 驱动句柄

    def close(self):
        raise NotImplementedError   # TODO: 关闭句柄

    def read_reg(self, addr: int, reg: int, length: int) -> bytes:
        raise NotImplementedError   # TODO: 通过 I2C 从 addr 设备的 reg 读 length 字节

    def write_reg(self, addr: int, reg: int, value: int) -> None:
        raise NotImplementedError   # TODO: 通过 I2C 向 addr 设备的 reg 写 1 字节


# ---- 参考实现：CH347T Mode2 HID（agx347 官方 libch347.so，驱动不同时整段替换即可） ----
class CH347HIDTransport(IMUTransport):
    VID = 0x1A86
    PID = 0x55DC

    def __init__(self, lib_path: str = "libch347.so"):
        self.lib = ctypes.CDLL(lib_path)
        self.lib.CH347OpenDevice.restype = ctypes.c_int
        self.lib.CH347OpenDevice.argtypes = [ctypes.c_char_p]
        self.lib.CH347CloseDevice.restype = None
        self.lib.CH347CloseDevice.argtypes = [ctypes.c_int]
        self.lib.CH347I2C_Set.restype = ctypes.c_int
        self.lib.CH347I2C_Set.argtypes = [ctypes.c_int, ctypes.c_ulong]
        self.lib.CH347StreamI2C.restype = ctypes.c_int
        self.lib.CH347StreamI2C.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p,
                                            ctypes.c_ulong, ctypes.c_void_p]
        self.fd = -1

    @staticmethod
    def _find_hidraw():
        fallback = None
        for hidraw in sorted(glob.glob("/sys/class/hidraw/hidraw*")):
            try:
                with open(os.path.join(hidraw, "device", "uevent")) as f:
                    uevent = f.read()
            except OSError:
                continue
            if f"VID_{CH347HIDTransport.VID:04X}" in uevent and f"PID_{CH347HIDTransport.PID:04X}" in uevent:
                dev = "/dev/" + os.path.basename(hidraw)
                if "MI_01" in uevent:
                    return dev
                if fallback is None:
                    fallback = dev
        return fallback

    def open(self):
        dev = self._find_hidraw()
        if not dev:
            raise RuntimeError("CH347 hidraw (VID:PID 1a86:55dc) not found")
        self.fd = self.lib.CH347OpenDevice(dev.encode())
        if self.fd < 0:
            raise RuntimeError("CH347OpenDevice failed")
        if not self.lib.CH347I2C_Set(self.fd, 1):
            raise RuntimeError("CH347I2C_Set failed")
        time.sleep(0.1)

    def close(self):
        if self.fd >= 0:
            self.lib.CH347CloseDevice(self.fd)
            self.fd = -1

    def read_reg(self, addr: int, reg: int, length: int) -> bytes:
        wbuf = bytes([addr << 1, reg])
        rbuf = (ctypes.c_ubyte * length)()
        if not self.lib.CH347StreamI2C(self.fd, len(wbuf), wbuf, length, rbuf):
            return None
        return bytes(rbuf)

    def write_reg(self, addr: int, reg: int, value: int) -> None:
        wbuf = bytes([addr << 1, reg, value])
        self.lib.CH347StreamI2C(self.fd, len(wbuf), wbuf, 0, None)


# ===========================================================================
# 2) LSM6DS3 逻辑（与驱动无关，基本不用改）
# ===========================================================================
IMU_ADDR = 0x6B
ACC_SENS = 2.0 / 32768.0     # TODO: 若你在驱动/配置里改了量程(FS)，这里要同步改
GYRO_SENS = 245.0 / 32768.0  # TODO: 同上（±2g / 245dps 对应值；±8g=0.244/1000*9.81 等）


def _s16(lo, hi):
    v = lo | (hi << 8)
    return v - 0x10000 if v & 0x8000 else v


def _identity_calib():
    return {"bias_gyro": [0.0, 0.0, 0.0], "bias_accel": [0.0, 0.0, 0.0],
            "matrix": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "gravity": 9.81}


class LSM6DS3_IMU:
    def __init__(self, transport: IMUTransport, imu_addr: int = IMU_ADDR,
                 calib_json: str = "imu_calibration.json"):
        self.t = transport
        self.imu_addr = imu_addr
        self.calib = self._load_calib(calib_json)
        self.source = log.SensorEventData.SensorSource.lsm6ds3

    def _load_calib(self, path):
        try:
            with open(path) as f:
                c = json.load(f)
            return {
                "bias_gyro": c.get("imuBiasGyro", [0.0, 0.0, 0.0]),
                "bias_accel": c.get("imuBiasAccel", [0.0, 0.0, 0.0]),
                "matrix": c.get("imuCalibMatrix", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
                "gravity": c.get("gravity", 9.81),
            }
        except (OSError, ValueError):
            return _identity_calib()

    def init(self):
        self.t.open()
        who = self.t.read_reg(0x0F, 1)            # WHO_AM_I 期望 0x69
        if not who or who[0] != 0x69:
            self.imu_addr = 0x6A                  # TODO: 地址不对就换 0x6A 重试
            who = self.t.read_reg(0x0F, 1)
            if not who or who[0] != 0x69:
                raise RuntimeError("LSM6DS3 WHO_AM_I != 0x69")
        self.t.write_reg(0x12, 0x01); time.sleep(0.1)   # soft reset
        self.t.write_reg(0x10, 0x60)                    # CTRL1_XL: 416Hz, +-2g
        self.t.write_reg(0x11, 0x60)                    # CTRL2_G : 416Hz, 245dps
        self.t.write_reg(0x12, 0x44)                    # CTRL3_C : BDU + 地址自增
        time.sleep(0.2)

    def read_raw(self):
        d = self.t.read_reg(0x22, 12)              # gyro(0x22-0x27) + accel(0x28-0x2D) 连读
        if not d or len(d) != 12:
            raise RuntimeError("IMU read error")
        g = [_s16(d[i], d[i + 1]) * GYRO_SENS for i in (0, 2, 4)]
        a = [_s16(d[i], d[i + 1]) * ACC_SENS for i in (6, 8, 10)]
        return g, a

    def _apply_calib(self, g, a):
        gc = [g[i] - self.calib["bias_gyro"][i] for i in range(3)]
        a0 = [a[i] - self.calib["bias_accel"][i] for i in range(3)]
        m = self.calib["matrix"]
        a_unit = [
            m[0] * a0[0] + m[1] * a0[1] + m[2] * a0[2],
            m[3] * a0[0] + m[4] * a0[1] + m[5] * a0[2],
            m[6] * a0[0] + m[7] * a0[1] + m[8] * a0[2],
        ]
        gx = [x * math.pi / 180.0 for x in gc]
        ax = [x / self.calib.get("gravity", 9.81) for x in a_unit]
        return gx, ax

    def get_gyro_event(self, ts: int | None = None):
        if ts is None:
            ts = time.time_ns()
        g, a = self.read_raw()
        gc, _ = self._apply_calib(g, a)
        event = log.SensorEventData.new_message()
        event.timestamp = ts
        event.version = 2
        event.sensor = 5
        event.type = 16
        event.source = self.source
        gg = event.init('gyroUncalibrated')
        gg.v = gc
        gg.status = 1
        return event

    def get_accel_event(self, ts: int | None = None):
        if ts is None:
            ts = time.time_ns()
        g, a = self.read_raw()
        _, ac = self._apply_calib(g, a)
        event = log.SensorEventData.new_message()
        event.timestamp = ts
        event.version = 1
        event.sensor = 1
        event.type = 1
        event.source = self.source
        aa = event.init('acceleration')
        aa.v = ac
        aa.status = 1
        return event

    def shutdown(self):
        self.t.close()


# ===========================================================================
# 3) 接入 sensord 的薄封装（与现有 openpilot sensord 接口一致）
# ===========================================================================
class CH347_Gyro:
    def __init__(self, imu: LSM6DS3_IMU):
        self.imu = imu

    def init(self):
        self.imu.init()

    def reset(self):
        pass

    def shutdown(self):
        self.imu.shutdown()

    def is_data_valid(self):
        return True

    def get_event(self, ts: int | None = None):
        return self.imu.get_gyro_event(ts)


class CH347_Accel:
    def __init__(self, imu: LSM6DS3_IMU):
        self.imu = imu

    def init(self):
        self.imu.init()

    def reset(self):
        pass

    def shutdown(self):
        self.imu.shutdown()

    def is_data_valid(self):
        return True

    def get_event(self, ts: int | None = None):
        return self.imu.get_accel_event(ts)
