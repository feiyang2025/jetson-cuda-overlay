#!/usr/bin/env python3
import os
import time
import ctypes
import math
import threading
import glob
import platform

import cereal.messaging as messaging
from cereal import log
from openpilot.common.swaglog import cloudlog
from openpilot.system.sensord.sensors.i2c_sensor import Sensor


def get_ch347_lib_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    lib_dir = os.path.join(project_root, "third_party", "ch347", "lib")

    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        arch_dir = "x64"
    elif machine.startswith("arm"):
        if machine.endswith("v7l") or machine.endswith("hf"):
            arch_dir = "arm-gnueabihf"
        else:
            arch_dir = "arm-gnueabi"
    elif machine.startswith("aarch64"):
        arch_dir = "aarch64"
    else:
        arch_dir = "x86"

    lib_path = os.path.join(lib_dir, arch_dir, "dynamic", "libch347.so")
    if not os.path.exists(lib_path):
        lib_path = os.path.join(lib_dir, "x64", "dynamic", "libch347.so")
    return lib_path


CH347_DEFAULT_LIB_PATH = get_ch347_lib_path()
CH347_WHO_AM_I_REG = 0x0F
CH347_LSM6_ADDRS = (0x6B, 0x6A)
CH347_WHO_AM_I_IDS = (0x69, 0x6A)

# Calibration schema matches sunnypilot-cuda tools/imu_calib (imu_calibration.json):
#   "imuBiasGyro"    : [gx, gy, gz]  (deg/s, dps)
#   "imuBiasAccel"   : [ax, ay, az]  (g)
#   "imuCalibMatrix" : [m0..m8]      (row-major 3x3: axis coupling + scale)
# Apply: gyro_rad = raw_rad - imuBiasGyro * pi/180
#        accel_ms2 = imuCalibMatrix @ (raw_mps2 - 9.81 * imuBiasAccel)
DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi


def calib_json_path():
    env = os.getenv("IMU_CALIB_JSON")
    if env:
        return env
    return os.path.join(os.getcwd(), "imu_calibration.json")


def _find_json_key(json: str, key: str) -> int:
    i = 0
    pat = f'"{key}"'
    while True:
        i = json.find(pat, i)
        if i < 0:
            return -1
        j = i + len(pat)
        while j < len(json) and json[j] in " \t\r\n":
            j += 1
        if j < len(json) and json[j] == ":":
            return j + 1
        i += len(pat)


def _parse_json_array(json: str, start: int, max_count: int) -> list[float]:
    i = start
    while i < len(json) and json[i] in " \t\r\n":
        i += 1
    if i >= len(json) or json[i] != "[":
        return []
    i += 1
    out = []
    while len(out) < max_count and i < len(json):
        while i < len(json) and json[i] in " \t\r\n,":
            i += 1
        if i >= len(json) or json[i] == "]":
            break
        j = i
        while j < len(json) and json[j] not in ",]\r\n":
            j += 1
        try:
            out.append(float(json[i:j]))
        except ValueError:
            out.append(0.0)
        i = j
    return out


def load_calibration() -> dict:
    calib = {
        "gyro_bias_dps": [0.0, 0.0, 0.0],
        "accel_bias_g": [0.0, 0.0, 0.0],
        "matrix": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    }
    path = calib_json_path()
    if not os.path.exists(path):
        return calib
    try:
        with open(path, encoding="utf-8") as f:
            json = f.read()
    except OSError as e:
        cloudlog.warning(f"IMU: could not read {path}: {e}")
        return calib
    pos = _find_json_key(json, "imuBiasGyro")
    if pos >= 0:
        for k, v in enumerate(_parse_json_array(json, pos, 3)):
            calib["gyro_bias_dps"][k] = v
    pos = _find_json_key(json, "imuBiasAccel")
    if pos >= 0:
        for k, v in enumerate(_parse_json_array(json, pos, 3)):
            calib["accel_bias_g"][k] = v
    pos = _find_json_key(json, "imuCalibMatrix")
    if pos >= 0:
        for k, v in enumerate(_parse_json_array(json, pos, 9)):
            calib["matrix"][k] = v
    return calib


class CH347LSM6:
    def __init__(self, dev_path: str, lib_path: str) -> None:
        self.dev_path = dev_path
        self.lib = ctypes.CDLL(lib_path)
        self.fd = -1
        self.addr = None
        self.source = None

    def _stream_read(self, write_bytes: list[int], read_len: int) -> bytes:
        wbuf = (ctypes.c_ubyte * len(write_bytes))(*write_bytes)
        rbuf = (ctypes.c_ubyte * read_len)()
        ack = ctypes.c_ubyte()
        ok = self.lib.CH347StreamI2C_RetAck(self.fd, len(write_bytes), wbuf, read_len, rbuf, ctypes.byref(ack))
        if not ok:
            raise OSError(f"CH347 I2C read failed, ack={ack.value}")
        return bytes(rbuf)

    def _stream_write(self, write_bytes: list[int]) -> None:
        wbuf = (ctypes.c_ubyte * len(write_bytes))(*write_bytes)
        ok = self.lib.CH347StreamI2C(self.fd, len(write_bytes), wbuf, 0, None)
        if not ok:
            raise OSError("CH347 I2C write failed")

    def read_u8(self, reg: int) -> int:
        return self._stream_read([self.addr << 1, reg], 1)[0]

    def write_u8(self, reg: int, val: int) -> None:
        self._stream_write([self.addr << 1, reg, val])

    def read_block(self, start_reg: int, length: int) -> bytes:
        write_bytes = [self.addr << 1 | 0x00, start_reg]
        wbuf = (ctypes.c_ubyte * len(write_bytes))(*write_bytes)
        rbuf = (ctypes.c_ubyte * length)()
        ack = ctypes.c_ubyte()
        ok = self.lib.CH347StreamI2C_RetAck(self.fd, len(write_bytes), wbuf, 0, None, ctypes.byref(ack))
        if not ok:
            raise OSError(f"CH347 I2C block write address failed, ack={ack.value}")
        wbuf_read = (ctypes.c_ubyte * 1)(self.addr << 1 | 0x01)
        ok = self.lib.CH347StreamI2C_RetAck(self.fd, 1, wbuf_read, length, rbuf, ctypes.byref(ack))
        if not ok:
            raise OSError(f"CH347 I2C block read data failed, ack={ack.value}")
        return bytes(rbuf)

    def open(self) -> None:
        self.fd = self.lib.CH347OpenDevice(self.dev_path.encode())
        if self.fd < 0:
            raise OSError(f"failed to open {self.dev_path}")
        if not self.lib.CH34xSetTimeout(self.fd, 2000, 2000):
            raise OSError("CH34xSetTimeout failed")
        if not self.lib.CH347I2C_Set(self.fd, 0x01):
            raise OSError("CH347I2C_Set failed")
        self.lib.CH347I2C_SetIgnoreNack(self.fd, 1)
        self.lib.CH347I2C_SetStretch(self.fd, True)
        time.sleep(0.02)

        for _ in range(8):
            for addr in CH347_LSM6_ADDRS:
                try:
                    who = self._stream_read([addr << 1, CH347_WHO_AM_I_REG], 1)[0]
                    if who in CH347_WHO_AM_I_IDS:
                        self.addr = addr
                        self.source = log.SensorEventData.SensorSource.lsm6ds3trc if who == 0x6A else log.SensorEventData.SensorSource.lsm6ds3
                        cloudlog.info(f"CH347 LSM6DS3 detected at addr=0x{addr:02X}, who=0x{who:02X}")
                        return
                except Exception:
                    continue
            time.sleep(0.02)

        cloudlog.error(f"CH347 LSM6DS3 not detected on any of these addresses: {CH347_LSM6_ADDRS}")
        raise OSError("CH347 LSM6DS3 not detected on 0x6A/0x6B")

    def init_sensor(self) -> None:
        self.write_u8(0x12, 0x01)
        time.sleep(0.1)
        self.write_u8(0x12, 0x04)
        self.write_u8(0x10, 0x40)
        self.write_u8(0x11, 0x40)

    def shutdown(self) -> None:
        if self.fd >= 0:
            try:
                self.write_u8(0x10, 0x00)
                self.write_u8(0x11, 0x00)
            except Exception:
                pass
            self.lib.CH347CloseDevice(self.fd)
            self.fd = -1


def detect_ch347_backend() -> tuple[str, str] | None:
    dev = os.getenv("SENSORD_CH347_DEV")
    if not dev:
        devs = sorted(glob.glob("/dev/ch34x_pis*"))
        if not devs:
            cloudlog.warning("No CH347 devices found in /dev/ch34x_pis*")
            return None
        cloudlog.info(f"Found CH347 devices: {devs}, using {devs[0]}")
        dev = devs[0]

    lib_path = os.getenv("SENSORD_CH347_LIB", CH347_DEFAULT_LIB_PATH)
    if not os.path.exists(dev):
        cloudlog.error(f"CH347 device does not exist: {dev}")
        return None
    if not os.path.exists(lib_path):
        cloudlog.error(f"CH347 library does not exist: {lib_path}")
        return None

    cloudlog.info(f"CH347 backend: dev={dev}, lib={lib_path}")
    return dev, lib_path


def ch347_loop(dev_path: str, lib_path: str, event: threading.Event) -> None:
    sensor = CH347LSM6(dev_path, lib_path)
    pm = messaging.PubMaster(["accelerometer", "gyroscope", "temperatureSensor"])
    from openpilot.common.realtime import Ratekeeper
    rk = Ratekeeper(104, print_delay_threshold=None)
    temp_ticks = 0
    try:
        sensor.open()
        sensor.init_sensor()
        cloudlog.info(f"Using CH347 backend: dev={dev_path}")

        calib = load_calibration()
        gyro_bias_rad = [v * DEG_TO_RAD for v in calib["gyro_bias_dps"]]
        cloudlog.info(f"IMU: calibration loaded gyro_bias_dps={calib['gyro_bias_dps']} "
                      f"accel_bias_g={calib['accel_bias_g']}")

        while not event.is_set():
            status = sensor.read_u8(0x1E)

            if status & 0x01:
                b = sensor.read_block(0x28, 6)
                x = Sensor.parse_16bit(b[0], b[1])
                y = Sensor.parse_16bit(b[2], b[3])
                z = Sensor.parse_16bit(b[4], b[5])
                scale = 9.81 * 2.0 / (1 << 15)

                raw = [y * scale, -x * scale, z * scale]
                centered = [raw[k] - 9.81 * calib["accel_bias_g"][k] for k in range(3)]
                m = calib["matrix"]

                msg = messaging.new_message("accelerometer", valid=True)
                msg.accelerometer.version = 1
                msg.accelerometer.sensor = 1
                msg.accelerometer.type = 1
                msg.accelerometer.source = sensor.source
                msg.accelerometer.timestamp = msg.logMonoTime
                acc = msg.accelerometer.init("acceleration")
                acc.v = [
                    m[0] * centered[0] + m[1] * centered[1] + m[2] * centered[2],
                    m[3] * centered[0] + m[4] * centered[1] + m[5] * centered[2],
                    m[6] * centered[0] + m[7] * centered[1] + m[8] * centered[2],
                ]
                acc.status = 1
                pm.send("accelerometer", msg)

            if status & 0x02:
                b = sensor.read_block(0x22, 6)
                x = Sensor.parse_16bit(b[0], b[1])
                y = Sensor.parse_16bit(b[2], b[3])
                z = Sensor.parse_16bit(b[4], b[5])
                scale = (8.75 / 1000.0) * (math.pi / 180.0)

                msg = messaging.new_message("gyroscope", valid=True)
                msg.gyroscope.version = 2
                msg.gyroscope.sensor = 5
                msg.gyroscope.type = 16
                msg.gyroscope.source = sensor.source
                msg.gyroscope.timestamp = msg.logMonoTime
                gyro = msg.gyroscope.init("gyroUncalibrated")
                gyro.v = [y * scale - gyro_bias_rad[0], -x * scale - gyro_bias_rad[1], z * scale - gyro_bias_rad[2]]
                gyro.status = 1
                pm.send("gyroscope", msg)

            temp_ticks += 1

            if temp_ticks >= 10:
                temp_ticks = 0
                tb = sensor.read_block(0x20, 2)
                raw_t = Sensor.parse_16bit(tb[0], tb[1])
                tscale = 16.0 if sensor.source == log.SensorEventData.SensorSource.lsm6ds3 else 256.0

                msg = messaging.new_message("temperatureSensor", valid=True)
                msg.temperatureSensor.version = 1
                msg.temperatureSensor.source = sensor.source
                msg.temperatureSensor.timestamp = msg.logMonoTime
                msg.temperatureSensor.temperature = 25 + (raw_t / tscale)
                pm.send("temperatureSensor", msg)

            rk.keep_time()
    finally:
        sensor.shutdown()
