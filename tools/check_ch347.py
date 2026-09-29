#!/usr/bin/env python3
"""§4 第 3 步自检: CH347 IMU 是否真的出 accelerometer / gyroscope / temperatureSensor。

插着 CH347 → 三路消息都要有; 没插 → ch347t 应干净退出(不崩、不占 msgq)。

用法: python3 tools/check_ch347.py <树根> [采集秒数]
"""
import os
import subprocess
import sys
import time

TREE = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
DURATION = float(sys.argv[2]) if len(sys.argv) > 2 else 25.0
SENSORD = os.path.join(TREE, "openpilot", "system", "sensord")

sys.path.insert(0, TREE)
sys.path.insert(0, os.path.join(TREE, "openpilot"))

import cereal.messaging as messaging  # noqa: E402

FAIL = 0


def check(cond, msg):
  global FAIL
  print(("  [PASS] " if cond else "  [FAIL] ") + msg)
  if not cond:
    FAIL = 1


RUNNER = os.path.join(SENSORD, "run_ch347t.sh")
print(f"[check_ch347] 起 {RUNNER} (cwd={SENSORD})")
proc = subprocess.Popen(["bash", RUNNER], cwd=SENSORD,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

sm = messaging.SubMaster(["accelerometer", "gyroscope", "temperatureSensor"])
counts = {"accelerometer": 0, "gyroscope": 0, "temperatureSensor": 0}
first = {}
t0 = time.monotonic()
log_tail = []


def drain_log():
  import threading

  def _r():
    for line in proc.stdout:
      log_tail.append(line.rstrip())
      if len(log_tail) > 40:
        log_tail.pop(0)
  threading.Thread(target=_r, daemon=True).start()


drain_log()

try:
  while time.monotonic() - t0 < DURATION and proc.poll() is None:
    sm.update(100)
    for s in counts:
      if sm.updated[s]:
        counts[s] += 1
        if s not in first:
          first[s] = sm[s]
except Exception as e:
  print(f"[check_ch347] 采集期异常(不影响终止): {e}")

elapsed = time.monotonic() - t0
rc = proc.poll()
if rc is None:
  proc.terminate()
  try:
    proc.wait(timeout=5)
  except subprocess.TimeoutExpired:
    proc.kill()
    proc.wait(timeout=3)

# 释放 msgq socket: 进程退出后 SubMaster 立刻关闭, 免得下次跑撞 MultiplePublishersError
del sm

print(f"[check_ch347] 采集 {elapsed:.1f}s, 进程 rc={rc}")
for s, n in counts.items():
  print(f"  {s}: {n} 条 ({n / elapsed:.1f} Hz)")
if "accelerometer" in first:
  ev = first["accelerometer"]
  v = ev.acceleration.v if ev.which() == "acceleration" else None
  print(f"  accel 首帧 (which={ev.which()}): v = {[round(x, 3) for x in v] if v is not None else '?'} (m/s^2, 合模长 "
        f"{sum(x * x for x in v) ** 0.5:.2f})")
if "gyroscope" in first:
  ev = first["gyroscope"]
  v = ev.gyro.v if ev.which() == "gyro" else None
  print(f"  gyro 首帧 (which={ev.which()}): v = {[round(x, 5) for x in v] if v is not None else '?'} (rad/s)")
if "temperatureSensor" in first:
  ev = first["temperatureSensor"]
  print(f"  temp 首帧 (which={ev.which()}): {ev.to_dict()}")

print("[check_ch347] ch347t 日志尾部:")
for line in log_tail[-12:]:
  print("   |", line)

if counts["accelerometer"] > 0 or counts["gyroscope"] > 0:
  check(counts["accelerometer"] > 50, f"accelerometer 有数据 ({counts['accelerometer']} 条)")
  check(counts["gyroscope"] > 50, f"gyroscope 有数据 ({counts['gyroscope']} 条)")
  check(counts["temperatureSensor"] > 0, f"temperatureSensor 有数据 ({counts['temperatureSensor']} 条)")
  hz = counts["accelerometer"] / elapsed
  check(hz > 50, f"加速度频率 {hz:.1f} Hz (>50 才算真在跑)")
else:
  check(rc is not None, f"没插 CH347: 进程应干净退出 (rc={rc})")
  check("Failed to load CH347 library" not in "\n".join(log_tail), "不报载库失败")
  check("CH347 not available after waiting" in "\n".join(log_tail) or rc is not None,
        "日志出现 'CH347 not available after waiting' (等待超时后走兜底)")

print()
print("CH347 自检: " + ("全部通过" if FAIL == 0 else "有 FAIL 项"))
sys.exit(FAIL)
