#!/usr/bin/env python3
"""§4 第 5/6 步 + WideCalibMode 换流真机验证。

一次跑完四件事:
  1. 双摄 V4L2 能打开、出图 (§4 第 5 步)
  2. road/wide 帧时间戳、frameId 配对、稳定 ~20Hz (§4 第 6 步)
  3. 翻 WideCalibMode 后: 内参互换 + 备份生成 + 输出 msg/stream 打到对侧 (camerad 侧真机)
  4. 关回去: 内参还原 + 备份清理

会**临时**写入 FcamIntrinsics/EcamIntrinsics 测试值(因为当前是空的, 不写就没东西可换),
结束时按跑之前抓到的原值原样放回(原本没有就删掉), 不留痕。

用法: python3 tools/check_camerad_widecalib.py <树根> [每阶段秒数]
"""
import os
import signal
import subprocess
import sys
import threading
import time

TREE = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
PHASE = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0

sys.path.insert(0, TREE)
sys.path.insert(0, os.path.join(TREE, "openpilot"))

from openpilot.common.params import Params  # noqa: E402
import cereal.messaging as messaging  # noqa: E402

MSGS = ["narrowRoadCameraState", "wideRoadCameraState"]
# JSON 型 param: put 必须传 dict 且 block=True, 传字符串会 TypeError
FCAM_TEST = {"fl": 2520, "cx": 672.0, "cy": 380.0, "test": True}
ECAM_TEST = {"fl": 670, "cx": 672.0, "cy": 380.0, "test": True}

FAIL = 0


def check(cond, msg):
  global FAIL
  print(("  [PASS] " if cond else "  [FAIL] ") + msg)
  if not cond:
    FAIL = 1


def pstr(p, k):
  v = p.get(k)
  if isinstance(v, (bytes, bytearray)):
    return v.decode()
  return v


params = Params()
orig = {k: pstr(params, k) for k in
        ("FcamIntrinsics", "EcamIntrinsics", "WideCalibIntrinsicsFcamBackup",
         "WideCalibIntrinsicsEcamBackup", "WideCalibMode")}
print("[check] 跑之前的状态: " + ", ".join(f"{k}={'<空>' if v is None else v}" for k, v in orig.items()))
if orig["FcamIntrinsics"] is None:
  params.put("FcamIntrinsics", FCAM_TEST, block=True)
if orig["EcamIntrinsics"] is None:
  params.put("EcamIntrinsics", ECAM_TEST, block=True)
params.remove("WideCalibIntrinsicsFcamBackup")
params.remove("WideCalibIntrinsicsEcamBackup")
params.put_bool("WideCalibMode", False, block=True)

env = dict(os.environ)
env.update({"USE_WEBCAM": "1", "ROAD_CAM": "0", "WIDE_CAM": "1",
            "CAM_WIDTH": "1344", "CAM_HEIGHT": "760",
            "PYTHONPATH": f"{TREE}:{TREE}/openpilot"})


def other_camerads():
  """扫 /proc 找已在跑的 webcamerad (不依赖 pkill, 不会误伤自己)。"""
  out = []
  me = os.getpid()
  for pid in os.listdir("/proc"):
    if not pid.isdigit() or int(pid) == me:
      continue
    try:
      with open(f"/proc/{pid}/cmdline", "rb") as f:
        cmd = f.read().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
      continue
    if "webcam.camerad" in cmd:
      out.append((pid, cmd.strip()[:90]))
  return out


busy = other_camerads()
if busy:
  print("[check] 已经有 camerad 在跑, 会抢 /dev/videoN 和 msgq —— 先停掉再跑:")
  for pid, cmd in busy:
    print(f"    pid {pid}: {cmd}")
  print("    (精确 kill 这些 PID; 崩过的测试会留下孤儿)")
  sys.exit(2)

log_path = "/tmp/camerad_widecalib.log"
logf = open(log_path, "w")
print(f"[check] 起 camerad, 日志 -> {log_path}")
proc = subprocess.Popen([sys.executable, "-u", "-m", "openpilot.system.camerad.webcam.camerad"],
                        cwd=TREE, env=env, stdout=logf, stderr=subprocess.STDOUT)
log_lines = []


def _reader():
  with open(log_path, "r", errors="replace") as f:
    while True:
      line = f.readline()
      if not line:
        if proc.poll() is not None:
          return
        time.sleep(0.2)
        continue
      log_lines.append(line.rstrip())
      print("   |camerad| " + line.rstrip()[:160])


threading.Thread(target=_reader, daemon=True).start()


def measure(seconds, label):
  sm = messaging.SubMaster(MSGS)
  stats = {m: {"n": 0, "mono": [], "fid": [], "ts": []} for m in MSGS}
  t0 = time.monotonic()
  while time.monotonic() - t0 < seconds:
    sm.update(100)
    for m in MSGS:
      if sm.updated[m]:
        ev = sm[m]
        stats[m]["n"] += 1
        # 这个 fork: SubMaster 不把 logMonoTime 放进 FrameData, 走 sm.logMonoTime[topic]
        stats[m]["mono"].append(sm.logMonoTime[m])
        stats[m]["fid"].append(ev.frameId)
        stats[m]["ts"].append(ev.timestampEof)
  el = time.monotonic() - t0
  print(f"[check] {label} ({el:.1f}s):")
  for m in MSGS:
    s = stats[m]
    hz = s["n"] / el
    print(f"    {m}: {s['n']} 帧 ({hz:.1f} Hz)"
          + (f", frameId [{s['fid'][0]}..{s['fid'][-1]}], timestampEof 首帧 {s['ts'][0]}" if s["n"] else ""))
  del sm
  return stats, el


# ---------------- 阶段 1: 双摄出图 + 帧率
print()
print("[阶段1] 双摄出图 / 帧率 / 时间戳")
s1, el1 = measure(PHASE, "基线")
check(all(s1[m]["n"] > PHASE * 10 for m in MSGS), "两路都稳定出帧 (>10Hz 判定为真的在跑)")
for m in MSGS:
  hz = s1[m]["n"] / el1
  check(15.0 <= hz <= 25.0, f"{m} 帧率 {hz:.1f} Hz 落在 15~25 (目标 20Hz)")
  ids = s1[m]["mono"]
  check(all(b > a for a, b in zip(ids, ids[1:])), f"{m} logMonoTime 单调递增")
  fid = s1[m]["fid"]
  if m == MSGS[0]:
    # road 是 dense 发号方, 必须严格递增
    check(all(b > a for a, b in zip(fid, fid[1:])), f"{m} frameId 严格递增")
  else:
    # wide 在 decoupled 模式下"跟随 road 最新号", 允许复用同一号重发 → 只要求不回退
    check(all(b >= a for a, b in zip(fid, fid[1:])), f"{m} frameId 不回退 (允许重复)")
ts1 = s1[MSGS[0]]["ts"]
ts2 = s1[MSGS[1]]["ts"]
if ts1 and ts2:
  n = min(len(ts1), len(ts2))
  diffs = sorted(abs(ts1[i] - ts2[i]) / 1e6 for i in range(n))
  p50 = diffs[len(diffs) // 2]
  print(f"    road/wide timestampEof 差: p50 {p50:.2f} ms (max {diffs[-1]:.2f} ms)")
  check(p50 < 60.0, f"双路配对相位差 p50 {p50:.2f} ms 正常 (<60ms)")
f1 = s1[MSGS[0]]["fid"]
f2 = s1[MSGS[1]]["fid"]
if f1 and f2:
  steps = [b - a for a, b in zip(f2, f2[1:])]
  print(f"    wide frameId 步长集合: {sorted(set(steps))[:6]}"
        f" (含 0 = 复用 road 最新号重发, 是 decoupled 同步的设计行为)")
  check(all(s >= 0 for s in steps), "wide frameId 不回退")
  lo, hi = min(f1), max(f1)
  check(all(lo <= x <= hi for x in f2), "wide frameId 始终落在 road 已发号的区间内")

# ---------------- 阶段 2: 开 WideCalibMode
print()
print("[阶段2] 开 WideCalibMode (road/wide 换流)")
before = (pstr(params, "FcamIntrinsics"), pstr(params, "EcamIntrinsics"))
params.put_bool("WideCalibMode", True, block=True)
time.sleep(4.0)  # camerad 轮询周期 ~2s
after = (pstr(params, "FcamIntrinsics"), pstr(params, "EcamIntrinsics"))
bk_f = pstr(params, "WideCalibIntrinsicsFcamBackup")
bk_e = pstr(params, "WideCalibIntrinsicsEcamBackup")


def brief(v):
  if v is None:
    return "<空>"
  if isinstance(v, dict):
    return f"fl={v.get('fl')} cx={v.get('cx')} cy={v.get('cy')} 键={list((v.get('calibrations') or {}).keys())}"
  return str(v)[:60]


print(f"    Fcam: {brief(before[0])}")
print(f"      ->  {brief(after[0])}")
print(f"    Ecam: {brief(before[1])}")
print(f"      ->  {brief(after[1])}")
check(after[0] == before[1], "FcamIntrinsics 已换成原 Ecam 值")
check(after[1] == before[0], "EcamIntrinsics 已换成原 Fcam 值")
check(bk_f == before[0] and bk_e == before[1], "两份备份都写了原始值")
check(any("[WideCalib] Swap" in l for l in log_lines), "camerad 日志出现 '[WideCalib] Swap: ECAM -> road stream'")
s2, el2 = measure(PHASE, "换流后")
check(all(s2[m]["n"] > PHASE * 10 for m in MSGS), "换流后两路仍稳定出帧 (换的是输出身份, 不是停流)")

# ---------------- 阶段 3: 关 WideCalibMode
print()
print("[阶段3] 关 WideCalibMode (还原)")
params.put_bool("WideCalibMode", False, block=True)
time.sleep(4.0)
restored = (pstr(params, "FcamIntrinsics"), pstr(params, "EcamIntrinsics"))
print(f"    Fcam 还原为: {brief(restored[0])}")
print(f"    Ecam 还原为: {brief(restored[1])}")
check(restored[0] == before[0], "FcamIntrinsics 已还原")
check(restored[1] == before[1], "EcamIntrinsics 已还原")
check(pstr(params, "WideCalibIntrinsicsFcamBackup") is None, "Fcam 备份已清理")
check(pstr(params, "WideCalibIntrinsicsEcamBackup") is None, "Ecam 备份已清理")
check(any("[WideCalib] Intrinsics restored" in l for l in log_lines), "camerad 日志出现 'Intrinsics restored'")
s3, el3 = measure(PHASE / 2, "还原后")
check(s3[MSGS[0]]["n"] > 0, "还原后 road 流仍然出帧")

# ---------------- 收尾
print()
print("[check] 收尾: 停 camerad + 恢复跑之前的 param 状态")
proc.send_signal(signal.SIGINT)
try:
  proc.wait(timeout=10)
except subprocess.TimeoutExpired:
  proc.kill()
  proc.wait(timeout=5)
logf.close()
time.sleep(1)
for k, v in orig.items():
  if k == "WideCalibMode":
    params.put_bool(k, bool(v) if v else False, block=True)
  elif v is None:
    params.remove(k)
  else:
    params.put(k, v, block=True)
print("    param 已还原: " + ", ".join(f"{k}={'<空>' if v is None else '原值'}" for k, v in orig.items()))
print(f"    camerad 完整日志: {log_path}")

print()
print("相机 + 换流 自检: " + ("全部通过" if FAIL == 0 else "有 FAIL 项"))
sys.exit(FAIL)
