#!/usr/bin/env python3
"""CalibSession.feed/ 的实数据自测 (用录制日志, 不需要车/相机)。

为什么单独测这一层: --scan 走的是"离线回放 + 直接调三个校准器"的老路径, 而 calibd 走的是
`CalibSession.feed()` 这条**实时**路径 (从 SubMaster 取 carState/radarState/modelV2)。
feed() 里任何字段名写错, 单元测试和 scan 都发现不了 —— 只有真数据能发现。

做法: 用 LogReader 读一段日志, 包一个最小的 SubMaster 壳 (updated/seen/[]), 逐条喂进去,
检查样本数、分方法样本数, 并打印一趟的融合结论。

用法: .venv/bin/python3 tools/selftest_calib_session.py <树根> [<日志段目录>]
"""
import os
import sys

TREE = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
sys.path.insert(0, TREE)
sys.path.insert(0, os.path.join(TREE, "tools", "calib"))
os.environ.setdefault("CAM_WIDTH", "1344")
os.environ.setdefault("CAM_HEIGHT", "760")

from openpilot.tools.lib.logreader import LogReader  # noqa: E402
import self_calibrator as SC  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'OK ' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


base = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/.comma/media/0/realdata")
segs = sorted(os.path.join(base, d) for d in os.listdir(base) if d and os.path.isdir(os.path.join(base, d)))
segs = [s for s in segs if os.path.exists(os.path.join(s, "rlog.zst")) or os.path.exists(os.path.join(s, "qlog.zst"))]
if not segs:
    print(f"找不到日志段 (base={base})")
    sys.exit(1)
seg = segs[-1]
print(f"日志段: {seg}")


class FakeSM:
    """最小 SubMaster 壳: 只提供 feed() 用到的 updated / seen / [] 三个接口。"""

    def __init__(self):
        self.data = {}
        self.updated = {}
        self.seen = {}

    def push(self, name, msg):
        self.data[name] = msg
        self.seen[name] = True
        self.updated = {k: (k == name) for k in ("carState", "radarState", "modelV2")}

    def __getitem__(self, name):
        return self.data[name]


wanted = {"carState", "radarState", "modelV2"}
sess = SC.CalibSession(ecam=False)
sm = FakeSM()
n_msgs = 0
# 数据可用性计数: 用来区分"管道没接上"和"这段日志本来就没有可用场景" (台架段就是后者)
n_lead = n_lane = n_moving = 0
for msg in LogReader(os.path.join(seg, "qlog.zst") if os.path.exists(os.path.join(seg, "qlog.zst")) else os.path.join(seg, "rlog.zst")):
    which = msg.which()
    if which in wanted:
        n_msgs += 1
        m = getattr(msg, which)
        if which == "modelV2":
            try:
                if m.leadsV3[0].prob > 0.5:
                    n_lead += 1
                probs = list(m.laneLineProbs)
                if len(probs) >= 3 and probs[1] > 0.5 and probs[2] > 0.5:
                    n_lane += 1
            except Exception:
                pass
        elif which == "carState":
            try:
                if m.vEgo > 1.0:
                    n_moving += 1
            except Exception:
                pass
        sm.push(which, m)
        sess.feed(sm)

rv, lw, cv = sess.counts
print(f"喂入 {n_msgs} 条相关消息, 帧数={sess.frames}; 场景可用性: 有前车 {n_lead} 帧 / "
      f"有双车道线 {n_lane} 帧 / 运动 {n_moving} 帧")
check("feed() 不抛异常", True)
usable = n_lead > 0 or n_lane > 0
if usable:
    check("有可用场景时确实产生了样本", sess.total > 0, f"rv={rv} lw={lw} cv={cv}")
else:
    print("  [跳过] 本段没有任何可用场景 (台架静止日志) -> 只验管道通畅, 不要求样本")
    check("无可用场景时样本为 0 (而不是乱记)", sess.total == 0, f"rv={rv} lw={lw} cv={cv}")
check("一帧一帧推进 (frames > 0)", sess.frames > 0, str(sess.frames))

result = sess.finish(write_params=False)
f = result.get("fused") or {}
print(f"融合结论: accepted={f.get('accepted')} fl_current={f.get('fl_current')} "
      f"fl_suggested={f.get('fl_suggested')} 原因={f.get('reason')}")
check("finish() 返回结构完整", all(k in result for k in ("fused", "methods", "timestamp", "camera")))
check("三个方法都在结果里", set(result["methods"]) == {"radar_vision_ratio", "lane_width", "curvature"})
check("未通过门限时 fl_suggested 为空 (不写参数)", f.get("accepted") or f.get("fl_suggested") is None,
      f"accepted={f.get('accepted')} suggested={f.get('fl_suggested')}")

print()
if FAILS:
    print(f"FAILED {len(FAILS)}: {FAILS}")
    sys.exit(1)
print("全部通过")
