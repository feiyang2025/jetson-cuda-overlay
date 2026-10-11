#!/usr/bin/env python3
"""统计日志段里"有没有可用场景" (体检包第 6 节 --deep 用)。

对每个段打印: 有前车帧数 / 双车道线帧数 / 运动帧数 —— 标定三个方法分别吃这些场景,
全是 0 就说明这段是被标定流程判定为"没数据"的 (台架停车段就是如此)。
"""
import os
import sys

from openpilot.tools.lib.logreader import LogReader


def scan(d: str) -> str:
    f = os.path.join(d, "qlog.zst")
    if not os.path.exists(f):
        f = os.path.join(d, "rlog.zst")
    if not os.path.exists(f):
        return f"  {os.path.basename(d)}: (无 qlog/rlog)"
    lead = lane = mov = 0
    for m in LogReader(f):
        w = m.which()
        if w == "modelV2":
            try:
                if m.modelV2.leadsV3[0].prob > 0.5:
                    lead += 1
                pr = list(m.modelV2.laneLineProbs)
                if len(pr) >= 3 and pr[1] > 0.5 and pr[2] > 0.5:
                    lane += 1
            except Exception:
                pass
        elif w == "carState":
            try:
                if m.carState.vEgo > 1.0:
                    mov += 1
            except Exception:
                pass
    return f"  {os.path.basename(d)}: 有前车 {lead} 帧 / 双车道线 {lane} 帧 / 运动 {mov} 帧"


for d in sys.argv[1:]:
    print(scan(d))
