#!/usr/bin/env python3
"""给 master-c3 的 zh-CHS 翻译注入"摄像头标定面板"的词条 (幂等)。

为什么不直接覆盖 app_zh-CHS.po: 上游会整份重写这个文件 (2272 行), 覆盖式 kit
会在下次同步时把上游的翻译一起打回去。所以只维护自己的一个带 MARKER 的尾块,
每次重写这个块(旧词条能被新版替换), 上游已有的 msgid 一律不碰。

译文用 sp(参考项目) `selfdrive/ui/translations/main_zh-CHS.ts` 里的官方用词,
面板源码的 msgid 也与 sp 的 qt developer_panel.cc 逐字对齐, 这样中文和参考项目一致:
  长焦标定 / 长焦在线标定 / 广角标定 / 摄像头标定参数

用法: python3 tools/patch_translations.py <树根>
"""
import re
import sys
from pathlib import Path

MARKER = "# ---- 摄像头标定面板 (overlay 移植, 见 calibration_panel.py) ----"
REF = "#: openpilot\\selfdrive\\ui\\sunnypilot\\calibration_panel.py"

# (msgid, msgstr) —— 前 4 组是 sp developer_panel.cc 原文, 译文取自 sp 的官方 zh-CHS
ENTRIES = [
  ("FCAM Calibration", "长焦标定"),
  ("RUN", "运行"),
  ("RUNNING...", "运行中..."),
  ("Analyze radar-vision lead distance, lane width, and IMU-model curvature to verify camera intrinsic calibration. Requires recent drive logs.",
   "分析雷达-视觉前车距离、车道宽度和IMU-模型曲率来验证摄像头内参标定。需要最近的驾驶日志。"),
  ("FCAM Live Calibration", "长焦在线标定"),
  ("Toggle ON before driving: collects radar-vision lead distance, lane width, and IMU-model curvature data during the drive. Auto-computes FCAM calibration on shutdown and saves result.",
   "驾驶前打开：行驶中采集雷达-视觉前车距离、车道宽度和IMU-模型曲率数据。openpilot关闭时自动计算长焦内参标定并保存结果。"),
  ("Wide Calibration", "广角标定"),
  ("Switch to wide-angle camera for ECAM calibration. START: resets extrinsic calibration, swaps ECAM to road stream, and runs live radar-vision data collection. Monitor collection status in the description (rv≥30 + lw≥10 + cv≥10 needed). STOP: computes ECAM intrinsics; if data is insufficient no save prompt will appear.",
   "切换到广角摄像头进行标定。开始：清除外参校准、将广角画面切换为主路摄像头，实时采集雷达-视觉数据。观察按钮下方采集状态（rv≥30 + lw≥10 + cv≥10 达标后停止）。停止：计算广角内参；数据不足则不弹出保存提示。"),
  ("Camera Calibration", "摄像头标定参数"),
  ("VIEW", "查看"),
  ("View current FCAM and ECAM intrinsic focal lengths, and extrinsic calibration (roll/pitch/yaw).",
   "查看当前长焦(FCAM)和广角(ECAM)的内参焦距，以及外参校准（侧倾角/俯仰角/横摆角）。"),
  ("START", "开始"),
  ("STOP", "停止"),
  ("SWITCHING...", "切换摄像头..."),
  ("COMPUTING...", "计算中..."),
  ("NO DATA", "无数据"),
  ("FAILED", "失败"),
  # 下面这些 sp 里没有, 是移植时新引入的状态串
  ("script not found", "未找到标定脚本"),
  ("calibration result", "标定结果"),
  ("current", "当前"),
  ("Update intrinsics? Extrinsic calibration will be reset.", "更新内参？外参校准会被清除。"),
  ("Update", "更新"),
  ("not calibrated", "未标定"),
  ("No calibration data", "无标定数据"),
  ("Extrinsics", "外参"),
  ("unreadable", "无法读取"),
]

PO_REL = "openpilot/selfdrive/ui/translations/app_zh-CHS.po"


def existing_msgids(text: str) -> set:
  out = set()
  for m in re.finditer(r'^msgid ((?:\"(?:[^\"\\\\]|\\\\.)*\"\s*)+)', text, re.M):
    out.add(''.join(re.findall(r'"((?:[^"\\\\]|\\\\.)*)"', m.group(1))))
  return out


def strip_block(text: str) -> str:
  """删掉上一次注入的尾块 (MARKER 到文件末尾), 让本轮可以整块重写。"""
  i = text.find(MARKER)
  if i < 0:
    return text
  return text[:i].rstrip("\n") + "\n"


def main():
  root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
  po = root / PO_REL
  if not po.exists():
    print("[patch_translations] 找不到 %s, 跳过" % po)
    return 0
  text = strip_block(po.read_text(encoding="utf-8"))
  have = existing_msgids(text)  # 上游自带的翻译优先, 不覆盖

  lines = ["", MARKER, ""]
  written = skipped = 0
  for msgid, msgstr in ENTRIES:
    if msgid in have:
      skipped += 1
      continue
    lines.append(REF)
    lines.append('msgid "%s"' % msgid.replace('"', '\\"'))
    lines.append('msgstr "%s"' % msgstr)
    lines.append("")
    written += 1

  if not text.endswith("\n"):
    text += "\n"
  po.write_text(text + "\n".join(lines) + "\n", encoding="utf-8")
  print("[patch_translations] zh-CHS: 写入 %d 条 (上游已有 %d 条不覆盖)" % (written, skipped))
  return 0


if __name__ == "__main__":
  sys.exit(main())
