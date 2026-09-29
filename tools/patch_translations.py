#!/usr/bin/env python3
"""给 master-c3 的 zh-CHS 翻译注入"摄像头标定面板"的词条 (幂等)。

为什么不直接覆盖 app_zh-CHS.po: 上游会整份重写这个文件 (2272 行), 覆盖式 kit
会在下次同步时把上游的翻译一起打回去。所以按 msgid 逐个补, 已有的不碰。

用法: python3 tools/patch_translations.py <树根>
"""
import re
import sys
from pathlib import Path

MARKER = "# ---- 摄像头标定面板 (overlay 移植, 见 calibration_panel.py) ----"
REF = "#: openpilot\\selfdrive\\ui\\sunnypilot\\calibration_panel.py"

ENTRIES = [
  ("FCAM Calibration", "FCAM 标定"),
  ("RUN", "运行"),
  ("RUNNING...", "运行中…"),
  ("Analyze radar-vision lead distance, lane width, and IMU-model curvature to verify camera intrinsic calibration. Requires recent drive logs.",
   "通过雷达视觉前车距离、车道宽度与 IMU 模型曲率校验相机内参标定。需要近期的行车日志。"),
  ("FCAM Live Calibration", "FCAM 实时标定"),
  ("Toggle ON before driving: collects radar-vision lead distance, lane width and IMU-model curvature during the drive. Computes FCAM calibration on shutdown and saves the result.",
   "行车前打开：行车中持续采集雷达视觉前车距离、车道宽度与 IMU 模型曲率样本，退出时计算并保存 FCAM 标定结果。"),
  ("Wide Calibration", "宽角标定"),
  ("Switch the wide-angle camera onto the road stream for ECAM calibration. START resets extrinsic calibration and starts live sample collection (needs rv>=30 + lw>=10 + cv>=10). STOP computes ECAM intrinsics; if data is insufficient nothing is saved.",
   "把宽角相机切到 road 流做 ECAM 标定。START 会清空外参标定并开始实时采样本（需要 rv>=30、lw>=10、cv>=10）；STOP 计算 ECAM 内参，样本不足则不保存。"),
  ("Camera Calibration", "相机标定"),
  ("View current FCAM and ECAM intrinsic focal lengths and the extrinsic calibration (roll/pitch/yaw).",
   "查看当前 FCAM / ECAM 内参焦距与外参标定（横滚/俯仰/偏航）。"),
  ("START", "开始"),
  ("STOP", "停止"),
  ("SWITCHING...", "切换中…"),
  ("COMPUTING...", "计算中…"),
  ("NO DATA", "无数据"),
  ("FAILED", "失败"),
  ("script not found", "未找到标定脚本"),
  ("calibration result", "标定结果"),
  ("current", "当前"),
  ("Update intrinsics? Extrinsic calibration will be reset.", "更新内参？外参标定会被清除。"),
  ("Update", "更新"),
  ("not calibrated", "未标定"),
  ("No calibration data", "无标定数据"),
  ("Extrinsics", "外参"),
  ("unreadable", "无法读取"),
]

PO_REL = "openpilot/selfdrive/ui/translations/app_zh-CHS.po"


def existing_msgids(text: str) -> set:
  out = set()
  for m in re.finditer(r'^msgid ((?:"(?:[^"\\]|\\.)*"\s*)+)', text, re.M):
    out.add(''.join(re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))))
  return out


def main():
  root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
  po = root / PO_REL
  if not po.exists():
    print("[patch_translations] 找不到 %s, 跳过" % po)
    return 0
  text = po.read_text(encoding="utf-8")
  have = existing_msgids(text)

  lines = ["", MARKER, ""]
  added = 0
  for msgid, msgstr in ENTRIES:
    if msgid in have:
      continue
    lines.append(REF)
    lines.append('msgid "%s"' % msgid.replace('"', '\\"'))
    lines.append('msgstr "%s"' % msgstr)
    lines.append("")
    added += 1

  if added == 0:
    print("[patch_translations] zh-CHS: 已就绪 (0 条新增, 共 %d 条已有翻译)" % len(have))
    return 0

  if not text.endswith("\n"):
    text += "\n"
  po.write_text(text + "\n".join(lines) + "\n", encoding="utf-8")
  print("[patch_translations] zh-CHS: 新增 %d 条标定面板翻译" % added)
  return 0


if __name__ == "__main__":
  sys.exit(main())
