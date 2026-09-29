#!/usr/bin/env python3
"""限速牌位置自检: 用纯几何算术验证 CarrotPanelSide 三档不会撞到别的元素。

背景(2026-09-29 实测): 限速牌画成 白圆+红环, 圆的半径是 (width+18)/2 —— 比它自己的
布局盒左右各多出 9px。MAX/设定车速面板在 rect.x+46..246(y 45..249); 车速大数字是
**水平居中**画的(speed_renderer: rect.x + rect.width/2, y=180, 字号 176)。
所以 CarrotPanelSide=0(左) 时限速牌 x=rect.x+60 → 圆横跨 rect.x+51..269, 正好压在
MAX 面板上 —— 这就是"红圈压住最大限速图标"的原因。

用法: python3 tools/check_sl_panel_pos.py
"""
import sys

SET_SPEED_WIDTH_METRIC = 200
SET_SPEED_HEIGHT = 204
SIGN_W = SET_SPEED_WIDTH_METRIC
SIGN_H = SET_SPEED_HEIGHT + 12
RADIUS = (SIGN_W + 18) / 2.0          # = 109
CIRCLE_PAD = RADIUS - SIGN_W / 2.0    # = 9  —— 圆比布局盒每侧多出的量
MAX_PANEL = (46, 246)                 # rect.x+46 .. rect.x+246
SPEED_FONT = 176
SPEED_DIGIT_W = 0.55 * SPEED_FONT     # 单字符近似宽度


def sign_x(side, rect_w):
  if side == 0:
    return 60.0
  if side == 1:
    return 60 + SIGN_W + 30 - 6
  x = rect_w * 0.30 - SIGN_W / 2.0                      # 中间
  return x if x >= 60 + SET_SPEED_WIDTH_METRIC + 10 + 8 else 60 + SIGN_W + 30 - 6


def overlaps(a, b):
  return not (a[1] <= b[0] or a[0] >= b[1])


def check(side, rect_w, ndigits):
  x = sign_x(side, rect_w)
  circ = (x - CIRCLE_PAD, x + SIGN_W + CIRCLE_PAD)
  ctr = rect_w / 2.0
  half = ndigits * SPEED_DIGIT_W / 2.0
  digits_box = (ctr - half, ctr + half)
  hit_max = overlaps(circ, MAX_PANEL)
  hit_spd = overlaps(circ, digits_box)
  tag = {0: "左", 1: "右", 2: "中间"}[side]
  ok = (not hit_max) and (not hit_spd)
  print(f"  [{'PASS' if ok else 'FAIL'}] side={tag} rect_w={rect_w:5.0f} 圆x=[{circ[0]:6.1f},{circ[1]:6.1f}] "
        f"撞MAX={hit_max} 撞车速数字={hit_spd} (数字{ndigits}位 x=[{digits_box[0]:.0f},{digits_box[1]:.0f}])")
  return ok


# 实际 rect 宽: BIG=1 时窗口约 1826px, 相机内容区约 1675px(实测自截图)
print("限速牌位置自检 (rect 宽 1675/1440; 数字 1 位与 3 位两种最宽):")
allok = True
for rw in (1675, 1440):
  for nd in (1, 3):
    allok &= check(2, rw, nd)              # 本次采用的"中间"档
print("  对照(旧行为 = 左档, 已知会压 MAX 面板):")
check(0, 1675, 3)
print("  窄窗极限(rect 只有 447px, 三元素放不下, 属已知无解):")
check(2, 447, 3)
print("\n结论: " + ("中间档在实际分辨率下既不压 MAX 面板也不压车速数字 ✓"
                 if allok else "仍有重叠, 需要再调"))
sys.exit(0 if allok else 1)
