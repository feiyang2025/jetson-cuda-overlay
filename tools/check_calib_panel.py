#!/usr/bin/env python3
"""标定面板自检 (需要 X/display: 会开一个窗口闪几秒)。

验证 4 个按钮能真的构建出来并跑几帧 update() —— 这是纯逻辑测试查不到的
(button_item/tr/gui_app 都依赖 raylib 窗口已 init)。

用法: DISPLAY=:0 XAUTHORITY=... python3 tools/check_calib_panel.py <树根>
"""
import os
import sys
import threading

TREE = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
sys.path.insert(0, TREE)
sys.path.insert(0, os.path.join(TREE, "openpilot"))

from openpilot.system.ui.lib.application import gui_app, FontWeight  # noqa: E402
from openpilot.system.ui.widgets import Widget  # noqa: E402
from openpilot.system.ui.widgets.scroller_tici import Scroller  # noqa: E402
from openpilot.selfdrive.ui.sunnypilot.calibration_panel import CalibrationPanel  # noqa: E402

FAIL = 0


def check(cond, msg):
  global FAIL
  print(("  [PASS] " if cond else "  [FAIL] ") + msg)
  if not cond:
    FAIL = 1


gui_app.init_window("calib-panel-selftest")

panel = CalibrationPanel()
check(len(panel.items) == 4, f"构建出 4 个标定项 ({list(panel.items.keys())})")
for key, item in panel.items.items():
  check(bool(item.title), f"{key}: 标题非空 -> {item.title}")

# 跑 10 帧: 按 DeveloperLayoutSP 的方式把标定项放进 Scroller, 每帧 update()+render
FRAMES = [0]


class Harness(Widget):
  def __init__(self, panel):
    super().__init__()
    self._panel = panel
    self._scroller = Scroller(list(panel.items.values()), line_separator=True, spacing=0)

  def _update_state(self):
    self._panel.update()
    self._scroller._update_state()
    FRAMES[0] += 1
    if FRAMES[0] >= 10:
      gui_app.request_close()

  def _render(self, rect):
    self._scroller.render(rect)


gui_app.push_widget(Harness(panel))
threading.Timer(20.0, gui_app.request_close).start()  # 兜底, 别挂死
try:
  # render() 是 generator: 真实 UI 在 main() 里 for 它
  for _ in gui_app.render():
    if FRAMES[0] >= 10:
      break
  check(FRAMES[0] >= 9, f"跑了 {FRAMES[0]} 帧 update()+render 无异常")
except Exception as e:
  check(False, f"render/update 抛异常: {e}")

# 按钮文字状态机: START/STOP/闪示
panel.items["fcam"].action_item.set_text("FLASH-TEST")
check(panel.items["fcam"].action_item.text == "FLASH-TEST", "action_item.set_text 生效")

print()
print("标定面板自检: " + ("全部通过" if FAIL == 0 else "有 FAIL 项"))
gui_app.close()
sys.exit(FAIL)
