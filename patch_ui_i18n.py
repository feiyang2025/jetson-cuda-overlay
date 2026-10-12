#!/usr/bin/env python3
"""中文 UI 三件事一起装 (幂等, 可反复跑):

1. 字体回退: 装 OpenpilotCJK-Regular.fnt/.png (文泉驿正黑烘的中文字形图集) 并让
   system/ui/lib/application.py 在中文时回退到它。原实现回退到 KaiGenGothicKR-Bold.fnt(韩文字体),
   那张图集里汉字不全 -> 「态/电/通/轻/柔」被画成 "?"; 退回 unifont 虽然不缺字, 但 unifont 是
   16px 点阵字体, 中文看着像"数码管"。图集重建: kits/ui/i18n/bake_cjk_font.py。
2. 面板取值翻译: settings/usbgpu.py 的"值"原来直接返回英文 token(not detected/not connected/...),
   usbgpu_text.py 里的 tr_usbgpu_* 没人调用 -> 界面全英文。改成过 tr_usbgpu_*。
3. 日期行的星期: onroad/hud_renderer.py 里硬编码韩文星期(일월화수목금토), 中文界面下没有韩文字形
   -> 显示 "10-12(?)"。改成按界面语言取(中文 日一二三四五六, 韩文保持原样)。

外加把两个中文 .po 里"空 msgstr"补齐(上游留空 -> 界面显示英文/带 ? 的英文 FAQ):
   app_zh-CHS.po 4 条(数据/Firehose 页描述+说明、Reverse Gear), app_zh-CHT.po 219 条
   (用 zh-CHS 已有的译文经 OpenCC s2t 转繁体, 见 kits/ui/i18n/zh_cht_fill.json)。
只填空的, 上游已译好的一律不碰。

用法: python3 patch_ui_i18n.py <树根>
"""
import json
import re
import sys
from pathlib import Path

OVERLAY = Path(__file__).resolve().parent
KIT = OVERLAY / "kits" / "ui" / "i18n"

FONT_W = '  CJK = "OpenpilotCJK-Regular.fnt"\n'

APP_HELPERS = '''
# 只有含中日韩/泰文的文本才需要换字体; 纯拉丁/符号串(带 ✓ ° … 等)继续用 Inter 图集画。
def _needs_cjk_font(text: str) -> bool:
  return any(0x2E80 <= ord(c) <= 0x9FFF or 0xAC00 <= ord(c) <= 0xD7AF or 0x0E00 <= ord(c) <= 0x0E7F
             for c in text)


_CJK_FONT_BY_LANGUAGE = {
  "zh-CHS": FontWeight.CJK,
  "zh-CHT": FontWeight.CJK,
}


def font_fallback(font: rl.Font, text: str = "") -> rl.Font:
  """CJK 语言回退到中文字形图集(OpenpilotCJK), 再退 unifont, 最后才是 DISPLAY。"""
  if multilang.requires_unifont():
    if text and not _needs_cjk_font(text):
      return font
    fw = _CJK_FONT_BY_LANGUAGE.get(multilang.language)
    if fw is not None and gui_app.has_font(fw):
      return gui_app.font(fw)
    if gui_app.has_font(FontWeight.UNIFONT):
      return gui_app.font(FontWeight.UNIFONT)
    if gui_app.has_font(FontWeight.DISPLAY):
      return gui_app.font(FontWeight.DISPLAY)
  return font
'''

HUD_WEEKDAYS = '''WEEKDAYS_KO = ("일", "월", "화", "수", "목", "금", "토")
# 日期行 "10-12(금)" 的星期原为韩文硬编码: 中文界面下中文字体图集没有韩文字形 -> 画成 "?"。
# 按界面语言取(韩文仍作默认, 保持原样)。
WEEKDAYS_BY_LANGUAGE = {
  "zh-CHS": ("日", "一", "二", "三", "四", "五", "六"),
  "zh-CHT": ("日", "一", "二", "三", "四", "五", "六"),
  "en": ("SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT"),
}


def weekdays_for_ui_language() -> tuple[str, ...]:
  try:
    from openpilot.system.ui.lib.multilang import multilang as _ml
    return WEEKDAYS_BY_LANGUAGE.get(_ml.language, WEEKDAYS_KO)
  except Exception:
    return WEEKDAYS_KO
'''


def _read(p: Path) -> str:
  return p.read_text(encoding="utf-8")


def _find_tree(root: Path, *cands: str) -> Path | None:
  for c in cands:
    p = root / c
    if p.exists():
      return p
  return None


def install_font(root: Path) -> None:
  dst_dir = _find_tree(root, "openpilot/selfdrive/assets/fonts", "selfdrive/assets/fonts")
  if dst_dir is None:
    print("  [ui-i18n] 找不到 assets/fonts, 跳过字体")
    return
  for f in ("OpenpilotCJK-Regular.fnt", "OpenpilotCJK-Regular.png"):
    (dst_dir / f).write_bytes((KIT / f).read_bytes())
  print(f"  + 字体图集 -> {dst_dir}/OpenpilotCJK-Regular.{{fnt,png}}")


def patch_application(root: Path) -> None:
  p = _find_tree(root, "openpilot/system/ui/lib/application.py", "system/ui/lib/application.py")
  if p is None:
    print("  [ui-i18n] 找不到 application.py, 跳过")
    return
  s = _read(p)
  if "OpenpilotCJK" in s:
    print("  - application.py 已是 CJK 版, 不动")
    return

  # 1) 加 FontWeight.CJK
  m = re.search(r'^  DISPLAY = "(.*)"\n', s, re.M)
  if not m:
    print("  [ui-i18n] application.py 里没找到 DISPLAY 权重, 跳过")
    return
  s = s[:m.end()] + FONT_W + s[m.end():]

  # 2) 换掉整个 font_fallback (老的返回值是 DISPLAY, 有的一版没有 text 参数)
  m = re.search(r"^def font_fallback\(.*?\n(?=\n\nclass |\n\ndef |\Z)", s, re.M | re.S)
  if not m:
    print("  [ui-i18n] 没找到 font_fallback, 跳过")
    return
  s = s[:m.start()] + APP_HELPERS.lstrip("\n") + s[m.end():]

  # 3) 从源字体现场烘焙时也带上 CJK 码位
  s = re.sub(r"if font_weight in \(FontWeight\.DISPLAY, FontWeight\.UNIFONT[^)]*\):",
             "if font_weight in (FontWeight.DISPLAY, FontWeight.UNIFONT, FontWeight.CJK):", s)

  # 4) draw_text_ex 包装里把 text 传进回退函数(否则纯拉丁串也会被换成中文图集)
  s = s.replace("font = font_fallback(font)\n", "font = font_fallback(font, text)\n")
  p.write_text(s, encoding="utf-8")
  print("  + application.py: FontWeight.CJK + 中文回退 + 文本判据")


def patch_usbgpu(root: Path) -> None:
  p = _find_tree(root, "openpilot/selfdrive/ui/layouts/settings/usbgpu.py",
                "selfdrive/ui/layouts/settings/usbgpu.py")
  if p is None:
    print("  [ui-i18n] 找不到 settings/usbgpu.py, 跳过")
    return

  # usbgpu_text.py 有的树里没有(该面板却要 import 它) —— 缺了就装上, 否则界面导入即崩
  text_py = p.parent / "usbgpu_text.py"
  if not text_py.exists() and (KIT / "usbgpu_text.py").exists():
    text_py.write_text((KIT / "usbgpu_text.py").read_text(encoding="utf-8"), encoding="utf-8")
    print(f"  + 装上 usbgpu_text.py (原树缺这个文件, 面板会 import 它) -> {text_py}")

  s = _read(p)
  if "tr_usbgpu_status" in s:
    print("  - usbgpu.py 已接过翻译, 不动")
  else:
    s = s.replace("from openpilot.selfdrive.ui.ui_state import ui_state",
                  "from openpilot.selfdrive.ui.layouts.settings.usbgpu_text import tr_usbgpu_check, tr_usbgpu_status\n"
                  "from openpilot.selfdrive.ui.ui_state import ui_state", 1)
    s = s.replace('text_item(lambda: tr("eGPU Status"), self._status),',
                  'text_item(lambda: tr("eGPU Status"), lambda: tr_usbgpu_status(self._status())),')
    s = s.replace('text_item(lambda: tr("USB Link"), self._link),',
                  'text_item(lambda: tr("USB Link"), lambda: tr_usbgpu_status(self._link())),')
    s = s.replace('text_item(lambda: tr("Connection Check"), lambda: self._check_result),',
                  'text_item(lambda: tr("Connection Check"), lambda: tr_usbgpu_check(self._check_result)),')
    p.write_text(s, encoding="utf-8")
    print("  + usbgpu.py: 面板取值改走 tr_usbgpu_*(原来值是英文 token)")

  if text_py.exists():
    s = _read(text_py)
    if '"not connected"' not in s:
      s = s.replace('  tr_noop("USB not connected"),',
                    '  tr_noop("USB not connected"),\n  tr_noop("not connected"),\n'
                    '  tr_noop("not checked"),\n  tr_noop("checking..."),', 1)
      text_py.write_text(s, encoding="utf-8")
      print("  + usbgpu_text.py: 补 not connected / not checked / checking... 三个 token")


def patch_hud(root: Path) -> None:
  p = _find_tree(root, "openpilot/selfdrive/ui/onroad/hud_renderer.py", "selfdrive/ui/onroad/hud_renderer.py")
  if p is not None:
    s = _read(p)
    if "WEEKDAYS_BY_LANGUAGE" in s:
      print("  - hud_renderer.py(大UI) 星期已本地化, 不动")
    elif 'WEEKDAYS_KO = ("일", "월", "화", "수", "목", "금", "토")\n' in s:
      s = s.replace('WEEKDAYS_KO = ("일", "월", "화", "수", "목", "금", "토")\n', HUD_WEEKDAYS, 1)
      s = s.replace("weekday = WEEKDAYS_KO[(now.tm_wday + 1) % 7]",
                    "weekday = weekdays_for_ui_language()[(now.tm_wday + 1) % 7]")
      p.write_text(s, encoding="utf-8")
      print("  + hud_renderer.py(大UI): 星期改按界面语言")

  p = _find_tree(root, "openpilot/selfdrive/ui/mici/onroad/hud_renderer.py", "selfdrive/ui/mici/onroad/hud_renderer.py")
  if p is not None:
    s = _read(p)
    if "weekdays_for_ui_language" in s:
      print("  - hud_renderer.py(mici) 星期已本地化, 不动")
    elif 'weekdays_ko = ["일", "월", "화", "수", "목", "금", "토"]' in s:
      s = s.replace('      weekdays_ko = ["일", "월", "화", "수", "목", "금", "토"]\n',
                    '      from openpilot.selfdrive.ui.onroad.hud_renderer import weekdays_for_ui_language\n'
                    '      weekdays = weekdays_for_ui_language()\n', 1)
      s = s.replace("weekday = weekdays_ko[(now.weekday() + 1) % 7]",
                    "weekday = weekdays[(now.weekday() + 1) % 7]")
      p.write_text(s, encoding="utf-8")
      print("  + hud_renderer.py(mici): 星期改按界面语言")


def _unquote(lines: str) -> str:
  return "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', lines)).replace("\\n", "\n").replace('\\"', '"')


def _quote(text: str) -> list[str]:
  esc = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
  if len(esc) <= 72:
    return ['msgstr "%s"' % esc]
  chunks = [esc[i:i + 72] for i in range(0, len(esc), 72)]
  return ['msgstr "%s"' % chunks[0]] + ['"%s"' % c for c in chunks[1:]]


def fill_empty_msgstr(root: Path, po_rel: str, table_name: str) -> None:
  po = root / po_rel
  if not po.exists():
    print(f"  [ui-i18n] 找不到 {po_rel}, 跳过")
    return
  table = json.loads((KIT / table_name).read_text(encoding="utf-8"))
  blocks = _read(po).split("\n\n")
  filled = 0
  for i, blk in enumerate(blocks):
    mi = re.search(r'^msgid ((?:"(?:[^"\\]|\\.)*"\s*)+)', blk, re.M)
    ms = re.search(r'^msgstr ((?:"(?:[^"\\]|\\.)*"\s*)+)', blk, re.M)
    if not mi or not ms:
      continue
    mid = _unquote(mi.group(1))
    if not mid or mid not in table or _unquote(ms.group(1)).strip():
      continue
    lines = blk.split("\n")
    start = sum(1 for _ in lines[:ms.start()]) if False else blk[:ms.start()].count("\n")
    end = blk[:ms.end()].count("\n")
    blocks[i] = "\n".join(lines[:start] + _quote(table[mid]) + lines[end + 1:])
    filled += 1
  if filled:
    po.write_text("\n\n".join(blocks), encoding="utf-8")
  print(f"  + {po_rel.split('/')[-1]}: 补了 {filled} 条空 msgstr (想补 {len(table)})")


def main() -> int:
  root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
  print("[ui-i18n] 中文 UI 字体/翻译/星期 修复")
  install_font(root)
  patch_application(root)
  patch_usbgpu(root)
  patch_hud(root)
  fill_empty_msgstr(root, "openpilot/selfdrive/ui/translations/app_zh-CHS.po", "zh_chs_fill.json")
  fill_empty_msgstr(root, "openpilot/selfdrive/ui/translations/app_zh-CHT.po", "zh_cht_fill.json")
  return 0


if __name__ == "__main__":
  sys.exit(main())
