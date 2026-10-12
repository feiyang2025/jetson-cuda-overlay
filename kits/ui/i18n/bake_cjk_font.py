#!/usr/bin/env python3
"""烘一张只覆盖"本树中文 UI 真正用到的字"的 CJK 图集(体积可控), 然后接到 UI 的字体回退上。

码位集合 = 拉丁 + process.EXTRA_CHARS + 语言名 + app_zh-CHS.po/app_zh-CHT.po 里的所有字符。
(不烘全 BMP/Hangul —— 那个会得到 8192x4096 的图集, 加载又慢又占显存)
"""
import pathlib
import sys

FONT_DIR = pathlib.Path("/home/nvidia/公共/openpilot_cp_egpucp/openpilot/openpilot/selfdrive/assets/fonts")
sys.path.insert(0, str(FONT_DIR))
import process  # noqa: E402
import pyray as rl  # noqa: E402

FONT_SIZE = 48
PADDING = 2
STEM = "OpenpilotCJK-Regular"


def main() -> int:
  src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/wqy-zenhei-face0.ttf")

  base_cp, _ = process._char_sets()
  cps = set(base_cp)                       # base_cp 已经是 int 码位
  cps.update(ord(c) for c in map(chr, range(32, 127)))
  cps.update(ord(c) for c in process.EXTRA_CHARS)
  for code in ("zh-CHS", "zh-CHT"):
    po = process.TRANSLATIONS_DIR / f"app_{code}.po"
    cps.update(ord(c) for c in po.read_text(encoding="utf-8"))
  cps.update(ord(c) for c in "中文（简体）中文（繁體）")   # 语言名本身
  cps = tuple(sorted(cps))
  print(f"码位集合: {len(cps)}")

  data = src.read_bytes()
  buf = rl.ffi.new("unsigned char[]", data)
  cp_buffer = rl.ffi.new("int[]", cps)
  cp_ptr = rl.ffi.cast("int *", cp_buffer)
  glyph_count = rl.ffi.new("int *", len(cps))
  glyphs = rl.load_font_data(rl.ffi.cast("unsigned char *", buf), len(data), FONT_SIZE,
                             cp_ptr, len(cps), rl.FontType.FONT_DEFAULT, glyph_count)
  if glyphs == rl.ffi.NULL:
    raise RuntimeError("raylib 读不了这个字体(只吃 TrueType 轮廓, 不吃 .ttc/CFF)")
  print(f"烘出字形: {glyph_count[0]}")

  rects_ptr = rl.ffi.new("Rectangle **")
  image = rl.gen_image_font_atlas(glyphs, rects_ptr, glyph_count[0], FONT_SIZE, PADDING, 0)
  entries, line_height, base = process._glyph_metrics(glyphs, rects_ptr[0], glyph_count[0])
  atlas_name = f"{STEM}.png"
  if not rl.export_image(image, (FONT_DIR / atlas_name).as_posix()):
    raise RuntimeError("导出 png 失败")
  process._write_bmfont(FONT_DIR / f"{STEM}.fnt", FONT_SIZE, STEM, atlas_name,
                        line_height, base, (image.width, image.height), entries)
  print(f"图集 {image.width}x{image.height} -> {STEM}.fnt/.png")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
