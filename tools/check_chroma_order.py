#!/usr/bin/env python3
"""判断 twgmsl packed 帧的 U/V 字节序: 对**同一帧**跑两种色度顺序, 比哪个更像真实场景。

为什么需要它: 均值类判据(Y.std / U.mean / G/R)对 **U/V 互换是瞎的** —— U 和 V 的均值
都贴 128, 互换前后统计一模一样。必须用**色相可信度**来判:
  · 夜间室内真实场景里, 最亮的像素(灯/白纸/白墙)应当偏暖或中性 (R >= B);
    明显 "G 一家独大" 不自然 —— 那正是 U/V 互换把暖白变绿的特征;
  · 蓝→紫、橘黄→绿 是 U/V 互换的教科书症状(BT.601 下橙色 (255,165,0) 互换后算出
    (54,209,255) 青绿; 蓝色 (0,0,255) 互换后偏品红)。

用法: python3 tools/check_chroma_order.py <树根>
需要设备空闲(camerad 没在跑), 否则 REQBUFS 会 EBUSY。
"""
import os
import sys

TREE = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
sys.path.insert(0, TREE)
sys.path.insert(0, os.path.join(TREE, "openpilot"))
os.environ.setdefault("USE_WEBCAM", "1")
os.environ.setdefault("ROAD_CAM", "0")
os.environ.setdefault("WIDE_CAM", "1")
os.environ.setdefault("GMSL_CHROMA_LAYOUT", "twgmsl")

import numpy as np  # noqa: E402

from openpilot.system.camerad.webcam.v4l2_dmabuf_camera import Camera  # noqa: E402
from openpilot.system.camerad.webcam.camerad import CudaUyvyConverter  # noqa: E402

OUT_W, OUT_H = 1344, 760
ALIGNED_STRIDE, ALIGNED_Y_ROWS = 1408, 768


def stats(nv12, tag):
  """紧密 NV12 -> 亮部色相可信度 + 最饱和像素均值。"""
  a = np.asarray(nv12, dtype=np.uint8)
  y = a[:OUT_W * OUT_H].reshape(OUT_H, OUT_W).astype(np.float64)
  uv = a[OUT_W * OUT_H:].reshape(OUT_H // 2, OUT_W).astype(np.float64)
  u, v = uv[:, 0::2], uv[:, 1::2]
  yb = y.reshape(OUT_H // 2, 2, OUT_W // 2, 2).mean(axis=(1, 3))
  r = yb + 1.402 * (v - 128)
  g = yb - 0.344 * (u - 128) - 0.714 * (v - 128)
  b = yb + 1.772 * (u - 128)
  rgb = np.clip(np.stack([r, g, b], -1), 0, 255)
  lum = yb
  bright = lum >= np.percentile(lum, 90)
  br = rgb[bright]
  g_dom = float(((br[..., 1] > br[..., 0] + 6) & (br[..., 1] > br[..., 2] + 6)).mean())
  warm = float(((br[..., 0] > br[..., 2] + 6) & (br[..., 0] >= br[..., 1])).mean())
  sat = rgb.max(-1) - rgb.min(-1)
  s = rgb[sat >= np.percentile(sat, 98)].mean(0) if (sat >= np.percentile(sat, 98)).any() else rgb.mean(0)
  print(f"  [{tag}] 亮部绿主导 {g_dom * 100:5.1f}%  暖(R>B) {warm * 100:5.1f}%  "
        f"最饱和2% RGB=({s[0]:5.1f},{s[1]:5.1f},{s[2]:5.1f})  全图 RGB=({rgb[..., 0].mean():5.1f},"
        f"{rgb[..., 1].mean():5.1f},{rgb[..., 2].mean():5.1f})")
  return g_dom


def grab(device, label):
  print(f"\n[{label}] {device}")
  cam = Camera("narrowRoadCameraState", 0, device)
  try:
    vc = cam.cam
    conv = CudaUyvyConverter(OUT_W, OUT_H, src_w=vc.cam_active_w, src_h=vc.cam_active_h,
                             dst_stride=ALIGNED_STRIDE, y_plane_rows=ALIGNED_Y_ROWS,
                             layout=(1 if str(vc.cam_format_name).upper() == 'UYVY' else 0),
                             src_stride=vc.cam_active_w * 2)
    it = cam.read_frames()
    data = next(it).data
    src = np.ascontiguousarray(np.frombuffer(data, dtype=np.uint8))
    gd = {}
    for swap in (0, 1):
      # 两条路都要覆盖: resize 路取 self.chroma_swap, 普通路取 .so 里的全局开关
      conv.chroma_swap = bool(swap)
      conv._packed.packed_to_nv12_set_chroma_swap(swap)
      nv12 = conv.convert(src)
      gd[swap] = stats(nv12, f"chroma_swap={swap}")
    best = 0 if gd[0] < gd[1] else 1
    print(f"  → 亮部绿主导更少的一侧更像真实场景: chroma_swap={best}")
    return best
  finally:
    cam.cam.close()


verdicts = [grab("/dev/video0", "road / dev/video0"), grab("/dev/video1", "wide / dev/video1")]
print("\n色度顺序判定: " + ("两路一致 → chroma_swap=1 (当前 U/V 是互换的, 需要打开)"
                       if verdicts == [1, 1] else
                       "两路一致 → chroma_swap=0 (当前顺序已经对, 不要开)"
                       if verdicts == [0, 0] else
                       f"两路不一致 {verdicts} —— 需要分别判断, 别盲目全局开"))
