#!/usr/bin/env python3
"""相机颜色数值自检 (判据取自 相机链路_绿色与撕裂_20260929.md §2.2)。

不看画面靠猜, 直接抓一帧、走 camerad 的同一条 convert 路径拿到 NV12,
再算指标:

  指标            正确            错误(绿)
  Y.std            ~30 (有明暗)    ~7-10 (亮度被压平)
  U/V.mean         ~128-133        ~78-85
  U.std            小(3-13)        ~33 (=亮度std)
  G/R              ~0.9            ~3.1-3.7

G/R 只在读到零填充(4K 画布的空白区)时会飙到 3+; 字节序解错表现为 Y.std 被压平。

需要设备空闲(camerad 没在跑), 否则 REQBUFS 会 EBUSY。
用法: python3 tools/check_camera_color.py <树根>
"""
import os
import sys

TREE = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
sys.path.insert(0, TREE)
sys.path.insert(0, os.path.join(TREE, "openpilot"))
os.environ.setdefault("USE_WEBCAM", "1")
os.environ.setdefault("ROAD_CAM", "0")
os.environ.setdefault("WIDE_CAM", "1")

import numpy as np  # noqa: E402

from openpilot.system.camerad.webcam.v4l2_dmabuf_camera import Camera  # noqa: E402
from openpilot.system.camerad.webcam.camerad import CudaUyvyConverter  # noqa: E402

OUT_W, OUT_H = 1344, 760
ALIGNED_STRIDE, ALIGNED_Y_ROWS = 1408, 768

ok = True


def check(cond, msg):
  global ok
  print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
  if not cond:
    ok = False


def yuv_metrics(nv12: np.ndarray) -> dict:
  """NV12 -> Y/U/V 统计 + 由 YUV 还原 RGB 后的 G/R。

  注意: convert() 走 host resize 分支(源 != 输出尺寸)时输出的是**紧密** NV12
  (stride=W, rows=H); 补成 Venus 对齐布局是 _send_nv12 干的活。所以这里按紧密解。
  """
  y = nv12[:OUT_W * OUT_H].reshape(OUT_H, OUT_W).astype(np.float64)
  uv = nv12[OUT_W * OUT_H:].reshape(OUT_H // 2, OUT_W).astype(np.float64)
  u, v = uv[:, 0::2], uv[:, 1::2]
  # BT.601 还原 (与文档截图分析的算法一致)
  r = (y.reshape(OUT_H // 2, 2, OUT_W // 2, 2).mean(axis=(1, 3)) + 1.402 * (v - 128))
  g = (y.reshape(OUT_H // 2, 2, OUT_W // 2, 2).mean(axis=(1, 3)) - 0.344 * (u - 128) - 0.714 * (v - 128))
  b = (y.reshape(OUT_H // 2, 2, OUT_W // 2, 2).mean(axis=(1, 3)) + 1.772 * (u - 128))
  # G/R 只在像素共有的区域比较: 取 Y 的每个 2x2 块均值
  yb = y.reshape(OUT_H // 2, 2, OUT_W // 2, 2).mean(axis=(1, 3))
  gr = float((g.mean() / r.mean())) if r.mean() else float("nan")
  return {"Y_mean": float(yb.mean()), "Y_std": float(yb.std()),
          "U_mean": float(u.mean()), "U_std": float(u.std()),
          "V_mean": float(v.mean()), "G/R": gr,
          "R": float(r.mean()), "G": float(g.mean()), "B": float(b.mean())}


def grab(device, label):
  print(f"\n[{label}] 打开 {device} 抓帧...")
  cam = Camera("narrowRoadCameraState", 0, device)
  try:
    vc = cam.cam
    print(f"    画布: cam_w={vc.cam_w} cam_h={vc.cam_h} stride={vc.cam_bytesperline} "
          f"有效区={vc.cam_active_w}x{vc.cam_active_h} 格式={vc.cam_format_name} "
          f"实测bytesused={getattr(vc, '_expected_bytesused', None)}")
    it = cam.read_frames()
    vb = next(it)
    data = vb.data
    print(f"    帧: {len(data)} 字节 (= {len(data) / 2 / vc.cam_active_h:.0f} 字节/行 packed)")
    conv = CudaUyvyConverter(OUT_W, OUT_H, src_w=vc.cam_active_w, src_h=vc.cam_active_h,
                             dst_stride=ALIGNED_STRIDE, y_plane_rows=ALIGNED_Y_ROWS,
                             layout=(1 if str(vc.cam_format_name).upper() == 'UYVY' else 0),
                             src_stride=vc.cam_active_w * 2)
    nv12 = conv.convert(np.ascontiguousarray(np.frombuffer(data, dtype=np.uint8)))
    check(nv12 is not None, "convert 返回 NV12 (没崩、没失败)")
    if nv12 is None:
      return
    check(len(nv12) == OUT_W * OUT_H * 3 // 2,
          f"NV12 大小 {len(nv12)} = 紧密 {OUT_W}x{OUT_H} ({OUT_W * OUT_H * 3 // 2})")
    m = yuv_metrics(np.asarray(nv12))
    print(f"    Y={m['Y_mean']:.1f}(std {m['Y_std']:.1f})  U={m['U_mean']:.1f}(std {m['U_std']:.1f})  "
          f"V={m['V_mean']:.1f}  RGB=({m['R']:.1f},{m['G']:.1f},{m['B']:.1f})  G/R={m['G/R']:.2f}")
    check(m['Y_std'] > 20.0, f"Y.std={m['Y_std']:.1f} > 20 (亮度有明暗层次, 未被压平)")
    check(100.0 < m['U_mean'] < 160.0, f"U.mean={m['U_mean']:.1f} 贴中性 128 附近")
    check(m['U_std'] < 20.0, f"U.std={m['U_std']:.1f} < 20 (色度是 2x 下采样, 不该等于亮度 std)")
    check(m['G/R'] < 1.5, f"G/R={m['G/R']:.2f} < 1.5 (绿色暴增的指纹是 3+)")
    check(m['G'] < m['R'] * 1.6 and m['G'] < m['B'] * 1.6, "G 通道没有一家独大")
  finally:
    cam.cam.close()


grab("/dev/video0", "road / narrowRoadCameraState")
grab("/dev/video1", "wide / wideRoadCameraState")

print()
print("相机颜色自检: " + ("全部通过 (色彩判据达标)" if ok else "有 FAIL 项"))
sys.exit(0 if ok else 1)
