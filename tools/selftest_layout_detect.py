#!/usr/bin/env python3
"""packed 4:2:2 字节序判据自测 (合成帧, 不需要相机)。

验的是 v4l2_dmabuf_camera._measure_packed_layout 的本事:
  1. 白天正常场景 (亮度 72±35, 色度 130±8) 判得对;
  2. **夜间近乎全黑** (亮度 22±14, 色度 130±11) 判得对 —— 这正是 2026-10-12 凌晨翻车的场景:
     旧判据"哪组 std 大就是亮度"在这里会错 (色度噪声方差 > 亮度方差) → 画面变绿;
  3. 极暗 + 色度噪声更大的最恶劣情况 (亮度 15±6, 色度 129±12) 判得对;
  4. 真值 UYVY 的数据能被判成 UYVY (对称性);
  5. 4K 虚标画布 (bytesused=16588800, 行距 7680, 其余零填充) 下也判得对 —— 零填充不得影响判据;
  6. 退化路径: 强制环境变量、以及"判据不明确 -> 回退 G_FMT/记忆值"的行为。

用法: .venv/bin/python3 tools/selftest_layout_detect.py <树根>
"""
import os
import sys

TREE = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
sys.path.insert(0, TREE)
sys.path.insert(0, os.path.join(TREE, "openpilot"))

import numpy as np  # noqa: E402

from openpilot.system.camerad.webcam.v4l2_dmabuf_camera import V4L2_PIX_FMT_UYVY, V4L2_PIX_FMT_YUYV, V4L2Camera  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'OK ' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


class FakeCam(V4L2Camera):
    """绕过 __init__/设备打开: 只为调用判据函数。"""

    def __init__(self, device="/dev/video0", bytesused=4147200, stride=3840,
                 gfmt=V4L2_PIX_FMT_YUYV, rows=1080):
        self.fd = None
        self.device = device
        self.cam_bytesperline = stride
        self.cam_h = rows
        self.cam_w = stride // 2
        self.cam_format_name = "YUYV" if gfmt == V4L2_PIX_FMT_YUYV else "UYVY"
        self._gfmt_pixelformat = gfmt

    def close(self):      # 不碰真设备
        pass

    def __del__(self):
        pass


def make_frame(order, y_mean, y_sd, c_mean, c_sd, h=1080, w=1920, stride=None, canvas=None, seed=7):
    """造一帧 packed 4:2:2 数据。

    真实 4:2:2 里色度是水平 2:1 下采样的, 所以色度样本要比亮度平滑: 这里用低频正弦
    生成色度、用带噪声的图案生成亮度, 逼近真实统计特征 (而不是纯高斯白噪声)。
    """
    rng = np.random.default_rng(seed)
    stride = stride or w * 2
    rows = canvas[1] if canvas else h
    cols = canvas[0] * 2 if canvas else stride
    buf = np.zeros((rows, cols), dtype=np.uint8)

    xs = np.arange(w)
    yy = (y_mean + y_sd * (0.6 * np.sin(xs / 37.0) + 0.4 * rng.standard_normal((h, w)))).clip(0, 255).astype(np.uint8)
    # 色度: 真实 4:2:2 里每 2 列才有一个样本, 且低频平滑 → (h, w/2)
    c_half = np.clip(c_mean + c_sd * np.sin(np.arange(w // 2) / 90.0), 0, 255).astype(np.uint8)
    u_half = np.repeat(c_half[None, :], h, axis=0)
    v_half = u_half

    if order == "YUYV":      # Y U Y V
        buf[:h, 0:2 * w:4] = yy[:, 0::2]
        buf[:h, 1:2 * w:4] = u_half
        buf[:h, 2:2 * w:4] = yy[:, 1::2]
        buf[:h, 3:2 * w:4] = v_half
    else:                    # U Y V Y
        buf[:h, 0:2 * w:4] = u_half
        buf[:h, 1:2 * w:4] = yy[:, 0::2]
        buf[:h, 2:2 * w:4] = v_half
        buf[:h, 3:2 * w:4] = yy[:, 1::2]
    return buf.tobytes(), stride


print("1) 白天场景 (亮度 72±35 / 色度 130±8)")
cam = FakeCam()
for truth in ("YUYV", "UYVY"):
    frame, stride = make_frame(truth, 72, 35, 130, 8)
    got = cam._measure_packed_layout(frame, 4147200)
    check(f"真值 {truth} -> 判定 {got[0] if got else None}", got and got[0] == truth,
          f"y_sd={got[1]:.1f}" if got else "None")

print("2) 夜间近乎全黑 (亮度 22±14 / 色度 130±11) —— 旧判据翻车的场景")
for truth in ("YUYV", "UYVY"):
    frame, stride = make_frame(truth, 22, 14, 130, 11)
    got = cam._measure_packed_layout(frame, 4147200)
    check(f"真值 {truth} -> 判定 {got[0] if got else None}", got and got[0] == truth,
          f"y_sd={got[1]:.1f}" if got else "None")

print("3) 最恶劣: 极暗低对比 (亮度 15±6) + 色度噪声比亮度方差还大 (色度 129±12)")
for truth in ("YUYV", "UYVY"):
    frame, stride = make_frame(truth, 15, 6, 129, 12)
    got = cam._measure_packed_layout(frame, 4147200)
    check(f"真值 {truth} -> 判定 {got[0] if got else None}", got and got[0] == truth,
          f"y_sd={got[1]:.1f}" if got else "None")

print("4) 4K 虚标画布 (bytesused=16588800, 行距 7680, 右侧+下方零填充)")
for truth in ("YUYV", "UYVY"):
    frame, stride = make_frame(truth, 60, 30, 130, 8, stride=7680, canvas=(3840, 2160))
    got = cam._measure_packed_layout(frame, 16588800)
    check(f"真值 {truth} -> 判定 {got[0] if got else None}", got and got[0] == truth,
          f"y_sd={got[1]:.1f}" if got else "None")

print("5) 强制开关 (CAM_PACKED_LAYOUT / CAM_YUV_LAYOUT)")
frame, _ = make_frame("YUYV", 72, 35, 130, 8)
for env_name in ("CAM_PACKED_LAYOUT", "CAM_YUV_LAYOUT"):
    os.environ[env_name] = "UYVY"
    got = cam._measure_packed_layout(frame, 4147200)
    check(f"{env_name}=UYVY 覆盖实测", got and got[0] == "UYVY", str(got and got[0]))
    del os.environ[env_name]

print("6) 判据不明确时的回退 (色度对分与亮度对分几乎相同, 且都很差)")
# 全灰帧: 两组分数都接近 0/很小 -> 不明确
flat = bytes([128]) * (4147200)
got = cam._measure_packed_layout(flat, 4147200)
check("全灰帧 -> 回退到 G_FMT(YUYV) 而不是乱猜", got and got[0] == "YUYV", str(got and got[0]))
cam2 = FakeCam(gfmt=V4L2_PIX_FMT_UYVY)
got2 = cam2._measure_packed_layout(flat, 4147200)
check("全灰帧 -> 回退跟随 G_FMT(UYVY)", got2 and got2[0] == "UYVY", str(got2 and got2[0]))

print()
if FAILS:
    print(f"FAILED {len(FAILS)}: {FAILS}")
    sys.exit(1)
print("全部通过")
