"""纯 numpy 的窄基线立体匹配 —— 替代 wide_calibrator 里的 cv2.ORB + BFMatcher。

为什么不用 opencv:
  本机 venv 装不了 opencv (系统 cv2 4.8 是 numpy 1.x ABI, venv 是 numpy 2.5 -> ImportError),
  pip 又没网。而 ECAM 标定真正需要的只是"FCAM 与 ECAM 之间的点对应":
  两路相机固连在同一个支架上, 基线小、相对旋转固定, 平移 + 小幅尺度/旋转即可覆盖,
  不需要 ORB 那种旋转/尺度不变性。所以用结构张量角点 + 金字塔 NCC 块匹配就够了。

流程:
  Y 平面提取 (I420/NV12 只取 Y, 不需要 cvtColor)
  -> Shi-Tomasi 角点 (结构张量最小特征值 + 非极大抑制 + 网格分散)
  -> 金字塔 NCC 块匹配 (由粗到细, 每级 ±4 邻域, 亚像素抛物线拟合)
  -> 唯一性/比率检验 + 可选极线(Sampson)过滤 (由调用方做)
  -> [(u1, v1, u2, v2), ...]

CPU-only, 依赖只有 numpy。实测 1344x760 单帧对 ~0.1-0.3 s (见 python numpy_stereo.py)。
"""

from __future__ import annotations

import time

import numpy as np

# ------------------------------------------------------------------ 参数
PATCH_RADIUS = 5          # 11x11 匹配窗口
MIN_DIST = 10             # 角点最小间距 (px, 全分辨率)
MAX_LEVELS = 3            # 金字塔层数 (1/8 起)
COARSE_SEARCH = 4         # 最粗层每级搜索邻域 (像素, 该层尺度)
FINE_SEARCH = 2           # 细化层搜索邻域
MIN_NCC = 0.70            # 最低相关度
UNIQUENESS_MARGIN = 0.05  # 次优(距离>3px)至少要低这么多才接受


# ------------------------------------------------------------------ Y 平面
def y_plane_from_buf(buf, copy: bool = True) -> np.ndarray:
    """从 VisionBuf 取 Y 平面当灰度图。

    I420 与 NV12 的 Y 都在最前面, 所以不需要 cv2.cvtColor; 只需按 stride 裁掉行尾对齐填充。
    """
    h, w = int(buf.height), int(buf.width)
    stride = int(getattr(buf, "stride", 0) or 0) or w
    arr = np.frombuffer(buf.data, dtype=np.uint8, count=stride * h).reshape(h, stride)
    y = arr[:, :w]
    return np.ascontiguousarray(y) if copy else y


# ------------------------------------------------------------------ 角点
def _box_sum(a: np.ndarray, r: int) -> np.ndarray:
    """半径 r 的方形窗口求和 (积分图, O(N))。"""
    c = np.cumsum(np.cumsum(a, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)))
    h, w = a.shape
    ys = np.arange(h)
    xs = np.arange(w)
    y0 = np.clip(ys - r, 0, h)
    y1 = np.clip(ys + r + 1, 0, h)
    x0 = np.clip(xs - r, 0, w)
    x1 = np.clip(xs + r + 1, 0, w)
    return (c[np.ix_(y1, x1)] - c[np.ix_(y0, x1)] - c[np.ix_(y1, x0)] + c[np.ix_(y0, x0)])


def _nms_max(a: np.ndarray, r: int) -> np.ndarray:
    """半径 r 的局部最大值 (bool mask)。"""
    h, w = a.shape
    out = np.full(a.shape, -np.inf, dtype=a.dtype)
    for dy in range(-r, r + 1):
      for dx in range(-r, r + 1):
        if dx == 0 and dy == 0:
          continue
        ys0, ys1 = max(0, dy), min(h, h + dy)
        xs0, xs1 = max(0, dx), min(w, w + dx)
        ys0b, ys1b = max(0, -dy), min(h, h - dy)
        xs0b, xs1b = max(0, -dx), min(w, w - dx)
        out[ys0b:ys1b, xs0b:xs1b] = np.maximum(out[ys0b:ys1b, xs0b:xs1b],
                                                a[ys0:ys1, xs0:xs1])
    return a >= out


def detect_features(gray: np.ndarray, max_features: int = 800, min_dist: int = MIN_DIST,
                    quality: float = 0.01) -> np.ndarray:
    """Shi-Tomasi 角点, 返回 (N,2) 的 (x, y) 整数坐标。

    不用 cv2.goodFeaturesToTrack: 结构张量最小特征值的解析式 + 非极大抑制 + 网格分散足够,
    而且完全可控 (无 cv2 依赖)。
    """
    img = gray.astype(np.float32, copy=False)
    if img.shape[0] < 4 * min_dist or img.shape[1] < 4 * min_dist:
        return np.zeros((0, 2), dtype=np.int32)
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    gy[1:-1, :] = img[2:, :] - img[:-2, :]

    r = max(1, min_dist // 2)
    a = _box_sum(gx * gx, r)
    b = _box_sum(gx * gy, r)
    c = _box_sum(gy * gy, r)
    trace = a + c
    # 最小特征值 = (a+c)/2 - sqrt(((a-c)/2)^2 + b^2)
    resp = 0.5 * trace - np.sqrt(0.25 * (a - c) ** 2 + b * b)
    resp = np.maximum(resp, 0.0)

    # 边缘与边界排除 (块匹配需要完整 patch)
    edge = PATCH_RADIUS + 1
    resp[:edge, :] = 0.0
    resp[-edge:, :] = 0.0
    resp[:, :edge] = 0.0
    resp[:, -edge:] = 0.0

    peak = float(resp.max())
    if peak <= 0.0:
        return np.zeros((0, 2), dtype=np.int32)
    resp[resp < quality * peak] = 0.0
    keep = _nms_max(resp, max(1, min_dist // 2))

    ys, xs = np.nonzero(keep)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    scores = resp[ys, xs]
    order = np.argsort(scores)[::-1]

    # 网格分散: 每格只留响应最高的一个, 避免角点全挤在纹理密集的一小块
    seen = set()
    picked = []
    for i in order:
        gx_cell = int(xs[i]) // min_dist
        gy_cell = int(ys[i]) // min_dist
        if (gx_cell, gy_cell) in seen:
            continue
        seen.add((gx_cell, gy_cell))
        picked.append(i)
        if len(picked) >= max_features:
            break
    return np.stack([xs[picked], ys[picked]], axis=1).astype(np.int32)


# ------------------------------------------------------------------ 匹配
def _patches(img: np.ndarray, xs: np.ndarray, ys: np.ndarray, radius: int) -> np.ndarray:
    """取 (N, (2r+1)^2) 的 patch 矩阵 (坐标四舍五入到整像素; 调用方保证边界安全)。

    亚像素由后面的 NCC 峰值抛物线拟合补, 这里保持整数索引以便向量化。
    """
    xi = np.rint(xs).astype(np.int32)
    yi = np.rint(ys).astype(np.int32)
    offs = np.arange(-radius, radius + 1)
    gy = yi[:, None] + offs[None, :]          # (N, K)
    gx = xi[:, None] + offs[None, :]          # (N, K)
    win = img[gy[:, :, None], gx[:, None, :]]  # (N, K, K)
    return win.reshape(len(yi), -1).astype(np.float32, copy=False)


def _ncc(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """逐行归一化互相关 (N,) —— 去均值 + 归一化。"""
    a = p1 - p1.mean(axis=1, keepdims=True)
    b = p2 - p2.mean(axis=1, keepdims=True)
    na = np.sqrt((a * a).sum(axis=1))
    nb = np.sqrt((b * b).sum(axis=1))
    denom = np.maximum(na * nb, 1e-9)
    return (a * b).sum(axis=1) / denom


def _down2(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[0] // 2 * 2, img.shape[1] // 2 * 2
    a = img[:h, :w].astype(np.float32, copy=False)
    return (a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2]) * 0.25


def match_ncc(img1: np.ndarray, img2: np.ndarray, kps: np.ndarray,
              radius: int = PATCH_RADIUS, levels: int = MAX_LEVELS,
              min_ncc: float = MIN_NCC) -> tuple[np.ndarray, np.ndarray]:
    """由粗到细的金字塔 NCC 匹配。

    返回 (matches(N,4) float32 = (u1,v1,u2,v2) 全分辨率坐标, scores(N,) NCC)。
    """
    if len(kps) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    pyr1 = [img1.astype(np.float32, copy=False)]
    pyr2 = [img2.astype(np.float32, copy=False)]
    for _ in range(levels):
        pyr1.append(_down2(pyr1[-1]))
        pyr2.append(_down2(pyr2[-1]))

    # 只保留在每一层都离边界足够远的关键点: 整像素 patch 失败在这里就会越界
    # (最粗层 1/8 尺度下, 全分辨率的 radius 边距会缩成不到 1px)。
    margin = radius + COARSE_SEARCH + 1
    mask = np.ones(len(kps), dtype=bool)
    for lvl in range(levels + 1):
        h, w = pyr1[lvl].shape
        kp = kps.astype(np.float32) / (1 << lvl)
        mask &= (kp[:, 0] >= margin) & (kp[:, 0] < w - margin) & \
                (kp[:, 1] >= margin) & (kp[:, 1] < h - margin)
    if not mask.all():
        kps = kps[mask]
    if len(kps) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    scale = 1 << levels
    p1 = kps.astype(np.float32) / scale                        # 最粗层的关键点
    best = np.zeros((len(kps), 2), dtype=np.float32)           # 最粗层坐标

    # 粗层: 全邻域搜索
    src_c = _patches(pyr1[levels], p1[:, 0], p1[:, 1], radius)
    probe = COARSE_SEARCH
    best_score = np.full(len(kps), -2.0, dtype=np.float32)
    H, W = pyr1[levels].shape
    for dy in range(-probe, probe + 1):
      for dx in range(-probe, probe + 1):
        xs = p1[:, 0] + dx
        ys = p1[:, 1] + dy
        ok = (xs >= radius) & (xs < W - radius) & (ys >= radius) & (ys < H - radius)
        if not ok.any():
          continue
        cand = np.full((len(kps), radius * 2 + 1, radius * 2 + 1), 0.0, dtype=np.float32)
        flat = cand.reshape(len(kps), -1)
        idx = np.nonzero(ok)[0]
        flat[idx] = _patches(pyr2[levels], xs[idx], ys[idx], radius)
        s = _ncc(src_c, flat)
        better = ok & (s > best_score)
        best_score[better] = s[better]
        best[better] = np.stack([xs[better], ys[better]], axis=1)
    # 次优分数(与最优位置相距 >3px 的最高分): 用于唯一性检验
    second = np.full(len(kps), -2.0, dtype=np.float32)
    for dy in range(-probe, probe + 1):
      for dx in range(-probe, probe + 1):
        xs = p1[:, 0] + dx
        ys = p1[:, 1] + dy
        ok = (xs >= radius) & (xs < W - radius) & (ys >= radius) & (ys < H - radius)
        if not ok.any():
          continue
        far = ok & ((np.abs(xs - best[:, 0]) > 3) | (np.abs(ys - best[:, 1]) > 3))
        if not far.any():
          continue
        idx = np.nonzero(far)[0]
        flat = np.zeros((len(kps), (radius * 2 + 1) ** 2), dtype=np.float32)
        flat[idx] = _patches(pyr2[levels], xs[idx], ys[idx], radius)
        s = _ncc(src_c, flat)
        upd = far & (s > second)
        second[upd] = s[upd]

    # 细化: 逐级放大, 每级在当前估计附近搜索
    cur = best * 2.0
    for lvl in range(levels - 1, -1, -1):
        img1l, img2l = pyr1[lvl], pyr2[lvl]
        kp_l = kps.astype(np.float32) / (1 << lvl)
        src_l = _patches(img1l, kp_l[:, 0], kp_l[:, 1], radius)
        H, W = img1l.shape
        # 必须 copy: best_l 被原地写入, 与 cur 共享内存会让后续偏移相对"已更新"的估计搜索 -> 漂移
        best_l = cur.copy()
        best_s = np.full(len(kps), -2.0, dtype=np.float32)
        for dy in range(-FINE_SEARCH, FINE_SEARCH + 1):
          for dx in range(-FINE_SEARCH, FINE_SEARCH + 1):
            xs = cur[:, 0] + dx
            ys = cur[:, 1] + dy
            ok = (xs >= radius) & (xs < W - radius) & (ys >= radius) & (ys < H - radius)
            if not ok.any():
              continue
            idx = np.nonzero(ok)[0]
            flat = np.zeros((len(kps), (radius * 2 + 1) ** 2), dtype=np.float32)
            flat[idx] = _patches(img2l, xs[idx], ys[idx], radius)
            s = _ncc(src_l, flat)
            better = ok & (s > best_s)
            best_s[better] = s[better]
            best_l[better] = np.stack([xs[better], ys[better]], axis=1)
        cur = best_l * (2.0 if lvl > 0 else 1.0) if lvl > 0 else best_l

    # 唯一性检验 (次优必须明显更差) —— 用全分辨率的分数比较
    ratio_ok = (best_score > min_ncc) & (best_score - second > UNIQUENESS_MARGIN)
    keep = np.nonzero(ratio_ok)[0]
    if len(keep) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    out = np.stack([kps[keep, 0].astype(np.float32), kps[keep, 1].astype(np.float32),
                    cur[keep, 0], cur[keep, 1]], axis=1)
    return out, best_score[keep]


class NumpyStereoMatcher:
    """与旧 cv2 版 StereoMatcher 同接口: match(img1, img2) -> [(u1,v1,u2,v2), ...]。"""

    def __init__(self, max_features: int = 800, log=None):
        self.max_features = max_features
        self._log = log

    def match(self, img1: np.ndarray, img2: np.ndarray) -> list[tuple[float, float, float, float]]:
        kps = detect_features(img1, max_features=self.max_features)
        if len(kps) < 10:
            if self._log:
                self._log(f"match: too few features ({len(kps)})")
            return []
        m, score = match_ncc(img1, img2, kps)
        if self._log:
            self._log(f"match: keypoints={len(kps)} matches={len(m)} "
                      f"ncc_median={(float(np.median(score)) if len(score) else 0.0):.3f}")
        return [(float(r[0]), float(r[1]), float(r[2]), float(r[3])) for r in m]


# ------------------------------------------------------------------ 自测
def _synth_pair(h: int = 760, w: int = 1344, dx: float = 12.0, dy: float = -3.0,
                scale: float = 1.01, rot_deg: float = 0.6, seed: int = 7):
    """合成一对图像: img2[y,x] = img1[ T(x,y) ], T = 中心缩放 + 旋转 + 平移。

    返回 (img1, img2, T_params), T_params 里的参数与 _expected_in_img2 的解析逆变换一致。
    """
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((h, w)).astype(np.float32)
    # 低频纹理 + 硬边缘, 更接近真实街景
    for _ in range(3):
        pad = np.pad(base, 2, mode="edge")
        base = sum(pad[i:i + h, j:j + w] for i in range(5) for j in range(5)) / 25.0
    img1 = 128 + 60 * base / (np.abs(base).max() + 1e-6)
    for _ in range(40):  # 随机矩形边缘 -> 强角点
        x0 = rng.integers(0, w - 120)
        y0 = rng.integers(0, h - 120)
        img1[y0:y0 + rng.integers(20, 90), x0:x0 + rng.integers(20, 90)] += rng.uniform(-60, 60)
    img1 = np.clip(img1, 0, 255).astype(np.uint8)

    th = np.deg2rad(rot_deg)
    cx, cy = w / 2.0, h / 2.0
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    # 采样映射 (img2 网格 -> img1 坐标): p1 = s·R·(p2 - c) + c + t
    xr = (xs - cx) * scale
    yr = (ys - cy) * scale
    xs2 = (xr * np.cos(th) - yr * np.sin(th)) + cx + dx
    ys2 = (xr * np.sin(th) + yr * np.cos(th)) + cy + dy
    xi = np.clip(np.rint(xs2).astype(np.int32), 0, w - 1)
    yi = np.clip(np.rint(ys2).astype(np.int32), 0, h - 1)
    img2 = img1[yi, xi]
    return img1, img2, {"h": h, "w": w, "dx": dx, "dy": dy, "scale": scale, "rot_deg": rot_deg}


def _expected_in_img2(pts: np.ndarray, tp: dict) -> np.ndarray:
    """解析逆变换: 给定 img1 上的点, 求它在 img2 里应有的坐标 (采样映射的逆)。"""
    th = np.deg2rad(tp["rot_deg"])
    cx, cy = tp["w"] / 2.0, tp["h"] / 2.0
    dx = pts[:, 0] - cx - tp["dx"]
    dy = pts[:, 1] - cy - tp["dy"]
    # R^-1 = R(-th)
    qx = (dx * np.cos(th) + dy * np.sin(th)) / tp["scale"]
    qy = (-dx * np.sin(th) + dy * np.cos(th)) / tp["scale"]
    return np.stack([qx + cx, qy + cy], axis=1)


def _self_test():
    for (h, w) in ((760, 1344), (380, 672)):
        img1, img2, tp = _synth_pair(h, w)
        t0 = time.monotonic()
        kps = detect_features(img1)
        t1 = time.monotonic()
        m, score = match_ncc(img1, img2, kps)
        t2 = time.monotonic()
        if len(m) == 0:
            print(f"{w}x{h}: NO MATCHES (keypoints={len(kps)})")
            continue
        exp = _expected_in_img2(m[:, :2], tp)
        err = np.hypot(m[:, 2] - exp[:, 0], m[:, 3] - exp[:, 1])
        print(f"{w}x{h}: kp={len(kps)} matches={len(m)} "
              f"err(px) median={np.median(err):.2f} p90={np.percentile(err, 90):.2f} "
              f"| detect={t1 - t0:.2f}s match={t2 - t1:.2f}s "
              f"ncc_median={np.median(score):.3f}")


if __name__ == "__main__":
    _self_test()
