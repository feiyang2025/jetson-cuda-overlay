"""标定工具的 fl 基准 —— 全部对齐 sp-cuda 那套「厂商标定」标准, 且与相机模型同源。

为什么需要这个文件 (历史教训):
  标定工具曾把"当前 fl"硬编码成 2520 / 605 (comma 相机的数字、而且是**模型空间**的 fl),
  而本机 AGX 移植的相机模型 (openpilot/common/transformations/camera.py) 用的是**物理**
  fl: FCAM(road 流) = 961.76, ECAM(wide 流) = 1740.0。两套差 ~2.6-2.9 倍 —— 标定算出来的
  绝对建议值一旦写进 FcamIntrinsics, 相机模型会当成物理焦距用, 画面 warp 与视觉测距直接崩。
  所以三处 (工具 / 面板 / camera.py) 必须共用同一个来源: 本模块。

sp-cuda 标准 (该树连跑 1 个多月, 2026-08-30 厂商标定) 的 **镜头数值**:
  * 120° 广角镜头 (带 fisheye 畸变, 模型路径不做畸变校正): pinhole fl ≈ 961.76 @1920x1080
  * 60° 窄角镜头: fx = fy = 1740, cx = 960, cy = 540
按 *本机* 实测的流↔镜头对应 (2026-10-12, ORB 尺度比 road/wide=1.735):
  * road 流 = /dev/video0 = twgmsl 9-0028 = **60° 窄角** → fl = 1740 (× 宽度比)
  * wide 流 = /dev/video1 = twgmsl 9-002a = **120° 广角** → fl = 961.76 (× 宽度比)
  (sp-cuda 那台把 120° 那只接在 road 流上, 所以它写的是 FCAM=961.76 / ECAM=1740;
   本机两只节点的角色相反 —— 数值不变, 归属跟着镜头走。)
  * 运行时可覆盖: Params 的 FcamIntrinsics / EcamIntrinsics; 缺省即上述厂商标定值
  * 分辨率经 env ROAD_CAM_WIDTH/HEIGHT, WIDE_CAM_WIDTH/HEIGHT 参与缩放 (fl 随宽度线性缩放)

读取优先级 (与相机模型保持一致):
  1. openpilot.common.transformations.camera 的 DEVICE_CAMERAS[("pc","unknown")]
     —— 已经做完了 Params 覆盖 + env 缩放, 是唯一权威来源;
  2. 退化路径 (transformations 还没编译好时): 直接读 Params;
  3. 再退化: sp-cuda 厂商标定常量。
"""

from __future__ import annotations

import os

# --- 厂商标定标准 (base 分辨率 1920x1080) ---
STD_BASE_W = 1920
STD_BASE_H = 1080
STD_FCAM_FL = 1740.0     # road 流 / 本机是 60° 窄角镜头 (tvide0 9-0028)
STD_ECAM_FL = 961.76    # wide 流 / 本机是 120° 广角镜头 (video1 9-002a)

# 名称映射: 工具沿用 FCAM/ECAM 的叫法, 对应 camera.py 的字段与 Params 键
_CAM_META = {
    "fcam": {"attr": "narrow_road", "param": "FcamIntrinsics", "env_w": "ROAD_CAM_WIDTH", "env_h": "ROAD_CAM_HEIGHT", "std_fl": STD_FCAM_FL},
    "ecam": {"attr": "wide_road", "param": "EcamIntrinsics", "env_w": "WIDE_CAM_WIDTH", "env_h": "WIDE_CAM_HEIGHT", "std_fl": STD_ECAM_FL},
}


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key) or default)
    except (TypeError, ValueError):
        return default


def effective(cam: str = "fcam") -> dict:
    """返回该相机当前**生效**的内参: {fl, width, height, source, param_key}。

    source: "camera.py" | "params" | "std" —— 用来判断这个数是从哪来的, 方便日志与排查。
    """
    meta = _CAM_META[cam]
    # 1) 相机模型 (权威): 已含 Params 覆盖与 env 缩放
    try:
        from openpilot.common.transformations.camera import DEVICE_CAMERAS
        cfg = getattr(DEVICE_CAMERAS[("pc", "unknown")], meta["attr"])
        if cfg.focal_length:
            return {"fl": float(cfg.focal_length), "width": int(cfg.width), "height": int(cfg.height),
                    "source": "camera.py", "param_key": meta["param"]}
    except Exception:
        pass

    # 2) 直接读 Params (按 env 宽度缩放, 语义同 camera.py)
    w = _env_int(meta["env_w"], STD_BASE_W)
    h = _env_int(meta["env_h"], STD_BASE_H)
    try:
        from params_json import params_json_get
        d = params_json_get(meta["param"]) or {}
        fl = d.get("fl")
        if fl:
            fl = float(fl) * (w / STD_BASE_W)
            return {"fl": fl, "width": w, "height": h, "source": "params", "param_key": meta["param"]}
    except Exception:
        pass

    # 3) 厂商标定常量
    return {"fl": meta["std_fl"] * (w / STD_BASE_W), "width": w, "height": h,
            "source": "std", "param_key": meta["param"]}


def effective_fl(cam: str = "fcam") -> float:
    return effective(cam)["fl"]


def describe() -> str:
    """一行日志用: 两台相机当前生效的内参及其来源。"""
    out = []
    for cam in ("fcam", "ecam"):
        e = effective(cam)
        out.append(f"{cam.upper()} fl={e['fl']:.2f} @{e['width']}x{e['height']} ({e['source']})")
    return " | ".join(out)


if __name__ == "__main__":
    print(describe())
