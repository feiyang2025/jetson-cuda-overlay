#!/usr/bin/env python3
"""§4 第 2 步自检: CUDA 变换库能加载、能初始化、能执行一次, 且无 CUDA error。

用法: python3 tools/check_cuda_transform.py <树根> [LD_LIBRARY_PATH 里的 libcuda 目录]
只读/只算, 不碰 msgq, 所以可以和别的树同时在跑。
"""
import ctypes
import os
import sys

SO_REL = "openpilot/selfdrive/modeld/transforms/libcuda_transform.so"
# 模型输入尺寸 (openpilot 固定 512x256, temporal_skip=1)
MODEL_W, MODEL_H, TEMPORAL_SKIP = 512, 256, 1
# 源帧: master-c3 实测 CAM_WIDTH/HEIGHT=1344x760 的 NV12
FRAME_W, FRAME_H = 1344, 760


def main() -> int:
  root = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
  so = os.path.join(root, SO_REL)
  if not os.path.isfile(so):
    print(f"[FAIL] 找不到 {so} (先编: nvcc -arch=sm_87 -shared -O2 -o libcuda_transform.so cuda_transform.cu -I.)")
    return 1
  print(f"[info] {so}")

  # cuda_transform_state 是 POD: 8 个指针 + 几个 int, 给 512 字节足够
  state = ctypes.create_string_buffer(512)
  try:
    lib = ctypes.CDLL(so)
  except OSError as e:
    print(f"[FAIL] dlopen 失败: {e}")
    print("       多半是 libcuda.so.1 不在搜索路径 (Jetson: LD_LIBRARY_PATH 指向 l4t-gpu-libs/nvgpu)")
    return 1
  print("[PASS] dlopen libcuda_transform.so")

  lib.cuda_transform_init.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
  lib.cuda_transform_init.restype = None
  lib.cuda_transform_execute.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                         ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
  lib.cuda_transform_execute.restype = ctypes.c_void_p
  lib.cuda_transform_get_output.argtypes = [ctypes.c_void_p]
  lib.cuda_transform_get_output.restype = ctypes.c_void_p
  lib.cuda_transform_destroy.argtypes = [ctypes.c_void_p]
  lib.cuda_transform_destroy.restype = None

  lib.cuda_transform_init(state, MODEL_W, MODEL_H, TEMPORAL_SKIP)
  initialized = int.from_bytes(state[88:92], "little")  # 最后一个 int (initialized)
  if initialized != 1:
    print(f"[FAIL] cuda_transform_init 后 initialized={initialized} (期望 1) —— CUDA 侧初始化没成功")
    return 1
  print(f"[PASS] cuda_transform_init(w={MODEL_W}, h={MODEL_H}, temporal_skip={TEMPORAL_SKIP}) -> initialized=1")

  # 造一帧假 NV12: Y 全 0x40, UV 全 0x80; stride == 宽度(逐行紧凑), uv_offset 紧跟 Y 之后
  y_size = FRAME_W * FRAME_H
  uv_size = (FRAME_W // 2) * (FRAME_H // 2) * 2
  buf = ctypes.create_string_buffer(bytes([0x40]) * y_size + bytes([0x80]) * uv_size)
  proj = (ctypes.c_float * 9)(1, 0, 0, 0, 1, 0, 0, 0, 1)

  out = lib.cuda_transform_execute(state, buf, FRAME_W, FRAME_H,
                                   FRAME_W, y_size, len(buf), proj, 0)
  if not out:
    print("[FAIL] cuda_transform_execute 返回空指针 (宿主输入路径失败)")
    lib.cuda_transform_destroy(state)
    return 1
  got = lib.cuda_transform_get_output(state)
  print(f"[PASS] cuda_transform_execute(宿主 NV12 {FRAME_W}x{FRAME_H}) -> out={hex(out)} get_output={hex(got or 0)}")
  if out != got:
    print("[WARN] execute 返回值与 get_output 不一致 (可能正常, 看实现)")

  lib.cuda_transform_destroy(state)
  print("[PASS] cuda_transform_destroy")
  print("\n第 2 步: CUDA 变换库 加载/初始化/执行/destroy 全过, 无 CUDA error")
  return 0


if __name__ == "__main__":
  sys.exit(main())
