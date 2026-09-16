import ctypes
import os
from pathlib import Path

import numpy as np
from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor


class CudaTransform:
  def __init__(self, model_w: int, model_h: int, temporal_skip: int, library_path: str | None = None):
    path = library_path or os.getenv("CUDA_TRANSFORM_LIBRARY")
    if path is None:
      path = str(Path(__file__).resolve().parents[3] / "selfdrive" / "modeld" / "transforms" / "libcuda_transform.so")
    self._cuda = ctypes.CDLL("libcuda.so.1")
    self._library = ctypes.CDLL(path)
    self._library.cuda_transform_init.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    self._library.cuda_transform_init.restype = None
    self._library.cuda_transform_execute.argtypes = [
      ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
      ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ]
    self._library.cuda_transform_execute.restype = ctypes.c_void_p
    self._library.cuda_transform_destroy.argtypes = [ctypes.c_void_p]
    self._library.cuda_transform_destroy.restype = None
    self._cuda.cuMemHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
    self._cuda.cuMemHostRegister.restype = ctypes.c_int
    self._cuda.cuMemHostUnregister.argtypes = [ctypes.c_void_p]
    self._cuda.cuMemHostUnregister.restype = ctypes.c_int
    self._registered: set[tuple[str, int]] = set()
    self._states: dict[str, ctypes.Array] = {}
    self._model_w = model_w
    self._model_h = model_h
    self._temporal_skip = temporal_skip
    # TRT engine creation may leave a different (or destroyed) context current.
    # Always restore the tinygrad primary context before our raw driver calls.
    self._cuda.cuCtxGetCurrent.argtypes = [ctypes.c_void_p]
    self._cuda.cuCtxGetCurrent.restype = ctypes.c_int
    self._cuda.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
    self._cuda.cuCtxSetCurrent.restype = ctypes.c_int
    self._primary_ctx = None
    try:
      from tinygrad import Device
      self._primary_ctx = int(ctypes.cast(Device["CUDA"].context, ctypes.c_void_p).value or 0)
    except Exception:
      self._primary_ctx = None

  def _restore_ctx(self) -> None:
    if not self._primary_ctx:
      return
    cur = ctypes.c_ulonglong()
    self._cuda.cuCtxGetCurrent(ctypes.byref(cur))
    if cur.value != self._primary_ctx:
      self._cuda.cuCtxSetCurrent(ctypes.c_void_p(self._primary_ctx))

  def _state(self, camera: str) -> ctypes.Array:
    if camera not in self._states:
      self._restore_ctx()
      state = ctypes.create_string_buffer(128)
      self._library.cuda_transform_init(
        ctypes.byref(state), self._model_w, self._model_h, self._temporal_skip,
      )
      self._states[camera] = state
    return self._states[camera]

  def __call__(self, camera: str, frame, projection: np.ndarray, shape: tuple[int, ...]) -> Tensor:
    self._restore_ctx()
    data = np.frombuffer(frame.data, dtype=np.uint8)
    ptr = int(data.ctypes.data)
    # Pin the NV12 frame buffer (CUDA_MEMHOSTREGISTER_DEVICEMAP=0x02) so the
    # transform kernel can safely read it on AGX Orin's unified memory.
    key = (camera, ptr)
    if key not in self._registered:
      ret = self._cuda.cuMemHostRegister(ctypes.c_void_p(ptr), data.nbytes, 0x02)
      if ret != 0:
        raise RuntimeError(f"cuMemHostRegister failed: {ret}")
      self._registered.add(key)
    output = self._library.cuda_transform_execute(
      ctypes.byref(self._state(camera)), ctypes.c_void_p(ptr),
      frame.width, frame.height, frame.stride, frame.uv_offset,
      projection.astype(np.float32, copy=False).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    if not output:
      raise RuntimeError("cuda_transform_execute returned null")
    self.last_ptr = int(output)
    return Tensor.from_blob(output, shape, dtype=dtypes.uint8, device="CUDA")

  def close(self) -> None:
    # Release pinned-memory registrations before tearing down the transform
    # states (CUDA_MEMHOSTREGISTER_DEVICEMAP pins the pages until unregistered).
    for _, ptr in list(self._registered):
      self._cuda.cuMemHostUnregister(ctypes.c_void_p(ptr))
    self._registered.clear()
    for state in self._states.values():
      self._library.cuda_transform_destroy(ctypes.byref(state))
    self._states.clear()

  def __del__(self):
    try:
      self.close()
    except Exception:
      pass
