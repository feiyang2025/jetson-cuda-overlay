#!/usr/bin/env python3
from __future__ import annotations

import ctypes
import ctypes.util
import math
from pathlib import Path
from typing import Mapping

import numpy as np

# CUDA memory/stream plumbing for the TensorRT big-model backend.
#
# This used to pull in torch solely for its allocator, CUDA stream and a handful
# of tensor ops. On the Jetson AGX Orin that is a ~2-3 GB dependency with no
# prebuilt wheel for JetPack 7 / CUDA 13, so the tiny amount of work torch did is
# done here with ctypes against libcudart instead:
#   cudaMalloc/Free          -> device buffers
#   cudaMemcpy H2D/D2H       -> staging the (small) model inputs and outputs
#   cudaStreamCreate/Sync    -> the same execution stream TensorRT runs on
# The ring-buffer rotation, frame subsampling and the desire amax are pure host
# numpy; every one of those buffers is tiny (<=1.5 MB), so one upload per frame
# is cheaper than the device-side indexing it replaces.

_CUDA_MEMCPY_HOST_TO_DEVICE = 1
_CUDA_MEMCPY_DEVICE_TO_HOST = 2
_CUDA_HOST_ALLOC_DEFAULT = 0
_CUDA_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1

_LIBCUDART_CANDIDATES = (
  "libcudart.so",
  "libcudart.so.13",
  "libcudart.so.12",
  "/usr/local/cuda/lib64/libcudart.so",
  "/usr/lib/aarch64-linux-gnu/libcudart.so",
)


def _load_cudart() -> ctypes.CDLL:
  last_error: OSError | None = None
  for name in _LIBCUDART_CANDIDATES:
    try:
      return ctypes.CDLL(name)
    except OSError as exc:
      last_error = exc
  found = ctypes.util.find_library("cudart")
  if found:
    return ctypes.CDLL(found)
  raise RuntimeError(f"libcudart not found; TensorRT backend unavailable ({last_error})")


class CudaRuntime:
  """Minimal libcudart binding: device buffers + one execution stream."""

  def __init__(self):
    self.lib = _load_cudart()
    lib = self.lib
    lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    lib.cudaMalloc.restype = ctypes.c_int
    lib.cudaFree.argtypes = [ctypes.c_void_p]
    lib.cudaFree.restype = ctypes.c_int
    lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    lib.cudaMemcpy.restype = ctypes.c_int
    lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.cudaStreamCreate.restype = ctypes.c_int
    lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
    lib.cudaStreamSynchronize.restype = ctypes.c_int
    lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
    lib.cudaStreamDestroy.restype = ctypes.c_int
    lib.cudaHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
    lib.cudaHostAlloc.restype = ctypes.c_int
    lib.cudaFreeHost.argtypes = [ctypes.c_void_p]
    lib.cudaFreeHost.restype = ctypes.c_int
    lib.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                    ctypes.c_int, ctypes.c_void_p]
    lib.cudaMemcpyAsync.restype = ctypes.c_int
    lib.cudaStreamBeginCapture.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.cudaStreamBeginCapture.restype = ctypes.c_int
    lib.cudaStreamEndCapture.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.cudaStreamEndCapture.restype = ctypes.c_int
    lib.cudaGraphInstantiateWithFlags.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
                                                  ctypes.c_ulonglong]
    lib.cudaGraphInstantiateWithFlags.restype = ctypes.c_int
    lib.cudaGraphLaunch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.cudaGraphLaunch.restype = ctypes.c_int
    lib.cudaGraphExecDestroy.argtypes = [ctypes.c_void_p]
    lib.cudaGraphExecDestroy.restype = ctypes.c_int
    lib.cudaGraphDestroy.argtypes = [ctypes.c_void_p]
    lib.cudaGraphDestroy.restype = ctypes.c_int

  @staticmethod
  def _check(code: int, what: str) -> None:
    if code != 0:
      raise RuntimeError(f"{what} failed with cudaError {code}")

  def malloc(self, nbytes: int) -> ctypes.c_void_p:
    ptr = ctypes.c_void_p()
    self._check(self.lib.cudaMalloc(ctypes.byref(ptr), nbytes), f"cudaMalloc({nbytes})")
    return ptr

  def free(self, ptr: ctypes.c_void_p) -> None:
    if ptr:
      self.lib.cudaFree(ptr)

  def upload(self, ptr: ctypes.c_void_p, host: np.ndarray) -> None:
    self._check(self.lib.cudaMemcpy(ptr, host.ctypes.data_as(ctypes.c_void_p),
                                    host.nbytes, _CUDA_MEMCPY_HOST_TO_DEVICE), "cudaMemcpy H2D")

  def download(self, ptr: ctypes.c_void_p, host: np.ndarray) -> None:
    self._check(self.lib.cudaMemcpy(host.ctypes.data_as(ctypes.c_void_p), ptr,
                                    host.nbytes, _CUDA_MEMCPY_DEVICE_TO_HOST), "cudaMemcpy D2H")

  def create_stream(self) -> ctypes.c_void_p:
    stream = ctypes.c_void_p()
    self._check(self.lib.cudaStreamCreate(ctypes.byref(stream)), "cudaStreamCreate")
    return stream

  def sync(self, stream: ctypes.c_void_p) -> None:
    self._check(self.lib.cudaStreamSynchronize(stream), "cudaStreamSynchronize")

  def destroy_stream(self, stream: ctypes.c_void_p) -> None:
    if stream:
      self.lib.cudaStreamDestroy(stream)

  def host_alloc(self, nbytes: int) -> ctypes.c_void_p:
    ptr = ctypes.c_void_p()
    self._check(self.lib.cudaHostAlloc(ctypes.byref(ptr), nbytes, _CUDA_HOST_ALLOC_DEFAULT),
                f"cudaHostAlloc({nbytes})")
    return ptr

  def host_free(self, ptr: ctypes.c_void_p) -> None:
    if ptr:
      self.lib.cudaFreeHost(ptr)

  def upload_async(self, ptr: ctypes.c_void_p, host: np.ndarray, stream: ctypes.c_void_p) -> None:
    self._check(self.lib.cudaMemcpyAsync(ptr, host.ctypes.data_as(ctypes.c_void_p),
                                         host.nbytes, _CUDA_MEMCPY_HOST_TO_DEVICE, stream), "cudaMemcpyAsync H2D")

  def download_async(self, ptr: ctypes.c_void_p, host: np.ndarray, stream: ctypes.c_void_p) -> None:
    self._check(self.lib.cudaMemcpyAsync(host.ctypes.data_as(ctypes.c_void_p), ptr,
                                         host.nbytes, _CUDA_MEMCPY_DEVICE_TO_HOST, stream), "cudaMemcpyAsync D2H")

  def begin_capture(self, stream: ctypes.c_void_p) -> None:
    self._check(self.lib.cudaStreamBeginCapture(stream, _CUDA_STREAM_CAPTURE_MODE_THREAD_LOCAL),
                "cudaStreamBeginCapture")

  def end_capture(self, stream: ctypes.c_void_p) -> ctypes.c_void_p:
    graph = ctypes.c_void_p()
    self._check(self.lib.cudaStreamEndCapture(stream, ctypes.byref(graph)), "cudaStreamEndCapture")
    return graph

  def graph_instantiate(self, graph: ctypes.c_void_p) -> ctypes.c_void_p:
    graph_exec = ctypes.c_void_p()
    self._check(self.lib.cudaGraphInstantiateWithFlags(ctypes.byref(graph_exec), graph, 0),
                "cudaGraphInstantiateWithFlags")
    return graph_exec

  def graph_launch(self, graph_exec: ctypes.c_void_p, stream: ctypes.c_void_p) -> None:
    self._check(self.lib.cudaGraphLaunch(graph_exec, stream), "cudaGraphLaunch")

  def graph_destroy(self, graph: ctypes.c_void_p, graph_exec: ctypes.c_void_p) -> None:
    if graph_exec:
      self.lib.cudaGraphExecDestroy(graph_exec)
    if graph:
      self.lib.cudaGraphDestroy(graph)

  @staticmethod
  def stream_handle(stream: ctypes.c_void_p) -> int:
    return int(stream.value or 0)


class TensorRTPolicyRunner:
  """TensorRT execution for the driving policy, preserving modeld queue semantics."""

  INPUT_DTYPES = {
    "img": "float32",
    "big_img": "float32",
    "desire_pulse": "float16",
    "traffic_convention": "float16",
    "action_t": "float16",
    "features_buffer": "float16",
  }

  _TRT_TO_NUMPY = {
    "FLOAT": np.float32,
    "HALF": np.float16,
    "INT32": np.int32,
    "INT8": np.int8,
    "BOOL": np.bool_,
  }

  def __init__(self, engine_path: str | Path, input_shapes: Mapping[str, tuple[int, ...]], frame_skip: int):
    import importlib
    import sys

    # tensorrt ships in the system dist-packages on the Jetson image.
    for _p in ("/usr/lib/python3.12/dist-packages", "/home/nvidia/.local/lib/python3.12/site-packages"):
      if _p not in sys.path:
        sys.path.insert(0, _p)
    trt = importlib.import_module("tensorrt")

    self.trt = trt
    self.frame_skip = frame_skip
    self.input_shapes = {k: tuple(v) for k, v in input_shapes.items()}
    self.cuda = CudaRuntime()

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
    if engine is None:
      raise RuntimeError(f"failed to deserialize TensorRT engine: {engine_path}")
    context = engine.create_execution_context()
    if context is None:
      raise RuntimeError("failed to create TensorRT execution context")

    self._logger = logger
    self._runtime = runtime
    self.engine = engine
    self.context = context
    self.stream = self.cuda.create_stream()

    # Host-side staging buffers, one per engine binding, filled per frame.
    # Pinned (page-locked) memory keeps the H2D/D2H copies asynchronous and
    # capturable by the CUDA graph below instead of staging through pageable RAM.
    self._host: dict[str, np.ndarray] = {}
    self._dev: dict[str, ctypes.c_void_p] = {}
    self._host_pinned: list[ctypes.c_void_p] = []
    for i in range(engine.num_io_tensors):
      name = engine.get_tensor_name(i)
      shape = tuple(context.get_tensor_shape(name))
      dtype = self._TRT_TO_NUMPY[engine.get_tensor_dtype(name).name]
      self._host[name] = self._pinned_array(shape, dtype)
      self._dev[name] = self.cuda.malloc(int(np.prod(shape)) * np.dtype(dtype).itemsize)
      if not context.set_tensor_address(name, self._dev[name].value):
        raise RuntimeError(f"failed to bind TensorRT tensor {name}")

    # CUDA graph state. The graph is captured once after a warmup frame (so
    # TensorRT's lazy allocations are not part of the capture) and replayed
    # afterwards. Any failure permanently disables the graph and falls back to
    # the direct per-frame path, so a capture issue can never break the big model.
    self._graph: ctypes.c_void_p | None = None
    self._graph_exec: ctypes.c_void_p | None = None
    self._graph_disabled = False
    self._warmed = False

    if "outputs" not in self._host:
      raise RuntimeError(f"TensorRT engine has no 'outputs' tensor: {sorted(self._host)}")
    self.output = self._host.pop("outputs")
    self.output_dev = self._dev.pop("outputs")

    expected_inputs = set(self.INPUT_DTYPES)
    if set(self._host) != expected_inputs:
      raise RuntimeError(f"unexpected TensorRT inputs: {sorted(self._host)} != {sorted(expected_inputs)}")

    # Ring buffers, mirroring the queue layout the tinygrad model uses.
    img = self.input_shapes["img"]
    self.n_frames = img[1] // 6
    image_shape = (frame_skip * (self.n_frames - 1) + 1, 6, img[2], img[3])
    fb = self.input_shapes["features_buffer"]
    dp = self.input_shapes["desire_pulse"]

    self.img_q = np.zeros(image_shape, dtype=np.uint8)
    self.big_img_q = np.zeros(image_shape, dtype=np.uint8)
    self.desire_q = np.zeros((frame_skip * dp[1], dp[0], dp[2]), dtype=np.float16)
    self.feat_q = np.zeros((frame_skip * fb[1], fb[0], math.prod(fb[2:])), dtype=np.float16)
    self._heads = {"img": 0, "big_img": 0, "desire": 0, "feat": 0}

  def __del__(self):
    cuda = getattr(self, "cuda", None)
    if cuda is None:
      return
    cuda.graph_destroy(getattr(self, "_graph", None), getattr(self, "_graph_exec", None))
    for stream in (getattr(self, "stream", None),):
      if stream:
        cuda.destroy_stream(stream)
    for ptr in getattr(self, "_dev", {}).values():
      cuda.free(ptr)
    if getattr(self, "output_dev", None):
      cuda.free(self.output_dev)
    for ptr in getattr(self, "_host_pinned", ()):
      cuda.host_free(ptr)

  def _pinned_array(self, shape: tuple[int, ...], dtype) -> np.ndarray:
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    ptr = self.cuda.host_alloc(nbytes)
    self._host_pinned.append(ptr)
    buffer = (ctypes.c_char * nbytes).from_address(ptr.value)
    return np.frombuffer(buffer, dtype=dtype).reshape(shape)

  # -- queue helpers (port of the torch tensor ops) -------------------------
  def _append(self, name: str, queue: np.ndarray, value: np.ndarray) -> None:
    head = self._heads[name]
    queue[head] = value
    self._heads[name] = (head + 1) % queue.shape[0]

  def _ordered(self, name: str, queue: np.ndarray) -> np.ndarray:
    head = self._heads[name]
    if head == 0:
      return queue
    return np.concatenate((queue[head:], queue[:head]), axis=0)

  def _to_input(self, name: str, host: np.ndarray) -> None:
    self._host[name][...] = host.reshape(self._host[name].shape)

  def run(self, warped: np.ndarray, desire: np.ndarray, traffic_convention: np.ndarray,
          action_t: np.ndarray, prev_feat: np.ndarray) -> np.ndarray:
    img = self.input_shapes["img"]
    if warped.shape != (2, 6, img[2], img[3]):
      raise ValueError(f"unexpected warped shape: {warped.shape}")

    self._append("img", self.img_q, warped[0])
    self._append("big_img", self.big_img_q, warped[1])
    self._append("desire", self.desire_q, np.asarray(desire, dtype=np.float16).reshape(1, -1))
    self._append("feat", self.feat_q, np.asarray(prev_feat, dtype=np.float16).reshape(1, -1))

    img_q = self._ordered("img", self.img_q)
    big_img_q = self._ordered("big_img", self.big_img_q)
    desire_q = self._ordered("desire", self.desire_q)
    feat_q = self._ordered("feat", self.feat_q)

    # _to_input assigns into the pinned fp16/fp32 host buffer, so the explicit
    # astype() casts (and their per-frame full-size allocations) are redundant.
    sampled = lambda q: q[::self.frame_skip]  # noqa: E731 - sampled frames only
    self._to_input("img", sampled(img_q).reshape(img))
    self._to_input("big_img", sampled(big_img_q).reshape(self.input_shapes["big_img"]))

    # desire is a pulse: keep the strongest value in each frame_skip window
    desire_sampled = desire_q.reshape(-1, self.frame_skip, *desire_q.shape[1:]).max(axis=1)
    self._to_input("desire_pulse", desire_sampled)
    self._to_input("features_buffer", sampled(feat_q).reshape(self.input_shapes["features_buffer"]))
    self._to_input("traffic_convention", np.asarray(traffic_convention))
    self._to_input("action_t", np.asarray(action_t))

    if self._graph_exec is not None:
      self.cuda.graph_launch(self._graph_exec, self.stream)
    elif self._graph_disabled:
      self._enqueue()
    elif self._warmed:
      # Second frame onwards: capture once, then replay instead of re-enqueueing
      # every kernel launch. Any failure permanently disables the graph.
      try:
        self._capture_graph()
        self.cuda.graph_launch(self._graph_exec, self.stream)
      except Exception:
        self._graph_disabled = True
        self._graph = None
        self._graph_exec = None
        self._enqueue()
    else:
      self._enqueue()
      self._warmed = True

    self.cuda.sync(self.stream)
    return self.output.astype(np.float32, copy=False)[0]

  def _enqueue(self) -> None:
    for name in self.INPUT_DTYPES:
      self.cuda.upload_async(self._dev[name], self._host[name], self.stream)
    if not self.context.execute_async_v3(CudaRuntime.stream_handle(self.stream)):
      raise RuntimeError("TensorRT execute_async_v3 failed")
    self.cuda.download_async(self.output_dev, self.output, self.stream)

  def _capture_graph(self) -> None:
    self.cuda.begin_capture(self.stream)
    try:
      self._enqueue()
      graph = self.cuda.end_capture(self.stream)
    except Exception:
      # A failed capture leaves the stream in capture mode; end it before re-raising.
      try:
        self.cuda.end_capture(self.stream)
      except Exception:
        pass
      raise
    self._graph = graph
    try:
      self._graph_exec = self.cuda.graph_instantiate(graph)
    except Exception:
      self.cuda.graph_destroy(graph, None)
      self._graph = None
      raise
