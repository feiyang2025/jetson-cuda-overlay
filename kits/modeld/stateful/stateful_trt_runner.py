#!/usr/bin/env python3
"""TensorRT execution for the STATEFUL (cinque-v3) driving policy.

ctypes/libcudart port of the stateful TRT runner (the original imported torch,
which is not available on the Jetson JetPack 7 / CUDA 13 image).

  inputs : new_img, desire, traffic_convention, action_t   (state_* are internal)
  outputs: outputs                                         (next_state_* are internal)

The recurrent state lives on the DEVICE as two ping-pong buffers per state: each
frame the engine reads one slot and writes next_state_* into the other, then the
slots swap. Nothing is copied host<->device for the state, which is what keeps
the per-frame cost down (the state is ~10 MB).
"""
from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Mapping

import numpy as np

from openpilot.selfdrive.modeld.tensorrt_policy_runner import CudaRuntime

_TRT_TO_NUMPY = {
  "FLOAT": np.float32,
  "HALF": np.float16,
  "INT32": np.int32,
  "INT8": np.int8,
  "BOOL": np.bool_,
}


def _import_tensorrt():
  import importlib
  import sys
  for _p in ("/usr/lib/python3.12/dist-packages", "/home/nvidia/.local/lib/python3.12/site-packages",
             "/usr/lib/python3/dist-packages"):
    if _p not in sys.path:
      sys.path.insert(0, _p)
  return importlib.import_module("tensorrt")


class StatefulTensorRTPolicyRunner:
  INPUT_NAMES = ("new_img", "desire", "traffic_convention", "action_t",
                 "state_img_q", "state_desire_q", "state_feat_q")
  OUTPUT_NAMES = ("outputs", "next_state_img_q", "next_state_desire_q", "next_state_feat_q")
  STATE_PAIRS = {"next_state_img_q": "state_img_q",
                 "next_state_desire_q": "state_desire_q",
                 "next_state_feat_q": "state_feat_q"}

  def __init__(self, engine_path: str | Path, input_shapes: Mapping[str, tuple[int, ...]]):
    trt = _import_tensorrt()
    self.trt = trt
    self.input_shapes = {k: tuple(v) for k, v in input_shapes.items()}
    self.cuda = CudaRuntime()
    self.stream = self.cuda.create_stream()

    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
    if engine is None:
      raise RuntimeError(f"failed to deserialize TensorRT engine: {engine_path}")
    context = engine.create_execution_context()
    if context is None:
      raise RuntimeError("failed to create TensorRT execution context")
    self._runtime, self.engine, self.context = runtime, engine, context

    self._host: dict[str, np.ndarray] = {}
    self._dev: dict[str, ctypes.c_void_p] = {}
    self._sbuf: dict[str, list] = {}       # state_* input -> [devA, devB], device-resident
    self._sstate: dict[str, tuple] = {}    # state_* input -> (shape, dtype)
    self.input_names: list[str] = []
    self.output_names: list[str] = []
    for i in range(engine.num_io_tensors):
      name = engine.get_tensor_name(i)
      shape = tuple(context.get_tensor_shape(name))
      dtype = _TRT_TO_NUMPY[engine.get_tensor_dtype(name).name]
      (self.input_names if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
       else self.output_names).append(name)
      if name in self.STATE_PAIRS.values():
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        self._sbuf[name] = [self.cuda.malloc(nbytes), self.cuda.malloc(nbytes)]
        self._sstate[name] = (shape, dtype)
        continue
      if name in self.STATE_PAIRS:
        continue  # next_state_* output is an alias of the ping-pong buffers, bound per frame
      self._host[name] = np.zeros(shape, dtype=dtype)
      self._dev[name] = self.cuda.malloc(self._host[name].nbytes)
      if not context.set_tensor_address(name, self._dev[name].value):
        raise RuntimeError(f"failed to bind TensorRT tensor {name}")

    self._sidx = {name: 0 for name in self.STATE_PAIRS.values()}
    self.reset_state()

  def _to(self, name: str, value: np.ndarray) -> None:
    self._host[name][...] = np.asarray(value).reshape(self._host[name].shape)

  def reset_state(self) -> None:
    """Zero the on-device recurrent state (cold start, or after a model change)."""
    for name, (shape, dtype) in self._sstate.items():
      zeros = np.zeros(shape, dtype=dtype)
      for buf in self._sbuf[name]:
        self.cuda.upload(buf, zeros)
    for name in self._sidx:
      self._sidx[name] = 0

  def run(self, new_img: np.ndarray, desire: np.ndarray, traffic_convention: np.ndarray,
          action_t: np.ndarray) -> np.ndarray:
    for name, value in (("new_img", new_img), ("desire", desire),
                        ("traffic_convention", traffic_convention), ("action_t", action_t)):
      self._to(name, value)
      self.cuda.upload(self._dev[name], self._host[name])

    # Read state_* from one slot, write next_state_* into the other, then swap: the
    # next frame reads what was just written. No state ever crosses host<->device.
    for next_name, state_name in self.STATE_PAIRS.items():
      i = self._sidx[state_name]
      self.context.set_tensor_address(state_name, self._sbuf[state_name][i].value)
      self.context.set_tensor_address(next_name, self._sbuf[state_name][1 - i].value)

    if not self.context.execute_async_v3(int(self.stream.value or 0)):
      raise RuntimeError("TensorRT execute_async_v3 failed")
    self.cuda.sync(self.stream)
    for state_name in self._sidx:
      self._sidx[state_name] ^= 1
    self.cuda.download(self._dev["outputs"], self._host["outputs"])
    return self._host["outputs"].astype(np.float32, copy=False)[0]

  def __del__(self):
    cuda = getattr(self, "cuda", None)
    if cuda is None:
      return
    stream = getattr(self, "stream", None)
    if stream:
      cuda.destroy_stream(stream)
    for ptr in getattr(self, "_dev", {}).values():
      cuda.free(ptr)
    for bufs in getattr(self, "_sbuf", {}).values():
      for ptr in bufs:
        cuda.free(ptr)
