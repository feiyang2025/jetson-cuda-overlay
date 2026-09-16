import ctypes
import os
from pathlib import Path

import numpy as np
from openpilot.common.swaglog import cloudlog

from openpilot.sunnypilot.modeld_v2.gpu_backend.cuda_transform import CudaTransform
from openpilot.sunnypilot.modeld_v2.gpu_backend.trt_runner import TrtRunner
from openpilot.sunnypilot.modeld_v2.gpu_backend.profile import ModelProfile


def is_agx_orin() -> bool:
  if not os.path.exists("/etc/nv_tegra_release"):
    return False
  try:
    with open("/etc/nv_tegra_release") as f:
      content = f.read()
  except Exception:
    content = ""
  if any(tok in content.upper() for tok in ("ORIN", "AGX", "ORING", "T234")):
    return True
  try:
    with open("/proc/device-tree/model") as f:
      model = f.read().rstrip("\0")
  except Exception:
    model = ""
  return "Orin" in model or "T234" in model or "AGX" in model.upper()


def trt_available() -> bool:
  try:
    ctypes.CDLL("libnvinfer.so")
    return True
  except OSError:
    return False


def cuda_transform_library() -> Path | None:
  path = os.getenv("CUDA_TRANSFORM_LIBRARY")
  if path:
    return Path(path)
  candidate = Path(__file__).resolve().parents[3] / "selfdrive" / "modeld" / "transforms" / "libcuda_transform.so"
  return candidate if candidate.exists() else None


class GpuBackend:
  def __init__(self, profile: ModelProfile):
    self.profile = profile
    lib = cuda_transform_library()
    if lib is None:
      raise RuntimeError("libcuda_transform.so not found; build it on-device or set CUDA_TRANSFORM_LIBRARY")
    self.transform = CudaTransform(profile.model_w, profile.model_h, 4, str(lib))
    if profile.mode == "merged":
      engine = Path(profile.engine_dir) / profile.merged_engine
      self.vision_runner: TrtRunner | None = TrtRunner(str(engine))
      self.policy_runner: TrtRunner | None = None
    else:
      v_engine = Path(profile.engine_dir) / profile.vision_engine
      p_engine = Path(profile.engine_dir) / profile.policy_engine
      self.vision_runner = TrtRunner(str(v_engine))
      self.policy_runner = TrtRunner(str(p_engine))
    self._vision_ptrs: dict[str, int] = {}
    self._vision_outs: dict[str, np.ndarray] = {}

  def preprocess(self, frames: dict[str, object], transforms: dict[str, np.ndarray]) -> None:
    self._vision_ptrs.clear()
    self._vision_outs.clear()
    for key in self.profile.vision_input_names:
      if key not in frames:
        continue
      frame = frames[key]
      proj = transforms[key]
      shape = self.profile.vision_shape_for(key)
      tensor = self.transform(key, frame, proj, shape)
      self._vision_ptrs[key] = self.transform.last_ptr

  def infer_merged(self, scalar_inputs: dict[str, np.ndarray]) -> np.ndarray:
    kwargs: dict = {k: self._vision_ptrs[k] for k in self.profile.vision_input_names if k in self._vision_ptrs}
    kwargs.update(scalar_inputs)
    return self.vision_runner(**kwargs)

  def infer_split(self, scalar_inputs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    vision_kwargs: dict = {k: self._vision_ptrs[k] for k in self.profile.vision_input_names if k in self._vision_ptrs}
    vision_out = self.vision_runner(**vision_kwargs)
    if "hidden_state" in self.profile.vision_slices:
      hidden = vision_out[self.profile.vision_slices["hidden_state"]]
      scalar_inputs["features_buffer"][0, :-1] = scalar_inputs["features_buffer"][0, 1:]
      scalar_inputs["features_buffer"][0, -1] = hidden[: self.profile.temporal.features_len]
    policy_out = self.policy_runner(**scalar_inputs)
    return vision_out, policy_out

  def close(self) -> None:
    for runner in (self.vision_runner, self.policy_runner):
      if runner is not None:
        runner.close()
    if self.transform is not None:
      self.transform.close()

  def __del__(self):
    try:
      self.close()
    except Exception:
      pass


def create_gpu_backend(profile: ModelProfile) -> GpuBackend | None:
  if not is_agx_orin():
    cloudlog.warning("Not AGX Orin; CUDA backend disabled")
    return None
  if not trt_available():
    cloudlog.warning("TensorRT unavailable; CUDA backend disabled")
    return None
  if os.getenv("DISABLE_CUDA_BACKEND", "0") == "1":
    cloudlog.warning("CUDA backend disabled by env")
    return None
  try:
    return GpuBackend(profile)
  except Exception as e:
    cloudlog.warning(f"CUDA backend init failed: {e}")
    return None
