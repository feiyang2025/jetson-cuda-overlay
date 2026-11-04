import ctypes
import os
import re
from pathlib import Path

import numpy as np
from openpilot.common.swaglog import cloudlog

from openpilot.sunnypilot.modeld_v2.gpu_backend.cuda_transform import CudaTransform
from openpilot.sunnypilot.modeld_v2.gpu_backend.trt_runner import TrtRunner
from openpilot.sunnypilot.modeld_v2.gpu_backend.profile import ModelProfile


def is_agx_orin() -> bool:
  """True on a Jetson AGX Orin. Matches on the SoC name, not a pinned L4T
  version - the old check hardcoded 'ORIN'/'40.1'/'35.' and went false on any
  future JetPack."""
  release = Path("/etc/nv_tegra_release")
  if not release.exists():
    return False
  try:
    content = release.read_text()
  except OSError:
    return False
  if "ORIN" in content.upper():
    return True
  # AGX Orin module model string, e.g. "NVIDIA Jetson AGX Orin 32GB"
  device_tree = Path("/sys/firmware/devicetree/base/model")
  if device_tree.exists():
    try:
      return "orin" in device_tree.read_bytes().decode(errors="ignore").lower()
    except OSError:
      pass
  return bool(re.search(r"\b(orin|t234)\b", content, re.IGNORECASE))


def trt_available() -> str | None:
  """Return None when TensorRT is loadable, else the reason it is not."""
  try:
    ctypes.CDLL("libnvinfer.so")
    return None
  except OSError as e:
    return str(e)


def cuda_transform_library() -> Path | None:
  path = os.getenv("CUDA_TRANSFORM_LIBRARY")
  if path:
    return Path(path)
  candidate = Path(__file__).resolve().parents[3] / "selfdrive" / "modeld" / "transforms" / "libcuda_transform.so"
  return candidate if candidate.exists() else None


class GpuBackend:
  def __init__(self, profile: ModelProfile):
    missing = profile.missing_reason()
    if missing is not None:
      raise RuntimeError(f"profile {profile.name} not usable: {missing}")

    lib = cuda_transform_library()
    if lib is None:
      raise RuntimeError(
        "libcuda_transform.so not found; build it on-device "
        "(nvcc -arch=sm_87 -shared -O2 -o libcuda_transform.so cuda_transform.cu) or set CUDA_TRANSFORM_LIBRARY"
      )

    if profile.mode == "merged":
      engine = Path(profile.engine_dir) / profile.merged_engine
      self.vision_runner: TrtRunner | None = TrtRunner(str(engine))
      self.policy_runner: TrtRunner | None = None
    else:
      root = Path(profile.engine_dir)
      self.vision_runner = TrtRunner(str(root / profile.vision_engine))
      self.policy_runner = TrtRunner(str(root / profile.policy_engine))

    # The vision tensor dtype comes from the engine, not from a constant.
    # dtype 从引擎绑定读, 不写死: BigCombo=uint8, FiletOFish 也实测 uint8, 但绑定名不同
    # (img vs input_imgs), 所以必须按本 profile 的第一个 vision 输入名取, 否则取不到
    # 会静默退回 uint8 —— 换模型/换引擎时就是无声的错。
    _vision_key = profile.vision_input_names[0]
    vision_dtype = self.vision_runner.input_dtypes.get(_vision_key, np.dtype(np.uint8))
    self.transform = CudaTransform(profile.model_w, profile.model_h, profile.temporal_skip, str(lib),
                                   dtype=vision_dtype)

    self.profile = profile
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
      self.transform(key, frame, proj, shape)
      self._vision_ptrs[key] = self.transform.last_ptr

  def _vision_kwargs(self, scalar_inputs: dict[str, np.ndarray]) -> dict:
    # Vision pointers go in last and win. A caller that passes a full input dict
    # (GpuModelState builds one from the profile) would otherwise overwrite the
    # transform's device pointer with a zero array and the engine would run on a
    # black image without a single error.
    kwargs: dict = dict(scalar_inputs)
    for key in self.profile.vision_input_names:
      if key in self._vision_ptrs:
        kwargs[key] = self._vision_ptrs[key]
    return kwargs

  def infer_merged(self, scalar_inputs: dict[str, np.ndarray]) -> np.ndarray:
    return self.vision_runner(**self._vision_kwargs(scalar_inputs))

  def infer_split(self, scalar_inputs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Run vision then policy.

    Deliberately does NOT shift features_buffer: temporal bookkeeping belongs to
    the modeld layer (GpuModelState._update_temporal), which shifts exactly once
    per frame using whichever engine actually produced hidden_state.
    """
    vision_out = self.vision_runner(**self._vision_kwargs(scalar_inputs))
    policy_out = self.policy_runner(**scalar_inputs)
    return vision_out, policy_out

  def close(self) -> None:
    for runner in (self.vision_runner, self.policy_runner):
      if runner is not None:
        runner.close()
    if getattr(self, "transform", None) is not None:
      self.transform.close()

  def __del__(self):
    try:
      self.close()
    except Exception:
      pass


def create_gpu_backend(profile: ModelProfile) -> GpuBackend | None:
  """Returns None when the CUDA backend cannot be used. Every rejection logs the
  concrete reason - a bare 'falling back' would hide a broken install."""
  if os.getenv("DISABLE_CUDA_BACKEND", "0") == "1":
    cloudlog.warning("CUDA backend disabled by DISABLE_CUDA_BACKEND=1")
    return None
  if not is_agx_orin():
    cloudlog.warning("Not a Jetson AGX Orin (no /etc/nv_tegra_release match); CUDA backend disabled")
    return None
  trt_error = trt_available()
  if trt_error is not None:
    cloudlog.warning(f"TensorRT unavailable ({trt_error}); CUDA backend disabled")
    return None
  try:
    # TrtRunner allocates through the CUDA driver and needs a context that already
    # exists. modeld normally has one from the tinygrad runner, but a bare process
    # (tests, a one-off script) does not, and the failure is a bare
    # "No active CUDA context" that reads like a broken install. tinygrad owns that
    # context for the whole process, so go through it rather than making our own.
    from tinygrad import Device
    Device["CUDA"]
  except Exception as e:
    cloudlog.warning(f"could not initialise a CUDA context ({type(e).__name__}: {e}); CUDA backend disabled")
    return None
  try:
    return GpuBackend(profile)
  except Exception as e:
    cloudlog.warning(f"CUDA backend init failed, falling back to tinygrad: {type(e).__name__}: {e}")
    return None
