"""FrogPilotGpuState — run FrogPilot's modeld business logic on CUDA/TensorRT.

Why this exists and where it plugs in:

FrogPilot's modeld comes in three processes (system/manager/process_config.py):

  * NativeProcess "modeld"         -> selfdrive/modeld/. C++ supercombo/thneed.
  * NativeProcess "classic_modeld" -> frogpilot/selfdrive/modeld/classic/. thneed.
  * PythonProcess "tinygrad_modeld" -> frogpilot/selfdrive/modeld/tinygrad_modeld.py.

None of the C++/thneed ones can run on AGX Orin. The Python one's `main()` touches
the model through exactly four members:

  model.vision_input_names, model.desire_type, model.use_lateral_control_params,
  model.run(bufs, transforms, inputs, prepare_only)

and constructs it as `ModelState(cl_context, model_name)`.

This class is the seam. It owns the overlay's `GpuModelState` (CUDA transform +
TensorRT, chosen through the overlay's profile registry) and presents the four
members above. It does NOT reimplement temporal bookkeeping: GpuModelState
already owns the desire pulse and the features_buffer shift, and its `run` takes
nearly the same dict arguments FrogPilot passes.

Two FrogPilot-specific differences the shim handles:

* `prepare_only` (a dropped frame still updates the image transform / temporal
  state but must not run inference). GpuModelState has no such mode, so the shim
  runs its preprocess step alone in that case.
* `.desire_type` is a name FrogPilot reads to key its inputs dict; it is exposed
  from the resolved profile (`desire` vs `desire_pulse`).

The model bundle comes from the overlay's compiled models directory, NOT from
FrogPilot's own downloads: FrogPilot-Resources only ships thneed (v1-v6) and
tinygrad pkl (v8+) artifacts, neither of which runs on Orin. Point
`FP_CUDA_MODEL_DIR` at the models directory (or at one bundle inside it).
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np

# GpuModelState / registry module names, new layout first. sys.modules is checked
# before importing: sitecustomize runs in every process, and a module that is
# already loaded must not be imported twice (that would re-run its device setup).
STATE_MODULES = ("openpilot.sunnypilot.modeld_v2.gpu_model_state",)
REGISTRY_MODULES = ("openpilot.sunnypilot.modeld_v2.gpu_backend.models.registry",)

_state_cls: Any = None
_registry: Any = None


def _import_first(names) -> Any:
  for name in names:
    module = sys.modules.get(name)
    if module is not None:
      return module
  for name in names:
    try:
      return importlib.import_module(name)
    except Exception:
      continue
  return None


def _gpu_state_class():
  global _state_cls
  if _state_cls is None:
    module = _import_first(STATE_MODULES)
    _state_cls = getattr(module, "GpuModelState", False) if module is not None else False
  return cast(Any, _state_cls) or None


def _registry_module():
  global _registry
  if _registry is None:
    _registry = _import_first(REGISTRY_MODULES)
  return _registry


def _models_root() -> Path | None:
  """Models directory to resolve bundles from.

  `FP_CUDA_MODEL_DIR` may point at a models directory (models/<Name>/) or
  directly at one bundle. Without it, the tree that owns this file is searched
  upwards for a selfdrive/modeld/models directory.
  """
  env_dir = os.getenv("FP_CUDA_MODEL_DIR")
  if env_dir:
    base = Path(env_dir)
    if base.is_dir():
      # A bundle directory itself (has metadata pkl) is its own models root.
      if any(base.glob("*metadata*.pkl")):
        return base.parent
      return base
  here = Path(__file__).resolve()
  for parent in here.parents:
    for relative in ("selfdrive/modeld/models", "openpilot/selfdrive/modeld/models"):
      candidate = parent / relative
      if candidate.is_dir():
        return candidate
  return None


def resolve_model_name(model: str | None) -> str | None:
  """Map FrogPilot's model name onto an overlay bundle name.

  FrogPilot names are its own catalogue (`dark-souls`, `starry-night`, ...);
  overlay bundles are named after the network (`FiletOFish`, `TGC`, ...). The
  env var wins, then an exact bundle-name match, then the fork name, and finally
  the registry's own preference (None = let the registry decide).
  """
  forced = os.getenv("FP_CUDA_BUNDLE")
  if forced:
    return forced
  root = _models_root()
  if root is not None and model and (root / model).is_dir():
    return model
  return None


class FrogPilotGpuState:
  """Drop-in for FrogPilot's ModelState backed by the overlay CUDA/TensorRT state.

  Constructor mirrors the fork's `(context, model[, model_version])` shapes;
  `context` (an OpenCL CLContext) is ignored - the CUDA path has no use for it.
  """

  def __init__(self, context=None, model: str | None = None, model_version: str | None = None):
    GpuState = _gpu_state_class()
    if GpuState is None:
      raise RuntimeError("overlay GpuModelState not importable (gpu_model_state.py missing?)")
    registry = _registry_module()
    if registry is None:
      raise RuntimeError("overlay model registry not importable (gpu_backend/models/registry.py missing?)")

    root = _models_root()
    requested = resolve_model_name(model)
    profile = registry.resolve_profile(requested, root=root) if root is not None \
      else registry.resolve_profile(requested)
    if profile is None:
      raise RuntimeError(f"no usable CUDA bundle for model {model!r} (root={root})")

    self.profile = profile
    self._core = GpuState(cam_w=profile.model_w, cam_h=profile.model_h,
                          chestnut=False, profile=profile)
    warmup = getattr(self._core, "warmup", None)
    if callable(warmup):
      warmup()

    self.model_name = model or profile.name
    self.bundle_name = profile.name
    # The four members FrogPilot's main() uses:
    self.vision_input_names = list(self._core.vision_input_names)
    self.desire_type = self._core.desire_key
    self.use_lateral_control_params = "lateral_control_params" in profile.input_shapes
    self.desire_len = profile.input_shapes.get(self.desire_type, (1, 25, 8))[-1]
    # classic_modeld's fill_model_msg reads this; the CUDA path derives curvature
    # from the plan, so the dedicated head is not in use.
    self.desired_curvature_head = False
    self.chestnut = False
    self.frame_drop_ratio = 0.0

  def run(self, bufs: dict[str, object], transforms: dict[str, np.ndarray],
          inputs: dict[str, np.ndarray], prepare_only: bool) -> dict[str, np.ndarray] | None:
    """FrogPilot's 4-arg run(bufs, transforms, inputs, prepare_only).

    A dropped frame must still refresh the transform/temporal state without an
    inference, which GpuModelState has no mode for: run its preprocess step and
    return, exactly what the dropped-frame path needs.
    """
    if prepare_only:
      frames = {k: bufs[k] for k in self.vision_input_names if k in bufs}
      self._core.backend.preprocess(frames, transforms)
      return None
    return self._core.run(bufs, transforms, inputs)


def patch_tinygrad_modeld(module) -> bool:
  """Replace FrogPilot's tinygrad ModelState/CLContext with the CUDA state.

  `module` is the fully-imported `frogpilot...tinygrad_modeld` module. Only the
  class its main() instantiates is swapped; nothing else is touched.
  """
  if getattr(module, "_fork_fp_cuda", False):
    return False
  if getattr(module, "ModelState", None) is None:
    return False
  module.ModelState = FrogPilotGpuState
  module._fork_fp_cuda = True
  # CLContext is only used to build VisionIpcClient buffers; the CUDA state
  # ignores it. A no-op stub keeps the fork's main() from dying on the missing
  # OpenCL stack - an unused argument must not stop the process.
  if getattr(module, "CLContext", None) is not None:
    class _NullCLContext:
      def __init__(self, *a, **kw):
        pass
    module.CLContext = _NullCLContext
  return True