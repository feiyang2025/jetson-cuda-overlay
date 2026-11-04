"""Fork-adaptive CUDA/TensorRT runtime patch layer for openpilot-family forks.

The branch-adaptation layer of the overlay: instead of overwriting a fork's
modeld.py / camerad.py wholesale and hoping the anchors still match, it hooks at
import time and swaps only the classes that must run on AGX Orin, leaving every
other line of the fork's code alone.

What it patches, and why the patches need to exist at all:

* `ModelState.__init__` -> CUDA first, tinygrad fallback. The constructor
  signature differs between forks (sunnypilot new layout: `(cam_w, cam_h,
  chestnut)`; FrogPilot 0.9.7: `(context, model, model_version)`;
  StarPilot/master-c3 variants: `(cam_w, cam_h, chestnut)` with or without more
  kwargs). The adapter reads the signature and only wraps shapes it understands;
  anything else is reported and left as upstream wrote it - so "patch after
  import" never breaks a fork it does not know.
* `camerad.Camera` -> the V4L2/DMABUF camera on Linux, wherever the fork keeps
  its webcam module (`system/camerad/webcam/` or `tools/webcam/`).

Rules for anything added here (same discipline as the eGPU stack's patch layer):

* never import the modules being patched at startup - use `patch_after_import()`.
  modeld must set its tinygrad device before the first tinygrad import;
* one patch per concern, idempotent, and silent on failure: this module runs in
  every python process on the device (pytest and the modeld worker included) and
  must never be able to break one;
* `OPENPILOT_CUDA_PATCH=0` disables everything, `DISABLE_CUDA_BACKEND=1` keeps
  the hooks but makes them fall through to the stock class. Both are read at
  call time, not import time.

The registry it drives (`sunnypilot/modeld_v2/gpu_backend/models/registry.py`)
scans `models/<Name>/` bundles, picks by explicit name -> MODEL_NAME ->
`.trt_order` -> Params "Model" -> curated order -> scanned bundles, and remembers
failed bundles in `models/.trt_failed` so a bad engine cannot restart-loop
modeld.

Run `python3 <this file> --diagnose <tree-root>` (with the tree on PYTHONPATH)
for a snapshot of every candidate bundle and why it is or is not usable.
"""
from __future__ import annotations

import functools
import importlib.machinery
import inspect
import os
import sys
from collections.abc import Callable

ENABLED_ENV = "OPENPILOT_CUDA_PATCH"

# Module names modeld can live under, per fork layout. The adapter hooks all of
# them; hooks for modules the fork never imports simply never fire.
MODELD_MODULES = (
  "openpilot.sunnypilot.modeld_v2.modeld",   # sunnypilot new layout / master-c3
  "openpilot.selfdrive.modeld.modeld",       # official openpilot + most forks
  "selfdrive.modeld.modeld",                 # old-layout forks (sp-cuda, dp, ajouatom)
)
CAMERAD_MODULES = (
  "openpilot.system.camerad.webcam.camerad",  # new layout
  "tools.webcam.camerad",                     # old layout
)

# State shared with apply()/diagnose(): what the last pass did and why.
_PATCH_STATE: dict[str, str] = {}


def enabled() -> bool:
  """Read at call time, not import time (the escape hatch must stay live)."""
  return os.environ.get(ENABLED_ENV, "1") != "0"


def patch_after_import(module_name: str, callback: Callable[[object], bool]) -> bool:
  """Call callback(module) as soon as module_name finishes importing.

  Used instead of importing the target here: the call sites import tinygrad and
  the modeld modules lazily, and importing them at interpreter start would be a
  behaviour change (startup cost, device ordering), not a patch.
  """
  if (module := sys.modules.get(module_name)) is not None:
    try:
      callback(module)
    except Exception:
      pass
    return True

  class _Finder:
    def find_spec(self, fullname, path=None, target=None):
      if fullname != module_name:
        return None
      sys.meta_path.remove(self)
      try:
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
      except Exception:
        return None
      if spec is None or spec.loader is None:
        return None
      execute = spec.loader.exec_module

      def exec_module(module):
        execute(module)
        try:
          callback(module)
        except Exception:
          pass

      try:
        spec.loader.exec_module = exec_module  # type: ignore[method-assign]
      except Exception:
        return None
      return spec

  sys.meta_path.insert(0, _Finder())
  return True


def _constructor_shape(form) -> str | None:
  """'stock' when ModelState.__init__ accepts (cam_w, cam_h, chestnut[, ...]),
  else None - the adapter leaves the fork's class untouched.

  Known stock-shaped variants:
    (cam_w, cam_h, chestnut)                        sunnypilot new layout
    (cam_w, cam_h, chestnut, profile)               master-c3 TRT entry
  FrogPilot 0.9.7's `(context, model, model_version)` is NOT stock-shaped: its
  seam is ModelState.run, which needs the FrogPilot-specific adapter (see
  kits/patch/README.md) - wrapping its __init__ by accident would crash it.
  """
  if not isinstance(form, inspect.Signature) or not form.parameters:
    return None
  params = list(form.parameters.values())
  if params[0].name not in ("self", "cls"):
    return None
  names = [p.name for p in params[1:]]
  if not names:
    return None
  if names[0] in ("cam_w", "width", "cam_h", "height"):
    return "stock"
  return None


def _cuda_enabled_by_env() -> bool:
  return os.environ.get("DISABLE_CUDA_BACKEND", "0") != "1"


def _registry_module():
  """Profile registry across layouts (new first, compat fallback)."""
  try:
    from openpilot.sunnypilot.modeld_v2.gpu_backend.models.registry import (
      mark_failed, resolve_profile, snapshot,
    )
    return mark_failed, resolve_profile, snapshot
  except Exception:
    try:
      from openpilot.sunnypilot.modeld_v2.compat.registry import resolve_profile
      return None, resolve_profile, None
    except Exception:
      return None, None, None


def _pick_profile(chestnut: bool, requested: str | None):
  _, resolve_profile, _ = _registry_module()
  if resolve_profile is None:
    return None
  if requested:
    return resolve_profile(requested)
  # chestnut: big (merged) models win when set, small split otherwise.
  # resolve_profile's own ordering handles the rest.
  return resolve_profile()


def patch_modeld_state(modeld: object) -> bool:
  """Swap a fork's tinygrad ModelState for GpuModelState on a Jetson.

  Wraps `__init__` only when the constructor signature matches a layout this
  adapter understands; otherwise records why and leaves the class untouched.
  On any failure the stock path runs unchanged, and the failed bundle is
  remembered in `.trt_failed` so a bricked engine cannot restart-loop modeld.
  """
  if getattr(modeld, "_fork_cuda_modelstate", False):
    return False
  if not _cuda_enabled_by_env():
    return False
  model_state = getattr(modeld, "ModelState", None)
  if model_state is None:
    return False

  try:
    form = inspect.signature(model_state.__init__)
  except (TypeError, ValueError):
    form = None
  if _constructor_shape(form) is None:
    names = list(form.parameters) if form is not None else []
    _PATCH_STATE["modeld"] = (
      f"signature unsupported {names[:5]}...; fork needs its own adapter"
    )
    return False

  original_init = model_state.__init__

  def _positional(names, idx, default):
    return names[idx] if len(names) > idx else default

  @functools.wraps(original_init)
  def __init__(self, *args, **kwargs):
    profile = None
    try:
      from openpilot.sunnypilot.modeld_v2.gpu_model_state import GpuModelState

      if "chestnut" in kwargs:
        chestnut = bool(kwargs["chestnut"])
      else:
        chestnut = bool(_positional(args, 2, False))

      cam_w = kwargs.get("cam_w", _positional(args, 0, None))
      cam_h = kwargs.get("cam_h", _positional(args, 1, None))
      if cam_w is None or cam_h is None:
        raise TypeError("ModelState constructor did not receive cam_w/cam_h")

      profile = _pick_profile(chestnut, os.getenv("MODEL_NAME") or None)
      if profile is None:
        _PATCH_STATE["modeld"] = "no usable CUDA profile; using tinygrad"
        original_init(self, *args, **kwargs)
        return
      try:
        GpuModelState.__init__(self, cam_w=cam_w, cam_h=cam_h,
                               chestnut=chestnut, profile=profile)
      except TypeError:
        # Fork whose GpuModelState has no profile kwarg: let it resolve its own.
        GpuModelState.__init__(self, cam_w=cam_w, cam_h=cam_h, chestnut=chestnut)
      self.signal_shadow = getattr(self, "signal_shadow", None)
      _PATCH_STATE["modeld"] = f"cuda profile={profile.name} mode={profile.mode}"
    except Exception as e:
      reason = f"{type(e).__name__}: {e}"
      _PATCH_STATE["modeld"] = f"{reason}; using tinygrad"
      try:
        # Remember the failure so modeld does not restart-loop on this engine.
        if profile is not None:
          mark_failed, _, _ = _registry_module()
          if mark_failed is not None:
            mark_failed(profile.name, reason)
      except Exception:
        pass
      original_init(self, *args, **kwargs)

  modeld.ModelState.__init__ = __init__
  modeld._fork_cuda_modelstate = True
  return True


def patch_camerad_camera(camerad_module: object) -> bool:
  """Use the V4L2/DMABUF camera instead of cv2 on Linux (new + old layouts)."""
  if getattr(camerad_module, "_fork_cuda_camera", False):
    return False
  if os.environ.get("USE_V4L2_CAMERA", "1") != "1" or sys.platform == "darwin":
    return False
  if getattr(camerad_module, "Camera", None) is None:
    return False

  for import_path in (
    "openpilot.system.camerad.webcam.v4l2_dmabuf_camera",
    "system.camerad.webcam.v4l2_dmabuf_camera",
    "tools.webcam.v4l2_dmabuf_camera",
  ):
    try:
      v4l2_mod = __import__(import_path, fromlist=[""])
      v4l2_camera = getattr(v4l2_mod, "Camera", None)
      if v4l2_camera is not None:
        camerad_module.Camera = v4l2_camera
        camerad_module._fork_cuda_camera = True
        _PATCH_STATE["camera"] = f"v4l2_dmabuf_camera.Camera ({import_path})"
        return True
    except Exception:
      continue
  _PATCH_STATE["camera"] = "v4l2_dmabuf_camera unavailable; keeping stock Camera"
  return False


def apply() -> list[str]:
  """Apply every patch, returns the names that changed something."""
  if not enabled():
    return []
  applied: list[str] = []
  hooks = [(f"modeld:{m}", m, patch_modeld_state) for m in MODELD_MODULES]
  hooks += [(f"camerad:{m}", m, patch_camerad_camera) for m in CAMERAD_MODULES]
  for label, module, patch in hooks:
    try:
      if patch_after_import(module, patch):
        applied.append(f"hook:{label}")
    except Exception:
      pass
  if applied and os.environ.get("CUDA_PATCH_VERBOSE", "0") == "1":
    # 默认不打日志: 这条分支会 import swaglog -> common.hardware 链, 而 apply() 在
    # manager 每个子进程的 interpreter 启动阶段就运行, 不该为一行日志付这个开销。
    # 需要看钩子是否装上时: CUDA_PATCH_VERBOSE=1 (或直接 --diagnose)。
    try:
      from openpilot.common.swaglog import cloudlog
      cloudlog.warning(f"cuda patches active: {applied} {_PATCH_STATE}")
    except Exception:
      pass
  return applied


def diagnose(tree_root: str | None = None) -> None:
  """Print every model bundle candidate and why it is or is not usable."""
  import json
  try:
    _, _, snapshot = _registry_module()
    if snapshot is None:
      print("no registry available (not an openpilot tree?)")
      return
    print(json.dumps(snapshot(), indent=2, ensure_ascii=False))
  except Exception as e:
    print(f"diagnose failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
  if len(sys.argv) > 1 and sys.argv[1] == "--diagnose":
    root = sys.argv[2] if len(sys.argv) > 2 else None
    if root:
      sys.path.insert(0, root)
      if os.path.isdir(os.path.join(root, "openpilot")):
        sys.path.insert(0, os.path.join(root, "openpilot"))
    diagnose(root)
  else:
    raise SystemExit(apply())