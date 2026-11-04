"""Contract tests for the FrogPilot CUDA adapter (kits/patch/frogpilot).

What must hold: FrogPilot's modeld main() uses exactly four members of the model
object, and the adapter must present them over the overlay's GpuModelState whose
run() has a different dropped-frame contract. A stub GpuModelState + a stub
registry stand in for TensorRT, so this runs on any machine.

Run with:  python3 -m unittest discover -s kits/patch/tests -v
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

OVERLAY = Path(__file__).resolve().parents[3]
FP_ADAPTER = OVERLAY / "kits" / "patch" / "frogpilot" / "fp_gpu_state.py"

STATE_MODULE = "openpilot.sunnypilot.modeld_v2.gpu_model_state"
REGISTRY_MODULE = "openpilot.sunnypilot.modeld_v2.gpu_backend.models.registry"


class FakeProfile:
  """The parts of a ModelProfile this adapter reads."""

  def __init__(self, name="FiletOFish", mode="split", vision_names=("input_imgs", "big_input_imgs"),
               desire_key="desire", lateral=True, desire_len=8):
    self.name = name
    self.mode = mode
    self._vision_names = list(vision_names)
    self._desire_key = desire_key
    self.input_shapes = {
      desire_key: (1, 25, desire_len),
      "traffic_convention": (1, 2),
      "features_buffer": (1, 25, 512),
    }
    if lateral:
      self.input_shapes["lateral_control_params"] = (1, 2)

  @property
  def model_w(self):
    return 512

  @property
  def model_h(self):
    return 256


class FakeGpuState:
  """Stand-in for the overlay's GpuModelState."""

  created: list["FakeGpuState"] = []
  resolve_args: list = []

  def __init__(self, cam_w, cam_h, chestnut, profile):
    self.cam_w, self.cam_h, self.chestnut, self.profile = cam_w, cam_h, chestnut, profile
    self.vision_input_names = list(profile._vision_names)
    self.run_calls = []
    self.preprocess_calls = []
    self.warmed = False
    self.backend = types.SimpleNamespace(preprocess=self._preprocess)
    FakeGpuState.created.append(self)

  @property
  def desire_key(self):
    return self.profile._desire_key

  def _preprocess(self, frames, transforms):
    self.preprocess_calls.append((frames, transforms))

  def warmup(self):
    self.warmed = True

  def run(self, bufs, transforms, inputs, after_enqueue=None):
    self.run_calls.append(dict(bufs=bufs, transforms=transforms, inputs=inputs,
                               after_enqueue=after_enqueue))
    return {"plan": np.zeros((1, 33, 6), dtype=np.float32)}


class FakeRegistry:
  def __init__(self, profile=None):
    self.profile = profile if profile is not None else FakeProfile()
    self.calls = []

  def resolve_profile(self, name=None, root=None, **kw):
    self.calls.append({"name": name, "root": root})
    return self.profile


def _load_adapter():
  spec = importlib.util.spec_from_file_location("fp_gpu_state_under_test", FP_ADAPTER)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  module._state_cls = None
  module._registry = None
  return module


class FrogPilotAdapterTests(unittest.TestCase):
  def setUp(self):
    FakeGpuState.created.clear()
    self.adapter = _load_adapter()
    self._saved = {n: sys.modules.get(n) for n in (STATE_MODULE, REGISTRY_MODULE)}
    self._tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self._tmp.cleanup)
    self.models = Path(self._tmp.name) / "models"
    self.bundle = self.models / "FiletOFish"
    self.bundle.mkdir(parents=True)
    (self.bundle / "driving_vision_metadata.pkl").write_bytes(b"x")

  def tearDown(self):
    for name, module in self._saved.items():
      if module is None:
        sys.modules.pop(name, None)
      else:
        sys.modules[name] = module

  def _install(self, profile=None):
    state_mod = types.ModuleType(STATE_MODULE)
    state_mod.GpuModelState = FakeGpuState
    registry = FakeRegistry(profile)
    reg_mod = types.ModuleType(REGISTRY_MODULE)
    reg_mod.resolve_profile = registry.resolve_profile
    sys.modules[STATE_MODULE] = state_mod
    sys.modules[REGISTRY_MODULE] = reg_mod
    self.adapter._state_cls = None
    self.adapter._registry = None
    return registry

  def _make(self, model="FiletOFish", env_dir=None):
    env = {"FP_CUDA_MODEL_DIR": str(env_dir or self.models)}
    with mock.patch.dict(os.environ, env):
      return self.adapter.FrogPilotGpuState(None, model)

  # --- the four members FrogPilot's main() reads ---------------------------

  def test_exposes_frogpilot_contract(self):
    self._install()
    state = self._make()
    self.assertEqual(state.vision_input_names, ["input_imgs", "big_input_imgs"])
    self.assertEqual(state.desire_type, "desire")
    self.assertTrue(state.use_lateral_control_params)
    self.assertEqual(state.desire_len, 8)
    self.assertFalse(state.chestnut)
    self.assertFalse(state.desired_curvature_head)

  def test_desire_type_follows_the_bundle(self):
    # A merged bundle names the input desire_pulse; FrogPilot keys its inputs
    # dict by this attribute, so reporting the wrong name would feed the model
    # zeros forever.
    self._install(FakeProfile(desire_key="desire_pulse"))
    state = self._make()
    self.assertEqual(state.desire_type, "desire_pulse")
    self.assertEqual(state.desire_len, 8)

  def test_missing_lateral_params_is_reported(self):
    self._install(FakeProfile(lateral=False))
    state = self._make()
    self.assertFalse(state.use_lateral_control_params)

  def test_profile_drives_cam_dimensions_and_warmup(self):
    self._install()
    state = self._make()
    core = state._core
    self.assertEqual((core.cam_w, core.cam_h), (512, 256))
    self.assertTrue(core.warmed, "modeld warms up before the first frame")

  # --- selection ----------------------------------------------------------

  def test_bundle_name_matching_a_directory_is_requested(self):
    registry = self._install()
    self._make("FiletOFish")
    self.assertEqual(registry.calls[0]["name"], "FiletOFish")

  def test_foreign_frogpilot_name_defers_to_registry(self):
    # FrogPilot's catalogue names (dark-souls, ...) are not overlay bundles.
    registry = self._install()
    self._make("dark-souls")
    self.assertIsNone(registry.calls[0]["name"])

  def test_bundle_env_override_wins(self):
    registry = self._install()
    with mock.patch.dict(os.environ, {"FP_CUDA_MODEL_DIR": str(self.models),
                                      "FP_CUDA_BUNDLE": "TGC"}):
      self.adapter.FrogPilotGpuState(None, "dark-souls")
    self.assertEqual(registry.calls[0]["name"], "TGC")

  def test_no_usable_profile_raises(self):
    self._install()
    sys.modules[REGISTRY_MODULE].resolve_profile = lambda *a, **k: None
    self.adapter._registry = None
    with self.assertRaises(RuntimeError):
      self._make()

  def test_missing_gpu_state_module_raises(self):
    sys.modules.pop(STATE_MODULE, None)
    self.adapter._state_cls = False
    with self.assertRaises(RuntimeError):
      self.adapter.FrogPilotGpuState(None, "FiletOFish")

  # --- run() shim ---------------------------------------------------------

  def test_run_forwards_dicts_and_never_passes_prepare_only_as_callback(self):
    self._install()
    state = self._make()
    bufs = {"input_imgs": "ROAD", "big_input_imgs": "WIDE"}
    tfms = {"input_imgs": np.eye(3), "big_input_imgs": np.eye(3)}
    out = state.run(bufs, tfms, {"desire": np.zeros(8)}, False)
    call = state._core.run_calls[0]
    self.assertEqual(call["bufs"], bufs)
    self.assertEqual(call["transforms"], tfms)
    # GpuModelState's 4th argument is a callback, not a bool: passing the bool
    # through would call it and crash modeld on every frame.
    self.assertIsNone(call["after_enqueue"])
    self.assertIn("plan", out)

  def test_prepare_only_preprocesses_without_inference(self):
    self._install()
    state = self._make()
    bufs = {"input_imgs": "R", "big_input_imgs": "W"}
    tfms = {"input_imgs": np.eye(3), "big_input_imgs": np.eye(3)}
    self.assertIsNone(state.run(bufs, tfms, {"desire": np.zeros(8)}, True))
    self.assertEqual(len(state._core.preprocess_calls), 1)
    self.assertEqual(state._core.preprocess_calls[0][0], bufs)
    self.assertEqual(state._core.run_calls, [], "a dropped frame must not run the model")

  def test_prepare_only_ignores_missing_frames(self):
    self._install()
    state = self._make()
    state.run({"input_imgs": "ONLY"}, {"input_imgs": np.eye(3)}, {"desire": np.zeros(8)}, True)
    self.assertEqual(state._core.preprocess_calls[0][0], {"input_imgs": "ONLY"})


class PatchTinygradModeldTests(unittest.TestCase):
  def setUp(self):
    self.adapter = _load_adapter()

  def _fake_module(self):
    module = types.ModuleType("frogpilot.selfdrive.modeld.tinygrad.tinygrad_modeld")

    class ModelState:
      def __init__(self, context, model):
        pass

    class CLContext:
      pass

    module.ModelState = ModelState
    module.CLContext = CLContext
    return module

  def test_swaps_modelstate_and_stubs_cl_context(self):
    module = self._fake_module()
    self.assertTrue(self.adapter.patch_tinygrad_modeld(module))
    self.assertIs(module.ModelState, self.adapter.FrogPilotGpuState)
    self.assertIsNotNone(module.CLContext())
    self.assertFalse(hasattr(module.CLContext, "device_id"))

  def test_idempotent(self):
    module = self._fake_module()
    self.assertTrue(self.adapter.patch_tinygrad_modeld(module))
    self.assertFalse(self.adapter.patch_tinygrad_modeld(module))

  def test_module_without_modelstate_is_ignored(self):
    self.assertFalse(self.adapter.patch_tinygrad_modeld(types.ModuleType("empty")))


if __name__ == "__main__":
  unittest.main()