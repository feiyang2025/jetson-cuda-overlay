"""Tests for the fork-adaptive patch layer (cuda_adapter.py).

The adapter must never touch a fork whose ModelState it does not understand, and
must be idempotent/silent because it runs in every python process on the device.

Run with:  python3 -m unittest discover -s kits/patch/tests -v
"""
from __future__ import annotations

import builtins
import importlib
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

OVERLAY = Path(__file__).resolve().parents[3]
ADAPTER_SRC = OVERLAY / "kits" / "patch" / "cuda_adapter.py"

PKG_PREFIXES = ("openpilot", "tools", "selfdrive")


def _load_adapter():
  spec = importlib.util.spec_from_file_location("cuda_adapter_under_test", ADAPTER_SRC)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _stub_openpilot_modules(*names):
  """Install empty stubs so logging/import probes never touch a real tree.

  Without this, `apply()`'s "log what was patched" branch imports the real
  openpilot package (which pulls tinygrad on a checkout root) and the test stops
  being about the adapter.
  """
  created = {}
  for name in names:
    module = types.ModuleType(name)
    module.__path__ = []
    created[name] = module
  return created


class ConstructorShapeTests(unittest.TestCase):
  """Signature detection decides whether a fork gets the CUDA state at all."""

  def setUp(self):
    self.adapter = _load_adapter()
    import inspect
    self.inspect = inspect

  def _shape_of(self, fn):
    return self.adapter._constructor_shape(self.inspect.signature(fn))

  def test_sunnypilot_new_layout(self):
    def init(self, cam_w, cam_h, chestnut=False):
      pass
    self.assertEqual(self._shape_of(init), "stock")

  def test_master_c3_trt_entry_with_profile(self):
    def init(self, cam_w, cam_h, chestnut=False, profile=None):
      pass
    self.assertEqual(self._shape_of(init), "stock")

  def test_frogpilot_shape_is_unsupported(self):
    def init(self, context, model, model_version):
      pass
    self.assertIsNone(self._shape_of(init))

  def test_positional_variadic_stock_shape(self):
    def init(self, cam_w, cam_h, *args, **kwargs):
      pass
    self.assertEqual(self._shape_of(init), "stock")

  def test_garbage_signature_is_unsupported(self):
    self.assertIsNone(self.adapter._constructor_shape(None))


class PatchBehaviourTests(unittest.TestCase):
  def setUp(self):
    self.adapter = _load_adapter()
    self.adapter._PATCH_STATE.clear()
    self._env = mock.patch.dict(os.environ, {}, clear=False)
    self._env.start()
    for key in ("DISABLE_CUDA_BACKEND", "OPENPILOT_CUDA_PATCH"):
      os.environ.pop(key, None)
    self._module_snapshot = set(sys.modules)

  def tearDown(self):
    # Anything an import probe left behind belongs to the probe, not the test run.
    for name in set(sys.modules) - self._module_snapshot:
      if name.startswith(PKG_PREFIXES):
        del sys.modules[name]
    self._env.stop()

  def test_patch_after_import_fires_on_later_import(self):
    with tempfile.TemporaryDirectory() as tmp:
      pkg = Path(tmp) / "latepkg"
      pkg.mkdir()
      (pkg / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
      sys.path.insert(0, tmp)
      self.addCleanup(sys.path.remove, tmp)

      seen = []
      self.assertTrue(self.adapter.patch_after_import("latepkg.mod", lambda m: seen.append(m) or True))
      (pkg / "mod.py").write_text("X = 42\n", encoding="utf-8")
      importlib.import_module("latepkg.mod")
      self.addCleanup(sys.modules.pop, "latepkg.mod", None)
      self.addCleanup(sys.modules.pop, "latepkg", None)
      self.assertEqual(len(seen), 1)
      self.assertEqual(seen[0].X, 42)

  def test_frogpilot_modelstate_is_left_untouched(self):
    fake = types.ModuleType("fake_modeld")

    class ModelState:
      def __init__(self, context, model, model_version):
        self.touched = True

    fake.ModelState = ModelState
    self.assertFalse(self.adapter.patch_modeld_state(fake))
    self.assertIn("signature unsupported", self.adapter._PATCH_STATE["modeld"])
    self.assertTrue(ModelState("ctx", "model", "v1").touched)

  def test_patch_is_idempotent(self):
    fake = types.ModuleType("fake_modeld2")

    class ModelState:
      def __init__(self, cam_w, cam_h, chestnut=False):
        pass

    fake.ModelState = ModelState
    with mock.patch.dict(os.environ, {"DISABLE_CUDA_BACKEND": "1"}):
      self.assertFalse(self.adapter.patch_modeld_state(fake))
    self.assertTrue(self.adapter.patch_modeld_state(fake))
    self.assertFalse(self.adapter.patch_modeld_state(fake))  # already patched

  def test_module_without_modelstate_is_ignored(self):
    self.assertFalse(self.adapter.patch_modeld_state(types.ModuleType("empty")))

  def test_camera_patch_skipped_on_darwin(self):
    fake = types.ModuleType("fake_camerad")

    class Camera:
      pass

    fake.Camera = Camera
    with mock.patch.object(sys, "platform", "darwin"):
      self.assertFalse(self.adapter.patch_camerad_camera(fake))
    self.assertIs(fake.Camera, Camera)

  def test_camera_patch_without_v4l2_module_keeps_stock(self):
    fake = types.ModuleType("fake_camerad2")

    class Camera:
      pass

    fake.Camera = Camera

    real_import = builtins.__import__

    def no_v4l2(name, *args, **kwargs):
      if "v4l2_dmabuf_camera" in name:
        raise ImportError("not installed in this test")
      return real_import(name, *args, **kwargs)

    with mock.patch.object(builtins, "__import__", side_effect=no_v4l2):
      self.assertFalse(self.adapter.patch_camerad_camera(fake))
    self.assertIs(fake.Camera, Camera)
    self.assertIn("unavailable", self.adapter._PATCH_STATE["camera"])

  def test_kill_switch_disables_apply(self):
    with mock.patch.dict(os.environ, {"OPENPILOT_CUDA_PATCH": "0"}):
      self.assertEqual(self.adapter.apply(), [])

  def test_apply_registers_hooks_without_importing_targets(self):
    stubs = _stub_openpilot_modules("openpilot", "openpilot.common",
                                    "openpilot.common.swaglog")
    with mock.patch.dict(sys.modules, stubs):
      before = set(sys.modules)
      applied = self.adapter.apply()
      imported = {m for m in set(sys.modules) - before if m.startswith(PKG_PREFIXES)}
    self.assertTrue(applied, "apply() should register hooks")
    self.assertTrue(all(name.startswith("hook:") for name in applied), applied)
    self.assertEqual(imported, set(), f"apply() must not import fork modules: {imported}")


if __name__ == "__main__":
  unittest.main()
