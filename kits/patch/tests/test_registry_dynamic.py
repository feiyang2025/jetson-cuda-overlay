"""Tests for the dynamic bundle registry (models/<Name>/ scan + ordered selection).

Run with:  python3 -m unittest discover -s kits/patch/tests -v
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import pickle
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

OVERLAY = Path(__file__).resolve().parents[3]
REGISTRY_SRC = OVERLAY / "openpilot/sunnypilot/modeld_v2/gpu_backend/models/registry.py"
PROFILE_SRC = OVERLAY / "openpilot/sunnypilot/modeld_v2/gpu_backend/profile.py"

PACKAGE_PATH = ("openpilot", "openpilot.sunnypilot", "openpilot.sunnypilot.modeld_v2",
                "openpilot.sunnypilot.modeld_v2.gpu_backend",
                "openpilot.sunnypilot.modeld_v2.gpu_backend.models")


def _load_registry(tmp: Path):
  """Import registry.py under a throwaway package rooted at `tmp`.

  The package is built in sys.modules explicitly instead of relying on sys.path
  order: running the tests from a checkout root otherwise lets the *real*
  openpilot package win (and drag tinygrad in), which makes the test depend on
  what happens to sit in the working directory. Every fake module is removed
  again afterwards, so the adapter tests still import a pristine environment.
  """
  dirs = {
    "openpilot": tmp / "openpilot",
    "openpilot.sunnypilot": tmp / "openpilot" / "sunnypilot",
    "openpilot.sunnypilot.modeld_v2": tmp / "openpilot" / "sunnypilot" / "modeld_v2",
    "openpilot.sunnypilot.modeld_v2.gpu_backend":
      tmp / "openpilot" / "sunnypilot" / "modeld_v2" / "gpu_backend",
    "openpilot.sunnypilot.modeld_v2.gpu_backend.models":
      tmp / "openpilot" / "sunnypilot" / "modeld_v2" / "gpu_backend" / "models",
  }
  for path in dirs.values():
    path.mkdir(parents=True, exist_ok=True)

  (dirs["openpilot.sunnypilot.modeld_v2.gpu_backend"] / "profile.py").write_text(
    PROFILE_SRC.read_text(encoding="utf-8"), encoding="utf-8")
  (dirs["openpilot.sunnypilot.modeld_v2.gpu_backend.models"] / "registry.py").write_text(
    REGISTRY_SRC.read_text(encoding="utf-8"), encoding="utf-8")

  saved = {name: sys.modules[name] for name in PACKAGE_PATH if name in sys.modules}
  try:
    for name in PACKAGE_PATH:
      module = types.ModuleType(name)
      module.__path__ = [str(dirs[name])]
      module.__package__ = name
      sys.modules[name] = module
    importlib.invalidate_caches()
    return importlib.import_module("openpilot.sunnypilot.modeld_v2.gpu_backend.models.registry")
  finally:
    for name in PACKAGE_PATH:
      sys.modules.pop(name, None)
    sys.modules.update(saved)


def _metadata(merged: bool, features_windows: int = 24, output_width: int = 2580):
  if merged:
    return {
      "input_shapes": {
        "img": (1, 12, 128, 256),
        "big_img": (1, 12, 128, 256),
        "desire_pulse": (1, 25, 8),
        "traffic_convention": (1, 2),
        "features_buffer": (1, features_windows, 512),
      },
      "output_slices": {
        "lane_lines": slice(0, 528),
        "plan": slice(917, 1907),
        "lead": slice(1907, 2051),
        "hidden_state": slice(2066, output_width - 2),
        "pad": slice(output_width - 2, output_width),
      },
    }
  return {
    "input_shapes": {
      "desire": (1, 25, 8),
      "traffic_convention": (1, 2),
      "features_buffer": (1, features_windows, 512),
    },
    "output_slices": {
      "lane_lines": slice(0, 528),
      "plan": slice(917, 1907),
      "lead": slice(1907, 2051),
      "hidden_state": slice(2066, 2578),
    },
  }


def _write_bundle(root: Path, name: str, kind: str, *, engines: bool = True,
                  zero_byte_fp16: bool = False) -> Path:
  d = root / name
  d.mkdir(parents=True, exist_ok=True)
  if kind == "merged":
    (d / "driving_supercombo_metadata.pkl").write_bytes(pickle.dumps(_metadata(True)))
    if engines:
      (d / "driving_supercombo_fp16.plan").write_bytes(b"plan")
  else:
    vision_meta = {"input_shapes": {"input_imgs": (1, 12, 128, 256),
                                    "big_input_imgs": (1, 12, 128, 256)},
                   "output_slices": _metadata(False)["output_slices"]}
    (d / "driving_vision_metadata.pkl").write_bytes(pickle.dumps(vision_meta))
    (d / "driving_policy_metadata.pkl").write_bytes(pickle.dumps(_metadata(False)))
    if engines:
      (d / "driving_vision_fp16.plan").write_bytes(b"vision")
      if zero_byte_fp16:
        (d / "driving_policy_fp16.plan").write_bytes(b"")
        (d / "driving_policy.plan").write_bytes(b"policy")
      else:
        (d / "driving_policy_fp16.plan").write_bytes(b"policy")
  return d


class RegistryTests(unittest.TestCase):
  def setUp(self):
    self._tmp = tempfile.TemporaryDirectory()
    tmp = Path(self._tmp.name)
    self.models = tmp / "openpilot" / "selfdrive" / "modeld" / "models"
    self.models.mkdir(parents=True)
    self.reg = _load_registry(tmp)
    # Curated entries are built from the overlay's own (absent) models dir at
    # import; start empty so each test exercises scanning deterministically.
    self.reg.REGISTRY.clear()
    self.reg._SCAN_CACHE.update(key=None, profiles={})
    self.reg.MODELS_ROOT = self.models

  def tearDown(self):
    self._tmp.cleanup()

  # --- scan ---------------------------------------------------------------

  def test_scan_finds_split_and_merged(self):
    _write_bundle(self.models, "Alpha", "split")
    _write_bundle(self.models, "Beta", "merged")
    found = self.reg.scan_profiles(refresh=True)
    self.assertEqual(sorted(found), ["Alpha", "Beta"])
    self.assertEqual(found["Alpha"].mode, "split")
    self.assertEqual(found["Beta"].mode, "merged")
    self.assertTrue(found["Alpha"].is_ready())
    self.assertTrue(found["Beta"].is_ready())

  def test_scan_reads_temporal_geometry_from_metadata(self):
    _write_bundle(self.models, "Beta", "merged")
    profile = self.reg.scan_profiles(refresh=True)["Beta"]
    self.assertEqual(profile.temporal.features_windows, 24)
    self.assertEqual(profile.temporal.desire_windows, 25)
    self.assertEqual(profile.temporal.features_len, 512)
    # The image tensor must not be treated as a scalar input, or inference runs
    # on a black frame (see _scalars docstring).
    self.assertNotIn("img", profile.scalar_input_shapes)
    self.assertIn("features_buffer", profile.scalar_input_shapes)

  def test_scan_ignores_non_model_directories(self):
    (self.models / "not_a_model").mkdir()
    self.reg.write_order(["Alpha"])
    self.assertEqual(self.reg.scan_profiles(refresh=True), {})

  def test_incomplete_bundle_is_scanned_but_not_ready(self):
    _write_bundle(self.models, "Half", "split", engines=False)
    found = self.reg.scan_profiles(refresh=True)
    self.assertIn("Half", found)
    self.assertFalse(found["Half"].is_ready())
    self.assertIn("engine missing", found["Half"].missing_reason())

  def test_zero_byte_engine_falls_back_to_next_candidate(self):
    _write_bundle(self.models, "Alpha", "split", zero_byte_fp16=True)
    profile = self.reg.scan_profiles(refresh=True)["Alpha"]
    self.assertEqual(profile.policy_engine, "driving_policy.plan")
    self.assertTrue(profile.is_ready())

  # --- selection -----------------------------------------------------------

  def test_order_file_wins_over_alphabetical(self):
    _write_bundle(self.models, "Alpha", "split")
    _write_bundle(self.models, "Beta", "merged")
    self.reg.write_order(["Beta", "Alpha"])
    self.assertEqual(self.reg.resolve_profile().name, "Beta")
    self.assertEqual(self.reg.resolve_profile("Alpha").name, "Alpha")
    self.assertEqual(self.reg.read_order(), ["Beta", "Alpha"])

  def test_falls_back_to_sorted_scan_without_order_file(self):
    _write_bundle(self.models, "Zeta", "merged")
    _write_bundle(self.models, "Alpha", "split")
    self.assertEqual(self.reg.resolve_profile().name, "Alpha")

  def test_unready_bundle_is_skipped(self):
    _write_bundle(self.models, "Aaa", "split", engines=False)
    _write_bundle(self.models, "Zzz", "merged")
    self.assertEqual(self.reg.resolve_profile().name, "Zzz")
    # A broken named model must not take modeld down: another one is used...
    self.assertEqual(self.reg.resolve_profile("Aaa").name, "Zzz")
    # ...unless the caller explicitly asks for "this one or nothing".
    self.assertIsNone(self.reg.resolve_profile("Aaa", strict=True))
    self.assertIsNone(self.reg.resolve_profile("Nope", strict=True))

  def test_scan_can_be_disabled(self):
    _write_bundle(self.models, "Alpha", "split")
    self.reg._SCAN_CACHE.update(key=None, profiles={})
    old = os.environ.get("DISABLE_GPU_REGISTRY_SCAN")
    os.environ["DISABLE_GPU_REGISTRY_SCAN"] = "1"
    try:
      self.assertEqual(self.reg.scan_profiles(refresh=True), {})
      self.assertIsNone(self.reg.resolve_profile())
    finally:
      if old is None:
        os.environ.pop("DISABLE_GPU_REGISTRY_SCAN", None)
      else:
        os.environ["DISABLE_GPU_REGISTRY_SCAN"] = old

  def test_curated_profile_keeps_its_name_over_scanned_bundle(self):
    _write_bundle(self.models, "Alpha", "split")
    scanned = self.reg.scan_profiles(refresh=True)["Alpha"]
    self.reg.REGISTRY["Alpha"] = scanned
    _write_bundle(self.models, "Alpha", "merged")  # same name on disk, other kind
    self.assertIs(self.reg.get_profile("Alpha"), scanned)
    self.reg.REGISTRY.clear()

  # --- failure memory ------------------------------------------------------

  def test_failed_bundle_is_skipped_until_cooldown(self):
    _write_bundle(self.models, "Alpha", "split")
    _write_bundle(self.models, "Beta", "merged")
    self.reg.write_order(["Alpha", "Beta"])
    profile = self.reg.scan_profiles(refresh=True)["Alpha"]
    self.reg.mark_failed(profile, "cudaErrorIllegalAddress")

    self.assertEqual(self.reg.resolve_profile().name, "Beta")
    self.assertIn("cudaErrorIllegalAddress", self.reg.failure_reason("Alpha", profile) or "")

    # Expire the marker the way the cooldown would, without waiting.
    self.reg.FAILED_COOLDOWN_S = -1.0
    try:
      self.assertEqual(self.reg.resolve_profile().name, "Alpha")
    finally:
      self.reg.FAILED_COOLDOWN_S = 900.0

  def test_rebuilt_engine_clears_a_fresh_marker(self):
    d = _write_bundle(self.models, "Alpha", "split")
    profile = self.reg.scan_profiles(refresh=True)["Alpha"]
    self.reg.mark_failed(profile, "boom")

    future = time.time() + 120
    for engine in (d / "driving_vision_fp16.plan", d / "driving_policy_fp16.plan"):
      os.utime(engine, (future, future))
    refreshed = self.reg.scan_profiles(refresh=True)["Alpha"]
    self.assertIsNone(self.reg.failure_reason("Alpha", refreshed))
    self.assertEqual(self.reg.resolve_profile().name, "Alpha")

  def test_clear_failed(self):
    _write_bundle(self.models, "Alpha", "split")
    profile = self.reg.scan_profiles(refresh=True)["Alpha"]
    self.reg.mark_failed(profile, "boom")
    self.assertIsNotNone(self.reg.failure_reason("Alpha", profile))
    self.reg.clear_failed("Alpha")
    self.assertIsNone(self.reg.failure_reason("Alpha", profile))

  # --- snapshot ------------------------------------------------------------

  def test_snapshot_reports_every_candidate(self):
    _write_bundle(self.models, "Alpha", "split")
    _write_bundle(self.models, "Beta", "merged", engines=False)
    rows = {r["name"]: r for r in self.reg.snapshot()}
    self.assertEqual(sorted(rows), ["Alpha", "Beta"])
    self.assertTrue(rows["Alpha"]["ready"])
    self.assertTrue(rows["Alpha"]["selected"])
    self.assertFalse(rows["Beta"]["ready"])
    self.assertIn("engine missing", rows["Beta"]["reason"])


if __name__ == "__main__":
  unittest.main()
