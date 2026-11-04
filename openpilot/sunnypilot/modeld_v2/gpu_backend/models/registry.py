"""Model profile registry: curated profiles + on-disk bundle discovery.

Two layers, on purpose:

* curated factories (`_supercombo`, `_big_combo`, `_tgc`, `_filet_ofish`) know the
  quirks of the models this port was developed against (engine binding names,
  fp32-only supercombo, TGC's unusual temporal geometry). They stay because the
  names are used in env/Params/UI and must keep resolving.
* `scan_profiles()` builds a profile from whatever `models/<Name>/` directories
  exist, reading shapes, slices and temporal geometry out of the metadata pkl.
  That is what makes a newly compiled bundle usable without editing this file:
  `engine_build.py` writes `<models>/<Name>/{driving_vision_fp16.plan,...}` and
  a `*_metadata.pkl`, which is all the scan needs.

Selection (`resolve_profile`) is ordered, never "first folder alphabetically":

    explicit name -> MODEL_NAME env -> .trt_order file -> Params "Model"
    -> curated order -> scanned bundles sorted by name

A bundle that failed to initialise is remembered in `models/.trt_failed` and
skipped until the cooldown expires or its engines are rebuilt, so a bad plan
cannot turn modeld into a restart loop. `snapshot()` reports every candidate and
why it is or is not usable, for logs and for the UI.

Nothing here imports openpilot modules beyond the profile dataclasses, so this
file is unit-testable with a temporary models directory (see
kits/patch/tests/test_registry_dynamic.py).
"""
import json
import logging
import os
import pickle
import time
from pathlib import Path

from openpilot.sunnypilot.modeld_v2.gpu_backend.profile import ModelProfile, TemporalMeta

log = logging.getLogger(__name__)

MODELS_ROOT = Path(__file__).parents[4] / "selfdrive" / "modeld" / "models"

# Output heads produced by the policy stage of a split bundle. Everything else in
# a merged supercombo output belongs to the vision stage.
POLICY_KEYS = frozenset((
  "plan",
  "planplus",
  "lead",
  "lead_prob",
  "desire_state",
  "action",
  "hidden_state",
  "pad",
))

VISION_ENGINE_SHAPES = {
  # BigCombo (merged) 引擎的绑定名
  "img": (1, 12, 128, 256),
  "big_img": (1, 12, 128, 256),
  # FiletOFish (split) 引擎实测绑定名 —— engine 里就叫 input_imgs/big_input_imgs。
  # TrtRunner 按引擎实际绑定名取 kwargs (trt_runner.py:196), 所以每个 profile 必须
  # 用自己 engine 的真实名字, 否则 kwargs.get('img') 拿到 None 直接报错。
  "input_imgs": (1, 12, 128, 256),
  "big_input_imgs": (1, 12, 128, 256),
}

# Engine file names a bundle can carry. Order is preference: fp16 first, then a
# plain/fp32 build. A zero-byte plan is a trtexec run that died partway through
# and must not win the search (see `_engine`).
MERGED_ENGINE_CANDIDATES = ("driving_supercombo_fp16.plan", "driving_supercombo_fp32.plan",
                            "driving_supercombo.plan", "big_driving_supercombo_fp16.plan")
SPLIT_VISION_ENGINES = ("driving_vision_fp16.plan", "driving_vision_fp32.plan", "driving_vision.plan")
SPLIT_POLICY_ENGINES = ("driving_policy_fp16.plan", "driving_policy_fp32.plan", "driving_policy.plan")

MERGED_METADATA_NAMES = ("driving_supercombo_metadata.pkl", "big_driving_supercombo_metadata.pkl")
SPLIT_VISION_METADATA = "driving_vision_metadata.pkl"
SPLIT_POLICY_METADATA = "driving_policy_metadata.pkl"

# A directory without these is not selectable at all (the same rule StarPilot
# applies with REQUIRED_FILES), which keeps an interrupted copy or download from
# shadowing a working model.
REQUIRED_SPLIT_FILES = (SPLIT_VISION_METADATA, SPLIT_POLICY_METADATA)

ORDER_FILE = MODELS_ROOT / ".trt_order"
FAILED_FILE = MODELS_ROOT / ".trt_failed"
FAILED_COOLDOWN_S = float(os.getenv("TRT_FAIL_COOLDOWN_S", "900"))

# Curated default preference, used when nothing else asks for a model.
# `supercombo` is the small model the bundle ships and is what runs unless a big
# model is asked for by name; TGC is that big model. FiletOFish is a different
# split model (632 + 5884) kept for the profile project3 ran on the Orin - it is
# NOT the bundle's small model, so it stays last.
CURATED_ORDER = ("supercombo", "BigCombo", "TGC", "FiletOFish")


class MetadataError(RuntimeError):
  pass


def _load_metadata(metadata_path: Path) -> dict:
  if not metadata_path.is_file():
    raise MetadataError(f"metadata not found: {metadata_path}")
  try:
    with open(metadata_path, "rb") as f:
      metadata = pickle.load(f)
  except Exception as e:
    raise MetadataError(f"cannot unpickle {metadata_path}: {e}") from e
  if not isinstance(metadata, dict):
    raise MetadataError(f"{metadata_path} holds {type(metadata).__name__}, expected dict")
  return metadata


def _output_slices(metadata: dict, source: Path) -> dict[str, slice]:
  slices = metadata.get("output_slices")
  if not isinstance(slices, dict) or not slices:
    raise MetadataError(f"{source} has no usable output_slices (got {type(slices).__name__})")
  bad = [k for k, v in slices.items() if not isinstance(v, slice)]
  if bad:
    raise MetadataError(f"{source} output_slices has non-slice entries: {bad}")
  return slices


def _scalars(shapes: dict, vision_names) -> dict:
  """Metadata input_shapes lists the image tensors too.

  Those are not scalars: they are filled by the CUDA transform, which hands the
  backend a device pointer. Leaving them in this dict makes callers build a zero
  array for them and then overwrite the pointer, so every inference silently runs
  on a black image. They are dropped here once instead of at each call site.
  """
  return {k: v for k, v in shapes.items() if k not in set(vision_names)}


def _split_slices(slices: dict[str, slice]) -> tuple[dict[str, slice], dict[str, slice]]:
  vision = {k: v for k, v in slices.items() if k not in POLICY_KEYS}
  policy = {k: v for k, v in slices.items() if k in POLICY_KEYS}
  if not vision or not policy:
    raise MetadataError(
      f"cannot split output_slices: vision={sorted(vision)} policy={sorted(policy)}; "
      f"POLICY_KEYS may not match this engine"
    )
  return vision, policy


def np_prod(values) -> int:
  total = 1
  for v in values:
    total *= int(v)
  return total


def _temporal_from_metadata(metadata: dict, source: Path, features_key: str | None = None,
                            desire_key: str | None = None) -> TemporalMeta:
  """Temporal geometry from the policy metadata.

  Key names differ between checkpoints (`desire` / `desire_pulse`), so both are
  accepted and the caller may pass an explicit choice. Axis 1 is the time axis
  for both (confirmed from the ONNX graph: features_buffer is consumed by
  Concat(axis=1), desire by Gather(axis=1)).
  """
  shapes = metadata.get("input_shapes")
  if not isinstance(shapes, dict):
    raise MetadataError(f"{source} has no input_shapes")

  features_key = features_key or next((k for k in ("features_buffer",) if k in shapes), None)
  if features_key is None:
    raise MetadataError(f"{source} input_shapes has no features_buffer; has {sorted(shapes)}")
  desire_key = desire_key or next((k for k in ("desire_pulse", "desire") if k in shapes), None)
  if desire_key is None:
    raise MetadataError(f"{source} input_shapes has no desire/desire_pulse; has {sorted(shapes)}")

  fb = tuple(int(v) for v in shapes[features_key])
  dr = tuple(int(v) for v in shapes[desire_key])
  if len(fb) < 2 or len(dr) < 2:
    raise MetadataError(f"{source} expects >=2D {features_key}={fb} and {desire_key}={dr}")
  features_len = int(np_prod(fb[2:])) if len(fb) > 2 else fb[1]
  features_windows = fb[1]
  desire_len = int(np_prod(dr[2:])) if len(dr) > 2 else dr[1]
  desire_windows = dr[1]
  if features_windows < 2:
    raise MetadataError(f"{source} features_buffer has {features_windows} time slots, needs >= 2 to shift")
  return TemporalMeta(
    features_len=features_len,
    features_windows=features_windows,
    desire_len=desire_len,
    desire_windows=desire_windows,
    features_includes_current=False,
    desire_includes_current=True,
  )


def _engine(engine_dir: Path, *names: str) -> str | None:
  # A zero-byte engine is a trtexec run that died partway through; it must not win
  # the candidate search, or it shadows a perfectly good plan sitting later in the
  # list and the profile just reports itself unready.
  for name in names:
    path = engine_dir / name
    if path.is_file() and path.stat().st_size > 0:
      return name
  return None


def _vision_names_from_shapes(shapes: dict) -> list[str]:
  """Vision tensor names of a merged engine, from its metadata.

  Known names win (they are what the engines actually bind); otherwise any 4-D
  input is treated as an image. This is what lets a new checkpoint work without
  being added to VISION_ENGINE_SHAPES.
  """
  known = [k for k in VISION_ENGINE_SHAPES if k in shapes]
  if known:
    return known
  return [k for k, v in shapes.items() if isinstance(v, (tuple, list)) and len(v) == 4]


# --- curated profiles -------------------------------------------------------

def _filet_ofish() -> ModelProfile:
  engine_dir = MODELS_ROOT / "FiletOFish"
  vision_engine = _engine(engine_dir, *SPLIT_VISION_ENGINES)
  policy_engine = _engine(engine_dir, *SPLIT_POLICY_ENGINES)
  if vision_engine is None or policy_engine is None:
    raise MetadataError(f"FiletOFish: missing split engines in {engine_dir}")

  vision_meta = _load_metadata(engine_dir / SPLIT_VISION_METADATA)
  policy_meta = _load_metadata(engine_dir / SPLIT_POLICY_METADATA)

  return ModelProfile(
    name="FiletOFish",
    mode="split",
    engine_dir=str(engine_dir),
    vision_engine=vision_engine,
    policy_engine=policy_engine,
    vision_input_names=["input_imgs", "big_input_imgs"],
    input_shapes=dict(VISION_ENGINE_SHAPES),
    vision_slices=_output_slices(vision_meta, engine_dir / SPLIT_VISION_METADATA),
    policy_slices=_output_slices(policy_meta, engine_dir / SPLIT_POLICY_METADATA),
    temporal=_temporal_from_metadata(policy_meta, engine_dir / SPLIT_POLICY_METADATA,
                                     "features_buffer", "desire"),
    scalar_input_shapes=_scalars(policy_meta.get("input_shapes") or {}, ["input_imgs", "big_input_imgs"]),
  )


def _merged_profile(name: str, engine_dir: Path, metadata_name: str | None = None) -> ModelProfile:
  # A missing engine is NOT an error here: metadata may be in place while trtexec
  # is still building, and `missing_reason()` reports that as not-ready so the
  # bundle stays visible (and selectable the moment the plan lands).
  merged_engine = _engine(engine_dir, *MERGED_ENGINE_CANDIDATES)

  if metadata_name is None:
    for candidate in MERGED_METADATA_NAMES:
      if (engine_dir / candidate).is_file():
        metadata_name = candidate
        break
  if metadata_name is None:
    raise MetadataError(f"{name}: no supercombo metadata in {engine_dir}")
  metadata_path = engine_dir / metadata_name
  metadata = _load_metadata(metadata_path)
  slices = _output_slices(metadata, metadata_path)
  vision, policy = _split_slices(slices)

  shapes = metadata.get("input_shapes") or {}
  if not shapes:
    raise MetadataError(f"{metadata_path} has no input_shapes; cannot size the scalar inputs")
  vision_names = _vision_names_from_shapes(shapes)

  return ModelProfile(
    name=name,
    mode="merged",
    engine_dir=str(engine_dir),
    merged_engine=merged_engine,
    vision_input_names=vision_names,
    input_shapes={**VISION_ENGINE_SHAPES, **shapes},
    vision_slices=vision,
    policy_slices=policy,
    temporal=_temporal_from_metadata(metadata, metadata_path),
    scalar_input_shapes=_scalars(shapes, vision_names),
  )


def _big_combo() -> ModelProfile:
  return _merged_profile("BigCombo", MODELS_ROOT / "BigCombo")


def _supercombo() -> ModelProfile:
  # The small model the bundle actually ships (driving_supercombo.onnx): one merged
  # engine, FLOAT16 img inputs after the UINT8 rewrite, features_buffer (1, 24, 512),
  # desire_pulse (1, 25, 8), output 2576 wide. Same layout as BigCombo/TGC, so this
  # is that code path with a different directory.
  # No _fp16 engine on purpose: the 20 Gelu(approximate=0) nodes have no fp16
  # implementation on sm_87, so trtexec fails with "Could not find any
  # implementation for node_conv2d_1 + node_gelu_1". The fp32 engine builds and
  # runs at 14.49 ms / 69 Hz, well inside the 20 Hz budget.
  return _merged_profile("supercombo", MODELS_ROOT / "supercombo")


def _tgc() -> ModelProfile:
  # The big model sunnypilot currently publishes as `big_driving_supercombo.onnx`
  # is TGC-shaped, not the older BigCombo: desire_pulse is (1, 33, 8),
  # features_buffer is (1, 32, 32, 512) and the output is 18452 wide. Same input
  # names and the same merged single-engine layout, so it reuses the BigCombo
  # code path; only the temporal geometry differs and that is read from metadata.
  return _merged_profile("TGC", MODELS_ROOT / "TGC")


def _build(name: str, factory) -> ModelProfile | None:
  try:
    return factory()
  except Exception as e:
    log.warning(f"profile {name} unusable: {e}")
    return None


REGISTRY: dict[str, ModelProfile | None] = {
  "supercombo": _build("supercombo", _supercombo),
  "BigCombo": _build("BigCombo", _big_combo),
  "TGC": _build("TGC", _tgc),
  "FiletOFish": _build("FiletOFish", _filet_ofish),
}


# --- dynamic bundle discovery ----------------------------------------------

def _bundle_kind(bundle_dir: Path) -> str | None:
  """'split', 'merged', or None when the directory holds no model at all."""
  if all((bundle_dir / f).is_file() for f in REQUIRED_SPLIT_FILES):
    return "split"
  if any((bundle_dir / n).is_file() for n in MERGED_METADATA_NAMES):
    return "merged"
  for candidate in bundle_dir.glob("*supercombo_metadata*.pkl"):
    if candidate.is_file():
      return "merged"
  return None


def bundle_from_dir(bundle_dir: Path) -> ModelProfile | None:
  """Build a profile from an on-disk bundle, or None when it is not a model.

  Raises nothing: a directory that cannot be understood reports itself through
  `snapshot()` instead of taking the registry down.
  """
  kind = _bundle_kind(bundle_dir)
  if kind is None:
    return None
  name = bundle_dir.name

  if kind == "merged":
    return _merged_profile(name, bundle_dir)

  vision_engine = _engine(bundle_dir, *SPLIT_VISION_ENGINES)
  policy_engine = _engine(bundle_dir, *SPLIT_POLICY_ENGINES)
  vision_meta = _load_metadata(bundle_dir / SPLIT_VISION_METADATA)
  policy_meta = _load_metadata(bundle_dir / SPLIT_POLICY_METADATA)

  vision_shapes = vision_meta.get("input_shapes") or {}
  vision_names = [k for k, v in vision_shapes.items()
                  if isinstance(v, (tuple, list)) and len(v) == 4] or ["input_imgs", "big_input_imgs"]

  return ModelProfile(
    name=name,
    mode="split",
    engine_dir=str(bundle_dir),
    vision_engine=vision_engine,
    policy_engine=policy_engine,
    vision_input_names=vision_names,
    input_shapes={**VISION_ENGINE_SHAPES, **vision_shapes, **(policy_meta.get("input_shapes") or {})},
    vision_slices=_output_slices(vision_meta, bundle_dir / SPLIT_VISION_METADATA),
    policy_slices=_output_slices(policy_meta, bundle_dir / SPLIT_POLICY_METADATA),
    temporal=_temporal_from_metadata(policy_meta, bundle_dir / SPLIT_POLICY_METADATA),
    scalar_input_shapes=_scalars(policy_meta.get("input_shapes") or {}, vision_names),
  )


_SCAN_CACHE: dict[str, object] = {"key": None, "profiles": {}}


def _scan_key(root: Path) -> tuple:
  try:
    return (str(root), root.stat().st_mtime_ns)
  except OSError:
    return (str(root), 0)


def scan_profiles(root: Path | None = None, *, refresh: bool = False) -> dict[str, ModelProfile]:
  """Every readable bundle under `root`, name -> profile.

  Cached per process and invalidated when the models directory changes, so a
  freshly compiled plan is picked up without a restart while the common path
  costs one stat(). `DISABLE_GPU_REGISTRY_SCAN=1` restricts resolution to the
  curated profiles (the pre-scan behaviour), as an escape hatch.
  """
  root = Path(root or MODELS_ROOT)
  if os.getenv("DISABLE_GPU_REGISTRY_SCAN", "0") == "1":
    return {}
  key = _scan_key(root)
  if not refresh and _SCAN_CACHE["key"] == key:
    return dict(_SCAN_CACHE["profiles"])  # type: ignore[arg-type]

  found: dict[str, ModelProfile] = {}
  try:
    entries = sorted(p for p in root.iterdir() if p.is_dir())
  except OSError:
    entries = []
  for bundle_dir in entries:
    # Skip dot-dirs (order/marker files) and bookkeeping/pycache directories.
    if bundle_dir.name.startswith((".", "__")):
      continue
    try:
      profile = bundle_from_dir(bundle_dir)
    except Exception as e:
      log.debug(f"bundle {bundle_dir.name} unusable: {e}")
      continue
    if profile is not None:
      found[profile.name] = profile

  _SCAN_CACHE["key"] = key
  _SCAN_CACHE["profiles"] = found
  return dict(found)


# --- failure memory ---------------------------------------------------------

def _read_failed(root: Path | None = None) -> dict:
  path = Path(root or MODELS_ROOT) / FAILED_FILE.name
  try:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
      return value
  except (OSError, ValueError):
    pass
  return {}


def _write_failed(value: dict, root: Path | None = None) -> None:
  path = Path(root or MODELS_ROOT) / FAILED_FILE.name
  try:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
  except OSError:
    pass


def _engines_mtime(profile: ModelProfile) -> float:
  newest = 0.0
  for engine in profile._engine_paths():
    try:
      newest = max(newest, engine.stat().st_mtime)
    except OSError:
      pass
  return newest


def engines_mtime(profile: ModelProfile) -> float:
  """Newest engine mtime; a rebuild makes a failure marker stale."""
  return _engines_mtime(profile)


def mark_failed(profile: ModelProfile | str, reason: str = "", root: Path | None = None) -> None:
  """Remember that this bundle failed to initialise.

  Keyed by bundle name, stamped with the engines' mtime: rebuilding the plan
  (a newer file) makes the entry stale, so a fixed engine is retried without
  anyone deleting a file by hand.
  """
  name = profile if isinstance(profile, str) else profile.name
  mtime = 0.0 if isinstance(profile, str) else _engines_mtime(profile)
  value = _read_failed(root)
  value[name] = {"epoch": time.time(), "reason": reason[:500], "engines_mtime": mtime}
  _write_failed(value, root)


def clear_failed(name: str | None = None, root: Path | None = None) -> None:
  value = _read_failed(root)
  if name is None:
    value = {}
  else:
    value.pop(name, None)
  _write_failed(value, root)


def failure_reason(name: str, profile: ModelProfile | None = None, root: Path | None = None) -> str | None:
  """Why `name` is being skipped, or None when it is not marked failed.

  A marker is ignored (and left to expire on its own) when the cooldown passed
  or the engines are newer than the marker.
  """
  entry = _read_failed(root).get(name)
  if not isinstance(entry, dict):
    return None
  try:
    epoch = float(entry.get("epoch", 0))
    recorded_mtime = float(entry.get("engines_mtime", 0))
  except (TypeError, ValueError):
    return None
  if time.time() - epoch > FAILED_COOLDOWN_S:
    return None
  if profile is not None:
    current = _engines_mtime(profile)
    if current > recorded_mtime + 1.0:
      return None  # engine rebuilt after the failure: retry it
  reason = entry.get("reason") or "previous initialisation failed"
  return f"marked failed: {reason}"


# --- selection --------------------------------------------------------------

def read_order(root: Path | None = None) -> list[str]:
  """The user's preferred order (`models/.trt_order`, one name per line)."""
  path = Path(root or MODELS_ROOT) / ORDER_FILE.name
  names: list[str] = []
  try:
    for line in path.read_text(encoding="utf-8").splitlines():
      name = line.strip()
      if name and not name.startswith("#") and name not in names:
        names.append(name)
  except OSError:
    pass
  return names


def write_order(names, root: Path | None = None) -> None:
  path = Path(root or MODELS_ROOT) / ORDER_FILE.name
  try:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(names) + ("\n" if names else ""), encoding="utf-8")
  except OSError:
    pass


def _params_model() -> str | None:
  """`Model` Param, if this tree has one (old-layout forks may not)."""
  try:
    from openpilot.common.params import Params
    value = Params().get("Model", encoding="utf-8")
    return value or None
  except Exception:
    return None


def _all_profiles(refresh: bool = False) -> dict[str, ModelProfile]:
  profiles: dict[str, ModelProfile] = {}
  for name, profile in REGISTRY.items():
    if profile is not None:
      profiles[name] = profile
  # Scanned bundles fill in names the curated set does not cover; a curated
  # profile always wins for its own name (it knows the engine's binding quirks).
  for name, profile in scan_profiles(refresh=refresh).items():
    profiles.setdefault(name, profile)
  return profiles


def get_profile(name: str | None) -> ModelProfile | None:
  if not name:
    return None
  return _all_profiles().get(name)


def _usable(name: str, profiles: dict[str, ModelProfile], root: Path | None) -> ModelProfile | None:
  profile = profiles.get(name)
  if profile is None:
    return None
  if profile.missing_reason() is not None:
    return None
  if failure_reason(name, profile, root) is not None:
    return None
  return profile


def resolve_profile(name: str | None = None, root: Path | None = None, *,
                    strict: bool = False) -> ModelProfile | None:
  """Pick the first profile that is actually usable on this machine.

  Priority: explicit name -> MODEL_NAME env -> .trt_order -> Params Model ->
  curated order -> scanned bundles by name.

  `name` is a PREFERENCE, not a constraint: when the tree or the UI asks for a
  model whose engines are missing, a working one is used instead and the caller
  logs the substitution (modeld must not fall to tinygrad just because the
  selected bundle has not been compiled yet). Pass `strict=True` for the other
  reading - "the named model or nothing" - which is what a UI "test this model"
  action wants.

  Returns None when nothing is usable; the caller then falls back to the stock
  tinygrad ModelState.
  """
  profiles = _all_profiles()
  candidates: list[str] = []
  for candidate in (name, os.getenv("MODEL_NAME"), *_order_candidates(root)):
    if candidate and candidate not in candidates:
      candidates.append(candidate)
  if not strict:
    candidates += [n for n in CURATED_ORDER if n not in candidates]
    candidates += [n for n in sorted(profiles) if n not in candidates]

  for candidate in candidates:
    profile = _usable(candidate, profiles, root)
    if profile is not None:
      return profile
  return None


def _order_candidates(root: Path | None) -> list[str]:
  candidates = list(read_order(root))
  param = _params_model()
  if param:
    candidates.append(param)
  return candidates


def snapshot(root: Path | None = None) -> list[dict]:
  """Every candidate with its verdict, for logs and the web UI.

  [{'name', 'mode', 'ready', 'reason', 'failed', 'selected'}]
  """
  profiles = _all_profiles()
  chosen = resolve_profile(root=root)
  rows = []
  for name in sorted(profiles):
    profile = profiles[name]
    missing = profile.missing_reason()
    failed = failure_reason(name, profile, root)
    rows.append({
      "name": name,
      "mode": profile.mode,
      "ready": missing is None,
      "reason": missing or failed or "",
      "failed": bool(failed),
      "selected": chosen is not None and chosen.name == name,
    })
  return rows