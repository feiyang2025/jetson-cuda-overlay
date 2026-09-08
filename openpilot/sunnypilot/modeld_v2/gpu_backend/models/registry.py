import pickle
from pathlib import Path

from openpilot.sunnypilot.modeld_v2.gpu_backend.profile import ModelProfile, TemporalMeta

MODELS_ROOT = Path(__file__).parents[4] / "selfdrive" / "modeld" / "models"


def _load_slices(metadata_path: Path) -> dict[str, slice] | None:
  if not metadata_path.exists():
    return None
  try:
    with open(metadata_path, "rb") as f:
      metadata = pickle.load(f)
    slices = metadata.get("output_slices")
    if isinstance(slices, dict):
      return slices
  except Exception:
    return None
  return None


def _temporal() -> TemporalMeta:
  return TemporalMeta(
    features_len=512,
    features_windows=25,
    desire_len=8,
    desire_windows=25,
    features_includes_current=False,
    desire_includes_current=True,
  )


def _filet_ofish() -> ModelProfile:
  engine_dir = MODELS_ROOT / "FiletOFish"
  vision_slices = _load_slices(engine_dir / "driving_vision_metadata.pkl")
  policy_slices = _load_slices(engine_dir / "driving_policy_metadata.pkl")
  return ModelProfile(
    name="FiletOFish",
    mode="split",
    engine_dir=str(engine_dir),
    vision_engine="driving_vision_fp16.plan",
    policy_engine="driving_policy_fp16.plan",
    vision_input_names=['img', 'big_img'],
    input_shapes={
      'img': (1, 12, 128, 256),
      'big_img': (1, 12, 128, 256),
      'desire': (1, 25, 8),
      'traffic_convention': (1, 2),
      'features_buffer': (1, 24, 512),
    },
    vision_slices=vision_slices or {},
    policy_slices=policy_slices or {},
    temporal=_temporal(),
  )


def _big_combo() -> ModelProfile:
  engine_dir = MODELS_ROOT / "BigCombo"
  merged_slices = _load_slices(engine_dir / "driving_supercombo_metadata.pkl")
  if merged_slices is None:
    merged_slices = {
      'lane_lines': slice(0, 528),
      'lane_lines_prob': slice(528, 536),
      'road_edges': slice(536, 800),
      'meta': slice(800, 855),
      'desire_pred': slice(855, 887),
      'pose': slice(887, 899),
      'wide_from_device_euler': slice(899, 905),
      'road_transform': slice(905, 917),
      'plan': slice(917, 1907),
      'lead': slice(1907, 2051),
      'lead_prob': slice(2051, 2054),
      'desire_state': slice(2054, 2062),
      'action': slice(2062, 2066),
      'hidden_state': slice(2066, 2578),
      'pad': slice(2578, 2580),
    }
  vision_slices = {k: v for k, v in merged_slices.items() if k not in (
    'plan', 'lead', 'lead_prob', 'desire_state', 'action', 'hidden_state', 'pad',
  )}
  policy_slices = {k: v for k, v in merged_slices.items() if k not in vision_slices}
  return ModelProfile(
    name="BigCombo",
    mode="merged",
    engine_dir=str(engine_dir),
    merged_engine="driving_supercombo_fp16.plan",
    vision_input_names=['img', 'big_img'],
    input_shapes={
      'img': (1, 12, 128, 256),
      'big_img': (1, 12, 128, 256),
      'desire_pulse': (1, 25, 8),
      'traffic_convention': (1, 2),
      'action_t': (1, 2),
      'features_buffer': (1, 24, 512),
    },
    vision_slices=vision_slices,
    policy_slices=policy_slices,
    temporal=_temporal(),
  )


REGISTRY: dict[str, ModelProfile] = {
  "FiletOFish": _filet_ofish(),
  "BigCombo": _big_combo(),
}


def get_profile(name: str | None) -> ModelProfile | None:
  return REGISTRY.get(name or "FiletOFish")


def resolve_profile(name: str | None = None) -> ModelProfile | None:
  """Pick the first profile that is actually usable on this machine.

  Priority:
    1. explicit `name` (from Params Model / env) if its engines exist;
    2. any ready profile in REGISTRY order;
    3. None -> caller falls back to tinygrad.
  """
  if name:
    profile = get_profile(name)
    if profile is not None and profile.is_ready():
      return profile
  for profile in REGISTRY.values():
    if profile.is_ready():
      return profile
  return None