# Compat model profile registry for old-layout forks
# Resolves model name to a ModelProfile

import os
from pathlib import Path


class ModelProfile:
  def __init__(self, name: str, engine_dir: str, mode: str = "split"):
    self.name = name
    self.engine_dir = engine_dir
    self.mode = mode
    self.model_w = 512
    self.model_h = 256

    # Vision inputs
    self.vision_input_names = ["biginput", "input"]

    # Input shapes matching compat constants (old-layout supercombo-style).
    self.input_shapes = {
      "input": (1, 12, 128, 256),
      "biginput": (1, 12, 128, 256),
      "desire": (1, 32),
      "traffic_convention": (1, 2),
      "features_buffer": (1, 24, 512),
    }

    # Vision shape
    self.vision_shape = (1, 12, self.model_h, self.model_w)

    # Slices - simplified for compat
    self.vision_slices = {
      "hidden_state": slice(0, 512),
    }
    self.policy_slices = {
      "plan": slice(0, 2580),
    }
    self.temporal = type('obj', (object,), {'features_len': 512})()

    # Engine files (try _fp16 first, then plain name)
    engine_dir_path = Path(engine_dir)
    if mode == "merged":
      self.merged_engine = self._find_engine(engine_dir_path, ["driving_supercombo_fp16.plan", "driving_supercombo.plan"])
      self.vision_engine = None
      self.policy_engine = None
    else:
      self.merged_engine = None
      self.vision_engine = self._find_engine(engine_dir_path, ["driving_vision_fp16.plan", "driving_vision.plan"])
      self.policy_engine = self._find_engine(engine_dir_path, ["driving_policy_fp16.plan", "driving_policy.plan"])

  def _find_engine(self, engine_dir: Path, candidates: list[str]) -> str | None:
    for name in candidates:
      if (engine_dir / name).exists():
        return name
    return candidates[0]  # Return first candidate as default

  def vision_shape_for(self, key: str) -> tuple:
    return self.vision_shape

  def is_ready(self) -> bool:
    engine_dir = Path(self.engine_dir)
    if self.mode == "merged":
      return (engine_dir / self.merged_engine).exists()
    else:
      return (engine_dir / self.vision_engine).exists() and (engine_dir / self.policy_engine).exists()


def resolve_profile(model_name: str | None) -> ModelProfile | None:
  """Resolve model name to a profile. Auto-detect split vs merged."""
  # Default model directory
  model_dir = os.getenv("MODEL_DIR")
  if model_dir is None:
    # Try common locations
    candidates = [
      Path(__file__).resolve().parents[4] / "selfdrive" / "modeld" / "models",
      Path("/data/openpilot/models"),
    ]
    for c in candidates:
      if c.exists():
        model_dir = str(c)
        break
  
  if model_dir is None:
    return None
  
  # Check for merged engine
  profile = ModelProfile(model_name or "default", model_dir, mode="merged")
  if profile.is_ready():
    return profile
  
  # Check for split engines
  profile = ModelProfile(model_name or "default", model_dir, mode="split")
  if profile.is_ready():
    return profile
  
  return None
