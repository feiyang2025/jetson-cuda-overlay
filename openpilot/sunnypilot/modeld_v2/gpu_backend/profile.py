import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class TemporalMeta:
  features_len: int
  features_windows: int
  desire_len: int
  desire_windows: int
  features_includes_current: bool
  desire_includes_current: bool


@dataclass
class ModelProfile:
  name: str
  mode: Literal["split", "merged"]
  engine_dir: str
  vision_engine: str | None = None
  policy_engine: str | None = None
  merged_engine: str | None = None
  vision_input_names: list[str] = field(default_factory=list)
  input_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
  vision_slices: dict[str, slice] = field(default_factory=dict)
  policy_slices: dict[str, slice] = field(default_factory=dict)
  temporal: TemporalMeta | None = None
  scalar_input_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)

  def vision_shape_for(self, key: str) -> tuple[int, ...]:
    if key not in self.input_shapes:
      raise KeyError(f"profile {self.name} has no input shape for {key!r}; known: {sorted(self.input_shapes)}")
    return self.input_shapes[key]

  @property
  def model_w(self) -> int:
    first = self.vision_input_names[0]
    return self.input_shapes[first][-1] * 2

  @property
  def model_h(self) -> int:
    first = self.vision_input_names[0]
    return self.input_shapes[first][-2] * 2

  @property
  def temporal_skip(self) -> int:
    if self.temporal is None:
      raise RuntimeError(f"profile {self.name} has no temporal metadata; refusing to guess temporal_skip")
    return self.temporal.features_windows

  def _engine_paths(self) -> list[Path]:
    if self.mode == "merged":
      return [Path(self.engine_dir) / self.merged_engine] if self.merged_engine else []
    root = Path(self.engine_dir)
    return [p for p in (root / self.vision_engine if self.vision_engine else None,
                        root / self.policy_engine if self.policy_engine else None) if p is not None]

  def missing_reason(self) -> str | None:
    """Why this profile cannot run, or None when it is ready."""
    root = Path(self.engine_dir)
    if not root.is_dir():
      return f"engine dir missing: {root}"
    # A bundle can carry valid metadata while its engine has not been compiled
    # yet (trtexec still running, or the build failed). That is a "not ready"
    # state the UI must see, not a reason to drop the bundle from the list.
    if self.mode == "merged" and not self.merged_engine:
      return f"engine missing: no merged engine in {root}"
    if self.mode == "split":
      if not self.vision_engine:
        return f"engine missing: no vision engine in {root}"
      if not self.policy_engine:
        return f"engine missing: no policy engine in {root}"
    for engine in self._engine_paths():
      if not engine.is_file():
        return f"engine missing: {engine}"
      if engine.stat().st_size == 0:
        return f"engine is empty: {engine}"
    if not self.vision_slices:
      return "vision output slices missing from metadata"
    if self.mode == "split" and not self.policy_slices:
      return "policy output slices missing from metadata"
    if self.temporal is None:
      return "temporal metadata missing"
    if self.temporal.features_windows < 2:
      return f"features_buffer has {self.temporal.features_windows} time slots, needs >= 2"
    for key in self.vision_input_names:
      if key not in self.input_shapes:
        return f"no input shape for vision input {key!r}"
    return None

  def is_ready(self) -> bool:
    return self.missing_reason() is None


EngineMode = ModelProfile


def cuda_transform_library_default() -> Path:
  return Path(__file__).resolve().parents[3] / "selfdrive" / "modeld" / "transforms" / "libcuda_transform.so"


def is_cuda_disabled_by_env() -> bool:
  return os.getenv("DISABLE_CUDA_BACKEND", "0") == "1"
