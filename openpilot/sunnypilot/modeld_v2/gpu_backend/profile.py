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

  def vision_shape_for(self, key: str) -> tuple[int, ...]:
    return self.input_shapes[key]

  @property
  def model_w(self) -> int:
    first = self.vision_input_names[0]
    return self.input_shapes[first][-1] * 2

  @property
  def model_h(self) -> int:
    first = self.vision_input_names[0]
    return self.input_shapes[first][-2] * 2

  def is_ready(self) -> bool:
    root = Path(self.engine_dir)
    if self.mode == "merged":
      return bool(self.merged_engine and (root / self.merged_engine).is_file())
    return bool(self.vision_engine and self.policy_engine and
                (root / self.vision_engine).is_file() and
                (root / self.policy_engine).is_file() and
                self.vision_slices and self.policy_slices)


EngineMode = ModelProfile