import os

import numpy as np
from openpilot.cereal import log
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.controls.lib.drive_helpers import get_accel_from_plan, get_curvature_from_plan, smooth_value, should_stop
from openpilot.sunnypilot.modeld_v2.modeld_base import ModelStateBase

from openpilot.sunnypilot.modeld_v2.gpu_backend.backend import create_gpu_backend
from openpilot.sunnypilot.modeld_v2.gpu_backend.models.registry import resolve_profile
from openpilot.sunnypilot.modeld_v2.constants import ModelConstants, Plan
from openpilot.sunnypilot.modeld_v2.parse_model_outputs import Parser as CombinedParser
from openpilot.sunnypilot.modeld_v2.parse_model_outputs_split import Parser as SplitParser


class GpuModelState(ModelStateBase):
  def __init__(self, cam_w: int, cam_h: int, chestnut: bool = False):
    ModelStateBase.__init__(self)
    self.chestnut = chestnut
    model_name = os.getenv("MODEL_NAME") or Params().get("Model", encoding="utf-8")
    self.profile = resolve_profile(model_name)
    if self.profile is None:
      raise RuntimeError(f"No usable CUDA model profile for {model_name or 'default'}")
    self.backend = create_gpu_backend(self.profile)
    if self.backend is None:
      raise RuntimeError("CUDA backend unavailable")
    self.constants = ModelConstants
    self.vision_input_names = list(self.profile.vision_input_names)
    self.road_key = next(k for k in self.vision_input_names if "big" not in k)
    self.wide_key = next(k for k in self.vision_input_names if "big" in k)
    self.numpy_inputs: dict[str, np.ndarray] = {}
    for k, shape in self.profile.input_shapes.items():
      if k in self.vision_input_names:
        continue
      self.numpy_inputs[k] = np.zeros(shape, dtype=np.float32)
    desire_candidates = [k for k in self.numpy_inputs if k.startswith("desire")]
    self.desire_key = desire_candidates[0] if desire_candidates else "desire"
    self.prev_desire = np.zeros(self.constants.DESIRE_LEN, dtype=np.float32)
    self.lat_delay = 0.0
    self.LAT_SMOOTH_SECONDS = 0.0
    self.LONG_SMOOTH_SECONDS = 0.0
    self.MIN_LAT_CONTROL_SPEED = 0.3
    self.PLANPLUS_CONTROL = 1.0
    self.full_features_buffer: np.ndarray | None = None
    self.parser = SplitParser() if self.profile.mode == "split" else CombinedParser()
    cloudlog.warning(f"GpuModelState initialized: {model_name} mode={self.profile.mode}")

  @property
  def mlsim(self) -> bool:
    return False

  def slice_outputs(self, model_output: np.ndarray, slices: dict[str, slice]) -> dict[str, np.ndarray]:
    return {k: model_output[np.newaxis, v] for k, v in slices.items()}

  def _update_temporal(self, raw_output: np.ndarray) -> None:
    hidden = raw_output[self.profile.policy_slices["hidden_state"]]
    feats = self.numpy_inputs.get("features_buffer")
    if feats is None:
      return
    feats[0, :-1] = feats[0, 1:]
    feats[0, -1] = hidden[: self.profile.temporal.features_len]

  def run(self, bufs: dict[str, object], transforms: dict[str, np.ndarray],
                inputs: dict[str, np.ndarray], prepare_only: bool) -> dict[str, np.ndarray] | None:
    frames = {k: bufs[k] for k in self.vision_input_names if k in bufs}
    self.backend.preprocess(frames, transforms)

    if self.desire_key in inputs:
      cur = np.asarray(inputs[self.desire_key]).reshape(-1)
      current = cur[-self.constants.DESIRE_LEN:]
      pulse = np.where(current - self.prev_desire > 0.99, current, 0)
      target = self.numpy_inputs[self.desire_key]
      if target.ndim >= 2:
        target[..., :-1, :] = target[..., 1:, :]
        target[..., -1, :] = pulse
      else:
        target[:] = pulse
      self.prev_desire[:] = current

    for k, v in inputs.items():
      if k in self.numpy_inputs and k != self.desire_key:
        self.numpy_inputs[k][:] = v

    scalar_inputs = {k: v for k, v in self.numpy_inputs.items()}

    if self.profile.mode == "merged":
      raw = self.backend.infer_merged(scalar_inputs)
    else:
      v_out, p_out = self.backend.infer_split(scalar_inputs)
      raw = np.concatenate([v_out, p_out])

    if self.profile.mode == "merged":
      sliced = self.slice_outputs(raw, {**self.profile.vision_slices, **self.profile.policy_slices})
      outputs = self.parser.parse_outputs(sliced)
    else:
      vision_output = raw[: self._split_vision_size()]
      policy_output = raw[self._split_vision_size():]
      vision_sliced = self.slice_outputs(vision_output, self.profile.vision_slices)
      policy_sliced = self.slice_outputs(policy_output, self.profile.policy_slices)
      outputs = self.parser.parse_vision_outputs(vision_sliced)
      outputs.update(self.parser.parse_policy_outputs(policy_sliced))

    self._update_temporal(raw)

    if self.chestnut and not np.all(np.isfinite(outputs.get("plan", np.array([0.0])))):
      cloudlog.error("model output not finite, dropping frame")
      return None
    return outputs

  def _split_vision_size(self) -> int:
    return sum(s.stop - s.start for s in self.profile.vision_slices.values())

  def get_action_from_model(self, model_output: dict[str, np.ndarray], prev_action: log.ModelDataV2.Action,
                            lat_action_t: float, long_action_t: float, v_ego: float) -> log.ModelDataV2.Action:
    if "action" not in model_output:
      plan = model_output["plan"][0]
      desired_accel = get_accel_from_plan(plan[:, Plan.VELOCITY][:, 0], plan[:, Plan.ACCELERATION][:, 0],
                                          self.constants.T_IDXS, action_t=long_action_t)
      desired_curvature = get_curvature_from_plan(plan[:, Plan.T_FROM_CURRENT_EULER][:, 2],
                                                  plan[:, Plan.ORIENTATION_RATE][:, 2],
                                                  self.constants.T_IDXS, v_ego, action_t=lat_action_t)
    else:
      desired_accel = model_output["action"][0, 1]
      desired_curvature = model_output["action"][0, 0] / (max(1.0, v_ego)) ** 2

    stop = should_stop(v_ego, desired_accel)
    desired_accel = smooth_value(desired_accel, prev_action.desiredAcceleration, self.LONG_SMOOTH_SECONDS)
    if v_ego > self.MIN_LAT_CONTROL_SPEED:
      desired_curvature = smooth_value(desired_curvature, prev_action.desiredCurvature, self.LAT_SMOOTH_SECONDS)
    else:
      desired_curvature = prev_action.desiredCurvature
    return log.ModelDataV2.Action(
      desiredCurvature=float(desired_curvature),
      desiredAcceleration=float(desired_accel),
      shouldStop=bool(stop),
    )
