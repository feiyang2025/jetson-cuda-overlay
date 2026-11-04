import os

import numpy as np
from openpilot.cereal import log
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.controls.lib.drive_helpers import get_accel_from_plan, get_curvature_from_plan, smooth_value, should_stop
from openpilot.sunnypilot.modeld_v2.modeld_base import ModelStateBase
from openpilot.sunnypilot.modeld_v2.constants import ModelConstants, Plan

from openpilot.sunnypilot.modeld_v2.gpu_backend.backend import create_gpu_backend
from openpilot.sunnypilot.modeld_v2.gpu_backend.models.registry import resolve_profile
from openpilot.sunnypilot.modeld_v2.parse_model_outputs import Parser

# Outputs whose finiteness decides whether the frame is usable for control.
# Checked unconditionally: a NaN that only trips the big model is exactly the
# kind of failure that survives until the car is already moving wrong.
CONTROL_OUTPUTS = ("plan", "planplus", "desired_curvature", "action", "desired_accel")


class _WarmupFrame:
  """Minimal stand-in for a camera frame.

  The CUDA transform wants frame.width/height/stride/uv_offset next to the data,
  which the tinygrad path can infer from a bare array and this one cannot.
  """

  def __init__(self, data: np.ndarray, width: int, height: int, stride: int, uv_offset: int):
    self.data = memoryview(data)
    self.width = width
    self.height = height
    self.stride = stride
    self.uv_offset = uv_offset


class GpuModelState(ModelStateBase):
  """CUDA/TensorRT backend for the stock modeld process.

  Drop-in for `openpilot/selfdrive/modeld/modeld.py:ModelState`: same
  constructor signature and the same contract the caller relies on -
  `vision_input_names`, `chestnut`, `lat_delay`, `frame_drop_ratio`,
  `in_control`, `signal_shadow`, `warmup()`, and
  `run(bufs, transforms, inputs, after_enqueue)`.

  Falls back to the stock class whenever a CUDA profile is not usable - see
  `cuda_patches.patch_modeld_state`.
  """

  def __init__(self, cam_w: int, cam_h: int, chestnut: bool, profile=None):
    ModelStateBase.__init__(self)
    self.chestnut = chestnut
    self.W, self.H = cam_w, cam_h
    self.frame_drop_ratio = 0
    self.in_control = False
    self.signal_shadow = None

    # The caller (cuda_patches) has already decided which profile fits chestnut or
    # not; re-resolving here would pick the registry default and silently run the
    # big model as the small one, so only resolve when nobody passed one in.
    if profile is None:
      model_name = os.getenv("MODEL_NAME") or Params().get("Model")
      profile = resolve_profile(model_name)
      if profile is None:
        raise RuntimeError(f"no usable CUDA model profile for {model_name or 'default'}")
    self.profile = profile
    self.backend = create_gpu_backend(self.profile)
    if self.backend is None:
      raise RuntimeError("CUDA backend unavailable (see the create_gpu_backend warning for the reason)")

    self.constants = ModelConstants
    self.LAT_SMOOTH_SECONDS = 0.0
    self.LONG_SMOOTH_SECONDS = 0.0
    self.MIN_LAT_CONTROL_SPEED = 0.3
    self.PLANPLUS_CONTROL = 1.0

    self._vision_input_names = list(self.profile.vision_input_names)
    self._road_key = next(k for k in self._vision_input_names if "big" not in k)
    self._wide_key = next(k for k in self._vision_input_names if "big" in k)

    self.input_shapes = dict(self.profile.scalar_input_shapes)
    self.output_slices = {**self.profile.vision_slices, **self.profile.policy_slices}
    if "hidden_state" not in self.output_slices:
      raise RuntimeError(f"profile {self.profile.name} metadata has no hidden_state slice")

    # Scalars are held in whatever dtype the engine actually binds. Hardcoding
    # float32 made every real inference fail the runner's size/dtype check
    # ("expects (1, 24, 512) float16, got float32") - the engine, not this class,
    # decides the width.
    # Split profiles bind the scalars on the policy engine, not the vision one:
    # looking them up on vision_runner alone left every scalar at float32 and the
    # engine rejected shape (1, 25, 8) float32 against (1, 25, 8) float16.
    scalar_runner = self.backend.policy_runner if self.profile.mode == "split" else self.backend.vision_runner
    engine_dtypes = dict(getattr(scalar_runner, "input_dtypes", {}) or {})
    self.npy: dict[str, np.ndarray] = {
      k: np.zeros(shape, dtype=engine_dtypes.get(k, np.float32))
      for k, shape in self.input_shapes.items()
    }
    self.numpy_inputs = self.npy

    # The stock big model takes a desire pulse and a previous-feature buffer.
    # Naming them from metadata keeps this aligned with whatever the engine wants
    # instead of assuming the shape the old overlay assumed.
    self._desire_out = next((k for k in self.input_shapes if k.startswith("desire")), None)
    if self._desire_out is None:
      raise RuntimeError(f"profile {self.profile.name} declares no desire input; got {sorted(self.input_shapes)}")
    self._prev_feat_key = "prev_feat" if "prev_feat" in self.input_shapes else None

    self.prev_desire = np.zeros(self.constants.DESIRE_LEN, dtype=np.float32)
    self.parser = Parser()
    self._warm = False

    cloudlog.warning(
      f"GpuModelState initialized: profile={self.profile.name} chestnut={chestnut} "
      f"mode={self.profile.mode} scalars={sorted(self.input_shapes)}"
    )

  @property
  def mlsim(self) -> bool:
    return False

  @property
  def vision_input_names(self) -> list[str]:
    return self._vision_input_names

  @property
  def road_key(self) -> str:
    return self._road_key

  @property
  def wide_key(self) -> str:
    return self._wide_key

  @property
  def desire_key(self) -> str:
    return self._desire_out

  def warmup(self) -> None:
    if self._warm:
      return
    # The CUDA transform reads frame.width/height/stride/uv_offset, so a bare numpy
    # array is not enough - it raised AttributeError and took the whole GPU path
    # down on startup, since modeld always warms up before the first frame.
    # NV12 sizing here is only for the throwaway frame; real frames carry their own
    # geometry and their own uv_offset (0 for packed UYVY).
    frame_bytes = self.W * self.H * 3 // 2
    blank = np.zeros(frame_bytes, dtype=np.uint8)
    frames = {}
    for k in self._vision_input_names:
      frames[k] = _WarmupFrame(blank, self.W, self.H, self.W, self.W * self.H)
    transforms = {k: np.eye(3, dtype=np.float32) for k in (self._road_key, self._wide_key)}
    inputs = {
      "desire_pulse": np.zeros(self.constants.DESIRE_LEN, dtype=np.float32),
      "traffic_convention": np.zeros(2, dtype=np.float32),
      "action_t": np.zeros(2, dtype=np.float32),
    }
    self.run(frames, transforms, inputs)
    for v in self.npy.values():
      v[:] = 0
    self.prev_desire[:] = 0
    self._warm = True

  def slice_outputs(self, model_outputs: np.ndarray, output_slices: dict[str, slice]) -> dict[str, np.ndarray]:
    return {k: model_outputs[np.newaxis, v] for k, v in output_slices.items()}

  def run(self, bufs: dict[str, object], transforms: dict[str, np.ndarray],
          inputs: dict[str, np.ndarray], after_enqueue=None) -> dict[str, np.ndarray]:
    frames = {k: bufs[k] for k in self._vision_input_names if k in bufs}
    if len(frames) != len(self._vision_input_names):
      raise RuntimeError(f"missing vision frames: {sorted(set(self._vision_input_names) - set(frames))}")
    self.backend.preprocess(frames, transforms)

    desire_in = inputs.get("desire_pulse")
    if desire_in is not None:
      pulse = np.asarray(desire_in, dtype=np.float32).reshape(-1)
      pulse = np.where(pulse - self.prev_desire > 0.99, pulse, 0).astype(np.float32)
      self.npy[self._desire_out][:] = pulse
      self.prev_desire[:] = pulse

    for key in ("desire", "traffic_convention", "action_t"):
      if key in self.npy and key in inputs:
        self.npy[key][:] = inputs[key]

    for key, src in (("tfm", self._road_key), ("big_tfm", self._wide_key)):
      if key in self.npy and src in transforms:
        self.npy[key][:, :] = transforms[src].reshape(3, 3)

    if self.profile.mode == "merged":
      raw = self.backend.infer_merged(dict(self.npy))
      outputs = self.parser.parse_outputs(self.slice_outputs(raw, self.output_slices))
      self._update_temporal(raw, None)
    else:
      vision_out, policy_out = self.backend.infer_split(dict(self.npy))
      outputs = self.parser.parse_vision_outputs(self.slice_outputs(vision_out, self.profile.vision_slices))
      outputs.update(self.parser.parse_policy_outputs(self.slice_outputs(policy_out, self.profile.policy_slices)))
      self._update_temporal(vision_out, policy_out)

    if after_enqueue is not None:
      after_enqueue()

    if not self._outputs_finite(outputs):
      cloudlog.error("model output not finite")
      raise RuntimeError("model output not finite")
    return outputs

  def _update_temporal(self, vision_output: np.ndarray, policy_output: np.ndarray | None) -> None:
    """Store prev_feat once per frame from the engine that produced hidden_state.

    The slices are relative to that engine's own output, never to a
    concatenation of both, so the source is chosen rather than assumed.
    """
    if self._prev_feat_key is None:
      return
    slices, source = (self.profile.vision_slices, vision_output)
    if "hidden_state" not in slices and policy_output is not None:
      slices, source = (self.profile.policy_slices, policy_output)
    if "hidden_state" not in slices:
      return
    hidden = source[slices["hidden_state"]]
    buf = self.npy[self._prev_feat_key]
    if hidden.size != buf.size:
      raise RuntimeError(
        f"hidden_state holds {hidden.size} values but {self._prev_feat_key} holds {buf.size}; "
        f"engine and metadata disagree"
      )
    buf[:] = hidden.reshape(buf.shape)

  def _outputs_finite(self, outputs: dict[str, np.ndarray]) -> bool:
    for key in CONTROL_OUTPUTS:
      if key not in outputs:
        continue
      value = np.asarray(outputs[key], dtype=np.float64)
      if value.size and not np.all(np.isfinite(value)):
        cloudlog.error(f"non-finite model output in {key}")
        return False
    return True

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
