#!/usr/bin/env python3
"""FrogPilot modeld main — 100% FrogPilot business logic on the CUDA/TRT engine.

This is `frogpilot/selfdrive/modeld/tinygrad/tinygrad_modeld.py` with the model
construction swapped from the tinygrad ModelState to FrogPilotGpuState (overlay
kits/patch/frogpilot/fp_gpu_state.py). Everything else - the message loop,
frogpilotModelV2 publishing, DesireHelper, get_action_from_model, fill messages,
toggles refresh, dropped-frame bookkeeping - is unchanged FrogPilot logic, so the
fork's behaviour on the road is preserved.

The original imports `DrivingModelFrame/CLContext` (OpenCL) and tinygrad; this
file imports none of those. It is registered in process_config.py in place of
the C++ modeld by kits/patch/install_fp.py.
"""
import os
import time

import numpy as np
from setproctitle import setproctitle

import cereal.messaging as messaging
from cereal import log
from cereal import car  # 下面 CarParams.from_bytes 用的就是它; 漏了这行会 NameError: name 'car'
from cereal.messaging import PubMaster, SubMaster
from msgq.visionipc import VisionIpcClient, VisionStreamType
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process, DT_MDL
from openpilot.common.swaglog import cloudlog
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.frogpilot.common import frogpilot_paths, frogpilot_utilities, frogpilot_variables
from openpilot.frogpilot.selfdrive.modeld.tinygrad.constants import ModelConstants, Plan
from openpilot.frogpilot.selfdrive.modeld.tinygrad.fill_model_msg import fill_model_msg, fill_pose_msg, PublishState
from openpilot.frogpilot.selfdrive.modeld.tinygrad.parse_model_outputs import Parser
from openpilot.selfdrive.car.car_helpers import get_demo_car_params
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper
from openpilot.selfdrive.controls.lib.drive_helpers import curv_from_psis, get_accel_from_plan, smooth_value

from openpilot.frogpilot.selfdrive.modeld.tinygrad.fp_modeld_gpu import FrogPilotGpuState

PROCESS_NAME = "frogpilot.selfdrive.modeld.modeld_gpu"
SEND_RAW_PRED = os.getenv('SEND_RAW_PRED')

LONG_SMOOTH_SECONDS = 0.3
MIN_LAT_CONTROL_SPEED = 0.3
MIN_STABLE_DELAY = 0.3
V_EGO_STOPPING = 0.3


def get_current_speed_accel_from_plan(speeds, accels, action_t, min_stable_delay):
  v_now = speeds[0]
  if action_t < min_stable_delay:
    v_target = v_now + (action_t / min_stable_delay) * (np.interp(min_stable_delay, ModelConstants.T_IDXS, speeds) - v_now)
  else:
    v_target = np.interp(action_t, ModelConstants.T_IDXS, speeds)

  a_target = 2 * (v_target - v_now) / action_t - accels[0]
  should_stop = v_now < V_EGO_STOPPING and a_target < 0.1
  return a_target, should_stop


def get_stable_curvature_from_plan(yaws, yaw_rates, v_ego, action_t, min_stable_delay):
  if action_t < min_stable_delay:
    psi_target = (action_t / min_stable_delay) * np.interp(min_stable_delay, ModelConstants.T_IDXS, yaws)
  else:
    psi_target = np.interp(action_t, ModelConstants.T_IDXS, yaws)

  return curv_from_psis(psi_target, yaw_rates[0], v_ego, action_t)


def get_action_from_model(model_output: dict[str, np.ndarray], prev_action: log.ModelDataV2.Action,
                          lat_action_t: float, long_action_t: float, v_ego: float, use_curvature_from_plan: bool,
                          lat_smooth_seconds: float, current_speed_stop: bool, min_stable_delay: float) -> log.ModelDataV2.Action:
    plan = model_output['plan'][0]
    if current_speed_stop:
      desired_accel, should_stop = get_current_speed_accel_from_plan(plan[:,Plan.VELOCITY][:,0],
                                                                     plan[:,Plan.ACCELERATION][:,0],
                                                                     long_action_t,
                                                                     min_stable_delay)
    else:
      desired_accel, should_stop = get_accel_from_plan(plan[:,Plan.VELOCITY][:,0],
                                                       plan[:,Plan.ACCELERATION][:,0],
                                                       ModelConstants.T_IDXS,
                                                       action_t=long_action_t)
    desired_accel = smooth_value(desired_accel, prev_action.desiredAcceleration, LONG_SMOOTH_SECONDS)

    if use_curvature_from_plan:
      desired_curvature = get_stable_curvature_from_plan(plan[:,Plan.T_FROM_CURRENT_EULER][:,2],
                                                         plan[:,Plan.ORIENTATION_RATE][:,2],
                                                         v_ego,
                                                         lat_action_t,
                                                         min_stable_delay)
    else:
      desired_curvature = model_output['desired_curvature'][0, 0]

    if v_ego > MIN_LAT_CONTROL_SPEED:
      desired_curvature = smooth_value(desired_curvature, prev_action.desiredCurvature, lat_smooth_seconds)
    else:
      desired_curvature = prev_action.desiredCurvature

    return log.ModelDataV2.Action(desiredCurvature=float(desired_curvature),
                                  desiredAcceleration=float(desired_accel),
                                  shouldStop=bool(should_stop))


class FrameMeta:
  frame_id: int = 0
  timestamp_sof: int = 0
  timestamp_eof: int = 0

  def __init__(self, vipc=None):
    if vipc is not None:
      self.frame_id, self.timestamp_sof, self.timestamp_eof = vipc.frame_id, vipc.timestamp_sof, vipc.timestamp_eof


def main(demo=False):
  # FrogPilot variables
  frogpilot_toggles = frogpilot_variables.get_frogpilot_toggles()

  model_name = frogpilot_toggles.model
  lat_smooth_seconds = frogpilot_utilities.get_model_lat_smooth_seconds(model_name)
  use_curvature_from_plan = not frogpilot_utilities.use_desired_curvature_action(model_name)
  current_speed_stop = frogpilot_utilities.use_current_speed_stop(model_name)

  action_delay = 0
  if frogpilot_utilities.use_action_delay(model_name):
    action_delay = DT_MDL / 2

  min_stable_delay = 0
  if frogpilot_utilities.use_min_stable_delay(model_name):
    min_stable_delay = MIN_STABLE_DELAY

  setproctitle(PROCESS_NAME)
  cloudlog.warning("frogpilot modeld_gpu init")

  config_realtime_process(7, 54)

  st = time.monotonic()
  cloudlog.warning("loading CUDA/TensorRT model")
  model = FrogPilotGpuState()
  cloudlog.warning(f"CUDA model loaded in {time.monotonic() - st:.1f}s, modeld_gpu starting")

  # visionipc clients (no CLContext: CUDA path ignores it)
  while True:
    available_streams = VisionIpcClient.available_streams("camerad", block=False)
    if available_streams:
      use_extra_client = VisionStreamType.VISION_STREAM_WIDE_ROAD in available_streams and VisionStreamType.VISION_STREAM_ROAD in available_streams
      main_wide_camera = VisionStreamType.VISION_STREAM_ROAD not in available_streams
      break
    time.sleep(.1)

  vipc_client_main_stream = VisionStreamType.VISION_STREAM_WIDE_ROAD if main_wide_camera else VisionStreamType.VISION_STREAM_ROAD
  vipc_client_main = VisionIpcClient("camerad", vipc_client_main_stream, True, None)
  vipc_client_extra = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD, False, None)
  cloudlog.warning(f"vision stream set up, main_wide_camera: {main_wide_camera}, use_extra_client: {use_extra_client}")

  while not vipc_client_main.connect(False):
    time.sleep(0.1)
  while use_extra_client and not vipc_client_extra.connect(False):
    time.sleep(0.1)

  cloudlog.warning(f"connected main cam with buffer size: {vipc_client_main.buffer_len} ({vipc_client_main.width} x {vipc_client_main.height})")
  if use_extra_client:
    cloudlog.warning(f"connected extra cam with buffer size: {vipc_client_extra.buffer_len} ({vipc_client_extra.width} x {vipc_client_extra.height})")

  # messaging
  pm = PubMaster(["modelV2", "drivingModelData", "cameraOdometry", "frogpilotModelV2"])
  sm = SubMaster(["deviceState", "carState", "roadCameraState", "liveCalibration", "driverMonitoringState", "carControl", "liveDelay", "frogpilotPlan"])

  publish_state = PublishState()
  params = Params()

  # setup filter to track dropped frames
  frame_dropped_filter = FirstOrderFilter(0., 10., 1. / ModelConstants.MODEL_RUN_FREQ)
  frame_id = 0
  last_vipc_frame_id = 0
  run_count = 0

  model_transform_main = np.zeros((3, 3), dtype=np.float32)
  model_transform_extra = np.zeros((3, 3), dtype=np.float32)
  live_calib_seen = False
  buf_main, buf_extra = None, None
  meta_main = FrameMeta()
  meta_extra = FrameMeta()

  if demo:
    CP = get_demo_car_params()
  else:
    with car.CarParams.from_bytes(params.get("CarParams", block=True)) as msg:
      CP = msg
  cloudlog.info("modeld_gpu got CarParams: %s", CP.carName)

  prev_action = log.ModelDataV2.Action()

  DH = DesireHelper()

  while True:
    # Keep receiving frames until we are at least 1 frame ahead of previous extra frame.
    # 注意: 每轮 recv() 都会 acquire 一个 buffer; 覆盖 buf_main 之前必须先 release 上一个,
    # 否则这些"追赶丢弃"的 frame 只进不出, ref_count 会累加到 ~18/19,
    # camerad 的 get_buffer 每次轮转到这些 buffer 都白等 200ms → 帧率塌到 ~5Hz。
    while meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
      next_main = vipc_client_main.recv()
      if next_main is None:
        break
      if buf_main is not None:
        buf_main.release()
      buf_main = next_main
      meta_main = FrameMeta(vipc_client_main)

    if buf_main is None:
      cloudlog.debug("vipc_client_main no frame")
      continue

    if use_extra_client:
      # Keep receiving extra frames until frame id matches main camera
      while True:
        next_extra = vipc_client_extra.recv()
        if next_extra is None:
          break
        if buf_extra is not None:
          buf_extra.release()
        buf_extra = next_extra
        meta_extra = FrameMeta(vipc_client_extra)
        if meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
          break

      if buf_extra is None:
        cloudlog.debug("vipc_client_extra no frame")
        # buf_main 已经 recv(acquire) 了, 直接 continue 会把它的 ref 泄漏掉,
        # 池子几轮就耗尽 -> camerad 卡在 get_buffer。
        buf_main.release()
        continue

      if abs(meta_main.timestamp_sof - meta_extra.timestamp_sof) > 10000000:
        cloudlog.error(f"frames out of sync! main: {meta_main.frame_id} ({meta_main.timestamp_sof / 1e9:.5f}),\
                         extra: {meta_extra.frame_id} ({meta_extra.timestamp_sof / 1e9:.5f})")

    else:
      # Use single camera
      buf_extra = buf_main
      meta_extra = meta_main

    sm.update(0)
    desire = DH.desire
    is_rhd = sm["driverMonitoringState"].isRHD
    frame_id = sm["roadCameraState"].frameId
    v_ego = max(sm["carState"].vEgo, 0.)
    lat_delay = sm["liveDelay"].lateralDelay + lat_smooth_seconds
    lateral_control_params = np.array([v_ego, lat_delay], dtype=np.float32)
    long_delay = frogpilot_toggles.longitudinalActuatorDelay + LONG_SMOOTH_SECONDS
    if sm.updated["liveCalibration"] and sm.seen['roadCameraState'] and sm.seen['deviceState']:
      device_from_calib_euler = np.array(sm["liveCalibration"].rpyCalib, dtype=np.float32)
      dc = DEVICE_CAMERAS[(str(sm['deviceState'].deviceType), str(sm['roadCameraState'].sensor))]
      model_transform_main = get_warp_matrix(device_from_calib_euler, dc.ecam.intrinsics if main_wide_camera else dc.fcam.intrinsics, False).astype(np.float32)
      model_transform_extra = get_warp_matrix(device_from_calib_euler, dc.ecam.intrinsics, True).astype(np.float32)
      live_calib_seen = True

    traffic_convention = np.zeros(2)
    traffic_convention[int(is_rhd)] = 1

    vec_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)
    if desire >= 0 and desire < ModelConstants.DESIRE_LEN:
      vec_desire[desire] = 1

    # tracked dropped frames
    vipc_dropped_frames = max(0, meta_main.frame_id - last_vipc_frame_id - 1)
    frames_dropped = frame_dropped_filter.update(min(vipc_dropped_frames, 10))
    if run_count < 10:  # let frame drops warm up
      frame_dropped_filter.x = 0.
      frames_dropped = 0.
    run_count = run_count + 1

    frame_drop_ratio = frames_dropped / (1 + frames_dropped)
    prepare_only = vipc_dropped_frames > 0
    if prepare_only:
      cloudlog.error(f"skipping model eval. Dropped {vipc_dropped_frames} frames")

    bufs = {name: buf_extra if 'big' in name else buf_main for name in model.vision_input_names}
    transforms = {name: model_transform_extra if 'big' in name else model_transform_main for name in model.vision_input_names}
    inputs = {
      model.desire_type: vec_desire,
      'traffic_convention': traffic_convention,
      **({'lateral_control_params': lateral_control_params} if model.use_lateral_control_params else {}),
    }

    mt1 = time.perf_counter()
    model_output = model.run(bufs, transforms, inputs, prepare_only)
    mt2 = time.perf_counter()
    model_execution_time = mt2 - mt1

    if model_output is not None:
      modelv2_send = messaging.new_message('modelV2')
      frogpilot_modelv2_send = messaging.new_message('frogpilotModelV2')
      drivingdata_send = messaging.new_message('drivingModelData')
      posenet_send = messaging.new_message('cameraOdometry')

      action = get_action_from_model(model_output, prev_action, lat_delay + DT_MDL + action_delay, long_delay + DT_MDL + action_delay, v_ego,
                                     use_curvature_from_plan, lat_smooth_seconds, current_speed_stop, min_stable_delay)
      prev_action = action
      fill_model_msg(drivingdata_send, modelv2_send, model_output, action,
                     publish_state, meta_main.frame_id, meta_extra.frame_id, frame_id,
                     frame_drop_ratio, meta_main.timestamp_eof, model_execution_time, live_calib_seen)

      desire_state = modelv2_send.modelV2.meta.desireState
      l_lane_change_prob = desire_state[log.Desire.laneChangeLeft]
      r_lane_change_prob = desire_state[log.Desire.laneChangeRight]
      lane_change_prob = l_lane_change_prob + r_lane_change_prob
      DH.update(sm['carState'], sm['carControl'].latActive, lane_change_prob, sm['frogpilotPlan'], frogpilot_toggles)
      modelv2_send.modelV2.meta.laneChangeState = DH.lane_change_state
      modelv2_send.modelV2.meta.laneChangeDirection = DH.lane_change_direction
      frogpilot_modelv2_send.valid = modelv2_send.valid
      frogpilot_modelv2_send.frogpilotModelV2.turnDirection = DH.turn_direction
      drivingdata_send.drivingModelData.meta.laneChangeState = DH.lane_change_state
      drivingdata_send.drivingModelData.meta.laneChangeDirection = DH.lane_change_direction

      fill_pose_msg(posenet_send, model_output, meta_main.frame_id, vipc_dropped_frames, meta_main.timestamp_eof, live_calib_seen)
      pm.send('modelV2', modelv2_send)
      pm.send('frogpilotModelV2', frogpilot_modelv2_send)
      pm.send('drivingModelData', drivingdata_send)
      pm.send('cameraOdometry', posenet_send)
    last_vipc_frame_id = meta_main.frame_id

    # Update FrogPilot variables
    frogpilot_toggles = frogpilot_variables.get_frogpilot_toggles(sm)

    # 归还 VisionIPC buffer。缺这两行会让 buffer 池只出不进:
    # camerad 的 get_buffer 卡在 "buffer busy for >200ms (ref=1)" 上,
    # 采集+发布从 20Hz 掉到 2Hz (实测: 屏蔽 modeld 时 20.00fps, 有 modeld 时 2.53fps)。
    # 本版 visionipc 的 API 是 VisionBuf.release() (没有 VisionIpcClient.release_buffer),
    # recv() 内部会 acquire, 所以这里 release 是配对的。
    if buf_main is not None:
      buf_main.release()
    if use_extra_client and buf_extra is not None and buf_extra is not buf_main:
      buf_extra.release()


if __name__ == "__main__":
  try:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='A boolean for demo mode.')
    args = parser.parse_args()
    main(demo=args.demo)
  except KeyboardInterrupt:
    cloudlog.warning("got SIGINT")