#!/usr/bin/env python3
"""BigCombo merged vision+policy modeld — standalone module (2026-09-02).

单引擎大模型分支，完全独立于 selfdrive/modeld/modeld.py 的 FiletOFish 分离架构：
- modeld.py 只加 2 行 dispatch（Params Model == "BigCombo" 时转发到这里），原路径零改动
- 单引擎: driving_supercombo_fp16.plan (1.76GB, TRT 10.16.2, fp16)
- 输入: img/big_img [1,12,128,256] uint8 + desire_pulse[1,25,8] + traffic_convention[1,2]
        + action_t[1,2] + features_buffer[1,24,512] (fp16, runner 自动转换)
- 输出: 2580 维单 Concat（切片表来自 ONNX 官方 output_slices，已验证 debug_bigcombo_layout.py PASS）
- 时序: features_buffer 24 格不含当前帧（图内自动拼当前 hidden），desire_pulse 25 格含当前帧
- action_t v1: [vEgo, steerDelay+0.2]（语义未定，影响极小；真车回放 A/B 后校准）
- action 输出头(4维) v1 忽略：desiredCurvature 走 plan 回退路径（fill_model_msg 现有支持）

复用清单（与 modeld.py 完全同款，避免行为分叉）：
  DrivingModelFrame/CLContext, CUDA transform zero-copy, TensorRTModel, Parser,
  fill_model_msg/fill_pose_msg/PublishState, DesireHelper, VisionIpcClient,
  FrameMeta, drop-filter, Model param 热切换(每100帧重启)
差异点（相对 modeld.py）：
  ① 单 runner（use_trt 单引擎，无 policy 引擎/无 tinygrad policy pkl）
  ② features temporal_idxs = slice(-1-(4*23), None, 4) → 24 格
  ③ vision/policy 切片独立（2580 单数组切两次再合并，不能对同数组重复解析）
  ④ 喂 action_t 输入
"""
import os
os.environ['LRU'] = '0'  # Disable LRU GPU alloc cache early (tinygrad import reads it)
import ctypes
import pickle
import time
import numpy as np
from pathlib import Path

from openpilot.system.hardware import TICI
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.device import Device
if TICI:
  from openpilot.selfdrive.modeld.runners.tinygrad_helpers import qcom_tensor_from_opencl_address
  os.environ['QCOM'] = '1'
else:
  os.environ['CUDA'] = '1'

import cereal.messaging as messaging
from cereal import car, log
from setproctitle import setproctitle
from cereal.messaging import PubMaster, SubMaster
from msgq.visionipc import VisionIpcClient, VisionStreamType, VisionBuf
from opendbc.car.car_helpers import get_demo_car_params
from openpilot.common.swaglog import cloudlog
from openpilot.common.params import Params
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import config_realtime_process, DT_MDL
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.system import sentry
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper


def _get_lat_delay(sm) -> float:
  # 本地 params 白名单无 LagdValueCache 键(上游专用), 直接用 lagd 实时发布的 lateralDelay;
  # 无效/未就绪时回退 0.2s 经验值
  try:
    v = float(sm["liveDelay"].lateralDelay)
  except Exception:
    return 0.2
  return v if (np.isfinite(v) and v > 0.0) else 0.2

from openpilot.selfdrive.modeld.parse_model_outputs import Parser
from openpilot.selfdrive.modeld.fill_model_msg import fill_model_msg, fill_pose_msg, PublishState
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.modeld.models.commonmodel_pyx import DrivingModelFrame, CLContext

PROCESS_NAME = "selfdrive.modeld.modeld_bigcombo"
SEND_RAW_PRED = os.getenv('SEND_RAW_PRED')

LAT_SMOOTH_SECONDS = 0.0
LONG_SMOOTH_SECONDS = 0.3

DEFAULT_MODEL_DIR = Path(__file__).parent / 'models'

# ── 2580-dim merged output layout (verified vs ONNX official output_slices, 2026-09-02) ──
VISION_SLICES = {
  'lane_lines': slice(0, 528),
  'lane_lines_prob': slice(528, 536),
  'road_edges': slice(536, 800),
  'meta': slice(800, 855),
  'desire_pred': slice(855, 887),
  'pose': slice(887, 899),
  'wide_from_device_euler': slice(899, 905),
  'road_transform': slice(905, 917),
}
POLICY_SLICES = {
  'plan': slice(917, 1907),
  'lane_lines': slice(0, 528),
  'lane_lines_prob': slice(528, 536),
  'road_edges': slice(536, 800),
  'lead': slice(1907, 2051),
  'lead_prob': slice(2051, 2054),
  'desire_state': slice(2054, 2062),
  'action': slice(2062, 2066),
  'hidden_state': slice(2066, 2578),
  'pad': slice(2578, 2580),
}
OUTPUT_TOTAL = 2580
TRT_PLAN_NAME = 'driving_supercombo_fp16.plan'

# TRT 加载兜底闭环参数(与 modeld.py 一致, 可用环境变量覆盖)
TRT_LOAD_ATTEMPTS = int(os.getenv("TRT_LOAD_ATTEMPTS", "3"))
TRT_LOAD_RETRY_INTERVAL = float(os.getenv("TRT_LOAD_RETRY_INTERVAL", "2.0"))
METADATA_NAME = 'big_driving_supercombo_metadata.pkl'


def get_model_name():
  return Params().get("Model", encoding='utf-8') or "FiletOFish"


def is_bigcombo():
  return get_model_name() == "BigCombo"


class FrameMeta:
  frame_id: int = 0
  timestamp_sof: int = 0
  timestamp_eof: int = 0

  def __init__(self, vipc=None):
    if vipc is not None:
      self.frame_id, self.timestamp_sof, self.timestamp_eof = vipc.frame_id, vipc.timestamp_sof, vipc.timestamp_eof


class BigComboTrtUnavailable(Exception):
  """BigCombo 合并引擎 TRT 加载彻底失败, 无 tinygrad fallback, 需降级 FiletOFish。"""


class ModelState:
  frames: dict[str, DrivingModelFrame]
  inputs: dict[str, np.ndarray]
  output: np.ndarray
  prev_desire: np.ndarray  # for tracking the rising edge of the pulse

  def __init__(self, context: CLContext):
    model_dir = DEFAULT_MODEL_DIR / "BigCombo"
    metadata_path = model_dir / METADATA_NAME

    with open(metadata_path, 'rb') as f:
      metadata = pickle.load(f)
      self.input_shapes = metadata['input_shapes']
      self.output_slices = metadata['output_slices']  # official; values match VISION/POLICY_SLICES above
      output_size = metadata['output_shapes']['outputs'][1]
    assert output_size == OUTPUT_TOTAL, f'BigCombo output size {output_size} != {OUTPUT_TOTAL}'

    self.input_names = list(self.input_shapes.keys())
    self.vision_input_names = [n for n in self.input_names if n in ('img', 'big_img')]

    # ── dimensions derived from metadata ──
    self.temporal_skip = ModelConstants.TEMPORAL_SKIP  # 4
    self.desire_len = self.input_shapes['desire_pulse'][2]           # 8
    self.desire_slots = self.input_shapes['desire_pulse'][1]         # 25 (incl current)
    self.feature_len = self.input_shapes['features_buffer'][2]       # 512
    self.feature_slots = self.input_shapes['features_buffer'][1]     # 24 (excl current — graph concats hidden in-graph)
    self.full_history_buffer_len = ModelConstants.FULL_HISTORY_BUFFER_LEN  # 100
    self.full_features_buffer = np.zeros((1, self.full_history_buffer_len, self.feature_len), dtype=np.float32)
    self.full_desire = np.zeros((1, self.full_history_buffer_len, self.desire_len), dtype=np.float32)
    # desire: 25 slots incl current  → idxs -1-(4*24) step 4
    self.desire_idxs = slice(-1 - (self.temporal_skip * (self.desire_slots - 1)), None, self.temporal_skip)
    # features: 24 slots excl current → idxs -1-(4*23) step 4  (past hidden only)
    self.features_idxs = slice(-1 - (self.temporal_skip * (self.feature_slots - 1)), None, self.temporal_skip)

    # ── frame pipeline (two streams, same as modeld.py) ──
    self.frames = {name: DrivingModelFrame(context, self.temporal_skip) for name in ('input_imgs', 'big_input_imgs')}
    self.prev_desire = np.zeros(self.desire_len, dtype=np.float32)

    self.numpy_inputs = {name: np.zeros(shape, dtype=np.float32) for name, shape in self.input_shapes.items()}

    # img buffers are managed in CUDA transform code
    self.vision_input_ptrs: dict[str, int] = {}  # CUDA device ptrs for TRT zero-copy
    self.output = np.zeros(output_size, dtype=np.float32)
    self.parser = Parser()

    # ── CUDA transform pipeline (bypass OpenCL on non-TICI) — identical to modeld.py ──
    self._cuda_transform = None
    if not TICI:
      try:
        transform_dir = Path(__file__).parent / 'transforms'
        lib_path = transform_dir / 'libcuda_transform.so'
        if lib_path.exists():
          self._cu_transform = ctypes.CDLL(str(lib_path))
          self._cu_transform.cuda_transform_init.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
          self._cu_transform.cuda_transform_init.restype = None
          self._cu_transform.cuda_transform_execute.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
          self._cu_transform.cuda_transform_execute.restype = ctypes.c_void_p
          self._cu_transform.cuda_transform_destroy.argtypes = [ctypes.c_void_p]
          self._cu_transform.cuda_transform_destroy.restype = None

          self._cuda_states = {}
          # Derive model dims from vision input shape: (1, channels, h, w)
          # Each sub-plane in transform output is (w, h) → model input channel is (h, w)
          # MODEL_W = w * 2, MODEL_H = h * 2  (identical to modeld.py derivation)
          _model_input_shape = self.input_shapes['img']
          _model_h, _model_w = _model_input_shape[2], _model_input_shape[3]
          self._model_w = _model_w * 2
          self._model_h = _model_h * 2
          for key in self.frames:
            state = ctypes.create_string_buffer(128)
            self._cu_transform.cuda_transform_init(
              ctypes.byref(state), self._model_w, self._model_h, self.temporal_skip
            )
            self._cuda_states[key] = state
          self._cuda_transform = True
          cloudlog.warning("[BigCombo] CUDA transform initialized (OpenCL bypassed)")
      except Exception as e:
        cloudlog.warning(f"[BigCombo] CUDA transform not available: {e}")
        self._cuda_transform = False

    # ── Force tinygrad CUDA device init BEFORE TRT (context sharing, same as modeld.py) ──
    _ = Device['CUDA']

    # ── Single TRT engine (mandatory for BigCombo) ──
    # 兜底闭环: 重试 TRT_LOAD_ATTEMPTS 次(间隔 TRT_LOAD_RETRY_INTERVAL); 彻底失败 →
    # 抛 BigComboTrtUnavailable 由 main() 切回 FiletOFish 重启 (合并引擎无 tinygrad fallback)。
    trt_engine_path = model_dir / TRT_PLAN_NAME
    self.trt_fail_reason = ""
    for attempt in range(1, TRT_LOAD_ATTEMPTS + 1):
      try:
        from openpilot.selfdrive.modeld.runners.tensorrt_runner import TensorRTModel
        self.combo_run = TensorRTModel(str(trt_engine_path))
        cloudlog.warning(f"[BigCombo] Using TensorRT merged engine (FP16): {trt_engine_path}")
        break
      except Exception as e:
        self.trt_fail_reason = f"{e}"
        if attempt < TRT_LOAD_ATTEMPTS:
          cloudlog.warning(f"[BigCombo] TRT load failed ({e}), retry {attempt}/{TRT_LOAD_ATTEMPTS} in {TRT_LOAD_RETRY_INTERVAL}s")
          print(f"[WARN] [BigCombo] TRT load failed: {e}, retry {attempt}/{TRT_LOAD_ATTEMPTS}")
          time.sleep(TRT_LOAD_RETRY_INTERVAL)
        else:
          cloudlog.warning(f"[BigCombo] TRT load failed after {TRT_LOAD_ATTEMPTS} attempts ({e}); merged engine has no tinygrad fallback, switching to FiletOFish")
          print(f"[WARN] [BigCombo] TRT load failed after {TRT_LOAD_ATTEMPTS} attempts: {e}; switching to FiletOFish")
          raise BigComboTrtUnavailable(str(e))

    # engine input order/dtype sanity (uint8 imgs + fp16 scalars; runner auto-converts)
    eng_names = set(self.combo_run.input_names)
    if eng_names != set(self.input_names):
      raise RuntimeError(f"[BigCombo] engine inputs {eng_names} != metadata inputs {set(self.input_names)}")

  def run(self, buf: VisionBuf, wbuf: VisionBuf, transform: np.ndarray, transform_wide: np.ndarray,
          inputs: dict[str, np.ndarray], prepare_only: bool) -> dict[str, np.ndarray] | None:
    # Model decides when action is completed, so desire input is just a pulse triggered on rising edge
    inputs['desire'][0] = 0
    new_desire = np.where(inputs['desire'] - self.prev_desire > .99, inputs['desire'], 0)
    self.prev_desire[:] = inputs['desire']

    self.full_desire[0, :-1] = self.full_desire[0, 1:]
    self.full_desire[0, -1] = new_desire

    # policy-ish inputs by name
    desire_hist = self.full_desire.reshape((1, self.desire_slots, self.temporal_skip, -1)).max(axis=2)
    if 'desire_pulse' in self.numpy_inputs:
      self.numpy_inputs['desire_pulse'][:] = desire_hist
    if 'traffic_convention' in self.numpy_inputs:
      self.numpy_inputs['traffic_convention'][:] = inputs['traffic_convention']
    if 'action_t' in self.numpy_inputs:
      # 上游官方语义: [lat_action_t, long_action_t] 秒级时间量(0.2~0.6s), 由 main() 每帧传入
      self.numpy_inputs['action_t'][:] = inputs.get('action_t', np.array([0.35, 0.65], dtype=np.float32))

    if TICI:
      imgs = {'input_imgs': self.frames['input_imgs'].prepare(buf, transform.flatten()),
              'big_input_imgs': self.frames['big_input_imgs'].prepare(wbuf, transform_wide.flatten())}
      eng_keys = {'input_imgs': 'img', 'big_input_imgs': 'big_img'}
      for frame_key, eng_key in eng_keys.items():
        if eng_key not in self.vision_input_ptrs:
          self.vision_input_ptrs[eng_key] = imgs[frame_key].mem_address  # qcom: opencl mem == cuda mem on TICI
    elif self._cuda_transform:
      for frame_key, eng_key, fbuf, proj in (
        ('input_imgs', 'img', buf, transform),
        ('big_input_imgs', 'big_img', wbuf, transform_wide),
      ):
        state = self._cuda_states[frame_key]
        c_ptr = fbuf.data.ctypes.data
        buf_len = fbuf.data.nbytes
        output_ptr = self._cu_transform.cuda_transform_execute(
          ctypes.byref(state), ctypes.c_void_p(c_ptr),
          fbuf.width, fbuf.height, fbuf.stride, fbuf.uv_offset,
          buf_len,
          proj.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
          0  # input is host (SHM) memory
        )
        # 修复: BigCombo 单引擎下, TRT 上下文读不到 transform 缓冲的每帧更新
        # (FOF 双引擎无此问题), 现象=输出比特级冻结。将 transform 输出 D2H 拷回主机,
        # 走 runner 的 H2D 路径喂引擎(离线已证可靠), 每帧仅 393KB。
        if not hasattr(self, '_img_host'):
          self._img_host = {}
        if eng_key not in self._img_host:
          self._img_host[eng_key] = np.zeros(self.input_shapes[eng_key], dtype=np.uint8)
        h = self._img_host[eng_key]
        runner = self.combo_run
        cu = runner.cu
        ret = cu.cuMemcpyDtoHAsync_v2(ctypes.c_void_p(h.ctypes.data),
                                      int(output_ptr),
                                      h.nbytes, runner.stream)
        if ret != 0:
          raise RuntimeError(f"[BigCombo] D2H copy failed for {eng_key}: CUDA error {ret}")
        cu.cuStreamSynchronize(runner.stream)
        self.vision_input_ptrs[eng_key] = h  # numpy → runner H2D
    else:
      # pure-CPU fallback (no CUDA transform lib): tinygrad CL path like modeld.py else-branch
      imgs = {'input_imgs': self.frames['input_imgs'].prepare(buf, transform.flatten()),
              'big_input_imgs': self.frames['big_input_imgs'].prepare(wbuf, transform_wide.flatten())}
      eng_keys = {'input_imgs': 'img', 'big_input_imgs': 'big_img'}
      for frame_key, eng_key in eng_keys.items():
        frame_input = self.frames[frame_key].buffer_from_cl(imgs[frame_key]).reshape(self.input_shapes[eng_key])
        self.vision_input_ptrs[eng_key] = frame_input.ctypes.data  # host ptr → runner H2D copy

    if prepare_only:
      return None

    # ── Single merged inference (zero-copy imgs + numpy scalars) ──
    # 修复(0902晚真根因): numpy_inputs 曾包含 img/big_img 全零数组, update 时覆盖了
    # vision 指针 → 引擎每帧吃全零图 → 输出冻结/彩虹不显示/车道线固定。
    # 现在标量字典排除图像键, 并让图像指针最后覆盖, 保证引擎吃到每帧新鲜图像。
    kwargs = {k: v for k, v in self.numpy_inputs.items() if k not in self.vision_input_ptrs}
    kwargs.update(self.vision_input_ptrs)
    self.output = np.asarray(self.combo_run(**kwargs)).ravel()
    assert self.output.size == OUTPUT_TOTAL and np.isfinite(self.output).all(), "[BigCombo] output diverged"

    # ── Parse: vision slices first, then policy slices independently (merged-array rule) ──
    vision_outs = {k: self.output[v][np.newaxis] for k, v in VISION_SLICES.items()}
    policy_outs = {k: self.output[v][np.newaxis] for k, v in POLICY_SLICES.items()}
    vision_dict = self.parser.parse_vision_outputs(vision_outs)
    policy_dict = self.parser.parse_policy_outputs(policy_outs)
    combined = {**vision_dict, **policy_dict}
    if SEND_RAW_PRED:
      combined['raw_pred'] = self.output.copy()

    # ── temporal feedback: hidden_state (current frame, from THIS run) → features_buffer ──
    self.full_features_buffer[0, :-1] = self.full_features_buffer[0, 1:]
    self.full_features_buffer[0, -1] = combined['hidden_state'][0, :]
    if 'features_buffer' in self.numpy_inputs:
      self.numpy_inputs['features_buffer'][:] = self.full_features_buffer[0, self.features_idxs]

    return combined


def main(demo=False):
  cloudlog.warning("[BigCombo] modeld init")
  sentry.set_tag("daemon", PROCESS_NAME)
  cloudlog.bind(daemon=PROCESS_NAME)
  setproctitle(PROCESS_NAME)
  # BigCombo 单帧贴预算(51ms/50ms), 独占 core 8 + SCHED_FIFO 54:
  # core 7 是原版 modeld 惯例位(camerad/webcamerad 同在 0-7 抢核), 用 8 隔离更稳
  config_realtime_process(8, 54)

  cloudlog.warning("[BigCombo] setting up CL context")
  cl_context = CLContext()
  cloudlog.warning("[BigCombo] CL context ready; loading model")
  try:
    model = ModelState(cl_context)
  except BigComboTrtUnavailable as e:
    # 降级闭环: 合并引擎 TRT 不可用 → 切 Params Model=FiletOFish 并退出,
    # manager 重启 modeld → modeld.py dispatch 走 FiletOFish 路径
    Params().put("Model", "FiletOFish")
    cloudlog.warning(f"[BigCombo] TRT unavailable, switched Model=FiletOFish, restarting modeld ({e})")
    print(f"[WARN] [BigCombo] TRT unavailable, switched Model=FiletOFish, restarting modeld: {e}")
    raise SystemExit(0)
  cloudlog.warning("[BigCombo] models loaded, modeld starting")

  # visionipc clients (same as modeld.py)
  while True:
    available_streams = VisionIpcClient.available_streams("camerad", block=False)
    if available_streams:
      use_extra_client = VisionStreamType.VISION_STREAM_WIDE_ROAD in available_streams and VisionStreamType.VISION_STREAM_ROAD in available_streams
      main_wide_camera = VisionStreamType.VISION_STREAM_ROAD not in available_streams
      break
    time.sleep(.1)

  vipc_client_main_stream = VisionStreamType.VISION_STREAM_WIDE_ROAD if main_wide_camera else VisionStreamType.VISION_STREAM_ROAD
  vipc_client_main = VisionIpcClient("camerad", vipc_client_main_stream, True, cl_context)
  vipc_client_extra = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD, True, cl_context)
  cloudlog.warning(f"[BigCombo] vision stream set up, main_wide_camera: {main_wide_camera}, use_extra_client: {use_extra_client}")

  while not vipc_client_main.connect(False):
    time.sleep(0.1)
  while use_extra_client and not vipc_client_extra.connect(False):
    time.sleep(0.1)

  cloudlog.warning(f"[BigCombo] connected main cam with buffer size: {vipc_client_main.buffer_len} ({vipc_client_main.width} x {vipc_client_main.height})")
  if use_extra_client:
    cloudlog.warning(f"[BigCombo] connected extra cam with buffer size: {vipc_client_extra.buffer_len} ({vipc_client_extra.width} x {vipc_client_extra.height})")

  # messaging
  pm = PubMaster(["modelV2", "drivingModelData", "cameraOdometry"])
  sm = SubMaster(["deviceState", "carState", "roadCameraState", "liveCalibration", "driverMonitoringState", "carControl", "liveDelay"])

  publish_state = PublishState()
  params = Params()

  loaded_model_name = get_model_name()
  model_check_counter = 0

  # setup filter to track dropped frames
  frame_dropped_filter = FirstOrderFilter(0., 10., 1. / ModelConstants.MODEL_FREQ)
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
    CP = messaging.log_from_bytes(params.get("CarParams", block=True), car.CarParams)
  cloudlog.info("[BigCombo] modeld got CarParams: %s", CP.brand)

  steer_delay = CP.steerActuatorDelay + .2
  long_delay = CP.longitudinalActuatorDelay + LONG_SMOOTH_SECONDS

  DH = DesireHelper()

  while True:
    # Check if model was changed via params (same hot-switch mechanism as modeld.py)
    model_check_counter += 1
    if model_check_counter % 100 == 0:
      current_model_name = params.get("Model", encoding='utf-8') or "FiletOFish"
      if current_model_name != loaded_model_name:
        cloudlog.warning(f"[BigCombo] Model changed from {loaded_model_name} to {current_model_name}, restarting")
        raise SystemExit(0)

    buf_main = vipc_client_main.recv()
    meta_main = FrameMeta(vipc_client_main)
    if buf_main is None:
      cloudlog.debug("vipc_client_main no frame")
      continue

    if use_extra_client:
      buf_extra = vipc_client_extra.recv()
      meta_extra = FrameMeta(vipc_client_extra)
      if buf_extra is None:
        cloudlog.debug("vipc_client_extra no frame")
        continue
      # Decoupled FrameSync: ±1 skew is steady state; >1 is an anomaly
      if meta_main.frame_id != meta_extra.frame_id:
        if abs(meta_main.frame_id - meta_extra.frame_id) > 1:
          cloudlog.warning(f"[BigCombo] frame_id mismatch (anomaly): main={meta_main.frame_id} extra={meta_extra.frame_id}")
        else:
          cloudlog.debug(f"[BigCombo] frame_id skew 1 (steady state): main={meta_main.frame_id} extra={meta_extra.frame_id}")
    else:
      buf_extra = buf_main
      meta_extra = meta_main

    sm.update(0)
    desire = DH.desire
    is_rhd = sm["driverMonitoringState"].isRHD
    frame_id = sm["roadCameraState"].frameId
    v_ego = max(sm["carState"].vEgo, 0.)
    if sm.updated["liveCalibration"] and sm.seen['roadCameraState'] and sm.seen['deviceState']:
      device_from_calib_euler = np.array(sm["liveCalibration"].rpyCalib, dtype=np.float32)
      dc = DEVICE_CAMERAS[(str(sm['deviceState'].deviceType), str(sm['roadCameraState'].sensor))]
      model_transform_main = get_warp_matrix(device_from_calib_euler, dc.ecam.intrinsics if main_wide_camera else dc.fcam.intrinsics, False).astype(np.float32)
      model_transform_extra = get_warp_matrix(device_from_calib_euler, dc.ecam.intrinsics, True).astype(np.float32)
      live_calib_seen = True

    traffic_convention = np.zeros(2)
    traffic_convention[int(is_rhd)] = 1

    vec_desire = np.zeros(model.desire_len, dtype=np.float32)
    if desire >= 0 and desire < model.desire_len:
      vec_desire[desire] = 1

    # tracked dropped frames
    vipc_dropped_frames = max(0, meta_main.frame_id - last_vipc_frame_id - 1)
    frames_dropped = frame_dropped_filter.update(min(vipc_dropped_frames, 10))
    if run_count < 10:
      frame_dropped_filter.x = 0.
      frames_dropped = 0.
    run_count = run_count + 1

    frame_drop_ratio = frames_dropped / (1 + frames_dropped)
    prepare_only = vipc_dropped_frames > 0
    if prepare_only:
      cloudlog.error(f"[BigCombo] skipping model eval. Dropped {vipc_dropped_frames} frames")

    # action_t 上游官方语义: [lat_action_t, long_action_t] 都是秒级时间量
    # (修复: v1 曾误喂 [v_ego, steer_delay], 车速被当成时间导致输出全歪——彩虹/车道线都不正常的根因)
    frame_delay = DT_MDL   # 补偿帧采集到当前的时间差(平均50ms)
    action_delay = DT_MDL / 2  # 模型输出与下一帧之间的中点
    lat_action_t = _get_lat_delay(sm) + frame_delay + action_delay
    long_action_t = long_delay + frame_delay + action_delay

    inputs: dict[str, np.ndarray] = {
      'desire': vec_desire,
      'traffic_convention': traffic_convention,
      'action_t': np.array([lat_action_t, long_action_t], dtype=np.float32),
    }

    mt1 = time.perf_counter()
    model_output = model.run(buf_main, buf_extra, model_transform_main, model_transform_extra, inputs, prepare_only)
    mt2 = time.perf_counter()
    model_execution_time = mt2 - mt1

    if model_output is not None:
      modelv2_send = messaging.new_message('modelV2')
      drivingdata_send = messaging.new_message('drivingModelData')
      posenet_send = messaging.new_message('cameraOdometry')
      fill_model_msg(drivingdata_send, modelv2_send, model_output, v_ego, steer_delay,
                     publish_state, meta_main.frame_id, meta_extra.frame_id, frame_id,
                     frame_drop_ratio, meta_main.timestamp_eof, model_execution_time, live_calib_seen)

      desire_state = modelv2_send.modelV2.meta.desireState
      l_lane_change_prob = desire_state[log.Desire.laneChangeLeft]
      r_lane_change_prob = desire_state[log.Desire.laneChangeRight]
      lane_change_prob = l_lane_change_prob + r_lane_change_prob
      DH.update(sm['carState'], sm['carControl'].latActive, lane_change_prob)
      modelv2_send.modelV2.meta.laneChangeState = DH.lane_change_state
      modelv2_send.modelV2.meta.laneChangeDirection = DH.lane_change_direction
      drivingdata_send.drivingModelData.meta.laneChangeState = DH.lane_change_state
      drivingdata_send.drivingModelData.meta.laneChangeDirection = DH.lane_change_direction

      fill_pose_msg(posenet_send, model_output, meta_main.frame_id, vipc_dropped_frames, meta_main.timestamp_eof, live_calib_seen)
      pm.send('modelV2', modelv2_send)
      pm.send('drivingModelData', drivingdata_send)
      pm.send('cameraOdometry', posenet_send)
    last_vipc_frame_id = meta_main.frame_id


if __name__ == "__main__":
  try:
    main(demo=os.getenv('DEMO', '0') == '1')
  except KeyboardInterrupt:
    cloudlog.warning(f"child process {PROCESS_NAME} received SIGINT")
    raise SystemExit(0)
