#!/usr/bin/env python3
import os
os.environ['LRU'] = '0'  # Disable LRU GPU alloc cache early (tinygrad import reads it)
import ctypes
try:
  from openpilot.system.hardware import TICI
except ImportError:
  # master-c3 系硬件抽象无 TICI 常量: 非 TICI 平台 → 走 CUDA 路径
  TICI = False
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.device import Device
if TICI:
  from openpilot.selfdrive.modeld.runners.tinygrad_helpers import qcom_tensor_from_opencl_address
  os.environ['QCOM'] = '1'
else:
  os.environ['CUDA'] = '1'
import time
import pickle
import numpy as np
try:
  import openpilot.cereal.messaging as messaging
  from openpilot.cereal import log
  from opendbc.car.structs import car
except ImportError:
  import cereal.messaging as messaging
  from cereal import car, log
from pathlib import Path
from setproctitle import setproctitle
try:
  from openpilot.cereal.messaging import PubMaster, SubMaster
except ImportError:
  from cereal.messaging import PubMaster, SubMaster
from msgq.visionipc import VisionIpcClient, VisionBuf
try:
  from msgq.visionipc import VisionStreamType
  _VST_ROAD = VisionStreamType.VISION_STREAM_ROAD
  _VST_DRIVER = VisionStreamType.VISION_STREAM_DRIVER
  _VST_WIDE_ROAD = VisionStreamType.VISION_STREAM_WIDE_ROAD
except ImportError:
  # 无类型 msgq (新代, master-c3 系): 数值与 sp 枚举一致, kit 内部自洽
  _VST_ROAD, _VST_DRIVER, _VST_WIDE_ROAD = 0, 1, 2
try:
  from openpilot.cereal import log as _cereal_log
except ImportError:
  from cereal import log as _cereal_log
# cereal schema 自适应 (master-c3: roadCameraState→narrowRoadCameraState, 无 liveCalibration)
_EV_FIELDS = set(_cereal_log.Event.schema.fields.keys())
_MSG_ROAD = 'roadCameraState' if 'roadCameraState' in _EV_FIELDS else 'narrowRoadCameraState'
_MSG_LIVECAL = 'liveCalibration' if 'liveCalibration' in _EV_FIELDS else None
from opendbc.car.car_helpers import get_demo_car_params
from openpilot.common.swaglog import cloudlog
from openpilot.common.params import Params
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import config_realtime_process
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.system import sentry
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper
from openpilot.selfdrive.modeld.parse_model_outputs import Parser
from openpilot.selfdrive.modeld.fill_model_msg import fill_model_msg, fill_pose_msg, PublishState
from openpilot.selfdrive.modeld.constants import ModelConstants, Plan
from openpilot.selfdrive.modeld.models.commonmodel_pyx import DrivingModelFrame, CLContext

# TRT 加载兜底闭环参数(可用环境变量覆盖)
TRT_LOAD_ATTEMPTS = int(os.getenv("TRT_LOAD_ATTEMPTS", "3"))
TRT_LOAD_RETRY_INTERVAL = float(os.getenv("TRT_LOAD_RETRY_INTERVAL", "2.0"))

PROCESS_NAME = "selfdrive.modeld.modeld"
SEND_RAW_PRED = os.getenv('SEND_RAW_PRED')

# 兼容 master-c3 系: 原生 modeld.py 导出此常量, controlsd.py import 它
LAT_SMOOTH_SECONDS = 0.0

DEFAULT_MODEL_DIR = Path(__file__).parent / 'models'


def get_model_name():
  return Params().get("Model", encoding='utf-8') or "FiletOFish"


def get_model_dir():
  return DEFAULT_MODEL_DIR / get_model_name()


class FrameMeta:
  frame_id: int = 0
  timestamp_sof: int = 0
  timestamp_eof: int = 0

  def __init__(self, vipc=None):
    if vipc is not None:
      self.frame_id, self.timestamp_sof, self.timestamp_eof = vipc.frame_id, vipc.timestamp_sof, vipc.timestamp_eof

class ModelState:
  frames: dict[str, DrivingModelFrame]
  inputs: dict[str, np.ndarray]
  output: np.ndarray
  prev_desire: np.ndarray  # for tracking the rising edge of the pulse

  def __init__(self, context: CLContext):
    model_dir = get_model_dir()

    vision_metadata_path = model_dir / 'driving_vision_metadata.pkl'
    policy_metadata_path = model_dir / 'driving_policy_metadata.pkl'
    vision_pkl_path = model_dir / 'driving_vision_tinygrad.pkl'
    policy_pkl_path = model_dir / 'driving_policy_tinygrad.pkl'

    with open(vision_metadata_path, 'rb') as f:
      vision_metadata = pickle.load(f)
      self.vision_input_shapes = vision_metadata['input_shapes']
      self.vision_output_slices = vision_metadata['output_slices']
      vision_output_size = vision_metadata['output_shapes']['outputs'][1]
    self.vision_input_names = list(self.vision_input_shapes.keys())

    with open(policy_metadata_path, 'rb') as f:
      policy_metadata = pickle.load(f)
      self.policy_input_shapes = policy_metadata['input_shapes']
      self.policy_output_slices = policy_metadata['output_slices']
      policy_output_size = policy_metadata['output_shapes']['outputs'][1]

    # Derive all dimensions dynamically from metadata shapes
    # (different models use different key names, so search by pattern)
    self.temporal_skip = ModelConstants.TEMPORAL_SKIP
    _temporal_key = next((k for k in ['desire', 'desire_pulse', 'features_buffer'] if k in self.policy_input_shapes), None)
    _temporal_shape = self.policy_input_shapes[_temporal_key]
    self.input_history_buffer_len = _temporal_shape[1]
    _desire_key = next((k for k in ['desire', 'desire_pulse'] if k in self.policy_input_shapes), None)
    self.desire_len = self.policy_input_shapes[_desire_key][2] if _desire_key is not None else ModelConstants.DESIRE_LEN
    self.feature_len = self.policy_input_shapes['features_buffer'][2]
    self.full_history_buffer_len = self.input_history_buffer_len * self.temporal_skip

    # Derive model dimensions from vision input shape: (1, channels, h, w)
    # Each sub-plane in transform output is (w, h) → model input channel is (h, w)
    # MODEL_W = w * 2, MODEL_H = h * 2
    _model_input_shape = self.vision_input_shapes[self.vision_input_names[0]]
    _model_h, _model_w = _model_input_shape[2], _model_input_shape[3]
    self._model_w = _model_w * 2
    self._model_h = _model_h * 2

    self.frames = {
      'input_imgs': DrivingModelFrame(context, self.temporal_skip),
      'big_input_imgs': DrivingModelFrame(context, self.temporal_skip)
    }
    self.prev_desire = np.zeros(self.desire_len, dtype=np.float32)

    self.full_features_buffer = np.zeros((1, self.full_history_buffer_len, self.feature_len), dtype=np.float32)
    self.full_desire = np.zeros((1, self.full_history_buffer_len, self.desire_len), dtype=np.float32)
    self.full_prev_desired_curv = None
    self.temporal_idxs = slice(-1 - (self.temporal_skip * (self.input_history_buffer_len - 1)), None, self.temporal_skip)

    # Create policy inputs dynamically from model metadata
    self.numpy_inputs = {name: np.zeros(shape, dtype=np.float32) for name, shape in self.policy_input_shapes.items()}

    # Initialize prev_desired_curv buffer using actual model shape
    prev_curv_shape = self.numpy_inputs.get('prev_desired_curv', np.zeros((1,), dtype=np.float32)).shape
    self.full_prev_desired_curv = np.zeros((1, self.full_history_buffer_len, prev_curv_shape[-1]), dtype=np.float32)

    # img buffers are managed in openCL transform code
    self.vision_inputs: dict[str, Tensor] = {}
    self.vision_input_ptrs: dict[str, int] = {}  # CUDA device ptrs for TRT zero-copy
    self.vision_output = np.zeros(vision_output_size, dtype=np.float32)
    self.policy_inputs = {k: Tensor(v, device='NPY').realize() for k,v in self.numpy_inputs.items()}
    self.policy_output = np.zeros(policy_output_size, dtype=np.float32)
    self.parser = Parser()

    # Initialize CUDA transform pipeline (bypass OpenCL on non-TICI)
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
          for key in self.frames:
            state = ctypes.create_string_buffer(128)
            self._cu_transform.cuda_transform_init(
              ctypes.byref(state), self._model_w, self._model_h, self.temporal_skip
            )
            self._cuda_states[key] = state
          self._cuda_transform = True
          cloudlog.warning("CUDA transform initialized (OpenCL bypassed)")
          # 零拷贝: VisionIPC buffer 是 cudaHostRegister 映射的, 用设备指针喂 transform 省掉 H2D
          self._vipc_devptr = {}
          self._cudart_zc = None
          try:
            self._cudart_zc = ctypes.CDLL('libcudart.so')
            self._cudart_zc.cudaHostGetDevicePointer.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint]
            self._cudart_zc.cudaHostGetDevicePointer.restype = ctypes.c_int
          except Exception as e:
            cloudlog.warning(f"zerocopy device ptr unavailable: {e}")
      except Exception as e:
        cloudlog.warning(f"CUDA transform not available: {e}")
        self._cuda_transform = False

    # ── Force tinygrad CUDA device init BEFORE TRT ──
    # TRT will piggyback on tinygrad's context so they share the same CUDA context.
    # Without this, TRT creates its own context and tinygrad creates a different one,
    # causing 'invalid resource handle' on TRT inference.
    _ = Device['CUDA']

    # Try TensorRT engine first (much faster, ~2ms vs ~50ms)
    # 兜底闭环: plan 存在但加载失败 → 按 TRT_LOAD_ATTEMPTS 重试(间隔 TRT_LOAD_RETRY_INTERVAL) →
    # 仍失败才降级 tinygrad, 并记录失败原因供 cloudlog 汇总排查(不再静默只打一行 warning)。
    trt_engine_path = model_dir / 'driving_vision_fp16.plan'
    self.use_trt = trt_engine_path.exists()
    self.trt_vision_fail_reason = ""
    if self.use_trt:
      for attempt in range(1, TRT_LOAD_ATTEMPTS + 1):
        try:
          from openpilot.selfdrive.modeld.runners.tensorrt_runner import TensorRTModel
          self.vision_run = TensorRTModel(str(trt_engine_path))
          cloudlog.warning("Using TensorRT for vision model (FP16)")
          print("[INFO] Using TensorRT for vision model")
          break
        except Exception as e:
          self.trt_vision_fail_reason = f"{e}"
          if attempt < TRT_LOAD_ATTEMPTS:
            cloudlog.warning(f"TRT vision load failed ({e}), retry {attempt}/{TRT_LOAD_ATTEMPTS} in {TRT_LOAD_RETRY_INTERVAL}s")
            print(f"[WARN] TRT vision load failed: {e}, retry {attempt}/{TRT_LOAD_ATTEMPTS}")
            time.sleep(TRT_LOAD_RETRY_INTERVAL)
          else:
            cloudlog.warning(f"TRT vision load failed after {TRT_LOAD_ATTEMPTS} attempts ({e}), falling back to tinygrad")
            print(f"[WARN] TRT vision load failed after {TRT_LOAD_ATTEMPTS} attempts: {e}, falling back to tinygrad")
            self.use_trt = False
    if not self.use_trt:
      with open(vision_pkl_path, "rb") as f:
        self.vision_run = pickle.load(f)

    # Try TensorRT for policy model too (same pattern as vision)
    policy_trt_path = model_dir / 'driving_policy_fp16.plan'
    self.use_trt_policy = policy_trt_path.exists()
    self.trt_policy_fail_reason = ""
    if self.use_trt_policy:
      for attempt in range(1, TRT_LOAD_ATTEMPTS + 1):
        try:
          from openpilot.selfdrive.modeld.runners.tensorrt_runner import TensorRTModel
          self.policy_run = TensorRTModel(str(policy_trt_path))
          cloudlog.warning("Using TensorRT for policy model (FP16)")
          print("[INFO] Using TensorRT for policy model")
          break
        except Exception as e:
          self.trt_policy_fail_reason = f"{e}"
          if attempt < TRT_LOAD_ATTEMPTS:
            cloudlog.warning(f"Policy TRT load failed ({e}), retry {attempt}/{TRT_LOAD_ATTEMPTS} in {TRT_LOAD_RETRY_INTERVAL}s")
            print(f"[WARN] Policy TRT load failed: {e}, retry {attempt}/{TRT_LOAD_ATTEMPTS}")
            time.sleep(TRT_LOAD_RETRY_INTERVAL)
          else:
            cloudlog.warning(f"Policy TRT load failed after {TRT_LOAD_ATTEMPTS} attempts ({e}), falling back to tinygrad")
            print(f"[WARN] Policy TRT load failed after {TRT_LOAD_ATTEMPTS} attempts: {e}, falling back to tinygrad")
            self.use_trt_policy = False
    if not self.use_trt_policy:
      with open(policy_pkl_path, "rb") as f:
        self.policy_run = pickle.load(f)

  def slice_outputs(self, model_outputs: np.ndarray, output_slices: dict[str, slice]) -> dict[str, np.ndarray]:
    parsed_model_outputs = {k: model_outputs[np.newaxis, v] for k,v in output_slices.items()}
    return parsed_model_outputs

  def run(self, buf: VisionBuf, wbuf: VisionBuf, transform: np.ndarray, transform_wide: np.ndarray,
                inputs: dict[str, np.ndarray], prepare_only: bool) -> dict[str, np.ndarray] | None:
    # Model decides when action is completed, so desire input is just a pulse triggered on rising edge
    inputs['desire'][0] = 0
    new_desire = np.where(inputs['desire'] - self.prev_desire > .99, inputs['desire'], 0)
    self.prev_desire[:] = inputs['desire']

    self.full_desire[0,:-1] = self.full_desire[0,1:]
    self.full_desire[0,-1] = new_desire

    # Populate policy inputs dynamically by name
    desire_hist = self.full_desire.reshape((1, self.input_history_buffer_len, self.temporal_skip, -1)).max(axis=2)
    if 'desire' in self.numpy_inputs:
      self.numpy_inputs['desire'][:] = desire_hist
    if 'desire_pulse' in self.numpy_inputs:
      self.numpy_inputs['desire_pulse'][:] = desire_hist
    if 'traffic_convention' in self.numpy_inputs:
      self.numpy_inputs['traffic_convention'][:] = inputs['traffic_convention']
    if 'lateral_control_params' in self.numpy_inputs:
      self.numpy_inputs['lateral_control_params'][:] = inputs['lateral_control_params']

    if TICI:
      imgs_cl = {'input_imgs': self.frames['input_imgs'].prepare(buf, transform.flatten()),
                 'big_input_imgs': self.frames['big_input_imgs'].prepare(wbuf, transform_wide.flatten())}
      for i, cl_key in enumerate(['input_imgs', 'big_input_imgs']):
        model_key = self.vision_input_names[i]
        if model_key not in self.vision_inputs:
          self.vision_inputs[model_key] = qcom_tensor_from_opencl_address(imgs_cl[cl_key].mem_address, self.vision_input_shapes[model_key], dtype=dtypes.uint8)
    elif self._cuda_transform:
      for i, (key, fbuf) in enumerate([('input_imgs', buf), ('big_input_imgs', wbuf)]):
        model_key = self.vision_input_names[i]
        state = self._cuda_states[key]
        proj = transform if key == 'input_imgs' else transform_wide
        # 零拷贝: buffer 若是映射内存, 用设备指针, transform 内部跳过 H2D
        c_ptr = fbuf.data.ctypes.data
        input_is_device = 0
        if self._cudart_zc is not None:
          dptr = self._vipc_devptr.get(c_ptr)
          if dptr is None:
            d = ctypes.c_void_p()
            if self._cudart_zc.cudaHostGetDevicePointer(ctypes.byref(d), ctypes.c_void_p(c_ptr), 0) == 0 and d.value:
              dptr = d.value
              self._vipc_devptr[c_ptr] = dptr
          if dptr is not None:
            c_ptr = dptr
            input_is_device = 1
        buf_len = fbuf.data.nbytes
        output_ptr = self._cu_transform.cuda_transform_execute(
          ctypes.byref(state), ctypes.c_void_p(c_ptr),
          fbuf.width, fbuf.height, fbuf.stride, fbuf.uv_offset,
          buf_len,
          proj.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
          input_is_device
        )
        self.vision_inputs[model_key] = Tensor.from_blob(output_ptr, self.vision_input_shapes[model_key], dtype=dtypes.uint8, device='CUDA')
        self.vision_input_ptrs[model_key] = output_ptr  # save for TRT zero-copy
    else:
      imgs_cl = {'input_imgs': self.frames['input_imgs'].prepare(buf, transform.flatten()),
                 'big_input_imgs': self.frames['big_input_imgs'].prepare(wbuf, transform_wide.flatten())}
      for i, cl_key in enumerate(['input_imgs', 'big_input_imgs']):
        model_key = self.vision_input_names[i]
        frame_input = self.frames[cl_key].buffer_from_cl(imgs_cl[cl_key]).reshape(self.vision_input_shapes[model_key])
        self.vision_inputs[model_key] = Tensor(frame_input, dtype=dtypes.uint8).realize()

    if prepare_only:
      return None

    if self.use_trt:
      # ── Zero-copy: pass saved CUDA device pointers directly ──
      # These are saved at transform time: no tinygrad internal API needed.
      if self.vision_input_ptrs:
        # Have CUDA pointers saved from CUDA transform path
        input_ptrs = {k: self.vision_input_ptrs[k] for k in self.vision_input_names}
        self.vision_output = self.vision_run(**input_ptrs)
      else:
        # Fallback: pass numpy arrays (TRT runner does H2D copy internally)
        np_inputs = {k: self.vision_inputs[k].numpy() for k in self.vision_input_names}
        self.vision_output = self.vision_run(**np_inputs)
    else:
      self.vision_output = self.vision_run(**self.vision_inputs).numpy().flatten()
    vision_outputs_dict = self.parser.parse_vision_outputs(self.slice_outputs(self.vision_output, self.vision_output_slices))

    self.full_features_buffer[0,:-1] = self.full_features_buffer[0,1:]
    self.full_features_buffer[0,-1] = vision_outputs_dict['hidden_state'][0, :]
    if 'features_buffer' in self.numpy_inputs:
      self.numpy_inputs['features_buffer'][:] = self.full_features_buffer[0, self.temporal_idxs]

    if self.use_trt_policy:
      self.policy_output = self.policy_run(**self.numpy_inputs)
      # 模型健康报警: 平时静默, 仅在输出发散/NaN/顶 fp16 上限时打印 (velocity 乱跳的预兆)
      fb = self.numpy_inputs.get('features_buffer')
      po = np.asarray(self.policy_output).ravel().astype(np.float32)
      po_finite = np.isfinite(po)
      po_finite_pct = po_finite.mean() * 100.0
      po_absmax = np.abs(po[po_finite]).max() if po_finite.any() else -1.0
      if fb is not None and (po_finite_pct < 100.0 or po_absmax > 60000.0):
        print(f'[MODELD-ALERT] fb_std={fb.std():.5f} fb_absmax={np.abs(fb).max():.4f} '
              f'po_finite={po_finite_pct:.0f}% po_absmax={po_absmax:.1f}', flush=True)
    else:
      self.policy_output = self.policy_run(**self.policy_inputs).numpy().flatten()
    policy_outputs_dict = self.parser.parse_policy_outputs(self.slice_outputs(self.policy_output, self.policy_output_slices))

    if 'prev_desired_curv' in self.numpy_inputs:
      self.full_prev_desired_curv[0,:-1] = self.full_prev_desired_curv[0,1:]
      if 'desired_curvature' in policy_outputs_dict:
        self.full_prev_desired_curv[0,-1,0] = policy_outputs_dict['desired_curvature'][0, 0]
      else:
        plan_last = policy_outputs_dict['plan'][0, -1] if policy_outputs_dict['plan'].ndim == 3 else policy_outputs_dict['plan'][0, 0, -1]
        vx = float(plan_last[Plan.VELOCITY][0])
        yaw_rate = float(plan_last[Plan.ORIENTATION_RATE][2])
        self.full_prev_desired_curv[0,-1,0] = yaw_rate / max(vx, 0.01)
      self.numpy_inputs['prev_desired_curv'][:] = self.full_prev_desired_curv[0, self.temporal_idxs]

    combined_outputs_dict = {**vision_outputs_dict, **policy_outputs_dict}
    if SEND_RAW_PRED:
      combined_outputs_dict['raw_pred'] = np.concatenate([self.vision_output.copy(), self.policy_output.copy()])

    return combined_outputs_dict


def main(demo=False):
  # BigCombo dispatch: merged single-engine model runs its own module (FiletOFish path untouched).
  # NOTE: manager 的 launcher 直接调用 mod.main()，不经过 __main__，所以分流必须放在 main() 里。
  if (Params().get("Model", encoding='utf-8') or "FiletOFish") == "BigCombo":
    import selfdrive.modeld.modeld_bigcombo as _bigcombo
    _bigcombo.main(demo=demo)
    raise SystemExit(0)

  cloudlog.warning("modeld init")

  sentry.set_tag("daemon", PROCESS_NAME)
  cloudlog.bind(daemon=PROCESS_NAME)
  setproctitle(PROCESS_NAME)
  config_realtime_process(7, 54)

  cloudlog.warning("setting up CL context")
  cl_context = CLContext()
  cloudlog.warning("CL context ready; loading model")
  model = ModelState(cl_context)
  cloudlog.warning("models loaded, modeld starting")
  cloudlog.warning(f"backend: vision_trt={model.use_trt} policy_trt={model.use_trt_policy} "
                   f"(vision_fail={model.trt_vision_fail_reason or 'none'} policy_fail={model.trt_policy_fail_reason or 'none'})")

  # visionipc clients
  while True:
    available_streams = VisionIpcClient.available_streams("camerad", block=False)
    if available_streams:
      use_extra_client = _VST_WIDE_ROAD in available_streams and _VST_ROAD in available_streams
      main_wide_camera = _VST_ROAD not in available_streams
      break
    time.sleep(.1)

  vipc_client_main_stream = _VST_WIDE_ROAD if main_wide_camera else _VST_ROAD
  vipc_client_main = VisionIpcClient("camerad", vipc_client_main_stream, True, cl_context)
  vipc_client_extra = VisionIpcClient("camerad", _VST_WIDE_ROAD, True, cl_context)
  cloudlog.warning(f"vision stream set up, main_wide_camera: {main_wide_camera}, use_extra_client: {use_extra_client}")

  while not vipc_client_main.connect(False):
    time.sleep(0.1)
  while use_extra_client and not vipc_client_extra.connect(False):
    time.sleep(0.1)

  cloudlog.warning(f"connected main cam with buffer size: {vipc_client_main.buffer_len} ({vipc_client_main.width} x {vipc_client_main.height})")
  if use_extra_client:
    cloudlog.warning(f"connected extra cam with buffer size: {vipc_client_extra.buffer_len} ({vipc_client_extra.width} x {vipc_client_extra.height})")

  # messaging
  pm = PubMaster(["modelV2", "drivingModelData", "cameraOdometry"])
  sm = SubMaster(["deviceState", "carState", _MSG_ROAD] + ([_MSG_LIVECAL] if _MSG_LIVECAL else []) + ["driverMonitoringState", "carControl"])

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
  cloudlog.info("modeld got CarParams: %s", CP.brand)

  # TODO this needs more thought, use .2s extra for now to estimate other delays
  steer_delay = CP.steerActuatorDelay + .2

  DH = DesireHelper()

  while True:
    # Check if model was changed via params
    model_check_counter += 1
    if model_check_counter % 100 == 0:
      current_model_name = params.get("Model", encoding='utf-8') or "FiletOFish"
      if current_model_name != loaded_model_name:
        cloudlog.warning(f"Model changed from {loaded_model_name} to {current_model_name}, restarting")
        raise SystemExit(0)

    # Receive next main frame (FrameSync ensures matching frame_ids across cameras)
    buf_main = vipc_client_main.recv()
    meta_main = FrameMeta(vipc_client_main)
    if buf_main is None:
      cloudlog.debug("vipc_client_main no frame")
      continue

    if use_extra_client:
      # Receive extra frame — FrameSync guarantees same global_frame_id
      buf_extra = vipc_client_extra.recv()
      meta_extra = FrameMeta(vipc_client_extra)
      if buf_extra is None:
        cloudlog.debug("vipc_client_extra no frame")
        if buf_main is not None:
          buf_main.release()  # 归还已 acquire 的 main buffer, 避免 ref_count 泄漏
        continue

      # Decoupled FrameSync: wide 稳定落后 main 1 帧是设计稳态 (claim/follow 交错, camerad.py),
      # 差 1 只走 debug 进日志文件; 只有真实异常 (|差|>1, 含 wide 超前) 才 warning 刷屏。
      if meta_main.frame_id != meta_extra.frame_id:
        if abs(meta_main.frame_id - meta_extra.frame_id) > 1:
          cloudlog.warning(f"frame_id mismatch (anomaly): main={meta_main.frame_id} extra={meta_extra.frame_id}")
        else:
          cloudlog.debug(f"frame_id skew 1 (steady state): main={meta_main.frame_id} extra={meta_extra.frame_id}")

    else:
      # Use single camera
      buf_extra = buf_main
      meta_extra = meta_main

    sm.update(0)
    desire = DH.desire
    is_rhd = sm["driverMonitoringState"].isRHD
    frame_id = sm[_MSG_ROAD].frameId
    v_ego = max(sm["carState"].vEgo, 0.)
    lateral_control_params = np.array([v_ego, steer_delay], dtype=np.float32)
    if _MSG_LIVECAL and sm.updated[_MSG_LIVECAL] and sm.seen[_MSG_ROAD] and sm.seen['deviceState']:
      device_from_calib_euler = np.array(sm[_MSG_LIVECAL].rpyCalib, dtype=np.float32)
      dc = DEVICE_CAMERAS[(str(sm['deviceState'].deviceType), str(sm[_MSG_ROAD].sensor))]
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
    if run_count < 10: # let frame drops warm up
      frame_dropped_filter.x = 0.
      frames_dropped = 0.
    run_count = run_count + 1

    frame_drop_ratio = frames_dropped / (1 + frames_dropped)
    prepare_only = vipc_dropped_frames > 0
    if prepare_only:
      cloudlog.error(f"skipping model eval. Dropped {vipc_dropped_frames} frames")

    inputs:dict[str, np.ndarray] = {
      'desire': vec_desire,
      'traffic_convention': traffic_convention,
      'lateral_control_params': lateral_control_params,
      }

    mt1 = time.perf_counter()
    model_output = model.run(buf_main, buf_extra, model_transform_main, model_transform_extra, inputs, prepare_only)
    mt2 = time.perf_counter()
    model_execution_time = mt2 - mt1

    # 推理/读取完成, 释放 vipc buffer 供 camerad 复用 (与 client recv 的 acquire 配对)
    if buf_extra is not None and buf_extra is not buf_main:
      buf_extra.release()
    if buf_main is not None:
      buf_main.release()

    # Log GPU memory periodically (every 100 frames) on CUDA/Orin
    if not TICI and frame_id % 100 == 0:
      try:
        import ctypes
        cuda = ctypes.CDLL('libcuda.so.1')
        free_mem = ctypes.c_size_t()
        total_mem = ctypes.c_size_t()
        cuda.cuMemGetInfo_v2(ctypes.byref(free_mem), ctypes.byref(total_mem))
        cloudlog.warning(f"GPU mem: free={free_mem.value>>20}MB total={total_mem.value>>20}MB used={(total_mem.value - free_mem.value)>>20}MB")
      except Exception:
        pass

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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='A boolean for demo mode.')
    args = parser.parse_args()
    main(demo=args.demo)  # BigCombo dispatch is inside main() (manager calls main() directly)
  except KeyboardInterrupt:
    cloudlog.warning(f"child {PROCESS_NAME} got SIGINT")
  except Exception:
    sentry.capture_exception()
    raise
