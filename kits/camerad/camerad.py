#!/usr/bin/env python3
import threading
import os
import platform
import time
import ctypes
import numpy as np
from pathlib import Path
from collections import namedtuple

from msgq.visionipc import VisionIpcServer, VisionStreamType
try:
  from openpilot.cereal import messaging
except ImportError:
  from cereal import messaging


# ---------------------------------------------------------------------------
# FrameSync: 多相机帧同步 – 快帧等慢帧，统一分配 global frame_id
# 每个相机线程拿到帧后 call wait()，直到所有相机都到齐才一同放行，
# 此时分配同一个 global_frame_id，确保 modeld 收到的 road/wide 帧 id 匹配。
# ---------------------------------------------------------------------------
class FrameSync:
  def __init__(self):
    self._lock = threading.Lock()
    self._cv = threading.Condition(self._lock)
    self.n_cameras = 0
    self.arrived = 0
    self.round = 0
    self.global_frame_id = 0
    self.latest = 0

  def init(self, n):
    self.n_cameras = n
    if n > 1:
      print(f"[FRAME_SYNC] {n} cameras: decoupled mode (road dense id, wide follows road latest)", flush=True)
    else:
      print(f"[FRAME_SYNC] Python FrameSync initialized with {n} cameras, barrier mode", flush=True)

  def claim(self):
    """主相机(road)发号: 独立致密递增。modeld 的丢帧统计只看主镜头 id,
    id 必须每交付一帧 +1, 否则会被误判为丢帧 (49.9% frames dropped 根因)。"""
    with self._lock:
      gid = self.global_frame_id
      self.global_frame_id += 1
      self.latest = gid
      return gid

  def follow(self):
    """副相机(wide)取号: 非阻塞跟随 road 最新号, 仅作帧配对参考。
    两路同在 VI 33.3ms 网格的 66.7ms 双步窗口交付, 相位差 <=33ms,
    偶发差 1 号只触发 modeld 的 mismatch 告警, 不影响任何统计。"""
    with self._lock:
      return self.latest

  def wait(self):
    """兼容单相机: 直接发号, 无等待。"""
    return self.claim()


g_frame_sync = FrameSync()

GMSL_WEBCAM = os.getenv("GMSL_WEBCAM", "0") == "1"
ROAD_CAM = os.getenv("ROAD_CAM", "1" if GMSL_WEBCAM else "0")
WIDE_CAM = os.getenv("WIDE_CAM", "2" if GMSL_WEBCAM else "")
DRIVER_CAM = os.getenv("DRIVER_CAM")
USE_NVMM = os.getenv("SP_VISIONIPC_NVMM", "0") == "1"
USE_V4L2 = os.getenv("USE_V4L2_CAMERA", "1" if platform.system() != "Darwin" else "0") == "1"
REPO_ROOT = Path(__file__).resolve().parents[2]

def _find_repo_root():
  """自适应 repo 根: 兼容老布局(<root>/tools/webcam)与新布局(<root>/openpilot/system/camerad/webcam)。"""
  p = Path(__file__).resolve()
  for parent in p.parents:
    if (parent / 'selfdrive').is_dir():
      return parent
  return p.parents[2]

REPO_ROOT = _find_repo_root()

CameraType = namedtuple("CameraType", ["msg_name", "stream_type", "cam_id"])


def _normalize_cam_id(cam_id):
  if cam_id is None:
    return None
  cam_id = str(cam_id).strip()
  if cam_id == "":
    return ""
  if cam_id.startswith("/dev/video"):
    cam_id = cam_id[len("/dev/video"):]
  elif cam_id.startswith("video"):
    cam_id = cam_id[len("video"):]
  return cam_id


def _build_cameras():
  cameras = [
    CameraType("roadCameraState", VisionStreamType.VISION_STREAM_ROAD, _normalize_cam_id(ROAD_CAM)),
  ]
  if WIDE_CAM:
    cameras.append(CameraType("wideRoadCameraState", VisionStreamType.VISION_STREAM_WIDE_ROAD, _normalize_cam_id(WIDE_CAM)))
  if DRIVER_CAM:
    cameras.append(CameraType("driverCameraState", VisionStreamType.VISION_STREAM_DRIVER, _normalize_cam_id(DRIVER_CAM)))

  non_empty_ids = [c.cam_id for c in cameras if c.cam_id not in (None, "")]
  if len(non_empty_ids) > 1 and len(set(non_empty_ids)) == 1 and not GMSL_WEBCAM:
    remapped = []
    next_id = int(non_empty_ids[0])
    for idx, c in enumerate(cameras):
      if c.cam_id in (None, ""):
        remapped.append(c)
      else:
        remapped.append(CameraType(c.msg_name, c.stream_type, str(next_id + idx)))
    cameras = remapped
    print(f"[camerad] duplicate camera ids detected, remapped to {[c.cam_id for c in cameras]}", flush=True)

  return cameras


CAMERAS = _build_cameras()


def _packed_yuv_to_nv12(yuv_data, width, height, pixel_format='UYVY'):
  """packed YUV -> NV12.

  twgmsl 的 UYVY 节点实际是特殊 packed 两字节布局：有效亮度在奇数字节，
  色度交错也来自奇数字节；按标准 UYVY 读取会得到 U/V≈0，画面纯绿。
  通过 GMSL_CHROMA_LAYOUT=twgmsl 选择与 CP 相同的通道归一化。
  """
  yuv = np.frombuffer(yuv_data, dtype=np.uint8).reshape(height, width * 2)
  nv12 = np.zeros(height * width * 3 // 2, dtype=np.uint8)
  y_plane = nv12[:height * width].reshape(height, width)
  uv_plane = nv12[height * width:].reshape(height // 2, width)

  if pixel_format == 'UYVY' and os.environ.get('GMSL_CHROMA_LAYOUT', 'twgmsl').lower() == 'twgmsl':
    # 与 CP openpilot/tools/webcam/camera.py 完全一致：
    # 实测 twgmsl packed 布局中，Y 在偶数字节；V/U 位于奇数字节的两条色度 lane。
    # CP 原实现：y=a[0::2], u=a[3::4], v=a[1::4]。
    for i in range(height):
      row = yuv[i]
      y_plane[i] = row[0::2]
      if i % 2 == 0:
        # 当前 Qt shader 按标准 NV12: U 在偶数位、V 在奇数位。
        # 实测原始 lane 与 CP 的历史命名相反；为恢复红/蓝正确方向，交换两条 lane。
        uv_plane[i // 2, 0::2] = row[1::4]  # U
        uv_plane[i // 2, 1::2] = row[3::4]  # V
    return nv12

  if pixel_format == 'UYVY':
    for i in range(height):
      row = yuv[i]
      y_plane[i] = row[1::2]
      if i % 2 == 0:
        uv_plane[i // 2, 0::2] = row[0::4]
        uv_plane[i // 2, 1::2] = row[2::4]
  else:  # YUYV
    for i in range(height):
      row = yuv[i]
      y_plane[i] = row[0::2]
      if i % 2 == 0:
        uv_plane[i // 2, 0::2] = row[1::4]
        uv_plane[i // 2, 1::2] = row[3::4]
  return nv12


class CudaUyvyConverter:
  def __init__(self, width, height, src_w=1920, src_h=1080):
    self.W = width
    self.H = height
    self.src_w = src_w      # 源 packed 采集尺寸 (NvBufSurface, 1920x1080)
    self.src_h = src_h
    # cp 适配开关 (env): SP_CAM_FLIP=1 180°翻转; SP_CHROMA_SWAP=1 色度 U/V 交换
    self.flip = os.environ.get('SP_CAM_FLIP', '0') == '1'
    self.chroma_swap = os.environ.get('SP_CHROMA_SWAP', '0') == '1'
    self.uyvy_size = width * height * 2
    self.nv12_size = width * height * 3 // 2
    self.stride = width * 2
    self._use_cuda = False
    self._packed = None

    # 优先: 已验证的 GMSL packed→NV12 CUDA kernel (与下方 CPU 路径字节规则一致)
    # 部署位候选: 与 camerad.py 同目录 → 老布局 tools/webcam/ → 新布局 openpilot/tools/webcam/
    _so_candidates = [
      Path(__file__).resolve().parent / 'libpacked_to_nv12.so',
      REPO_ROOT / 'tools' / 'webcam' / 'libpacked_to_nv12.so',
      REPO_ROOT / 'openpilot' / 'tools' / 'webcam' / 'libpacked_to_nv12.so',
    ]
    packed_so = next((c for c in _so_candidates if c.exists()), None)
    if packed_so is not None and packed_so.exists() and os.environ.get('GMSL_CHROMA_LAYOUT', 'twgmsl').lower() == 'twgmsl':
      try:
        self._packed = ctypes.CDLL(str(packed_so))
        self._packed.packed_to_nv12.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self._packed.packed_to_nv12.restype = ctypes.c_int
        # 零拷贝入口: 直接写到设备指针, 不 D2H
        self._packed.packed_to_nv12_device.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self._packed.packed_to_nv12_device.restype = ctypes.c_int
        self._packed.packed_to_nv12_device_to_device.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self._packed.packed_to_nv12_device_to_device.restype = ctypes.c_int
        # resize+flip 零拷贝 (cp 链路: 1920x1080 源 → 1344x760 输出, 双线性)
        self._packed.packed_to_nv12_resize_device_to_device.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self._packed.packed_to_nv12_resize_device_to_device.restype = ctypes.c_int
        self.nv12_cpu = np.zeros(self.nv12_size, dtype=np.uint8)
        self._use_cuda = True
        print(f"[CudaUyvyConverter] packed CUDA kernel OK: {width}x{height}", flush=True)
        return
      except Exception as e:
        print(f"[CudaUyvyConverter] packed CUDA init failed: {e}, fallback", flush=True)
        self._packed = None

    lib_path = REPO_ROOT / 'selfdrive' / 'modeld_v2' / 'libuyvy_convert.so'
    if not lib_path.exists():
      alt_path = REPO_ROOT / 'sunnypilot' / 'modeld_v2' / 'libuyvy_convert.so'
      if alt_path.exists():
        lib_path = alt_path
      else:
        # 新布局: fork 有 openpilot/ 子目录时 selfdrive 在其下
        alt2 = REPO_ROOT / 'openpilot' / 'selfdrive' / 'modeld_v2' / 'libuyvy_convert.so'
        alt3 = REPO_ROOT / 'openpilot' / 'sunnypilot' / 'modeld_v2' / 'libuyvy_convert.so'
        if alt2.exists():
          lib_path = alt2
        elif alt3.exists():
          lib_path = alt3
        else:
          print(f"[CudaUyvyConverter] libuyvy_convert.so not found at {lib_path}, falling back to CPU UYVY→NV12", flush=True)
          return

    try:
      self.cudart = ctypes.CDLL('libcudart.so.12')
      self.cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_ulonglong]
      self.cudart.cudaMalloc.restype = ctypes.c_int
      self.cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_int]
      self.cudart.cudaMemcpy.restype = ctypes.c_int

      self.lib = ctypes.CDLL(str(lib_path))
      self.lib.uyvy_to_nv12.argtypes = [ctypes.c_ulonglong, ctypes.c_ulonglong, ctypes.c_int, ctypes.c_int, ctypes.c_int]
      self.lib.uyvy_to_nv12.restype = ctypes.c_int

      self.uyvy_gpu = ctypes.c_ulonglong()
      self.nv12_gpu = ctypes.c_ulonglong()
      r1 = self.cudart.cudaMalloc(ctypes.byref(self.uyvy_gpu), self.uyvy_size)
      r2 = self.cudart.cudaMalloc(ctypes.byref(self.nv12_gpu), self.nv12_size)
      if r1 != 0 or r2 != 0:
        raise RuntimeError(f"cudaMalloc failed: r1={r1} r2={r2}")

      self.nv12_cpu = np.zeros(self.nv12_size, dtype=np.uint8)
      self._use_cuda = True
      print(f"[CudaUyvyConverter] init OK: W={width} H={height} uyvy_gpu={self.uyvy_gpu.value:#x} nv12_gpu={self.nv12_gpu.value:#x}", flush=True)
    except Exception as e:
      print(f"[CudaUyvyConverter] CUDA init failed: {e}, falling back to CPU", flush=True)
      self._use_cuda = False

  def convert(self, uyvy_data):
    if self._packed is not None:
      src = np.ascontiguousarray(uyvy_data, dtype=np.uint8)
      ret = self._packed.packed_to_nv12(ctypes.c_void_p(src.ctypes.data),
                                        ctypes.c_void_p(self.nv12_cpu.ctypes.data),
                                        self.W, self.H)
      if ret != 0:
        print(f"[CUDA] packed_to_nv12 failed ret={ret}", flush=True)
        return None
      return self.nv12_cpu
    return self._convert_fallback(uyvy_data)

  def _convert_fallback(self, uyvy_data):
    if not self._use_cuda:
      return _packed_yuv_to_nv12(uyvy_data.tobytes() if hasattr(uyvy_data, 'tobytes') else bytes(uyvy_data), self.W, self.H, 'UYVY')
    ret_copy = self.cudart.cudaMemcpy(ctypes.c_void_p(self.uyvy_gpu.value), ctypes.c_void_p(uyvy_data.ctypes.data), self.uyvy_size, 1)
    if ret_copy != 0:
      print(f"[CUDA] H2D failed ret={ret_copy}", flush=True)
      return None
    ret_kernel = self.lib.uyvy_to_nv12(self.uyvy_gpu.value, self.nv12_gpu.value, self.W, self.H, self.stride)
    if ret_kernel != 0:
      print(f"[CUDA] kernel failed ret={ret_kernel}", flush=True)
      return None
    ret_copy2 = self.cudart.cudaMemcpy(ctypes.c_void_p(self.nv12_cpu.ctypes.data), ctypes.c_void_p(self.nv12_gpu.value), self.nv12_size, 2)
    if ret_copy2 != 0:
      print(f"[CUDA] D2H failed ret={ret_copy2}", flush=True)
      return None
    return self.nv12_cpu

  def convert_to_device(self, uyvy_data, dst_device_ptr):
    """零拷贝: NV12 直接写到 dst_device_ptr (VisionIPC buffer 的设备指针), 不 D2H。"""
    if self._packed is None:
      return False
    src = np.ascontiguousarray(uyvy_data, dtype=np.uint8)
    ret = self._packed.packed_to_nv12_device(ctypes.c_void_p(src.ctypes.data),
                                             ctypes.c_void_p(dst_device_ptr),
                                             self.W, self.H)
    if ret != 0:
      print(f"[CUDA] packed_to_nv12_device failed ret={ret}", flush=True)
      return False
    return True

  def convert_device_to_device(self, src_device_ptr, dst_device_ptr):
    """完整零拷贝: packed 源和 NV12 目标都在 CUDA device pointer 上。"""
    if self._packed is None:
      return False
    ret = self._packed.packed_to_nv12_device_to_device(ctypes.c_void_p(src_device_ptr),
                                                       ctypes.c_void_p(dst_device_ptr),
                                                       self.W, self.H)
    if ret != 0:
      print(f"[CUDA] packed_to_nv12_device_to_device failed ret={ret}", flush=True)
      return False
    return True

  def convert_resize_device_to_device(self, src_device_ptr, dst_device_ptr,
                                      src_w=None, src_h=None, dst_w=None, dst_h=None,
                                      flip=None, chroma_swap=None):
    """resize+flip 零拷贝: 源采集尺寸 → 目标输出尺寸 (cp 链路: 1920x1080 → 1344x760)。
    参数缺省取 self 的构造值(env 可配)。flip=1 时 180°翻转, chroma_swap=1 时 U/V 交换。"""
    if self._packed is None:
      return False
    sw = src_w if src_w is not None else self.src_w
    sh = src_h if src_h is not None else self.src_h
    dw = dst_w if dst_w is not None else self.W
    dh = dst_h if dst_h is not None else self.H
    fl = self.flip if flip is None else flip
    cs = self.chroma_swap if chroma_swap is None else chroma_swap
    ret = self._packed.packed_to_nv12_resize_device_to_device(ctypes.c_void_p(src_device_ptr),
                                                              ctypes.c_void_p(dst_device_ptr),
                                                              sw, sh, dw, dh,
                                                              1 if fl else 0, 1 if cs else 0)
    if ret != 0:
      print(f"[CUDA] packed_to_nv12_resize_device_to_device failed ret={ret} {sw}x{sh}->{dw}x{dh}", flush=True)
      return False
    return True


class Camerad:
  def __init__(self):
    self.pm = messaging.PubMaster([c.msg_name for c in CAMERAS])
    self.vipc_server = VisionIpcServer("camerad")
    self.use_v4l2 = USE_V4L2
    self.use_nvmm = USE_NVMM
    self.cameras = []
    self.converters = {}
    # 零拷贝: stream_type -> (host_ptr -> device_ptr) 缓存
    self._devptr_cache = {}
    self._zerocopy = os.environ.get('SP_ZEROCOPY', '1') == '1'
    self._nvbuf_zerocopy = os.environ.get('SP_NVBUF_ZEROCOPY', '0') == '1'
    self._src_devptr = {}
    self._staging_devptr = {}   # stream_type -> CUDA 暂存 NV12 device ptr
    self._staging_size = {}     # stream_type -> NV12 staging 字节数
    self._cudart = None
    self._cuda = None
    if self._zerocopy or self._nvbuf_zerocopy:
      try:
        self._cudart = ctypes.CDLL('libcudart.so')
        self._cudart.cudaHostGetDevicePointer.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint]
        self._cudart.cudaHostGetDevicePointer.restype = ctypes.c_int
        self._cudart.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
        self._cudart.cudaHostRegister.restype = ctypes.c_int
        self._cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self._cudart.cudaMalloc.restype = ctypes.c_int
        self._cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self._cudart.cudaMemcpy.restype = ctypes.c_int
        if self._nvbuf_zerocopy:
          # libnvbuf_import.so 部署位候选(与 packed_so 同规则)
          _imp_candidates = [
            Path(__file__).resolve().parent / 'libnvbuf_import.so',
            REPO_ROOT / 'tools' / 'webcam' / 'libnvbuf_import.so',
            REPO_ROOT / 'openpilot' / 'tools' / 'webcam' / 'libnvbuf_import.so',
          ]
          _imp_so = next((c for c in _imp_candidates if c.exists()), None)
          if _imp_so is None:
            raise RuntimeError("libnvbuf_import.so not found in any deploy location")
          self._nvbuf_import = ctypes.CDLL(str(_imp_so))
          self._nvbuf_import.nvbuf_import_fd.argtypes = [ctypes.c_int, ctypes.c_ulonglong, ctypes.POINTER(ctypes.c_ulonglong)]
          self._nvbuf_import.nvbuf_import_fd.restype = ctypes.c_int
      except Exception as e:
        print(f"[camerad] zerocopy disabled, cudart load failed: {e}", flush=True)
        self._zerocopy = False
        self._nvbuf_zerocopy = False

    if (self._zerocopy or self._nvbuf_zerocopy) and not hasattr(self.vipc_server, 'write_and_send'):
      print("[camerad] msgq visionipc 无 write_and_send(目标 fork 未打零拷贝补丁), 回退主机路径", flush=True)
      self._zerocopy = False
      self._nvbuf_zerocopy = False

    if self.use_v4l2:
      try:
        from openpilot.system.camerad.webcam.v4l2_dmabuf_camera import Camera
      except ImportError:
        from openpilot.tools.webcam.v4l2_dmabuf_camera import Camera
      print("[camerad] Using V4L2 DMABUF camera (UYVY → CUDA NV12 → send)", flush=True)
    else:
      from openpilot.tools.webcam.camera import Camera
      print("[camerad] Using OpenCV camera (fallback mode)", flush=True)

    print(f"[camerad] config: repo_root={REPO_ROOT} gmsl={GMSL_WEBCAM} ROAD_CAM={ROAD_CAM} WIDE_CAM={WIDE_CAM} use_v4l2={self.use_v4l2} use_nvmm={self.use_nvmm}", flush=True)

    for c in CAMERAS:
      cam_device = f"/dev/video{c.cam_id}" if platform.system() != "Darwin" else c.cam_id
      cam = Camera(c.msg_name, c.stream_type, cam_device)
      self.cameras.append(cam)
      # 20 个共享 buffer: 配合 refcount 同步 (server 写前等 ref_count==0),
      # 轮转周期 1s @20fps, 消除读写竞争撕裂; modeld/UI 慢时 camerad 限流而非撕裂。
      self.vipc_server.create_buffers(c.stream_type, 20, cam.W, cam.H)
      vic_enabled = getattr(getattr(cam, 'cam', None), '_vic', False)
      use_dmabuf = getattr(getattr(cam, 'cam', None), '_use_dmabuf', False)
      print(f"[camerad] camera={c.msg_name} device={cam_device} size={cam.W}x{cam.H} vic={vic_enabled} dmabuf={use_dmabuf}", flush=True)
      if self.use_v4l2 and not vic_enabled:
        self.converters[c.stream_type] = CudaUyvyConverter(cam.W, cam.H)
      if self._nvbuf_zerocopy:
        stage = ctypes.c_void_p()
        nv12_size = cam.W * cam.H * 3 // 2
        r = self._cudart.cudaMalloc(ctypes.byref(stage), nv12_size)
        if r != 0 or not stage.value:
          raise RuntimeError(f"cudaMalloc staging failed r={r}")
        self._staging_devptr[c.stream_type] = stage.value
        self._staging_size[c.stream_type] = nv12_size
        print(f"[camerad] staging NV12 device ptr {c.stream_type}: {stage.value:#x} size={nv12_size}", flush=True)

    # Initialize frame sync barrier
    n_v4l2 = len([c for c in self.cameras if self.use_v4l2])
    g_frame_sync.init(n_v4l2 if n_v4l2 > 0 else len(self.cameras))

    self.vipc_server.start_listener()

  def _send_nv12(self, nv12_data, frame_id, pub_type, yuv_type, timestamp_sof=0, timestamp_eof=0):
    if timestamp_eof <= 0:
      timestamp_eof = timestamp_sof if timestamp_sof > 0 else int(time.monotonic_ns())
    self.vipc_server.send(yuv_type, nv12_data, frame_id, timestamp_sof, timestamp_eof)
    self._publish_camera_state(frame_id, pub_type, timestamp_sof, timestamp_eof)

  def _send_nv12_zerocopy(self, raw, converter, frame_id, pub_type, yuv_type, timestamp_sof=0, timestamp_eof=0):
    """kernel 直接把 NV12 写进 VisionIPC 共享内存的设备映射, 跳过 D2H 和 memcpy。
    write_and_send 保证写入和发送是同一个 buffer。"""
    if timestamp_eof <= 0:
      timestamp_eof = timestamp_sof if timestamp_sof > 0 else int(time.monotonic_ns())

    def fill(dst):
      host_ptr = dst.ctypes.data
      cache = self._devptr_cache.setdefault(yuv_type, {})
      dptr = cache.get(host_ptr)
      if dptr is None:
        d = ctypes.c_void_p()
        rc = self._cudart.cudaHostGetDevicePointer(ctypes.byref(d), ctypes.c_void_p(host_ptr), 0)
        if rc != 0 or not d.value:
          # 诊断+补救 (cp 实车踩坑): 可能 C++ 层 cudaHostRegister 未生效
          # (进程内早期 CUDA 状态干扰), 此处补注册后重试
          reg = self._cudart.cudaHostRegister(ctypes.c_void_p(host_ptr), dst.nbytes, 2)  # 2 = cudaHostRegisterMapped
          rc2 = self._cudart.cudaHostGetDevicePointer(ctypes.byref(d), ctypes.c_void_p(host_ptr), 0)
          print(f"[camerad] cudaHostGetDevicePointer rc={rc} reg={reg} retry_rc={rc2} dptr={d.value and hex(d.value)} host={host_ptr:#x} len={dst.nbytes}", flush=True)
          if rc2 != 0 or not d.value:
            raise RuntimeError(f"cudaHostGetDevicePointer rc={rc} reg={reg} retry={rc2}")
        dptr = d.value
        cache[host_ptr] = dptr
      if not converter.convert_to_device(raw, dptr):
        raise RuntimeError("convert_to_device failed")

    try:
      self.vipc_server.write_and_send(yuv_type, fill, frame_id, timestamp_sof, timestamp_eof)
    except Exception as e:
      print(f"[camerad] zerocopy fill failed: {e}", flush=True)
      return False
    self._publish_camera_state(frame_id, pub_type, timestamp_sof, timestamp_eof)
    return True

  def _import_nvbuf(self, dmabuf_fd, size):
    dptr = self._src_devptr.get(dmabuf_fd)
    if dptr is not None:
      return dptr
    fd2 = os.dup(dmabuf_fd)
    dev = ctypes.c_ulonglong()
    rc = self._nvbuf_import.nvbuf_import_fd(fd2, size, ctypes.byref(dev))
    if rc != 0 or not dev.value:
      raise RuntimeError(f"nvbuf_import_fd failed fd={dmabuf_fd} rc={rc}")
    self._src_devptr[dmabuf_fd] = dev.value
    return dev.value

  def _send_nv12_nvbuf(self, vision_buf, converter, frame_id, pub_type, yuv_type, timestamp_sof=0, timestamp_eof=0):
    if timestamp_eof <= 0:
      timestamp_eof = timestamp_sof if timestamp_sof > 0 else int(time.monotonic_ns())
    staging = self._staging_devptr.get(yuv_type)
    if staging is None:
      raise RuntimeError(f"no staging buffer for {yuv_type}")
    try:
      # 第一步：kernel 把相机 buffer 直接转换到独立暂存区（这是唯一读相机 buffer 的时刻）。
      src = self._import_nvbuf(vision_buf.fd, os.fstat(vision_buf.fd).st_size)
      if not converter.convert_device_to_device(src, staging):
        raise RuntimeError("convert_device_to_device failed")
    finally:
      # 相机 buffer 用完立刻归还，不等 VisionIPC。
      if vision_buf.v4l2_index is not None:
        for cam in self.cameras:
          if getattr(cam, "stream_type", None) == yuv_type and hasattr(getattr(cam, "cam", None), "_requeue_buffer"):
            cam.cam._requeue_buffer(vision_buf.v4l2_index)
            break

    # 第二步：把暂存区 NV12 D2D 拷进 VisionIPC 自己的 device ptr，然后发送。
    def fill(dst):
      host_ptr = dst.ctypes.data
      cache = self._devptr_cache.setdefault(yuv_type, {})
      dptr = cache.get(host_ptr)
      if dptr is None:
        d = ctypes.c_void_p()
        rc = self._cudart.cudaHostGetDevicePointer(ctypes.byref(d), ctypes.c_void_p(host_ptr), 0)
        if rc != 0 or not d.value:
          raise RuntimeError(f"cudaHostGetDevicePointer rc={rc}")
        dptr = d.value
        cache[host_ptr] = dptr
      rc = self._cudart.cudaMemcpy(dptr, staging, converter.nv12_size, 3)  # 3 = cudaMemcpyDeviceToDevice; 2 是 DeviceToHost 会写错地方
      if rc != 0:
        raise RuntimeError(f"staging D2D copy failed rc={rc}")

    self.vipc_server.write_and_send(yuv_type, fill, frame_id, timestamp_sof, timestamp_eof)
    self._publish_camera_state(frame_id, pub_type, timestamp_sof, timestamp_eof)
    return True

  def _publish_camera_state(self, frame_id, pub_type, timestamp_sof=0, timestamp_eof=0):
    dat = messaging.new_message(pub_type, valid=True)
    if timestamp_eof <= 0:
      timestamp_eof = timestamp_sof if timestamp_sof > 0 else int(time.monotonic_ns())

    msg = {
      "frameId": frame_id,
      "timestampSof": timestamp_sof,
      "timestampEof": timestamp_eof,
      # sensor 是 ImageSensor 枚举 (ar0231/ox03c10/os04c10), twgmsl 不在其中。
      # 勿写任意字符串 (会 capnp enum 报错崩 camerad)。IMX390 内参改为在
      # common/transformations/camera.py 的 DEVICE_CAMERAS 里按 ("pc","unknown") 修正。
      "transform": [1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0],
    }
    setattr(dat, pub_type, msg)
    self.pm.send(pub_type, dat)

  def _vision_buf_to_nv12(self, cam, vision_buf):
    if vision_buf is None or vision_buf.data is None:
      return None
    pixel_format = getattr(vision_buf, 'pixel_format', 'UYVY').upper()
    if pixel_format == 'NV12':
      return vision_buf.data

    converter = self.converters.get(cam.stream_type)
    if converter is None:
      print(f"[camerad] unsupported pixel format for {cam.cam_type_state}: {pixel_format}", flush=True)
      return None
    raw = np.frombuffer(vision_buf.data, dtype=np.uint8)
    if pixel_format == 'YUYV':
      # YUYV (Y0 U0 Y1 V0) → UYVY (U0 Y0 V0 Y1): swap each adjacent byte pair
      raw = raw.reshape(-1, 2)[:, ::-1].ravel()
    # UYVY → NV12 converter handles UYVY natively, no swap needed
    nv12 = converter.convert(np.ascontiguousarray(raw))
    if nv12 is None:
      print(f"[camerad] CUDA convert failed for {cam.cam_type_state}", flush=True)
    return nv12

  def camera_runner(self, cam, cam_idx):
    # VI 传感器实测恒定 30fps (2026-08-31 不限速实验: sof_dt=33.3ms 无空窗)。
    # sleep 节流与 33.3ms 帧网格不可通约, 锁相后产生 57.4ms 拍频 → 16Hz 根因。
    # 改为帧计数丢弃: 30fps 输入每 3 帧交付 2 帧 = 精确 20fps, 无 sleep。
    # 注意: 若更换传感器输出帧率(如 25fps), 此比例需同步修改。
    target_fps = 20.0
    frame_interval = 1.0 / target_fps
    if self.use_v4l2:
      in_count = 0
      # nvbuf 零拷贝模式下 read_frame 的 data=None 分支(770行)不归还相机 buffer,
      # 归还点只剩 _send_nv12_nvbuf 的 finally → 丢帧 continue / 异常降级空转 都会漏还,
      # 4 块 buffer 在 ~0.4s (12帧) 内被扣死 → DQBUF 永久 BlockingIOError。
      # 兜底: 每帧 try/finally, 除 _send_nv12_nvbuf 已归还外, 一律补还。
      cam_nvbuf = bool(getattr(getattr(cam, "cam", None), "_nvbuf_zerocopy", False))
      for vision_buf in cam.read_frames():
        in_count += 1
        requeued = False
        try:
          if in_count % 3 == 0:
            continue  # 丢弃第 3 帧: 不进 FrameSync, 不占 VisionIPC buffer(相机 buffer 由 finally 兜底归还)
          # Frame sync: road 致密发号(丢帧统计依赖), wide 跟随 road 最新号
          if cam.cam_type_state == "roadCameraState":
            sync_frame_id = g_frame_sync.claim()
          elif g_frame_sync.n_cameras > 1:
            sync_frame_id = g_frame_sync.follow()
          else:
            sync_frame_id = g_frame_sync.wait()
          converter = self.converters.get(cam.stream_type)
          sent = False
          if self._nvbuf_zerocopy and converter is not None and cam_nvbuf:
            try:
              sent = self._send_nv12_nvbuf(vision_buf, converter, sync_frame_id, cam.cam_type_state, cam.stream_type, vision_buf.timestamp_sof, vision_buf.timestamp_eof)
              self._nvbuf_fail = 0
            except Exception as e:
              # staging 方案下失败极罕见(import/convert/memcpy), 单帧跳过重试即可,
              # 不轻易整体降级(降级后 read_frame 仍走 data=None, 发送链全空, 需重启恢复)。
              self._nvbuf_fail = getattr(self, '_nvbuf_fail', 0) + 1
              if self._nvbuf_fail >= 10:
                print(f"[camerad] nvbuf zerocopy failed {self._nvbuf_fail}x consecutively, disabling: {e}", flush=True)
                self._nvbuf_zerocopy = False
              else:
                print(f"[camerad] nvbuf zerocopy frame skipped ({self._nvbuf_fail}x): {e}", flush=True)
            finally:
              requeued = True  # _send_nv12_nvbuf 的 finally 已归还(成功或异常都会执行)
          if (not sent) and self._zerocopy and converter is not None and converter._packed is not None and vision_buf is not None and vision_buf.data is not None:
            raw = np.frombuffer(vision_buf.data, dtype=np.uint8)
            sent = self._send_nv12_zerocopy(raw, converter, sync_frame_id, cam.cam_type_state, cam.stream_type, vision_buf.timestamp_sof, vision_buf.timestamp_eof)
            if not sent and not getattr(self, '_zc_warned', False):
              print("[camerad] zerocopy send failed, falling back to host path", flush=True)
              self._zc_warned = True
              self._zerocopy = False
          if not sent:
            nv12 = self._vision_buf_to_nv12(cam, vision_buf)
            if nv12 is not None:
              self._send_nv12(nv12, sync_frame_id, cam.cam_type_state, cam.stream_type, vision_buf.timestamp_sof, vision_buf.timestamp_eof)
        finally:
          # 相机 buffer 归还兜底: 丢帧/降级空转时 _send_nv12_nvbuf 未被调用, 靠这里补还;
          # 已走 _send_nv12_nvbuf 的帧不重复还(避免同一 index 双 QBUF 乱序)。
          # 非 nvbuf 模式 read_frame 内部已归还(mmap/VIC 分支), 这里不碰。
          if cam_nvbuf and not requeued and vision_buf is not None and vision_buf.v4l2_index is not None:
            try:
              cam.cam._requeue_buffer(vision_buf.v4l2_index)
            except Exception as e:
              print(f"[camerad] requeue failed idx={vision_buf.v4l2_index}: {e}", flush=True)
      # 无 sleep: 以传感器原速 30fps 消费, 每 3 帧交付 2 帧 = 精确 20fps。
      # 输出节奏锁定 VI 的 33.3ms 硬件网格 (66.7ms 双步), 零拍频零积压。
    else:
      for yuv in cam.read_frames():
        sync_frame_id = g_frame_sync.wait() if g_frame_sync.n_cameras > 1 else cam.cur_frame_id
        self._send_nv12(yuv, sync_frame_id, cam.cam_type_state, cam.stream_type)
        if g_frame_sync.n_cameras <= 1:
          cam.cur_frame_id += 1

  def run(self):
    threads = []
    for i, cam in enumerate(self.cameras):
      cam_thread = threading.Thread(target=self.camera_runner, args=(cam, i))
      cam_thread.start()
      threads.append(cam_thread)

    for t in threads:
      t.join()


def main():
  camerad = Camerad()
  camerad.run()


if __name__ == "__main__":
  main()
