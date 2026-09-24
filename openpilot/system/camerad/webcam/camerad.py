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
  def __init__(self, width, height):
    self.W = width
    self.H = height
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
    self._cudart = None
    if self._zerocopy:
      try:
        self._cudart = ctypes.CDLL('libcudart.so')
        self._cudart.cudaHostGetDevicePointer.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint]
        self._cudart.cudaHostGetDevicePointer.restype = ctypes.c_int
      except Exception as e:
        print(f"[camerad] zerocopy disabled, cudart load failed: {e}", flush=True)
        self._zerocopy = False

    if self._zerocopy and not hasattr(self.vipc_server, 'write_and_send'):
      print("[camerad] msgq visionipc 无 write_and_send(目标 fork 未打零拷贝补丁), 回退主机路径", flush=True)
      self._zerocopy = False

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
          raise RuntimeError(f"cudaHostGetDevicePointer rc={rc}")
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
      for vision_buf in cam.read_frames():
        in_count += 1
        if in_count % 3 == 0:
          continue  # 丢弃第 3 帧: 不进 FrameSync, 不占 VisionIPC buffer
        # Frame sync: road 致密发号(丢帧统计依赖), wide 跟随 road 最新号
        if cam.cam_type_state == "roadCameraState":
          sync_frame_id = g_frame_sync.claim()
        elif g_frame_sync.n_cameras > 1:
          sync_frame_id = g_frame_sync.follow()
        else:
          sync_frame_id = g_frame_sync.wait()
        converter = self.converters.get(cam.stream_type)
        sent = False
        if self._zerocopy and converter is not None and converter._packed is not None and vision_buf is not None and vision_buf.data is not None:
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
