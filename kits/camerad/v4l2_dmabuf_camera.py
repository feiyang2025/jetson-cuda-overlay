#!/usr/bin/env python3
"""V4L2 摄像头 + Jetson VIC 硬件零拷贝 UYVY→NV12 降采样"""
import fcntl
import ctypes
import struct
import os
import time
import mmap
import numpy as np


def v4l2_fourcc(a, b, c, d):
  return ord(a) | (ord(b) << 8) | (ord(c) << 16) | (ord(d) << 24)

# V4L2 ioctl 常量
VIDIOC_QUERYCAP    = 0x80685600
VIDIOC_QUERYBUF    = 0xC0585609
VIDIOC_G_FMT       = 0xC0CC5604
VIDIOC_REQBUFS     = 0xC0145608
VIDIOC_QBUF        = 0xC058560F
VIDIOC_DQBUF       = 0xC0585611
VIDIOC_STREAMON    = 0x40045612
VIDIOC_STREAMOFF   = 0x40045613
VIDIOC_EXPBUF      = 0xC0405610
VIDIOC_S_CTRL      = 0xC008561C

V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_MEMORY_MMAP            = 1
V4L2_MEMORY_DMABUF          = 4
V4L2_CAP_STREAMING = 0x04000000
V4L2_BUF_FLAG_ERROR = 0x00000040
V4L2_PIX_FMT_UYVY = v4l2_fourcc('U', 'Y', 'V', 'Y')
V4L2_PIX_FMT_NV12 = v4l2_fourcc('N', 'V', '1', '2')

# tegra-video 驱动的 exposure 控制 ID（0x009a200a，无 auto_exposure）
V4L2_CID_EXPOSURE_TEGRA = 0x009a200a

# NvBufSurface 常量（来自 nvbufsurface.h）
NVBUF_COLOR_FORMAT_UYVY = 10
NVBUF_COLOR_FORMAT_NV12 = 6
NVBUF_LAYOUT_PITCH      = 0
NVBUF_MEM_DEFAULT       = 0
NVBUF_MAP_READ          = 1

# NvBufSurfTransform 常量（来自 nvbufsurftransform.h）
NVBUFSURF_TRANSFORM_CROP_SRC = 1
NVBUFSURF_TRANSFORM_FILTER   = 4
NVBUFSURF_INTER_SMART        = 4   # NvBufSurfTransformInter_Algo3 (VIC-Smart)
NVBUFSURF_COMPUTE_VIC        = 2   # NvBufSurfTransformCompute_VIC

# NvBufSurface 结构体偏移（gcc offsetof 验证）
SURFACELIST_OFFSET = 24      # NvBufSurface.surfaceList
PARAMS_DATASIZE    = 32      # NvBufSurfaceParams.dataSize
PARAMS_PITCH       = 8       # NvBufSurfaceParams.pitch
PARAMS_MAPPEDADDR  = 280     # NvBufSurfaceParams.mappedAddr
MAPPEDADDR_ADDR0   = 0       # NvBufSurfaceMappedAddr.addr[0]


class v4l2_requestbuffers(ctypes.Structure):
  _fields_ = [
    ("count", ctypes.c_uint32), ("type", ctypes.c_uint32),
    ("memory", ctypes.c_uint32), ("capabilities", ctypes.c_uint32),
    ("reserved", ctypes.c_uint32),
  ]


class v4l2_exportbuffer(ctypes.Structure):
  _fields_ = [
    ("type", ctypes.c_uint32), ("index", ctypes.c_uint32),
    ("plane", ctypes.c_uint32), ("flags", ctypes.c_uint32),
    ("fd", ctypes.c_int32), ("reserved", ctypes.c_uint32 * 11),
  ]


class v4l2_timeval(ctypes.Structure):
  _fields_ = [("tv_sec", ctypes.c_long), ("tv_usec", ctypes.c_long)]


class v4l2_timecode(ctypes.Structure):
  _fields_ = [
    ("type", ctypes.c_uint32), ("flags", ctypes.c_uint32),
    ("frames", ctypes.c_uint8), ("seconds", ctypes.c_uint8),
    ("minutes", ctypes.c_uint8), ("hours", ctypes.c_uint8),
    ("userbits", ctypes.c_uint8 * 4),
  ]


class v4l2_buffer(ctypes.Structure):
  _fields_ = [
    ("index", ctypes.c_uint32), ("type", ctypes.c_uint32),
    ("bytesused", ctypes.c_uint32), ("flags", ctypes.c_uint32),
    ("field", ctypes.c_uint32), ("timestamp", v4l2_timeval),
    ("timecode", v4l2_timecode), ("sequence", ctypes.c_uint32),
    ("memory", ctypes.c_uint32), ("m", ctypes.c_uint64),
    ("length", ctypes.c_uint32), ("input", ctypes.c_uint32),
    ("reserved", ctypes.c_uint32 * 4),
  ]


class v4l2_control(ctypes.Structure):
  _fields_ = [
    ("id", ctypes.c_uint32),
    ("value", ctypes.c_int32),
  ]


class NvBufSurfaceCreateParams(ctypes.Structure):
  _fields_ = [
    ("gpuId", ctypes.c_uint32),
    ("width", ctypes.c_uint32),
    ("height", ctypes.c_uint32),
    ("size", ctypes.c_uint32),
    ("isContiguous", ctypes.c_bool),
    ("_pad", ctypes.c_uint8 * 3),
    ("colorFormat", ctypes.c_uint32),
    ("layout", ctypes.c_uint32),
    ("memType", ctypes.c_uint32),
  ]


class _NvBufSurfaceParamsLite(ctypes.Structure):
  _fields_ = [
    ("width", ctypes.c_uint32),
    ("height", ctypes.c_uint32),
    ("pitch", ctypes.c_uint32),
    ("colorFormat", ctypes.c_uint32),
    ("layout", ctypes.c_uint32),
    ("bufferDesc", ctypes.c_uint64),
    ("dataSize", ctypes.c_uint32),
    ("_pad", ctypes.c_uint32),
    ("dataPtr", ctypes.c_void_p),
  ]


class _NvBufSurfaceLite(ctypes.Structure):
  _fields_ = [
    ("gpuId", ctypes.c_uint32),
    ("batchSize", ctypes.c_uint32),
    ("numFilled", ctypes.c_uint32),
    ("isContiguous", ctypes.c_bool),
    ("_pad", ctypes.c_uint8 * 3),
    ("memType", ctypes.c_uint32),
    ("surfaceList", ctypes.POINTER(_NvBufSurfaceParamsLite)),
  ]


class NvBufSurfTransformRect(ctypes.Structure):
  _fields_ = [
    ("top", ctypes.c_uint32), ("left", ctypes.c_uint32),
    ("width", ctypes.c_uint32), ("height", ctypes.c_uint32),
    # 真实 C 结构含 void* reserved[STRUCTURE_PADDING=4]（32B）→ 总 48B
    ("reserved", ctypes.c_void_p * 4),
  ]


class NvBufSurfTransformParams(ctypes.Structure):
  _fields_ = [
    ("transform_flag", ctypes.c_uint32),
    ("transform_flip", ctypes.c_uint32),
    ("transform_filter", ctypes.c_uint32),
    ("_pad", ctypes.c_uint32),
    ("src_rect", ctypes.POINTER(NvBufSurfTransformRect)),
    ("dst_rect", ctypes.POINTER(NvBufSurfTransformRect)),
    # 真实 C 结构含 void* reserved[STRUCTURE_PADDING=4]（32B）→ 总 64B
    ("reserved", ctypes.c_void_p * 4),
  ]


class NvBufSurfTransformConfigParams(ctypes.Structure):
  _fields_ = [
    ("compute_mode", ctypes.c_uint32),
    ("gpu_id", ctypes.c_int32),
    ("cuda_stream", ctypes.c_void_p),
  ]


class VisionBuf:
  def __init__(self, dmabuf_fd, width, height, stride, data_size, frame_id, pixel_format='NV12', data=None, timestamp_sof=0, timestamp_eof=0, v4l2_index=None):
    self.fd = dmabuf_fd
    self.width = width
    self.height = height
    self.stride = stride
    self.data_size = data_size
    self.len = data_size
    self.idx = 0
    self.addr = None
    self.nvmm_surf = None
    self.nvmm_owner = False
    self.frame_id_storage = frame_id
    self.frame_id = self.frame_id_storage
    self.pixel_format = pixel_format
    self.data = data
    self.timestamp_sof = timestamp_sof
    self.timestamp_eof = timestamp_eof
    self.v4l2_index = v4l2_index


class V4L2Camera:
  """V4L2 摄像头 + Jetson VIC 硬件零拷贝 UYVY→NV12 降采样"""
  # twgmsl VI 按 DTB 虚标 4K 出 buffer; IMX390 传感器真实 1920x1080。
  # 2026-08-30 探针实测 (16588800B buffer): 真实数据只占左上角 ——
  # 行距 7680B, 每行前 3840B (=1920 像素) 有效, 共 1080 行, 其余全零。
  CAMERA_NATIVE_WIDTH = 3840
  CAMERA_NATIVE_HEIGHT = 2160
  CAMERA_NATIVE_STRIDE = 7680
  CAMERA_NATIVE_SIZE = 16588800

  ACTIVE_WIDTH = 1920             # buffer 内真实图像区 (传感器有效像素)
  ACTIVE_HEIGHT = 1080

  OUTPUT_WIDTH = 1920             # 下游输出 NV12 宽 (env CAM_WIDTH/SP_CAM_OUT_W 可覆盖, cp 用 1344)
  OUTPUT_HEIGHT = 1080            # 下游输出 NV12 高 (env CAM_HEIGHT/SP_CAM_OUT_H 可覆盖, cp 用 760)
  OUTPUT_STRIDE = 1920
  OUTPUT_SIZE = 3110400           # 1920*1080*3//2

  def __init__(self, device, width=None, height=None, fps=20, num_buffers=4, exposure=33334):
    self.device = device
    # 输出尺寸: 优先显式参数, 其次 CAM_WIDTH/CAM_HEIGHT (cp 兼容), 最后默认 1920x1080
    if width is None:
      width = int(os.environ.get("CAM_WIDTH", os.environ.get("SP_CAM_OUT_W", str(self.OUTPUT_WIDTH))))
    if height is None:
      height = int(os.environ.get("CAM_HEIGHT", os.environ.get("SP_CAM_OUT_H", str(self.OUTPUT_HEIGHT))))
    self.target_w = width
    self.target_h = height
    self.W = width
    self.H = height
    self.cam_w = width
    self.cam_h = height
    self.fps = fps
    self.num_buffers = num_buffers
    self.exposure = exposure
    self.fd = None
    self.dmabuf_fds = []
    self.mmap_ptrs = []
    self.mmap_objs = []
    self.streaming = False
    self.cur_frame_id = 0
    self.cam_bytesperline = 0
    self.cam_sizeimage = 0
    self._vic = False
    self._external_dst_surfs = {}
    self._fd_cache_generation = 0
    self._nv12_size = width * height * 3 // 2
    self._open()

  def _probe_frame_bytesused(self):
    """open 阶段布局探测: 临时 MMAP 抓几帧, 返回稳定后的 bytesused。
    结束时 STREAMOFF + 释放全部临时资源 + munmap, 不留脏状态。"""
    probe = v4l2_requestbuffers()
    probe.count = 2
    probe.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
    probe.memory = V4L2_MEMORY_MMAP
    mms = []
    seen = []
    try:
      fcntl.ioctl(self.fd, VIDIOC_REQBUFS, probe)
      if probe.count == 0:
        return -1
      qbufs = []
      mms = []
      for i in range(probe.count):
        qb = v4l2_buffer()
        qb.index = i
        qb.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        qb.memory = V4L2_MEMORY_MMAP
        fcntl.ioctl(self.fd, VIDIOC_QUERYBUF, qb)
        mm = mmap.mmap(self.fd, qb.length, mmap.MAP_SHARED, mmap.PROT_READ, offset=qb.m)
        mms.append((mm, qb.length))
        qbufs.append((i, qb.length))
        buf = v4l2_buffer()
        buf.index = i
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        buf.memory = V4L2_MEMORY_MMAP
        fcntl.ioctl(self.fd, VIDIOC_QBUF, buf)
      fcntl.ioctl(self.fd, VIDIOC_STREAMON, struct.pack("I", V4L2_BUF_TYPE_VIDEO_CAPTURE))
      seen = []
      deadline = time.monotonic() + 2.0
      while len(seen) < 4 and time.monotonic() < deadline:
        buf = v4l2_buffer()
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        buf.memory = V4L2_MEMORY_MMAP
        try:
          fcntl.ioctl(self.fd, VIDIOC_DQBUF, buf)
        except BlockingIOError:
          time.sleep(0.002)
          continue
        seen.append(buf.bytesused)
        rq = v4l2_buffer()
        rq.index = buf.index
        rq.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        rq.memory = V4L2_MEMORY_MMAP
        fcntl.ioctl(self.fd, VIDIOC_QBUF, rq)
    except OSError as e:
      print(f"[V4L2Camera] {self.device}: 布局探测失败: {e}", flush=True)
      return -1
    finally:
      try:
        fcntl.ioctl(self.fd, VIDIOC_STREAMOFF, struct.pack("I", V4L2_BUF_TYPE_VIDEO_CAPTURE))
      except OSError:
        pass
      # 释放探测 REQBUFS(带已 mmap 的缓冲需先 munmap), 把链路还原到未申请状态
      try:
        for mm, length in mms:
          mm.close()
      except Exception:
        pass
      try:
        release = v4l2_requestbuffers()
        release.count = 0
        release.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        release.memory = V4L2_MEMORY_MMAP
        fcntl.ioctl(self.fd, VIDIOC_REQBUFS, release)
      except OSError:
        pass
    if not seen:
      return -1
    # 稳定值: 取众数 (前几帧可能有过渡)
    vals, counts = np.unique(np.array(seen), return_counts=True)
    return int(vals[np.argmax(counts)])

  def _open(self):
    self.fd = os.open(self.device, os.O_RDWR | os.O_NONBLOCK)

    cap = bytearray(104)
    fcntl.ioctl(self.fd, VIDIOC_QUERYCAP, cap)
    caps = struct.unpack_from("I", cap, 84)[0]
    dev_caps = struct.unpack_from("I", cap, 88)[0]
    if not (dev_caps & V4L2_CAP_STREAMING) and not (caps & V4L2_CAP_STREAMING):
      raise RuntimeError(f"{self.device} 不支持 STREAMING")

    self._gfmt_ok = False
    try:
      fmt = bytearray(204)
      struct.pack_into("I", fmt, 0, V4L2_BUF_TYPE_VIDEO_CAPTURE)
      fcntl.ioctl(self.fd, VIDIOC_G_FMT, fmt)
      # struct v4l2_format: type@0 (4B) → union 对齐到 8 → pix.width@8 height@12
      # pixelformat@16 bytesperline@24 sizeimage@28
      self.cam_w = struct.unpack_from("I", fmt, 8)[0]
      self.cam_h = struct.unpack_from("I", fmt, 12)[0]
      self.cam_pixelformat = struct.unpack_from("I", fmt, 16)[0]
      self.cam_bytesperline = struct.unpack_from("I", fmt, 24)[0]
      self.cam_sizeimage = struct.unpack_from("I", fmt, 28)[0]
      self.cam_format_name = {V4L2_PIX_FMT_UYVY: 'UYVY', V4L2_PIX_FMT_NV12: 'NV12'}.get(self.cam_pixelformat, f"0x{self.cam_pixelformat:08x}")
      self._gfmt_ok = True
      print(f"[V4L2Camera] {self.device}: G_FMT {self.cam_w}x{self.cam_h} {self.cam_format_name} stride={self.cam_bytesperline}")
    except OSError:
      self.cam_w = self.CAMERA_NATIVE_WIDTH
      self.cam_h = self.CAMERA_NATIVE_HEIGHT
      self.cam_pixelformat = V4L2_PIX_FMT_UYVY
      self.cam_format_name = 'UYVY'
      self.cam_bytesperline = self.cam_w * 2
      self.cam_sizeimage = self.cam_h * self.cam_bytesperline
      print(f"[V4L2Camera] {self.device}: G_FMT 失败 (twgmsl 不实现)，buffer 几何用 {self.cam_w}x{self.cam_h} UYVY")

    # G_FMT 失败 = twgmsl（驱动不实现 G_FMT）。布局用实测判定：驱动状态不同会发两种帧：
    #   ① 4K 画布: bytesused=16588800, 行距 7680, 真实 1920x1080 在左上（虚标状态）
    #   ② 紧凑 1080p: bytesused=4147200, 行距 3840, 整幅有效（真实状态，硬杀/重载后可能出现）
    # 靠猜会出 "1/4 画面" 或 "上半画面下半绿"，所以 open 时先抓一帧量 bytesused。
    if self._gfmt_ok:
      self.cam_active_w, self.cam_active_h = self.cam_w, self.cam_h
    else:
      probed = self._probe_frame_bytesused()
      if probed == self.CAMERA_NATIVE_SIZE:
        self.cam_w, self.cam_h = self.CAMERA_NATIVE_WIDTH, self.CAMERA_NATIVE_HEIGHT
        self.cam_bytesperline = self.cam_w * 2
        self.cam_sizeimage = self.CAMERA_NATIVE_SIZE
        self.cam_active_w, self.cam_active_h = self.ACTIVE_WIDTH, self.ACTIVE_HEIGHT
        print(f"[V4L2Camera] {self.device}: 实测 bytesused={probed} → 4K 画布布局, 有效区 {self.cam_active_w}x{self.cam_active_h}", flush=True)
      elif probed == self.ACTIVE_WIDTH * self.ACTIVE_HEIGHT * 2:
        self.cam_w, self.cam_h = self.ACTIVE_WIDTH, self.ACTIVE_HEIGHT
        self.cam_bytesperline = self.cam_w * 2
        self.cam_sizeimage = probed
        self.cam_active_w, self.cam_active_h = self.cam_w, self.cam_h
        print(f"[V4L2Camera] {self.device}: 实测 bytesused={probed} → 紧凑 1080p 布局, 整幅有效", flush=True)
      else:
        self.cam_active_w = min(self.cam_w, self.ACTIVE_WIDTH)
        self.cam_active_h = min(self.cam_h, self.ACTIVE_HEIGHT)
        print(f"[V4L2Camera] {self.device}: 探测失败 (bytesused={probed})，按 4K 画布布局兜底", flush=True)
      self._expected_bytesused = probed

    if self.exposure is not None:
      ctrl = v4l2_control()
      ctrl.id = V4L2_CID_EXPOSURE_TEGRA
      ctrl.value = self.exposure
      try:
        fcntl.ioctl(self.fd, VIDIOC_S_CTRL, ctrl)
        print(f"[V4L2Camera] {self.device}: exposure={self.exposure}", flush=True)
      except OSError as e:
        print(f"[V4L2Camera] {self.device}: 设曝光失败 {e}", flush=True)

    # 相机输出不是 NV12（或需要缩放）时，启用 VIC 硬件转换 UYVY→NV12
    # 注：taegra-camrtc 驱动不实现 G_FMT/S_FMT，G_FMT 失败时 cam_pixelformat 为 UYVY，
    # 因此同尺寸 1920x1080 也会走 VIC 硬件转换而非 CPU fallback
    # twgmsl 当前输出不是标准 UYVY：色度需按 GMSL_CHROMA_LAYOUT 归一化。
    # VIC 只能按标准 UYVY 解码，直接走 VIC 会把错误色度带入 NV12 造成纯绿；
    # 因此 GMSL 特殊布局默认禁用 VIC，除非显式 SP_ENABLE_VIC_GMSL=1。
    gmsl_special_layout = os.environ.get('GMSL_CHROMA_LAYOUT', 'twgmsl').lower() == 'twgmsl'
    vic_allowed = (not gmsl_special_layout) or os.environ.get('SP_ENABLE_VIC_GMSL') == '1'
    if vic_allowed and os.environ.get("SP_FORCE_NO_VIC") != "1" and (self.cam_pixelformat != V4L2_PIX_FMT_NV12 or (self.cam_active_w, self.cam_active_h) != (self.target_w, self.target_h)):
      print(f"[V4L2Camera] {self.device}: VIC 裁剪/转换 buffer {self.cam_w}x{self.cam_h} 有效区 {self.cam_active_w}x{self.cam_active_h} → {self.target_w}x{self.target_h} NV12", flush=True)
      try:
        self.W = self.target_w
        self.H = self.target_h
        self._init_vic()
      except Exception as e:
        print(f"[V4L2Camera] {self.device}: VIC 初始化失败，回退 MMAP+CUDA: {e}", flush=True)
        self._vic = False
        self.W = self.target_w
        self.H = self.target_h
    else:
      # 200万像素契约: 无论 V4L2 画布是 1080p 还是虚标 4K, 下游 NV12 都是 1920x1080。
      # 有效像素由 _numpy_downsample 按 cam_active 裁出。
      self.W = self.target_w
      self.H = self.target_h

    self._src_surfs = []
    self._nvbuf_fds = []
    self._nvbuf_zerocopy = os.environ.get("SP_NVBUF_ZEROCOPY", "0") == "1" and not self._vic
    if self._nvbuf_zerocopy:
      try:
        self._nvbuf = ctypes.CDLL("/usr/lib/aarch64-linux-gnu/nvidia/libnvbufsurface.so")
        self._nvbuf.NvBufSurfaceCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32, ctypes.POINTER(NvBufSurfaceCreateParams)]
        self._nvbuf.NvBufSurfaceCreate.restype = ctypes.c_int
        self._nvbuf.NvBufSurfaceDestroy.argtypes = [ctypes.c_void_p]
        self._nvbuf.NvBufSurfaceDestroy.restype = ctypes.c_int
        for i in range(self.num_buffers):
          src_params = NvBufSurfaceCreateParams()
          src_params.gpuId = 0
          # 源尺寸 = 相机实际采集尺寸 (cam_active = 探测的有效区, 1920x1080) —
          # 驱动按 sensor 真实尺寸写 DMA; 下游输出尺寸(target_w/h)可能不同(cp: 1344x760),
          # 由 CUDA kernel resize, 不能拿输出尺寸建源 surface(会溢出/越界)。
          src_params.width = self.cam_active_w
          src_params.height = self.cam_active_h
          src_params.size = 0
          src_params.isContiguous = True
          src_params.colorFormat = NVBUF_COLOR_FORMAT_UYVY
          src_params.layout = NVBUF_LAYOUT_PITCH
          src_params.memType = NVBUF_MEM_DEFAULT
          src_surf = ctypes.c_void_p()
          r = self._nvbuf.NvBufSurfaceCreate(ctypes.byref(src_surf), 1, ctypes.byref(src_params))
          if r != 0:
            raise RuntimeError(f"NvBufSurfaceCreate(src) failed: {r}")
          self._src_surfs.append(src_surf)
          params = ctypes.cast(src_surf, ctypes.POINTER(_NvBufSurfaceLite)).contents.surfaceList.contents
          self._nvbuf_fds.append(int(params.bufferDesc))
          print(f"[V4L2Camera] {self.device}: NVBUF zerocopy buffer {i} fd={params.bufferDesc} size={params.dataSize} pitch={params.pitch}", flush=True)
      except Exception as e:
        print(f"[V4L2Camera] {self.device}: NVBUF zerocopy init failed, fallback MMAP: {e}", flush=True)
        for surf in self._src_surfs:
          try:
            self._nvbuf.NvBufSurfaceDestroy(surf)
          except Exception:
            pass
        self._src_surfs = []
        self._nvbuf_fds = []
        self._nvbuf_zerocopy = False
    if self._vic:
      try:
        for i in range(self.num_buffers):
          src_params = NvBufSurfaceCreateParams()
          src_params.gpuId = 0
          src_params.width = self.cam_w
          src_params.height = self.cam_h
          src_params.size = 0
          src_params.isContiguous = False
          src_params.colorFormat = NVBUF_COLOR_FORMAT_UYVY
          src_params.layout = NVBUF_LAYOUT_PITCH
          src_params.memType = NVBUF_MEM_DEFAULT

          src_surf = ctypes.c_void_p()
          r = self._nvbuf.NvBufSurfaceCreate(ctypes.byref(src_surf), 1, ctypes.byref(src_params))
          if r != 0:
            raise RuntimeError(f"NvBufSurfaceCreate(src) failed: {r}")
          ctypes.c_uint32.from_address(src_surf.value + 8).value = 1
          self._src_surfs.append(src_surf)

          sl_ptr = ctypes.c_void_p.from_address(src_surf.value + SURFACELIST_OFFSET).value
          dmabuf_fd = ctypes.c_int32.from_address(sl_ptr + 24).value
          self._nvbuf_fds.append(dmabuf_fd)
          print(f"[VIC] src buffer {i}: NvBufSurface fd={dmabuf_fd}", flush=True)
      except Exception as e:
        print(f"[V4L2Camera] {self.device}: VIC src buffer 创建失败，回退 MMAP+CUDA: {e}", flush=True)
        self._vic = False
        self._src_surfs = []
        self._nvbuf_fds = []
        self.W = self.cam_w
        self.H = self.cam_h

    req = v4l2_requestbuffers()
    req.count = self.num_buffers
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
    if (self._vic or self._nvbuf_zerocopy) and self._nvbuf_fds:
      req.memory = V4L2_MEMORY_DMABUF
    else:
      req.memory = V4L2_MEMORY_MMAP
    try:
      fcntl.ioctl(self.fd, VIDIOC_REQBUFS, req)
      use_dmabuf = (self._vic or self._nvbuf_zerocopy) and self._nvbuf_fds
    except OSError:
      print(f"[V4L2Camera] DMABUF 不支持，回退 MMAP", flush=True)
      req.memory = V4L2_MEMORY_MMAP
      fcntl.ioctl(self.fd, VIDIOC_REQBUFS, req)
      use_dmabuf = False
      self._vic = False

    if req.count < 2:
      raise RuntimeError(f"{self.device} 只分配了 {req.count} 个缓冲区")
    self.num_buffers = req.count
    self._use_dmabuf = use_dmabuf

    if use_dmabuf:
      for i in range(self.num_buffers):
        buf = v4l2_buffer()
        buf.index = i
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        buf.memory = V4L2_MEMORY_DMABUF
        buf.m = self._nvbuf_fds[i]
        fcntl.ioctl(self.fd, VIDIOC_QBUF, buf)
      print(f"[V4L2Camera] DMABUF 模式: {self.num_buffers} buffers", flush=True)
    else:
      for i in range(self.num_buffers):
        expbuf = v4l2_exportbuffer()
        expbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        expbuf.index = i
        expbuf.plane = 0
        expbuf.flags = 0
        expbuf.fd = -1
        fcntl.ioctl(self.fd, VIDIOC_EXPBUF, expbuf)
        self.dmabuf_fds.append(expbuf.fd)
        mm = mmap.mmap(expbuf.fd, self.cam_sizeimage, mmap.MAP_SHARED, mmap.PROT_READ, offset=0)
        self.mmap_objs.append(mm)
        self.mmap_ptrs.append(mm)
        print(f"[V4L2Camera] Buffer {i}: EXPBUF fd={expbuf.fd}")
      for i in range(self.num_buffers):
        buf = v4l2_buffer()
        buf.index = i
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        buf.memory = V4L2_MEMORY_MMAP
        fcntl.ioctl(self.fd, VIDIOC_QBUF, buf)

    buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE
    fcntl.ioctl(self.fd, VIDIOC_STREAMON, struct.pack("I", buf_type))
    self.streaming = True
    print(f"[V4L2Camera] Streaming started on {self.device}")

  def _init_vic(self):
    self._nvbuf = ctypes.CDLL('libnvbufsurface.so')
    self._nvtf = ctypes.CDLL('libnvbufsurftransform.so')

    self._nvbuf.NvBufSurfaceCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32, ctypes.POINTER(NvBufSurfaceCreateParams)]
    self._nvbuf.NvBufSurfaceCreate.restype = ctypes.c_int
    self._nvbuf.NvBufSurfaceFromFd.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
    self._nvbuf.NvBufSurfaceFromFd.restype = ctypes.c_int
    self._nvbuf.NvBufSurfaceDestroy.argtypes = [ctypes.c_void_p]
    self._nvbuf.NvBufSurfaceDestroy.restype = ctypes.c_int
    self._nvbuf.NvBufSurfaceMap.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_uint32]
    self._nvbuf.NvBufSurfaceMap.restype = ctypes.c_int
    self._nvbuf.NvBufSurfaceUnMap.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    self._nvbuf.NvBufSurfaceUnMap.restype = ctypes.c_int
    self._nvbuf.NvBufSurfaceSyncForCpu.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    self._nvbuf.NvBufSurfaceSyncForCpu.restype = ctypes.c_int

    self._nvtf.NvBufSurfTransformSetSessionParams.argtypes = [ctypes.POINTER(NvBufSurfTransformConfigParams)]
    self._nvtf.NvBufSurfTransformSetSessionParams.restype = ctypes.c_int
    self._nvtf.NvBufSurfTransform.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(NvBufSurfTransformParams)]
    self._nvtf.NvBufSurfTransform.restype = ctypes.c_int

    create_params = NvBufSurfaceCreateParams()
    create_params.gpuId = 0
    create_params.width = self.target_w
    create_params.height = self.target_h
    create_params.size = 0
    create_params.isContiguous = False
    create_params.colorFormat = NVBUF_COLOR_FORMAT_NV12
    create_params.layout = NVBUF_LAYOUT_PITCH
    create_params.memType = NVBUF_MEM_DEFAULT

    self._dst_surf = ctypes.c_void_p()
    r = self._nvbuf.NvBufSurfaceCreate(ctypes.byref(self._dst_surf), 1, ctypes.byref(create_params))
    if r != 0:
      raise RuntimeError(f"NvBufSurfaceCreate failed: {r}")
    ctypes.c_uint32.from_address(self._dst_surf.value + 8).value = 1
    print(f"[VIC] dst_surf created: {self.target_w}x{self.target_h} NV12", flush=True)

    self._nvtf.NvBufSurfTransformSetDefaultSession()
    print(f"[VIC] Default session set", flush=True)

    self._src_rect = NvBufSurfTransformRect(0, 0, self.cam_active_w, self.cam_active_h)
    self._dst_rect = NvBufSurfTransformRect(0, 0, self.target_w, self.target_h)
    self._tf_params = NvBufSurfTransformParams()
    self._tf_params.transform_flag = NVBUFSURF_TRANSFORM_CROP_SRC | NVBUFSURF_TRANSFORM_FILTER
    self._tf_params.transform_flip = 0
    self._tf_params.transform_filter = NVBUFSURF_INTER_SMART
    self._tf_params.src_rect = ctypes.pointer(self._src_rect)
    self._tf_params.dst_rect = ctypes.pointer(self._dst_rect)

    self._dst_np = np.empty(self._nv12_size, dtype=np.uint8)
    self._vic = True
    print(f"[VIC] Init done: nv12_size={self._nv12_size}", flush=True)

  def _clear_external_dst_cache(self, reason):
    if self._external_dst_surfs:
      print(f"[V4L2Camera] {self.device}: clearing external dst fd cache reason={reason} entries={len(self._external_dst_surfs)} gen={self._fd_cache_generation}", flush=True)
    self._external_dst_surfs.clear()
    if hasattr(self, '_external_dst_cache_hit_logged'):
      delattr(self, '_external_dst_cache_hit_logged')

  def _get_external_dst_surf(self, dst_fd):
    if dst_fd in self._external_dst_surfs:
      if not hasattr(self, '_external_dst_cache_hit_logged'):
        print(f"[V4L2Camera] {self.device}: external dst fd cache hit fd={dst_fd} entries={len(self._external_dst_surfs)} gen={self._fd_cache_generation}", flush=True)
        self._external_dst_cache_hit_logged = True
      return self._external_dst_surfs[dst_fd]

    if len(self._external_dst_surfs) >= self.num_buffers:
      self._clear_external_dst_cache(f"fd rollover new_fd={dst_fd}")

    dst_surf = ctypes.c_void_p()
    r = self._nvbuf.NvBufSurfaceFromFd(int(dst_fd), ctypes.byref(dst_surf))
    if r != 0 or not dst_surf.value:
      self._clear_external_dst_cache(f"import failed fd={dst_fd} ret={r}")
      raise RuntimeError(f"NvBufSurfaceFromFd(dst_fd={dst_fd}) failed: {r}")

    ctypes.c_uint32.from_address(dst_surf.value + 8).value = 1
    self._external_dst_surfs[dst_fd] = dst_surf
    self._fd_cache_generation += 1
    print(f"[V4L2Camera] {self.device}: external dst fd imported fd={dst_fd} entries={len(self._external_dst_surfs)} gen={self._fd_cache_generation}", flush=True)
    return dst_surf

  def _vic_transform(self, src_surf, dst_surf):
    r = self._nvtf.NvBufSurfTransform(src_surf, dst_surf, ctypes.byref(self._tf_params))
    if r != 0:
      raise RuntimeError(f"NvBufSurfTransform failed: {r}")
    if not hasattr(self, '_vic_tf_ok'):
      self._vic_tf_ok = True

  def vic_convert_to_fd(self, buf_index, dst_fd):
    if not self._vic:
      raise RuntimeError("VIC is not initialized")
    src_surf = self._src_surfs[buf_index]
    dst_surf = self._get_external_dst_surf(dst_fd)
    self._vic_transform(src_surf, dst_surf)
    return True

  def _vic_convert(self, buf_index):
    src_surf = self._src_surfs[buf_index]
    self._vic_transform(src_surf, self._dst_surf)

    self._nvbuf.NvBufSurfaceMap(self._dst_surf, 0, -1, NVBUF_MAP_READ)
    self._nvbuf.NvBufSurfaceSyncForCpu(self._dst_surf, 0, -1)

    surf_ptr = self._dst_surf.value
    surfaceList_ptr = ctypes.c_void_p.from_address(surf_ptr + SURFACELIST_OFFSET).value
    data_ptr = ctypes.c_void_p.from_address(surfaceList_ptr + PARAMS_MAPPEDADDR + MAPPEDADDR_ADDR0).value
    dst_pitch = ctypes.c_uint32.from_address(surfaceList_ptr + PARAMS_PITCH).value

    if data_ptr and dst_pitch > 0:
      uv_ptr = ctypes.c_void_p.from_address(surfaceList_ptr + PARAMS_MAPPEDADDR + 8).value
      y_rows = self.target_h
      uv_rows = self.target_h // 2
      row_bytes = self.target_w
      y_size = row_bytes * y_rows

      y_src = np.ctypeslib.as_array((ctypes.c_uint8 * (dst_pitch * y_rows)).from_address(data_ptr)).reshape(y_rows, dst_pitch)
      self._dst_np[:y_size].reshape(y_rows, row_bytes)[:, :] = y_src[:, :row_bytes]

      uv_base = uv_ptr if uv_ptr else data_ptr + dst_pitch * self.target_h
      uv_src = np.ctypeslib.as_array((ctypes.c_uint8 * (dst_pitch * uv_rows)).from_address(uv_base)).reshape(uv_rows, dst_pitch)
      self._dst_np[y_size:].reshape(uv_rows, row_bytes)[:, :] = uv_src[:, :row_bytes]

    self._nvbuf.NvBufSurfaceUnMap(self._dst_surf, 0, -1)
    return self._dst_np

  @staticmethod
  def _buf_timestamp_ns(buf):
    return int(buf.timestamp.tv_sec) * 1_000_000_000 + int(buf.timestamp.tv_usec) * 1_000

  def _is_bad_buffer(self, buf):
    return buf.index >= self.num_buffers or (buf.bytesused == 0 and not self._use_dmabuf)

  def _timestamp_ns_or_now(self, buf):
    timestamp_ns = self._buf_timestamp_ns(buf)
    if timestamp_ns > 0:
      return timestamp_ns
    if not hasattr(self, '_zero_timestamp_seen'):
      self._zero_timestamp_seen = True
      print(f"[V4L2Camera] {self.device}: V4L2 timestamp is zero; using monotonic clock for subsequent zero timestamp buffers", flush=True)
    return time.monotonic_ns()

  def _note_error_flag(self, buf):
    if (buf.flags & V4L2_BUF_FLAG_ERROR) and not hasattr(self, '_error_flag_seen'):
      self._error_flag_seen = True
      print(f"[V4L2Camera] {self.device}: V4L2 error flag is set on otherwise usable buffers flags=0x{buf.flags:x} bytesused={buf.bytesused} ts={self._buf_timestamp_ns(buf)}", flush=True)

  def _requeue_buffer(self, index):
    qbuf = v4l2_buffer()
    qbuf.index = index
    qbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
    if self._use_dmabuf:
      qbuf.memory = V4L2_MEMORY_DMABUF
      qbuf.m = self._nvbuf_fds[index]
    else:
      qbuf.memory = V4L2_MEMORY_MMAP
    fcntl.ioctl(self.fd, VIDIOC_QBUF, qbuf)

  def read_frame_to_fd(self, dst_fd):
    if not self._vic or not self._use_dmabuf:
      raise RuntimeError("External dst_fd path requires VIC + V4L2 DMABUF")

    frame_id = self.cur_frame_id
    timestamp_sof = 0
    timestamp_eof = 0
    buf = v4l2_buffer()
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
    buf.memory = V4L2_MEMORY_DMABUF
    while True:
      try:
        fcntl.ioctl(self.fd, VIDIOC_DQBUF, buf)
        index = buf.index
        if self._is_bad_buffer(buf):
          if index < self.num_buffers:
            self._requeue_buffer(index)
          print(f"[V4L2Camera] {self.device}: dropped bad buffer index={index} flags=0x{buf.flags:x} bytesused={buf.bytesused} ts={self._buf_timestamp_ns(buf)}", flush=True)
          continue
        self._note_error_flag(buf)
        timestamp_sof = self._timestamp_ns_or_now(buf)
        timestamp_eof = timestamp_sof
        break
      except BlockingIOError:
        time.sleep(0.001)
    index = buf.index

    try:
      self.vic_convert_to_fd(index, dst_fd)
      if not hasattr(self, '_nvmm_path_logged'):
        print(f"[V4L2Camera] {self.device}: NVMM path active src_index={index} src_fd={self._nvbuf_fds[index]} dst_fd={dst_fd}", flush=True)
        self._nvmm_path_logged = True
    except Exception:
      if dst_fd in self._external_dst_surfs:
        self._external_dst_surfs.pop(dst_fd, None)
        print(f"[V4L2Camera] {self.device}: dropped cached external dst fd after transform failure fd={dst_fd}", flush=True)
      raise
    finally:
      self._requeue_buffer(index)

    return frame_id, timestamp_sof, timestamp_eof

  def read_frame(self):
    _t0 = time.monotonic()
    buf = v4l2_buffer()
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
    buf.memory = V4L2_MEMORY_DMABUF if self._use_dmabuf else V4L2_MEMORY_MMAP
    _bad = 0
    _eagain = 0
    while True:
      try:
        fcntl.ioctl(self.fd, VIDIOC_DQBUF, buf)
      except BlockingIOError:
        _eagain += 1
        time.sleep(0.001)
        continue
      index = buf.index
      if self._is_bad_buffer(buf):
        if index < self.num_buffers:
          self._requeue_buffer(index)
        _bad += 1
        print(f"[V4L2Camera] {self.device}: dropped bad buffer index={index} flags=0x{buf.flags:x} bytesused={buf.bytesused} ts={self._buf_timestamp_ns(buf)}", flush=True)
        continue
      self._note_error_flag(buf)
      timestamp_sof = self._timestamp_ns_or_now(buf)
      timestamp_eof = timestamp_sof
      break
    _t1 = time.monotonic()  # DQBUF 完成

    if self._vic:
      data = self._vic_convert(index)
      self._requeue_buffer(index)
    elif self._nvbuf_zerocopy:
      data = None
    else:
      mm = self.mmap_ptrs[index]
      raw_data = bytes(mm[:self.cam_sizeimage])
      self._requeue_buffer(index)
      if self.cam_pixelformat == V4L2_PIX_FMT_NV12:
        data = raw_data
      else:
        data = self._numpy_downsample(raw_data)
    _t2 = time.monotonic()  # VIC 转换+回队 完成

    # 诊断计时(可通过 SP_TIMING=1 开启): dq等待/EAGAIN空转/VIC耗时/SOF硬件时间戳间隔,
    # 每60帧写 /tmp/vic_timing.log。平时关闭, 零开销。
    if os.environ.get("SP_TIMING"):
      if not hasattr(self, '_tstat'):
        self._tstat = {'dq': 0.0, 'vic': 0.0, 'bad': 0, 'n': 0, 'sof_dt': 0.0, 'sof_n': 0, 'last_sof': None, 'eagain': 0}
        self._tstat_reset = dict(self._tstat)
      self._tstat['dq'] += _t1 - _t0
      self._tstat['vic'] += _t2 - _t1
      self._tstat['bad'] += _bad
      self._tstat['eagain'] += _eagain
      self._tstat['n'] += 1
      if self._tstat['last_sof'] is not None and timestamp_sof > self._tstat['last_sof']:
        self._tstat['sof_dt'] += (timestamp_sof - self._tstat['last_sof']) / 1e9
        self._tstat['sof_n'] += 1
      self._tstat['last_sof'] = timestamp_sof
      if self._tstat['n'] >= 60:
        n = self._tstat['n']
        sof_dt = self._tstat['sof_dt'] / self._tstat['sof_n'] * 1000 if self._tstat['sof_n'] else 0
        sof_fps = 1000.0 / sof_dt if sof_dt > 0 else 0
        line = (f"[TIMING] {self.device} n={n} avg_cycle_dq={self._tstat['dq']/n*1000:.1f}ms "
                f"vic={self._tstat['vic']/n*1000:.1f}ms bad={self._tstat['bad']} eagain={self._tstat['eagain']} "
                f"total={((self._tstat['dq']+self._tstat['vic'])/n)*1000:.1f}ms "
                f"sof_dt={sof_dt:.1f}ms sof_fps={sof_fps:.2f}\n")
        print(line, flush=True)
        try:
          with open('/tmp/vic_timing.log', 'a') as f:
            f.write(line)
        except Exception:
          pass
        self._tstat = dict(self._tstat_reset)

    dmabuf_fd = self._nvbuf_fds[index] if self._use_dmabuf else self.dmabuf_fds[index]
    return dmabuf_fd, data, index, timestamp_sof, timestamp_eof

  def _numpy_downsample(self, raw_data):
    """CPU 回退: 从 VI buffer 裁出真实图像区 (active), 必要时整数倍抽点到目标尺寸"""
    raw = np.frombuffer(raw_data, dtype=np.uint8)
    aw, ah = self.cam_active_w, self.cam_active_h
    rows = raw[:self.cam_h * self.cam_bytesperline].reshape(self.cam_h, self.cam_bytesperline)
    uyvy = np.ascontiguousarray(rows[:ah, :aw * 2])
    if aw == self.target_w and ah == self.target_h:
      return uyvy.tobytes()
    sx = max(1, aw // self.target_w)
    sy = max(1, ah // self.target_h)
    uyvy = uyvy[:, :self.target_w * sx * 2]
    macro = uyvy.reshape(ah, self.target_w * sx // 2, 4)
    out = np.ascontiguousarray(macro[::sy, ::sx])
    return out.reshape(out.shape[0], self.target_w * 2).tobytes()

  def read_frames(self):
    while self.streaming:
      try:
        dmabuf_fd, data, index, timestamp_sof, timestamp_eof = self.read_frame()
        if self._nvbuf_zerocopy or (dmabuf_fd is not None and dmabuf_fd >= 0):
          is_nv12 = self._vic or self.cam_pixelformat == V4L2_PIX_FMT_NV12
          yield VisionBuf(
            dmabuf_fd=dmabuf_fd,
            width=self.W,
            height=self.H,
            stride=self.W if is_nv12 else self.W * 2,
            data_size=self._nv12_size if is_nv12 else self.W * self.H * 2,
            frame_id=self.cur_frame_id,
            pixel_format='NV12' if is_nv12 else self.cam_format_name,
            data=data,
            timestamp_sof=timestamp_sof,
            timestamp_eof=timestamp_eof,
            v4l2_index=index,
          )
        self.cur_frame_id += 1
      except BlockingIOError:
        time.sleep(0.001)

  def close(self):
    if self.fd is None:
      return
    if self.streaming:
      buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE
      fcntl.ioctl(self.fd, VIDIOC_STREAMOFF, struct.pack("I", buf_type))
    self.mmap_ptrs.clear()
    for mm in self.mmap_objs:
      try:
        mm.close()
      except BufferError:
        pass
    self.mmap_objs.clear()
    self._clear_external_dst_cache("camera close")
    for i, dmabuf_fd in enumerate(self.dmabuf_fds):
      if dmabuf_fd >= 0:
        os.close(dmabuf_fd)
        self.dmabuf_fds[i] = -1
    if self._vic and hasattr(self, '_dst_surf'):
      self._nvbuf.NvBufSurfaceDestroy(self._dst_surf)
    self._external_dst_surfs.clear()
    if self.fd is not None:
      os.close(self.fd)
    self.fd = None
    self.streaming = False

  def __del__(self):
    self.close()


class Camera:
  def __init__(self, cam_type_state, stream_type, cam_device):
    self.cam_type_state = cam_type_state
    self.stream_type = stream_type
    self.cam_device = cam_device
    self.cur_frame_id = 0
    self.cam = V4L2Camera(cam_device, width=V4L2Camera.OUTPUT_WIDTH, height=V4L2Camera.OUTPUT_HEIGHT, fps=20)
    self.W = self.cam.W
    self.H = self.cam.H

  def read_frames(self):
    for vision_buf in self.cam.read_frames():
      yield vision_buf
      self.cur_frame_id = self.cam.cur_frame_id

  def read_frame_to_fd(self, dst_fd):
    return self.cam.read_frame_to_fd(dst_fd)

  def get_dmabuf_fd(self):
    return self.cam.dmabuf_fds[0] if self.cam.dmabuf_fds else -1

  def get_pitch(self):
    return self.cam.W * 2

  def get_data_size(self):
    return self.cam.W * self.cam.H * 2
