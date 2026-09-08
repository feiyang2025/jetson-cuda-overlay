import ctypes
import os

LIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'selfdrive', 'modeld', 'transforms')

class CudaJpegDecoder:
  """JPEG -> NV12 decoder on GPU using nvJPEG + RGB->NV12 CUDA kernel."""

  def __init__(self):
    self._lib = None
    self._state = None
    self._cu = None
    self._tg_ctx_raw = None

  def _ensure_cuda(self):
    try:
      if not self._cu:
        self._cu = ctypes.CDLL('libcuda.so.1')
      from tinygrad.device import Device
      dev = Device['CUDA']
      self._tg_ctx_raw = ctypes.c_void_p(ctypes.addressof(dev.context.contents))
      self._cu.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
      self._cu.cuCtxSetCurrent.restype = ctypes.c_int
      self._cu.cuCtxSetCurrent(self._tg_ctx_raw)
      return True
    except Exception:
      return False

  def create(self, width, height) -> bool:
    lib_path = os.path.join(LIB_DIR, 'libcuda_jpeg_decoder.so')
    if not os.path.exists(lib_path):
      return False
    self._lib = ctypes.CDLL(lib_path)
    if not self._ensure_cuda():
      return False
    self._lib.jpeg_decoder_init.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    self._lib.jpeg_decoder_init.restype = None
    self._lib.jpeg_decoder_set_output.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    self._lib.jpeg_decoder_set_output.restype = None
    self._lib.jpeg_decoder_decode.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    self._lib.jpeg_decoder_decode.restype = ctypes.c_void_p
    self._lib.jpeg_decoder_destroy.argtypes = [ctypes.c_void_p]
    self._lib.jpeg_decoder_destroy.restype = None

    self._state = ctypes.create_string_buffer(128)
    self._lib.jpeg_decoder_init(ctypes.byref(self._state), width, height)
    self._width = width
    self._height = height
    self._frame_size = width * height * 3 // 2
    return True

  def set_external_output(self, device_ptr, size):
    """Redirect decode output to an externally-owned GPU buffer (e.g. CUDA IPC)."""
    self._lib.jpeg_decoder_set_output(ctypes.byref(self._state),
                                      ctypes.c_void_p(device_ptr), size)
    self._frame_size = size

  def decode_to_device(self, jpeg_bytes: bytes):
    """Decode JPEG to NV12 on GPU. Returns CUDA device pointer (int)."""
    self._ensure_cuda()
    buf = ctypes.create_string_buffer(jpeg_bytes, len(jpeg_bytes))
    dev_ptr = self._lib.jpeg_decoder_decode(
      ctypes.byref(self._state),
      buf, len(jpeg_bytes)
    )
    if not dev_ptr:
      return None
    # Synchronize GPU to ensure nvJPEG decode + RGB->NV12 kernel complete
    self._cu.cuCtxSynchronize.restype = ctypes.c_int
    self._cu.cuCtxSynchronize()
    return dev_ptr

  def copy_to_host(self, dev_ptr):
    """Copy decoded NV12 from GPU to CPU bytes. dev_ptr from decode_to_device."""
    out = ctypes.create_string_buffer(self._frame_size)
    self._cu.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    self._cu.cuMemcpyDtoH_v2.restype = ctypes.c_int
    self._cu.cuMemcpyDtoH_v2(out, ctypes.c_void_p(dev_ptr), self._frame_size)
    return out.raw

  def decode_to_host(self, jpeg_bytes: bytes):
    """Decode JPEG to NV12 on GPU, copy result to CPU bytes."""
    dev_ptr = self.decode_to_device(jpeg_bytes)
    if not dev_ptr:
      return None
    return self.copy_to_host(dev_ptr)

  def close(self):
    if self._lib and self._state:
      self._ensure_cuda()
      self._lib.jpeg_decoder_destroy(ctypes.byref(self._state))
