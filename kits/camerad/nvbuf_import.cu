#include <cuda.h>
#include <cstdio>
extern "C" int nvbuf_import_fd(int fd, unsigned long long size, unsigned long long* dev_out) {
  CUdevice dev; CUcontext ctx; CUresult r;
  r = cuInit(0); if (r) return 10 + (int)r;
  r = cuDeviceGet(&dev, 0); if (r) return 20 + (int)r;
  r = cuDevicePrimaryCtxRetain(&ctx, dev); if (r) return 30 + (int)r;
  r = cuCtxSetCurrent(ctx); if (r) return 40 + (int)r;
  CUDA_EXTERNAL_MEMORY_HANDLE_DESC desc = {};
  desc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
  desc.handle.fd = fd;
  desc.size = size;
  CUexternalMemory ext = nullptr;
  r = cuImportExternalMemory(&ext, &desc);
  if (r) return 50 + (int)r;
  CUDA_EXTERNAL_MEMORY_BUFFER_DESC buf = {};
  buf.offset = 0;
  buf.size = size;
  CUdeviceptr ptr = 0;
  r = cuExternalMemoryGetMappedBuffer(&ptr, ext, &buf);
  if (r) return 60 + (int)r;
  *dev_out = (unsigned long long)ptr;
  return 0;
}
