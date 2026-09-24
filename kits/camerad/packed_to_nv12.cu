// packed GMSL (twgmsl) -> NV12
// 已用画面验证的取值, 与 tools/webcam/camerad.py CPU 路径一致:
//   Y[i]     = raw[i * 2]
//   U[row,c] = raw[row * stride + c * 4 + 1]   (仅偶数行)
//   V[row,c] = raw[row * stride + c * 4 + 3]
// raw 布局: 每行 width*2 字节, 每 4 字节一个宏像素 [Y U Y V]

#include <cuda_runtime.h>
#include <cstdint>

__global__ void packed_to_nv12_kernel(const uint8_t* __restrict__ src,
                                      uint8_t* __restrict__ dst,
                                      int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  int stride = width * 2;
  const uint8_t* row = src + y * stride;

  dst[y * width + x] = row[x * 2];

  if ((y & 1) == 0 && (x & 1) == 0) {
    int c = x >> 1;
    uint8_t* uv = dst + width * height + (y >> 1) * width;
    uv[c * 2]     = row[c * 4 + 1];
    uv[c * 2 + 1] = row[c * 4 + 3];
  }
}

extern "C" int packed_to_nv12(const uint8_t* src_host, uint8_t* dst_host,
                              int width, int height) {
  size_t src_bytes = (size_t)width * height * 2;
  size_t dst_bytes = (size_t)width * height * 3 / 2;

  uint8_t *d_src = nullptr, *d_dst = nullptr;
  if (cudaMalloc(&d_src, src_bytes) != cudaSuccess) return 1;
  if (cudaMalloc(&d_dst, dst_bytes) != cudaSuccess) { cudaFree(d_src); return 2; }

  if (cudaMemcpy(d_src, src_host, src_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(d_src); cudaFree(d_dst); return 3;
  }

  dim3 block(32, 8);
  dim3 grid((width + 31) / 32, (height + 7) / 8);
  packed_to_nv12_kernel<<<grid, block>>>(d_src, d_dst, width, height);

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) { cudaFree(d_src); cudaFree(d_dst); return 4; }

  if (cudaMemcpy(dst_host, d_dst, dst_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
    cudaFree(d_src); cudaFree(d_dst); return 5;
  }
  cudaFree(d_src);
  cudaFree(d_dst);
  return 0;
}

// 零拷贝版: NV12 直接写到调用方给的设备指针 (VisionIPC buffer 的
// cudaHostGetDevicePointer), 不再分配输出缓冲, 不再 D2H。
// dst_device 由调用方保证容量 >= width*height*3/2 且 GPU 可写。
extern "C" int packed_to_nv12_device(const uint8_t* src_host, uint8_t* dst_device,
                                      int width, int height) {
  size_t src_bytes = (size_t)width * height * 2;
  uint8_t* d_src = nullptr;
  if (cudaMalloc(&d_src, src_bytes) != cudaSuccess) return 1;
  if (cudaMemcpy(d_src, src_host, src_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(d_src); return 3;
  }

  dim3 block(32, 8);
  dim3 grid((width + 31) / 32, (height + 7) / 8);
  packed_to_nv12_kernel<<<grid, block>>>(d_src, dst_device, width, height);

  cudaError_t err = cudaDeviceSynchronize();
  cudaFree(d_src);
  return err == cudaSuccess ? 0 : 4;
}

// 完整零拷贝版: src/dst 都是 CUDA device pointer。
// src_device 来自 NvBufSurface dma-buf 的 CUDA import，dst_device 来自 VisionIPC mapped buffer。
extern "C" int packed_to_nv12_device_to_device(const uint8_t* src_device, uint8_t* dst_device,
                                               int width, int height) {
  dim3 block(32, 8);
  dim3 grid((width + 31) / 32, (height + 7) / 8);
  packed_to_nv12_kernel<<<grid, block>>>(src_device, dst_device, width, height);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return 6;
  err = cudaDeviceSynchronize();
  return err == cudaSuccess ? 0 : 7;
}
