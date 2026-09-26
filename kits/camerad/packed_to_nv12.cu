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

// ============ resize + flip 版 (cp 链路: 源 1920x1080 UYVY -> 目标任意尺寸 NV12) ============
// src: packed UYVY 4:2:2, 每宏像素 4 字节 [Y U Y V] (chroma_swap=0) 或 [Y V Y U] (chroma_swap=1)
// dst: NV12 dst_w x dst_h
// flip: 1 = 180 度翻转 (水平+垂直镜像, cp 相机倒装用)
// 采样: Y/U/V 各自双线性插值
__device__ __forceinline__ float _clampf(float v, float lo, float hi) {
  return fminf(fmaxf(v, lo), hi);
}

__device__ __forceinline__ uint8_t _bilinear_y(const uint8_t* src, int sw, int sh,
                                               float sx, float sy) {
  int stride = sw * 2;
  int x0 = (int)floorf(sx), y0 = (int)floorf(sy);
  x0 = max(0, min(x0, sw - 1));
  y0 = max(0, min(y0, sh - 1));
  int x1 = min(x0 + 1, sw - 1), y1 = min(y0 + 1, sh - 1);
  float fx = sx - x0, fy = sy - y0;
  float v00 = (float)src[y0 * stride + x0 * 2];
  float v10 = (float)src[y0 * stride + x1 * 2];
  float v01 = (float)src[y1 * stride + x0 * 2];
  float v11 = (float)src[y1 * stride + x1 * 2];
  float top = v00 * (1.0f - fx) + v10 * fx;
  float bot = v01 * (1.0f - fx) + v11 * fx;
  return (uint8_t)(top * (1.0f - fy) + bot * fy + 0.5f);
}

// sx 单位 = 宏像素列 (0 .. sw/2-1), sy 单位 = 行 (0 .. sh-1); pick = 0 -> U, 1 -> V
__device__ __forceinline__ uint8_t _bilinear_chroma(const uint8_t* src, int sw, int sh,
                                                    float sx, float sy, int pick, int chroma_swap) {
  int half_w = sw / 2;
  int stride = sw * 2;
  int x0 = (int)floorf(sx), y0 = (int)floorf(sy);
  x0 = max(0, min(x0, half_w - 1));
  y0 = max(0, min(y0, sh - 1));
  int x1 = min(x0 + 1, half_w - 1), y1 = min(y0 + 1, sh - 1);
  float fx = sx - x0, fy = sy - y0;
  int uoff = chroma_swap ? 3 : 1;
  int voff = chroma_swap ? 1 : 3;
  int off = pick == 0 ? uoff : voff;
  float v00 = (float)src[y0 * stride + x0 * 4 + off];
  float v10 = (float)src[y0 * stride + x1 * 4 + off];
  float v01 = (float)src[y1 * stride + x0 * 4 + off];
  float v11 = (float)src[y1 * stride + x1 * 4 + off];
  float top = v00 * (1.0f - fx) + v10 * fx;
  float bot = v01 * (1.0f - fx) + v11 * fx;
  return (uint8_t)(top * (1.0f - fy) + bot * fy + 0.5f);
}

__global__ void packed_to_nv12_resize_kernel(const uint8_t* __restrict__ src,
                                             uint8_t* __restrict__ dst,
                                             int sw, int sh, int dw, int dh,
                                             int flip, int chroma_swap,
                                             int dst_stride, int y_plane_rows) {
  int dx = blockIdx.x * blockDim.x + threadIdx.x;
  int dy = blockIdx.y * blockDim.y + threadIdx.y;
  if (dx >= dw || dy >= dh) return;

  // Y: 源坐标 (未翻转), 双线性
  float sx = (dx + 0.5f) * (float)sw / (float)dw - 0.5f;
  float sy = (dy + 0.5f) * (float)sh / (float)dh - 0.5f;
  if (flip) { sx = (float)(sw - 1) - sx; sy = (float)(sh - 1) - sy; }
  dst[dy * dst_stride + dx] = _bilinear_y(src, sw, sh, sx, sy);

  // UV: 偶数输出行/列各算一对
  if ((dy & 1) == 0 && (dx & 1) == 0) {
    int ux = dx >> 1;            // 输出 U 网格列 (0 .. dw/2-1)
    int uy = dy >> 1;            // 输出 U 网格行 (0 .. dh/2-1)
    float ux_src = (ux + 0.5f) * (float)(sw / 2) / (float)(dw / 2) - 0.5f;
    float uy_src = (uy + 0.5f) * (float)sh / (float)(dh / 2) - 0.5f;
    if (flip) { ux_src = (float)(sw / 2 - 1) - ux_src; uy_src = (float)(sh - 1) - uy_src; }
    uint8_t u = _bilinear_chroma(src, sw, sh, ux_src, uy_src, 0, chroma_swap);
    uint8_t v = _bilinear_chroma(src, sw, sh, ux_src, uy_src, 1, chroma_swap);
    // uv 平面偏移 = dst_stride * y_plane_rows (对齐时 y_plane_rows = align(dh,32), 紧密时 = dh)
    uint8_t* uv = dst + (size_t)dst_stride * y_plane_rows + (size_t)uy * dst_stride;
    uv[ux * 2] = u;
    uv[ux * 2 + 1] = v;
  }
}

// resize+flip 零拷贝: src 设备指针 (NvBufSurface import) -> dst 设备指针 (VisionIPC)
// src 尺寸 sw x sh (packed UYVY, sw*sh*2 字节), dst 尺寸 dw x dh (NV12)
// dst_stride: 输出行距 (紧密=dw, Venus 对齐=align(dw,128)); y_plane_rows: y 平面行数 (紧密=dh, 对齐=align(dh,32))
extern "C" int packed_to_nv12_resize_device_to_device(const uint8_t* src_device, uint8_t* dst_device,
                                                      int sw, int sh, int dw, int dh,
                                                      int flip, int chroma_swap,
                                                      int dst_stride, int y_plane_rows) {
  dim3 block(32, 8);
  dim3 grid((dw + 31) / 32, (dh + 7) / 8);
  packed_to_nv12_resize_kernel<<<grid, block>>>(src_device, dst_device, sw, sh, dw, dh,
                                                flip ? 1 : 0, chroma_swap ? 1 : 0,
                                                dst_stride, y_plane_rows);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return 6;
  err = cudaDeviceSynchronize();
  return err == cudaSuccess ? 0 : 7;
}

// host 版 resize: packed (sw x sh) -> NV12 (dw x dh), 紧密输出 (dst_stride=dw, y_plane_rows=dh)
// 供 camerad.py convert() 在源采集尺寸 != 输出尺寸时使用 (master-c3 链路: 1920x1080 -> 1344x760)
// 修复: 原 convert() 用输出尺寸读源 packed 导致行距错位花屏
extern "C" int packed_to_nv12_resize(const uint8_t* src_host, uint8_t* dst_host,
                                     int sw, int sh, int dw, int dh,
                                     int flip, int chroma_swap) {
  size_t src_bytes = (size_t)sw * sh * 2;
  size_t dst_bytes = (size_t)dw * dh * 3 / 2;
  uint8_t *d_src = nullptr, *d_dst = nullptr;
  if (cudaMalloc(&d_src, src_bytes) != cudaSuccess) return 1;
  if (cudaMalloc(&d_dst, dst_bytes) != cudaSuccess) { cudaFree(d_src); return 2; }
  if (cudaMemcpy(d_src, src_host, src_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(d_src); cudaFree(d_dst); return 3;
  }
  dim3 block(32, 8);
  dim3 grid((dw + 31) / 32, (dh + 7) / 8);
  packed_to_nv12_resize_kernel<<<grid, block>>>(d_src, d_dst, sw, sh, dw, dh,
                                                flip ? 1 : 0, chroma_swap ? 1 : 0,
                                                dw, dh);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) { cudaFree(d_src); cudaFree(d_dst); return 6; }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) { cudaFree(d_src); cudaFree(d_dst); return 7; }
  if (cudaMemcpy(dst_host, d_dst, dst_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
    cudaFree(d_src); cudaFree(d_dst); return 5;
  }
  cudaFree(d_src); cudaFree(d_dst);
  return 0;
}
