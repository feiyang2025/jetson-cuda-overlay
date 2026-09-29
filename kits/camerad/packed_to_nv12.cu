// packed GMSL (twgmsl) -> NV12
//
// 2026-09-28 抓帧实测 (G_FMT) 发现两个相机节点的 packed 布局并不相同:
//   /dev/video0 (road) 1920x1080 pf=0x56595559 = YUYV  stride=3840 bytesused=4147200
//   /dev/video1 (wide) 1920x1080 pf=0x59565955 = UYVY  stride=3840 bytesused=4147200
// 两者都是 4:2:2 宏像素 4 字节, 只是字节序不同, 所以 layout 必须由调用方按
// 实际 G_FMT 传进来, 不能写死一种。
//
// layout=0 YUYV: [Y0 U0 Y1 V0]  亮度在每 2 字节对的第 0 个, U/V 在宏像素的 1/3
// layout=1 UYVY: [U0 Y0 V0 Y1]  亮度在每 2 字节对的第 1 个, U/V 在宏像素的 0/2
//
// 判据 (video0 抓帧, YUYV): Y 均值 85.2 std 31.7; U 均值 128.5 std 3.6 范围 104..145;
// V 均值 130.1 std 2.8 范围 115..155。色度 std 明显低于亮度且均值贴 128 = 真实 4:2:2
// 色度下采样的特征。若按 16bit LE 解读, 均值 33183 / min 26740, 数值上不可能是亮度。
//
// 历史坑: 曾误判成 "16bit LE、亮度在奇数字节、无真实色度", 把 U/V 写死 128 ——
// 结果 road 还能看但 wide 全灰或全绿。现在按 layout 读真实色度。

#include <cuda_runtime.h>
#include <cstdint>

#define LAYOUT_YUYV 0
#define LAYOUT_UYVY 1

// 亮度字节在 2 字节对内的偏移
__device__ __forceinline__ int _y_off(int layout) { return layout == LAYOUT_UYVY ? 1 : 0; }
// U/V 在 4 字节宏像素内的偏移 (chroma_swap 只交换 U/V, 不动亮度)
//   YUYV [Y0 U0 Y1 V0]: U=1 V=3   UYVY [U0 Y0 V0 Y1]: U=0 V=2
//   chroma_swap=1 时两者互换
__device__ __forceinline__ int _u_off(int layout, int chroma_swap) {
  if (chroma_swap) return layout == LAYOUT_UYVY ? 2 : 3;
  return layout == LAYOUT_UYVY ? 0 : 1;
}
__device__ __forceinline__ int _v_off(int layout, int chroma_swap) {
  if (chroma_swap) return layout == LAYOUT_UYVY ? 0 : 1;
  return layout == LAYOUT_UYVY ? 2 : 3;
}

__global__ void packed_to_nv12_kernel(const uint8_t* __restrict__ src,
                                      uint8_t* __restrict__ dst,
                                      int width, int height, int layout, int src_stride) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  // src_stride: 源 buffer 实际行距。紧凑 1080p = width*2 = 3840;
  // 4K 虚标画布 = 7680 (有效 width 像素在每行行首, 其余是零填充)。
  int stride = src_stride > 0 ? src_stride : width * 2;
  int yo = _y_off(layout);
  const uint8_t* row = src + y * stride;

  dst[y * width + x] = row[x * 2 + yo];

  if ((y & 1) == 0 && (x & 1) == 0) {
    int c = x >> 1;
    uint8_t* uv = dst + width * height + (y >> 1) * width;
    uv[c * 2]     = row[c * 4 + _u_off(layout, 0)];
    uv[c * 2 + 1] = row[c * 4 + _v_off(layout, 0)];
  }
}

extern "C" int packed_to_nv12(const uint8_t* src_host, uint8_t* dst_host,
                              int width, int height, int layout, int src_stride) {
  size_t src_bytes = (size_t)(src_stride > 0 ? src_stride : width * 2) * height;
  size_t dst_bytes = (size_t)width * height * 3 / 2;

  uint8_t *d_src = nullptr, *d_dst = nullptr;
  if (cudaMalloc(&d_src, src_bytes) != cudaSuccess) return 1;
  if (cudaMalloc(&d_dst, dst_bytes) != cudaSuccess) { cudaFree(d_src); return 2; }

  if (cudaMemcpy(d_src, src_host, src_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(d_src); cudaFree(d_dst); return 3;
  }

  dim3 block(32, 8);
  dim3 grid((width + 31) / 32, (height + 7) / 8);
  packed_to_nv12_kernel<<<grid, block>>>(d_src, d_dst, width, height, layout, src_stride);

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
                                      int width, int height, int layout, int src_stride) {
  size_t src_bytes = (size_t)(src_stride > 0 ? src_stride : width * 2) * height;
  uint8_t* d_src = nullptr;
  if (cudaMalloc(&d_src, src_bytes) != cudaSuccess) return 1;
  if (cudaMemcpy(d_src, src_host, src_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(d_src); return 3;
  }

  dim3 block(32, 8);
  dim3 grid((width + 31) / 32, (height + 7) / 8);
  packed_to_nv12_kernel<<<grid, block>>>(d_src, dst_device, width, height, layout, src_stride);

  cudaError_t err = cudaDeviceSynchronize();
  cudaFree(d_src);
  return err == cudaSuccess ? 0 : 4;
}

// 完整零拷贝版: src/dst 都是 CUDA device pointer。
// src_device 来自 NvBufSurface dma-buf 的 CUDA import，dst_device 来自 VisionIPC mapped buffer。
extern "C" int packed_to_nv12_device_to_device(const uint8_t* src_device, uint8_t* dst_device,
                                               int width, int height, int layout, int src_stride) {
  dim3 block(32, 8);
  dim3 grid((width + 31) / 32, (height + 7) / 8);
  packed_to_nv12_kernel<<<grid, block>>>(src_device, dst_device, width, height, layout, src_stride);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return 6;
  err = cudaDeviceSynchronize();
  return err == cudaSuccess ? 0 : 7;
}

// ============ resize + flip 版 (cp 链路: 源 1920x1080 packed -> 目标任意尺寸 NV12) ============
// src: packed 4:2:2 (YUYV / UYVY 由 layout 决定), sw*sh*2 字节
// dst: NV12 dw x dh
// flip: 1 = 180 度翻转 (水平+垂直镜像, cp 相机倒装用)
// chroma_swap: 1 = U/V 交换
// 采样: Y/U/V 各自双线性插值
// src_stride = 源 buffer 每行字节数。twgmsl 有两种驱动状态:
//   紧凑 1080p: 真实图 1920x1080 整幅有效, stride = 3840
//   4K 虚标画布: buffer 3840x2160 / stride 7680, 真实图只在左上 1920x1080, 其余全零
// 写死 sw*2 只在紧凑状态成立; 4K 状态必须传 7680, 否则读到的行全是零填充
// (Y=0,U=0,V=0 经 BT.601 还原后 R=0 G=135 B=0 = 纯绿, 正是"1/4 画面其余绿色")。
__device__ __forceinline__ uint8_t _bilinear_y(const uint8_t* src, int sw, int sh,
                                               float sx, float sy, int layout, int src_stride) {
  int stride = src_stride > 0 ? src_stride : sw * 2;
  int yo = _y_off(layout);
  int x0 = (int)floorf(sx), y0 = (int)floorf(sy);
  x0 = max(0, min(x0, sw - 1));
  y0 = max(0, min(y0, sh - 1));
  int x1 = min(x0 + 1, sw - 1), y1 = min(y0 + 1, sh - 1);
  float fx = sx - x0, fy = sy - y0;
  float v00 = (float)src[y0 * stride + x0 * 2 + yo];
  float v10 = (float)src[y0 * stride + x1 * 2 + yo];
  float v01 = (float)src[y1 * stride + x0 * 2 + yo];
  float v11 = (float)src[y1 * stride + x1 * 2 + yo];
  float top = v00 * (1.0f - fx) + v10 * fx;
  float bot = v01 * (1.0f - fx) + v11 * fx;
  return (uint8_t)(top * (1.0f - fy) + bot * fy + 0.5f);
}

// sx 单位 = 宏像素列 (0 .. sw/2-1), sy 单位 = 行 (0 .. sh-1); pick = 0 -> U, 1 -> V
__device__ __forceinline__ uint8_t _bilinear_chroma(const uint8_t* src, int sw, int sh,
                                                    float sx, float sy, int pick,
                                                    int layout, int chroma_swap, int src_stride) {
  int half_w = sw / 2;
  int stride = src_stride > 0 ? src_stride : sw * 2;
  int off = pick == 0 ? _u_off(layout, chroma_swap) : _v_off(layout, chroma_swap);
  int x0 = (int)floorf(sx), y0 = (int)floorf(sy);
  x0 = max(0, min(x0, half_w - 1));
  y0 = max(0, min(y0, sh - 1));
  int x1 = min(x0 + 1, half_w - 1), y1 = min(y0 + 1, sh - 1);
  float fx = sx - x0, fy = sy - y0;
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
                                             int flip, int chroma_swap, int layout,
                                             int dst_stride, int y_plane_rows,
                                             int src_stride) {
  int dx = blockIdx.x * blockDim.x + threadIdx.x;
  int dy = blockIdx.y * blockDim.y + threadIdx.y;
  if (dx >= dw || dy >= dh) return;

  // Y: 源坐标 (未翻转), 双线性。src_stride 是源 packed 的实际行距:
  // 紧凑 1080p = 3840, 4K 虚标画布 = 7680 (有效 1920px 在每行行首)。
  float sx = (dx + 0.5f) * (float)sw / (float)dw - 0.5f;
  float sy = (dy + 0.5f) * (float)sh / (float)dh - 0.5f;
  if (flip) { sx = (float)(sw - 1) - sx; sy = (float)(sh - 1) - sy; }
  dst[dy * dst_stride + dx] = _bilinear_y(src, sw, sh, sx, sy, layout, src_stride);

  // UV: 偶数输出行/列各算一对, 走真实色度双线性 (不是中性 128, 否则全灰)
      if ((dy & 1) == 0 && (dx & 1) == 0) {
        int ux = dx >> 1;            // 输出 U 网格列 (0 .. dw/2-1)
        int uy = dy >> 1;            // 输出 U 网格行 (0 .. dh/2-1)
        // 色度样本对应源里第 2*ux / 2*uy 个全分辨率像素, 再折成宏像素坐标
        float csx = ((float)(ux * 2) + 0.5f) * (float)sw / (float)dw - 0.5f;
        float csy = ((float)(uy * 2) + 0.5f) * (float)sh / (float)dh - 0.5f;
        if (flip) { csx = (float)(sw - 1) - csx; csy = (float)(sh - 1) - csy; }
        float macro_x = csx * 0.5f;   // _bilinear_chroma 的 x 单位是宏像素列
        uint8_t u = _bilinear_chroma(src, sw, sh, macro_x, csy, 0, layout, chroma_swap, src_stride);
        uint8_t v = _bilinear_chroma(src, sw, sh, macro_x, csy, 1, layout, chroma_swap, src_stride);
        // uv 平面偏移 = dst_stride * y_plane_rows (对齐时 y_plane_rows = align(dh,32), 紧密时 = dh)
        uint8_t* uv = dst + (size_t)dst_stride * y_plane_rows + (size_t)uy * dst_stride;
        uv[ux * 2] = u;
        uv[ux * 2 + 1] = v;
      }
}

// resize+flip 零拷贝: src 设备指针 (NvBufSurface import) -> dst 设备指针 (VisionIPC)
// src 有效区 sw x sh (packed 4:2:2, 字节序由 layout 定), 行距 src_stride
//   (紧凑 1080p 传 3840; 4K 虚标画布传 7680, 真实图在每行行首)
// dst 尺寸 dw x dh (NV12)
// dst_stride: 输出行距 (紧密=dw, Venus 对齐=align(dw,128)); y_plane_rows: y 平面行数 (紧密=dh, 对齐=align(dh,32))
extern "C" int packed_to_nv12_resize_device_to_device(const uint8_t* src_device, uint8_t* dst_device,
                                                      int sw, int sh, int dw, int dh,
                                                      int flip, int chroma_swap, int layout,
                                                      int dst_stride, int y_plane_rows,
                                                      int src_stride) {
  dim3 block(32, 8);
  dim3 grid((dw + 31) / 32, (dh + 7) / 8);
  packed_to_nv12_resize_kernel<<<grid, block>>>(src_device, dst_device, sw, sh, dw, dh,
                                                flip ? 1 : 0, chroma_swap ? 1 : 0, layout,
                                                dst_stride, y_plane_rows, src_stride);
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
                                     int flip, int chroma_swap, int layout, int src_stride) {
  size_t src_bytes = (size_t)(src_stride > 0 ? src_stride : sw * 2) * sh;
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
                                                flip ? 1 : 0, chroma_swap ? 1 : 0, layout,
                                                dw, dh, src_stride);
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
