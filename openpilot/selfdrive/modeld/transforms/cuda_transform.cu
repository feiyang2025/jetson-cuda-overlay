#include "cuda_transform.h"

// ---- Constants (matching OpenCL transform.cl) ----
#define INTER_BITS 5
#define INTER_TAB_SIZE (1 << INTER_BITS)
#define INTER_REMAP_COEF_BITS 15
#define INTER_REMAP_COEF_SCALE (1 << INTER_REMAP_COEF_BITS)

// ---- warpPerspective (matches OpenCL transform.cl exactly) ----
// For each output pixel (dx, dy), computes source coordinate via perspective
// transform and bilinear interpolation.
__global__ void warp_perspective_kernel(const uint8_t *src,
    int src_row_stride, int src_px_stride, int src_offset,
    int src_rows, int src_cols,
    uint8_t *dst, int dst_row_stride, int dst_offset,
    int dst_rows, int dst_cols,
    const float *M)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx >= dst_cols || dy >= dst_rows) return;

    float X0 = M[0] * dx + M[1] * dy + M[2];
    float Y0 = M[3] * dx + M[4] * dy + M[5];
    float W = M[6] * dx + M[7] * dy + M[8];
    W = (W != 0.0f) ? (float)INTER_TAB_SIZE / W : 0.0f;
    int X = __float2int_rn(X0 * W), Y = __float2int_rn(Y0 * W);

    int sx = X >> INTER_BITS;
    int sy = Y >> INTER_BITS;

    sx = max(0, min(sx, src_cols - 1));
    int sx_p1 = max(0, min(sx + 1, src_cols - 1));
    sy = max(0, min(sy, src_rows - 1));
    int sy_p1 = max(0, min(sy + 1, src_rows - 1));

    int v0 = src[sy * src_row_stride + src_offset + sx * src_px_stride];
    int v1 = src[sy * src_row_stride + src_offset + sx_p1 * src_px_stride];
    int v2 = src[sy_p1 * src_row_stride + src_offset + sx * src_px_stride];
    int v3 = src[sy_p1 * src_row_stride + src_offset + sx_p1 * src_px_stride];

    short ay = (short)(Y & (INTER_TAB_SIZE - 1));
    short ax = (short)(X & (INTER_TAB_SIZE - 1));
    float taby = (1.0f / INTER_TAB_SIZE) * ay;
    float tabx = (1.0f / INTER_TAB_SIZE) * ax;

    int dst_index = dy * dst_row_stride + dst_offset + dx;

    int itab0 = __float2int_rn((1.0f - taby) * (1.0f - tabx) * INTER_REMAP_COEF_SCALE);
    int itab1 = __float2int_rn((1.0f - taby) * tabx * INTER_REMAP_COEF_SCALE);
    int itab2 = __float2int_rn(taby * (1.0f - tabx) * INTER_REMAP_COEF_SCALE);
    int itab3 = __float2int_rn(taby * tabx * INTER_REMAP_COEF_SCALE);

    int val = v0 * itab0 + v1 * itab1 + v2 * itab2 + v3 * itab3;
    uint8_t pix = (uint8_t)max(0, min(255,
        (val + (1 << (INTER_REMAP_COEF_BITS - 1))) >> INTER_REMAP_COEF_BITS));

    dst[dst_index] = pix;
}

// ---- loadys (matches OpenCL loadys exactly) ----
// Rearranges Y plane (W×H) into 4 sub-planes at (W/2 × H/2) each.
// Output layout: y0, y1, y2, y3, where:
//   y0 = even rows, even cols
//   y1 = odd rows, even cols (only for odd rows in original)
//   y2 = even rows, odd cols
//   y3 = odd rows, odd cols (only for odd rows in original)
__global__ void loadys_kernel(const uint8_t *Y, uint8_t *out, int out_offset,
    int width, int height)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_work = (width * height) / 8;
    if (gid >= total_work) return;

    int uv_size = (width / 2) * (height / 2);

    int ois = gid * 8;
    int oy = ois / width;
    int ox = ois % width;

    // Read 8 consecutive Y values
    uint8_t ys0 = Y[gid * 8 + 0];
    uint8_t ys1 = Y[gid * 8 + 1];
    uint8_t ys2 = Y[gid * 8 + 2];
    uint8_t ys3 = Y[gid * 8 + 3];
    uint8_t ys4 = Y[gid * 8 + 4];
    uint8_t ys5 = Y[gid * 8 + 5];
    uint8_t ys6 = Y[gid * 8 + 6];
    uint8_t ys7 = Y[gid * 8 + 7];

    int half_row = oy / 2;
    int half_col = ox / 2;

    if ((oy & 1) == 0) {
        // Even row → y0 and y2
        uint8_t *outy0 = out + out_offset;
        uint8_t *outy2 = out + out_offset + uv_size * 2;

        outy0[half_row * (width / 2) + half_col + 0] = ys0;
        outy0[half_row * (width / 2) + half_col + 1] = ys2;
        outy0[half_row * (width / 2) + half_col + 2] = ys4;
        outy0[half_row * (width / 2) + half_col + 3] = ys6;

        outy2[half_row * (width / 2) + half_col + 0] = ys1;
        outy2[half_row * (width / 2) + half_col + 1] = ys3;
        outy2[half_row * (width / 2) + half_col + 2] = ys5;
        outy2[half_row * (width / 2) + half_col + 3] = ys7;
    } else {
        // Odd row → y1 and y3
        uint8_t *outy1 = out + out_offset + uv_size;
        uint8_t *outy3 = out + out_offset + uv_size * 3;

        outy1[half_row * (width / 2) + half_col + 0] = ys0;
        outy1[half_row * (width / 2) + half_col + 1] = ys2;
        outy1[half_row * (width / 2) + half_col + 2] = ys4;
        outy1[half_row * (width / 2) + half_col + 3] = ys6;

        outy3[half_row * (width / 2) + half_col + 0] = ys1;
        outy3[half_row * (width / 2) + half_col + 1] = ys3;
        outy3[half_row * (width / 2) + half_col + 2] = ys5;
        outy3[half_row * (width / 2) + half_col + 3] = ys7;
    }
}

// ---- loaduv (matches OpenCL loaduv exactly) ----
// Copies UV data to output after Y planes
__global__ void loaduv_kernel(const uint8_t *in, uint8_t *out, int out_offset,
    int total_work)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_work) return;
    out[gid + out_offset] = in[gid];
}

// ---- copy (matches OpenCL copy exactly) ----
// Copies data between buffers (for temporal ring buffer management)
__global__ void copy_kernel(const uint8_t *in, uint8_t *out,
    int in_offset, int out_offset, int total_work)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_work) return;
    out[gid + out_offset] = in[gid + in_offset];
}

// ---- C++ API ----

extern "C" {

void cuda_transform_init(CUDATransformState *s, int width, int height, int temporal_skip)
{
    s->width = width;
    s->height = height;
    s->temporal_skip = temporal_skip;

    int uv_size = (width / 2) * (height / 2);
    s->frame_size = width * height * 3 / 2;  // MODEL_FRAME_SIZE
    s->buf_size = s->frame_size * 2;

    // Allocate device buffers
    cudaMalloc(&s->d_y, width * height);
    cudaMalloc(&s->d_u, uv_size);
    cudaMalloc(&s->d_v, uv_size);

    // Temporal ring buffer: holds (temporal_skip + 1) frames
    cudaMalloc(&s->d_img_buffer, (temporal_skip + 1) * s->frame_size);

    // Output: 2 frames concatenated
    cudaMalloc(&s->d_output, s->buf_size);

    // Projection matrices
    cudaMalloc(&s->d_proj_y, 9 * sizeof(float));
    cudaMalloc(&s->d_proj_uv, 9 * sizeof(float));

    s->initialized = 1;
}

void cuda_transform_destroy(CUDATransformState *s)
{
    if (!s->initialized) return;
    cudaFree(s->d_y);
    cudaFree(s->d_u);
    cudaFree(s->d_v);
    cudaFree(s->d_img_buffer);
    cudaFree(s->d_output);
    cudaFree(s->d_proj_y);
    cudaFree(s->d_proj_uv);
    s->initialized = 0;
}

uint8_t* cuda_transform_execute(CUDATransformState *s,
                                 const uint8_t *input_nv12,
                                 int frame_width, int frame_height,
                                 int frame_stride, int frame_uv_offset,
                                 const float *projection)
{
    int w = s->width, h = s->height;
    int uv_size = (w / 2) * (h / 2);
    int frame_size = s->frame_size;

    // Upload projection matrix
    // Build half-resolution projection for UV
    float proj_uv[9];
    float scale = 0.5f;
    for (int i = 0; i < 3; i++) {
        proj_uv[i*3+0] = projection[i*3+0] * scale;
        proj_uv[i*3+1] = projection[i*3+1] * scale;
        proj_uv[i*3+2] = projection[i*3+2] * scale;
    }
    cudaMemcpy(s->d_proj_y, projection, 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(s->d_proj_uv, proj_uv, 9 * sizeof(float), cudaMemcpyHostToDevice);

    // ---- Step 1: Warp perspective Y ----
    {
        dim3 block(16, 16);
        dim3 grid((w + 15) / 16, (h + 15) / 16);
        warp_perspective_kernel<<<grid, block>>>(
            input_nv12,
            frame_stride,        // src_row_stride
            1,                   // src_px_stride (Y: stride=1)
            0,                   // src_offset
            frame_height,        // src_rows
            frame_width,         // src_cols
            s->d_y,              // dst
            w,                   // dst_row_stride
            0,                   // dst_offset
            h,                   // dst_rows
            w,                   // dst_cols
            s->d_proj_y          // M
        );
    }

    // ---- Step 2: Warp perspective U ----
    {
        dim3 block(16, 16);
        dim3 grid(((w/2) + 15) / 16, ((h/2) + 15) / 16);
        warp_perspective_kernel<<<grid, block>>>(
            input_nv12,
            frame_stride,        // src_row_stride
            2,                   // src_px_stride (UV: stride=2)
            frame_uv_offset,     // src_offset (U offset)
            frame_height / 2,    // src_rows (UV half height)
            frame_width / 2,     // src_cols (UV half width)
            s->d_u,              // dst
            w / 2,               // dst_row_stride
            0,                   // dst_offset
            h / 2,               // dst_rows
            w / 2,               // dst_cols
            s->d_proj_uv         // M
        );
    }

    // ---- Step 3: Warp perspective V ----
    {
        dim3 block(16, 16);
        dim3 grid(((w/2) + 15) / 16, ((h/2) + 15) / 16);
        warp_perspective_kernel<<<grid, block>>>(
            input_nv12,
            frame_stride,        // src_row_stride
            2,                   // src_px_stride
            frame_uv_offset + 1, // src_offset (V offset = U offset + 1)
            frame_height / 2,
            frame_width / 2,
            s->d_v,              // dst
            w / 2,
            0,
            h / 2,
            w / 2,
            s->d_proj_uv
        );
    }

    // ---- Step 4: Shift temporal ring buffer ----
    // Copy d_img_buffer[1..temporal_skip] → d_img_buffer[0..temporal_skip-1]
    for (int i = s->temporal_skip - 1; i >= 0; i--) {
        int total_work = frame_size;
        int block_size = 256;
        int grid_size = (total_work + block_size - 1) / block_size;
        copy_kernel<<<grid_size, block_size>>>(
            s->d_img_buffer, s->d_img_buffer,
            (i + 1) * frame_size,   // src_offset
            i * frame_size,          // dst_offset
            total_work
        );
    }

    // ---- Step 5: LoadYUV (loadys + loaduv) ----
    // Write current frame to d_img_buffer[temporal_skip * frame_size]

    uint8_t *current_out = s->d_img_buffer + s->temporal_skip * frame_size;

    // 5a: loadys (Y → 4 sub-planes)
    {
        int total_work = (w * h) / 8;
        int block_size = 256;
        int grid_size = (total_work + block_size - 1) / block_size;
        loadys_kernel<<<grid_size, block_size>>>(s->d_y, current_out, 0, w, h);
    }

    // 5b: loaduv (U → output after Y sub-planes)
    {
        int y_planes_size = 4 * uv_size;  // 4 Y sub-planes
        int total_work = uv_size;
        int block_size = 256;
        int grid_size = (total_work + block_size - 1) / block_size;
        loaduv_kernel<<<grid_size, block_size>>>(s->d_u, current_out, y_planes_size, total_work);
    }

    // 5c: loaduv (V → output after U)
    {
        int y_planes_size = 4 * uv_size;
        int total_work = uv_size;
        int block_size = 256;
        int grid_size = (total_work + block_size - 1) / block_size;
        loaduv_kernel<<<grid_size, block_size>>>(s->d_v, current_out, y_planes_size + uv_size, total_work);
    }

    // ---- Step 6: Copy previous frame to output ----
    // img_buffer[0] is the previous frame (already shifted)
    {
        int total_work = frame_size;
        int block_size = 256;
        int grid_size = (total_work + block_size - 1) / block_size;
        copy_kernel<<<grid_size, block_size>>>(
            s->d_img_buffer, s->d_output,
            0,            // src_offset (frame 0 = previous frame)
            0,            // dst_offset
            total_work
        );
    }

    // ---- Step 7: Copy current frame to output ----
    {
        int total_work = frame_size;
        int block_size = 256;
        int grid_size = (total_work + block_size - 1) / block_size;
        copy_kernel<<<grid_size, block_size>>>(
            s->d_img_buffer, s->d_output,
            s->temporal_skip * frame_size,  // src_offset (current frame)
            frame_size,                     // dst_offset (second half of output)
            total_work
        );
    }

    cudaDeviceSynchronize();
    return s->d_output;
}

uint8_t* cuda_transform_get_output(CUDATransformState *s)
{
    return s->d_output;
}

} // extern "C"
