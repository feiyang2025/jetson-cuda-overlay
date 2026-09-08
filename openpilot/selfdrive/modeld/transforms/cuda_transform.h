#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// All state for the CUDA transform pipeline
typedef struct {
  // Device buffers for Y/U/V transform outputs
  uint8_t *d_y, *d_u, *d_v;
  // Temporal ring buffer (holds previous frames)
  uint8_t *d_img_buffer;
  // Output buffer (2 frames concatenated, same layout as OpenCL input_frames_cl)
  uint8_t *d_output;
  // Device copy of projection matrix
  float *d_proj_y, *d_proj_uv;

  int width, height;           // Model width/height (e.g. 512x256)
  int temporal_skip;           // Number of temporal skip frames
  int frame_size;              // MODEL_FRAME_SIZE
  int buf_size;                // frame_size * 2

  // Whether the state has been initialized
  int initialized;
} CUDATransformState;

// Initialize CUDA transform state
void cuda_transform_init(CUDATransformState *s, int width, int height, int temporal_skip);

// Destroy CUDA transform state and free all GPU memory
void cuda_transform_destroy(CUDATransformState *s);

// Execute the full transform pipeline:
//   input_nv12_device - CUDA device pointer to NV12 frame data (or host registered ptr)
//   frame_width, frame_height, frame_stride, frame_uv_offset - source frame params
//   projection - 3x3 perspective transform matrix
//
// After this call:
//   s->d_output contains 2 temporal frames in model input format
//   Get the device pointer via cuda_transform_get_output(s)
uint8_t* cuda_transform_execute(CUDATransformState *s,
                                 const uint8_t *input_nv12,
                                 int frame_width, int frame_height,
                                 int frame_stride, int frame_uv_offset,
                                 const float *projection);

// Get the output buffer device pointer (for from_blob)
uint8_t* cuda_transform_get_output(CUDATransformState *s);

#ifdef __cplusplus
}
#endif
