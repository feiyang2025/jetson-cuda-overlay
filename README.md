# jetson-cuda-overlay

AGX Orin CUDA/TensorRT + V4L2/VIC camera overlay for **sunnypilot / openpilot**
forks (old and new layout). Lets CP / SP / DP / FP / master-c3 reuse the same
CUDA inference backend and GMSL camera path on AGX Orin.

## What it installs

| Component | Path | Purpose |
|---|---|---|
| gpu_backend | `openpilot/sunnypilot/modeld_v2/gpu_backend/` | CUDA transform + TRT runner + model profiles |
| gpu_model_state | `openpilot/sunnypilot/modeld_v2/gpu_model_state.py` | CUDA-first ModelState (tinygrad fallback) |
| cuda_transform | `openpilot/selfdrive/modeld/transforms/cuda_transform.*` | NV12 -> 12-channel GPU warp kernels |
| tensorrt_runner | `openpilot/selfdrive/modeld/runners/tensorrt_runner.py` | TRT .plan loader (zero-copy) |
| V4L2/VIC camera | `openpilot/system/camerad/webcam/v4l2_dmabuf_camera.py` + `v4l2_camera.py` | GMSL IMX390 UYVY -> VIC -> NV12 |

## Usage (any project)

```bash
# first time or after upstream update:
cd <target-project-root>
bash /path/to/jetson-cuda-overlay/apply_cuda.sh .

# repeat after every `git pull upstream`
```

`apply_cuda.sh` is **idempotent** and version-independent: it copies the overlay
files and runs `patch_modeld.py` + `patch_camerad.py` to wire in `_make_model()`
(CUDA-first, tinygrad fallback) and prefer the V4L2 camera on Linux.

## Models

Both split (FiletOFish vision+policy) and merged (BigCombo single-engine 1.7GB)
are supported via `ModelProfile`. Engine selection is automatic:
`Params Model` -> profile is_ready() -> first ready profile -> tinygrad.

## On-device steps (AGX Orin)

```bash
# 1. CUDA transform library
cd openpilot/selfdrive/modeld/transforms
nvcc -arch=sm_87 -shared -O2 -o libcuda_transform.so cuda_transform.cu -I.

# 2. model metadata (per model dir)
python3 openpilot/selfdrive/modeld/get_model_metadata.py <model.onnx>

# 3. TensorRT engines must exist under openpilot/selfdrive/modeld/models/<Name>/
#    e.g. driving_vision_fp16.plan / driving_policy_fp16.plan (split)
#         driving_supercombo_fp16.plan        (merged/BigCombo)

# 4. run
USE_V4L2_CAMERA=1   # default on Linux
# CUDA backend is on by default; disable with DISABLE_CUDA_BACKEND=1
```

## Requirements (Orin)

- CUDA toolkit (nvcc) + TensorRT
- V4L2 GMSL driver (`tegra-camrtc`) + `libnvbufsurface.so` / `libnvbufsurftransform.so`
- optional: `tw_camera_cfg` for SG2/IMX390 serdes init
