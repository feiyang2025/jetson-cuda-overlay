# jetson-cuda-overlay

AGX Orin CUDA/TensorRT + V4L2/VIC camera overlay for **sunnypilot / openpilot**
forks (old and new layout). Lets CP / SP / DP / FP / master-c3 reuse the same
CUDA inference backend and GMSL camera path on AGX Orin.

## What it installs

| component | path | purpose |
|---|---|---|
| gpu_backend | `openpilot/sunnypilot/modeld_v2/gpu_backend/` | CUDA transform + TRT runner + model profiles |
| gpu_model_state | `openpilot/sunnypilot/modeld_v2/gpu_model_state.py` | CUDA-first ModelState (tinygrad fallback) |
| cuda_transform | `openpilot/selfdrive/modeld/transforms/cuda_transform.*` | NV12 -> 12-channel GPU warp kernels |
| tensorrt_runner | `openpilot/selfdrive/modeld/runners/tensorrt_runner.py` | TRT .plan loader (zero-copy) |
| v4l2/vic camera | `openpilot/system/camerad/webcam/v4l2_dmabuf_camera.py` + `v4l2_camera.py` | GMSL IMX390 UYVY -> VIC -> NV12 |
| ch347 imu | `openpilot/system/sensord/ch347t.cc` + `third_party/ch347/` | USB-I2C LSM6DS3 IMU daemon w/ auto zero-bias calib |

## CH347 IMU (optional, auto-exits if absent)

The overlay registers `sensord_ch347` as an optional daemon. It:
- talks to an external CH347 USB-I2C (LSM6DS3) break-out board — useful on AGX
  Orin / PCs that have no on-board IMU;
- publishes `accelerometer` / `gyroscope` / `temperatureSensor` cereal messages,
  so the project's own `imu_calibrationd` / `calibrationd` can consume them;
- runs a **boot-time auto zero-rate bias calibration**: skips the first second,
  collects ~5 s, and if the gyro magnitude 1-sigma is < 0.015 rad/s (stationary)
  it writes the bias to `imu_calibration.json` (or `IMU_CALIB_JSON`);
- if no CH347 device (`/dev/ch34x_pis*` / `/dev/ttyACM*`) is present it exits
  cleanly and does not affect the system.

`run_ch347t.sh` builds `ch347t` on first run (needs `g++` + `libzmq` + `capnp`
dev packages), then execs it. On-board sensors (comma-style I2C LSM6DS3) keep
working via the project's stock `sensord`.

## usage (any project)

```bash
# first time or after upstream update:
cd <target-project-root>
bash /path/to/jetson-cuda-overlay/apply_cuda.sh .

# repeat after every `git pull upstream`
```

`apply_cuda.sh` is **idempotent** and version-independent: it copies the overlay
files and runs `patch_modeld.py` + `patch_camerad.py` + `patch_ch347_manager.py`
to wire in `_make_model()` (CUDA-first, tinygrad fallback), prefer the V4L2/VIC
camera on Linux, and register the optional CH347 daemon.

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
