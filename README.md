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

## AGX Orin validation checklist

Run the following in order on the device after applying the overlay:

1. Confirm the target project starts normally with the original backend:
   `DISABLE_CUDA_BACKEND=1`.
2. Build `libcuda_transform.so` and verify it loads without CUDA errors.
3. Build `ch347t` and confirm `accelerometer`, `gyroscope`, and
   `temperatureSensor` messages when the CH347 device is connected.
4. Confirm the boot gyro calibration either updates or safely retains
   `imu_calibration.json` when the device is moving.
5. Confirm both cameras open through V4L2 and report the expected active image
   size, pixel format, stride, and NV12 output.
6. Confirm road/wide frame timestamps, frame IDs, pair delta, and stable 20 Hz.
7. Test FiletOFish split engines and BigCombo merged engine separately.
8. Confirm TensorRT inputs use GPU pointers for image tensors and that output is
   finite and physically reasonable.
9. Confirm modelV2, cameraOdometry, radard, controlsd, Panda/CAN, and UI remain
   healthy for a stationary test before any driving test.
10. Run a short controlled drive and inspect frame drops, model execution time,
    camera quality, IMU values, radar lead data, and disengagement events.

Expected steady-state targets:

```text
road camera: 20 Hz
wide camera: 20 Hz
modelV2: 20 Hz
frame drops: 0 during steady state
CUDA transform: no illegal access or non-finite output
TensorRT: target Orin engine, no silent CPU model fallback
```

## Update and repair workflow

The overlay is maintained separately from each target project. The normal loop is:

```bash
# On the device: update the target project
cd <target-project-root>
git fetch upstream
git merge upstream/<target-branch>

# Reapply the shared hardware layer
bash /path/to/jetson-cuda-overlay/apply_cuda.sh .

# Rebuild device-specific artifacts and run the validation checklist
bash openpilot/sunnypilot/modeld_v2/gpu_backend/deploy.sh
```

When a device test exposes a problem:

```text
1. Save the exact project commit, overlay commit, command, and complete log.
2. Report the failing stage and error without deleting the working tree.
3. Fix the corresponding file in jetson-cuda-overlay, not by editing every fork.
4. Run the offline syntax/logic checks and review the diff.
5. Commit and push a new overlay commit.
6. On the device, pull the new overlay and rerun apply_cuda.sh.
7. Rebuild affected .so/engine artifacts and repeat the checklist.
```

Device-specific `.plan`, `.so`, calibration files, logs, and generated metadata
must not be committed to the overlay. Keep them on the AGX Orin or in a separate
artifact store. If an upstream update changes a model interface, update the
profile/metadata handling and regenerate the engine on the target Orin.
