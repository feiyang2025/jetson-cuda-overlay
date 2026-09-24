# jetson-cuda-overlay

AGX Orin CUDA/TensorRT + V4L2/VIC camera overlay for **sunnypilot / openpilot**
forks (old and new layout). Lets CP / SP / DP / FP / master-c3 reuse the same
CUDA inference backend and GMSL camera path on AGX Orin.

> **New here?** Read **[FEATURES.md](FEATURES.md)** — what each component can do,
> what it can NOT do, and the exact command for every task (apply / calibrate /
> models / IMU / Panda unification / health check).
> **[USAGE.md](USAGE.md)** is the longer manual.

## One-command entry points

```bash
bash apply_cuda.sh <fork-root>          # apply everything (idempotent)
bash panda/install.sh                   # install the cross-fork Panda tooling + start-up hooks
bash tools/self_check.sh <fork-root>    # camerad/modeld kit 完整性自检 (PASS/FAIL 清单)
python3 doctor.py                       # health-check the whole chain
```

## Kit structure (2026-09-24, sp 实车验证版)

采集→推理链路按"功能"打包成 kit, 与 fork 树路径解耦, 任何分支 apply 即用:

| kit | 内容 | 对应契约文档 |
|---|---|---|
| `kits/camerad/` | camerad.py (跨分支自适应版) + v4l2_dmabuf_camera.py + v4l2_camera.py + camera_cuda.py + packed_to_nv12.cu + `msgq/0001-visionipc-zerocopy.patch` (write_and_send + refcount) | `kits/camerad/CAMERAD_CONTRACT.md` |
| `kits/modeld/` | modeld.py + modeld_bigcombo.py (TRT 加载兜底闭环: 重试→降级 tinygrad / BigCombo→FiletOFish) + tensorrt_runner.py + cuda_transform.{cu,h} + `sconstruct_jetson.diff` | `kits/modeld/MODELD_CONTRACT.md` |
| `patches/` | 本地散改补丁 (git 操作会冲掉的): sp_longitudinal_planner_scc.patch | — |

契约文档钉死"不要再改"的规则: twgmsl 色度 (Y=raw[0::2], U=raw[1::4], V=raw[3::4])、20fps 输出、
20 buffer refcount、零拷贝检测回退、入口 CPU 拷贝硬边界 (videobuf2 CMA 内存)、TRT 兜底参数等。

## What it installs

| component | path | purpose |
|---|---|---|
| gpu_backend | `openpilot/sunnypilot/modeld_v2/gpu_backend/` | CUDA transform + TRT runner + model profiles |
| gpu_model_state | `openpilot/sunnypilot/modeld_v2/gpu_model_state.py` | CUDA-first ModelState (tinygrad fallback) |
| cuda_transform | `openpilot/selfdrive/modeld/transforms/cuda_transform.*` | NV12 -> 12-channel GPU warp kernels |
| tensorrt_runner | `openpilot/selfdrive/modeld/runners/tensorrt_runner.py` | TRT .plan loader (zero-copy) |
| v4l2/cuda camera | `openpilot/system/camerad/webcam/{v4l2_dmabuf_camera.py,camerad.py,packed_to_nv12.cu}` | GMSL IMX390 packed UYVY -> twgmsl 色度归一化 -> CUDA packed→NV12 (VIC 默认禁, 零拷贝可选) |
| ch347 imu | `openpilot/system/sensord/ch347t.cc` + `third_party/ch347/` | USB-I2C LSM6DS3 IMU daemon w/ auto zero-bias calib |
| camera calib | `tools/calib/*` + `patch_calib.py` | FCAM/ECAM intrinsic + ECAM extrinsic calibration toolkit and its Param plumbing |
| panda unify | `panda/*` (installed to `/data/openpilot/panda_版本核对/`) | keep every fork on one Panda firmware/protocol -> no reflashing |

## Camera calibration toolkit (ported from the primary tree)

`apply_cuda.sh` copies `tools/calib/` (log-based, no hardcoded paths) and runs
`patch_calib.py`, which wires in the plumbing those tools need:

- `common/transformations/camera.py`: parameterised intrinsics
  (`_read_calib_from_params` / `_camera_config_from_params`) and the AGX Orin GMSL
  entry for `("pc", "unknown")` — road fl 961.76 / wide fl 1740.0 @1920x1080 as the
  first-run default, live values live in the `FcamIntrinsics` / `EcamIntrinsics`
  Params (resolution-keyed JSON). Override size with
  `ROAD_CAM_WIDTH/HEIGHT` / `WIDE_CAM_WIDTH/HEIGHT`.
- `common/params_keys.h`: adds `FcamIntrinsics`, `EcamIntrinsics`, `FcamCalibResult`,
  `WideCalibResult`, `PendingCalibReset`, `WideCalibActive`, `WideCalibMode`,
  `FcamLiveActive`, `WideCalibIntrinsics{Fcam,Ecam}Backup`, in whatever syntax the
  fork uses (`{"K", FLAGS}` or `{"K", {FLAGS, TYPE}}`). `Params.check_key()` rejects
  unknown keys, so this has to be compiled in.
- `tools/calib/*.py`: the originals hardcoded the author's log directory; the patcher
  rewrites the `--base-dir` default to `Paths.log_root()` (this device's realdata).

Because `params_keys.h` is compiled, rebuild the params module **in the target tree**:

```bash
source .venv/bin/activate && scons -j8 common/
```

Tools — run with the project venv (only `wide_calibrator.py` needs `cv2`, which the
system python3 has while some fork venvs do not):

| tool | what | needs |
|---|---|---|
| `self_calibrator.py --scan` | FCAM intrinsics from drive logs | logs |
| `self_calibrator.py --live` / `--ecam` | same, live collection | running stack |
| `ecam_analyze.py` | split FCAM/ECAM frames, estimate ECAM fl | logs |
| `estimate_extrinsics.py` | ECAM extrinsics (pitch/height) from radar-vision mismatch | logs |
| `plot_calib.py` | three-method comparison plot (+ lane-width correction) | logs, matplotlib |
| `wide_calibrator.py` | ECAM intrinsics via live stereo matching | live stack + cv2 |

Verified on this rig 2026-09-12 (installed into the `ajouatom` tree): Param plumbing OK
(`FcamIntrinsics` = 961.76 / `EcamIntrinsics` = 1740.0 as 1920x1080 JSON, readable via
`Params().get`) and `estimate_extrinsics.py` ran over 150 real segments
(pitch 5.2° down, RMSE 4.98 m). Note `estimate_extrinsics.py` / `ecam_analyze.py` only
scan `--dir` (or `--route` together with `--base-dir`), so point them at a directory
whose subdirs each contain `rlog.zst`.

## CH347 IMU (optional, auto-exits if absent)

The overlay registers `sensord_ch347` as an optional daemon. It:
- talks to an external CH347 USB-I2C (LSM6DS3) break-out board — useful on AGX
  Orin / PCs that have no on-board IMU;
- publishes `accelerometer` / `gyroscope` / `temperatureSensor` cereal messages,
  so the project's own `imu_calibrationd` / `calibrationd` can consume them;
- loads the full calibration from `imu_calibration.json` (or `IMU_CALIB_JSON`)
  and applies it on every sample:
  `gyro_rad = raw_rad - imuBiasGyro * pi/180` and
  `accel_ms2 = imuCalibMatrix @ (raw_mps2 - 9.81 * imuBiasAccel)`
  (schema: `imuBiasGyro` in deg/s, `imuBiasAccel` in g, `imuCalibMatrix`
  row-major 3x3 — compatible with the sunnypilot-cuda `tools/imu_calib`
  multi-pose ellipsoid calibrator);
- runs a **boot-time auto zero-rate bias calibration**: skips the first second,
  collects ~5 s, and if the gyro magnitude 1-sigma is < 0.015 rad/s (stationary)
  it updates `imuBiasGyro` only — a pre-existing `imuBiasAccel` / `imuCalibMatrix`
  from a multi-pose calibration is preserved, never overwritten;
- if no CH347 device (`/dev/ch34x_pis*` / `/dev/ttyACM*`) is present it exits
  cleanly and does not affect the system.

`run_ch347t.sh` builds `ch347t` on first run (needs `g++` + `libzmq` + `capnp`
dev packages), then execs it. On-board sensors (comma-style I2C LSM6DS3) keep
working via the project's stock `sensord`. The Python fallback `ch347t.py`
mirrors the same calibration load/apply path.

## usage (any project)

```bash
# first time or after upstream update:
cd <target-project-root>
bash /path/to/jetson-cuda-overlay/apply_cuda.sh .

# repeat after every `git pull upstream`
```

`apply_cuda.sh` is **idempotent** and version-independent: it copies the overlay
files and runs `patch_modeld.py` + `patch_camerad.py` + `patch_ch347_manager.py`
+ `patch_calib.py` to wire in `_make_model()` (CUDA-first, tinygrad fallback),
prefer the V4L2/VIC camera on Linux, register the optional CH347 daemon, and set
up the camera-calibration toolkit (tools + intrinsics plumbing + Param keys).

After apply, rebuild the compiled Param key list once:

```bash
source .venv/bin/activate && scons -j8 common/
```

**See [USAGE.md](USAGE.md)** for what each component can do, the exact commands,
the on-device steps, the Panda firmware/protocol unification workflow, and the
honest list of what is NOT included.

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
