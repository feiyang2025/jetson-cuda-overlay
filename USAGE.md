# 使用说明 —— 组件清单、能做什么、怎么用

本文件说明 `jetson-cuda-overlay` 里每个组件**能实现什么功能**、**怎么用**、**验证口径**。
（README.md 是概览，本文件是操作手册。）

适用硬件：NVIDIA AGX Orin（Ubuntu 24.04 / aarch64 / CUDA 13.2 / TensorRT 10.16.2，
森云 SG2 = IMX390 GMSL 双摄，Panda H7 走 USB-A）。
适用对象：sunnypilot / openpilot 的各类分支（CP / SP / DP / FP / master-c3，新布局或旧布局）。

---

## 0. 一分钟上手

```bash
cd <你的分支仓库根>
bash /path/to/jetson-cuda-overlay/apply_cuda.sh .      # 幂等，可反复执行
source .venv/bin/activate && scons -j8 common/         # 标定组件改了 params_keys.h, 需重编一次
# 然后按本文件 §2.1 编 CUDA 变换库 / §3 跑相机 / §4 跑模型
```

`apply_cuda.sh` 每次都做三件事：拷贝文件 → 运行补丁器（幂等注入）→ 打印还需要手动做的事。
上传上游更新（`git pull`）之后**再跑一次**即可恢复全部改动。

---

## 1. 组件清单（能给你什么）

| # | 组件 | 文件（overlay 内） | 安装到目标树 | 功能 |
|---|---|---|---|---|
| 1 | CUDA 模型后端 | `openpilot/sunnypilot/modeld_v2/gpu_backend/`、`gpu_model_state.py` | `sunnypilot/modeld_v2/...`（旧布局走 `compat/`） | 用 CUDA/TensorRT 跑模型推理，替换 tinygrad；失败自动回退 |
| 2 | CUDA 图像变换 | `openpilot/selfdrive/modeld/transforms/cuda_transform.{cu,h}` | `selfdrive/modeld/transforms/` | 相机 NV12 → 模型要的 12 通道 GPU warp，替代 pocl/OpenCL |
| 3 | TensorRT 运行器 | `.../runners/tensorrt_runner.py`、`trt_c_api.so` | `selfdrive/modeld/runners/` | 直接吃 TRT `.plan`（零拷贝），25× 于 tinygrad |
| 4 | V4L2/VIC 相机适配 | `openpilot/system/camerad/webcam/v4l2_dmabuf_camera.py` 等 5 个 | `system/camerad/webcam/`（旧布局 `tools/webcam/`） | 让 GMSL IMX390 UYVY 走 V4L2 → VIC → NV12，供 openpilot 出图 |
| 5 | CH347 IMU | `openpilot/system/sensord/ch347t.{cc,py}`、`third_party/ch347/` | `system/sensord/` + `third_party/ch347/` | 外置 USB-I2C LSM6DS3（板载无 IMU 时用），带开机零偏自动校准，没插就自动退出 |
| 6 | **相机标定工具链** | `openpilot/tools/calib/*`、`patch_calib.py` | `tools/calib/`、并打补丁到 `common/transformations/camera.py` + `common/params_keys.h` | FCAM/ECAM 内参 + ECAM 外参标定，结果按分辨率存 Params 供全栈使用 |
| 7 | **Panda 固件/协议统一** | `/data/openpilot/panda_版本核对/`（设备侧脚本） | 不进 overlay，运行在 `/data/openpilot` | 让所有分支共用同一份 panda 固件与协议，**不再来回刷固件** |

> 6/7 是本轮新增。7 不落在 overlay 仓库里，因为它是"跨分支运维脚本"，作用于
> `/data/openpilot/*` 三个仓库（panda 固件字节按 gitignore 属构建产物，不适合放进 overlay 仓库）。
> 它的用法规格见 §5。

---

## 2. 跑起来要做的设备侧步骤

### 2.1 CUDA 变换库（每台设备编一次）

```bash
cd openpilot/selfdrive/modeld/transforms
nvcc -arch=sm_87 -shared -O2 -o libcuda_transform.so cuda_transform.cu -I.
```

### 2.2 模型引擎（每台设备编一次，跨机型不可复用）

- 分体模型（FiletOFish 等）：`driving_vision_fp16.plan` + `driving_policy_fp16.plan`
- 合并模型（BigCombo）：`driving_supercombo_fp16.plan`

```bash
trtexec --onnx=model.onnx --saveEngine=xxx_fp16.plan --fp16
# 引擎必须在部署设备上编（AGX Orin = sm_87，TensorRT 版本也要一致）
```

引擎放到 `selfdrive/modeld/models/<模型名>/`，`Params Model` 指向模型名即可。
运行开关：`DISABLE_CUDA_BACKEND=1` 关 CUDA 后端（回 tinygrad）；`USE_V4L2_CAMERA=1` 用 V4L2 相机（Linux 默认）。

### 2.3 相机驱动初始化（每次开机）

```bash
sudo tw_camera_cfg bring      # 森云 serdes 初始化, 必须先于 camerad
```

---

## 3. 相机标定工具链（本轮新增，重点）

### 3.1 它能解决什么

| 问题 | 用什么工具 | 输出 |
|---|---|---|
| 视觉测距系统性偏大/偏小（fl 不对） | `self_calibrator.py --scan` / `--live` | FCAM 焦距 fl、cx/cy → `FcamIntrinsics` |
| 第二路相机（WIDE/ECAM）焦距未知 | `ecam_analyze.py`、`wide_calibrator.py` | ECAM 焦距 → `EcamIntrinsics` |
| 俯仰/安装高度不对导致测距偏差 | `estimate_extrinsics.py` | ECAM pitch / 高度（外参） |
| 想一眼看三种方法是否一致 | `plot_calib.py` | 对比曲线图（含车道宽度校正） |

标定结果写入 Params（按分辨率键值，例如 `1920x1080`），全栈（modeld/UI/controls）自动读取；
换分辨率时按宽度比例自动缩放，不需要重标。

### 3.2 怎么用（离线，最常用）

```bash
cd <目标树> && source .venv/bin/activate
export PYTHONPATH=$PWD

# 1) 用行车日志算 FCAM 内参（三种方法融合）
python3 tools/calib/self_calibrator.py --scan --base-dir ~/.commaspold/media/0/realdata

# 2) 分离 FCAM/ECAM 帧并估 ECAM 焦距（需 --dir，子目录里各含 rlog.zst）
python3 tools/calib/ecam_analyze.py --dir <route目录>

# 3) ECAM 外参（俯仰/高度）——推荐汇总多段一起算
mkdir -p /tmp/calib && i=0
for f in $(ls -t ~/.commaspold/media/0/realdata/*/rlog.zst | head -150); do
  d=/tmp/calib/seg$i; mkdir -p $d; ln -sf "$f" $d/rlog.zst; i=$((i+1));
done
python3 tools/calib/estimate_extrinsics.py --dir /tmp/calib --fl-init 961

# 4) 画对比图（需 matplotlib，用系统 python3 也行）
python3 tools/calib/plot_calib.py --dir <route目录> --output calib.png
```

### 3.3 怎么用（车上实时，可选）

- `self_calibrator.py --live`：实时采一段（SIGTERM 结束）算 FCAM 内参。
- `self_calibrator.py --ecam` / `wide_calibrator.py`：ECAM 标定；`wide_calibrator` 需要 `cv2`
  且需要标定时把 ECAM 顶上 ROAD 流（cuda 主版本用 developer_panel 的 Wide Calibration 按钮做，
  **overlay 目前未移植该 UI/流交换**，见 §6 未包含项）。

### 3.4 已实测口径（本机 AGX Orin, 2026-09-12）

- 环境：`FcamIntrinsics` = 961.76 / `EcamIntrinsics` = 1740.0 @1920x1080（首次运行自动创建，可覆盖）。
- 150 段真实日志：`estimate_extrinsics.py` 输出 pitch 5.2° down、RMSE 4.98m。
- 注意事项：这两个脚本只扫 `--dir`（或 `--route` + `--base-dir`），不会自己遍历 realdata，
  所以要按 §3.2 第 3 步那样拼一个"每段一个子目录、里面叫 rlog.zst"的目录。

---

## 4. 模型/相机/IMU 的运行验证顺序

1. 原始后端能起：`DISABLE_CUDA_BACKEND=1` 跑一次。
2. CUDA 变换库能加载，无 CUDA error。
3. CH347 插着时能出 `accelerometer` / `gyroscope` / `temperatureSensor`；没插时进程干净退出。
4. IMU 开机零偏校准：静止时写入 `imu_calibration.json`，运动时保留旧值。
5. 双摄 V4L2 打开，分辨率/格式/stride/NV12 正确。
6. road/wide 帧时间戳、frame id、配对差值、稳定 20Hz。
7. 分体引擎与合并引擎分别测试。
8. TRT 图像输入用 GPU 指针，输出有限且量级合理。
9. 静止状态下 modelV2 / cameraOdometry / radard / controlsd / Panda / UI 全正常，再上路。
10. 短途试驾：看丢帧、模型耗时、画质、IMU、雷达 lead、脱管事件。

---

## 5. Panda 固件/协议统一（跨分支，不进 overlay）

目的：**不管跑哪个分支，panda 都不再重新刷固件**（刷得多了容易坏），做法是让三个分支
期望的固件签名完全一致。

### 5.1 一次性状态（已完成）

三个分支 `panda/board/obj/panda_h7.bin.signed` 字节相同（md5 `67bb24b2…`，签名 `1eddee73…`），
协议统一到 HEALTH **v17**（py 库 / `board/health.h` / C++ 结构 57 字节三方一致）。

### 5.2 日常怎么用

```bash
# 启动时自动：4 个启动脚本已挂自检 (aunch_pc.sh / launch_pc.sh / ajouatom / dp 的 launch_chffrplus.sh)
#   0.06s: 固件字节自动对齐 + 协议检查(不一致只告警, 不阻塞启动)

# 拉取/合并新分支之后跑一次（把该做的都做掉）
bash /data/openpilot/panda_版本核对/panda_维护.sh
#   ① 固件字节对齐 ② 幂等协议补丁(含补 qt3.py) ③ 自动重编源码有变的 ./pandad ④ 全量校验

# 只想看当前状态
bash /data/openpilot/panda_版本核对/panda_boot_check.sh
```

### 5.3 验证口径

插上 panda（USB-A）后任意分支启动，日志里应是
`signature 1eddee736255554a, expected 1eddee736255554a`，且没有
`Panda firmware out of date` / `pandad.uncaught_exception`。
回滚点：`/data/openpilot/panda_版本核对/固件备份/`。

---

## 6. 未包含 / 待办（诚实清单）

- **标定 UI**：`developer_panel` 的 4 个标定按钮、`annotated_camera` 动态内参重载、中文翻译
  —— 未移植（各分支 UI 框架不同：ajouatom 是 Qt `developer_panel.cc`，dp 是新式 `ui/layouts/settings/developer.py`）。
- **ECAM 流交换**：cuda 主版本在 `camerad_thread.cc` 里 `WideCalibMode` 交换 road/wide 流；
  旧分支是 `camerad_usb.cc`、dp 是 `jetson_camerad.py`，结构不同，未移植 —— 因此"车上按按钮实时标
  Wide"这条路暂不可用，但离线标定（§3.2）不受影响。
- **模型引擎与 libcuda_transform.so**：属设备侧产物，overlay 不带，按 §2 自己编。
- 运行 `wide_calibrator.py` 需要 `cv2`：dp 的 venv 和系统 python3 有，cuda/ajouatom 的 venv 没有
  （用系统 python3 跑，或自行 `pip install opencv-python`）。

---

## 7. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `No tool module 'qt' found` | 目标树缺 scons 的 qt3 工具：`cp /usr/lib/python3/dist-packages/SCons/Tool/qt3.py <树>/site_scons/site_tools/` |
| `UnknownKeyName: FcamIntrinsics` | params_keys.h 改了没重编：`scons -j8 common/` |
| `cythonize: not found` | scons 要在 venv 激活状态下跑 |
| 引擎加载失败 `Platform specific tag mismatch` | 引擎不是在 AGX Orin 上编的（必须 sm_87 + 同 TensorRT 版本） |
| 切分支后 panda 又开始刷固件 | 跑 `panda_维护.sh`（谁单独重编过固件，签名就会不一致） |
| 标定脚本说 `No rlog files found` | 需要 `--dir` 指向"子目录里各含 rlog.zst"的目录，见 §3.2 第 3 步 |
