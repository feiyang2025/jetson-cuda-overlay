# 功能边界 + 调用手册（给 AI agent 与人工看）

> 本文件回答三个问题：**能干什么 / 干不了什么 / 每条功能怎么调用**。
> 命令全部可复制粘贴执行；括号里标注"实测"= 已在本机 AGX Orin 上跑通并核对过。
> 适配方：NVIDIA AGX Orin + Ubuntu 24.04 + CUDA/TensorRT，openpilot 类分支（CP/SP/DP/FP/master-c3，新/旧布局）。

---

## 0. 黄金路径（新分支从零到能跑）

```bash
# ① 应用到目标分支（幂等，git pull 后重跑一次）
cd /data/openpilot/<分支>
bash /data/openpilot/jetson-cuda-overlay/apply_cuda.sh .

# ② 重编 params（标定组件新增了 Param 键；params_keys.h 是编译进去的）
source .venv/bin/activate && scons -j8 common/

# ③ 编 CUDA 变换库（每台设备一次）
cd openpilot/selfdrive/modeld/transforms
nvcc -arch=sm_87 -shared -O2 -o libcuda_transform.so cuda_transform.cu -I.

# ④ 放模型引擎（.plan 必须在 Orin 上编，跨机型不可用）
#   分体: driving_vision_fp16.plan + driving_policy_fp16.plan
#   合并: driving_supercombo_fp16.plan  (BigCombo 1.7G)
#   目录: selfdrive/modeld/models/<模型名>/  然后 Params Model=<模型名>

# ⑤ 相机初始化（每次开机，先于 camerad）
sudo tw_camera_cfg bring

# ⑥ 体检（一条命令看全链路）
python3 /data/openpilot/jetson-cuda-overlay/doctor.py
```

---

## 1. 功能边界（诚实清单）

### 1.1 确有其事（可以对外这么说）

| 能力 | 具体做了什么 | 证据 |
|---|---|---|
| 自动适配分支 | 自动识别新布局（`openpilot/sunnypilot/modeld_v2`）与旧布局（`selfdrive/modeld`），幂等、版本无关 | `apply_cuda.sh` 实测（cuda/ajouatom/dp 三个分支） |
| 相机链路 | GMSL IMX390（森云 SG2）packed UYVY → V4L2 → twgmsl 色度归一化 → CUDA packed→NV12（VIC 默认禁）；4 个适配文件 + packed_to_nv12.cu + 自动 patch camerad | 实机出图 20Hz（20fps 锁定，颜色平衡实车验证，零拷贝可选） |
| CUDA 推理后端 | CUDA 图像变换（NV12→12 通道 warp，绕开 pocl/OpenCL）+ TensorRT 运行器（零拷贝），tinygrad 自动回退 | 实机 `Using TensorRT` ～2ms/帧 |
| 模型兼容 | **小模型（分体 vision+policy，FiletOFish 类）与 1.7G 大模型（BigCombo 单引擎合并）都支持**，走 ModelProfile 注册表自动选 | 本机两套引擎都跑过（BigCombo 240 帧漂移测试通过） |
| CH347 IMU | 外置 USB-I2C LSM6DS3 守护进程：发 accelerometer/gyroscope/temperatureSensor，开机零偏自动校准，**没插设备自动干净退出** | 实机验证；`process_config` 注册可选守护进程 |
| 相机标定工具链 | FCAM/ECAM 内参、ECAM 外参（离线用行车日志）；参数化内参按分辨率存 Params，全栈自动读取 | 150 段真实日志跑出 pitch 5.2°/RMSE 4.98m（ajouatom + dp 都验过） |
| Panda 跨分支统一 | 所有分支共用同一份 panda 固件字节（签名一致）→ **任何分支启动都不再刷固件**；协议三方一致（py库/health.h/C结构） | 三个分支签名统一为 `1eddee73…`；启动自检 0.06s |
| 体检/排障 | `doctor.py` 一条命令过一遍环境/分支/相机/CUDA/IMU/标定/Panda | 本机跑通（4 项提示均为"未插设备"类） |

### 1.2 有限制（别对外说满）

| 项 | 限制 |
|---|---|
| "自动"是两档 | 文件/补丁是真自动；**设备侧仍需手动 3 步**：编 `libcuda_transform.so`、准备 `.plan` 引擎、重编一次 `scons -j8 common/` |
| 相机外参"自动" | rpy 外参由 openpilot 自带 `calibrationd` 在线自动估（overlay 未改）；overlay 提供的是**离线标定** FCAM/ECAM 内参 + ECAM 外参（pitch/高度） |
| 标定需要在车上实时按按钮的那条路 | 未移植（见 1.3） |
| 引擎/固件字节不进仓库 | `.plan`、`panda_h7.bin.signed`、`libcuda_transform.so` 都是设备相关产物，仓库只带工具与补丁 |
| `wide_calibrator.py` | 需要 `cv2`：系统 python3 与 dp 的 venv 有，cuda/ajouatom 的 venv 没有（用系统 python3 或自行 pip 装 opencv） |
| CH347 IMU | 是**外置** USB 板，不插就没有 IMU（板载 IMU 走系统自带 sensord，不受影响） |

### 1.3 未包含（可继续移植的候选）

- 相机标定 **UI**：`developer_panel` 的 4 个按钮（FCAM / FCAM Live / Wide / 查看）、`annotated_camera` 动态内参重载、中文翻译。
- **ECAM 实时标定的流交换**：主版本在 `camerad_thread.cc` 里用 `WideCalibMode` 交换 road/wide 流；旧分支是 `camerad_usb.cc`、dp 是 `jetson_camerad.py`，结构不同未移植 → "车上按按钮实时标 Wide"暂不可用（离线标定不受影响）。

---

## 2. 调用手册（按场景）

### 2.1 应用到任意分支（每次 pull 后）

```bash
cd /data/openpilot/<分支>
bash /data/openpilot/jetson-cuda-overlay/apply_cuda.sh .        # 幂等
source .venv/bin/activate && scons -j8 common/                 # 有改动时它会提示
```
卸载/回滚：改动都在工作区（未提交），`git checkout -- <文件>` 即可回退；`.bak_orig` 只在固件对齐时生成。

### 2.2 相机标定（离线，最常用）

```bash
cd /data/openpilot/<分支> && source .venv/bin/activate
export PYTHONPATH=$PWD

# FCAM 内参（扫历史日志，三方法融合）
python3 tools/calib/self_calibrator.py --scan

# ECAM 焦距估计 / FCAM-ECAM 帧分离（--dir 指向"子目录里各含 rlog.zst"的目录）
python3 tools/calib/ecam_analyze.py --dir <route目录>

# ECAM 外参（俯仰/高度）——推荐汇总多段
mkdir -p /tmp/calib && i=0
for f in $(ls -t ~/.commaspold/media/0/realdata/*/rlog.zst | head -150); do
  d=/tmp/calib/seg$i; mkdir -p $d; ln -sf "$f" $d/rlog.zst; i=$((i+1))
done
python3 tools/calib/estimate_extrinsics.py --dir /tmp/calib --fl-init 961

# 对比图（需 matplotlib）
python3 tools/calib/plot_calib.py --dir <route目录> --output calib.png
```

标定结果落到 Params（`FcamIntrinsics` / `EcamIntrinsics`，按 `1920x1080` 键值存 JSON），
换分辨率时按宽度比例自动缩放。查看当前值：

```bash
cat ~/.commaspold/params/d/FcamIntrinsics; echo; cat ~/.commaspold/params/d/EcamIntrinsics
```

### 2.3 模型（小模型 / 大模型）

```bash
# 看当前分支有哪些引擎
python3 /data/openpilot/jetson-cuda-overlay/doctor.py | sed -n '/CUDA 后端/,/CH347/p'

# 选模型（重启 modeld 生效；也可以 echo 到参数里热切换）
echo -n FiletOFish > ~/.commaspold/params/d/Model      # 分体小模型
echo -n BigCombo   > ~/.commaspold/params/d/Model      # 1.7G 合并大模型

# 强制走 tinygrad（排障用）
DISABLE_CUDA_BACKEND=1 ./launch_pc.sh
```

### 2.4 CH347 IMU

```bash
# 不插设备时自动退出，无需干预；插上后：
ls /dev/ch34x_pis* /dev/ttyACM*                 # 看设备节点
ls /data/openpilot/<分支>/imu_calibration.json  # 零偏校准结果
```
构建与运行脚本由 overlay 一起安装（`system/sensord/{build,run}_ch347t.sh`），首次运行自动编译。

### 2.5 Panda 跨分支统一（傻瓜式）

```bash
# 安装/更新工具到设备规范位置 + 给所有分支启动脚本挂自检勾子（幂等）
bash /data/openpilot/jetson-cuda-overlay/panda/install.sh

# 拉取/合并新分支之后跑一次（固件对齐 → 协议补丁 → 需要就重编 → 校验）
bash /data/openpilot/panda_版本核对/panda_维护.sh

# 只想看当前状态（不改任何东西）
bash /data/openpilot/panda_版本核对/panda_boot_check.sh
```

- 脚本会**自动发现**所有分支（扫 `/data/openpilot/*`、`/opt/openpilot/*`，可用 `PANDA_FORK_ROOTS` 覆盖），
  **自动判定**"设备在跑的那份固件"（读 swaglog 里 pandad 报的签名），不需要任何配置。
- 启动脚本里的勾子只做两件事：固件字节对齐 + 协议检查（0.06s，不一致只告警，不阻塞启动）。
- 验证口径：插上 panda（USB-A）后日志应是
  `signature 1eddee736255554a, expected 1eddee736255554a`，且没有 `out of date` / `uncaught_exception`。

### 2.6 体检与排障

```bash
python3 /data/openpilot/jetson-cuda-overlay/doctor.py           # 全量体检
python3 /data/openpilot/jetson-cuda-overlay/doctor.py --short   # 只看有问题的项
```

| 现象 | 处理 |
|---|---|
| `No tool module 'qt' found` | 目标分支缺 scons qt3 工具 → 跑 `panda/install.sh` 或手动拷 `/usr/lib/python3/dist-packages/SCons/Tool/qt3.py` 到 `<分支>/site_scons/site_tools/` |
| `UnknownKeyName: FcamIntrinsics` | params_keys.h 改了没重编 → `scons -j8 common/` |
| `cythonize: not found` | scons 要在 venv 激活状态下跑 |
| 引擎报 `Platform specific tag mismatch` | 引擎不是在 Orin 上编的（必须 sm_87 + 同 TensorRT 版本） |
| 切分支后又开始刷 panda | 跑 `panda_维护.sh`（谁单独重编过固件，签名就会不一致） |
| 标定脚本说 `No rlog files found` | 需要 `--dir` 指向"子目录里各含 rlog.zst"的目录（见 2.2） |

---

## 3. 目录速查

| 位置 | 内容 |
|---|---|
| `jetson-cuda-overlay/apply_cuda.sh` | 一键应用（入口） |
| `jetson-cuda-overlay/patch_calib.py` | 标定组件补丁器（camera.py 内参管道 + params_keys 键） |
| `jetson-cuda-overlay/openpilot/tools/calib/` | 标定工具本体（随 apply 装到目标分支） |
| `jetson-cuda-overlay/panda/` | panda 跨分支统一工具（`install.sh` 装到设备、`panda_维护.sh` 日常用） |
| `jetson-cuda-overlay/doctor.py` | 一键体检 |
| `/data/openpilot/panda_版本核对/` | panda 工具在设备上的规范位置（启动脚本勾子引用此处） |
