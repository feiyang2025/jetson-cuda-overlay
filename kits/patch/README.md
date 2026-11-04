# kits/patch — 分支适配补丁层

overlay 的“分支适配补丁层”。动机与判定表见下；一句话总结：
**能整文件覆盖就整文件覆盖（sunnypilot 系），不能就只打补丁**——绝不把别人
fork 自己的 modeld 业务逻辑静默覆盖掉。

## 三块内容

| 文件 | 作用 |
|---|---|
| `detect_fork.py` | 纯 AST 判定 fork 的 ModelState 形状，不导入任何东西。`apply_cuda.sh` 用它决定走整文件模式还是补丁模式 |
| `cuda_adapter.py` | import 时注入（patch_after_import）：ModelState.__init__ -> GpuModelState（CUDA/TRT，失败回退 tinygrad）+ camerad.Camera -> V4L2/DMABUF。只认 stock 形状签名，别的 fork 一律不碰 |
| `install_patch.py` | 把 adapter 放进 `<树>/<pkg>/selfdrive/modeld/`，把钩子写进 `sitecustomize.py`（已有自研钩子的树保留原样，仅追加/跳过） |
| `tests/` | 27 个单测：registry 扫描/选择/失败冷却 + adapter 签名判定/幂等/静默 |

## fork 判定表（实测 2026-11-04）

| 项目 | detect_fork 结果 | apply_cuda.sh 行为 |
|---|---|---|
| 新 sunnypilot（`/home/nvidia/公共/sunnypilot`） | sunnypilot-stock | 整文件覆盖 + 补丁层 |
| master-c3（carrot 系，咱们主树） | sunnypilot-stock | 整文件覆盖 + 补丁层（它自带 cuda_patches，sitecustomize 保留） |
| StarPilot | sunnypilot-stock | 整文件覆盖 + 补丁层 |
| sunnypilot-cuda（老布局） | unknown | 只补丁层 |
| egpu（cp-cuda-main） | frogpilot 形状 | 只补丁层 |
| FrogPilot 0.9.7 | frogpilot (thneed) | 只补丁层 |

“整文件覆盖安全”的判据：该 fork 的 `ModelState.__init__` 是不是
`(cam_w, cam_h, chestnut...)` 形状。FrogPilot 是 `(context, model, model_version)`，
它的模型注册/切换逻辑在 ModelState 内部，覆盖 = 删功能。

## ModelState 形状与补丁行为

| 形状 | 示例 | 补丁层行为 |
|---|---|---|
| stock | `(cam_w, cam_h, chestnut)` | __init__ 包一层 → GpuModelState，失败回退原类并记 `.trt_failed` |
| frogpilot | `(context, model, model_version)` | 不碰（记录 signature unsupported）；接入点在 ModelState.run，需 ForkPilot 专用 adapter |
| unknown | — | 不碰，只在日志说明 |

## 模型目录动态注册（gpu_backend/models/registry.py）

比“硬编码三家 profile”多出来的能力（StarPilot bundle 语义的移植）：

- `scan_profiles()`：扫 `models/<Name>/`，shape/slices/temporal 全从 metadata pkl 读，
  新模型编译完放进目录即可用，不用改代码；
- 选择顺序：显式名 -> `MODEL_NAME` env -> `models/.trt_order` -> Params `Model` ->
  手工清单 -> 扫描目录按名字排序；
- `models/.trt_failed`：初始化失败的 bundle 被跳过，冷却 900s 或引擎 mtime 变新
  （重编译）后自动复活——坏引擎不会让 modeld 重启循环；
- `snapshot()`：每个候选 + 就绪原因，喂日志/Web UI；
- `DISABLE_GPU_REGISTRY_SCAN=1` 退回“只用手工清单”的旧行为。

## 使用

```bash
# 单独装补丁层（apply_cuda.sh 已自动调用）
python3 kits/patch/install_patch.py <树根>

# 看一台机器的模型就绪状况（树要在 PYTHONPATH 上）
PYTHONPATH=<树根> python3 kits/patch/cuda_adapter.py --diagnose <树根>

# 单测
python3 -m unittest discover -s kits/patch/tests -v
```

环境开关：`OPENPILOT_CUDA_PATCH=0` 全关；`DISABLE_CUDA_BACKEND=1` 钩子挂着但回退原类；
`TRT_FAIL_COOLDOWN_S` 调失败冷却秒数。