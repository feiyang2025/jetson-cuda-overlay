# stateful TRT runner (参考实现, 来自 openpilot_cp_egpucp cp-cuda-main)

**用途**：带 recurrent 状态的新代 driving policy（cinque-v3 这类）在 TensorRT 上的执行器。
主 kit modeld.py 的 features_buffer 走 host numpy 移位 + 每帧 upload；本 runner 把状态
放设备驻留乒乓缓冲，状态永不跨 host<->device（状态 ~10MB/帧，省的就是这份带宽）。

**为什么收进 overlay**（2026-11-04 对比结论）：
master-c3 当前 FiletOFish 是 split 无状态模型，用不到它；但上游下一代模型若带状态
（如 cinque-v3），这是现成的接入路径，不必从零再造。Orin 是统一内存，收益小于离散卡，
但结构正确更重要。

## 文件

| 文件 | 内容 |
|---|---|
| `stateful_trt_runner.py` | `StatefulTensorRTPolicyRunner`：引擎 IO 绑定、state ping-pong 缓冲、reset_state、run() |
| `tensorrt_policy_runner.py` | `CudaRuntime`：ctypes/libcudart 封装（malloc/upload/download/stream/host_alloc + CUDA Graph） |

## 依赖

- `stateful_trt_runner.py` 只 import 同目录的 `tensorrt_policy_runner.py`（CudaRuntime），无其他外部依赖
- Python TRT 模块从 (`/usr/lib/python3.12/dist-packages`, `~/.local/lib/python3.12/site-packages`,
  `/usr/lib/python3/dist-packages`) 里找；本机 TRT 10.16.2.10 可用
- 输入列表：`new_img, desire, traffic_convention, action_t`；状态：`state_img_q, state_desire_q, state_feat_q`
  （引擎内部 ping-pong，每帧读一个槽写另一个，然后交换）

## 接入流程（未来用到时）

1. 拷贝两文件到目标树 `selfdrive/modeld/`（apply_cuda.sh 的 `--install-stateful` 钩子正在做这个）
2. 模型 profile 加 `stateful=True` 分支：`input_names` 从引擎读，`STATE_PAIRS` 按引擎输出 `next_state_*` 声明
3. modeld 侧：每帧调 `runner.run(new_img, desire, traffic_convention, action_t)`，返回 `outputs[0]`
4. 精度对照：与 host-移位版跑同一段数据逐帧对比 <1%

## 与 kit 主路径的区别

| | kit modeld.py (master-c3) | stateful runner (egpu) |
|---|---|---|
| 状态 | host numpy `features_buffer` 移位 + upload | 设备驻留乒乓缓冲, 永不 H2D |
| 输入 dtype | 从 C API `trt_engine_input_dims` 读 (uint8/fp16) | 从 python TRT `engine.get_tensor_dtype` 读 |
| TRT 入口 | `trt_c_api.so` + libcuda ctypes | python `tensorrt` + libcudart ctypes |
| 适用 | split 无状态 (FiletOFish/CD210) | stateful (cinque-v3 等新代) |

## 与 egpu 其它部分的关联

- egpu 的 modeld 主体是 **tinygrad warp + 可选 TRT policy**（`modeld_trt_candidate.py` ModelState 里
  `if self.trt_policy is not None` 分支），不是全 TRT——本 overlay 走全 TRT + CUDA kernel，性能上限更高。
- `egpu_patches.py` 的 `patch_after_import` 幂等补丁机制是"不改上游文件"的适配思路；
  本 overlay 的 apply_cuda.sh 目前是**覆盖文件**，patch 机制列为后续增强（见 COMPARE_egpu.md）。
- `modeld_trt_candidate.py` 的 NV12 紧密→Venus 自动转换是为 USB eGPU 离散卡设计的，Orin 集成 GPU 不需要。