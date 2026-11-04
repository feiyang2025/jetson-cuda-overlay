# overlay vs egpu 项目模型推理对比 (2026-11-04)

对比对象：`/home/nvidia/公共/openpilot_cp_egpucp/openpilot/openpilot`（分支 cp-cuda-main）
结论一句话：**不用换路线**——egpu 是 tinygrad warp + 可选 TRT policy，overlay 是全 TRT + CUDA kernel，性能上限和确定性更高。

## 路线差异

| 环节 | egpu (cp-cuda-main) | overlay (master-c3 兼容) |
|---|---|---|
| 图像 warp | tinygrad 模型自带 warp（jits） | 正式 CUDA kernel（cuda_transform.cu，Orin 580 staging 已修） |
| 推理 | tinygrad 为主，`modeld_trt_candidate.py` ModelState.run 里 `if self.trt_policy is not None` 可选切 TRT | vision + policy 全 TensorRT（FiletOFish split），TRT_LOAD 兜底→tinygrad |
| 状态模型 | `stateful_trt_runner.py`：设备驻留 ping-pong 状态缓冲（~10MB 状态永不跨 host↔device） | features_buffer host numpy 移位 + 每帧 upload（split 模型无状态，够用） |
| 分支适配 | `egpu_patches.py`：patch_after_import 幂等补丁，不碰上游文件 | apply_cuda.sh 覆盖文件 + new/old 布局检测 |
| 设备 | WARP_DEV/QUEUE_DEV + NV12 紧密→Venus 自动转换（为 USB eGPU 离散卡设计） | Orin 集成 GPU 专用（统一内存） |

## 为什么 overlay 更优

1. **全 TRT + CUDA kernel**：egpu 的 warp 还是 tinygrad jit，每次推理走 tinygrad 的算子调度；overlay 把图像变换和推理都落到 CUDA/TRT，跨内核开销小、时序确定性高（实测 23ms/帧 @20Hz）。
2. **Orin 上 staging 已修**：cuda_transform.cu 处理了 Orin driver ≥ 580 的 cuMemHostRegister 不可读问题（host→device staging），egpu 的 from_blob 直接映射在离散卡上是错的（它自己注释也承认 PC 源要拷贝）。

## egpu 值得吸收的三点（已做/已记录）

1. ✅ **stateful_trt_runner.py + CudaRuntime 封装** → 收进 `kits/modeld/stateful/`（见该目录 README.md）。
   apply_cuda.sh 加了可选安装钩子：目标树有 stateful_model_state.py 时自动装，否则跳过。
   - 对后期模型推理更有利：上游下一代若带状态（cinque-v3 这类），接入路径现成，Orin 统一内存下收益小但结构正确。
2. ⏸ **patch_after_import 幂等补丁机制**（egpu_patches.py）——不直接改上游文件、import 后注入、静默失败。
   本 overlay 目前是覆盖文件 + .orig_local 备份。下游如果要接多个分支，建议把"覆盖"升级为"覆盖 + 补丁双模"，冲突面收敛到补丁层。列为后续增强，不是今天必须。
3. 📝 **NV12 布局自动转换**（modeld_trt_candidate.py 的紧密→Venus）——camerad 侧我们已对齐（get_nv12_info 同源），消费侧转换留作参考。

## 关联文件

- `/home/nvidia/公共/openpilot_cp_egpucp/openpilot/openpilot/selfdrive/modeld/stateful_trt_runner.py`
- `/home/nvidia/公共/openpilot_cp_egpucp/openpilot/openpilot/selfdrive/modeld/tensorrt_policy_runner.py`
- `/home/nvidia/公共/openpilot_cp_egpucp/openpilot/openpilot/selfdrive/modeld/egpu_patches.py`
- `/home/nvidia/公共/openpilot_cp_egpucp/openpilot/openpilot/selfdrive/modeld/modeld_trt_candidate.py`
- `/home/nvidia/公共/openpilot_cp_egpucp/openpilot/docs/big_model_local_gpu_baseline.md`（模型目录可用性判据）