# MODELD_KIT 契约 (2026-09-24, TRT 兜底闭环版, 融合自 CP)

本 kit = VisionIPC 帧 → CUDA transform → TensorRT FP16 推理 → modelV2 (20Hz)。
兜底闭环从 CP (ajouatom) 的「推理资源兜底闭环」融合而来, 已在 SP 实车验证。

## 边界与输出契约

- 输入: VisionIPC road/wide 帧 (NV12 1920x1080), 零拷贝时直接吃设备指针
- 处理: cuda_transform (CUDA, 替代 OpenCL/pocl) -> TensorRT FP16
- 输出: modelV2 ~20Hz。**20Hz 输出契约不动** (模型/下游全链路按 20Hz 标定, 30Hz 得不偿失)
- 明确不移植 CP 的: OpenCV 采集、合成时间戳 (SP 真实 VI 时间戳更好)、官方 while 配对循环

## TRT 加载兜底闭环 (核心规则)

参数 (环境变量可覆盖):
- `TRT_LOAD_ATTEMPTS` 默认 3: plan 存在但加载失败的重试次数
- `TRT_LOAD_RETRY_INTERVAL` 默认 2.0s: 重试间隔

规则:
1. FiletOFish 分体模型 (driving_vision_fp16.plan / driving_policy_fp16.plan):
   TRT 加载失败 → 按 TRT_LOAD_ATTEMPTS 重试 → 仍失败才降级 tinygrad (pickle 加载),
   失败原因记录在 model.use_trt / trt_vision_fail_reason / trt_policy_fail_reason
2. BigCombo 合并引擎: 同样重试闭环 (原本硬失败直接崩 modeld);
   彻底失败 → 抛 BigComboTrtUnavailable → main() 捕获 → Params Model 切回 FiletOFish
   + cloudlog + SystemExit(0) → manager 重启 modeld 走 FiletOFish 路径
   (BigCombo 无 tinygrad 单引擎 fallback, 降级目标只能是 FiletOFish)
3. 全程可观测: 每次重试/最终降级都有日志 (含 attempt 计数与原因);
   启动汇总行 `backend: vision_trt=.. policy_trt=.. (vision_fail=.. policy_fail=..)`

验证口径:
- 正常: 无降级日志, modelV2 ~20Hz 持续, frameId 递增, valid=True alive=True
- 故障注入 (plan 改名/损坏): 看到重试日志 → 降级 tinygrad → 进程不崩, modelV2 仍出

## 配套改动 (零拷贝消费侧)

- modeld 对收到的 VisionIPC buffer 取设备指针, cuda_transform_execute(input_is_device=1),
  推理后 release buffer
- SConstruct aarch64 分支必须加 `-D__JETSON__` + CUDA include (sconstruct_jetson.diff),
  否则 visionbuf_jetson.cc 的 CUDA 映射被宏裁掉, d_addr 一直为空
- 本机 cythonize 3.3 的 `--cplus` 不生效: site_scons/site_tools/cython.py 需直接调 `cython --cplus`
  (msgq 是子模块, pyx 改动在 msgq_repo 里, 升级子模块会冲掉)

## 设备侧产物 (不随 kit, 每台设备编一次)

- .plan 引擎: trtexec --onnx=model.onnx --saveEngine=xxx_fp16.plan --fp16,
  必须在部署设备上编 (AGX Orin = sm_87, TensorRT 版本一致), 跨机型不可复用
- libcuda_transform.so: nvcc -arch=sm_87 -shared -O2 -o libcuda_transform.so cuda_transform.cu -I.
- trt_c_api.so: 设备侧 artifact

## 模型与后端开关

- DISABLE_CUDA_BACKEND=1: 关 CUDA 后端回 tinygrad
- USE_V4L2_CAMERA=1: 用 V4L2 相机 (Linux 默认)
- 模型: Params Model 指向模型名, 引擎放 selfdrive/modeld/models/<模型名>/
