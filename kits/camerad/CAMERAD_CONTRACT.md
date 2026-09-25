# CAMERAD_KIT 契约 (2026-09-24, sp 实车验证版)

本 kit = 相机采集 → NV12 → VisionIPC 发布的完整链路, 任何 fork apply 即用。
**下面是钉死的规则, 不是建议。改动前先问自己: 是不是又要重踩一遍实车验证过的坑。**

## 边界

```
IMX390 200万像素 -> MAX9295 -> MAX96712 -> MIPI CSI-2 -> Orin VI5 -> twgmsl
  -> /dev/video0 = road, /dev/video1 = wide
  -> V4L2 DQBUF packed UYVY 1920x1080 (4147200 字节/帧)
  -> 色度归一化 (twgmsl 专属, 见下) + CUDA packed->NV12 kernel (2.4ms, 缺则 CPU 4.7ms)
  -> VisionIPC 20 buffer + refcount -> cameraState + 帧 buffer 发布
本 kit 到 VisionIPC 发布为止。modeld 消费侧见 MODELD_KIT。

不要用 /dev/video2 当 wide: 没有实际帧, modeld 会一直等 wide 不出 modelV2。
```

## 设备契约 (与上游无关, 由硬件决定)

- /dev/video0 = road, /dev/video1 = wide (实车验证组合, launch 里 ROAD_CAM=0 WIDE_CAM=1)
- V4L2 只出 UYVY 4:2:2 (1920x1080@30 等档位); 下游 NV12 默认 1920x1080
  (4K 是驱动虚标画布, 有效像素 200 万); **输出尺寸可 env 覆盖**:
  CAM_WIDTH/CAM_HEIGHT (cp 用 1344x760) 或 SP_CAM_OUT_W/SP_CAM_OUT_H,
  resize 由 CUDA kernel 双线性完成; 源尺寸永远用相机实际采集尺寸 (cam_active),
  不能拿输出尺寸建源 surface (会溢出/越界)
- 启动先决: `sudo tw_camera_cfg bring` (serdes 初始化) 必须先于 camerad
- 内核: twgmsl 是 tegra-capture-vi 老框架 + 第三方驱动, 不是 SIPL

## 颜色规则 (不要再改!)

twgmsl 这路不是标准 UYVY。标准 UYVY 直转会发绿、雾、红蓝反。
当前正确取值 (camerad.py 内):

```python
Y = raw[0::2]
U = raw[1::4]
V = raw[3::4]
```

- `GMSL_CHROMA_LAYOUT=twgmsl` (默认)。改下标前先逐字节对比验证。
- VIC 只认标准 UYVY, 默认关。只有实验才 `SP_ENABLE_VIC_GMSL=1`。

## 输出契约 (下游 modeld/UI 都按这个吃)

- 20fps 节流: 30fps 源每 3 帧交付 2 帧, 无 sleep 无拍频
- **帧时间戳必须用 CLOCK_BOOTTIME** (openpilot 全链路时钟, 与 logMonoTime/nanos_since_boot
  一致)。twgmsl 的 V4L2 buffer timestamp 不是 BOOTTIME 体系 (实测快 ~36s): 直接使用会让
  cameraOdometry 时间错位, locationd 把正确的 IMU 观测判为 "observation too old",
  标定永远无法收敛。帧间差值 dt 仍正确, modeld 帧率/延迟补偿不受影响。(cp 实车修复)
- frame_id: road 致密递增发号 (modeld 丢帧统计只看主镜头 id, 不能跳号), wide 跟随配对
- 20 个共享 buffer + refcount (客户端 recv acquire / 用完 release) — 撕裂根因是 4 buffer + 无同步, 已修
- 零拷贝两级 (camerad.py 读 env, 任一不满足自动回退, 不影响出图):
  - `SP_NVBUF_ZEROCOPY=1` (推荐/默认): **完整零拷贝** — V4L2 REQBUFS DMABUF 接收自建
    NvBufSurface 的 dma-buf → nvbuf_import_fd 导入 CUDA → kernel(src→staging) 转换完
    **立即归还相机 buffer** (不等 VisionIPC refcount) → write_and_send fill 里 staging→dst
    D2D。采集到模型输入全程无 CPU 拷贝。依赖: libnvbuf_import.so + msgq write_and_send。
  - `SP_ZEROCOPY=1` (默认): NV12 后零拷贝 (入口一次 CPU 拷贝, 硬边界已被 nvbuf 方案覆盖,
    此级为无 nvbuf 硬件/驱动时的回退)。
- VisionIPC 客户端用 `msgq.visionipc` (不是 openpilot.common.visionipc)

## 硬边界 (实验已证明, 不要再试)

- ~~入口那一次 CPU 拷贝 (V4L2 mmap -> bytes() -> H2D) 去不掉~~ **已被推翻 (2026-09-24 晚)**:
  早期结论"twgmsl 不支持 V4L2_MEMORY_DMABUF 接收"只对 videobuf2 自带缓冲成立;
  自建 NvBufSurface (MEM_DEFAULT) 的 dma-buf fd 走 REQBUFS DMABUF 注入, tegra 驱动
  直接写 NVMM 连续内存, CUDA 可 import 可映射可读 → 完整零拷贝成立, 见输出契约。
- 关停时 `get_buffer busy for >200ms` 是进程退出时引用未归零, 不是推理失败。

## 跨分支适配 (kit 与 fork 本地版的区别)

camerad.py 是跨分支自适应版, 相对 fork 本地版多了:
- `_find_repo_root()`: 自适应老布局 (tools/webcam) / 新布局 (openpilot/system/camerad/webcam)
- libpacked_to_nv12.so / libuyvy_convert.so 多候选路径探测
- import 多路径: openpilot.system.camerad.webcam / openpilot.tools.webcam / tools.webcam
- cereal messaging 前缀自适应: try openpilot.cereal -> except cereal (胡萝卜系补救, cp 实车)
- 零拷贝能力检测回退
- cudaHostRegister 补注册诊断: cudaHostGetDevicePointer 失败时进程内补注册后重试
  (cp 实车踩坑: C++ 层 cudaHostRegister 可能因进程内早期 CUDA 状态未生效)
- packed_to_nv12.cu 含 resize + flip 180 版 (cp 相机倒装用, 双线性插值)

fork 本地版 = 单分支直连 (import 写死)。**不要用 fork 本地版替换 kit 版。**

## msgq 配套 (零拷贝必需)

- msgq/0001-visionipc-zerocopy.patch: write_and_send + 共享 refcount (6 文件 70 行,
  含 visionbuf_jetson.cc 的 __JETSON__ CUDA 映射)。目标 fork 的 msgq 子模块未打则
  SP_ZEROCOPY 与 SP_NVBUF_ZEROCOPY **都**自动回退 (camerad 检测 write_and_send 属性缺失)。
- 坑: get_buffer 是轮转的, 分开 get 和 send 会拿到两个不同 buffer (必须 write_and_send 一次完成)。
- 坑: cudaMemcpy kind 枚举 — D2D 必须是 3 (cudaMemcpyDeviceToDevice), 2 是 DeviceToHost (踩过)。
- 坑: nvbuf 相机 buffer 归还要在 kernel 转换完成后立即做, 不许等 write_and_send;
      丢帧/异常路径必须 finally 兜底归还 + requeued 标志防双还 (4 块 buffer 12 帧=0.4s 扣死过)。

## 环境变量清单

| 变量 | 默认 | 作用 |
|---|---|---|
| USE_WEBCAM | 1 | 走 webcamerad python 链路 |
| ROAD_CAM / WIDE_CAM | 0 / 1 | 相机设备索引 |
| GMSL_CHROMA_LAYOUT | twgmsl | 色度布局 (非 twgmsl 时走 CPU 标准转换) |
| SP_NVBUF_ZEROCOPY | 1 | 完整零拷贝 (V4L2 DMABUF+NvBufSurface; 依赖 so+msgq 补丁, 缺则自动回退) |
| SP_ZEROCOPY | 1 | NV12 后零拷贝通道 (无 write_and_send 自动回退) |
| SP_ENABLE_VIC_GMSL | 0 | 实验性 VIC 硬件转换, 默认关 |
| SP_CAM_FLIP | 0 | 1=180° 翻转 (cp 相机倒装用, resize kernel 双线性) |
| SP_CHROMA_SWAP | 0 | 1=U/V 交换 (个别相机色度布局差异) |
| DISABLE_CUDA_TRANSFORM | 0 | 不要设 1 (那是 USB 摄像头 PC 模式) |
