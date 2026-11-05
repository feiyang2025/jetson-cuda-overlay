# CAMERA_SYNC_CONTRACT — T906G 双路 IMX390 reset 同步对齐层

状态: 2026-11-05 台架实测验证（跨 3 次冷启动 spread 40µs），挂接 launcher，非侵入。

## 1. 这个 kit 解决什么

T906G（AGX Orin + 2× SG2-IMX390C-GMSL2 + MAX9295A）双摄 free-run 相位差
随机落 0~7ms。IMX390 在模组内部、无 host I2C 通路，无法配外部触发；
des 内部 FSYNC 引擎（0x3E0）一开就杀 CSI-PLL（0 帧）。
唯一可注入的同步点是 **MFP0 (RESET) 释放时刻**：

    双路 MAX9295A 0x02BE 同时 0x00(进 reset) -> hold -> 同时 0x10(释放)
    => 帧相位差从 2~7ms 压到 0.15~0.22ms，抖动 0.5µs

## 2. 铁律（改这里之前先读）

1. **必须用单次 ioctl 双 msg 写两颗 ser**（`wr2()`），绝不用两条独立 i2cset /
   两个独立 ioctl —— 间隔越大相位残差越大。
2. **绝不写 sensor 任何寄存器**。IMX390 无 host 通路本来就是设计；此层只动
   MAX9295A GPIO0(0x02BE)。
3. **绝不碰 des FSYNC 引擎**（0x3E0 等）：开了杀 CSI-PLL，链路就 0 帧。
4. **运行时机 = tw_camera_cfg(服务模式) 就绪之后、camerad 启动之前**。
   reset 期间视频中断 ~3s；camerad 已在跑会吃断流/报错。
5. **失败不阻塞 openpilot 启动**。无同步 = 回到随机相位，不影响功能，只影响
   双摄同时性。launcher 里必须容忍 exit!=0（reset_sync.sh 已抹平为 0）。
6. 自带硬件识别：bus9 上无 des 0x4A / ser 0x41,0x42 / 0x02BE 不可读 -> SKIP
   （exit 0）。非 T906G 平台跑它零副作用。

## 3. 与既有契约的关系

- `kits/camerad/CAMERAD_CONTRACT.md`: "tw_camera_cfg 必须先于 camerad" ——
  本层插在它之后，不改变该前提。
- tw_camera_cfg 会覆盖寄存器：bring/重启后 0x02BE 回到 0x18(GPIO0 高=reset
  释放)。所以**每次启动都要重做对齐**，这是特性不是 bug。

## 4. 部署

```bash
bash <overlay>/install_camera_sync.sh <tree-root>   # 装 tools/t906g_camera/ + 打 launcher 补丁
# 或全量: bash <overlay>/apply_cuda.sh <tree-root>
```

launcher（launch_chffrplus.sh）注入位置：`ensure_camera_driver` 之后、
`start_fake_panda` 之前，调用 `run_camera_sync`。

## 5. 验证

```bash
sudo <tree>/tools/t906g_camera/reset_sync.py --dry-run --json   # 计划
sudo <tree>/tools/t906g_camera/reset_sync.py --wait-link --json # 实跑
# 期望 JSON: {"status": "ok", "link": "0x62/0x62/0x61"}
# 相位测量(可选): 文档/fsync_reset_exp/dual_ts.py (60帧, video1-video0 应 <0.3ms)
```