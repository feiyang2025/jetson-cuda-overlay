# CH347T + LSM6DS3 — AGX Orin (Linux aarch64) 部署包

## 文件清单

```
agx347/
├── README.md                  本说明
├── lib/
│   └── libch347.so            官方库 (aarch64, HID模式直连hidraw)
├── include/
│   └── CH347LIB.h             API头文件（编译C程序用）
├── demo/
│   └── CH347PAR_Demo.c        官方C示例源码
├── scripts/
│   ├── imu_lsm6ds3.py         LSM6DS3 I2C扫描 + 数据流读取（Python）
│   └── install.sh             一键部署脚本
├── udev/
│   └── 99-ch347.rules         hidraw权限规则
└── driver/                    （可选）内核驱动源码
    ├── ch34x_pis.c            注意：只支持Mode1/Mode3，Mode2(HID)用不到
    └── Makefile
```

## 硬件要求

- CH347T 开发板（MUSE LAB LXC-CH347）
- **拨码开关：S1=OFF, S2=ON → Mode 2（HID免驱）**
- LSM6DS3 模块接 I2C 引脚：3V3/GND/SCL/SDA（SDO接高→地址0x6B）

## AGX Orin 部署步骤

```bash
# 1. 解压后进入目录，一键安装（复制so库 + udev权限）
sudo bash scripts/install.sh

# 2. 插上CH347，确认设备识别（应显示 1a86:55dc）
lsusb | grep 1a86

# 3. I2C 扫描
cd scripts
python3 imu_lsm6ds3.py --scan

# 4. IMU 数据流
python3 imu_lsm6ds3.py
```

## 无需内核驱动！

Mode 2 是 HID 免驱模式：
- Linux 内核自带 `hid-generic` 驱动，设备自动出现为 `/dev/hidraw*`
- `libch347.so` 直接读写 hidraw 设备
- **不需要** 编译内核模块（driver/ 目录仅备用，Mode1/Mode3 才需要）

## Python 脚本说明

`imu_lsm6ds3.py` 通过 ctypes 调用 `libch347.so`，API 与 Windows 的
CH347DLLA64.dll 完全一致：

| 函数 | 作用 |
|------|------|
| `CH347OpenDevice("/dev/hidrawX")` | 打开设备（传hidraw路径） |
| `CH347I2C_Set(fd, 1)` | I2C 100kHz（0=20K 1=100K 2=400K 3=750K） |
| `CH347StreamI2C(fd, wlen, wbuf, rlen, rbuf)` | I2C 读写 |

LSM6DS3 寄存器配置：
- `0x0F` WHO_AM_I = 0x69
- `0x10` CTRL1_XL = 0x60（加速度计 416Hz ±2g）
- `0x11` CTRL2_G  = 0x60（陀螺仪 416Hz 245dps）
- `0x12` CTRL3_C  = 0x44（BDU + 地址自增）
- `0x22`~`0x2D` 陀螺仪+加速度计数据（12字节连读）

## 常见问题

| 现象 | 解决 |
|------|------|
| `hidraw device not found` | 检查拨码开关是否 Mode2，重新插拔USB |
| `CH347OpenDevice failed` | 用 sudo 运行，或确认 udev 规则已加载 |
| `WHO_AM_I read failed` | 检查接线 SCL/SDA/3V3/GND；试 --addr 0x6A |
| 权限错误 | `sudo usermod -aG dialout $USER` 后重新登录 |
