#!/usr/bin/env python3
"""T906G 双路 IMX390 reset 同步对齐（仅 T906G/MAX9295A 拓扑，其它硬件自动跳过）。

原理（2026-11-05 台架实测验证，见 文档/T906G_IMX390传感器墙攻坚记录_20261105.md 第十节）:
  IMX390 帧相位 = f(reset 释放时刻)。双路同时释放 reset -> 相位差 0.15~0.22ms
  （跨 3 次冷启动 spread 40us，抖动 0.5us），等效外触发级同步。
  Sensing 模组 MFP0=RESET，由 MAX9295A GPIO0 控制，寄存器 0x02BE：
    bit4 GPIO_OUT 驱动值, bit3 GPIO_IN 引脚电平回读(只读), bit2 GPIO_RX_EN,
    bit1 GPIO_TX_EN, bit0 GPIO_OUT_DIS。
  写 0x00 = 驱动低(进 reset), 写 0x10 = 驱动高(释放); 回读 bit3 自证动作真实。

安全属性（本工具刻意不做的）:
  * 不写 sensor 任何寄存器（IMX390 在模组内部，无 host I2C 通路）
  * 不碰 des FSYNC 引擎（0x3E0 必须保持 off: 开了会杀 CSI-PLL 导致 0 帧）
  * 不改任何链路配置; reset 后模组自恢复(~3s)，无需断电/重 bring

用法:
  sudo python3 reset_sync.py [--wait-link] [--hold-ms 100] [--settle-ms 3000]
                             [--timeout 8] [--dry-run] [--json]
  exit code: 0 = ok / skip(非 T906G 或未确认链路), 2 = fail
  调用方(launcher)决定: 同步失败不阻塞 openpilot 启动(记录日志继续跑)。
"""
import argparse
import ctypes
import fcntl
import json
import os
import sys
import time

I2C_RDWR = 0x0707
I2C_M_RD = 0x0001
BUS = 9
DES, S1, S2 = 0x4A, 0x41, 0x42
GPIO0 = 0x02BE
REG_PIPE_Y, REG_PIPE_Z, REG_CSIPLL = 0x011A, 0x012C, 0x0308
LINK_OK = {REG_PIPE_Y: 0x62, REG_PIPE_Z: 0x62, REG_CSIPLL: 0x61}


class i2c_msg(ctypes.Structure):
    _fields_ = [("addr", ctypes.c_uint16), ("flags", ctypes.c_uint16),
                ("len", ctypes.c_uint16), ("buf", ctypes.POINTER(ctypes.c_uint8))]


class i2c_rdwr_ioctl_data(ctypes.Structure):
    _fields_ = [("msgs", ctypes.POINTER(i2c_msg)), ("nmsgs", ctypes.c_uint32)]


def open_bus(bus=BUS):
    return os.open(f"/dev/i2c-{bus}", os.O_RDWR)


def _rw(fd, msgs):
    arr = (i2c_msg * len(msgs))(*msgs)
    data = i2c_rdwr_ioctl_data(arr, len(msgs))
    fcntl.ioctl(fd, I2C_RDWR, data)


def rd(fd, addr, reg):
    rb = (ctypes.c_uint8 * 2)((reg >> 8) & 0xFF, reg & 0xFF)
    vb = (ctypes.c_uint8 * 1)()
    _rw(fd, [i2c_msg(addr, 0, 2, rb), i2c_msg(addr, I2C_M_RD, 1, vb)])
    return vb[0]


def wr2(fd, a1, reg1, v1, a2, reg2, v2):
    """一次 ioctl 两条写 msg(不同地址) -> 双路写入间隔 = 总线时间量级(~0.1ms)。
    铁律: 绝不能用两次独立 ioctl 代替(实测间隔 0.5ms 仍可用,但单 ioctl 更稳)。
    """
    b1 = (ctypes.c_uint8 * 3)((reg1 >> 8) & 0xFF, reg1 & 0xFF, v1 & 0xFF)
    b2 = (ctypes.c_uint8 * 3)((reg2 >> 8) & 0xFF, reg2 & 0xFF, v2 & 0xFF)
    _rw(fd, [i2c_msg(a1, 0, 3, b1), i2c_msg(a2, 0, 3, b2)])


def link_state(fd):
    try:
        return {REG_PIPE_Y: rd(fd, DES, REG_PIPE_Y),
                REG_PIPE_Z: rd(fd, DES, REG_PIPE_Z),
                REG_CSIPLL: rd(fd, DES, REG_CSIPLL)}
    except OSError:
        return None


def wait_pin(fd, addr, mask, want, tries=10, delay=0.01):
    """轮询等待引脚回读达到期望值，返回 (ok, 读到的值)。
    坑(2026-11-05 实测): 写完 0x10 立即读 bit3 会读到旧值 0x0 -> 误判失败; ser 内部
    GPIO 状态更新有几个 ms 的延迟。必须轮询等, 不能用固定 sleep 也不臆断。
    """
    v = rd(fd, addr, GPIO0)
    for _ in range(tries):
        if (v & mask) == want:
            return True, v
        time.sleep(delay)
        v = rd(fd, addr, GPIO0)
    return False, v


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wait-link", action="store_true", help="先等链路 0x62/0x62/0x61 再对齐")
    ap.add_argument("--timeout", type=float, default=8.0, help="链路等待超时秒(默认8)")
    ap.add_argument("--hold-ms", type=float, default=100.0, help="reset 保持毫秒(默认100)")
    ap.add_argument("--settle-ms", type=float, default=3000.0, help="释放后静置毫秒(默认3000)")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划不动硬件")
    ap.add_argument("--json", action="store_true", help="最后输出一行 JSON 结果")
    args = ap.parse_args()
    result = {"tool": "reset_sync", "bus": BUS, "status": "unknown",
              "hold_ms": args.hold_ms, "settle_ms": args.settle_ms}

    # ---- 硬件识别: 非 T906G 拓扑一律 skip(exit 0)，不干扰其它平台 ----
    # 重要(2026-11-06 实测): launcher 里 tw_camera_cfg 刚 i2c 就绪时, serdes 往往还没
    # 建链完成(des/ser 读不到)。此时按"识别失败"直接 skip 会把同步静默丢掉。
    # 所以给了 --wait-link 时, 先轮询等拓扑出现+链路就绪, 再统一判 skip/fail。
    try:
        fd = open_bus()
    except OSError as e:
        result.update(status="skip", reason=f"no i2c-{BUS}: {e}")
        log(f"[reset-sync] SKIP: /dev/i2c-{BUS} 不存在 ({e}) —— 非 T906G 相机平台")
        print(json.dumps(result, ensure_ascii=False)) if args.json else None
        return 0

    st = s1 = s2 = None
    if args.wait_link:
        deadline = time.monotonic() + args.timeout
        while True:
            try:
                st = link_state(fd)
                if st is not None:
                    s1 = rd(fd, S1, GPIO0)
                    s2 = rd(fd, S2, GPIO0)
                    if st[REG_PIPE_Y] == LINK_OK[REG_PIPE_Y] and st[REG_PIPE_Z] == LINK_OK[REG_PIPE_Z] \
                            and st[REG_CSIPLL] == LINK_OK[REG_CSIPLL]:
                        break
            except OSError:
                st = None
            if time.monotonic() >= deadline:
                break
            time.sleep(0.2)

    if st is None or s1 is None or s2 is None:
        try:
            st = link_state(fd)
            if st is not None:
                s1 = rd(fd, S1, GPIO0)
                s2 = rd(fd, S2, GPIO0)
        except OSError:
            st = None
    if st is None:
        result.update(status="skip", reason="no des 0x4A / ser 0x41 0x42 on bus9")
        log("[reset-sync] SKIP: bus9 上没有 T906G 拓扑 (des 0x4A + ser 0x41/0x42) —— 跳过")
        os.close(fd)
        print(json.dumps(result, ensure_ascii=False)) if args.json else None
        return 0
    log(f"[reset-sync] T906G 拓扑确认: des 0x{st[REG_PIPE_Y]:02x}/0x{st[REG_PIPE_Z]:02x}/"
        f"0x{st[REG_CSIPLL]:02x}, ser1 0x02BE=0x{s1:02x}, ser2=0x{s2:02x}")

    # ---- 链路就绪判定(--wait-link 时上面已轮询等待) ----
    if args.wait_link:
        if not (st[REG_PIPE_Y] == LINK_OK[REG_PIPE_Y] and st[REG_PIPE_Z] == LINK_OK[REG_PIPE_Z]
                and st[REG_CSIPLL] == LINK_OK[REG_CSIPLL]):
            result.update(status="fail",
                          reason=f"link not ready in {args.timeout}s "
                                 f"(011A={st[REG_PIPE_Y]:#x} 012C={st[REG_PIPE_Z]:#x} "
                                 f"0308={st[REG_CSIPLL]:#x})")
            log(f"[reset-sync] FAIL: 链路 {args.timeout}s 内未就绪")
            os.close(fd)
            print(json.dumps(result, ensure_ascii=False)) if args.json else None
            return 2
        log("[reset-sync] 链路就绪")

    # ---- 干跑 ----
    if args.dry_run:
        log(f"[reset-sync] DRY-RUN: wr2(0x41/0x42 0x02BE=0x00) -> hold {args.hold_ms}ms -> "
            f"wr2(0x02BE=0x10) -> settle {args.settle_ms}ms -> 复检链路")
        os.close(fd)
        result.update(status="dry-run")
        print(json.dumps(result, ensure_ascii=False)) if args.json else None
        return 0

    # ---- 执行: assert - hold - release - settle - 复检 ----
    pre_s1 = rd(fd, S1, GPIO0)
    pre_s2 = rd(fd, S2, GPIO0)
    if (pre_s1 & 0x08) != 0x08 or (pre_s2 & 0x08) != 0x08:
        result.update(status="fail", reason=f"reset pin not high before sync (0x{pre_s1:02x}/0x{pre_s2:02x})")
        log("[reset-sync] FAIL: 预检——reset 引脚回读非高(应已释放), 有外部占用? 中止")
        os.close(fd)
        print(json.dumps(result, ensure_ascii=False)) if args.json else None
        return 2
    wr2(fd, S1, GPIO0, 0x00, S2, GPIO0, 0x00)
    time.sleep(args.hold_ms / 1000.0)
    ok_lo1, lo1 = wait_pin(fd, S1, 0x08, 0x00)
    ok_lo2, lo2 = wait_pin(fd, S2, 0x08, 0x00)
    wr2(fd, S1, GPIO0, 0x10, S2, GPIO0, 0x10)
    ok_hi1, hi1 = wait_pin(fd, S1, 0x08, 0x08)
    ok_hi2, hi2 = wait_pin(fd, S2, 0x08, 0x08)
    log(f"[reset-sync] 自证: assert 后 bit3=0x{lo1 >> 3 & 1:x}/0x{lo2 >> 3 & 1:x}(应0) "
        f"release 后=0x{hi1 >> 3 & 1:x}/0x{hi2 >> 3 & 1:x}(应1)")
    if not (ok_lo1 and ok_lo2 and ok_hi1 and ok_hi2):
        result.update(status="fail",
                      reason=f"reset pin self-check failed (assert {lo1:#04x}/{lo2:#04x} "
                             f"release {hi1:#04x}/{hi2:#04x})")
        log("[reset-sync] FAIL: 引脚自证失败 —— 写寄存器和引脚动作不一致")
        os.close(fd)
        print(json.dumps(result, ensure_ascii=False)) if args.json else None
        return 2
    time.sleep(args.settle_ms / 1000.0)
    st = link_state(fd)
    ok = st and st[REG_PIPE_Y] == LINK_OK[REG_PIPE_Y] and st[REG_PIPE_Z] == LINK_OK[REG_PIPE_Z] \
        and st[REG_CSIPLL] == LINK_OK[REG_CSIPLL]
    result.update(status="ok" if ok else "fail",
                  link=f"0x{st[REG_PIPE_Y]:02x}/0x{st[REG_PIPE_Z]:02x}/0x{st[REG_CSIPLL]:02x}"
                  if st else "n/a")
    log(f"[reset-sync] 链路复检: {result['link']} -> {'OK 双路同步' if ok else 'FAIL'}")
    os.close(fd)
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    return 0 if ok else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)