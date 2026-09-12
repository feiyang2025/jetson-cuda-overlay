#!/usr/bin/env python3
"""幂等修复各分支的 panda 协议层，使其等于"设备在跑的那一份"（v17 / 57B）。

改这些文件（只改锚点能对上的，对不上会明确报"需人工"，绝不乱改）：
  panda/board/health.h          HEALTH_PACKET_VERSION 16->17, 删 fan_stall_count, spi 字段改名
  panda/python/__init__.py      版本/结构/health() 字典下标
  selfdrive/pandad/pandad.cc    两行字段引用
  selfdrive/pandad/pandad.py    旧版本缺 SKIP_PANDA_VERSION_CHECK / EnablePandaFlash 门控时补上
  site_scons/site_tools/qt3.py  旧分支缺 scons 的 qt3 工具模块时自动补（否则编译报 No tool module 'qt'）

用法：
    python3 repair_protocol.py                 # 所有分支
    python3 repair_protocol.py <分支目录...>    # 指定分支（测试用）
退出码：0=无需改动；2=有改动（需重编 ./pandad）；1=有锚点对不上/出错
"""
from __future__ import annotations

import os
import shutil
import sys

try:
    from env import FORKS
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from env import FORKS

QT3_SRC = "/usr/lib/python3/dist-packages/SCons/Tool/qt3.py"

PATCHES = [
    ("panda/board/health.h", [
        ("#define HEALTH_PACKET_VERSION 16", "#define HEALTH_PACKET_VERSION 17"),
        ("  uint16_t spi_checksum_error_count_pkt;\n  uint8_t fan_stall_count;\n",
         "  uint16_t spi_error_count_pkt;\n"),
    ], ["#define HEALTH_PACKET_VERSION 17", "spi_error_count_pkt", "!fan_stall_count"]),
    ("panda/python/__init__.py", [
        ("HEALTH_PACKET_VERSION = 16", "HEALTH_PACKET_VERSION = 17"),
        ('HEALTH_STRUCT = struct.Struct("<IIIIIIIIBBBBBHBBBHfBBHBHHB")',
         'HEALTH_STRUCT = struct.Struct("<IIIIIIIIBBBBBHBBBHfBBHHHB")'),
        ('      "spi_checksum_error_count": a[21],\n'
         '      "fan_stall_count": a[22],\n'
         '      "sbu1_voltage_mV": a[23],\n'
         '      "sbu2_voltage_mV": a[24],\n'
         '      "som_reset_triggered": a[25],\n',
         '      "spi_error_count": a[21],\n'
         '      "sbu1_voltage_mV": a[22],\n'
         '      "sbu2_voltage_mV": a[23],\n'
         '      "som_reset_triggered": a[24],\n'),
    ], ["HEALTH_PACKET_VERSION = 17", '"som_reset_triggered": a[24]', '!fan_stall_count']),
    ("selfdrive/pandad/pandad.cc", [
        ("  ps.setFanStallCount(health.fan_stall_count);\n", ""),
        ("ps.setSpiChecksumErrorCount(health.spi_checksum_error_count_pkt);",
         "ps.setSpiChecksumErrorCount(health.spi_error_count_pkt);"),
    ], ["health.spi_error_count_pkt", "!health.spi_checksum_error_count_pkt",
        "!health.fan_stall_count"]),
]

PANDAD_PY_GATES = [
    ("def flash_panda(panda_serial: str) -> Panda:",
     'def skip_version_check() -> bool:\n'
     '  return os.environ.get("SKIP_PANDA_VERSION_CHECK") == "1"\n\n'
     'def flash_panda(panda_serial: str, disable_flasher: bool = False) -> Panda:'),
    ("    HARDWARE.recover_internal_panda()\n    raise\n\n  fw_signature = get_expected_signature(panda)",
     "    HARDWARE.recover_internal_panda()\n    raise\n\n"
     "  if disable_flasher:\n"
     '    cloudlog.info("Panda flashing is disabled, skipping flash")\n'
     "    return panda\n\n"
     "  # Skip version check if environment variable is set\n"
     "  if skip_version_check():\n"
     '    cloudlog.info("Panda version check is disabled, skipping")\n'
     "    return panda\n\n"
     "  fw_signature = get_expected_signature(panda)"),
    ("      # Flash pandas\n      pandas: list[Panda] = []\n"
     "      for serial in panda_serials:\n        pandas.append(flash_panda(serial))",
     "      enable_flasher = params.get_bool(\"EnablePandaFlash\")\n\n"
     "      # Flash pandas\n"
     "      pandas: list[Panda] = []\n"
     "      for serial in panda_serials:\n"
     "        pandas.append(flash_panda(serial, not enable_flasher))"),
]


def satisfied(txt: str, markers) -> bool:
    return all((m[1:] not in txt) if m.startswith("!") else (m in txt) for m in markers)


def apply_file(fork: str, rel: str, subs, markers) -> tuple[str, str]:
    path = os.path.join(fork, rel)
    if not os.path.isfile(path):
        return ("MISSING", f"{rel} 文件不存在")
    txt = open(path, errors="replace").read()
    if satisfied(txt, markers):
        return ("OK", f"{rel} 已是基准")
    hits, missed = 0, []
    for old, new in subs:
        if old and old in txt:
            txt = txt.replace(old, new, 1)
            hits += 1
        elif old:
            missed.append(old.strip().splitlines()[0][:60])
    if hits:
        open(path, "w").write(txt)
        txt = open(path, errors="replace").read()
    if not satisfied(txt, markers):
        bad = [m for m in markers if not ((m[1:] not in txt) if m.startswith("!") else (m in txt))]
        return ("NEEDS_MANUAL", f"{rel} 锚点不匹配 {missed or ''} 仍缺 {bad}")
    return ("CHANGED" if hits else "OK", f"{rel} 已修 ({hits} 处)")


def ensure_qt3(fork: str):
    dst = os.path.join(fork, "site_scons/site_tools/qt3.py")
    if os.path.isfile(dst) or not os.path.isdir(os.path.dirname(dst)):
        return None
    if not os.path.isfile(QT3_SRC):
        return ("NEEDS_MANUAL", f"{dst} 缺失且系统无 {QT3_SRC}")
    shutil.copyfile(QT3_SRC, dst)
    return ("CHANGED", "补齐 site_scons/site_tools/qt3.py")


def main() -> int:
    forks = sys.argv[1:] or FORKS
    changed, problems = [], []
    for fork in forks:
        print(f"[{os.path.basename(fork.rstrip('/'))}]")
        r = ensure_qt3(fork)
        if r:
            print(f"   {r[0]:<12} {r[1]}")
            (changed if r[0] == "CHANGED" else problems).append(f"{fork}:{r[1]}")
        for rel, subs, markers in PATCHES:
            st, msg = apply_file(fork, rel, subs, markers)
            print(f"   {st:<12} {msg}")
            if st == "CHANGED":
                changed.append(f"{fork}:{rel}")
            elif st in ("NEEDS_MANUAL", "MISSING"):
                problems.append(f"{fork}:{rel}")
        py = os.path.join(fork, "selfdrive/pandad/pandad.py")
        if os.path.isfile(py):
            txt = open(py, errors="replace").read()
            if "SKIP_PANDA_VERSION_CHECK" in txt:
                print(f"   {'OK':<12} selfdrive/pandad/pandad.py 已有门控")
            else:
                for a, b in PANDAD_PY_GATES:
                    if a in txt:
                        txt = txt.replace(a, b, 1)
                if "SKIP_PANDA_VERSION_CHECK" in txt:
                    open(py, "w").write(txt)
                    print(f"   {'CHANGED':<12} selfdrive/pandad/pandad.py 已补防回刷门控")
                    changed.append(f"{fork}:selfdrive/pandad/pandad.py")
                else:
                    print(f"   {'NEEDS_MANUAL':<12} selfdrive/pandad/pandad.py 锚点不匹配")
                    problems.append(f"{fork}:selfdrive/pandad/pandad.py")
    print()
    print(f"changed={len(changed)} problems={len(problems)}")
    if problems:
        print("需人工处理: " + "; ".join(problems))
        return 1
    return 2 if changed else 0


if __name__ == "__main__":
    raise SystemExit(main())
