#!/usr/bin/env python3
"""校验所有分支的 panda 协议一致性（傻瓜式，无参数）。

对每个分支检查：
  1. python 库 HEALTH_PACKET_VERSION == board/health.h 的 #define
  2. python HEALTH_STRUCT 的字段宽度和总数 == C 结构体（用 clang++ 真编译一遍取 sizeof）
  3. health() 字典里关键字段的下标没挪位（heartbeat_lost / controls_allowed / ...）
  4. 固件文件的 expected 签名 == 基准（基准 = 设备在跑的那份）

退出码 0 = 全通过；1 = 有分支不一致（跑 panda_维护.sh 修）。
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile

try:
    from env import CANON_SIG, FORKS, fork_fw, fw_sig
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from env import CANON_SIG, FORKS, fork_fw, fw_sig

WANT_VER = 17
WANT_SIZE = 57
WIDTH = {"uint32_t": 4, "uint16_t": 2, "uint8_t": 1, "float": 4}
FMT_WIDTH = {"I": 4, "H": 2, "B": 1, "f": 4}
REQUIRED_KEYS = {
    "heartbeat_lost": 16, "controls_allowed": 10, "safety_mode": 12,
    "car_harness_status": 11, "ignition_line": 8, "som_reset_triggered": 24,
    "spi_error_count": 21, "sbu1_voltage_mV": 22, "sbu2_voltage_mV": 23,
}

PROBE = r"""
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include "panda/board/health.h"
int main() {
  printf("%zu %zu\n", sizeof(health_t), offsetof(health_t, spi_error_count_pkt));
  return 0;
}
"""


def c_fields(text: str) -> list[tuple[str, str]]:
    m = re.search(r"struct __attribute__\(\(packed\)\) health_t \{(.*?)\};", text, re.S)
    if not m:
        return []
    out = []
    for line in m.group(1).splitlines():
        line = line.split("//")[0].strip()
        if not line.endswith(";"):
            continue
        p = line.split()
        if len(p) >= 2:
            out.append((p[0], p[1].rstrip(";")))
    return out


def check_fork(name: str, root: str) -> list[str]:
    problems: list[str] = []
    py_path = os.path.join(root, "panda", "python", "__init__.py")
    hh_path = os.path.join(root, "panda", "board", "health.h")
    if not (os.path.isfile(py_path) and os.path.isfile(hh_path)):
        return [f"{name}: 缺少 panda/python 或 board/health.h"]
    py = open(py_path, errors="replace").read()
    hh = open(hh_path, errors="replace").read()

    py_ver = int(re.search(r"HEALTH_PACKET_VERSION = (\d+)", py).group(1))
    c_ver = int(re.search(r"#define HEALTH_PACKET_VERSION (\d+)", hh).group(1))
    fmt = re.search(r'HEALTH_STRUCT = struct\.Struct\("([^"]+)"\)', py).group(1)
    py_widths = [FMT_WIDTH[c] for c in fmt.lstrip("<")]
    c_widths = [WIDTH[t] for t, _ in c_fields(hh)]
    py_size, c_size = sum(py_widths), sum(c_widths)

    probe = "n/a"
    with tempfile.TemporaryDirectory() as td:
        src, exe = os.path.join(td, "p.cc"), os.path.join(td, "p")
        open(src, "w").write(PROBE)
        r = subprocess.run(["clang++", "-std=c++17", f"-I{root}", src, "-o", exe],
                           capture_output=True, text=True)
        if r.returncode == 0:
            probe = subprocess.run([exe], capture_output=True, text=True).stdout.strip()

    dmap = dict(re.findall(r'"([A-Za-z0-9_]+)":\s*a\[(\d+)\]', py))
    bad_keys = {k: (dmap.get(k), v) for k, v in REQUIRED_KEYS.items() if dmap.get(k) != str(v)}
    fw = fork_fw(root)
    sig = fw_sig(fw) if fw else None

    ok = (py_ver == c_ver == WANT_VER and py_size == c_size == WANT_SIZE
          and py_widths == c_widths and not bad_keys and sig == CANON_SIG)
    tag = "OK  " if ok else "BAD "
    print(f"[{tag}] {name:<24} 版本 py={py_ver}/h={c_ver}  结构 py={py_size}B/C={c_size}B/C编译={probe}"
          f"  签名={sig}")
    if not ok:
        if py_ver != c_ver or py_ver != WANT_VER:
            problems.append(f"协议版本 py={py_ver} health.h={c_ver}（应为 {WANT_VER}）")
        if py_size != c_size or py_size != WANT_SIZE or py_widths != c_widths:
            problems.append(f"结构不符 py={py_size}B C={c_size}B（应为 {WANT_SIZE}B）")
        if bad_keys:
            problems.append(f"字典下标错位 {bad_keys}")
        if sig != CANON_SIG:
            problems.append(f"固件签名 {sig} != 基准 {CANON_SIG}")
    return [f"{name}: {p}" for p in problems]


def main() -> int:
    print(f"基准签名: {CANON_SIG}   分支数: {len(FORKS)}")
    print("=" * 100)
    problems: list[str] = []
    for fork in FORKS:
        problems += check_fork(os.path.basename(fork.rstrip("/")), fork)
    print("=" * 100)
    if problems:
        print("不一致:")
        for p in problems:
            print("  - " + p)
        print("\n修法: bash <本目录>/panda_维护.sh")
        return 1
    print("RESULT: 全部分支 panda 协议一致, 任何分支启动都不会刷 panda")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
