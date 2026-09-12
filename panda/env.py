#!/usr/bin/env python3
"""env.py —— panda 运维脚本的共用"环境自发现"模块（傻瓜式：无需任何配置）

自动发现：
  * PANDAUTIL_DIR  本工具目录（脚本自身所在目录，或设备上的 /data/openpilot/panda_版本核对）
  * FORKS          所有"带 panda 子目录的 openpilot 分支仓库"（默认扫 /data/openpilot/* 和 /opt/openpilot/*）
  * PRIMARY        主分支 = 当前设备在跑的那份固件所属分支（读 swaglog 里 pandad 的
                   "signature X, expected X" 行，X 与哪个分支的 panda_h7.bin.signed 后 128 字节
                   一致就是它；找不到就退化为名字含 sunnypilot/cuda 的分支，再退化为第一个）
  * CANON_SIG/MD5  基准固件签名/md5 = PRIMARY 分支固件文件的当前值
  * REALDATA       行车日志目录（Paths.log_root()，取不到就 $HOME/.commaspold/media/0/realdata）

用法（其它脚本）：
    from env import FORKS, PRIMARY, CANON_SIG, CANON_MD5, REALDATA, find_fork, fork_fw
    python3 env.py            # 打印当前环境(排障用)
"""
from __future__ import annotations

import glob
import hashlib
import os
import re
import subprocess

PANDAUTIL_DIR = os.path.dirname(os.path.abspath(__file__))

# 设备上的规范位置：启动脚本勾子固定引用这里
DEVICE_DIR = "/data/openpilot/panda_版本核对"

# 分支仓库扫描位置（可按需 export PANDA_FORK_ROOTS="/a:/b"）
DEFAULT_ROOTS = ["/data/openpilot", "/opt/openpilot", os.path.expanduser("~/openpilot")]


def _fork_roots() -> list[str]:
    env_roots = os.environ.get("PANDA_FORK_ROOTS")
    if env_roots:
        return [r for r in env_roots.split(os.pathsep) if r]
    return DEFAULT_ROOTS


def fork_fw(fork: str) -> str | None:
    """分支会被实际刷进去的固件文件（H7 app）。"""
    consts = os.path.join(fork, "panda", "python", "constants.py")
    if not os.path.isfile(consts):
        return None
    try:
        txt = open(consts, errors="replace").read()
    except OSError:
        return None
    m = re.search(r"H7Config = McuConfig\((.*?)\n\)", txt, re.S)
    if not m:
        return None
    names = re.findall(r'"([^"]+\.bin[^"]*)"', m.group(1))
    if not names:
        return None
    path = os.path.join(fork, "panda", "board", "obj", names[0])
    return path if os.path.isfile(path) else None


def fw_sig(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            f.seek(-128, 2)
            return f.read(128).hex()[:16]
    except OSError:
        return None


def fw_md5(path: str) -> str | None:
    try:
        return hashlib.md5(open(path, "rb").read()).hexdigest()
    except OSError:
        return None


def _discover_forks() -> list[str]:
    out = []
    for root in _fork_roots():
        if not os.path.isdir(root):
            continue
        for entry in sorted(os.listdir(root)):
            fork = os.path.join(root, entry)
            if os.path.isdir(os.path.join(fork, "panda", "python")) and fork not in out:
                out.append(fork)
    return out


FORKS = _discover_forks()


def find_fork(name_or_path: str) -> str | None:
    """按完整路径或目录名挑一个分支。"""
    for f in FORKS:
        if f == name_or_path or os.path.basename(f) == name_or_path:
            return f
    return None


def _signature_owner_from_logs(sig_by_fork: dict[str, str]) -> str | None:
    """swaglog 里 pandad 报过的固件签名 -> 哪个分支的固件文件是一致的。"""
    logs = sorted(glob.glob(os.path.expanduser("~/.commaspold/log/swaglog.*"))) or \
        sorted(glob.glob(os.path.join(REALDATA, "..", "..", "log", "swaglog.*")))
    seen: list[str] = []
    for fn in logs[-4:]:                     # 最近几个日志文件足够
        try:
            with open(fn, errors="replace") as f:
                for line in f:
                    if '"daemon": "pandad"' not in line or "signature" not in line:
                        continue
                    m = re.search(r"connected, version: (\S+?), signature ([0-9a-f]+), expected ([0-9a-f]+)", line)
                    if m:
                        seen.append(m.group(2)[:16])
        except OSError:
            continue
    for sig in reversed(seen):               # 最近的优先
        for fork, fsig in sig_by_fork.items():
            if fsig and fsig == sig:
                return fork
    return None


def _realdata() -> str:
    home = os.path.expanduser("~")
    try:
        import sys
        sys.path.insert(0, "")
        from openpilot.system.hardware.hw import Paths  # type: ignore
        p = Paths.log_root()
        if p:
            return p
    except Exception:
        pass
    return os.path.join(home, ".commaspold", "media", "0", "realdata")


REALDATA = _realdata()


def _pick_primary() -> str | None:
    sig_by_fork = {f: (fw_sig(fork_fw(f)) or "") for f in FORKS}
    owner = _signature_owner_from_logs(sig_by_fork)
    if owner:
        return owner
    for f in FORKS:                          # 退化 1：名字像主分支
        b = os.path.basename(f).lower()
        if "sunnypilot" in b or "cuda" in b:
            return f
    # 退化 2：固件签名出现次数最多的那份
    from collections import Counter
    c = Counter(v for v in sig_by_fork.values() if v)
    if c:
        top = c.most_common(1)[0][0]
        for f, s in sig_by_fork.items():
            if s == top:
                return f
    return FORKS[0] if FORKS else None


PRIMARY = _pick_primary()
_PRIMARY_FW = fork_fw(PRIMARY) if PRIMARY else None
CANON_SIG = fw_sig(_PRIMARY_FW) if _PRIMARY_FW else None
CANON_MD5 = fw_md5(_PRIMARY_FW) if _PRIMARY_FW else None
PRIMARY_FW = _PRIMARY_FW


def fork_names() -> list[str]:
    return [os.path.basename(f) for f in FORKS]


def health_version(fork: str) -> int | None:
    """分支 python 库声明的 HEALTH_PACKET_VERSION。"""
    py = os.path.join(fork, "panda", "python", "__init__.py")
    try:
        txt = open(py, errors="replace").read()
    except OSError:
        return None
    m = re.search(r"HEALTH_PACKET_VERSION = (\d+)", txt)
    return int(m.group(1)) if m else None


def health_struct_size(fork: str) -> int | None:
    py = os.path.join(fork, "panda", "python", "__init__.py")
    try:
        txt = open(py, errors="replace").read()
    except OSError:
        return None
    m = re.search(r'HEALTH_STRUCT = struct\.Struct\("([^"]+)"\)', txt)
    if not m:
        return None
    w = {"I": 4, "H": 2, "B": 1, "f": 4}
    return sum(w[c] for c in m.group(1).lstrip("<"))


def canonical_source() -> str | None:
    """基准固件源文件：优先设备备份目录里的那支，其次主分支的固件文件。"""
    cands = []
    fwdir = os.path.join(DEVICE_DIR, "固件备份")
    if CANON_MD5:
        cands += sorted(glob.glob(os.path.join(fwdir, f"*{CANON_SIG}.bin.signed"))) if CANON_SIG else []
    if PRIMARY_FW:
        cands.append(PRIMARY_FW)
    for c in cands:
        if os.path.isfile(c) and CANON_MD5 and fw_md5(c) == CANON_MD5:
            return c
    return None


def _self_test() -> int:
    print("PANDAUTIL_DIR :", PANDAUTIL_DIR)
    print("DEVICE_DIR    :", DEVICE_DIR)
    print("分支扫描根    :", _fork_roots())
    print("FORKS         :")
    for f in FORKS:
        fw = fork_fw(f)
        print(f"   - {os.path.basename(f):<24} health_v={health_version(f)} size={health_struct_size(f)}B"
              f" fw_sig={fw_sig(fw) if fw else 'n/a'} md5={(fw_md5(fw) or 'n/a')[:8]}")
    print("PRIMARY       :", os.path.basename(PRIMARY) if PRIMARY else None)
    print("CANON_SIG     :", CANON_SIG)
    print("CANON_MD5     :", CANON_MD5)
    print("canonical源   :", canonical_source())
    print("REALDATA      :", REALDATA, "(存在)" if os.path.isdir(REALDATA) else "(不存在)")
    return 0 if FORKS and CANON_SIG else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
