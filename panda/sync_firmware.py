#!/usr/bin/env python3
"""统一所有分支的 panda 固件字节到"设备在跑的那一份"（傻瓜式，无参数）。

对比的是每个分支 panda/board/obj/<app_fn> 的字节：
  * 已是基准 -> OK
  * 不一致   -> 备份成 .bak_orig 后覆盖为基准
  * 文件缺失 -> 从基准复制（git clean 之后会出现）

用法：
    python3 sync_firmware.py            # 对齐 + 校验
    python3 sync_firmware.py --check    # 只看不改
    python3 sync_firmware.py --root /x  # 只处理某个分支目录(测试用)
"""
from __future__ import annotations

import os
import shutil
import sys

try:
    from env import CANON_MD5, CANON_SIG, FORKS, canonical_source, fork_fw, fw_md5, fw_sig
except ImportError:  # 允许从其它目录运行
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from env import CANON_MD5, CANON_SIG, FORKS, canonical_source, fork_fw, fw_md5, fw_sig


def sync_one(fork: str, src: str, check_only: bool) -> str:
    fw = fork_fw(fork)
    if fw is None:
        # 目录存在但固件名解析不出来：用主分支同名文件兜底
        fw = os.path.join(fork, "panda", "board", "obj", os.path.basename(src))
    if not os.path.isdir(os.path.dirname(fw)):
        return "SKIP(无 board/obj)"
    if os.path.isfile(fw) and fw_md5(fw) == CANON_MD5:
        return "OK(已一致)"
    state = f"DIFF(现在 sig={fw_sig(fw) or 'n/a'})" if os.path.isfile(fw) else "MISS(缺失)"
    if check_only:
        return state
    if os.path.isfile(fw) and not os.path.isfile(fw + ".bak_orig"):
        shutil.copy2(fw, fw + ".bak_orig")
    shutil.copy2(src, fw)
    return "SYNC(已对齐)" if fw_md5(fw) == CANON_MD5 else "FAIL(覆盖失败)"


def main() -> int:
    check_only = "--check" in sys.argv
    only = None
    if "--root" in sys.argv:
        only = sys.argv[sys.argv.index("--root") + 1]

    src = canonical_source()
    if not src:
        print("!! 找不到基准固件（主分支的 panda_h7.bin.signed 不存在？）")
        return 1
    print(f"基准固件: {src}  (sig {CANON_SIG}, md5 {CANON_MD5})")
    print()

    forks = [only] if only else FORKS
    rc = 0
    for fork in forks:
        name = os.path.basename(fork.rstrip("/"))
        res = sync_one(fork, src, check_only)
        print(f"  {name:<24} {res}")
        if res.startswith(("FAIL", "SKIP")) or (check_only and res.startswith(("DIFF", "MISS"))):
            rc = 2
    if check_only and rc == 2:
        print("\n有分支不是基准 -> 直接再跑一次(不带 --check)即可修好")
    if rc == 0:
        print("\n结论: 所有分支的 panda 固件字节一致, 任何分支启动都不会刷 panda")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
