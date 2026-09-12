#!/usr/bin/env bash
# panda 运维工具 一键安装/更新（傻瓜式，无参数）
#
# 做两件事：
#   1. 把本目录的脚本装到设备的规范位置 /data/openpilot/panda_版本核对/
#      （如果那里是旧目录：先备份成 .bak_<时间戳> 再安装；软链/已是最新则跳过）
#   2. 给各分支的启动脚本挂上"启动自检"勾子（幂等，已挂就不重复）
#
# 用法:  bash install.sh          # 安装/更新 + 挂勾子
#        bash install.sh --check  # 只报告当前状态
set -uo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)"
DEST=/data/openpilot/panda_版本核对
CHECK=0
[ "${1:-}" = "--check" ] && CHECK=1

echo "来源: $SRC"
echo "目标: $DEST"
echo

# ---------- 1) 安装工具 ----------
if [ "$CHECK" = "0" ]; then
  if [ -L "$DEST" ]; then
    echo "[工具] $DEST 已是软链 -> $(readlink -f "$DEST")  (跳过安装)"
  else
    if [ -d "$DEST" ]; then
      bak="$DEST.bak_$(date +%Y%m%d_%H%M%S)"
      echo "[工具] 已有旧目录, 备份到 $bak"
      mv "$DEST" "$bak"
    fi
    mkdir -p "$DEST"
    cp -rf "$SRC/." "$DEST/"
    rm -rf "$DEST/__pycache__"
    echo "[工具] 已安装到 $DEST"
  fi
  chmod +x "$DEST"/*.sh "$DEST"/*.py 2>/dev/null || true
else
  [ -d "$DEST" ] && echo "[工具] 已安装于 $DEST" || echo "[工具] 未安装(缺 $DEST)"
fi
echo

# ---------- 2) 给启动脚本挂勾子 ----------
HOOK_BODY='if [ -f /data/openpilot/panda_版本核对/panda_boot_check.sh ]; then
  bash /data/openpilot/panda_版本核对/panda_boot_check.sh || true
fi'

hook_file() { # <启动脚本> <插入锚点前的标记行>
  local f="$1" anchor="$2"
  [ -f "$f" ] || { echo "[勾子] 跳过(无此文件): $f"; return; }
  if grep -q "panda_boot_check.sh" "$f"; then
    echo "[勾子] 已有: $f"
    return
  fi
  if [ "$CHECK" = "1" ]; then
    echo "[勾子] 缺失: $f"
    return
  fi
  # 在锚点行之前插入
  python3 - "$f" "$anchor" <<'PY'
import sys
path, anchor = sys.argv[1], sys.argv[2]
hook = '''  # ---- panda 基准自检(快 <0.1s): 固件字节对齐 + 协议检查(防分支互刷) ----
  # 不一致时只告警, 不阻塞启动; 修法: bash /data/openpilot/panda_版本核对/panda_维护.sh
  if [ -f /data/openpilot/panda_版本核对/panda_boot_check.sh ]; then
    bash /data/openpilot/panda_版本核对/panda_boot_check.sh || true
  fi

'''
s = open(path, encoding="utf-8", errors="replace").read()
if anchor not in s:
    print(f"  !! 锚点不匹配, 未插入: {path} ({anchor!r})")
    sys.exit(0)
s = s.replace(anchor, hook + anchor, 1)
open(path, "w", encoding="utf-8").write(s)
print(f"  + 已挂勾子: {path}")
PY
}

# 自动发现所有分支的启动脚本（apply_cuda.sh 改过的分支名可能不同）
for f in $(ls -d /data/openpilot/*/ 2>/dev/null); do
  for s in aunch_pc.sh launch_pc.sh launch_chffrplus.sh launch_chffrphus.sh; do
    [ -f "$f$s" ] && hook_file "$f$s" "  # start manager"
  done
done

echo
if [ "$CHECK" = "0" ]; then
  echo "完成。验证: bash $DEST/panda_维护.sh"
else
  echo "（--check 模式，未改动任何文件）"
fi
