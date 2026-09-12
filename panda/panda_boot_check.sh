#!/usr/bin/env bash
# 启动自检（0.1 秒，傻瓜式）：固件字节自动对齐 + 协议检查。
# 各分支启动脚本(以及用户)都可以直接调它；不一致时只告警，不阻塞启动。
#
#   设备上的规范位置: /data/openpilot/panda_版本核对/panda_boot_check.sh
#   (用 panda/install.sh 安装/更新到这个位置)
set -uo pipefail

# 工具目录：优先设备规范位置，其次脚本自身所在目录（便于在仓库里直接跑）
if [ -f /data/openpilot/panda_版本核对/env.py ]; then
  D=/data/openpilot/panda_版本核对
else
  D="$(cd "$(dirname "$0")" && pwd)"
fi

CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

if [ "$CHECK_ONLY" = "0" ]; then
  out=$(python3 "$D/sync_firmware.py" 2>&1)
  if echo "$out" | grep -q "SYNC"; then
    echo "[panda] 固件字节已重新对齐(避免分支互刷):"
    echo "$out" | grep -E "SYNC|FAIL" | sed 's/^/        /'
  elif echo "$out" | grep -q "FAIL\|找不到基准"; then
    echo "[panda] 固件对齐有问题:"; echo "$out" | sed 's/^/        /'
  fi
fi

if ! python3 "$D/verify_protocol.py"; then
  echo "[panda] 修法: bash $D/panda_维护.sh"
  exit 1
fi
exit 0
