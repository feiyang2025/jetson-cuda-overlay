#!/usr/bin/env bash
# T906G 双路 IMX390 reset 同步对齐 —— launcher 调用入口（薄封装）
# 用法: sudo bash reset_sync.sh [--wait-link] [--extra-args...]
# 失败不阻塞调用方(exit 0), 但会留下 /tmp/reset_sync.log 供排障。
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG=/tmp/camera_reset_sync.log

if ! command -v python3 >/dev/null 2>&1; then
  echo "[reset-sync] 无 python3, 跳过" | tee "$LOG"
  exit 0
fi

echo "=== camera reset sync $(date '+%F %T') ===" | tee "$LOG"
# 失败不阻塞: launcher 继续起 openpilot(无同步仍可工作, 只是相位回到随机 ~2-7ms)
if ! python3 "$DIR/reset_sync.py" "$@" --json 2>>"$LOG"; then
  ec=$?
  echo "[reset-sync] 非致命失败(exit=$ec), 继续启动 openpilot。日志: $LOG" | tee -a "$LOG"
  exit 0
fi
exit 0