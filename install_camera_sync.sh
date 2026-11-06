#!/usr/bin/env bash
# 独立安装入口: 只装 T906G 相机同步层(不动 overlay 其它部分)
# 用法: bash install_camera_sync.sh <tree-root>
#   1. 复制 kits/camera_sync/{reset_sync.py,reset_sync.sh,CAMERA_SYNC_CONTRACT.md}
#      -> <tree>/tools/t906g_camera/
#   2. 给 <tree>/launch_chffrplus.sh 打 run_camera_sync 补丁(幂等, 兼容已打旧版)
# 注意: spcuda(有 aunch_pc.sh 的那棵树)主启动不走 launch_chffrplus.sh 时,
#       需手动在 aunch_pc.sh 的相机就绪后加一行(见脚本末尾输出)。
set -euo pipefail
OVERLAY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TREE="${1:-$(pwd)}"
TREE="$(cd "$TREE" && pwd -P)"

echo "[camera-sync] overlay=$OVERLAY"
echo "[camera-sync] target=$TREE"

DST="$TREE/tools/t906g_camera"
mkdir -p "$DST"
cp -f "$OVERLAY/kits/camera_sync/reset_sync.py" "$DST/reset_sync.py"
cp -f "$OVERLAY/kits/camera_sync/reset_sync.sh" "$DST/reset_sync.sh"
cp -f "$OVERLAY/kits/camera_sync/CAMERA_SYNC_CONTRACT.md" "$DST/CAMERA_SYNC_CONTRACT.md"
chmod +x "$DST/reset_sync.sh"
echo "  + tools/t906g_camera/{reset_sync.py,reset_sync.sh,CAMERA_SYNC_CONTRACT.md}"

python3 "$OVERLAY/patch_launcher_camera.py" "$TREE"
python3 "$OVERLAY/patch_camera_gate.py" "$TREE"

echo
echo "[camera-sync] 完成。验证:"
echo "  sudo $DST/reset_sync.py --dry-run --json"
echo "  sudo $DST/reset_sync.py --wait-link --json   # 期望 {\"status\": \"ok\"}"
if [ -f "$TREE/aunch_pc.sh" ] && [ "$(grep -c 'run_camera_sync' "$TREE/aunch_pc.sh" 2>/dev/null || true)" = "0" ]; then
  echo
  echo "[camera-sync] 注意: 此树有 aunch_pc.sh 且未见 run_camera_sync, 若它以 aunch_pc.sh 为主启动,"
  echo "  请在 aunch_pc.sh 中 '相机驱动就绪' 之后、启动 manager 之前插入:"
  echo
  echo "  sudo bash $DST/reset_sync.sh --wait-link"
  echo
fi