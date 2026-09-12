#!/usr/bin/env bash
# panda 一键维护（傻瓜式）：拉取/合并新分支之后跑一次，把该做的都做掉。
#
#   ① 固件字节对齐（所有分支统一到"设备在跑的那一份"）
#   ② 协议补丁（幂等；旧分支缺 qt3.py 也自动补）
#   ③ 源码有变动的分支自动重编 ./pandad
#   ④ 全量校验（版本/结构/偏移/字典/签名）
#
# 用法: bash panda_维护.sh
set -uo pipefail

if [ -f /data/openpilot/panda_版本核对/env.py ]; then
  D=/data/openpilot/panda_版本核对
else
  D="$(cd "$(dirname "$0")" && pwd)"
fi

echo "=========== 环境自发现 ==========="
FORK_LIST=$(python3 - "$D" <<'PY'
import sys, os
sys.path.insert(0, sys.argv[1])
from env import CANON_SIG, PRIMARY, FORKS
print(os.path.basename(PRIMARY or "") or "-")
for f in FORKS:
    print(f)
print(CANON_SIG)
PY
)
PRIMARY_NAME=$(echo "$FORK_LIST" | head -1)
CANON_SIG=$(echo "$FORK_LIST" | tail -1)
FORKS=$(echo "$FORK_LIST" | sed -n '2,$p' | head -n -1)
echo "  主分支(设备在跑的固件来源): $PRIMARY_NAME    基准签名: $CANON_SIG"
echo "  巡查分支: $(echo $FORKS | tr '\n' ' ')"
echo

echo "=========== ① 固件字节对齐 ==========="
python3 "$D/sync_firmware.py" || true
echo

echo "=========== ② 协议补丁(幂等) ==========="
python3 "$D/repair_protocol.py"
rc=$?
[ "$rc" = "1" ] && echo "!! 上面的 NEEDS_MANUAL 需要人工处理后再继续"
echo

echo "=========== ③ 重编源码有变动的 ./pandad ==========="
for f in $FORKS; do
  name=$(basename "$f")
  bin="$f/selfdrive/pandad/pandad"
  if [ ! -f "$bin" ]; then
    echo "--- $name: 没有 ./pandad, 首次构建"
  else
    newer=$(find "$f/selfdrive/pandad" -maxdepth 1 \( -name '*.cc' -o -name '*.pyx' \) -newer "$bin" 2>/dev/null)
    [ "$f/panda/board/health.h" -nt "$bin" ] && newer="$newer"$'\n'"health.h"
    if [ -z "${newer// /}" ]; then
      echo "--- $name: ./pandad 已是最新, 不动"; continue
    fi
    echo "--- $name: 源码比 ./pandad 新 -> 重编"
  fi
  log="/tmp/panda_build_$name.log"
  if ( cd "$f" && source .venv/bin/activate && scons -j8 selfdrive/pandad >"$log" 2>&1 ); then
    echo "    重编 OK   (日志 $log)"
  else
    echo "    重编失败! (日志 $log):"; tail -6 "$log" | sed 's/^/      /'
  fi
done
echo

echo "=========== ④ 全量校验 ==========="
if python3 "$D/verify_protocol.py"; then
  echo
  echo "✅ 完成: 所有分支 panda 基准一致, 任何分支启动都不会刷 panda"
  exit 0
fi
echo
echo "❌ 校验未通过（见上面输出）"
exit 1
