#!/usr/bin/env bash
set -uo pipefail
# sync_upstream.sh — 上游更新同步: 锁定文件报警, 其余自动跟进
#
# 思路 (2026-09-24 讨论定稿):
#   1. 每个 fork 记一个"适配基线" commit SHA (不是日期)
#   2. 锁定清单 = 本地适配文件 (上游动了这些文件必须人工决策, 不准静默)
#   3. 同步 = fetch upstream -> diff 基线..上游 求交集 ->
#        空: merge + apply_cuda.sh + panda_维护.sh + self_check.sh -> 更新基线
#        非空: 报警列出文件, 人工处理后再继续
#
# 用法:
#   bash tools/sync_upstream.sh <fork-根> [--lockfile <清单>] [--baseline <SHA>]
#
# 配置 (fork 根下 .overlay_sync.conf, 不存在则自动提示):
#   UPSTREAM_URL=https://github.com/sunnypilot/sunnypilot.git
#   UPSTREAM_BRANCH=master
#   锁定清单默认 tools/locklist/<仓库名>.txt

OVERLAY_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "${1:-$(pwd)}" && pwd -P)"
LOCKFILE=""
BASELINE_OVERRIDE=""
ACCEPT_REWRITE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --lockfile) LOCKFILE="$2"; shift 2 ;;
    --baseline) BASELINE_OVERRIDE="$2"; shift 2 ;;
    --accept-rewrite) ACCEPT_REWRITE=1; shift ;;
    *) shift ;;
  esac
done

CONF="$REPO_ROOT/.overlay_sync.conf"
if [ ! -f "$CONF" ]; then
  echo "缺少 $CONF, 先按模板创建:" >&2
  cat <<'EOF' >&2
UPSTREAM_URL=<上游仓库 URL>
UPSTREAM_BRANCH=<上游分支, 如 master>
EOF
  exit 1
fi
# shellcheck disable=SC1090
source "$CONF"

REPO_NAME="$(basename "$REPO_ROOT")"
[ -z "$LOCKFILE" ] && LOCKFILE="$OVERLAY_DIR/tools/locklist/${REPO_NAME}.txt"

fail() { echo "  [FAIL] $*" >&2; exit 1; }

echo "== sync_upstream: $REPO_ROOT =="
echo "   上游: $UPSTREAM_URL ($UPSTREAM_BRANCH)"
echo "   锁定清单: $LOCKFILE"

# --- 0. 代理坑: Clash 注入 http_proxy 会劫持 git 命令 ---
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

# --- 1. 锁定清单必须存在 ---
[ -f "$LOCKFILE" ] || fail "锁定清单不存在: $LOCKFILE (先建 tools/locklist/${REPO_NAME}.txt)"
LOCKED=$(grep -v '^#' "$LOCKFILE" | grep -v '^$' | sort -u)
[ -n "$LOCKED" ] || fail "锁定清单为空"

# --- 2. 基线: --baseline 覆盖 > conf 里的 BASELINE > 上次记录的基线文件 ---
BASELINE="$BASELINE_OVERRIDE"
if [ -z "$BASELINE" ]; then
  BASELINE_FILE="$REPO_ROOT/.overlay_baseline"
  [ -f "$BASELINE_FILE" ] && BASELINE="$(cat "$BASELINE_FILE")"
fi
if [ -z "$BASELINE" ]; then
  echo "首次同步没有基线文件。" >&2
  echo "  基线 = 上次被吸收的上游 commit; 不能用适配 commit 本身" >&2
  echo "  (它含本地适配改动, diff 基线..上游 会把自家改动误报为上游改动)。" >&2
  echo "  推荐: --baseline HEAD~1 (= 适配 commit 的父, 即上游原版)" >&2
  echo "  当前 HEAD 父: $(git -C "$REPO_ROOT" rev-parse HEAD~1 2>/dev/null || echo N/A)" >&2
  exit 1
fi
echo "   基线: ${BASELINE:0:12}"
git -C "$REPO_ROOT" cat-file -e "$BASELINE^{commit}" 2>/dev/null || fail "基线 SHA 无效: $BASELINE"

# --- 3. 确保 upstream remote 存在 ---
cd "$REPO_ROOT"
if ! git remote | grep -qx "upstream"; then
  [ -n "$UPSTREAM_URL" ] || fail "没有 upstream remote 且配置里没有 UPSTREAM_URL"
  git remote add upstream "$UPSTREAM_URL"
  echo "  + added upstream -> $UPSTREAM_URL"
fi

# --- 4. fetch + 上游改动集 ---
echo "  fetching upstream/$UPSTREAM_BRANCH ..."
git fetch upstream "$UPSTREAM_BRANCH" || fail "fetch 失败 (网络? 代理没 unset?)"
UPSTREAM_HEAD="$(git rev-parse "upstream/$UPSTREAM_BRANCH")"
echo "  上游 HEAD: ${UPSTREAM_HEAD:0:12}"

# --- 4.5 上游历史被强推/改写? ---
# 基线必须是上游 HEAD 的祖先。不是的话 "diff 基线..上游" 反映的是两条线的差异,
# merge 会退化成几十处冲突的大杂烩 (master-c3 2026-09-29 实例: 56 处冲突)。
if ! git merge-base --is-ancestor "$BASELINE" "$UPSTREAM_HEAD" 2>/dev/null; then
  REWRITE_MB="$(git merge-base "$BASELINE" "$UPSTREAM_HEAD" 2>/dev/null || echo '')"
  echo
  echo "  [STOP] 上游历史被改写: 基线 ${BASELINE:0:12} 不是上游 HEAD ${UPSTREAM_HEAD:0:12} 的祖先" >&2
  echo "         共同祖先: ${REWRITE_MB:0:12}" >&2
  echo "         本地独有: $(git rev-list --count "$UPSTREAM_HEAD..$BASELINE") 条 / 上游独有: $(git rev-list --count "$BASELINE..$UPSTREAM_HEAD") 条" >&2
  echo >&2
  echo "  推荐做法 (以新上游为基准重放 overlay, 而不是 merge):" >&2
  echo "    git -C $REPO_ROOT branch backup-\$(date +%Y%m%d%H%M%S) HEAD   # 留退路" >&2
  echo "    git -C $REPO_ROOT reset --hard upstream/$UPSTREAM_BRANCH" >&2
  echo "    git -C $REPO_ROOT submodule sync --recursive && git -C $REPO_ROOT submodule update --init --recursive" >&2
  echo "    bash $OVERLAY_DIR/apply_cuda.sh $REPO_ROOT" >&2
  echo "    bash $OVERLAY_DIR/tools/self_check.sh $REPO_ROOT" >&2
  echo "  前提: 本地改动都在 overlay 里可重放 (untracked 产物不受 reset 影响)。" >&2
  echo "  要脚本自动做完上面这套, 加 --accept-rewrite 重跑 (会先建 backup 分支)。" >&2
  echo >&2
  if [ "$ACCEPT_REWRITE" != "1" ]; then
    exit 1
  fi

  BK="backup-$(git rev-parse --abbrev-ref HEAD)-$(date +%Y%m%d%H%M%S)"
  echo "  --accept-rewrite: 建备份分支 $BK, 然后 reset --hard 到上游 HEAD"
  git branch "$BK" HEAD || fail "建备份分支失败"
  git reset --hard "$UPSTREAM_HEAD" || fail "reset --hard 失败"
  echo "  + 备份分支: $BK (回退: git reset --hard $BK)"
  echo "  syncing submodules ..."
  git submodule sync --recursive >/dev/null 2>&1 || true
  git submodule update --init --recursive || echo "  [WARN] submodule update 非零退出, 见上"
  git rev-parse "$UPSTREAM_HEAD" > .overlay_baseline
  echo "  + 基线已更新: $(git rev-parse --short "$UPSTREAM_HEAD")"
  echo "  re-applying overlay ..."
  bash "$OVERLAY_DIR/apply_cuda.sh" "$REPO_ROOT" || echo "  [WARN] apply_cuda.sh 有非零退出, 见上"
  for PM in /data/openpilot/panda_版本核对/panda_维护.sh "$REPO_ROOT/panda/panda_维护.sh"; do
    [ -f "$PM" ] && { echo "  running $PM"; bash "$PM" || echo "  [WARN] panda_维护.sh 非零退出"; break; }
  done
  echo "  self-check ..."
  bash "$OVERLAY_DIR/tools/self_check.sh" "$REPO_ROOT" || fail "self_check 不通过, 升级未完成! 检查上面 FAIL 项"
  echo
  echo "== 改写同步完成: $(git -C "$REPO_ROOT" log --oneline -1) =="
  echo "  提醒: 设备侧产物 (.plan 引擎 / .so) 不随 reset 变化;"
  echo "        SConstruct / params_keys.h 变了要重编: source .venv/bin/activate && scons -j8 openpilot/common/"
  exit 0
fi

# --- 5. 交集检测: 上游动了锁定文件? ---
U="$(git diff --name-only "$BASELINE..$UPSTREAM_HEAD" | sort -u)"
HIT="$(comm -12 <(printf '%s\n' "$U") <(printf '%s\n' "$LOCKED"))"
if [ -n "$HIT" ]; then
  echo
  echo "  [STOP] 上游更新动了以下锁定文件, 必须人工决策:" >&2
  printf '    %s\n' $HIT >&2
  echo >&2
  echo "  处理方式:" >&2
  echo "    1. 看每个文件的上游 diff: git diff $BASELINE..$UPSTREAM_HEAD -- <文件>" >&2
  echo "    2. 决定吸收(改 kit/补丁)还是拒绝(更新清单), 处理完再跑本脚本 merge" >&2
  echo "    merge 命令: git -C $REPO_ROOT merge upstream/$UPSTREAM_BRANCH --no-edit" >&2
  exit 1
fi
echo "  [OK] 上游未动锁定文件 ($(printf '%s' "$U" | grep -c .) 个文件有改动, 与清单零交集)"

# --- 6. merge + 重放 + 自检 ---
echo "  merging upstream/$UPSTREAM_BRANCH ..."
git merge "upstream/$UPSTREAM_BRANCH" --no-edit || fail "merge 冲突! 人工解决后: git merge --continue, 再重跑本脚本做 apply+自检"
# 基线 = 最近被吸收的上游 commit (merge 后 upstream 指针; 不能用 HEAD:
# "Already up to date" 时 HEAD 还是适配 commit, 下次 diff 会误报自家改动)
git rev-parse "upstream/$UPSTREAM_BRANCH" > .overlay_baseline
echo "  + 基线已更新: $(git rev-parse --short "upstream/$UPSTREAM_BRANCH")"

echo "  re-applying overlay ..."
bash "$OVERLAY_DIR/apply_cuda.sh" "$REPO_ROOT" || echo "  [WARN] apply_cuda.sh 有非零退出, 见上"

# panda 维护 (存在才跑)
for PM in /data/openpilot/panda_版本核对/panda_维护.sh "$REPO_ROOT/panda/panda_维护.sh"; do
  [ -f "$PM" ] && { echo "  running $PM"; bash "$PM" || echo "  [WARN] panda_维护.sh 非零退出"; break; }
done

echo "  self-check ..."
bash "$OVERLAY_DIR/tools/self_check.sh" "$REPO_ROOT" || fail "self_check 不通过, 升级未完成! 检查上面 FAIL 项"

echo
echo "== 同步完成: $(git -C "$REPO_ROOT" log --oneline -1) =="
echo "  冒烟建议: 启动 fork (launch 脚本) 看 modelV2 是否 ~20Hz 持续输出"