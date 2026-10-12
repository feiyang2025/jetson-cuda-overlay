#!/usr/bin/env python3
"""Give the fake-panda bench run the env it needs to reach CarParams (and thus inference).

Idempotent. Ports the fake-panda environment from sunnypilot-cuda/aunch_pc.sh into this
tree's launch_chffrplus.sh.

Why: with a fake panda there is no OBD VIN responder. opendbc's
`car/car_helpers.py:fingerprint()` does, unless SKIP_FW_QUERY is set:

    carlog.warning("Getting VIN & FW versions")
    set_obd_multiplexing(True)
    vin_rx_addr, vin_rx_bus, vin = get_vin(can_recv, can_send, (0, 1))   # retries forever

so `card` never writes CarParams. modeld then blocks on
`params.get("CarParams", block=True)` in its main() and never publishes modelV2 — the UI
shows camera frames but no model, and nothing in the log is an error.

What it sets (all overridable from the outside):
  NOBOARD=1           manager ignores pandad (manager.py reads NOBOARD); the fake panda
                      publishes pandaStates + can itself
  SKIP_FW_QUERY=1     skip the VIN/FW query entirely, go straight to the CAN fingerprint
  FINGERPRINT         opendbc's fingerprint() prefers it over the CAN fingerprint
  BLOCK=<...>,pandad  belt and braces alongside NOBOARD
  FAKE_CAN_PROFILE=toyota / FAKE_CAN_HZ=100 / FAKE_PANDA_SAFETY_MODEL=toyota
                      TSS2 parsing wants KINEMATICS/WHEEL_SPEEDS at ~80Hz (50 is not enough)

Usage: python3 patch_launcher_fake_panda.py <repo-root>
"""
import os
import sys

SENTINEL = "export SKIP_FW_QUERY=1"

BLOCK = r'''    # 台架假 panda 的配套环境 (对照 sunnypilot-cuda/aunch_pc.sh)。
    # 关键: 假 CAN 没有 OBD VIN 应答, 不设 SKIP_FW_QUERY 时 opendbc 的 fingerprint()
    # 会进 get_vin() 无限重试 -> card 永远不写 CarParams -> modeld 阻塞在
    # params.get("CarParams", block=True) -> 相机有画面但没有 modelV2。
    export NOBOARD=1                                        # manager 跳过 pandad (manager.py 读 NOBOARD)
    export SKIP_FW_QUERY=1                                  # 跳过 VIN/FW 查询, 直接走 CAN 指纹
    export FINGERPRINT="${FINGERPRINT:-LEXUS_ES_TSS2}"       # opendbc fingerprint() 优先用它
    export BLOCK="${BLOCK:+$BLOCK,}pandad"                   # 与 NOBOARD 双保险
    export FAKE_CAN_PROFILE="${FAKE_CAN_PROFILE:-toyota}"
    export FAKE_CAN_HZ="${FAKE_CAN_HZ:-100}"                 # TSS2 解析要 ~80Hz, 50 不够
    export FAKE_PANDA_SAFETY_MODEL="${FAKE_PANDA_SAFETY_MODEL:-toyota}"

'''

ANCHOR = '    if pgrep -f "[f]ake_panda_state" > /dev/null 2>&1; then\n'


def main():
  root = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
  path = os.path.join(root, "launch_chffrplus.sh")
  if not os.path.isfile(path):
    print(f"!! launch_chffrplus.sh not found under {root}", file=sys.stderr)
    return 1
  s = open(path, encoding="utf-8").read()
  if SENTINEL in s:
    print("  - launch_chffrplus.sh: fake-panda env already set (no change)")
    return 0
  if ANCHOR not in s:
    print("!! anchor (fake_panda_state pgrep) not found -> SKIP (patch by hand)")
    return 1
  s = s.replace(ANCHOR, BLOCK + ANCHOR, 1)
  open(path, "w", encoding="utf-8").write(s)
  print("  + launch_chffrplus.sh: fake-panda env (NOBOARD/SKIP_FW_QUERY/FINGERPRINT/...) added")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
