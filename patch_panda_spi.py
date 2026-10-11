#!/usr/bin/env python3
"""Disable SPI panda probing on AGX Orin / dev boxes (panda is USB-only there).

Idempotent. Mirrors the panda handling already present in the sunnypilot-cuda tree.

Why: pandad calls PandaDFU.list() -> spi_list() at startup. On the Orin there is no
SPI panda on the bus, and the probe can wedge the Tegra spidev controller in an
uninterruptible D-state; in the common case /dev/spidev* is root:gpio 0660, so the
raw PermissionError escapes dfu.py's `except PandaSpiException` guard and crashes
pandad entirely. panda here is connected over USB, so SPI probing should be off.

What it does:
  * panda/python/spi.py : SpiDevice.__init__ refuses to open /dev/spidev* unless
                          PANDA_SPI_ENABLE=1 (escape hatch to force probing).
  * panda/python/dfu.py : widen the two SPI guards from `except PandaSpiException`
                          to `except Exception`, so any SPI failure degrades to
                          "no panda" instead of killing pandad.

Usage: python3 patch_panda_spi.py <repo-root>
"""
import os
import sys

GATE = '''    if os.environ.get("PANDA_SPI_ENABLE", "") != "1":
      # AGX Orin / dev boxes: no SPI panda on the bus, and SPI probing can
      # wedge the Tegra spidev controller (uninterruptible D-state hang).
      # Panda here is USB-only; set PANDA_SPI_ENABLE=1 to force SPI probing.
      raise PandaSpiUnavailable("SPI panda probing disabled (PANDA_SPI_ENABLE!=1)")

'''

ANCHOR = "  def __init__(self, speed=MAX_SPEED):\n    assert speed <= self.MAX_SPEED\n\n"


def patch_spi(path):
  s = open(path, encoding="utf-8").read()
  if "PANDA_SPI_ENABLE" in s:
    return "  - spi.py: already gated (no change)"
  if ANCHOR not in s:
    return "!! spi.py: SpiDevice.__init__ anchor not found -> SKIP (patch by hand)"
  s = s.replace(ANCHOR, ANCHOR + GATE, 1)
  open(path, "w", encoding="utf-8").write(s)
  return "  + spi.py: SPI probing gated behind PANDA_SPI_ENABLE=1"


def patch_dfu(path):
  s = open(path, encoding="utf-8").read()
  n = s.count("    except PandaSpiException:")
  if n == 0:
    return "  - dfu.py: already widened (no change)"
  s = s.replace("    except PandaSpiException:", "    except Exception:")
  open(path, "w", encoding="utf-8").write(s)
  return f"  + dfu.py: widened {n} SPI guard(s) to `except Exception`"


def main():
  root = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
  panda = os.path.join(root, "panda", "python")
  if not os.path.isdir(panda):
    print(f"!! panda/python not found under {root}", file=sys.stderr)
    return 1
  print("Patching panda SPI handling (panda is USB-only on this board)...")
  print(patch_spi(os.path.join(panda, "spi.py")))
  print(patch_dfu(os.path.join(panda, "dfu.py")))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
