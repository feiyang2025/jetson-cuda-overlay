#!/usr/bin/env python3
"""Print the manager's process list only when it changes.

Idempotent. Upstream prints the coloured process list on every manager loop (~1 Hz),
which floods the console / launch log on a long-running board. A change (a process
starting, dying, or flipping its liveness colour) is the actual signal, so gate the
print on that: problems still show up, normal operation stays quiet.

Usage: python3 patch_manager_proclist.py <repo-root>
"""
import os
import sys

PRE_OLD = """  started_prev = False
  ignition_prev = False

  while True:
"""

PRE_NEW = """  started_prev = False
  ignition_prev = False
  running_prev = ''

  while True:
"""

# raw strings: the file contains a literal backslash-u-0-0-1-b, not an ESC byte
LOOP_OLD = r"""    running = ' '.join("{}{}\u001b[0m".format("\u001b[32m" if p.proc.is_alive() else "\u001b[31m", p.name)
                       for p in managed_processes.values() if p.proc)
    print(running)
    cloudlog.debug(running)
"""

LOOP_NEW = r"""    running = ' '.join("{}{}\u001b[0m".format("\u001b[32m" if p.proc.is_alive() else "\u001b[31m", p.name)
                       for p in managed_processes.values() if p.proc)
    # Only emit the process list when it changes (a process started, died, or changed
    # liveness colour): the 1 Hz repeat is noise, a change is the signal worth keeping.
    if running != running_prev:
      print(running)
    running_prev = running
    cloudlog.debug(running)
"""


def main():
  root = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
  path = None
  for rel in ("openpilot/system/manager/manager.py", "system/manager/manager.py"):
    p = os.path.join(root, rel)
    if os.path.isfile(p):
      path = p
      break
  if path is None:
    print(f"!! manager.py not found under {root}", file=sys.stderr)
    return 1

  s = open(path, encoding="utf-8").read()
  if "running_prev" in s:
    print("  - manager.py: process list already change-gated (no change)")
    return 0
  if PRE_OLD not in s or LOOP_OLD not in s:
    print("!! manager.py: upstream anchors not found -> SKIP (patch by hand)", file=sys.stderr)
    return 1

  s = s.replace(PRE_OLD, PRE_NEW, 1).replace(LOOP_OLD, LOOP_NEW, 1)
  open(path, "w", encoding="utf-8").write(s)
  print("  + manager.py: process list printed only on change")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
