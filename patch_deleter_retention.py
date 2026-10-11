#!/usr/bin/env python3
"""Add age-based route retention to loggerd's deleter.

Idempotent. Upstream deletes only under free-space pressure (MIN_BYTES = 5GB /
MIN_PERCENT = 10%). On a large dev-box disk that means hundreds of GB can accumulate
before the first delete, so recorded routes pile up for months. This adds
"also delete routes older than N days", N from LOGGERD_MAX_ROUTE_AGE_DAYS
(default 7, 0 = upstream behaviour only).

Usage: python3 patch_deleter_retention.py <repo-root>
"""
import os
import sys

CONST_OLD = """MIN_BYTES = 5 * 1024 * 1024 * 1024
MIN_PERCENT = 10
"""

CONST_NEW = """MIN_BYTES = 5 * 1024 * 1024 * 1024
MIN_PERCENT = 10

# Upstream only prunes under free-space pressure (MIN_BYTES/MIN_PERCENT). On a large
# dev-box disk that can mean hundreds of GB accumulated before the first delete, so
# routes pile up for months. Also delete routes whose newest write is older than this
# many days. 0 = upstream behaviour only. Override with LOGGERD_MAX_ROUTE_AGE_DAYS.
MAX_ROUTE_AGE_DAYS = int(os.environ.get("LOGGERD_MAX_ROUTE_AGE_DAYS", "7"))
"""

LOOP_OLD = """    out_of_bytes = get_available_bytes(default=MIN_BYTES + 1) < MIN_BYTES
    out_of_percent = get_available_percent(default=MIN_PERCENT + 1) < MIN_PERCENT

    if out_of_percent or out_of_bytes:
      dirs = listdir_by_creation(Paths.log_root())
      preserved_dirs = get_preserved_segments(dirs)
"""

LOOP_NEW = """    out_of_bytes = get_available_bytes(default=MIN_BYTES + 1) < MIN_BYTES
    out_of_percent = get_available_percent(default=MIN_PERCENT + 1) < MIN_PERCENT
    out_of_space = out_of_percent or out_of_bytes

    # Age-based pruning (see MAX_ROUTE_AGE_DAYS). A route still being written has a
    # fresh mtime and is therefore never the oldest, so the active one is safe.
    too_old = False
    if not out_of_space and MAX_ROUTE_AGE_DAYS > 0:
      old_dirs = listdir_by_creation(Paths.log_root())
      if old_dirs:
        try:
          oldest_mtime = os.path.getmtime(os.path.join(Paths.log_root(), old_dirs[0]))
          too_old = (time.time() - oldest_mtime) > MAX_ROUTE_AGE_DAYS * 86400
        except OSError:
          too_old = False

    if out_of_space or too_old:
      dirs = listdir_by_creation(Paths.log_root())
      preserved_dirs = get_preserved_segments(dirs)
"""


def find_deleter(root):
  for rel in ("openpilot/system/loggerd/deleter.py", "system/loggerd/deleter.py"):
    p = os.path.join(root, rel)
    if os.path.isfile(p):
      return p
  return None


def main():
  root = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
  path = find_deleter(root)
  if path is None:
    print(f"!! deleter.py not found under {root}", file=sys.stderr)
    return 1

  s = open(path, encoding="utf-8").read()
  if "MAX_ROUTE_AGE_DAYS" in s:
    print("  - deleter.py: already has age retention (no change)")
    return 0
  if CONST_OLD not in s or LOOP_OLD not in s:
    print("!! deleter.py: upstream anchors not found -> SKIP (patch by hand)", file=sys.stderr)
    return 1

  s = s.replace(CONST_OLD, CONST_NEW, 1).replace(LOOP_OLD, LOOP_NEW, 1)
  open(path, "w", encoding="utf-8").write(s)
  print("  + deleter.py: age-based route retention added (default 7d)")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
