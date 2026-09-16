#!/usr/bin/env python3
"""Register sensord_ch347 as an optional NativeProcess in process_config.py.

Idempotent. The daemon auto-exits when no CH347 device is present, so it does
not break projects without the hardware. Build is run via a bash wrapper so no
prebuilt binary is required.
"""
import sys

path = sys.argv[1]
s = open(path, encoding="utf-8").read()

if "sensord_ch347" in s:
  print("  - sensord_ch347 already registered")
  raise SystemExit(0)

marker = "managed_processes = {p.name: p for p in procs}"
if marker not in s:
  print("ERROR: managed_processes line not found", file=sys.stderr)
  raise SystemExit(1)

if "def always_run" not in s:
  print("ERROR: always_run helper not found in process_config", file=sys.stderr)
  raise SystemExit(1)

append = (
  '\n'
  '# CH347 USB-I2C IMU (AGX Orin). Builds on first run; exits cleanly if no device.\n'
  'procs += [NativeProcess("sensord_ch347", "openpilot/system/sensord",\n'
  '              ["bash", "run_ch347t.sh"],\n'
  '              always_run, enabled=PC, sigkill=False)]\n'
)

idx = s.find(marker)
s = s[:idx] + append + s[idx:]
open(path, "w", encoding="utf-8").write(s)
print("  + registered sensord_ch347 (optional)")