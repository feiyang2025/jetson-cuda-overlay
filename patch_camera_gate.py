#!/usr/bin/env python3
"""Add the explicit T906G/desktop camera-only gate override.

SP_FORCE_CAMERA=1 enables camera processes while the device is offroad/no ignition.
Default behavior is unchanged. This is intentionally a process-config gate only;
it does not fake ignition, car state, or onroad controls.
"""
import os
import sys


def main():
    root = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
    candidates = [
        os.path.join(root, "openpilot/system/manager/process_config.py"),
        os.path.join(root, "system/manager/process_config.py"),
    ]
    path = next((p for p in candidates if os.path.isfile(p)), None)
    if path is None:
        print("  - camera gate: process_config.py not found (skip)")
        return 0
    text = open(path, encoding="utf-8").read()
    old = 'return started or params.get_bool("IsDriverViewEnabled")'
    new = old + ' or os.getenv("SP_FORCE_CAMERA") == "1"'
    if 'os.getenv("SP_FORCE_CAMERA")' in text:
        print(f"  - {path}: SP_FORCE_CAMERA already present")
        return 0
    if old not in text:
        print(f"  - {path}: driverview anchor not found (skip)")
        return 0
    text = text.replace(old, new, 1)
    open(path, "w", encoding="utf-8").write(text)
    print(f"  + {path}: SP_FORCE_CAMERA=1 camera-only gate added")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
