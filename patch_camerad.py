#!/usr/bin/env python3
import sys

path = sys.argv[1]
s = open(path, encoding="utf-8").read()

# old-layout forks use tools/webcam/camera.py; new-layout use
# openpilot/system/camerad/webcam/camera.py. Handle both so the V4L2
# camera is never silently skipped on the old layout.
variants = [
  ("from openpilot.system.camerad.webcam.camera import Camera",
   "openpilot.system.camerad.webcam.camera",
   "openpilot.system.camerad.webcam.v4l2_camera"),
  ("from tools.webcam.camera import Camera",
   "tools.webcam.camera",
   "tools.webcam.v4l2_camera"),
]

def build_block(stock: str, v4l2: str) -> str:
  return f'''if os.getenv("USE_V4L2_CAMERA", "1") == "1" and platform.system() != "Darwin":
  try:
    from {v4l2} import Camera
  except Exception:
    from {stock} import Camera
else:
  from {stock} import Camera'''

if "v4l2_camera import Camera" in s:
  print("camera.py already patched")
  raise SystemExit(0)

for old, stock, v4l2 in variants:
  if old in s:
    s = s.replace(old, build_block(stock, v4l2), 1)
    open(path, "w", encoding="utf-8").write(s)
    print("camera.py patched")
    raise SystemExit(0)

raise SystemExit("camera import line not found")