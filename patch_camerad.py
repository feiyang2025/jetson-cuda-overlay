#!/usr/bin/env python3
import sys

path = sys.argv[1]
s = open(path, encoding="utf-8").read()
old = "from openpilot.system.camerad.webcam.camera import Camera"
new = '''if os.getenv("USE_V4L2_CAMERA", "1") == "1" and platform.system() != "Darwin":
  try:
    from openpilot.system.camerad.webcam.v4l2_camera import Camera
  except Exception:
    from openpilot.system.camerad.webcam.camera import Camera
else:
  from openpilot.system.camerad.webcam.camera import Camera'''
if "v4l2_camera import Camera" not in s:
  if old not in s:
    raise SystemExit("camera import line not found")
  s = s.replace(old, new, 1)
  open(path, "w", encoding="utf-8").write(s)
  print("camera.py patched")
else:
  print("camera.py already patched")