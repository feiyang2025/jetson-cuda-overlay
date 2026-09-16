#!/usr/bin/env python3
"""Adapter exposing the V4L2 + VIC (UYVY -> NV12) camera as the new-layout
webcam Camera interface (read_frames() yields raw NV12 bytes).

Lives under openpilot/system/camerad/webcam/ so the stock camerad.py can use
it without changes:  NARROW_ROAD_CAM / WIDE_CAM select /dev/video* devices.
"""
import numpy as np

from openpilot.system.camerad.webcam.v4l2_dmabuf_camera import V4L2Camera


class Camera:
  def __init__(self, cam_type_state, stream_type, camera_id):
    try:
      camera_id = int(camera_id)
    except ValueError:
      pass
    # openpilot passes a numeric camera index in some forks; turn it into a
    # device node so V4L2Camera's os.open() gets a path (int would crash).
    if isinstance(camera_id, int):
      camera_id = f"/dev/video{camera_id}"
    self.cam_type_state = cam_type_state
    self.stream_type = stream_type
    self.cur_frame_id = 0
    self.cam = V4L2Camera(camera_id, width=V4L2Camera.OUTPUT_WIDTH, height=V4L2Camera.OUTPUT_HEIGHT, fps=20)
    self.W = self.cam.W
    self.H = self.cam.H

  def read_frames(self):
    for buf in self.cam.read_frames():
      data = buf.data
      if hasattr(data, 'tobytes'):
        data = data.tobytes()
      yield bytes(data)
      self.cur_frame_id = buf.frame_id

  def close(self):
    if getattr(self, 'cam', None) is not None:
      self.cam.close()

  def __del__(self):
    try:
      self.close()
    except Exception:
      pass
