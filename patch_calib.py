#!/usr/bin/env python3
"""Camera-calibration patcher used by apply_cuda.sh (idempotent, layout-agnostic).

Brings the "camera calibration" strength of the primary tree (sunnypilot-cuda) to any
sunnypilot/openpilot fork:

  1. common/transformations/camera.py
       - injects the parameterised-intrinsics helpers
         (_read_calib_from_params / _env_or / _camera_config_from_params)
       - registers the AGX Orin GMSL entry for ("pc", "unknown")
         (fcam = road 120deg, ecam = wide/narrow) driven by the
         FcamIntrinsics / EcamIntrinsics Params, defaulting to the vendor values
         961.76 (fcam @1920x1080) and 1740.0 (ecam). Override with
         ROAD_CAM_WIDTH/HEIGHT / WIDE_CAM_WIDTH/HEIGHT env vars.
  2. common/params_keys.h
       - adds the calibration Param keys in whatever syntax that fork uses
         (`{"Key", FLAGS}` for sunnypilot-style, `{"Key", {FLAGS, TYPE}}` for
         carrot/dragonpilot-style), because Params.check_key() raises
         UnknownKeyName for anything not listed here.

The calibration *tools* themselves (tools/calib/*.py) are plain file copies done by
apply_cuda.sh; they are layout-agnostic (no hardcoded paths).

NOTE: params_keys.h is compiled into common (libcommon + params_pyx.so), so after this
patcher reports changes you MUST rebuild it in the target tree:

    source .venv/bin/activate && scons -j8 common/

Usage:
  python3 patch_calib.py <target-repo-root>
Exit codes: 0 = nothing to do, 2 = changes applied (rebuild params!), 1 = error.
"""
from __future__ import annotations

import os
import re
import sys

# ---------------------------------------------------------------- camera.py

HELPERS = '''

def _read_calib_from_params(param_key, width, height, default_fl):
  from openpilot.common.params import Params
  import json
  raw = Params().get(param_key)
  if raw is None:
    key = f"{width}x{height}"
    data = {"fl": default_fl, "cx": width / 2.0, "cy": height / 2.0,
            "calibrations": {key: {"fl": default_fl, "cx": width / 2.0, "cy": height / 2.0}}}
    Params().put(param_key, json.dumps(data))
  else:
    data = json.loads(raw)
  calibs = data.get("calibrations")
  if calibs:
    key = f"{width}x{height}"
    entry = calibs.get(key)
    if entry is not None:
      return float(entry.get("fl", default_fl)), float(entry.get("cx", width / 2.0)), float(entry.get("cy", height / 2.0))
    # No exact match: scale from first available calibration entry by width ratio
    for cal_key, cal_entry in calibs.items():
      try:
        cal_w = int(cal_key.split("x")[0])
        scale = width / cal_w
        scaled_fl = float(cal_entry.get("fl", default_fl)) * scale
        return scaled_fl, float(cal_entry.get("cx", width / 2.0)), float(cal_entry.get("cy", height / 2.0))
      except (ValueError, KeyError, IndexError):
        continue
  fl = float(data.get("fl", default_fl))
  return fl, float(data.get("cx", width / 2.0)), float(data.get("cy", height / 2.0))


def _env_or(val, env_key, cast=int):
  import os
  env = os.environ.get(env_key)
  return cast(env) if env else val


def _camera_config_from_params(width_env: str, height_env: str, base_fl: float, param_key: str, base_w: int = 1920) -> CameraConfig:
  width = _env_or(1920, width_env)
  height = _env_or(1080, height_env)
  fl = base_fl * (width / base_w)
  try:
    calib_fl, _, _ = _read_calib_from_params(param_key, width, height, fl)
    return CameraConfig(width, height, calib_fl)
  except Exception:
    return CameraConfig(width, height, fl)

'''

PC_OVERRIDE = '''
# ---- AGX Orin + Sensing/twgmsl IMX390 GMSL dual camera (PC webcamerad path) ----
#   fcam = road camera: vendor pinhole fl ~= 961.76 @1920x1080
#   ecam = the second camera (exposed as WIDE stream): vendor fl 1740.0 @1920x1080
# Live values are stored in the FcamIntrinsics / EcamIntrinsics Params
# (resolution-keyed JSON written by tools/calib/*); the numbers below are the
# fallback used the first time those Params are created.
# camerad's roadCameraState.sensor enum cannot report imx390, so the ("pc",
# "unknown") entry is replaced instead - the PC path only hits this one entry.
_imx390_fcam = _camera_config_from_params("ROAD_CAM_WIDTH", "ROAD_CAM_HEIGHT", 961.76, "FcamIntrinsics")
_imx390_ecam = _camera_config_from_params("WIDE_CAM_WIDTH", "WIDE_CAM_HEIGHT", 1740.0, "EcamIntrinsics")
DEVICE_CAMERAS[("pc", "unknown")] = DeviceCameraConfig(_imx390_fcam, _imx390_ecam, _imx390_ecam)
'''

ANCHOR_BEFORE_CONFIG = re.compile(r"^_ar_ox_fisheye = ", re.M)
ANCHOR_AFTER_UPD = re.compile(r"^DEVICE_CAMERAS\.update\(.*\)\s*$", re.M)


def patch_camera_py(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        s = f.read()
    changed = []

    if "_read_calib_from_params" not in s:
        m = ANCHOR_BEFORE_CONFIG.search(s)
        if not m:
            return [f"!! {path}: anchor '_ar_ox_fisheye = ' not found (needs manual merge)"]
        s = s[:m.start()] + HELPERS.lstrip("\n") + "\n" + s[m.start():]
        changed.append("camera.py: intrinsics helpers injected")

    if "_imx390_fcam" not in s:
        m = ANCHOR_AFTER_UPD.search(s)
        if not m:
            return [f"!! {path}: anchor 'DEVICE_CAMERAS.update(...)' not found (needs manual merge)"]
        s = s[:m.end()] + "\n" + PC_OVERRIDE + s[m.end():]
        changed.append('camera.py: ("pc","unknown") GMSL entry added')

    if changed:
        with open(path, "w", encoding="utf-8") as f:
            f.write(s)
    return changed


# ------------------------------------------------------------ params_keys.h

# name -> (flags, type)   type is only used by the typed (carrot/dragonpilot) syntax
CALIB_KEYS = [
    ("FcamIntrinsics", "PERSISTENT", "JSON"),
    ("EcamIntrinsics", "PERSISTENT", "JSON"),
    ("FcamCalibResult", "PERSISTENT", "JSON"),
    ("WideCalibResult", "PERSISTENT", "JSON"),
    ("PendingCalibReset", "PERSISTENT", "BOOL"),
    ("WideCalibActive", "CLEAR_ON_MANAGER_START", "BOOL"),
    ("WideCalibMode", "CLEAR_ON_MANAGER_START", "BOOL"),
    ("FcamLiveActive", "PERSISTENT", "BOOL"),
    ("WideCalibIntrinsicsFcamBackup", "CLEAR_ON_MANAGER_START", "JSON"),
    ("WideCalibIntrinsicsEcamBackup", "CLEAR_ON_MANAGER_START", "JSON"),
]
ENTRY_RE = re.compile(r'^\s*\{"([A-Za-z0-9_]+)",')


def detect_style(text: str) -> str:
    if re.search(r'\{"[A-Za-z0-9_]+", \{[A-Z]', text):
        return "typed"
    if re.search(r'\{"[A-Za-z0-9_]+", [A-Z]', text):
        return "flags"
    return "unknown"


def patch_params_keys(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")

    style = detect_style("\n".join(lines))
    if style == "unknown":
        return [f"!! {path}: cannot detect params_keys syntax"]

    existing = {m.group(1) for m in (ENTRY_RE.match(l) for l in lines) if m}
    missing = [(n, fl, ty) for n, fl, ty in CALIB_KEYS if n not in existing]
    if not missing:
        return []

    # keep the file's alphabetical order: insert each key before the first entry that sorts after it
    keys_in_order = [m.group(1) for m in (ENTRY_RE.match(l) for l in lines) if m]
    sorted_file = keys_in_order == sorted(keys_in_order)
    changed = []
    for name, flags, ty in missing:
        entry = f'    {{"{name}", {{{flags}, {ty}}}}},' if style == "typed" else f'    {{"{name}", {flags}}},'
        idx = None
        if sorted_file:
            for i, l in enumerate(lines):
                m = ENTRY_RE.match(l)
                if m and m.group(1) > name:
                    idx = i
                    break
        if idx is None:                      # append at the end of the map
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip() == "};":
                    idx = i
                    break
        if idx is None:
            return [f"!! {path}: closing '}};' not found"]
        lines.insert(idx, entry)
        changed.append(f"params_keys.h: +{name} ({style} syntax)")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return changed


# ------------------------------------------------------------ tools/calib/*

REALDATA_HELPER = '''

def _default_realdata():
  from openpilot.system.hardware.hw import Paths
  return Paths.log_root()
'''
HARDCODED_REALDATA = 'default="/home/dengjian/realdata"'


def patch_calib_tool(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        s = f.read()
    if HARDCODED_REALDATA not in s:
        return []
    if "_default_realdata" not in s:
        m = re.search(r"^def ", s, re.M)
        if not m:
            return [f"!! {path}: no 'def' anchor for _default_realdata"]
        s = s[:m.start()] + REALDATA_HELPER.lstrip("\n") + "\n\n" + s[m.start():]
    s = s.replace(HARDCODED_REALDATA, 'default=_default_realdata()')
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)
    return [f"{os.path.basename(path)}: --base-dir default -> device realdata (Paths.log_root())"]


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: patch_calib.py <target-repo-root>", file=sys.stderr)
        return 2
    root = os.path.abspath(sys.argv[1])
    out: list[str] = []
    # 布局候选: 老布局 common/ 在根, 新布局在 openpilot/common/
    root_candidates = [root, os.path.join(root, "openpilot"),
                       os.path.join(root, "openpilot", "sunnypilot")]
    def _resolve(rel: str) -> str | None:
        for rc in root_candidates:
            p = os.path.join(rc, rel)
            if os.path.isfile(p):
                return p
        return None
    targets = []
    for rel in ("common/transformations/camera.py", "common/params_keys.h"):
        p = _resolve(rel)
        if p is not None:
            targets.append(p)
        else:
            out.append(f"!! missing {rel} (checked {[os.path.join(rc, rel) for rc in root_candidates]})")
    if not targets:
        return 1
    for t in targets:
        if not os.path.isfile(t):
            out.append(f"!! missing {t}")
            continue
        try:
            out += patch_camera_py(t) if t.endswith("camera.py") else patch_params_keys(t)
        except Exception as e:  # noqa: BLE001
            out.append(f"!! {t}: {type(e).__name__}: {e}")

    # tools/calib/*: replace the original author's hardcoded log path with the device one
    calib_dir = os.path.join(root, "tools/calib")
    for fn in sorted(os.listdir(calib_dir)) if os.path.isdir(calib_dir) else []:
        if not fn.endswith(".py"):
            continue
        try:
            out += patch_calib_tool(os.path.join(calib_dir, fn))
        except Exception as e:  # noqa: BLE001
            out.append(f"!! {calib_dir}/{fn}: {type(e).__name__}: {e}")

    if not out:
        print("[patch_calib] already applied, nothing to do")
        return 0
    for line in out:
        print("[patch_calib] " + line)
    if any(l.startswith("!!") for l in out):
        return 1
    print("[patch_calib] done -> REBUILD the params module now:")
    print(f"[patch_calib]   cd {root} && source .venv/bin/activate && scons -j8 common/")
    return 2


if __name__ == "__main__":
    sys.exit(main())
