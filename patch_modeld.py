#!/usr/bin/env python3
"""Cross-platform patcher used by apply_cuda.sh.

Injects ``_make_model()`` before ``def main(demo=False):`` and rewrites every
``ModelState(cam_w=..., cam_h=..., chestnut=...)`` call to
``_make_model(...)`` so the CUDA/TRT backend is used by default on AGX Orin,
with tinygrad fallback.

Idempotent: safe to re-run after ``git pull`` upstream.

Usage:
  python3 patch_modeld.py <modeld.py>
"""
import sys


HELPER = (
  'def _make_model(cam_w: int, cam_h: int, chestnut: bool = False):\n'
  '  if not chestnut and os.getenv("DISABLE_CUDA_BACKEND", "0") != "1":\n'
  '    try:\n'
  '      from openpilot.sunnypilot.modeld_v2.gpu_model_state import GpuModelState\n'
  '      model = GpuModelState(cam_w=cam_w, cam_h=cam_h, chestnut=False)\n'
  '      cloudlog.warning("Using GpuModelState (CUDA/TensorRT)")\n'
  '      return model\n'
  '    except Exception:\n'
  '      cloudlog.exception("GpuModelState init failed, falling back to tinygrad ModelState")\n'
  '  return ModelState(cam_w=cam_w, cam_h=cam_h, chestnut=chestnut)\n\n\n'
)


def main() -> int:
  if len(sys.argv) != 2:
    print("usage: patch_modeld.py <modeld.py>", file=sys.stderr)
    return 2
  path = sys.argv[1]
  with open(path, encoding="utf-8") as f:
    s = f.read()

  changed = False
  old = "ModelState(cam_w=vipc_client_main.width, cam_h=vipc_client_main.height, chestnut="
  new = "_make_model(cam_w=vipc_client_main.width, cam_h=vipc_client_main.height, chestnut="

  # Rewrite only the original source before adding the helper, so its fallback
  # ModelState call is never rewritten recursively.
  if "def _make_model" not in s:
    call_sites = s.count(old)
    if call_sites:
      s = s.replace(old, new)
      changed = True
      print(f"  + rewrote {call_sites} call site(s)")
    marker = "def main(demo=False):"
    idx = s.find(marker)
    if idx < 0:
      print("ERROR: 'def main(demo=False):' not found", file=sys.stderr)
      return 2
    s = s[:idx] + HELPER + s[idx:]
    changed = True
    print("  + injected _make_model")

  if changed:
    with open(path, "w", encoding="utf-8") as f:
      f.write(s)
  else:
    print("  - already patched")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())