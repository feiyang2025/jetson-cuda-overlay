#!/usr/bin/env python3
"""Install the CUDA/TensorRT path into a FrogPilot tree.

What FrogPilot needs on AGX Orin, and what this installer changes:

1. **Which processes run.** FrogPilot 0.9.7 starts its model stack as C++
   binaries (`NativeProcess "modeld"`, `NativeProcess "classic_modeld"`,
   `NativeProcess "camerad"`). None of them build or run on Orin (OpenCL/QCOM).
   They are rewritten to Python processes:
     modeld / classic_modeld -> frogpilot...tinygrad.fp_modeld_main
     camerad                 -> openpilot.tools.webcam.camerad (V4L2/DMABUF kit)
   The fork's own process gating (`run_new_modeld`, `run_classic_modeld`, ...)
   is preserved, so model selection still behaves as FrogPilot intends.

2. **The engine.** `fp_modeld_gpu.py` (from kits/patch/frogpilot/fp_gpu_state.py)
   wraps the overlay's CUDA/TRT ModelState and exposes the four members
   FrogPilot's main() uses. `fp_modeld_main.py` is FrogPilot's own tinygrad main
   loop with only the model construction swapped, so behaviour is unchanged.

3. **The model bundle.** FrogPilot-Resources ships thneed/tinygrad artifacts
   that cannot run here. `--models-from <dir>` links a compiled overlay bundle
   (models/<Name>/driving_*_fp16.plan + metadata) into the fork, which is what
   the CUDA state loads.

Backups are written next to every file it replaces (`.before_fp_cuda`).
Idempotent: re-running after `git pull` restores the patched state.

Usage:
  python3 install_fp.py <frogpilot-tree> [--models-from <dir>] [--bundle Name] [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

BACKUP = ".before_fp_cuda"

CAMERAD_KIT_FILES = ("camerad.py", "v4l2_dmabuf_camera.py", "v4l2_camera.py", "camera_cuda.py",
                     "packed_to_nv12.cu", "nvbuf_import.cu")

# Exact process_config.py rewrite table: (old line fragment, new line).
# Anchored on the fork's real lines so a changed upstream line is skipped loudly
# instead of being mangled.
PROCESS_REWRITES = (
  (
    'NativeProcess("modeld", "selfdrive/modeld", ["./modeld"], run_new_modeld)',
    'PythonProcess("modeld", "frogpilot.selfdrive.modeld.tinygrad.fp_modeld_main", run_new_modeld)',
  ),
  (
    'NativeProcess("classic_modeld", "frogpilot/selfdrive/modeld/classic", ["./classic_modeld"], run_classic_modeld)',
    'PythonProcess("classic_modeld", "frogpilot.selfdrive.modeld.tinygrad.fp_modeld_main", run_classic_modeld)',
  ),
  (
    'NativeProcess("camerad", "system/camerad", ["./camerad"], driverview)',
    'PythonProcess("camerad", "openpilot.tools.webcam.camerad", driverview, enabled=PC)',
  ),
  (
    'PythonProcess("tinygrad_modeld", "frogpilot.selfdrive.modeld.tinygrad.tinygrad_modeld", run_tinygrad_modeld)',
    'PythonProcess("tinygrad_modeld", "frogpilot.selfdrive.modeld.tinygrad.fp_modeld_main", run_tinygrad_modeld)',
  ),
)


def pkg_root(tree: Path) -> Path:
  """Directory that holds the `openpilot` package (tree itself, usually)."""
  if (tree / "openpilot" / "selfdrive").is_dir():
    return tree
  return tree


def frogpilot_dir(tree: Path) -> Path | None:
  # Real directory first: FrogPilot's `openpilot/frogpilot` is a symlink back to
  # `frogpilot/`, so editing through it is fine but paths in logs are clearer.
  for candidate in (tree / "frogpilot", tree / "openpilot" / "frogpilot"):
    if candidate.is_dir():
      return candidate
  return None


def _install_file(src: Path, dst: Path, changed: list[str], dry: bool) -> None:
  if not src.is_file():
    raise FileNotFoundError(f"kit file missing: {src}")
  if dst.is_file():
    if dst.read_bytes() == src.read_bytes():
      print(f"  - {dst} 已是最新, 不动")
      return
    if not dry:
      shutil.copy2(dst, dst.with_name(dst.name + BACKUP))
    action = "updated"
  else:
    action = "installed"
  if dry:
    print(f"  ~ would {action}: {dst}")
    return
  dst.parent.mkdir(parents=True, exist_ok=True)
  shutil.copy2(src, dst)
  print(f"  + {action}: {dst}")
  changed.append(str(dst))


def patch_process_config(tree: Path, dry: bool) -> bool:
  pconfig = None
  for candidate in (tree / "openpilot" / "system" / "manager" / "process_config.py",
                    tree / "system" / "manager" / "process_config.py"):
    if candidate.is_file():
      pconfig = candidate
      break
  if pconfig is None:
    print("  [WARN] process_config.py 未找到, 跳过进程替换")
    return False

  text = pconfig.read_text(encoding="utf-8")
  original = text
  applied = 0
  skipped = []

  for old, new in PROCESS_REWRITES:
    if new in text:
      continue                      # already rewritten
    if old in text:
      text = text.replace(old, new)
      applied += 1
    else:
      skipped.append(old.split(",")[0] + "...")

  if text == original:
    print(f"  - process_config.py 无需改动 ({'已在目标状态' if applied == 0 else ''})")
    return True

  if dry:
    print(f"  ~ would rewrite {applied} process line(s) in {pconfig}")
    return True

  shutil.copy2(pconfig, pconfig.with_name(pconfig.name + BACKUP))
  pconfig.write_text(text, encoding="utf-8")
  print(f"  + process_config.py: 改写 {applied} 处进程定义 (备份 {BACKUP})")
  for s in skipped:
    print(f"    [SKIP] 锚点未匹配: {s}  (上游改过? 手工处理)")
  return True


def link_models(tree: Path, models_from: Path | None, bundle: str | None, dry: bool) -> bool:
  if models_from is None:
    print("  - 未给 --models-from: 跳过模型目录 (可稍后手工放置或设 FP_CUDA_MODEL_DIR)")
    return True
  if not models_from.is_dir():
    print(f"  [WARN] --models-from 不存在: {models_from}")
    return False

  dst_root = tree / "openpilot" / "selfdrive" / "modeld" / "models"
  if not (tree / "openpilot").is_dir():
    dst_root = tree / "selfdrive" / "modeld" / "models"
  dst_root.mkdir(parents=True, exist_ok=True)

  bundles = [bundle] if bundle else [p.name for p in sorted(models_from.iterdir())
                                     if p.is_dir() and not p.name.startswith((".", "__"))]
  # Only the split bundle is worth linking by default: it is what the CUDA state
  # runs as FrogPilot's small model. Linking every bundle would silently make the
  # registry pick whichever sorts first.
  if not bundle:
    preferred = "FiletOFish"
    bundles = [preferred] if (models_from / preferred).is_dir() else bundles[:1]
  linked = 0
  for name in bundles:
    if name is None:
      continue
    src = models_from / name
    if not src.is_dir():
      print(f"  [WARN] bundle 不存在: {src}")
      continue
    dst = dst_root / name
    if dst.exists():
      print(f"  - models/{name} 已存在, 不动")
      continue
    if dry:
      print(f"  ~ would link: {dst} -> {src}")
      continue
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
      # Hardlink: the .plan files are hundreds of MB and never edited in place,
      # and both trees live on the same filesystem here.
      dst.symlink_to(src, target_is_directory=True)
      print(f"  + models/{name} -> {src} (symlink)")
    except OSError as e:
      shutil.copytree(src, dst)
      print(f"  + models/{name} 已拷贝 (symlink 失败: {e})")
    linked += 1
  print(f"  -> 模型目录: {dst_root} ({linked} 个 bundle)")
  return True


def main() -> int:
  ap = argparse.ArgumentParser(description="Install the CUDA/TensorRT path into a FrogPilot tree")
  ap.add_argument("tree", help="FrogPilot checkout root")
  ap.add_argument("--models-from", help="overlay models dir holding models/<Name>/ with .plan engines")
  ap.add_argument("--bundle", help="only link this bundle name")
  ap.add_argument("--dry-run", action="store_true")
  args = ap.parse_args()

  tree = Path(args.tree).expanduser().resolve()
  kit = Path(__file__).resolve().parent
  print(f"[install_fp] tree={tree}")

  fp = frogpilot_dir(tree)
  if fp is None:
    print("ERROR: 没有 frogpilot/ 目录, 这不是 FrogPilot 树")
    return 1

  tinygrad_dir = fp / "selfdrive" / "modeld" / "tinygrad"
  if not tinygrad_dir.is_dir():
    print(f"ERROR: {tinygrad_dir} 不存在 (FrogPilot 版本不含 tinygrad modeld)")
    return 1

  changed: list[str] = []

  # 1. the two Python modules FrogPilot's modeld becomes
  _install_file(kit / "fp_gpu_state.py", tinygrad_dir / "fp_modeld_gpu.py", changed, args.dry_run)
  _install_file(kit / "fp_modeld_main.py", tinygrad_dir / "fp_modeld_main.py", changed, args.dry_run)

  # 2. camera adapter (V4L2/DMABUF + twgmsl chroma) into the fork's webcam dir
  cam_dst = None
  for candidate in (tree / "openpilot" / "tools" / "webcam", tree / "tools" / "webcam"):
    if candidate.is_dir():
      cam_dst = candidate
      break
  if cam_dst is None:
    print("  [WARN] 无 tools/webcam 目录, 跳过相机适配")
  else:
    cam_kit = Path(__file__).resolve().parent.parent / "camerad"
    for f in CAMERAD_KIT_FILES:
      src = cam_kit / f
      if src.is_file():
        _install_file(src, cam_dst / f, changed, args.dry_run)

  # 3. process_config: C++ binaries -> Python processes
  patch_process_config(tree, args.dry_run)

  # 4. model bundle
  link_models(tree, Path(args.models_from).expanduser() if args.models_from else None,
              args.bundle, args.dry_run)

  print(f"[install_fp] done. changed={len(changed)}")
  if not args.dry_run:
    print("\nNext:")
    print("  1. 编 CUDA 变换库 (每台一次):")
    print(f"       cd {tree}/openpilot/selfdrive/modeld/transforms && nvcc -arch=sm_87 -shared -O2 -o libcuda_transform.so cuda_transform.cu -I.")
    print("  2. Params 里确认 Model 指向合法 bundle, 或设环境变量 FP_CUDA_MODEL_DIR")
    print("  3. 启动后用 logcat 看 'frogpilot modeld_gpu init' 与 modelV2 是否发布")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())