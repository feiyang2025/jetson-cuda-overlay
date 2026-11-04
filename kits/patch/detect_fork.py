#!/usr/bin/env python3
"""Detect what a fork's modeld expects, without importing anything.

`apply_cuda.sh` uses this to decide whether it may replace modeld.py wholesale or
must switch to the patch layer only. Whole-file replacement is safe exactly when
the fork's `ModelState.__init__` is the sunnypilot-shaped
`(cam_w, cam_h, chestnut...)`: that is the class the kit's modeld expects to
build. FrogPilot (0.9.7) and anything else get the patch layer instead, because
their ModelState carries fork-specific state (model registry, metadata pkl,
frogpilot_toggles) that a copy would silently delete.

Prints one line:  fork=<kind> modeld=<path> engine=<what it runs on>

  fork=sunnypilot-stock    stock-shaped ModelState, whole-file replacement OK
  fork=frogpilot           thneed/OpenCL ModelState(context, model, version)
  fork=unknown             unreadable or unrecognised; patch layer only

Exit code is always 0 unless no modeld.py was found at all (1).
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

MODELD_CANDIDATES = (
  "openpilot/sunnypilot/modeld_v2/modeld.py",   # sunnypilot new layout
  "openpilot/selfdrive/modeld/modeld.py",       # master-c3 / official new layout
  "selfdrive/modeld/modeld.py",                 # old layout
)


def find_modeld(tree: Path) -> Path | None:
  for rel in MODELD_CANDIDATES:
    candidate = tree / rel
    if candidate.is_file():
      return candidate
  return None


def _class_init(tree_ast, class_name: str):
  for node in ast.walk(tree_ast):
    if isinstance(node, ast.ClassDef) and node.name == class_name:
      for item in node.body:
        if isinstance(item, ast.FunctionDef) and item.name == "__init__":
          params = [a.arg for a in item.args.args]
          return params
  return None


def _engine_hint(tree_ast) -> str:
  source = ast.dump(tree_ast)
  if "commonmodel_pyx" in source and "thneed" in source:
    return "thneed-opencl"
  if "GpuModelState" in source:
    return "cuda-trt"
  if "tinygrad" in source:
    return "tinygrad"
  return "unknown"


def classify(modeld: Path) -> tuple[str, str]:
  try:
    source = modeld.read_text(encoding="utf-8", errors="ignore")
    tree_ast = ast.parse(source)
  except (OSError, SyntaxError):
    return "unknown", "unknown"

  params = _class_init(tree_ast, "ModelState") or []
  engine = _engine_hint(tree_ast)

  if len(params) >= 2 and params[1] in ("cam_w", "width", "cam_h", "height"):
    return "sunnypilot-stock", engine
  if len(params) >= 3 and params[1] == "context":
    return "frogpilot", engine
  return "unknown", engine


def main() -> int:
  tree = Path(sys.argv[1] if len(sys.argv) > 1 else ".").expanduser().resolve()
  modeld = find_modeld(tree)
  if modeld is None:
    print(f"fork=none modeld=- engine=- (no modeld.py under {tree})")
    return 1
  kind, engine = classify(modeld)
  print(f"fork={kind} modeld={modeld.relative_to(tree)} engine={engine}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
