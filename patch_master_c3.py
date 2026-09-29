#!/usr/bin/env python3
"""master-c3 (carrot 系) fork 的 AGX Orin 收口补丁器 —— 幂等。

apply_cuda.sh 覆盖不到、但本机 master-c3 树必需的 6 处改动：
  1. SConstruct        : env 加 HOME / PARAMS_ROOT (Path::comma_home() 拼出 /.comma 会导致 errno=13 编译失败)
  2. launch_chffrplus.sh: setup_python_path 追加树内 .venv site-packages (uv 装的依赖否则找不到 capnp)
  3. launch_env.sh     : 追加 AGX 实验环境默认开关 (USE_WEBCAM/ROAD_CAM/WIDE_CAM/CAM_*/BIG/ENABLE_FAKE_PANDA)
  4. pc/hardware.h     : 补 get_voltage/get_current (pandad.cc 读 hwmon, PC 无此硬件, 缺了编译失败)
  5. ui/onroad/cameraview.py: NV12 通道采样 .ra -> .rg (官方 shader 会绿屏)
  6. manager/process_config.py: webcamerad restart_if_crash=True

ch347t.cc 的 master-c3 cereal schema 适配不走这里 —— 它直接改在 overlay 自己的
openpilot/system/sensord/ch347t.cc 里（注释掉 schema 已删除的 setVersion/setSensor/setType/setStatus）。

每处都按"锚点是否存在"决定是否动手，锚点不在就 SKIP，不会碰其他 fork 的写法。
用法: python3 patch_master_c3.py <树根>   （apply_cuda.sh 末尾会自动调用）
"""
import re
import sys
from pathlib import Path

MARK = "AGX Orin 适配"
ENV_MARK = "AGX Orin 实验环境默认"


def edit(path: Path, fn):
  """fn(text) -> new_text | None(不改)"""
  if not path.exists():
    return f"SKIP  {path.name}: 文件不存在"
  src = path.read_text()
  out = fn(src)
  if out is None:
    return f"   -  {path.name}: 已就绪"
  path.write_text(out)
  return f"   +  {path.name}"


def sconstruct(text: str):
  if "PARAMS_ROOT" in text:
    return None
  old = '    "TERA_PATH": acados.TERA_PATH\n'
  if old not in text:
    return None
  new = ('    "TERA_PATH": acados.TERA_PATH,\n'
         '    # %s: 编译期 C++ 工具 (params/acados) 需要 HOME 定位 ~/.comma;\n'
         '    # 否则 Path::comma_home() 拼出 /.comma, errno=13 编译失败\n'
         '    "HOME": os.environ.get("HOME", ""),\n'
         '    "PARAMS_ROOT": os.environ.get("PARAMS_ROOT", os.path.join(os.environ.get("HOME", ""), ".comma/params")),\n'
         % MARK)
  return text.replace(old, new, 1)


def launch_chffrplus(text: str):
  if ".venv/lib/python3.12/site-packages" in text:
    return None
  anchor = '  [ -d "$venv_site" ] && py_path="$py_path:$venv_site"\n'
  if anchor not in text:
    return None
  add = ('  # AGX/Jetson: 依赖装在树内 .venv (uv), 必须加入搜索路径, 否则 capnp 等找不到\n'
         '  [ -d "$root/.venv/lib/python3.12/site-packages" ] && py_path="$py_path:$root/.venv/lib/python3.12/site-packages"\n')
  return text.replace(anchor, anchor + add, 1)


AGX_ENV_BLOCK = '''
# ---- %s (master-c3 适配): 相机/USE_WEBCAM/全屏/输出尺寸 ----
# 全部用 ${VAR:-default} 形式, 命令行显式赋值可覆盖。
# USE_WEBCAM=1 -> webcamerad(Python) 替代原生 camerad 二进制(未编译, FileNotFoundError)
# CAM_WIDTH/HEIGHT=1344x760 -> 原生 modeld 只认编译过的 1928x1208 / 1344x760
# BIG=1 -> raylib UI 2160x1080 自动缩放全屏 (默认是 536x240 小窗)
# ENABLE_FAKE_PANDA=1 -> 无 panda 卡时用假 panda (ignition/CAN)
export USE_WEBCAM="${USE_WEBCAM:-1}"
export ROAD_CAM="${ROAD_CAM:-0}"
export WIDE_CAM="${WIDE_CAM:-1}"
export CAM_WIDTH="${CAM_WIDTH:-1344}"
export CAM_HEIGHT="${CAM_HEIGHT:-760}"
export BIG="${BIG:-1}"
export ENABLE_FAKE_PANDA="${ENABLE_FAKE_PANDA:-1}"
''' % ENV_MARK


def launch_env(text: str):
  if ENV_MARK in text:
    return None
  if 'export STAGING_ROOT="/data/safe_staging"' not in text:
    return None
  return text.rstrip("\n") + "\n" + AGX_ENV_BLOCK


def hardware_h(text: str):
  if "get_voltage" in text:
    return None
  anchor = "  static bool PC() { return true; }\n};"
  if anchor not in text:
    return None
  add = ('  static bool PC() { return true; }\n'
         '  // %s: pandad.cc 读 hwmon 电压/电流, PC 无此硬件, 返回 0 触发\n'
         '  // pandad 的 fallback (用 panda 自测电压电流)。缺这两个方法会编译失败。\n'
         '  static int get_voltage() { return 0; }\n'
         '  static int get_current() { return 0; }\n};' % MARK)
  return text.replace(anchor, add, 1)


def cameraview(text: str):
  # 曾经把上游的 .ra 改成 .rg 并注释"AGX Orin 适配 / .ra 会绿屏" —— 那是错的。
  # 本机 raylib 的色度纹理走 GL_LUMINANCE_ALPHA: .r 与 .g **都等于第一个字节(=U)**,
  # 所以 .rg 取到的是 (U,U), 代进 BT.601 的后果是精确的
  #   蓝 → 品红紫 (R/B 同涨、G 掉)  橘黄 → 绿 (R/B 同降、G 涨)
  # 正确取法是 .r=U / .a=V, 即上游原版的 .ra。这条只做"把错的改回来",
  # 上游本来就是 .ra 时返回 None(幂等)。
  bad = "vec2 uv = texture(texture1, fragTexCoord).rg - 0.5;  // %s: NV12 通道顺序, .ra 会绿屏" % MARK
  if bad in text:
    return text.replace(bad, "vec2 uv = texture(texture1, fragTexCoord).ra - 0.5;", 1)
  return None


def process_config(text: str):
  old = 'PythonProcess("webcamerad", "openpilot.system.camerad.webcam.camerad", driverview, enabled=WEBCAM)'
  # 用整行做锚点: 别用全局 "restart_if_crash=True" 判断, 树里别的进程早就有这个参数了
  if old not in text:
    return None
  new = ('PythonProcess("webcamerad", "openpilot.system.camerad.webcam.camerad", driverview, '
         'enabled=WEBCAM, restart_if_crash=True)')
  return text.replace(old, new, 1)


# ------------------------------------------------ developer 页的摄像头标定按钮

CALIB_IMPORT_ANCHOR = "from openpilot.system.ui.sunnypilot.widgets.list_view import toggle_item_sp\n"
CALIB_IMPORT = ("from openpilot.selfdrive.ui.sunnypilot.calibration_panel import CalibrationPanel"
                "  # %s: 摄像头标定 4 按钮\n" % MARK)

CALIB_ITEMS_ANCHOR = ('    self.items: list = [self.show_advanced_controls, self.enable_github_runner_toggle, '
                      'self.enable_copyparty_toggle, self.prebuilt_toggle, self.error_log_btn,]\n')
CALIB_ITEMS = ('    # %s: 摄像头标定 (FCAM RUN / FCAM Live / Wide / VIEW)\n'
               '    # 实现见 openpilot/selfdrive/ui/sunnypilot/calibration_panel.py\n'
               '    self._calib_panel = CalibrationPanel(ui_state.params)\n'
               '    self.items.extend(self._calib_panel.items.values())\n' % MARK)

CALIB_UPDATE_ANCHOR = "    self.error_log_btn.set_visible(not self._is_release_branch)\n"
CALIB_UPDATE = "    self._calib_panel.update()\n"


def developer_py(text: str):
  """把标定面板接进开发者页 (幂等)。"""
  if "CalibrationPanel" in text:
    return None
  if CALIB_ITEMS_ANCHOR not in text or CALIB_UPDATE_ANCHOR not in text:
    return None
  out = text
  if CALIB_IMPORT_ANCHOR in out:
    out = out.replace(CALIB_IMPORT_ANCHOR, CALIB_IMPORT_ANCHOR + CALIB_IMPORT, 1)
  else:
    m = re.search(r"^(?:from|import) .*\n", out, re.M)
    if not m:
      return None
    out = out[:m.start()] + CALIB_IMPORT + out[m.start():]
  out = out.replace(CALIB_ITEMS_ANCHOR, CALIB_ITEMS_ANCHOR + CALIB_ITEMS, 1)
  out = out.replace(CALIB_UPDATE_ANCHOR, CALIB_UPDATE_ANCHOR + CALIB_UPDATE, 1)
  return out


def main():
  root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
  print("[patch_master_c3] target=%s" % root)
  targets = [
    (root / "SConstruct", sconstruct),
    (root / "openpilot" / "SConstruct", sconstruct),
    (root / "launch_chffrplus.sh", launch_chffrplus),
    (root / "launch_env.sh", launch_env),
    (root / "openpilot" / "common" / "hardware" / "pc" / "hardware.h", hardware_h),
    (root / "openpilot" / "selfdrive" / "ui" / "onroad" / "cameraview.py", cameraview),
    (root / "openpilot" / "system" / "manager" / "process_config.py", process_config),
    (root / "openpilot" / "selfdrive" / "ui" / "sunnypilot" / "layouts" / "settings" / "developer.py", developer_py),
  ]
  for path, fn in targets:
    try:
      print(edit(path, fn))
    except Exception as e:
      print("   !  %s: %s" % (path.name, e))
      raise


if __name__ == "__main__":
  main()
