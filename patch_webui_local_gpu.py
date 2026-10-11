#!/usr/bin/env python3
"""Make the carrot web UI tell the truth on this port: local GPU + locally-available big model.

Idempotent, anchor-based. Four files, five edits. Preserves CRLF line endings byte-for-byte
(the webui sources are CRLF; plain text-mode read/write would silently rewrite the whole file).

Why:
  * The "eGPU / Big Model" card is upstream's Chestnut concept. `_egpu_state()` bailed out with
    `return None` when `chestnutPresent` is false, so on a Jetson the card rendered
    "eGPU 未激活 / 未检测到 eGPU（Chestnut）硬件" even though the Orin's own GPU runs all the
    TRT inference and the BigCombo engine (1.76 GB) is already on disk. It looked like the
    hardware had not been detected at all.
  * The Bluetooth card reported "未检测到蓝牙射频硬件" because `carrot_bluetooth` was gated
    `enabled=COMMA_HARDWARE` (AGNOS only) and therefore never wrote
    `/dev/shm/carrot-bluetooth/status.json`, which is what the card reads. The daemon itself
    works fine on Ubuntu over BlueZ D-Bus, so the gate is lifted here.

Usage: python3 patch_webui_local_gpu.py <repo-root>
"""
import os
import sys

PC = 'openpilot/system/manager/process_config.py'
SA = 'webui/server/bridge/state_api.py'
PJ = 'webui/web/static/js/panels.js'
I18N = 'webui/web/static/js/i18n.js'

PC_OLD = ('  PythonProcess("carrot_bluetooth", "openpilot.sunnypilot.carrot.bluetooth.daemon", '
          'always_run, enabled=COMMA_HARDWARE, restart_if_crash=True),')
PC_NEW = ('''  # 本移植: 门控从 COMMA_HARDWARE 放开 —— AGX Orin 上(Ubuntu)也带蓝牙, 守护进程
  # 走 BlueZ D-Bus(system bus) + /data/bluetooth/ENABLED, 不依赖 AGNOS 的 evdev 节点。
  # 不放开时: 守护进程不启动 → 不写 /dev/shm/carrot-bluetooth/status.json →
  # 网页"蓝牙"卡片报"未检测到蓝牙射频硬件"(而 hciconfig/rfkill 显示硬件一切正常)。
''' + PC_OLD.replace(', enabled=COMMA_HARDWARE', ''))

SA_OLD = '''def _egpu_state(ds, started: bool, sm=None) -> dict[str, Any] | None:
  """Mirror sunnypilot sidebarSP _get_home_icon eGPU logic."""
  try:
    from openpilot.common.params import Params
    p = Params()
    chestnut_present = bool(ds.chestnutPresent) if hasattr(ds, "chestnutPresent") else False
    if not chestnut_present:
      return None'''

SA_NEW = '''def _big_model_available() -> bool:
  """本机是否已备好大模型引擎 (BigCombo 的 TRT plan)。任何异常都当"没有"。"""
  try:
    from openpilot.selfdrive.modeld.helpers import MODELS_DIR
    return (MODELS_DIR / 'BigCombo' / 'driving_supercombo_fp16.plan').exists()
  except Exception:
    return False


def _egpu_state(ds, started: bool, sm=None) -> dict[str, Any] | None:
  """Mirror sunnypilot sidebarSP _get_home_icon eGPU logic."""
  try:
    from openpilot.common.params import Params
    p = Params()
    chestnut_present = bool(ds.chestnutPresent) if hasattr(ds, "chestnutPresent") else False
    if not chestnut_present:
      # 本移植(非 comma 硬件): 没有 Chestnut 也有 Orin 自带 GPU, 大模型引擎也就在本机
      # (是否启用由 `Model` 参数决定)。原逻辑这里直接 return None, 卡片就落成
      # "eGPU 未激活 / 未检测到硬件" —— 让人误以为本机 GPU 没被识别。如实报告本机情况。
      try:
        from openpilot.common.hardware import COMMA_HARDWARE
      except Exception:
        COMMA_HARDWARE = True
      if COMMA_HARDWARE:
        return None  # comma 机器上保持上游语义
      try:
        model_name = p.get("Model")
      except Exception:
        model_name = None
      if isinstance(model_name, (bytes, bytearray)):
        model_name = bytes(model_name).decode("utf-8", "replace")
      model_name = model_name or "FiletOFish"   # 本 fork 的 Params.get 不吃 encoding= 参数
      big_available = _big_model_available()
      selected = model_name == "BigCombo"
      return {
        "state": "green" if (selected and big_available) else "gray",
        "localGpu": True,
        "bigModel": model_name,
        "bigModelAvailable": big_available,
        "bigModelSelected": selected,
        "chestnutExpected": False,
      }'''

PJ_OLD_WARN = '''  // Chestnut info
  if (!globalState?.device?.chestnutPresent) {'''
PJ_NEW_WARN = '''  // Chestnut info —— 只在"该有 eGPU"的平台上提示; 本移植在非 comma 硬件上改报本地 GPU
  const chestnutExpected = st ? st.chestnutExpected !== false : true;
  if (chestnutExpected && !globalState?.device?.chestnutPresent) {'''

PJ_OLD_ROWS = '''    { label: t("Description"), value: t(EGPU_STATE_DESCS[state] || "") },
  ];'''
PJ_NEW_ROWS = '''    { label: t("Description"), value: t(EGPU_STATE_DESCS[state] || "") },
  ];

  // 本移植: 没有 Chestnut 时如实报告本机 GPU 与大模型(BigCombo)实况, 免得看着像"硬件没认出来"
  if (st?.localGpu) {
    stateItems.push({ label: t("Local GPU"), value: t("Active (TensorRT FP16)") });
    stateItems.push({
      label: t("Big Model"),
      value: `${st.bigModel || ""} · ${st.bigModelSelected ? t("Selected") : t("Not selected (default model running)")}`,
    });
    stateItems.push({ label: t("Big Model Engine"), value: st.bigModelAvailable ? t("Available locally") : t("Not found") });
    stateItems.push({
      label: t("Note"),
      value: t("This port runs TRT inference on the Orin's own GPU; eGPU/Chestnut is a separate external GPU. The big model is switched by the Model param (FiletOFish / BigCombo)."),
    });
  }'''

# 两张语言表各自唯一的两行锚点 (简体=4 空格缩进 / 繁体=2 空格) —— 不能用纯缩进区分,
# 因为 '  ' 是 '    ' 的子串, 会串表。
ZH_ANCHOR = '    "Current State": "当前状态",\n    "eGPU Available": "eGPU 可用",'
TW_ANCHOR = '  "Current State": "當前狀態",\n  "eGPU Available": "eGPU 可用",'

I18N_KEYS = [
    ('Local GPU', '本地 GPU', '本機 GPU'),
    ('Active (TensorRT FP16)', '已启用（TensorRT FP16）', '已啟用（TensorRT FP16）'),
    # ('Big Model', ...) 已在两张语言表里存在(模型选择 UI 早就翻译过), 不再重复添加
    ('Selected', '已选中', '已選中'),
    ('Not selected (default model running)', '未选中（当前跑默认模型）', '未選中（目前跑預設模型）'),
    ('Big Model Engine', '大模型引擎', '大模型引擎'),
    ('Available locally', '本机已有', '本機已有'),
    ('Not found', '未找到', '未找到'),
    ('Note', '说明', '說明'),
    ('This port runs TRT inference on the Orin\'s own GPU; eGPU/Chestnut is a separate external GPU. '
     'The big model is switched by the Model param (FiletOFish / BigCombo).',
     '本移植用 Orin 自带 GPU 跑 TRT 推理；eGPU/Chestnut 是另一块外接 GPU。大模型由 Model 参数切换（FiletOFish / BigCombo）。',
     '本移植用 Orin 自帶 GPU 跑 TRT 推理；eGPU/Chestnut 是另一塊外接 GPU。大模型由 Model 參數切換（FiletOFish / BigCombo）。'),
]


def _read(p):
  # newline='' → 不做换行翻译, 保住原文件的 CRLF
  return open(p, encoding='utf-8', newline='').read()


def _write(p, s):
  open(p, 'w', encoding='utf-8', newline='').write(s)


def edit(path, old, new, sentinel):
  s = _read(path)
  name = os.path.basename(path)
  if sentinel in s:
    return '  - %s: already patched' % name
  if old in s:
    pass
  elif old.replace('\n', '\r\n') in s:
    old, new = old.replace('\n', '\r\n'), new.replace('\n', '\r\n')
  else:
    return '!! %s: anchor not found -> SKIP (patch by hand)' % name
  _write(path, s.replace(old, new, 1))
  return '  + %s patched' % name


def main():
  root = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else '.')
  out = [edit(os.path.join(root, PC), PC_OLD, PC_NEW, '门控从 COMMA_HARDWARE 放开'),
         edit(os.path.join(root, SA), SA_OLD, SA_NEW, 'chestnutExpected'),
         edit(os.path.join(root, PJ), PJ_OLD_WARN, PJ_NEW_WARN, 'chestnutExpected'),
         edit(os.path.join(root, PJ), PJ_OLD_ROWS, PJ_NEW_ROWS, 'st?.localGpu')]

  p = os.path.join(root, I18N)
  s = _read(p)
  nl = '\r\n' if '\r\n' in s else '\n'
  added = 0
  for anchor_nl, indent, idx in ((ZH_ANCHOR, '    ', 1), (TW_ANCHOR, '  ', 2)):
    anchor = anchor_nl.replace('\n', nl)   # 文件是 CRLF 时锚点也要用 CRLF
    if anchor not in s:
      out.append('!! i18n.js: anchor %r not found -> SKIP' % anchor.strip()[:28])
      continue
    at = s.index(anchor)
    window = s[at:at + 1600]          # 锚点后面那一段就是本语言表的键区
    lines = []
    for entry in I18N_KEYS:
      line = '%s"%s": "%s",' % (indent, entry[0], entry[idx])
      if line in window:
        continue                     # 本表已有这一条 (同名键可能在别处/别的表存在, 不能只看全文件)
      lines.append(line)
    if lines:
      s = s[:at + len(anchor)] + nl + nl.join(lines) + s[at + len(anchor):]
      added += len(lines)
  if added:
    _write(p, s)
  out.append('  + i18n.js: %d entry(ies) added (zh-Hans + zh-Hant)' % added)
  print('\n'.join(out))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
