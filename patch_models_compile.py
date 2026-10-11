#!/usr/bin/env python3
"""下载 → 编译 自动化: 把本移植新增的"TRT 引擎编译"能力打进 fork(幂等, 锚点式)。

背景: 上游 sunnypilot 的模型体系(bundle)只提供 tinygrad 的预编译 pkl(runner=="tinygrad"),
既不是 ONNX 也不是 TRT 引擎。本移植走 TensorRT, 所以"下载完成"之后必须多一步: 本机
trtexec 把 bundle 里的 ONNX 编成 .plan。本脚本把这一步接到 fork 的三个面上:
  1. 执行器        openpilot/sunnypilot/models/engine_build.py       (整文件, 由 apply_cuda.sh 拷贝)
  2. 自动触发      openpilot/sunnypilot/models/manager.py            (下载收尾 → 可选自动编译)
  3. 网页          models_api.py(状态) / system_api.py(动作) / panel_catalog.py(按钮)
                   / panels.js + i18n.js(面板行与文案)
  4. 参数注册      openpilot/common/params_keys.h                    (ModelAutoCompile)
                                                                    ← 改完需重编 libparams_c.so

幂等: 每个文件先查 marker, 已打过就跳过; 锚点找不到会明确报错(不静默改错地方)。
换行: JS/h i18n 文件按原文件的 CRLF/LF 写入, 不做规范化。
用法: python3 patch_models_compile.py <REPO_ROOT>
"""

import sys
from pathlib import Path


def _read(p: Path) -> str:
  with open(p, encoding="utf-8", newline="") as f:
    return f.read()


def _write(p: Path, s: str) -> None:
  with open(p, "w", encoding="utf-8", newline="") as f:
    f.write(s)


def _nl(s: str) -> str:
  return "\r\n" if "\r\n" in s else "\n"


def edit(root: Path, rel: str, marker: str, edits, out):
  """edits: [(old, new)]; 全部命中才写盘。marker 存在即视为已打过。"""
  p = root / rel
  if not p.is_file():
    out.append("!! %s: 文件不存在, 跳过" % rel)
    return
  s = _read(p)
  if marker in s:
    out.append("  = %-46s 已打过" % rel)
    return
  nl = _nl(s)
  for i, (old, new) in enumerate(edits):
    old = old.replace("\n", nl)
    new = new.replace("\n", nl)
    if s.count(old) != 1:
      out.append("!! %s: 第 %d 处锚点匹配 %d 次(应为 1), 未改动" % (rel, i + 1, s.count(old)))
      return
    s = s.replace(old, new, 1)
  _write(p, s)
  out.append("  + %-46s 已打补丁" % rel)


def main() -> int:
  root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
  out: list[str] = []
  out.append("[patch_models_compile] root=%s" % root)

  # ── 1) manager.py: 下载收尾 → 可选自动编译 ──
  helper = '''

_AUTO_COMPILE_DEBOUNCE_S = 45.0
_last_auto_compile = 0.0


def _maybe_auto_compile(params) -> None:
  """本移植专用: 上游 bundle 下的是 tinygrad 预编译件(runner=="tinygrad"), 走 TensorRT 必须
  本机把 ONNX 编成 .plan。下载收尾后若 ModelAutoCompile 开启, 就在后台线程里编 ——
  engine_build 自身幂等(ONNX/TRT 版本没变就跳过)且串行(全局锁), 重复触发无副作用。"""
  global _last_auto_compile
  try:
    import threading
    import time as _t
    if not params.get_bool("ModelAutoCompile"):
      return
    now = _t.time()
    if now - _last_auto_compile < _AUTO_COMPILE_DEBOUNCE_S:
      return
    _last_auto_compile = now
    from openpilot.sunnypilot.models import engine_build as eb
    model_dir = eb.models_root() / str(params.get("Model") or "FiletOFish")
    if not model_dir.is_dir():
      return
    threading.Thread(target=eb.compile_model, args=(model_dir,), daemon=True).start()
  except Exception:
    pass
'''
  mgr = root / "openpilot/sunnypilot/models/manager.py"
  if mgr.is_file() and "_maybe_auto_compile" not in _read(mgr):
    s = _read(mgr)
    nl = _nl(s)
    marker = "\n\nclass "
    i = s.index(marker) if marker in s else -1
    if i < 0:
      out.append("!! manager.py: 找不到首个 class 定义, 跳过")
    else:
      s = s[:i] + helper.replace("\n", nl) + nl + "class " + s[i + len(marker):]
      for ind, anchor in (("        ", '''        artifact.downloadProgress.status = custom.ModelManagerSP.DownloadStatus.cached
        artifact.downloadProgress.progress = 100
        artifact.downloadProgress.eta = 0'''),
                          ("      ", '''      artifact.downloadProgress.status = custom.ModelManagerSP.DownloadStatus.downloaded
      artifact.downloadProgress.progress = 100
      artifact.downloadProgress.eta = 0''')):
        a = anchor.replace("\n", nl)
        if s.count(a) == 1:
          s = s.replace(a, a + nl + ind + "_maybe_auto_compile(self.params)", 1)
      _write(mgr, s)
      out.append("  + %-46s 已打补丁" % "openpilot/sunnypilot/models/manager.py")
  else:
    out.append("  = %-46s 已打过" % "openpilot/sunnypilot/models/manager.py")

  # ── 2) models_api.py: 状态里带 compile 块 ──
  api_helper = '''

def _bundle_compile_state() -> dict[str, Any]:
  """本移植专用环节: 上游 bundle 下的是 tinygrad 预编译 pkl(runner=="tinygrad"), 既不是 ONNX
  也不是 TRT 引擎 —— TRT 路线必须多一步: 拿 bundle 里的 ONNX 在本机 trtexec 编成 .plan。
  这里把 engine_build 的实时状态 + 当前模型各 net 的编译情况一起交给前端画进度/按钮。"""
  try:
    from openpilot.common.params import Params as _Params
    from openpilot.sunnypilot.models import engine_build as eb
    name = str(_Params().get("Model") or "FiletOFish")
    model_dir = eb.models_root() / name
    st = eb.status()
    sv = eb.survey(model_dir) if model_dir.is_dir() else {"nets": []}
    return {
      "available": True,
      "harness": "tensorrt",
      "model": name,
      "status": st.get("status"),
      "stage": st.get("stage"),
      "net": st.get("net"),
      "error": st.get("error"),
      "verified": st.get("verified"),
      "log": st.get("log"),
      "elapsed_s": st.get("elapsed_s"),
      "eta_s": st.get("eta_s"),
      "trtexec": sv.get("trtexec"),
      "trt_version": sv.get("trt_version"),
      "has_trtexec": sv.get("has_trtexec"),
      "nets": sv.get("nets", []),
      "needs_build": any(x.get("needs_build") for x in sv.get("nets", [])),
    }
  except Exception as exc:
    return {"available": False, "error": str(exc)}
'''
  edit(root, "webui/server/bridge/models_api.py", "_bundle_compile_state", [
    ("  download = _bundle_download(mm)\n  return {\n    \"ok\": True,",
     "  download = _bundle_download(mm)\n  compile_state = _bundle_compile_state()\n  return {\n    \"ok\": True,"),
    ("    \"download\": download,", "    \"download\": download,\n    \"compile\": compile_state,"),
    ("\n\ndef models_status() -> dict[str, Any]:", api_helper + "\n\ndef models_status() -> dict[str, Any]:"),
  ], out)

  # ── 3) system_api.py: 两个动作 ──
  edit(root, "webui/server/bridge/system_api.py", "models_compile", [
    ('''    if action == "models_cancel_download":
      p.remove("ModelManager_DownloadRef")
      return {"ok": True, "action": action}''',
     '''    if action == "models_cancel_download":
      p.remove("ModelManager_DownloadRef")
      return {"ok": True, "action": action}

    # 本移植: 上游只下 tinygrad pkl; TRT 路线要本机把 ONNX 编成 .plan(见 models/engine_build.py)
    if action == "models_compile":
      import threading
      from openpilot.sunnypilot.models import engine_build as eb
      st = eb.status().get("status")
      if st in ("building", "verifying"):
        return {"ok": False, "action": action, "error": "已有编译在进行中"}
      name = str(payload.get("model") or p.get("Model") or "FiletOFish")
      model_dir = eb.models_root() / name
      nets = payload.get("nets") or None
      force = bool(payload.get("force"))
      threading.Thread(target=eb.compile_model, args=(model_dir, nets),
                       kwargs={"force": force}, daemon=True).start()
      return {"ok": True, "action": action, "model": name, "nets": nets, "force": force}

    if action == "models_cancel_compile":
      from openpilot.sunnypilot.models import engine_build as eb
      return {"ok": True, "action": action, **eb.cancel()}'''),
  ], out)

  # ── 4) panel_catalog.py: 面板按钮 + 自动编译开关 ──
  edit(root, "webui/server/bridge/panel_catalog.py", "Build TRT Engines", [
    ('      {"type": "action", "action": "models_sync", "label": "Refresh Model List", "button": "REFRESH"},',
     '''      {"type": "action", "action": "models_sync", "label": "Refresh Model List", "button": "REFRESH"},
      {"type": "action", "action": "models_compile", "label": "Build TRT Engines (ONNX to .plan)", "button": "COMPILE",
       "desc": "This port runs TensorRT, while upstream bundles ship tinygrad pickles: the ONNX must be compiled into .plan on this device. A load + I/O self-check runs after each build."},
      {"type": "bool", "param": "ModelAutoCompile", "label": "Auto-build TRT engines after download",
       "desc": "Compile the downloaded model's ONNX into TensorRT engines automatically, about 1-3 minutes per engine."},'''),
  ], out)

  # ── 5) panels.js: 编译行 + 按钮 ──
  js = '''function paintModelsCompile(container, status) {
  const c = status && status.compile;
  if (!c || c.available === false) return;

  let root = container.querySelector("[data-models-compile='1']");
  if (!root) {
    const anchor = container.querySelector("[data-models-download='1']")
      || container.querySelector("[data-models-status-row]");
    root = document.createElement("div");
    root.dataset.modelsCompile = "1";
    if (anchor && anchor.parentElement) anchor.parentElement.insertBefore(root, anchor.nextSibling);
    else container.appendChild(root);
  }
  root.innerHTML = "";

  const st = c.status || "idle";
  const building = st === "building" || st === "verifying";
  const nets = c.nets || [];
  const pending = nets.filter((n) => n.needs_build).length;

  const row = document.createElement("div");
  row.className = "opui-sp-row opui-sp-row--has-action";
  let stateText;
  if (building) {
    stateText = `${t("Compiling")}${c.stage ? " · " + c.stage : ""}`
      + (c.elapsed_s ? ` · ${Math.round(c.elapsed_s)}s` : "")
      + (c.eta_s ? ` / ~${Math.round(c.eta_s)}s` : "");
  } else if (st === "failed") {
    stateText = `${t("Compile failed")}${c.error ? ": " + c.error : ""}`;
  } else if (pending) {
    stateText = `${t("Needs compiling")}: ${pending}`;
  } else {
    stateText = t("Compiled (verified)");
  }
  row.innerHTML = `
    <div class="opui-sp-row-text"><div class="opui-sp-row-title">${escapeHtml(t("TRT Engine"))}</div>
      <div class="opui-sp-row-desc">${escapeHtml(stateText)}${c.trt_version ? " · TensorRT " + escapeHtml(String(c.trt_version)) : ""}</div></div>
    <button type="button" class="opui-btn opui-btn--action${building ? " danger" : ""}">${escapeHtml(t(building ? "Cancel" : "Compile"))}</button>`;
  row.querySelector("button")?.addEventListener("click", async () => {
    if (building) {
      await apiPost("/api/opui/action/models_cancel_compile");
      toast(t("Compile cancelled"));
    } else {
      const r = await apiPost("/api/opui/action/models_compile", { model: c.model, force: false });
      toast(r && r.ok ? t("Compiling started") : ((r && r.error) || t("Failed")));
    }
    const cont2 = document.getElementById("panel-content");
    if (cont2 && panelDataRef) await renderModelsPanel(cont2, panelDataRef);
  });
  root.appendChild(row);

  const types = { driving_vision: t("Vision Model"), driving_policy: t("Policy Model"),
                  driving_supercombo: t("Driving Model") };
  for (const n of nets) {
    if (!n.needs_build && !building && st !== "failed") continue;
    const sub = document.createElement("div");
    sub.className = "opui-sp-row";
    sub.innerHTML = `<div class="opui-sp-row-text"><div class="opui-sp-row-title">${escapeHtml(types[n.net] || n.net)}</div>
      <div class="opui-sp-row-desc">${escapeHtml(n.reason || "")}</div></div>`;
    root.appendChild(sub);
  }
}

'''
  edit(root, "webui/web/static/js/panels.js", "paintModelsCompile", [
    ("function syncModelsExtras(status, container) {", js + "function syncModelsExtras(status, container) {"),
    ("function syncModelsExtras(status, container) {\n  paintModelsDownload(container, status);",
     "function syncModelsExtras(status, container) {\n  paintModelsDownload(container, status);\n  paintModelsCompile(container, status);"),
    ("  paintModelsDownload(container, m);\n  updateModelsCacheValue(container, m);",
     "  paintModelsDownload(container, m);\n  paintModelsCompile(container, m);\n  updateModelsCacheValue(container, m);"),
  ], out)

  # ── 6) i18n.js: 文案(简繁) ──
  keys = [("TRT Engine", "TRT 引擎", "TRT 引擎"),
          ("Compile", "编译", "編譯"), ("Compiling", "编译中", "編譯中"),
          ("Compiled (verified)", "已编译 · 自检通过", "已編譯 · 自檢通過"),
          ("Needs compiling", "待编译", "待編譯"), ("Compile failed", "编译失败", "編譯失敗"),
          ("Compile cancelled", "已取消编译", "已取消編譯"),
          ("Compiling started", "已开始编译", "已開始編譯"), ("Failed", "失败", "失敗"),
          ("Build TRT Engines (ONNX to .plan)", "编译 TRT 引擎（ONNX → .plan）", "編譯 TRT 引擎（ONNX → .plan）"),
          ("Auto-build TRT engines after download", "下载后自动编译 TRT 引擎", "下載後自動編譯 TRT 引擎"),
          ("Compile the downloaded model's ONNX into TensorRT engines automatically, about 1-3 minutes per engine.",
           "自动把下载模型的 ONNX 编成 TensorRT 引擎，每个引擎约 1-3 分钟。",
           "自動把下載模型的 ONNX 編成 TensorRT 引擎，每個引擎約 1-3 分鐘。"),
          ("This port runs TensorRT, while upstream bundles ship tinygrad pickles: the ONNX must be compiled into .plan on this device. A load + I/O self-check runs after each build.",
           "本移植走 TensorRT，而上游下的是 tinygrad 预编译件：必须在本机把 ONNX 编成 .plan。每次编译后会做一次加载 + I/O 自检。",
           "本移植走 TensorRT，而上游下的是 tinygrad 預編譯件：必須在本機把 ONNX 編成 .plan。每次編譯後會做一次載入 + I/O 自檢。")]
  i18n = root / "webui/web/static/js/i18n.js"
  if not i18n.is_file():
    out.append("!! i18n.js 不存在")
  elif '"TRT 引擎"' in _read(i18n):
    out.append("  = %-46s 已打过" % "webui/web/static/js/i18n.js")
  else:
    s = _read(i18n)
    nl = _nl(s)
    added = 0
    for anchor, indent, idx in (('    "Model Status": "模型状态",', "    ", 1),
                                ('  "Model Status": "模型狀態",', "  ", 2)):
      if anchor not in s:
        out.append("!! i18n.js: 锚点缺失 %s" % anchor.strip())
        return 1
      lines = ['%s"%s": "%s",' % (indent, en, (zh, tw)[idx - 1]) for en, zh, tw in keys
               if '%s"%s": "%s",' % (indent, en, (zh, tw)[idx - 1]) not in s]
      if lines:
        s = s.replace(anchor, nl.join([anchor] + lines), 1)
        added += len(lines)
    _write(i18n, s)
    out.append("  + %-46s 已打补丁(+%d 条)" % ("webui/web/static/js/i18n.js", added))

  # ── 7) params_keys.h: 注册 ModelAutoCompile ──
  edit(root, "openpilot/common/params_keys.h", "ModelAutoCompile", [
    ('    {"Model", {PERSISTENT | BACKUP, STRING, "FiletOFish"}},   // classic modeld / BigCombo selector',
     '    {"Model", {PERSISTENT | BACKUP, STRING, "FiletOFish"}},   // classic modeld / BigCombo selector\n'
     '    {"ModelAutoCompile", {PERSISTENT | BACKUP, BOOL, "0"}},        // 下载后是否自动把 ONNX 编成 TRT 引擎'),
  ], out)

  print("\n".join(out))
  print("[patch_models_compile] 提醒: 改过 params_keys.h 后要重编 libparams_c.so:")
  print("  PATH=$PWD/.venv/bin:$PATH PYTHONPATH=$PWD .venv/bin/scons -j$(nproc)")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
