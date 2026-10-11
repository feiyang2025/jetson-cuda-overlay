"""ONNX -> TensorRT .plan 编译器 (Jetson CUDA 移植)。

为什么需要这一步: 上游 sunnypilot 的模型体系(bundle)里 artifact 是 **tinygrad 的预编译 pkl**
(bundle.runner == "tinygrad"), 那是"另一个运行时的成品" —— 既不是 ONNX 也不是 TRT 引擎。
本移植走 TensorRT, 所以"下载完成"之后必须多一步: 拿 bundle 里的 ONNX, 在本机 trtexec 编成 .plan。

本模块的设计原则(全部来自真机踩坑):
  1) 幂等 + 版本绑定: .plan 与 TensorRT 版本 / GPU 绑定, 跨机器拷贝不可靠。
     引擎旁写 sidecar(<plan>.json, 记录 onnx 指纹 + trtexec/TRT 版本 + GPU 名),
     指纹或版本变了才重编 —— 把"祈祷能加载"变成"可验证"。
  2) 串行: 全局锁, 两个编译不重叠。
  3) 诚实进度: TensorRT 不吐进度条, 只上报阶段 + 用上次同 net 的实测时长估 ETA。
  4) 编完自检(关键): 真加载引擎 + 核对 I/O binding 名字, 以及"元数据的切片名是否覆盖
     解析器实际要求的名字"(模型代际 vs 代码代际, 本移植最容易踩的那类坑)。
     自检不通过就标 unavailable, 绝不留给运行时崩。

调用方: sunnypilot/models/manager.py (下载收尾后按 ModelAutoCompile 自动触发)、
        webui 的 /api/opui/action/models_compile。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

# split 双引擎(bundle 里 vision/policy 两个 net, 也就是网页上那两条进度条)
NETS = ("driving_vision", "driving_policy")

# merged 单引擎: ONNX 在 models 根目录, 引擎放在按模型分的子目录(与本移植 modeld_bigcombo 一致)
MERGED = {
  "driving_supercombo": {"onnx": "<models>/driving_supercombo.onnx", "out_dir": "BigCombo"},
}

TRTEXEC = shutil.which("trtexec") or "/usr/bin/trtexec"
WORKSPACE_MB = int(os.getenv("TRT_BUILD_WORKSPACE_MB", "4096"))
PRECISION = os.getenv("TRT_BUILD_PRECISION", "--fp16")

_lock = threading.Lock()          # 串行化编译
_proc: subprocess.Popen | None = None
_cancel = threading.Event()
_state: dict = {
  "status": "idle",               # idle | building | verifying | done | failed | cancelled
  "net": None, "model": None, "stage": None,
  "started": 0.0, "eta_s": None, "log": None, "error": None, "verified": None,
}
_last_durations: dict[str, float] = {}   # net -> 上次实测时长(秒), 用于 ETA


# ───────────────────────── 基础设施 ─────────────────────────

def models_root() -> Path:
  """<tree>/openpilot/selfdrive/modeld/models"""
  here = Path(__file__).resolve()
  for parent in here.parents:
    cand = parent / "openpilot" / "selfdrive" / "modeld" / "models"
    if cand.is_dir():
      return cand
  raise RuntimeError("models dir not found from %s" % here)


def _hash_file(path: Path) -> str:
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def _trt_version() -> str:
  """从 libnvinfer 的 soname 取 TensorRT 主版本, 拿不到就退回 trtexec --version。"""
  for pat in ("/usr/lib/aarch64-linux-gnu/libnvinfer.so.*", "/usr/lib/*/libnvinfer.so.*"):
    hits = sorted(Path("/").glob(pat.lstrip("/")))
    if hits:
      return hits[-1].name.split(".so.")[-1]
  try:
    out = subprocess.run([TRTEXEC, "--version"], capture_output=True, text=True, timeout=20).stdout
    m = re.search(r"TensorRT[^\d]*([\d.]+)", out)
    return m.group(1) if m else "unknown"
  except Exception:
    return "unknown"


def _gpu_name() -> str:
  try:
    return subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=15).stdout.strip()[:80]
  except Exception:
    return "unknown"


def onnx_path(model_dir: Path, net: str) -> Path:
  if net in MERGED:
    return models_root() / Path(MERGED[net]["onnx"]).name
  return Path(model_dir) / f"{net}.onnx"


def plan_path(model_dir: Path, net: str) -> Path:
  if net in MERGED:
    return models_root() / MERGED[net]["out_dir"] / f"{net}_fp16.plan"
  return Path(model_dir) / f"{net}_fp16.plan"


def metadata_path(model_dir: Path, net: str) -> Path:
  if net in MERGED:
    return models_root() / MERGED[net]["out_dir"] / "big_driving_supercombo_metadata.pkl"
  return Path(model_dir) / f"{net}_metadata.pkl"


def _sidecar(plan: Path) -> Path:
  return plan.with_suffix(plan.suffix + ".json")


def buildable_nets(model_dir: Path) -> list[str]:
  """该模型目录里能编的 net(有 ONNX 且不是坏文件)。"""
  out = []
  for net in NETS:
    if onnx_path(model_dir, net).is_file():
      out.append(net)
  if (models_root() / "driving_supercombo.onnx").is_file() and Path(model_dir).name == MERGED["driving_supercombo"]["out_dir"]:
    out.append("driving_supercombo")
  return out


# ───────────────────────── 判断"要不要编" ─────────────────────────

def plan_state(model_dir: Path, net: str) -> dict:
  """返回 {exists, needs_build, reason, onnx_sha256, trt_version, built_at, verified}"""
  onnx = onnx_path(Path(model_dir), net)
  plan = plan_path(Path(model_dir), net)
  st = {"net": net, "onnx": str(onnx), "plan": str(plan), "exists": plan.is_file(),
        "needs_build": False, "reason": "", "onnx_sha256": None, "trt_version": _trt_version(),
        "built_at": None, "verified": None}
  if not onnx.is_file():
    st.update(needs_build=False, reason="没有 ONNX(该模型不是可编译形态)")
    return st

  st["onnx_sha256"] = _hash_file(onnx)
  side = _sidecar(plan)
  if not plan.is_file():
    st["needs_build"] = True
    st["reason"] = "缺 .plan(只有 ONNX)"
    return st
  try:
    meta = json.loads(side.read_text())
  except Exception:
    st["needs_build"] = True
    st["reason"] = "缺 sidecar(无法确认引擎与 ONNX/TRT 版本的对应关系)"
    return st
  st["built_at"] = meta.get("built_at")
  st["verified"] = meta.get("verified")
  if meta.get("onnx_sha256") != st["onnx_sha256"]:
    st.update(needs_build=True, reason="ONNX 变了(引擎过期)")
    return st
  if str(meta.get("trt_version")) != str(st["trt_version"]):
    st.update(needs_build=True, reason="TensorRT 版本变了(%s -> %s)" % (meta.get("trt_version"), st["trt_version"]))
    return st
  if meta.get("verified") is False:
    st.update(needs_build=True, reason="上次自检未通过: %s" % (meta.get("verify_detail") or "未知"))
    return st
  st["reason"] = "已编译且自检通过"
  return st


def survey(model_dir: Path) -> dict:
  nets = buildable_nets(Path(model_dir))
  return {"model_dir": str(model_dir), "nets": [plan_state(model_dir, n) for n in nets],
          "trtexec": TRTEXEC, "trt_version": _trt_version(),
          "has_trtexec": os.access(TRTEXEC, os.X_OK) if TRTEXEC else False}


# ───────────────────────── 自检 ─────────────────────────

_PARSER_RE = re.compile(r"parse_(?:mdn|binary_crossentropy|categorical_crossentropy)\(\s*'([a-z_]+)'")


def parser_required_names() -> dict[str, set[str]]:
  """从 parse_model_outputs.py 源码里抽出解析器真正要求的输出名(不硬编码, 解析器改了就跟着变)。"""
  src_file = Path(__file__).resolve().parents[2] / "selfdrive" / "modeld" / "parse_model_outputs.py"
  try:
    src = src_file.read_text()
  except Exception:
    return {}
  vision = set()
  policy = set()
  for m in re.finditer(r"def (parse_vision_outputs|parse_policy_outputs)\(self.*?(?=\n  def |\Z)", src, re.S):
    body, names = m.group(0), set(_PARSER_RE.findall(m.group(0)))
    (vision if m.group(1) == "parse_vision_outputs" else policy).update(names)
  return {"driving_vision": vision, "driving_policy": policy}


_SELFCHECK = r'''
import json, pickle, sys
sys.path.insert(0, %(root)r)
out = {"ok": False, "detail": ""}
try:
  import ctypes
  # 与运行时同一条路径: 沿用当前 context, 不 push primary
  try:
    from tinygrad.device import Device
    _ = Device['CUDA']
  except Exception:
    pass
  from openpilot.selfdrive.modeld.runners.tensorrt_runner import TensorRTModel
  m = TensorRTModel(%(plan)r)
  ins = sorted(getattr(m, "input_names", []) or [])
  outs = sorted(getattr(m, "output_names", []) or [])
  md = pickle.load(open(%(meta)r, "rb"))
  want_in = sorted(md.get("input_shapes", {}).keys())
  missing_in = [k for k in want_in if k not in ins]
  n_out_want = md.get("output_shapes", {}).get("outputs", (1, 0))[1]
  slices = md.get("output_slices") or {}
  sl_names = list(slices.keys()) if isinstance(slices, dict) else list(slices)
  need = set(%(need)r)
  missing_slices = sorted(need - set(sl_names)) if need else []
  out.update(ok=not missing_in and not missing_slices,
             input_names=ins, output_names=outs, want_inputs=want_in,
             missing_inputs=missing_in, missing_slices=missing_slices,
             output_width_meta=n_out_want)
  if missing_in:
    out["detail"] = "引擎缺输入绑定: %%s" %% missing_in
  elif missing_slices:
    out["detail"] = "模型代际不符, 解析器要的切片元数据里没有: %%s" %% missing_slices
  else:
    out["detail"] = "加载成功, 绑定名与解析器要求一致"
except Exception as e:
  out["detail"] = "%%s: %%s" %% (type(e).__name__, e)
print(json.dumps(out))
'''


def self_check(model_dir: Path, net: str) -> dict:
  """真加载引擎 + 核对绑定名与解析器要求。跑在子进程里, 崩了也拖不垮调用方。"""
  plan, meta = plan_path(Path(model_dir), net), metadata_path(Path(model_dir), net)
  need = sorted(parser_required_names().get(net, set()))
  root = str(Path(__file__).resolve().parents[3])
  code = _SELFCHECK % {"plan": str(plan), "meta": str(meta), "root": root, "need": need}
  try:
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
    line = (r.stdout or "").strip().splitlines()[-1] if r.stdout else ""
    res = json.loads(line) if line.startswith("{") else {"ok": False, "detail": (r.stderr or "")[-300:]}
  except Exception as e:
    res = {"ok": False, "detail": "%s: %s" % (type(e).__name__, e)}
  return res


# ───────────────────────── 编译 ─────────────────────────

def _trtexec_once(onnx: Path, plan: Path, log: Path, precision: str) -> tuple[bool, str]:
  cmd = ([TRTEXEC, "--onnx=%s" % onnx, "--saveEngine=%s" % plan] + ([precision] if precision else [])
         + ["--memPoolSize=workspace:%d" % WORKSPACE_MB, "--noDataTransfers"])
  with open(log, "ab", buffering=0) as lf:
    lf.write(("\n$ " + " ".join(cmd) + "\n").encode())
    global _proc
    _proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
    try:
      rc = _proc.wait()
    finally:
      _proc = None
  return rc == 0, ("exit=%s" % rc)


def _plan_ok(plan: Path) -> bool:
  # 实测坑: trtexec 可能 rc=0 却留下 0 字节引擎(conv+gelu 挑不到 tactic: "Created engine
  # with size: 0 MiB" + assertion failure), 必须按大小再判一次, 否则会把空引擎当成功。
  try:
    return plan.stat().st_size > (1 << 20)
  except OSError:
    return False


def _run_trtexec(onnx: Path, plan: Path, log: Path) -> tuple[bool, str, str]:
  """先按 PRECISION 编; 命中 tactic 缺失时去掉精度标志重试一次。

  本移植实测: CD210 的 vision 在 --fp16 下报
  'Could not find any implementation for node node_conv2d_1 + node_gelu_1',
  去掉 --fp16 后正常编出 80MB 引擎(80s)。手册另记 fp16/fp32 编出的引擎 MD5 相同
  (没有真正的 FP16 kernel), 所以这种降级不损失性能, 可放心作为兜底。
  """
  plan.parent.mkdir(parents=True, exist_ok=True)
  detail, tried = "unknown", []
  for precision in (PRECISION, ""):
    plan.unlink(missing_ok=True)
    ok, detail = _trtexec_once(onnx, plan, log, precision)
    usable = ok and _plan_ok(plan)
    tried.append("%s:%s" % (precision or "--fp32(tf32)", "ok" if usable else "fail"))
    if usable:
      return True, detail, precision or "--fp32(tf32)"
    if ok:
      detail = "引擎为空文件(rc=0 但未产出可用引擎)"
    if not precision:
      break
    try:
      tail = log.read_text(errors="replace")[-6000:]
    except Exception:
      tail = ""
    if "Could not find any implementation" not in tail:
      break        # 不是 tactic 缺失 → 换精度也没用, 不再白跑一遍
  plan.unlink(missing_ok=True)
  return False, detail, "none (%s)" % ",".join(tried)


def compile_net(model_dir: Path, net: str, log_dir: Path | None = None) -> dict:
  """编译单个 net(同步, 已加锁)。返回 {ok, net, reason, verify}"""
  model_dir = Path(model_dir)
  onnx, plan = onnx_path(model_dir, net), plan_path(model_dir, net)
  if not onnx.is_file():
    return {"ok": False, "net": net, "reason": "没有 ONNX: %s" % onnx}
  if _cancel.is_set():
    return {"ok": False, "net": net, "reason": "已取消"}

  trt = _trt_version()
  log = (log_dir or Path("/tmp")) / ("trt_build_%s.log" % net)
  _state.update(status="building", net=net, model=model_dir.name, stage="构建引擎",
                started=time.time(), eta_s=_last_durations.get(net), log=str(log), error=None)
  t0 = time.time()
  ok, detail, precision_used = _run_trtexec(onnx, plan, log)
  dur = time.time() - t0
  if ok:
    _last_durations[net] = dur
  if _cancel.is_set():
    plan.unlink(missing_ok=True)
    _state.update(status="cancelled", stage=None)
    return {"ok": False, "net": net, "reason": "已取消"}
  if not ok:
    _state.update(status="failed", error="trtexec 失败 (%s), 见 %s" % (detail, log))
    return {"ok": False, "net": net, "reason": "trtexec 失败", "log": str(log)}

  # 自检 → 通过才写 sidecar 并标 verified
  _state.update(status="verifying", stage="自检(加载引擎 + 核对绑定名)")
  check = self_check(model_dir, net)
  side = {
    "onnx": str(onnx), "onnx_sha256": _hash_file(onnx), "trt_version": trt,
    "gpu": _gpu_name(), "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "duration_s": round(dur, 1), "precision": precision_used, "verified": bool(check.get("ok")),
    "verify_detail": check.get("detail"), "input_names": check.get("input_names"),
    "output_names": check.get("output_names"),
  }
  _sidecar(plan).write_text(json.dumps(side, ensure_ascii=False, indent=1))
  _state.update(status="done" if check.get("ok") else "failed",
                verified=bool(check.get("ok")),
                error=None if check.get("ok") else "自检未通过: %s" % check.get("detail"))
  return {"ok": bool(check.get("ok")), "net": net, "plan": str(plan), "duration_s": round(dur, 1), "precision": precision_used,
          "verify": check, "log": str(log)}


def compile_model(model_dir: Path, nets: list[str] | None = None, force: bool = False) -> dict:
  """编译一个模型的若干 net(串行、幂等)。"""
  with _lock:
    _cancel.clear()
    model_dir = Path(model_dir)
    todo = nets or buildable_nets(model_dir)
    results = []
    for net in todo:
      st = plan_state(model_dir, net)
      if not st["needs_build"] and not force:
        results.append({"ok": True, "net": net, "skipped": True, "reason": st["reason"]})
        continue
      results.append(compile_net(model_dir, net))
      if _cancel.is_set():
        break
    ok = all(r.get("ok") for r in results) if results else False
    _state.update(status="done" if ok else "failed", net=None, stage=None,
                  error=None if ok else next((r.get("reason") or r.get("verify", {}).get("detail")
                                             for r in results if not r.get("ok")), None))
    return {"ok": ok, "model": model_dir.name, "results": results}


def cancel() -> dict:
  _cancel.set()
  p = _proc
  if p and p.poll() is None:
    p.terminate()
  return {"ok": True, "cancelled": True}


def status() -> dict:
  st = dict(_state)
  if st["status"] == "building" and st.get("started"):
    st["elapsed_s"] = round(time.time() - st["started"], 1)
  return st


import sys  # noqa: E402  (子进程自检用)
