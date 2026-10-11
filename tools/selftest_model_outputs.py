#!/usr/bin/env python3
"""modeld 输出解析链路自测 (用真实 bundle 的 metadata, 不需要相机/引擎)。

背景: modeld 那次崩溃 `ValueError: Missing output lane_lines` 是"解析器按老的单模型布局
写, 而 bundle 是 driving_vision + driving_policy 分离布局"造成的。这个自测直接拿
<树>/openpilot/selfdrive/modeld/models/<Model param> 下的 metadata pkl 造出与引擎等长的
原始输出, 走一遍 modeld 的 slice_outputs + parse_vision/policy, 检查:

  1. 不再抛异常;
  2. fill_model_msg 需要的每一个键都在, 且形状对 (plan/lane_lines/road_edges/lead/
     desire_state/desire_pred/meta/...);
  3. vision 的 hidden_state 长度与 policy 的 features_buffer 最后一维一致 (视觉喂策略的接线);
  4. 反向对照: 真的缺必要输出 (比如 pose) 时仍然报错, 不静默兜底。

用法: .venv/bin/python3 tools/selftest_model_outputs.py <树根>
"""
import os
import pickle
import sys

TREE = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
sys.path.insert(0, TREE)

import numpy as np  # noqa: E402

from openpilot.selfdrive.modeld.constants import ModelConstants as MC  # noqa: E402
from openpilot.selfdrive.modeld.parse_model_outputs import Parser  # noqa: E402
from openpilot.selfdrive.modeld.modeld import get_model_dir  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'OK ' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


model_dir = get_model_dir()
print(f"bundle: {model_dir}")
if not model_dir.exists():
    print(f"!! 模型目录不存在: {model_dir}")
    sys.exit(1)


def load(name):
    with open(model_dir / name, "rb") as f:
        return pickle.load(f)


def fake_outputs(meta, value=0.5):
    """按 metadata 的 output_shapes + output_slices 造原始输出 (与引擎输出等长, 1-D)。

    用常量填充: 解析只做 reshape/softmax/exp, 不关心数值, 只要长度对就能暴露形状错误。
    """
    n = meta["output_shapes"]["outputs"][1]
    flat = np.full(n, value, dtype=np.float32)
    return flat, meta["output_slices"]


def slice_outputs(flat, slices):
    return {k: flat[np.newaxis, v] for k, v in slices.items()}


v_meta = load("driving_vision_metadata.pkl")
p_meta = load("driving_policy_metadata.pkl")
parser = Parser()

v_flat, v_sl = fake_outputs(v_meta)
p_flat, p_sl = fake_outputs(p_meta)

print("1) 视觉解析 (本 bundle 的视觉输出没有 lane_lines, 老代码在这里崩)")
try:
    v_outs = parser.parse_vision_outputs(slice_outputs(v_flat, v_sl))
    check("vision 解析不抛异常", True)
except Exception as e:
    v_outs = {}
    check("vision 解析不抛异常", False, f"{type(e).__name__}: {e}")

print("2) 策略解析 + 合并 (modeld 是 {**vision, **policy})")
try:
    p_outs = parser.parse_policy_outputs(slice_outputs(p_flat, p_sl))
    check("policy 解析不抛异常", True)
except Exception as e:
    p_outs = {}
    check("policy 解析不抛异常", False, f"{type(e).__name__}: {e}")

outs = {**v_outs, **p_outs}

print("3) fill_model_msg 需要的键与形状")
expect = {
    "plan": (1, MC.IDX_N, MC.PLAN_WIDTH),
    "plan_stds": (1, MC.IDX_N, MC.PLAN_WIDTH),
    "lane_lines": (1, MC.NUM_LANE_LINES, MC.IDX_N, MC.LANE_LINES_WIDTH),
    "lane_lines_stds": (1, MC.NUM_LANE_LINES, MC.IDX_N, MC.LANE_LINES_WIDTH),
    "lane_lines_prob": (1, 2 * MC.NUM_LANE_LINES),
    "road_edges": (1, MC.NUM_ROAD_EDGES, MC.IDX_N, MC.LANE_LINES_WIDTH),
    "road_edges_stds": (1, MC.NUM_ROAD_EDGES, MC.IDX_N, MC.LANE_LINES_WIDTH),
    "lead": (1, MC.LEAD_MHP_SELECTION, MC.LEAD_TRAJ_LEN, MC.LEAD_WIDTH),
    "lead_stds": (1, MC.LEAD_MHP_SELECTION, MC.LEAD_TRAJ_LEN, MC.LEAD_WIDTH),
    "lead_prob": (1, MC.LEAD_MHP_SELECTION),
    "desire_state": (1, MC.DESIRE_PRED_WIDTH),
    "desire_pred": (1, MC.DESIRE_PRED_LEN, MC.DESIRE_PRED_WIDTH),
    # meta 的维度由 metadata 的 slice 长度决定 (55), 不写死
    "meta": (1, v_sl["desire_pred"].start - v_sl["meta"].start),
    "pose": (1, MC.POSE_WIDTH),
    "wide_from_device_euler": (1, MC.WIDE_FROM_DEVICE_WIDTH),
}
for k, shape in expect.items():
    got = outs.get(k)
    ok = got is not None and tuple(got.shape) == tuple(shape)
    check(f"{k:24s} 形状 {tuple(shape)}", ok, f"实际 {None if got is None else tuple(got.shape)}")

print("4) 视觉喂策略的接线 (hidden_state 长度 vs features_buffer 最后一维)")
hs = v_outs.get("hidden_state")
feat_dim = p_meta["input_shapes"]["features_buffer"][-1]
check(f"hidden_state 存在且长度 = {feat_dim}", hs is not None and hs.shape[-1] == feat_dim,
      f"hidden_state {None if hs is None else hs.shape} vs features_buffer 尾维 {feat_dim}")

print("5) desired_curvature 解析成 2-D (modeld 用 [0,0] 索引, 原始 slice 是 1-D 会 IndexError)")
dc = outs.get("desired_curvature")
ok = dc is not None and dc.ndim == 2
try:
    _ = dc[0, 0]
except Exception as e:
    ok = False
    print(f"      索引失败: {e}")
check("desired_curvature 可 [0,0] 索引", ok, f"shape={None if dc is None else dc.shape}")

print("6) 反向对照: 真的缺必要输出时仍要报错 (不静默兜底)")
bad = slice_outputs(v_flat, v_sl)
del bad["pose"]
try:
    Parser().parse_vision_outputs(bad)
    check("缺 pose 时抛错", False, "没有报错")
except ValueError as e:
    check("缺 pose 时抛错", True, str(e))

print("7) 反向对照: 老布局 (单模型, 驾驶头在视觉里) 也要能解析")
old = {**slice_outputs(v_flat, v_sl), **slice_outputs(p_flat, p_sl)}   # 两套合起来 = 单模型
try:
    old_outs = Parser().parse_vision_outputs(dict(old))
    ll = old_outs.get("lane_lines")
    check("老布局兼容 (lane_lines 在视觉字典里)", ll is not None and tuple(ll.shape) == (1, MC.NUM_LANE_LINES, MC.IDX_N, MC.LANE_LINES_WIDTH),
          f"lane_lines={None if ll is None else tuple(ll.shape)}")
except ValueError as e:
    check("老布局兼容 (lane_lines 在视觉字典里)", False, str(e))

print()
if FAILS:
    print(f"FAILED {len(FAILS)}: {FAILS}")
    sys.exit(1)
print("全部通过")
