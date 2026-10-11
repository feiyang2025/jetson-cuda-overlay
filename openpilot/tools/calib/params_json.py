"""Params 读写小工具, 屏蔽本 fork 与老 fork 的 JSON 型 param 差异。

背景 (两条都踩过):
- 本 fork (sunnypilot 新布局) 的 param key 带类型, FcamIntrinsics / EcamIntrinsics /
  FcamCalibResult / WideCalibResult 注册为 JSON 型:
    * get(key) 直接返回 dict —— 老代码的 json.loads(...) 会 TypeError, 被 except 吞掉,
      于是"读不到已标定的 fl", 悄悄回落到默认焦距。
    * put(key, ...) 只接受 dict/list —— 老代码的 put(key, json.dumps(x)) 会 TypeError,
      被 except 吞掉, 于是标定结果永远写不进 Params, UI 面板一直显示 NO DATA。
- 老 fork 的 Params 没有类型, 读写都是 JSON 字符串。

所以统一走这两个入口: 读时容忍 dict / str / bytes; 写时先按 dict 写, 失败再退回字符串。
"""

from __future__ import annotations

import json


def params_json_get(key: str, default=None, params=None):
    """读 JSON 型 param, 返回 dict/list; 读不到返回 default。

    params 不传就自己开一个 (Params 构造有 ctypes 开销, 热路径上请复用实例)。
    """
    try:
        if params is None:
            from openpilot.common.params import Params
            params = Params()
        raw = params.get(key)
    except Exception:
        return default
    if raw is None:
        return default
    if isinstance(raw, (dict, list)):
        return raw           # 本 fork: 已经是解析好的对象
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode(errors="replace")
    if isinstance(raw, str):
        if not raw.strip():
            return default
        try:
            return json.loads(raw)
        except Exception:
            return default
    return default


def params_json_put(key: str, obj, params=None) -> bool:
    """写 JSON 型 param。成功返回 True。

    先试 put(dict) (本 fork 的唯一正确写法), 再退回 put(json 字符串) (老 fork)。
    """
    try:
        if params is None:
            from openpilot.common.params import Params
            params = Params()
    except Exception:
        return False
    if isinstance(obj, (dict, list)):
        try:
            params.put(key, obj)
            return True
        except Exception:
            pass
    try:
        params.put(key, json.dumps(obj, ensure_ascii=False, default=str))
        return True
    except Exception:
        return False
