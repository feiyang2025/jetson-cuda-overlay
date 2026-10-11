#!/usr/bin/env python3
"""把标定相关参数全量打印出来 (体检包第 3 节用)。"""
import json
import sys

from openpilot.common.params import Params

KEYS = ["AutoCalib", "AutoCalibHistory", "AutoCalibBackup", "AutoCalibRollback",
        "FcamIntrinsics", "EcamIntrinsics", "FcamCalibResult", "WideCalibResult",
        "CalibrationParams", "Model", "UseWideCamera", "WideCalibActive", "FcamLiveActive"]

p = Params()
for k in KEYS:
    try:
        v = p.get(k)
    except Exception as e:
        v = f"(读不到: {type(e).__name__}: {e})"
    if k == "CalibrationParams":
        v = "(存在)" if v is not None else None
    if isinstance(v, (dict, list)):
        v = json.dumps(v, ensure_ascii=False, indent=1)
    print(f"--- {k} = {v}")
sys.exit(0)
