import argparse
import json

import numpy as np


def load_samples(path, cols):
    raw = np.loadtxt(path, delimiter=",", comments="#")
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if cols is None:
        idx = list(range(6))
    else:
        idx = list(cols)
    gyro = raw[:, idx[0]:idx[0] + 3]
    accel = raw[:, idx[3]:idx[3] + 3]
    return gyro, accel


def stationary_mask(gyro, thr):
    return np.linalg.norm(gyro, axis=1) < thr


def gyro_bias(gyro, mask):
    if mask.sum() > 10:
        return np.mean(gyro[mask], axis=0)
    return np.mean(gyro, axis=0)


def accel_ellipsoid(accel, gravity):
    ax, ay, az = accel[:, 0], accel[:, 1], accel[:, 2]
    D = np.column_stack([
        ax * ax, ay * ay, az * az,
        2 * ax * ay, 2 * ax * az, 2 * ay * az,
        2 * ax, 2 * ay, 2 * az,
    ])
    rhs = np.ones_like(ax)
    p, *_ = np.linalg.lstsq(D, rhs, rcond=None)
    v11, v22, v33, v12, v13, v23, u1, u2, u3 = p
    V = np.array([[v11, v12, v13], [v12, v22, v23], [v13, v23, v33]])
    u = np.array([u1, u2, u3])
    w = -1.0
    bias = -np.linalg.solve(V, u)
    A = V / max(abs(float(bias @ (V @ bias) - w)), 1e-12)
    A = (A + A.T) / 2.0
    vals, vecs = np.linalg.eigh(A)
    vals = np.abs(vals)
    A = (vecs * vals) @ vecs.T
    A += np.eye(3) * 1e-9
    L = np.linalg.cholesky(A)
    M = np.linalg.inv(L)
    return bias, M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--cols", type=int, nargs=6, default=None,
                    help="gx gy gz ax ay az column indices in the CSV")
    ap.add_argument("--g", type=float, default=9.81)
    ap.add_argument("--gyro-thr", type=float, default=0.05)
    ap.add_argument("--out", default="imu_calibration.json")
    args = ap.parse_args()

    gyro, accel = load_samples(args.file, args.cols)
    mask = stationary_mask(gyro, args.gyro_thr)

    g_bias = gyro_bias(gyro, mask)
    if mask.sum() > 10:
        accel_fit = accel[mask]
    else:
        accel_fit = accel

    a_bias, M = accel_ellipsoid(accel_fit, args.g)
    imu_calib_matrix = M.flatten().tolist()

    result = {
        "imuBiasGyro": g_bias.tolist(),
        "imuBiasAccel": a_bias.tolist(),
        "imuCalibMatrix": imu_calib_matrix,
        "gravity": args.g,
        "note": "backend: accel_ms2 = (imuCalibMatrix @ (raw_accel_g - imuBiasAccel)) / gravity; "
                "gyro_rad = (raw_dps - imuBiasGyro) * pi/180",
    }

    print(json.dumps(result, indent=2))
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
