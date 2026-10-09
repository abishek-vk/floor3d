"""Small geometry helpers shared by the Mode B stages.

Conventions: cameras are OpenCV (x right, y down, z forward). A pose is stored as
world-to-camera (R, t): x_cam = R @ x_world + t. Intrinsics K are 3x3.
"""
from __future__ import annotations

import numpy as np


def rodrigues(r: np.ndarray) -> np.ndarray:
    """(N,3) axis-angle -> (N,3,3) rotation matrices (vectorised)."""
    r = np.atleast_2d(r)
    th = np.linalg.norm(r, axis=1, keepdims=True)
    k = r / np.maximum(th, 1e-12)
    kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]
    z = np.zeros_like(kx)
    Kx = np.stack([z, -kz, ky, kz, z, -kx, -ky, kx, z], 1).reshape(-1, 3, 3)
    th = th[:, :, None]
    return np.eye(3)[None] + np.sin(th) * Kx + (1 - np.cos(th)) * (Kx @ Kx)


def rot_to_rvec(R: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.Rodrigues(np.asarray(R, np.float64))[0].ravel()


def rotate_points(rvec: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Rotate each X[i] by axis-angle rvec[i] (Rodrigues formula, vectorised)."""
    th = np.linalg.norm(rvec, axis=1, keepdims=True)
    k = rvec / np.maximum(th, 1e-12)
    c, s = np.cos(th), np.sin(th)
    return X * c + np.cross(k, X) * s + k * np.sum(k * X, 1, keepdims=True) * (1 - c)


def cam_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return -R.T @ t


def project(K: np.ndarray, R: np.ndarray, t: np.ndarray, X: np.ndarray):
    """World points (N,3) -> pixels (N,2), depth (N,)."""
    Xc = X @ R.T + t
    z = Xc[:, 2]
    uv = (Xc[:, :2] / np.maximum(z[:, None], 1e-9)) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    return uv, z


def backproject(K: np.ndarray, depth: np.ndarray, stride: int = 1):
    """Depth map -> camera-frame points (H*W,3) and pixel grid."""
    h, w = depth.shape
    v, u = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[::stride, ::stride]
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, z], -1).reshape(-1, 3), u.ravel(), v.ravel()


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Least-squares similarity dst ~ s R src + t (Umeyama 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / a.var(0).sum() if with_scale else 1.0
    return s, R, mu_d - s * R @ mu_s


def triangulate_dlt(Ps: list[np.ndarray], xs: list[np.ndarray]) -> np.ndarray:
    """Multi-view linear triangulation of one point; Ps are 3x4 normalised projections."""
    A = []
    for P, x in zip(Ps, xs):
        A.append(x[0] * P[2] - P[0])
        A.append(x[1] * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.asarray(A))
    X = Vt[-1]
    return X[:3] / X[3]
