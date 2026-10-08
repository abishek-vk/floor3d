"""Video -> keyframes (+ held-out test views).

Keyframes are picked one per uniform time bin, choosing the sharpest frame in each bin
(variance of the Laplacian), so motion-blurred frames are skipped. Two passes over the
video (sharpness on thumbnails, then decode only the keyframes) keep memory flat for long
clips. With `holdout=k` every k-th keyframe becomes a held-out test view (LLFF convention):
it is never used for geometry or appearance, only registered afterwards for evaluation.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from core.log import RunContext

VIDEO_EXT = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm")


@dataclass
class Frames:
    images: list[np.ndarray]          # RGB uint8, processing resolution
    index: list[int]                  # source frame number in the video
    train: list[int]                  # positions in `images`
    test: list[int]
    K: np.ndarray                     # 3x3 at processing resolution
    K_given: bool                     # intrinsics supplied by the user (else a guess, refined by BA)
    scale: float                      # processing / source resolution
    src_size: tuple[int, int]         # (w, h) of the source video
    sharpness: list[float] = field(default_factory=list)


def _iter_bgr(path: str):
    if os.path.isdir(path):  # a folder of images also works
        for n in sorted(os.listdir(path)):
            if n.lower().endswith((".jpg", ".jpeg", ".png")):
                yield cv2.imread(os.path.join(path, n))
        return
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open video {path}")
    try:
        while True:
            ok, f = cap.read()
            if not ok:
                break
            yield f
    finally:
        cap.release()


def load_frames(path: str, ctx: RunContext, n_key: int = 150, max_side: int = 640,
                holdout: int = 0, intrinsics: list[float] | None = None, fov_deg: float | None = None) -> Frames:
    sharp, src_size = [], None
    for bgr in _iter_bgr(path):
        if bgr is None:
            sharp.append(-1.0)
            continue
        src_size = src_size or (bgr.shape[1], bgr.shape[0])
        k = 320 / max(bgr.shape[:2])
        g = cv2.cvtColor(cv2.resize(bgr, None, fx=k, fy=k, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        sharp.append(float(cv2.Laplacian(g, cv2.CV_32F).var()))
    if src_size is None:
        raise IOError(f"no frames could be decoded from {path}")
    n = len(sharp)
    if n > n_key:
        bins = np.linspace(0, n, n_key + 1).astype(int)
        keep = [int(b0 + np.argmax(sharp[b0:b1])) for b0, b1 in zip(bins[:-1], bins[1:]) if b1 > b0]
    else:
        keep = [i for i in range(n) if sharp[i] >= 0]
    scale = min(1.0, max_side / max(src_size))
    size = (round(src_size[0] * scale), round(src_size[1] * scale))
    want, images = set(keep), []
    for i, bgr in enumerate(_iter_bgr(path)):
        if i in want:
            if scale != 1.0:
                bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
            images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if len(images) < 8:
        ctx.fallback("frames", f"only {len(images)} usable frames; reconstruction will be weak")
    h, w = images[0].shape[:2]
    if intrinsics is not None:  # fx fy cx cy in source pixels
        fx, fy, cx, cy = intrinsics
        K = np.array([[fx * scale, 0, cx * scale], [0, fy * scale, cy * scale], [0, 0, 1]])
        given = True
    else:
        # Unknown camera: start from a typical phone FOV; bundle adjustment refines f.
        f = 0.5 * max(w, h) / np.tan(np.radians(fov_deg or 65.0) / 2)
        K = np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1]])
        given = fov_deg is not None
    pos = list(range(len(images)))
    test = pos[holdout // 2::holdout] if holdout and holdout > 1 else []
    train = [p for p in pos if p not in set(test)]
    return Frames(images, keep, train, test, K, given, scale, src_size, [sharp[k] for k in keep])
