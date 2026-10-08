"""Load PNG/JPG/PDF floor plans and normalize them for detection."""
from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

from core.log import RunContext

# Long-side bounds for the working image. CubiCasa5K "F1_scaled" images mostly
# fall in this range, which is what the pretrained network was trained on.
MIN_SIDE, MAX_SIDE = 800, 1800


@dataclass
class Ingested:
    rgb: np.ndarray        # working image, RGB uint8 (deskewed, resized)
    binary: np.ndarray     # ink mask, uint8 {0,255}, ink=255
    scale: float           # working px = input px * scale
    skew_deg: float        # rotation applied (deg, CCW) before scaling
    input_size: tuple[int, int]  # (w, h) of the input image


def read_any(path: str, dpi: int = 200) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(path)
        img = pdf[0].render(scale=dpi / 72).to_numpy()  # first page
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    data = np.fromfile(path, dtype=np.uint8)  # handles non-ASCII Windows paths
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"cannot decode image: {path}")
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:  # composite transparency onto white
        a = img[:, :, 3:4].astype(np.float32) / 255
        img = (img[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def binarize(rgb: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    g = cv2.medianBlur(g, 3)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return b


def estimate_skew(binary: np.ndarray) -> float:
    """Dominant line angle modulo 90 deg, in (-45, 45]. Small angles only matter."""
    lines = cv2.HoughLinesP(binary, 1, np.pi / 720, threshold=100,
                            minLineLength=max(binary.shape) // 15, maxLineGap=3)
    if lines is None:
        return 0.0
    l = lines.reshape(-1, 4).astype(np.float64)
    ang = np.degrees(np.arctan2(l[:, 3] - l[:, 1], l[:, 2] - l[:, 0]))
    ang = (ang + 45) % 90 - 45
    w = np.hypot(l[:, 2] - l[:, 0], l[:, 3] - l[:, 1])
    hist, edges = np.histogram(ang, bins=180, range=(-45, 45), weights=w)
    i = int(np.argmax(hist))
    return float((edges[i] + edges[i + 1]) / 2)


def ingest(path: str, ctx: RunContext, deskew: bool = True) -> Ingested:
    rgb = read_any(path)
    h, w = rgb.shape[:2]
    ctx.save_img("input", rgb)

    skew = 0.0
    if deskew:
        skew = estimate_skew(binarize(rgb))
        if abs(skew) > 0.5:  # ignore sub-degree noise; scanned plans need it
            M = cv2.getRotationMatrix2D((w / 2, h / 2), skew, 1.0)
            rgb = cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderValue=(255, 255, 255))
            ctx.fallback("ingest", f"deskewed by {skew:.2f} deg")
        else:
            skew = 0.0

    long_side = max(h, w)
    scale = 1.0
    if long_side > MAX_SIDE:
        scale = MAX_SIDE / long_side
    elif long_side < MIN_SIDE:
        scale = MIN_SIDE / long_side
    if scale != 1.0:
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        rgb = cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=interp)

    # Light denoise for scans; near-lossless on clean vector renders.
    rgb = cv2.fastNlMeansDenoisingColored(rgb, None, 3, 3, 7, 21) if _is_noisy(rgb) else rgb
    binary = binarize(rgb)
    ctx.save_img("binary", binary)
    return Ingested(rgb=rgb, binary=binary, scale=scale, skew_deg=skew, input_size=(w, h))


def _is_noisy(rgb: np.ndarray) -> bool:
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Fraction of mid-gray pixels: high for scans/photos, ~0 for clean renders.
    mid = np.mean((g > 60) & (g < 200))
    return mid > 0.08
