"""Monocular metric depth (Depth Anything V2, Metric-Hypersim ViT-S, via onnxruntime) aligned to the SfM model.

Two things come out of the alignment:
  * the global metric scale of the SfM reconstruction: the robust median of
    mono_depth / sfm_depth over every SfM observation (SfM is scale-free; the depth network
    was trained in metres on indoor scenes), and
  * per-frame dense depth consistent with the SfM geometry:
        median  one scale per frame (baseline)
        affine  robust a*d + b per frame
        grid    a smooth per-frame log-scale field on a coarse bilinear grid, ridge-regularised
                towards the frame's median, which absorbs the low-frequency warping that
                monocular depth shows on long walls (ours)
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

from core.log import RunContext, log

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DA_REPO = os.path.join(ROOT, "third_party", "Depth-Anything-V2", "metric_depth")
WEIGHTS = os.path.join(ROOT, "data", "weights")
ENCODERS = {"vits": ([48, 96, 192, 384], 64), "vitb": ([96, 192, 384, 768], 128)}
TRAIN_HFOV_DEG = 60.0  # every Hypersim image is rendered with a 60 degree horizontal FOV
INPUT_SIZE = 518       # short side fed to the network (multiple of 14); 364 is 2.7x faster, 35% worse
ENCODER = "vits"
_SESS: dict = {}


def _session(encoder: str, nh: int, nw: int):
    """onnxruntime session (3.7x faster than eager torch on a laptop CPU). Exported once per
    input shape from the official checkpoint and cached next to it: the ViT's positional-
    embedding interpolation is traced as constants, so one graph serves exactly one shape."""
    key = (encoder, nh, nw)
    if key in _SESS:
        return _SESS[key]
    import onnxruntime as ort
    pth = os.path.join(WEIGHTS, f"depth_anything_v2_metric_hypersim_{encoder}.pth")
    onx = pth[:-4] + f"_{nh}x{nw}.onnx"
    if not os.path.exists(onx):
        import torch
        if DA_REPO not in sys.path:
            sys.path.insert(0, DA_REPO)
        from depth_anything_v2.dpt import DepthAnythingV2
        ch, feat = ENCODERS[encoder]
        m = DepthAnythingV2(encoder=encoder, features=feat, out_channels=ch, max_depth=20.0)
        m.load_state_dict(torch.load(pth, map_location="cpu"))
        m.eval()
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(m, torch.randn(1, 3, nh, nw), onx, input_names=["x"], output_names=["d"],
                              opset_version=17, dynamo=False)
    so = ort.SessionOptions()
    so.intra_op_num_threads = max(1, min(10, (os.cpu_count() or 4) - 2))
    _SESS[key] = ort.InferenceSession(onx, so, providers=["CPUExecutionProvider"])
    return _SESS[key]


def infer(rgb: np.ndarray, input_size: int = INPUT_SIZE, encoder: str = ENCODER) -> np.ndarray:
    """Metric depth (metres) at the image's resolution; preprocessing as the reference infer_image."""
    h, w = rgb.shape[:2]
    k = max(input_size / h, input_size / w)

    def fit(x):
        y = int(np.round(x * k / 14) * 14)
        return y if y >= input_size else int(np.ceil(x * k / 14) * 14)
    nh, nw = fit(h), fit(w)
    x = cv2.resize(rgb.astype(np.float32) / 255.0, (nw, nh), interpolation=cv2.INTER_CUBIC)
    x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    x = np.ascontiguousarray(x.transpose(2, 0, 1)[None], np.float32)
    d = _session(encoder, nh, nw).run(None, {"x": x})[0][0]
    return cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)


def predict(images: list[np.ndarray], ids: list[int], cache_dir: str | None = None,
            input_size: int = INPUT_SIZE, encoder: str = ENCODER) -> dict[int, np.ndarray]:
    out = {}
    for i in ids:
        p = os.path.join(cache_dir, f"depth_{encoder}{input_size}_{i:04d}.npy") if cache_dir else None
        if p and os.path.exists(p):
            out[i] = np.load(p).astype(np.float32)
            continue
        out[i] = infer(images[i], input_size, encoder).astype(np.float32)
        if p:
            os.makedirs(cache_dir, exist_ok=True)
            np.save(p, out[i].astype(np.float16))
    return out


def _sample(d: np.ndarray, uv: np.ndarray) -> np.ndarray:
    return cv2.remap(d, uv[:, 0].astype(np.float32).reshape(-1, 1), uv[:, 1].astype(np.float32).reshape(-1, 1),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()


def sparse_depths(rec, im: int):
    """(uv, z) of the SfM points observed in image `im`, in SfM units."""
    sel = rec.obs_img == im
    uv = rec.obs_uv[sel]
    X = rec.X[rec.obs_pt[sel]]
    z = (X @ rec.R[im].T + rec.t[im])[:, 2]
    ok = z > 0
    return uv[ok], z[ok]


def metric_scale(rec, images: list[np.ndarray], K: np.ndarray, ids: list[int], canonical: bool = True,
                 n_frames: int = 60, mono: dict[int, np.ndarray] | None = None) -> tuple[float, dict]:
    """Metres per SfM unit = median over frames of median(mono depth / SfM depth).

    canonical=True runs the network on a centre crop that has the training camera's 60 deg
    horizontal FOV (resampled to the network's input size), so the metric prior sees the
    projection it was trained on; only SfM points inside the crop vote. A wide-angle phone
    or a synthetic 90 deg camera otherwise makes everything look farther away. Cameras
    narrower than 60 deg cannot be cropped wider and use the full frame (logged)."""
    ids = [i for i in ids if i in rec.R]
    if not ids:
        return 1.0, {"n_frames": 0}
    ids = [ids[k] for k in np.linspace(0, len(ids) - 1, min(n_frames, len(ids))).astype(int)]
    h, w = images[ids[0]].shape[:2]
    fx = K[0, 0]
    cw = 2 * fx * np.tan(np.radians(TRAIN_HFOV_DEG) / 2)
    crop = canonical and cw < w - 2
    ratios = []
    for im in ids:
        uv, z = sparse_depths(rec, im)
        if crop:
            ch = min(h, cw * 0.75)
            x0, y0 = K[0, 2] - cw / 2, K[1, 2] - ch / 2
            sel = (uv[:, 0] >= x0) & (uv[:, 0] < x0 + cw) & (uv[:, 1] >= y0) & (uv[:, 1] < y0 + ch)
            if sel.sum() < 10:
                continue
            xi, yi = int(round(max(0, x0))), int(round(max(0, y0)))
            sub = images[im][yi:yi + int(round(ch)), xi:xi + int(round(cw))]
            d = infer(sub)
            r = _sample(d, uv[sel] - [xi, yi]) / z[sel]
        else:
            if len(z) < 10:
                continue
            d = mono[im] if mono is not None and im in mono else predict(images, [im])[im]
            r = _sample(d, uv) / z
        ratios.append(np.median(r))
    if not ratios:
        return 1.0, {"n_frames": 0}
    ratios = np.array(ratios)
    s = float(np.median(ratios))
    return s, {"n_frames": len(ratios), "scale": s, "canonical_crop": bool(crop),
               "hfov_deg": round(float(np.degrees(2 * np.arctan(w / 2 / fx))), 1),
               "frame_spread_iqr": float(np.subtract(*np.percentile(ratios, [75, 25])) / s)}


def _grid_basis(uv: np.ndarray, w: int, h: int, gx: int, gy: int) -> np.ndarray:
    """Bilinear interpolation weights of each pixel on a gx x gy node grid -> (N, gx*gy)."""
    x = np.clip(uv[:, 0] / max(w - 1, 1) * (gx - 1), 0, gx - 1 - 1e-6)
    y = np.clip(uv[:, 1] / max(h - 1, 1) * (gy - 1), 0, gy - 1 - 1e-6)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    B = np.zeros((len(uv), gx * gy))
    r = np.arange(len(uv))
    for dx, dy, wt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        np.add.at(B, (r, (y0 + dy) * gx + x0 + dx), wt)
    return B


def align(rec, mono: dict[int, np.ndarray], scale: float, mode: str, ctx: RunContext,
          grid: tuple[int, int] = (4, 3), lam: float = 2.0) -> dict[int, np.ndarray]:
    """Mono depth -> depth consistent with the (metric-scaled) SfM model, per frame."""
    out = {}
    n_fb = 0
    for im, d in mono.items():
        if im not in rec.R:
            continue
        uv, z = sparse_depths(rec, im)
        z = z * scale
        dm = _sample(d, uv) if len(z) else np.zeros(0)
        ok = (dm > 0.05) & (z > 0.05)
        uv, z, dm = uv[ok], z[ok], dm[ok]
        if len(z) < 15:
            out[im] = d.copy()  # nothing to align to: trust the metric network as is
            n_fb += 1
            continue
        lr = np.log(z / dm)
        med = float(np.median(lr))
        if mode == "median":
            out[im] = d * np.exp(med)
        elif mode == "affine":
            a, b = np.exp(med), 0.0
            for _ in range(10):  # IRLS with Huber weights on z = a*dm + b
                r = z - (a * dm + b)
                s = 1.4826 * np.median(np.abs(r)) + 1e-6
                wgt = np.minimum(1.0, 1.5 * s / np.maximum(np.abs(r), 1e-9))
                A = np.stack([dm, np.ones_like(dm)], 1) * np.sqrt(wgt)[:, None]
                a, b = np.linalg.lstsq(A, z * np.sqrt(wgt), rcond=None)[0]
            if a <= 0:
                a, b = np.exp(med), 0.0
            out[im] = np.maximum(a * d + b, 0.0)
        elif mode == "grid":
            h, w = d.shape
            gx, gy = grid
            B = _grid_basis(uv, w, h, gx, gy)
            c = np.full(gx * gy, med)
            for _ in range(8):  # IRLS Huber, ridge towards the median log-scale
                r = lr - B @ c
                s = 1.4826 * np.median(np.abs(r)) + 1e-6
                wgt = np.minimum(1.0, 1.5 * s / np.maximum(np.abs(r), 1e-9))
                BtW = B.T * wgt
                c = np.linalg.solve(BtW @ B + lam * np.eye(gx * gy), BtW @ lr + lam * med)
            yy, xx = np.mgrid[0:h, 0:w]
            field = (_grid_basis(np.stack([xx.ravel(), yy.ravel()], 1), w, h, gx, gy) @ c).reshape(h, w)
            out[im] = d * np.exp(field).astype(np.float32)
        else:
            raise ValueError(mode)
    out = {k: v.astype(np.float32) for k, v in out.items()}
    if n_fb:
        ctx.fallback("depth", f"{n_fb} frame(s) had too few SfM points; used raw metric depth")
    log.info("depth aligned (%s) for %d frames", mode, len(out))
    return out


def consistency_filter(depths: dict[int, np.ndarray], K: np.ndarray, R: dict, t: dict,
                       n_nb: int = 6, rel_tol: float = 0.04, min_agree: int = 2) -> dict[int, np.ndarray]:
    """Multi-view geometric consistency: keep a pixel only if its 3D point lands on the
    depth surface of at least `min_agree` neighbouring views (relative depth error < rel_tol)
    or is not visible in them. Invalid pixels become 0. Removes flying pixels at depth
    edges and frame-specific hallucinations of the monocular network."""
    ids = sorted(depths)
    C = {i: -R[i].T @ t[i] for i in ids}
    out = {}
    for i in ids:
        d = depths[i]
        h, w = d.shape
        v, u = np.mgrid[0:h, 0:w]
        x = np.stack([(u - K[0, 2]) / K[0, 0] * d, (v - K[1, 2]) / K[1, 1] * d, d], -1).reshape(-1, 3)
        Xw = (x - t[i]) @ R[i]
        nbs = sorted((j for j in ids if j != i), key=lambda j: abs(j - i))[:n_nb]
        agree = np.zeros(h * w, np.int16)
        seen = np.zeros(h * w, np.int16)
        for j in nbs:
            Xc = Xw @ R[j].T + t[j]
            z = Xc[:, 2]
            ok = z > 0.05
            pu = np.where(ok, Xc[:, 0] / np.maximum(z, 1e-6) * K[0, 0] + K[0, 2], -1)
            pv = np.where(ok, Xc[:, 1] / np.maximum(z, 1e-6) * K[1, 1] + K[1, 2], -1)
            inb = ok & (pu >= 0) & (pu <= w - 1) & (pv >= 0) & (pv <= h - 1)
            dj = np.zeros(h * w, np.float32)
            dj[inb] = depths[j][np.round(pv[inb]).astype(int), np.round(pu[inb]).astype(int)]
            valid = inb & (dj > 0)
            rel = np.abs(dj - z) / np.maximum(z, 1e-6)
            seen += valid
            agree += valid & (rel < rel_tol)
        keep = (agree >= min_agree) | (seen < min_agree)  # never seen elsewhere: can't judge, keep
        out[i] = np.where(keep.reshape(h, w), d, 0.0).astype(np.float32)
    return out
