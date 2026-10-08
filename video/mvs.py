"""Mono-prior-guided plane-sweep stereo (ours).

Monocular depth gets the layout of a room right but its *shape* is soft (~9 cm median error
on Replica even after an oracle per-frame scale). Multi-view photometric evidence is sharp
but ambiguous on texture-less walls. We combine them per pixel:

    hypotheses   d = d_prior * exp(delta), delta in [-range, +range]  (a narrow band around
                 the aligned monocular depth instead of the whole frustum)
    cost         1 - NCC over a 7x7 window between the reference image and each source view
                 warped through the hypothesis; sources are chosen for a useful triangulation
                 angle; the mean of the best half of the sources is kept (occlusion-robust)
    decision     winner-take-all with parabolic sub-sample refinement, accepted only where the
                 reference patch is textured and the cost curve has a clear minimum; elsewhere
                 the prior is kept.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as Fn

from core.log import log


def _gray(img: np.ndarray) -> torch.Tensor:
    g = torch.from_numpy(img.astype(np.float32)) @ torch.tensor([0.299, 0.587, 0.114])
    return g / 255.0


def _box(x: torch.Tensor, r: int) -> torch.Tensor:
    return Fn.avg_pool2d(x[None, None], 2 * r + 1, 1, r, count_include_pad=False)[0, 0]


def pick_sources(i: int, ids: list[int], C: dict, fwd: dict, depth_med: float, n_src: int = 4,
                 ang_lo: float = 3.0, ang_hi: float = 25.0) -> list[int]:
    cand = []
    for j in ids:
        if j == i:
            continue
        if fwd[i] @ fwd[j] < np.cos(np.radians(45)):  # must look at roughly the same thing
            continue
        b = np.linalg.norm(C[j] - C[i])
        ang = np.degrees(2 * np.arctan(b / 2 / max(depth_med, 1e-3)))
        if ang < 0.5 * ang_lo:
            continue
        pen = 0 if ang_lo <= ang <= ang_hi else min(abs(ang - ang_lo), abs(ang - ang_hi))
        cand.append((pen, abs(j - i), j))
    cand.sort()
    return [j for _, _, j in cand[:n_src]]


def refine_depth(i: int, prior: np.ndarray, images: list[np.ndarray], K, R: dict, t: dict, srcs: list[int],
                 rel_range: float = 0.25, n_hyp: int = 33, r: int = 3, min_std: float = 0.02,
                 max_cost: float = 0.35):
    h, w = prior.shape
    ref = _gray(images[i])
    mu_r = _box(ref, r)
    sd_r = (_box(ref * ref, r) - mu_r ** 2).clamp(min=0).sqrt()
    deltas = torch.linspace(-rel_range, rel_range, n_hyp)
    d0 = torch.from_numpy(prior.astype(np.float32))
    v, u = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    rays = torch.stack([(u - float(K[0, 2])) / float(K[0, 0]), (v - float(K[1, 2])) / float(K[1, 1]), torch.ones_like(u)], -1)
    Ri, ti = torch.from_numpy(R[i].astype(np.float32)), torch.from_numpy(t[i].astype(np.float32))
    costs = torch.empty(len(srcs), n_hyp, h, w)
    for s_k, j in enumerate(srcs):
        Rj, tj = torch.from_numpy(R[j].astype(np.float32)), torch.from_numpy(t[j].astype(np.float32))
        Rrel = Rj @ Ri.T
        trel = tj - Rrel @ ti
        src = _gray(images[j])[None, None]
        a = rays @ Rrel.T  # direction part, (h,w,3)
        for k, dl in enumerate(deltas):
            d = d0 * torch.exp(dl)
            P = a * d[..., None] + trel
            z = P[..., 2].clamp(min=1e-4)
            gx = (P[..., 0] / z * float(K[0, 0]) + float(K[0, 2])) / (w - 1) * 2 - 1
            gy = (P[..., 1] / z * float(K[1, 1]) + float(K[1, 2])) / (h - 1) * 2 - 1
            warped = Fn.grid_sample(src, torch.stack([gx, gy], -1)[None], mode="bilinear",
                                    padding_mode="zeros", align_corners=True)[0, 0]
            valid = (gx.abs() <= 1) & (gy.abs() <= 1) & (P[..., 2] > 0)
            mu_s = _box(warped, r)
            sd_s = (_box(warped * warped, r) - mu_s ** 2).clamp(min=0).sqrt()
            cov = _box(ref * warped, r) - mu_r * mu_s
            ncc = cov / (sd_r * sd_s + 1e-4)
            costs[s_k, k] = torch.where(valid, 1 - ncc, torch.full_like(ncc, 2.0))
    # best half of the sources per hypothesis
    m = max(1, len(srcs) // 2)
    cost = torch.sort(costs, 0).values[:m].mean(0)
    best = cost.argmin(0)
    cmin = cost.gather(0, best[None])[0]
    # parabolic refinement
    bi = best.clamp(1, n_hyp - 2)
    c_m = cost.gather(0, (bi - 1)[None])[0]
    c_0 = cost.gather(0, bi[None])[0]
    c_p = cost.gather(0, (bi + 1)[None])[0]
    den = c_m - 2 * c_0 + c_p
    off = torch.where(den > 1e-6, 0.5 * (c_m - c_p) / den, torch.zeros_like(den)).clamp(-0.5, 0.5)
    step = float(deltas[1] - deltas[0])
    dl = deltas[bi] + off * step
    # a clear minimum: interior, sharp (curvature), low cost, textured reference
    interior = (best > 0) & (best < n_hyp - 1)
    conf = interior & (cmin < max_cost) & (sd_r > min_std) & (den > 0.02)
    out = torch.where(conf, d0 * torch.exp(dl), d0)
    return out.numpy(), conf.numpy()


def refine_all(depths: dict[int, np.ndarray], images: list[np.ndarray], K, R: dict, t: dict,
               all_ids: list[int] | None = None, **kw):
    """Refine every depth map; source views may be any posed frame in `all_ids` (they need
    an image and a pose, not a depth map)."""
    ids = sorted(depths)
    pool = sorted(set(ids) | set(all_ids or []))
    C = {i: -R[i].T @ t[i] for i in pool}
    fwd = {i: R[i][2] for i in pool}
    out, confs, frac = {}, {}, []
    for i in ids:
        dm = float(np.median(depths[i][depths[i] > 0])) if (depths[i] > 0).any() else 2.0
        srcs = pick_sources(i, all_ids or ids, C, fwd, dm)
        if len(srcs) < 2:
            out[i], confs[i] = depths[i], np.zeros(depths[i].shape, bool)
            continue
        out[i], confs[i] = refine_depth(i, depths[i], images, K, R, t, srcs, **kw)
        frac.append(confs[i].mean())
    log.info("mvs: refined %.0f%% of pixels on average", 100 * float(np.mean(frac)) if frac else 0)
    return out, confs, float(np.mean(frac)) if frac else 0.0


def reanchor(mono: np.ndarray, ref: np.ndarray, conf: np.ndarray, grid: tuple[int, int] = (8, 6),
             lam: float = 1.0, max_pts: int = 20000) -> np.ndarray:
    """Re-fit the monocular depth's smooth log-scale field to the confident stereo depths
    (tens of thousands of anchors instead of a few hundred SfM points)."""
    from video.depth import _grid_basis
    v, u = np.nonzero(conf & (mono > 0.05) & (ref > 0.05))
    if len(v) < 200:
        return None
    if len(v) > max_pts:
        k = np.random.default_rng(0).choice(len(v), max_pts, replace=False)
        v, u = v[k], u[k]
    h, w = mono.shape
    gx, gy = grid
    lr = np.log(ref[v, u] / mono[v, u])
    med = float(np.median(lr))
    B = _grid_basis(np.stack([u, v], 1).astype(float), w, h, gx, gy)
    c = np.full(gx * gy, med)
    for _ in range(6):
        r = lr - B @ c
        s = 1.4826 * np.median(np.abs(r)) + 1e-6
        wgt = np.minimum(1.0, 1.5 * s / np.maximum(np.abs(r), 1e-9))
        BtW = B.T * wgt
        c = np.linalg.solve(BtW @ B + lam * np.eye(gx * gy), BtW @ lr + lam * med)
    yy, xx = np.mgrid[0:h, 0:w]
    field = (_grid_basis(np.stack([xx.ravel(), yy.ravel()], 1).astype(float), w, h, gx, gy) @ c).reshape(h, w)
    return (mono * np.exp(field)).astype(np.float32)
