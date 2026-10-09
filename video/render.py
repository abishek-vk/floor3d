"""CPU rendering for evaluation and view synthesis (torch, no OpenGL).

rasterize   exact z-buffered triangle rasteriser: every (triangle, pixel-in-bbox) pair is
            tested with edge functions; the nearest hit per pixel wins via a scatter-min on a
            packed (depth, face id) int64 key. Perspective-correct barycentrics.
render_mesh vertex colours / per-face labels of a mesh seen from a camera
render_ibr  image-based rendering on top of the reconstructed geometry (ours): every pixel
            of the novel view is re-projected into the nearest training views, occlusion-
            tested against their rendered depth, and blended with unstructured-lumigraph
            weights (ray-angle and distance penalties). Pixels no training view sees fall
            back to the mesh's own colours, so generated regions keep their generated texture.
"""
from __future__ import annotations

import numpy as np
import torch

PAIR_BUDGET = 12_000_000  # (triangle, pixel) pairs per chunk
NEAR = 0.05


def rasterize(V: np.ndarray, F: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray, h: int, w: int):
    """-> depth (h,w) float32 (0 = empty), face (h,w) int64 (-1 = empty), bary (h,w,3) float32."""
    Vt = torch.from_numpy(np.asarray(V, np.float32))
    Ft = torch.from_numpy(np.asarray(F, np.int64))
    Xc = Vt @ torch.from_numpy(np.asarray(R, np.float32)).T + torch.from_numpy(np.asarray(t, np.float32))
    z = Xc[:, 2]
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    zs = torch.clamp(z, min=1e-6)
    u = Xc[:, 0] / zs * fx + cx
    v = Xc[:, 1] / zs * fy + cy
    zf = z[Ft]
    keep = (zf > NEAR).all(1)
    uf, vf = u[Ft], v[Ft]
    x0 = torch.clamp(torch.ceil(uf.min(1).values), min=0)
    x1 = torch.clamp(torch.floor(uf.max(1).values), max=w - 1)
    y0 = torch.clamp(torch.ceil(vf.min(1).values), min=0)
    y1 = torch.clamp(torch.floor(vf.max(1).values), max=h - 1)
    keep &= (x1 >= x0) & (y1 >= y0)
    fid = torch.nonzero(keep).squeeze(1)
    bw = (x1 - x0 + 1)[fid].long()
    bh = (y1 - y0 + 1)[fid].long()
    cnt = bw * bh
    INF = torch.iinfo(torch.int64).max
    zbuf = torch.full((h * w,), INF, dtype=torch.int64)
    # chunk the faces so the number of (face, pixel) pairs stays bounded
    csum = torch.cumsum(cnt, 0)
    starts = [0]
    while starts[-1] < len(fid):
        base = csum[starts[-1] - 1] if starts[-1] > 0 else 0
        nxt = int(torch.searchsorted(csum, base + PAIR_BUDGET, right=True))
        starts.append(max(nxt, starts[-1] + 1))
    starts[-1] = len(fid)
    for a, b in zip(starts[:-1], starts[1:]):
        f = fid[a:b]
        c = cnt[a:b]
        rep = torch.repeat_interleave(torch.arange(len(f)), c)
        off = torch.arange(int(c.sum())) - torch.repeat_interleave(torch.cumsum(c, 0) - c, c)
        bwr = bw[a:b][rep]
        px = x0[f][rep] + (off % bwr).float()
        py = y0[f][rep] + torch.div(off, bwr, rounding_mode="floor").float()
        ff = f[rep]
        l0, l1, l2, area = _bary(uf[ff], vf[ff], px, py)
        inside = (l0 >= -1e-4) & (l1 >= -1e-4) & (l2 >= -1e-4) & (area.abs() > 1e-12)
        zt = zf[ff]
        inv = l0 / zt[:, 0] + l1 / zt[:, 1] + l2 / zt[:, 2]
        inside &= inv > 0
        depth = 1.0 / torch.clamp(inv, min=1e-9)
        pix = (py * w + px).long()
        key = (torch.clamp(depth * 1e4, max=2**30).long() << 32) | ff
        zbuf.scatter_reduce_(0, pix[inside], key[inside], reduce="amin")
    hit = zbuf != INF
    face = torch.full((h * w,), -1, dtype=torch.int64)
    face[hit] = zbuf[hit] & 0xFFFFFFFF
    # exact barycentrics/depth for the winners
    pidx = torch.nonzero(hit).squeeze(1)
    fw = face[pidx]
    px, py = (pidx % w).float(), torch.div(pidx, w, rounding_mode="floor").float()
    l0, l1, l2, _ = _bary(uf[fw], vf[fw], px, py)
    zt = zf[fw]
    q = torch.stack([l0 / zt[:, 0], l1 / zt[:, 1], l2 / zt[:, 2]], 1)
    s = q.sum(1, keepdim=True)
    depth = torch.zeros(h * w)
    depth[pidx] = (1.0 / s).squeeze(1)
    bary = torch.zeros(h * w, 3)
    bary[pidx] = q / s
    return depth.reshape(h, w).numpy(), face.reshape(h, w).numpy(), bary.reshape(h, w, 3).numpy()


def _bary(uf, vf, px, py):
    ax, ay, bx, by, cx, cy = uf[:, 0], vf[:, 0], uf[:, 1], vf[:, 1], uf[:, 2], vf[:, 2]
    area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    sa = torch.where(area.abs() > 1e-12, area, torch.ones_like(area))
    l0 = ((bx - px) * (cy - py) - (by - py) * (cx - px)) / sa
    l1 = ((cx - px) * (ay - py) - (cy - py) * (ax - px)) / sa
    return l0, l1, 1 - l0 - l1, area


def render_mesh(V, F, vcol, K, R, t, h, w, face_label=None, bg=0):
    """-> rgb (h,w,3) uint8, depth (h,w), label (h,w) int (-1 = empty)."""
    depth, face, bary = rasterize(V, F, K, R, t, h, w)
    rgb = np.full((h, w, 3), bg, np.float32)
    hit = face >= 0
    fv = F[face[hit]]
    c = vcol[:, :3].astype(np.float32)
    rgb[hit] = (c[fv] * bary[hit][:, :, None]).sum(1)
    lab = np.full((h, w), -1, np.int64)
    if face_label is not None:
        lab[hit] = face_label[face[hit]]
    return np.clip(rgb + 0.5, 0, 255).astype(np.uint8), depth, lab


def _sample_rgb(img: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    h, w = img.shape[:2]
    g = torch.stack([u / (w - 1) * 2 - 1, v / (h - 1) * 2 - 1], -1).view(1, 1, -1, 2)
    out = torch.nn.functional.grid_sample(img.permute(2, 0, 1)[None], g, mode="bilinear", align_corners=True)
    return out[0, :, 0].T


def render_ibr(V, F, vcol, K, R, t, h, w, src_imgs: dict, src_R: dict, src_t: dict, src_depth: dict,
               n_src: int = 4, face_label=None, depth_tol: float = 0.03, sigma_deg: float = 12.0):
    """Novel view by blending re-projected training views over our geometry (see module doc)."""
    base, depth, lab = render_mesh(V, F, vcol, K, R, t, h, w, face_label)
    hit = depth > 0
    if not hit.any() or not src_imgs:
        return base, depth, lab, np.zeros((h, w), bool)
    C = -R.T @ t
    fwd = R[2]
    ids = list(src_imgs)
    score = []
    for j in ids:
        Cj = -src_R[j].T @ src_t[j]
        score.append(np.degrees(np.arccos(np.clip(src_R[j][2] @ fwd, -1, 1))) + 20 * np.linalg.norm(Cj - C))
    ids = [ids[k] for k in np.argsort(score)[:n_src]]
    v, u = np.nonzero(hit)
    z = depth[v, u]
    Xc = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1)
    Xw = torch.from_numpy(((Xc - t) @ R).astype(np.float32))
    ray_t = Xw - torch.from_numpy(C.astype(np.float32))
    ray_t = ray_t / ray_t.norm(dim=1, keepdim=True)
    acc = torch.zeros(len(Xw), 3)
    wsum = torch.zeros(len(Xw))
    for j in ids:
        Rj = torch.from_numpy(src_R[j].astype(np.float32))
        tj = torch.from_numpy(src_t[j].astype(np.float32))
        Xj = Xw @ Rj.T + tj
        zj = Xj[:, 2]
        uj = Xj[:, 0] / zj.clamp(min=1e-6) * float(K[0, 0]) + float(K[0, 2])
        vj = Xj[:, 1] / zj.clamp(min=1e-6) * float(K[1, 1]) + float(K[1, 2])
        hj, wj = src_depth[j].shape
        ok = (zj > NEAR) & (uj >= 0) & (uj <= wj - 1) & (vj >= 0) & (vj <= hj - 1)
        dj = torch.from_numpy(src_depth[j])
        dsrc = dj[vj.round().long().clamp(0, hj - 1), uj.round().long().clamp(0, wj - 1)]
        ok &= (dsrc > 0) & ((dsrc - zj).abs() < depth_tol * zj + 0.02)
        Cj = torch.from_numpy((-src_R[j].T @ src_t[j]).astype(np.float32))
        ray_j = Xw - Cj
        dist_j = ray_j.norm(dim=1)
        ray_j = ray_j / dist_j[:, None]
        ang = torch.rad2deg(torch.arccos(torch.clamp((ray_t * ray_j).sum(1), -1, 1)))
        dist_t = (Xw - torch.from_numpy(C.astype(np.float32))).norm(dim=1)
        wj_ = torch.exp(-(ang / sigma_deg) ** 2) / (1 + 2 * torch.clamp(dist_j / dist_t - 1, min=0))
        wj_ = torch.where(ok, wj_, torch.zeros_like(wj_))
        col = _sample_rgb(torch.from_numpy(src_imgs[j]).float(), uj, vj)
        acc += col * wj_[:, None]
        wsum += wj_
    out = base.astype(np.float32)
    covered = (wsum > 1e-3).numpy()
    blend = (acc / wsum.clamp(min=1e-9)[:, None]).numpy()
    # fade to the mesh colour where the evidence is weak
    a = np.clip(wsum.numpy() / 0.3, 0, 1)[:, None]
    out[v, u] = np.where(covered[:, None], a * blend + (1 - a) * out[v, u], out[v, u])
    ibr_mask = np.zeros((h, w), bool)
    ibr_mask[v[covered], u[covered]] = True
    return np.clip(out + 0.5, 0, 255).astype(np.uint8), depth, lab, ibr_mask
