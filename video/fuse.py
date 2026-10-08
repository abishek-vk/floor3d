"""TSDF fusion of aligned depth maps (torch, CPU) -> observed surface mesh.

Every voxel keeps a truncated signed distance, an integration weight and a colour. A voxel
with weight > 0 was *observed* (some camera saw the surface or the free space through it),
which is what the honesty bookkeeping downstream is built on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from skimage.measure import marching_cubes

from core.log import RunContext, log

MAX_VOXELS = 30_000_000


@dataclass
class Volume:
    origin: np.ndarray       # world position of voxel (0,0,0) centre
    voxel: float
    tsdf: np.ndarray         # (X,Y,Z) float32 in [-1, 1]
    weight: np.ndarray       # (X,Y,Z) float32
    color: np.ndarray        # (X,Y,Z,3) float32 0-255
    trunc: float

    def world_to_index(self, P: np.ndarray) -> np.ndarray:
        return (P - self.origin) / self.voxel


def scene_bounds(depths, K, R, t, stride: int = 8, max_depth: float = 8.0):
    pts = []
    for i, d in depths.items():
        h, w = d.shape
        v, u = np.mgrid[0:h:stride, 0:w:stride]
        z = d[::stride, ::stride]
        ok = (z > 0) & (z < max_depth)
        x = np.stack([(u[ok] - K[0, 2]) / K[0, 0] * z[ok], (v[ok] - K[1, 2]) / K[1, 1] * z[ok], z[ok]], 1)
        pts.append((x - t[i]) @ R[i])
        pts.append((-R[i].T @ t[i])[None])
    P = np.concatenate(pts)
    lo, hi = np.percentile(P, 0.5, 0), np.percentile(P, 99.5, 0)
    pad = 0.1 * (hi - lo) + 0.1
    return lo - pad, hi + pad


def integrate(depths: dict[int, np.ndarray], images: list[np.ndarray], K, R, t, ctx: RunContext,
              voxel: float = 0.02, trunc_vox: float = 4.0, max_depth: float = 8.0, bounds=None) -> Volume:
    lo, hi = bounds if bounds is not None else scene_bounds(depths, K, R, t, max_depth=max_depth)
    dims = np.ceil((hi - lo) / voxel).astype(int)
    if dims.prod() > MAX_VOXELS:
        k = (dims.prod() / MAX_VOXELS) ** (1 / 3)
        voxel *= k
        dims = np.ceil((hi - lo) / voxel).astype(int)
        ctx.fallback("fuse", f"scene too large for the voxel budget; voxel enlarged to {voxel * 100:.1f} cm")
    trunc = trunc_vox * voxel
    log.info("tsdf grid %s voxels of %.1f cm", dims.tolist(), voxel * 100)
    n = int(dims.prod())
    tsdf = torch.ones(n)
    wgt = torch.zeros(n)
    col = torch.zeros(n, 3)
    ii, jj, kk = torch.meshgrid(*[torch.arange(d, dtype=torch.float32) for d in dims], indexing="ij")
    P = torch.stack([ii.reshape(-1), jj.reshape(-1), kk.reshape(-1)], 1) * voxel + torch.tensor(lo, dtype=torch.float32)
    del ii, jj, kk
    fx, fy, cx, cy = (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
    for i, d in depths.items():
        h, w = d.shape
        Rt = torch.tensor(R[i], dtype=torch.float32)
        tt = torch.tensor(t[i], dtype=torch.float32)
        Xc = P @ Rt.T + tt
        z = Xc[:, 2]
        front = (z > 0.05) & (z < max_depth + trunc)
        idx = torch.nonzero(front).squeeze(1)
        z = z[idx]
        u = torch.round(Xc[idx, 0] / z * fx + cx).long()
        v = torch.round(Xc[idx, 1] / z * fy + cy).long()
        inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        idx, z, u, v = idx[inb], z[inb], u[inb], v[inb]
        dt = torch.from_numpy(np.ascontiguousarray(d, np.float32))
        dd = dt[v, u]
        sdf = dd - z
        ok = (dd > 0) & (dd < max_depth) & (sdf > -trunc)
        idx, sdf, u, v, dd = idx[ok], sdf[ok], u[ok], v[ok], dd[ok]
        s = torch.clamp(sdf / trunc, max=1.0)
        wn = 1.0 / torch.clamp(dd, min=0.3)  # nearer observations are sharper
        w0 = wgt[idx]
        tsdf[idx] = (tsdf[idx] * w0 + s * wn) / (w0 + wn)
        near = sdf < trunc  # colour only near the surface
        ci = idx[near]
        cw = wn[near]
        rgb = torch.from_numpy(images[i]).float()[v[near], u[near]]
        cw0 = wgt[ci]
        col[ci] = (col[ci] * cw0[:, None] + rgb * cw[:, None]) / (cw0 + cw)[:, None]
        wgt[idx] = w0 + wn
    shape = tuple(dims.tolist())
    return Volume(np.asarray(lo, float), voxel, tsdf.reshape(shape).numpy(), wgt.reshape(shape).numpy(),
                  col.reshape(shape + (3,)).numpy(), trunc)


def extract_mesh(vol: Volume, min_weight: float = 0.5, min_component_faces: int = 200) -> trimesh.Trimesh:
    """Marching cubes on observed voxels; vertex colours from the colour grid; floaters removed."""
    mask = vol.weight >= min_weight
    if mask.sum() < 8 or vol.tsdf[mask].min() > 0 or vol.tsdf[mask].max() < 0:
        return trimesh.Trimesh()
    verts, faces, normals, _ = marching_cubes(vol.tsdf, level=0.0, spacing=(vol.voxel,) * 3, mask=mask,
                                              allow_degenerate=False)
    gi = np.clip(np.round(verts / vol.voxel).astype(int), 0, np.array(vol.tsdf.shape) - 1)
    vc = vol.color[gi[:, 0], gi[:, 1], gi[:, 2]]
    verts = verts + vol.origin
    m = trimesh.Trimesh(verts, faces, vertex_colors=np.c_[np.clip(vc, 0, 255), np.full(len(vc), 255)].astype(np.uint8),
                        process=False)
    return remove_small_components(m, min_component_faces)


def remove_small_components(m: trimesh.Trimesh, min_faces: int) -> trimesh.Trimesh:
    if len(m.faces) == 0:
        return m
    f = m.faces
    nv = len(m.vertices)
    A = coo_matrix((np.ones(3 * len(f)), (np.r_[f[:, 0], f[:, 1], f[:, 2]], np.r_[f[:, 1], f[:, 2], f[:, 0]])),
                   shape=(nv, nv))
    _, lab = connected_components(A, directed=False)
    fl = lab[f[:, 0]]
    cnt = np.bincount(fl)
    keep = cnt[fl] >= min_faces
    m.update_faces(keep)
    m.remove_unreferenced_vertices()
    return m
