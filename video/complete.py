"""Completion of the unseen room shell, with explicit provenance (ours).

Every shell surface (floor, ceiling, each wall of the layout) is rasterised into texels and
each texel gets exactly one label:

    OBSERVED   reconstructed surface lies on the plane here (kept as captured, never replaced)
    OPENING    the camera saw *through* the plane here (doorway, window, open side): free
               space on the plane and observed space behind it. Never filled.
    GENERATED  nobody saw this texel (behind furniture, outside the video's view, a ceiling
               the camera never pointed at). Filled with plane geometry and a synthesised
               texture: low frequencies from inpainting the observed texels, high frequencies
               transferred from observed patches of the same surface.

Generated geometry is emitted as a separate mesh with per-face provenance, so the exporter,
viewer and evaluator can always tell it apart from what was observed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import trimesh

OBSERVED, OPENING, GENERATED, OUTSIDE = 1, 2, 3, 0
TEXEL = 0.03


@dataclass
class Surface:
    name: str
    origin: np.ndarray      # room-frame point of texel (0,0) corner
    eu: np.ndarray          # room-frame unit vector along texel columns
    ev: np.ndarray          # room-frame unit vector along texel rows
    normal: np.ndarray      # room-frame inward normal
    shape: tuple[int, int]  # rows, cols
    inside: np.ndarray      # bool (rows, cols): texel belongs to the surface
    label: np.ndarray | None = None
    color: np.ndarray | None = None
    stats: dict = field(default_factory=dict)

    def centers(self) -> np.ndarray:
        r, c = np.mgrid[0:self.shape[0], 0:self.shape[1]]
        return self.origin + ((c + 0.5) * TEXEL)[..., None] * self.eu + ((r + 0.5) * TEXEL)[..., None] * self.ev


def shell_surfaces(rf) -> list[Surface]:
    from matplotlib.path import Path
    out = []
    poly = rf.polygon
    lo, hi = poly.min(0), poly.max(0)
    cols, rows = np.ceil((hi - lo) / TEXEL).astype(int)
    r, c = np.mgrid[0:rows, 0:cols]
    xz = np.stack([lo[0] + (c + 0.5) * TEXEL, lo[1] + (r + 0.5) * TEXEL], -1).reshape(-1, 2)
    inside = Path(poly).contains_points(xz).reshape(rows, cols)
    ex, ez = np.array([1.0, 0, 0]), np.array([0, 0, 1.0])
    out.append(Surface("floor", np.array([lo[0], rf.floor_y, lo[1]]), ex, ez, np.array([0, 1.0, 0]), (rows, cols), inside))
    out.append(Surface("ceiling", np.array([lo[0], rf.ceil_y, lo[1]]), ex, ez, np.array([0, -1.0, 0]), (rows, cols), inside.copy()))
    H = rf.ceil_y - rf.floor_y
    for k, w in enumerate(rf.wall_planes):
        p0, p1 = np.array(w["p0"]), np.array(w["p1"])
        L = np.linalg.norm(p1 - p0)
        if L < 2 * TEXEL:
            continue
        d = (p1 - p0) / L
        eu = np.array([d[0], 0, d[1]])
        n = np.zeros(3)
        n[w["axis"]] = w["normal"]
        rows_w, cols_w = int(np.ceil(H / TEXEL)), int(np.ceil(L / TEXEL))
        out.append(Surface(f"wall_{k}", np.array([p0[0], rf.floor_y, p0[1]]), eu, np.array([0, 1.0, 0]), n,
                           (rows_w, cols_w), np.ones((rows_w, cols_w), bool)))
    return out


def _vol_sample(vol, P_world: np.ndarray):
    idx = np.round((P_world - vol.origin) / vol.voxel).astype(int)
    ok = np.all((idx >= 0) & (idx < np.array(vol.tsdf.shape)), 1)
    w = np.zeros(len(P_world))
    s = np.ones(len(P_world))
    w[ok] = vol.weight[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
    s[ok] = vol.tsdf[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
    return w, s


def classify(surfs: list[Surface], m: trimesh.Trimesh, rf, vol, band: float = 0.10):
    """Label every texel OBSERVED / OPENING / GENERATED and collect observed colours."""
    P = m.vertices @ rf.Rm.T
    N = m.vertex_normals @ rf.Rm.T
    col = m.visual.vertex_colors[:, :3].astype(np.float32)
    for S in surfs:
        d = (P - S.origin) @ S.normal
        sel = (np.abs(d) < band) & (N @ S.normal > 0.5)
        uv = np.stack([(P[sel] - S.origin) @ S.eu, (P[sel] - S.origin) @ S.ev], 1) / TEXEL
        rows, cols = S.shape
        ci, ri = np.floor(uv[:, 0]).astype(int), np.floor(uv[:, 1]).astype(int)
        ok = (ci >= 0) & (ci < cols) & (ri >= 0) & (ri < rows)
        acc = np.zeros((rows, cols, 3), np.float32)
        cnt = np.zeros((rows, cols), np.float32)
        np.add.at(acc, (ri[ok], ci[ok]), col[sel][ok])
        np.add.at(cnt, (ri[ok], ci[ok]), 1)
        obs = cnt > 0
        obs = cv2.morphologyEx(obs.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
        color = np.where(cnt[..., None] > 0, acc / np.maximum(cnt, 1)[..., None], 0)
        # openings: free on the plane, and space observed 30 cm behind it (well past the depth
        # noise that leaks behind solid walls); at least ~33 cm wide
        Cw = S.centers().reshape(-1, 3)
        w_on, s_on = _vol_sample(vol, (Cw + 0.03 * S.normal) @ rf.Rm)
        w_bk, s_bk = _vol_sample(vol, (Cw - 0.30 * S.normal) @ rf.Rm)
        opening = ((w_on > 0) & (s_on > 0.9) & (w_bk > 0) & (s_bk > -0.5)).reshape(rows, cols)
        opening = cv2.morphologyEx(opening.astype(np.uint8), cv2.MORPH_OPEN, np.ones((11, 11), np.uint8)).astype(bool)
        lab = np.full((rows, cols), OUTSIDE, np.uint8)
        lab[S.inside] = GENERATED
        lab[S.inside & opening & ~obs] = OPENING
        lab[S.inside & obs] = OBSERVED
        S.label, S.color = lab, color
        n_in = max(int(S.inside.sum()), 1)
        S.stats = {"area_m2": round(n_in * TEXEL ** 2, 3),
                   "observed": round(float((lab == OBSERVED).sum() / n_in), 4),
                   "opening": round(float((lab == OPENING).sum() / n_in), 4),
                   "generated": round(float((lab == GENERATED).sum() / n_in), 4)}


def synthesize(S: Surface, rng: np.random.Generator, fallback_rgb: np.ndarray, patch: int = 8):
    """Texture for GENERATED texels: inpainted low frequencies + observed high-frequency patches."""
    lab = S.label
    obs = lab == OBSERVED
    gen = lab == GENERATED
    if not gen.any():
        return
    tex = S.color.copy()
    if obs.sum() < 50:  # nothing to learn from on this surface: flat prior colour
        tex[gen] = fallback_rgb
        S.color = tex
        S.stats["texture"] = "prior_colour"
        return
    # low frequency: inpaint a 4x-downsampled copy, so large holes get smooth trends, not smears
    k = 4
    # low frequencies are inpainted from typical surface texels only (outliers like a TV excluded)
    med0 = np.median(tex[obs], 0)
    obs_lf = obs & (np.linalg.norm(tex - med0, axis=-1) < 40)
    obs_lf = obs_lf if obs_lf.sum() >= 50 else obs
    small = cv2.resize(np.where(obs_lf[..., None], tex, 0).astype(np.float32), None, fx=1 / k, fy=1 / k, interpolation=cv2.INTER_AREA)
    wsm = cv2.resize(obs_lf.astype(np.float32), None, fx=1 / k, fy=1 / k, interpolation=cv2.INTER_AREA)
    small = np.where(wsm[..., None] > 0.05, small / np.maximum(wsm, 1e-6)[..., None], 0)
    hole = (wsm <= 0.05).astype(np.uint8)
    small8 = np.clip(small, 0, 255).astype(np.uint8)
    low = cv2.inpaint(small8, hole, 5, cv2.INPAINT_TELEA) if hole.any() else small8
    low = cv2.resize(low.astype(np.float32), (lab.shape[1], lab.shape[0]), interpolation=cv2.INTER_LINEAR)
    low = cv2.GaussianBlur(low, (0, 0), 2.0)
    # high frequency: residuals of fully observed patches, pasted at random
    lo_obs = cv2.GaussianBlur(np.where(obs[..., None], tex, low).astype(np.float32), (0, 0), 2.0)
    resid = tex - lo_obs
    rows, cols = lab.shape
    # only patches that look like the surface itself (not a TV, picture or furniture in front)
    med = np.median(tex[obs], 0)
    cand = [(r, c) for r in range(0, rows - patch + 1, patch // 2) for c in range(0, cols - patch + 1, patch // 2)
            if obs[r:r + patch, c:c + patch].all()
            and np.linalg.norm(tex[r:r + patch, c:c + patch].reshape(-1, 3).mean(0) - med) < 25]
    hf = np.zeros_like(tex)
    if cand:
        for r in range(0, rows, patch):
            for c in range(0, cols, patch):
                if not gen[r:r + patch, c:c + patch].any():
                    continue
                pr, pc = cand[rng.integers(len(cand))]
                h_, w_ = min(patch, rows - r), min(patch, cols - c)
                hf[r:r + h_, c:c + w_] = resid[pr:pr + h_, pc:pc + w_]
    tex[gen] = np.clip(low[gen] + hf[gen], 0, 255)
    S.color = tex
    S.stats["texture"] = "inpaint+patches" if cand else "inpaint"


def generated_mesh(surfs: list[Surface], rf) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Quads for every GENERATED texel (world frame, vertex colours) + per-face surface index."""
    Vs, Fs, Cs, Ss = [], [], [], []
    nv = 0
    for si, S in enumerate(surfs):
        rr, cc = np.nonzero(S.label == GENERATED)
        if not len(rr):
            continue
        # shared corner grid for this surface
        rows, cols = S.shape
        cid = -np.ones((rows + 1, cols + 1), int)
        corners = np.unique(np.concatenate([np.stack([rr + a, cc + b], 1) for a in (0, 1) for b in (0, 1)]), axis=0)
        cid[corners[:, 0], corners[:, 1]] = np.arange(len(corners)) + nv
        P = S.origin + (corners[:, 1] * TEXEL)[:, None] * S.eu + (corners[:, 0] * TEXEL)[:, None] * S.ev
        # corner colour = mean of adjacent generated texels
        acc = np.zeros((rows + 1, cols + 1, 3))
        cnt = np.zeros((rows + 1, cols + 1))
        for a in (0, 1):
            for b in (0, 1):
                np.add.at(acc, (rr + a, cc + b), S.color[rr, cc])
                np.add.at(cnt, (rr + a, cc + b), 1)
        col = acc[corners[:, 0], corners[:, 1]] / cnt[corners[:, 0], corners[:, 1]][:, None]
        v00, v01, v10, v11 = cid[rr, cc], cid[rr, cc + 1], cid[rr + 1, cc], cid[rr + 1, cc + 1]
        f = np.concatenate([np.stack([v00, v01, v11], 1), np.stack([v00, v11, v10], 1)])
        # orient faces so their normal is the inward normal (toward the room)
        if np.dot(np.cross(S.eu, S.ev), S.normal) < 0:
            f = f[:, ::-1]
        Vs.append(P @ rf.Rm)  # room frame -> world
        Fs.append(f)
        Cs.append(col)
        Ss.append(np.full(len(f), si))
        nv += len(corners)
    if not Vs:
        return trimesh.Trimesh(), np.zeros(0, int)
    V = np.concatenate(Vs)
    C = np.concatenate(Cs)
    m = trimesh.Trimesh(V, np.concatenate(Fs), vertex_colors=np.c_[np.clip(C, 0, 255), np.full(len(C), 255)].astype(np.uint8),
                        process=False)
    return m, np.concatenate(Ss)


def complete(m: trimesh.Trimesh, rf, vol, seed: int = 0):
    surfs = shell_surfaces(rf)
    classify(surfs, m, rf, vol)
    rng = np.random.default_rng(seed)
    walls_obs = [S.color[S.label == OBSERVED] for S in surfs if S.name.startswith("wall")]
    walls_obs = np.concatenate(walls_obs) if walls_obs and sum(len(x) for x in walls_obs) else np.zeros((0, 3))
    wall_rgb = np.median(walls_obs, 0) if len(walls_obs) else np.array([225.0, 222, 215])
    for S in surfs:
        prior = wall_rgb if not S.name == "ceiling" else np.clip(wall_rgb * 1.05 + 10, 0, 255)
        synthesize(S, rng, prior)
    gm, face_surface = generated_mesh(surfs, rf)
    report = {S.name: S.stats for S in surfs}
    tot = sum(S.stats["area_m2"] for S in surfs)
    report["_total"] = {k: round(sum(S.stats[k] * S.stats["area_m2"] for S in surfs) / max(tot, 1e-9), 4)
                        for k in ("observed", "opening", "generated")}
    report["_total"]["area_m2"] = round(tot, 2)
    return gm, face_surface, surfs, report
