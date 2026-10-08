"""Mode B (our method): room video -> posed keyframes -> metric dense depth -> TSDF mesh ->
room layout -> planar regularisation -> completion of the unseen shell with provenance.

Ablation flags (see eval/video_eval.py:CONFIGS); defaults are the full method:
    depth_ba  False        monocular depth as a scale-drift prior in BA (experimental: mixed on dev)
    scale     "canonical"  metric scale from FOV-canonical crops (ours) | "naive" full frame
    align     "grid"       smooth per-frame scale field (ours) | "median" one scale per frame
    mvs       True         mono-prior-guided plane-sweep stereo + dense re-anchoring (ours)
    filter    True         multi-view depth consistency filter
    snap      True         snap observed surfaces to the layout planes (ours)
    complete  True         fill the unseen room shell, labelled as generated (ours)
The baseline is SfM + per-frame median-scaled monocular depth + TSDF fusion (no layout
reasoning, no completion), with the scale read naively from the full frame.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle

import numpy as np

from core.log import RunContext, log
from video import depth as D

FULL = {"depth_ba": False, "scale": "canonical", "align": "grid", "mvs": True, "filter": True, "snap": True, "complete": True}
BASELINE = {"depth_ba": False, "scale": "naive", "align": "median", "mvs": False, "filter": False, "snap": False, "complete": False}


def _key(path: str, **kw) -> str:
    st = os.stat(path) if os.path.isfile(path) else None
    s = json.dumps([os.path.abspath(path), st.st_size if st else 0, st.st_mtime if st else 0, kw], sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:10]


def reconstruct(video: str, ctx: RunContext, abl: dict | None = None, cache_dir: str | None = None,
                n_key: int = 150, max_side: int = 640, holdout: int = 0, intrinsics=None, fov_deg=None,
                voxel: float = 0.02, depth_stride: int = 2) -> dict:
    from video.frames import load_frames
    from video.fuse import extract_mesh, integrate
    from video.sfm import run_sfm
    abl = {**FULL, **(abl or {})}
    cache_dir = cache_dir or os.path.join(ctx.out_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    key = _key(video, n_key=n_key, max_side=max_side, holdout=holdout, intr=intrinsics, fov=fov_deg)

    with ctx.stage("frames"):
        fr = load_frames(video, ctx, n_key=n_key, max_side=max_side, holdout=holdout,
                         intrinsics=intrinsics, fov_deg=fov_deg)
        ctx.save_img("keyframes", _contact_sheet([fr.images[i] for i in fr.train[::max(1, len(fr.train) // 12)]]))

    with ctx.stage("sfm"):
        pk = os.path.join(cache_dir, f"sfm_{key}.pkl")
        # monocular depth on every other train frame; it is a scale-drift prior inside BA (ours,
        # flag "depth_ba") and is reused below as the dense depth prior
        dids = fr.train[::depth_stride] if depth_stride > 1 else list(fr.train)
        mono = D.predict(fr.images, dids, os.path.join(cache_dir, f"depth_{key}"))
        pk = os.path.join(cache_dir, f"sfm_{key}_{'dba' if abl['depth_ba'] else 'plain'}.pkl")
        if os.path.exists(pk):
            rec = pickle.load(open(pk, "rb"))
        else:
            rec = run_sfm(fr.images, fr.train, fr.test, fr.K, not fr.K_given, ctx,
                          mono=mono if abl["depth_ba"] else None)
            pickle.dump(rec, open(pk, "wb"))
        K = rec.K
        train = [i for i in fr.train if i in rec.R]
        log.info("sfm: %s", rec.stats)

    with ctx.stage("depth"):
        dids = [i for i in dids if i in rec.R]
        mono = {i: mono[i] for i in dids}
        sc_cache = os.path.join(cache_dir, f"scale_{key}_{int(abl['depth_ba'])}_{abl['scale']}.json")
        if os.path.exists(sc_cache):
            s, sst = json.load(open(sc_cache))
        else:
            s, sst = D.metric_scale(rec, fr.images, K, train, canonical=abl["scale"] == "canonical", mono=mono)
            json.dump([s, sst], open(sc_cache, "w"))
        R = {i: rec.R[i] for i in rec.R}
        T = {i: rec.t[i] * s for i in rec.t}
        dkey = os.path.join(cache_dir, f"dense_{key}_{int(abl['depth_ba'])}_{abl['scale']}_{abl['align']}_{int(abl['mvs'])}{int(abl['filter'])}.npz")
        mvs_frac = None
        if os.path.exists(dkey):
            z = np.load(dkey)
            depths = {int(k[1:]): z[k].astype(np.float32) for k in z.files if k.startswith("d")}
            mvs_frac = float(z["mvs_frac"]) if "mvs_frac" in z.files else None
        else:
            depths = D.align(rec, mono, s, abl["align"], ctx)
            if abl["mvs"]:
                from video.mvs import reanchor, refine_all
                ref, confs, mvs_frac = refine_all(depths, fr.images, K, R, T, all_ids=train, max_cost=0.5, min_std=0.01)
                # the confident stereo depths re-anchor the monocular prior; stereo wins where confident
                for i in list(ref):
                    ra = reanchor(mono[i], ref[i], confs[i])
                    depths[i] = np.where(confs[i], ref[i], ra).astype(np.float32) if ra is not None else ref[i]
            if abl["filter"]:
                depths = D.consistency_filter(depths, K, R, T)
            extra = {"mvs_frac": mvs_frac} if mvs_frac is not None else {}
            np.savez_compressed(dkey, **{f"d{i}": d.astype(np.float16) for i, d in depths.items()}, **extra)
        i0 = dids[len(dids) // 2]
        ctx.save_img("depth_example", np.concatenate([fr.images[i0], _colorize(mono[i0]), _colorize(depths[i0])], 1))

    with ctx.stage("fuse"):
        vol = integrate(depths, fr.images, K, R, T, ctx, voxel=voxel)
        mesh = extract_mesh(vol)
        if len(mesh.faces) == 0:
            raise RuntimeError("fusion produced an empty surface (too little parallax or texture)")

    with ctx.stage("layout"):
        from video.layout import estimate, snap_to_planes, to_layout
        Cw = {i: -R[i].T @ T[i] for i in train}
        rf = estimate(mesh, {i: R[i] for i in train}, Cw, vol, ctx)
        snapped = None
        if abl["snap"]:
            # layout-guided depth refinement, then a second fusion pass (ours)
            from video.layout import refine_depths_with_shell
            depths, shell_st = refine_depths_with_shell(depths, mono, rf, K, R, T)
            log.info("shell refinement: %s", shell_st)
            vol = integrate(depths, fr.images, K, R, T, ctx, voxel=voxel, bounds=(vol.origin,
                            vol.origin + np.array(vol.tsdf.shape) * vol.voxel))
            mesh = extract_mesh(vol)
            mesh, snapped = snap_to_planes(mesh, rf)
            from video.layout import crop_to_room
            mesh, cropped = crop_to_room(mesh, rf)
            log.info("cropped %.1f%% of faces outside the room shell", 100 * cropped)
        L = to_layout(rf)
        ctx.save_img("layout_topdown", _topdown(mesh, rf))

    gen, face_surface, surfs, prov = None, None, None, None
    with ctx.stage("complete"):
        from video.complete import complete
        g, fs, surfs, prov = complete(mesh, rf, vol)
        if abl["complete"]:
            gen, face_surface = g, fs
        else:
            prov = {**prov, "_note": "completion disabled: generated fractions are what WOULD be missing"}
        ctx.save_img("shell_provenance", _provenance_sheet(surfs))

    return {"frames": fr, "rec": rec, "K": K, "R": R, "T": T, "train": train, "test": [i for i in fr.test if i in R],
            "scale": s, "scale_stats": sst, "vol": vol, "mesh": mesh, "generated": gen, "face_surface": face_surface,
            "rf": rf, "layout": L, "provenance": prov, "mvs_frac": mvs_frac, "snapped_frac": snapped, "abl": abl,
            "depth_ids": dids}


# ----------------------------------------------------------------------------- debug images
def _colorize(d: np.ndarray) -> np.ndarray:
    import cv2
    v = d[d > 0]
    lo, hi = (np.percentile(v, 2), np.percentile(v, 98)) if len(v) else (0, 1)
    x = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    c = cv2.applyColorMap((255 * (1 - x)).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1].copy()
    c[d <= 0] = 0
    return c


def _contact_sheet(ims: list[np.ndarray], cols: int = 4) -> np.ndarray:
    import cv2
    h, w = ims[0].shape[:2]
    k = 240 / w
    ims = [cv2.resize(i, (240, int(h * k))) for i in ims]
    while len(ims) % cols:
        ims.append(np.zeros_like(ims[0]))
    return np.concatenate([np.concatenate(ims[r:r + cols], 1) for r in range(0, len(ims), cols)], 0)


def _topdown(mesh, rf, res: float = 0.02) -> np.ndarray:
    import cv2
    P = mesh.vertices @ rf.Rm.T
    xz = P[:, [0, 2]]
    lo = np.minimum(xz.min(0), rf.polygon.min(0)) - 0.2
    hi = np.maximum(xz.max(0), rf.polygon.max(0)) + 0.2
    W, H = np.ceil((hi - lo) / res).astype(int) + 1
    img = np.full((H, W, 3), 255, np.uint8)
    ij = ((xz - lo) / res).astype(int)
    hgt = np.clip((P[:, 1] - rf.floor_y) / max(rf.ceil_y - rf.floor_y, 1e-3), 0, 1)
    col = mesh.visual.vertex_colors[:, :3]
    order = np.argsort(hgt)  # higher points drawn last
    img[ij[order, 1], ij[order, 0]] = col[order]
    pts = ((rf.polygon - lo) / res).astype(np.int32)
    cv2.polylines(img, [pts], True, (47, 111, 222), 3)
    return img


def _provenance_sheet(surfs) -> np.ndarray:
    import cv2
    from video.complete import GENERATED, OBSERVED, OPENING
    tiles = []
    for S in surfs:
        lab = S.label
        img = np.full(lab.shape + (3,), 255, np.uint8)
        img[lab == OBSERVED] = np.clip(S.color[lab == OBSERVED], 0, 255).astype(np.uint8)
        img[lab == GENERATED] = (224, 64, 154)
        img[lab == OPENING] = (60, 200, 90)
        k = 160 / max(img.shape[:2])
        img = cv2.resize(img, (max(1, int(img.shape[1] * k)), max(1, int(img.shape[0] * k))), interpolation=cv2.INTER_NEAREST)
        tile = np.full((180, 170, 3), 255, np.uint8)
        tile[:img.shape[0], :img.shape[1]] = img
        cv2.putText(tile, S.name, (2, 176), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        tiles.append(tile)
    return _contact_sheet(tiles, cols=min(6, len(tiles)))
