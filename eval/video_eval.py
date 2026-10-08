"""Mode B evaluation on Replica: baseline + ablations, novel views, geometry, dimensions, completion.

    python eval/video_eval.py --scenes office0 --out results_video/dev          # development
    python eval/video_eval.py --scenes room0 room1 room2 office2 office3 --out results_video/test

Protocol
  * Input: the scene's video.mp4 (every 4th Replica frame); 150 keyframes, every 8th held out.
    Camera intrinsics are self-calibrated (not read from the dataset) unless --gt-intrinsics.
  * Alignment: Sim(3) (Umeyama) from estimated to GT train-camera centres. Its scale factor is
    the metric-scale error of our pipeline; ATE is reported after alignment.
  * Geometry (observed region): GT mesh points visible from at least one train camera (z-test
    against GT depth rendered with our rasteriser). Accuracy, completeness, Chamfer-L1, F-score
    @5 cm, after Sim(3) ("sim3") and after rigid alignment with OUR metric scale ("metric").
  * Unseen region: GT points no train camera saw. Completeness of observed-only vs observed +
    generated geometry, and accuracy of the generated geometry itself (hallucination check).
  * Dimensions: room width/depth/height from the strongest wall/floor/ceiling planes, measured
    the same way on the prediction (our metric scale, rigid alignment) and on the GT mesh.
  * Novel views: held-out keyframes, registered by PnP against the train-only model. PSNR /
    SSIM / LPIPS(alex) on the full frame; PSNR also split by observed vs generated pixels.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from video.geom import umeyama  # noqa: E402
from video.render import rasterize, render_ibr, render_mesh  # noqa: E402

_B = {"depth_ba": False, "scale": "naive", "align": "median", "mvs": False, "filter": False, "snap": False, "complete": False}
CONFIGS = {  # cumulative, like Mode A's ablation rows; the last row is a separate experiment
    "baseline": _B,
    "+canon_scale": {**_B, "scale": "canonical"},
    "+grid_align": {**_B, "scale": "canonical", "align": "grid"},
    "+mvs": {**_B, "scale": "canonical", "align": "grid", "mvs": True, "filter": True},
    "+snap": {**_B, "scale": "canonical", "align": "grid", "mvs": True, "filter": True, "snap": True},
    "full": {**_B, "scale": "canonical", "align": "grid", "mvs": True, "filter": True, "snap": True, "complete": True},
    # depth prior inside BA: mixed on dev (office0 worse, room0 better), so not part of "full"
    "full+depth_ba": {"depth_ba": True, "scale": "canonical", "align": "grid", "mvs": True, "filter": True,
                      "snap": True, "complete": True},
}
F_TAU = 0.05
N_SAMPLES = 400_000


# ----------------------------------------------------------------------------- ground truth
def load_gt(scene_dir: str):
    traj = np.loadtxt(os.path.join(scene_dir, "traj.txt")).reshape(-1, 4, 4)
    fid = json.load(open(os.path.join(scene_dir, "frames.json")))["frame_ids"]
    cam = json.load(open(os.path.join(scene_dir, "cam_params.json")))["camera"]
    mesh = trimesh.load(os.path.join(scene_dir, "gt_mesh.ply"), process=False)
    return traj, fid, cam, mesh


def gt_pose(traj, fid, video_frame):
    c2w = traj[fid[video_frame]]
    R = c2w[:3, :3].T
    return R, -R @ c2w[:3, 3]


def gt_visibility(mesh, pts, poses, K, h, w, tol=0.03):
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    seen = np.zeros(len(pts), bool)
    for R, t in poses:
        d, _, _ = rasterize(V, F, K, R, t, h, w)
        Xc = pts @ R.T + t
        z = Xc[:, 2]
        ok = z > 0.05
        u = np.round(Xc[:, 0] / np.maximum(z, 1e-6) * K[0, 0] + K[0, 2]).astype(int)
        v = np.round(Xc[:, 1] / np.maximum(z, 1e-6) * K[1, 1] + K[1, 2]).astype(int)
        ok &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
        dz = np.full(len(pts), np.inf)
        dz[ok] = d[v[ok], u[ok]]
        seen |= ok & (np.abs(dz - z) < tol + 0.01 * z)
    return seen


def plane_dims(mesh: trimesh.Trimesh, axes: np.ndarray, min_area: float = 0.8) -> dict:
    """Width/depth along axes[0]/axes[2] between the outermost strong wall planes; height between
    the lowest strong up-facing and highest strong down-facing planes. axes rows: X, up, Z."""
    n, a, c = mesh.face_normals, mesh.area_faces, mesh.triangles_center
    out = {}

    def peaks(v, w):
        if not len(v):
            return []
        hist, e = np.histogram(v, bins=max(3, int(np.ptp(v) / 0.02) + 1), weights=w)
        hist = np.convolve(hist, np.ones(5), "same")
        cen = (e[:-1] + e[1:]) / 2
        return [cen[k] for k in range(len(hist)) if hist[k] >= min_area and hist[k] == hist[max(0, k - 3):k + 4].max()]

    for name, ax in (("width_m", axes[0]), ("depth_m", axes[2])):
        sel = np.abs(n @ ax) > 0.95
        pk = peaks(c[sel] @ ax, a[sel])
        out[name] = float(max(pk) - min(pk)) if len(pk) >= 2 else None
    up = axes[1]
    fl = peaks(c[n @ up > 0.95] @ up, a[n @ up > 0.95])
    ce = peaks(c[n @ up < -0.95] @ up, a[n @ up < -0.95])
    out["height_m"] = float(max(ce) - min(fl)) if fl and ce else None
    return out


# ----------------------------------------------------------------------------- metrics
def geo_metrics(pred_pts, gt_pts, tau=F_TAU):
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return {"acc_cm": None, "comp_cm": None, "chamfer_cm": None, "fscore": 0.0}
    d_pg = cKDTree(gt_pts).query(pred_pts)[0]
    d_gp = cKDTree(pred_pts).query(gt_pts)[0]
    p, r = float((d_pg < tau).mean()), float((d_gp < tau).mean())
    return {"acc_cm": 100 * float(d_pg.mean()), "comp_cm": 100 * float(d_gp.mean()),
            "chamfer_cm": 50 * float(d_pg.mean() + d_gp.mean()), "precision": p, "recall": r,
            "fscore": 2 * p * r / max(p + r, 1e-9)}


def psnr(a, b, mask=None):
    d = (a.astype(np.float64) - b.astype(np.float64)) / 255.0
    mse = (d ** 2).mean() if mask is None else (d[mask] ** 2).mean() if mask.any() else np.nan
    return float(10 * np.log10(1.0 / max(mse, 1e-12))) if mse == mse else None


_LPIPS = None


def lpips_score(a, b):
    global _LPIPS
    import torch
    if _LPIPS is None:
        import lpips
        _LPIPS = lpips.LPIPS(net="alex", verbose=False)
    ta = torch.from_numpy(a).permute(2, 0, 1)[None].float() / 127.5 - 1
    tb = torch.from_numpy(b).permute(2, 0, 1)[None].float() / 127.5 - 1
    with torch.no_grad():
        return float(_LPIPS(ta, tb).item())


def ssim(a, b):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(a, b, channel_axis=2, data_range=255))


def sample(m: trimesh.Trimesh | None, n: int) -> np.ndarray:
    if m is None or len(m.faces) == 0:
        return np.zeros((0, 3))
    return np.asarray(m.sample(n, seed=0))


# ----------------------------------------------------------------------------- one scene
def eval_scene(scene: str, data_dir: str, out_dir: str, configs: list[str], gt_intr: bool, save_renders: bool,
               n_key: int = 150, holdout: int = 8, video_name: str = "video.mp4") -> list[dict]:
    from run_video import run
    sd = os.path.join(data_dir, scene)
    traj, fid, cam, gtm = load_gt(sd)
    cache = os.path.join(out_dir, "cache", scene)
    rows = []
    gt_cache = {}
    for cfg in configs:
        t0 = time.time()
        od = os.path.join(out_dir, "runs", cfg, scene)
        intr = [cam["fx"], cam["fy"], cam["cx"], cam["cy"]] if gt_intr else None
        rep, res = run(os.path.join(sd, video_name), od, ablation=CONFIGS[cfg], holdout=holdout, intrinsics=intr,
                       n_key=n_key, max_side=600, cache_dir=cache, viewer=(cfg in ("baseline", "full")))
        fr, K = res["frames"], res["K"]
        h, w = fr.images[0].shape[:2]
        train, test = res["train"], res["test"]
        # --- alignment
        Cest = np.array([-res["R"][i].T @ res["T"][i] for i in train])
        Cgt = np.array([-gt_pose(traj, fid, fr.index[i])[0].T @ gt_pose(traj, fid, fr.index[i])[1] for i in train])
        s, Ra, ta = umeyama(Cest, Cgt)
        ate = np.linalg.norm(s * Cest @ Ra.T + ta - Cgt, axis=1)
        _, Rr, tr = umeyama(Cest, Cgt, with_scale=False)  # rigid, our metric scale
        # --- GT sampling + visibility (once per scene)
        if "pts" not in gt_cache:
            gpts = sample(gtm, N_SAMPLES)
            Kgt = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]]) * [[fr.scale], [fr.scale], [1]]
            vis = gt_visibility(gtm, gpts, [gt_pose(traj, fid, fr.index[i]) for i in train], Kgt, h, w)
            gt_cache.update(pts=gpts, vis=vis)
        gpts, vis = gt_cache["pts"], gt_cache["vis"]
        obs = res["mesh"]
        gen = res["generated"]
        po = sample(obs, 200_000)
        pg = sample(gen, 100_000)
        row = {"scene": scene, "config": cfg, "n_train": len(train), "n_test": len(test),
               "n_registered": rep["sfm"]["n_registered"], "focal_err_pct": 100 * (K[0, 0] / (cam["fx"] * fr.scale) - 1),
               "ate_cm": 100 * float(np.sqrt((ate ** 2).mean())), "gt_visible_frac": float(vis.mean())}
        # scale_err: our metres per true metre - 1  (GT = s * ours  =>  ours/true = 1/s)
        row["scale_err_pct"] = 100 * (1 / s - 1)
        for tag, (ss, RR, tt) in (("sim3", (s, Ra, ta)), ("metric", (1.0, Rr, tr))):
            P = ss * po @ RR.T + tt
            g = geo_metrics(P, gpts[vis])
            row.update({f"{tag}_{k}": v for k, v in g.items()})
        # unseen region: completeness with and without generated geometry, generated accuracy
        unseen = gpts[~vis]
        Pobs = s * po @ Ra.T + ta
        Pgen = s * pg @ Ra.T + ta if len(pg) else np.zeros((0, 3))
        if len(unseen) and len(Pobs):
            d_obs = cKDTree(Pobs).query(unseen)[0]
            row["unseen_recall_obs"] = float((d_obs < 2 * F_TAU).mean())
            if len(Pgen):
                d_all = np.minimum(d_obs, cKDTree(Pgen).query(unseen)[0])
                row["unseen_recall_all"] = float((d_all < 2 * F_TAU).mean())
                dg = cKDTree(gpts).query(Pgen)[0]
                row["gen_acc_cm"] = 100 * float(dg.mean())
                row["gen_precision10"] = float((dg < 2 * F_TAU).mean())
            else:
                row["unseen_recall_all"] = row["unseen_recall_obs"]
        row["gen_frac_shell"] = (rep.get("provenance") or {}).get("generated")
        # --- dimensions: our reported (layout) and plane-rule on our mesh, vs plane-rule on GT
        rf = res["rf"]
        axes_gt = (Ra @ rf.Rm.T).T  # room axes expressed in the GT frame
        gd = plane_dims(gtm, axes_gt)
        pd_ = plane_dims(obs, rf.Rm)
        lay = {"width_m": rep["room"]["width_m"], "depth_m": rep["room"]["depth_m"], "height_m": rep["room"]["height_m"]}
        for k in ("width_m", "depth_m", "height_m"):
            row[f"gt_{k}"] = gd[k]
            row[f"plane_{k}"] = pd_[k]
            row[f"layout_{k}"] = lay[k]
        row["dim_err_plane_cm"] = _dim_err(pd_, gd)
        row["dim_err_layout_cm"] = _dim_err(lay, gd)
        # --- novel views
        nv = novel_views(res, cfg, os.path.join(od, "renders") if save_renders else None)
        row.update(nv)
        row["time_s"] = round(time.time() - t0, 1)
        rows.append(row)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    return rows


def _dim_err(pred: dict, gt: dict):
    e = [abs(pred[k] - gt[k]) for k in ("width_m", "depth_m", "height_m") if pred.get(k) and gt.get(k)]
    return 100 * float(np.mean(e)) if e else None


def novel_views(res: dict, cfg: str, save_dir: str | None) -> dict:
    import cv2
    fr, K = res["frames"], res["K"]
    h, w = fr.images[0].shape[:2]
    obs, gen = res["mesh"], res["generated"]
    if gen is not None and len(gen.faces):
        V = np.vstack([obs.vertices, gen.vertices])
        F = np.vstack([obs.faces, gen.faces + len(obs.vertices)])
        C = np.vstack([obs.visual.vertex_colors, gen.visual.vertex_colors])
        lab = np.r_[np.zeros(len(obs.faces), int), np.ones(len(gen.faces), int)]
    else:
        V, F, C = obs.vertices, obs.faces, obs.visual.vertex_colors
        lab = np.zeros(len(F), int)
    renderers = ["vertex", "ibr"]
    src = {}
    if "ibr" in renderers:
        for j in res["train"]:
            d, _, _ = rasterize(V, F, K, res["R"][j], res["T"][j], h, w)
            src[j] = d
    out = {r: {"psnr": [], "ssim": [], "lpips": [], "psnr_obs": [], "psnr_gen": [], "hole": []} for r in renderers}
    for q in res["test"]:
        gt = fr.images[q]
        for r in renderers:
            if r == "vertex":
                img, depth, lb = render_mesh(V, F, C, K, res["R"][q], res["T"][q], h, w, face_label=lab)
            else:
                img, depth, lb, _ = render_ibr(V, F, C, K, res["R"][q], res["T"][q], h, w,
                                               {j: fr.images[j] for j in res["train"]}, res["R"], res["T"], src,
                                               face_label=lab)
            o = out[r]
            o["psnr"].append(psnr(img, gt))
            o["ssim"].append(ssim(img, gt))
            o["lpips"].append(lpips_score(img, gt))
            o["psnr_obs"].append(psnr(img, gt, lb == 0))
            o["psnr_gen"].append(psnr(img, gt, lb == 1))
            o["hole"].append(float((lb < 0).mean()))
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                tint = img.copy()
                tint[lb == 1] = (0.5 * tint[lb == 1] + [112, 32, 77]).astype(np.uint8)
                cv2.imwrite(os.path.join(save_dir, f"{q:03d}_{r}.jpg"),
                            cv2.cvtColor(np.concatenate([gt, img, tint], 1), cv2.COLOR_RGB2BGR))
    row = {}
    for r, o in out.items():
        for k, v in o.items():
            v = [x for x in v if x is not None]
            row[f"nvs_{r}_{k}"] = float(np.mean(v)) if v else None
    return row


def make_partial(scene_dir: str, frac: float) -> str:
    """First `frac` of the video (frame ids still map through frames.json, which starts at 0)."""
    import cv2
    name = f"video_p{int(round(frac * 100))}.mp4"
    out = os.path.join(scene_dir, name)
    if not os.path.exists(out):
        cap = cv2.VideoCapture(os.path.join(scene_dir, "video.mp4"))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        vw = None
        for k in range(int(n * frac)):
            ok, f = cap.read()
            if not ok:
                break
            if vw is None:
                vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (f.shape[1], f.shape[0]))
            vw.write(f)
        vw.release()
        cap.release()
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["office0"])
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "replica"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results_video", "dev"))
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--gt-intrinsics", action="store_true")
    ap.add_argument("--video", default="video.mp4")
    ap.add_argument("--save-renders", action="store_true")
    ap.add_argument("--partial", type=float, default=None,
                    help="use only the first fraction of each video (partial-scan protocol)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for sc in a.scenes:
        vname = make_partial(os.path.join(a.data, sc), a.partial) if a.partial else a.video
        rows += eval_scene(sc, a.data, a.out, a.configs, a.gt_intrinsics, a.save_renders, video_name=vname)
        json.dump(rows, open(os.path.join(a.out, "rows.json"), "w"), indent=1)
    print(f"wrote {os.path.join(a.out, 'rows.json')}")


if __name__ == "__main__":
    main()
