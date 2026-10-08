"""Incremental structure-from-motion for a room video (OpenCV + scipy; no COLMAP).

    features   RootSIFT on every keyframe
    pairs      temporal window + loop-closure candidates from a tf-idf bag of visual words
    verify     fundamental matrix (MAGSAC), so verification does not depend on the focal guess
    tracks     union-find over verified matches; tracks seen twice in one image are dropped
    init       the best-connected nearby pair with enough parallax (essential matrix)
    grow       PnP-RANSAC registration, multi-view triangulation, local + periodic global BA
    BA         robust (Huber) sparse least squares over poses, points and, when the camera is
               unknown, a shared focal length
Held-out test views are registered last by PnP against the finished model; their pixels
never touch the reconstruction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix

from core.log import RunContext, log
from video.geom import rodrigues, rot_to_rvec, rotate_points

N_FEAT = 4000
WINDOW = 6
N_LOOP = 5
MIN_INLIERS = 30
REPROJ_PX = 3.0
MIN_ANGLE_DEG = 1.5
JUMP_STEPS = 3.0  # max distance to a temporal neighbour, in median steps per (index gap + 1)


@dataclass
class Recon:
    K: np.ndarray
    R: dict[int, np.ndarray] = field(default_factory=dict)   # image -> world-to-camera rotation
    t: dict[int, np.ndarray] = field(default_factory=dict)
    X: np.ndarray | None = None                                # (M,3) points
    obs_img: np.ndarray | None = None                          # (O,) image index per observation
    obs_pt: np.ndarray | None = None                           # (O,) point index
    obs_uv: np.ndarray | None = None                           # (O,2) pixel
    obs_kp: np.ndarray | None = None                           # (O,) keypoint index in that image
    color: np.ndarray | None = None                            # (M,3) uint8
    stats: dict = field(default_factory=dict)

    def registered(self) -> list[int]:
        return sorted(self.R)


# --------------------------------------------------------------------------- features
def extract_features(images: list[np.ndarray]):
    sift = cv2.SIFT_create(nfeatures=N_FEAT, contrastThreshold=0.02)
    kps, descs = [], []
    for im in images:
        g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        k, d = sift.detectAndCompute(g, None)
        if d is None:
            k, d = [], np.zeros((0, 128), np.float32)
        d = np.sqrt(d / np.maximum(d.sum(1, keepdims=True), 1e-9)).astype(np.float32)  # RootSIFT
        kps.append(np.array([p.pt for p in k], np.float32).reshape(-1, 2))
        descs.append(d)
    return kps, descs


def _vocab_hist(descs: list[np.ndarray], k: int = 128) -> np.ndarray:
    rng = np.random.default_rng(0)
    pool = np.concatenate([d[rng.choice(len(d), min(len(d), 400), replace=False)] for d in descs if len(d)])
    k = min(k, max(2, len(pool) // 20))
    _, _, C = cv2.kmeans(pool, k, None, (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 20, 1e-3),
                         2, cv2.KMEANS_PP_CENTERS)
    H = np.zeros((len(descs), k), np.float32)
    for i, d in enumerate(descs):
        if len(d):
            w = np.argmax(d @ C.T - 0.5 * (C * C).sum(1), 1)
            H[i] = np.bincount(w, minlength=k)
    idf = np.log(len(descs) / np.maximum((H > 0).sum(0), 1))
    H = H * idf
    return H / np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-9)


def candidate_pairs(descs: list[np.ndarray], window: int = WINDOW, n_loop: int = N_LOOP) -> list[tuple[int, int]]:
    n = len(descs)
    pairs = {(i, j) for i in range(n) for j in range(i + 1, min(n, i + window + 1))}
    if n > 2 * window:
        H = _vocab_hist(descs)
        S = H @ H.T
        for i in range(n):
            S[i, max(0, i - window):i + window + 1] = -1
            for j in np.argsort(-S[i])[:n_loop]:
                if S[i, j] > 0:
                    pairs.add((min(i, j), max(i, j)))
    return sorted(pairs)


def match_pair(d1, d2, ratio: float = 0.8) -> np.ndarray:
    if len(d1) < 8 or len(d2) < 8:
        return np.zeros((0, 2), int)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    m12 = bf.knnMatch(d1, d2, k=2)
    m21 = bf.knnMatch(d2, d1, k=1)
    back = {m[0].queryIdx: m[0].trainIdx for m in m21 if m}
    out = [(a.queryIdx, a.trainIdx) for a, b in (m for m in m12 if len(m) == 2)
           if a.distance < ratio * b.distance and back.get(a.trainIdx) == a.queryIdx]
    return np.array(out, int).reshape(-1, 2)


def verify(k1, k2, m, return_F: bool = False):
    if len(m) < MIN_INLIERS:
        return (m[:0], None) if return_F else m[:0]
    try:
        F, inl = cv2.findFundamentalMat(k1[m[:, 0]], k2[m[:, 1]], cv2.USAC_MAGSAC, 1.0, 0.999, 10000)
    except cv2.error:  # degenerate sample set inside USAC
        F, inl = cv2.findFundamentalMat(k1[m[:, 0]], k2[m[:, 1]], cv2.FM_RANSAC, 1.0, 0.999)
    if F is None or inl is None or F.shape != (3, 3):
        return (m[:0], None) if return_F else m[:0]
    m = m[inl.ravel() > 0]
    m = m if len(m) >= MIN_INLIERS else m[:0]
    return (m, F) if return_F else m


def focal_from_F(Fs: list[np.ndarray], w: int, h: int) -> tuple[float, float]:
    """Self-calibration of a shared focal length (principal point at the centre, square pixels)
    by the Mendonca-Cipolla criterion: for the true K, E = K^T F K has two equal non-zero
    singular values. 1-D search over the FOV, robust (median) over image pairs."""
    cx, cy = (w - 1) / 2, (h - 1) / 2
    fov = np.radians(np.linspace(25, 130, 211))
    fs = 0.5 * max(w, h) / np.tan(fov / 2)
    cost = []
    for f in fs:
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
        c = []
        for F in Fs:
            sv = np.linalg.svd(K.T @ F @ K, compute_uv=False)
            c.append((sv[0] - sv[1]) / max(sv[0] + sv[1], 1e-12))
        cost.append(np.median(c))
    cost = np.array(cost)
    k = int(np.argmin(cost))
    return float(fs[k]), float(cost[k])


# --------------------------------------------------------------------------- tracks
def build_tracks(kps, matches: dict[tuple[int, int], np.ndarray]):
    off = np.cumsum([0] + [len(k) for k in kps])
    parent = np.arange(off[-1])

    def find(a):
        r = a
        while parent[r] != r:
            r = parent[r]
        while parent[a] != r:
            parent[a], a = r, parent[a]
        return r

    for (i, j), m in matches.items():
        for a, b in m:
            ra, rb = find(off[i] + a), find(off[j] + b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(a) for a in range(off[-1])])
    img_of = np.repeat(np.arange(len(kps)), np.diff(off))
    kp_of = np.arange(off[-1]) - off[img_of]
    order = np.argsort(roots, kind="stable")
    r_sorted = roots[order]
    starts = np.flatnonzero(np.r_[True, r_sorted[1:] != r_sorted[:-1]])
    tracks = []
    for s, e in zip(starts, np.r_[starts[1:], len(order)]):
        if e - s < 2:
            continue
        nodes = order[s:e]
        imgs = img_of[nodes]
        if len(np.unique(imgs)) != len(imgs):  # inconsistent: two keypoints in one image
            continue
        tracks.append(np.stack([imgs, kp_of[nodes]], 1))
    return tracks


# --------------------------------------------------------------------------- bundle adjustment
DEPTH_W = 1.0     # depth-prior residual weight: 10% depth disagreement ~ 0.1 px
FOCAL_W = 50.0    # focal prior weight: 3% from the self-calibrated focal ~ 1.5 px


def bundle_adjust(K, R: dict, t: dict, X: np.ndarray, obs_img, obs_pt, obs_uv,
                  free_cams=None, free_pts=None, refine_f: bool = False, max_nfev: int = 40,
                  fixed_cam: int | None = None, obs_dm: np.ndarray | None = None, f_prior: float | None = None):
    """Robust BA in place. Only `free_cams`/`free_pts` move (default: all except `fixed_cam`).

    obs_dm (optional, ours): monocular metric depth at each observation (0 = none). Adds a
    robust residual log(z / (a * d_mono)) with ONE global factor a, so the reconstruction's
    scale cannot drift from one part of the video to another (monocular SfM's classic
    failure on weakly connected sequences). f_prior keeps a free focal near its
    self-calibrated value."""
    cams = sorted(R)
    if free_cams is None:
        free_cams = [c for c in cams if c != fixed_cam]
    free_cams = [c for c in free_cams if c in R]
    free_pts = np.unique(obs_pt) if free_pts is None else np.asarray(free_pts)
    if len(free_pts) == 0 and not free_cams:
        return K
    keep = np.isin(obs_pt, free_pts) | np.isin(obs_img, free_cams)
    oi, op, uv = obs_img[keep], obs_pt[keep], obs_uv[keep]
    dm = obs_dm[keep] if obs_dm is not None else np.zeros(len(oi))
    ci = {c: k for k, c in enumerate(free_cams)}
    pi = {p: k for k, p in enumerate(free_pts.tolist())}
    nc, npnt = len(free_cams), len(free_pts)
    cam_k = np.array([ci.get(c, -1) for c in oi])
    pt_k = np.array([pi.get(p, -1) for p in op])

    rv_all = {c: rot_to_rvec(R[c]) for c in R}
    rv_obs = np.array([rv_all[c] for c in oi])
    t_obs = np.array([t[c] for c in oi])
    X_obs = X[op].copy()
    f0 = K[0, 0]
    cx, cy = K[0, 2], K[1, 2]
    n_f = 1 if refine_f else 0
    dsel = np.flatnonzero(dm > 0.05)
    n_a = 1 if len(dsel) >= 20 else 0
    la0 = 0.0
    if n_a:
        Xc0 = rotate_points(rv_obs[dsel], X_obs[dsel]) + t_obs[dsel]
        ok = Xc0[:, 2] > 0
        la0 = float(np.median(np.log(Xc0[ok, 2] / dm[dsel][ok]))) if ok.any() else 0.0

    x0 = np.concatenate([np.concatenate([np.r_[rv_all[c], t[c]] for c in free_cams]) if nc else np.zeros(0),
                         X[free_pts].ravel(), [f0] * n_f, [la0] * n_a])
    cm, pm = cam_k >= 0, pt_k >= 0
    i_f = 6 * nc + 3 * npnt
    i_a = i_f + n_f

    def unpack(x):
        rv, tt, XX = rv_obs.copy(), t_obs.copy(), X_obs.copy()
        if nc:
            cp = x[:6 * nc].reshape(nc, 6)
            rv[cm], tt[cm] = cp[cam_k[cm], :3], cp[cam_k[cm], 3:]
        if npnt:
            XX[pm] = x[6 * nc:i_f].reshape(npnt, 3)[pt_k[pm]]
        f = x[i_f] if n_f else f0
        return rv, tt, XX, f

    def resid(x):
        rv, tt, XX, f = unpack(x)
        Xc = rotate_points(rv, XX) + tt
        z = np.maximum(Xc[:, 2], 1e-3)
        u = f * Xc[:, 0] / z + cx
        v = f * Xc[:, 1] / z + cy
        out = [u - uv[:, 0], v - uv[:, 1]]
        if n_a:
            out.append(DEPTH_W * (np.log(z[dsel] / dm[dsel]) - x[i_a]))
        if n_f and f_prior:
            out.append([FOCAL_W * np.log(x[i_f] / f_prior)])
        return np.concatenate(out)

    no = len(oi)
    rows, cols = [], []
    r2 = np.arange(no)
    for k in range(6):
        rows += [r2[cm], r2[cm] + no]
        cols += [6 * cam_k[cm] + k] * 2
    for k in range(3):
        rows += [r2[pm], r2[pm] + no]
        cols += [6 * nc + 3 * pt_k[pm] + k] * 2
    if n_f:
        rows += [r2, r2 + no]
        cols += [np.full(no, i_f)] * 2
    n_rows = 2 * no
    if n_a:
        rd = n_rows + np.arange(len(dsel))
        dc, dp = cam_k[dsel], pt_k[dsel]
        for k in range(6):
            rows.append(rd[dc >= 0])
            cols.append(6 * dc[dc >= 0] + k)
        for k in range(3):
            rows.append(rd[dp >= 0])
            cols.append(6 * nc + 3 * dp[dp >= 0] + k)
        rows.append(rd)
        cols.append(np.full(len(dsel), i_a))
        n_rows += len(dsel)
    if n_f and f_prior:
        rows.append(np.array([n_rows]))
        cols.append(np.array([i_f]))
        n_rows += 1
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_rows, len(x0)))
    res = least_squares(resid, x0, jac_sparsity=A, loss="huber", f_scale=1.5, x_scale="jac",
                        max_nfev=max_nfev, method="trf", tr_solver="lsmr", verbose=0)
    x = res.x
    if nc:
        cp = x[:6 * nc].reshape(nc, 6)
        Rs = rodrigues(cp[:, :3])
        for c, k in ci.items():
            R[c], t[c] = Rs[k], cp[k, 3:].copy()
    if npnt:
        X[free_pts] = x[6 * nc:i_f].reshape(npnt, 3)
    if n_f:
        K = K.copy()
        K[0, 0] = K[1, 1] = x[i_f]
    return K


# --------------------------------------------------------------------------- incremental
class _Builder:
    def __init__(self, K, kps, tracks, ctx: RunContext, refine_f: bool, mono: dict | None = None,
                 f_prior: float | None = None):
        self.K, self.kps, self.tracks, self.ctx, self.refine_f = K, kps, tracks, ctx, refine_f
        self.f_prior = f_prior
        # monocular depth at every keypoint of the images that have a depth map (0 = none)
        self.kp_dm = [np.zeros(len(k), np.float32) for k in kps]
        for im, d in (mono or {}).items():
            if len(kps[im]):
                self.kp_dm[im] = cv2.remap(d, kps[im][:, 0].reshape(-1, 1), kps[im][:, 1].reshape(-1, 1),
                                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()
        self.R, self.t = {}, {}
        self.X = np.full((len(tracks), 3), np.nan)
        # per image: kp -> track
        self.kp2tr = [dict() for _ in kps]
        for ti, tr in enumerate(tracks):
            for im, kp in tr:
                self.kp2tr[im][kp] = ti
        self.bad = np.zeros(len(tracks), bool)  # observations rejected per track handled via mask
        self.obs_ok: dict[tuple[int, int], bool] = {}

    def P(self, im):
        return np.hstack([self.R[im], self.t[im][:, None]])

    def obs_arrays(self, with_kp: bool = False):
        oi, op, ok = [], [], []
        for ti in np.flatnonzero(~np.isnan(self.X[:, 0])):
            for im, kp in self.tracks[ti]:
                if im in self.R and self.obs_ok.get((ti, im), True):
                    oi.append(im)
                    op.append(ti)
                    ok.append(kp)
        uv = np.array([self.kps[i][k] for i, k in zip(oi, ok)], float).reshape(-1, 2)
        out = np.array(oi, int), np.array(op, int), uv
        return out + (np.array(ok, int),) if with_kp else out

    def triangulate(self, tracks_idx):
        Kinv = np.linalg.inv(self.K)
        n_new = 0
        for ti in tracks_idx:
            tr = [(im, kp) for im, kp in self.tracks[ti] if im in self.R]
            if len(tr) < 2:
                continue
            xs = [(Kinv @ np.r_[self.kps[im][kp], 1.0])[:2] for im, kp in tr]
            A = []
            for (im, _), x in zip(tr, xs):
                P = self.P(im)
                A += [x[0] * P[2] - P[0], x[1] * P[2] - P[1]]
            Xh = np.linalg.svd(np.asarray(A))[2][-1]
            if abs(Xh[3]) < 1e-12:
                continue
            Xw = Xh[:3] / Xh[3]
            ok, centers = True, []
            for im, kp in tr:
                xc = self.R[im] @ Xw + self.t[im]
                if xc[2] <= 0:
                    ok = False
                    break
                u = self.K[:2, :2] @ (xc[:2] / xc[2]) + self.K[:2, 2]
                if np.linalg.norm(u - self.kps[im][kp]) > REPROJ_PX:
                    ok = False
                    break
                centers.append(-self.R[im].T @ self.t[im])
            if not ok:
                continue
            rays = np.array([Xw - c for c in centers])
            rays /= np.linalg.norm(rays, axis=1, keepdims=True)
            cosang = np.clip(rays @ rays.T, -1, 1)
            if np.degrees(np.arccos(cosang.min())) < MIN_ANGLE_DEG:
                continue
            self.X[ti] = Xw
            n_new += 1
        return n_new

    def initialize(self, matches) -> bool:
        cand = sorted(((len(m), i, j) for (i, j), m in matches.items() if 2 <= j - i <= 3 * WINDOW), reverse=True)
        for _, i, j in cand[:40]:
            m = matches[(i, j)]
            p1, p2 = self.kps[i][m[:, 0]], self.kps[j][m[:, 1]]
            E, inl = cv2.findEssentialMat(p1, p2, self.K, cv2.RANSAC, 0.999, 1.0)
            if E is None or E.shape != (3, 3):
                continue
            n, Rr, tr, inl2 = cv2.recoverPose(E, p1, p2, self.K, mask=inl)
            if n < 100:
                continue
            self.R = {i: np.eye(3), j: Rr}
            self.t = {i: np.zeros(3), j: tr.ravel()}
            tids = {self.kp2tr[i].get(a) for a in m[:, 0]} - {None}
            self.X[:] = np.nan
            if self.triangulate(sorted(tids)) < 80:
                continue
            med = self._median_angle(i, j)
            if med < 2.0:
                continue
            log.info("sfm init pair %d-%d: %d points, median angle %.1f deg", i, j,
                     int((~np.isnan(self.X[:, 0])).sum()), med)
            self.init_pair = (i, j)
            return True
        self.R, self.t = {}, {}
        return False

    def _median_angle(self, i, j):
        ci, cj = -self.R[i].T @ self.t[i], -self.R[j].T @ self.t[j]
        Xs = self.X[~np.isnan(self.X[:, 0])]
        a, b = Xs - ci, Xs - cj
        cos = np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        return float(np.degrees(np.median(np.arccos(np.clip(cos, -1, 1)))))

    def register(self, im) -> bool:
        pts3, pts2 = [], []
        for kp, ti in self.kp2tr[im].items():
            if not np.isnan(self.X[ti, 0]):
                pts3.append(self.X[ti])
                pts2.append(self.kps[im][kp])
        if len(pts3) < 20:
            return False
        pts3, pts2 = np.array(pts3), np.array(pts2)
        ok, rv, tv, inl = cv2.solvePnPRansac(pts3, pts2, self.K, None, iterationsCount=1000,
                                             reprojectionError=REPROJ_PX * 1.5, confidence=0.999,
                                             flags=cv2.SOLVEPNP_SQPNP)
        if not ok or inl is None or len(inl) < max(15, 0.2 * len(pts3)):
            return False
        inl = inl.ravel()
        rv, tv = cv2.solvePnPRefineLM(pts3[inl], pts2[inl], self.K, None, rv, tv)
        R, t = cv2.Rodrigues(rv)[0], tv.ravel()
        if not self.plausible(im, -R.T @ t):
            return False
        self.R[im], self.t[im] = R, t
        return True

    def _median_step(self):
        reg = sorted(self.R)
        C = np.array([-self.R[i].T @ self.t[i] for i in reg])
        st = [np.linalg.norm(C[k + 1] - C[k]) / (reg[k + 1] - reg[k]) for k in range(len(reg) - 1)]
        return float(np.median(st)) if st else None

    def plausible(self, im: int, C: np.ndarray, k_steps: float = JUMP_STEPS, exclude_self: bool = False) -> bool:
        """Video prior: a camera cannot teleport. Its centre must be within k median per-frame
        steps x (index gap + 1) of its nearest registered temporal neighbours."""
        step = self._median_step()
        if step is None or step <= 0:
            return True
        nb = [j for j in sorted(self.R, key=lambda j: abs(j - im)) if j != im and abs(j - im) <= 4][:2]
        if not nb:
            return True
        for j in nb:
            Cj = -self.R[j].T @ self.t[j]
            if np.linalg.norm(C - Cj) <= k_steps * step * (abs(j - im) + 1):
                return True
        return False

    def drop_implausible(self) -> list[int]:
        bad = [im for im in sorted(self.R) if not self.plausible(im, -self.R[im].T @ self.t[im])]
        for im in bad:
            del self.R[im], self.t[im]
        return bad

    def filter_outliers(self):
        oi, op, uv = self.obs_arrays()
        if not len(oi):
            return 0
        Rs = np.stack([self.R[i] for i in oi])
        ts = np.stack([self.t[i] for i in oi])
        Xc = np.einsum("nij,nj->ni", Rs, self.X[op]) + ts
        z = Xc[:, 2]
        u = Xc[:, :2] / np.maximum(z[:, None], 1e-9) * [self.K[0, 0], self.K[1, 1]] + self.K[:2, 2]
        err = np.linalg.norm(u - uv, axis=1)
        bad = (err > 2 * REPROJ_PX) | (z <= 0)
        for i, p in zip(oi[bad], op[bad]):
            self.obs_ok[(p, i)] = False
        # points left with <2 good observations are removed
        oi, op, _ = self.obs_arrays()
        cnt = np.bincount(op, minlength=len(self.X))
        drop = (~np.isnan(self.X[:, 0])) & (cnt < 2)
        self.X[drop] = np.nan
        return int(bad.sum())

    def ba(self, local: list[int] | None = None, max_pts: int | None = None, nfev: int = 30):
        oi, op, uv, okp = self.obs_arrays(with_kp=True)
        if not len(oi):
            return
        dm = np.array([self.kp_dm[i][k] for i, k in zip(oi, okp)], np.float64)
        fixed = self.init_pair[0]
        if local is not None:
            pts = np.unique(op[np.isin(oi, local)])
            sel = np.isin(op, pts)
            self.K = bundle_adjust(self.K, self.R, self.t, self.X, oi[sel], op[sel], uv[sel],
                                   free_cams=[c for c in local if c != fixed], free_pts=pts,
                                   refine_f=False, max_nfev=nfev, obs_dm=dm[sel])
            return
        pts = np.unique(op)
        if max_pts and len(pts) > max_pts:  # intermediate global BA on the best-observed points
            cnt = np.bincount(op)
            pts = pts[np.argsort(-cnt[pts])[:max_pts]]
            sel = np.isin(op, pts)
            oi, op, uv, dm = oi[sel], op[sel], uv[sel], dm[sel]
        self.K = bundle_adjust(self.K, self.R, self.t, self.X, oi, op, uv, fixed_cam=fixed,
                               refine_f=self.refine_f, max_nfev=nfev, obs_dm=dm, f_prior=self.f_prior)


def run_sfm(images: list[np.ndarray], train: list[int], test: list[int], K0: np.ndarray,
            refine_f: bool, ctx: RunContext, mono: dict[int, np.ndarray] | None = None) -> Recon:
    """mono: optional {image index: monocular metric depth} used as a scale-drift prior in BA."""
    kps, descs = extract_features(images)
    tr_kps = [kps[i] for i in train]
    tr_desc = [descs[i] for i in train]
    pairs = candidate_pairs(tr_desc)
    matches, Fs = {}, {}
    for i, j in pairs:
        m, F = verify(tr_kps[i], tr_kps[j], match_pair(tr_desc[i], tr_desc[j]), return_F=True)
        if len(m):
            matches[(i, j)] = m
            Fs[(i, j)] = F
    f_init = None
    if refine_f:
        # pairs with plenty of inliers and some baseline (pure rotations make F ill-posed)
        good = [F for (i, j), F in Fs.items() if j - i >= 2 and len(matches[(i, j)]) >= 100]
        if len(good) >= 5:
            h, w = images[0].shape[:2]
            f_init, mc = focal_from_F(good, w, h)
            log.info("focal self-calibration: %.1f px (MC cost %.4f, %d pairs)", f_init, mc, len(good))
            K0 = K0.copy()
            K0[0, 0] = K0[1, 1] = f_init
        else:
            ctx.fallback("sfm", "too few strong pairs to self-calibrate the focal; using the FOV prior")
    tracks = build_tracks(tr_kps, matches)
    log.info("sfm: %d images, %d verified pairs, %d tracks", len(train), len(matches), len(tracks))
    pos = {im: k for k, im in enumerate(train)}
    mono_local = {pos[i]: d for i, d in (mono or {}).items() if i in pos}
    B = _Builder(K0.copy(), tr_kps, tracks, ctx, refine_f, mono_local, f_init)
    if not B.initialize(matches):
        raise RuntimeError("SfM could not find an initial pair with enough parallax "
                           "(move the camera sideways, not just rotate)")
    B.ba(nfev=50)
    failed = set()
    last_global = len(B.R)
    while True:
        cand = []
        for im in range(len(train)):
            if im in B.R or im in failed:
                continue
            n = sum(1 for ti in B.kp2tr[im].values() if not np.isnan(B.X[ti, 0]))
            if n >= 20:
                cand.append((n, im))
        if not cand:
            break
        cand.sort(reverse=True)
        added = []
        for n, im in cand[:3]:
            if B.register(im):
                added.append(im)
            else:
                failed.add(im)
        if not added:
            continue
        for im in added:
            B.triangulate([ti for ti in B.kp2tr[im].values() if np.isnan(B.X[ti, 0])])
        near = sorted((c for c in B.R if c not in added), key=lambda c: min(abs(c - a) for a in added))
        B.ba(local=added + near[:6], nfev=15)
        if len(B.R) >= 1.3 * last_global or (B.refine_f and len(B.R) in (8, 15, 25)):
            B.ba(max_pts=6000, nfev=25)
            B.filter_outliers()
            last_global = len(B.R)
            failed.clear()  # retry stragglers against the improved model
    B.ba(nfev=60)
    bad = B.drop_implausible()
    if bad:
        ctx.fallback("sfm", f"dropped {len(bad)} keyframe(s) whose pose jumped away from their neighbours")
    B.filter_outliers()
    B.ba(nfev=30)
    # retriangulate anything newly possible after the final BA
    B.triangulate([ti for ti in range(len(tracks)) if np.isnan(B.X[ti, 0])])

    n_reg = len(B.R)
    if n_reg < 0.7 * len(train):
        ctx.fallback("sfm", f"only {n_reg}/{len(train)} keyframes registered")
    oi, op, uv, okp = B.obs_arrays(with_kp=True)
    good = np.unique(op)
    remap = -np.ones(len(B.X), int)
    remap[good] = np.arange(len(good))
    rec = Recon(K=B.K, X=B.X[good], obs_img=np.array([train[i] for i in oi], int),
                obs_pt=remap[op], obs_uv=uv, obs_kp=okp)
    rec.R = {train[i]: B.R[i] for i in B.R}
    rec.t = {train[i]: B.t[i] for i in B.t}
    col = np.zeros((len(good), 3))
    cnt = np.zeros(len(good))
    for im, p, (u, v) in zip(rec.obs_img, rec.obs_pt, uv):
        img = images[im]
        col[p] += img[min(int(v), img.shape[0] - 1), min(int(u), img.shape[1] - 1)]
        cnt[p] += 1
    rec.color = (col / np.maximum(cnt, 1)[:, None]).astype(np.uint8)
    err = _reproj_err(rec)
    rec.stats = {"n_train": len(train), "n_registered": n_reg, "n_points": len(good), "n_obs": len(oi),
                 "n_pairs": len(matches), "mean_reproj_px": round(float(err.mean()), 3) if len(err) else None,
                 "focal_px": round(float(B.K[0, 0]), 2), "focal_init_px": round(float(K0[0, 0]), 2),
                 "focal_selfcal_px": round(f_init, 2) if f_init else None}

    # held-out views: localise against the finished model (pose only)
    if test:
        _register_test(rec, images, kps, descs, train, test, ctx)
    return rec


def _reproj_err(rec: Recon) -> np.ndarray:
    if rec.obs_img is None or not len(rec.obs_img):
        return np.zeros(0)
    Rs = np.stack([rec.R[i] for i in rec.obs_img])
    ts = np.stack([rec.t[i] for i in rec.obs_img])
    Xc = np.einsum("nij,nj->ni", Rs, rec.X[rec.obs_pt]) + ts
    u = Xc[:, :2] / Xc[:, 2:3] * [rec.K[0, 0], rec.K[1, 1]] + rec.K[:2, 2]
    return np.linalg.norm(u - rec.obs_uv, axis=1)


def _register_test(rec: Recon, images, kps, descs, train, test, ctx: RunContext):
    # descriptor of each 3D point's observation, per train image
    by_img: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for im in set(rec.obs_img.tolist()):
        sel = rec.obs_img == im
        by_img[im] = (rec.obs_kp[sel], rec.obs_pt[sel])
    reg = sorted(rec.R)
    n_ok = 0
    for q in test:
        near = sorted(reg, key=lambda r: abs(r - q))[:6]
        p3, p2 = [], []
        for r in near:
            if r not in by_img:
                continue
            kp_idx, pts = by_img[r]
            m = match_pair(descs[q], descs[r][kp_idx])
            for a, b in m:
                p3.append(rec.X[pts[b]])
                p2.append(kps[q][a])
        if len(p3) < 20:
            ctx.fallback("sfm", f"held-out view {q} could not be localised")
            continue
        p3, p2 = np.array(p3), np.array(p2)
        ok, rv, tv, inl = cv2.solvePnPRansac(p3, p2, rec.K, None, iterationsCount=2000,
                                             reprojectionError=REPROJ_PX, confidence=0.999,
                                             flags=cv2.SOLVEPNP_SQPNP)
        if not ok or inl is None or len(inl) < 15:
            ctx.fallback("sfm", f"held-out view {q} could not be localised")
            continue
        inl = inl.ravel()
        rv, tv = cv2.solvePnPRefineLM(p3[inl], p2[inl], rec.K, None, rv, tv)
        rec.R[q], rec.t[q] = cv2.Rodrigues(rv)[0], tv.ravel()
        n_ok += 1
    rec.stats["n_test_localised"] = n_ok
    rec.stats["n_test"] = len(test)
