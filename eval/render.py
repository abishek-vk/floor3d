"""Headless GLB -> PNG snapshot (matplotlib, flat shaded) for reports and sanity checks.

    python eval/render.py out/model.glb out/model.png [--elev 55 --azim -60]
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402


def render(glb: str, png: str, elev: float = 55, azim: float = -60, size: int = 9) -> None:
    scene = trimesh.load(glb, force="scene")
    light = np.array([0.4, 0.8, 0.3])
    light /= np.linalg.norm(light)
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111, projection="3d")
    allv = []
    for node in scene.graph.nodes_geometry:
        T, gname = scene.graph[node]
        m = scene.geometry[gname].copy()
        m.apply_transform(T)
        try:
            c = np.array(m.visual.material.baseColorFactor, float).ravel()
            c = c / 255 if c.max() > 1 else c
            if c.size < 3:
                raise ValueError
        except Exception:
            try:  # some loaders expose the colour as main_color (uint8)
                c = np.array(m.visual.material.main_color, float).ravel() / 255
            except Exception:
                c = np.array([0.8, 0.8, 0.8, 1])
        c = c.copy()
        c[:3] = np.where(c[:3] <= 0.0031308, c[:3] * 12.92, 1.055 * c[:3] ** (1 / 2.4) - 0.055)  # linear -> sRGB
        shade = 0.45 + 0.55 * np.abs(m.face_normals @ light)
        cols = np.clip(c[None, :3] * shade[:, None], 0, 1)
        alpha = c[3] if len(c) > 3 else 1
        # world (x, y_up, z) -> plot (x, z, y) so "up" is up and the plan isn't mirrored
        v = m.vertices[m.faces][:, :, [0, 2, 1]] * np.array([1, -1, 1])
        ax.add_collection3d(Poly3DCollection(v, facecolors=np.c_[cols, np.full(len(cols), alpha)],
                                             linewidths=0))
        allv.append(v.reshape(-1, 3))
    V = np.concatenate(allv)
    lo, hi = V.min(0), V.max(0)
    c, r = (lo + hi) / 2, (hi - lo).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(0, 2 * r)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    fig.savefig(png, dpi=110, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("glb")
    ap.add_argument("png")
    ap.add_argument("--elev", type=float, default=55)
    ap.add_argument("--azim", type=float, default=-60)
    a = ap.parse_args()
    render(a.glb, a.png, a.elev, a.azim)
