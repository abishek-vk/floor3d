"""Mode B outputs: GLB/OBJ/PLY with observed and generated geometry kept apart, a provenance
report, Mode A layout files, and a self-contained viewer.

Export frame: the room frame (Y up, X/Z along the walls, metres) shifted so the floor is at
Y = 0 and the Layout's pixel (0,0) is at X = Z = 0. Mode A's extruded layout model lives in
exactly this frame, so the two overlay without any transform.
"""
from __future__ import annotations

import base64
import json
import os

import numpy as np
import trimesh

TEMPLATE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "viewer", "template_video.html")


def decimate(m: trimesh.Trimesh, cell: float) -> trimesh.Trimesh:
    """Vertex clustering on a `cell`-sized grid (colours averaged, degenerate faces dropped)."""
    if len(m.faces) == 0 or cell <= 0:
        return m
    q = np.floor(m.vertices / cell).astype(np.int64)
    _, inv, cnt = np.unique(q, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    n = len(cnt)
    V = np.zeros((n, 3))
    np.add.at(V, inv, m.vertices)
    V /= cnt[:, None]
    C = np.zeros((n, 4))
    np.add.at(C, inv, m.visual.vertex_colors.astype(float))
    C /= cnt[:, None]
    F = inv[m.faces]
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
    _, first = np.unique(np.sort(F, 1), axis=0, return_index=True)  # drop duplicate faces
    F = F[np.sort(first)]
    out = trimesh.Trimesh(V, F, vertex_colors=C.astype(np.uint8), process=False)
    out.remove_unreferenced_vertices()
    return out


def to_export_frame(rf) -> np.ndarray:
    """4x4: world -> export frame."""
    T = np.eye(4)
    T[:3, :3] = rf.Rm
    T[:3, 3] = -np.array([rf.origin_xz[0], rf.floor_y, rf.origin_xz[1]])
    return T


def build_scene(observed: trimesh.Trimesh, generated: trimesh.Trimesh | None, T: np.ndarray,
                export_cell: float = 0.03) -> tuple[trimesh.Scene, dict]:
    sc = trimesh.Scene()
    info = {}
    obs = decimate(observed, export_cell)
    obs.apply_transform(T)
    obs.metadata["provenance"] = "observed"
    sc.add_geometry(obs, node_name="observed_scan", geom_name="observed_scan")
    info["observed_faces"] = int(len(obs.faces))
    if generated is not None and len(generated.faces):
        g = generated.copy()
        g.apply_transform(T)
        g.metadata["provenance"] = "generated"
        sc.add_geometry(g, node_name="generated_shell", geom_name="generated_shell")
        info["generated_faces"] = int(len(g.faces))
    return sc, info


def export_all(sc: trimesh.Scene, out_dir: str) -> dict:
    paths = {"glb": os.path.join(out_dir, "scene.glb")}
    sc.export(paths["glb"], file_type="glb")
    _tag_glb_extras(paths["glb"])
    try:
        paths["obj"] = os.path.join(out_dir, "scene.obj")
        sc.export(paths["obj"], file_type="obj")
    except Exception as e:  # secondary format
        paths.pop("obj")
        paths["obj_error"] = str(e)
    for name, g in sc.geometry.items():  # one PLY per provenance class (CloudCompare/MeshLab friendly)
        p = os.path.join(out_dir, f"{name}.ply")
        g.export(p)
        paths[f"ply_{name}"] = p
    return paths


def _tag_glb_extras(path: str) -> None:
    """Write {"provenance": ...} into each node's glTF `extras`, so any glTF tool can filter
    generated geometry without relying on our node names."""
    import struct
    data = open(path, "rb").read()
    jlen = struct.unpack_from("<I", data, 12)[0]
    gl = json.loads(data[20:20 + jlen])
    for n in gl.get("nodes", []):
        nm = n.get("name", "")
        if nm.startswith(("observed", "generated")):
            n.setdefault("extras", {})["provenance"] = "generated" if nm.startswith("generated") else "observed"
    gl.setdefault("asset", {})["extras"] = {"note": "Mode B reconstruction; nodes carry extras.provenance"}
    js = json.dumps(gl, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    rest = data[20 + jlen:]
    out = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + len(rest)) + struct.pack("<II", len(js), 0x4E4F534A) + js + rest
    open(path, "wb").write(out)


def build_viewer(out_html: str, glb_path: str, info: dict) -> str:
    html = open(TEMPLATE, encoding="utf8").read()
    glb = base64.b64encode(open(glb_path, "rb").read()).decode()
    js = json.dumps(info).replace("</", "<\\/")
    html = html.replace("__INFO_JSON__", js).replace("__GLB_B64__", glb)
    with open(out_html, "w", encoding="utf8") as f:
        f.write(html)
    return out_html
