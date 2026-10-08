"""Layout -> 3D scene (trimesh) -> GLB / OBJ.

Openings are cut without CSG: walls are extruded in horizontal bands whose 2D
footprints differ:
    [0, sill)      walls - doors
    [sill, head)   walls - doors - windows
    [head, H]      walls
This is exact for axis-aligned openings, robust, and fast.

World frame: Y up, metres. Plan pixel (x, y) -> world (x*s, ., y*s), so the
top-down view matches the input image.
"""
from __future__ import annotations

import numpy as np
import trimesh
from shapely.geometry import LineString, MultiPolygon, Polygon, box
from shapely.ops import unary_union
from shapely import affinity
from shapely.geometry import Point
from shapely.prepared import prep
from shapely.validation import make_valid

from core.layout import Furniture, Layout, Opening, Room, Wall

# Component colours (sRGB, 0-1), chosen so every component reads differently.
# Plain tuple = colour; nested = (colour, alpha, metallic, roughness).
PALETTE = {
    "walls_exterior": (0.78, 0.47, 0.36),            # terracotta
    "walls_interior": (0.93, 0.89, 0.80),            # warm cream
    "railing": ((0.28, 0.31, 0.36), 1.0, 0.6, 0.35),  # dark metal
    "door": (0.60, 0.38, 0.20),                      # wood leaf
    "door_frame": (0.36, 0.22, 0.12),                # dark wood
    "glass": ((0.55, 0.78, 0.96), 0.4, 0.0, 0.05),   # translucent blue
    "window_frame": (0.22, 0.26, 0.31),              # charcoal
    "sill": (0.97, 0.97, 0.97),                      # white
    "slab": (0.58, 0.58, 0.60),                      # concrete
    "ceiling": (0.98, 0.95, 0.88),                   # soft cream
}
ROOM_COLORS = {  # floors, by room type
    "Living Room": (0.80, 0.62, 0.40), "Bed Room": (0.55, 0.65, 0.90), "Kitchen": (0.96, 0.78, 0.35),
    "Bath": (0.40, 0.78, 0.82), "Entry": (0.78, 0.72, 0.64), "Storage": (0.66, 0.60, 0.52),
    "Garage": (0.50, 0.50, 0.52), "Outdoor": (0.52, 0.74, 0.42), "Undefined": (0.86, 0.83, 0.77),
}
FURNITURE_COLORS = {
    "closet": (0.72, 0.55, 0.36), "appliance": (0.76, 0.78, 0.82), "toilet": (0.97, 0.98, 1.0),
    "sink": (0.97, 0.98, 1.0), "bathtub": (0.97, 0.98, 1.0), "sauna_bench": (0.86, 0.68, 0.44),
    "fireplace": (0.42, 0.28, 0.24), "chimney": (0.50, 0.38, 0.32),
}
# plan (x, y, z_up) -> world (x, z_up, y). A reflection, so faces are re-wound.
_PLAN_TO_WORLD = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)


def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _mat(name, rgb, alpha=1.0, metal=0.0, rough=0.85):
    # palette values are sRGB (what you see); glTF baseColorFactor is linear
    lin = [_srgb_to_linear(c) for c in rgb]
    return trimesh.visual.material.PBRMaterial(
        name=name, baseColorFactor=[*lin, alpha], metallicFactor=metal, roughnessFactor=rough,
        alphaMode="BLEND" if alpha < 1 else "OPAQUE", doubleSided=alpha < 1)


def _clean(g):
    if g.is_empty:
        return []
    g = make_valid(g)
    if isinstance(g, Polygon):
        polys = [g]
    else:
        polys = [p for p in getattr(g, "geoms", []) if isinstance(p, Polygon)]
        polys += [q for p in getattr(g, "geoms", []) if isinstance(p, MultiPolygon) for q in p.geoms]
    return [p for p in polys if p.area > 1e-4]


def _extrude(geom, z0: float, z1: float, material) -> trimesh.Trimesh | None:
    meshes = []
    for p in _clean(geom):
        try:
            m = trimesh.creation.extrude_polygon(p, z1 - z0)
        except Exception:
            continue
        m.apply_translation([0, 0, z0])
        meshes.append(m)
    if not meshes:
        return None
    m = trimesh.util.concatenate(meshes)
    m.apply_transform(_PLAN_TO_WORLD)
    m.invert()
    m.visual = trimesh.visual.TextureVisuals(material=material)
    return m


def wall_footprint(walls, s: float):
    parts = []
    for w in walls:
        if w.polygon is not None and len(w.polygon) >= 3:
            p = Polygon(np.array(w.polygon) * s, [np.array(h) * s for h in (w.holes or [])])
            parts.append(p.buffer(0))
        else:
            ls = LineString([np.array(w.p0) * s, np.array(w.p1) * s])
            if ls.length > 0:
                # square caps fill L-corners; mitre keeps corners sharp
                parts.append(ls.buffer(w.thickness * s / 2, cap_style="square", join_style="mitre"))
    return unary_union(parts) if parts else Polygon()


def opening_footprint(o, s: float, wall_t_m: float):
    p0, p1 = np.array(o.p0) * s, np.array(o.p1) * s
    L = float(np.linalg.norm(p1 - p0))
    if L < 1e-3:
        return Polygon(), (p0 + p1) / 2, 0.0, 0.0
    across = max(o.thickness * s, wall_t_m) * 2.0  # generous: must cut fully through
    ang = np.degrees(np.arctan2(*(p1 - p0)[::-1]))
    c = (p0 + p1) / 2
    r = affinity.rotate(box(-L / 2, -across / 2, L / 2, across / 2), ang, origin=(0, 0))
    return affinity.translate(r, *c), c, L, ang


SLAB = 0.2  # m between storeys


def _level_layout(L: Layout, level: int, offset) -> Layout:
    """Elements of one storey, shifted by its alignment offset (px)."""
    dx, dy = offset

    def sp(p):
        return [p[0] + dx, p[1] + dy]

    def sr(r):
        return [sp(p) for p in r] if r is not None else None
    walls = [Wall(w.id, sp(w.p0), sp(w.p1), w.thickness, sr(w.polygon), w.kind,
                  [sr(h) for h in w.holes] if w.holes else None, w.level) for w in L.walls if w.level == level]
    ops = [Opening(o.id, o.type, sp(o.p0), sp(o.p1), o.thickness, o.wall_id, o.offset, o.confidence, o.level)
           for o in L.openings if o.level == level]
    rooms = [Room(r.id, sr(r.polygon), r.type, r.label, r.dims_text, r.level) for r in L.rooms if r.level == level]
    furn = [Furniture(f.id, f.type, sr(f.polygon), f.level) for f in L.furniture if f.level == level]
    return Layout(L.width, L.height, L.meters_per_px, L.scale_source, walls, ops, rooms, furn, {})


def build_scene(layout: Layout, wall_height: float = 2.7, sill: float = 0.9, head: float = 2.1,
                ceiling: bool = False, floor_t: float = 0.02) -> trimesh.Scene:
    """Single storey: as drawn. Several storeys (Layout.meta["levels"]): each one is
    aligned over the ground floor and raised by (wall height + slab); nodes are
    prefixed L0_, L1_, ... so viewers can show/hide floors."""
    levels = layout.meta.get("levels") or []
    if len(levels) < 2:
        return _build_single(layout, wall_height, sill, head, ceiling, floor_t)
    scene = trimesh.Scene()
    storey = wall_height + SLAB
    for lv in levels:
        i = lv["index"]
        sub = _build_single(_level_layout(layout, i, lv["offset"]), wall_height, sill, head,
                            ceiling, floor_t if i == 0 else SLAB)
        for name, g in sub.geometry.items():
            g.apply_translation([0, i * storey, 0])
            scene.add_geometry(g, node_name=f"L{i}_{name}", geom_name=f"L{i}_{name}")
    scene.metadata["levels"] = [{"index": lv["index"], "name": lv["name"], "elevation_m": lv["index"] * storey}
                                for lv in levels]
    return scene


def _wall_is_exterior(w, rooms) -> bool:
    """Exterior = rooms on at most one side. Probes just beside the wall, both sides."""
    if not rooms:
        return True
    a, b = np.array(w.p0, float), np.array(w.p1, float)
    r = b - a
    L = np.linalg.norm(r)
    if L < 1e-6:
        return True
    u = r / L
    n = np.array([-u[1], u[0]])
    d = w.thickness / 2 + 4
    sides = []
    for sgn in (1, -1):
        hits = sum(any(p.contains(Point(*(a + t * r + sgn * d * n))) for p in rooms)
                   for t in np.linspace(0.15, 0.85, 5))
        sides.append(hits >= 2)
    return not (sides[0] and sides[1])


def _build_single(layout: Layout, wall_height: float = 2.7, sill: float = 0.9, head: float = 2.1,
                  ceiling: bool = False, floor_t: float = 0.02) -> trimesh.Scene:
    s = layout.meters_per_px
    head = min(head, wall_height - 0.05)
    sill = min(sill, head - 0.1)
    scene = trimesh.Scene()
    M = {k: (_mat(k, *v) if isinstance(v[0], tuple) else _mat(k, v)) for k, v in PALETTE.items()}

    def add(mesh, name):
        if mesh is not None:
            scene.add_geometry(mesh, node_name=name, geom_name=name)

    rooms_px = [prep(Polygon(r.polygon).buffer(0)) for r in layout.rooms if len(r.polygon) >= 3]
    solid = [w for w in layout.walls if w.kind != "railing"]
    ext = [w for w in solid if _wall_is_exterior(w, rooms_px)]
    ext_ids = {id(w) for w in ext}
    wall_t_m = float(np.median([w.thickness for w in layout.walls])) * s if layout.walls else 0.15
    ext_fp = wall_footprint(ext, s)
    int_fp = wall_footprint([w for w in solid if id(w) not in ext_ids], s).difference(ext_fp)
    rail_fp = wall_footprint([w for w in layout.walls if w.kind == "railing"], s)

    doors, windows = [], []
    for o in layout.openings:
        fp, c, L, ang = opening_footprint(o, s, wall_t_m)
        if fp.is_empty:
            continue
        (doors if o.type == "door" else windows).append((o, fp, c, L, ang))
    door_u = unary_union([d[1] for d in doors]) if doors else Polygon()
    win_u = unary_union([d[1] for d in windows]) if windows else Polygon()

    for name, fp in (("walls_exterior", ext_fp), ("walls_interior", int_fp)):
        bands = [(0.0, sill, fp.difference(door_u)),
                 (sill, head, fp.difference(door_u).difference(win_u)),
                 (head, wall_height, fp)]
        wm = [m for z0, z1, g in bands if (m := _extrude(g, z0, z1, M[name])) is not None]
        if wm:
            add(trimesh.util.concatenate(wm), name)
    if not rail_fp.is_empty:
        add(_extrude(rail_fp, 0, 1.0, M["railing"]), "railings")

    def placed(g, c, ang):
        return affinity.translate(affinity.rotate(g, ang, origin=(0, 0)), *c)

    fw, t = 0.06, wall_t_m
    for o, fp, c, L, ang in doors:
        frame = [_extrude(placed(g, c, ang), z0, z1, M["door_frame"]) for g, z0, z1 in (
            (box(-L / 2, -t / 2, -L / 2 + fw, t / 2), 0, head),
            (box(L / 2 - fw, -t / 2, L / 2, t / 2), 0, head),
            (box(-L / 2, -t / 2, L / 2, t / 2), head - fw, head))]
        # door leaf, hinged at one jamb and swung 90 degrees open (doorway stays walkable)
        leaf_w = max(L - 2 * fw, 0.3)
        hx, hy = -L / 2 + fw, t / 2
        leaf = box(hx, hy, hx + 0.04, hy + leaf_w)
        frame = [m for m in frame if m is not None]
        if frame:  # one material per node: merging materials would bake a texture
            add(trimesh.util.concatenate(frame), f"door_{o.id}")
        add(_extrude(placed(leaf, c, ang), 0.01, head - fw - 0.01, M["door"]), f"doorleaf_{o.id}")
    wf = 0.05
    for o, fp, c, L, ang in windows:
        add(_extrude(placed(box(-L / 2, -0.012, L / 2, 0.012), c, ang), sill, head, M["glass"]),
            f"window_{o.id}")
        fr = [_extrude(placed(g, c, ang), z0, z1, M["window_frame"]) for g, z0, z1 in (
            (box(-L / 2, -0.035, -L / 2 + wf, 0.035), sill, head),
            (box(L / 2 - wf, -0.035, L / 2, 0.035), sill, head),
            (box(-L / 2, -0.035, L / 2, 0.035), head - wf, head),
            (box(-L / 2, -0.035, L / 2, 0.035), sill, sill + wf))]
        fr = [m for m in fr if m is not None]
        if fr:
            add(trimesh.util.concatenate(fr), f"windowframe_{o.id}")
        add(_extrude(placed(box(-L / 2 - 0.03, -t * 0.65, L / 2 + 0.03, t * 0.65), c, ang),
                     sill - 0.04, sill, M["sill"]), f"sill_{o.id}")

    all_rooms = []
    finish = 0.02
    for r in layout.rooms:
        if len(r.polygon) < 3:
            continue
        p = Polygon(np.array(r.polygon) * s).buffer(0)
        all_rooms.append(p)
        key = f"floor_{r.type}"
        if key not in M:
            M[key] = _mat(key, ROOM_COLORS.get(r.type, ROOM_COLORS["Undefined"]), rough=0.7)
        name = f"floor_{r.id}_{r.type.replace(' ', '_')}"
        add(_extrude(p, -finish, 0.0, M[key]), name)
        scene.metadata.setdefault("rooms", {})[name] = {"label": r.label, "type": r.type}
    if floor_t > finish and all_rooms:  # structural slab under an upper storey
        add(_extrude(unary_union(all_rooms + [ext_fp, int_fp]), -floor_t, -finish, M["slab"]), "slab")

    if layout.furniture:
        from vectorize.furniture import HEIGHT
        for f in layout.furniture:
            key = f"furniture_{f.type}"
            if key not in M:
                M[key] = _mat(key, FURNITURE_COLORS.get(f.type, (0.75, 0.72, 0.68)), rough=0.5)
            p = Polygon(np.array(f.polygon) * s).buffer(0)
            add(_extrude(p, 0.0, min(HEIGHT.get(f.type, 0.8), wall_height), M[key]),
                f"furniture_{f.id}_{f.type}")

    if ceiling and all_rooms:
        add(_extrude(unary_union(all_rooms + [ext_fp, int_fp]), wall_height, wall_height + 0.02,
                     M["ceiling"]), "ceiling")
    return scene


def export(scene: trimesh.Scene, out_prefix: str) -> dict:
    paths = {"glb": out_prefix + ".glb", "obj": out_prefix + ".obj"}
    scene.export(paths["glb"], file_type="glb")
    try:
        scene.export(paths["obj"], file_type="obj")
    except Exception as e:  # OBJ is a secondary format; never fail the run on it
        paths["obj_error"] = str(e)
        paths.pop("obj")
    return paths
