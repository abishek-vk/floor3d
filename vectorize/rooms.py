"""Rooms = enclosed free space between vectorized walls, typed by learned class votes."""
from __future__ import annotations

import cv2
import numpy as np

from core.layout import ROOM_CLASSES, Opening, Room, Wall
from detect.cubicasa import SegOutput

NON_ROOM = {ROOM_CLASSES.index(c) for c in ("Background", "Wall", "Railing")}
ROOM_IDX = [i for i in range(len(ROOM_CLASSES)) if i not in NON_ROOM]


def wall_quad(w: Wall, cap: float | None = None) -> np.ndarray:
    a, b = np.array(w.p0, float), np.array(w.p1, float)
    r = b - a
    L = np.linalg.norm(r)
    u = r / L if L > 1e-6 else np.array([1.0, 0.0])
    n = np.array([-u[1], u[0]])
    e = w.thickness / 2 if cap is None else cap
    h = w.thickness / 2
    a, b = a - u * e, b + u * e
    return np.array([a + n * h, b + n * h, b - n * h, a - n * h])


def barrier_mask(shape, walls: list[Wall], ops: list[Opening], extra: np.ndarray | None = None) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    for w in walls:
        cv2.fillPoly(m, [np.round(wall_quad(w)).astype(np.int32)], 1)
    for o in ops:  # openings close the room boundary in 2D
        q = wall_quad(Wall(0, o.p0, o.p1, o.thickness), cap=0)
        cv2.fillPoly(m, [np.round(q).astype(np.int32)], 1)
    if extra is not None:
        m |= extra.astype(np.uint8)
    return m


def _polygon(m: np.ndarray, eps: float) -> np.ndarray | None:
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    poly = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(float)
    return poly if len(poly) >= 3 else None


def _reflex_cuts(m: np.ndarray, eps: float) -> np.ndarray:
    """Rays extended from each reflex corner of region m along its incident edges."""
    cut = np.zeros(m.shape, np.uint8)
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return cut
    P = cv2.approxPolyDP(max(cs, key=cv2.contourArea), eps, True).reshape(-1, 2).astype(float)
    n = len(P)
    if n < 5:
        return cut
    area2 = np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
    h, w = m.shape
    for i in range(n):
        a, v, b = P[i - 1], P[i], P[(i + 1) % n]
        cr = (v[0] - a[0]) * (b[1] - v[1]) - (v[1] - a[1]) * (b[0] - v[0])
        if cr * area2 >= 0:  # convex corner (same turn as polygon orientation)
            continue
        for d in (v - a, v - b):  # continue each incident edge past the corner
            L = np.linalg.norm(d)
            if L < 1e-6:
                continue
            d = d / L
            t, last = 2.0, None
            while True:
                q = v + t * d
                x, y = int(round(q[0])), int(round(q[1]))
                if not (0 <= x < w and 0 <= y < h) or not m[y, x]:
                    break
                last = (x, y)
                t += 1.0
            if last is not None:
                cv2.line(cut, tuple(np.int32(np.round(v))), last, 1, 2)
    return cut & m.astype(np.uint8)


def split_by_semantics(m: np.ndarray, seg: SegOutput, min_area: float, t_typ: float
                       ) -> list[tuple[np.ndarray, int]]:
    """Split one free-space region into rooms along *straight virtual boundaries*.

    Candidate boundaries are rays extended from the region's reflex corners (where
    open-plan spaces meet); the network only decides which cells belong together
    (adjacent cells with the same majority class merge). A convex/rectangular
    region therefore can never be split. Returns [(mask, class_idx)].
    """
    whole = ROOM_IDX[int(np.argmax(seg.rooms[ROOM_IDX][:, m].sum(1)))]
    cut = _reflex_cuts(m, eps=max(2.0, 0.75 * t_typ))
    if not cut.any():
        return [(m, whole)]
    n, lab, st, _ = cv2.connectedComponentsWithStats((m & (cut == 0)).astype(np.uint8), connectivity=4)
    cells = [k for k in range(1, n) if st[k, cv2.CC_STAT_AREA] >= 0.25 * min_area]
    if len(cells) < 2:
        return [(m, whole)]
    # Hand cut pixels and sliver cells to the nearest real cell first, so that
    # adjacency is not blocked by a discarded sliver between two parallel rays.
    d = np.stack([cv2.distanceTransform((lab != k).astype(np.uint8), cv2.DIST_L2, 3) for k in cells])
    full = np.where(m, np.array(cells)[np.argmin(d, 0)], 0)
    cls = {k: ROOM_IDX[int(np.argmax(seg.rooms[ROOM_IDX][:, full == k].sum(1)))] for k in cells}
    parent = {k: k for k in cells}

    def find(k):
        while parent[k] != k:
            k = parent[k]
        return k
    ker = np.ones((3, 3), np.uint8)
    for k in cells:
        nb = np.unique(full[cv2.dilate((full == k).astype(np.uint8), ker) > 0])
        for j in nb:
            if j in cls and j != k and cls[j] == cls[k]:
                parent[find(j)] = find(k)
    groups = {}
    for k in cells:
        groups.setdefault(find(k), []).append(k)
    if len(groups) < 2:
        return [(m, whole)]
    gm = [(np.isin(full, ks), cls[r]) for r, ks in groups.items()]
    # small groups are not rooms: fold them into neighbours
    gm = [g for g in gm if g[0].sum() >= min_area] or [(m, whole)]
    if len(gm) < 2:
        return [(m, whole)]
    d = np.stack([cv2.distanceTransform((~g).astype(np.uint8), cv2.DIST_L2, 3) for g, _ in gm])
    owner = np.argmin(d, 0)
    return [((owner == i) & m, c) for i, (_, c) in enumerate(gm)]


def rooms_from_walls(walls: list[Wall], ops: list[Opening], seg: SegOutput, railing: np.ndarray,
                     t_typ: float, semantic_split: bool = True, learned_fill: bool = True
                     ) -> tuple[list[Room], np.ndarray]:
    h, w = seg.room_label.shape
    bar = barrier_mask((h, w), walls, ops, railing)
    free = (1 - bar).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    min_area = max(9 * t_typ * t_typ, 1e-4 * h * w)
    rooms = []
    for k in range(1, n):
        x, y, bw, bh, area = stats[k]
        if area < min_area:
            continue
        if x == 0 or y == 0 or x + bw >= w or y + bh >= h:
            continue  # touches the border: exterior
        m = lab == k
        probs = seg.rooms[:, m].mean(1)
        if probs[0] > 0.5:  # mostly background: exterior pocket / courtyard
            continue
        parts = split_by_semantics(m, seg, min_area, t_typ) if semantic_split else \
            [(m, ROOM_IDX[int(np.argmax(probs[ROOM_IDX]))])]
        for pm, cls in parts:
            # wall-bounded regions are already straight; split boundaries need more smoothing
            poly = _polygon(pm, 1.5)
            if poly is not None:
                rooms.append(Room(len(rooms), poly.tolist(), ROOM_CLASSES[cls]))
    if learned_fill:
        rooms += _learned_fill(rooms, bar, seg, min_area, t_typ, len(rooms))
    return rooms, bar


def _learned_fill(rooms: list[Room], bar: np.ndarray, seg: SegOutput, min_area: float,
                  t_typ: float, start_id: int) -> list[Room]:
    """Spaces walls cannot enclose (balconies behind thin railings, rooms open to the
    outside, missed walls): take the network's room regions that no wall-bounded room
    covers, minus the walls themselves."""
    h, w = bar.shape
    covered = bar.astype(bool).copy()
    for r in rooms:
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [np.round(np.asarray(r.polygon)).astype(np.int32)], 1)
        covered |= m.astype(bool)
    k = max(3, int(round(t_typ)) | 1)
    lab = seg.room_label
    out = []
    for cls in ROOM_IDX:
        cm = ((lab == cls) & ~covered).astype(np.uint8)
        cm = cv2.morphologyEx(cm, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
        n, cl, st, _ = cv2.connectedComponentsWithStats(cm, connectivity=4)
        for j in range(1, n):
            if st[j, cv2.CC_STAT_AREA] < min_area:
                continue
            poly = _polygon(cl == j, 2.0)
            if poly is not None:
                out.append(Room(start_id + len(out), poly.tolist(), ROOM_CLASSES[cls]))
    return out
