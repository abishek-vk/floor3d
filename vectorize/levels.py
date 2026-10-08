"""Multi-storey sheets: detect floors drawn side by side, order them, align them for stacking.

A sheet often shows every storey of one building next to each other. Read naively, that
becomes several buildings on the same ground. Here we:
  1. group walls + rooms into separate footprints (closing gap ~0.25 m),
  2. accept them as storeys only if the drawing labels them ("1. kerros", "2nd floor",
     "Ground floor", "Basement", ...) or there are >= 2 footprints of similar size
     (so a small detached garage or shed is not stacked on the house),
  3. order them (labels first; otherwise the larger footprint is the ground floor),
  4. align each storey onto the ground floor by phase-correlating their wall drawings.
Results: `level` on every element, and Layout.meta["levels"] =
  [{"index", "name", "offset": [dx, dy] px to add so the storey sits over the ground floor}].
"""
from __future__ import annotations

import re

import cv2
import numpy as np

from core.layout import Layout
from core.log import RunContext
from vectorize.rooms import wall_quad

ORDINALS = {"basement": -1, "cellar": -1, "kellari": -1, "lower ground": -0.5, "ground": 0,
            "pohja": 0, "main": 0, "first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3,
            "3rd": 3, "upper": 5, "attic": 9, "ullakko": 9, "loft": 9}
NUM_RE = re.compile(r"(?:^|\D)([0-9])\s*[.:]?\s*(?:kerros|krs|floor|fl\b|storey|story|etage|og\b|level)", re.I)
WORD_RE = re.compile(r"(basement|cellar|kellari|lower ground|ground|pohja|main|first|1st|second|2nd|third|3rd|"
                     r"upper|attic|ullakko|loft)\w*\s*(?:kerros|floor|fl\b|storey|story|level)?", re.I)
FLOORISH = re.compile(r"kerros|krs\b|floor|storey|story|etage|level|basement|cellar|kellari|attic|ullakko|loft", re.I)


def floor_key(text: str) -> float | None:
    """Sort key for a floor label, or None if the text is not a floor label."""
    if not FLOORISH.search(text):
        return None
    m = NUM_RE.search(text)
    if m:
        return float(m.group(1))
    m = WORD_RE.search(text)
    if m:
        w = m.group(1).lower()
        return float(next(v for k, v in ORDINALS.items() if w.startswith(k)))
    if re.search(r"kellari", text, re.I):
        return -1.0
    return None


def _footprint_mask(L: Layout, shape) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    for w in L.walls:
        cv2.fillPoly(m, [np.round(wall_quad(w)).astype(np.int32)], 1)
    for r in L.rooms:
        if len(r.polygon) >= 3:
            cv2.fillPoly(m, [np.round(np.asarray(r.polygon)).astype(np.int32)], 1)
    return m


def _wall_mask(L: Layout, shape, level: int) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    for w in L.walls:
        if w.level == level and w.kind == "wall":
            cv2.fillPoly(m, [np.round(wall_quad(w)).astype(np.int32)], 1)
    return m


def _reread_label(rgb: np.ndarray, c, size: int) -> float | None:
    """A floor word was read without its number (vertical text often loses "2."):
    OCR a crop around it in all four orientations and look for a full floor label."""
    from scale.ocr import _engine
    x, y = int(c[0]), int(c[1])
    h, w = rgb.shape[:2]
    crop = rgb[max(0, y - size):min(h, y + size), max(0, x - size):min(w, x + size)]
    if crop.size == 0:
        return None
    eng = _engine()
    for rot in (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180):
        im = crop if rot is None else cv2.rotate(crop, rot)
        im = cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)  # small text
        r = eng(im)
        for t in (r.txts or []):
            k = floor_key(t)
            if k is not None:
                return k
    return None


def detect_levels(L: Layout, ctx: RunContext, rgb: np.ndarray | None = None) -> Layout:
    try:
        return _detect(L, ctx, rgb)
    except Exception as e:  # never fail the run over storey detection
        ctx.fallback("levels", f"storey detection skipped: {e}")
        return L


def _detect(L: Layout, ctx: RunContext, rgb: np.ndarray | None) -> Layout:
    h, w = L.height, L.width
    if not L.rooms or not L.walls:
        return L
    fp = _footprint_mask(L, (h, w))
    # walls and rooms of one building already touch; only bridge hairline gaps (~0.25 m),
    # since storeys on one sheet can be drawn as little as half a metre apart
    gap = int(np.clip(0.25 / max(L.meters_per_px, 1e-6), 3, max(h, w) / 40))
    closed = cv2.morphologyEx(fp, cv2.MORPH_CLOSE, np.ones((gap, gap), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if n <= 2:
        return L
    comps = sorted(range(1, n), key=lambda k: -st[k, cv2.CC_STAT_AREA])
    big = st[comps[0], cv2.CC_STAT_AREA]

    # floor labels from OCR, assigned to the footprint they sit in or next to
    labels = {}
    for it in L.meta.get("ocr_items", []):
        key = floor_key(it["text"])
        if key is None and FLOORISH.search(it["text"]) and rgb is not None:
            key = _reread_label(rgb, it["c"], size=int(max(60, 0.06 * max(h, w))))
            if key is not None:
                it = dict(it, text=f"{int(key)}. {it['text']}" if key >= 1 else it["text"])
        if key is None:
            continue
        x, y = (int(round(v)) for v in it["c"])
        best, bd = None, None
        for k in comps:
            x0, y0, bw, bh = st[k, :4]
            d = max(x0 - x, 0, x - (x0 + bw)) + max(y0 - y, 0, y - (y0 + bh))
            if bd is None or d < bd:
                best, bd = k, d
        if best is not None and bd < 0.25 * max(h, w) and best not in labels:
            labels[best] = (key, it["text"])

    main = [k for k in comps if st[k, cv2.CC_STAT_AREA] >= 0.35 * big]
    labelled = [k for k in main if k in labels]
    if len(labelled) >= 2:
        floors = labelled
    else:
        # unlabelled: similar size and shape required (a garage or shed is not a storey)
        b0 = st[comps[0], 2:4].astype(float)
        floors = [k for k in main if st[k, cv2.CC_STAT_AREA] >= 0.5 * big and
                  np.all(np.abs(np.sort(st[k, 2:4]) / np.sort(b0) - 1) < 0.45)]
        if len(floors) < 2:
            return L

    if all(k in labels for k in floors):
        floors.sort(key=lambda k: labels[k][0])
    else:  # larger footprint at the bottom; tie -> lower on the sheet
        floors.sort(key=lambda k: (-st[k, cv2.CC_STAT_AREA], -(st[k, 1] + st[k, 3])))

    # every element goes to the floor footprint it lies in (or the nearest one)
    dist = np.stack([cv2.distanceTransform((lab != k).astype(np.uint8), cv2.DIST_L2, 3) for k in floors])

    def level_of(pts) -> int:
        c = np.mean(np.asarray(pts, float), 0)
        x, y = int(np.clip(round(c[0]), 0, w - 1)), int(np.clip(round(c[1]), 0, h - 1))
        return int(np.argmin(dist[:, y, x]))
    for wl in L.walls:
        wl.level = level_of([wl.p0, wl.p1])
    for o in L.openings:
        o.level = level_of([o.p0, o.p1])
    for r in L.rooms:
        r.level = level_of(r.polygon)
    for f in L.furniture:
        f.level = level_of(f.polygon)

    # align each storey onto the ground floor: phase correlation of wall masks
    base = 0
    x0, y0, bw, bh = st[floors[base], :4]
    ref = _wall_mask(L, (h, w), base)[y0:y0 + bh, x0:x0 + bw].astype(np.float32)
    levels = []
    for i, k in enumerate(floors):
        name = labels[k][1] if k in labels else ("Ground floor" if i == 0 else f"Floor {i + 1}")
        if i == base:
            levels.append({"index": i, "name": name, "offset": [0.0, 0.0]})
            continue
        xi, yi, wi, hi = st[k, :4]
        mov = _wall_mask(L, (h, w), i)[yi:yi + hi, xi:xi + wi].astype(np.float32)
        H, W = max(bh, hi), max(bw, wi)
        a = np.zeros((H, W), np.float32)
        b = np.zeros((H, W), np.float32)
        a[:bh, :bw] = ref
        b[:hi, :wi] = mov
        win = cv2.createHanningWindow((W, H), cv2.CV_32F)
        fa = np.zeros((H, W), np.float32)
        fb = np.zeros((H, W), np.float32)
        fa[:bh, :bw] = (lab[y0:y0 + bh, x0:x0 + bw] == floors[base])
        fb[:hi, :wi] = (lab[yi:yi + hi, xi:xi + wi] == k)
        # candidate shifts (local coords of this storey -> ground floor's): phase correlation
        # of walls and of filled footprints, plus edge/centre alignments of the footprints
        cands = []
        for A, B in ((a, b), (fa, fb)):
            (sx, sy), _ = cv2.phaseCorrelate(B * win, A * win)
            cands.append((sx, sy))
        for ax in (0.0, 0.5, 1.0):
            for ay in (0.0, 0.5, 1.0):
                cands.append((ax * (bw - wi), ay * (bh - hi)))
        # score = overlap of (slightly dilated) wall drawings after the shift
        ker = np.ones((5, 5), np.uint8)
        A = cv2.dilate((a > 0).astype(np.uint8), ker)
        Bm = cv2.dilate((b > 0).astype(np.uint8), ker)

        def score(sx, sy):
            M = np.float32([[1, 0, sx], [0, 1, sy]])
            moved = cv2.warpAffine(Bm, M, (W, H), flags=cv2.INTER_NEAREST)
            u = np.count_nonzero(A | moved)
            return np.count_nonzero(A & moved) / u if u else 0.0
        scored = sorted(((score(*c), c) for c in cands), reverse=True)
        best, (sx, sy) = scored[0]
        ctx.timings[f"levels_wall_overlap_{i}"] = round(float(best), 3)
        dx, dy = (x0 - xi) + sx, (y0 - yi) + sy
        if best < 0.15:
            ctx.fallback("levels", f"storey '{name}' only loosely aligned with the floor below "
                                   f"(wall overlap {best:.0%})")
        levels.append({"index": i, "name": name, "offset": [float(dx), float(dy)]})
    L.meta["levels"] = levels
    ctx.fallback("levels", f"{len(levels)} storeys found on one sheet: "
                           + ", ".join(lv["name"] for lv in levels) + "; stacked vertically")
    return L
