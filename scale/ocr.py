"""Metric scale from OCR'd annotations, by robust voting.

Every parsed annotation proposes metres-per-pixel hypotheses:
  pair   "5.66 m x 3.83 m" / 12'6" x 10'  inside a room -> room inner dims
  area   "ASH 16.5" / "12.4 m2" inside a room           -> sqrt(area / px area)
  length "7200" / "3.60 m" / 12'6"                      -> every wall-to-wall span
         along the text's axis that covers the text position (many hypotheses,
         at most one is right)
The scale is the peak of a kernel density over log(m/px). Wrong pairings scatter;
correct ones agree. Inlier annotations become constraints for the solver.
Falls back to door width (then a default prior) when support is too weak.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field

import cv2
import numpy as np
from shapely.geometry import Point, Polygon

from core.layout import Layout
from core.log import RunContext
from vectorize.walls import orientation

_ENGINE = None


@dataclass
class TextItem:
    text: str
    score: float
    box: np.ndarray        # (4, 2) in working px
    axis: str              # "x" if text runs horizontally, else "y"

    @property
    def center(self) -> np.ndarray:
        return self.box.mean(0)


@dataclass
class Vote:
    log_s: float
    weight: float
    src: int               # index of the text item
    kind: str
    info: dict = field(default_factory=dict)


def _engine():
    global _ENGINE
    if _ENGINE is None:
        import logging
        from rapidocr import RapidOCR
        try:
            _ENGINE = RapidOCR(params={"Global.log_level": "error"})
        except Exception:
            _ENGINE = RapidOCR()
        lg = logging.getLogger("RapidOCR")  # its constructor resets the level; silence after
        lg.setLevel(logging.ERROR)
        for h in lg.handlers:
            h.setLevel(logging.ERROR)
    return _ENGINE


CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache", "ocr")


def run_ocr(rgb: np.ndarray, rotated_pass: bool = True) -> list[TextItem]:
    """OCR with a per-image disk cache (OCR is deterministic and the slowest stage)."""
    key = hashlib.sha1(rgb.tobytes() + bytes(str(rgb.shape) + str(rotated_pass), "ascii")).hexdigest()
    cp = os.path.join(CACHE_DIR, key + ".json")
    if os.path.exists(cp):
        try:
            return [TextItem(d["text"], d["score"], np.array(d["box"]), d["axis"])
                    for d in json.load(open(cp, encoding="utf8"))]
        except Exception:
            pass
    items = _run_ocr(rgb, rotated_pass)
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cp, "w", encoding="utf8") as f:
            json.dump([{"text": it.text, "score": it.score, "box": it.box.tolist(), "axis": it.axis}
                       for it in items], f)
    except OSError:
        pass
    return items


def _run_ocr(rgb: np.ndarray, rotated_pass: bool = True) -> list[TextItem]:
    eng = _engine()
    h = rgb.shape[0]
    items: list[TextItem] = []
    r = eng(rgb)
    if r.boxes is not None:
        for b, t, s in zip(r.boxes, r.txts, r.scores):
            b = np.asarray(b, float)
            w_, h_ = np.ptp(b[:, 0]), np.ptp(b[:, 1])
            items.append(TextItem(t, float(s), b, "x" if w_ >= h_ else "y"))
    if rotated_pass:
        r = eng(cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE))
        if r.boxes is not None:
            for b, t, s in zip(r.boxes, r.txts, r.scores):
                b = np.asarray(b, float)
                ob = np.column_stack([b[:, 1], h - 1 - b[:, 0]])  # back to original frame
                c = ob.mean(0)
                if any(np.linalg.norm(c - it.center) < 12 and _norm(it.text) == _norm(t) for it in items):
                    continue
                items.append(TextItem(t, float(s), ob, "y"))
    return [it for it in items if it.score >= 0.6]


def _norm(t: str) -> str:
    return re.sub(r"\s+", "", t)


# ---------------------------------------------------------------- parsing
FT_IN = r"(\d{1,3})\s*['’′]\s*-?\s*(?:(\d{1,2}(?:\.\d+)?)\s*(?:\"|”|″|''))?"
NUM = r"(\d+(?:[.,]\d+)?)"
PAIR_RE = re.compile(rf"^\s*(?:{FT_IN}|{NUM}\s*(m|cm|mm)?)\s*[xX×*]\s*(?:{FT_IN}|{NUM}\s*(m|cm|mm)?)\s*$")
FT_RE = re.compile(rf"^\s*{FT_IN}\s*$")
UNIT_RE = re.compile(rf"^\s*{NUM}\s*(m|cm|mm)\s*$")
MM_RE = re.compile(r"^\s*(\d{1,2}\s?\d{3}|\d{3,5})\s*$")
M_RE = re.compile(r"^\s*(\d{1,2}[.,]\d{2})\s*$")
# "ASH 16.5": number separated from the room name by a space ("APK4,5" is an appliance code)
AREA_RE = re.compile(r"(?:^|\s)(\d{1,3}[.,]\d)\s*(?:m2|m²)?\s*$", re.I)
AREA_UNIT_RE = re.compile(r"(\d{1,4}(?:[.,]\d{1,2})?)\s*(?:m2|m²|sq\.?\s*m)", re.I)


def _f(x: str) -> float:
    return float(x.replace(",", ".").replace(" ", ""))


def _unit(v: float, u: str | None) -> float:
    return {"m": v, "cm": v / 100, "mm": v / 1000, None: v}[u]


def _ftin(ft, inch) -> float:
    return (float(ft) * 12 + (float(inch) if inch else 0.0)) * 0.0254


def parse(text: str) -> tuple[str, tuple] | None:
    text = re.sub(r"(?<=\d)[:;](?=\d{3}\b)", "", text)  # OCR reads "12 000" as "12:000"
    t =text.strip().replace("O", "0") if re.fullmatch(r"[\dO\s.,]+", text.strip()) else text.strip()
    if t.startswith(("(", "+", "-", "±")) or "/" in t:
        return None  # levels like (+0.590), window codes like 15/12
    m = PAIR_RE.match(t)
    if m:
        g = m.groups()
        a = _ftin(g[0], g[1]) if g[0] else _unit(_f(g[2]), g[3])
        b = _ftin(g[4], g[5]) if g[4] else _unit(_f(g[6]), g[7])
        if g[2] and not g[3] and g[6] and not g[7]:  # bare numbers: mm if large, else m
            a, b = (a / 1000, b / 1000) if max(a, b) > 100 else (a, b)
        if 0.5 <= a <= 50 and 0.5 <= b <= 50:
            return "pair", (a, b)
        return None
    m = FT_RE.match(t)
    if m and ("'" in t or "’" in t or "′" in t):
        v = _ftin(*m.groups())
        return ("length", (v,)) if 0.3 <= v <= 100 else None
    m = UNIT_RE.match(t)
    if m:
        v = _unit(_f(m.group(1)), m.group(2))
        return ("length", (v,)) if 0.3 <= v <= 100 else None
    m = MM_RE.match(t)
    if m:
        v = _f(m.group(1)) / 1000
        return ("length", (v,)) if 0.3 <= v <= 100 else None
    m = M_RE.match(t)
    if m:
        v = _f(m.group(1))
        return ("length", (v,)) if 0.3 <= v <= 50 else None
    mu = AREA_UNIT_RE.search(t)
    m = mu or AREA_RE.search(t)
    if m:
        v = _f(m.group(1))
        # second value: 1.0 if the unit (m2) was written, else 0.0
        return ("area", (v, 1.0 if mu else 0.0)) if 1.0 <= v <= 500 else None
    return None


# ---------------------------------------------------------------- voting
def _room_dims(poly) -> tuple[float, float]:
    (_, _), (w, h), _ = cv2.minAreaRect(np.asarray(poly, np.float32))
    return w, h


def _axis_coords(L: Layout, axis: str) -> tuple[np.ndarray, list[tuple[int, float]]]:
    """Candidate x (or y) coordinates of wall faces and centrelines, with the
    (wall id, offset in half-thicknesses) each one came from."""
    cs = []
    want = "v" if axis == "x" else "h"
    k = 0 if axis == "x" else 1
    for w in L.walls:
        if w.kind != "wall" or orientation(w) != want:
            continue
        c = (w.p0[k] + w.p1[k]) / 2
        for off in (-1.0, 0.0, 1.0):
            cs.append((c + off * w.thickness / 2, w.id, off))
    cs.sort()
    keep = []
    for c in cs:
        if not keep or c[0] - keep[-1][0] > 1.5:
            keep.append(c)
    return np.array([c[0] for c in keep]), [(c[1], c[2]) for c in keep]


def collect_votes(items: list[TextItem], L: Layout) -> list[Vote]:
    votes: list[Vote] = []
    rooms = [(r, Polygon(r.polygon)) for r in L.rooms if len(r.polygon) >= 3]
    coords = {ax: _axis_coords(L, ax) for ax in ("x", "y")}
    total_px = sum(abs(p.area) for r, p in rooms if r.type != "Outdoor")
    for i, it in enumerate(items):
        p = parse(it.text)
        if p is None:
            continue
        kind, vals = p
        c = it.center
        host = next((r for r, poly in rooms if poly.buffer(2).contains(Point(*c))), None)
        if kind == "pair" and host is not None:
            w, h = _room_dims(host.polygon)
            a, b = vals
            for s1, s2 in ((a / w, b / h), (b / w, a / h)):
                if abs(np.log(s1 / s2)) < 0.15:
                    votes.append(Vote(float(np.log((s1 + s2) / 2)), 3.0, i, "pair",
                                      {"room": host.id, "a": a, "b": b}))
        elif kind == "area":
            # An area label is either the host room's area or the whole unit's area.
            if host is not None:
                A_px = abs(Polygon(host.polygon).area)
                if A_px > 0:
                    votes.append(Vote(float(0.5 * np.log(vals[0] / A_px)), 1.5, i, "area",
                                      {"room": host.id, "area": vals[0], "explicit": vals[1]}))
            if total_px > 0:
                votes.append(Vote(float(0.5 * np.log(vals[0] / total_px)), 1.0, i, "total_area",
                                  {"area": vals[0], "explicit": vals[1]}))
        elif kind == "length":
            k = 0 if it.axis == "x" else 1
            cs, refs = coords[it.axis]
            pos = c[k]
            lo_i = np.nonzero(cs < pos)[0][-12:]
            hi_i = np.nonzero(cs > pos)[0][:12]
            spans = [(a_, b_) for a_ in lo_i for b_ in hi_i if cs[b_] - cs[a_] > 10]
            if not spans:
                continue
            wgt = 1.0 / np.sqrt(len(spans))
            for a_, b_ in spans:
                votes.append(Vote(float(np.log(vals[0] / (cs[b_] - cs[a_]))), wgt, i, "length",
                                  {"axis": it.axis, "lo": float(cs[a_]), "hi": float(cs[b_]),
                                   "lo_ref": refs[a_], "hi_ref": refs[b_], "m": vals[0]}))
    return votes


def mode_scale(votes: list[Vote], bw: float = 0.02, prior: float | None = None,
               prior_sigma: float = 0.35) -> tuple[float | None, list[Vote], float]:
    """Peak of the weighted vote density over log(m/px), optionally times a broad
    log-normal prior (from door widths) so coincidental span agreements far from
    any plausible scale cannot win."""
    if not votes:
        return None, [], 0.0
    x = np.array([v.log_s for v in votes])
    w = np.array([v.weight for v in votes])
    grid = np.arange(x.min() - 0.1, x.max() + 0.1, bw / 4)
    dens = (w[None, :] * np.exp(-0.5 * ((grid[:, None] - x[None, :]) / bw) ** 2)).sum(1)
    if prior is not None:
        dens = dens * np.exp(-0.5 * ((grid - np.log(prior)) / prior_sigma) ** 2)
    peak = grid[int(np.argmax(dens))]
    inl = [v for v in votes if abs(v.log_s - peak) < 2 * bw]
    # one vote per text item (its best one) so a text cannot support itself twice
    best = {}
    for v in inl:
        if v.src not in best or abs(v.log_s - peak) < abs(best[v.src].log_s - peak):
            best[v.src] = v
    inl = list(best.values())
    support = sum(v.weight for v in inl)
    if not inl:
        return None, [], 0.0
    xs = np.array([v.log_s for v in inl])
    ws = np.array([v.weight for v in inl])
    return float(np.exp(np.average(xs, weights=ws))), inl, float(support)


# Acceptance rule (selected on the val split; see README): v1 | v2 | v3
RULE = os.environ.get("FP3D_SCALE_RULE", "v2")
GATE = float(os.environ.get("FP3D_SCALE_GATE", "0.25"))  # max |log(ocr / door)|, chosen on val


def _independent(items: list[TextItem], srcs: list[int]) -> int:
    """Number of distinct annotations among text indices: the same parsed value read
    twice within 60 px (both OCR passes, or a duplicated label) counts once."""
    groups: list[tuple[tuple, np.ndarray]] = []
    for i in srcs:
        p = parse(items[i].text)
        key = (p[0], round(p[1][0], 2)) if p else (items[i].text,)
        c = items[i].center
        if not any(k == key and np.linalg.norm(c - cc) < 60 for k, cc in groups):
            groups.append((key, c))
    return len(groups)


def estimate_scale(ing, L: Layout, ctx: RunContext) -> Layout:
    try:
        items = run_ocr(ing.rgb)
    except Exception as e:  # OCR engine missing/broken: keep the existing scale
        ctx.fallback("ocr", f"OCR failed ({e}); keeping {L.scale_source} scale")
        return L
    # Text positions are reused later, e.g. to find floor labels ("2. kerros", "Ground floor").
    L.meta["ocr_items"] = [{"text": it.text, "c": [float(it.center[0]), float(it.center[1])]}
                           for it in items]
    # Room labels: non-numeric text inside a room polygon.
    polys = [(r, Polygon(r.polygon).buffer(2)) for r in L.rooms if len(r.polygon) >= 3]
    for it in items:
        # Room names start with a capital ("KEITTIÖ", "Bedroom"); lowercase strings are
        # drawing notes ("siirr", "säleikko").
        if parse(it.text) is None and re.match(r"^[A-ZÄÖÅ][A-Za-zÄÖÅäöå]", it.text.strip()):
            r = next((r for r, p in polys if p.contains(Point(*it.center))), None)
            if r is None:
                continue
            key = _norm(it.text).upper()[:5]
            if r.label is not None and key in _norm(r.label).upper():
                continue  # same word read twice (both OCR passes)
            r.label = it.text if r.label is None else f"{r.label} {it.text}"
        elif parse(it.text) is not None:
            r = next((r for r, p in polys if p.contains(Point(*it.center))), None)
            if r is not None:
                r.dims_text.append(it.text)

    votes = collect_votes(items, L)
    if RULE == "v3":
        votes = [v for v in votes if v.kind != "total_area"]
    prior = L.meters_per_px if L.scale_source == "door_width" else None
    s, inl, support = mode_scale(votes, prior=prior)
    n_src = len({v.src for v in inl})
    n_ind = _independent(items, [v.src for v in inl])
    if RULE == "v1":
        # one annotation may decide alone if it is a size pair or an area with a unit
        strong = any(v.kind == "pair" or (v.kind in ("area", "total_area") and v.info.get("explicit"))
                     for v in inl)
        ok = n_src >= 2 or (strong and support >= 1.0)
    else:
        # need two *independent* agreeing annotations (the same label read by both OCR
        # passes counts once); only a room size pair "a x b" may decide alone
        ok = n_ind >= 2 or any(v.kind == "pair" for v in inl)
    L.meta["ocr"] = {"n_text": len(items), "n_votes": len(votes), "support": support,
                     "n_inlier_texts": n_src, "n_independent": n_ind, "rule": RULE,
                     "texts": [it.text for it in items][:200]}
    # Two independent cues must not disagree wildly. On val, OCR beat the door-width cue on
    # 10/13 plans, and every gross OCR failure disagreed with it by > 28%.
    if s is not None and ok and prior is not None and abs(np.log(s / prior)) > GATE:
        ctx.fallback("scale", f"OCR scale {s:.5f} disagrees with door-width scale {prior:.5f} "
                              f"by {abs(s / prior - 1):.0%}; keeping door width")
        L.meta["ocr"]["rejected_scale"] = s
        return L
    if s is None or not ok:
        ctx.fallback("scale", f"OCR scale support too weak ({support:.1f} from {n_src} texts); "
                              f"keeping {L.scale_source}")
        return L
    # Sanity: implied building size must be plausible.
    ext = max(L.width, L.height) * s
    if not (3 <= ext <= 300):
        ctx.fallback("scale", f"OCR scale implies {ext:.0f} m plan extent; rejected")
        return L
    L.meters_per_px, L.scale_source = s, "ocr"
    L.meta["dim_constraints"] = [{"kind": v.kind, **v.info, "text": items[v.src].text} for v in inl]
    return L
