"""Shared structured-layout representation passed between every pipeline stage.

All 2D coordinates are in *pixels of the input image* (x right, y down).
`meters_per_px` converts to metric. The 3D exporter maps plan (x, y) to
world (X=x*s, Z=y*s) with Y up, so a top-down view matches the image.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

ROOM_CLASSES = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room", "Bed Room",
                "Bath", "Entry", "Railing", "Storage", "Garage", "Undefined"]
ICON_CLASSES = ["No Icon", "Window", "Door", "Closet", "Electrical Appliance", "Toilet",
                "Sink", "Sauna Bench", "Fire Place", "Bathtub", "Chimney"]


@dataclass
class Wall:
    id: int
    p0: list[float]          # centerline start [x, y] px
    p1: list[float]          # centerline end [x, y] px
    thickness: float         # px
    polygon: list[list[float]] | None = None  # explicit footprint (baseline uses mask contours)
    kind: str = "wall"       # wall | railing
    holes: list[list[list[float]]] | None = None  # interior rings of `polygon`
    level: int = 0           # storey index (0 = ground); see Layout.meta["levels"]


@dataclass
class Opening:
    id: int
    type: str                # door | window
    p0: list[float]          # endpoints along the host wall centerline, px
    p1: list[float]
    thickness: float         # px, extent across the wall
    wall_id: int | None = None
    offset: float | None = None  # px from host wall p0 to opening center
    confidence: float = 1.0
    level: int = 0


@dataclass
class Room:
    id: int
    polygon: list[list[float]]   # exterior ring, px
    type: str = "Undefined"      # one of ROOM_CLASSES
    label: str | None = None     # OCR text, if any
    dims_text: list[str] = field(default_factory=list)
    level: int = 0


@dataclass
class Furniture:
    id: int
    type: str
    polygon: list[list[float]]
    level: int = 0


@dataclass
class Layout:
    width: int
    height: int
    meters_per_px: float
    scale_source: str = "default"
    walls: list[Wall] = field(default_factory=list)
    openings: list[Opening] = field(default_factory=list)
    rooms: list[Room] = field(default_factory=list)
    furniture: list[Furniture] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf8") as f:
            json.dump(self.to_dict(), f, indent=1)

    @staticmethod
    def from_dict(d: dict) -> "Layout":
        return Layout(
            width=d["width"], height=d["height"], meters_per_px=d["meters_per_px"],
            scale_source=d.get("scale_source", "default"),
            walls=[Wall(**w) for w in d.get("walls", [])],
            openings=[Opening(**o) for o in d.get("openings", [])],
            rooms=[Room(**r) for r in d.get("rooms", [])],
            furniture=[Furniture(**f) for f in d.get("furniture", [])],
            meta=d.get("meta", {}),
        )

    @staticmethod
    def load(path: str) -> "Layout":
        with open(path, encoding="utf8") as f:
            return Layout.from_dict(json.load(f))

    def scaled(self, k: float) -> "Layout":
        """Return a copy with all pixel coordinates multiplied by k."""
        def sp(p):
            return [p[0] * k, p[1] * k]

        def sr(r):
            return [sp(p) for p in r] if r is not None else None

        return Layout(
            width=round(self.width * k), height=round(self.height * k),
            meters_per_px=self.meters_per_px / k, scale_source=self.scale_source,
            walls=[Wall(w.id, sp(w.p0), sp(w.p1), w.thickness * k, sr(w.polygon), w.kind,
                        [sr(h) for h in w.holes] if w.holes else None, w.level)
                   for w in self.walls],
            openings=[Opening(o.id, o.type, sp(o.p0), sp(o.p1), o.thickness * k, o.wall_id,
                              None if o.offset is None else o.offset * k, o.confidence, o.level)
                      for o in self.openings],
            rooms=[Room(r.id, sr(r.polygon), r.type, r.label, list(r.dims_text), r.level)
                   for r in self.rooms],
            furniture=[Furniture(f.id, f.type, sr(f.polygon), f.level) for f in self.furniture],
            meta=_scale_meta(self.meta, k),
        )


def _scale_meta(meta: dict, k: float) -> dict:
    m = dict(meta)
    if "levels" in m:  # per-level pixel offsets scale with the coordinates
        m["levels"] = [dict(lv, offset=[lv["offset"][0] * k, lv["offset"][1] * k]) for lv in m["levels"]]
    return m
