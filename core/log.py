"""Run context: stage timing, fallback log, and debug-image dumping."""
from __future__ import annotations

import json
import logging
import os
import random
import time
from contextlib import contextmanager

import cv2
import numpy as np

log = logging.getLogger("fp3d")


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


class RunContext:
    def __init__(self, out_dir: str, debug: bool = True, progress=None):
        self.out_dir = out_dir
        self.progress = progress  # optional callable(stage_name), e.g. for a UI
        self.debug_dir = os.path.join(out_dir, "debug")
        self.debug = debug
        os.makedirs(self.debug_dir, exist_ok=True)
        self.fallbacks: list[dict] = []
        self.timings: dict[str, float] = {}
        self._n = 0

    def fallback(self, stage: str, msg: str) -> None:
        """Record that a fallback path fired. Never raises."""
        log.warning("[fallback] %s: %s", stage, msg)
        self.fallbacks.append({"stage": stage, "msg": msg})

    @contextmanager
    def stage(self, name: str):
        t = time.perf_counter()
        log.info("-> %s", name)
        if self.progress is not None:
            try:
                self.progress(name)
            except Exception:
                pass
        try:
            yield
        finally:
            self.timings[name] = round(time.perf_counter() - t, 3)

    def save_img(self, name: str, img: np.ndarray) -> None:
        if not self.debug or img is None:
            return
        self._n += 1
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(self.debug_dir, f"{self._n:02d}_{name}.png"), img)

    def report(self) -> dict:
        r = {"fallbacks": self.fallbacks, "timings_s": self.timings,
             "total_s": round(sum(self.timings.values()), 3)}
        with open(os.path.join(self.out_dir, "run_report.json"), "w") as f:
            json.dump(r, f, indent=1)
        return r
