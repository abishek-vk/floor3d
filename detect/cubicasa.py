"""Wrapper around the pretrained CubiCasa5K multi-task network (Kalervo et al. 2019).

Outputs per-pixel probabilities for 12 room classes and 11 icon classes plus
21 junction heatmaps, at the working-image resolution.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CC_REPO = os.path.join(ROOT, "third_party", "CubiCasa5k")
WEIGHTS = os.path.join(ROOT, "data", "weights", "model_best_val_loss_var.pkl")
SPLIT = (21, 12, 11)

_MODEL = None


@dataclass
class SegOutput:
    heatmaps: np.ndarray   # (21, H, W) float32, raw regression output
    rooms: np.ndarray      # (12, H, W) softmax probs
    icons: np.ndarray      # (11, H, W) softmax probs

    @property
    def room_label(self) -> np.ndarray:
        return self.rooms.argmax(0).astype(np.uint8)

    @property
    def icon_label(self) -> np.ndarray:
        return self.icons.argmax(0).astype(np.uint8)


def available() -> bool:
    return os.path.exists(WEIGHTS) and os.path.isdir(CC_REPO)


def _load():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    import torch
    if CC_REPO not in sys.path:
        sys.path.append(CC_REPO)  # append: its eval.py must not shadow our eval package
    from floortrans.models.hg_furukawa_original import hg_furukawa_original

    n = sum(SPLIT)
    model = hg_furukawa_original(n_classes=51)  # skip init_weights(): full checkpoint below
    model.conv4_ = torch.nn.Conv2d(256, n, bias=True, kernel_size=1)
    model.upsample = torch.nn.ConvTranspose2d(n, n, kernel_size=4, stride=4)
    ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    _MODEL = model
    return model


def predict(rgb: np.ndarray, tta: bool = False, max_side: int = 1024) -> SegOutput:
    """Run the network. Inference happens at <= max_side and is resized back.

    tta=True averages the 4 rotations as in the CubiCasa paper (4x slower).
    """
    import torch
    import torch.nn.functional as F

    model = _load()
    h, w = rgb.shape[:2]
    k = min(1.0, max_side / max(h, w))
    x = torch.from_numpy(rgb).float().permute(2, 0, 1)[None]
    x = 2 * (x / 255.0) - 1
    if k < 1.0:
        x = F.interpolate(x, scale_factor=k, mode="bilinear", align_corners=False)

    with torch.inference_mode():
        if not tta:
            pred = model(x)
        else:
            sys.path.append(CC_REPO) if CC_REPO not in sys.path else None
            from floortrans.loaders.augmentations import RotateNTurns
            rot = RotateNTurns()
            preds = []
            for fwd, back in [(0, 0), (1, -1), (2, 2), (-1, 1)]:
                p = model(rot(x, "tensor", fwd))
                p = rot(p, "tensor", back)
                p = rot(p, "points", back)
                preds.append(F.interpolate(p, size=x.shape[2:], mode="bilinear",
                                           align_corners=True))
            pred = torch.stack(preds).mean(0)
        pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=True)[0]

    a, b, _ = SPLIT
    return SegOutput(
        heatmaps=pred[:a].numpy(),  # raw, as CubiCasa's post-processing expects
        rooms=F.softmax(pred[a:a + b], 0).numpy(),
        icons=F.softmax(pred[a + b:], 0).numpy(),
    )
