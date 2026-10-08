"""Fetch Replica scenes (Mode B evaluation data) straight out of the 12 GB NICE-SLAM zip
using HTTP range requests, so we never download the whole archive.

    python data/fetch_replica.py --scenes room0 office0 --step 4 --out data/replica

Writes <out>/<scene>/{video.mp4, frames/frame*.jpg, traj.txt, gt_mesh.ply, cam_params.json}.
`video.mp4` is what the Mode B pipeline consumes (the frames are rendered from a smooth
handheld-like trajectory); traj.txt (camera-to-world, OpenCV axes, one 4x4 per line) and the
GT mesh are only read by the evaluator.
"""
import argparse
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_cubicasa import HttpRangeFile  # noqa: E402

URL = "https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip"
SCENES = ["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]


def main():
    import zipfile
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["room0"], choices=SCENES)
    ap.add_argument("--step", type=int, default=4, help="keep every k-th of the 2000 frames")
    ap.add_argument("--fps", type=float, default=30 / 4)
    ap.add_argument("--out", default="data/replica")
    a = ap.parse_args()

    z = zipfile.ZipFile(HttpRangeFile(URL))
    names = set(z.namelist())
    cam = json.loads(z.read("Replica/cam_params.json"))
    for sc in a.scenes:
        dst = os.path.join(a.out, sc)
        os.makedirs(os.path.join(dst, "frames"), exist_ok=True)
        json.dump(cam, open(os.path.join(dst, "cam_params.json"), "w"), indent=1)
        for src, name in ((f"Replica/{sc}/traj.txt", "traj.txt"), (f"Replica/{sc}_mesh.ply", "gt_mesh.ply")):
            p = os.path.join(dst, name)
            if not os.path.exists(p):
                open(p, "wb").write(z.read(src))
        frames = sorted(n for n in names if n.startswith(f"Replica/{sc}/results/frame") and n.endswith(".jpg"))
        frames = frames[::a.step]
        paths = []
        for i, n in enumerate(frames):
            p = os.path.join(dst, "frames", os.path.basename(n))
            if not os.path.exists(p):
                open(p, "wb").write(z.read(n))
            paths.append(p)
            if i % 50 == 0:
                print(f"{sc}: {i + 1}/{len(frames)}", flush=True)
        im = cv2.imread(paths[0])
        vw = cv2.VideoWriter(os.path.join(dst, "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), a.fps,
                             (im.shape[1], im.shape[0]))
        for p in paths:
            vw.write(cv2.imread(p))
        vw.release()
        json.dump({"step": a.step, "frame_ids": [int(os.path.basename(n)[5:11]) for n in frames]},
                  open(os.path.join(dst, "frames.json"), "w"))
        print(f"{sc}: done, {len(paths)} frames")


if __name__ == "__main__":
    main()
