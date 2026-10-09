"""Build a static, demo-only copy of the web UI for Vercel (or any static host).

Runs the full pipeline once per demo plan and writes the outputs plus JSON
stand-ins for the API, so the page works without the Python server:

    python deploy/build_static.py            # -> web/
    python deploy/build_static.py --reuse    # skip demos already built

Uploads are disabled in the static build (the page says so).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "app"))

import server  # noqa: E402  (reuses _public() so the JSON matches the live API)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "web"))
    ap.add_argument("--wall-height", type=float, default=2.7)
    ap.add_argument("--reuse", action="store_true", help="keep demo outputs that already exist")
    a = ap.parse_args()

    from run import run

    out = a.out
    jobs_dir = os.path.join(out, "jobs")
    api_dir = os.path.join(out, "api")
    os.makedirs(os.path.join(api_dir, "jobs"), exist_ok=True)
    server.JOBS_DIR = jobs_dir

    html = open(os.path.join(server.STATIC, "index.html"), encoding="utf8").read()
    html = html.replace("<head>", "<head>\n<script>window.FP3D_STATIC = 1;</script>", 1)
    open(os.path.join(out, "index.html"), "w", encoding="utf8").write(html)
    shutil.copytree(server.DEMOS, os.path.join(out, "demos"), dirs_exist_ok=True)

    demos = json.load(open(os.path.join(server.DEMOS, "demos.json"), encoding="utf8"))
    json.dump(demos, open(os.path.join(api_dir, "demos.json"), "w", encoding="utf8"))
    json.dump({"accept": sorted(server.FORMATS), "max_mb": server.MAX_BYTES // (1024 * 1024),
               "video_accept": sorted(server.VIDEO_FORMATS),
               "video_max_mb": server.MAX_VIDEO_BYTES // (1024 * 1024)},
              open(os.path.join(api_dir, "formats.json"), "w", encoding="utf8"))

    for d in demos:
        jid, job_out = d["id"], os.path.join(jobs_dir, d["id"])
        report_path = os.path.join(job_out, "run_report.json")
        t0 = time.time()
        if a.reuse and os.path.isfile(report_path):
            rep = json.load(open(report_path, encoding="utf8"))
            print(f"[reuse] {jid}")
        else:
            print(f"[run]   {jid} ...", flush=True)
            rep = run(os.path.join(server.DEMOS, jid + ".png"), job_out, "full", wall_height=a.wall_height)
        job = {"id": jid, "source": "demo", "name": d["title"], "mode": "plan", "wall_height": a.wall_height,
               "stage": "done", "status": "done", "error": None, "created": t0, "started": t0,
               "finished": t0 + float(rep.get("wall_clock_s") or 0), "report": rep}
        json.dump(server._public(job), open(os.path.join(api_dir, "jobs", jid + ".json"), "w", encoding="utf8"))

    print(f"static site written to {out}")


if __name__ == "__main__":
    main()
