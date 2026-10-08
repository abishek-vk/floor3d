"""Local web UI for floorplan3d: pick a demo blueprint or upload your own, get a 3D model.

    python app/server.py            # then open http://127.0.0.1:8000
    python app/server.py --port 9000

Standard library only. One analysis runs at a time (the model stays loaded between
runs, so later jobs are faster). Results are kept in app_data/jobs/<job id>/.
"""
from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import os
import re
import sys
import threading
import time
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

APP = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(APP)
sys.path.insert(0, ROOT)

STATIC = os.path.join(APP, "static")
DEMOS = os.path.join(APP, "demos")
DATA = os.path.join(ROOT, "app_data")
JOBS_DIR = os.path.join(DATA, "jobs")

MAX_BYTES = 25 * 1024 * 1024
# extension -> magic-byte prefixes; the content must match, not just the name
FORMATS = {
    ".png": [b"\x89PNG\r\n\x1a\n"],
    ".jpg": [b"\xff\xd8\xff"],
    ".jpeg": [b"\xff\xd8\xff"],
    ".pdf": [b"%PDF-"],
}
STAGES = ["queued", "ingest", "detect", "vectorize", "extrude", "viewer", "done"]
STAGE_LABEL = {
    "queued": "Waiting to start",
    "ingest": "Reading and cleaning the image",
    "detect": "Detecting walls, rooms, doors and windows",
    "vectorize": "Vectorising walls, building rooms, reading dimensions, solving for metric scale",
    "extrude": "Building the 3D model",
    "viewer": "Preparing the 3D viewer",
    "done": "Done",
}

log = logging.getLogger("fp3d.app")
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
RUN_LOCK = threading.Lock()  # the pipeline is CPU-heavy: one job at a time


def new_job(source: str, name: str, input_path: str, wall_height: float) -> str:
    jid = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[jid] = {"id": jid, "source": source, "name": name, "input": input_path,
                     "wall_height": wall_height, "stage": "queued", "status": "queued",
                     "error": None, "created": time.time(), "report": None}
    threading.Thread(target=_work, args=(jid,), daemon=True).start()
    return jid


def _set(jid: str, **kw) -> None:
    with JOBS_LOCK:
        JOBS[jid].update(kw)


def _work(jid: str) -> None:
    from run import run
    job = JOBS[jid]
    out = os.path.join(JOBS_DIR, jid)
    with RUN_LOCK:
        _set(jid, status="running", stage="ingest", started=time.time())
        try:
            rep = run(job["input"], out, "full", wall_height=job["wall_height"],
                      progress=lambda s: _set(jid, stage=s))
            _set(jid, status="done", stage="done", report=rep, finished=time.time())
        except Exception as e:
            log.error("job %s failed: %s", jid, e)
            _set(jid, status="error", error=str(e), trace=traceback.format_exc(), finished=time.time())


def _public(job: dict) -> dict:
    j = {k: job.get(k) for k in ("id", "source", "name", "stage", "status", "error", "wall_height")}
    j["stage_label"] = STAGE_LABEL.get(job["stage"], job["stage"])
    j["stage_index"] = STAGES.index(job["stage"]) if job["stage"] in STAGES else 0
    j["n_stages"] = len(STAGES) - 1
    t0 = job.get("started") or job["created"]
    j["elapsed_s"] = round((job.get("finished") or time.time()) - t0, 1)
    if job["status"] == "done":
        base = f"/jobs/{job['id']}/"
        j["files"] = {"viewer": base + "viewer.html", "glb": base + "model.glb", "obj": base + "model.obj",
                      "mtl": base + "material.mtl", "layout": base + "layout.json",
                      "report": base + "run_report.json"}
        dbg = os.path.join(JOBS_DIR, job["id"], "debug")
        j["debug"] = [base + "debug/" + f for f in sorted(os.listdir(dbg))] if os.path.isdir(dbg) else []
        j["report"] = job["report"]
    return j


def _safe_join(base: str, rel: str) -> str | None:
    p = os.path.realpath(os.path.join(base, rel))
    return p if p.startswith(os.path.realpath(base) + os.sep) and os.path.isfile(p) else None


class Handler(BaseHTTPRequestHandler):
    server_version = "floorplan3d"

    def log_message(self, fmt, *args):  # quieter console
        log.info("%s %s", self.address_string(), fmt % args)

    # ---------------------------------------------------------------- helpers
    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: str | None):
        if path is None:
            return self._json({"error": "not found"}, 404)
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        if path.endswith(".glb"):
            ctype = "model/gltf-binary"
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        if path.endswith((".glb", ".obj", ".mtl", ".json")) and "download=1" in (self.path or ""):
            self.send_header("Content-Disposition", f'attachment; filename="{os.path.basename(path)}"')
        self.end_headers()
        self.wfile.write(data)

    # ---------------------------------------------------------------- routes
    def do_GET(self):
        u = urlparse(self.path)
        p = unquote(u.path)
        if p in ("/", "/index.html"):
            return self._file(os.path.join(STATIC, "index.html"))
        if p == "/api/demos":
            return self._json(json.load(open(os.path.join(DEMOS, "demos.json"), encoding="utf8")))
        if p == "/api/formats":
            return self._json({"accept": sorted(FORMATS), "max_mb": MAX_BYTES // (1024 * 1024)})
        m = re.fullmatch(r"/api/jobs/([0-9a-f]{12})", p)
        if m:
            job = JOBS.get(m.group(1))
            return self._json(_public(job)) if job else self._json({"error": "unknown job"}, 404)
        if p.startswith("/demos/"):
            return self._file(_safe_join(DEMOS, p[len("/demos/"):]))
        m = re.fullmatch(r"/jobs/([0-9a-f]{12})/(.+)", p)
        if m:
            return self._file(_safe_join(os.path.join(JOBS_DIR, m.group(1)), m.group(2)))
        if p.startswith("/static/"):
            return self._file(_safe_join(STATIC, p[len("/static/"):]))
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        p = urlparse(self.path).path
        try:
            wall_h = float(self.headers.get("X-Wall-Height", "2.7"))
        except ValueError:
            wall_h = 2.7
        wall_h = min(max(wall_h, 2.0), 6.0)

        if p == "/api/analyze/demo":
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n) or b"{}")
            demos = {d["id"]: d for d in json.load(open(os.path.join(DEMOS, "demos.json"), encoding="utf8"))}
            d = demos.get(body.get("id"))
            if d is None:
                return self._json({"error": "unknown demo"}, 400)
            jid = new_job("demo", d["title"], os.path.join(DEMOS, d["id"] + ".png"), wall_h)
            return self._json({"job": jid})

        if p == "/api/analyze/upload":
            n = int(self.headers.get("Content-Length", "0"))
            name = unquote(self.headers.get("X-Filename", "upload"))
            ext = os.path.splitext(name)[1].lower()
            if ext not in FORMATS:
                return self._json({"error": f"Unsupported file type '{ext or '?'}'. "
                                            f"Use PNG, JPG/JPEG or PDF."}, 400)
            if n <= 0:
                return self._json({"error": "The file is empty."}, 400)
            if n > MAX_BYTES:
                return self._json({"error": f"File is {n / 1048576:.1f} MB; the limit is "
                                            f"{MAX_BYTES // 1048576} MB."}, 413)
            data = self.rfile.read(n)
            if not any(data.startswith(m) for m in FORMATS[ext]):
                return self._json({"error": f"The file is named {ext} but its contents are not a "
                                            f"valid {ext[1:].upper()} file."}, 400)
            up = os.path.join(DATA, "uploads", uuid.uuid4().hex[:12])
            os.makedirs(up, exist_ok=True)
            path = os.path.join(up, "input" + ext)
            with open(path, "wb") as f:
                f.write(data)
            safe_name = re.sub(r"[^\w.\- ]", "_", os.path.basename(name))[:80]
            jid = new_job("upload", safe_name, path, wall_h)
            return self._json({"job": jid})

        return self._json({"error": "not found"}, 404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    os.makedirs(JOBS_DIR, exist_ok=True)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"floorplan3d UI running at http://{a.host}:{a.port}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
