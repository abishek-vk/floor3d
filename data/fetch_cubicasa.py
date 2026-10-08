"""Fetch a subset of CubiCasa5K straight out of the 5.4 GB Zenodo zip using HTTP
range requests, so we never download the whole archive.

    python data/fetch_cubicasa.py --split test --n 30 --out data/cubicasa5k

Writes <out>/<folder>/{F1_scaled.png, F1_original.png, model.svg} plus the
split .txt files. Deterministic: a seeded random sample (seed 0) of N folders from the split.
"""
import argparse
import io
import os
import random
import sys
import urllib.request
import zipfile

URL = "https://zenodo.org/records/2613548/files/cubicasa5k.zip?download=1"
WANTED = ("F1_scaled.png", "F1_original.png", "model.svg")


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests, with a small block cache."""

    BLOCK = 1 << 20

    def __init__(self, url):
        self.url = url
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as r:
            self.url = r.url  # follow redirects once
            self.size = int(r.headers["Content-Length"])
        self.pos = 0
        self.cache = {}

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def _fetch(self, start, end):
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end - 1}"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    return r.read()
            except Exception as e:  # flaky network: retry
                print(f"  retry {attempt}: {e}", file=sys.stderr)
        raise IOError(f"range fetch failed {start}-{end}")

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        # Large reads go straight through; small ones use the block cache.
        if n > self.BLOCK:
            data = self._fetch(self.pos, self.pos + n)
        else:
            out = bytearray()
            p, end = self.pos, self.pos + n
            while p < end:
                b = p // self.BLOCK
                if b not in self.cache:
                    s = b * self.BLOCK
                    self.cache[b] = self._fetch(s, min(s + self.BLOCK, self.size))
                blk = self.cache[b]
                o = p - b * self.BLOCK
                take = min(end - p, len(blk) - o)
                out += blk[o:o + take]
                p += take
            data = bytes(out)
        self.pos += len(data)
        return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--out", default="data/cubicasa5k")
    args = ap.parse_args()

    f = HttpRangeFile(URL)
    z = zipfile.ZipFile(f)
    names = z.namelist()
    root = names[0].split("/")[0]
    os.makedirs(args.out, exist_ok=True)
    for s in ("train", "val", "test"):
        p = f"{root}/{s}.txt"
        if p in names:
            with open(os.path.join(args.out, f"{s}.txt"), "wb") as fh:
                fh.write(z.read(p))
    folders = [l.strip() for l in open(os.path.join(args.out, f"{args.split}.txt")) if l.strip()]
    random.Random(0).shuffle(folders := sorted(folders))
    folders = folders[args.offset:args.offset + args.n]
    for i, fol in enumerate(folders):
        dst = os.path.join(args.out, fol.strip("/"))
        os.makedirs(dst, exist_ok=True)
        for w in WANTED:
            src = f"{root}{fol}{w}"
            out = os.path.join(dst, w)
            if os.path.exists(out) or src not in names:
                continue
            with open(out, "wb") as fh:
                fh.write(z.read(src))
        print(f"[{i + 1}/{len(folders)}] {fol}")


if __name__ == "__main__":
    main()
