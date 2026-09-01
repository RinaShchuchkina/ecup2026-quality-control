#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")
MAX_IMGS = 5


def data_dir():
    return Path(os.environ.get("ECUP_DATA", "data"))


def out_dir():
    return Path(os.environ.get("ECUP_OUT", "runs"))


def load(job):
    i, p, s = job
    try:
        im = Image.open(p).convert("RGB")
        w, h = im.size
        r = s / max(w, h)
        nw, nh = max(1, int(round(w * r))), max(1, int(round(h * r)))
        im = im.resize((nw, nh), Image.BICUBIC)
        cv = Image.new("RGB", (s, s), (255, 255, 255))
        cv.paste(im, ((s - nw) // 2, (s - nh) // 2))
        return i, np.asarray(cv, dtype=np.uint8)
    except Exception:
        return i, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(data_dir() / "images"))
    ap.add_argument("--labels", default=str(out_dir() / "labels_folds.csv"))
    ap.add_argument("--out", default=str(out_dir() / "cache"))
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("ECUP_NPROC", "8")))
    args = ap.parse_args()

    img_root = Path(args.images)
    out = Path(args.out)
    s = args.size
    out.mkdir(parents=True, exist_ok=True)

    lab = pd.read_csv(args.labels)
    rows = []
    for pid in lab["id"].to_numpy():
        d = img_root / str(pid)
        if not d.is_dir():
            continue
        fs = sorted([f for f in os.listdir(d) if f.lower().endswith(IMG_EXTS)],
                    key=lambda x: (len(x), x))
        for f in fs[:MAX_IMGS]:
            rows.append((int(pid), str(d / f)))
    idx = pd.DataFrame(rows, columns=["id", "path"])
    print("images:", len(idx), "products with >=1 img:", idx["id"].nunique(), flush=True)

    mm = np.lib.format.open_memmap(str(out / f"imgs_{s}.npy"), mode="w+",
                                   dtype=np.uint8, shape=(len(idx), s, s, 3))
    jobs = [(i, p, s) for i, p in enumerate(idx["path"])]
    bad = 0
    with Pool(args.workers) as pool:
        for k, (i, arr) in enumerate(pool.imap_unordered(load, jobs, chunksize=64)):
            if arr is None:
                bad += 1
                arr = np.full((s, s, 3), 255, np.uint8)
            mm[i] = arr
            if k % 5000 == 0:
                print(k, flush=True)
    mm.flush()
    idx.to_csv(out / f"index_{s}.csv", index=False)
    print("done, bad:", bad, "shape", mm.shape, flush=True)


if __name__ == "__main__":
    sys.exit(main())
