#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import hashlib
import json
import os
import pickle
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prompts import RULES_BAD, RULES_LV

BAD = "БАД"
LV = "Легковоспламеняющиеся"
TAG_RE = re.compile(r"<[^>]+>")
NONWORD_RE = re.compile(r"[^a-zа-яё0-9]+")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def data_dir():
    return Path(os.environ.get("ECUP_DATA", "data"))


def out_dir():
    return Path(os.environ.get("ECUP_OUT", "runs"))


def norm_text(name, desc):
    t = f"{name} {desc}".lower()
    t = TAG_RE.sub(" ", t)
    t = NONWORD_RE.sub(" ", t)
    return " ".join(t.split())


def product_images(images_dir, pid):
    d = images_dir / str(pid)
    if not d.is_dir():
        return []
    fs = sorted([f for f in os.listdir(d) if f.lower().endswith(IMG_EXTS)])
    return [str(d / f) for f in fs]


def build_imghash(ids, images_dir, cache_path):
    if cache_path.exists():
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)
    imghash = {}
    for i, pid in enumerate(ids):
        d = images_dir / str(pid)
        hs = []
        if d.is_dir():
            for f in sorted(os.listdir(d)):
                try:
                    with open(d / f, "rb") as fh:
                        hs.append(hashlib.md5(fh.read()).hexdigest())
                except OSError:
                    continue
        imghash[pid] = hs
        if i % 2000 == 0:
            print(f"  hashing {i}/{len(ids)}", flush=True)
    with open(cache_path, "wb") as fh:
        pickle.dump(imghash, fh)
    return imghash


def build_families(df, imghash):
    parent = list(range(len(df)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    ids = df["id"].tolist()
    by_text, by_img = {}, {}
    for row, (pid, tk) in enumerate(zip(ids, df["text_key"])):
        if tk:
            by_text.setdefault(tk, []).append(row)
        for h in imghash.get(pid, []):
            by_img.setdefault(h, []).append(row)
    for rows in list(by_text.values()) + list(by_img.values()):
        for r in rows[1:]:
            union(rows[0], r)
    return [find(r) for r in range(len(df))]


def assign_folds(df, n_folds, seed):
    df["fold_family"] = -1
    df["fold_iid"] = -1
    for _, sub in df.groupby("category"):
        idx = sub.index
        y = sub["label"].values
        sgkf = StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed)
        for k, (_, va) in enumerate(sgkf.split(idx, y, groups=sub["family_id"].values)):
            df.loc[idx[va], "fold_family"] = k
        skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
        for k, (_, va) in enumerate(skf.split(idx, y)):
            df.loc[idx[va], "fold_iid"] = k
    return df


def write_sft(df, images_dir, path, fold_col, desc_chars, max_imgs):
    n_img = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r in df.itertuples(index=False):
            imgs = product_images(images_dir, r.id)[:max_imgs]
            n_img += len(imgs)
            rec = {
                "id": int(r.id),
                "category": r.category,
                "label": int(r.label),
                "fold": int(getattr(r, fold_col)),
                "family_id": int(r.family_id),
                "rules": RULES_BAD if r.category == BAD else RULES_LV,
                "text": f"Название: {r.name}\nОписание: {str(r.description)[:desc_chars]}",
                "images": imgs,
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return n_img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-csv", default=str(data_dir() / "data.csv"))
    ap.add_argument("--images", default=str(data_dir() / "images"))
    ap.add_argument("--out", default=str(out_dir()))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fold-column", default="fold_family",
                    choices=["fold_family", "fold_iid"])
    ap.add_argument("--desc-chars", type=int, default=2500)
    ap.add_argument("--max-imgs", type=int, default=5)
    args = ap.parse_args()

    images_dir = Path(args.images)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data_csv, index_col=0)
    df["name"] = df["name"].fillna("")
    df["description"] = df["description"].fillna("")
    df["text_key"] = [norm_text(n, d) for n, d in zip(df["name"], df["description"])]
    print(f"строк: {len(df)}, категорий: {df['category'].nunique()}", flush=True)

    imghash = build_imghash(df["id"].tolist(), images_dir, out / "imghash.pkl")
    all_h = [h for v in imghash.values() for h in v]
    print(f"изображений: {len(all_h)}, уникальных md5: {len(set(all_h))}", flush=True)

    df["family_id"] = build_families(df, imghash)
    fam_sizes = df.groupby("family_id").size()
    multi = (df.groupby("family_id")["id"].transform("size") > 1).sum()
    print(f"семейств: {len(fam_sizes)}, товаров в семьях >1: {multi}", flush=True)

    df = assign_folds(df, args.folds, args.seed)
    for cat, sub in df.groupby("category"):
        print(f"{cat}: позитивов по фолдам {sub.groupby('fold_family')['label'].sum().tolist()}",
              flush=True)

    fam = df[["id", "name", "description", "category", "label", "text_key",
              "family_id", "fold_family", "fold_iid"]]
    with open(out / "families.pkl", "wb") as fh:
        pickle.dump(fam, fh, protocol=4)
    print("saved:", out / "families.pkl", flush=True)

    lab = df[["id", "category", "label", "fold_family", "family_id"]]
    lab.to_csv(out / "labels_folds.csv", index=False)
    print("saved:", out / "labels_folds.csv", flush=True)

    n_img = write_sft(df, images_dir, out / "sft_data.jsonl",
                      args.fold_column, args.desc_chars, args.max_imgs)
    print(f"saved: {out / 'sft_data.jsonl'} (строк {len(df)}, картинок {n_img})", flush=True)


if __name__ == "__main__":
    sys.exit(main())
