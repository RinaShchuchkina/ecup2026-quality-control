#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

BAD, LV = "БАД", "Легковоспламеняющиеся"
CATMAP = {"BAD": BAD, "LV": LV}
CAT_TAG = {BAD: "BAD", LV: "LV"}
GRAY = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)


def out_dir():
    return Path(os.environ.get("ECUP_OUT", "runs"))


def weights_dir():
    return Path(os.environ.get("ECUP_WEIGHTS", "weights"))


class RawDS(Dataset):
    def __init__(self, arr, rows, labels, jpeg_p=0.0, jpeg_q=(40, 95)):
        self.mm, self.rows, self.labels = arr, rows, labels
        self.jpeg_p, self.jpeg_q = jpeg_p, jpeg_q

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        a = self.mm[self.rows[i]]
        if self.jpeg_p > 0 and np.random.rand() < self.jpeg_p:
            q = int(np.random.randint(self.jpeg_q[0], self.jpeg_q[1] + 1))
            buf = io.BytesIO()
            Image.fromarray(a).save(buf, format="JPEG", quality=q)
            buf.seek(0)
            a = np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)
        else:
            a = np.ascontiguousarray(a)
        return torch.from_numpy(a), np.float32(self.labels[i])


def gpu_batch(xb, size, train, dev, mean, std):
    x = xb.to(dev, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.0)
    B = x.shape[0]
    if train:
        s = torch.empty(B, device=dev).uniform_(0.65, 1.0).sqrt()
        r = torch.empty(B, device=dev).uniform_(np.log(0.85), np.log(1.18)).exp().sqrt()
        sx, sy = (s * r).clamp(max=1.0), (s / r).clamp(max=1.0)
        tx = (1 - sx) * (torch.rand(B, device=dev) * 2 - 1)
        ty = (1 - sy) * (torch.rand(B, device=dev) * 2 - 1)
        flip = torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0)
        th = torch.zeros(B, 2, 3, device=dev)
        th[:, 0, 0] = sx * flip
        th[:, 0, 2] = tx
        th[:, 1, 1] = sy
        th[:, 1, 2] = ty
        grid = F.affine_grid(th, (B, 3, size, size), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
        g = GRAY.to(dev)
        br = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.85, 1.15)
        x = x * br
        m = (x * g).sum(1, keepdim=True)
        mm_ = m.mean(dim=(2, 3), keepdim=True)
        ct = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.85, 1.15)
        x = mm_ + (x - mm_) * ct
        st = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.85, 1.15)
        x = m + (x - m) * st
        x = x.clamp_(0, 1)
    else:
        if x.shape[-1] != size:
            x = F.interpolate(x, size=(size, size), mode="bicubic",
                              align_corners=False).clamp_(0, 1)
    return ((x - mean) / std).contiguous(memory_format=torch.channels_last)


def build(args):
    import timm
    kw = dict(pretrained=True, num_classes=1, drop_rate=args.drop, drop_path_rate=args.drop_path)
    try:
        m = timm.create_model(args.model, img_size=args.size, **kw)
    except TypeError:
        m = timm.create_model(args.model, **kw)
    cfg = m.pretrained_cfg
    mean = torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406))).view(1, 3, 1, 1)
    std = torch.tensor(cfg.get("std", (0.229, 0.224, 0.225))).view(1, 3, 1, 1)
    if args.grad_ckpt:
        try:
            m.set_grad_checkpointing(True)
        except Exception as e:
            print("grad_ckpt fail", e, flush=True)
    return m, mean, std


def make_opt(m, args):
    from timm.optim import create_optimizer_v2
    return create_optimizer_v2(m, opt="adamw", lr=args.lr, weight_decay=args.wd,
                               layer_decay=args.llrd if args.llrd > 0 else None)


def run_fold(args, cat, fold, idx, lab, ARR, dev, oof_dir, wdir):
    tag = CAT_TAG[cat]
    fp = oof_dir / f"oof_{args.tag}_{tag}_f{fold}.csv"
    if fp.exists():
        o = pd.read_csv(fp)
        print(f"  skip {tag} f{fold}", flush=True)
        return o, float(o["auc"].iloc[0])
    prod = lab[lab.category == cat]
    tr_pid = set(prod.loc[prod.fold_family != fold, "id"])
    va_pid = set(prod.loc[prod.fold_family == fold, "id"])
    y_by_pid = dict(zip(prod["id"], prod["label"]))
    sub = idx[idx["id"].isin(set(prod["id"]))]
    rows = sub.index.to_numpy()
    pids = sub["id"].to_numpy()
    ys = np.array([y_by_pid[p] for p in pids], np.float32)
    tr = np.array([i for i, p in enumerate(pids) if p in tr_pid])
    va = np.array([i for i, p in enumerate(pids) if p in va_pid])
    tr_rows, tr_y = rows[tr], ys[tr]
    if args.oversample > 1:
        pos = np.where(tr_y == 1)[0]
        rep = np.concatenate([np.arange(len(tr_rows))] + [pos] * (args.oversample - 1))
        tr_rows, tr_y = tr_rows[rep], tr_y[rep]
    print(f"  {tag} f{fold}: train img {len(tr_rows)} (pos {int(tr_y.sum())}), "
          f"val img {len(va)} / товаров {len(va_pid)}", flush=True)
    ltr = DataLoader(RawDS(ARR, tr_rows, tr_y, args.jpeg_p), batch_size=args.bs, shuffle=True,
                     num_workers=args.workers, pin_memory=True, drop_last=True,
                     persistent_workers=args.workers > 0,
                     prefetch_factor=4 if args.workers else None)
    lva = DataLoader(RawDS(ARR, rows[va], ys[va]), batch_size=max(1, args.bs), shuffle=False,
                     num_workers=max(2, args.workers // 2), pin_memory=True)
    m, mean, std = build(args)
    m = m.to(dev).to(memory_format=torch.channels_last)
    mean, std = mean.to(dev), std.to(dev)
    opt = make_opt(m, args)
    steps = max(1, len(ltr) // args.accum) * args.epochs
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in opt.param_groups],
                                              total_steps=steps, pct_start=args.warm)
    lossf = nn.BCEWithLogitsLoss()
    allp = {}
    va_pids = pids[va]
    last_auc = 0.5
    for ep in range(args.epochs):
        m.train()
        t0 = time.time()
        tot = 0.0
        nb = 0
        k = 0
        opt.zero_grad(set_to_none=True)
        for xb, y in ltr:
            x = gpu_batch(xb, args.size, True, dev, mean, std)
            y = y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = lossf(m(x).squeeze(1), y)
            (loss / args.accum).backward()
            k += 1
            if k % args.accum == 0:
                if args.clip > 0:
                    torch.nn.utils.clip_grad_norm_(m.parameters(), args.clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                if sch.last_epoch < steps - 1:
                    sch.step()
            tot += loss.item()
            nb += 1
        m.eval()
        ps = []
        with torch.no_grad():
            for xb, _ in lva:
                x = gpu_batch(xb, args.size, False, dev, mean, std)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    ps.append(torch.sigmoid(m(x).squeeze(1)).float().cpu().numpy())
        p = np.concatenate(ps)
        dfp = pd.DataFrame({"id": va_pids, "p": p, "y": ys[va]})
        g = dfp.groupby("id").agg(p=("p", "mean"), y=("y", "first"))
        auc = roc_auc_score(g.y, g.p) if g.y.nunique() > 1 else 0.5
        ap = average_precision_score(g.y, g.p) if g.y.nunique() > 1 else 0.0
        print(f"  [{tag} f{fold}] ep{ep} loss {tot/max(nb,1):.4f} AUC {auc:.4f} AP {ap:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        allp[f"p_ep{ep}"] = p.copy()
        last_auc = auc
        if ep == args.epochs - 1 and args.save_w:
            from safetensors.torch import save_file
            sd = {kk: vv.detach().cpu().to(torch.float16).contiguous()
                  for kk, vv in m.state_dict().items()}
            wdir.mkdir(parents=True, exist_ok=True)
            save_file(sd, str(wdir / f"{args.tag}_{tag}_f{fold}.safetensors"))
    o = pd.DataFrame({"id": va_pids, "p_img": allp[f"p_ep{args.epochs-1}"],
                      "fold": fold, "cat": cat, "auc": last_auc})
    for kk, vv in allp.items():
        o[kk] = vv
    o.to_csv(fp, index=False)
    del m, opt
    torch.cuda.empty_cache()
    return o, last_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="eva02_base_patch14_448.mim_in22k_ft_in22k_in1k")
    ap.add_argument("--tag", default="eva224")
    ap.add_argument("--cats", default="BAD,LV")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--llrd", type=float, default=0.75)
    ap.add_argument("--warm", type=float, default=0.25)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--drop", type=float, default=0.0)
    ap.add_argument("--drop_path", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--oversample", type=int, default=1)
    ap.add_argument("--oversample_lv", type=int, default=8)
    ap.add_argument("--jpeg_p", type=float, default=0.5)
    ap.add_argument("--grad_ckpt", type=int, default=0)
    ap.add_argument("--save_w", type=int, default=1)
    ap.add_argument("--cache", default=str(out_dir() / "cache"))
    ap.add_argument("--labels", default=str(out_dir() / "labels_folds.csv"))
    ap.add_argument("--oof", default=str(out_dir() / "oof_vision"))
    ap.add_argument("--weights", default=str(weights_dir() / "eva"))
    ap.add_argument("--npy", default="")
    ap.add_argument("--index", default="")
    ap.add_argument("--mmap", type=int, default=1)
    args = ap.parse_args()

    dev = "cuda"
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    print("GPU", torch.cuda.get_device_name(), flush=True)

    cache = Path(args.cache)
    oof_dir = Path(args.oof)
    wdir = Path(args.weights)
    oof_dir.mkdir(parents=True, exist_ok=True)

    mmpath = args.npy or str(cache / "imgs_256.npy")
    t0 = time.time()
    ARR = np.load(mmpath, mmap_mode="r" if args.mmap else None)
    print(f"кэш {mmpath} {ARR.shape} за {time.time()-t0:.0f}с", flush=True)
    idxname = args.index or str(cache / f"index_{ARR.shape[1]}.csv")
    idx = pd.read_csv(idxname)
    lab = pd.read_csv(args.labels)

    meta = []
    for ckey in args.cats.split(","):
        cat = CATMAP[ckey]
        args.oversample = args.oversample_lv if cat == LV else 1
        for fold in [int(f) for f in args.folds.split(",")]:
            t1 = time.time()
            o, auc = run_fold(args, cat, fold, idx, lab, ARR, dev, oof_dir, wdir)
            meta.append(dict(cat=CAT_TAG[cat], fold=fold, auc=auc,
                             secs=round(time.time() - t1, 1)))
            print(f"FOLD DONE {CAT_TAG[cat]} {fold} AUC {auc:.4f} {time.time()-t1:.0f}s",
                  flush=True)
    with open(oof_dir / f"meta_{args.tag}_{args.cats.replace(',', '')}.json", "w") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
