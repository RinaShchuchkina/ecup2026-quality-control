# -*- coding: utf-8 -*-
import math
import os
import re
import unicodedata

import numpy as np
from pathlib import Path

TAG_RE = re.compile(r"<[^>]+>")
NONWORD_RE = re.compile(r"[^a-zа-яё0-9]+")
CAT_TAG = {"БАД": "категория бад",
           "Легковоспламеняющиеся": "категория легковоспламеняющиеся"}


def _norm(s):
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).lower().replace("ё", "е")
    s = TAG_RE.sub(" ", s)
    s = NONWORD_RE.sub(" ", s)
    return " ".join(s.split())


def _encode(tok, cat, name, desc, max_len):
    pre = tok(f"{CAT_TAG.get(cat, cat)} | название: {_norm(name)} | описание:",
              add_special_tokens=False)["input_ids"][:288]
    budget = max_len - 2 - len(pre)
    d = tok(_norm(desc), add_special_tokens=False)["input_ids"]
    if len(d) > budget:
        h = budget // 2
        d = d[:h] + d[-(budget - h):]
    return [tok.cls_token_id] + pre + d + [tok.sep_token_id]


def encoder_branch(test, model_dir, max_len=1024, bs=64, device="cuda"):
    root = Path(str(model_dir))
    subs = sorted(d for d in root.iterdir()
                  if d.is_dir() and (d / "config.json").exists()) if root.is_dir() else []
    if not (root / "config.json").exists() and subs:
        acc = None
        for d in subs:
            v = _one_model(test, d, max_len, bs, device)
            acc = v if acc is None else acc + v
        return acc / len(subs)
    return _one_model(test, root, max_len, bs, device)


def _one_model(test, model_dir, max_len=1024, bs=64, device="cuda"):
    import torch
    from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

    model_dir = str(model_dir)
    tok = AutoTokenizer.from_pretrained(model_dir)
    cfg = AutoConfig.from_pretrained(model_dir)
    for k in ("reference_compile", "compile_model"):
        if hasattr(cfg, k):
            setattr(cfg, k, False)
    lim = int(getattr(cfg, "max_position_embeddings", 0) or 0)
    if lim and lim < 4096:
        max_len = min(max_len, lim - 4)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, config=cfg, attn_implementation="sdpa")
    use_cuda = device == "cuda" and torch.cuda.is_available()
    model = model.eval()
    amp_dtype = torch.float32
    if use_cuda:
        model = model.cuda()
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = model.float()

    enc = [_encode(tok, c, n, d, max_len) for c, n, d in
           zip(test["category"], test["name"], test["description"])]
    lens = np.array([len(e) for e in enc])
    order = np.argsort(lens, kind="stable")
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    out = np.full(len(enc), np.nan, dtype=np.float64)

    SPREAD = 48
    batches, cur = [], []
    for i in order:
        if cur and (len(cur) >= bs or lens[i] - lens[cur[0]] > SPREAD):
            batches.append(cur); cur = []
        cur.append(i)
    if cur:
        batches.append(cur)

    def run(sel, mult=8):
        n = int(math.ceil(max(lens[i] for i in sel) / mult) * mult)
        ids = np.full((len(sel), n), pad, dtype=np.int64)
        am = np.zeros((len(sel), n), dtype=np.int64)
        for j, i in enumerate(sel):
            ids[j, : lens[i]] = enc[i]
            am[j, : lens[i]] = 1
        t_ids = torch.from_numpy(ids); t_am = torch.from_numpy(am)
        if use_cuda:
            t_ids, t_am = t_ids.cuda(), t_am.cuda()
            with torch.autocast("cuda", dtype=amp_dtype):
                lg = model(input_ids=t_ids, attention_mask=t_am).logits[:, 0]
        else:
            lg = model(input_ids=t_ids, attention_mask=t_am).logits[:, 0]
        return torch.sigmoid(lg.float()).cpu().numpy()

    with torch.no_grad():
        for sel in batches:
            v = run(sel)
            bad = ~np.isfinite(v)
            if bad.any():
                for j in np.where(bad)[0]:
                    v[j] = run([sel[j]], mult=1)[0]
            out[sel] = v
    if not np.isfinite(out).all():
        raise RuntimeError(f"encoder produced {int((~np.isfinite(out)).sum())} non-finite probs")
    del model
    if use_cuda:
        torch.cuda.empty_cache()
    return out
