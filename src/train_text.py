#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analyzers import char_wb_ngrams, rule_flags, transform_tfidf, word_ngrams

BAD = "БАД"
LV = "Легковоспламеняющиеся"
TAG_RE = re.compile(r"<[^>]+>")
NONWORD_RE = re.compile(r"[^a-zа-яё0-9]+")
DIGIT_RE = re.compile(r"\d+")
MAX_FEATURES_W = 60000
MAX_FEATURES_C = 100000
C_REG = 4
MAX_ITER = 2000
RULE_SCALE = 2.0
THR_GRID = np.arange(0.05, 0.95, 0.01)


def data_dir():
    return Path(os.environ.get("ECUP_DATA", "data"))


def out_dir():
    return Path(os.environ.get("ECUP_OUT", "runs"))


def weights_dir():
    return Path(os.environ.get("ECUP_WEIGHTS", "weights"))


def norm_text(name, desc):
    t = f"{name} {desc}".lower()
    t = TAG_RE.sub(" ", t)
    t = NONWORD_RE.sub(" ", t)
    return " ".join(t.split())


def key_nd(name):
    return " ".join(DIGIT_RE.sub(" ", norm_text(name, "")).split())


def key_w5(name):
    return " ".join(norm_text(name, "").split()[:5])


def make_vecs():
    return (TfidfVectorizer(max_features=MAX_FEATURES_W, sublinear_tf=True,
                            analyzer=word_ngrams),
            TfidfVectorizer(max_features=MAX_FEATURES_C, sublinear_tf=True,
                            analyzer=char_wb_ngrams))


def build_X(texts, vec_w, vec_c, fit=False):
    rules = csr_matrix(np.vstack([rule_flags(t) for t in texts])) * RULE_SCALE
    if fit:
        return hstack([vec_w.fit_transform(texts), vec_c.fit_transform(texts), rules]).tocsr()
    return hstack([vec_w.transform(texts), vec_c.transform(texts), rules]).tocsr()


def dhash_list(images_dir, pid):
    from PIL import Image
    out = []
    d = images_dir / str(pid)
    if not d.is_dir():
        return out
    for f in sorted(os.listdir(d))[:5]:
        try:
            im = Image.open(d / f).convert("L").resize((9, 8), Image.BILINEAR)
        except Exception:
            continue
        a = np.asarray(im, dtype=np.int16)
        bits = (a[:, 1:] > a[:, :-1]).flatten()
        v = 0
        for b in bits:
            v = (v << 1) | int(b)
        out.append(v)
    return out


def build_dhash_votes(df, images_dir):
    per_cat = {}
    for cat, sub in df.groupby("category"):
        agg = defaultdict(lambda: [0, 0])
        for n, (pid, y) in enumerate(zip(sub["id"], sub["label"])):
            for hv in dhash_list(images_dir, pid):
                rec = agg[int(hv)]
                rec[0] += 1
                rec[1] += int(y)
            if n % 2000 == 0:
                print(f"  dhash {cat} {n}/{len(sub)}", flush=True)
        per_cat[cat] = {k: (v[0], v[1]) for k, v in agg.items()}
        print(f"  dhash {cat}: хэшей {len(per_cat[cat])}", flush=True)
    return per_cat


def oof_threshold(sub, texts, y, folds, imghash):
    prob = np.zeros(len(sub))
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        vw, vc = make_vecs()
        Xtr = build_X([texts[i] for i in np.where(tr)[0]], vw, vc, fit=True)
        Xva = build_X([texts[i] for i in np.where(va)[0]], vw, vc)
        clf = LogisticRegression(max_iter=MAX_ITER, C=C_REG, class_weight="balanced")
        clf.fit(Xtr, y[tr])
        prob[va] = clf.predict_proba(Xva)[:, 1]
    ids, tks = sub["id"].values, sub["text_key"].values
    p_blend = prob.copy()
    for k in np.unique(folds):
        bt, bi = {}, {}
        for i in np.where(folds != k)[0]:
            if tks[i]:
                bt.setdefault(tks[i], []).append(y[i])
            for h in imghash.get(ids[i], []):
                bi.setdefault(h, []).append(y[i])
        for i in np.where(folds == k)[0]:
            votes = list(bt.get(tks[i], []))
            for h in imghash.get(ids[i], []):
                votes.extend(bi.get(h, []))
            if votes:
                v = float(np.mean(votes))
                p_blend[i] = v if v in (0.0, 1.0) else 0.5 * v + 0.5 * prob[i]
    f1s = [f1_score(y, p_blend > t) for t in THR_GRID]
    return float(THR_GRID[int(np.argmax(f1s))]), float(max(f1s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default=str(out_dir() / "families.pkl"))
    ap.add_argument("--imghash", default=str(out_dir() / "imghash.pkl"))
    ap.add_argument("--images", default=str(data_dir() / "images"))
    ap.add_argument("--out", default=str(weights_dir() / "model_v1.pkl"))
    ap.add_argument("--fold-column", default="fold_iid",
                    choices=["fold_iid", "fold_family"])
    ap.add_argument("--no-dhash", action="store_true")
    args = ap.parse_args()

    images_dir = Path(args.images)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.families, "rb") as fh:
        df = pickle.load(fh)
    with open(args.imghash, "rb") as fh:
        imghash = pickle.load(fh)

    export = {"categories": {}, "text_index": {}, "img_index": {},
              "text_index_nd": {}, "text_index_w5": {}}

    ti, ii, tnd, tw5 = {}, {}, {}, {}
    for pid, nm, tk, lab in zip(df["id"], df["name"], df["text_key"], df["label"]):
        if tk:
            ti.setdefault(tk, []).append(int(lab))
        knd, kw5 = key_nd(nm), key_w5(nm)
        if knd:
            tnd.setdefault(knd, []).append(int(lab))
        if kw5:
            tw5.setdefault(kw5, []).append(int(lab))
        for h in imghash.get(pid, []):
            ii.setdefault(h, []).append(int(lab))
    export["text_index"] = {k: float(np.mean(v)) for k, v in ti.items()}
    export["img_index"] = {k: float(np.mean(v)) for k, v in ii.items()}
    export["text_index_nd"] = {k: float(np.mean(v)) for k, v in tnd.items()}
    export["text_index_w5"] = {k: float(np.mean(v)) for k, v in tw5.items()}
    print(f"индексы: text {len(export['text_index'])}, nd {len(export['text_index_nd'])}, "
          f"w5 {len(export['text_index_w5'])}, img {len(export['img_index'])}", flush=True)

    for cat, sub in df.groupby("category"):
        sub = sub.reset_index(drop=True)
        texts = (sub["name"].fillna("") + " " + sub["description"].fillna("")).tolist()
        y = sub["label"].values
        folds = sub[args.fold_column].values

        thr, f1 = oof_threshold(sub, texts, y, folds, imghash)
        print(f"{cat}: OOF blend F1={f1:.4f} @ thr={thr:.2f}", flush=True)

        vw, vc = make_vecs()
        X = build_X(texts, vw, vc, fit=True)
        clf = LogisticRegression(max_iter=MAX_ITER, C=C_REG, class_weight="balanced")
        clf.fit(X, y)

        vocab_w = {t: int(i) for t, i in vw.vocabulary_.items()}
        vocab_c = {t: int(i) for t, i in vc.vocabulary_.items()}
        sample = texts[:200]
        dw = float(np.abs(vw.transform(sample)
                          - transform_tfidf(sample, vocab_w, vw.idf_, word_ngrams)).max())
        dc = float(np.abs(vc.transform(sample)
                          - transform_tfidf(sample, vocab_c, vc.idf_, char_wb_ngrams)).max())
        print(f"  manual vs sklearn: word {dw:.2e}, char {dc:.2e}", flush=True)
        assert dw < 1e-9 and dc < 1e-9, "manual transform расходится со sklearn"

        export["categories"][cat] = dict(
            vocab_w=vocab_w,
            idf_w=vw.idf_.astype(np.float64),
            vocab_c=vocab_c,
            idf_c=vc.idf_.astype(np.float64),
            coef=clf.coef_[0].astype(np.float64),
            intercept=float(clf.intercept_[0]),
            thr=thr,
        )

        if cat == LV:
            names = [norm_text(n, "") for n in sub["name"].fillna("")]
            export["fuzzy_lv"] = {
                "X": transform_tfidf(names, vocab_c, vc.idf_, char_wb_ngrams),
                "y": y.astype(np.int64),
            }
            print(f"  fuzzy_lv: {export['fuzzy_lv']['X'].shape}", flush=True)

    if not args.no_dhash:
        per_cat = build_dhash_votes(df, images_dir)
        export["dhash_votes_cat"] = per_cat
        export["dhash_votes"] = per_cat.get(LV, {})

    with open(out_path, "wb") as fh:
        pickle.dump(export, fh, protocol=4)
    print("saved:", out_path, f"({os.path.getsize(out_path) / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
