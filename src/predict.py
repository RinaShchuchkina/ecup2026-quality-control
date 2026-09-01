import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pickle

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent if HERE.name == "src" else HERE
for _p in (REPO_ROOT, HERE, REPO_ROOT / "vendor", HERE / "vendor"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from src.analyzers import word_ngrams, char_wb_ngrams, rule_flags, transform_tfidf
from src.comments import make_comment, format_result, validate_submission
from src.prompts import RULES_BAD, RULES_LV
from src.rules_lv import route as lv_route
from src.encoder_branch import encoder_branch

from scipy.sparse import hstack, csr_matrix

TAG_RE = re.compile(r"<[^>]+>")
NONWORD_RE = re.compile(r"[^a-zа-яё0-9]+")
MAX_PIXELS = 262144
MAX_LEN = 3072

WEIGHTS_ENV = "ECUP_WEIGHTS"
CONFIG_ENV = "ECUP_CONFIG"
DEFAULT_WEIGHTS_DIR = "weights"
DEFAULT_CONFIG_NAME = "v2_config.json"
_weights_dir = os.environ.get(WEIGHTS_ENV) or DEFAULT_WEIGHTS_DIR


def set_weights_dir(value):
    global _weights_dir
    if value and not os.environ.get(WEIGHTS_ENV):
        _weights_dir = str(value)


def weights_roots():
    p = Path(_weights_dir)
    if p.is_absolute():
        return [p]
    return [REPO_ROOT / p, HERE / p, Path.cwd() / p]


def asset_path(name):
    p = Path(str(name))
    if p.is_absolute():
        return p
    roots = weights_roots()
    candidates = [r / p for r in roots]
    candidates += [HERE / p, REPO_ROOT / p]
    candidates += [r / p.name for r in roots]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def config_path(explicit=None):
    candidates = []
    for c in (explicit, os.environ.get(CONFIG_ENV)):
        if c:
            candidates.append(Path(c))
    candidates += [HERE / DEFAULT_CONFIG_NAME,
                   REPO_ROOT / DEFAULT_CONFIG_NAME,
                   REPO_ROOT / "configs" / DEFAULT_CONFIG_NAME]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        f"конфигурация не найдена: {[str(c) for c in candidates]}; "
        f"передайте --config или задайте {CONFIG_ENV}")

def norm_text(name, desc):
    t = f"{name} {desc}".lower()
    t = TAG_RE.sub(" ", t)
    t = NONWORD_RE.sub(" ", t)
    return " ".join(t.split())

DIGIT_RE = re.compile(r"\d+")
NOT_BAD_RE = re.compile(
    r"\bне\s+является\s+"
    r"(?:(?:лекарственн\w+\s+средств\w*|лекарств\w+)\s+(?:и|,)\s+)?"
    r"(?:бад(?:ом)?\b|биологически\s+активн\w*\s+добавк\w*)"
    r"|\bне\s+относится\s+к\s+(?:категории\s+)?"
    r"(?:бад\b|биологически\s+активн\w*\s+добавк\w*)"
    r"|\b(?:данный\s+продукт\s+)?не\s+бад(?:ом)?\b",
    re.I,
)

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

def _popcount64(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)) & np.uint64(0xFF)

def dhash_positive(model, images_dir, pid, min_votes, cat=None):
    byc = model.get("dhash_votes_cat") or {}
    idx = byc.get(cat) if cat in byc else model.get("dhash_votes")
    if not idx:
        return None
    per_img = []
    for hv in dhash_list(images_dir, pid):
        rec = idx.get(hv)
        if rec is None:
            continue
        cnt, sm = rec
        if sm == 0:
            per_img.append((0, cnt))
        elif sm == cnt:
            per_img.append((1, cnt))
        else:
            per_img.append((-1, cnt))
    if not per_img:
        return None
    if any(v == -1 for v, _ in per_img):
        return None
    vals = {v for v, _ in per_img}
    if len(vals) != 1:
        return None
    if sum(c for _, c in per_img) < min_votes:
        return None
    return vals.pop()

def dhash_votes_legacy(model, images_dir, pid, tol):
    idx = model.get("dhash_index") or {}
    arr = model.get("dhash_arr")
    lab = model.get("dhash_lab")
    votes = []
    for hv in dhash_list(images_dir, pid):
        if hv in idx:
            votes.append(idx[hv])
            continue
        if arr is not None and tol > 0:
            dist = _popcount64(np.uint64(hv) ^ arr).astype(np.int16)
            m = dist <= tol
            if m.any():
                votes.append(float(lab[m].mean()))
    return votes

def retrieval_key(idx_name, name, desc):
    if idx_name == "text_index":
        return norm_text(name, desc)
    base = norm_text(name, "")
    if idx_name == "text_index_nd":
        return " ".join(DIGIT_RE.sub(" ", base).split())
    if idx_name == "text_index_w5":
        return " ".join(base.split()[:5])
    return norm_text(name, desc)

def image_hashes(images_dir, pid):
    d = images_dir / str(pid)
    out = []
    if d.is_dir():
        for f in sorted(os.listdir(d)):
            try:
                with open(d / f, "rb") as fh:
                    out.append(hashlib.md5(fh.read()).hexdigest())
            except OSError:
                pass
    return out

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def text_branch(model, test):
    preds_prob = np.zeros(len(test))
    for cat, idx in test.groupby("category").groups.items():
        idx = np.asarray(idx)
        cb = model["categories"].get(cat) or next(iter(model["categories"].values()))
        texts = (test["name"].iloc[idx] + " " + test["description"].iloc[idx]).tolist()
        Xw = transform_tfidf(texts, cb["vocab_w"], cb["idf_w"], word_ngrams)
        Xc = transform_tfidf(texts, cb["vocab_c"], cb["idf_c"], char_wb_ngrams)
        Xr = csr_matrix(np.vstack([rule_flags(t) for t in texts])) * 2.0
        X = hstack([Xw, Xc, Xr]).tocsr()
        preds_prob[idx] = sigmoid(X @ cb["coef"] + cb["intercept"])
    return preds_prob

def find_base_model(candidates):
    shared = os.environ.get("SHARED_MODELS_PATH", "/shared_models")
    for c in candidates:
        p = Path(shared) / c
        if p.is_dir():
            return str(p)
    raise FileNotFoundError(f"базовая модель не найдена в {shared}: {candidates}")

def lora_branch(cfg, test, images_dir, adapter_dir, acfg):
    import torch
    from transformers import AutoProcessor
    import transformers
    from peft import PeftModel
    from src.vision import collect_images

    max_imgs = int(acfg.get("max_imgs", 4))
    max_pixels = int(acfg.get("max_pixels", MAX_PIXELS))
    max_len = int(acfg.get("max_len", MAX_LEN))
    base = find_base_model(acfg.get("model_candidates") or cfg["model_candidates"])
    print("base model:", base, "adapter:", adapter_dir.name,
          f"(imgs<={max_imgs}, px<={max_pixels}, len<={max_len})", flush=True)
    processor = AutoProcessor.from_pretrained(base, trust_remote_code=True)
    tok = processor.tokenizer
    mtype = json.load(open(os.path.join(base, "config.json"))).get("model_type", "")
    cls = None
    if "gemma" in mtype:
        cls = getattr(transformers, "Gemma4ForConditionalGeneration", None)
    elif "qwen3_5" in mtype:
        cls = getattr(transformers, "Qwen3_5ForConditionalGeneration", None)
        if cls is None:
            raise RuntimeError(
                f"transformers {transformers.__version__} не знает {mtype}: "
                "Qwen3_5ForConditionalGeneration отсутствует, откат недопустим")
    if cls is None:
        cls = getattr(transformers, "Qwen3VLForConditionalGeneration",
                      transformers.AutoModelForImageTextToText)
    model = cls.from_pretrained(base, dtype=torch.bfloat16,
                                attn_implementation="sdpa", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model = model.merge_and_unload().eval().cuda()
    yes_id = tok.encode("да", add_special_tokens=False)[0]
    no_id = tok.encode("нет", add_special_tokens=False)[0]

    def build_messages(row):
        rules = RULES_BAD if row["category"] == "БАД" else RULES_LV
        text = f"Название: {row['name']}\nОписание: {str(row['description'])[:2500]}"
        d = images_dir / str(row["id"])
        content = []
        if d.is_dir():
            imgs = [str(d / f) for f in sorted(os.listdir(d))[:max_imgs]
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            content += [{"type": "image", "image": p, "max_pixels": max_pixels} for p in imgs]
        content.append({"type": "text", "text": text})
        return [
            {"role": "system", "content": [{"type": "text", "text": rules}]},
            {"role": "user", "content": content},
        ]

    probs = np.full(len(test), np.nan)
    bs = int(cfg.get("batch_size", 8))
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(test), bs):
            rows = [test.iloc[i] for i in range(s, min(s + bs, len(test)))]
            convs = [build_messages(r) for r in rows]
            try:
                texts = processor.apply_chat_template(convs, tokenize=False,
                                                      add_generation_prompt=True,
                                                      enable_thinking=False)
            except TypeError:
                texts = processor.apply_chat_template(convs, tokenize=False,
                                                      add_generation_prompt=True)
            images = collect_images(convs)
            enc = processor(text=texts, images=images if images else None,
                            padding=True, truncation=True, max_length=max_len,
                            return_tensors="pt")
            enc = {k: v.cuda() for k, v in enc.items()}
            if "pixel_values" in enc:
                enc["pixel_values"] = enc["pixel_values"].to(torch.bfloat16)
            logits = model(**enc).logits
            lens = enc["attention_mask"].sum(1) - 1
            for j in range(logits.shape[0]):
                lg = logits[j, lens[j]]
                two = torch.stack([lg[yes_id], lg[no_id]]).float()
                probs[s + j] = torch.softmax(two, dim=0)[0].item()
            if s % 200 == 0:
                el = time.time() - t0
                print(f"lora scored {s}/{len(test)} ({el:.0f}s)", flush=True)
    print(f"lora total: {time.time()-t0:.0f}s", flush=True)
    return probs

def knn_branch(cfg, test, images_dir, view="joint", cats=None):
    import torch
    import torch.nn.functional as F
    from transformers import AutoProcessor
    import transformers
    from src.vision import collect_images

    kc = cfg["knn"]
    base = find_base_model(kc["model_candidates"])
    print("knn embed model:", base, flush=True)
    processor = AutoProcessor.from_pretrained(base, trust_remote_code=True)
    model = None
    for cls_name in ("AutoModelForImageTextToText", "Qwen3VLForConditionalGeneration", "AutoModel"):
        try:
            model = getattr(transformers, cls_name).from_pretrained(
                base, dtype=torch.bfloat16, trust_remote_code=True).eval().cuda()
            print("knn loaded via", cls_name, flush=True)
            break
        except Exception as e:
            print(f"knn {cls_name} failed: {type(e).__name__}: {str(e)[:100]}", flush=True)
    if model is None:
        raise RuntimeError("embedding model load failed")
    instr = kc["instruction"]
    max_px = int(kc.get("max_pixels", 261120))

    def conv(row):
        d = images_dir / str(row["id"])
        content = []
        if view == "joint" and d.is_dir():
            imgs = [str(d / f) for f in sorted(os.listdir(d))[:5]
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            content += [{"type": "image", "image": p, "max_pixels": max_px} for p in imgs]
        text = f"Название: {row['name']}\nКатегория: {row['category']}\nОписание: {row['description']}"[:6000]
        content.append({"type": "text", "text": text})
        return [{"role": "system", "content": [{"type": "text", "text": instr}]},
                {"role": "user", "content": content}]

    rows_idx = [i for i in range(len(test)) if (cats is None or test["category"].iloc[i] in cats)]
    embs = []
    bs = int(kc.get("batch_size", 8))
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(rows_idx), bs):
            rows = [test.iloc[i] for i in rows_idx[s:s + bs]]
            convs = [conv(r) for r in rows]
            texts = processor.apply_chat_template(convs, add_generation_prompt=True, tokenize=False)
            images = collect_images(convs)
            enc = processor(text=texts, images=images if images else None, padding=True,
                            truncation=True, max_length=4096, return_tensors="pt")
            enc = {k: v.cuda() for k, v in enc.items()}
            for k, v in enc.items():
                if torch.is_floating_point(v):
                    enc[k] = v.to(model.dtype)
            out = model(**enc, output_hidden_states=True)
            hidden = out.hidden_states[-1]
            am = enc["attention_mask"]
            col = am.shape[1] - am.flip(dims=[1]).argmax(dim=1) - 1
            e = hidden[torch.arange(hidden.shape[0], device=hidden.device), col]
            embs.append(F.normalize(e.float(), p=2, dim=-1).cpu().numpy())
            if s % 400 == 0:
                print(f"knn[{view}] embedded {s}/{len(rows_idx)} ({time.time()-t0:.0f}s)", flush=True)
    votes = np.full(len(test), np.nan)
    if not rows_idx:
        return votes
    Etest = np.vstack(embs)
    tr = np.load(asset_path("train_embeds.npz" if view == "joint" else "train_embeds_text.npz"))
    Etr = tr["emb"].astype(np.float32)
    ytr = tr["label"].astype(np.float32)
    ctr = tr["cat"].astype(int)
    cat_code = {"БАД": 0, "Легковоспламеняющиеся": 1}
    k = int(kc.get("k", 3))
    S = Etest @ Etr.T
    for r, i in enumerate(rows_idx):
        c = cat_code.get(test["category"].iloc[i])
        thr = kc["thr" if view == "joint" else "thr_text"].get(test["category"].iloc[i])
        if c is None or thr is None:
            continue
        sims = np.where(ctr == c, S[r], -1.0)
        idx = np.argpartition(-sims, k)[:k]
        idx = idx[sims[idx] >= thr]
        if len(idx) == 0:
            continue
        w = sims[idx]
        votes[i] = float((w * ytr[idx]).sum() / w.sum())
    print(f"knn[{view}]: покрыто {int(np.isfinite(votes).sum())}/{len(test)}, {time.time()-t0:.0f}s", flush=True)
    return votes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_data_path", "-i", required=True)
    ap.add_argument("--output_path", "--output-path", "-o", dest="output_path", required=True)
    ap.add_argument("--config", "-c", dest="config", default=None)
    args = ap.parse_args()

    cfg = json.load(open(config_path(args.config), encoding="utf-8"))
    set_weights_dir(cfg.get("weights_dir"))
    model = pickle.load(open(asset_path("model_v1.pkl"), "rb"))

    test = pd.read_csv(args.test_data_path)
    if test.columns[0].startswith("Unnamed"):
        test = test.drop(columns=test.columns[0])
    test["description"] = test["description"].fillna("")
    test["name"] = test["name"].fillna("")
    images_dir = Path(args.test_data_path).resolve().parent / "images"

    p_text = text_branch(model, test)

    vlm_probs = {}
    for name in cfg.get("adapters", ["adapter"]):
        adir = asset_path(name)
        if not adir.is_dir():
            msg = f"ADAPTER MISSING IN ARCHIVE: {name} ({adir})"
            if cfg.get("strict_adapters"):
                raise RuntimeError(msg)
            print(msg, file=sys.stderr, flush=True)
            continue
        cats = [c for c, ws in cfg.get("vlm_weights", {}).items()
                if float(ws.get(name, 0)) > 0]
        mask = test["category"].isin(cats).values if cats else np.ones(len(test), bool)
        if not mask.any():
            continue
        try:
            import torch
            acfg = cfg.get("adapter_cfg", {}).get(name, {})
            sub = test[mask].reset_index(drop=True)
            p_sub = lora_branch(cfg, sub, images_dir, adir, acfg)
            p_full = np.full(len(test), np.nan)
            p_full[np.where(mask)[0]] = p_sub
            vlm_probs[name] = p_full
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"LORA BRANCH {name} FAILED ({type(e).__name__}): {e}",
                  file=sys.stderr, flush=True)

    for _n in cfg.get("adapters", []):
        _st = "OK" if _n in vlm_probs else "НЕ ЗАГРУЖЕН (fallback на текст)"
        print(f"branch {_n}: {_st}" + (f", строк {int(np.isfinite(vlm_probs[_n]).sum())}" if _n in vlm_probs else ""), flush=True)

    p_final = p_text.copy()
    if vlm_probs:
        for cat, idx in test.groupby("category").groups.items():
            idx = np.asarray(idx)
            weights = dict(cfg["vlm_weights"].get(cat, {}))
            avail = {n: w for n, w in weights.items()
                     if n in vlm_probs and w > 0}
            if not avail:
                continue
            w_text = max(0.0, 1.0 - sum(avail.values()))
            acc = w_text * p_text[idx]
            wsum = w_text
            for n, w in avail.items():
                pl = vlm_probs[n][idx]
                pl = np.where(np.isnan(pl), p_text[idx], pl)
                acc = acc + w * pl
                wsum += w
            p_final[idx] = acc / max(wsum, 1e-9)

    enc_w = cfg.get("enc_weights") or {}
    enc_dirs = {c: cfg["encoder_dirs"][c] for c in cfg["encoder_dirs"]} \
        if cfg.get("encoder_dirs") else \
        {c: cfg.get("encoder_dir", "encoder") for c in enc_w}
    if enc_w:
        try:
            t_enc = time.time()
            cache, used = {}, {}
            for cat, idx in test.groupby("category").groups.items():
                e = float(enc_w.get(cat, 0.0))
                if e <= 0:
                    continue
                d = asset_path(enc_dirs.get(cat, cfg.get("encoder_dir", "encoder")))
                if not d.is_dir():
                    raise FileNotFoundError(f"encoder dir missing in archive: {d}")
                if d not in cache:
                    cache[d] = encoder_branch(
                        test, d, max_len=int(cfg.get("encoder_max_len", 1024)),
                        bs=int(cfg.get("encoder_bs", 64)))
                idx = np.asarray(idx)
                p_final[idx] = (1.0 - e) * p_final[idx] + e * cache[d][idx]
                used[cat] = (d.name, e)
            print(f"encoder branch: OK, {time.time() - t_enc:.1f}s, "
                  f"{len(cache)} model(s), {used}", flush=True)
        except Exception as e:
            print(f"ENCODER BRANCH FAILED ({type(e).__name__}): {e}",
                  file=sys.stderr, flush=True)

    eva_w = cfg.get("eva_weights") or {}
    eva_live = {}
    if eva_w:
        try:
            t_eva = time.time()
            from src.eva_branch import eva_scores
            wdir = asset_path(cfg.get("eva_dir", "eva_weights"))
            if not wdir.is_dir():
                raise FileNotFoundError(f"eva dir missing in archive: {wdir}")
            used = {}
            for cat, idx in test.groupby("category").groups.items():
                w = float(eva_w.get(cat, 0.0))
                if w <= 0:
                    continue
                idx = np.asarray(idx)
                ids = [test["id"].iloc[i] for i in idx]
                from src.eva_branch import CAT_TAG
                sc = eva_scores(ids, str(images_dir), str(wdir), CAT_TAG.get(cat, cat[:3]),
                                max_folds=int(cfg.get("eva_max_folds", 5)))
                if not sc:
                    raise RuntimeError(f"ветвь EVA не дала ни одного предсказания для {cat}")
                eva_live[cat] = w
                pv = np.array([sc.get(pid, np.nan) for pid in ids], dtype=np.float64)
                ok = ~np.isnan(pv)
                p_final[idx[ok]] = (1.0 - w) * p_final[idx[ok]] + w * pv[ok]
                used[cat] = (w, int(ok.sum()), len(ids))
            print(f"eva branch: OK, {time.time() - t_eva:.1f}s, {used}", flush=True)
        except Exception as e:
            print(f"EVA BRANCH FAILED ({type(e).__name__}): {e}",
                  file=sys.stderr, flush=True)
            if cfg.get("strict_branches", True):
                raise

    for _cat in sorted(set(test["category"])):
        _decl = cfg.get("vlm_weights", {}).get(_cat, {}) or {}
        _live = {n: float(w) for n, w in _decl.items() if n in vlm_probs and float(w) > 0}
        _dead = sorted(set(n for n, w in _decl.items() if float(w) > 0) - set(_live))
        _wv = sum(_live.values())
        _wx = float(eva_live.get(_cat, 0.0))
        _shrink = 1.0 - _wx
        _parts = " ".join(f"{n}={w * _shrink:.4f}" for n, w in sorted(_live.items()))
        print(f"ВЕСА[{_cat}] ЖИВЫЕ: tfidf={(1.0 - _wv) * _shrink:.4f} {_parts} "
              f"eva={_wx:.4f}" + (f" | ОТВАЛИЛИСЬ: {_dead}" if _dead else ""), flush=True)
        if _dead and cfg.get("strict_branches", True):
            raise RuntimeError(f"ветви {_dead} заявлены для {_cat}, но не загрузились")

    nb_cats = set(cfg.get("not_bad_rule_cats", []))
    if nb_cats:
        n_nb = 0
        for i in np.where(test["category"].isin(nb_cats).values)[0]:
            t = f"{test['name'].iloc[i]} {test['description'].iloc[i]}".lower()
            if NOT_BAD_RE.search(t):
                p_final[i] = 0.01
                n_nb += 1
        print(f"not_bad rule -> 0: {n_nb}", flush=True)

    exact_cats = set(cfg.get("exact_retrieval_cats", ["БАД", "Легковоспламеняющиеся"]))
    rkeys = cfg.get("retrieval_keys", {})
    img_cats = set(cfg.get("retrieval_img_cats", ["БАД", "Легковоспламеняющиеся"]))
    dhash_tol = int(cfg.get("dhash_tol", 0))
    n_exact = 0
    exact_hit = np.zeros(len(test), dtype=bool)
    for i in range(len(test)):
        if test["category"].iloc[i] not in exact_cats:
            continue
        votes = []
        fallback = []
        for idx_name in rkeys.get(test["category"].iloc[i], ["text_index"]):
            if idx_name == "dhash_index":
                continue
            idx = model.get(idx_name)
            if not idx:
                continue
            tk = retrieval_key(idx_name, test["name"].iloc[i], test["description"].iloc[i])
            if tk in idx:
                votes.append(idx[tk])
        if not votes:
            votes = fallback
        if test["category"].iloc[i] in img_cats:
            for h in image_hashes(images_dir, test["id"].iloc[i]):
                if h in model["img_index"]:
                    votes.append(model["img_index"][h])
        if votes:
            v = float(np.mean(votes))
            if v in (0.0, 1.0):
                p_final[i] = v
                exact_hit[i] = True
                n_exact += 1
    print(f"exact retrieval override: {n_exact}", flush=True)

    dh_cats = set(cfg.get("dhash_cats", []))
    dh_sym_cats = set(cfg.get("dhash_sym_cats", []))
    if dh_cats or dh_sym_cats:
        dh_min = int(cfg.get("dhash_min_votes", 2))
        n_dh = n_dh0 = 0
        try:
            all_dh = dh_cats | dh_sym_cats
            for i in np.where(test["category"].isin(all_dh).values & ~exact_hit)[0]:
                cat_i = test["category"].iloc[i]
                v = dhash_positive(model, images_dir, test["id"].iloc[i], dh_min, cat_i)
                if v == 1:
                    p_final[i] = 0.99
                    exact_hit[i] = True
                    n_dh += 1
                elif v == 0 and cat_i in dh_sym_cats:
                    p_final[i] = 0.01
                    exact_hit[i] = True
                    n_dh0 += 1
        except Exception as e:
            print(f"DHASH FAILED ({type(e).__name__}): {e}", file=sys.stderr, flush=True)
        print(f"dhash override: ->1 {n_dh}, ->0 {n_dh0}", flush=True)

    fz = model.get("fuzzy_lv")
    fuzzy_cats = set(cfg.get("fuzzy_cats", []))
    if fz is not None and fuzzy_cats:
        fthr = float(cfg.get("fuzzy_thr", 0.90))
        fuzzy_only_pos = bool(cfg.get("fuzzy_only_positive", True))
        n_fuzzy = 0
        try:
            cand = np.where(test["category"].isin(fuzzy_cats).values & ~exact_hit)[0]
            if len(cand):
                cbf = model["categories"][sorted(fuzzy_cats)[0]]
                Xtr = fz["X"].tocsr()
                ytr = np.asarray(fz["y"])
                for st in range(0, len(cand), 512):
                    part = cand[st:st + 512]
                    q = [norm_text(test["name"].iloc[i], "") for i in part]
                    Xq = transform_tfidf(q, cbf["vocab_c"], cbf["idf_c"], char_wb_ngrams)
                    S = (Xq @ Xtr.T).toarray()
                    for r, i in enumerate(part):
                        nb = np.where(S[r] >= fthr)[0]
                        if nb.size:
                            vv = ytr[nb]
                            if vv.min() == vv.max():
                                val = int(vv[0])
                                if val == 1 or not fuzzy_only_pos:
                                    p_final[i] = 0.99 if val == 1 else 0.01
                                    n_fuzzy += 1
        except Exception as e:
            print(f"FUZZY FAILED ({type(e).__name__}): {e}", file=sys.stderr, flush=True)
        print(f"fuzzy retrieval override: {n_fuzzy}", flush=True)

    if cfg.get("knn", {}).get("enabled"):
        kc = cfg["knn"]
        votes = np.full(len(test), np.nan)
        try:
            if kc.get("thr"):
                votes = knn_branch(cfg, test, images_dir, "joint", cats=set(kc["thr"]))
        except Exception as e:
            print(f"KNN JOINT FAILED ({type(e).__name__}): {e}", file=sys.stderr, flush=True)
        try:
            if kc.get("thr_text"):
                vt = knn_branch(cfg, test, images_dir, "text", cats=set(kc["thr_text"]))
                fill = ~np.isfinite(votes) & np.isfinite(vt)
                votes[fill] = vt[fill]
        except Exception as e:
            print(f"KNN TEXT FAILED ({type(e).__name__}): {e}", file=sys.stderr, flush=True)
        m = np.isfinite(votes)
        p_final[m] = np.where(votes[m] >= 0.5, 0.99, 0.01)
        print(f"knn override: {int(m.sum())}", flush=True)

    if cfg.get("lv_router"):
        n_routed = n_skip = 0
        for i in range(len(test)):
            if test["category"].iloc[i] != "Легковоспламеняющиеся":
                continue
            if exact_hit[i]:
                n_skip += 1
                continue
            text = f"{test['name'].iloc[i]} {test['description'].iloc[i]}".lower()
            ovr, rule = lv_route(text, name_lower=str(test["name"].iloc[i]).lower())
            if ovr is not None:
                p_final[i] = 0.99 if ovr == 1 else 0.01
                n_routed += 1
        print(f"lv_router: переопределено {n_routed} (пропущено покрытых retrieval: {n_skip})", flush=True)

    for cat_f, val in (cfg.get("force_class") or {}).items():
        idx = test.index[test["category"] == cat_f].to_numpy()
        p_final[idx] = 0.99 if int(val) == 1 else 0.01
        print(f"force_class: {cat_f} -> {val} ({len(idx)} строк)", flush=True)

    preds = np.zeros(len(test), dtype=int)
    for cat, idx in test.groupby("category").groups.items():
        idx = np.asarray(idx)
        thr = float(cfg["thr"].get(cat, 0.5))
        preds[idx] = (p_final[idx] > thr).astype(int)

    known = set(model["categories"])
    fallback_cat = next(iter(known))
    result = [
        format_result(make_comment(str(n), str(d), c if c in known else fallback_cat, int(p)), int(p))
        for n, d, c, p in zip(test["name"], test["description"], test["category"], preds)
    ]
    out = pd.DataFrame({"id": test["id"], "result": result})
    errs = validate_submission(out)
    if errs:
        print("FORMAT ERRORS:", errs, file=sys.stderr)
        bad = ~out["result"].astype(str).str.match(
            r"^<комментарий>.{50,300}?<вердикт>(бан|не бан)$", na=False)
        for i in np.where(bad.values)[0]:
            p = int(preds[i]); cat = test["category"].iloc[i]
            out.iloc[i, out.columns.get_loc("result")] = format_result(
                make_comment("", "", cat if cat in known else fallback_cat, p), p)
    out.to_csv(args.output_path, index=False)
    print(f"ok: {len(out)} rows -> {args.output_path}")

if __name__ == "__main__":
    main()
