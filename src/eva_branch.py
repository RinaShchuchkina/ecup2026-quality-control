import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(HERE, "vendor") not in sys.path:
    sys.path.insert(0, os.path.join(HERE, "vendor"))

MODEL = "eva02_base_patch14_224.mim_in22k"
CACHE_S = 256
SIZE = 224
MAX_IMGS = 5
EXTS = (".jpg", ".jpeg", ".png", ".webp")
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


def product_files(img_root, pid):
    d = os.path.join(img_root, str(pid))
    if not os.path.isdir(d):
        return []
    fs = sorted([f for f in os.listdir(d) if f.lower().endswith(EXTS)],
                key=lambda x: (len(x), x))
    return [os.path.join(d, f) for f in fs[:MAX_IMGS]]


def _to_canvas(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    r = CACHE_S / max(w, h)
    nw, nh = max(1, int(round(w * r))), max(1, int(round(h * r)))
    im = im.resize((nw, nh), Image.BICUBIC)
    cv = Image.new("RGB", (CACHE_S, CACHE_S), (255, 255, 255))
    cv.paste(im, ((CACHE_S - nw) // 2, (CACHE_S - nh) // 2))
    return np.asarray(cv, dtype=np.uint8)


class ImgDS(Dataset):
    def __init__(self, paths):
        self.p = paths

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        try:
            a = _to_canvas(self.p[i])
        except Exception:
            a = np.full((CACHE_S, CACHE_S, 3), 255, np.uint8)
        return torch.from_numpy(a.copy())


CAT_TAG = {"БАД": "BAD", "Легковоспламеняющиеся": "LV"}


def load_models(wdir, cat_tag, dev, max_folds=5):
    import glob

    import timm
    from safetensors.torch import load_file

    files = sorted(glob.glob(os.path.join(wdir, f"eva224_{cat_tag}_f*.safetensors")))[:max_folds]
    if not files:
        have = sorted(os.listdir(wdir)) if os.path.isdir(wdir) else []
        raise FileNotFoundError(
            f"нет весов EVA по маске eva224_{cat_tag}_f* в {wdir}; лежит: {have}")
    ms = []
    for f in files:
        m = timm.create_model(MODEL, pretrained=False, num_classes=1)
        m.load_state_dict(load_file(f), strict=True)
        ms.append(m.eval().to(dev).to(torch.float16))
    return ms


@torch.no_grad()
def predict(paths, models, dev, bs=128, workers=4):
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    dl = DataLoader(ImgDS(paths), batch_size=bs, num_workers=workers, pin_memory=True)
    out = []
    for xb in dl:
        x = xb.to(dev, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
        x = F.interpolate(x, size=(SIZE, SIZE), mode="bicubic", align_corners=False).clamp_(0, 1)
        x = ((x - mean) / std).to(torch.float16).contiguous(memory_format=torch.channels_last)
        p = torch.stack([torch.sigmoid(m(x).float().squeeze(-1)) for m in models]).mean(0)
        out.append(p.cpu().numpy())
    return np.concatenate(out).astype(np.float32) if out else np.zeros(0, np.float32)


def eva_scores(ids, img_root, wdir, cat_tag, dev=None, max_folds=5, bs=128, workers=4):
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(wdir, cat_tag, dev, max_folds)
    paths, owner = [], []
    for pid in ids:
        for f in product_files(img_root, pid):
            paths.append(f)
            owner.append(pid)
    if not paths:
        return {}
    p = predict(paths, models, dev, bs=bs, workers=workers)
    acc = {}
    for pid, v in zip(owner, p):
        acc.setdefault(pid, []).append(float(v))
    return {k: float(np.mean(v)) for k, v in acc.items()}
