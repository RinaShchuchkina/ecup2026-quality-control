#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN = re.compile(r"/home/|/Users/|hf_cache|conda_envs|venvs/")


def collect_weights(cfg):
    items = []
    wdir = ROOT / cfg.get("weights_dir", "weights")
    for name in cfg.get("adapters", []):
        src = wdir / name
        if not src.is_dir():
            raise FileNotFoundError(f"нет каталога адаптера: {src}")
        items.append((src, name))
    eva = cfg.get("eva_weights") or {}
    if eva:
        src = ROOT / cfg.get("eva_dir", "weights/eva")
        if not src.is_dir():
            raise FileNotFoundError(f"нет каталога визуальной ветви: {src}")
        items.append((src, "eva_weights"))
    model = wdir / "model_v1.pkl"
    if not model.is_file():
        raise FileNotFoundError(f"нет текстовой модели: {model}")
    items.append((model, "model_v1.pkl"))
    return items


def build(cfg_path, out_path, max_mb):
    cfg = json.loads(Path(cfg_path).read_text())
    stage = Path(tempfile.mkdtemp(prefix="submit_"))
    try:
        (stage / "src").mkdir()
        for p in sorted((ROOT / "src").glob("*.py")):
            if p.name in {"predict.py", "pack_submit.py"} or p.name.startswith("train_"):
                continue
            if p.name in {"prepare_data.py", "build_image_cache.py"}:
                continue
            shutil.copy2(p, stage / "src" / p.name)
        (stage / "run.py").write_text((ROOT / "src" / "predict.py").read_text())
        (stage / "src" / "__init__.py").touch(exist_ok=True)
        shutil.copy2(ROOT / "src" / "metadata.json", stage / "metadata.json")

        packed = dict(cfg)
        packed.pop("weights_dir", None)
        if packed.get("eva_weights"):
            packed["eva_dir"] = "eva_weights"
        (stage / "v2_config.json").write_text(
            json.dumps(packed, ensure_ascii=False, indent=1) + "\n")

        for src, name in collect_weights(cfg):
            dst = stage / name
            if src.is_dir():
                shutil.copytree(src, dst, ignore=shutil.ignore_patterns("README.md"))
            else:
                shutil.copy2(src, dst)

        shutil.copytree(ROOT / "vendor", stage / "vendor",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

        for p in stage.rglob("*"):
            if not p.is_file():
                continue
            if any(ord(ch) > 127 for ch in p.name):
                raise ValueError(f"не-ASCII имя файла: {p.relative_to(stage)}")
            if p.suffix in {".py", ".json", ".md", ".txt"}:
                if FORBIDDEN.search(p.read_text(errors="ignore")):
                    raise ValueError(f"локальный путь в {p.relative_to(stage)}")

        out = Path(out_path)
        if out.exists():
            raise FileExistsError(f"{out} уже существует")
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(stage.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(stage))

        size_mb = out.stat().st_size / 1048576
        if size_mb > max_mb:
            raise ValueError(f"архив {size_mb:.0f} МБ превышает лимит {max_mb} МБ")

        probe = Path(tempfile.mkdtemp(prefix="probe_"))
        try:
            with zipfile.ZipFile(out) as z:
                z.extractall(probe)
            r = subprocess.run([sys.executable, "run.py", "--help"], cwd=probe,
                               capture_output=True, text=True, timeout=300)
            if r.returncode != 0:
                raise RuntimeError("распакованный архив не запускается:\n"
                                   + (r.stderr or r.stdout)[-800:])
            print("  проверка запуска: run.py стартует из распакованного архива")
        finally:
            shutil.rmtree(probe, ignore_errors=True)
        with zipfile.ZipFile(out) as z:
            broken = z.testzip()
            if broken:
                raise ValueError(f"битый файл в архиве: {broken}")
            names = z.namelist()
            digest = {}
            for name in names:
                if name.endswith("adapter_model.safetensors"):
                    digest[name] = hashlib.sha256(z.read(name)).hexdigest()[:12]
        print(f"{out} — {size_mb:.0f} МБ, файлов {len(names)}")
        for k, v in sorted(digest.items()):
            print(f"  {k}: sha256 {v}")
        for cat, w in cfg.get("vlm_weights", {}).items():
            wx = (cfg.get("eva_weights") or {}).get(cat, 0.0)
            scale = 1.0 - wx
            parts = " ".join(f"{n}={v * scale:.2f}" for n, v in sorted(w.items()))
            print(f"  {cat}: tfidf={(1 - sum(w.values())) * scale:.2f} {parts} eva={wx:.2f}")
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-mb", type=int, default=5000)
    args = ap.parse_args()
    build(args.config, args.out, args.max_mb)


if __name__ == "__main__":
    main()
