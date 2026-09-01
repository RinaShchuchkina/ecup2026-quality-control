#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_BASE_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
LV = "Легковоспламеняющиеся"
MAX_PIXELS = 262144
MAX_LEN = 3072
MAX_IMGS = 4


def data_dir():
    return Path(os.environ.get("ECUP_DATA", "data"))


def out_dir():
    return Path(os.environ.get("ECUP_OUT", "runs"))


def weights_dir():
    return Path(os.environ.get("ECUP_WEIGHTS", "weights"))


def load_items(jsonl_path):
    items = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


class SftDataset(Dataset):
    def __init__(self, items, processor, with_answer=True):
        self.items = items
        self.processor = processor
        self.with_answer = with_answer

    def __len__(self):
        return len(self.items)

    def build_messages(self, it, with_answer):
        content = [{"type": "image", "image": p, "max_pixels": MAX_PIXELS}
                   for p in it["images"][:MAX_IMGS]]
        content.append({"type": "text", "text": it["text"]})
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": it["rules"]}]},
            {"role": "user", "content": content},
        ]
        if with_answer:
            ans = "да" if it["label"] == 1 else "нет"
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": ans}]})
        return msgs

    def __getitem__(self, i):
        from qwen_vl_utils import process_vision_info
        it = self.items[i]
        msgs_full = self.build_messages(it, self.with_answer)
        msgs_prompt = self.build_messages(it, False)

        try:
            text_full = self.processor.apply_chat_template(
                msgs_full, tokenize=False, add_generation_prompt=not self.with_answer,
                enable_thinking=False)
        except TypeError:
            text_full = self.processor.apply_chat_template(
                msgs_full, tokenize=False, add_generation_prompt=not self.with_answer)
        images, _ = process_vision_info([msgs_full])
        enc = self.processor(text=[text_full], images=images, return_tensors="pt",
                             truncation=True, max_length=MAX_LEN)
        input_ids = enc["input_ids"][0]
        labels = torch.full_like(input_ids, -100)
        if self.with_answer:
            try:
                text_prompt = self.processor.apply_chat_template(
                    msgs_prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                text_prompt = self.processor.apply_chat_template(
                    msgs_prompt, tokenize=False, add_generation_prompt=True)
            enc_p = self.processor(text=[text_prompt], images=images, return_tensors="pt",
                                   truncation=True, max_length=MAX_LEN)
            plen = enc_p["input_ids"].shape[1]
            assert input_ids.shape[0] > plen, (
                f"answer truncated: id={it['id']}, prompt={plen}, full={input_ids.shape[0]}")
            labels[plen:] = input_ids[plen:]
        out = dict(input_ids=input_ids, attention_mask=enc["attention_mask"][0],
                   labels=labels)
        for k, v in enc.items():
            if k in out or not torch.is_tensor(v):
                continue
            out[k] = v[0] if (v.dim() >= 1 and v.shape[0] == 1 and v.dim() == 2
                              and v.shape[1] == input_ids.shape[0]) else v
        return out


def collate(batch, pad_id):
    maxlen = max(b["input_ids"].shape[0] for b in batch)
    pad_val = {"input_ids": pad_id, "attention_mask": 0, "labels": -100}
    n0 = batch[0]["input_ids"].shape[0]
    out = {}
    for k in list(batch[0].keys()):
        v0 = batch[0][k]
        per_token = torch.is_tensor(v0) and v0.dim() == 1 and v0.shape[0] == n0
        if per_token:
            pv = pad_val.get(k, 0)
            out[k] = torch.stack([torch.nn.functional.pad(
                b[k], (0, maxlen - b[k].shape[0]), value=pv) for b in batch])
        else:
            parts = [b[k] for b in batch if k in b and torch.is_tensor(b[k]) and b[k].dim() > 0]
            if parts:
                out[k] = torch.cat(parts, dim=0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(out_dir() / "sft_data.jsonl"))
    ap.add_argument("--base", default=os.environ.get("ECUP_BASE_MODEL", DEFAULT_BASE_MODEL))
    ap.add_argument("--val-fold", type=int, default=-1)
    ap.add_argument("--out", default=str(out_dir() / "lora"))
    ap.add_argument("--adapter-dir", default=str(weights_dir() / "adapter"))
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--max-imgs", type=int, default=4)
    ap.add_argument("--max-pixels", type=int, default=262144)
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--save-steps", type=int, default=150)
    ap.add_argument("--lv-oversample", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    global MAX_PIXELS, MAX_LEN, MAX_IMGS
    MAX_PIXELS, MAX_LEN, MAX_IMGS = args.max_pixels, args.max_len, args.max_imgs
    print(f"cfg: imgs<={MAX_IMGS}, px<={MAX_PIXELS}, len<={MAX_LEN}, "
          f"epochs={args.epochs}, data={args.data}", flush=True)

    from transformers import AutoConfig, AutoProcessor, Trainer, TrainingArguments
    import transformers
    from peft import LoraConfig, PeftModel, get_peft_model

    import transformers.utils.import_utils as _iu
    import transformers.trainer as _tr
    _iu.check_torch_load_is_safe = lambda: None
    _tr.check_torch_load_is_safe = lambda: None

    model_path = args.base
    print("base model:", model_path, flush=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tok = processor.tokenizer

    def load_base():
        mtype = getattr(AutoConfig.from_pretrained(model_path, trust_remote_code=True),
                        "model_type", "")
        if "gemma" in mtype:
            pref = ("Gemma4ForConditionalGeneration",)
        elif "qwen3_5" in mtype or "qwen3.5" in mtype:
            pref = ("Qwen3_5ForConditionalGeneration",)
        else:
            pref = ()
        for cls_name in pref + ("Qwen3VLForConditionalGeneration", "AutoModelForImageTextToText"):
            try:
                cls = getattr(transformers, cls_name)
                m = cls.from_pretrained(model_path, dtype=torch.bfloat16,
                                        attn_implementation="sdpa", trust_remote_code=True)
                print(f"loaded via {cls_name}", flush=True)
                return m
            except (AttributeError, ValueError, OSError) as e:
                print(f"{cls_name}: {type(e).__name__}: {str(e)[:100]}", flush=True)
        raise RuntimeError("model load failed")

    items = load_items(args.data)
    train_items = [it for it in items if it["fold"] != args.val_fold]
    val_items = [it for it in items if it["fold"] == args.val_fold]
    extra = [it for it in train_items
             if it["category"] == LV and it["label"] == 1] * max(0, args.lv_oversample - 1)
    train_items = train_items + extra
    random.Random(args.seed).shuffle(train_items)
    print(f"train={len(train_items)} (oversample LV x{args.lv_oversample}), "
          f"val={len(val_items)}", flush=True)

    run_dir = Path(args.out)
    adapter_dir = Path(args.adapter_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_train:
        model = load_base()
        model.config.use_cache = False
        target = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
        try:
            lcfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r,
                              lora_dropout=0.05, target_modules=target, bias="none",
                              task_type="CAUSAL_LM")
            model = get_peft_model(model, lcfg)
        except ValueError:
            lcfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r,
                              lora_dropout=0.05,
                              target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                              "gate_proj", "up_proj", "down_proj"],
                              bias="none", task_type="CAUSAL_LM")
            model = get_peft_model(model, lcfg)
        model.print_trainable_parameters()
        model.enable_input_require_grads()

        train_ds = SftDataset(train_items, processor, with_answer=True)
        import inspect
        import math
        total_steps = math.ceil(len(train_items) / (args.batch * args.grad_accum)) * args.epochs
        kw = dict(
            output_dir=str(run_dir), num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr, lr_scheduler_type="cosine",
            warmup_steps=int(0.1 * total_steps),
            bf16=True, logging_steps=20, save_steps=args.save_steps, save_total_limit=2,
            report_to=[], dataloader_num_workers=4, remove_unused_columns=False,
            gradient_checkpointing=True, seed=args.seed,
        )
        allowed = set(inspect.signature(TrainingArguments.__init__).parameters)
        dropped = [k for k in kw if k not in allowed]
        if dropped:
            print("TrainingArguments: пропускаю неподдерживаемые", dropped, flush=True)
        targs = TrainingArguments(**{k: v for k, v in kw.items() if k in allowed})
        trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                          data_collator=lambda b: collate(b, tok.pad_token_id))
        ckpts = sorted(run_dir.glob("checkpoint-*"))
        trainer.train(resume_from_checkpoint=str(ckpts[-1]) if (args.resume and ckpts) else None)
        model.save_pretrained(str(adapter_dir))
        print("saved adapter:", adapter_dir, flush=True)
        del model, trainer
        torch.cuda.empty_cache()

    if not val_items:
        print("val fold пуст, скоринг пропущен")
        print("DONE")
        return
    print("=== scoring val fold ===", flush=True)
    model = load_base()
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model = model.merge_and_unload().eval().cuda()

    yes_id = tok.encode("да", add_special_tokens=False)[0]
    no_id = tok.encode("нет", add_special_tokens=False)[0]
    print("token ids:", yes_id, no_id, flush=True)

    val_ds = SftDataset(val_items, processor, with_answer=False)
    probs, ids = [], []
    bs = 4
    with torch.no_grad():
        for s in range(0, len(val_ds), bs):
            batch = [val_ds[i] for i in range(s, min(s + bs, len(val_ds)))]
            enc = collate(batch, tok.pad_token_id)
            enc = {k: v.cuda() for k, v in enc.items() if k != "labels"}
            if "pixel_values" in enc:
                enc["pixel_values"] = enc["pixel_values"].to(torch.bfloat16)
            logits = model(**enc).logits
            lens = enc["attention_mask"].sum(1) - 1
            for j in range(logits.shape[0]):
                lg = logits[j, lens[j]]
                two = torch.stack([lg[yes_id], lg[no_id]]).float()
                probs.append(torch.softmax(two, dim=0)[0].item())
            ids.extend(it["id"] for it in val_items[s:s + bs])
            if s % 400 == 0:
                print(f"scored {s}/{len(val_ds)}", flush=True)
    np.savez(run_dir / "val_probs.npz", ids=np.array(ids), probs=np.array(probs))

    import pandas as pd
    from sklearn.metrics import f1_score
    v = pd.DataFrame({"id": ids, "p": probs}).merge(
        pd.DataFrame([(it["id"], it["category"], it["label"]) for it in val_items],
                     columns=["id", "category", "label"]), on="id")
    for cat, g in v.groupby("category"):
        best = max((f1_score(g["label"], g["p"] > t), t) for t in np.arange(0.05, 0.95, 0.02))
        print(f"{cat}: F1@0.5={f1_score(g['label'], g['p'] > 0.5):.4f} "
              f"best={best[0]:.4f}@{best[1]:.2f}", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
