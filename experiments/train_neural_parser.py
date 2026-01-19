from __future__ import annotations

import argparse
import re
import json
import os
import random
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def load_jsonl(path: Path) -> List[Dict]:
    out: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m experiments.train_neural_parser",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- keep original positional interface ----
    p.add_argument("train_jsonl", type=str, help="Training dataset (.jsonl)")
    p.add_argument("test_jsonl", type=str, help="Evaluation dataset (.jsonl)")
    p.add_argument("out_dir", type=str, help="Output directory")
    p.add_argument("seed", type=int, help="Random seed")

    # ---- new knobs (optional) ----
    p.add_argument(
        "--model",
        type=str,
        default=os.environ.get("NEURAL_PARSER_MODEL", "distilbert-base-uncased"),
        help="HF model name or local path. Env NEURAL_PARSER_MODEL also supported.",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=os.environ.get("RESUME", ""),
        help="Resume checkpoint directory (or blank). Env RESUME also supported.",
    )
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--train_bs", type=int, default=64)
    p.add_argument("--eval_bs", type=int, default=128)
    p.add_argument("--max_len", type=int, default=96)
    p.add_argument("--token_cache", action="store_true", help="Cache tokenized datasets under out_dir/token_cache/")
    p.add_argument("--eval_steps", type=int, default=1000)
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--logging_steps", type=int, default=200)
    p.add_argument("--weight_decay", type=float, default=0.01)

    return p


def main() -> int:
    args = _argparser().parse_args()

    train_path = Path(args.train_jsonl)
    test_path = Path(args.test_jsonl)
    out_dir = Path(args.out_dir)
    seed = int(args.seed)

    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.model
    resume = args.resume.strip() or None

    # If resuming: load from that checkpoint
    # Else: load from the model base
    load_dir = resume if resume else model_name


    tok = AutoTokenizer.from_pretrained(load_dir, use_fast=False)
    random.seed(seed)
    torch.manual_seed(seed)

    train_rows = load_jsonl(train_path)
    test_rows = load_jsonl(test_path)

    # NOTE: sid → label
    def _safe_name(x: str) -> str:
        import re
        return re.sub(r"[^A-Za-z0-9._-]+", "_", x)

    cache_root = Path(out_dir) / "token_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_tag = f"{_safe_name(load_dir)}_len{int(args.max_len)}"
    cache_dir = cache_root / cache_tag

    if args.token_cache and cache_dir.exists():
        dd = load_from_disk(str(cache_dir))
        ds_train = dd["train"]
        ds_test = dd["test"]
        print(f"[CACHE] loaded: {cache_dir}")
    else:
        ds_train = Dataset.from_list([{"text": r["text"], "label": int(r["sid"])} for r in train_rows])
        ds_test = Dataset.from_list([{"text": r["text"], "label": int(r["sid"])} for r in test_rows])

        def tokenize(batch: Dict[str, Any]) -> Dict[str, Any]:
            return tok(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=int(args.max_len),
            )

        ds_train = ds_train.map(tokenize, batched=True)
        ds_test = ds_test.map(tokenize, batched=True)

        if args.token_cache:
            DatasetDict({"train": ds_train, "test": ds_test}).save_to_disk(str(cache_dir))
            print(f"[CACHE] saved: {cache_dir}")
    ds_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])    
    ds_test.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    # VSC has 256 clean state IDs
    model = AutoModelForSequenceClassification.from_pretrained(load_dir, num_labels=256)

    train_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=int(args.train_bs),
        per_device_eval_batch_size=int(args.eval_bs),
        learning_rate=float(args.lr),
        num_train_epochs=float(args.epochs),
        weight_decay=float(args.weight_decay),
        eval_strategy="steps",
        eval_steps=int(args.eval_steps),
        save_strategy="steps",
        save_steps=int(args.save_steps),
        logging_steps=int(args.logging_steps),
        seed=seed,
        report_to=[],
        load_best_model_at_end=False,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        acc = float((preds == labels).mean())
        return {"acc": acc}

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds_train,
        eval_dataset=ds_test,
        tokenizer=tok,
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))

    metrics = trainer.evaluate()
    (out_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"[PASS] TRAIN_NEURAL_PARSER: model={model_name} out={out_dir} metrics={metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
