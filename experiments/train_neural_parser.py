from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def load_jsonl(path: Path) -> List[Dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def main() -> int:
    if len(sys.argv) != 5:
        print("USAGE: python -m experiments.train_neural_parser TRAIN.jsonl TEST.jsonl OUT_DIR SEED", file=sys.stderr)
        return 2

    train_path = Path(sys.argv[1])
    test_path = Path(sys.argv[2])
    out_dir = Path(sys.argv[3])
    seed = int(sys.argv[4])

    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = os.environ.get("NEURAL_PARSER_MODEL", "distilbert-base-uncased")

    random.seed(seed)
    torch.manual_seed(seed)

    train_rows = load_jsonl(train_path)
    test_rows = load_jsonl(test_path)

    ds_train = Dataset.from_list([{"text": r["text"], "label": int(r["sid"])} for r in train_rows])
    ds_test = Dataset.from_list([{"text": r["text"], "label": int(r["sid"])} for r in test_rows])

    tok = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tok(batch["text"], truncation=True, padding="max_length", max_length=64)

    ds_train = ds_train.map(tokenize, batched=True)
    ds_test = ds_test.map(tokenize, batched=True)

    ds_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    ds_test.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=256)

    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=64,
        per_device_eval_batch_size=128,
        learning_rate=5e-5,
        num_train_epochs=1,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        logging_steps=200,
        seed=seed,
        report_to=[],
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        acc = float((preds == labels).mean())
        return {"acc": acc}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_test,
        tokenizer=tok,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))

    metrics = trainer.evaluate()
    (out_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"[PASS] TRAIN_NEURAL_PARSER: model={model_name} out={out_dir} metrics={metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
