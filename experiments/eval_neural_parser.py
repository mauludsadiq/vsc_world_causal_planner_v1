from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_jsonl(path: Path):
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def main() -> int:
    if len(sys.argv) != 4:
        print("USAGE: python -m experiments.eval_neural_parser MODEL_DIR TEST.jsonl OUT.json", file=sys.stderr)
        return 2

    model_dir = Path(sys.argv[1])
    test_path = Path(sys.argv[2])
    out_path = Path(sys.argv[3])

    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(test_path)
    ds = Dataset.from_list([{"text": r["text"], "label": int(r["sid"])} for r in rows])

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    def tokenize(batch):
        return tok(batch["text"], truncation=True, padding="max_length", max_length=64)

    ds = ds.map(tokenize, batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    correct = 0
    total = 0

    with torch.no_grad():
        for ex in ds:
            input_ids = ex["input_ids"].unsqueeze(0)
            attn = ex["attention_mask"].unsqueeze(0)
            label = int(ex["label"])
            logits = model(input_ids=input_ids, attention_mask=attn).logits
            pred = int(torch.argmax(logits, dim=-1).item())
            correct += int(pred == label)
            total += 1

    acc = float(correct) / float(total) if total > 0 else 0.0

    out = {
        "NEURAL_PARSER_EVAL": {
            "model_dir": str(model_dir),
            "test_path": str(test_path),
            "n": total,
            "acc": acc,
        }
    }

    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[PASS] EVAL_NEURAL_PARSER: acc={acc:.6f} n={total} out={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
