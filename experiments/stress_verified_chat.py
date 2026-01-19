from __future__ import annotations

import argparse
import glob
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _find_latest_checkpoint(root: str) -> str:
    pat = os.path.join(root, "checkpoint-*")
    cks = sorted(glob.glob(pat), key=lambda p: int(os.path.basename(p).split("-")[-1]))
    if not cks:
        raise FileNotFoundError(f"no checkpoints found under {root!r} (expected {pat!r})")
    return cks[-1]


def _softmax_max(logits: torch.Tensor) -> Tuple[float, int]:
    probs = torch.softmax(logits, dim=-1)
    pmax, idx = torch.max(probs, dim=-1)
    return float(pmax.item()), int(idx.item())


def _load_jsonl(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if limit is not None and len(out) >= limit:
                break
    return out


def _extract_text(row: Dict[str, Any]) -> str:
    text = None
    for k in ("text", "input", "utterance", "prompt", "surface"):
        if k in row and isinstance(row[k], str):
            text = row[k]
            break
    if text is None:
        raise KeyError(
            "could not find text field in row; expected one of: text/input/utterance/prompt/surface"
        )
    return text


def _best_gold_label(row: Any, num_labels: int) -> Optional[int]:
    # Recursive scan: pick the best int candidate in [0, num_labels)
    # Prefer keys like sid_out/label/target/action_id when present.
    best: Optional[Tuple[int, int]] = None  # (score, value)

    def score_key(k: str) -> int:
        kk = (k or "").lower()
        if "sid_out" in kk:
            return 100
        if kk == "sid":
            return 95
        if "label" in kk:
            return 90
        if "target" in kk:
            return 80
        if "action_id" in kk:
            return 75
        if kk == "y":
            return 70
        if kk == "action":
            return 60
        if "sid" in kk:
            return 50
        return 0

    def visit(obj: Any, key_hint: str = "") -> None:
        nonlocal best
        if isinstance(obj, dict):
            for k, v in obj.items():
                visit(v, k)
        elif isinstance(obj, list):
            for v in obj:
                visit(v, key_hint)
        else:
            val = None
            if isinstance(obj, int):
                val = obj
            elif isinstance(obj, str) and obj.isdigit():
                val = int(obj)

            if val is None:
                return

            if 0 <= val < num_labels:
                s = score_key(key_hint)
                if best is None or (s > best[0]):
                    best = (s, val)

    visit(row)
    return best[1] if best is not None else None


def _get_text_and_gold(row: Dict[str, Any], num_labels: int) -> Tuple[str, Optional[int]]:
    text = _extract_text(row)
    gold = _best_gold_label(row, num_labels)
    return text, gold
@dataclass
class TraceRow:
    i: int
    decision: str
    pmax: float
    a_hat: int
    a_star: Optional[int]
    correct: Optional[bool]
    wrong_accept: Optional[bool]
    text: str


def main() -> int:
    ap = argparse.ArgumentParser(prog="stress_verified_chat", add_help=True)

    ap.add_argument("out_json", type=str)
    ap.add_argument("seed", type=int)
    ap.add_argument("tau_conf", type=float)

    ap.add_argument("--data_jsonl", type=str, default=None)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--shuffle", action="store_true")

    ap.add_argument("--ckpt_root", type=str, default=os.environ.get("PARSER_CKPT_ROOT", "models/neural_parser"))
    ap.add_argument("--ckpt_path", type=str, default=os.environ.get("PARSER_CKPT_PATH", ""))

    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    ckpt_path = args.ckpt_path.strip() or _find_latest_checkpoint(args.ckpt_root)

    tok = AutoTokenizer.from_pretrained(ckpt_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_path)
    model.eval()

    rows: List[Dict[str, Any]] = []
    if args.data_jsonl:
        rows = _load_jsonl(args.data_jsonl, limit=None)
    else:
        raise SystemExit(
            "data_jsonl is required for â vs a* + wrong-accept statistics. "
            "Pass --data_jsonl data/enhanced/test.jsonl (or similar)."
        )

    if args.shuffle:
        random.shuffle(rows)

    if args.limit is not None:
        rows = rows[: max(1, int(args.limit))]

    traces: List[TraceRow] = []

    n_total = 0
    n_accept = 0
    n_reject = 0

    n_gold_known = 0
    n_accept_gold_known = 0

    n_wrong_accept = 0
    n_correct_accept = 0

    for i, row in enumerate(rows):
        # finite action space cardinality for gold-label validation
        num_labels = 256

        text, gold = _get_text_and_gold(row, num_labels)

        with torch.no_grad():
            enc = tok(text, return_tensors="pt", truncation=True, max_length=256)
            logits = model(**enc).logits.squeeze(0)
            pmax, a_hat = _softmax_max(logits)

        decision = "ACCEPT" if pmax >= float(args.tau_conf) else "REJECT"

        a_star = gold if gold is not None else None
        correct = (a_hat == a_star) if a_star is not None else None

        wrong_accept = None
        if a_star is not None:
            n_gold_known += 1
            if decision == "ACCEPT":
                n_accept_gold_known += 1
                wrong_accept = (a_hat != a_star)
                if wrong_accept:
                    n_wrong_accept += 1
                else:
                    n_correct_accept += 1

        n_total += 1
        if decision == "ACCEPT":
            n_accept += 1
        else:
            n_reject += 1

        tr = TraceRow(
            i=i,
            decision=decision,
            pmax=pmax,
            a_hat=a_hat,
            a_star=a_star,
            correct=correct,
            wrong_accept=wrong_accept,
            text=text,
        )
        traces.append(tr)

        if a_star is None:
            print(f"{decision}  pmax={pmax:.6f}  a_hat={a_hat}  a_star=?  text={text}")
        else:
            mark = "OK" if correct else "BAD"
            print(f"{decision}  pmax={pmax:.6f}  a_hat={a_hat}  a_star={a_star}  {mark}  text={text}")

    gamma = (n_accept / n_total) if n_total > 0 else 0.0
    R_sel = (n_wrong_accept / n_accept_gold_known) if n_accept_gold_known > 0 else 0.0
    P_W = (n_wrong_accept / n_total) if n_total > 0 else 0.0

    summary = {
        "seed": int(args.seed),
        "tau_conf": float(args.tau_conf),
        "checkpoint": ckpt_path,
        "counts": {
            "n_total": int(n_total),
            "n_accept": int(n_accept),
            "n_reject": int(n_reject),
            "n_gold_known": int(n_gold_known),
            "n_accept_gold_known": int(n_accept_gold_known),
            "n_wrong_accept": int(n_wrong_accept),
            "n_correct_accept": int(n_correct_accept),
        },
        "metrics": {
            "gamma": float(gamma),
            "R_sel": float(R_sel),
            "P_W": float(P_W),
            "identity_PW_equals_gamma_times_Rsel": float(abs(P_W - (gamma * R_sel))),
        },
        "definition": {
            "W": "ACCEPT and (a_hat != a_star)",
            "ACCEPT": "pmax >= tau_conf",
            "pmax": "max_a p_theta(a|x)",
            "P(W)": "n_wrong_accept / n_total",
            "gamma": "n_accept / n_total",
            "R_sel": "n_wrong_accept / n_accept_gold_known",
            "identity": "P(W) = gamma * R_sel (exact via counts)",
        },
        "traces": [
            {
                "i": t.i,
                "decision": t.decision,
                "pmax": t.pmax,
                "a_hat": t.a_hat,
                "a_star": t.a_star,
                "correct": t.correct,
                "wrong_accept": t.wrong_accept,
                "text": t.text,
            }
            for t in traces
        ],
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("")
    print("SUMMARY")
    print(f"n_total={n_total}  n_accept={n_accept}  n_reject={n_reject}")
    print(f"gamma=P(ACCEPT)={gamma:.6f}")
    print(f"R_sel=P(wrong | ACCEPT)={R_sel:.6f}")
    print(f"P(W)=P(wrong accept)={P_W:.6f}")
    print(f"CHECK  |P(W) - gamma*R_sel| = {abs(P_W - (gamma*R_sel)):.12f}")
    print(f"WROTE_JSON={args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
