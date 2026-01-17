from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import math
import numpy as np

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None


@dataclass(frozen=True)
class NeuralInverse:
    model_dir: str
    tok: Any
    model: Any
    device: str


def softmax_np(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)


def entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def load_neural_inverse(model_dir: str) -> NeuralInverse:
    """
    Loads a HF sequence classifier checkpoint dir (e.g., models/neural_parser_resume or checkpoint-*).
    This MUST be a directory containing config.json with model_type and the classifier head.
    """
    if torch is None:
        raise RuntimeError("torch/transformers not available in this environment")

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    model.to(device)
    return NeuralInverse(model_dir=model_dir, tok=tok, model=model, device=device)


def _neural_decode(neural: NeuralInverse, text: str, max_len: int = 96) -> Dict[str, Any]:
    """
    Returns:
      - sid_hat: int
      - p_top1: float
      - p_top2: float
      - margin: float = p1 - p2
      - entropy: float
    """
    import torch  # local import (guarantees toolchain exists)

    enc = neural.tok(
        text,
        truncation=True,
        max_length=max_len,
        padding="max_length",
        return_tensors="pt",
    )
    enc = {k: v.to(neural.device) for k, v in enc.items()}

    with torch.no_grad():
        out = neural.model(**enc)
        logits = out.logits.detach().float().cpu().numpy()[0]

    p = softmax_np(logits)
    top = np.argsort(-p)
    sid1 = int(top[0])
    sid2 = int(top[1]) if len(top) > 1 else sid1

    p1 = float(p[sid1])
    p2 = float(p[sid2])
    return {
        "sid_hat": sid1,
        "p_top1": p1,
        "p_top2": p2,
        "margin": float(p1 - p2),
        "entropy": entropy(p),
    }


def decode_text_to_sid(
    text: str,
    *,
    seed: int,
    symbolic_first: bool,
    parser_clean: Callable[[str], Optional[int]],
    neural: Optional[NeuralInverse],
    tau_p: float = 0.90,
    tau_margin: float = 0.10,
) -> Dict[str, Any]:
    """
    Deterministic decode contract.

    Modes:
      - symbolic: parse_sentence_clean succeeded
      - neural: symbolic failed, neural succeeded AND confidence passes thresholds
      - reject: both failed OR neural confidence below thresholds
    """
    # 1) Symbolic-first
    if symbolic_first:
        try:
            sid = parser_clean(text)
        except Exception:
            sid = None
        if sid is not None:
            return {
                "mode": "symbolic",
                "sid_hat": int(sid),
                "p_top1": 1.0,
                "p_top2": 0.0,
                "margin": 1.0,
                "entropy": 0.0,
                "tau_p": tau_p,
                "tau_margin": tau_margin,
                "seed": int(seed),
            }

    # 2) Neural fallback
    if neural is None:
        return {
            "mode": "reject",
            "sid_hat": None,
            "reason": "no_neural_model_loaded",
            "tau_p": tau_p,
            "tau_margin": tau_margin,
            "seed": int(seed),
        }

    nd = _neural_decode(neural, text)
    ok = (nd["p_top1"] >= tau_p) and (nd["margin"] >= tau_margin)

    if ok:
        return {
            "mode": "neural",
            **nd,
            "tau_p": tau_p,
            "tau_margin": tau_margin,
            "seed": int(seed),
        }

    return {
        "mode": "reject",
        "sid_hat": int(nd["sid_hat"]),
        "p_top1": float(nd["p_top1"]),
        "p_top2": float(nd["p_top2"]),
        "margin": float(nd["margin"]),
        "entropy": float(nd["entropy"]),
        "reason": "low_confidence",
        "tau_p": tau_p,
        "tau_margin": tau_margin,
        "seed": int(seed),
    }
