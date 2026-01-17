from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from text_world.neural_inverse import NeuralInverse, decode_text_to_sid


@dataclass
class _FakeTok:
    def __call__(self, text: str, truncation: bool, max_length: int, padding: str, return_tensors: str) -> Dict[str, Any]:
        ids = torch.zeros((1, max_length), dtype=torch.long)
        mask = torch.ones((1, max_length), dtype=torch.long)
        return {"input_ids": ids, "attention_mask": mask}


@dataclass
class _Out:
    logits: torch.Tensor


@dataclass
class _FakeModel:
    logits_row: torch.Tensor

    def __call__(self, **kwargs: Any) -> _Out:
        return _Out(logits=self.logits_row)

    def eval(self) -> "_FakeModel":
        return self

    def to(self, device: str) -> "_FakeModel":
        return self


def test_decode_symbolic_first_wins() -> None:
    def parser_clean(s: str) -> Optional[int]:
        return 7

    out = decode_text_to_sid(
        "anything",
        seed=0,
        symbolic_first=True,
        parser_clean=parser_clean,
        neural=None,
    )
    assert out["mode"] == "symbolic"
    assert out["sid_hat"] == 7
    assert out["p_top1"] == 1.0
    assert out["entropy"] == 0.0


def test_decode_neural_fallback_accepts_high_confidence() -> None:
    def parser_clean(s: str) -> Optional[int]:
        return None

    logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    neural = NeuralInverse(
        model_dir="FAKE",
        tok=_FakeTok(),
        model=_FakeModel(logits_row=logits),
        device="cpu",
    )

    out = decode_text_to_sid(
        "hello world",
        seed=0,
        symbolic_first=True,
        parser_clean=parser_clean,
        neural=neural,
        tau_p=0.90,
        tau_margin=0.10,
    )

    assert out["mode"] == "neural"
    assert out["sid_hat"] == 0
    assert out["p_top1"] >= 0.90
    assert out["margin"] >= 0.10
    assert out["entropy"] >= 0.0


def test_decode_rejects_low_confidence() -> None:
    def parser_clean(s: str) -> Optional[int]:
        return None

    logits = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    neural = NeuralInverse(
        model_dir="FAKE",
        tok=_FakeTok(),
        model=_FakeModel(logits_row=logits),
        device="cpu",
    )

    out = decode_text_to_sid(
        "hello world",
        seed=0,
        symbolic_first=True,
        parser_clean=parser_clean,
        neural=neural,
        tau_p=0.90,
        tau_margin=0.10,
    )

    assert out["mode"] == "reject"
    assert out["reason"] == "low_confidence"
    assert out["sid_hat"] is not None
    assert out["p_top1"] <= 0.90
