from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

class _LazyTorch:
    def __init__(self) -> None:
        self._torch = None

    @property
    def torch(self):
        if self._torch is None:
            import torch
            self._torch = torch
        return self._torch


@dataclass(frozen=True)
class SidParserOut:
    sid_ids: List[int]
    scores: List[float]


class NeuralParser:
    """
    NeuralParser is a 256-way SID classifier.
    It predicts SID ids (0..255) from text.
    It does not predict action ids (0..8).
    """

    def __init__(self, model_dir: str, device: str = "cpu") -> None:
        self.device = device
        self._lazy = _LazyTorch()

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

        self.num_labels = int(self.model.config.num_labels)

    @staticmethod
    def default_model_dir() -> str:
        return "models/neural_parser_resume"

    @property
    def torch(self):
        return self._lazy.torch

    def predict_sid256_topk(
        self,
        text: str,
        k: int = 5,
        seed: int = 0,
    ) -> SidParserOut:
        """
        Return top-k SID ids in [0, num_labels).
        Deterministic:
          - manual_seed
          - eval() + no_grad()
        """
        torch = self.torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        )

        for key in enc:
            enc[key] = enc[key].to(self.device)

        with torch.no_grad():
            out = self.model(**enc)
            logits = out.logits.squeeze(0)

        if logits.dim() != 1:
            logits = logits.view(-1)

        kk = int(min(max(1, int(k)), int(logits.numel())))

        vals, idxs = torch.topk(logits, k=kk)

        sid_ids = [int(i) for i in idxs.detach().cpu().tolist()]
        scores = [float(v) for v in vals.detach().cpu().tolist()]
        return SidParserOut(sid_ids=sid_ids, scores=scores)
