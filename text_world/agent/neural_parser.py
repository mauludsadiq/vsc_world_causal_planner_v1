from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import os


@dataclass(frozen=True)
class ParserOut:
    action_ids: List[int]
    scores: List[float]  # higher = more confident


class NeuralParser:
    """
    Tiny loader wrapper for a trained HF sequence classification checkpoint.

    Expected:
      - model is a classifier over discrete action ids [0..num_labels-1]
      - checkpoint dir contains config.json, tokenizer files, model weights

    Determinism:
      - inference uses CPU by default
      - eval() + no_grad()
      - torch manual seed set in predict_topk()
    """

    def __init__(self, model_dir: str, device: str = "cpu") -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except Exception as e:
            raise RuntimeError(
                "NeuralParser requires torch + transformers installed in this venv."
            ) from e

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()

        self.device = device
        self.model.to(self.device)

        # Cache num_labels if present
        try:
            self.num_labels = int(getattr(self.model.config, "num_labels"))
        except Exception:
            self.num_labels = None

    @staticmethod
    def default_model_dir() -> str:
        # Allow override without code edits
        return os.environ.get("PARSER_MODEL_DIR", "models/neural_parser_resume")

    def predict_topk(
        self,
        text: str,
        k: int = 5,
        seed: int = 0,
        allowed_actions: Optional[Sequence[int]] = None,
    ) -> ParserOut:
        """
        Return top-k action ids.

        Mandatory masking:
          - If caller does not supply allowed_actions, we default to text_world.actions.ALL_ACTIONS.
          - This prevents accidental leakage from classifier label space (e.g. 256) into planner action space (e.g. 9).

        Determinism:
          - manual_seed
          - eval() + no_grad()
        """
        torch = self.torch

        # Deterministic inference
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

        for k_ in enc:
            enc[k_] = enc[k_].to(self.device)

        with torch.no_grad():
            out = self.model(**enc)
            logits = out.logits.squeeze(0)  # [num_labels]

        if logits.dim() != 1:
            logits = logits.view(-1)

        # ---------------------------
        # Mandatory action-space mask
        # ---------------------------
        if allowed_actions is None:
            from text_world.actions import ALL_ACTIONS
            allowed_actions = ALL_ACTIONS

        allowed = sorted({int(a) for a in allowed_actions})
        if len(allowed) == 0:
            raise RuntimeError("NeuralParser.predict_topk: allowed_actions is empty")

        # Keep only actions that exist inside the classifier label space.
        allowed_in_range = [a for a in allowed if 0 <= a < logits.numel()]
        if len(allowed_in_range) == 0:
            raise RuntimeError(
                "NeuralParser.predict_topk: no allowed actions are within model label range"
            )

        mask = torch.full_like(logits, float("-inf"))
        for a in allowed_in_range:
            mask[a] = logits[a]
        logits = mask

        # Clamp k to the number of allowed actions so we never emit -inf indices.
        kk = int(min(max(1, int(k)), len(allowed_in_range)))

        vals, idxs = torch.topk(logits, k=kk)

        action_ids = [int(i) for i in idxs.detach().cpu().tolist()]
        scores = [float(v) for v in vals.detach().cpu().tolist()]

        return ParserOut(action_ids=action_ids, scores=scores)
