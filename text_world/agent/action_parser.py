from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import re
import random

from text_world.actions import ALL_ACTIONS
from text_world.agent.neural_parser import NeuralParser


@dataclass(frozen=True)
class ActionParserOut:
    action_ids: List[int]
    scores: List[float]
    mode: str


def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _symbolic_action(text: str) -> int:
    t = _normalize(text)

    if any(x in t for x in ["exit", "quit", "leave"]):
        return 8

    if any(x in t for x in ["look", "look around", "inspect", "examine"]):
        return 2

    if any(x in t for x in ["open door", "open the door", "unlock door"]):
        return 0

    if any(x in t for x in ["close door", "shut the door"]):
        return 1

    if any(x in t for x in ["take", "pick up", "grab"]):
        return 3

    if any(x in t for x in ["drop", "put down"]):
        return 4

    if any(x in t for x in ["use", "activate"]):
        return 5

    if any(x in t for x in ["talk", "speak"]):
        return 6

    if any(x in t for x in ["help", "what can i do"]):
        return 7

    return -1


_CANONICAL_BANK: Dict[int, List[str]] = {
    0: ["open the door", "unlock the door", "open door"],
    1: ["close the door", "shut the door"],
    2: ["look around", "inspect the room", "examine area"],
    3: ["take the key", "pick up the item", "grab it"],
    4: ["drop the item", "put it down"],
    5: ["use the key", "activate the device"],
    6: ["talk to them", "speak to the person"],
    7: ["help", "what can i do"],
    8: ["exit", "quit", "leave"],
}


def _sid_signature(p: NeuralParser, text: str, seed: int, k: int = 32) -> Dict[int, float]:
    out = p.predict_sid256_topk(text, k=k, seed=seed)
    sig: Dict[int, float] = {}
    for sid, score in zip(out.sid_ids, out.scores):
        sig[int(sid)] = float(score)
    return sig


def _dot(sig_a: Dict[int, float], sig_b: Dict[int, float]) -> float:
    keys = set(sig_a.keys()) & set(sig_b.keys())
    return float(sum(sig_a[k] * sig_b[k] for k in keys))


def predict_action9_topk(
    text: str,
    k: int = 5,
    seed: int = 0,
    parser: NeuralParser | None = None,
) -> ActionParserOut:
    """
    Deterministic action parser into action ids in ALL_ACTIONS.

    Layer 1: symbolic rules
    Layer 2: kNN fallback using SID signature similarity (no new training)
    """

    sym = _symbolic_action(text)
    if sym in ALL_ACTIONS:
        topk = [sym] + [a for a in ALL_ACTIONS if a != sym]
        topk = topk[: max(1, min(int(k), len(ALL_ACTIONS)))]
        scores = [1.0] + [0.0] * (len(topk) - 1)
        return ActionParserOut(action_ids=topk, scores=scores, mode="symbolic")

    if parser is None:
        parser = NeuralParser(NeuralParser.default_model_dir(), device="cpu")

    qsig = _sid_signature(parser, text, seed=seed, k=32)

    scored: List[Tuple[int, float]] = []
    for act in ALL_ACTIONS:
        phrases = _CANONICAL_BANK.get(int(act), [])
        if len(phrases) == 0:
            scored.append((int(act), float("-inf")))
            continue

        best = float("-inf")
        for ph in phrases:
            psig = _sid_signature(parser, ph, seed=seed, k=32)
            s = _dot(qsig, psig)
            if s > best:
                best = s
        scored.append((int(act), float(best)))

    rng = random.Random(int(seed))
    jittered = [(a, sc + (rng.random() * 1e-9)) for (a, sc) in scored]
    jittered.sort(key=lambda x: x[1], reverse=True)

    kk = max(1, min(int(k), len(ALL_ACTIONS)))
    top = jittered[:kk]

    return ActionParserOut(
        action_ids=[int(a) for (a, _) in top],
        scores=[float(sc) for (_, sc) in top],
        mode="knn_sid_signature",
    )
