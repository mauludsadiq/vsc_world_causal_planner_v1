from __future__ import annotations

import random
from typing import Any, List, Tuple

from text_world.env_block import sample_transition


def _state_return(obj: Any) -> float:
    if hasattr(obj, "kappa"):
        try:
            return float(getattr(obj, "kappa"))
        except Exception:
            return 0.0
    return 0.0


def _state_risk(obj: Any) -> float:
    if hasattr(obj, "kappa"):
        try:
            return 1.0 if float(getattr(obj, "kappa")) == 0.0 else 0.0
        except Exception:
            return 0.0
    if hasattr(obj, "hazard"):
        try:
            return 1.0 if bool(getattr(obj, "hazard")) else 0.0
        except Exception:
            return 0.0
    if hasattr(obj, "risk"):
        try:
            return float(getattr(obj, "risk"))
        except Exception:
            return 0.0
    return 0.0


class BlockWorldAdapter:
    def __init__(self, world: Any):
        self.world = world

    def enumerate_actions(self, s: int) -> List[int]:
        w = self.world
        if hasattr(w, "actions"):
            return [int(a) for a in list(w.actions)]
        raise AttributeError("env_block world expected to expose .actions")

    def step(self, s: int, a: int, rng: random.Random) -> Tuple[int, float, float]:
        sp = sample_transition(self.world, int(s), int(a), rng)
        obj = self.world.states[int(sp)]
        dr = _state_return(obj)
        rk = _state_risk(obj)
        return int(sp), float(dr), float(rk)
