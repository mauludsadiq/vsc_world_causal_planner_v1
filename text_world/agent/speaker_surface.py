from __future__ import annotations

from typing import Literal
import random


Strength = Literal["plain", "soft", "firm"]


def surface_reply(base: str, *, seed: int, turn: int, sid: int, strength: Strength) -> str:
    r = random.Random((int(seed) * 1000003) + (int(turn) * 1009) + int(sid))
    if strength == "plain":
        return base
    if strength == "soft":
        opts = [
            base,
            "I think this is the claim: " + base,
            "Here is the statement: " + base,
        ]
        return opts[r.randrange(len(opts))]
    if strength == "firm":
        opts = [
            base,
            "This is the claim: " + base,
            "The system asserts: " + base,
        ]
        return opts[r.randrange(len(opts))]
    return base
