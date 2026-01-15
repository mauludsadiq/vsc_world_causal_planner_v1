from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Tuple

REQ_FACTS = ("f0", "f1", "f2", "f3")

STYLE_NEUTRAL = 0
STYLE_FORMAL = 1

LEN_SHORT = 0
LEN_MED = 1
LEN_LONG = 2
LEN_XLONG = 3

@dataclass(frozen=True)
class SentenceState:
    fact_mask: int
    contradiction: int
    style: int
    length: int

    def hazard(self) -> int:
        return 1 if self.contradiction == 1 else 0

    def has_fact(self, i: int) -> bool:
        return (self.fact_mask >> i) & 1 == 1

    def with_fact(self, i: int) -> "SentenceState":
        return SentenceState(self.fact_mask | (1 << i), self.contradiction, self.style, self.length)

    def toggle_contradiction(self) -> "SentenceState":
        return SentenceState(self.fact_mask, 1 - self.contradiction, self.style, self.length)

    def set_style(self, style: int) -> "SentenceState":
        return SentenceState(self.fact_mask, self.contradiction, style, self.length)

    def shorten(self) -> "SentenceState":
        return SentenceState(self.fact_mask, self.contradiction, self.style, max(LEN_SHORT, self.length - 1))

    def lengthen(self) -> "SentenceState":
        return SentenceState(self.fact_mask, self.contradiction, self.style, min(LEN_XLONG, self.length + 1))

def enumerate_states() -> List[SentenceState]:
    out: List[SentenceState] = []
    for m in range(16):
        for c in (0, 1):
            for s in (STYLE_NEUTRAL, STYLE_FORMAL):
                for l in (LEN_SHORT, LEN_MED, LEN_LONG, LEN_XLONG):
                    out.append(SentenceState(m, c, s, l))
    return out
