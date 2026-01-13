from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import re

from text_world.state import SentenceState

@dataclass(frozen=True)
class RealTextFeatures:
    contradiction: int
    style: int
    length: int
    fact_mask: int

def extract_features(text: str) -> RealTextFeatures:
    t = text.strip()
    style = 1 if (len(t) > 0 and t[0].isupper()) else 0
    sents = [x.strip() for x in re.split(r"[.!?]+", t) if x.strip()]
    length = 0 if len(sents) <= 1 else 1 if len(sents) == 2 else 2
    neg = 1 if re.search(r"\bnot\b|\bnever\b|\bno\b", t.lower()) else 0
    contradiction = 1 if (neg == 1 and re.search(r"\bbut\b|\bhowever\b|\byet\b", t.lower())) else 0
    fact_mask = 0
    if re.search(r"\bwireless\b|\bcharging\b", t.lower()):
        fact_mask |= 1
    if re.search(r"\bbattery\b|\ball-day\b|\blasts\b", t.lower()):
        fact_mask |= 2
    if re.search(r"\bshatter\b|\bglass\b|\bscreen\b|\bdisplay\b", t.lower()):
        fact_mask |= 4
    return RealTextFeatures(contradiction=contradiction, style=style, length=length, fact_mask=fact_mask)

def to_sentence_state(text: str) -> SentenceState:
    f = extract_features(text)
    return SentenceState(fact_mask=f.fact_mask, contradiction=f.contradiction, style=f.style, length=f.length)
