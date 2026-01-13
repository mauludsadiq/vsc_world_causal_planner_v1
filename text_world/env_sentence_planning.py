from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

from text_world.actions import A_ADD_F0, A_ADD_F1, A_ADD_F2, A_NOOP

Prob = float
Dist = Dict[int, Prob]

@dataclass(frozen=True)
class PlanningWorld:
    states: List[int]
    T: Dict[Tuple[int, int], Dist]

def build_sentence_planning_world() -> PlanningWorld:
    states = list(range(8))  # fact_mask only

    T: Dict[Tuple[int, int], Dist] = {}
    for m in states:
        for a in (A_ADD_F0, A_ADD_F1, A_ADD_F2, A_NOOP):
            if a == A_NOOP:
                T[(m, a)] = {m: 1.0}
            else:
                i = a  # 0,1,2 correspond to fact indices
                mp = m | (1 << i)
                T[(m, a)] = {mp: 1.0}
    return PlanningWorld(states=states, T=T)

def facts_count(mask: int) -> int:
    return (mask & 1) + ((mask >> 1) & 1) + ((mask >> 2) & 1)
