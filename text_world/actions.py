from __future__ import annotations
from dataclasses import dataclass
from typing import List

A_ADD_F0 = 0
A_ADD_F1 = 1
A_ADD_F2 = 2
A_TOGGLE_CONTRADICTION = 3
A_SET_FORMAL = 4
A_SET_NEUTRAL = 5
A_SHORTEN = 6
A_LENGTHEN = 7
A_NOOP = 8

ACTION_NAMES = {
    A_ADD_F0: "ADD_FACT_0",
    A_ADD_F1: "ADD_FACT_1",
    A_ADD_F2: "ADD_FACT_2",
    A_TOGGLE_CONTRADICTION: "TOGGLE_CONTRADICTION",
    A_SET_FORMAL: "SET_FORMAL",
    A_SET_NEUTRAL: "SET_NEUTRAL",
    A_SHORTEN: "SHORTEN",
    A_LENGTHEN: "LENGTHEN",
    A_NOOP: "NOOP",
}

ALL_ACTIONS: List[int] = list(ACTION_NAMES.keys())

def action_name(a: int) -> str:
    return ACTION_NAMES[a]
