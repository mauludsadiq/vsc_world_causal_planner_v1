from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import random

from text_world.state import SentenceState, enumerate_states, STYLE_FORMAL, LEN_LONG, LEN_SHORT, LEN_MED
from text_world.actions import (
    ALL_ACTIONS,
    A_ADD_F0, A_ADD_F1, A_ADD_F2,
    A_TOGGLE_CONTRADICTION,
    A_SET_FORMAL, A_SET_NEUTRAL,
    A_SHORTEN, A_LENGTHEN, A_NOOP,
)

Prob = float
Dist = Dict[int, Prob]

@dataclass(frozen=True)
class SentenceWorld:
    states: List[SentenceState]
    index_of: Dict[SentenceState, int]
    T: Dict[Tuple[int, int], Dist]

def _normalize(d: Dist) -> Dist:
    s = sum(d.values())
    if s <= 0:
        raise ValueError("distribution has nonpositive mass")
    return {k: v / s for k, v in d.items()}

def _contradiction_injection_prob(st: SentenceState) -> float:
    base = 0.08
    if st.style == STYLE_FORMAL:
        base *= 0.6
    if st.length == LEN_LONG:
        base *= 0.7
    if st.length == LEN_SHORT:
        base *= 1.2
    return min(0.20, max(0.0, base))

def build_sentence_world() -> SentenceWorld:
    states = enumerate_states()
    index_of = {s: i for i, s in enumerate(states)}
    T: Dict[Tuple[int, int], Dist] = {}

    for sid, st in enumerate(states):
        for a in ALL_ACTIONS:
            dist: Dist = {}
            if a in (A_ADD_F0, A_ADD_F1, A_ADD_F2):
                i = a
                st2 = st.with_fact(i)
                p_inj = _contradiction_injection_prob(st2)
                if st2.contradiction == 1:
                    dist[index_of[st2]] = 1.0
                else:
                    dist[index_of[st2]] = 1.0 - p_inj
                    dist[index_of[st2.toggle_contradiction()]] = p_inj

            elif a == A_TOGGLE_CONTRADICTION:
                st2 = st.toggle_contradiction()
                dist[index_of[st2]] = 1.0

            elif a == A_SET_FORMAL:
                st2 = st.set_style(STYLE_FORMAL)
                dist[index_of[st2]] = 1.0

            elif a == A_SET_NEUTRAL:
                st2 = st.set_style(0)
                dist[index_of[st2]] = 1.0

            elif a == A_SHORTEN:
                st2 = st.shorten()
                dist[index_of[st2]] = 1.0

            elif a == A_LENGTHEN:
                st2 = st.lengthen()
                dist[index_of[st2]] = 1.0

            elif a == A_NOOP:
                p_typo = 0.01
                if st.contradiction == 1:
                    dist[sid] = 1.0
                else:
                    dist[sid] = 1.0 - p_typo
                    dist[index_of[st.toggle_contradiction()]] = p_typo

            else:
                raise ValueError(f"unknown action {a}")

            T[(sid, a)] = _normalize(dist)

    return SentenceWorld(states=states, index_of=index_of, T=T)

def sample_transition(world: SentenceWorld, sid: int, a: int, rng: random.Random) -> int:
    dist = world.T[(sid, a)]
    r = rng.random()
    cum = 0.0
    for nsid, p in dist.items():
        cum += p
        if r <= cum:
            return nsid
    return nsid

def mle_estimate_T(world: SentenceWorld, transitions: List[Tuple[int, int, int]]) -> Dict[Tuple[int, int], Dict[int, float]]:
    counts: Dict[Tuple[int, int], Dict[int, int]] = {}
    totals: Dict[Tuple[int, int], int] = {}
    for s, a, sp in transitions:
        key = (s, a)
        if key not in counts:
            counts[key] = {}
            totals[key] = 0
        counts[key][sp] = counts[key].get(sp, 0) + 1
        totals[key] += 1

    hat: Dict[Tuple[int, int], Dict[int, float]] = {}
    for key, d in counts.items():
        tot = totals[key]
        hat[key] = {sp: c / tot for sp, c in d.items()}
    return hat

def mean_l1_distance(world: SentenceWorld, hat: Dict[Tuple[int, int], Dict[int, float]]) -> float:
    acc = 0.0
    n = 0
    for (s, a), true_dist in world.T.items():
        est = hat.get((s, a), {})
        l1 = 0.0
        keys = set(true_dist.keys()) | set(est.keys())
        for k in keys:
            l1 += abs(true_dist.get(k, 0.0) - est.get(k, 0.0))
        acc += l1
        n += 1
    return acc / max(1, n)
