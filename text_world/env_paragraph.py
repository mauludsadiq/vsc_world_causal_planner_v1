from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import random

from text_world.env_sentence import build_sentence_world
from text_world.paragraph import normalize_paragraph, ParagraphState

Prob = float
Dist = Dict[int, Prob]

def encode_action(slot: int, a: int) -> int:
    return slot * 16 + a

def decode_action(act: int) -> Tuple[int, int]:
    return (act // 16, act % 16)

@dataclass
class ParagraphWorldLazy:
    sw: object
    states: List[ParagraphState]
    index_of: Dict[ParagraphState, int]
    actions: List[int]

    # cache transitions only for visited (s,act)
    _T: Dict[Tuple[int, int], Dist]

def build_paragraph_world() -> ParagraphWorldLazy:
    sw = build_sentence_world()

    actions: List[int] = []
    for slot in (0, 1, 2):
        for a in range(9):
            actions.append(encode_action(slot, a))

    states: List[ParagraphState] = []
    index_of: Dict[ParagraphState, int] = {}

    # canonical start paragraph: (first sentence state)^3 normalized
    p0 = normalize_paragraph(sw.states[0], sw.states[0], sw.states[0])
    index_of[p0] = 0
    states.append(p0)

    return ParagraphWorldLazy(
        sw=sw,
        states=states,
        index_of=index_of,
        actions=actions,
        _T={},
    )

def _get_state_id(world: ParagraphWorldLazy, p: ParagraphState) -> int:
    pid = world.index_of.get(p)
    if pid is None:
        pid = len(world.states)
        world.index_of[p] = pid
        world.states.append(p)
    return pid

def transition_dist(world: ParagraphWorldLazy, s: int, act: int) -> Dist:
    key = (s, act)
    cached = world._T.get(key)
    if cached is not None:
        return cached

    slot, a = decode_action(act)
    p = world.states[s]
    sw = world.sw

    base = p.s1 if slot == 0 else (p.s2 if slot == 1 else p.s3)
    base_idx = sw.index_of[base]
    dist_sent = sw.T[(base_idx, a)]  # sentence-index -> prob

    dist_para: Dist = {}
    for sp_sent_idx, prob in dist_sent.items():
        sp_sent = sw.states[sp_sent_idx]
        if slot == 0:
            pp = normalize_paragraph(sp_sent, p.s2, p.s3)
        elif slot == 1:
            pp = normalize_paragraph(p.s1, sp_sent, p.s3)
        else:
            pp = normalize_paragraph(p.s1, p.s2, sp_sent)
        pid2 = _get_state_id(world, pp)
        dist_para[pid2] = dist_para.get(pid2, 0.0) + prob

    world._T[key] = dist_para
    return dist_para

def sample_transition(world: ParagraphWorldLazy, s: int, act: int, rng: random.Random) -> int:
    dist = transition_dist(world, s, act)
    r = rng.random()
    acc = 0.0
    for sp, p in dist.items():
        acc += p
        if r <= acc:
            return sp
    return next(iter(dist.keys()))

def mle_estimate_T(world: ParagraphWorldLazy, transitions: List[Tuple[int, int, int]]) -> Dict[Tuple[int, int], Dist]:
    counts: Dict[Tuple[int, int], Dict[int, int]] = {}
    totals: Dict[Tuple[int, int], int] = {}

    for s, act, sp in transitions:
        key = (s, act)
        if key not in counts:
            counts[key] = {}
            totals[key] = 0
        counts[key][sp] = counts[key].get(sp, 0) + 1
        totals[key] += 1

    hat: Dict[Tuple[int, int], Dist] = {}
    for key, m in counts.items():
        tot = totals[key]
        hat[key] = {sp: c / tot for sp, c in m.items()}
    return hat

def mean_l1_over_keys(world: ParagraphWorldLazy, hat: Dict[Tuple[int, int], Dist]) -> float:
    keys = list(hat.keys())
    if not keys:
        return 1.0
    total = 0.0
    for (s, act) in keys:
        true = transition_dist(world, s, act)
        est = hat[(s, act)]
        all_keys = set(true.keys()) | set(est.keys())
        l1 = 0.0
        for k in all_keys:
            l1 += abs(true.get(k, 0.0) - est.get(k, 0.0))
        total += l1
    return total / len(keys)
