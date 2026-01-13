from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import random

from text_world.env_paragraph import (
    ParagraphWorldLazy,
    build_paragraph_world,
    transition_dist as para_transition_dist,
)

from text_world.paragraph import ParagraphState

Prob = float
Dist = Dict[int, Prob]

def encode_block_action(slot: int, para_act: int) -> int:
    return slot * 64 + para_act

def decode_block_action(act: int) -> Tuple[int, int]:
    return (act // 64, act % 64)

@dataclass(frozen=True)
class BlockState:
    paras: Tuple[ParagraphState, ...]
    kappa: int  # 0,1,2

def paragraph_style(p: ParagraphState) -> int:
    return p.s1.style

def paragraph_hazard(p: ParagraphState) -> int:
    return int(p.s1.hazard() == 1 or p.s2.hazard() == 1 or p.s3.hazard() == 1)

def block_coherence_bucket(paras: Tuple[ParagraphState, ...]) -> int:
    any_haz = 0
    for p in paras:
        if paragraph_hazard(p):
            any_haz = 1
            break
    if any_haz:
        return 0
    rho_all_good = int(all(p.rho == 2 for p in paras))
    styles = [paragraph_style(p) for p in paras]
    styles_match = int(all(s == styles[0] for s in styles))
    if rho_all_good and styles_match:
        return 2
    return 1

def normalize_block(paras: Tuple[ParagraphState, ...]) -> BlockState:
    k = block_coherence_bucket(paras)
    return BlockState(paras=paras, kappa=k)

@dataclass
class BlockWorldLazy:
    pw: ParagraphWorldLazy
    n: int
    states: List[BlockState]
    index_of: Dict[BlockState, int]
    actions: List[int]
    _T: Dict[Tuple[int, int], Dist]

def build_block_world(n: int) -> BlockWorldLazy:
    if n <= 0:
        raise ValueError("n must be >= 1")
    pw = build_paragraph_world()

    p0 = pw.states[0]
    paras = tuple([p0] * n)
    b0 = normalize_block(paras)

    states = [b0]
    index_of = {b0: 0}

    actions: List[int] = []
    for slot in range(n):
        for para_act in pw.actions:
            actions.append(encode_block_action(slot, para_act))

    return BlockWorldLazy(pw=pw, n=n, states=states, index_of=index_of, actions=actions, _T={})

def _get_state_id(world: BlockWorldLazy, b: BlockState) -> int:
    bid = world.index_of.get(b)
    if bid is None:
        bid = len(world.states)
        world.index_of[b] = bid
        world.states.append(b)
    return bid

def _pid_of(world: BlockWorldLazy, p: ParagraphState) -> int:
    pid = world.pw.index_of.get(p)
    if pid is None:
        pid = len(world.pw.states)
        world.pw.index_of[p] = pid
        world.pw.states.append(p)
    return pid

def transition_dist(world: BlockWorldLazy, s: int, act: int) -> Dist:
    key = (s, act)
    cached = world._T.get(key)
    if cached is not None:
        return cached

    slot, para_act = decode_block_action(act)
    b = world.states[s]

    # paragraph id of target slot (ensure present in pw)
    target_pid = _pid_of(world, b.paras[slot])

    dist_para = para_transition_dist(world.pw, target_pid, para_act)

    dist_block: Dist = {}
    for pid2, prob in dist_para.items():
        new_paras = list(b.paras)
        new_paras[slot] = world.pw.states[pid2]
        b2 = normalize_block(tuple(new_paras))
        bid2 = _get_state_id(world, b2)
        dist_block[bid2] = dist_block.get(bid2, 0.0) + prob

    world._T[key] = dist_block
    return dist_block

def sample_transition(world: BlockWorldLazy, s: int, act: int, rng: random.Random) -> int:
    dist = transition_dist(world, s, act)
    r = rng.random()
    acc = 0.0
    for sp, p in dist.items():
        acc += p
        if r <= acc:
            return sp
    return next(iter(dist.keys()))

def mle_estimate_T(world: BlockWorldLazy, transitions: List[Tuple[int, int, int]]) -> Dict[Tuple[int, int], Dist]:
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

def mean_l1_over_keys(world: BlockWorldLazy, hat: Dict[Tuple[int, int], Dist]) -> float:
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
