from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import random

from text_world.env_paragraph import (
    ParagraphWorldLazy,
    build_paragraph_world,
    sample_transition as sample_para_transition,
    transition_dist as para_transition_dist,
)

from text_world.paragraph import ParagraphState

Prob = float
Dist = Dict[int, Prob]

def encode_doc_action(block_idx: int, para_act: int) -> int:
    return block_idx * 64 + para_act

def decode_doc_action(act: int) -> Tuple[int, int]:
    return (act // 64, act % 64)

@dataclass(frozen=True)
class DocumentState:
    p0: ParagraphState
    p1: ParagraphState
    p2: ParagraphState
    p3: ParagraphState
    kappa: int  # 0,1,2

def paragraph_style(p: ParagraphState) -> int:
    # style bucket from first sentence (all can be mixed, but this is a deterministic probe)
    return p.s1.style

def paragraph_hazard(p: ParagraphState) -> int:
    return int(p.s1.hazard() == 1 or p.s2.hazard() == 1 or p.s3.hazard() == 1)

def doc_coherence_bucket(p0: ParagraphState, p1: ParagraphState, p2: ParagraphState, p3: ParagraphState) -> int:
    any_haz = paragraph_hazard(p0) or paragraph_hazard(p1) or paragraph_hazard(p2) or paragraph_hazard(p3)
    if any_haz:
        return 0
    rho_all_good = int(p0.rho == 2 and p1.rho == 2 and p2.rho == 2 and p3.rho == 2)
    styles = [paragraph_style(p0), paragraph_style(p1), paragraph_style(p2), paragraph_style(p3)]
    styles_match = int(all(s == styles[0] for s in styles))
    if rho_all_good and styles_match:
        return 2
    return 1

def normalize_document(p0: ParagraphState, p1: ParagraphState, p2: ParagraphState, p3: ParagraphState) -> DocumentState:
    k = doc_coherence_bucket(p0, p1, p2, p3)
    return DocumentState(p0=p0, p1=p1, p2=p2, p3=p3, kappa=k)

@dataclass
class DocumentWorldLazy:
    pw: ParagraphWorldLazy
    states: List[DocumentState]
    index_of: Dict[DocumentState, int]
    actions: List[int]
    _T: Dict[Tuple[int, int], Dist]

def build_document_world() -> DocumentWorldLazy:
    pw = build_paragraph_world()

    # start document: paragraph id 0 replicated
    p0 = pw.states[0]
    d0 = normalize_document(p0, p0, p0, p0)

    states = [d0]
    index_of = {d0: 0}

    actions: List[int] = []
    for block_idx in (0, 1, 2, 3):
        for para_act in pw.actions:
            actions.append(encode_doc_action(block_idx, para_act))

    return DocumentWorldLazy(pw=pw, states=states, index_of=index_of, actions=actions, _T={})

def _get_state_id(world: DocumentWorldLazy, d: DocumentState) -> int:
    did = world.index_of.get(d)
    if did is None:
        did = len(world.states)
        world.index_of[d] = did
        world.states.append(d)
    return did

def transition_dist(world: DocumentWorldLazy, s: int, act: int) -> Dist:
    key = (s, act)
    cached = world._T.get(key)
    if cached is not None:
        return cached

    block_idx, para_act = decode_doc_action(act)
    d = world.states[s]
    pw = world.pw

    # map paragraph object -> paragraph id in pw (lazy: ensure it exists)
    def pid_of(p: ParagraphState) -> int:
        pid = pw.index_of.get(p)
        if pid is None:
            pid = len(pw.states)
            pw.index_of[p] = pid
            pw.states.append(p)
        return pid

    pids = [pid_of(d.p0), pid_of(d.p1), pid_of(d.p2), pid_of(d.p3)]
    target_pid = pids[block_idx]

    dist_para = para_transition_dist(pw, target_pid, para_act)

    dist_doc: Dist = {}
    for pid2, prob in dist_para.items():
        new_paras = [d.p0, d.p1, d.p2, d.p3]
        new_paras[block_idx] = pw.states[pid2]
        d2 = normalize_document(new_paras[0], new_paras[1], new_paras[2], new_paras[3])
        did2 = _get_state_id(world, d2)
        dist_doc[did2] = dist_doc.get(did2, 0.0) + prob

    world._T[key] = dist_doc
    return dist_doc

def sample_transition(world: DocumentWorldLazy, s: int, act: int, rng: random.Random) -> int:
    dist = transition_dist(world, s, act)
    r = rng.random()
    acc = 0.0
    for sp, p in dist.items():
        acc += p
        if r <= acc:
            return sp
    return next(iter(dist.keys()))

def mle_estimate_T(world: DocumentWorldLazy, transitions: List[Tuple[int, int, int]]) -> Dict[Tuple[int, int], Dist]:
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

def mean_l1_over_keys(world: DocumentWorldLazy, hat: Dict[Tuple[int, int], Dist]) -> float:
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
