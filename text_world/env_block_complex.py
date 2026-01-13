from __future__ import annotations
from dataclasses import dataclass
import random
from typing import Dict, List, Tuple

from text_world.state import SentenceState
from text_world.env_sentence import enumerate_states as enum_sentence_states


@dataclass(frozen=True)
class ParagraphTagX:
    s1: SentenceState
    s2: SentenceState


@dataclass(frozen=True)
class BlockTagX:
    paras: Tuple[ParagraphTagX, ...]
    kappa: int
    tower: int
    gripper: int
    fragile: int


@dataclass(frozen=True)
class ComplexCfg:
    n: int
    towers: int
    fragile_prob: float
    grippers: int


def _mk_paragraphs(n: int) -> List[ParagraphTagX]:
    base = enum_sentence_states()
    paras: List[ParagraphTagX] = []
    for i in range(n):
        s1 = base[i % len(base)]
        s2 = base[(i * 7 + 3) % len(base)]
        paras.append(ParagraphTagX(s1=s1, s2=s2))
    return paras


def build_block_world_complex(cfg: ComplexCfg) -> Dict[str, object]:
    paras = _mk_paragraphs(cfg.n)

    states: List[BlockTagX] = []
    for kappa in (0, 1):
        for tower in range(cfg.towers):
            for gripper in range(cfg.grippers):
                for fragile in (0, 1):
                    states.append(BlockTagX(paras=tuple(paras), kappa=kappa, tower=tower, gripper=gripper, fragile=fragile))

    actions = list(range(cfg.n * 27))
    return {"states": states, "actions": actions, "n_paras": cfg.n, "cfg": cfg}


def _kappa_next(kappa: int, op: int) -> int:
    k = kappa
    if op in (0, 1, 2, 3, 4, 5):
        k = 1
    if op in (6, 7):
        k = 0
    return k


def sample_transition_complex(world: Dict[str, object], s: int, a: int, rng: random.Random) -> int:
    st: BlockTagX = world["states"][s]  # type: ignore[assignment]
    cfg: ComplexCfg = world["cfg"]  # type: ignore[assignment]
    op = a % 27

    k2 = _kappa_next(st.kappa, op)

    if st.fragile == 1 and rng.random() < cfg.fragile_prob:
        k2 = 0

    candidates = [i for i, x in enumerate(world["states"]) if x.kappa == k2]  # type: ignore[index]
    return candidates[(s + a) % len(candidates)]


def mle_estimate_T_from_anchors(
    world: Dict[str, object],
    anchors: List[int],
    reps_per_key: int,
    seed: int,
) -> Dict[Tuple[int, int], Dict[int, float]]:
    rng = random.Random(seed)
    counts: Dict[Tuple[int, int], Dict[int, int]] = {}
    actions: List[int] = world["actions"]  # type: ignore[assignment]

    for s in anchors:
        for a in actions:
            key = (s, a)
            inner: Dict[int, int] = {}
            for _ in range(reps_per_key):
                sp = sample_transition_complex(world, s, a, rng)
                inner[sp] = inner.get(sp, 0) + 1
            counts[key] = inner

    T_hat: Dict[Tuple[int, int], Dict[int, float]] = {}
    for key, inner in counts.items():
        tot = sum(inner.values())
        T_hat[key] = {sp: c / tot for sp, c in inner.items()}
    return T_hat


def mean_l1_over_anchors(world: Dict[str, object], anchors: List[int], reps_per_key: int, seed: int) -> float:
    T_hat = mle_estimate_T_from_anchors(world, anchors, reps_per_key, seed)
    actions: List[int] = world["actions"]  # type: ignore[assignment]

    l1_sum = 0.0
    denom = 0

    for s in anchors:
        for a in actions:
            key = (s, a)
            p_hat = T_hat[key]

            rng = random.Random(seed + 999)
            p_true: Dict[int, float] = {}
            for _ in range(reps_per_key):
                sp = sample_transition_complex(world, s, a, rng)
                p_true[sp] = p_true.get(sp, 0.0) + 1.0
            tot = sum(p_true.values())
            p_true = {sp: c / tot for sp, c in p_true.items()}

            support = set(p_hat.keys()) | set(p_true.keys())
            l1 = 0.0
            for sp in support:
                l1 += abs(p_true.get(sp, 0.0) - p_hat.get(sp, 0.0))

            l1_sum += l1
            denom += 1

    return l1_sum / max(1, denom)
