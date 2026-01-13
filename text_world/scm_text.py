from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import random

@dataclass(frozen=True)
class TextSCMParams:
    p_z1: float = 0.55
    p_x1_z0: float = 0.20
    p_x1_z1: float = 0.80
    p_y1_x0_z0: float = 0.15
    p_y1_x1_z0: float = 0.55
    p_y1_x0_z1: float = 0.40
    p_y1_x1_z1: float = 0.85

def sample_observational(rng: random.Random, n: int, p: TextSCMParams):
    counts = {(z, x, y): 0 for z in (0,1) for x in (0,1) for y in (0,1)}
    for _ in range(n):
        z = 1 if rng.random() < p.p_z1 else 0
        px = p.p_x1_z1 if z == 1 else p.p_x1_z0
        x = 1 if rng.random() < px else 0
        if z == 0 and x == 0:
            py = p.p_y1_x0_z0
        elif z == 0 and x == 1:
            py = p.p_y1_x1_z0
        elif z == 1 and x == 0:
            py = p.p_y1_x0_z1
        else:
            py = p.p_y1_x1_z1
        y = 1 if rng.random() < py else 0
        counts[(z, x, y)] += 1
    return counts

def sample_interventional_doX(rng: random.Random, n: int, x_do: int, p: TextSCMParams):
    counts = {(z, y): 0 for z in (0,1) for y in (0,1)}
    for _ in range(n):
        z = 1 if rng.random() < p.p_z1 else 0
        x = x_do
        if z == 0 and x == 0:
            py = p.p_y1_x0_z0
        elif z == 0 and x == 1:
            py = p.p_y1_x1_z0
        elif z == 1 and x == 0:
            py = p.p_y1_x0_z1
        else:
            py = p.p_y1_x1_z1
        y = 1 if rng.random() < py else 0
        counts[(z, y)] += 1
    return counts

def estimate_pz(counts_zyx: Dict[Tuple[int,int,int], int]) -> Dict[int, float]:
    tot = sum(counts_zyx.values())
    ztot = {0: 0, 1: 0}
    for (z, x, y), c in counts_zyx.items():
        ztot[z] += c
    return {z: ztot[z] / tot for z in (0,1)}

def estimate_p_y_given_xz(counts_zyx: Dict[Tuple[int,int,int], int]) -> Dict[Tuple[int,int], float]:
    num = {(x,z): 0 for x in (0,1) for z in (0,1)}
    den = {(x,z): 0 for x in (0,1) for z in (0,1)}
    for (z, x, y), c in counts_zyx.items():
        den[(x,z)] += c
        if y == 1:
            num[(x,z)] += c
    out = {}
    for x in (0,1):
        for z in (0,1):
            out[(x,z)] = num[(x,z)] / max(1, den[(x,z)])
    return out

def estimate_naive_p_y_given_x(counts_zyx: Dict[Tuple[int,int,int], int]) -> Dict[int, float]:
    num = {0: 0, 1: 0}
    den = {0: 0, 1: 0}
    for (z, x, y), c in counts_zyx.items():
        den[x] += c
        if y == 1:
            num[x] += c
    return {x: num[x] / max(1, den[x]) for x in (0,1)}

def backdoor_do_effect(counts_zyx: Dict[Tuple[int,int,int], int]) -> Dict[int, float]:
    pz = estimate_pz(counts_zyx)
    py_xz = estimate_p_y_given_xz(counts_zyx)
    out = {}
    for x in (0,1):
        out[x] = sum(py_xz[(x,z)] * pz[z] for z in (0,1))
    return out

def true_do_effect_from_interventional(counts_zy: Dict[Tuple[int,int], int]) -> float:
    tot = sum(counts_zy.values())
    y1 = sum(c for (z,y), c in counts_zy.items() if y == 1)
    return y1 / max(1, tot)
