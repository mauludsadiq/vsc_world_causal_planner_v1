from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np

@dataclass(frozen=True)
class ConstraintResult:
    chosen_policy: np.ndarray
    chosen_return: float
    chosen_risk: float

def estimate_risk_mc(T: np.ndarray, policy: np.ndarray, harm_states: Tuple[int, ...], start_dist: np.ndarray,
                     horizon: int = 30, n_mc: int = 20000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    harm_set = set(int(s) for s in harm_states)
    nS = T.shape[0]
    hits = 0
    for _ in range(n_mc):
        s = int(rng.choice(nS, p=start_dist))
        hit = False
        for _t in range(horizon):
            a = int(policy[s])
            s = int(rng.choice(nS, p=T[s, a]))
            if s in harm_set:
                hit = True
                break
        hits += 1 if hit else 0
    return float(hits / n_mc)

def constrained_select(policies: list[np.ndarray], returns: list[float], risks: list[float], epsilon: float) -> ConstraintResult:
    best_idx = None
    best_ret = -1e100
    for i in range(len(policies)):
        if risks[i] <= epsilon and returns[i] > best_ret:
            best_ret = returns[i]
            best_idx = i
    if best_idx is None:
        best_idx = int(np.argmin(np.array(risks)))
    return ConstraintResult(
        chosen_policy=policies[best_idx],
        chosen_return=float(returns[best_idx]),
        chosen_risk=float(risks[best_idx]),
    )
