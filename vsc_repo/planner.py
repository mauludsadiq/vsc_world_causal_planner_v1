from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Tuple
import numpy as np

@dataclass(frozen=True)
class PlanResult:
    V: np.ndarray
    Q: np.ndarray
    policy: np.ndarray

def value_iteration(T: np.ndarray, R: np.ndarray, gamma: float = 0.95, tol: float = 1e-10, max_iter: int = 100000) -> PlanResult:
    nS, nA, _ = T.shape
    V = np.zeros(nS, float)
    for _ in range(max_iter):
        Q = R + gamma * (T @ V)
        V_new = np.max(Q, axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    Q = R + gamma * (T @ V)
    policy = np.argmax(Q, axis=1).astype(int)
    return PlanResult(V=V, Q=Q, policy=policy)

def eval_policy_exact(T: np.ndarray, R: np.ndarray, policy: np.ndarray, gamma: float = 0.95, start_dist: np.ndarray | None = None) -> float:
    nS = T.shape[0]
    P_pi = np.zeros((nS, nS), float)
    r_pi = np.zeros(nS, float)
    for s in range(nS):
        a = int(policy[s])
        P_pi[s, :] = T[s, a, :]
        r_pi[s] = R[s, a]
    V = np.linalg.solve(np.eye(nS) - gamma * P_pi, r_pi)
    if start_dist is None:
        start_dist = np.ones(nS, float) / nS
    return float(start_dist @ V)

def brute_force_optimal_policy(T: np.ndarray, R: np.ndarray, gamma: float = 0.95, start_dist: np.ndarray | None = None) -> Tuple[np.ndarray, float]:
    nS, nA, _ = T.shape
    best_J = -1e100
    best_pi = None
    for acts in product(range(nA), repeat=nS):
        pi = np.array(acts, int)
        J = eval_policy_exact(T, R, pi, gamma=gamma, start_dist=start_dist)
        if J > best_J + 1e-12:
            best_J = J
            best_pi = pi
    return best_pi, float(best_J)
