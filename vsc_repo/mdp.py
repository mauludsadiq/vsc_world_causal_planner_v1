from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np

@dataclass(frozen=True)
class MDPEnv:
    T: np.ndarray
    R: np.ndarray
    harm_states: Tuple[int, ...]
    start_dist: np.ndarray

    @property
    def nS(self) -> int:
        return int(self.T.shape[0])

    @property
    def nA(self) -> int:
        return int(self.T.shape[1])

def step_env(env: MDPEnv, s: int, a: int, rng: np.random.Generator):
    probs = env.T[s, a]
    s2 = int(rng.choice(env.nS, p=probs))
    r = float(env.R[s, a])
    harm = (s2 in set(env.harm_states))
    return s2, r, harm

def rollout(env: MDPEnv, horizon: int, seed: int = 0, policy: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    s = int(rng.choice(env.nS, p=env.start_dist))
    S_list, A_list, S2_list, R_list, H_list = [], [], [], [], []
    for _t in range(horizon):
        a = int(rng.integers(0, env.nA)) if policy is None else int(policy[s])
        s2, r, harm = step_env(env, s, a, rng)
        S_list.append(s); A_list.append(a); S2_list.append(s2); R_list.append(r); H_list.append(1 if harm else 0)
        s = s2
    return {"s": np.array(S_list, int), "a": np.array(A_list, int), "s2": np.array(S2_list, int),
            "r": np.array(R_list, float), "harm": np.array(H_list, int)}

def collect_rollouts(env: MDPEnv, n_rollouts: int, horizon: int, seed: int = 0) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    chunks = []
    for _i in range(n_rollouts):
        chunks.append(rollout(env, horizon=horizon, seed=int(rng.integers(0, 2**31-1))))
    out = {k: np.concatenate([c[k] for c in chunks], axis=0) for k in chunks[0].keys()}
    return out

def learn_model_from_transitions(trans: Dict[str, np.ndarray], nS: int, nA: int, laplace: float = 1.0):
    s = trans["s"]; a = trans["a"]; s2 = trans["s2"]; r = trans["r"]
    T_counts = np.zeros((nS, nA, nS), float)
    R_sum = np.zeros((nS, nA), float)
    R_cnt = np.zeros((nS, nA), float)
    for i in range(len(s)):
        T_counts[int(s[i]), int(a[i]), int(s2[i])] += 1.0
        R_sum[int(s[i]), int(a[i])] += float(r[i])
        R_cnt[int(s[i]), int(a[i])] += 1.0
    T_hat = T_counts + laplace
    T_hat = T_hat / T_hat.sum(axis=2, keepdims=True)
    R_hat = np.zeros((nS, nA), float)
    seen = R_cnt > 0
    R_hat[seen] = R_sum[seen] / R_cnt[seen]
    return T_hat, R_hat

def l1_transition_error(T_true: np.ndarray, T_hat: np.ndarray) -> float:
    return float(np.mean(np.abs(T_true - T_hat)))
