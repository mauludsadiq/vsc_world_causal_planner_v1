from __future__ import annotations
from typing import Dict, List, Tuple
from text_world.actions import A_ADD_F0, A_ADD_F1, A_ADD_F2, A_NOOP
from text_world.env_sentence_planning import PlanningWorld, facts_count

ACTIONS = (A_ADD_F0, A_ADD_F1, A_ADD_F2, A_NOOP)

def vi_planning_world(world: PlanningWorld, H: int) -> Tuple[List[float], List[int]]:
    nS = len(world.states)
    V_prev = [0.0] * nS
    policy = [A_NOOP] * nS

    for _ in range(1, H + 1):
        V = [0.0] * nS
        for s in range(nS):
            m = world.states[s]
            best = -1e18
            besta = A_NOOP
            for a in ACTIONS:
                exp = 0.0
                for sp, p in world.T[(m, a)].items():
                    exp += p * (facts_count(sp) + V_prev[sp])
                if exp > best:
                    best = exp
                    besta = a
            V[s] = best
            policy[s] = besta
        V_prev = V
    return V_prev, policy

def eval_policy_exact(world: PlanningWorld, policy: List[int], s0: int, H: int) -> float:
    nS = len(world.states)
    dist = [0.0] * nS
    dist[s0] = 1.0
    ret = 0.0

    for _t in range(H):
        next_dist = [0.0] * nS
        for s in range(nS):
            if dist[s] == 0.0:
                continue
            m = world.states[s]
            a = policy[s]
            for mp, p in world.T[(m, a)].items():
                next_dist[mp] += dist[s] * p
        for mp in range(nS):
            if next_dist[mp] != 0.0:
                ret += next_dist[mp] * facts_count(mp)
        dist = next_dist

    return ret

def brute_force_best_stationary(world: PlanningWorld, s0: int, H: int) -> Tuple[List[int], float]:
    nS = len(world.states)
    best_pi = None
    best_ret = -1e18

    # enumerate all stationary deterministic policies: ACTIONS^nS
    # 4^8 = 65536
    def rec(i: int, pi: List[int]):
        nonlocal best_pi, best_ret
        if i == nS:
            r = eval_policy_exact(world, pi, s0=s0, H=H)
            if r > best_ret:
                best_ret = r
                best_pi = pi.copy()
            return
        for a in ACTIONS:
            pi[i] = a
            rec(i + 1, pi)

    rec(0, [A_NOOP] * nS)
    return best_pi, best_ret
