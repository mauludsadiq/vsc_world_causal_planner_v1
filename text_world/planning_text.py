from __future__ import annotations
from typing import Dict, List, Tuple

from text_world.env_sentence import SentenceWorld
from text_world.actions import ALL_ACTIONS
from text_world.state import SentenceState

def reward_of_state(st: SentenceState) -> float:
    facts = (st.fact_mask & 1) + ((st.fact_mask >> 1) & 1) + ((st.fact_mask >> 2) & 1)
    return 1.0 * facts - 3.0 * st.contradiction - 0.1 * st.length

def finite_horizon_value_iteration(world: SentenceWorld, H: int) -> Tuple[List[float], List[int]]:
    nS = len(world.states)
    V_prev = [0.0] * nS
    policy = [ALL_ACTIONS[0]] * nS

    for _ in range(1, H + 1):
        V = [0.0] * nS
        for s in range(nS):
            best_val = -1e18
            best_a = ALL_ACTIONS[0]
            for a in ALL_ACTIONS:
                exp = 0.0
                for sp, p in world.T[(s, a)].items():
                    exp += p * (reward_of_state(world.states[sp]) + V_prev[sp])
                if exp > best_val:
                    best_val = exp
                    best_a = a
            V[s] = best_val
            policy[s] = best_a
        V_prev = V
    return V_prev, policy

def eval_policy_exact(world: SentenceWorld, policy: List[int], s0: int, H: int) -> float:
    nS = len(world.states)
    dist = [0.0] * nS
    dist[s0] = 1.0
    ret = 0.0

    for _t in range(H):
        next_dist = [0.0] * nS
        for s in range(nS):
            if dist[s] == 0.0:
                continue
            a = policy[s]
            for sp, p in world.T[(s, a)].items():
                next_dist[sp] += dist[s] * p
        for sp in range(nS):
            if next_dist[sp] != 0.0:
                ret += next_dist[sp] * reward_of_state(world.states[sp])
        dist = next_dist

    return ret

def brute_force_best_constant_exact(world: SentenceWorld, s0: int, H: int) -> Tuple[List[int], float]:
    nS = len(world.states)
    best_pi = None
    best_ret = -1e18
    for a in ALL_ACTIONS:
        pi = [a] * nS
        r = eval_policy_exact(world, pi, s0=s0, H=H)
        if r > best_ret:
            best_ret = r
            best_pi = pi
    return best_pi, best_ret

def eval_risk_exact(world: SentenceWorld, policy: List[int], s0: int, H: int) -> float:
    nS = len(world.states)
    dist = [0.0] * nS
    dist[s0] = 1.0
    risk_sum = 0.0
    for _t in range(H):
        next_dist = [0.0] * nS
        for s in range(nS):
            if dist[s] == 0.0:
                continue
            a = policy[s]
            for sp, p in world.T[(s, a)].items():
                next_dist[sp] += dist[s] * p
        haz_prob = 0.0
        for sp in range(nS):
            if next_dist[sp] != 0.0 and world.states[sp].hazard() == 1:
                haz_prob += next_dist[sp]
        risk_sum += haz_prob
        dist = next_dist
    return risk_sum / H

def select_policy_under_risk(world: SentenceWorld, candidates: List[List[int]], s0: int, H: int, epsilon: float) -> Dict[str, object]:
    scored = []
    for pi in candidates:
        r = eval_policy_exact(world, pi, s0=s0, H=H)
        rk = eval_risk_exact(world, pi, s0=s0, H=H)
        scored.append((pi, r, rk))

    feasible = [t for t in scored if t[2] <= epsilon]
    if feasible:
        chosen = max(feasible, key=lambda t: t[1])
        mode = "feasible_best_return"
    else:
        chosen = min(scored, key=lambda t: t[2])
        mode = "fallback_min_risk"

    best_unconstrained = max(scored, key=lambda t: t[1])
    return {
        "mode": mode,
        "chosen_return": chosen[1],
        "chosen_risk": chosen[2],
        "chosen_action": chosen[0][0],
        "opt_return": best_unconstrained[1],
        "opt_risk": best_unconstrained[2],
        "epsilon": epsilon,
    }
