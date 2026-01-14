import math
import random

from text_world.planning_hier_block import hier_beam_search_block


class TinyRiskWorld:
    def __init__(self):
        self.S = [0, 1, 2]
        self.A = [0, 1]

    def enumerate_actions(self, s):
        return list(self.A)

    def step(self, s, a, rng):
        s = int(s)
        a = int(a)
        if s == 0 and a == 0:
            return 1, 1.0, 0.0
        if s == 0 and a == 1:
            return 2, 2.0, 0.9
        if s == 1 and a == 0:
            return 1, 0.0, 0.0
        if s == 1 and a == 1:
            return 2, 1.0, 0.2
        if s == 2 and a == 0:
            return 2, 0.0, 0.0
        if s == 2 and a == 1:
            return 2, 0.0, 0.0
        return s, 0.0, 0.0


def brute_force_best(world, s0, H, epsilon):
    best = None
    best_ret = -1e9
    best_risk = None
    actions = list(world.enumerate_actions(0))
    seqs = [[]]
    for _ in range(int(H)):
        seqs = [seq + [a] for seq in seqs for a in actions]

    for seq in seqs:
        s = int(s0)
        ret = 0.0
        risk_max = 0.0
        for t, a in enumerate(seq):
            sp, dr, rk = world.step(s, int(a), random.Random(1234 + 17 * t + int(a)))
            s = int(sp)
            ret += float(dr)
            risk_max = float(max(risk_max, float(rk)))
        if risk_max <= float(epsilon) and ret > best_ret:
            best_ret = ret
            best = seq
            best_risk = risk_max

    if best is None:
        best_ret = -1e9
        best_risk = 0.0
        for seq in seqs:
            s = int(s0)
            ret = 0.0
            risk_max = 0.0
            for t, a in enumerate(seq):
                sp, dr, rk = world.step(s, int(a), random.Random(1234 + 17 * t + int(a)))
                s = int(sp)
                ret += float(dr)
                risk_max = float(max(risk_max, float(rk)))
            if ret > best_ret:
                best_ret = ret
                best = seq
                best_risk = risk_max

    return best, best_ret, best_risk


def test_hier_planner_respects_epsilon_and_matches_reference_small():
    world = TinyRiskWorld()
    seed = 0
    epsilon = 0.3
    H = 2
    beam = 16

    out = hier_beam_search_block(world, 0, seed=seed, beam=beam, H_macro=H, epsilon=epsilon)
    plan = out["HIER_BLOCK_PLAN"]

    assert plan["risk_max"] <= epsilon
    assert len(plan["micro_traces"]) == H

    ref_seq, ref_ret, ref_risk = brute_force_best(world, 0, H, epsilon)

    assert math.isfinite(plan["ret_sum"])
    assert abs(plan["ret_sum"] - ref_ret) <= 1e-9
    assert ref_risk <= epsilon
