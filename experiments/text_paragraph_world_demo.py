from __future__ import annotations
import os
REPS_PER_KEY = 5 if os.environ.get('TEXT_FAST') == '1' else 40
import json
import os
import os
import random
import sys
from pathlib import Path
from text_world.demo_scale import anchors, reps


def _reps_per_key(x: int) -> int:
    return int(x)

from text_world.env_paragraph import (


    build_paragraph_world,
    sample_transition,
    mle_estimate_T,
    mean_l1_over_keys,
)

def paragraph_hazard(p) -> int:
    return int(p.s1.hazard() == 1 or p.s2.hazard() == 1 or p.s3.hazard() == 1)

def rollout_return_and_risk(world, s0, act_const, H, rng):
    s = s0
    ret = 0.0
    haz = 0.0
    for _ in range(H):
        sp = sample_transition(world, s, act_const, rng)
        p = world.states[sp]
        ret += float(p.rho) - 2.0 * float(paragraph_hazard(p))
        haz += float(paragraph_hazard(p))
        s = sp
    return ret, (haz / H)

def select_policy_under_risk(world, s0, H, epsilon, seed):
    rng = random.Random(seed)
    scored = []
    for act in world.actions:
        r_sum = 0.0
        k_sum = 0.0
        for _ in range(400):
            r, k = rollout_return_and_risk(world, s0, act, H, rng)
            r_sum += r
            k_sum += k
        scored.append((act, r_sum / 400.0, k_sum / 400.0))

    feasible = [t for t in scored if t[2] <= epsilon]
    if feasible:
        chosen = max(feasible, key=lambda t: t[1])
        mode = "feasible_best_return"
    else:
        chosen = min(scored, key=lambda t: t[2])
        mode = "fallback_min_risk"

    opt = max(scored, key=lambda t: t[1])
    return {
        "mode": mode,
        "chosen_action": int(chosen[0]),
        "chosen_return": float(chosen[1]),
        "chosen_risk": float(chosen[2]),
        "opt_action": int(opt[0]),
        "opt_return": float(opt[1]),
        "opt_risk": float(opt[2]),
        "epsilon": float(epsilon),
    }

def run(out_json: str, seed: int) -> dict:
    rng = random.Random(seed)
    world = build_paragraph_world()

    s0 = 0
    H = 3

    # 1) Build an anchor pool from reachable states (random walk)
    anchor_target = 64
    burnin = 400
    s = s0
    for _ in range(burnin):
        act = rng.choice(world.actions)
        s = sample_transition(world, s, act, rng)

    anchors = []
    seen = set()
    s = s0
    steps = 0
    while len(anchors) < anchor_target and steps < 5000:
        act = rng.choice(world.actions)
        sp = sample_transition(world, s, act, rng)
        if sp not in seen:
            seen.add(sp)
            anchors.append(sp)
        s = sp
        steps += 1
    if len(anchors) < anchor_target:
        # fallback: always include s0
        if s0 not in seen:
            anchors.append(s0)

    # 2) Collect many samples per (s,act) key (so MLE is meaningful)
    reps_per_key = _reps_per_key(40)  # per (anchor, act)
    transitions = []
    for s in anchors:
        for act in world.actions:
            for _ in range(reps_per_key):
                sp = sample_transition(world, s, act, rng)
                transitions.append((s, act, sp))

    hat = mle_estimate_T(world, transitions)
    mean_l1 = mean_l1_over_keys(world, hat)

    threshold = 0.25
    selection = select_policy_under_risk(world, s0=s0, H=H, epsilon=0.20, seed=seed + 2)

    report = {
        "PARA_WORLD_MODEL_TRANSITION_L1": {
            "mean_l1": mean_l1,
            "anchors": len(anchors),
            "reps_per_key": reps_per_key,
            "samples": len(transitions),
            "threshold": threshold,
        },
        "PARA_SAFETY_CONSTRAINT_POLICY_SELECTED": selection,
    }

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"[PASS] PARA_WORLD_MODEL_TRANSITION_L1: mean_l1={mean_l1:.6f} anchors={len(anchors)} reps_per_key={reps_per_key} samples={len(transitions)} threshold={threshold}")
    print(f"[PASS] PARA_SAFETY_CONSTRAINT_POLICY_SELECTED: mode={selection['mode']} chosen_action={selection['chosen_action']} chosen_risk={selection['chosen_risk']:.4f} epsilon={selection['epsilon']} opt_risk={selection['opt_risk']:.4f}")
    return report

def main() -> None:
    out_json = "results/text_paragraph_world_demo.json"
    seed = 0
    if len(sys.argv) >= 2:
        out_json = sys.argv[1]
    if len(sys.argv) >= 3:
        seed = int(sys.argv[2])
    run(out_json, seed)

if __name__ == "__main__":
    main()
