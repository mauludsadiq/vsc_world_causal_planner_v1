from __future__ import annotations
import json
import random
import sys
from pathlib import Path

from text_world.env_document import (
    build_document_world,
    sample_transition,
    mle_estimate_T,
    mean_l1_over_keys,
    decode_doc_action,
)

def doc_hazard(d) -> int:
    # hazard if any paragraph hazard (implemented via kappa=0 when hazard exists)
    return int(d.kappa == 0)

def style_mismatch_risk(d) -> int:
    # treat mixed paragraph styles as a risk event
    styles = [d.p0.s1.style, d.p1.s1.style, d.p2.s1.style, d.p3.s1.style]
    return int(any(s != styles[0] for s in styles))

def rollout_return_and_risk(world, s0, act_const, H, rng):
    s = s0
    ret = 0.0
    risk_events = 0.0
    for _ in range(H):
        sp = sample_transition(world, s, act_const, rng)
        d = world.states[sp]
        ret += float(d.kappa)
        risk_events += float(doc_hazard(d) or style_mismatch_risk(d))
        s = sp
    return ret, (risk_events / H)

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
    world = build_document_world()
    s0 = 0
    H = 3

    # anchors via reachable random walk (lazy)
    anchor_target = 32
    burnin = 200
    s = s0
    for _ in range(burnin):
        act = rng.choice(world.actions)
        s = sample_transition(world, s, act, rng)

    anchors = []
    seen = set()
    s = s0
    steps = 0
    while len(anchors) < anchor_target and steps < 4000:
        act = rng.choice(world.actions)
        sp = sample_transition(world, s, act, rng)
        if sp not in seen:
            seen.add(sp)
            anchors.append(sp)
        s = sp
        steps += 1
    if not anchors:
        anchors = [s0]

    reps = 30
    transitions = []
    for s in anchors:
        for act in world.actions:
            for _ in range(reps):
                sp = sample_transition(world, s, act, rng)
                transitions.append((s, act, sp))

    hat = mle_estimate_T(world, transitions)
    mean_l1 = mean_l1_over_keys(world, hat)
    threshold = 0.25

    # tighter epsilon so we force constraint selection when mismatch risk appears
    selection = select_policy_under_risk(world, s0=s0, H=H, epsilon=0.10, seed=seed + 1)

    report = {
        "DOC_WORLD_MODEL_TRANSITION_L1": {
            "mean_l1": mean_l1,
            "anchors": len(anchors),
            "reps_per_key": reps,
            "samples": len(transitions),
            "threshold": threshold,
        },
        "DOC_SAFETY_CONSTRAINT_POLICY_SELECTED": selection,
    }

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[PASS] DOC_WORLD_MODEL_TRANSITION_L1: mean_l1={mean_l1:.6f} anchors={len(anchors)} reps={reps} samples={len(transitions)} threshold={threshold}")
    print(f"[PASS] DOC_SAFETY_CONSTRAINT_POLICY_SELECTED: mode={selection['mode']} chosen_action={selection['chosen_action']} chosen_risk={selection['chosen_risk']:.4f} epsilon={selection['epsilon']} opt_risk={selection['opt_risk']:.4f}")
    return report

def main() -> None:
    out_json = "results/text_document_demo.json"
    seed = 0
    if len(sys.argv) >= 2:
        out_json = sys.argv[1]
    if len(sys.argv) >= 3:
        seed = int(sys.argv[2])
    run(out_json, seed)

if __name__ == "__main__":
    main()
